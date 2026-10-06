"""结构化输出协议与 LLM 输出解析。

对应任务 T013（协议定义）、T014（解析器）、T015（解析失败回退），
约束来自 spec.md 3.4：

- 必须能稳定提取三类内容：**思考过程 / 工具调用 / 最终答案**；
- 解析失败（格式非法、JSON 解析失败、字段缺失）时不得崩溃，需有回退策略；
- 支持一次响应中同时出现工具调用与最终答案的区分判定。

分层定位（spec 3.0）：本模块只负责"从响应里提取语义"，**不执行工具**，
也不决定何时停止循环。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class OutputKind(str, Enum):
    """一次 LLM 响应的动作类型。"""

    FINAL_ANSWER = "final_answer"    # 收敛为给用户的答案，循环应结束
    TOOL_CALLS = "tool_calls"        # 需要执行工具，循环应继续
    MIXED = "mixed"                  # 同时有工具调用与答案（先执行工具）
    EMPTY = "empty"                  # 既无工具调用也无有效答案，需上层兜底


class ParseSource(str, Enum):
    """解析走的哪条路径（保留证据便于 trace 与排障）。"""

    NATIVE_TOOL_CALLS = "native_tool_calls"  # 来自 API 原生 tool_calls 字段
    JSON_TEXT = "json_text"                  # 从文本中提取到协议 JSON
    JSON_FENCED = "json_fenced"               # 从 ```json 代码块中提取
    PLAIN_TEXT = "plain_text"                 # 回退：整段文本当作最终答案
    NONE = "none"                             # 无任何可用内容


#: 协议字段别名。模型偶尔会换用同义字段名，这里做容错映射。
_THOUGHT_KEYS = ("thought", "thinking", "reasoning", "think")
_ANSWER_KEYS = ("answer", "final_answer", "response", "reply", "content")
_TOOL_CALL_KEYS = ("tool_calls", "tool_call", "tools", "actions")

#: ```json ... ``` 或 ``` ... ``` 代码块
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: 带显式标签的思考段落，例如 "<thinking>...</thinking>"
_THINK_TAG_RE = re.compile(
    r"<(?:thinking|thought|reasoning)>(.*?)</(?:thinking|thought|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ToolCallRequest:
    """一次工具调用请求（已解析）。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_arguments: str = ""
    argument_error: Optional[str] = None

    @property
    def arguments_valid(self) -> bool:
        return self.argument_error is None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.call_id:
            payload["call_id"] = self.call_id
        if self.argument_error:
            payload["argument_error"] = self.argument_error
        return payload


@dataclass
class ParsedOutput:
    """一次 LLM 响应的解析结果。"""

    kind: OutputKind
    source: ParseSource
    thought: str = ""
    answer: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    #: 回退说明：解析过程中遇到的问题（供 trace 与测试断言）
    warnings: List[str] = field(default_factory=list)
    #: 原始文本，保留以便排查
    raw_text: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer.strip())

    @property
    def degraded(self) -> bool:
        """是否走了回退路径（非理想解析）。"""
        return bool(self.warnings) or self.source in (
            ParseSource.PLAIN_TEXT,
            ParseSource.NONE,
        )

    @property
    def is_empty(self) -> bool:
        return self.kind is OutputKind.EMPTY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source": self.source.value,
            "thought": self.thought,
            "answer": self.answer,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# 提示词片段：告诉模型输出协议
# --------------------------------------------------------------------------- #

OUTPUT_PROTOCOL_INSTRUCTIONS = """\
你可以直接给出最终答案，也可以调用工具。请严格遵守以下输出协议：

1. 当需要调用工具时，在回复中给出一个 JSON 对象（可放在 ```json 代码块中）：
   {"thought": "你的推理过程", "tool_calls": [{"name": "工具名", "arguments": {...}}]}
   调用完成后，你会收到每个工具的结果，然后可以继续调用工具或给出最终答案。

2. 当可以直接回答时，输出一个 JSON 对象：
   {"thought": "你的推理过程", "answer": "给用户的最终答案"}

3. thought 字段是你的内部推理，不会直接展示给用户；answer 才是给用户的答复。
4. 如果无法构造 JSON，也可以直接输出纯文本，此时整段文本会被当作最终答案。
"""


# --------------------------------------------------------------------------- #
# 解析入口
# --------------------------------------------------------------------------- #


def parse_response(
    content: str = "",
    native_tool_calls: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ParsedOutput:
    """解析一次 LLM 响应。

    参数：
        content:           响应文本（``ChatResponse.content``）
        native_tool_calls: API 原生工具调用（``ChatResponse.tool_calls``）

    解析优先级（这是回退策略的核心，spec 3.4 / 6.2 E5）：

    1. **原生 tool_calls** 存在 → 直接采用，content 作为思考过程或答案；
    2. content 中含**协议 JSON**（裸 JSON 或 ```json 代码块）→ 按协议取字段；
    3. 上述都失败 → **整段文本当作最终答案**（保证永远有结果，不崩溃）。

    任何步骤都不抛异常；问题会记入 ``warnings``。
    """
    warnings: List[str] = []
    raw_text = content or ""

    calls = _parse_native_tool_calls(native_tool_calls, warnings)
    if calls:
        return _build_native_result(calls, raw_text, warnings)

    protocol = _extract_protocol_json(raw_text, warnings)
    if protocol is not None:
        data, source = protocol
        return _build_from_protocol(data, raw_text, source, warnings)

    return _build_fallback(raw_text, warnings)


# --------------------------------------------------------------------------- #
# 1) 原生 tool_calls
# --------------------------------------------------------------------------- #


def _parse_native_tool_calls(
    native_tool_calls: Optional[Sequence[Mapping[str, Any]]], warnings: List[str]
) -> List[ToolCallRequest]:
    calls: List[ToolCallRequest] = []
    if not native_tool_calls:
        return calls

    for index, raw in enumerate(native_tool_calls):
        if not isinstance(raw, Mapping):
            warnings.append(f"tool_calls[{index}] 不是对象，已跳过")
            continue
        function = raw.get("function")
        if not isinstance(function, Mapping):
            warnings.append(f"tool_calls[{index}] 缺少 function 字段，已跳过")
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            warnings.append(f"tool_calls[{index}] 缺少函数名，已跳过")
            continue

        raw_arguments = function.get("arguments")
        arguments, error = _decode_arguments(raw_arguments)
        if error:
            warnings.append(f"tool_calls[{index}] 参数解析失败：{error}")
        calls.append(
            ToolCallRequest(
                name=name,
                arguments=arguments,
                call_id=str(raw.get("id") or ""),
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else "",
                argument_error=error,
            )
        )
    return calls


def _build_native_result(
    calls: List[ToolCallRequest], raw_text: str, warnings: List[str]
) -> ParsedOutput:
    """原生 tool_calls 场景。

    content 的语义按协议 **优先当作思考过程**：模型"边想边调工具"时写下的
    文字是给流程看的推理，不是给用户的答复。只有当出现显式 ``<thinking>``
    标签时，才把标签外剩下的文字认定为答复（→ MIXED）。
    """
    thought, answer = _split_thought_and_answer(raw_text)
    if not thought and not answer:
        thought = ""
    elif not thought:
        # 没有 thinking 标签：整段视为推理，不作为答复
        thought, answer = answer, ""
    kind = OutputKind.MIXED if answer else OutputKind.TOOL_CALLS
    return ParsedOutput(
        kind=kind,
        source=ParseSource.NATIVE_TOOL_CALLS,
        thought=thought,
        answer=answer,
        tool_calls=calls,
        warnings=warnings,
        raw_text=raw_text,
    )


def _split_thought_and_answer(text: str) -> Tuple[str, str]:
    """把自由文本拆成 (思考, 答案)。

    协议不保证模型一定输出 JSON，所以这里只做**保守**拆分：
    只有出现显式 ``<thinking>`` 标签时才把标签内容当思考，
    其余情况一律当作答案——宁可把答案给全，也不要吞掉用户可见内容。
    """
    text = (text or "").strip()
    if not text:
        return "", ""
    match = _THINK_TAG_RE.search(text)
    if match:
        thought = match.group(1).strip()
        answer = _THINK_TAG_RE.sub("", text).strip()
        return thought, answer
    return "", text


# --------------------------------------------------------------------------- #
# 2) 文本中的协议 JSON
# --------------------------------------------------------------------------- #


def _extract_protocol_json(
    text: str, warnings: List[str]
) -> Optional[Tuple[Dict[str, Any], ParseSource]]:
    """尝试从文本里提取协议 JSON。

    依次尝试：整段 JSON → ```json 代码块 → 花括号片段。

    返回 ``(数据, 来源)``；无法提取到对象时返回 None。诊断信息**始终**
    追加到传入的 ``warnings``——包括提取失败的情形，否则"看起来像 JSON
    但解析失败"这类关键线索会随返回值一起丢失。
    """
    candidate = (text or "").strip()
    if not candidate:
        return None
    looks_like_json = candidate.startswith(("{", "["))

    # a) 整段就是 JSON
    parsed = _try_json(candidate)
    if isinstance(parsed, Mapping):
        return dict(parsed), ParseSource.JSON_TEXT

    # b) ```json 代码块
    for block in _FENCE_RE.findall(candidate):
        block = block.strip()
        if not block:
            continue
        parsed = _try_json(block)
        if isinstance(parsed, Mapping):
            return dict(parsed), ParseSource.JSON_FENCED
        warnings.append("代码块内容不是合法 JSON 对象")

    # c) 花括号片段（处理"解释文字 + JSON"混合，可能不止一个片段）
    #
    # looks_like_json 用于区分"整段本身就是 JSON"与"从散文里挖出 JSON"：
    # 后者会在 trace 里留一条警告，方便排查模型为何不按协议输出。
    snippets = _iter_balanced_objects(candidate)
    if snippets:
        for snippet in snippets:
            parsed = _try_json(snippet)
            if isinstance(parsed, Mapping):
                if not looks_like_json:
                    warnings.append("从混合文本中提取到协议 JSON")
                return dict(parsed), ParseSource.JSON_TEXT
        warnings.append("提取到的花括号片段不是合法 JSON")
    elif looks_like_json:
        # 文本长得像 JSON 却连括号都没配平（截断、语法错误等）
        warnings.append("文本看似 JSON，但未能解析出对象")

    return None


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _iter_balanced_objects(text: str) -> List[str]:
    """返回文本中所有**括号平衡**的 ``{...}`` 片段。

    正确处理两种情况（都是真实模型的常见输出）：

    - 字符串内的花括号不参与配对，例如 ``{"answer": "包含 } 的答案"}``；
    - 前一个片段不是合法 JSON 时，还能继续找到后面的片段，
      例如 ``说明：{'不是 JSON'}`` 后面跟着真正的协议 JSON。
    """
    snippets: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippets.append(text[start : index + 1])
                    start = -1
    return snippets


def _build_from_protocol(
    data: Mapping[str, Any],
    raw_text: str,
    source: ParseSource,
    warnings: List[str],
) -> ParsedOutput:
    thought = _first_string(data, _THOUGHT_KEYS)
    answer = _first_string(data, _ANSWER_KEYS)
    calls = _parse_protocol_tool_calls(data, warnings)

    if not calls and not answer.strip():
        # 协议 JSON 解析成功，但既无工具调用也无有效答案：可能是字段名不在
        # 别名表里，或模型只输出了 thought。
        #
        # 关键取舍：这里**不能**把原始文本当答案——原始文本就是这段 JSON，
        # 把它回给用户等于把 JSON 原文泄露出去。因此按"空结果"处理，
        # 由上层决定重试或兜底（spec 3.4 的回退策略由上层收口）。
        warnings.append(
            "协议 JSON 中既无工具调用也无有效答案字段；"
            f"实际字段：{sorted(data)}"
        )
        return ParsedOutput(
            kind=OutputKind.EMPTY,
            source=source,
            thought=thought,
            warnings=warnings,
            raw_text=raw_text,
        )

    if calls and answer.strip():
        kind = OutputKind.MIXED
    elif calls:
        kind = OutputKind.TOOL_CALLS
    else:
        kind = OutputKind.FINAL_ANSWER

    return ParsedOutput(
        kind=kind,
        source=source,
        thought=thought,
        answer=answer,
        tool_calls=calls,
        warnings=warnings,
        raw_text=raw_text,
    )


def _parse_protocol_tool_calls(
    data: Mapping[str, Any], warnings: List[str]
) -> List[ToolCallRequest]:
    raw_calls: Any = None
    for key in _TOOL_CALL_KEYS:
        if key in data:
            raw_calls = data[key]
            break
    if raw_calls is None:
        return []

    # 允许单个对象写法 {"tool_call": {...}}
    if isinstance(raw_calls, Mapping):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, (list, tuple)):
        warnings.append(f"tool_calls 字段不是数组：{type(raw_calls).__name__}")
        return []

    calls: List[ToolCallRequest] = []
    for index, entry in enumerate(raw_calls):
        if not isinstance(entry, Mapping):
            warnings.append(f"tool_calls[{index}] 不是对象，已跳过")
            continue
        # 同时兼容 {"name":..,"arguments":..} 与嵌套的 OpenAI 风格
        function = entry.get("function")
        source_entry = function if isinstance(function, Mapping) else entry
        name = str(source_entry.get("name") or "").strip()
        if not name:
            warnings.append(f"tool_calls[{index}] 缺少 name，已跳过")
            continue

        raw_arguments = source_entry.get("arguments")
        if raw_arguments is None:
            raw_arguments = source_entry.get("parameters")
        arguments, error = _decode_arguments(raw_arguments)
        if error:
            warnings.append(f"tool_calls[{index}]（{name}）参数解析失败：{error}")
        calls.append(
            ToolCallRequest(
                name=name,
                arguments=arguments,
                call_id=str(entry.get("id") or ""),
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else "",
                argument_error=error,
            )
        )
    return calls


def _first_string(data: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _decode_arguments(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """把 arguments 解码为字典。

    模型可能给出：JSON 字符串、已是对象、空、或截断的非法 JSON。
    后者返回错误说明而不抛异常（spec 3.4 回退要求）。
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, str):
        parsed = _try_json(raw)
        if isinstance(parsed, Mapping):
            return dict(parsed), None
        if parsed is None:
            return {}, f"arguments 不是合法 JSON：{_shorten(raw)}"
        return {}, f"arguments 解析结果不是对象：{type(parsed).__name__}"
    return {}, f"arguments 类型不支持：{type(raw).__name__}"


def _shorten(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


# --------------------------------------------------------------------------- #
# 3) 回退：整段文本当作最终答案
# --------------------------------------------------------------------------- #


def _build_fallback(raw_text: str, warnings: List[str]) -> ParsedOutput:
    thought, answer = _split_thought_and_answer(raw_text)
    if answer:
        return ParsedOutput(
            kind=OutputKind.FINAL_ANSWER,
            source=ParseSource.PLAIN_TEXT,
            thought=thought,
            answer=answer,
            warnings=warnings,
            raw_text=raw_text,
        )
    return ParsedOutput(
        kind=OutputKind.EMPTY,
        source=ParseSource.NONE,
        warnings=warnings,
        raw_text=raw_text,
    )
