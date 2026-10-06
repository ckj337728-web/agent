"""Context 有效管理：组装策略、追问支持、基础压缩。

对应任务 T019（组装策略）、T020（纯对话追问）、T021（带工具追问）、
T022（基础压缩）、T023（超长工具输出截断）。

约束与设计依据（spec.md 3.5）：

- **哪些信息入 context**——本模块把策略显式化为 :class:`ContextPolicy`，
  并附「为什么这样放」的说明（见各字段注释与 :meth:`ContextBuilder.build` 文档）；
- **最大轮次限制**：loop 上限属主循环（T028），本模块负责历史长度控制；
- **状态记忆**：同一 session 内持续对话要记住此前状态，包括此前的工具调用结果；
- **追问支持**：纯对话追问与带工具追问都要能正确关联前文；
- **基础压缩**：context 过长时按轮次/字符数截断或摘要，不要求复杂压缩。

关键取舍（写进 README 的 memory 说明）：

1. **用户输入必放**——它是对话的锚点，任何压缩策略都必须保留最近若干轮的用户输入；
2. **工具执行结果要放**——"带工具的追问"依赖它（先查天气，再问"那明天呢"），
   但**注入前按 ``tool_result_max_chars`` 截断**，避免单条超长结果挤占上下文；
3. **思考过程默认不放回**——它是模型的内部推理，价值随时间快速衰减，
   且极易膨胀；默认只保留最终答复与工具结果。需要时可开启，
   并同样受长度上限约束；
4. **session 状态以独立 system 块置顶**——待办等状态是"当前事实"而非历史，
   放进历史容易被压缩丢掉，单独成块可保证始终可见。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .llm import ChatMessage
from .parser import OUTPUT_PROTOCOL_INSTRUCTIONS
from .session import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    Message,
    Session,
)

#: 与 agent.tools.todo 共享的状态键后缀（避免在此重复硬编码 "todos"）
TODOS_STATE_KEY = "todos"

#: 单条工具结果注入时的默认长度上限（spec 6.3 B4）
DEFAULT_TOOL_RESULT_MAX_CHARS = 2000

#: 默认 context 字符上限（约对应 8k~12k tokens 的中文文本）
DEFAULT_MAX_CONTEXT_CHARS = 24000

#: 摘要提示的前缀
_SUMMARY_HEADER = "[以下为较早对话的压缩摘要]"
_ELIDED_HEADER = "[以下为较早对话，因上下文过长已省略]"


@dataclass
class ContextPolicy:
    """context 组装与压缩策略（全部可配置）。

    对应 spec 3.5 的"哪些信息要塞入 context 更合适"——
    这里把每一项决定变成显式配置，便于解释与调参。
    """

    #: 是否把 assistant 的思考过程放回 context。
    #: 默认 False：思考过程价值衰减快、体积大，且工具调用协议不依赖它。
    include_reasoning: bool = False

    #: 单条工具结果注入上限；超长结果会被截断并标注省略字数（T023）
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS

    #: context 字符总量上限；超出即触发基础压缩（T022）
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS

    #: 压缩时**至少**保留的最近轮次数（这几轮永远完整保留）
    keep_recent_turns: int = 3

    #: 是否把 session 状态（如待办）作为独立 system 块注入
    include_session_state: bool = True

    #: 是否注入协议说明（工具调用与输出格式）
    include_protocol_instructions: bool = True

    #: 压缩时每条被省略轮次的用户输入保留长度（生成摘要用）
    summary_snippet_chars: int = 120

    #: 摘要总长度上限。摘要本身也在 context 里，必须有限，
    #: 否则"省略 30 轮"反而把摘要撑成新的超长文本。
    max_summary_chars: int = 600

    def __post_init__(self) -> None:
        if self.tool_result_max_chars <= 0:
            raise ValueError("tool_result_max_chars 必须为正数")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars 必须为正数")
        if self.keep_recent_turns < 1:
            raise ValueError("keep_recent_turns 至少为 1")
        if self.max_summary_chars <= 0:
            raise ValueError("max_summary_chars 必须为正数")


@dataclass
class ContextBuildResult:
    """一次 context 构建的结果与统计（便于测试与 trace）。"""

    messages: List[ChatMessage] = field(default_factory=list)
    total_chars: int = 0
    tool_results_truncated: int = 0
    elided_turns: int = 0
    compressed: bool = False
    kept_turns: int = 0
    summary: str = ""


class ContextBuilder:
    """把 session 历史组装成发给 LLM 的消息列表。"""

    def __init__(self, policy: Optional[ContextPolicy] = None) -> None:
        self._policy = policy or ContextPolicy()

    @property
    def policy(self) -> ContextPolicy:
        return self._policy

    # -- 主入口 ------------------------------------------------------------ #

    def build(self, session: Session, system_prompt: str = "") -> ContextBuildResult:
        """组装 context。

        顺序固定为：**系统提示 → 协议说明 → session 状态快照 → 对话历史**。

        之所以把状态与历史分开：状态是"当前事实"（会被工具改写），
        历史是"发生过什么"（只增不改）。混在一起时压缩历史会顺带
        丢掉状态，导致"切回旧窗口后待办消失"。
        """
        result = ContextBuildResult()
        prefix: List[ChatMessage] = []

        if system_prompt:
            prefix.append(ChatMessage(role=ROLE_SYSTEM, content=system_prompt))
        if self._policy.include_protocol_instructions:
            prefix.append(
                ChatMessage(role=ROLE_SYSTEM, content=OUTPUT_PROTOCOL_INSTRUCTIONS)
            )

        state_block = self._render_session_state(session)
        if state_block:
            prefix.append(ChatMessage(role=ROLE_SYSTEM, content=state_block))

        turns = _group_into_turns(session.messages)
        history, stats = self._render_turns(turns, result)
        history = _drop_orphan_tool_messages(history)

        result.messages = prefix + history
        result.kept_turns = _count_user_turns(history)
        result.total_chars = _total_chars(result.messages)

        # 压缩后再量一次；若仍超限（例如单轮就超长），记入压缩标记
        if result.total_chars > self._policy.max_context_chars:
            result.compressed = True

        return result

    # -- 历史渲染 ---------------------------------------------------------- #

    def _render_turns(
        self,
        turns: List["_Turn"],
        result: ContextBuildResult,
    ) -> tuple:
        """把轮次渲染为消息列表，必要时丢弃最旧的轮次。

        压缩策略（基础版，spec 3.5）：从**最旧**的轮次开始整轮丢弃，
        直到总字符数落回上限；被丢弃的用户输入折叠成一段摘要保留。
        整轮丢弃而不是逐条丢弃，是为了保持
        "assistant tool_calls ↔ tool 结果"的配对关系不被打断。

        两阶段丢弃：先丢到 ``keep_recent_turns`` 为止；若仍超限
        （例如某轮本身就极长），继续丢到只剩最后一轮，保证不超上限。
        """
        kept = list(turns)
        dropped: List["_Turn"] = []

        # 阶段一：整轮丢弃，但始终保留 keep_recent_turns 轮
        while (
            len(kept) > self._policy.keep_recent_turns
            and self._estimate_chars(kept) > self._policy.max_context_chars
        ):
            dropped.append(kept.pop(0))

        # 阶段二：仍超限则突破 keep_recent_turns 下限，保底留最后一轮
        while (
            len(kept) > 1
            and self._estimate_chars(kept) > self._policy.max_context_chars
        ):
            dropped.append(kept.pop(0))

        result.elided_turns += len(dropped)

        messages, truncated = self._turns_to_messages(kept)
        result.tool_results_truncated = truncated

        if dropped:
            result.compressed = True
            result.summary = self._summarize(dropped)
            messages.insert(0, ChatMessage(role=ROLE_SYSTEM, content=result.summary))

        return messages, result

    def _estimate_chars(self, turns: Sequence["_Turn"]) -> int:
        """估算这些轮次渲染后的字符数（不需要完整渲染）。"""
        return sum(turn.char_size(self._policy.tool_result_max_chars) for turn in turns)

    def _turns_to_messages(self, turns: Iterable["_Turn"]) -> tuple:
        messages: List[ChatMessage] = []
        truncated = 0
        for turn in turns:
            for message in turn.messages:
                converted, was_truncated = self._to_chat_message(message)
                if converted is None:
                    continue
                if was_truncated:
                    truncated += 1
                messages.append(converted)
        return messages, truncated

    def _to_chat_message(self, message: Message) -> tuple:
        """把一条历史消息转成 ChatMessage。返回 ``(消息, 是否截断)``。"""
        if message.role == ROLE_TOOL:
            content, was_truncated = truncate_tool_result(
                message.content, self._policy.tool_result_max_chars
            )
            return (
                ChatMessage(
                    role=ROLE_TOOL,
                    content=content,
                    tool_call_id=message.tool_call_id or None,
                    name=message.name or None,
                ),
                was_truncated,
            )

        if message.role == ROLE_ASSISTANT:
            content = message.content or ""
            if self._policy.include_reasoning and message.thought:
                content = f"<thinking>{message.thought}</thinking>\n{content}".strip()
            payload: Dict[str, Any] = {}
            if message.tool_calls:
                # 回放工具调用，保证 assistant↔tool 配对完整
                payload["tool_calls"] = [
                    {
                        "id": call.call_id or f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for index, call in enumerate(message.tool_calls)
                ]
            return (
                ChatMessage(
                    role=ROLE_ASSISTANT,
                    content=content,
                    tool_calls=payload.get("tool_calls"),
                ),
                False,
            )

        return ChatMessage(role=message.role, content=message.content or ""), False

    # -- session 状态 ------------------------------------------------------ #

    def _render_session_state(self, session: Session) -> str:
        """把 session 状态渲染成置顶的 system 块。

        ``todo`` 工具的列表在此渲染（状态键形如 ``"<session_id>:todos"``），
        使状态不依赖历史轮次是否被压缩——这正是 spec 6.3 B6
        「连续多轮追问后回到早期话题，早期状态仍可召回」的实现方式。
        """
        if not self._policy.include_session_state or not session.state:
            return ""
        lines: List[str] = ["[当前会话状态]"]
        rendered_any = False

        for key, value in session.state.items():
            if not isinstance(key, str) or not key.endswith(f":{TODOS_STATE_KEY}"):
                continue
            if not isinstance(value, list) or not value:
                continue
            rendered_any = True
            session_id = key[: -len(f":{TODOS_STATE_KEY}")]
            lines.append(f"待办清单（会话 {session_id}）：")
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                mark = "x" if item.get("status") == "done" else " "
                lines.append(f"  [{mark}] #{item.get('id')} {item.get('content')}")

        # 其他状态键做通用渲染，便于后续工具复用同一机制
        for key, value in session.state.items():
            if isinstance(key, str) and key.endswith(f":{TODOS_STATE_KEY}"):
                continue
            rendered_any = True
            lines.append(f"{key}：{_compact(value)}")

        if not rendered_any:
            return ""
        lines.append("（以上状态由工具维护，请勿凭记忆编造）")
        return "\n".join(lines)

    # -- 摘要 -------------------------------------------------------------- #

    def _summarize(self, dropped: Sequence["_Turn"]) -> str:
        """把被省略的轮次折叠成一段摘要。

        基础版只保留每轮用户输入的片段——足以让模型知道"聊过这些话题"，
        不追求语义级摘要（spec 2.2 明确排除复杂压缩算法）。

        ``max_summary_chars`` 是**硬上限**：先逐条拼接，最后统一裁剪。
        用"先构造、后裁剪"而不是"边算预算边拼"，是因为片段本身可能带
        省略号（``user_snippet`` 追加 ``...``），逐字符预算极易差几个字符，
        而摘要超限恰好会抵消压缩的意义。
        """
        lines = [_ELIDED_HEADER, f"共省略 {len(dropped)} 轮较早对话，要点："]
        for turn in dropped:
            snippet = turn.user_snippet(self._policy.summary_snippet_chars)
            if snippet:
                lines.append(f"  - {snippet}")
        text = "\n".join(lines)

        if len(text) <= self._policy.max_summary_chars:
            return text

        marker = "\n  - ...（更多较早轮次已省略）"
        keep = self._policy.max_summary_chars - len(marker) - 1
        if keep <= len(_ELIDED_HEADER):
            # 预算过小：至少保留标题，便于排查"为什么历史没了"
            return _ELIDED_HEADER[: self._policy.max_summary_chars]
        return f"{text[:keep]}{marker}"


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


@dataclass
class _Turn:
    """一轮对话：一条用户输入 + 其引发的全部消息。"""

    turn: int
    messages: List[Message] = field(default_factory=list)

    def char_size(self, tool_result_max_chars: int) -> int:
        total = 0
        for message in self.messages:
            text = message.content or ""
            if message.role == ROLE_TOOL:
                text, _ = truncate_tool_result(text, tool_result_max_chars)
            total += len(text)
        return total

    def user_snippet(self, limit: int) -> str:
        for message in self.messages:
            if message.role == ROLE_USER:
                text = (message.content or "").strip().replace("\n", " ")
                if len(text) > limit:
                    return f"{text[:limit]}..."
                return text
        return ""


def _group_into_turns(messages: Sequence[Message]) -> List[_Turn]:
    """按 turn 编号把消息分组，保证整轮一起保留或一起丢弃。"""
    turns: List[_Turn] = []
    index: Dict[int, _Turn] = {}
    for message in messages:
        key = message.turn
        turn = index.get(key)
        if turn is None:
            turn = _Turn(turn=key)
            index[key] = turn
            turns.append(turn)
        turn.messages.append(message)
    return turns


def _count_user_turns(messages: Sequence[ChatMessage]) -> int:
    return sum(1 for message in messages if message.role == ROLE_USER)


def _drop_orphan_tool_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """丢弃没有对应 assistant tool_call 的 tool 消息。

    OpenAI 兼容接口要求 ``role=tool`` 的消息必须紧跟其对应的
    ``assistant.tool_calls``；历史中出现孤儿 tool 消息会导致请求被拒。
    压缩/截断等操作可能破坏配对，因此在组装末尾统一兜底清理。
    """
    announced = set()
    cleaned: List[ChatMessage] = []
    for message in messages:
        if message.role == ROLE_ASSISTANT and message.tool_calls:
            for index, call in enumerate(message.tool_calls):
                announced.add(str(call.get("id") or f"call_{index}"))
            cleaned.append(message)
            continue
        if message.role == ROLE_TOOL:
            key = message.tool_call_id or ""
            if key and key in announced:
                cleaned.append(message)
            # 没有 tool_call_id 或未被声明过 → 视为孤儿，丢弃
            continue
        cleaned.append(message)
    return cleaned


def _total_chars(messages: Sequence[ChatMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


def _total_length(lines: Sequence[str]) -> int:
    """多行文本的总长度（含换行符）。"""
    return sum(len(line) for line in lines) + max(0, len(lines) - 1)


def _compact(value: Any, limit: int = 200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


def truncate_tool_result(text: str, max_chars: int) -> tuple:
    """截断超长工具结果，并标注省略了多少字符（T023，spec 6.3 B4）。

    返回 ``(截断后文本, 是否发生截断)``。
    """
    if text is None:
        return "", False
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    # 保留头部：工具结果的结论通常在前面（如计算结果、检索命中列表）
    return f"{text[:max_chars]}\n...[已截断，省略 {omitted} 字符]", True
