"""测试共用工具：提供可离线运行的 fake LLM 传输与客户端。

对应任务 T034，约束来自 spec.md 4.3 / 6.4 D1：
测试必须**不依赖真实网络与真实密钥**即可确定性重复运行。

跑法（在项目根目录执行）：

    python -m unittest discover -s tests -t . -v

``-t .`` 指定顶层目录为项目根，使 ``tests`` 作为包被导入，
从而 ``from tests.helpers import ...`` 可用。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# 允许直接 `python tests/xxx.py` 单文件跑测时也能找到 agent 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.config import LLMConfig  # noqa: E402
from agent.llm import ChatMessage, ChatResponse, LLMClient  # noqa: E402
from agent.tools.base import BaseTool, ToolContext, ToolResult  # noqa: E402
from agent.trace import Tracer  # noqa: E402


def sample_config(**overrides: Any) -> LLMConfig:
    """构造一份测试用配置（密钥为假值，绝不含真实密钥）。"""
    params: Dict[str, Any] = {
        "api_key": "test-key-not-real",
        "base_url": "https://example.invalid/v1",
        "model": "test-model",
        "timeout_seconds": 5.0,
        "max_retries": 0,
    }
    params.update(overrides)
    return LLMConfig(**params)


def chat_completion_payload(
    content: str = "",
    tool_calls: Optional[Sequence[Mapping[str, Any]]] = None,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """构造一个 OpenAI 兼容的成功响应体。"""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [dict(tc) for tc in tool_calls]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def tool_call(
    name: str, arguments: Any, call_id: str = "call_1"
) -> Dict[str, Any]:
    """构造一个 tool_call 结构；arguments 会被序列化为 JSON 字符串。"""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeTransport:
    """可编程的传输层替身：按顺序返回预置结果，并记录收到的请求。

    每个 item 可以是：
      - ``(status, payload)`` 元组：作为一次响应返回；
      - ``Exception`` 实例：抛出以模拟网络/超时错误；
      - 可调用对象：接收 (url, headers, body, timeout) 并返回上述任意一种。
    """

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = list(items)
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> Tuple[int, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if not self._items:
            raise AssertionError(
                "FakeTransport 收到超出预期的第 "
                f"{len(self.calls)} 次调用；请补足预置响应。"
            )
        item = self._items.pop(0)
        if callable(item) and not isinstance(item, tuple):
            item = item(url, headers, body, timeout)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    @property
    def remaining(self) -> int:
        return len(self._items)


def make_client(
    items: Sequence[Any],
    tracer: Optional[Tracer] = None,
    **config_overrides: Any,
) -> Tuple[LLMClient, FakeTransport]:
    """构造注入 fake transport 的客户端；退避等待被替换为空操作。"""
    transport = FakeTransport(items)
    client = LLMClient(
        sample_config(**config_overrides),
        transport=transport,
        tracer=tracer,
        sleep=lambda _seconds: None,
    )
    return client, transport


class AgentTestCase(unittest.TestCase):
    """项目内测试基类：统一路径与常用构造。"""

    maxDiff = None

    def sample_config(self, **overrides: Any) -> LLMConfig:
        return sample_config(**overrides)

    def make_client(self, items: Sequence[Any], tracer: Optional[Tracer] = None, **kw: Any):
        return make_client(items, tracer=tracer, **kw)


# --------------------------------------------------------------------------- #
# 主循环测试用：按脚本返回响应的假 LLM
# --------------------------------------------------------------------------- #


class ScriptedLLM:
    """按脚本依次返回响应的假 LLM 客户端。

    与 :class:`FakeTransport` 的区别：这一层直接替掉 ``LLMClient``，
    让主循环测试不必构造 HTTP 报文；同时记录每次调用收到的消息与工具列表，
    便于断言"context 里到底放了什么"。

    每个脚本项可以是：
      - ``str``：作为 content 返回（可用于测试协议 JSON 与纯文本回退）；
      - ``dict``：形如 ``{"content": ..., "tool_calls": [...]}``；
      - ``Exception``：抛出以模拟 LLM 失败。
    """

    def __init__(self, script: Sequence[Any], model: str = "scripted-model") -> None:
        self._script = list(script)
        self._model = model
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        trace_id: str = "",
        temperature: float = 0.0,
    ) -> ChatResponse:
        self.calls.append(
            {
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "tool_calls": m.tool_calls,
                        "tool_call_id": m.tool_call_id,
                        "name": m.name,
                    }
                    for m in messages
                ],
                "tools": [dict(t) for t in (tools or [])],
                "trace_id": trace_id,
            }
        )
        if not self._script:
            raise AssertionError(
                f"ScriptedLLM 收到第 {len(self.calls)} 次调用，但脚本已用尽；"
                "请补足预置响应。"
            )
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return ChatResponse(content=item, model=self._model, attempts=1)
        if isinstance(item, Mapping):
            return ChatResponse(
                content=str(item.get("content") or ""),
                tool_calls=[dict(tc) for tc in (item.get("tool_calls") or [])],
                finish_reason=str(item.get("finish_reason") or ""),
                model=self._model,
                attempts=1,
            )
        raise AssertionError(f"不支持的脚本项类型：{type(item).__name__}")

    @property
    def remaining(self) -> int:
        return len(self._script)

    # -- 断言辅助 ---------------------------------------------------------- #

    def messages_sent(self, index: int = -1) -> List[Dict[str, Any]]:
        return self.calls[index]["messages"]

    def joined_text(self, index: int = -1) -> str:
        return " ".join(m["content"] or "" for m in self.messages_sent(index))

    def tool_names_sent(self, index: int = -1) -> List[str]:
        return [t["function"]["name"] for t in self.calls[index]["tools"]]


def scripted_json(payload: Mapping[str, Any]) -> str:
    """把协议 JSON 序列化为 content 文本。"""
    return json.dumps(payload, ensure_ascii=False)


def protocol_tool_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    """构造 ``{"tool_calls": [...]}`` 形式的协议文本。"""
    return scripted_json({"tool_calls": [dict(c) for c in calls]})


def native_tool_call(name: str, arguments: Any, call_id: str = "c1") -> Dict[str, Any]:
    """构造 API 原生 tool_call 结构。"""
    return tool_call(name, arguments, call_id)


class RecordingTool(BaseTool):
    """记录调用参数的测试工具，可指定耗时或直接抛异常。"""

    name = "recorder"
    description = "记录每次调用的工具，用于主循环测试。"
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "任意值"},
            "delay": {"type": "number", "description": "模拟耗时秒数", "default": 0},
        },
        "required": ["value"],
    }

    def __init__(self, fail: bool = False) -> None:
        self.invocations: List[Dict[str, Any]] = []
        self.fail = fail

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.invocations.append(dict(arguments))
        delay = float(arguments.get("delay") or 0)
        if delay:
            import time as _time

            _time.sleep(delay)
        if self.fail:
            raise RuntimeError("模拟工具内部故障")
        return ToolResult.success(
            content=f"recorded:{arguments.get('value')}",
            data={"value": arguments.get("value")},
        )
