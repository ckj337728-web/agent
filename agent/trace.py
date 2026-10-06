"""trace / 执行日志基础设施。

对应任务 T006，约束来自 spec.md：

- 3.7 / 6.1 N11：每次 LLM 调用与工具调用都必须留下**可读、可回溯**的记录，
  用于回答"为什么调了这个工具""为什么给出了这个答案"；
- 3.1：工具调用的入参、原始输出（或截断后输出）、耗时、错误必须进入 trace。

设计要点：

- 零依赖，仅用标准库 `json` / `time` / `dataclasses`；
- 一次用户输入触发的处理过程共享一个 ``trace_id``，便于把同一轮的
  LLM 调用、工具调用、循环步骤串起来回放；
- 每条记录既给人读（默认单行文本），也可切换为 JSONL 便于机器解析；
- 超长内容默认截断，避免 trace 本身失控。
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, TextIO

# 事件类型
EVENT_SESSION = "session"
EVENT_LOOP_STEP = "loop_step"
EVENT_LLM_CALL = "llm_call"
EVENT_TOOL_CALL = "tool_call"
EVENT_ERROR = "error"

DEFAULT_MAX_TEXT = 500


def truncate_text(value: Any, max_length: int = DEFAULT_MAX_TEXT) -> str:
    """把任意值转为单行文本并做长度截断（spec 6.3 B4 的通用处理习惯）。"""
    if value is None:
        return ""
    text = value if isinstance(value, str) else _to_json_text(value)
    text = text.replace("\r\n", "\n").replace("\n", "\\n")
    if max_length >= 0 and len(text) > max_length:
        return f"{text[:max_length]}...<truncated {len(text) - max_length} chars>"
    return text


def _to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class TraceRecord:
    """一条 trace 记录。"""

    event: str
    name: str
    trace_id: str = ""
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 6),
            "trace_id": self.trace_id,
            "event": self.event,
            "name": self.name,
            "payload": self.payload,
        }

    def to_line(self, max_text: int = DEFAULT_MAX_TEXT) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        parts = [stamp, f"[{self.event}]", self.name]
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        for key, value in self.payload.items():
            parts.append(f"{key}={truncate_text(value, max_text)}")
        return " ".join(parts)


class Tracer:
    """记录并输出 trace。默认写入 stderr，避免污染给用户的最终答案。

    参数：
        stream:      输出目标，默认 ``sys.stderr``
        fmt:         ``"text"``（默认，人读）或 ``"jsonl"``（机器解析）
        enabled:     关闭后所有记录变为空操作，供测试静默使用
        max_text:    单字段最大字符数，超长截断
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        fmt: str = "text",
        enabled: bool = True,
        max_text: int = DEFAULT_MAX_TEXT,
    ) -> None:
        if fmt not in ("text", "jsonl"):
            raise ValueError(f"不支持的 trace 格式：{fmt!r}，仅支持 'text' 或 'jsonl'")
        self._stream = stream if stream is not None else sys.stderr
        self._fmt = fmt
        self._enabled = enabled
        self._max_text = max_text
        self._records: List[TraceRecord] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def records(self) -> List[TraceRecord]:
        """本次运行累计的记录（便于测试断言 trace 完整性，spec 5.7）。"""
        return list(self._records)

    def new_trace_id(self) -> str:
        """为一次用户输入的处理过程生成 trace id。"""
        return uuid.uuid4().hex[:8]

    def record(self, event: str, name: str, trace_id: str = "", **payload: Any) -> TraceRecord:
        """记录一条事件。payload 中的 None 值会被丢弃，保持输出简洁。"""
        entry = TraceRecord(
            event=event,
            name=name,
            trace_id=trace_id,
            payload={k: v for k, v in payload.items() if v is not None},
        )
        self._records.append(entry)
        if self._enabled:
            self._write(entry)
        return entry

    def session_event(self, name: str, trace_id: str = "", **payload: Any) -> TraceRecord:
        return self.record(EVENT_SESSION, name, trace_id, **payload)

    def loop_step(self, name: str, trace_id: str = "", **payload: Any) -> TraceRecord:
        return self.record(EVENT_LOOP_STEP, name, trace_id, **payload)

    def llm_call(self, name: str = "chat.completions", trace_id: str = "", **payload: Any) -> TraceRecord:
        return self.record(EVENT_LLM_CALL, name, trace_id, **payload)

    def tool_call(self, name: str, trace_id: str = "", **payload: Any) -> TraceRecord:
        """记录一次工具调用：入参、输出、耗时、错误（spec 3.1 要求四要素齐全）。"""
        return self.record(EVENT_TOOL_CALL, name, trace_id, **payload)

    def error(self, name: str, trace_id: str = "", **payload: Any) -> TraceRecord:
        return self.record(EVENT_ERROR, name, trace_id, **payload)

    @contextmanager
    def timeit(self) -> Iterator[Dict[str, float]]:
        """测量代码块耗时：``with tracer.timeit() as t: ...; t["duration_ms"]``。"""
        box: Dict[str, float] = {}
        started = time.perf_counter()
        try:
            yield box
        finally:
            box["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def _write(self, entry: TraceRecord) -> None:
        try:
            if self._fmt == "jsonl":
                self._stream.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            else:
                self._stream.write(entry.to_line(self._max_text) + "\n")
            self._stream.flush()
        except (OSError, ValueError):
            # trace 写入失败绝不能影响主流程（spec 3.7：不产生未捕获崩溃）
            self._enabled = False


def null_tracer() -> Tracer:
    """静默 trace：记录仍进入内存，但不输出。供测试使用。"""
    return Tracer(enabled=False)
