"""LLM HTTP 客户端。

对应任务 T005，约束来自 spec.md：

- C-3：必须调用**真实** LLM API（本模块是唯一发出真实网络请求的地方）；
- 3.0 / 4.3：本层只负责"调用 API、超时重试、错误归一化"，
  不负责解析语义，也不负责任何工具调度；
- 6.2 E4：超时 / 网络错误 / 限流必须归一化为统一错误类型，
  重试耗尽后抛出可解释错误，不向上泄漏裸堆栈。

可测试性（spec 5.0 / 4.3：测试须能离线运行）：
真实 HTTP 调用被收敛到一个可注入的 ``transport`` 上。生产使用
:func:`urllib_transport`；测试注入 fake transport 即可覆盖超时、
限流、5xx 等分支，无需真实网络与密钥。
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .config import LLMConfig
from .trace import Tracer, truncate_text

# HTTP 状态码中值得重试的类别
RETRIABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
DEFAULT_BACKOFF_SECONDS = 0.5

# 传输层签名：(url, headers, json_body_bytes, timeout_seconds) -> (status, parsed_body)
Transport = Callable[[str, Mapping[str, str], bytes, float], "tuple[int, Any]"]


# --------------------------------------------------------------------------- #
# 错误归一化
# --------------------------------------------------------------------------- #


class LLMError(Exception):
    """所有 LLM 调用失败的基类。上层只需捕获这一个类型（spec 6.2 E4）。"""

    retriable = False

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class LLMTimeoutError(LLMError):
    """请求超时。"""

    retriable = True


class LLMNetworkError(LLMError):
    """连接失败、DNS 失败、连接被重置等网络层错误。"""

    retriable = True


class LLMRateLimitError(LLMError):
    """被限流（HTTP 429）。"""

    retriable = True


class LLMServerError(LLMError):
    """服务端错误（HTTP 5xx）。"""

    retriable = True


class LLMAuthError(LLMError):
    """鉴权失败（HTTP 401/403）——重试无意义。"""

    retriable = False


class LLMBadRequestError(LLMError):
    """请求非法（HTTP 4xx，除 401/403/429）——重试无意义。"""

    retriable = False


class LLMResponseFormatError(LLMError):
    """响应不是预期的 JSON 结构。"""

    retriable = False


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChatMessage:
    """发给 LLM 的一条消息，直接映射为 API 的 message 对象。"""

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ChatResponse:
    """一次 LLM 调用的原始响应（尚未做语义解析）。"""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    duration_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class LLMClient:
    """OpenAI 兼容的 chat completions 客户端。

    参数：
        config:    来自环境变量的配置（见 :mod:`agent.config`）
        transport: 可注入的传输层，默认真实 HTTP
        tracer:    可选 trace 记录器
        sleep:     可注入的等待函数（测试中替换以避免真实退避等待）
    """

    def __init__(
        self,
        config: LLMConfig,
        transport: Optional[Transport] = None,
        tracer: Optional[Tracer] = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self._config = config
        self._transport = transport or urllib_transport
        self._tracer = tracer
        self._sleep = sleep
        self._backoff_seconds = backoff_seconds

    @property
    def config(self) -> LLMConfig:
        return self._config

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        trace_id: str = "",
        temperature: float = 0.0,
    ) -> ChatResponse:
        """调用 LLM；失败时抛出 :class:`LLMError` 子类。

        重试策略：仅对 ``retriable=True`` 的错误重试，最多
        ``config.max_retries`` 次（即最多 1 + max_retries 次尝试）。
        """
        body = self._build_body(messages, tools, temperature)
        url = self._config.chat_completions_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")

        attempts_allowed = self._config.max_retries + 1
        last_error: Optional[LLMError] = None

        for attempt in range(1, attempts_allowed + 1):
            started = time.perf_counter()
            try:
                status, payload = self._transport(
                    url, headers, encoded, self._config.timeout_seconds
                )
                self._raise_for_status(status, payload)
                response = self._parse_success(payload)
                response.attempts = attempt
                response.duration_ms = round((time.perf_counter() - started) * 1000, 2)
                self._trace_llm(url, body, response, trace_id)
                return response
            except LLMError as exc:
                last_error = exc
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                self._trace_failure(url, exc, attempt, attempts_allowed, elapsed_ms, trace_id)
                if not exc.retriable or attempt >= attempts_allowed:
                    raise
                self._sleep(self._backoff_seconds * attempt)

        # 理论上不可达：循环内要么 return 要么 raise
        raise last_error or LLMError("LLM 调用失败：未知原因")

    # -- 内部实现 ---------------------------------------------------------- #

    def _build_body(
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[Mapping[str, Any]]],
        temperature: float,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self._config.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": temperature,
        }
        if tools:
            # 工具定义来自注册表生成的 Schema（spec 3.3：LLM 基于 Schema 自主决策）
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        return body

    def _raise_for_status(self, status: int, payload: Any) -> None:
        if 200 <= status < 300:
            return
        detail = _extract_error_detail(payload)
        if status in (401, 403):
            raise LLMAuthError(
                f"LLM 鉴权失败（HTTP {status}）：{detail}。"
                "请检查 LLM_API_KEY 是否有效。",
                status=status,
            )
        if status == 429:
            raise LLMRateLimitError(
                f"LLM 限流（HTTP 429）：{detail}", status=status
            )
        if status in RETRIABLE_STATUS_CODES or status >= 500:
            raise LLMServerError(
                f"LLM 服务端错误（HTTP {status}）：{detail}", status=status
            )
        raise LLMBadRequestError(
            f"LLM 请求被拒绝（HTTP {status}）：{detail}", status=status
        )

    def _parse_success(self, payload: Any) -> ChatResponse:
        if not isinstance(payload, Mapping):
            raise LLMResponseFormatError("LLM 响应不是 JSON 对象")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseFormatError("LLM 响应缺少 choices 字段")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMResponseFormatError("LLM 响应的 choices[0] 不是对象")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise LLMResponseFormatError("LLM 响应的 choices[0].message 缺失")

        content = message.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise LLMResponseFormatError("LLM 响应的 tool_calls 不是数组")

        usage = payload.get("usage")
        return ChatResponse(
            content=content,
            tool_calls=[dict(tc) for tc in raw_tool_calls if isinstance(tc, Mapping)],
            finish_reason=str(first.get("finish_reason") or ""),
            model=str(payload.get("model") or self._config.model),
            usage=dict(usage) if isinstance(usage, Mapping) else {},
        )

    def _trace_llm(
        self, url: str, body: Mapping[str, Any], response: ChatResponse, trace_id: str
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.llm_call(
            "chat.completions",
            trace_id,
            url=url,
            model=body.get("model"),
            messages=len(body.get("messages", [])),
            tools=len(body.get("tools", []) or []),
            attempt=response.attempts,
            duration_ms=response.duration_ms,
            finish_reason=response.finish_reason,
            tool_calls=len(response.tool_calls),
            content=truncate_text(response.content, 200),
        )

    def _trace_failure(
        self,
        url: str,
        exc: LLMError,
        attempt: int,
        attempts_allowed: int,
        elapsed_ms: float,
        trace_id: str,
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.error(
            "llm_call_failed",
            trace_id,
            url=url,
            error_type=type(exc).__name__,
            error=str(exc),
            status=exc.status,
            attempt=f"{attempt}/{attempts_allowed}",
            retriable=exc.retriable,
            duration_ms=elapsed_ms,
        )


# --------------------------------------------------------------------------- #
# 真实传输层
# --------------------------------------------------------------------------- #


def urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> "tuple[int, Any]":
    """基于标准库 urllib 的真实 HTTP 传输；把网络异常归一化为 LLMError。"""
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return int(response.status), _decode_json(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return int(exc.code), _decode_json(raw)
    except socket.timeout as exc:
        raise LLMTimeoutError(f"LLM 请求超时（>{timeout}s）") from exc
    except urllib.error.URLError as exc:
        # urlopen 会把 socket.timeout 包装进 URLError.reason
        if isinstance(exc.reason, socket.timeout):
            raise LLMTimeoutError(f"LLM 请求超时（>{timeout}s）") from exc
        raise LLMNetworkError(f"LLM 网络错误：{exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMTimeoutError(f"LLM 请求超时（>{timeout}s）") from exc
    except OSError as exc:
        raise LLMNetworkError(f"LLM 网络错误：{exc}") from exc


def _decode_json(raw: str) -> Any:
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _extract_error_detail(payload: Any) -> str:
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message")
            if message:
                return str(message)
            return json.dumps(error, ensure_ascii=False)
        if isinstance(error, str):
            return error
        if "raw" in payload:
            return str(payload["raw"])
    return truncate_text(payload, 300) or "无错误详情"
