"""Session 模型、持久化与生命周期。

对应任务 T016（模型与生命周期）、T017（消息与状态持久化），
约束来自 spec.md：

- 3.6：必须支持多窗口隔离与随时续聊。"用户 A 在窗口 1…用户 A 在窗口 2…
  两个窗口是独立 session，context、todo 等 session 级状态互不影响；
  用户 A 可随时回到窗口 1 或窗口 2 继续对话，历史上下文正确恢复"；
- 6.2 E6：访问不存在的 session 需有明确处理（新建或明确报错）；
- 6.1 N8：双窗口场景下状态互不影响。

设计取舍：
- **Session 保存完整历史**，不做截断。截断是 context 层的事（T019–T023）——
  历史完整才能"随时续聊并正确恢复"，而 prompt 的尺寸由 context 层控制。
- **session 级状态集中存放**在 ``Session.state``，供工具（如 ``todo``）按会话隔离读写。
  这样切回旧窗口时状态、消息都能一起恢复。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence


class SessionError(Exception):
    """session 层的可解释错误（spec 6.2 E6）。"""


class SessionNotFoundError(SessionError):
    """要求必须已存在的 session，但未找到。"""


# 消息角色
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

VALID_ROLES = frozenset({ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL})

#: 工具结果在消息里保存的长度上限（防止一条结果把存储撑爆）。
#: 这是"保存"策略，与 context 层的"注入"截断（T023）是两回事。
MAX_STORED_TOOL_RESULT = 20000


@dataclass
class ToolCallRecord:
    """一次工具调用的记录（入参 + 结果），与 trace 的四要素对应。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    result_ok: bool = True
    result_text: str = ""
    error_code: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "result_ok": self.result_ok,
            "result_text": self.result_text,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCallRecord":
        return cls(
            name=str(data.get("name") or ""),
            arguments=dict(data.get("arguments") or {}),
            call_id=str(data.get("call_id") or ""),
            result_ok=bool(data.get("result_ok", True)),
            result_text=str(data.get("result_text") or ""),
            error_code=data.get("error_code"),
            duration_ms=float(data.get("duration_ms") or 0.0),
        )


@dataclass
class Message:
    """一条会话消息。"""

    role: str
    content: str = ""
    #: 仅 assistant：模型的思考过程（是否入 context 由 context 策略决定）
    thought: str = ""
    #: 仅 assistant：本消息发起的工具调用
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    #: 仅 tool：对应的工具调用 id
    tool_call_id: str = ""
    #: 仅 tool：工具名
    name: str = ""
    timestamp: float = field(default_factory=time.time)
    #: 轮次编号：一次用户输入及其引发的所有消息共享同一个 turn
    turn: int = 0

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise SessionError(
                f"非法消息角色 {self.role!r}；允许：{sorted(VALID_ROLES)}"
            )

    @property
    def is_from_user(self) -> bool:
        return self.role == ROLE_USER

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "thought": self.thought,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "turn": self.turn,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message":
        return cls(
            role=str(data.get("role") or ""),
            content=str(data.get("content") or ""),
            thought=str(data.get("thought") or ""),
            tool_calls=[
                ToolCallRecord.from_dict(item)
                for item in (data.get("tool_calls") or [])
            ],
            tool_call_id=str(data.get("tool_call_id") or ""),
            name=str(data.get("name") or ""),
            timestamp=float(data.get("timestamp") or time.time()),
            turn=int(data.get("turn") or 0),
        )


@dataclass
class Session:
    """一个独立会话（对应题面里的一个"窗口"）。"""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    #: session 级状态（如 todo 列表），按会话隔离（spec 3.6）
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    #: 当前轮次编号，每次 append_user_message 时递增
    turn_counter: int = 0

    # -- 消息读写 ---------------------------------------------------------- #

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        self.updated_at = time.time()
        return message

    def append_user_message(self, content: str) -> Message:
        """追加一条用户输入，并开启新的一轮（turn）。"""
        self.turn_counter += 1
        return self.append(Message(role=ROLE_USER, content=content, turn=self.turn_counter))

    def append_assistant_message(
        self,
        content: str = "",
        thought: str = "",
        tool_calls: Optional[Sequence[ToolCallRecord]] = None,
    ) -> Message:
        return self.append(
            Message(
                role=ROLE_ASSISTANT,
                content=content,
                thought=thought,
                tool_calls=list(tool_calls or []),
                turn=self.turn_counter,
            )
        )

    def append_tool_message(
        self, name: str, content: str, tool_call_id: str = ""
    ) -> Message:
        return self.append(
            Message(
                role=ROLE_TOOL,
                content=content,
                name=name,
                tool_call_id=tool_call_id,
                turn=self.turn_counter,
            )
        )

    def user_messages(self) -> List[Message]:
        return [m for m in self.messages if m.role == ROLE_USER]

    def turn_count(self) -> int:
        """已完成/进行中的用户轮次数。"""
        return len(self.user_messages())

    def current_turn(self) -> int:
        return self.turn_counter

    def last_assistant_message(self) -> Optional[Message]:
        for message in reversed(self.messages):
            if message.role == ROLE_ASSISTANT:
                return message
        return None

    def reset(self) -> None:
        self.messages.clear()
        self.state.clear()
        self.turn_counter = 0
        self.updated_at = time.time()

    def __len__(self) -> int:
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_counter": self.turn_counter,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Session":
        session_id = str(data.get("session_id") or "")
        if not session_id:
            raise SessionError("持久化数据缺少 session_id")
        return cls(
            session_id=session_id,
            messages=[Message.from_dict(item) for item in (data.get("messages") or [])],
            state=dict(data.get("state") or {}),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            turn_counter=int(data.get("turn_counter") or 0),
        )


# --------------------------------------------------------------------------- #
# 存储
# --------------------------------------------------------------------------- #


class SessionStore:
    """session 存储。

    传 ``path`` 则把 session 落盘为一整个 JSON 文件（可跨进程恢复历史），
    否则纯内存。两种模式接口一致，测试与运行都复用同一套行为。
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._sessions: Dict[str, Session] = {}
        self._path = path
        if path:
            self._load()

    # -- 生命周期 ---------------------------------------------------------- #

    def create(self, session_id: Optional[str] = None) -> Session:
        """新建 session；不指定 id 时自动生成。"""
        sid = session_id or self.new_session_id()
        if sid in self._sessions:
            raise SessionError(f"session {sid!r} 已存在")
        session = Session(session_id=sid)
        self._sessions[sid] = session
        self._flush()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """获取 session；不存在返回 None。"""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        """获取 session，不存在则新建（spec 6.2 E6 的"新建"策略）。"""
        session = self._sessions.get(session_id)
        if session is None:
            session = self.create(session_id)
        return session

    def require(self, session_id: str) -> Session:
        """要求 session 必须存在，否则明确报错（spec 6.2 E6 的"报错"策略）。"""
        session = self._sessions.get(session_id)
        if session is None:
            known = ", ".join(self.session_ids()) or "（当前没有任何 session）"
            raise SessionNotFoundError(
                f"session {session_id!r} 不存在。已有 session：{known}"
            )
        return session

    def delete(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None)
        if removed is not None:
            self._flush()
            return True
        return False

    def session_ids(self) -> List[str]:
        return sorted(self._sessions)

    def sessions(self) -> List[Session]:
        return [self._sessions[sid] for sid in self.session_ids()]

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions

    def __iter__(self) -> Iterator[Session]:
        return iter(self.sessions())

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    # -- 持久化 ------------------------------------------------------------ #

    def path(self) -> Optional[str]:
        return self._path

    def save(self) -> None:
        """显式落盘（未配置 path 时为空操作）。"""
        self._flush()

    def _flush(self) -> None:
        if not self._path:
            return
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "sessions": [session.to_dict() for session in self.sessions()],
        }
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        # 先写临时文件再替换，避免中途失败留下半个文件
        temp_path = f"{self._path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self._path)

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise SessionError(
                    f"session 文件 {self._path!r} 不是合法 JSON：{exc}"
                ) from exc
        for item in payload.get("sessions") or []:
            session = Session.from_dict(item)
            self._sessions[session.session_id] = session
