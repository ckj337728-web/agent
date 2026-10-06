"""``todo`` 工具：会话级待办管理。

对应任务 T011，选自 spec 3.2 的 ``read_docs / todo / weather`` 三选一。

选择 ``todo`` 的原因：它是唯一具备**真实状态**语义的选项——待办写入后
必须能在同一 session 的后续对话中被召回，从而直接验证
spec 3.6 / 6.1 N8 的"两个窗口互不影响"。

状态隔离方式：待办以 ``session_id`` 为键存放在一个**普通 dict** 中；
工具自身不持有任何跨会话状态，会话标识由 :class:`~agent.tools.base.ToolContext`
注入，因此同一进程内不同窗口天然隔离。

关于与 Stage 4 的衔接（T017 / T019）：当 ``ToolContext.state`` 提供了
实现可变映射语义的容器时（例如 ``Session.state``），待办会直接写入其中，
键形如 ``"<session_id>:todos"``。这样做的好处是：

- 待办变成**可 JSON 序列化**的普通字典列表，可直接随 session 落盘，
  满足"随时回到旧窗口继续聊"的要求；
- context 层能把同一份状态渲染成置顶的 system 块（见
  :mod:`agent.context`），从而在历史被压缩后仍能召回早期待办。

未提供 ``state`` 时（例如单元测试直接构造工具），退化为工具自持字典，
两种模式的读写行为完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .base import BaseTool, ToolContext, ToolErrorCode, ToolResult, validate_arguments

STATUS_PENDING = "pending"
STATUS_DONE = "done"

#: session 状态中的待办键名
TODOS_STATE_KEY = "todos"


@dataclass
class TodoItem:
    """一条待办（对外视图；内部以普通字典存储以便序列化）。"""

    id: int
    content: str
    status: str = STATUS_PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "content": self.content, "status": self.status}


def _to_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    """把内部存储项统一成普通字典。"""
    result: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            result.append(dict(item))
        elif isinstance(item, TodoItem):
            result.append(item.to_dict())
        else:  # pragma: no cover - 防御性分支
            continue
    return result


class TodoStore:
    """按 session 隔离的待办存储。

    刻意使用普通 dict + ``get_or_create``：读取操作不会凭空创建会话，
    且不同 session 的列表对象彼此独立，杜绝串扰。

    ``state`` 为可选的可变映射；提供时所有读写都落在它上面
    （因此会随 session 一起持久化），否则落在工具自持的内存字典上。
    """

    def __init__(self, state: Optional[MutableMapping[str, Any]] = None) -> None:
        self._memory: Dict[str, List[Dict[str, Any]]] = {}
        self._state: Optional[MutableMapping[str, Any]] = state

    # -- 存储位置 ---------------------------------------------------------- #

    def bind(self, state: Optional[MutableMapping[str, Any]]) -> "TodoStore":
        """绑定状态容器（绑定同一个 state 的多个 TodoStore 视图一致）。"""
        self._state = state
        return self

    @property
    def state(self) -> Optional[MutableMapping[str, Any]]:
        return self._state

    @staticmethod
    def _state_key(session_id: str) -> str:
        """状态容器中的键：形如 ``"<session_id>:todos"``。"""
        return f"{session_id}:{TODOS_STATE_KEY}"

    def _get(self, session_id: str) -> Optional[List[Any]]:
        if self._state is None:
            return self._memory.get(session_id)
        return self._state.get(self._state_key(session_id))  # type: ignore[return-value]

    def _get_or_create(self, session_id: str) -> List[Any]:
        items = self._get(session_id)
        if items is None:
            items = []
            if self._state is None:
                self._memory[session_id] = items
            else:
                self._state[self._state_key(session_id)] = items
        return items

    def _next_id(self, items: List[Any]) -> int:
        return max((int(i.get("id", 0)) for i in _to_dicts(items)), default=0) + 1

    # -- 与既有 API 兼容的会话级操作 --------------------------------------- #

    def get_or_create(self, session_id: str) -> List[TodoItem]:
        return [TodoItem(**item) for item in _to_dicts(self._get_or_create(session_id))]

    def peek(self, session_id: str) -> List[TodoItem]:
        """只读访问：不存在时不创建空会话。"""
        return [TodoItem(**item) for item in _to_dicts(self._get(session_id) or [])]

    def session_ids(self) -> List[str]:
        if self._state is None:
            return sorted(self._memory)
        suffix = f":{TODOS_STATE_KEY}"
        return sorted(
            key[: -len(suffix)]
            for key in self._state
            if isinstance(key, str) and key.endswith(suffix)
        )

    def add(self, session_id: str, content: str) -> TodoItem:
        items = self._get_or_create(session_id)
        item = TodoItem(id=self._next_id(items), content=content)
        items.append(item.to_dict())
        return item

    def list(self, session_id: str, include_done: bool = True) -> List[TodoItem]:
        items = self.peek(session_id)
        if include_done:
            return items
        return [item for item in items if item.status == STATUS_PENDING]

    def complete(self, session_id: str, todo_id: int) -> Optional[TodoItem]:
        for raw in self._get(session_id) or []:
            if int(raw.get("id", 0)) == todo_id:
                raw["status"] = STATUS_DONE
                return TodoItem(**dict(raw))
        return None

    def remove(self, session_id: str, todo_id: int) -> Optional[TodoItem]:
        items = self._get_or_create(session_id)
        for index, raw in enumerate(items):
            if int(raw.get("id", 0)) == todo_id:
                removed = items.pop(index)
                return TodoItem(**dict(removed))
        return None

    def clear(self, session_id: str) -> int:
        items = self._get_or_create(session_id)
        count = len(items)
        items.clear()
        return count


TODO_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": (
                "操作类型：add 新增待办；list 列出待办；"
                "complete 把待办标记为已完成；remove 删除待办。"
            ),
            "enum": ["add", "list", "complete", "remove"],
        },
        "content": {
            "type": "string",
            "description": "待办内容，仅 action=add 时必填。",
            "minLength": 1,
            "maxLength": 500,
        },
        "todo_id": {
            "type": "integer",
            "description": "待办编号，action=complete 或 remove 时必填。",
            "minimum": 1,
        },
        "include_done": {
            "type": "boolean",
            "description": "action=list 时是否包含已完成项，默认 true。",
            "default": True,
        },
    },
    "required": ["action"],
}


class TodoTool(BaseTool):
    """会话级待办工具。

    ``store`` 由外部注入（``agent.tools.default_registry`` 使用全局共享实例），
    保证同一 session 的不同工具调用看到同一份状态。
    """

    name = "todo"
    description = (
        "管理当前会话的待办清单（按会话隔离）。当用户要求记待办、列出待办、"
        "标记完成或删除待办时使用。action=add 需提供 content；"
        "action=complete 或 remove 需提供 todo_id。"
    )
    parameters = TODO_PARAMETERS

    def __init__(self, store: Optional[TodoStore] = None) -> None:
        self._store = store or TodoStore()

    @property
    def store(self) -> TodoStore:
        return self._store

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        checked = validate_arguments(self.parameters, arguments)
        if not checked.ok:
            return checked

        action = checked.data["action"]
        session_id = context.ensure_session()

        # 若上下文提供了状态容器（Session.state），把待办写进它，
        # 这样状态会随 session 持久化，也能被 context 层渲染（T017/T019）。
        state = context.state
        if isinstance(state, MutableMapping):
            self._store.bind(state)

        if action == "add":
            return self._add(checked.data, session_id)
        if action == "list":
            return self._list(checked.data, session_id)
        if action in ("complete", "remove"):
            return self._mutate(action, checked.data, session_id)
        # enum 已在校验层拦住，这里只是兜底
        return ToolResult.failure(
            ToolErrorCode.INVALID_ARGUMENTS, f"不支持的操作：{action!r}"
        )

    # -- 各操作 ------------------------------------------------------------ #

    def _add(self, data: Dict[str, Any], session_id: str) -> ToolResult:
        content = data.get("content")
        if not content:
            return ToolResult.failure(
                ToolErrorCode.MISSING_PARAMETER,
                "action=add 时必须提供非空的 content（待办内容）",
            )
        item = self._store.add(session_id, content)
        return ToolResult.success(
            content=f"已记录待办 #{item.id}：{item.content}（当前共 "
            f"{len(self._store.peek(session_id))} 条待办）",
            data={"action": "add", "item": item.to_dict()},
        )

    def _list(self, data: Dict[str, Any], session_id: str) -> ToolResult:
        include_done = bool(data.get("include_done", True))
        items = self._store.list(session_id, include_done=include_done)
        if not items:
            return ToolResult.success(
                content="当前会话没有待办事项。",
                data={"action": "list", "count": 0, "items": []},
            )
        lines = [f"当前会话共 {len(items)} 条待办："]
        for item in items:
            mark = "x" if item.status == STATUS_DONE else " "
            lines.append(f"[{mark}] #{item.id} {item.content}")
        return ToolResult.success(
            content="\n".join(lines),
            data={
                "action": "list",
                "count": len(items),
                "items": [item.to_dict() for item in items],
            },
        )

    def _mutate(
        self, action: str, data: Dict[str, Any], session_id: str
    ) -> ToolResult:
        todo_id = data.get("todo_id")
        if todo_id is None:
            return ToolResult.failure(
                ToolErrorCode.MISSING_PARAMETER,
                f"action={action} 时必须提供 todo_id（待办编号）",
            )
        if action == "complete":
            item = self._store.complete(session_id, int(todo_id))
            if item is None:
                return self._not_found(session_id, int(todo_id), action)
            return ToolResult.success(
                content=f"已完成待办 #{item.id}：{item.content}",
                data={"action": action, "item": item.to_dict()},
            )

        item = self._store.remove(session_id, int(todo_id))
        if item is None:
            return self._not_found(session_id, int(todo_id), action)
        return ToolResult.success(
            content=f"已删除待办 #{item.id}：{item.content}",
            data={"action": action, "item": item.to_dict()},
        )

    def _not_found(self, session_id: str, todo_id: int, action: str) -> ToolResult:
        existing = [item.id for item in self._store.peek(session_id)]
        hint = (
            f"当前会话已有编号：{existing}"
            if existing
            else "当前会话还没有任何待办"
        )
        return ToolResult.failure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"待办 #{todo_id} 不存在，无法执行 {action}；{hint}",
        )


def session_todos(state: Mapping[str, Any], session_id: str) -> List[Dict[str, Any]]:
    """从 session 状态里读出待办（供 context 层与测试复用）。"""
    value = state.get(TodoStore._state_key(session_id))
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
