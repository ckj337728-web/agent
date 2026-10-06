"""工具注册表。

对应任务 T008（注册 / 列举 / 按名获取 / 导出 LLM Schema），
参数校验与错误规范化对应任务 T012。

约束来自 spec.md 3.3：

- 工具以注册表形式集中注册，**新增工具无需改动主循环**；
- LLM 可见的工具描述由注册表自动生成并注入请求体；
- 调用不存在的工具、参数缺失、参数类型错误都必须被识别为
  **可恢复错误**并回灌 LLM。

设计取舍：注册表是纯调度 + 错误规范化层，不实现任何业务工具，
也不决定"何时调用"（那是 LLM 的职责，见 spec 3.0 分层职责表）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .base import BaseTool, ToolContext, ToolErrorCode, ToolResult, validate_arguments


class ToolRegistry:
    """工具注册表。"""

    def __init__(self, tools: Optional[Iterable[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or ():
            self.register(tool)

    # -- 注册与查询 -------------------------------------------------------- #

    def register(self, tool: BaseTool) -> BaseTool:
        """注册一个工具；同名重复注册会报错，避免静默覆盖。"""
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError(f"工具必须有非空的 name：{tool!r}")
        if name in self._tools:
            raise ValueError(
                f"工具名 {name!r} 已被 {type(self._tools[name]).__name__} 注册，不能重复注册"
            )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[BaseTool]:
        """按名获取；不存在返回 None（调用方需转成可恢复错误，见 spec 6.2 E1）。"""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """按注册顺序列出全部工具。"""
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # -- 导出给 LLM -------------------------------------------------------- #

    def to_llm_schemas(self) -> List[Dict[str, Any]]:
        """导出 LLM 可调用的工具定义列表（spec 3.3）。"""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    # -- 调度与错误规范化 -------------------------------------------------- #

    def execute(
        self, name: str, arguments: Optional[Mapping[str, Any]], context: Optional[ToolContext] = None
    ) -> ToolResult:
        """执行一次工具调用，**永不抛出**未捕获异常。

        四种可恢复错误都在此收敛：
        1. 未知工具（spec 6.2 E1）
        2. 参数缺失（spec 6.2 E2）
        3. 参数类型错误（spec 6.2 E2）
        4. 工具内部异常（spec 6.2 E3）

        实现说明：这里先做一次校验（快速失败，并给出与工具相同的错误信息），
        各工具的 ``run`` 内部还会再校验一次。重复校验是有意的——它让
        工具可以被独立调用（例如单测直接调 ``tool.run``）而仍然安全，
        代价只是一次廉价的字典检查。
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(
                ToolErrorCode.UNKNOWN_TOOL,
                f"工具 {name!r} 不存在。可用工具：{', '.join(self.names()) or '无'}",
            )

        checked = validate_arguments(tool.parameters, arguments or {})
        if not checked.ok:
            return checked

        active_context = context or ToolContext()
        try:
            result = tool.run(checked.data, active_context)
        except Exception as exc:  # noqa: BLE001 - 工具异常必须归一化，不能冒泡
            return ToolResult.failure(
                ToolErrorCode.TOOL_ERROR,
                f"工具 {name} 执行失败：{type(exc).__name__}: {exc}",
            )

        if not isinstance(result, ToolResult):
            return ToolResult.failure(
                ToolErrorCode.TOOL_ERROR,
                f"工具 {name} 返回值类型非法：{type(result).__name__}",
            )
        return result


# --------------------------------------------------------------------------- #
# 默认注册表
# --------------------------------------------------------------------------- #


def build_default_registry() -> ToolRegistry:
    """构建包含内置工具（calculator / search / todo）的注册表。

    ``todo`` 的存储实例在此创建并共享，保证同一 session 的多次工具调用
    看到同一份状态（spec 3.6）。
    """
    from .calculator import CalculatorTool
    from .search import SearchTool
    from .todo import TodoStore, TodoTool

    store = TodoStore()
    return ToolRegistry([CalculatorTool(), SearchTool(), TodoTool(store)])


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def default_registry() -> ToolRegistry:
    """进程级默认注册表（懒加载单例），供入口与主循环复用。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY
