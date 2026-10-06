"""工具层：契约、注册表与内置工具。

对应 spec.md 第 3.2 / 3.3 节与任务 T007–T012。

模块划分：

- :mod:`agent.tools.base`       —— 统一工具契约 + 参数校验（T007、T012）
- :mod:`agent.tools.registry`   —— 工具注册表与错误规范化调度（T008、T012）
- :mod:`agent.tools.calculator` —— calculator 工具（T009）
- :mod:`agent.tools.search`     —— search 工具，mock 检索（T010）
- :mod:`agent.tools.todo`       —— todo 工具，会话级状态（T011）
"""

from .base import (
    BaseTool,
    ToolContext,
    ToolErrorCode,
    ToolResult,
    validate_arguments,
)
from .registry import ToolRegistry, build_default_registry, default_registry

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolErrorCode",
    "ToolResult",
    "ToolRegistry",
    "build_default_registry",
    "default_registry",
    "validate_arguments",
]
