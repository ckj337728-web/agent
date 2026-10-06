"""统一工具契约。

对应任务 T007，约束来自 spec.md：

- 3.3：每个工具必须包含 ``name``、``description``、``parameters``
  （JSON Schema 风格：type / properties / required）；
- 6.2 E1/E2：未知工具、参数缺失、参数类型错误都必须被识别为
  **可恢复错误**（返回结构化错误结果），而不是抛出未捕获异常。

本模块同时提供参数校验器（任务 T012），供注册表在调度前统一调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

# 参数 Schema 的取值类型（JSON Schema 子集）
SchemaType = Union[str, Sequence[str]]


class ToolErrorCode(str, Enum):
    """工具层错误码。错误以数据形式返回，不作为异常向上冒泡。"""

    UNKNOWN_TOOL = "unknown_tool"          # 调用了未注册的工具（spec 6.2 E1）
    MISSING_PARAMETER = "missing_parameter"  # 缺少 required 参数（spec 6.2 E2）
    INVALID_TYPE = "invalid_type"            # 参数类型不符（spec 6.2 E2）
    INVALID_ARGUMENTS = "invalid_arguments"  # 其他参数非法（枚举值、范围等）
    TOOL_ERROR = "tool_error"                # 工具内部执行失败
    TOOL_TIMEOUT = "tool_timeout"            # 工具执行超时（T032）
    INVALID_EXPRESSION = "invalid_expression"
    DIVISION_BY_ZERO = "division_by_zero"


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ``content`` 是给 LLM 看的文本；``data`` 是给程序用的结构化数据。
    失败时 ``ok=False``，并通过 ``error_code`` / ``error_message`` 描述原因。
    """

    ok: bool
    content: str = ""
    data: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, content: str, data: Any = None) -> "ToolResult":
        return cls(ok=True, content=content, data=data)

    @classmethod
    def failure(
        cls,
        code: ToolErrorCode,
        message: str,
        data: Any = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            content=f"ERROR [{code.value}]: {message}",
            data=data,
            error_code=code.value,
            error_message=message,
        )

    def to_llm_text(self) -> str:
        """回灌给 LLM 的文本形式（spec 3.1：工具结果需可回灌）。"""
        if not self.ok:
            return self.content or f"ERROR [{self.error_code}]: {self.error_message}"
        return self.content


@dataclass
class ToolContext:
    """工具执行上下文。

    工具本身不持有会话状态；需要"按会话隔离状态"的工具（如 ``todo``）
    通过此上下文取得 ``state`` 存储与 ``session_id``，从而在
    spec 3.6 / 6.1 N8 的双窗口场景下互不串扰。
    """

    session_id: str = "default"
    state: Any = None

    def ensure_session(self) -> str:
        return self.session_id or "default"


# 工具执行函数签名
ToolCallable = Callable[[Dict[str, Any], ToolContext], ToolResult]


class BaseTool:
    """工具基类。

    子类必须提供 ``name`` / ``description`` / ``parameters``，
    并实现 :meth:`run`。新增工具只需继承本类并注册到注册表，
    **无需修改主循环**（spec 3.3）。
    """

    #: 唯一名称，LLM 调用时使用的标识
    name: str = ""
    #: 自然语言描述，用于引导 LLM 何时调用
    description: str = ""
    #: JSON Schema 风格的参数定义
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", ""):
            raise TypeError(f"工具类 {cls.__name__} 必须定义非空的 name")
        if not getattr(cls, "description", ""):
            raise TypeError(f"工具类 {cls.__name__} 必须定义 description")

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具。异常会被注册表统一捕获为 ``TOOL_ERROR``。"""
        raise NotImplementedError

    def to_llm_schema(self) -> Dict[str, Any]:
        """导出为 LLM 工具调用所需的 Schema（spec 3.3）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------- #
# 参数校验（任务 T012）
# --------------------------------------------------------------------------- #


def validate_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> ToolResult:
    """按 Schema 校验参数。

    返回 ``ToolResult``：成功时携带 **补全默认值后** 的参数（放在 ``data``），
    失败时是结构化错误。这样调用方无需用异常处理参数问题，
    错误也能原样回灌给 LLM 让其自我修正（spec 3.3）。
    """
    if not isinstance(arguments, Mapping):
        return ToolResult.failure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"参数必须是 JSON 对象，收到 {type(arguments).__name__}",
        )

    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    cleaned: Dict[str, Any] = {}

    for key in required:
        if key not in arguments or arguments[key] is None:
            return ToolResult.failure(
                ToolErrorCode.MISSING_PARAMETER,
                f"缺少必填参数 {key!r}（{_describe_property(properties.get(key))}）",
            )

    for key, value in arguments.items():
        rule = properties.get(key)
        if rule is None:
            # 宽松策略：未知参数忽略而非报错，避免因 LLM 多带字段而整轮失败
            continue
        error = _check_value(key, value, rule)
        if error is not None:
            return ToolResult.failure(error[0], error[1])
        cleaned[key] = value

    # 补默认值与可空字段
    for key, rule in properties.items():
        if key in cleaned:
            continue
        if "default" in rule:
            cleaned[key] = rule["default"]
        elif key in required:
            return ToolResult.failure(
                ToolErrorCode.MISSING_PARAMETER,
                f"缺少必填参数 {key!r}（{_describe_property(rule)}）",
            )

    return ToolResult.success(content="", data=cleaned)


def _check_value(
    key: str, value: Any, rule: Mapping[str, Any]
) -> Optional[tuple]:
    """返回 ``(错误码, 错误信息)``；合法时返回 None。"""
    expected = rule.get("type")
    if expected is not None and not _matches_type(value, expected):
        return (
            ToolErrorCode.INVALID_TYPE,
            f"参数 {key!r} 类型应为 {_type_label(expected)}，"
            f"实际为 {_actual_type(value)}",
        )

    if "enum" in rule and value not in rule["enum"]:
        allowed = ", ".join(repr(v) for v in rule["enum"])
        return (
            ToolErrorCode.INVALID_ARGUMENTS,
            f"参数 {key!r} 只能取以下值之一：{allowed}；实际为 {value!r}",
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = rule.get("minimum")
        if minimum is not None and value < minimum:
            return (
                ToolErrorCode.INVALID_ARGUMENTS,
                f"参数 {key!r} 不能小于 {minimum}，实际为 {value}",
            )
        maximum = rule.get("maximum")
        if maximum is not None and value > maximum:
            return (
                ToolErrorCode.INVALID_ARGUMENTS,
                f"参数 {key!r} 不能大于 {maximum}，实际为 {value}",
            )

    if isinstance(value, str):
        min_length = rule.get("minLength")
        if min_length is not None and len(value) < min_length:
            return (
                ToolErrorCode.INVALID_ARGUMENTS,
                f"参数 {key!r} 长度不能少于 {min_length}，实际为 {len(value)}",
            )
        max_length = rule.get("maxLength")
        if max_length is not None and len(value) > max_length:
            return (
                ToolErrorCode.INVALID_ARGUMENTS,
                f"参数 {key!r} 长度不能超过 {max_length}，实际为 {len(value)}",
            )

    if isinstance(value, list) and isinstance(rule.get("items"), Mapping):
        for index, item in enumerate(value):
            error = _check_value(f"{key}[{index}]", item, rule["items"])
            if error is not None:
                return error

    return None


def _matches_type(value: Any, expected: SchemaType) -> bool:
    if isinstance(expected, (list, tuple)):
        return any(_matches_type(value, one) for one in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "null":
        return value is None
    return True  # 未知类型不做限制


def _type_label(expected: SchemaType) -> str:
    if isinstance(expected, (list, tuple)):
        return " 或 ".join(str(one) for one in expected)
    return str(expected)


def _actual_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _describe_property(rule: Optional[Mapping[str, Any]]) -> str:
    """为错误信息补充参数语义，帮助 LLM 自我修正。"""
    if not rule:
        return "无描述"
    parts: List[str] = []
    if rule.get("description"):
        parts.append(str(rule["description"]))
    if rule.get("type"):
        parts.append(f"类型 {_type_label(rule['type'])}")
    if rule.get("enum"):
        parts.append("可选值 " + "/".join(repr(v) for v in rule["enum"]))
    return "；".join(parts) if parts else "无描述"


# --------------------------------------------------------------------------- #
# 工具实现风格：函数 + 薄包装（便于单测直接调用纯函数）
# --------------------------------------------------------------------------- #


def tool_from_callable(
    tool_cls_name: str,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    func: ToolCallable,
) -> Type[BaseTool]:
    """由一个纯函数生成工具类。用于保持工具实现与契约解耦。"""

    def run(self: BaseTool, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        return func(arguments, context)

    return type(
        tool_cls_name,
        (BaseTool,),
        {
            "name": name,
            "description": description,
            "parameters": parameters,
            "run": run,
            "__doc__": description,
        },
    )
