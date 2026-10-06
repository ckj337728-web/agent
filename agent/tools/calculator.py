"""``calculator`` 工具：数学表达式求值。

对应任务 T009，约束来自 spec.md 3.2：
"数学表达式求值，需处理非法表达式、除零等异常"。

实现要点：
- 用标准库 ``ast`` 解析表达式树后自行求值，**不使用 eval**，
  因此不存在代码注入面；
- 非法语法、不支持的语法节点、除零、未知函数都返回结构化错误结果
  （spec 6.2 E3），不抛出未捕获异常。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Optional, Tuple

from .base import BaseTool, ToolContext, ToolErrorCode, ToolResult, validate_arguments

# 允许的二元运算符
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许调用的函数白名单（含所需参数个数区间）
_FUNCTIONS = {
    "abs": (abs, 1, 1),
    "round": (round, 1, 2),
    "sqrt": (math.sqrt, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
    "log": (math.log, 1, 2),
    "log10": (math.log10, 1, 1),
    "exp": (math.exp, 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "min": (min, 1, 8),
    "max": (max, 1, 8),
    "pow": (pow, 2, 2),
}

# 允许的常量
_CONSTANTS = {"pi": math.pi, "e": math.e}

# 幂运算的规模上限，避免 9**9**9 之类的表达式耗尽资源
_MAX_POWER_EXPONENT = 1000
_MAX_ABSOLUTE_RESULT = 1e100


def evaluate_expression(expression: str) -> Tuple[Optional[float], Optional[ToolResult]]:
    """求值表达式。

    成功返回 ``(数值, None)``；失败返回 ``(None, 结构化错误)``。
    """
    if not isinstance(expression, str):
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_TYPE, "参数 'expression' 必须是字符串"
        )

    text = expression.strip()
    if not text:
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION, "表达式为空，请提供如 '1+1' 的算式"
        )

    try:
        # mode="eval" 只接受单个表达式，天然拒绝语句、赋值、导入等
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION,
            f"表达式语法错误：{exc.msg}（位置 {exc.offset}）；表达式为 {text!r}",
        )

    try:
        value = _eval_node(tree.body)
    except _CalcError as exc:
        return None, ToolResult.failure(exc.code, exc.message)
    except (OverflowError, ValueError) as exc:
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION, f"无法计算该表达式：{exc}"
        )

    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION,
            f"表达式结果不是数字：{type(value).__name__}",
        )
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION, "表达式结果为非有限值（NaN 或 无穷）"
        )
    if abs(value) > _MAX_ABSOLUTE_RESULT:
        return None, ToolResult.failure(
            ToolErrorCode.INVALID_EXPRESSION, "表达式结果过大，已拒绝计算"
        )

    # 整数值的浮点结果还原为整数，避免 "1+1" 显示为 2.0
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value, None


class _CalcError(Exception):
    """内部求值错误，携带工具错误码。"""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"不支持的常量类型：{type(node.value).__name__}",
            )
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise _CalcError(
            ToolErrorCode.INVALID_EXPRESSION,
            f"不支持的名称 {node.id!r}；可用常量：{', '.join(sorted(_CONSTANTS))}",
        )

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        func = _BINARY_OPS.get(op_type)
        if func is None:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"不支持的运算符：{op_type.__name__}",
            )
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POWER_EXPONENT:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"指数过大（上限 {_MAX_POWER_EXPONENT}），已拒绝计算",
            )
        try:
            return func(left, right)
        except ZeroDivisionError:
            raise _CalcError(
                ToolErrorCode.DIVISION_BY_ZERO,
                f"除数为零：{_format(left)} {_op_symbol(op_type)} {_format(right)}",
            ) from None
        except OverflowError:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION, "计算溢出，结果超出可表示范围"
            ) from None

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        func = _UNARY_OPS.get(op_type)
        if func is None:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"不支持的一元运算符：{op_type.__name__}",
            )
        return func(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION, "只支持调用具名函数，不支持属性访问"
            )
        if node.keywords:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION, "函数调用不支持关键字参数"
            )
        entry = _FUNCTIONS.get(node.func.id)
        if entry is None:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"不支持的函数 {node.func.id!r}；可用函数："
                f"{', '.join(sorted(_FUNCTIONS))}",
            )
        func, min_args, max_args = entry
        args = [_eval_node(arg) for arg in node.args]
        if not min_args <= len(args) <= max_args:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"函数 {node.func.id} 需要 {_arity_label(min_args, max_args)} 参数，"
                f"实际传入 {len(args)} 个",
            )
        try:
            return func(*args)
        except ZeroDivisionError:
            raise _CalcError(ToolErrorCode.DIVISION_BY_ZERO, "除数为零") from None
        except (ValueError, TypeError) as exc:
            raise _CalcError(
                ToolErrorCode.INVALID_EXPRESSION,
                f"函数 {node.func.id} 调用失败：{exc}",
            ) from None

    raise _CalcError(
        ToolErrorCode.INVALID_EXPRESSION,
        f"不支持的语法：{type(node).__name__}",
    )


def _arity_label(min_args: int, max_args: int) -> str:
    return str(min_args) if min_args == max_args else f"{min_args}~{max_args} 个"


def _op_symbol(op_type: type) -> str:
    return {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "//",
        ast.Mod: "%",
        ast.Pow: "**",
    }.get(op_type, "?")


def _format(value: Any) -> str:
    return repr(value)


def _format_result(value: Any) -> str:
    """给人看的数值格式：整数加千分位，浮点最多保留 10 位有效小数。"""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.10g}"


CALCULATOR_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "要计算的数学表达式，例如 '(1+2)*3'、'sqrt(16)+2**10'。"
                "支持 + - * / // % ** 、括号，以及 abs/round/sqrt/floor/ceil/"
                "log/log10/exp/sin/cos/tan/min/max/pow 与常量 pi、e。"
            ),
            "minLength": 1,
        }
    },
    "required": ["expression"],
}


class CalculatorTool(BaseTool):
    """数学表达式求值工具。"""

    name = "calculator"
    description = (
        "计算数学表达式并返回数值结果。当用户的问题涉及算术、百分比、幂、"
        "开方、对数或三角函数时使用。输入必须是单个数学表达式字符串，"
        "不要传入自然语言描述。"
    )
    parameters = CALCULATOR_PARAMETERS

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        checked = validate_arguments(self.parameters, arguments)
        if not checked.ok:
            return checked

        expression = checked.data["expression"]
        value, error = evaluate_expression(expression)
        if error is not None:
            return error
        return ToolResult.success(
            content=f"{expression.strip()} = {_format_result(value)}",
            data={"expression": expression.strip(), "result": value},
        )
