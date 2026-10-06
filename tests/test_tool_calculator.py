"""T009 验证：calculator 工具。

覆盖 spec 3.2（数学表达式求值、非法表达式、除零）与 6.2 E3。
"""

from __future__ import annotations

import math
import unittest

from tests.helpers import AgentTestCase  # noqa: F401  (导入即完成 sys.path 设置)
from agent.tools.base import ToolContext, ToolErrorCode
from agent.tools.calculator import CalculatorTool, evaluate_expression


class TestEvaluateExpression(unittest.TestCase):
    def test_integer_arithmetic(self):
        for expression, expected in [
            ("1+1", 2),
            ("10-3", 7),
            ("6*7", 42),
            ("9/3", 3),
            ("7//2", 3),
            ("7%3", 1),
            ("2**10", 1024),
            ("(1+2)*3", 9),
            ("-5+3", -2),
            ("+5", 5),
            ("2**3**2", 512),  # 右结合
        ]:
            with self.subTest(expression=expression):
                value, error = evaluate_expression(expression)
                self.assertIsNone(error)
                self.assertEqual(value, expected)

    def test_float_result(self):
        value, error = evaluate_expression("1/4")
        self.assertIsNone(error)
        self.assertAlmostEqual(value, 0.25)

    def test_integral_float_is_normalized_to_int(self):
        value, error = evaluate_expression("4/2")
        self.assertIsNone(error)
        self.assertEqual(value, 2)
        self.assertIsInstance(value, int)

    def test_whitespace_tolerated(self):
        value, error = evaluate_expression("   1 +   2 * 3  ")
        self.assertIsNone(error)
        self.assertEqual(value, 7)

    def test_constants(self):
        value, _ = evaluate_expression("pi")
        self.assertAlmostEqual(value, math.pi)
        value, _ = evaluate_expression("e")
        self.assertAlmostEqual(value, math.e)

    def test_whitelisted_functions(self):
        cases = {
            "abs(-3)": 3,
            "round(2.6)": 3,
            "round(2.345, 2)": 2.35,
            "sqrt(16)": 4,
            "floor(2.9)": 2,
            "ceil(2.1)": 3,
            "log(1)": 0,
            "log(8, 2)": 3,
            "log10(100)": 2,
            "exp(0)": 1,
            "sin(0)": 0,
            "cos(0)": 1,
            "tan(0)": 0,
            "min(3,1,2)": 1,
            "max(3,1,2)": 3,
            "pow(2,5)": 32,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                value, error = evaluate_expression(expression)
                self.assertIsNone(error)
                self.assertAlmostEqual(value, expected)

    def test_complex_combined_expression(self):
        value, error = evaluate_expression("sqrt(16) + 2**3 * (10-7) - abs(-1)")
        self.assertIsNone(error)
        self.assertEqual(value, 27)


class TestInvalidExpressions(unittest.TestCase):
    """spec 6.2 E3：非法表达式返回可恢复错误，不抛异常。"""

    def _assert_failure(self, expression, expected_code=None):
        value, error = evaluate_expression(expression)
        self.assertIsNone(value, msg=f"{expression!r} 不应求出值")
        self.assertIsNotNone(error, msg=f"{expression!r} 应返回错误")
        self.assertFalse(error.ok)
        if expected_code:
            self.assertEqual(error.error_code, expected_code.value)

    def test_syntax_error(self):
        self._assert_failure("1+", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("(1+2", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("1 2", ToolErrorCode.INVALID_EXPRESSION)

    def test_empty_expression(self):
        self._assert_failure("", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("   ", ToolErrorCode.INVALID_EXPRESSION)

    def test_division_by_zero(self):
        self._assert_failure("1/0", ToolErrorCode.DIVISION_BY_ZERO)
        self._assert_failure("10//0", ToolErrorCode.DIVISION_BY_ZERO)
        self._assert_failure("5%0", ToolErrorCode.DIVISION_BY_ZERO)

    def test_statements_are_rejected(self):
        self._assert_failure("import os", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("x = 1", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("1; 2", ToolErrorCode.INVALID_EXPRESSION)

    def test_unknown_name_rejected(self):
        self._assert_failure("foo + 1", ToolErrorCode.INVALID_EXPRESSION)

    def test_unknown_function_rejected(self):
        self._assert_failure("eval('1+1')", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("open('x')", ToolErrorCode.INVALID_EXPRESSION)

    def test_attribute_access_rejected(self):
        self._assert_failure("(1).__class__", ToolErrorCode.INVALID_EXPRESSION)

    def test_keyword_arguments_rejected(self):
        self._assert_failure("round(2.5, ndigits=1)", ToolErrorCode.INVALID_EXPRESSION)

    def test_wrong_arity_rejected(self):
        self._assert_failure("sqrt(1, 2)", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("pow(2)", ToolErrorCode.INVALID_EXPRESSION)

    def test_huge_exponent_rejected(self):
        """资源保护：拒绝 9**9**9 这类表达式。"""
        self._assert_failure("9**9999", ToolErrorCode.INVALID_EXPRESSION)

    def test_string_literal_rejected(self):
        self._assert_failure("'abc'", ToolErrorCode.INVALID_EXPRESSION)

    def test_math_domain_error(self):
        self._assert_failure("sqrt(-1)", ToolErrorCode.INVALID_EXPRESSION)
        self._assert_failure("log(0)", ToolErrorCode.INVALID_EXPRESSION)

    def test_non_string_input_rejected(self):
        value, error = evaluate_expression(123)  # type: ignore[arg-type]
        self.assertIsNone(value)
        self.assertEqual(error.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_error_content_is_llm_readable(self):
        _value, error = evaluate_expression("1/0")
        self.assertIn("ERROR", error.to_llm_text())
        self.assertIn("division_by_zero", error.to_llm_text())


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.tool = CalculatorTool()
        self.context = ToolContext(session_id="s1")

    def test_contract_fields(self):
        """spec 3.3：name / description / parameters 三要素齐备。"""
        self.assertEqual(self.tool.name, "calculator")
        self.assertTrue(self.tool.description)
        self.assertEqual(self.tool.parameters["type"], "object")
        self.assertIn("expression", self.tool.parameters["properties"])
        self.assertEqual(self.tool.parameters["required"], ["expression"])

    def test_run_success(self):
        result = self.tool.run({"expression": "12*12"}, self.context)
        self.assertTrue(result.ok)
        self.assertIn("144", result.content)
        self.assertEqual(result.data["result"], 144)

    def test_run_formats_large_integer_with_separator(self):
        result = self.tool.run({"expression": "1000*1000"}, self.context)
        self.assertIn("1,000,000", result.content)

    def test_run_missing_expression(self):
        result = self.tool.run({}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_run_wrong_type(self):
        result = self.tool.run({"expression": 42}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_run_reports_invalid_expression(self):
        result = self.tool.run({"expression": "1/0"}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.error_code, ToolErrorCode.DIVISION_BY_ZERO.value
        )

    def test_llm_schema_shape(self):
        schema = self.tool.to_llm_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        self.assertIn("description", schema["function"])
        self.assertIn("parameters", schema["function"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
