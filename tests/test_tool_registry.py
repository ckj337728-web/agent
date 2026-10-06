"""T008 / T012 / T037 / T038 验证：工具注册表、Schema 导出与错误规范化。

覆盖 spec 3.3（注册机制、Schema 自动生成、新增工具不改主循环）、
spec 6.2 E1（未知工具）、E2（缺参 / 类型错误）、E3（工具内部异常）。
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from tests.helpers import AgentTestCase  # noqa: F401
from agent.tools.base import (
    BaseTool,
    ToolContext,
    ToolErrorCode,
    ToolResult,
    validate_arguments,
)
from agent.tools.registry import (
    ToolRegistry,
    build_default_registry,
    default_registry,
)


# --------------------------------------------------------------------------- #
# 测试用工具：验证"新增工具无需改动主循环"
# --------------------------------------------------------------------------- #


class EchoTool(BaseTool):
    name = "echo"
    description = "回显传入的文本，用于测试注册机制。"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要回显的文本"},
            "times": {
                "type": "integer",
                "description": "重复次数",
                "minimum": 1,
                "maximum": 5,
                "default": 1,
            },
        },
        "required": ["text"],
    }

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        text = arguments["text"] * arguments.get("times", 1)
        return ToolResult.success(content=text, data={"echo": text})


class ExplodingTool(BaseTool):
    name = "explode"
    description = "总是抛出异常，用于测试工具内部异常的归一化。"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        raise RuntimeError("内部炸了")


class BadReturnTool(BaseTool):
    name = "bad_return"
    description = "返回非 ToolResult，用于测试返回值校验。"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> Any:
        return "不是 ToolResult"


class TestRegistration(unittest.TestCase):
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        self.assertIsNotNone(registry.get("echo"))
        self.assertIn("echo", registry)
        self.assertEqual(len(registry), 1)

    def test_subclass_without_name_is_rejected(self):
        """契约在类定义时即校验，避免注册进注册表后才发现缺字段。"""
        with self.assertRaises(TypeError) as ctx:

            class NamelessTool(BaseTool):
                description = "无名字"

            _ = NamelessTool
        self.assertIn("name", str(ctx.exception))

    def test_subclass_without_description_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class NoDescriptionTool(BaseTool):
                name = "nodesc"

            _ = NoDescriptionTool
        self.assertIn("description", str(ctx.exception))

    def test_duplicate_name_rejected(self):
        registry = ToolRegistry([EchoTool()])
        with self.assertRaises(ValueError) as ctx:
            registry.register(EchoTool())
        self.assertIn("echo", str(ctx.exception))

    def test_get_unknown_returns_none(self):
        self.assertIsNone(ToolRegistry().get("nope"))

    def test_list_tools_and_names_keep_registration_order(self):
        registry = ToolRegistry([EchoTool(), ExplodingTool()])
        self.assertEqual(registry.names(), ["echo", "explode"])
        self.assertEqual([t.name for t in registry.list_tools()], ["echo", "explode"])

    def test_no_tools_has_empty_schemas(self):
        self.assertEqual(ToolRegistry().to_llm_schemas(), [])


class TestSchemaExport(unittest.TestCase):
    """spec 3.3：LLM 可见的工具描述由注册表自动生成。"""

    def test_schema_shape_for_each_tool(self):
        schemas = ToolRegistry([EchoTool()]).to_llm_schemas()
        self.assertEqual(len(schemas), 1)
        schema = schemas[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")
        self.assertTrue(schema["function"]["description"])
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["text"]["type"], "string"
        )
        self.assertEqual(schema["function"]["parameters"]["required"], ["text"])

    def test_default_registry_exposes_three_tools(self):
        """spec C-4：至少 3 个工具，覆盖 calculator / search / todo。"""
        registry = build_default_registry()
        self.assertGreaterEqual(len(registry), 3)
        self.assertEqual(set(registry.names()), {"calculator", "search", "todo"})

    def test_default_registry_schemas_are_all_valid(self):
        for schema in build_default_registry().to_llm_schemas():
            with self.subTest(tool=schema["function"]["name"]):
                function = schema["function"]
                self.assertTrue(function["name"])
                self.assertTrue(function["description"])
                parameters = function["parameters"]
                self.assertEqual(parameters["type"], "object")
                self.assertIn("properties", parameters)
                for key in parameters.get("required", []):
                    self.assertIn(key, parameters["properties"])
                    self.assertIn("description", parameters["properties"][key])

    def test_new_tool_needs_no_core_change(self):
        """spec 3.3：新增工具只需注册，主循环无需改动。"""
        registry = build_default_registry()
        before = len(registry.to_llm_schemas())
        registry.register(EchoTool())
        self.assertEqual(len(registry.to_llm_schemas()), before + 1)
        result = registry.execute("echo", {"text": "hi"})
        self.assertTrue(result.ok)

    def test_default_registry_is_singleton(self):
        self.assertIs(default_registry(), default_registry())


class TestExecuteErrorNormalization(unittest.TestCase):
    """spec 6.2 E1/E2/E3：全部收敛为结构化错误，不抛异常。"""

    def setUp(self):
        self.registry = ToolRegistry([EchoTool(), ExplodingTool(), BadReturnTool()])

    def test_unknown_tool(self):
        """spec 6.2 E1"""
        result = self.registry.execute("no_such_tool", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.UNKNOWN_TOOL.value)
        self.assertIn("no_such_tool", result.error_message)
        self.assertIn("echo", result.error_message)  # 提示可用工具，便于 LLM 修正

    def test_missing_required_parameter(self):
        """spec 6.2 E2"""
        result = self.registry.execute("echo", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)
        self.assertIn("text", result.error_message)

    def test_none_arguments_treated_as_missing(self):
        result = self.registry.execute("echo", None)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_explicit_none_value_treated_as_missing(self):
        result = self.registry.execute("echo", {"text": None})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_wrong_parameter_type(self):
        """spec 6.2 E2"""
        result = self.registry.execute("echo", {"text": 123})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)
        self.assertIn("string", result.error_message)      # 说明期望类型
        self.assertIn("integer", result.error_message)     # 说明实际类型

    def test_optional_parameter_wrong_type(self):
        result = self.registry.execute("echo", {"text": "a", "times": "三次"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_out_of_range_parameter(self):
        result = self.registry.execute("echo", {"text": "a", "times": 99})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_default_value_is_applied(self):
        result = self.registry.execute("echo", {"text": "ab"})
        self.assertTrue(result.ok)
        self.assertEqual(result.data["echo"], "ab")  # times 默认 1

    def test_arguments_not_an_object(self):
        result = self.registry.execute("echo", ["text"])
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_unknown_extra_parameter_is_ignored(self):
        """宽松策略：LLM 多带字段不应让整轮失败。"""
        result = self.registry.execute("echo", {"text": "a", "unknown": 1})
        self.assertTrue(result.ok)

    def test_tool_internal_exception_normalized(self):
        """spec 6.2 E3"""
        result = self.registry.execute("explode", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.TOOL_ERROR.value)
        self.assertIn("RuntimeError", result.error_message)
        self.assertIn("内部炸了", result.error_message)

    def test_bad_return_value_normalized(self):
        result = self.registry.execute("bad_return", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.TOOL_ERROR.value)
        self.assertIn("str", result.error_message)

    def test_errors_render_as_llm_readable_text(self):
        """spec 3.1：工具结果（含错误）需要能回灌给 LLM。"""
        for name, arguments in [
            ("no_such_tool", {}),
            ("echo", {}),
            ("explode", {}),
        ]:
            with self.subTest(tool=name):
                result = self.registry.execute(name, arguments)
                text = result.to_llm_text()
                self.assertTrue(text.startswith("ERROR ["))
                self.assertIn("]", text)

    def test_execute_passes_context_to_tool(self):
        registry = build_default_registry()
        first = registry.execute(
            "todo", {"action": "add", "content": "x"}, ToolContext(session_id="s1")
        )
        self.assertTrue(first.ok)
        other = registry.execute(
            "todo", {"action": "list"}, ToolContext(session_id="s2")
        )
        self.assertEqual(other.data["count"], 0, "不同 session 不应看到彼此的待办")

    def test_default_context_used_when_omitted(self):
        registry = build_default_registry()
        registry.execute("todo", {"action": "add", "content": "无上下文"})
        listed = registry.execute("todo", {"action": "list"})
        self.assertEqual(listed.data["count"], 1)
        self.assertEqual(listed.data["items"][0]["content"], "无上下文")


class TestValidateArgumentsDirectly(unittest.TestCase):
    """T012：校验器可独立使用。"""

    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "name": {"type": "string", "minLength": 2, "maxLength": 5},
            "mode": {"type": "string", "enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "flag": {"type": "boolean", "default": False},
        },
        "required": ["count"],
    }

    def test_success_returns_cleaned_arguments_with_defaults(self):
        result = validate_arguments(self.schema, {"count": 3})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"count": 3, "flag": False})

    def test_number_accepts_int_and_float(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        self.assertTrue(validate_arguments(schema, {"x": 1}).ok)
        self.assertTrue(validate_arguments(schema, {"x": 1.5}).ok)

    def test_integer_rejects_float(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        self.assertFalse(validate_arguments(schema, {"x": 1.5}).ok)

    def test_boolean_is_not_integer(self):
        """Python 中 bool 是 int 子类，必须显式排除。"""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = validate_arguments(schema, {"x": True})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_union_type_accepts_any_member(self):
        schema = {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
        self.assertTrue(validate_arguments(schema, {"x": "a"}).ok)
        self.assertTrue(validate_arguments(schema, {"x": None}).ok)
        self.assertFalse(validate_arguments(schema, {"x": 1}).ok)

    def test_enum_violation(self):
        result = validate_arguments(self.schema, {"count": 1, "mode": "c"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_min_length_violation(self):
        result = validate_arguments(self.schema, {"count": 1, "name": "a"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_max_length_violation(self):
        result = validate_arguments(self.schema, {"count": 1, "name": "abcdef"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_array_item_type_violation_reports_index(self):
        result = validate_arguments(self.schema, {"count": 1, "tags": ["ok", 5]})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)
        self.assertIn("tags[1]", result.error_message)

    def test_array_accepted(self):
        result = validate_arguments(self.schema, {"count": 1, "tags": ["a", "b"]})
        self.assertTrue(result.ok)

    def test_missing_required_reports_description_hint(self):
        schema = {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "检索词"}},
            "required": ["q"],
        }
        result = validate_arguments(schema, {})
        self.assertFalse(result.ok)
        self.assertIn("检索词", result.error_message)

    def test_empty_schema_accepts_anything_object(self):
        schema = {"type": "object", "properties": {}}
        self.assertTrue(validate_arguments(schema, {}).ok)
        self.assertTrue(validate_arguments(schema, {"whatever": 1}).ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
