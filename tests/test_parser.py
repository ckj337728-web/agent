"""T013 / T014 / T015 / T039 验证：输出协议与解析器。

覆盖 spec 3.4：

- 稳定提取思考过程 / 工具调用 / 最终答案；
- 解析失败（JSON 非法、字段缺失、格式异常）走回退路径且不崩溃；
- 支持一次响应中同时出现工具调用与最终答案的区分判定。
"""

from __future__ import annotations

import json
import unittest

from tests.helpers import AgentTestCase, tool_call  # noqa: F401
from agent.parser import (
    OUTPUT_PROTOCOL_INSTRUCTIONS,
    OutputKind,
    ParsedOutput,
    ParseSource,
    ToolCallRequest,
    parse_response,
)


class TestNativeToolCalls(unittest.TestCase):
    """路径 1：API 原生 tool_calls（首选，最可靠）。"""

    def test_single_native_call(self):
        result = parse_response(
            content="",
            native_tool_calls=[tool_call("calculator", {"expression": "1+1"})],
        )
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)
        self.assertEqual(result.source, ParseSource.NATIVE_TOOL_CALLS)
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0].name, "calculator")
        self.assertEqual(result.tool_calls[0].arguments, {"expression": "1+1"})
        self.assertTrue(result.tool_calls[0].arguments_valid)

    def test_multiple_native_calls_preserve_order(self):
        result = parse_response(
            content="",
            native_tool_calls=[
                tool_call("calculator", {"expression": "1+1"}, "c1"),
                tool_call("todo", {"action": "list"}, "c2"),
                tool_call("search", {"query": "x"}, "c3"),
            ],
        )
        self.assertEqual(
            [c.name for c in result.tool_calls], ["calculator", "todo", "search"]
        )
        self.assertEqual([c.call_id for c in result.tool_calls], ["c1", "c2", "c3"])

    def test_call_id_preserved(self):
        result = parse_response(
            native_tool_calls=[tool_call("calculator", {"expression": "1"}, "call_abc")]
        )
        self.assertEqual(result.tool_calls[0].call_id, "call_abc")

    def test_content_with_native_calls_becomes_thought(self):
        """原生调用 + 纯文本 content：按协议 content 是推理，不是答复。"""
        result = parse_response(
            content="我需要先算一下。",
            native_tool_calls=[tool_call("calculator", {"expression": "1+1"})],
        )
        self.assertEqual(result.thought, "我需要先算一下。")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)

    def test_thinking_tag_is_extracted_when_also_answering(self):
        """同一响应既有工具调用又有给用户的答复 → MIXED。"""
        result = parse_response(
            content="<thinking>先查一下</thinking>我这就帮你查。",
            native_tool_calls=[tool_call("search", {"query": "x"})],
        )
        self.assertEqual(result.kind, OutputKind.MIXED)
        self.assertEqual(result.thought, "先查一下")
        self.assertEqual(result.answer, "我这就帮你查。")
        self.assertTrue(result.has_tool_calls)
        self.assertTrue(result.has_answer)

    def test_arguments_already_object(self):
        result = parse_response(
            native_tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "todo", "arguments": {"action": "list"}},
                }
            ]
        )
        self.assertEqual(result.tool_calls[0].arguments, {"action": "list"})

    def test_empty_arguments_string_is_valid(self):
        result = parse_response(
            native_tool_calls=[
                {"id": "c1", "function": {"name": "todo", "arguments": ""}}
            ]
        )
        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertTrue(result.tool_calls[0].arguments_valid)


class TestNativeToolCallFallbacks(unittest.TestCase):
    """原生结构本身畸形时的降级处理。"""

    def test_malformed_arguments_json_is_flagged_not_raised(self):
        result = parse_response(
            native_tool_calls=[
                {"id": "c1", "function": {"name": "calculator", "arguments": '{"expr'}}
            ]
        )
        call = result.tool_calls[0]
        self.assertEqual(call.name, "calculator")
        self.assertFalse(call.arguments_valid)
        self.assertIn("不是合法 JSON", call.argument_error)
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)  # 仍交给主循环报参数错误

    def test_truncated_arguments_keep_raw_text(self):
        raw = '{"expression": "1+1'
        result = parse_response(
            native_tool_calls=[
                {"id": "c1", "function": {"name": "calculator", "arguments": raw}}
            ]
        )
        self.assertEqual(result.tool_calls[0].raw_arguments, raw)

    def test_arguments_not_an_object(self):
        result = parse_response(
            native_tool_calls=[
                {"id": "c1", "function": {"name": "calculator", "arguments": "[1,2]"}}
            ]
        )
        call = result.tool_calls[0]
        self.assertFalse(call.arguments_valid)
        self.assertIn("不是对象", call.argument_error)

    def test_entry_not_an_object_is_skipped_with_warning(self):
        result = parse_response(
            content="回退答案",
            native_tool_calls=[
                "垃圾数据",
                {"id": "c1", "function": {"name": "calculator", "arguments": "{}"}},
            ],
        )
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue(any("不是对象" in w for w in result.warnings))

    def test_missing_function_field_is_skipped(self):
        result = parse_response(
            content="回退答案", native_tool_calls=[{"id": "c1", "type": "function"}]
        )
        self.assertEqual(result.tool_calls, [])
        self.assertTrue(any("缺少 function" in w for w in result.warnings))
        self.assertEqual(result.kind, OutputKind.FINAL_ANSWER)
        self.assertEqual(result.answer, "回退答案")

    def test_missing_function_name_is_skipped(self):
        result = parse_response(
            content="x",
            native_tool_calls=[{"id": "c1", "function": {"arguments": "{}"}}],
        )
        self.assertEqual(result.tool_calls, [])
        self.assertTrue(any("缺少函数名" in w for w in result.warnings))


class TestProtocolJsonInText(unittest.TestCase):
    """路径 2：文本中的协议 JSON。"""

    def test_bare_json_with_answer(self):
        payload = json.dumps({"thought": "算一下", "answer": "等于 2"})
        result = parse_response(content=payload)
        self.assertEqual(result.kind, OutputKind.FINAL_ANSWER)
        self.assertEqual(result.source, ParseSource.JSON_TEXT)
        self.assertEqual(result.thought, "算一下")
        self.assertEqual(result.answer, "等于 2")
        self.assertFalse(result.degraded)

    def test_bare_json_with_tool_calls(self):
        payload = json.dumps(
            {
                "thought": "需要算数",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "2*3"}}],
            }
        )
        result = parse_response(content=payload)
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)
        self.assertEqual(result.thought, "需要算数")
        self.assertEqual(result.tool_calls[0].name, "calculator")
        self.assertEqual(result.tool_calls[0].arguments, {"expression": "2*3"})

    def test_fenced_json(self):
        payload = "```json\n" + json.dumps({"answer": "答案是 42"}) + "\n```"
        result = parse_response(content=payload)
        self.assertEqual(result.source, ParseSource.JSON_FENCED)
        self.assertEqual(result.answer, "答案是 42")

    def test_fenced_json_without_language_tag(self):
        payload = "```\n" + json.dumps({"answer": "ok"}) + "\n```"
        result = parse_response(content=payload)
        self.assertEqual(result.answer, "ok")

    def test_json_embedded_in_explanatory_text(self):
        payload = (
            "好的，我先调用工具。\n"
            + json.dumps({"tool_calls": [{"name": "todo", "arguments": {"action": "list"}}]})
            + "\n就这样。"
        )
        result = parse_response(content=payload)
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)
        self.assertEqual(result.tool_calls[0].name, "todo")
        self.assertTrue(any("混合文本" in w for w in result.warnings))

    def test_tool_calls_as_single_object(self):
        payload = json.dumps({"tool_calls": {"name": "todo", "arguments": {"action": "list"}}})
        result = parse_response(content=payload)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.kind, OutputKind.TOOL_CALLS)

    def test_openai_style_nested_call_in_json(self):
        payload = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "c9",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "agent"}),
                        },
                    }
                ]
            }
        )
        result = parse_response(content=payload)
        call = result.tool_calls[0]
        self.assertEqual(call.name, "search")
        self.assertEqual(call.arguments, {"query": "agent"})
        self.assertEqual(call.call_id, "c9")

    def test_parameters_alias_accepted(self):
        payload = json.dumps(
            {"tool_calls": [{"name": "todo", "parameters": {"action": "add"}}]}
        )
        result = parse_response(content=payload)
        self.assertEqual(result.tool_calls[0].arguments, {"action": "add"})

    def test_mixed_json_has_both_calls_and_answer(self):
        payload = json.dumps(
            {
                "thought": "先算再答",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}],
                "answer": "先帮你算，再总结。",
            }
        )
        result = parse_response(content=payload)
        self.assertEqual(result.kind, OutputKind.MIXED)
        self.assertTrue(result.has_tool_calls)
        self.assertTrue(result.has_answer)

    def test_alias_field_names(self):
        for key in ("answer", "final_answer", "response", "reply", "content"):
            with self.subTest(key=key):
                result = parse_response(content=json.dumps({key: "文本"}))
                self.assertEqual(result.answer, "文本")
        for key in ("thought", "thinking", "reasoning", "think"):
            with self.subTest(key=key):
                result = parse_response(content=json.dumps({key: "推理", "answer": "a"}))
                self.assertEqual(result.thought, "推理")

    def test_thought_without_answer_alone_is_empty(self):
        """只有 thought 没有 answer/tool_calls，不能冒充最终答案。

        尤其不能把协议 JSON 原文当作答复回给用户。
        """
        result = parse_response(content=json.dumps({"thought": "我在想"}))
        self.assertEqual(result.kind, OutputKind.EMPTY)
        self.assertEqual(result.thought, "我在想")
        self.assertEqual(result.answer, "")
        self.assertTrue(any("既无工具调用也无有效答案" in w for w in result.warnings))

    def test_unknown_fields_do_not_leak_json_as_answer(self):
        """字段名不在别名表里时，绝不把 JSON 原文当答复。"""
        result = parse_response(content=json.dumps({"结论": "应该是 42"}))
        self.assertEqual(result.kind, OutputKind.EMPTY)
        self.assertNotIn("{", result.answer)
        self.assertIn("实际字段", " ".join(result.warnings))


class TestFallbackPaths(unittest.TestCase):
    """路径 3：回退为纯文本答案（spec 3.4 / 6.2 E5 —— 不崩溃）。"""

    def test_plain_text_becomes_answer(self):
        result = parse_response(content="你好，有什么可以帮你？")
        self.assertEqual(result.kind, OutputKind.FINAL_ANSWER)
        self.assertEqual(result.source, ParseSource.PLAIN_TEXT)
        self.assertEqual(result.answer, "你好，有什么可以帮你？")
        self.assertTrue(result.degraded)

    def test_invalid_json_looking_text_falls_back(self):
        result = parse_response(content='{"thought": "没写完')
        self.assertEqual(result.kind, OutputKind.FINAL_ANSWER)
        self.assertEqual(result.source, ParseSource.PLAIN_TEXT)
        self.assertIn("没写完", result.answer)
        self.assertTrue(any("JSON" in w for w in result.warnings))

    def test_json_array_wrapper_is_unwrapped_to_inner_object(self):
        """模型偶发用数组包一层，内层协议对象仍应被识别。"""
        result = parse_response(content='[{"answer": "x"}]')
        self.assertEqual(result.kind, OutputKind.FINAL_ANSWER)
        self.assertEqual(result.answer, "x")
        self.assertEqual(result.source, ParseSource.JSON_TEXT)

    def test_thinking_tag_in_plain_text(self):
        result = parse_response(content="<thinking>先想想</thinking>答案是这个。")
        self.assertEqual(result.thought, "先想想")
        self.assertEqual(result.answer, "答案是这个。")

    def test_empty_content_is_empty_kind(self):
        result = parse_response(content="")
        self.assertEqual(result.kind, OutputKind.EMPTY)
        self.assertEqual(result.source, ParseSource.NONE)
        self.assertTrue(result.is_empty)
        self.assertFalse(result.has_answer)

    def test_whitespace_only_content_is_empty(self):
        result = parse_response(content="   \n  ")
        self.assertEqual(result.kind, OutputKind.EMPTY)

    def test_none_content_is_empty(self):
        result = parse_response(content=None)  # type: ignore[arg-type]
        self.assertEqual(result.kind, OutputKind.EMPTY)

    def test_no_inputs_at_all_is_empty(self):
        result = parse_response()
        self.assertEqual(result.kind, OutputKind.EMPTY)
        self.assertEqual(result.tool_calls, [])

    def test_never_raises_on_garbage(self):
        """回退策略的底线：任何输入都不抛异常。"""
        for garbage in [
            "{",
            "}",
            "{{{{",
            "}}}}",
            '{"a": }',
            "```json\n{broken\n```",
            "\x00\x01",
            "null",
            "true",
            "123",
            '"just a string"',
        ]:
            with self.subTest(content=garbage):
                result = parse_response(content=garbage)
                self.assertIsInstance(result, ParsedOutput)
                self.assertIsInstance(result.kind, OutputKind)


class TestBalancedObjectExtraction(unittest.TestCase):
    """花括号提取需忽略字符串内的括号。"""

    def test_braces_inside_string_values(self):
        payload = (
            "说明：{'不是 JSON'}\n"
            + json.dumps({"answer": "包含 } 和 { 的答案"})
        )
        result = parse_response(content=payload)
        # 首个平衡片段是 {'不是 JSON'}，不是合法 JSON → 继续到下一个
        self.assertIn("包含", result.answer)

    def test_escaped_quotes_do_not_break_extraction(self):
        payload = json.dumps({"answer": '他说 "你好" 然后走了'})
        result = parse_response(content=payload)
        self.assertEqual(result.answer, '他说 "你好" 然后走了')

    def test_nested_objects_extracted_correctly(self):
        payload = json.dumps(
            {
                "tool_calls": [
                    {"name": "todo", "arguments": {"action": "add", "content": "a{b}"}}
                ]
            }
        )
        result = parse_response(content=payload)
        self.assertEqual(result.tool_calls[0].arguments["content"], "a{b}")


class TestParsedOutputHelpers(unittest.TestCase):
    def test_to_dict_is_serializable(self):
        result = parse_response(
            native_tool_calls=[tool_call("calculator", {"expression": "1+1"}, "c1")]
        )
        payload = result.to_dict()
        json.dumps(payload)  # 不应抛异常
        self.assertEqual(payload["kind"], "tool_calls")
        self.assertEqual(payload["tool_calls"][0]["name"], "calculator")

    def test_tool_call_request_helpers(self):
        call = ToolCallRequest(name="x", arguments={"a": 1})
        self.assertTrue(call.arguments_valid)
        bad = ToolCallRequest(name="y", argument_error="坏了")
        self.assertFalse(bad.arguments_valid)

    def test_protocol_instructions_mention_all_three(self):
        text = OUTPUT_PROTOCOL_INSTRUCTIONS
        self.assertIn("thought", text)
        self.assertIn("tool_calls", text)
        self.assertIn("answer", text)


class TestProtocolSelectionPriority(unittest.TestCase):
    """spec 3.4：三类内容需被**稳定区分**，优先级必须明确。"""

    def test_native_tool_calls_win_over_text_json(self):
        result = parse_response(
            content=json.dumps({"answer": "文本里的答案"}),
            native_tool_calls=[tool_call("calculator", {"expression": "1"})],
        )
        self.assertEqual(result.source, ParseSource.NATIVE_TOOL_CALLS)
        self.assertEqual(result.tool_calls[0].name, "calculator")

    def test_protocol_json_wins_over_plain_text(self):
        payload = "前言。" + json.dumps({"answer": "结构化答案"})
        result = parse_response(content=payload)
        self.assertEqual(result.source, ParseSource.JSON_TEXT)
        self.assertEqual(result.answer, "结构化答案")

    def test_plain_text_is_last_resort(self):
        result = parse_response(content="既不是 JSON 也没有标签")
        self.assertEqual(result.source, ParseSource.PLAIN_TEXT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
