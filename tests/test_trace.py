"""T006 验证：trace / 执行日志基础设施。

覆盖 spec 3.7、3.1（工具调用四要素进 trace）、6.1 N11（可回溯）。
"""

from __future__ import annotations

import io
import json
import unittest

from tests.helpers import AgentTestCase  # noqa: F401  (导入即完成 sys.path 设置)
from agent.trace import (
    EVENT_ERROR,
    EVENT_LLM_CALL,
    EVENT_LOOP_STEP,
    EVENT_TOOL_CALL,
    Tracer,
    null_tracer,
    truncate_text,
)


class TestRecordAndOutput(unittest.TestCase):
    def setUp(self):
        self.buffer = io.StringIO()
        self.tracer = Tracer(stream=self.buffer)

    def test_writes_readable_single_line(self):
        self.tracer.tool_call("calculator", "t1", expression="1+1", result="2")
        line = self.buffer.getvalue().strip()
        self.assertIn("[tool_call]", line)
        self.assertIn("calculator", line)
        self.assertIn("trace=t1", line)
        self.assertIn("expression=1+1", line)
        self.assertIn("result=2", line)
        self.assertEqual(line.count("\n"), 0)

    def test_none_payload_values_are_dropped(self):
        record = self.tracer.record(EVENT_TOOL_CALL, "t", result=None, ok=True)
        self.assertEqual(record.payload, {"ok": True})
        self.assertNotIn("result=", self.buffer.getvalue())

    def test_records_are_kept_in_memory(self):
        self.tracer.tool_call("a")
        self.tracer.tool_call("b")
        self.assertEqual([r.name for r in self.tracer.records], ["a", "b"])

    def test_records_property_returns_copy(self):
        self.tracer.tool_call("a")
        snapshot = self.tracer.records
        snapshot.append("junk")
        self.assertEqual(len(self.tracer.records), 1)

    def test_event_helpers_use_expected_event_types(self):
        self.tracer.session_event("open")
        self.tracer.loop_step("step1")
        self.tracer.llm_call()
        self.tracer.tool_call("calculator")
        self.tracer.error("boom")
        self.assertEqual(
            [r.event for r in self.tracer.records],
            [
                "session",
                EVENT_LOOP_STEP,
                EVENT_LLM_CALL,
                EVENT_TOOL_CALL,
                EVENT_ERROR,
            ],
        )


class TestJsonlFormat(unittest.TestCase):
    def test_each_line_is_valid_json(self):
        buffer = io.StringIO()
        tracer = Tracer(stream=buffer, fmt="jsonl")
        tracer.tool_call("calculator", "t1", expression="1+1")
        tracer.llm_call("chat.completions", "t1", model="m")
        lines = [ln for ln in buffer.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            parsed = json.loads(line)
            self.assertIn("timestamp", parsed)
            self.assertIn("event", parsed)
            self.assertIn("name", parsed)
            self.assertIn("payload", parsed)

    def test_unknown_format_rejected(self):
        with self.assertRaises(ValueError):
            Tracer(stream=io.StringIO(), fmt="xml")


class TestTruncation(unittest.TestCase):
    """spec 6.3 B4：超长内容需被截断，避免 trace 本身失控。"""

    def test_long_text_is_truncated_with_marker(self):
        text = "x" * 100
        result = truncate_text(text, max_length=10)
        self.assertTrue(result.startswith("x" * 10))
        self.assertIn("<truncated 90 chars>", result)

    def test_short_text_untouched(self):
        self.assertEqual(truncate_text("abc", max_length=10), "abc")

    def test_newlines_are_escaped_to_keep_one_line(self):
        self.assertEqual(truncate_text("a\r\nb\nc", max_length=100), "a\\nb\\nc")

    def test_dict_is_serialized(self):
        self.assertEqual(truncate_text({"a": 1}, max_length=100), '{"a": 1}')

    def test_none_becomes_empty(self):
        self.assertEqual(truncate_text(None), "")

    def test_non_serializable_falls_back_to_repr(self):
        self.assertIn("object", truncate_text(object(), max_length=100))

    def test_tracer_applies_truncation(self):
        buffer = io.StringIO()
        tracer = Tracer(stream=buffer, max_text=5)
        tracer.tool_call("t", payload="y" * 50)
        self.assertIn("<truncated", buffer.getvalue())


class TestTimeit(unittest.TestCase):
    def test_measures_duration(self):
        tracer = null_tracer()
        with tracer.timeit() as box:
            pass
        self.assertIn("duration_ms", box)
        self.assertGreaterEqual(box["duration_ms"], 0.0)

    def test_records_duration_even_on_exception(self):
        tracer = null_tracer()
        box = {}
        with self.assertRaises(RuntimeError):
            with tracer.timeit() as box:
                raise RuntimeError("boom")
        self.assertIn("duration_ms", box)


class TestDisabledTracer(unittest.TestCase):
    def test_disabled_writes_nothing_but_still_records(self):
        buffer = io.StringIO()
        tracer = Tracer(stream=buffer, enabled=False)
        tracer.tool_call("calculator")
        self.assertEqual(buffer.getvalue(), "")
        self.assertEqual(len(tracer.records), 1)
        self.assertFalse(tracer.enabled)

    def test_write_failure_does_not_break_main_flow(self):
        """spec 3.7：trace 写入失败绝不能影响主流程。"""

        class BrokenStream:
            def write(self, _text):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

        tracer = Tracer(stream=BrokenStream())
        tracer.tool_call("calculator")  # 不应抛出
        self.assertFalse(tracer.enabled)

    def test_new_trace_id_is_short_and_unique(self):
        tracer = null_tracer()
        ids = {tracer.new_trace_id() for _ in range(20)}
        self.assertEqual(len(ids), 20)
        self.assertTrue(all(len(i) == 8 for i in ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
