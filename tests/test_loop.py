"""T024–T027 / T035 验证：Agent 主循环四步骤与正常路径。

覆盖 spec 3.1（四步骤可观测）、spec 6.1 N2/N3/N4（直接回复、工具调用、链式调用）、
spec 6.3 B2（单响应多工具调用）。
"""

from __future__ import annotations

import io
import unittest

from tests.helpers import (
    RecordingTool,
    ScriptedLLM,
    native_tool_call,
    protocol_tool_calls,
    scripted_json,
)
from agent.context import ContextBuilder, ContextPolicy
from agent.loop import (
    EVENT_ANSWER,
    EVENT_THOUGHT,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    STOP_FINAL_ANSWER,
    STOP_MAX_ITERATIONS,
    AgentLoop,
    LoopPolicy,
    build_agent,
)
from agent.session import SessionStore
from agent.tools.registry import ToolRegistry, build_default_registry
from agent.trace import Tracer


def make_loop(script, tools=None, policy=None, sessions=None, tracer=None):
    """装配一个用脚本化假 LLM 驱动的 AgentLoop。"""
    llm = ScriptedLLM(script)
    registry = (
        tools
        if isinstance(tools, ToolRegistry)
        else ToolRegistry(tools or [])
    )
    loop = AgentLoop(
        client=llm,
        registry=registry,
        # 用 is None 而非 or：空的 SessionStore 是 falsy，会被静默替换掉
        sessions=sessions if sessions is not None else SessionStore(),
        builder=ContextBuilder(ContextPolicy(include_protocol_instructions=False)),
        tracer=tracer,
        policy=policy,
    )
    return loop, llm


class TestDirectReply(unittest.TestCase):
    """spec 6.1 N2：不调工具，一轮内给出答案。"""

    def test_direct_answer_via_protocol_json(self):
        loop, llm = make_loop([scripted_json({"thought": "简单问题", "answer": "你好！"})])
        result = loop.run_turn("你好", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_FINAL_ANSWER)
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "你好！")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(llm.remaining, 0)

    def test_direct_answer_via_plain_text_fallback(self):
        """解析回退：纯文本整体作为答案。"""
        loop, _ = make_loop(["我是一个纯文本回答。"])
        result = loop.run_turn("你好", session_id="w1")
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "我是一个纯文本回答。")

    def test_no_tools_requested_means_no_tool_schema_sent(self):
        """注册表为空时不应把空 tools 传给 LLM。"""
        loop, llm = make_loop([scripted_json({"answer": "ok"})])
        loop.run_turn("hi", session_id="w1")
        self.assertEqual(llm.calls[0]["tools"], [])

    def test_tool_schemas_are_sent_when_tools_registered(self):
        """spec 3.3：LLM 基于注册表导出的 Schema 自主决策。"""
        loop, llm = make_loop(
            [scripted_json({"answer": "ok"})], tools=[RecordingTool()]
        )
        loop.run_turn("hi", session_id="w1")
        self.assertEqual(llm.tool_names_sent(), ["recorder"])

    def test_answer_written_to_session(self):
        loop, _ = make_loop([scripted_json({"answer": "记住了"})])
        loop.run_turn("你好", session_id="w1")
        session = loop.sessions.require("w1")
        self.assertEqual(len(session), 2)  # user + assistant
        self.assertEqual(session.messages[0].content, "你好")
        self.assertEqual(session.last_assistant_message().content, "记住了")

    def test_turn_counter_increments_per_input(self):
        loop, _ = make_loop(
            [scripted_json({"answer": "一"}), scripted_json({"answer": "二"})]
        )
        loop.run_turn("第一问", session_id="w1")
        loop.run_turn("第二问", session_id="w1")
        self.assertEqual(loop.sessions.require("w1").current_turn(), 2)


class TestToolCallPath(unittest.TestCase):
    """spec 6.1 N3：调用工具后回灌结果并收敛。"""

    def _loop_with_recorder(self, script, policy=None):
        tool = RecordingTool()
        loop, llm = make_loop(script, tools=[tool], policy=policy)
        return loop, llm, tool

    def test_single_tool_call_then_answer(self):
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "结果是 A"}),
        ]
        loop, llm, tool = self._loop_with_recorder(script)
        result = loop.run_turn("记录 A", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "结果是 A")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual([i["value"] for i in tool.invocations], ["A"])
        self.assertEqual(llm.remaining, 0)

    def test_tool_result_is_fed_back_to_llm(self):
        """Step 3 的结果必须回灌（spec 3.1 Step 4）。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "收到"}),
        ]
        loop, llm, _ = self._loop_with_recorder(script)
        loop.run_turn("记录 A", session_id="w1")

        second_call = llm.messages_sent(1)
        tool_messages = [m for m in second_call if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("recorded:A", tool_messages[0]["content"])
        self.assertEqual(tool_messages[0]["tool_call_id"], "c1")

    def test_assistant_tool_calls_replayed_for_pairing(self):
        """回放时必须带 assistant.tool_calls，否则接口会拒。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "收到"}),
        ]
        loop, llm, _ = self._loop_with_recorder(script)
        loop.run_turn("记录 A", session_id="w1")

        second_call = llm.messages_sent(1)
        with_calls = [m for m in second_call if m["tool_calls"]]
        self.assertEqual(len(with_calls), 1)
        self.assertEqual(with_calls[0]["tool_calls"][0]["function"]["name"], "recorder")

    def test_tool_call_via_protocol_json(self):
        """文本协议路径也能驱动工具调用（不依赖原生 tool_calls）。"""
        script = [
            protocol_tool_calls(
                [{"name": "recorder", "arguments": {"value": "B"}}]
            ),
            scripted_json({"answer": "完成"}),
        ]
        loop, _llm, tool = self._loop_with_recorder(script)
        result = loop.run_turn("记录 B", session_id="w1")
        self.assertTrue(result.ok)
        self.assertEqual([i["value"] for i in tool.invocations], ["B"])

    def test_multiple_tool_calls_in_one_response(self):
        """spec 6.3 B2：一次响应含多个调用，全部按序执行。"""
        script = [
            {
                "tool_calls": [
                    native_tool_call("recorder", {"value": "1"}, "c1"),
                    native_tool_call("recorder", {"value": "2"}, "c2"),
                    native_tool_call("recorder", {"value": "3"}, "c3"),
                ]
            },
            scripted_json({"answer": "三个都记完了"}),
        ]
        loop, llm, tool = self._loop_with_recorder(script)
        result = loop.run_turn("记录三个", session_id="w1")

        self.assertEqual([i["value"] for i in tool.invocations], ["1", "2", "3"])
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(result.iterations, 2)

    def test_multiple_tool_results_all_fed_back(self):
        script = [
            {
                "tool_calls": [
                    native_tool_call("recorder", {"value": "1"}, "c1"),
                    native_tool_call("recorder", {"value": "2"}, "c2"),
                ]
            },
            scripted_json({"answer": "done"}),
        ]
        loop, llm, _ = self._loop_with_recorder(script)
        loop.run_turn("记录", session_id="w1")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(
            [m["tool_call_id"] for m in tool_messages], ["c1", "c2"]
        )

    def test_chained_tool_calls_across_iterations(self):
        """spec 6.1 N4：链式调用——第一轮结果驱动第二次调用。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "first"}, "c1")]},
            {"tool_calls": [native_tool_call("recorder", {"value": "second"}, "c2")]},
            scripted_json({"answer": "链条完成"}),
        ]
        loop, _llm, tool = self._loop_with_recorder(script)
        result = loop.run_turn("两步任务", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "链条完成")
        self.assertEqual(result.iterations, 3)
        self.assertEqual([i["value"] for i in tool.invocations], ["first", "second"])
        self.assertEqual(len(result.steps), 3)

    def test_tool_records_attached_to_session(self):
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm, _tool = self._loop_with_recorder(script)
        loop.run_turn("记录 A", session_id="w1")

        session = loop.sessions.require("w1")
        assistant_with_calls = [
            m for m in session.messages if m.role == "assistant" and m.tool_calls
        ]
        self.assertEqual(len(assistant_with_calls), 1)
        record = assistant_with_calls[0].tool_calls[0]
        self.assertEqual(record.name, "recorder")
        self.assertEqual(record.arguments, {"value": "A"})
        self.assertTrue(record.result_ok)
        self.assertEqual(record.result_text, "recorded:A")
        self.assertGreaterEqual(record.duration_ms, 0.0)

    def test_protocol_tool_call_gets_synthetic_id(self):
        script = [
            protocol_tool_calls([{"name": "recorder", "arguments": {"value": "x"}}]),
            scripted_json({"answer": "ok"}),
        ]
        loop, llm, _tool = self._loop_with_recorder(script)
        loop.run_turn("记录 x", session_id="w1")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertTrue(tool_messages[0]["tool_call_id"])  # 合成 id 非空


class TestLoopWithRealRegistry(unittest.TestCase):
    """用内置三工具跑真实工具链，验证端到端联动。"""

    def test_calculator_end_to_end(self):
        script = [
            {"tool_calls": [native_tool_call("calculator", {"expression": "12*12"})]},
            scripted_json({"answer": "12×12 等于 144"}),
        ]
        loop, llm = make_loop(script, tools=build_default_registry())
        result = loop.run_turn("12*12 是多少", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "12×12 等于 144")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("144", tool_messages[0]["content"])

    def test_todo_state_persists_across_turns_in_same_session(self):
        """spec 3.6：同一 session 后续轮次能看到之前的待办。"""
        script = [
            {"tool_calls": [native_tool_call("todo", {"action": "add", "content": "查天气"})]},
            scripted_json({"answer": "已记下"}),
            scripted_json({"answer": "你的待办有：查天气"}),
        ]
        loop, llm = make_loop(script, tools=build_default_registry())
        loop.run_turn("帮我记个待办：查天气", session_id="w1")
        loop.run_turn("我有什么待办？", session_id="w1")

        # 第二轮发送的 context 里应包含置顶的状态块
        second_turn_messages = llm.messages_sent(-1)
        joined = " ".join(m["content"] or "" for m in second_turn_messages)
        self.assertIn("查天气", joined)
        self.assertIn("当前会话状态", joined)

    def test_dual_window_isolation_through_loop(self):
        """spec 6.1 N8：两个窗口各自记待办，互不影响。"""
        script = [
            {"tool_calls": [native_tool_call("todo", {"action": "add", "content": "窗口1 的事"})]},
            scripted_json({"answer": "窗口1 已记"}),
            {"tool_calls": [native_tool_call("todo", {"action": "add", "content": "窗口2 的事"})]},
            scripted_json({"answer": "窗口2 已记"}),
            scripted_json({"answer": "只看窗口1"}),
        ]
        loop, llm = make_loop(script, tools=build_default_registry())
        loop.run_turn("记待办", session_id="window-1")
        loop.run_turn("记待办", session_id="window-2")
        loop.run_turn("我的待办", session_id="window-1")

        final_messages = llm.messages_sent(-1)
        joined = " ".join(m["content"] or "" for m in final_messages)
        self.assertIn("窗口1 的事", joined)
        self.assertNotIn("窗口2 的事", joined)


class TestObservability(unittest.TestCase):
    """spec 3.1 / 6.1 N11：每一步可被 trace 与事件观测。"""

    def test_trace_records_four_steps(self):
        buffer = io.StringIO()
        tracer = Tracer(stream=buffer, fmt="jsonl")
        tool = RecordingTool()
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[tool], tracer=tracer)
        loop.run_turn("记录 A", session_id="w1")

        events = [r.name for r in tracer.records]
        self.assertIn("step1_receive_input", events)
        self.assertIn("step2_build_request", events)
        self.assertIn("step2_parse_result", events)
        self.assertIn("step4_finish", events)
        self.assertIn("turn_finished", events)
        # 工具调用 trace 必须记录四要素
        tool_records = [r for r in tracer.records if r.event == "tool_call"]
        self.assertEqual(len(tool_records), 1)
        payload = tool_records[0].payload
        self.assertIn("arguments", payload)
        self.assertIn("result", payload)
        self.assertIn("duration_ms", payload)
        self.assertIn("ok", payload)

    def test_same_trace_id_across_one_turn(self):
        tracer = Tracer(enabled=False)
        tool = RecordingTool()
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[tool], tracer=tracer)
        result = loop.run_turn("记录 A", session_id="w1")

        trace_ids = {r.trace_id for r in tracer.records}
        self.assertEqual(trace_ids, {result.trace_id})

    def test_observer_receives_events_in_order(self):
        seen = []
        tool = RecordingTool()
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "done"}),
        ]
        loop, _llm = make_loop(script, tools=[tool])
        loop.run_turn("记录 A", session_id="w1", observer=lambda e, p: seen.append(e))

        self.assertIn(EVENT_TOOL_CALL, seen)
        self.assertIn(EVENT_TOOL_RESULT, seen)
        self.assertIn(EVENT_ANSWER, seen)
        self.assertLess(seen.index(EVENT_TOOL_CALL), seen.index(EVENT_TOOL_RESULT))
        self.assertLess(seen.index(EVENT_TOOL_RESULT), seen.index(EVENT_ANSWER))

    def test_observer_failure_does_not_break_loop(self):
        def bad_observer(_event, _payload):
            raise RuntimeError("观察者坏了")

        loop, _llm = make_loop([scripted_json({"answer": "ok"})])
        result = loop.run_turn("hi", session_id="w1", observer=bad_observer)
        self.assertTrue(result.ok)

    def test_result_is_json_serializable(self):
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[RecordingTool()])
        result = loop.run_turn("记录 A", session_id="w1")
        import json as _json

        payload = _json.loads(_json.dumps(result.to_dict(), ensure_ascii=False))
        self.assertEqual(payload["stop_reason"], STOP_FINAL_ANSWER)
        self.assertEqual(len(payload["steps"]), 2)

    def test_step_records_parse_source_and_warnings(self):
        loop, _llm = make_loop(["纯文本回答"])
        result = loop.run_turn("hi", session_id="w1")
        self.assertEqual(result.steps[0].parse_source, "plain_text")
        self.assertEqual(result.steps[0].answer, "纯文本回答")


class TestBuildAgentFactory(unittest.TestCase):
    def test_build_agent_wires_default_registry(self):
        llm = ScriptedLLM([scripted_json({"answer": "ok"})])
        loop = build_agent(client=llm, sessions=SessionStore())
        self.assertEqual(
            set(loop.registry.names()), {"calculator", "search", "todo"}
        )
        result = loop.run_turn("hi", session_id="w1")
        self.assertTrue(result.ok)


class TestPolicyValidation(unittest.TestCase):
    def test_defaults(self):
        policy = LoopPolicy()
        self.assertGreaterEqual(policy.max_iterations, 1)
        self.assertGreater(policy.tool_timeout_seconds, 0)

    def test_invalid_values_rejected(self):
        for kwargs in (
            {"max_iterations": 0},
            {"max_parse_retries": -1},
            {"max_no_progress_rounds": 0},
            {"tool_timeout_seconds": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    LoopPolicy(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
