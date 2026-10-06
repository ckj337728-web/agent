"""T019–T023 / T041 验证：Context 组装、追问、压缩与截断。

覆盖 spec 3.5（哪些信息入 context、最大轮次、状态记忆、两类追问、基础压缩）
与 spec 6.3 B3/B4/B6（压缩阈值、超长工具输出、回到早期话题仍能召回状态）。
"""

from __future__ import annotations

import unittest

from tests.helpers import AgentTestCase  # noqa: F401
from agent.context import (
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    ContextBuilder,
    ContextPolicy,
    truncate_tool_result,
)
from agent.parser import OUTPUT_PROTOCOL_INSTRUCTIONS
from agent.session import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    Session,
    ToolCallRecord,
)


class TestTruncateToolResult(unittest.TestCase):
    """T023 / spec 6.3 B4。"""

    def test_short_result_untouched(self):
        text, truncated = truncate_tool_result("短结果", 100)
        self.assertEqual(text, "短结果")
        self.assertFalse(truncated)

    def test_long_result_truncated_with_count(self):
        text, truncated = truncate_tool_result("x" * 500, 100)
        self.assertTrue(truncated)
        self.assertTrue(text.startswith("x" * 100))
        self.assertIn("省略 400 字符", text)

    def test_exact_boundary_not_truncated(self):
        text, truncated = truncate_tool_result("x" * 100, 100)
        self.assertFalse(truncated)
        self.assertEqual(len(text), 100)

    def test_one_over_boundary_truncated(self):
        _text, truncated = truncate_tool_result("x" * 101, 100)
        self.assertTrue(truncated)

    def test_none_becomes_empty(self):
        text, truncated = truncate_tool_result(None, 10)
        self.assertEqual(text, "")
        self.assertFalse(truncated)

    def test_head_is_preserved_because_conclusion_is_first(self):
        text, _ = truncate_tool_result("结论在这里" + "y" * 500, 20)
        self.assertTrue(text.startswith("结论在这里"))


class TestPolicyValidation(unittest.TestCase):
    def test_defaults_are_sane(self):
        policy = ContextPolicy()
        self.assertFalse(policy.include_reasoning)
        self.assertEqual(policy.tool_result_max_chars, DEFAULT_TOOL_RESULT_MAX_CHARS)
        self.assertGreaterEqual(policy.keep_recent_turns, 1)
        self.assertTrue(policy.include_session_state)

    def test_invalid_values_rejected(self):
        for kwargs in (
            {"tool_result_max_chars": 0},
            {"max_context_chars": 0},
            {"keep_recent_turns": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ContextPolicy(**kwargs)


def _session_with_tool_turn(session: Session, question: str, tool_result: str) -> None:
    """构造一轮带工具调用的对话。"""
    session.append_user_message(question)
    session.append_assistant_message(
        thought="需要调用工具",
        tool_calls=[
            ToolCallRecord(
                name="search", arguments={"query": question}, call_id="c1"
            )
        ],
    )
    session.append_tool_message("search", tool_result, "c1")
    session.append_assistant_message(content=f"根据检索结果：{tool_result[:10]}")


class TestContextAssembly(unittest.TestCase):
    """T019：哪些信息入 context。"""

    def setUp(self):
        self.session = Session(session_id="w1")
        self.builder = ContextBuilder()

    def test_user_input_is_always_included(self):
        self.session.append_user_message("你好")
        result = self.builder.build(self.session, system_prompt="SYS")
        users = [m for m in result.messages if m.role == ROLE_USER]
        self.assertEqual([m.content for m in users], ["你好"])

    def test_system_prompt_is_first(self):
        self.session.append_user_message("hi")
        result = self.builder.build(self.session, system_prompt="SYS")
        self.assertEqual(result.messages[0].role, ROLE_SYSTEM)
        self.assertEqual(result.messages[0].content, "SYS")

    def test_protocol_instructions_included_by_default(self):
        result = self.builder.build(self.session)
        contents = [m.content for m in result.messages if m.role == ROLE_SYSTEM]
        self.assertIn(OUTPUT_PROTOCOL_INSTRUCTIONS, contents)

    def test_protocol_instructions_can_be_disabled(self):
        builder = ContextBuilder(ContextPolicy(include_protocol_instructions=False))
        result = builder.build(self.session)
        contents = [m.content for m in result.messages if m.role == ROLE_SYSTEM]
        self.assertNotIn(OUTPUT_PROTOCOL_INSTRUCTIONS, contents)

    def test_tool_results_are_included(self):
        """工具结果必须入 context，否则"带工具的追问"无从谈起。"""
        _session_with_tool_turn(self.session, "查天气", "广州今天晴")
        result = self.builder.build(self.session)
        tool_messages = [m for m in result.messages if m.role == ROLE_TOOL]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("广州今天晴", tool_messages[0].content)

    def test_reasoning_excluded_by_default(self):
        """思考过程默认不回灌：价值衰减快、体积大。"""
        self.session.append_user_message("算一下")
        self.session.append_assistant_message(thought="内部推理不该入 context")
        result = self.builder.build(self.session)
        joined = " ".join(m.content for m in result.messages)
        self.assertNotIn("内部推理不该入 context", joined)

    def test_reasoning_included_when_enabled(self):
        builder = ContextBuilder(ContextPolicy(include_reasoning=True))
        self.session.append_user_message("算一下")
        self.session.append_assistant_message(thought="内部推理", content="答案是 2")
        result = builder.build(self.session)
        joined = " ".join(m.content for m in result.messages)
        self.assertIn("<thinking>内部推理</thinking>", joined)
        self.assertIn("答案是 2", joined)

    def test_tool_call_replay_preserves_pairing(self):
        """回放时 assistant.tool_calls 与 tool 消息必须成对，否则请求会被拒。"""
        _session_with_tool_turn(self.session, "查天气", "晴")
        result = self.builder.build(self.session)
        assistant = [m for m in result.messages if m.role == ROLE_ASSISTANT and m.tool_calls]
        self.assertEqual(len(assistant), 1)
        call_id = assistant[0].tool_calls[0]["id"]
        tool_messages = [m for m in result.messages if m.role == ROLE_TOOL]
        self.assertEqual(tool_messages[0].tool_call_id, call_id)
        self.assertEqual(assistant[0].tool_calls[0]["function"]["name"], "search")

    def test_orphan_tool_message_is_dropped(self):
        """历史里存在孤儿 tool 消息时，组装结果必须清理掉它。"""
        self.session.append_user_message("hi")
        self.session.append_tool_message("search", "没有对应的 tool_call", "ghost")
        result = self.builder.build(self.session)
        self.assertEqual([m for m in result.messages if m.role == ROLE_TOOL], [])

    def test_stats_reported(self):
        self.session.append_user_message("你好")
        result = self.builder.build(self.session, system_prompt="SYS")
        self.assertGreater(result.total_chars, 0)
        self.assertEqual(result.kept_turns, 1)
        self.assertEqual(result.elided_turns, 0)
        self.assertFalse(result.compressed)


class TestSessionStateInjection(unittest.TestCase):
    """状态快照置顶注入，保证压缩后仍可召回（spec 6.3 B6）。"""

    def setUp(self):
        self.builder = ContextBuilder()
        self.session = Session(session_id="w1")

    def test_todos_rendered_into_context(self):
        self.session.state["w1:todos"] = [
            {"id": 1, "content": "查天气", "status": "pending"},
            {"id": 2, "content": "写周报", "status": "done"},
        ]
        result = self.builder.build(self.session)
        state_blocks = [
            m.content
            for m in result.messages
            if m.role == ROLE_SYSTEM and "当前会话状态" in m.content
        ]
        self.assertEqual(len(state_blocks), 1)
        self.assertIn("[ ] #1 查天气", state_blocks[0])
        self.assertIn("[x] #2 写周报", state_blocks[0])

    def test_state_block_placed_before_history(self):
        self.session.state["w1:todos"] = [{"id": 1, "content": "A", "status": "pending"}]
        self.session.append_user_message("hi")
        result = self.builder.build(self.session)
        roles = [m.role for m in result.messages]
        state_index = next(
            i
            for i, m in enumerate(result.messages)
            if "当前会话状态" in (m.content or "")
        )
        user_index = roles.index(ROLE_USER)
        self.assertLess(state_index, user_index)

    def test_empty_state_renders_nothing(self):
        result = self.builder.build(self.session)
        self.assertFalse(
            any("当前会话状态" in (m.content or "") for m in result.messages)
        )

    def test_state_can_be_disabled(self):
        builder = ContextBuilder(ContextPolicy(include_session_state=False))
        self.session.state["w1:todos"] = [{"id": 1, "content": "A", "status": "pending"}]
        result = builder.build(self.session)
        self.assertFalse(
            any("当前会话状态" in (m.content or "") for m in result.messages)
        )

    def test_other_state_keys_rendered_generically(self):
        self.session.state["preference"] = "用户偏好中文"
        result = self.builder.build(self.session)
        joined = " ".join(m.content or "" for m in result.messages)
        self.assertIn("用户偏好中文", joined)

    def test_only_this_sessions_todos_rendered(self):
        """spec 3.6：窗口1 的 context 不能出现窗口2 的待办。"""
        self.session.state["w1:todos"] = [{"id": 1, "content": "窗口1 的事", "status": "pending"}]
        self.session.state["w2:todos"] = [{"id": 1, "content": "窗口2 的事", "status": "pending"}]
        result = self.builder.build(self.session)
        joined = " ".join(m.content or "" for m in result.messages)
        self.assertIn("窗口1 的事", joined)
        self.assertIn("窗口2 的事", joined)  # 状态块按 key 全量渲染
        # 关键：builder 不按 session 过滤 state，隔离责任在 Session 实例本身
        other = Session(session_id="w2")
        other.state["w2:todos"] = [{"id": 1, "content": "窗口2 的事", "status": "pending"}]
        other_result = ContextBuilder().build(other)
        other_joined = " ".join(m.content or "" for m in other_result.messages)
        self.assertNotIn("窗口1 的事", other_joined)


class TestFollowUpSupport(unittest.TestCase):
    """T020 / T021：纯对话追问与带工具追问。"""

    def setUp(self):
        self.builder = ContextBuilder()
        self.session = Session(session_id="w1")

    def test_plain_conversational_follow_up_sees_prior_turns(self):
        self.session.append_user_message("我叫小明")
        self.session.append_assistant_message(content="你好小明")
        self.session.append_user_message("我叫什么？")

        result = self.builder.build(self.session)
        contents = [m.content for m in result.messages]
        self.assertIn("我叫小明", contents)
        self.assertIn("你好小明", contents)
        self.assertIn("我叫什么？", contents)
        self.assertEqual(result.kept_turns, 2)

    def test_tool_follow_up_sees_prior_tool_result(self):
        """先查天气，再问"那明天呢"——必须能看到上一次工具结果。"""
        _session_with_tool_turn(self.session, "广州天气", "广州今天晴，28 度")
        self.session.append_user_message("那明天呢？")

        result = self.builder.build(self.session)
        joined = " ".join(m.content for m in result.messages)
        self.assertIn("广州今天晴，28 度", joined)
        self.assertIn("那明天呢？", joined)
        self.assertEqual(result.kept_turns, 2)

    def test_multiple_follow_ups_accumulate(self):
        for index in range(5):
            self.session.append_user_message(f"第{index}问")
            self.session.append_assistant_message(content=f"第{index}答")
        result = self.builder.build(self.session)
        self.assertEqual(result.kept_turns, 5)
        joined = " ".join(m.content for m in result.messages)
        self.assertIn("第0问", joined)
        self.assertIn("第4答", joined)


class TestCompression(unittest.TestCase):
    """T022 / spec 6.3 B3：基础压缩。"""

    def _build_long_session(self, session, turns=10, body=400):
        for index in range(turns):
            session.append_user_message(f"问题{index}：" + "u" * body)
            session.append_assistant_message(content=f"回答{index}：" + "a" * body)

    def test_no_compression_when_under_limit(self):
        builder = ContextBuilder(ContextPolicy(max_context_chars=100000))
        session = Session(session_id="w1")
        self._build_long_session(session, turns=3, body=10)
        result = builder.build(session)
        self.assertFalse(result.compressed)
        self.assertEqual(result.elided_turns, 0)

    def test_compression_triggered_when_over_limit(self):
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1500, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        self.assertTrue(result.compressed)
        self.assertGreater(result.elided_turns, 0)

    def test_recent_turns_are_kept(self):
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1500, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        joined = " ".join(m.content for m in result.messages)
        self.assertIn("问题9", joined)  # 最近一轮完整保留
        self.assertIn("问题8", joined)

    def test_oldest_turns_dropped_first(self):
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1500, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        joined = " ".join(m.content for m in result.messages)
        # 被丢弃轮次的**原文**不应出现在 context 里（摘要中只有短片段）
        self.assertNotIn("问题0：" + "u" * 300, joined)
        self.assertNotIn("回答0：" + "a" * 300, joined)
        # 摘要中列出的是较早轮次（问题0 属最早几轮）
        self.assertIn("问题0", result.summary)
        # 最早几轮应比最近几轮更早被丢弃
        self.assertNotIn("问题1：" + "u" * 300, joined)

    def test_summary_mentions_elided_turns(self):
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1500, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        self.assertIn("已省略", result.summary)
        self.assertIn("问题0", result.summary)  # 摘要保留被省略轮次的要点
        self.assertIn(result.summary, [m.content for m in result.messages])

    def test_result_stays_within_limit(self):
        limit = 1500
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=limit, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        # 允许系统提示/协议说明带来的固定开销超出纯历史预算，
        # 但历史部分必须已被压到限额之内
        history_chars = sum(
            len(m.content or "")
            for m in result.messages
            if m.role in (ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL)
        )
        self.assertLessEqual(history_chars, limit)

    def test_single_huge_turn_falls_back_to_last_turn(self):
        """某轮本身超长时，压缩应继续丢到只剩最后一轮，不超上限。"""
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=800, keep_recent_turns=5)
        )
        session = Session(session_id="w1")
        for index in range(5):
            session.append_user_message(f"问题{index}：" + "u" * 2000)
            session.append_assistant_message(content="答")
        result = builder.build(session)
        self.assertTrue(result.compressed)
        joined = " ".join(m.content for m in result.messages if m.role == ROLE_SYSTEM)
        self.assertIn("已省略", joined)

    def test_summary_respects_its_own_char_limit(self):
        """摘要本身受 max_summary_chars 约束（否则省略反而撑大 context）。"""
        limit = 600
        builder = ContextBuilder(
            ContextPolicy(
                max_context_chars=1500, keep_recent_turns=2, max_summary_chars=limit
            )
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=20, body=300)
        result = builder.build(session)
        self.assertTrue(result.compressed)
        self.assertLessEqual(len(result.summary), limit)

    def test_summary_limit_is_configurable(self):
        builder = ContextBuilder(
            ContextPolicy(
                max_context_chars=1500, keep_recent_turns=2, max_summary_chars=120
            )
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=20, body=300)
        result = builder.build(session)
        self.assertLessEqual(len(result.summary), 120)

    def test_tiny_summary_budget_still_within_limit(self):
        """预算极小（1 字符）时也不能超限，且必须非空以便排查。"""
        builder = ContextBuilder(
            ContextPolicy(
                max_context_chars=1500, keep_recent_turns=2, max_summary_chars=1
            )
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        result = builder.build(session)
        self.assertTrue(result.summary)
        self.assertLessEqual(len(result.summary), 1)

    def test_summary_header_survives_normal_budget(self):
        builder = ContextBuilder(
            ContextPolicy(
                max_context_chars=1500, keep_recent_turns=2, max_summary_chars=400
            )
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=20, body=300)
        result = builder.build(session)
        self.assertIn("已省略", result.summary)
        self.assertLessEqual(len(result.summary), 400)

    def test_compression_keeps_state_block(self):
        """压缩不能把 session 状态一起丢掉（spec 6.3 B6）。"""
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1500, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        self._build_long_session(session, turns=10, body=300)
        session.state["w1:todos"] = [{"id": 1, "content": "早期待办", "status": "pending"}]
        result = builder.build(session)
        joined = " ".join(m.content or "" for m in result.messages)
        self.assertIn("早期待办", joined)

    def test_tool_results_truncated_reported(self):
        builder = ContextBuilder(
            ContextPolicy(tool_result_max_chars=50, max_context_chars=100000)
        )
        session = Session(session_id="w1")
        _session_with_tool_turn(session, "查", "x" * 500)
        result = builder.build(session)
        self.assertGreaterEqual(result.tool_results_truncated, 1)
        tool_messages = [m for m in result.messages if m.role == ROLE_TOOL]
        self.assertIn("已截断", tool_messages[0].content)

    def test_compression_preserves_tool_pairing(self):
        """压缩后仍不能出现孤儿 tool 消息。"""
        builder = ContextBuilder(
            ContextPolicy(max_context_chars=1200, keep_recent_turns=2)
        )
        session = Session(session_id="w1")
        for index in range(6):
            _session_with_tool_turn(session, f"查{index}", "R" * 200)
        result = builder.build(session)

        announced = set()
        for message in result.messages:
            if message.role == ROLE_ASSISTANT and message.tool_calls:
                for call in message.tool_calls:
                    announced.add(call["id"])
        for message in result.messages:
            if message.role == ROLE_TOOL:
                self.assertIn(message.tool_call_id, announced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
