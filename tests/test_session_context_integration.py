"""T018 / T040 集成验证：Session × Context × 工具 三方联动。

这是 spec 6.1 N8 的核心场景：用户 A 开两个窗口，
窗口1「查天气 + 记待办」、窗口2「写周报 + 记待办」，
两窗口的 context 与状态彼此隔离，且随时可以接着聊。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from tests.helpers import AgentTestCase  # noqa: F401
from agent.context import ContextBuilder, ContextPolicy
from agent.session import ROLE_SYSTEM, ROLE_TOOL, ROLE_USER, SessionStore, ToolCallRecord
from agent.tools.base import ToolContext
from agent.tools.registry import build_default_registry
from agent.tools.todo import session_todos


class TestSessionToolContextIntegration(unittest.TestCase):
    """T018：session 状态与工具状态必须是同一份。"""

    def setUp(self):
        self.registry = build_default_registry()
        self.store = SessionStore()

    def _tool_context(self, session):
        """构造带 session 状态的工具上下文（与主循环将采用的方式一致）。"""
        return ToolContext(session_id=session.session_id, state=session.state)

    def test_tool_writes_into_session_state(self):
        session = self.store.create("w1")
        result = self.registry.execute(
            "todo",
            {"action": "add", "content": "查天气"},
            self._tool_context(session),
        )
        self.assertTrue(result.ok)
        todos = session_todos(session.state, "w1")
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["content"], "查天气")

    def test_session_state_is_json_serializable(self):
        """状态必须可序列化，否则无法随 session 落盘。"""
        import json

        session = self.store.create("w1")
        self.registry.execute(
            "todo", {"action": "add", "content": "带伞"}, self._tool_context(session)
        )
        json.dumps(session.state)  # 不应抛异常

    def test_tool_state_appears_in_context(self):
        """工具写的状态必须能被 context 渲染（T019 的联动点）。"""
        session = self.store.create("w1")
        self.registry.execute(
            "todo",
            {"action": "add", "content": "查天气"},
            self._tool_context(session),
        )
        result = ContextBuilder().build(session)
        joined = " ".join(m.content or "" for m in result.messages)
        self.assertIn("查天气", joined)

    def test_two_sessions_with_one_registry_stay_isolated(self):
        """同一个注册表服务两个 session，状态不能串。"""
        w1 = self.store.create("w1")
        w2 = self.store.create("w2")
        self.registry.execute(
            "todo", {"action": "add", "content": "窗口1 的事"}, self._tool_context(w1)
        )
        self.registry.execute(
            "todo", {"action": "add", "content": "窗口2 的事"}, self._tool_context(w2)
        )

        w1_context = ContextBuilder().build(w1)
        w2_context = ContextBuilder().build(w2)
        w1_text = " ".join(m.content or "" for m in w1_context.messages)
        w2_text = " ".join(m.content or "" for m in w2_context.messages)

        self.assertIn("窗口1 的事", w1_text)
        self.assertNotIn("窗口2 的事", w1_text)
        self.assertIn("窗口2 的事", w2_text)
        self.assertNotIn("窗口1 的事", w2_text)

    def test_todo_complete_visible_in_context(self):
        session = self.store.create("w1")
        context = self._tool_context(session)
        added = self.registry.execute(
            "todo", {"action": "add", "content": "倒垃圾"}, context
        )
        todo_id = added.data["item"]["id"]
        self.registry.execute(
            "todo", {"action": "complete", "todo_id": todo_id}, context
        )
        built = ContextBuilder().build(session)
        joined = " ".join(m.content or "" for m in built.messages)
        self.assertIn("[x] #1 倒垃圾", joined)


class TestDualWindowScenario(unittest.TestCase):
    """spec 3.6 / 6.1 N8 的端到端场景（不含 LLM，纯状态与上下文层）。"""

    def setUp(self):
        self.registry = build_default_registry()
        self.store = SessionStore()
        self.builder = ContextBuilder()
        self.w1 = self.store.create("window-1")
        self.w2 = self.store.create("window-2")

    def _ctx(self, session):
        return ToolContext(session_id=session.session_id, state=session.state)

    def _run(self, session, tool, arguments):
        return self.registry.execute(tool, arguments, self._ctx(session))

    def _context_text(self, session):
        return " ".join(
            m.content or "" for m in self.builder.build(session).messages
        )

    def test_full_dual_window_flow(self):
        # --- 窗口1：查天气 + 记待办 ---
        self.w1.append_user_message("帮我查一下广州天气，并记个待办")
        weather = self._run(self.w1, "search", {"query": "广州 天气"})
        self.assertTrue(weather.ok)
        self.w1.append_assistant_message(
            thought="先检索天气",
            tool_calls=[],
        )
        self.w1.append_tool_message("search", weather.to_llm_text())
        self._run(self.w1, "todo", {"action": "add", "content": "查看广州天气"})
        self.w1.append_assistant_message(content="已查到广州天气，并记下待办。")

        # --- 窗口2：写周报 + 记待办 ---
        self.w2.append_user_message("帮我写个周报，并记个待办")
        report = self._run(self.w2, "search", {"query": "周报"})
        self.assertTrue(report.ok)
        self.w2.append_tool_message("search", report.to_llm_text())
        self._run(self.w2, "todo", {"action": "add", "content": "写周报"})
        self.w2.append_assistant_message(content="周报框架已整理，并记下待办。")

        # --- 断言：两个窗口的上下文互不污染 ---
        w1_text = self._context_text(self.w1)
        w2_text = self._context_text(self.w2)

        self.assertIn("查看广州天气", w1_text)
        self.assertNotIn("写周报", w1_text.split("待办清单")[-1] if "待办清单" in w1_text else w1_text)
        self.assertIn("写周报", w2_text)
        self.assertNotIn("查看广州天气", w2_text.split("待办清单")[-1] if "待办清单" in w2_text else w2_text)

    def test_resume_window_1_after_working_in_window_2(self):
        """随时回到窗口1，历史与待办都还在（spec 3.6）。"""
        self.w1.append_user_message("窗口1 第一句")
        self._run(self.w1, "todo", {"action": "add", "content": "窗口1 待办"})

        # 切到窗口2 干活
        self.w2.append_user_message("窗口2 第一句")
        self._run(self.w2, "todo", {"action": "add", "content": "窗口2 待办"})
        self.w2.append_assistant_message(content="窗口2 已完成")

        # 回到窗口1（模拟重新取 session）
        resumed = self.store.require("window-1")
        text = self._context_text(resumed)
        self.assertIn("窗口1 第一句", text)
        self.assertIn("窗口1 待办", text)
        self.assertNotIn("窗口2 待办", text)
        self.assertNotIn("窗口2 已完成", text)

    def test_resume_after_reload_from_disk(self):
        """落盘后重新加载，两个窗口仍各自隔离且状态完整。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.json")
            store = SessionStore(path)
            w1 = store.create("w1")
            w2 = store.create("w2")
            w1.append_user_message("窗口1")
            w2.append_user_message("窗口2")
            self.registry.execute(
                "todo",
                {"action": "add", "content": "A"},
                ToolContext(session_id="w1", state=w1.state),
            )
            self.registry.execute(
                "todo",
                {"action": "add", "content": "B"},
                ToolContext(session_id="w2", state=w2.state),
            )
            store.save()

            reloaded = SessionStore(path)
            builder = ContextBuilder()
            text1 = " ".join(
                m.content or "" for m in builder.build(reloaded.require("w1")).messages
            )
            text2 = " ".join(
                m.content or "" for m in builder.build(reloaded.require("w2")).messages
            )
            self.assertIn("窗口1", text1)
            self.assertIn("A", text1)
            self.assertNotIn("B", text1)
            self.assertIn("B", text2)
            self.assertNotIn("A", text2)


class TestContextBuilderWithRealHistory(unittest.TestCase):
    """验证 builder 在真实工具结果文本下的行为。"""

    def test_long_search_result_is_truncated_before_injection(self):
        registry = build_default_registry()
        session = SessionStore().create("w1")
        session.append_user_message("查一下 session")
        result = registry.execute(
            "search", {"query": "session"}, ToolContext(session_id="w1")
        )
        # 必须成对写入 tool_call 与 tool 结果，否则会被当作孤儿消息清理掉
        session.append_assistant_message(
            tool_calls=[
                ToolCallRecord(name="search", call_id="c1", arguments={"query": "session"})
            ]
        )
        session.append_tool_message("search", result.to_llm_text(), "c1")

        builder = ContextBuilder()
        # 默认上限 2000 字符：本次结果更短，不应截断
        built = builder.build(session)
        self.assertEqual(built.tool_results_truncated, 0)

        tiny = ContextBuilder(ContextPolicy(tool_result_max_chars=10))
        small = tiny.build(session)
        self.assertEqual(small.tool_results_truncated, 1)
        tool_messages = [m for m in small.messages if m.role == ROLE_TOOL]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("已截断", tool_messages[0].content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
