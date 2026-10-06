"""T016 / T017 / T040 验证：session 模型、持久化与隔离。

覆盖 spec 3.6（多窗口隔离与随时续聊）、spec 6.2 E6（不存在的 session 有明确处理）、
spec 6.1 N8（双窗口场景中状态互不影响）。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.helpers import AgentTestCase  # noqa: F401
from agent.session import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    MAX_STORED_TOOL_RESULT,
    Message,
    Session,
    SessionError,
    SessionNotFoundError,
    SessionStore,
    ToolCallRecord,
)


class TestMessage(unittest.TestCase):
    def test_valid_roles_accepted(self):
        for role in (ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, "system"):
            with self.subTest(role=role):
                self.assertEqual(Message(role=role, content="x").role, role)

    def test_invalid_role_rejected(self):
        with self.assertRaises(SessionError) as ctx:
            Message(role="robot", content="x")
        self.assertIn("robot", str(ctx.exception))

    def test_round_trip_through_dict(self):
        original = Message(
            role=ROLE_ASSISTANT,
            content="答复",
            thought="推理",
            tool_calls=[
                ToolCallRecord(
                    name="calculator",
                    arguments={"expression": "1+1"},
                    call_id="c1",
                    result_ok=True,
                    result_text="1+1 = 2",
                    duration_ms=1.5,
                )
            ],
            turn=3,
        )
        restored = Message.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.role, original.role)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.thought, original.thought)
        self.assertEqual(restored.turn, 3)
        self.assertEqual(restored.tool_calls[0].name, "calculator")
        self.assertEqual(restored.tool_calls[0].arguments, {"expression": "1+1"})
        self.assertEqual(restored.tool_calls[0].result_text, "1+1 = 2")
        self.assertEqual(restored.tool_calls[0].duration_ms, 1.5)

    def test_tool_call_record_failure_round_trip(self):
        record = ToolCallRecord(
            name="calculator", result_ok=False, error_code="division_by_zero"
        )
        restored = ToolCallRecord.from_dict(record.to_dict())
        self.assertFalse(restored.result_ok)
        self.assertEqual(restored.error_code, "division_by_zero")


class TestSession(unittest.TestCase):
    def setUp(self):
        self.session = Session(session_id="w1")

    def test_append_user_message_starts_new_turn(self):
        first = self.session.append_user_message("你好")
        self.assertEqual(first.turn, 1)
        second = self.session.append_user_message("在吗")
        self.assertEqual(second.turn, 2)
        self.assertEqual(self.session.turn_count(), 2)

    def test_messages_in_same_turn_share_turn_number(self):
        self.session.append_user_message("算一下")
        assistant = self.session.append_assistant_message(
            thought="需要工具",
            tool_calls=[ToolCallRecord(name="calculator", call_id="c1")],
        )
        tool = self.session.append_tool_message("calculator", "1+1 = 2", "c1")
        self.assertEqual(assistant.turn, 1)
        self.assertEqual(tool.turn, 1)
        self.assertEqual(self.session.current_turn(), 1)

    def test_last_assistant_message(self):
        self.session.append_user_message("hi")
        self.assertIsNone(self.session.last_assistant_message())
        self.session.append_assistant_message(content="hello")
        self.assertEqual(self.session.last_assistant_message().content, "hello")

    def test_reset_clears_everything(self):
        self.session.append_user_message("hi")
        self.session.state["x"] = 1
        self.session.reset()
        self.assertEqual(len(self.session), 0)
        self.assertEqual(self.session.state, {})
        self.assertEqual(self.session.current_turn(), 0)

    def test_round_trip_preserves_state_and_turns(self):
        self.session.append_user_message("查天气")
        self.session.state["w1:todos"] = [{"id": 1, "content": "带伞", "status": "pending"}]
        restored = Session.from_dict(json.loads(json.dumps(self.session.to_dict())))
        self.assertEqual(restored.session_id, "w1")
        self.assertEqual(restored.current_turn(), 1)
        self.assertEqual(restored.state["w1:todos"][0]["content"], "带伞")

    def test_from_dict_requires_session_id(self):
        with self.assertRaises(SessionError):
            Session.from_dict({"messages": []})


class TestSessionStoreLifecycle(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore()

    def test_create_and_get(self):
        session = self.store.create("w1")
        self.assertEqual(session.session_id, "w1")
        self.assertIs(self.store.get("w1"), session)
        self.assertIn("w1", self.store)
        self.assertEqual(len(self.store), 1)

    def test_create_auto_generates_id(self):
        session = self.store.create()
        self.assertTrue(session.session_id)
        self.assertIn(session.session_id, self.store)

    def test_create_duplicate_rejected(self):
        self.store.create("w1")
        with self.assertRaises(SessionError):
            self.store.create("w1")

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.store.get("nope"))

    def test_get_or_create_creates_when_missing(self):
        """spec 6.2 E6 的"新建"策略。"""
        session = self.store.get_or_create("fresh")
        self.assertEqual(session.session_id, "fresh")
        self.assertIs(self.store.get_or_create("fresh"), session)

    def test_require_raises_with_known_ids(self):
        """spec 6.2 E6 的"明确报错"策略。"""
        self.store.create("w1")
        with self.assertRaises(SessionNotFoundError) as ctx:
            self.store.require("missing")
        message = str(ctx.exception)
        self.assertIn("missing", message)
        self.assertIn("w1", message)  # 告知已有 session，便于纠正

    def test_require_message_when_no_sessions(self):
        with self.assertRaises(SessionNotFoundError) as ctx:
            self.store.require("x")
        self.assertIn("没有任何 session", str(ctx.exception))

    def test_delete(self):
        self.store.create("w1")
        self.assertTrue(self.store.delete("w1"))
        self.assertFalse(self.store.delete("w1"))
        self.assertIsNone(self.store.get("w1"))

    def test_session_ids_sorted(self):
        for sid in ("b", "a", "c"):
            self.store.create(sid)
        self.assertEqual(self.store.session_ids(), ["a", "b", "c"])

    def test_sessions_iteration(self):
        self.store.create("a")
        self.store.create("b")
        self.assertEqual([s.session_id for s in self.store], ["a", "b"])
        self.assertEqual(len(list(iter(self.store))), 2)

    def test_new_session_id_is_unique(self):
        ids = {SessionStore.new_session_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)


class TestSessionIsolation(unittest.TestCase):
    """spec 3.6 / 6.1 N8：两个窗口互相隔离。"""

    def setUp(self):
        self.store = SessionStore()
        self.w1 = self.store.create("w1")
        self.w2 = self.store.create("w2")

    def test_messages_do_not_leak_between_sessions(self):
        self.w1.append_user_message("窗口1：查天气")
        self.w2.append_user_message("窗口2：写周报")
        self.assertEqual([m.content for m in self.w1.messages], ["窗口1：查天气"])
        self.assertEqual([m.content for m in self.w2.messages], ["窗口2：写周报"])

    def test_state_does_not_leak_between_sessions(self):
        self.w1.state["w1:todos"] = [{"id": 1, "content": "A", "status": "pending"}]
        self.w2.state["w2:todos"] = [{"id": 1, "content": "B", "status": "pending"}]
        self.assertEqual(self.w1.state["w1:todos"][0]["content"], "A")
        self.assertEqual(self.w2.state["w2:todos"][0]["content"], "B")

    def test_turn_counters_are_independent(self):
        self.w1.append_user_message("a")
        self.w1.append_user_message("b")
        self.w2.append_user_message("c")
        self.assertEqual(self.w1.current_turn(), 2)
        self.assertEqual(self.w2.current_turn(), 1)

    def test_switch_back_and_forth_keeps_both_histories(self):
        """随时切回任一窗口，历史都还在。"""
        self.w1.append_user_message("窗口1 第一句")
        self.w2.append_user_message("窗口2 第一句")
        self.w1.append_user_message("窗口1 第二句")

        resumed = self.store.get("w1")
        self.assertEqual(
            [m.content for m in resumed.messages], ["窗口1 第一句", "窗口1 第二句"]
        )
        self.assertEqual(
            [m.content for m in self.store.get("w2").messages], ["窗口2 第一句"]
        )


class TestFilePersistence(unittest.TestCase):
    """T017：落盘后重新加载可恢复历史（模拟进程重启）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "sessions.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_path_is_exposed(self):
        self.assertEqual(SessionStore(self.path).path(), self.path)

    def test_history_survives_reload(self):
        store = SessionStore(self.path)
        session = store.create("w1")
        session.append_user_message("你好")
        session.append_assistant_message(content="你好，有什么可以帮你？")
        session.state["w1:todos"] = [{"id": 1, "content": "带伞", "status": "pending"}]
        store.save()

        reloaded = SessionStore(self.path)
        restored = reloaded.require("w1")
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.messages[0].content, "你好")
        self.assertEqual(restored.messages[1].content, "你好，有什么可以帮你？")
        self.assertEqual(restored.state["w1:todos"][0]["content"], "带伞")

    def test_reload_preserves_multiple_sessions(self):
        store = SessionStore(self.path)
        store.create("w1").append_user_message("一")
        store.create("w2").append_user_message("二")
        store.save()

        reloaded = SessionStore(self.path)
        self.assertEqual(reloaded.session_ids(), ["w1", "w2"])
        self.assertEqual(reloaded.require("w1").messages[0].content, "一")
        self.assertEqual(reloaded.require("w2").messages[0].content, "二")

    def test_reload_preserves_turn_counter(self):
        store = SessionStore(self.path)
        session = store.create("w1")
        session.append_user_message("a")
        session.append_user_message("b")
        store.save()

        self.assertEqual(SessionStore(self.path).require("w1").current_turn(), 2)

    def test_auto_saves_on_create_and_delete(self):
        store = SessionStore(self.path)
        store.create("w1")
        self.assertTrue(os.path.exists(self.path))
        store.delete("w1")
        self.assertEqual(SessionStore(self.path).session_ids(), [])

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(len(SessionStore(self.path)), 0)

    def test_corrupt_file_raises_session_error(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(SessionError) as ctx:
            SessionStore(self.path)
        self.assertIn("不是合法 JSON", str(ctx.exception))

    def test_creates_parent_directory(self):
        nested = os.path.join(self._tmp.name, "deep", "nested", "sessions.json")
        SessionStore(nested).create("w1")
        self.assertTrue(os.path.exists(nested))

    def test_no_temp_file_left_behind(self):
        store = SessionStore(self.path)
        store.create("w1")
        self.assertFalse(os.path.exists(f"{self.path}.tmp"))


class TestStoredToolResultCap(unittest.TestCase):
    def test_cap_constant_is_documented_and_positive(self):
        self.assertGreater(MAX_STORED_TOOL_RESULT, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
