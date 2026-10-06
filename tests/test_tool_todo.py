"""T011 验证：todo 工具与会话级状态隔离。

覆盖 spec 3.2（第三个工具具备真实状态语义）与
spec 3.6 / 6.1 N8（两个窗口互不影响、可随时续聊）。
"""

from __future__ import annotations

import unittest

from tests.helpers import AgentTestCase  # noqa: F401
from agent.tools.base import ToolContext, ToolErrorCode
from agent.tools.todo import (
    STATUS_DONE,
    STATUS_PENDING,
    TodoStore,
    TodoTool,
)


class TestTodoStore(unittest.TestCase):
    def setUp(self):
        self.items = {}
        self.store = TodoStore(self.items)

    def test_add_and_list(self):
        self.store.add("s1", "买牛奶")
        items = self.store.list("s1")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].content, "买牛奶")
        self.assertEqual(items[0].status, STATUS_PENDING)

    def test_ids_increment_per_session(self):
        first = self.store.add("s1", "a")
        second = self.store.add("s1", "b")
        self.assertEqual((first.id, second.id), (1, 2))

    def test_ids_are_independent_per_session(self):
        self.assertEqual(self.store.add("s1", "a").id, 1)
        self.assertEqual(self.store.add("s2", "b").id, 1, "新会话应重新编号")

    def test_sessions_are_isolated(self):
        """spec 3.6：不同 session 的状态互不影响。"""
        self.store.add("s1", "窗口1 的待办")
        self.store.add("s2", "窗口2 的待办")
        s1 = [i.content for i in self.store.list("s1")]
        s2 = [i.content for i in self.store.list("s2")]
        self.assertEqual(s1, ["窗口1 的待办"])
        self.assertEqual(s2, ["窗口2 的待办"])

    def test_peek_does_not_create_session(self):
        self.assertEqual(self.store.peek("never-used"), [])
        self.assertEqual(self.store.session_ids(), [])

    def test_complete(self):
        item = self.store.add("s1", "a")
        done = self.store.complete("s1", item.id)
        self.assertIsNotNone(done)
        self.assertEqual(done.status, STATUS_DONE)

    def test_complete_unknown_id_returns_none(self):
        self.store.add("s1", "a")
        self.assertIsNone(self.store.complete("s1", 999))

    def test_complete_in_other_session_does_not_leak(self):
        item = self.store.add("s1", "a")
        self.assertIsNone(self.store.complete("s2", item.id), "不能跨会话操作待办")
        self.assertEqual(self.store.list("s1")[0].status, STATUS_PENDING)

    def test_list_can_exclude_done(self):
        first = self.store.add("s1", "a")
        self.store.add("s1", "b")
        self.store.complete("s1", first.id)
        self.assertEqual(len(self.store.list("s1")), 2)
        self.assertEqual(len(self.store.list("s1", include_done=False)), 1)

    def test_remove(self):
        item = self.store.add("s1", "a")
        removed = self.store.remove("s1", item.id)
        self.assertIsNotNone(removed)
        self.assertEqual(self.store.list("s1"), [])

    def test_remove_unknown_id_returns_none(self):
        self.assertIsNone(self.store.remove("s1", 5))

    def test_clear(self):
        self.store.add("s1", "a")
        self.store.add("s1", "b")
        self.assertEqual(self.store.clear("s1"), 2)
        self.assertEqual(self.store.list("s1"), [])

    def test_clear_one_session_keeps_other(self):
        self.store.add("s1", "a")
        self.store.add("s2", "b")
        self.store.clear("s1")
        self.assertEqual(self.store.list("s1"), [])
        self.assertEqual(len(self.store.list("s2")), 1)


class TestTodoTool(unittest.TestCase):
    """工具通过 ToolContext 取会话标识，是实现隔离的关键。"""

    def setUp(self):
        self.store = TodoStore()
        self.tool = TodoTool(self.store)

    def _ctx(self, session_id):
        return ToolContext(session_id=session_id)

    def test_contract_fields(self):
        self.assertEqual(self.tool.name, "todo")
        self.assertTrue(self.tool.description)
        actions = self.tool.parameters["properties"]["action"]["enum"]
        self.assertEqual(actions, ["add", "list", "complete", "remove"])
        self.assertEqual(self.tool.parameters["required"], ["action"])

    def test_add_then_list(self):
        added = self.tool.run({"action": "add", "content": "写周报"}, self._ctx("w2"))
        self.assertTrue(added.ok)
        self.assertIn("写周报", added.content)

        listed = self.tool.run({"action": "list"}, self._ctx("w2"))
        self.assertTrue(listed.ok)
        self.assertIn("写周报", listed.content)
        self.assertEqual(listed.data["count"], 1)

    def test_two_windows_do_not_interfere(self):
        """spec 3.6 核心场景：窗口1 记待办不变，窗口2 记待办。"""
        self.tool.run({"action": "add", "content": "窗口1：查天气"}, self._ctx("w1"))
        self.tool.run({"action": "add", "content": "窗口2：写周报"}, self._ctx("w2"))

        listed1 = self.tool.run({"action": "list"}, self._ctx("w1"))
        listed2 = self.tool.run({"action": "list"}, self._ctx("w2"))

        self.assertEqual([i["content"] for i in listed1.data["items"]], ["窗口1：查天气"])
        self.assertEqual([i["content"] for i in listed2.data["items"]], ["窗口2：写周报"])

    def test_resume_session_recalls_earlier_todos(self):
        """spec 3.6：随时回到旧窗口继续对话，状态仍在。"""
        self.tool.run({"action": "add", "content": "记得带伞"}, self._ctx("w1"))
        # 模拟"切换到窗口2 聊天，再切回窗口1"
        self.tool.run({"action": "add", "content": "别的窗口的事"}, self._ctx("w2"))
        resumed = self.tool.run({"action": "list"}, self._ctx("w1"))
        self.assertEqual([i["content"] for i in resumed.data["items"]], ["记得带伞"])

    def test_empty_list_message(self):
        result = self.tool.run({"action": "list"}, self._ctx("fresh"))
        self.assertTrue(result.ok)
        self.assertIn("没有待办", result.content)
        self.assertEqual(result.data["count"], 0)

    def test_add_without_content_fails(self):
        result = self.tool.run({"action": "add"}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_complete_flow(self):
        added = self.tool.run({"action": "add", "content": "倒垃圾"}, self._ctx("s1"))
        todo_id = added.data["item"]["id"]

        done = self.tool.run(
            {"action": "complete", "todo_id": todo_id}, self._ctx("s1")
        )
        self.assertTrue(done.ok)
        self.assertEqual(done.data["item"]["status"], STATUS_DONE)

        listed = self.tool.run({"action": "list"}, self._ctx("s1"))
        self.assertEqual(listed.data["items"][0]["status"], STATUS_DONE)

    def test_complete_without_todo_id_fails(self):
        result = self.tool.run({"action": "complete"}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_complete_unknown_id_gives_actionable_error(self):
        result = self.tool.run({"action": "complete", "todo_id": 7}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error_message)
        self.assertIn("还没有任何待办", result.error_message)

    def test_remove_flow(self):
        added = self.tool.run({"action": "add", "content": "临时项"}, self._ctx("s1"))
        todo_id = added.data["item"]["id"]
        removed = self.tool.run({"action": "remove", "todo_id": todo_id}, self._ctx("s1"))
        self.assertTrue(removed.ok)
        listed = self.tool.run({"action": "list"}, self._ctx("s1"))
        self.assertEqual(listed.data["count"], 0)

    def test_invalid_action_rejected(self):
        result = self.tool.run({"action": "explode"}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_missing_action_rejected(self):
        result = self.tool.run({}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_todo_id_wrong_type_rejected(self):
        result = self.tool.run({"action": "complete", "todo_id": "第一个"}, self._ctx("s1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_include_done_false(self):
        added = self.tool.run({"action": "add", "content": "a"}, self._ctx("s1"))
        self.tool.run({"action": "add", "content": "b"}, self._ctx("s1"))
        self.tool.run(
            {"action": "complete", "todo_id": added.data["item"]["id"]}, self._ctx("s1")
        )
        listed = self.tool.run(
            {"action": "list", "include_done": False}, self._ctx("s1")
        )
        self.assertEqual(listed.data["count"], 1)
        self.assertEqual(listed.data["items"][0]["content"], "b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
