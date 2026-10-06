"""T010 验证：search 工具（mock 检索）。

覆盖 spec 3.2（mock、结构化结果）与 6.2 E3（无结果时明确返回而非报错）。
"""

from __future__ import annotations

import unittest

from tests.helpers import AgentTestCase  # noqa: F401
from agent.tools.base import ToolContext, ToolErrorCode
from agent.tools.search import (
    DEFAULT_DOCUMENTS,
    SearchBackend,
    SearchDocument,
    SearchTool,
)


class TestSearchBackend(unittest.TestCase):
    def setUp(self):
        self.backend = SearchBackend()

    def test_keyword_hit_returns_structured_results(self):
        results = self.backend.search("session 隔离")
        self.assertTrue(results)
        first = results[0]
        self.assertEqual(
            set(first), {"title", "snippet", "url", "score"}
        )
        self.assertIsInstance(first["score"], float)
        self.assertIn("session", first["title"].lower())

    def test_results_sorted_by_score_descending(self):
        results = self.backend.search("上下文 压缩")
        scores = [item["score"] for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_limit_is_respected(self):
        results = self.backend.search("工具", limit=1)
        self.assertLessEqual(len(results), 1)

    def test_no_match_returns_empty_list(self):
        """spec 6.2 E3：无结果需明确返回空，而不是报错。"""
        self.assertEqual(self.backend.search("zzzz-不存在的词-qqqq"), [])

    def test_empty_query_returns_empty_list(self):
        self.assertEqual(self.backend.search(""), [])
        self.assertEqual(self.backend.search("   "), [])

    def test_punctuation_only_query_returns_empty(self):
        self.assertEqual(self.backend.search("，。！？"), [])

    def test_mock_corpus_is_non_empty_and_well_formed(self):
        self.assertTrue(DEFAULT_DOCUMENTS)
        for document in DEFAULT_DOCUMENTS:
            self.assertTrue(document.title)
            self.assertTrue(document.snippet)
            self.assertTrue(document.url.startswith("http"))

    def test_backend_is_replaceable(self):
        """spec 3.3：mock 数据源可替换，不影响工具契约。"""
        custom = SearchBackend(
            [
                SearchDocument(
                    title="部署手册",
                    snippet="灰度发布与回滚流程。",
                    url="https://x.invalid/deploy",
                    keywords=("部署", "灰度", "deploy"),
                )
            ]
        )
        self.assertEqual(custom.search("完全不相关的词"), [])
        results = custom.search("部署")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "部署手册")

    def test_snippet_only_match_is_not_treated_as_relevant(self):
        """摘要命中不算相关，避免无关查询把不相干文档拉进结果（spec 6.2 E3）。"""
        backend = SearchBackend(
            [
                SearchDocument(
                    title="部署手册",
                    snippet="灰度发布与回滚流程。",
                    url="https://x.invalid/deploy",
                )
            ]
        )
        self.assertEqual(backend.search("灰度发布"), [])


class TestSearchTool(unittest.TestCase):
    def setUp(self):
        self.tool = SearchTool()
        self.context = ToolContext(session_id="s1")

    def test_contract_fields(self):
        self.assertEqual(self.tool.name, "search")
        self.assertTrue(self.tool.description)
        self.assertEqual(self.tool.parameters["required"], ["query"])
        self.assertIn("limit", self.tool.parameters["properties"])

    def test_run_success_returns_readable_content(self):
        result = self.tool.run({"query": "Agent 主循环"}, self.context)
        self.assertTrue(result.ok)
        self.assertIn("命中", result.content)
        self.assertGreaterEqual(result.data["count"], 1)
        self.assertEqual(len(result.data["results"]), result.data["count"])

    def test_run_default_limit_applied(self):
        result = self.tool.run({"query": "工具"}, self.context)
        self.assertTrue(result.ok)
        self.assertLessEqual(result.data["count"], 5)

    def test_run_no_result_is_success_not_error(self):
        result = self.tool.run({"query": "zzzz-不存在-qqqq"}, self.context)
        self.assertTrue(result.ok, "无结果不应是错误（spec 6.2 E3）")
        self.assertEqual(result.data["count"], 0)
        self.assertIn("未检索到", result.content)

    def test_run_missing_query(self):
        result = self.tool.run({}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.MISSING_PARAMETER.value)

    def test_run_blank_query_rejected_by_min_length(self):
        result = self.tool.run({"query": ""}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_run_wrong_type(self):
        result = self.tool.run({"query": 123}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_run_limit_out_of_range(self):
        result = self.tool.run({"query": "工具", "limit": 100}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS.value)

    def test_run_limit_wrong_type(self):
        result = self.tool.run({"query": "工具", "limit": "很多"}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_run_limit_not_integer(self):
        result = self.tool.run({"query": "工具", "limit": 2.5}, self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.INVALID_TYPE.value)

    def test_results_data_matches_content(self):
        result = self.tool.run({"query": "周报"}, self.context)
        self.assertTrue(result.ok)
        first = result.data["results"][0]
        self.assertIn(first["title"], result.content)

    def test_llm_schema_shape(self):
        schema = self.tool.to_llm_schema()
        self.assertEqual(schema["function"]["name"], "search")
        self.assertEqual(schema["function"]["parameters"]["type"], "object")


if __name__ == "__main__":
    unittest.main(verbosity=2)
