"""``search`` 工具：mock 检索。

对应任务 T010，约束来自 spec.md 3.2：
"允许 mock，返回结构化检索结果"，且 spec 6.2 E3 要求
查询无结果时给出明确的空结果而非报错。

mock 数据源刻意做成可替换的：真实检索只需换掉 ``SearchBackend``，
工具契约与主循环都不受影响（spec 3.3 的可扩展性要求）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import BaseTool, ToolContext, ToolResult, validate_arguments

# 相似度阈值：低于此分数视为无结果。
# 标题命中记 0.4、关键词命中记 0.35，因此阈值取 0.3 的效果是：
# "命中标题或关键词"才算相关，而"仅在关键词上做了子串弱匹配"（0.25）不算。
_MIN_SCORE = 0.3


@dataclass(frozen=True)
class SearchDocument:
    """一条可检索的 mock 文档。"""

    title: str
    snippet: str
    url: str
    keywords: Tuple[str, ...] = ()


#: 内置 mock 语料。内容为 deliberately 简化的示例数据，不代表真实事实。
DEFAULT_DOCUMENTS: Tuple[SearchDocument, ...] = (
    SearchDocument(
        title="Agent 主循环的四个步骤",
        snippet=(
            "一个最小可用的 Agent 循环包含：接收用户输入、判断直接回复还是调用工具、"
            "执行工具、根据工具结果决定继续循环还是返回答案。"
        ),
        url="https://example.com/docs/agent-loop",
        keywords=("agent", "循环", "loop", "主循环", "工具调用", "tool"),
    ),
    SearchDocument(
        title="工具注册机制与参数 Schema",
        snippet=(
            "每个工具需要提供名称、描述和参数 Schema；运行时把注册表中的工具"
            "导出为 LLM 可调用的函数定义，由模型自主决策何时调用。"
        ),
        url="https://example.com/docs/tool-registry",
        keywords=("工具", "注册", "registry", "schema", "参数", "function"),
    ),
    SearchDocument(
        title="Session 与多窗口隔离",
        snippet=(
            "不同窗口应是彼此独立的 session；消息、工具结果与会话级状态都按 "
            "session 隔离存储，用户可随时回到任一窗口继续对话。"
        ),
        url="https://example.com/docs/session",
        keywords=("session", "会话", "窗口", "隔离", "多轮", "上下文"),
    ),
    SearchDocument(
        title="上下文压缩的常见做法",
        snippet=(
            "上下文过长时可按轮次或字符数截断，或对较早的历史做摘要；"
            "重点是保留最近对话与关键工具结果。"
        ),
        url="https://example.com/docs/context",
        keywords=("上下文", "context", "压缩", "截断", "摘要", "token"),
    ),
    SearchDocument(
        title="广州天气与出行建议",
        snippet="广州属亚热带季风气候，夏季高温多雨，出行建议随身带伞并注意补水。",
        url="https://example.com/docs/weather-guangzhou",
        keywords=("广州", "天气", "气候", "出行", "guangzhou", "weather"),
    ),
    SearchDocument(
        title="周报的常见结构",
        snippet=(
            "周报通常包含本周完成事项、进行中事项、遇到的问题与下周计划四部分。"
        ),
        url="https://example.com/docs/weekly-report",
        keywords=("周报", "报告", "weekly", "总结", "计划"),
    ),
)


class SearchBackend:
    """基于关键词匹配的 mock 检索后端。

    评分规则：标题命中权重高于关键词命中，关键词命中高于摘要命中。
    真实检索只需提供同名 ``search`` 方法的替代实现。
    """

    def __init__(self, documents: Sequence[SearchDocument] = DEFAULT_DOCUMENTS) -> None:
        self._documents = tuple(documents)

    @property
    def documents(self) -> Tuple[SearchDocument, ...]:
        return self._documents

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        normalized = query.strip().lower()
        if not normalized:
            return []
        # 以空白与常见标点切分为检索词
        terms = [t for t in _tokenize(normalized) if t]
        if not terms:
            return []

        scored: List[Tuple[float, SearchDocument]] = []
        for document in self._documents:
            score = _score_document(document, normalized, terms)
            if score >= _MIN_SCORE:
                scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].title))
        return [
            {
                "title": document.title,
                "snippet": document.snippet,
                "url": document.url,
                "score": round(score, 3),
            }
            for score, document in scored[:limit]
        ]


def _tokenize(text: str) -> List[str]:
    for separator in "，。！？、；：,.!?;:()（）[]【】\"'“”‘’/\\|+-_=*&^%$#@~`<>":
        text = text.replace(separator, " ")
    return [token for token in text.split() if token]


def _score_document(
    document: SearchDocument, normalized_query: str, terms: Sequence[str]
) -> float:
    """给文档打分。

    评分原则：只有**标题或关键词**命中才算"相关"；摘要命中仅作为
    已相关基础上的加权。否则一句恰好出现在摘要里的无关查询会把
    毫不相干的文档拉进结果（spec 6.2 E3 要求无结果时明确返回空）。
    """
    title = document.title.lower()
    snippet = document.snippet.lower()
    keywords = tuple(k.lower() for k in document.keywords)

    strong = 0.0
    weak = 0.0
    if normalized_query in title:
        strong += 0.8
    for term in terms:
        if term in title:
            strong += 0.4
        if term in keywords:
            strong += 0.35
        elif any(term in keyword or keyword in term for keyword in keywords):
            strong += 0.25
        if term in snippet:
            weak += 0.15

    if strong <= 0.0:
        return 0.0
    return strong + weak


SEARCH_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "检索关键词或短语，例如 'Agent 主循环'、'session 隔离'。",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "最多返回多少条结果，默认 5，取值 1~20。",
            "minimum": 1,
            "maximum": 20,
            "default": 5,
        },
    },
    "required": ["query"],
}


class SearchTool(BaseTool):
    """mock 检索工具。"""

    name = "search"
    description = (
        "检索资料库并返回相关文档片段（标题、摘要、链接）。当用户询问概念、"
        "文档内容、背景知识或需要依据资料回答时使用。"
        "每条结果都带 url 与 score，score 越高越相关。"
    )
    parameters = SEARCH_PARAMETERS

    def __init__(self, backend: Optional[SearchBackend] = None) -> None:
        self._backend = backend or SearchBackend()

    @property
    def backend(self) -> SearchBackend:
        return self._backend

    def run(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        checked = validate_arguments(self.parameters, arguments)
        if not checked.ok:
            return checked

        query = checked.data["query"]
        limit = checked.data.get("limit", 5)
        matches = self._backend.search(query, limit=limit)

        if not matches:
            # spec 6.2 E3：无结果要返回明确的空结果，而不是报错
            return ToolResult.success(
                content=f'未检索到与 "{query}" 相关的结果。可以换用更通用的关键词重试。',
                data={"query": query, "count": 0, "results": []},
            )

        lines = [f'检索 "{query}" 命中 {len(matches)} 条：']
        for index, item in enumerate(matches, start=1):
            lines.append(
                f"{index}. {item['title']}（score={item['score']}）\n"
                f"   {item['snippet']}\n"
                f"   {item['url']}"
            )
        return ToolResult.success(
            content="\n".join(lines),
            data={"query": query, "count": len(matches), "results": matches},
        )
