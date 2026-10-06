"""最小可用 Agent Runtime。

本包从零实现 Agent 的核心运行时，不依赖任何现成 Agent 框架：

- 主循环（接收输入 → 决策 → 调用工具 → 终止判定）
- 工具注册与调度（name / description / parameters Schema）
- LLM 输出解析（思考过程 / 工具调用 / 最终答案）
- session 管理与 context 有效管理
- 异常处理与调用 trace

模块职责对应 spec.md 第 3.0 节的分层职责表。
"""

__version__ = "0.1.0"
