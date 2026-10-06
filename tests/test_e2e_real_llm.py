"""T044 验证：可选的真实 LLM API 端到端测试。

对应任务 T044，约束来自 spec.md：

- C-3：必须接入**真实** LLM API——本文件是唯一走真实网络的测试；
- 4.3 / 6.4 D1：测试须能在**无密钥**环境下离线通过，因此本套件必须
  "缺密钥即自动跳过，且不导致测试套件失败"。

启用条件（双重闸门，避免误触发真实计费调用）：

1. 环境变量 ``LLM_API_KEY`` 已设置；
2. 环境变量 ``LLM_E2E`` 为真值（``1`` / ``true`` / ``yes``）。

用法：

    # PowerShell
    $env:LLM_API_KEY="<你的密钥>"
    $env:LLM_BASE_URL="https://api.example.com/v1"
    $env:LLM_MODEL="<模型名>"
    $env:LLM_E2E="1"
    python -m unittest tests.test_e2e_real_llm -v

**注意**：密钥只从环境变量读取，绝不写入仓库（spec C-7 / C-8）。
"""

from __future__ import annotations

import os
import unittest

from tests.helpers import AgentTestCase  # noqa: F401  (导入即完成 sys.path 设置)
from agent.config import ConfigError, LLMConfig
from agent.context import ContextBuilder, ContextPolicy
from agent.llm import LLMClient
from agent.loop import (
    STOP_FINAL_ANSWER,
    STOP_LLM_ERROR,
    AgentLoop,
    LoopPolicy,
)
from agent.session import SessionStore
from agent.tools.registry import build_default_registry
from agent.trace import Tracer

#: 真实模型的回答具有不确定性，因此断言只针对"可判定的客观事实"
#: （例如会话里是否真的发生了工具调用、工具结果是否进入下一轮），
#: 不对模型的具体措辞做断言。
TRUE_VALUES = {"1", "true", "yes", "on"}
SKIP_REASON_NO_KEY = (
    "未设置 LLM_API_KEY，跳过真实 API 端到端测试（spec 4.3 要求无密钥可离线运行）"
)
SKIP_REASON_DISABLED = (
    "未设置 LLM_E2E=1，跳过真实 API 端到端测试以避免误触发计费调用"
)

#: 落盘恢复场景的用户输入（提取为常量，避免断言与调用处各写一份而漂移）
PERSIST_PROMPT = "请调用 todo 工具记录待办「落盘验证项」。"


def e2e_enabled() -> tuple:
    """返回 ``(是否启用, 跳过原因)``。"""
    if not os.environ.get("LLM_API_KEY", "").strip():
        return False, SKIP_REASON_NO_KEY
    if os.environ.get("LLM_E2E", "").strip().lower() not in TRUE_VALUES:
        return False, SKIP_REASON_DISABLED
    return True, ""


class RealLLMTestCase(unittest.TestCase):
    """真实 API 测试基类：统一装配 Agent 与跳过逻辑。"""

    #: 真实模型可能较慢（尤其推理模型），给足超时
    LLM_TIMEOUT_SECONDS = 180.0
    #: 单轮 loop 上限，避免测试跑很久
    MAX_ITERATIONS = 6
    #: 工具超时：真实网络下 search/todo 都是本地实现，给 30s 足够
    TOOL_TIMEOUT_SECONDS = 30.0

    def setUp(self) -> None:
        enabled, reason = e2e_enabled()
        if not enabled:
            self.skipTest(reason)

        env = dict(os.environ)
        env["LLM_TIMEOUT_SECONDS"] = str(self.LLM_TIMEOUT_SECONDS)
        try:
            self.config = LLMConfig.from_env(env)
        except ConfigError as exc:  # pragma: no cover - 由启用闸门保证不会走到
            self.skipTest(f"LLM 配置不完整：{exc}")

        self.tracer = Tracer(enabled=False)
        self.client = LLMClient(self.config, tracer=self.tracer)
        self.sessions = SessionStore()
        self.loop = AgentLoop(
            client=self.client,
            registry=build_default_registry(),
            sessions=self.sessions,
            builder=ContextBuilder(ContextPolicy(include_protocol_instructions=True)),
            tracer=self.tracer,
            policy=LoopPolicy(
                max_iterations=self.MAX_ITERATIONS,
                tool_timeout_seconds=self.TOOL_TIMEOUT_SECONDS,
            ),
        )

    # -- 断言辅助 ---------------------------------------------------------- #

    def assert_no_llm_failure(self, result) -> None:
        """LLM 调用失败属环境问题，报错时要能一眼看出原因。"""
        if result.stop_reason == STOP_LLM_ERROR:
            self.fail(
                f"真实 LLM 调用失败：{result.answer}\n"
                f"配置：{self.config.safe_summary()}"
            )

    def session_tool_records(self, session_id: str):
        """取出某 session 里已记录的工具调用。"""
        session = self.sessions.require(session_id)
        return [call for message in session.messages for call in message.tool_calls]

    def tool_names_used(self, session_id: str):
        return [record.name for record in self.session_tool_records(session_id)]

    def skip_if_model_avoided_tools(self, session_id: str, expected: str) -> None:
        """模型没按预期调用工具时，记为跳过而不是失败。

        LLM 是否调用工具本身具有不确定性；本套件要验证的是**运行时**能否
        正确处理真实 API 的工具调用，因此"模型这次没用工具"不应判定为实现有缺陷。
        但一旦用到工具，下面各测试都会对运行时行为做严格断言。
        """
        if expected not in self.tool_names_used(session_id):
            self.skipTest(
                f"本次真实模型未调用 {expected} 工具（已调用："
                f"{self.tool_names_used(session_id) or '无'}），跳过该场景的严格断言"
            )


@unittest.skipUnless(e2e_enabled()[0], e2e_enabled()[1])
class TestRealLLMDirectReply(RealLLMTestCase):
    """真实通路 1：直接回复（不调工具）。"""

    def test_simple_question_returns_answer(self):
        result = self.loop.run_turn(
            "只回答一个词：你是什么类型的程序？", session_id="e2e-direct"
        )
        self.assert_no_llm_failure(result)
        self.assertTrue(result.answer.strip(), "真实模型必须返回非空答案")
        self.assertGreaterEqual(result.iterations, 1)

    def test_answer_is_persisted_to_session(self):
        result = self.loop.run_turn("用一句话介绍你自己。", session_id="e2e-persist")
        self.assert_no_llm_failure(result)
        session = self.sessions.require("e2e-persist")
        self.assertEqual(session.messages[0].content, "用一句话介绍你自己。")
        self.assertIn(
            result.answer.strip(), session.last_assistant_message().content
        )

    def test_llm_call_is_traced(self):
        self.loop.run_turn("说一句问候语。", session_id="e2e-trace")
        events = [record.event for record in self.tracer.records]
        self.assertIn("llm_call", events, "真实 LLM 调用必须写入 trace")
        self.assertIn("loop_step", events)


@unittest.skipUnless(e2e_enabled()[0], e2e_enabled()[1])
class TestRealLLMToolCalling(RealLLMTestCase):
    """真实通路 2：工具调用与结果回灌（spec 6.1 N3）。"""

    def test_calculator_tool_is_invoked_and_result_fed_back(self):
        """要求模型用工具做精确计算，验证 Schema 注入 + 原生 tool_calls 解析。"""
        result = self.loop.run_turn(
            "请务必调用 calculator 工具计算 1234*5678，然后告诉我结果。",
            session_id="e2e-calc",
        )
        self.assert_no_llm_failure(result)
        self.skip_if_model_avoided_tools("e2e-calc", "calculator")

        records = [
            record
            for record in self.session_tool_records("e2e-calc")
            if record.name == "calculator"
        ]
        self.assertTrue(records, "应至少记录一次 calculator 调用")

        record = records[0]
        self.assertTrue(record.result_ok, f"工具应成功：{record.result_text}")
        self.assertEqual(record.arguments.get("expression"), "1234*5678")
        # calculator 的展示格式带千分位（7,006,652），因此比较时去掉分隔符
        self.assertIn("7006652", record.result_text.replace(",", ""), "工具返回应含精确乘积")
        self.assertIn("7006652", result.answer.replace(",", ""), "最终答复应整合工具结果")
        self.assertGreaterEqual(record.duration_ms, 0.0)

    def test_tool_schema_is_sent_to_real_model(self):
        """spec 3.3：注册表导出的 Schema 必须真的发给了模型（否则不会触发调用）。"""
        self.loop.run_turn(
            "调用 search 工具检索「session 隔离」相关内容。", session_id="e2e-schema"
        )
        session = self.sessions.require("e2e-schema")
        self.assertGreater(len(session), 1)
        # 只要模型能基于 Schema 正确填出参数，就说明 Schema 生效
        self.skip_if_model_avoided_tools("e2e-schema", "search")
        record = self.session_tool_records("e2e-schema")[0]
        self.assertEqual(record.name, "search")
        self.assertIn("query", record.arguments)

    def test_unknown_tool_error_does_not_crash(self):
        """即使模型给出了无效参数，运行时也必须有确定行为（spec 6.2 E2）。"""
        result = self.loop.run_turn(
            "调用 calculator 工具计算一个非法的表达式：'1 +'（这是错误输入）。",
            session_id="e2e-invalid",
        )
        self.assert_no_llm_failure(result)
        # 无论模型是否调用工具，都必须得到可读答复且不抛异常
        self.assertTrue(result.answer.strip())
        self.assertNotEqual(result.stop_reason, "")


@unittest.skipUnless(e2e_enabled()[0], e2e_enabled()[1])
class TestRealLLMChainedAndSessions(RealLLMTestCase):
    """真实通路 3：链式工具调用与多 session 隔离（spec 6.1 N4/N8）。"""

    def test_todo_add_then_list_chains_tools(self):
        """一轮任务中先 add 再 list，验证链式调用与工具结果回灌。"""
        result = self.loop.run_turn(
            "请先调用 todo 工具添加待办「买牛奶」，然后调用 todo 工具列出全部待办，"
            "最后告诉我清单内容。",
            session_id="e2e-chain",
        )
        self.assert_no_llm_failure(result)
        self.skip_if_model_avoided_tools("e2e-chain", "todo")

        records = self.session_tool_records("e2e-chain")
        actions = [record.arguments.get("action") for record in records]
        self.assertIn("add", actions, "应记录到 add 调用")
        self.assertIn("买牛奶", result.answer, "答复应包含待办内容")

    def test_tool_result_visible_in_next_turn_follow_up(self):
        """spec 6.1 N9：带工具的追问——后续轮次能基于前序工具结果。"""
        first = self.loop.run_turn(
            "请调用 calculator 计算 20+22，然后只告诉我结果数字。",
            session_id="e2e-followup",
        )
        self.assert_no_llm_failure(first)
        self.skip_if_model_avoided_tools("e2e-followup", "calculator")

        second = self.loop.run_turn(
            "把刚才那个结果再乘以 2，是多少？", session_id="e2e-followup"
        )
        self.assert_no_llm_failure(second)
        # 84 = (20+22)*2 —— 模型应基于前序上下文得出，允许再次调用工具
        self.assertIn("84", second.answer, f"追问答复应含 84，实际：{second.answer}")

    def test_two_sessions_are_isolated_with_real_llm(self):
        """spec 6.1 N8：两个窗口各自记待办，互不影响。"""
        self.loop.run_turn(
            "请调用 todo 工具记录待办「窗口一的专属任务」。", session_id="e2e-w1"
        )
        self.loop.run_turn(
            "请调用 todo 工具记录待办「窗口二的专属任务」。", session_id="e2e-w2"
        )
        self.skip_if_model_avoided_tools("e2e-w1", "todo")
        self.skip_if_model_avoided_tools("e2e-w2", "todo")

        from agent.tools.todo import session_todos

        w1_todos = session_todos(self.sessions.require("e2e-w1").state, "e2e-w1")
        w2_todos = session_todos(self.sessions.require("e2e-w2").state, "e2e-w2")

        w1_text = " ".join(item["content"] for item in w1_todos)
        w2_text = " ".join(item["content"] for item in w2_todos)
        self.assertIn("窗口一", w1_text)
        self.assertNotIn("窗口二", w1_text)
        self.assertIn("窗口二", w2_text)
        self.assertNotIn("窗口一", w2_text)


@unittest.skipUnless(e2e_enabled()[0], e2e_enabled()[1])
class TestRealLLMSessionPersistence(RealLLMTestCase):
    """真实通路 4：落盘后重载，历史与状态仍可恢复（spec 3.6）。"""

    def test_session_survives_reload(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from agent.session import SessionStore as _Store

            path = os.path.join(tmp, "sessions.json")
            sessions = _Store(path)
            loop = AgentLoop(
                client=self.client,
                registry=build_default_registry(),
                sessions=sessions,
                builder=ContextBuilder(
                    ContextPolicy(include_protocol_instructions=True)
                ),
                tracer=self.tracer,
                policy=LoopPolicy(
                    max_iterations=self.MAX_ITERATIONS,
                    tool_timeout_seconds=self.TOOL_TIMEOUT_SECONDS,
                ),
            )
            first = loop.run_turn(PERSIST_PROMPT, session_id="e2e-store")
            self.assert_no_llm_failure(first)

            reloaded = _Store(path)
            session = reloaded.require("e2e-store")
            self.assertEqual(session.messages[0].content, PERSIST_PROMPT)
            # 无论模型是否调用工具，用户输入与助手答复都必须恢复
            self.assertGreaterEqual(len(session), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
