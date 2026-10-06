"""T028–T032 / T042 / T043 验证：主循环的异常路径与边界情况。

覆盖 spec 6.2（E1/E2/E4/E5/E6/E7/E8）与 spec 6.3（B1/B5/B7/B8）。
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from tests.helpers import (
    RecordingTool,
    ScriptedLLM,
    native_tool_call,
    protocol_tool_calls,
    scripted_json,
)
from agent.config import ConfigError, LLMConfig
from agent.context import ContextBuilder, ContextPolicy
from agent.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent.loop import (
    EVENT_LIMIT,
    STOP_EMPTY_INPUT,
    STOP_LLM_ERROR,
    STOP_MAX_ITERATIONS,
    STOP_NO_PROGRESS,
    STOP_PARSE_ERROR,
    AgentLoop,
    LoopPolicy,
    build_agent,
)
from agent.session import SessionStore
from agent.tools.base import BaseTool, ToolContext, ToolErrorCode, ToolResult
from agent.tools.registry import ToolRegistry, build_default_registry
from agent.trace import Tracer


class SlowTool(BaseTool):
    """故意休眠的工具，用于验证执行超时保护（spec 6.2 E8）。"""

    name = "slow"
    description = "休眠指定秒数后返回。"
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {"type": "number", "description": "休眠秒数", "default": 1}
        },
    }

    def run(self, arguments, context):  # type: ignore[override]
        import time as _time

        _time.sleep(float(arguments.get("seconds") or 1))
        return ToolResult.success(content="终于醒了")


class FlakyThenOkTool(BaseTool):
    """第一次抛异常，之后正常——验证错误回灌后模型能自我修正。"""

    name = "flaky"
    description = "首次调用失败。"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments, context):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("第一次故意失败")
        return ToolResult.success(content="第二次成功")


def make_loop(script, tools=None, policy=None, sessions=None, tracer=None):
    llm = ScriptedLLM(script)
    registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
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


class TestEmptyInput(unittest.TestCase):
    """spec 6.3 B5：空输入不触发 LLM 调用。"""

    def test_empty_string_short_circuits(self):
        loop, llm = make_loop([scripted_json({"answer": "不该被调用"})])
        result = loop.run_turn("", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_EMPTY_INPUT)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(llm.calls), 0, "空输入不得触发 LLM 调用")
        self.assertIn("请输入有效内容", result.answer)

    def test_whitespace_only_short_circuits(self):
        loop, llm = make_loop([])
        for text in ("   ", "\n\t ", "\u3000"):
            with self.subTest(text=repr(text)):
                result = loop.run_turn(text, session_id="w1")
                self.assertEqual(result.stop_reason, STOP_EMPTY_INPUT)
        self.assertEqual(len(llm.calls), 0)

    def test_none_input_short_circuits(self):
        loop, llm = make_loop([])
        result = loop.run_turn(None, session_id="w1")  # type: ignore[arg-type]
        self.assertEqual(result.stop_reason, STOP_EMPTY_INPUT)
        self.assertEqual(len(llm.calls), 0)

    def test_empty_input_not_recorded_as_turn(self):
        loop, _llm = make_loop([])
        loop.run_turn("  ", session_id="w1")
        self.assertEqual(loop.sessions.require("w1").current_turn(), 0)


class TestLLMFailures(unittest.TestCase):
    """spec 6.2 E4：LLM 失败时终止并返回可解释错误。"""

    def test_timeout_produces_explainable_result(self):
        loop, _llm = make_loop([LLMTimeoutError("LLM 请求超时（>5.0s）")])
        result = loop.run_turn("你好", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_LLM_ERROR)
        self.assertTrue(result.degraded)
        self.assertIn("超时", result.answer)
        self.assertIn("调用模型失败", result.answer)

    def test_rate_limit_produces_explainable_result(self):
        loop, _llm = make_loop([LLMRateLimitError("LLM 限流（HTTP 429）")])
        result = loop.run_turn("你好", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_LLM_ERROR)
        self.assertIn("限流", result.answer)

    def test_auth_error_produces_explainable_result(self):
        loop, _llm = make_loop([LLMAuthError("LLM 鉴权失败（HTTP 401）")])
        result = loop.run_turn("你好", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_LLM_ERROR)
        self.assertIn("鉴权失败", result.answer)

    def test_no_uncaught_exception_escapes(self):
        loop, _llm = make_loop([LLMTimeoutError("boom")])
        try:
            result = loop.run_turn("你好", session_id="w1")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"主循环不应抛出未捕获异常，实际抛出：{exc!r}")
        self.assertTrue(result.answer)

    def test_error_is_traced(self):
        tracer = Tracer(enabled=False)
        loop, _llm = make_loop([LLMTimeoutError("boom")], tracer=tracer)
        loop.run_turn("你好", session_id="w1")
        names = [r.name for r in tracer.records]
        self.assertIn("step2_llm_failed", names)

    def test_failure_after_tool_success_still_reports(self):
        """工具成功但随后 LLM 失败：也要明确结束原因。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            LLMTimeoutError("第二次调用超时"),
        ]
        loop, _llm = make_loop(script, tools=[RecordingTool()])
        result = loop.run_turn("记录 A", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_LLM_ERROR)
        self.assertEqual(result.tool_calls, 1)


class TestToolFailures(unittest.TestCase):
    """spec 6.2 E1/E2/E3/E8：工具侧失败都可恢复。"""

    def test_unknown_tool_error_is_fed_back_and_model_recovers(self):
        """spec 6.2 E1：模型调用不存在的工具 → 错误回灌 → 自我修正。"""
        script = [
            {"tool_calls": [native_tool_call("no_such_tool", {})]},
            scripted_json({"answer": "抱歉，我换个方式回答"}),
        ]
        loop, llm = make_loop(script, tools=[RecordingTool()])
        result = loop.run_turn("做点什么", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "抱歉，我换个方式回答")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("unknown_tool", tool_messages[0]["content"])

    def test_missing_parameter_error_is_fed_back(self):
        """spec 6.2 E2：缺参 → 结构化错误回灌。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {})]},
            scripted_json({"answer": "已修正"}),
        ]
        loop, llm = make_loop(script, tools=[RecordingTool()])
        result = loop.run_turn("记录", session_id="w1")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("missing_parameter", tool_messages[0]["content"])
        self.assertTrue(result.ok)

    def test_wrong_type_error_is_fed_back(self):
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": 123})]},
            scripted_json({"answer": "已修正"}),
        ]
        loop, llm = make_loop(script, tools=[RecordingTool()])
        loop.run_turn("记录", session_id="w1")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("invalid_type", tool_messages[0]["content"])

    def test_tool_internal_exception_is_normalized(self):
        """spec 6.2 E3：工具内部异常不冒泡。"""
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"})]},
            scripted_json({"answer": "工具坏了，我直接回答"}),
        ]
        loop, llm = make_loop(script, tools=[RecordingTool(fail=True)])
        try:
            result = loop.run_turn("记录 A", session_id="w1")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"工具异常不应冒泡：{exc!r}")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("tool_error", tool_messages[0]["content"])
        self.assertIn("模拟工具内部故障", tool_messages[0]["content"])
        self.assertTrue(result.ok)

    def test_malformed_arguments_are_reported_not_crashed(self):
        """参数 JSON 截断 → 按参数错误回灌（spec 3.4 与 6.2 E2 衔接）。"""
        script = [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "recorder", "arguments": '{"value":'},
                    }
                ]
            },
            scripted_json({"answer": "参数我写错了"}),
        ]
        loop, llm = make_loop(script, tools=[RecordingTool()])
        result = loop.run_turn("记录", session_id="w1")
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn("参数解析失败", tool_messages[0]["content"])
        self.assertTrue(result.ok)

    def test_tool_timeout_does_not_block_loop(self):
        """spec 6.2 E8：工具超时被归一化，loop 继续。"""
        script = [
            {"tool_calls": [native_tool_call("slow", {"seconds": 2})]},
            scripted_json({"answer": "工具太慢了，先这样"}),
        ]
        loop, llm = make_loop(
            script,
            tools=[SlowTool()],
            policy=LoopPolicy(tool_timeout_seconds=0.15),
        )
        result = loop.run_turn("跑个慢工具", session_id="w1")

        self.assertTrue(result.ok)
        tool_messages = [m for m in llm.messages_sent(1) if m["role"] == "tool"]
        self.assertIn(ToolErrorCode.TOOL_TIMEOUT.value, tool_messages[0]["content"])
        self.assertIn("超过 0.15s", tool_messages[0]["content"])

    def test_flaky_tool_recovers_on_retry(self):
        """错误回灌后模型重试同一个工具 → 第二次成功。"""
        flaky = FlakyThenOkTool()
        script = [
            {"tool_calls": [native_tool_call("flaky", {}, "c1")]},
            {"tool_calls": [native_tool_call("flaky", {}, "c2")]},
            scripted_json({"answer": "第二次成功了"}),
        ]
        loop, _llm = make_loop(script, tools=[flaky])
        result = loop.run_turn("调用它", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(flaky.calls, 2)
        self.assertEqual(result.tool_calls, 2)

    def test_timeout_result_recorded_in_session(self):
        script = [
            {"tool_calls": [native_tool_call("slow", {"seconds": 2})]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(
            script, tools=[SlowTool()], policy=LoopPolicy(tool_timeout_seconds=0.15)
        )
        loop.run_turn("跑个慢工具", session_id="w1")
        records = [
            call
            for message in loop.sessions.require("w1").messages
            for call in message.tool_calls
        ]
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].result_ok)
        self.assertEqual(records[0].error_code, ToolErrorCode.TOOL_TIMEOUT.value)


class TestMaxIterations(unittest.TestCase):
    """spec 6.3 B1 / C-6：轮次上限。"""

    def test_limit_stops_endless_tool_calls(self):
        tool = RecordingTool()
        # 脚本远多于上限，模型"永不收敛"
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": str(i)}, f"c{i}")]}
            for i in range(20)
        ]
        loop, llm = make_loop(
            script, tools=[tool], policy=LoopPolicy(max_iterations=3)
        )
        result = loop.run_turn("无休止地调用工具", session_id="w1")

        self.assertEqual(result.stop_reason, STOP_MAX_ITERATIONS)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(len(tool.invocations), 3, "不得超过轮次上限")
        self.assertEqual(len(llm.calls), 3)

    def test_limit_answer_explains_and_includes_partial_results(self):
        tool = RecordingTool()
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "部分A"}, f"c{i}")]}
            for i in range(10)
        ]
        loop, _llm = make_loop(
            script, tools=[tool], policy=LoopPolicy(max_iterations=2)
        )
        result = loop.run_turn("停不下来", session_id="w1")

        self.assertIn("最大循环次数", result.answer)
        self.assertIn("2 次", result.answer)
        self.assertIn("部分结果", result.answer)
        self.assertIn("recorded:部分A", result.answer)
        self.assertIn("拆得更具体", result.answer)

    def test_limit_does_not_crash_without_tool_results(self):
        script = [
            protocol_tool_calls([{"name": "recorder", "arguments": {"value": "x"}}])
            for _ in range(5)
        ]
        loop, _llm = make_loop(
            script, tools=[RecordingTool()], policy=LoopPolicy(max_iterations=2)
        )
        result = loop.run_turn("跑", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_MAX_ITERATIONS)
        self.assertTrue(result.answer)

    def test_limit_emits_limit_event(self):
        seen = []
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "x"}, f"c{i}")]}
            for i in range(5)
        ]
        loop, _llm = make_loop(
            script, tools=[RecordingTool()], policy=LoopPolicy(max_iterations=2)
        )
        loop.run_turn("跑", session_id="w1", observer=lambda e, p: seen.append(e))
        self.assertIn(EVENT_LIMIT, seen)

    def test_limit_result_written_to_session(self):
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "x"}, f"c{i}")]}
            for i in range(5)
        ]
        loop, _llm = make_loop(
            script, tools=[RecordingTool()], policy=LoopPolicy(max_iterations=2)
        )
        result = loop.run_turn("跑", session_id="w1")
        self.assertEqual(
            loop.sessions.require("w1").last_assistant_message().content, result.answer
        )


class TestNoProgress(unittest.TestCase):
    """spec 6.3 B8：模型坚持不调用工具 / 反复无效时的收敛保护。"""

    def test_repeated_empty_responses_stop(self):
        """空响应（无工具也无答案）反复出现 → 兜底停止，不进入死循环。"""
        script = [scripted_json({"thought": "只有思考"}) for _ in range(5)]
        loop, llm = make_loop(script, policy=LoopPolicy(max_no_progress_rounds=2))
        result = loop.run_turn("随便问问", session_id="w1")

        self.assertEqual(result.stop_reason, STOP_PARSE_ERROR)
        self.assertEqual(len(llm.calls), 2, "第 1 次 + 1 次重试后应停止")
        self.assertIn("无法解析", result.answer)

    def test_empty_content_responses_use_parse_error_path(self):
        script = ["", "", ""]
        loop, llm = make_loop(script, policy=LoopPolicy(max_no_progress_rounds=1))
        result = loop.run_turn("你好", session_id="w1")
        self.assertIn(result.stop_reason, (STOP_PARSE_ERROR, STOP_NO_PROGRESS))
        self.assertLessEqual(len(llm.calls), 3)

    def test_model_refuses_tools_and_answers_directly(self):
        """spec 6.3 B8：模型拒绝调用工具、直接回答 → 正常收敛，不空转。"""
        script = [scripted_json({"thought": "不需要工具", "answer": "直接回答你"})]
        loop, llm = make_loop(script, tools=build_default_registry())
        result = loop.run_turn("你好", session_id="w1")
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "直接回答你")
        self.assertEqual(len(llm.calls), 1)

    def test_parse_failure_can_be_retried_once_then_success(self):
        """spec 3.4：解析失败 → 重新提示 → 第二次成功。"""
        script = [scripted_json({"thought": "只有思考"}), scripted_json({"answer": "这次对了"})]
        loop, llm = make_loop(script, policy=LoopPolicy(max_parse_retries=1))
        result = loop.run_turn("你好", session_id="w1")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "这次对了")
        self.assertEqual(len(llm.calls), 2)
        # 重新提示必须真的发出去
        retry_messages = llm.messages_sent(1)
        self.assertTrue(
            any("无法被解析" in (m["content"] or "") for m in retry_messages)
        )

    def test_parse_retry_limit_respected(self):
        script = [scripted_json({"thought": "只有思考"}) for _ in range(6)]
        loop, llm = make_loop(script, policy=LoopPolicy(max_parse_retries=2))
        result = loop.run_turn("你好", session_id="w1")
        self.assertEqual(result.stop_reason, STOP_PARSE_ERROR)
        self.assertEqual(len(llm.calls), 3)  # 1 次 + 2 次重试


class TestSessionErrors(unittest.TestCase):
    """spec 6.2 E6：session 不存在 / 存储异常都有明确处理。"""

    def test_unknown_session_is_created(self):
        loop, _llm = make_loop([scripted_json({"answer": "ok"})])
        result = loop.run_turn("hi", session_id="brand-new")
        self.assertTrue(result.ok)
        self.assertIsNotNone(loop.sessions.get("brand-new"))

    def test_corrupt_store_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{broken")
            with self.assertRaises(Exception) as ctx:
                SessionStore(path)
            self.assertIn("不是合法 JSON", str(ctx.exception))


class TestMissingConfig(unittest.TestCase):
    """spec 6.2 E7：缺配置时明确报错。"""

    def test_from_env_without_key_raises_config_error(self):
        with self.assertRaises(ConfigError):
            LLMConfig.from_env({})

    def test_cli_reports_config_error_without_traceback(self):
        from agent.__main__ import main

        captured = io.StringIO()
        import contextlib
        from unittest import mock

        # 必须显式清空环境变量：否则当外部（如真实 API E2E）设置了 LLM_API_KEY 时，
        # 本测试会真的发起网络调用，断言随之失效。
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_E2E")
        }
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with contextlib.redirect_stderr(captured):
                code = main(["--prompt", "hi"])

        self.assertEqual(code, 2)
        self.assertIn("配置未就绪", captured.getvalue())
        self.assertNotIn("Traceback", captured.getvalue())


class TestCLIArgumentValidation(unittest.TestCase):
    """spec 6.2 E7：CLI 参数非法时必须给出明确提示，而非异常堆栈外泄。

    回归背景：`--max-iterations 0` 曾触发 `LoopPolicy` 抛出的裸 `ValueError`
    堆栈（`_make_loop` 当时只捕获 SessionError），违反"不擅自外泄堆栈"的要求。
    """

    ENV = {"LLM_API_KEY": "sk-fake-for-cli-validation"}

    def _run_main(self, argv):
        """在受控环境变量下调用 CLI 入口，返回 (退出码, stdout+stderr)。"""
        import contextlib
        from unittest import mock
        from agent.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
        env.update(self.ENV)
        with mock.patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = main(argv)
                except SystemExit as exc:  # argparse 拒绝时以 SystemExit 退出
                    code = int(exc.code) if exc.code is not None else 0
        return code, out.getvalue() + err.getvalue()

    def test_zero_max_iterations_is_rejected_cleanly(self):
        code, output = self._run_main(["--max-iterations", "0", "--prompt", "hi"])
        self.assertNotEqual(code, 0, "非法参数不应被静默接受")
        self.assertNotIn("Traceback", output, "不应外泄堆栈")
        self.assertIn("正整数", output)

    def test_negative_max_iterations_is_rejected(self):
        code, output = self._run_main(["--max-iterations", "-3", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)

    def test_non_numeric_max_iterations_is_rejected(self):
        code, output = self._run_main(["--max-iterations", "abc", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)
        self.assertIn("整数", output)

    def test_zero_tool_timeout_is_rejected_cleanly(self):
        code, output = self._run_main(["--tool-timeout", "0", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)
        self.assertIn("正数", output)

    def test_negative_tool_timeout_is_rejected(self):
        code, output = self._run_main(["--tool-timeout", "-1", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)

    def test_zero_max_context_chars_is_rejected_cleanly(self):
        code, output = self._run_main(["--max-context-chars", "0", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)
        self.assertIn("正整数", output)

    def test_valid_arguments_still_accepted(self):
        """边界正向：合法值必须照常通过参数校验（此处只校验到 argparse 层）。"""
        from agent.__main__ import build_parser

        args = build_parser().parse_args(
            ["--max-iterations", "1", "--tool-timeout", "0.5", "--max-context-chars", "1"]
        )
        self.assertEqual(args.max_iterations, 1)
        self.assertEqual(args.tool_timeout, 0.5)
        self.assertEqual(args.max_context_chars, 1)


class TestMultiSessionThroughLoop(unittest.TestCase):
    """spec 6.3 B7：两 session 交替使用，始终隔离。"""

    def test_alternating_sessions_keep_separate_context(self):
        script = [
            scripted_json({"answer": "窗口1 答1"}),
            scripted_json({"answer": "窗口2 答1"}),
            scripted_json({"answer": "窗口1 答2"}),
            scripted_json({"answer": "窗口2 答2"}),
        ]
        loop, llm = make_loop(script)
        loop.run_turn("窗口1 问1", session_id="w1")
        loop.run_turn("窗口2 问1", session_id="w2")
        loop.run_turn("窗口1 问2", session_id="w1")
        loop.run_turn("窗口2 问2", session_id="w2")

        w1_text = " ".join(m["content"] or "" for m in llm.messages_sent(2))
        w2_text = " ".join(m["content"] or "" for m in llm.messages_sent(3))
        self.assertIn("窗口1 问1", w1_text)
        self.assertIn("窗口1 答1", w1_text)
        self.assertNotIn("窗口2 问1", w1_text)
        self.assertIn("窗口2 问1", w2_text)
        self.assertNotIn("窗口1 问1", w2_text)

    def test_sessions_persist_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.json")
            sessions = SessionStore(path)
            loop, _llm = make_loop(
                [scripted_json({"answer": "1"}), scripted_json({"answer": "2"})],
                sessions=sessions,
            )
            loop.run_turn("窗口1", session_id="w1")
            loop.run_turn("窗口2", session_id="w2")

            reloaded = SessionStore(path)
            self.assertEqual(reloaded.require("w1").messages[0].content, "窗口1")
            self.assertEqual(reloaded.require("w2").messages[0].content, "窗口2")


class TestTraceCompleteness(unittest.TestCase):
    """spec 5.7 / 6.1 N11：trace 记录完整性。"""

    def test_trace_covers_tools_and_loop_steps(self):
        tracer = Tracer(enabled=False)
        script = [
            {"tool_calls": [native_tool_call("no_such_tool", {}, "c1")]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[RecordingTool()], tracer=tracer)
        loop.run_turn("跑", session_id="w1")

        events = [r.event for r in tracer.records]
        # 注：ScriptedLLM 绕过了 LLMClient，因此这里没有 llm_call 事件；
        # llm_call 由上层的 LLMClient 发出，见 test_llm_call_traced_via_real_client。
        self.assertIn("tool_call", events)
        self.assertIn("loop_step", events)

    def test_llm_call_traced_via_real_client(self):
        """走真实 LLMClient（fake transport）时，LLM 调用必须进 trace。"""
        from tests.helpers import chat_completion_payload, make_client

        tracer = Tracer(enabled=False)
        client, _transport = make_client(
            [
                (200, chat_completion_payload(content=scripted_json({"answer": "ok"}))),
            ],
            tracer=tracer,
        )
        loop = AgentLoop(
            client=client,
            registry=ToolRegistry([]),
            sessions=SessionStore(),
            builder=ContextBuilder(ContextPolicy(include_protocol_instructions=False)),
            tracer=tracer,
        )
        result = loop.run_turn("你好", session_id="w1")
        self.assertTrue(result.ok)

        events = [r.event for r in tracer.records]
        self.assertIn("llm_call", events)
        self.assertIn("loop_step", events)

    def test_tool_trace_contains_error_code_on_failure(self):
        tracer = Tracer(enabled=False)
        script = [
            {"tool_calls": [native_tool_call("recorder", {}, "c1")]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[RecordingTool()], tracer=tracer)
        loop.run_turn("跑", session_id="w1")

        record = [r for r in tracer.records if r.event == "tool_call"][0]
        self.assertFalse(record.payload["ok"])
        self.assertEqual(
            record.payload["error_code"], ToolErrorCode.MISSING_PARAMETER.value
        )

    def test_each_iteration_is_traceable(self):
        tracer = Tracer(enabled=False)
        script = [
            {"tool_calls": [native_tool_call("recorder", {"value": "A"}, "c1")]},
            scripted_json({"answer": "ok"}),
        ]
        loop, _llm = make_loop(script, tools=[RecordingTool()], tracer=tracer)
        loop.run_turn("跑", session_id="w1")

        iterations = [
            r.payload.get("iteration")
            for r in tracer.records
            if r.name == "step2_build_request"
        ]
        self.assertEqual(iterations, [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
