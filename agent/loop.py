"""Agent 主循环（Core Runtime）。

对应任务 T024–T032，约束来自 spec.md：

- 3.1：必须显式实现四步骤，且每一步可被 trace 观测——
  Step 1 接收用户输入 → Step 2 判断直接回复还是调用工具 →
  Step 3 调用工具 → Step 4 根据工具结果判断继续 loop 还是返回结果；
- 3.0：本层负责**编排**（轮次控制、终止判定），不实现具体工具；
- 3.7 / 6.2：LLM 失败、工具异常、参数非法、轮次上限都必须有明确处理路径，
  不产生未捕获崩溃；
- 6.3 B1/B2/B5/B8：轮次上限、单响应多工具调用、空输入、LLM 拒绝调工具，
  都要有确定行为。

本模块复用已有各层，不重复实现：LLM 调用用 :class:`agent.llm.LLMClient`，
语义提取用 :func:`agent.parser.parse_response`，历史与状态用
:class:`agent.session.SessionStore`，提示词组装用 :class:`agent.context.ContextBuilder`，
工具调度用 :class:`agent.tools.registry.ToolRegistry`。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .context import ContextBuilder, ContextPolicy
from .llm import ChatMessage, LLMClient, LLMError
from .parser import OutputKind, ParsedOutput, parse_response
from .session import Session, SessionStore, ToolCallRecord
from .tools.base import ToolContext, ToolErrorCode, ToolResult
from .tools.registry import ToolRegistry
from .trace import Tracer

#: 默认系统提示词（角色与工具使用原则）
DEFAULT_SYSTEM_PROMPT = """\
你是一个可以调用工具来完成任务的助手。

行为要求：
1. 先判断是否需要工具。需要精确计算、检索资料、读写会话待办时，必须调用工具，
   不要凭记忆或猜测作答。
2. 工具返回结果后，再基于结果组织给用户的答复。
3. 一次可以请求多个工具调用；能并行获取的信息可以一起请求。
4. 工具返回 ERROR 时，先判断是参数写错还是问题本身无法完成：
   参数写错就修正参数重试；无法完成就如实说明，不要编造结果。
5. 最终答复要直接、完整地回答用户的问题。
"""

#: 终止原因
STOP_FINAL_ANSWER = "final_answer"          # 正常收敛
STOP_MAX_ITERATIONS = "max_iterations"      # 达到轮次上限（spec 6.3 B1）
STOP_NO_PROGRESS = "no_progress"            # 反复无进展，兜底收敛（spec 6.3 B8）
STOP_LLM_ERROR = "llm_error"                # LLM 调用失败且重试耗尽
STOP_PARSE_ERROR = "parse_error"            # 解析反复失败
STOP_EMPTY_INPUT = "empty_input"            # 空输入（spec 6.3 B5）


@dataclass
class LoopPolicy:
    """主循环策略（全部可配置）。"""

    #: 单轮用户输入最多允许的 loop 次数（spec C-6：必须有上限）
    max_iterations: int = 8

    #: 解析失败后允许的"重新提示"次数（spec 3.4 回退路径之一）
    max_parse_retries: int = 1

    #: 连续多少轮"既无有效工具调用也无答案"就停止（spec 6.3 B8）
    max_no_progress_rounds: int = 2

    #: 单次工具执行超时秒数（spec 6.2 E8）
    tool_timeout_seconds: float = 15.0

    #: 是否把 assistant 的思考过程写入 trace
    trace_thought: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations 至少为 1")
        if self.max_parse_retries < 0:
            raise ValueError("max_parse_retries 不能为负数")
        if self.max_no_progress_rounds < 1:
            raise ValueError("max_no_progress_rounds 至少为 1")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds 必须为正数")


@dataclass
class LoopStep:
    """一次 loop 迭代的可观测记录。"""

    iteration: int
    thought: str = ""
    tool_calls: List[str] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    answer: str = ""
    finish_reason: str = ""
    parse_source: str = ""
    parse_warnings: List[str] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "answer": self.answer,
            "finish_reason": self.finish_reason,
            "parse_source": self.parse_source,
            "parse_warnings": self.parse_warnings,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TurnResult:
    """一次 ``run_turn`` 的完整结果。"""

    session_id: str
    answer: str
    stop_reason: str
    iterations: int = 0
    steps: List[LoopStep] = field(default_factory=list)
    trace_id: str = ""
    tool_calls: int = 0

    @property
    def ok(self) -> bool:
        """是否正常收敛为答案。"""
        return self.stop_reason == STOP_FINAL_ANSWER

    @property
    def degraded(self) -> bool:
        """是否以非理想方式结束（超限、无进展、失败）。"""
        return self.stop_reason != STOP_FINAL_ANSWER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "trace_id": self.trace_id,
            "steps": [step.to_dict() for step in self.steps],
        }


#: 事件观察者：(事件名, 载荷) -> None。用于 CLI 实时输出与测试断言。
EventObserver = Callable[[str, Dict[str, Any]], None]

EVENT_THOUGHT = "thought"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_ANSWER = "answer"
EVENT_ERROR = "error"
EVENT_LIMIT = "limit"


# --------------------------------------------------------------------------- #
# 工具执行超时保护（T032）
# --------------------------------------------------------------------------- #


@dataclass
class _ToolOutcome:
    result: ToolResult
    duration_ms: float


def _run_with_timeout(
    func: Callable[[], ToolResult], timeout_seconds: float
) -> _ToolOutcome:
    """在守护线程中执行工具，超时即放弃等待并返回超时错误。

    为什么用线程而不是强制终止：Python 无法安全地杀死线程。因此超时后
    我们**放弃等待**并让守护线程自生自灭，主循环继续——这满足 spec 6.2 E8
    "有超时保护，不阻塞整个 loop"的要求。
    """
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = func()
        except Exception as exc:  # noqa: BLE001 - 兜底，注册表内部已归一化
            box["result"] = ToolResult.failure(
                ToolErrorCode.TOOL_ERROR, f"{type(exc).__name__}: {exc}"
            )

    started = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return _ToolOutcome(
            result=ToolResult.failure(
                ToolErrorCode.TOOL_TIMEOUT,
                f"工具执行超过 {timeout_seconds}s 未返回，已放弃等待",
            ),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    result = box.get("result")
    if not isinstance(result, ToolResult):
        result = ToolResult.failure(
            ToolErrorCode.TOOL_ERROR, "工具未返回有效结果"
        )
    return _ToolOutcome(result=result, duration_ms=duration_ms)


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #


class AgentLoop:
    """最小可用的 Agent 主循环。

    参数：
        client:     LLM 客户端（T005）
        registry:   工具注册表（T008）
        sessions:   session 存储（T016/T017）
        builder:    context 组装器（T019）
        tracer:     trace 记录器（T006）
        policy:     循环策略
        system_prompt: 系统提示词
    """

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        sessions: Optional[SessionStore] = None,
        builder: Optional[ContextBuilder] = None,
        tracer: Optional[Tracer] = None,
        policy: Optional[LoopPolicy] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._client = client
        self._registry = registry
        # 注意：必须用 `is None` 判断。SessionStore 实现了 __len__，
        # 一个空的（尚未建任何 session 的）store 是 falsy，用 `or` 会把它
        # 悄悄换成新的内存 store，导致落盘配置失效。
        self._sessions = sessions if sessions is not None else SessionStore()
        self._builder = builder if builder is not None else ContextBuilder()
        self._tracer = tracer if tracer is not None else Tracer(enabled=False)
        self._policy = policy if policy is not None else LoopPolicy()
        self._system_prompt = system_prompt

    # -- 只读访问（便于测试与入口组装） ------------------------------------ #

    @property
    def policy(self) -> LoopPolicy:
        return self._policy

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def builder(self) -> ContextBuilder:
        return self._builder

    # -- 主入口 ------------------------------------------------------------ #

    def run_turn(
        self,
        user_input: str,
        session_id: str,
        observer: Optional[EventObserver] = None,
    ) -> TurnResult:
        """处理一次用户输入，返回最终结果。

        Step 1（接收输入）→ Step 2（决策）→ Step 3（调用工具）→ Step 4（终止判定）
        四步在本方法内显式分节实现，每一步都写入 trace 与 ``steps``。
        """
        session = self._sessions.get_or_create(session_id)
        trace_id = self._tracer.new_trace_id()

        # ---------------- Step 1：接收用户输入 ----------------
        cleaned = (user_input or "").strip()
        if not cleaned:
            # spec 6.3 B5：空输入不触发 LLM 调用、不进入 loop
            self._tracer.loop_step(
                "empty_input_rejected", trace_id, session_id=session_id
            )
            return TurnResult(
                session_id=session_id,
                answer="请输入有效内容后再试（收到的输入为空或只有空白字符）。",
                stop_reason=STOP_EMPTY_INPUT,
                trace_id=trace_id,
            )

        session.append_user_message(cleaned)
        self._tracer.loop_step(
            "step1_receive_input",
            trace_id,
            session_id=session_id,
            turn=session.current_turn(),
            input_chars=len(cleaned),
        )
        self._emit(observer, EVENT_THOUGHT, iteration=0, text=f"[收到输入] {cleaned}")

        result = TurnResult(session_id=session_id, answer="", stop_reason="", trace_id=trace_id)
        no_progress_rounds = 0
        parse_retries = 0
        # 仅用于"解析失败后重新提示"的临时消息。工具结果**不**走这里：
        # 它们已作为 tool 消息写入 session，由 ContextBuilder 统一渲染，
        # 否则同一份结果会被注入两次。
        extra_messages: List[ChatMessage] = []

        for iteration in range(1, self._policy.max_iterations + 1):
            result.iterations = iteration
            step, parsed_or_none, llm_failed = self._run_iteration(
                session=session,
                session_id=session_id,
                trace_id=trace_id,
                iteration=iteration,
                observer=observer,
                extra_messages=extra_messages,
            )
            result.steps.append(step)

            if llm_failed:
                # spec 6.2 E4：LLM 失败且重试耗尽 → 终止并返回可解释错误
                result.stop_reason = STOP_LLM_ERROR
                result.answer = (
                    f"抱歉，调用模型失败：{step.error}。请稍后重试；"
                    f"（本轮已尝试 {iteration} 次循环）"
                )
                self._finalize_answer(session, result, trace_id)
                return result

            parsed = parsed_or_none

            # 解析失败 → 走"重新提示"回退（spec 3.4）
            if parsed is not None and parsed.is_empty:
                if parse_retries < self._policy.max_parse_retries:
                    parse_retries += 1
                    self._tracer.loop_step(
                        "parse_retry",
                        trace_id,
                        session_id=session_id,
                        attempt=f"{parse_retries}/{self._policy.max_parse_retries}",
                        raw=parsed.raw_text[:200],
                    )
                    extra_messages = [
                        ChatMessage(
                            role="system",
                            content=(
                                "你上一条回复无法被解析。请严格按协议重新输出："
                                '需要工具时输出 {"thought": "...", "tool_calls": '
                                '[{"name": "...", "arguments": {...}}]}；'
                                '可以直接回答时输出 {"thought": "...", "answer": "..."}。'
                            ),
                        )
                    ]
                    continue
                result.stop_reason = STOP_PARSE_ERROR
                result.answer = (
                    "抱歉，模型的回复格式反复无法解析，已停止本轮处理。"
                    "请换一种说法再试一次。"
                )
                self._finalize_answer(session, result, trace_id)
                return result

            # ---------------- Step 2/4：决策与终止判定 ----------------
            if parsed is not None and not parsed.has_tool_calls:
                # 无工具调用 → 收敛为最终答案，本轮结束
                answer = parsed.answer.strip()
                if answer:
                    result.answer = answer
                    result.stop_reason = STOP_FINAL_ANSWER
                    session.append_assistant_message(content=answer, thought=parsed.thought)
                    self._tracer.loop_step(
                        "step4_finish",
                        trace_id,
                        session_id=session_id,
                        iteration=iteration,
                        answer_chars=len(answer),
                    )
                    self._finalize_answer(session, result, trace_id)
                    return result
                # 既无工具也无答案：计入"无进展"（spec 6.3 B8 的收敛保护）
                no_progress_rounds += 1
                if no_progress_rounds >= self._policy.max_no_progress_rounds:
                    result.stop_reason = STOP_NO_PROGRESS
                    result.answer = (
                        "抱歉，模型未能给出有效答复。请把问题描述得更具体一些，"
                        "或换一种问法。"
                    )
                    self._finalize_answer(session, result, trace_id)
                    return result
                pending_tool_messages = []
                continue

            # ---------------- Step 3：调用工具 ----------------
            assert parsed is not None
            session.append_assistant_message(
                content=parsed.answer, thought=parsed.thought
            )
            pending_tool_messages, executed = self._execute_tool_calls(
                session=session,
                session_id=session_id,
                trace_id=trace_id,
                iteration=iteration,
                parsed=parsed,
                step=step,
                observer=observer,
            )
            result.tool_calls += executed

            if executed == 0:
                # 请求了工具但全部未能执行（如参数非法到无法构造调用）
                no_progress_rounds += 1
                if no_progress_rounds >= self._policy.max_no_progress_rounds:
                    result.stop_reason = STOP_NO_PROGRESS
                    result.answer = (
                        "抱歉，工具调用反复失败，无法继续。请检查问题描述后重试。"
                    )
                    self._finalize_answer(session, result, trace_id)
                    return result
            else:
                no_progress_rounds = 0

        # ---------------- 达到轮次上限（spec 6.3 B1） ----------------
        result.stop_reason = STOP_MAX_ITERATIONS
        result.answer = self._limit_answer(result)
        self._tracer.loop_step(
            "step4_limit_reached",
            trace_id,
            session_id=session_id,
            max_iterations=self._policy.max_iterations,
            tool_calls=result.tool_calls,
        )
        self._emit(
            observer,
            EVENT_LIMIT,
            iteration=result.iterations,
            text=f"达到最大轮次 {self._policy.max_iterations}",
        )
        self._finalize_answer(session, result, trace_id)
        return result

    # -- 单次迭代 ---------------------------------------------------------- #

    def _run_iteration(
        self,
        session: Session,
        session_id: str,
        trace_id: str,
        iteration: int,
        observer: Optional[EventObserver],
        extra_messages: Sequence[ChatMessage],
    ) -> tuple:
        """执行一次"组装请求 → 调用 LLM → 解析"。

        返回 ``(step, parsed_or_none, llm_failed)``。
        """
        step = LoopStep(iteration=iteration)

        built = self._builder.build(session, system_prompt=self._system_prompt)
        messages = list(built.messages) + list(extra_messages)

        self._tracer.loop_step(
            "step2_build_request",
            trace_id,
            session_id=session_id,
            iteration=iteration,
            messages=len(messages),
            tools=len(self._registry),
            context_chars=built.total_chars,
            compressed=built.compressed,
            elided_turns=built.elided_turns,
            truncated_tool_results=built.tool_results_truncated,
        )

        # Step 2：由 LLM 基于工具 Schema 自主决策
        try:
            response = self._client.chat(
                messages,
                tools=self._registry.to_llm_schemas(),
                trace_id=trace_id,
            )
        except LLMError as exc:
            step.error = str(exc)
            self._tracer.error(
                "step2_llm_failed",
                trace_id,
                session_id=session_id,
                iteration=iteration,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._emit(observer, EVENT_ERROR, iteration=iteration, text=str(exc))
            return step, None, True

        step.finish_reason = response.finish_reason
        parsed = parse_response(
            content=response.content, native_tool_calls=response.tool_calls
        )
        step.parse_source = parsed.source.value
        step.parse_warnings = list(parsed.warnings)
        step.thought = parsed.thought
        step.answer = parsed.answer

        self._tracer.loop_step(
            "step2_parse_result",
            trace_id,
            session_id=session_id,
            iteration=iteration,
            kind=parsed.kind.value,
            source=parsed.source.value,
            tool_calls=len(parsed.tool_calls),
            warnings=len(parsed.warnings),
        )
        if parsed.warnings:
            self._tracer.loop_step(
                "parse_warnings",
                trace_id,
                session_id=session_id,
                iteration=iteration,
                detail="; ".join(parsed.warnings)[:300],
            )
        if parsed.thought and self._policy.trace_thought:
            self._emit(
                observer, EVENT_THOUGHT, iteration=iteration, text=parsed.thought
            )
        if parsed.answer:
            self._emit(
                observer, EVENT_ANSWER, iteration=iteration, text=parsed.answer
            )
        return step, parsed, False

    # -- Step 3：工具执行 -------------------------------------------------- #

    def _execute_tool_calls(
        self,
        session: Session,
        session_id: str,
        trace_id: str,
        iteration: int,
        parsed: ParsedOutput,
        step: LoopStep,
        observer: Optional[EventObserver],
    ) -> tuple:
        """按序执行一次响应中的全部工具调用（spec 6.3 B2）。

        返回 ``(回灌给 LLM 的消息, 实际执行的调用数)``。
        """
        appended: List[ChatMessage] = []
        executed = 0
        records: List[ToolCallRecord] = []
        tool_context = ToolContext(session_id=session_id, state=session.state)

        for index, call in enumerate(parsed.tool_calls):
            call_id = call.call_id or f"call_{iteration}_{index}"
            step.tool_calls.append(call.name)
            self._emit(
                observer,
                EVENT_TOOL_CALL,
                iteration=iteration,
                name=call.name,
                arguments=call.arguments,
            )

            if not call.arguments_valid:
                # 参数解析失败：按参数错误回灌，让 LLM 自己修正
                outcome = _ToolOutcome(
                    result=ToolResult.failure(
                        ToolErrorCode.INVALID_ARGUMENTS,
                        f"参数解析失败：{call.argument_error}",
                    ),
                    duration_ms=0.0,
                )
            else:
                outcome = _run_with_timeout(
                    lambda name=call.name, args=call.arguments: self._registry.execute(
                        name, args, tool_context
                    ),
                    self._policy.tool_timeout_seconds,
                )
            executed += 1

            tool_text = outcome.result.to_llm_text()
            step.tool_results.append(tool_text)

            record = ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                call_id=call_id,
                result_ok=outcome.result.ok,
                result_text=tool_text,
                error_code=outcome.result.error_code,
                duration_ms=outcome.duration_ms,
            )
            records.append(record)

            # trace 四要素：入参、输出、耗时、错误（spec 3.1）
            self._tracer.tool_call(
                call.name,
                trace_id,
                session_id=session_id,
                iteration=iteration,
                arguments=call.arguments,
                ok=outcome.result.ok,
                error_code=outcome.result.error_code,
                duration_ms=outcome.duration_ms,
                result=tool_text,
            )
            self._emit(
                observer,
                EVENT_TOOL_RESULT,
                iteration=iteration,
                name=call.name,
                ok=outcome.result.ok,
                text=tool_text,
            )

            session.append_tool_message(call.name, tool_text, call_id)
            appended.append(
                ChatMessage(role="tool", content=tool_text, tool_call_id=call_id, name=call.name)
            )

        if records:
            # 把本轮的调用记录挂到最近的 assistant 消息上，保证回放时成对
            self._attach_records(session, records)

        return appended, executed

    @staticmethod
    def _attach_records(session: Session, records: Sequence[ToolCallRecord]) -> None:
        """把工具调用记录挂到刚写入的 assistant 消息上。"""
        for message in reversed(session.messages):
            if message.role == "assistant":
                message.tool_calls = list(records)
                return

    # -- 收尾 -------------------------------------------------------------- #

    def _limit_answer(self, result: TurnResult) -> str:
        """轮次超限时的收尾答复：给出已获得的部分结论 + 超限说明（spec 6.3 B1）。"""
        partial: List[str] = []
        for step in result.steps:
            for text in step.tool_results:
                if text and not text.startswith("ERROR"):
                    partial.append(text)
        lines = [
            f"已达到单轮最大循环次数（{self._policy.max_iterations} 次），"
            "为避免无限循环已停止继续调用工具。"
        ]
        if result.tool_calls:
            lines.append(f"本轮共执行 {result.tool_calls} 次工具调用。")
        if partial:
            lines.append("已获得的部分结果：")
            lines.extend(f"  - {_one_line(text, 200)}" for text in partial[-3:])
        lines.append("如需继续，请把问题拆得更具体一些再问我。")
        return "\n".join(lines)

    def _finalize_answer(self, session: Session, result: TurnResult, trace_id: str) -> None:
        """把最终答复写入 session 并落盘。"""
        if result.answer:
            last = session.last_assistant_message()
            if not (last and last.content == result.answer):
                session.append_assistant_message(content=result.answer)
        self._sessions.save()
        self._tracer.loop_step(
            "turn_finished",
            trace_id,
            session_id=session.session_id,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            answer_chars=len(result.answer),
        )

    @staticmethod
    def _emit(
        observer: Optional[EventObserver], event: str, **payload: Any
    ) -> None:
        if observer is None:
            return
        try:
            observer(event, payload)
        except Exception:  # noqa: BLE001 - 观察者故障不得影响主流程
            pass


def _one_line(text: str, limit: int) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


# --------------------------------------------------------------------------- #
# 组装默认运行时
# --------------------------------------------------------------------------- #


def build_agent(
    client: LLMClient,
    sessions: Optional[SessionStore] = None,
    tracer: Optional[Tracer] = None,
    context_policy: Optional[ContextPolicy] = None,
    loop_policy: Optional[LoopPolicy] = None,
    registry: Optional[ToolRegistry] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentLoop:
    """用内置工具装配一个可用的 Agent（供 CLI 与 E2E 使用）。"""
    from .tools.registry import build_default_registry

    return AgentLoop(
        client=client,
        registry=registry if registry is not None else build_default_registry(),
        sessions=sessions if sessions is not None else SessionStore(),
        builder=ContextBuilder(context_policy),
        tracer=tracer,
        policy=loop_policy,
        system_prompt=system_prompt,
    )
