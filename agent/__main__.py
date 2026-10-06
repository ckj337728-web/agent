"""可执行入口：``python -m agent``。

对应任务 T003（骨架与入口）与 T033（驱动主循环的最小 CLI），
约束来自 spec.md：

- 6.2 E7：配置未就绪时给出**明确提示**而非裸堆栈；
- 3.6：需能指定 session id，从而在同一进程内驱动两个"窗口"；
- spec 第 7 节未要求 Web/GUI，因此这里只做驱动主循环所需的最小交互入口。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .config import ConfigError, LLMConfig
from .context import ContextBuilder, ContextPolicy
from .llm import LLMClient
from .loop import (
    DEFAULT_SYSTEM_PROMPT,
    EVENT_ANSWER,
    EVENT_ERROR,
    EVENT_LIMIT,
    EVENT_THOUGHT,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    AgentLoop,
    LoopPolicy,
    build_agent,
)
from .session import SessionStore, SessionError
from .trace import Tracer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="最小可用 Agent Runtime（从零实现，不依赖任何 Agent 框架）",
    )
    parser.add_argument("--version", action="version", version=f"agent {__version__}")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只校验环境变量中的 LLM 配置，不进入对话",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="session id（对应题面里的一个「窗口」）；省略则自动生成",
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="session 落盘文件路径；省略则仅保存在内存中",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="TEXT",
        help="单次提问后退出（非交互模式），便于脚本化验证",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=LoopPolicy.max_iterations,
        help=f"单轮最大循环次数，需为正整数（默认 {LoopPolicy.max_iterations}）",
    )
    parser.add_argument(
        "--tool-timeout",
        type=_positive_float,
        default=LoopPolicy.tool_timeout_seconds,
        help=f"单次工具执行超时秒数，需为正数（默认 {LoopPolicy.tool_timeout_seconds}）",
    )
    parser.add_argument(
        "--max-context-chars",
        type=_positive_int,
        default=ContextPolicy.max_context_chars,
        help=f"context 字符上限，超出触发压缩，需为正整数（默认 {ContextPolicy.max_context_chars}）",
    )
    parser.add_argument(
        "--trace",
        choices=["off", "text", "jsonl"],
        default="text",
        help="trace 输出格式，默认 text（写入 stderr）",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出存储中的 session 后退出",
    )
    return parser


def _positive_int(value: str) -> int:
    """argparse 类型校验：必须为正整数（如 --max-iterations）。"""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"必须是整数，收到 {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"必须为正整数（≥1），收到 {number}")
    return number


def _positive_float(value: str) -> float:
    """argparse 类型校验：必须为正数（如 --tool-timeout）。"""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"必须是数字，收到 {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"必须为正数（>0），收到 {number}")
    return number


def _print_event(event: str, payload: Dict[str, Any]) -> None:
    """把主循环事件渲染到终端。

    输出到 stderr 的事件与最终答案（stdout）分开，便于脚本只取答案。
    """
    if event == EVENT_THOUGHT:
        print(f"  · 思考：{_first_line(payload.get('text'))}", file=sys.stderr)
    elif event == EVENT_TOOL_CALL:
        print(
            f"  → 调用工具 {payload.get('name')} 参数 {payload.get('arguments')}",
            file=sys.stderr,
        )
    elif event == EVENT_TOOL_RESULT:
        mark = "✓" if payload.get("ok") else "✗"
        print(
            f"  {mark} 工具返回：{_first_line(payload.get('text'), 300)}",
            file=sys.stderr,
        )
    elif event == EVENT_ERROR:
        print(f"  ! 错误：{_first_line(payload.get('text'))}", file=sys.stderr)
    elif event == EVENT_LIMIT:
        print(f"  ! {_first_line(payload.get('text'))}", file=sys.stderr)


def _first_line(text: Any, limit: int = 200) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").split("\n")[0]
    return value if len(value) <= limit else f"{value[:limit]}..."


def _make_loop(args: argparse.Namespace, config: LLMConfig) -> AgentLoop:
    tracer = Tracer(fmt=args.trace) if args.trace != "off" else Tracer(enabled=False)
    client = LLMClient(config, tracer=tracer)
    sessions = SessionStore(args.store)
    return build_agent(
        client=client,
        sessions=sessions,
        tracer=tracer,
        context_policy=ContextPolicy(max_context_chars=args.max_context_chars),
        loop_policy=LoopPolicy(
            max_iterations=args.max_iterations,
            tool_timeout_seconds=args.tool_timeout,
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = LLMConfig.from_env()
    except ConfigError as exc:
        # 明确提示，不打印堆栈（spec 6.2 E7）
        print(f"[配置未就绪] {exc}", file=sys.stderr)
        return 2

    if args.list_sessions:
        sessions = SessionStore(args.store)
        ids = sessions.session_ids()
        if not ids:
            print("（存储中没有任何 session）")
        for session_id in ids:
            session = sessions.require(session_id)
            print(
                f"{session_id}  轮次={session.current_turn()}  "
                f"消息={len(session)}  更新时间={session.updated_at:.0f}"
            )
        return 0

    if args.check_config:
        print(f"agent {__version__} 配置就绪：{config.safe_summary()}")
        print("配置校验通过。")
        return 0

    try:
        loop = _make_loop(args, config)
    except SessionError as exc:
        print(f"[session 错误] {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        # 策略参数非法（如 --max-iterations 0）。
        # spec 6.2 E7 要求"给出明确提示，而非静默失败或异常堆栈外泄"，
        # 因此这里必须捕获并转成可读提示，不能让它冒泡成裸堆栈。
        print(f"[参数错误] {exc}", file=sys.stderr)
        print("用 `python -m agent --help` 查看各参数取值要求。", file=sys.stderr)
        return 2

    session_id = args.session or SessionStore.new_session_id()
    print(f"agent {__version__} | session={session_id}", file=sys.stderr)
    if args.store:
        print(f"session 存储：{args.store}", file=sys.stderr)
    print(
        f"模型：{config.model} | 工具：{', '.join(loop.registry.names())}",
        file=sys.stderr,
    )

    if args.prompt is not None:
        return _run_once(loop, args.prompt, session_id)

    return _run_interactive(loop, session_id)


def _run_once(loop: AgentLoop, prompt: str, session_id: str) -> int:
    result = loop.run_turn(prompt, session_id=session_id, observer=_print_event)
    print(result.answer)
    if result.degraded:
        print(f"[结束原因：{result.stop_reason}]", file=sys.stderr)
    return 0 if result.ok else 1


def _run_interactive(loop: AgentLoop, session_id: str) -> int:
    print("输入问题后回车发送；:exit 退出，:new 新建 session，:help 查看帮助。", file=sys.stderr)

    while True:
        try:
            line = input(f"[{session_id}] > ")
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            return 0

        text = line.strip()
        if not text:
            continue
        if text in (":exit", ":quit"):
            return 0
        if text == ":help":
            print(
                ":exit 退出 | :new 新建 session | :id 显示当前 session id\n"
                "提示：用 --session 指定 id 即可在同一存储里开多个独立窗口。",
                file=sys.stderr,
            )
            continue
        if text == ":id":
            print(session_id, file=sys.stderr)
            continue
        if text == ":new":
            session_id = loop.sessions.create().session_id
            print(f"已切换到新 session：{session_id}", file=sys.stderr)
            continue

        try:
            result = loop.run_turn(text, session_id=session_id, observer=_print_event)
        except SessionError as exc:
            print(f"[session 错误] {exc}", file=sys.stderr)
            continue

        print(result.answer)
        if result.degraded:
            print(f"[结束原因：{result.stop_reason}]", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
