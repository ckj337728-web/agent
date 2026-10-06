# 对照 spec.md 第 6 节验收清单的逐条核验（T050）

> 核验方式：每条给出**结论 + 可复核证据**（测试方法名 / 文件 / 命令）。
> 测试命令：`python -m unittest discover -s tests -t . -v`（离线，385 用例）
> 真实 API 命令：另需 `LLM_E2E=1` 与密钥环境变量（10 用例）
> 结论取值：✅ 通过 / ⚠️ 通过但需说明边界 / ❌ 不通过

---

## 6.1 正常路径

| 编号 | 验收项 | 结论 | 证据 |
| --- | --- | --- | --- |
| N1 | 主循环四步骤均有实现且在 trace 中逐步可见 | ✅ | `agent/loop.py::AgentLoop.run_turn` 按 Step1–4 分节；`tests/test_loop.py::TestObservability::test_trace_records_four_steps` 断言 trace 含 `step1_receive_input`、`step2_build_request`、`step2_parse_result`、`step4_finish`、`turn_finished` |
| N2 | 直接回复路径可用，一轮内给出答案，不产生多余工具调用 | ✅ | `test_loop.py::TestDirectReply::test_direct_answer_via_protocol_json`（`iterations==1`、`tool_calls==0`）、`test_direct_answer_via_plain_text_fallback` |
| N3 | 工具调用路径可用：LLM 自主选工具并给合法参数，结果回灌后收敛 | ✅ | `test_loop.py::TestToolCallPath::test_single_tool_call_then_answer`、`test_tool_result_is_fed_back_to_llm`；真实 API：`test_e2e_real_llm.py::TestRealLLMToolCalling::test_calculator_tool_is_invoked_and_result_fed_back` |
| N4 | 链式/多工具调用可用，最终答案综合各工具结果 | ✅ | `test_loop.py::TestToolCallPath::test_chained_tool_calls_across_iterations`、`test_multiple_tool_calls_in_one_response`；真实 API：`TestRealLLMChainedAndSessions::test_todo_add_then_list_chains_tools` |
| N5 | ≥3 个工具，具备 name/description/参数 Schema 注册机制，新增工具无需改主循环 | ✅ | `agent/tools/calculator.py`、`search.py`、`todo.py`；`test_tool_registry.py::test_default_registry_exposes_three_tools`、`test_new_tool_needs_no_core_change`（注册 EchoTool 后自动出现在 LLM 工具列表） |
| N6 | 解析逻辑可稳定提取思考过程、工具调用、最终答案 | ✅ | `test_parser.py::TestProtocolSelectionPriority`（三级优先级）、`TestNativeToolCalls`、`TestProtocolJsonInText`、`TestFallbackPaths`（共 43 用例） |
| N7 | 已接入真实 LLM API，端到端 E2E 在配置密钥后可跑通 | ✅ | `agent/llm.py::urllib_transport` 为默认真实传输；**实测**：`LLM_API_KEY`+`LLM_E2E=1` 下 `Ran 385 tests … OK`；CLI 真实跑通「search+todo」双工具调用 |
| N8 | 双窗口 session 场景通过，互不影响且可随时续聊并恢复历史 | ✅ | 离线：`test_session_context_integration.py::test_full_dual_window_flow`、`test_resume_window_1_after_working_in_window_2`、`test_loop.py::test_dual_window_isolation_through_loop`；**真实 API 跨进程实测**：窗口1 待办仅「带伞」、窗口2 仅「写周报」 |
| N9 | 纯对话追问与带工具追问均正确工作 | ✅ | `test_context.py::TestFollowUpSupport::test_plain_conversational_follow_up_sees_prior_turns`、`test_tool_follow_up_sees_prior_tool_result`；真实 API：`test_tool_result_visible_in_next_turn_follow_up`（(20+22)×2=84） |
| N10 | 最大轮次限制与基础压缩均生效且可配置 | ✅ | 轮次：`test_loop_errors.py::TestMaxIterations::test_limit_stops_endless_tool_calls`；压缩：`test_context.py::TestCompression::test_compression_triggered_when_over_limit`、`test_result_stays_within_limit`；均可配置（`LoopPolicy` / `ContextPolicy`，CLI 暴露 `--max-iterations`、`--max-context-chars`） |
| N11 | 每次 LLM 与工具调用均有 trace/日志，可回溯 | ✅ | `test_loop_errors.py::TestTraceCompleteness`（四步骤、迭代可回溯、工具 trace 含 `error_code`）、`test_llm_call_traced_via_real_client`；工具 trace 四要素见 `test_loop.py::test_trace_records_four_steps` |
| N12 | README 含运行方式、系统设计、memory 说明；另附 AI Prompt 与问题解决记录 | ✅ | README 第 1/2/3 节；记录见 [`notes/ai-usage.md`](../notes/ai-usage.md) |

## 6.2 异常路径

| 编号 | 验收项 | 结论 | 证据 |
| --- | --- | --- | --- |
| E1 | 调用不存在的工具：不崩溃，错误可识别并回灌 LLM | ✅ | `test_tool_registry.py::test_unknown_tool`；端到端回灌：`test_loop_errors.py::test_unknown_tool_error_is_fed_back_and_model_recovers` |
| E2 | 参数缺失或类型错误：被 Schema 校验拦截，错误可读且可恢复 | ✅ | `test_tool_registry.py::test_missing_required_parameter`、`test_wrong_parameter_type`、`test_none_arguments_treated_as_missing`；回灌：`test_missing_parameter_error_is_fed_back`、`test_wrong_type_error_is_fed_back` |
| E3 | 工具内部异常被归一化，Agent 能给出合理回应 | ✅ | calculator：`test_tool_calculator.py::TestInvalidExpressions`（非法表达式、除零、域错误等 12 例）；search 无结果：`test_tool_search.py::test_run_no_result_is_success_not_error`；工具内部异常：`test_tool_registry.py::test_tool_internal_exception_normalized`、`test_loop_errors.py::test_tool_internal_exception_is_normalized` |
| E4 | LLM 失败：按重试策略处理，重试耗尽终止并返回可解释错误 | ✅ | `test_llm.py::TestErrorNormalization`（429/5xx 重试、401/4xx 不重试、重试次数与恢复）；`test_loop_errors.py::TestLLMFailures`（超时/限流/鉴权均返回可读答复、无未捕获异常、有 trace） |
| E5 | 解析失败（JSON 非法、字段缺失、格式异常）走回退，不崩溃 | ✅ | `test_parser.py::TestFallbackPaths::test_never_raises_on_garbage`（11 种畸形输入）、`test_invalid_json_looking_text_falls_back`；重试路径：`test_loop_errors.py::test_parse_failure_can_be_retried_once_then_success`、`test_parse_retry_limit_respected` |
| E6 | 访问不存在的 session：有明确处理，不产生隐式串扰 | ✅ | 两种策略：`test_session.py::test_get_or_create_creates_when_missing`（新建）、`test_require_raises_with_known_ids`（明确报错并列出已有 id）；`test_loop_errors.py::TestSessionErrors` |
| E7 | 缺少配置（无 API Key 等）：明确提示，不静默失败 | ✅ | `test_config.py::TestMissingConfig::test_error_message_is_actionable`（含变量名、可照抄示例、指向 `.env.example`）；CLI：`test_loop_errors.py::test_cli_reports_config_error_without_traceback`（退出码 2 且无 `Traceback`） |
| E8 | 工具执行超时：有超时保护，不阻塞整个 loop | ✅ | `test_loop_errors.py::test_tool_timeout_does_not_block_loop`、`test_timeout_result_recorded_in_session`（`slow` 工具休眠 2s、超时设 0.15s，loop 仍收敛） |

## 6.3 边界情况

| 编号 | 验收项 | 结论 | 证据 |
| --- | --- | --- | --- |
| B1 | 达到最大轮次上限：停止调用工具，返回部分结论并说明原因 | ✅ | `test_loop_errors.py::TestMaxIterations::test_limit_answer_explains_and_includes_partial_results`（含"最大循环次数"、"部分结果"、已获工具结果）、`test_limit_emits_limit_event`、`test_limit_result_written_to_session` |
| B2 | 单轮触发大量工具调用：全部处理且顺序/结果可追溯 | ✅ | `test_loop.py::test_multiple_tool_calls_in_one_response`（3 个调用按序执行）、`test_multiple_tool_results_all_fed_back`（`tool_call_id` 顺序为 c1,c2） |
| B3 | context 达压缩阈值：压缩触发且压缩后仍可追问前文 | ✅ | `test_context.py::test_compression_triggered_when_over_limit`、`test_recent_turns_are_kept`、`test_oldest_turns_dropped_first`、`test_result_stays_within_limit` |
| B4 | 超长工具输出被截断或摘要后入 context，不撑爆上下文 | ✅ | `test_context.py::TestTruncateToolResult`（含恰好等于边界不截断、超 1 字符即截断）、`test_tool_results_truncated_reported`、`test_session_context_integration.py::test_long_search_result_is_truncated_before_injection` |
| B5 | 空输入或纯空白输入：明确处理，不触发 LLM 调用或死循环 | ✅ | `test_loop_errors.py::TestEmptyInput`（空串/纯空白/全角空格/`None` 四种，均断言 `len(llm.calls) == 0`、`iterations == 0`、不计入 turn） |
| B6 | 连续多轮追问后回到早期话题：早期状态（如 todo）仍可被召回 | ✅ | `test_context.py::test_compression_keeps_state_block`（压缩丢弃 7 轮后，早期待办仍在 context）；状态以置顶 system 块注入，见 `TestSessionStateInjection` |
| B7 | 两 session 交替高频切换：状态与上下文始终按 session 隔离，无串扰 | ✅ | `test_loop_errors.py::test_alternating_sessions_keep_separate_context`、`test_sessions_persist_independently`；`test_session_context_integration.py::TestSessionToolContextIntegration`（同一注册表服务两 session 仍隔离） |
| B8 | LLM 拒绝调用工具（坚持直接回答）：不陷入无效循环，能正常收敛 | ✅ | `test_loop_errors.py::test_model_refuses_tools_and_answers_directly`（`tool_calls==0`、1 次调用即收敛）；无进展兜底：`test_repeated_empty_responses_stop` |

## 6.4 交付验收

| 编号 | 验收项 | 结论 | 证据 |
| --- | --- | --- | --- |
| D1 | 测试覆盖第 5 节功能点并全部通过；测试可离线运行 | ✅ | 离线 `Ran 385 tests … OK`；覆盖矩阵见 README 第 5 节；真实 API 用例默认跳过（`skipped=10`） |
| D2 | 仓库无硬编码密钥、无真实 API Key，附 `.env.example` | ✅ | 特征串全文扫描零命中；`.env.example` 仅占位；`.gitignore` 排除 `.env`；`safe_summary()` 打码 |
| D3 | 按 README 步骤在干净环境可复现运行与跑测 | ✅ | 零第三方依赖，无需 `pip install`；`python -m agent --check-config`、`python -m unittest discover -s tests -t .` 均按 README 原样可执行 |
| D4 | 提交 GitHub 链接且可访问 | ✅ | <https://github.com/ckj337728-web/agent-vibe_coding-test>，匿名访问 `HTTP 200`；远端 43 个文件与本地完全一致 |
| D5 | 对照第 4 节约束逐条自检无违反 | ✅ | 见 [`docs/requirements-check.md`](requirements-check.md)，C-1 ~ C-10 全部通过，无未处理项 |

---

## 核验结论

- **6.1 正常路径**：12/12 通过
- **6.2 异常路径**：8/8 通过
- **6.3 边界情况**：8/8 通过
- **6.4 交付验收**：5/5 通过

**合计 33/33 通过，无未闭环项。**

## 需提交者完成的一件事

1. **轮换测试用 API 密钥**：该密钥已出现在对话记录中，建议提交前吊销。

## 推送结果（D4 证据）

| 项 | 结果 |
| --- | --- |
| 仓库地址 | <https://github.com/ckj337728-web/agent-vibe_coding-test> |
| 分支 | `main`，已建立 upstream 跟踪 |
| 提交数 | 3（初始实现 → 文档脱敏 → task.md 完成标记） |
| 远端文件数 | 43 |
| 本地与远端一致性 | `git diff HEAD origin/main` 为空，完全一致 |
| 匿名可访问性 | 仓库页与 `raw.githubusercontent.com` 上的 README 均返回 `HTTP 200` |
| 远端密钥扫描 | 逐个文件从 GitHub 拉取扫描，真实密钥 / 供应商信息 / 长密钥样式串**零命中** |

## 复核命令

```bash
python -m compileall -q agent tests
python -m unittest discover -s tests -t .            # 385 passed, 10 skipped
# 密钥泄漏扫描：把 <密钥特征串> / <端点域名> 换成你实际使用的值
grep -rn "<密钥特征串>\|<端点域名>" . --exclude-dir=.git --exclude-dir=__pycache__   # 应无命中
```
