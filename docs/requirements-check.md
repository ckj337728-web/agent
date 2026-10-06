# 对照 spec.md 第 4 节约束的逐条自检（T049）

> 自检日期：阶段 7
> 自检对象：当前仓库全部提交内容
> 判定方式：每条约束给出**结论 + 证据**（文件、命令或测试），证据可独立复核。
> 结论取值：✅ 通过 / ❌ 不通过 / N/A 不适用

---

## 4.1 实现方式约束（硬性，违反即不合格）

| 编号 | 约束 | 结论 | 证据 |
| --- | --- | --- | --- |
| C-1 | **不得依赖现成 Agent 框架承载主流程** | ✅ | `requirements.txt` 仅含注释说明，零第三方依赖；`agent/loop.py`（主循环）、`agent/parser.py`（解析）、`agent/session.py`（会话）、`agent/context.py`（上下文）、`agent/tools/registry.py`（调度）均为自行实现。全仓库检索无 `langgraph`/`openhands`/`openclaw`/`langchain`/`autogen` 等引用 |
| C-2 | 允许用 AI 工具辅助，但不得替代核心 Runtime | ✅ | AI 使用与分工记录见 [`notes/ai-usage.md`](../notes/ai-usage.md)；核心逻辑经人工 review 与测试验证 |
| C-3 | **必须接入真实 LLM API** | ✅ | `agent/llm.py` 的 `urllib_transport` 为默认传输层，发真实 HTTP 请求。**已用真实端点实测**：工具调用、链式调用、双窗口隔离均跑通（见 `task.md` T044 记录） |
| C-4 | 至少 3 个工具，覆盖 calculator / search / read_docs\|todo\|weather | ✅ | `agent/tools/` 下实现 `calculator`、`search`、`todo` 三个；`test_tool_registry.py::test_default_registry_exposes_three_tools` 断言数量与名称 |
| C-5 | 工具须具备注册机制（name/description/参数 Schema），LLM 基于 Schema 自主决策 | ✅ | `BaseTool` 定义三字段；`ToolRegistry.to_llm_schemas()` 导出为 LLM 工具定义并在请求体中传入（`tool_choice=auto`）；`test_tool_registry.py` 校验 Schema 形状，E2E 中真实模型据此成功调用 |
| C-6 | 单轮 loop 必须受限，不得无限循环 | ✅ | `LoopPolicy.max_iterations`（默认 8）为硬上限；另有 `max_parse_retries` 与 `max_no_progress_rounds` 兜底。`test_loop_errors.py::TestMaxIterations` 用"永不收敛"脚本验证恰好执行 N 次即停止 |

**C-1 复核命令**：

```bash
# 依赖检索（应无第三方框架）
grep -rniE "langgraph|openhands|openclaw|langchain|autogen|llama_index" --include=*.py --include=*.txt .
```

**C-3 复核方式**：设置 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 后执行
`python -m agent --prompt "用工具计算 1234*5678"`，观察 trace 中 `[llm_call]` 的
`url=` 为真实端点且 `duration_ms` 为真实耗时。

---

## 4.2 交付与安全约束

| 编号 | 约束 | 结论 | 证据 |
| --- | --- | --- | --- |
| C-7 | **不要提供 API Key**，提交物不得含真实密钥 | ✅ | 全文特征串扫描（真实密钥片段、供应商域名、真实模型名）**零命中**；密钥仅经环境变量注入；`.env` 已被 `.gitignore` 排除且仓库内不存在 |
| C-8 | 仓库不得硬编码密钥，附 `.env.example` | ✅ | `.env.example` 仅含占位说明；`agent/config.py` 全部经 `os.environ` 读取；`safe_summary()` 对密钥打码只留末 4 位 |
| C-9 | 提交形式为 GitHub 链接，附 README 与 AI Prompt/问题解决记录 | ✅ | 仓库地址 <https://github.com/ckj337728-web/agent-vibe_coding-test>（已验证匿名 HTTP 200 可访问）；README 完整；AI 记录见 [`notes/ai-usage.md`](../notes/ai-usage.md) |
| C-10 | README 必须覆盖运行方式、系统设计、memory 召回时机与放置方式 | ✅ | README 第 1 节（运行方式）、第 2 节（系统设计）、第 3 节（memory） |

**C-7 复核命令**：

```bash
# 真实密钥特征串扫描（应无命中）
# 把 <密钥特征串> / <端点域名> 换成你实际使用的值再执行
grep -rn "<密钥特征串>\|<端点域名>" . --exclude-dir=.git --exclude-dir=__pycache__
```

---

## 4.3 兼容性与可运行性约束

| 约束 | 结论 | 证据 |
| --- | --- | --- |
| 干净环境按 README 可复现运行，依赖清单完整、无隐式本机依赖 | ✅ | 零第三方依赖，无需 `pip install`；`python -m agent --check-config` 可直接运行 |
| 自动化测试须能**离线**运行，不依赖真实网络与密钥 | ✅ | `python -m unittest discover -s tests -t .` → **385 passed**（无网络、无密钥）；LLM 层被 `FakeTransport` / `ScriptedLLM` 替换 |
| 真实 API 通路作为**可选** E2E | ✅ | `tests/test_e2e_real_llm.py` 需 `LLM_API_KEY` **且** `LLM_E2E=1` 才运行；缺任一条件自动跳过 |
| 配置缺失时给出明确报错，而非静默失败或堆栈外泄 | ✅ | `ConfigError` 携带可直接照抄的设置示例；CLI 捕获后打印提示并退出码 2；`test_loop_errors.py::TestMissingConfig` 断言输出中无 `Traceback` |
| 支持多 session 并发使用而不互相污染 | ✅ | `test_session.py`、`test_session_context_integration.py`、`test_loop.py::TestLoopWithRealRegistry::test_dual_window_isolation_through_loop`；真实 API 下跨进程实测窗口1/窗口2 待办互不影响 |

---

## 4.4 明确"不能修改或影响"的部分

| 约束 | 结论 | 说明 |
| --- | --- | --- |
| 不得修改题面要求本身 | ✅ | 四步骤、≥3 工具、session 隔离、context 管理、测试用例、交付物清单均完整实现，无裁剪 |
| 不得因引入第三方库而绕过 C-1 | ✅ | 未引入任何第三方库 |
| 不得改动既有笔试题材料（`1.txt` 只读） | ✅ | `1.txt` 未被修改（内容与最初读取一致） |
| 不得移除或弱化已有约束 | ✅ | `spec.md` 的约束条目完整保留，仅做过章节重排与内容补充 |
| 不得让测试以"关闭功能"方式通过 | ✅ | 压缩、trace、轮次上限、超时保护均有**正向**测试断言其生效（如 `test_compression_triggered_when_over_limit`、`test_limit_stops_endless_tool_calls`、`test_tool_timeout_does_not_block_loop`），不存在为过测而关闭功能的开关 |

---

## 自检结论

**第 4 节全部约束均通过，无不通过项。**

需要提交者本人完成的一件事（非代码问题）：

1. **轮换本次用于测试的 API 密钥**：该密钥已出现在对话记录中，
   建议在提交前吊销或轮换。

已完成（提交者操作 + 复核）：

- 仓库已推送至 <https://github.com/ckj337728-web/agent-vibe_coding-test>，
  匿名访问验证 `HTTP 200`；远端 43 个文件逐个拉取扫描，无密钥泄漏。
- 远端与本地 `origin/main` 内容完全一致，工作区 clean。

---

## 附：一键复核命令

```bash
# 编译检查
python -m compileall -q agent tests

# 离线全量测试（应 385 passed，10 skipped）
python -m unittest discover -s tests -t . 2>&1 | tail -3

# 真实 API 端到端（需密钥）
#   PowerShell: $env:LLM_API_KEY="..."; $env:LLM_BASE_URL="..."; $env:LLM_MODEL="..."; $env:LLM_E2E="1"
python -m unittest discover -s tests -t .

# 密钥泄漏扫描（应无命中真实密钥）
# 把 <密钥特征串> / <端点域名> 换成你实际使用的值再执行
grep -rn "<密钥特征串>\|<端点域名>" . --exclude-dir=.git --exclude-dir=__pycache__
```
