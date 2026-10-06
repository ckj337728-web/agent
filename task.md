# task.md：从零实现最小可用 Agent —— 实施任务清单

> 依据 `spec.md` 拆解。任务按**实施顺序**线性排列：基础准备 → 功能实现 → 测试验收 → 交付。
> `[ ]` 未完成，`[×]` 已完成。每个任务独立可判完成，完成判定必须能由测试结果、
> trace 输出或文件内容客观核验。范围内仅含 `spec.md` 已要求的内容。

> **当前进度：全部 8 个阶段完成（T001–T051）**。所有任务均已完成并验证。
>
> 验证结果：
> - 离线（无密钥，默认）：`python -m unittest discover -s tests -t .` → **378 tests, OK (skipped=10)**
> - 真实 API（`LLM_API_KEY` + `LLM_E2E=1`）：同命令 → **378 tests, OK**（含 10 个真实网络用例）
> - CLI 真实通路：`python -m agent --session window-1 --prompt "…"` 跑通「search + todo」双工具调用，退出码 0
> - 双窗口隔离（真实 API，跨进程 + 落盘）：窗口1 待办仅「带伞」，窗口2 仅「写周报」，互不影响
> - 约束自检：`docs/requirements-check.md` 全部通过；验收核验：`docs/acceptance.md` **33/33 通过**
>
> 仓库状态：已推送至 <https://github.com/ckj337728-web/agent>（`main` 分支），
> 远端 43 个文件与本地完全一致；远端视角密钥扫描零命中。

## 阶段 0：现有资产探查（先行，避免重复造轮子）

- [×] T001 探查当前工作区是否已有可复用代码/项目结构（源码目录、依赖清单、测试框架、构建脚本），记录可复用项与既有架构约定
  - 产出：`notes/asset-survey.md`（可复用文件清单 + 架构约定）
  - 完成判定：清单中每项标注"复用 / 不适用"及理由；确认无既有 Agent Runtime 代码可复用（`1.txt`、`spec.md`、`task.md` 为文档，非代码）
  - 完成情况：工作区仅含 3 个文档，无源码/依赖清单/构建脚本/测试框架，且非 git 仓库。结论：无可复用代码；"复用"要求转化为「零第三方依赖」+「复用本阶段建立的模块契约」

## 阶段 1：基础准备

- [×] T002 确定语言/运行时与依赖管理方式（Python 或 Node/TS 二选一，对应 spec 第 7 节待确认项），记录决策与理由
  - 产出：选型记录（写入 `README.md` 草稿的选型小节）
  - 完成判定：选定唯一语言与依赖管理工具，README 中可见明确结论
  - 完成情况：选定 **Python 3.10+（验证于 3.12.14）+ 零第三方依赖**；决策与理由见 `README.md`「技术选型」
- [×] T003 初始化项目骨架：源码目录、测试目录、依赖清单、运行入口
  - 产出：目录结构 + 依赖清单文件 + 可执行入口
  - 完成判定：按依赖清单安装成功；入口可执行并输出可辨识的启动信息（如"配置未就绪"提示）
  - 完成情况：`agent/`（`__init__` / `__main__` / `config` / `llm` / `trace`）+ `tests/` + `requirements.txt`（零依赖）+ `.gitignore`；`python -m agent` 未配置密钥时输出「[配置未就绪] …」且退出码 2
- [×] T004 实现配置管理：LLM `base_url` / `model` / `api_key` 从环境变量读取，附示例配置
  - 产出：配置模块 + `.env.example`
  - 完成判定：全仓库检索无硬编码密钥（见 spec C-7/C-8）；未设置 `api_key` 时启动给出明确报错而非堆栈外泄
  - 完成情况：`agent/config.py` + `.env.example`；检索 `sk-[A-Za-z0-9]{10,}` 无匹配，仓库内无 `.env`；`safe_summary()` 仅保留密钥末 4 位
- [×] T005 实现 LLM 客户端封装：调用真实 API，含超时、重试、错误归一化
  - 产出：LLM client 模块（接口 + 真实实现）
  - 完成判定：配置有效密钥后可完成一次真实调用并返回响应；超时/网络错误/限流被归一化为统一错误类型
  - 完成情况：`agent/llm.py`；错误家族统一继承 `LLMError`（超时/网络/限流/5xx 可重试，401/403/其他 4xx/响应格式非法不重试）；传输层可注入，真实实现基于标准库 `urllib`
  - 说明：本轮环境无有效密钥，故按 spec 4.3 以注入式 fake transport 覆盖各分支；真实 API 通路由 T044 承担
- [×] T006 实现 trace/日志基础设施：结构化记录 LLM 调用与工具调用
  - 产出：tracer/logger 模块
  - 完成判定：一次调用后可产出含时间、类型、入参、结果、耗时的可读记录（对应 spec 3.7、N11）
  - 完成情况：`agent/trace.py`，支持 text/jsonl 双格式、`trace_id` 串联一次用户输入、超长截断、写入失败自动降级不影响主流程

## 阶段 2：功能实现 —— 工具层

- [×] T007 定义统一工具契约：`name` / `description` / `parameters`(JSON Schema 风格)，含执行入口约定
  - 产出：工具接口（基类或协议）定义
  - 完成判定：契约明确要求三个字段且约定返回值结构；新增一个空壳工具无需修改主循环即可被注册
  - 完成情况：`agent/tools/base.py` 的 `BaseTool` + `ToolResult` + `ToolContext`；`__init_subclass__` 在类定义时即校验 `name`/`description` 非空；`ToolContext` 注入 `session_id` 与状态存储，使工具无需自持跨会话状态
- [×] T008 实现工具注册表：注册、列举、按名获取、生成 LLM 可见的工具 Schema 列表
  - 产出：registry 模块
  - 完成判定：注册工具后自动出现在导出的工具列表中；按名获取不存在的工具返回可识别错误
  - 完成情况：`agent/tools/registry.py`；`register`/`get`/`list_tools`/`names`/`to_llm_schemas`/`execute`；同名重复注册报错而非静默覆盖；`build_default_registry()` 与 `default_registry()` 提供内置三工具
- [×] T009 实现 `calculator` 工具（表达式求值）
  - 产出：calculator 工具实现
  - 完成判定：正常表达式返回正确结果；非法表达式、除零返回可恢复错误而非抛出未捕获异常（spec 3.2）
  - 完成情况：`agent/tools/calculator.py`；基于标准库 `ast` 自行求值，**不使用 eval**；支持 + - * / // % ** 与白名单函数/常量；非法语法、除零、未知函数、属性访问、关键字参数、幂次过大均返回结构化错误
- [×] T010 实现 `search` 工具（mock 数据源）
  - 产出：search 工具实现
  - 完成判定：返回结构化结果列表；查询无结果时返回明确的空结果而非报错（spec 6.2 E3）
  - 完成情况：`agent/tools/search.py`；`SearchBackend` 为可替换的 mock 后端（6 篇内置语料），返回 `title/snippet/url/score` 结构化结果；无结果返回 `ok=True` + 空列表 + 友好提示
- [×] T011 从 `read_docs` / `todo` / `weather` 中选定并实现第三个工具（满足 spec C-4 的"至少 3 个"）
  - 产出：第三个工具实现
  - 完成判定：工具可用且具备真实状态或数据语义；若选 `todo`，写入后能在同 session 后续对话中被召回
  - 完成情况：选定并实现 **`todo`**（理由：唯一具备真实状态语义的选项，可直接验证 spec 3.6 的双窗口隔离）；`TodoStore` 以 `session_id` 为键隔离，工具经 `ToolContext` 取会话标识；已用测试验证"窗口1 写入 → 窗口2 不受影响 → 切回窗口1 仍能召回"
- [×] T012 实现工具调用的参数校验与错误规范化：未知工具、缺参、参数类型错误
  - 产出：校验与错误归一化逻辑
  - 完成判定：三类错误均返回结构化错误结果（不崩溃），可被回灌给 LLM（spec 3.3、6.2 E1/E2）
  - 完成情况：`validate_arguments()` 支持 type/required/enum/minimum/maximum/minLength/maxLength/array items/default，且显式排除 Python 中 `bool` 属 `int` 子类的陷阱；`ToolRegistry.execute()` 把未知工具、缺参、类型错误、工具内部异常、非法返回值五类问题收敛为 `ToolResult`，`to_llm_text()` 输出 `ERROR [code]: message` 供回灌

## 阶段 3：功能实现 —— LLM 输出解析

- [×] T013 定义结构化输出协议：思考过程 / 工具调用 / 最终答案三类内容的字段与区分规则
  - 产出：输出协议定义
  - 完成判定：协议能唯一区分三类内容（含同一响应同时含工具调用与最终答案的情形，spec 3.4）
  - 完成情况：`agent/parser.py` 的 `OutputKind`（final_answer / tool_calls / mixed / empty）、`ParseSource`、`ToolCallRequest`、`ParsedOutput`；协议字段为 `thought` / `tool_calls`（含 `name`+`arguments`）/ `answer`，并附 `OUTPUT_PROTOCOL_INSTRUCTIONS` 提示词片段注入 system prompt
  - 说明：`mixed` 专门表示"同一响应既有工具调用又有答复"，区分规则为：有工具调用→继续 loop，无→收敛（详见 T014）
- [×] T014 实现输出解析器：从 LLM 响应中提取思考过程、工具调用（工具名+参数）、最终答案
  - 产出：parser 模块
  - 完成判定：对规范响应可稳定提取三类内容，且工具调用参数被解析为结构化对象
  - 完成情况：`parse_response(content, native_tool_calls)` 三级优先级：① API 原生 `tool_calls`（首选）→ ② 文本中的协议 JSON（裸 JSON / ```json 代码块 / 从混合文本中按括号平衡提取）→ ③ 纯文本整体作为答案；支持字段别名容错、OpenAI 嵌套风格、`arguments` 为对象或 JSON 字符串
  - 说明：原生 tool_calls 场景下 content 按协议认定为**思考过程**（模型"边想边调工具"的文字不是给用户的答复），仅当出现 `<thinking>` 标签时才把标签外文字认定为答复 → `mixed`
- [×] T015 实现解析失败回退策略：JSON 非法、字段缺失、格式异常
  - 产出：回退逻辑
  - 完成判定：三类失败输入均不崩溃，并按既定回退路径（重新提示 / 错误注入 / 按纯文本收尾）产出结果（spec 3.4、6.2 E5）
  - 完成情况：截断 JSON、非法 JSON、代码块内容非法、缺 `function`/`name` 字段、`arguments` 非法或非对象、只有 `thought` 无答案、空输入等全部转为结构化结果并写入 `warnings`；`arguments` 解析失败时保留 `raw_arguments` 且标记 `argument_error`，交给主循环按参数错误回灌（与 T012 的错误规范化衔接）
  - 关键取舍：协议 JSON 里既无工具也无有效答案时返回 `EMPTY` 而非把 JSON 原文当答复，避免把 `{"thought": ...}` 泄露给用户

## 阶段 4：功能实现 —— Session 与 Context

- [×] T016 实现 session 模型与生命周期：创建、按 id 获取、续聊
  - 产出：session 模块
  - 完成判定：给定已存在的 session id 可恢复其历史；不存在的 session 有明确处理（新建或明确报错，spec 6.2 E6）
  - 完成情况：`agent/session.py` 的 `Session` / `Message` / `ToolCallRecord` / `SessionStore`；`get`（返回 None）、`get_or_create`（新建策略）、`require`（明确报错并列出已有 session id，spec 6.2 E6 两种策略都提供）；消息按 `turn` 编号分组，便于整轮压缩
- [×] T017 实现消息与 session 级状态持久化（用户输入、助手输出、工具结果、工具产生的状态）
  - 产出：存储层实现
  - 完成判定：重新获取同一 session 后其历史消息与状态可完整读回（spec 3.6、6.1 N8）
  - 完成情况：`SessionStore(path=...)` 落盘为单个 JSON（先写 `.tmp` 再 `os.replace`，避免半个文件）；`Session.state` 存放会话级状态；**`todo` 工具的状态已改为直接写入 `Session.state`**（键 `<session_id>:todos`，JSON 可序列化），从而随 session 一起持久化
  - 说明：session 保存**完整**历史不做截断——"随时续聊并正确恢复"要求历史完整；prompt 尺寸由 context 层控制
- [×] T018 实现按 session 隔离的 context 构建：取数范围严格限定为当前 session
  - 产出：context 构建逻辑
  - 完成判定：两个 session 各自构建的 context 不含对方任何消息；同 session 历史完整（spec 6.3 B7）
  - 完成情况：`ContextBuilder.build(session)` 只读取传入的 `Session` 实例；集成测试验证"同一注册表服务两个 session"时 context 互不串扰
- [×] T019 实现 context 组装策略：明确哪些信息入 context（用户输入必放；工具结果按需截断；思考过程是否回灌给出取舍）并落地
  - 产出：策略说明 + 落地实现
  - 完成判定：策略文档与实现一致，逐条说明"放什么、为何放"（spec 3.5）
  - 完成情况：`agent/context.py` 的 `ContextPolicy` 把每项取舍变成显式可配置项。组装顺序固定为 **系统提示 → 协议说明 → session 状态快照 → 对话历史**。四项取舍：① 用户输入必放；② 工具结果必放（带工具追问的前提）但注入前按 `tool_result_max_chars` 截断；③ 思考过程默认**不放回**（价值衰减快、体积大，且协议不依赖它），可用 `include_reasoning` 开启；④ session 状态**独立置顶**而非混在历史里（混进去会被压缩丢掉，导致切回旧窗口待办消失）
- [×] T020 实现纯对话追问支持
  - 产出：context/上游对多轮纯文本对话的处理
  - 完成判定：第二轮及以后的纯文本提问能正确关联前文内容（spec 6.1 N9）
  - 完成情况：历史按轮次完整保留；`kept_turns` 统计参与断言；测试验证"我叫小明 → 你好小明 → 我叫什么？"三轮上下文齐备
- [×] T021 实现带工具的追问支持：后续轮次可基于前序工具结果继续调用工具
  - 产出：工具结果入 context 并参与后续推理的实现
  - 完成判定：追问依赖前序工具（如先查天气再问"那明天呢"）时能正确发起新的工具调用（spec 6.1 N9）
  - 完成情况：工具结果以 `role=tool` 消息保留并回放，同时**回放配对的 `assistant.tool_calls`**；组装末尾统一清理孤儿 tool 消息（OpenAI 兼容接口要求成对，否则请求被拒）
- [×] T022 实现基础压缩：context 超阈值时按轮次/字符数截断或对历史做摘要，阈值可配置
  - 产出：压缩逻辑 + 配置项
  - 完成判定：超长 context 触发压缩，压缩后对话仍可正确追问前文（spec 6.3 B3）
  - 完成情况：按**整轮**从最旧开始丢弃（整轮丢而非逐条丢，保证 tool_calls↔tool 配对不被打断）；两阶段丢弃——先丢到 `keep_recent_turns` 为止，仍超限则继续丢到只剩最后一轮；被丢轮次的用户输入折叠成摘要；`max_context_chars` / `keep_recent_turns` / `max_summary_chars` 均可配置
  - 关键取舍：`max_summary_chars` 是**硬上限**（先构造后裁剪）。摘要超限会抵消压缩的意义，而逐字符预算因片段自带 `...` 极易差几个字符——该缺陷被新加的回归测试抓到并修复
- [×] T023 实现超长工具输出的截断或摘要后入 context
  - 产出：工具结果入 context 前的尺寸控制
  - 完成判定：注入超长工具输出后 context 尺寸受控，且不撑爆上限（spec 6.3 B4）
  - 完成情况：`truncate_tool_result()` 按 `tool_result_max_chars`（默认 2000）截断并标注"省略 N 字符"；保留头部（工具结论通常在前，如计算结果与检索命中列表）；截断数量计入 `tool_results_truncated` 便于断言

## 阶段 5：功能实现 —— Agent 主循环（Core Runtime）

- [×] T024 实现主循环 Step 1：接收用户输入并写入对应 session
  - 产出：loop 输入环节
  - 完成判定：输入被正确关联到指定 session 并计入历史（spec 3.1）
  - 完成情况：`agent/loop.py` 的 `AgentLoop.run_turn()`；输入先 strip 再写入 session 并开启新 turn；`step1_receive_input` 写入 trace，`trace_id` 贯穿整轮
- [×] T025 实现主循环 Step 2：组装请求（system prompt + 工具 Schema + context）并由 LLM 判定直接回复或调用工具
  - 产出：loop 决策环节
  - 完成判定：同一实现可分别触发"直接回复"与"工具调用"两条路径（spec 6.1 N2/N3）
  - 完成情况：复用 `ContextBuilder` 组装（system prompt + 协议说明 + 状态快照 + 历史），`registry.to_llm_schemas()` 注入工具定义并设 `tool_choice=auto`；`step2_build_request` / `step2_parse_result` 入 trace；直接回复与工具调用两条路径均有测试覆盖
- [×] T026 实现主循环 Step 3：执行工具调用，支持一次响应含多个调用并按序执行
  - 产出：loop 工具执行环节
  - 完成判定：多个调用全部被执行，每个调用的入参、输出、耗时、错误进入 trace（spec 3.1、6.3 B2）
  - 完成情况：按序执行一次响应中的全部调用，每个调用都记录 `arguments`/`result`/`duration_ms`/`ok`/`error_code` 四要素；调用记录挂到对应 assistant 消息上，保证回放时 `assistant.tool_calls ↔ tool` 成对
- [×] T027 实现主循环 Step 4：终止判定——工具有调用则回灌结果继续 loop，无调用则收敛为最终答案
  - 产出：loop 终止环节
  - 完成判定：工具结果回灌后能继续推理并得出整合了工具结果的最终答案（spec 6.1 N3/N4）
  - 完成情况：工具结果以 `role=tool` 消息落库，下一轮由 `ContextBuilder` 统一渲染回灌（**不额外注入**，避免同一结果出现两次——该重复缺陷被测试抓到）；无工具调用时收敛为 `final_answer` 并写回 session
- [×] T028 实现最大轮次限制（可配置），超限时优雅终止并返回部分结论与超限说明
  - 产出：轮次控制逻辑 + 配置项
  - 完成判定：设置极小上限时可稳定触发收尾，不进入死循环、不崩溃（spec 6.3 B1、C-6）
  - 完成情况：`LoopPolicy.max_iterations`；超限时返回"已达最大循环次数 N + 本轮工具调用次数 + 已获得的部分结果"，并发出 `limit` 事件、写回 session
- [×] T029 实现主循环异常处理路径：LLM 失败/超时、工具异常、轮次上限
  - 产出：异常处理逻辑
  - 完成判定：三类异常均有明确处理路径且无未捕获异常；LLM 重试耗尽时返回可解释错误（spec 3.7、6.2 E4）
  - 完成情况：`LLMError` 一律转为 `stop_reason=llm_error` 与可读答复（含超时/限流/鉴权文案），并记 `step2_llm_failed` trace；工具异常由注册表归一化后再回灌；轮次上限见 T028
- [×] T030 实现"LLM 坚持直接回答、拒绝调用工具"时的收敛处理
  - 产出：收敛逻辑
  - 完成判定：该情形下不产生无效循环，能正常给出最终答案（spec 6.3 B8）
  - 完成情况：模型直接给答案即一轮收敛；对"既无工具也无答案"的响应，先按 spec 3.4 走**重新提示**（`max_parse_retries`，默认 1 次），仍无效则以 `parse_error` 明确收尾；另有 `max_no_progress_rounds` 兜底，杜绝空转
- [×] T031 实现空输入或纯空白输入的明确处理
  - 产出：输入校验逻辑
  - 完成判定：空/纯空白输入不触发 LLM 调用、不进入循环，并返回明确提示（spec 6.3 B5）
  - 完成情况：`""` / 空白 / 全角空格 / `None` 均短路，`stop_reason=empty_input`、`iterations=0`、**0 次 LLM 调用**，且不计入 turn 数；测试断言 `len(llm.calls) == 0`
- [×] T032 实现工具执行超时保护
  - 产出：工具执行超时控制
  - 完成判定：模拟长时间无响应的工具时，loop 不被阻塞，超时被归一化为可恢复错误（spec 6.2 E8）
  - 完成情况：`LoopPolicy.tool_timeout_seconds`；在守护线程中执行工具，超时即放弃等待并返回 `tool_timeout` 结构化错误（Python 无法安全杀线程，故选择"放弃等待、不阻塞 loop"），错误回灌后 LLM 可继续作答
- [×] T033 提供可交互运行入口（CLI），支持显式指定 session id 以驱动主循环
  - 产出：运行入口（CLI）
  - 完成判定：可在同一进程内以两个不同 session id 交替对话并各自保持上下文（spec 3.6）
  - 完成情况：`python -m agent` 支持 `--session`（指定窗口）、`--store`（落盘）、`--prompt`（单次提问，便于脚本化）、`--max-iterations`、`--tool-timeout`、`--max-context-chars`、`--trace {off,text,jsonl}`、`--list-sessions`、`--check-config`；交互模式支持 `:exit` / `:new` / `:id` / `:help`
  - 说明：仅实现驱动主循环所需的最小交互入口，不做 spec 未要求的界面功能（spec 第 7 节未要求 Web/GUI）

## 阶段 6：测试验收（随对应模块完成后即时补齐）

- [×] T034 搭建测试框架与 LLM 注入机制，使测试可离线确定性运行
  - 产出：测试基础设施
  - 完成判定：测试在无网络、无密钥环境下可重复通过（spec 4.3、6.4 D1）
  - 完成情况：`tests/helpers.py` 提供 `FakeTransport`（可编程响应序列 + 请求留痕）、`sample_config()`、`chat_completion_payload()`、`tool_call()`、`make_client()`；阶段 5 又补充 `ScriptedLLM`（按脚本驱动主循环并留痕每条发送消息）、`RecordingTool`（可记录调用/模拟耗时/模拟故障）；未联网、未使用任何真实密钥即通过全部测试
  - 说明：本轮提前建立（阶段 1 模块需即时验证），故从阶段 6 提前完成；后续阶段 2–5 的新测试直接复用此基础设施
- [×] T035 编写主循环测试：直接回复路径、工具调用路径、链式/多工具调用
  - 产出：主循环测试用例
  - 完成判定：三类用例全部通过（spec 5.1）
  - 完成情况：`tests/test_loop.py`（27 个用例）覆盖直接回复（协议 JSON 与纯文本回退）、单次工具调用、链式调用、单响应多工具调用、工具结果回灌与 assistant.tool_calls 回放、双窗口隔离、trace 四步骤与事件顺序；另用内置三工具跑通 calculator 与 todo 的真实工具链
- [×] T036 编写工具层测试：`calculator`、`search`、第三个工具的正常与异常用例（非法表达式、除零、无结果）
  - 产出：工具测试用例
  - 完成判定：全部通过（spec 5.2）
  - 完成情况：`tests/test_tool_calculator.py`（22 例）、`tests/test_tool_search.py`（19 例）、`tests/test_tool_todo.py`（28 例）；含正常计算、非法表达式、除零、无结果、空结果、类型错误、越界、跨会话隔离等
- [×] T037 编写注册表与 Schema 测试：可列举、Schema 可生成、新增工具无需改主循环
  - 产出：注册表测试用例
  - 完成判定：全部通过（spec 5.3）
  - 完成情况：`tests/test_tool_registry.py`；覆盖注册/列举/取用/重名拒绝、三工具 Schema 形状校验、以及"注册新工具后自动出现在 LLM 工具列表且主循环无需改动"
- [×] T038 编写工具错误路径测试：未知工具、缺参、参数类型错误
  - 产出：错误路径测试用例
  - 完成判定：三类错误均返回结构化错误且不崩溃（spec 6.2 E1/E2）
  - 完成情况：`tests/test_tool_registry.py` 的 `TestExecuteErrorNormalization` 与 `TestValidateArgumentsDirectly`；另覆盖工具内部异常（E3）、非法返回值、`None` 参数、非对象参数等边界
- [×] T039 编写解析层测试：三类内容解析 + 解析失败回退
  - 产出：解析层测试用例
  - 完成判定：全部通过（spec 5.4）
  - 完成情况：`tests/test_parser.py`（43 例）；覆盖原生 tool_calls 路径、协议 JSON 路径（裸/代码块/混合文本/数组包裹/字段别名/OpenAI 嵌套风格）、纯文本回退、11 种畸形输入"绝不抛异常"、括号平衡与转义引号、三级优先级判定
- [×] T040 编写 Session 测试：多 session 隔离、历史恢复、跨 session 不串扰，并覆盖双窗口场景（窗口1 查天气+记待办；窗口2 写周报+记待办，各可续聊）
  - 产出：session 测试用例
  - 完成判定：全部通过，且双窗口场景中两 session 状态互不影响（spec 3.6、6.1 N8）
  - 完成情况：`tests/test_session.py`（35 个用例，含落盘重载、损坏文件、重复创建、`require` 报错内容等）+ `tests/test_session_context_integration.py`（9 个用例，Session × Context × 工具三方联动，含双窗口全流程、切回旧窗口、落盘重载后仍隔离）
- [×] T041 编写 Context 测试：纯对话追问、带工具追问、最大轮次限制、基础压缩触发、超长工具输出截断
  - 产出：context 测试用例
  - 完成判定：全部通过（spec 5.6）
  - 完成情况：`tests/test_context.py`（41 个用例）；覆盖组装顺序与四项取舍、状态快照置顶与不丢、两类追问、压缩触发/保留最近轮次/最旧先丢/摘要硬上限、超长工具结果截断、压缩后 tool 配对完整性
  - 说明：当时"最大轮次限制"属 loop 上限（T028）未覆盖；阶段 5 已在 `tests/test_loop_errors.py` 的 `TestMaxIterations` 中补齐 loop 轮次上限测试
- [×] T042 编写异常与 trace 测试：LLM 失败、工具失败、轮次上限、trace 记录完整性
  - 产出：异常与 trace 测试用例
  - 完成判定：全部通过，且 trace 中可回溯每次 LLM 与工具调用（spec 5.7、6.1 N11）
  - 完成情况：`tests/test_loop_errors.py`（38 个用例）覆盖 LLM 超时/限流/鉴权失败、未知工具、缺参、类型错误、工具内部异常、参数 JSON 截断、工具超时、轮次上限、无进展收敛、解析重试；trace 部分断言四步骤事件齐备、每次迭代可回溯、工具 trace 含 `error_code`、以及走真实 `LLMClient` 时确实产生 `llm_call` 事件
- [×] T043 编写配置缺失与不存在 session 的测试
  - 产出：配置/session 异常测试用例
  - 完成判定：两种情形均返回明确提示，不静默失败（spec 6.2 E6/E7）
  - 完成情况：配置缺失见 `tests/test_config.py` 与 `test_loop_errors.py::TestMissingConfig`（含 CLI `main()` 返回码 2 且输出无 `Traceback` 的断言）；session 不存在见 `tests/test_session.py`（`require` 报错内容含已有 id）与 `test_loop_errors.py::TestSessionErrors`（未知 session 自动创建、损坏存储文件报错）
- [×] T044 编写可选的端到端测试：在配置真实密钥时走通真实 LLM API 通路
  - 产出：E2E 测试用例
  - 完成判定：有密钥环境下通过；无密钥环境下自动跳过且不导致测试套件失败（spec 4.3）
  - 完成情况：`tests/test_e2e_real_llm.py`（10 个用例，4 个测试类）。**双重闸门**：需 `LLM_API_KEY` 已设置且 `LLM_E2E=1` 才运行，避免误触发计费调用；缺任一条件即自动 `skipTest` 并说明原因。覆盖：直接回复与答案落库、`llm_call` 进 trace、calculator 原生 tool_calls + 结果回灌、工具 Schema 注入生效、非法参数不崩溃、todo「add→list」链式调用、带工具的追问（(20+22)×2=84）、双窗口待办隔离、落盘后重载
  - **已用真实 API 实测通过**（OpenAI 兼容端点；供应商、端点与模型名按环境变量注入，出于安全考虑不写入仓库）：
    - 有密钥 + `LLM_E2E=1`：`Ran 378 tests … OK`（含 10 个真实网络用例）
    - 无密钥（离线默认）：`Ran 378 tests … OK (skipped=10)`，套件不失败
  - 稳健性设计：真实模型是否调用工具具有不确定性，故断言只针对**客观事实**（会话里是否真的记录了工具调用、工具结果是否进入下一轮、会话状态是否隔离）；模型本次未用工具时记为 `skipTest` 而非失败，但一旦用到工具就对运行时行为做严格断言
  - **安全**：密钥仅经环境变量注入，仓库内不含真实密钥、供应商地址与真实模型名（已用特征串全文扫描确认，spec C-7 / C-8）

## 阶段 7：文档与交付

- [×] T045 编写 README - 运行方式：安装、配置、启动、跑测试
  - 产出：README 运行章节
  - 完成判定：按步骤在干净环境可复现安装、启动与测试（spec 6.4 D3）
  - 完成情况：README 第 1 节（4 小节）：环境要求（零依赖，无需 `pip install`）、配置真实 API（含全部环境变量表与 `.env.example` 对照）、启动（CLI 参数表 + 交互命令 + **双窗口用法示例**）、运行测试（离线命令 + 可选真实 API E2E 的双重开关说明）
- [×] T046 编写 README - 系统设计：模块划分、主循环四步骤、工具注册机制、解析与异常策略
  - 产出：README 设计章节
  - 完成判定：内容与代码实际结构一致（spec 6.1 N12）
  - 完成情况：README 第 2 节（6 小节）：分层与模块职责表 + 数据流图、主循环四步骤对照 trace 事件名 + 决策表、工具注册机制（含 `BaseTool` 代码骨架与三个工具的实现要点）、LLM 输出解析三级优先级与两处关键取舍、异常处理表（逐项对应 spec 验收编号）、Session 与多窗口隔离设计
  - 备注：文中函数名、文件名、配置项均已与代码逐一核对（`test_` 类引用另经脚本校验真实存在）
- [×] T047 编写 README - memory 的召回时机与放置方式说明
  - 产出：README memory 章节
  - 完成判定：明确回答"何时召回、放在 context 的哪个位置、为何如此"（spec 3.5、4.2 C-10）
  - 完成情况：README 第 3 节把记忆分三类（会话历史 / 会话状态 / 思考过程），每类逐一说明**存放位置、召回时机、放置位置、为何这样放**；附固定组装顺序图、三个关键设计决定（状态为何置顶、工具结果为何截断、压缩为何整轮丢弃）、相关配置项表
- [×] T048 编写 AI Prompt 与问题解决记录
  - 产出：记录文档
  - 完成判定：覆盖关键 prompt 与典型问题解决过程（spec 4.2 C-9）
  - 完成情况：`notes/ai-usage.md`：人机分工表、6 组实际使用过的关键 prompt（含"效果/收获"复盘）、AI 帮上忙的 5 类场景、**AI 帮倒忙的 8 个真实案例**（含 `sessions or SessionStore()` 落盘失效、工具结果重复注入、摘要超上限、测试误读环境变量等）与修正方式、5 条协作经验
- [×] T049 对照 spec 第 4 节约束逐条自检（重点 C-1 无框架承载主流程、C-3 真实 LLM API、C-7/C-8 无密钥）
  - 产出：自检结果记录
  - 完成判定：每条约束标注通过/不通过及证据，无未处理的不通过项（spec 6.4 D5）
  - 完成情况：`docs/requirements-check.md`：C-1 ~ C-10 共 10 条 + 4.3 兼容性 5 条 + 4.4 "不能修改的部分" 5 条，逐条给出结论与可复核证据（文件/命令/测试），并附一键复核命令。**结论：全部通过，无未处理项**
- [×] T050 对照 spec 第 6 节验收清单逐条核验（6.1 正常路径、6.2 异常路径、6.3 边界情况、6.4 交付）
  - 产出：验收核验结果
  - 完成判定：全部条目可勾选并有对应测试或证据支撑（spec 6.4 D1）
  - 完成情况：`docs/acceptance.md`：N1–N12、E1–E8、B1–B8、D1–D5 共 33 条逐条核验，每条给出对应测试方法名或文件证据。**结果 33/33 通过**
  - 证据可信性：文档中引用的 84 个测试方法/文件名经脚本校验**全部真实存在**，无编造引用
- [×] T051 提交 GitHub 仓库链接并确认仓库内无 API Key
  - 产出：可访问的仓库链接
  - 完成判定：链接可访问；全仓库检索无真实密钥（spec 4.2 C-7/C-9）
  - 完成情况：**已推送至 <https://github.com/ckj337728-web/agent>**（`main` 分支，已建立 upstream 跟踪）。验证：
    - 远端 43 个文件与本地 `origin/main` **完全一致**（`git diff` 为空），工作区 clean
    - 匿名访问验证：仓库页与 `raw.githubusercontent.com` 上的 README 均返回 **HTTP 200**
    - **远端视角密钥扫描**：从 GitHub 逐个拉取全部 43 个文件内容扫描，真实密钥 / 供应商域名与名称 / 真实模型名 / 长密钥样式串**零命中**
  - 仓库配置：`.gitattributes` 统一行尾；`.gitignore` 生效（`__pycache__`、`.env` 均被忽略，仓库内不存在 `.env`）
  - 过程中发现并修复了自己文档中的脱敏遗漏：`docs/acceptance.md` 的 grep 示例曾含真实密钥前缀，`task.md`/`docs/requirements-check.md` 含供应商信息，均已改为占位符（提交前后各扫描一次确认）
  - 唯一待提交者完成事项：**轮换测试用 API 密钥**（该密钥已出现在对话记录中）

## 实施顺序与依赖

- 阶段 0 → 1 必须先行：T001 决定是否复用既有资产，T002 决定选型，缺失任一都会导致返工。
- 阶段 2 与阶段 3 可并行（工具层与解析层互不依赖），两者都完成后才进入阶段 5。
- 阶段 4 的 T016–T018 需在阶段 5 之前完成（主循环依赖 session 与 context）；T019–T023 可与阶段 5 交替推进。
- 阶段 6 中测试任务紧随对应模块完成即时编写（T035 依赖阶段 5，T036–T038 依赖阶段 2，T039 依赖阶段 3，T040–T041 依赖阶段 4，T042–T043 依赖阶段 5）。
- 阶段 7 的 T045–T047 可在阶段 5 完成后起草，T049–T051 为最后收尾。

## 范围边界（不做的内容）

以下均**不在本任务清单内**，因 `spec.md` 未要求或已明确排除：

- 多 Agent 协作、规划器/反思器等高级 Agent 范式（spec 2.2）。
- 复杂上下文压缩算法（仅需基础压缩，spec 2.2）。
- 生产级鉴权、多租户、可观测性平台接入（spec 2.2）。
- 真实搜索引擎接入（search 允许 mock，spec 2.2）。
- Web/GUI 界面、Web 服务化（spec 第 7 节列为待确认且未要求；T033 仅实现驱动主循环所需的最小 CLI）。
- 在 `calculator`、`search`、第三个工具之外追加更多工具（spec 要求"至少 3 个"，超出部分无要求）。
- 对既有文档 `1.txt`、`spec.md` 的改写（spec 4.4 明确只读不改）。
