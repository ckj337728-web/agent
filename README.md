# 从零实现一个最小可用 Agent（Vibe Coding 笔试题）

一个**从零实现**的最小可用 Agent Runtime：主循环、工具注册与调度、LLM 输出解析、
session 管理与 context 压缩全部自行实现，**未使用任何 Agent 框架**承载主流程
（题面明确排除 langgraph / openhands / openclaw / PI）。

- 仓库地址：<https://github.com/ckj337728-web/agent>
- 代码行数：源码约 3600 行，测试约 4200 行
- 测试：**378 个用例**，离线全绿；另有 10 个真实 API 端到端用例
- 依赖：**零第三方依赖**，仅使用 Python 标准库

---

## 1. 运行方式

### 1.1 环境要求

| 项 | 要求 |
| --- | --- |
| Python | 3.10 及以上（开发与验证于 3.12.14） |
| 第三方依赖 | **无** |
| 网络 | 仅真实运行时需要（调用 LLM API）；跑测试**不需要**网络 |

无需 `pip install`。`requirements.txt` 刻意留空并注明原因。

### 1.2 配置真实 LLM API

密钥只能通过环境变量注入，**不会也不应写入仓库**（题面要求"不要给 API Key"）。

```bash
# 参考 .env.example 查看全部配置项；下面是最小配置
# Windows PowerShell
$env:LLM_API_KEY  = "<你的密钥>"
$env:LLM_BASE_URL = "https://<你的端点>/v1"     # 默认 https://api.openai.com/v1
$env:LLM_MODEL    = "<模型名>"                  # 默认 gpt-4o-mini

# Linux / macOS
export LLM_API_KEY="<你的密钥>"
export LLM_BASE_URL="https://<你的端点>/v1"
export LLM_MODEL="<模型名>"
```

本项目适配任意 **OpenAI 兼容**的 `/chat/completions` 端点（含原生 `tool_calls`）。

全部可用配置项：

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `LLM_API_KEY` | ✅ | — | 密钥；缺失时启动给出明确提示并以退出码 2 结束 |
| `LLM_BASE_URL` | | `https://api.openai.com/v1` | API 基地址 |
| `LLM_MODEL` | | `gpt-4o-mini` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | | `60` | 单次请求超时 |
| `LLM_MAX_RETRIES` | | `2` | 失败重试次数（仅对可重试错误生效） |
| `LLM_E2E` | | — | 设为 `1` 才启用真实 API 端到端测试 |

### 1.3 启动

```bash
python -m agent --check-config          # 只校验配置，不发起调用
python -m agent                          # 交互式对话（自动生成 session id）
python -m agent --session window-1       # 指定 session，即题面里的一个"窗口"
python -m agent --prompt "12*12 是多少"   # 单次提问后退出，便于脚本化
python -m agent --list-sessions          # 列出存储中的 session
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--session ID` | 指定 session（窗口）。用不同 id 即可在同一进程内开多个互不影响的窗口 |
| `--store PATH` | 把 session 落盘为 JSON；省略则仅保存在内存 |
| `--prompt TEXT` | 单次提问后退出（非交互模式） |
| `--trace {off,text,jsonl}` | trace 输出格式，默认 `text`（写入 stderr） |
| `--max-iterations N` | 单轮最大循环次数，默认 8 |
| `--tool-timeout S` | 单次工具执行超时秒数，默认 15 |
| `--max-context-chars N` | context 字符上限，超出即触发压缩，默认 24000 |

交互模式内置命令：`:exit` 退出、`:new` 新建 session、`:id` 显示当前 session id、`:help` 帮助。

**两个窗口的用法**（对应题面场景）：

```bash
python -m agent --store sessions.json --session window-1 --prompt "查广州天气并记待办带伞"
python -m agent --store sessions.json --session window-2 --prompt "写周报并记待办"
python -m agent --store sessions.json --session window-1 --prompt "我有哪些待办？"   # 只看到窗口1 的
```

### 1.4 运行测试

```bash
# 离线全量测试（不需要网络与密钥）
python -m unittest discover -s tests -t . -v

# 只跑某一层
python -m unittest tests.test_loop tests.test_parser -v
```

真实 API 端到端测试是**可选**的，需要双重开关，避免误触发计费：

```bash
$env:LLM_API_KEY="<密钥>"; $env:LLM_E2E="1"     # PowerShell
python -m unittest discover -s tests -t .        # 378 passed（含 10 个真实网络用例）
```

未设置 `LLM_API_KEY` 或未设置 `LLM_E2E=1` 时，这 10 个用例自动跳过，
套件仍为 `OK (skipped=10)`，不会失败。

---

## 2. 系统设计

### 2.1 分层与模块

严格按"关注点分离"分层，每层只做一件事，便于独立测试：

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| [`agent/config.py`](agent/config.py) | 从环境变量读取并校验配置 | 发起调用 |
| [`agent/llm.py`](agent/llm.py) | 调用真实 API、超时重试、错误归一化 | 解析语义、工具调度 |
| [`agent/parser.py`](agent/parser.py) | 从响应中提取思考 / 工具调用 / 最终答案 | 执行工具 |
| [`agent/tools/`](agent/tools/) | 工具契约、注册表、参数校验与执行 | 决定何时调用（由 LLM 决定） |
| [`agent/session.py`](agent/session.py) | 会话生命周期、消息与状态持久化、隔离 | LLM 调用、工具执行 |
| [`agent/context.py`](agent/context.py) | 选材、组装、截断/压缩后产出请求体 | 业务决策 |
| [`agent/loop.py`](agent/loop.py) | 编排以上各层、轮次控制、终止判定 | 具体工具实现 |
| [`agent/trace.py`](agent/trace.py) | 横切记录 LLM 与工具调用 | 参与业务逻辑 |

数据流：

```
CLI (agent/__main__.py)
  └─> AgentLoop.run_turn()
        ├─ SessionStore         取得 session（含历史与状态）
        ├─ ContextBuilder       组装 messages（system + 协议 + 状态 + 历史）
        ├─ LLMClient.chat()     调用真实 API（超时/重试/错误归一化）
        ├─ parse_response()     提取 thought / tool_calls / answer
        ├─ ToolRegistry.execute()  参数校验 → 执行 → 错误归一化
        └─ Tracer               全过程留痕
```

### 2.2 主循环四步骤

题面要求的四步在 [`AgentLoop.run_turn()`](agent/loop.py) 中显式分节实现，
每一步都写入 trace（事件名可直接在日志里检索）：

| 步骤 | 实现位置 | trace 事件 |
| --- | --- | --- |
| Step 1 接收用户输入 | 输入清洗后写入 session，开启新 turn | `step1_receive_input` |
| Step 2 判断直接回复还是调用工具 | 组装请求 → 调 LLM → 解析 | `step2_build_request` / `step2_parse_result` |
| Step 3 调用工具 | 按序执行一次响应中的全部调用，结果落库 | `tool_call`（含入参/输出/耗时/错误） |
| Step 4 判定继续还是返回 | 有工具调用 → 回灌继续；无 → 收敛 | `step4_finish` / `step4_limit_reached` |

决策表：

| 解析结果 | 处理 | 是否继续循环 |
| --- | --- | --- |
| `final_answer`（无工具调用） | 落库并返回用户 | 否 |
| `tool_calls` | 依次执行，结果作为 `role=tool` 消息回灌 | 是 |
| `mixed`（既有工具调用又有答复） | 先执行工具，答复留待工具结果合并后再产出 | 是 |
| `empty`（解析不出有效内容） | 先"重新提示"，重试耗尽则明确收尾 | 视重试而定 |
| LLM 调用失败 | 归一化错误，终止并返回可解释提示 | 否 |
| 达到轮次上限 | 停止调用工具，返回部分结论 + 超限说明 | 否 |

**三重防失控保护**：`max_iterations`（硬上限）、`max_parse_retries`（解析失败重试）、
`max_no_progress_rounds`（既无工具也无答案时兜底停止）。

**工具结果只回灌一次**：工具结果写入 session 成为 `role=tool` 消息，下一轮由
ContextBuilder 统一渲染。这一点很关键——早期实现既落库又在循环内额外注入，
导致 LLM 每轮收到两份相同结果，是被测试抓出来后修正的。

### 2.3 工具注册机制

每个工具是一个 `BaseTool` 子类，必须提供三个字段（[`agent/tools/base.py`](agent/tools/base.py)）：

```python
class CalculatorTool(BaseTool):
    name = "calculator"                      # 唯一名称，LLM 调用时使用
    description = "计算数学表达式…"            # 引导 LLM 何时调用
    parameters = {"type": "object", ...}     # JSON Schema 风格参数定义

    def run(self, arguments, context) -> ToolResult: ...
```

**新增工具无需改动主循环**：注册到 `ToolRegistry` 即自动出现在
`registry.to_llm_schemas()` 导出的工具列表中，随请求体一起发给 LLM。
`BaseTool.__init_subclass__` 会在**类定义时**就校验 `name` / `description` 非空，
避免带着缺字段的工具混进注册表。

内置三个工具（题面要求"至少三个"）：

| 工具 | 说明 | 关键实现 |
| --- | --- | --- |
| `calculator` | 数学表达式求值 | 基于标准库 `ast` 自行遍历求值树，**不使用 `eval`**；天然拒绝 `import`、赋值、属性访问；含幂次上限与结果上限防资源耗尽 |
| `search` | mock 检索 | 可替换的 `SearchBackend`（6 篇内置语料）；只认标题/关键词命中，避免无关查询被摘要命中拉进结果；无结果返回空列表而非报错 |
| `todo` | 会话级待办 | 状态直接写入 `Session.state`（键 `<session_id>:todos`），因此**随 session 落盘**，并能被 context 渲染 |

**参数校验**（[`agent/tools/base.py`](agent/tools/base.py) `validate_arguments`）支持
`type` / `required` / `enum` / `minimum` / `maximum` / `minLength` / `maxLength` /
数组 `items` / `default`，并显式排除 Python 中 `bool` 属于 `int` 子类的陷阱。

**错误规范化**：五类问题（未知工具、缺参、类型错误、工具内部异常、非法返回值）
在 `ToolRegistry.execute()` 收敛为 `ToolResult`，**永不抛出**未捕获异常，
错误文本形如 `ERROR [missing_parameter]: …` 并回灌给 LLM 自我修正。

### 2.4 LLM 输出解析

[`agent/parser.py`](agent/parser.py) 采用**三级优先级**，这是回退策略的核心：

1. **API 原生 `tool_calls`** —— 首选，最可靠；
2. **文本中的协议 JSON** —— 裸 JSON / ` ```json ` 代码块 / 从散文里按括号平衡提取
   （支持字段别名与 OpenAI 嵌套风格容错）；
3. **纯文本整体作为答案** —— 兜底，保证永远有结果。

协议（同时注入 system prompt 引导模型遵守）：

```json
{"thought": "内部推理，不展示给用户", "tool_calls": [{"name": "…", "arguments": {...}}]}
{"thought": "内部推理", "answer": "给用户的最终答复"}
```

两处关键取舍：

- **原生 `tool_calls` 场景下，content 按协议认定为思考过程**。模型"边想边调工具"时
  写下的文字是给流程看的推理，不是给用户的答复；只有当出现 `<thinking>` 标签时，
  才把标签外的文字认定为答复（→ `mixed`）。
- **协议 JSON 里既无工具也无有效答案时返回 `EMPTY`**，而不是把 JSON 原文当答复——
  否则 `{"thought": …}` 会直接泄露给用户。

任何畸形输入（截断 JSON、缺字段、代码块内容非法、空输入…）都不抛异常，
问题记入 `warnings` 并进入 trace。`arguments` 解析失败时保留 `raw_arguments`
并标记 `argument_error`，交给主循环按参数错误回灌。

### 2.5 异常处理与可观测性

| 场景 | 处理 | 对应验收项 |
| --- | --- | --- |
| LLM 超时 / 网络错误 / 限流 / 5xx | 归一化为统一错误家族并重试；重试耗尽返回可解释提示 | 6.2 E4 |
| LLM 鉴权失败 / 4xx | 不重试，直接明确报错 | 6.2 E4 |
| 未知工具 / 缺参 / 类型错误 | 结构化错误回灌，LLM 可自我修正 | 6.2 E1/E2 |
| 工具内部异常 | 归一化为 `tool_error`，不冒泡 | 6.2 E3 |
| 工具执行超时 | 守护线程执行 + 放弃等待，返回 `tool_timeout`，不阻塞 loop | 6.2 E8 |
| 解析失败 | 重新提示 → 仍失败则明确收尾 | 6.2 E5 |
| 不存在的 session | `get_or_create` 新建 或 `require` 明确报错（含已有 id） | 6.2 E6 |
| 缺少配置 | 启动即明确提示，退出码 2，无堆栈外泄 | 6.2 E7 |
| 空输入 | 短路，0 次 LLM 调用，返回明确提示 | 6.3 B5 |
| 达到轮次上限 | 返回部分结论 + 超限说明 | 6.3 B1 |

**trace**（[`agent/trace.py`](agent/trace.py)）支持 `text` 与 `jsonl` 两种格式，
一次用户输入共享同一个 `trace_id`，可完整回放。工具调用按题面要求记录**四要素**：
入参、输出、耗时、错误。

真实运行示例（一次"检索 + 记待办"）：

```
[loop_step] step1_receive_input trace=8e331816 session_id=window-1 turn=1 input_chars=40
[loop_step] step2_build_request trace=8e331816 iteration=1 messages=3 tools=3 context_chars=638
[llm_call]  chat.completions trace=8e331816 model=<模型名> attempt=1 duration_ms=19373.16
            finish_reason=tool_calls tool_calls=2
[loop_step] step2_parse_result kind=tool_calls source=native_tool_calls tool_calls=2 warnings=0
[tool_call] search  arguments={...} ok=true duration_ms=1.61 result=...
[tool_call] todo    arguments={...} ok=true duration_ms=1.04 result=...
[loop_step] step2_build_request iteration=2 messages=7 context_chars=843
[llm_call]  finish_reason=stop tool_calls=0
[loop_step] step4_finish iteration=2 answer_chars=41
[loop_step] turn_finished stop_reason=final_answer iterations=2 tool_calls=2
```

### 2.6 Session 与多窗口隔离

- `Session` 保存**完整历史**不做截断——"随时续聊并正确恢复"要求历史完整，
  而 prompt 尺寸由 context 层控制，两者职责分离。
- 消息按 `turn` 编号分组，使压缩可以**整轮**丢弃而不打断
  `assistant.tool_calls ↔ tool` 的配对关系。
- `Session.state` 存放会话级状态（如待办），随 session 一起落盘。
- `SessionStore` 支持内存与落盘两种模式；落盘采用"先写 `.tmp` 再 `os.replace`"，
  避免中途失败留下半个文件。
- 不存在的 session：`get_or_create` 走"新建"，`require` 走"明确报错（列出已有 id）"
  ——两种策略都提供，由调用方选择。

---

## 3. memory 的召回时机与放置方式

题面要求说明"memory 的召回时机与放置方式"。本项目不引入向量库或外部记忆服务
（spec 2.2 明确排除复杂实现），而是把记忆分成三类，各自有明确的**召回时机**与
**放置位置**：

| 记忆类型 | 存放位置 | 召回时机 | 放置位置 | 为什么这样放 |
| --- | --- | --- | --- | --- |
| **会话历史**（用户输入 / 助手答复 / 工具结果） | `Session.messages`（可落盘） | 每次组装 context 时 | 消息列表的**历史区**，按时间顺序 | 对话锚点与推理依据；顺序不能乱 |
| **会话状态**（待办等工具维护的事实） | `Session.state`（可落盘） | 每次组装 context 时 | **置顶的独立 system 块** | 状态是"当前事实"而非历史；混进历史会被压缩丢掉 |
| **思考过程**（`thought`） | 存入 `Session.messages` 的 assistant 消息 | 默认**不召回**；`include_reasoning=True` 时召回 | 包在 `<thinking>` 标签内，紧邻该条答复 | 价值衰减快、体积大，协议不依赖它 |

### 3.1 组装顺序（固定）

```
[system] 角色与工具使用原则（DEFAULT_SYSTEM_PROMPT）
[system] 输出协议说明（OUTPUT_PROTOCOL_INSTRUCTIONS）
[system] 当前会话状态快照   ← 状态置顶，压缩也丢不掉
[user]   …
[assistant] …（含 tool_calls 回放）
[tool]   …（工具结果，超长则截断）
[assistant] 最终答复
```

### 3.2 三个关键设计决定

**(1) 状态为什么置顶而不放进历史？**
状态（如待办清单）是"当前事实"，会被工具改写；历史是"发生过什么"，只增不改。
若把状态混在历史里，压缩历史时会顺带把状态丢掉，导致"切回旧窗口后待办消失"。
置顶成独立 system 块后，即使省略了 30 轮历史，状态依然可见——
这正是 spec 6.3 B6「回到早期话题仍能召回状态」的实现方式。

**(2) 工具结果为什么必须召回，但注入前要截断？**
"带工具的追问"（先查天气，再问"那明天呢"）完全依赖它，所以必须召回。
但单条检索结果可能很长，因此注入前按 `tool_result_max_chars`（默认 2000）截断并
标注"省略 N 字符"，保留头部（工具结论通常在前面）。

**(3) 压缩时为什么整轮丢弃而摘要？**
按**整轮**从最旧开始丢，避免把 `assistant.tool_calls` 与对应 `tool` 消息拆开
（OpenAI 兼容接口要求成对，否则请求被拒）。被丢弃轮次的用户输入折叠成一段摘要保留，
让模型知道"聊过这些话题"。摘要自身受 `max_summary_chars` **硬上限**约束——
摘要超限会抵消压缩的意义（该缺陷曾被回归测试抓到）。

### 3.3 相关配置

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `include_reasoning` | `False` | 是否把思考过程放回 context |
| `tool_result_max_chars` | `2000` | 单条工具结果注入上限 |
| `max_context_chars` | `24000` | context 字符总量上限，超出触发压缩 |
| `keep_recent_turns` | `3` | 压缩时至少保留的最近轮次数 |
| `max_summary_chars` | `600` | 摘要长度硬上限 |
| `include_session_state` | `True` | 是否注入会话状态块 |

---

## 4. 项目结构

```
agent-vibre_coding-test/
├── 1.txt                        # 题面原文（只读，不改动）
├── spec.md                      # 需求规格（含约束与验收标准）
├── task.md                      # 实施任务清单与完成状态
├── README.md                    # 本文件
├── requirements.txt             # 依赖清单（零第三方依赖，附说明）
├── .env.example                 # 环境变量示例（不含真实密钥）
├── .gitignore                   # 排除 .env 与 __pycache__
├── notes/
│   ├── asset-survey.md          # 现有资产探查记录
│   └── ai-usage.md              # AI Prompt 与问题解决记录
├── docs/
│   ├── requirements-check.md    # 对照 spec 约束的逐条自检
│   └── acceptance.md            # 对照 spec 验收清单的逐条核验
├── agent/
│   ├── __init__.py
│   ├── __main__.py              # CLI 入口
│   ├── config.py                # 配置（环境变量校验）
│   ├── llm.py                   # LLM 客户端（超时/重试/错误归一化）
│   ├── parser.py                # 输出协议与解析（含回退）
│   ├── session.py               # Session 模型与持久化
│   ├── context.py               # context 组装、压缩、截断
│   ├── loop.py                  # Agent 主循环
│   ├── trace.py                 # trace 与执行日志
│   └── tools/
│       ├── __init__.py
│       ├── base.py              # 工具契约 + 参数校验
│       ├── registry.py          # 工具注册表
│       ├── calculator.py        # 工具 1
│       ├── search.py            # 工具 2（mock）
│       └── todo.py              # 工具 3（会话级状态）
└── tests/
    ├── helpers.py                            # 测试基础设施（fake LLM / 脚本化 LLM / 记录工具）
    ├── test_config.py  test_llm.py  test_trace.py
    ├── test_parser.py
    ├── test_tool_calculator.py  test_tool_search.py  test_tool_todo.py  test_tool_registry.py
    ├── test_session.py  test_context.py  test_session_context_integration.py
    ├── test_loop.py  test_loop_errors.py
    └── test_e2e_real_llm.py                  # 可选的真实 API 端到端
```

---

## 5. 测试覆盖

`python -m unittest discover -s tests -t . -v` → **378 个用例全绿**（离线，无需密钥）。

| 测试文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `test_config.py` | 14 | 环境变量读取、默认值、缺配置明确报错、密钥打码 |
| `test_llm.py` | 27 | 成功/401/429/5xx/超时/网络错误、重试与耗尽、响应畸形、trace |
| `test_parser.py` | 43 | 三级优先级、字段别名、括号平衡、11 种畸形输入不抛异常 |
| `test_tool_calculator.py` | 28 | 正常计算、非法表达式、除零、注入防护、幂次上限 |
| `test_tool_search.py` | 21 | 结构化结果、排序、无结果、参数越界 |
| `test_tool_todo.py` | 27 | 增删改查、会话隔离、状态持久化 |
| `test_tool_registry.py` | 39 | 注册/列举/取用/重名、Schema 形状、五类错误规范化 |
| `test_session.py` | 35 | 生命周期、落盘重载、损坏文件、隔离、`require` 报错 |
| `test_context.py` | 41 | 组装顺序、状态置顶、两类追问、压缩与摘要硬上限、截断 |
| `test_session_context_integration.py` | 9 | Session × Context × 工具三方联动、双窗口全流程 |
| `test_loop.py` | 27 | 四步骤、直接回复、工具调用、链式调用、多工具、trace 与事件 |
| `test_loop_errors.py` | 38 | LLM/工具失败、轮次上限、无进展、解析重试、空输入、配置与 session 异常 |
| `test_e2e_real_llm.py` | 10 | 真实 API 端到端（默认跳过，需 `LLM_E2E=1`） |

测试可离线确定性运行：LLM 层被替换为可注入的 `FakeTransport`（伪造 HTTP 报文）
或 `ScriptedLLM`（按脚本返回响应），两者都会记录收到的请求，便于断言
"context 里到底放了什么"。

---

## 6. 约束遵守情况

| 约束 | 遵守方式 |
| --- | --- |
| 不依赖现成 Agent 框架承载主流程 | 主循环、工具调度、解析、session、context 全部自行实现；`requirements.txt` 零依赖 |
| 必须接入真实 LLM API | `agent/llm.py` 默认使用标准库 `urllib` 真实请求；已用真实端点实测通过 |
| 至少 3 个工具 + 注册机制 | `calculator` / `search` / `todo`，含 name/description/参数 Schema |
| 不要提供 API Key | 密钥仅经环境变量注入；已全文扫描确认仓库无真实密钥 |
| 提交 GitHub 链接 | <https://github.com/ckj337728-web/agent>；README 覆盖运行方式、系统设计、memory 说明 |
| 附 AI Prompt 与问题解决记录 | 见 [`notes/ai-usage.md`](notes/ai-usage.md) |

逐条自检见 [`docs/requirements-check.md`](docs/requirements-check.md)，
验收清单核验见 [`docs/acceptance.md`](docs/acceptance.md)。
