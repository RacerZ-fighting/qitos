# QitOS

<img src="assets/logo.png" alt="QitOS Logo" width="75%">

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-qitor.mintlify.app-0A66C2)](https://qitor.mintlify.app/)
[![PyPI](https://img.shields.io/pypi/v/qitos.svg)](https://pypi.org/project/qitos/)
[![Repo](https://img.shields.io/badge/github-Qitor%2Fqitos-black)](https://github.com/Qitor/qitos)

QitOS 是面向 agent 研究者的 torch-flavor 框架。

你可以在同一个 `AgentModule + Engine` 内核上原型化方法、运行 benchmark，并用内建 `qita` 检查长时轨迹。

QitOS 主仓库是小而清晰的核心框架。产品级 / 展示级应用会进入独立的 `qitos-zoo`，包括计划中的 `qitos-coder` 与 `qitos-cyber-agent`。

[快速开始](https://qitor.mintlify.app/zh/quickstart) · [教程课程](https://qitor.mintlify.app/zh/tutorials) · [基准测试](https://qitor.mintlify.app/zh/benchmarks/overview) · [CLI 参考](https://qitor.mintlify.app/zh/reference/cli) · [更新日志](CHANGELOG.md) · [English README](README.md)

## 最新进展

- **只保留一条 canonical 可观测路径**：重复的进程级 tracing provider 及其 W&B/MLflow
  processors 已删除。运行时事件、`TraceWriter`、Session Journal 和 `qita` 现在组成一条
  明确的 trace/replay 路径；Handoff 仍通过 canonical `HANDOFF_START`/`HANDOFF_END`
  事件完整可见。
- **完整 Run 与交互式 Run 共用官方 MCP 生命周期**：独立 server 会在固定并发与 timeout
  下连接并发现目录，再按 factory 顺序确定性发布成功目录。交互式 session 由
  `Engine.astep()` 懒启动并持续持有同一生命周期；同步 `step()` 会明确拒绝 MCP
  session。Stdio 与 Streamable HTTP 现在直接使用官方 Python SDK 和协议类型；每个
  SDK context 由一个专用 task 持有到关闭，QitOS 继续负责有界目录发布和稳定
  ToolResult 错误。PTY 改由 `ptyprocess` 建立，YAML 配置入口使用严格的 Pydantic
  边界校验。来自 incomplete 或 failed 模型终态的 ToolCall 只保留诊断，不会执行。
- **从 terminal fact 恢复完成通知**：后台 Child 与受管进程的完成输入现在都是 canonical Journal
  terminal 的确定性投影。Resume 无需第二份存储即可补上 terminal 到 Mailbox 之间的崩溃窗口，也不会重复投递前台 Child ToolResult；
  已消费 event id 保持幂等，Fork 继承的事实也不会变成新输入。canonical `ToolResult` 同时保存
  模型 `call_id`，Child factory 可以安全完成异步资源构造。
- **无 writer lease 的 Run 查询与 lineage**：`JsonlRunCatalog` 现在可以在 Engine
  仍持有 writer lease 时返回不可变、类型化的 Run 摘要、确定性列表、已校验祖先和直接
  Child。读取不会修复 canonical JSONL，也不会重建可丢弃的 SQLite 投影。继承到本地的
  committed boundary 可继续 fork；嵌套 fork 的 Engine 恢复不依赖祖先文件，也不会重放
  已完成工具。稳定 lineage id 会跨 fork 保留；terminal handle 还会给出最后一个非终态
  continuation boundary，让显式 follow-up fork 成为 resumable Run，而不是继承完成态。
- **带 revision 的原子文件工具**：有界读取现在会返回完整 UTF-8 文件的 SHA-256
  revision。Host 与 Docker 文件能力通过原子替换提交内容，并在每个 environment 内按
  相同路径串行化 mutation；`write_file` 支持 compare-and-swap，`edit_file` 默认校验
  它刚读到的 revision，因此并发修改会明确报冲突，不会静默覆盖。
- **后台进程完成后安全唤醒 Agent**：Host watcher 会先收完输出并持久化
  `process.terminal`，再通过既有 durable Mailbox 提交唯一的 `process.completed` 输入。
  活跃 Agent 只在下一 turn safe point 读取该事件；Mailbox 已关闭或拒绝时，terminal
  snapshot 仍可由进程控制工具查询。
- **带判别类型的模型流事件**：每个 Provider 事件现在都会明确声明它是文本、reasoning、
  ToolCall 增量、原生输出项、usage、生命周期、成功终态还是失败终态。含混的
  `ModelStreamChunk` 字段集合已经删除；Engine 把 `FAILED` 视为错误，只提交
  `COMPLETED` 事务。
- **可恢复模型请求与受校验的 continuation**：Engine 现在只向 Provider 传递一份
  不可变 `ModelRequest`，并把精确、已脱敏的请求快照写入 Journal。Responses 只有在
  Run、Provider、model、protocol、请求设置和 canonical input 前缀全部一致时才使用
  `previous_response_id`；resume 可以保留这项优化，fork、Provider 切换、压缩漂移或
  句柄过期都会回退到完整本地 transcript。恢复回归现在会让多轮 Run 依次经历压缩、
  取消、从 committed boundary fork 和 resume，并验证 canonical ToolCall/ToolResult
  transcript 始终完整。若所有 Tool terminal 已落盘、但进程在 step commit 前退出，
  恢复出的 step 也会保留与之匹配的 continuation。
- **同一个异步 turn 事务**：完整 Run 与交互式 step 现在共用一份不可变 turn
  事务。Parser、Critic 与 handoff 作为组合策略接入，不再各自占据或复制 Agent loop；
  Tool、Mailbox、MCP 与 Child 始终运行在调用方 event loop，取消会先等待已启动 handler
  清理并按输入顺序写完 terminal 结果。模型变化只会让下一 turn 重新解析推断协议。
- **Root/Child 共用一份 usage 总账**：产品运行时可把同一个 `BudgetLedger` 传给所有
  后代 Engine。每个完成的模型事务只向 Root JSONL 结算一次 token 与 cost；Child 预算
  只会进一步收紧共享余额，不会复制一份全局额度。结果同时保留共享总量、本地用量和
  完整性。首版在结算后阻止下一 turn，不为已经并发发出的请求预留 token。
- **Session Journal 只有一个 owner**：每个 Run 现在使用进程安全的 JSONL writer lease
  和明确的终止生命周期。replay 始终先验证 canonical JSONL；可丢弃的 SQLite 读取投影只有
  在与 JSONL 一致时才会保留，并会在过期或损坏后重建。payload 在 append 前先通过唯一的
  严格 JSON 边界；不支持的 schema 会给出升级错误；fork 失败也不会泄漏未返回的 child writer。
- **每 turn 冻结工具 exposure**：模型请求和动作分发器现在共用一份带 revision 的
  只读 `ToolExposure`。应用可以按 group 或策略筛选；registry 的后续变化只影响下一
  turn；Journal 会记录精确名称与 Schema 摘要，便于 replay 审计。
- **工具输入只认同一份严格 Schema**：展示给模型的 JSON Schema 现在也会在每次
  handler 调用前执行。自动生成的根对象默认拒绝未知字段；字段缺失、类型错误、枚举
  或边界不合法时会生成可持久化的未执行终态，不再进入自定义工具代码。
- **工具调用完整后才执行**：兼容 Chat 与 Anthropic Messages 只有在各自协议终态
  证明调用完整后才发布原生 ToolCall。因输出上限截断、block 未闭合或参数畸形的调用
  只保留诊断，不会进入工具 handler，也不会在 Provider replay 中复活。
- **完整的 Responses 流生命周期**：Responses adapter 现在保留 lifecycle/item
  事件、refusal、仅出现在终态的文本/reasoning，以及相互独立的交错函数参数。
  incomplete 或互相矛盾的终态不会变成可执行工具调用。
- **来源明确的 token 统计**：Engine context telemetry 现在区分 Provider 实测、
  本地估算与缺失 usage。Provider 返回的零值不会被本地估算覆盖；cache read/write
  与 reasoning 子项保持可见，同时不会被重复计入累计总量。
- **真实的模型能力快照**：模型 adapter 现在暴露不可变的 `ModelCapabilities`。
  Responses、Anthropic Messages 与兼容 Chat 只声明已经通过测试的原生工具、
  reasoning/replay、usage/cache 与多模态能力，不提前宣称 hosted tool；Responses
  还会声明已通过契约测试的受校验 continuation。终态 token 用量会收敛为类型化
  `ModelUsage`，同时保留 Provider
  原始明细。
- **模型家族与 wire 可独立选择**：同一 Kimi 家族配置可使用兼容 Chat Completions
  或 Anthropic Messages，且不会把一种 adapter 的请求默认值泄漏到另一种 wire。
  本地 trace 会保留模型身份、终态、类型化 usage 及其来源，同时继续遮蔽嵌套凭据。
- **上下文压缩不再错调异步模型**：同步 `CompactHistory.retrieve()` 遇到 QitOS
  异步 `Model` 时会使用有界 heuristic summary 并记录实际模式，不再把异步模型当
  普通函数调用，也不引入 sync/async event-loop 桥。
- **模型 delta 真实实时可见**：文本、reasoning 与工具调用分片到达后立即发布。
  只有首个 provider event 之前的失败可以重试，因此不会重复展示输出或重复工具调用；
  terminal chunk 仍会等到 provider EOF 后才提交，保留成功事务边界。
- **Anthropic 原生预设与 reasoning**：Anthropic 家族预设现在直接构造官方
  Messages adapter。Claude 4.5 的 reasoning 强度会映射为受输出上限约束的手动
  thinking budget，请求默认值会真实进入 wire payload，thinking 请求也不会再发送
  不兼容的 temperature 覆盖；预设默认使用原生 API tool schema。
- **Run-scoped MCP 工具**：向 Engine 传入为每个 Run 创建全新 transport 的
  `mcp_server_factory` 后，Engine 会完成连接、发现、暴露 `mcp__server__tool` 名称、在调用方 event
  loop 上执行调用，并在 run 结束时注销工具和关闭连接。目录变化只会在下一轮安全点
  原子发布；annotations、分页、类型化远端错误、取消和上一版有效目录都会保留；
  HTTP JSON/SSE、隔离的 GET 重连游标、取消安全的 cleanup 和 session 过期恢复都不会重放
  可能有副作用的 Tool 调用；默认空配置没有启动成本。
- **渐进式 bundled Skill**：应用可以给 `SkillToolSet` 配置只读资源根目录，先暴露
  有界目录，再按精确名称加载完整 `SKILL.md`。递归发现遵循显式根目录优先级，并返回
  强类型诊断；同一个 bundle revision 同时覆盖说明与资源。应用还可以传入冻结的运行时
  requirement 集合，在完整说明暴露给模型前拒绝当前环境不可用的工作流；整个过程无需
  调用 provider 或写安装 registry。
- **强类型 WorkPlan**：`WorkPlanState` 与 `update_plan` 提供经过校验、可随
  checkpoint 恢复的轻量清单，并带有纯 reducer 和确定性 Markdown 投影。coding preset
  不再直接修改自由格式的 todo metadata。
- **Terminal resume trace 正确收尾**：恢复已经终止的 checkpoint 不会执行模型或工具，
  新配置的空 trace manifest 也会立即进入终态。
- **唯一的 Shell 准入边界**：`run_command` 通过常规工具准入后直接执行，不再在
  handler 内重复做第二次权限判断。
- **长工具输出可继续读取**：配置 `FileArtifactStore` 后，Engine 会先保存完整的超长
  结果，再生成有界模型预览。Reducer 与 trace 仍能读取 canonical
  `ToolResult.output`，checkpoint 保存模型实际看到的 replacement，Agent 可通过现有
  `read_file` 按 workspace 相对路径继续分页读取。
- **可恢复的异步 checkpoint**：Engine 会在安全边界等待唯一的 `CheckpointStore`，
  包括第一次 Provider 请求前保存的初始化输入快照，并保存原始任务、完整模型历史前缀、
  状态与 fork 链路。恢复 terminal checkpoint 时直接返回该状态，不再调用模型或工具。SQLite
  操作不会阻塞 event loop，取消传播前会先等待数据库调用稳定结束；旧 JSON manager、
  会丢数据的 durability 线程、空壳 pending-write 层和 trace 桥接均已删除。
  `TraceWriter` 会先提交完整 step 的事件范围，再写入 step 标记；取消等生命周期事件
  仍会立即可见。内存与 SQLite store 使用相同的 JSON 边界。
- **模型运行时只保留一条原生异步路径**：`Engine.arun()` 与 `Engine.astep()` 统一负责
  从模型请求到终态响应的完整过程。OpenAI Responses、Anthropic Messages、兼容 Chat
  Completions、Gemini、LiteLLM 与 Ollama 都实现同一个异步流契约；旧的同步/异步类层级、
  `call_raw`、导入时注册和 daemon-thread `AsyncEngine` 桥接已经删除。
- **qita 同规格对比预检**：compare 页面会先核对模型、提示词、工具、环境、
  上下文策略、预算、源码版本与实验来源。配置不同或来源信息不完整的结果会明确标为
  只能描述、不能用于因果判断；配置一致也不会掩盖供应商或外部环境的非确定性。
- **真实且类型化的模型流**：Chat、Responses 与 Anthropic 流会保留供应商终止原因、
  reasoning 与工具调用分片、完整工具调用及 usage。不完整的流会明确失败，不再伪造
  完成；发生错误后 Engine handler 也不会再收到正常 `on_end`。
- **按调用准确统计 qita 工具状态**：工具次数与失败数现在来自 canonical action/result
  配对，不会再把同一步中的一个失败错误归到所有调用上。精确生命周期计数以及无法配对
  的 trace 证据都会保留供审计。
- **类式工具只保留一个执行契约**：类式工具现在只暴露
  `execute(args, runtime_context)`。`ToolRegistry` 只做精确 canonical 名称查找，
  校验、权限、超时/重试、调用和结果归一化统一由 `ActionExecutor` 负责。旧的
  `run`/`call` 适配、注册表直接执行、注册表自动名称别名、duck-typed 回退与隐式
  并发白名单已删除；只有显式 `concurrency_safe=True` 才允许并行。
- **工具生命周期结果不再失真**：统一的 `ToolResult` 投影会在执行记录、Observation、
  history、trace、摘要与成功率中保留 success、partial、running、error、拒绝/跳过、
  输入/审批、超时和取消。未知状态与旧别名都会 fail-closed；业务结果使用独立字段，
  不再挤占执行状态。
- **单一原生工具调用通道**：当模型 preset 优先使用服务商原生工具时，类型化调用会
  跳过文本解释器和 parser，API 请求不再附加重复的框架动作契约；每个获准或被拒绝
  的调用都只按原始 call id 与顺序提交一次结果。参数损坏的调用也会返回配对错误，
  不会执行工具。
- **OpenAI-compatible 实时流与有界重试**：Engine 由 QitOS 统一持有模型传输
  重试预算，并关闭 SDK 内部重试。连接和首事件前失败可在默认 300 秒恢复窗口内
  重试；首个 provider event 之后保持实时发布，失败时直接停止而不重放。事件空闲
  超时能识别卡住的流，又不会截断持续有输出的长响应。
- **qita 轨迹分析工作台**：run 页面现在默认进入失败诊断视图，用 Focus Navigator、Agent Behavior Story 和右侧 Inspector 引导用户先看关键证据。每步按照 `Input -> Thought -> Action Calls -> Environment Observation` 展示；每个 action 都和自己的完整参数、状态、耗时及 model-visible result 成对出现，canonical raw 与无法配对的证据仍可在 Inspector 审计。异常调用默认展开、成功调用默认折叠，长正文自动折行且绝不会只有截断预览。CyberGym 的预算耗尽和 `submit_poc` 验证失败会被提升为重点复盘信号；Light/Dark 主题覆盖 board、run、replay 与 compare 页面。
- **稳定的 same-spec 比较指纹**：纯文本任务使用内容生成的稳定 ID，qita 分别记录完整任务与运行配置指纹；挂钟时间不再造成假差异，不同任务或缺少 provenance 的旧 trace 也不会被误报为 same-spec。
- **立即取消状态保持一致**：Engine 识别立即取消后，State、任务/运行结果、END event 与 trace manifest 现在都会记录 `cancelled_immediate`；qita 会将该 manifest 视为 `stopped`，不再误判为正常完成。
- **结构化动作文本不再假完成**：当原生工具模型没有返回 `tool_calls`、却以文本输出了格式错误的动作字段时，QitOS 现在会保留 parser 恢复路径，而不会把动作文本当成最终答案；普通自然语言结论的行为保持不变。
- **窗口安全的原生工具历史**：当消息窗口裁掉 assistant 调用声明时，模型请求会移除对应的孤立工具结果，避免长时并行工具 Agent 发送非法 `tool_call_id` 链，同时保持完整轮次和原有恢复行为不变。
- **直接构造 Engine 时保留 preset 协议**：`Engine(agent=...)` 现在会采用 `build_model_for_preset(...)` 写入模型的协议，使 Kimi K3 等服务商别名继续使用 JSON/原生 API 工具交付，而不会静默回退到文本 ReAct。
- **空模型响应有界恢复**：既无有效文本也无工具调用的模型响应现在会被记录为可追踪的 `model_error`，重试一次后若仍为空则明确停止，不再伪装成 parser `wait` 并耗尽 Agent 步数预算。
- **可选 OpenAI Responses API 传输**：通过 `api_mode="responses"`（或 YAML `api_mode: responses`）保留类型化输出项、并行函数调用、`call_id` 工具结果、流式事件和可重放工具上下文。现有 Chat Completions 行为仍是默认值。

## v0.5.0 最新进展

- **12 个方法模板**：ReAct、PlanAct、SWE-Agent、Voyager、Debate、Manager-Worker、Planner-Executor、Self-Refine、Reflexion、LATS、MoA 和 Magentic-One — 每个都包含 paper.md、config.yaml 和 recipe 实现。
- **`qit new` CLI**：使用 `qit new --template <name>` 从内建模板脚手架新 agent 项目。
- **导出 API**：`EngineConfig`、`ToolPermissionSpec`、`CriticTrace` 和 `HandoffTrace` 用于程序化访问引擎配置和 trace 数据。
- **FamilyPreset 可扩展性**：`override()`、`recommended_*` 建议字段、`MaxTokensCriteria` 停止条件。
- **qita 成本面板**：运行概览中的 token 用量和成本指标。

详见 [CHANGELOG.md](CHANGELOG.md)。

## Live Terminal of QitOS for Code Review

<p align="center">
  <img src="demo.gif" alt="QitOS long-running agent demo" width="92%">
</p>

## QitOS 适合谁

- **方法研究者**：频繁改 prompt、parser、critic、tool 与 memory policy，但不想每次都重写 runtime。
- **benchmark 使用者**：希望 GAIA、Tau-Bench、CyBench 跑在和 agent 开发同一套内核上。
- **长时 agent 调试者**：更关心 trajectory review、replay、diff 与 context collapse，而不是先拼应用脚手架。

## 2 分钟跑通 QitOS

QitOS 里的 minimal agent 应该是一个最轻量的 **coding agent**。它会配置真实模型、进入 workspace、改代码、跑验证，并留下 qita 可检查的 trace。

```bash
pip install "qitos[models]"
export OPENAI_API_KEY="sk-..."
qit --version
qit demo minimal
qita board --logdir runs
```

OpenAI-compatible provider 常见补充配置：

```bash
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1/"
export QITOS_MODEL="Qwen/Qwen3-8B"
```

`qit demo minimal` 会先种一个最小 bug workspace，再让模型驱动的 coding agent 修复它、运行验证，并把轨迹写到 `./runs`。

接下来可以继续：

- 想看 ReAct：见 [`examples/patterns/react.py`](examples/patterns/react.py)
- 想看 coding agent：见 [`examples/real/coding_agent.py`](examples/real/coding_agent.py)
- 想看 benchmark：从 [评测总览](https://qitor.mintlify.app/zh/benchmarks/overview) 开始
- 想看方法模板：见 [方法模板指南](https://qitor.mintlify.app/zh/guides/method-templates)

## 为什么是 QitOS

| 如果你想要... | QitOS 提供... |
|---|---|
| 可复现的 agent 研究 | 稳定的 `AgentModule + Engine` 内核 |
| 方法 = Agent + Critic | 12 个内建方法模板，映射经典论文 |
| 强可观测性 | `qita` board、replay、export 与 trace 工件 |
| benchmark 工作流 | GAIA、Tau-Bench、CyBench 适配器 |
| 更少框架胶水 | 一条 canonical 执行主线 |

## 方法模板

QitOS 内置 12 个方法模板 — 每个都是实现经典 agentic 推理模式的 Agent + Critic 组合：

| 模板 | 模式 | 论文 |
|------|------|------|
| ReAct | 推理 + 行动 | Yao et al. 2023 |
| PlanAct | 先规划再执行 | — |
| SWE-Agent | 软件工程 | Princeton 2024 |
| Voyager | 开放探索 | Wang et al. 2023 |
| Debate | 多 Agent 辩论 | — |
| Manager-Worker | 编排与委派 | — |
| Planner-Executor | 计划分解 | — |
| Self-Refine | 生成 → 批评 → 改进 | Madaan et al. 2023 |
| Reflexion | 行动 → 反思 → 重试 | Shinn et al. 2023 |
| LATS | 蒙特卡洛树搜索 | Zhou et al. 2023 |
| MoA | 并行提议 + 聚合 | Wang et al. 2024 |
| Magentic-One | 编排器 + 专家 | Furtado et al. 2024 |

直接使用：

```python
from qitos.recipes.reflexion import ReflexionAgent, ReflexionCritic

agent = ReflexionAgent(llm=my_llm)
result = agent.run(
    task="Debug the failing test",
    critics=[ReflexionCritic(max_reflections=3)],
    max_steps=15,
    return_state=True,
)
```

或从任意模板脚手架新 agent：

```bash
pip install qitos[cookiecutter]
qit new --agent-name my_agent --agent-description "My custom agent"
qit list-templates
```

## 工具层布局

QitOS 将工具导入分为三层：

- `qitos.kit`：最简单的常用工具集入口
- `qitos.kit.toolset`：场景导向的预设和注册表构建器
- `qitos.kit.tool.<domain>`：高级原子能力导入

默认组合是列表优先：

```python
from qitos import ToolRegistry
from qitos.kit.tool.file import ReadFile
from qitos.kit.toolset import coding_tools

registry = ToolRegistry().include_toolset(
    [
        ReadFile(workspace_root="."),
        coding_tools(workspace_root="."),
    ]
)
```

安全敏感工具为显式 opt-in 导入，不在 `qitos`、`qitos.kit`、`qit demo` 或快速开始路径中。

## 文档地图

- 第一次接触： [简介](https://qitor.mintlify.app/zh/introduction)
- 第一条成功路径： [快速开始](https://qitor.mintlify.app/zh/quickstart)
- 安装方式： [安装](https://qitor.mintlify.app/zh/installation)
- 写自己的最小 coding agent： [构建第一个 Agent](https://qitor.mintlify.app/zh/guides/build-your-first-agent)
- 方法模板： [方法模板指南](https://qitor.mintlify.app/zh/guides/method-templates)
- 理解运行时： [AgentModule](https://qitor.mintlify.app/zh/concepts/agent-module) / [Engine](https://qitor.mintlify.app/zh/concepts/engine)
- 看 trace： [可观测性](https://qitor.mintlify.app/zh/guides/observability)
- 走完整课程： [教程](https://qitor.mintlify.app/zh/tutorials)
- 看 benchmark： [评测总览](https://qitor.mintlify.app/zh/benchmarks/overview)
- 看命令： [CLI 参考](https://qitor.mintlify.app/zh/reference/cli)
- 看 API： [API 参考](https://qitor.mintlify.app/zh/reference/api)

## 界面预览

<table>
  <tr>
    <td align="center"><strong>QitOS CLI</strong></td>
    <td align="center"><strong>qita Board</strong></td>
    <td align="center"><strong>qita Trajectory View</strong></td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/qitos_cli_snapshot.png">
        <img src="assets/qitos_cli_snapshot.png" alt="QitOS CLI" width="100%" />
      </a>
    </td>
    <td align="center">
      <a href="assets/qita_board_snapshot.png">
        <img src="assets/qita_board_snapshot.png" alt="qita Board" width="100%" />
      </a>
    </td>
    <td align="center">
      <a href="assets/qita_traj_snapshot.png">
        <img src="assets/qita_traj_snapshot.png" alt="qita Trajectory View" width="100%" />
      </a>
    </td>
  </tr>
</table>

## 当前阶段

QitOS 当前处于 **Beta**。

- 相对稳定：`AgentModule + Engine`、trace/qita、canonical examples、benchmark adapters，以及官方可复现 run 契约。
- 仍会演进：更高层 convenience API、部分 `kit` 模块、实验性 toolset。
- 如果你正在评估接入，建议从 kernel 与 examples 开始，而不是假设所有高层表面都已冻结。
- 持续演进和升级说明见 [CHANGELOG.md](CHANGELOG.md)。

## 安装与版本

- 支持的 Python 版本：**3.10+**
- 普通用户安装：`pip install "qitos[models]"`
- 版本检查：`qit --version`
- 最小 coding agent：`qit demo minimal`
- 常见 provider 配置：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`QITOS_MODEL`
- 仅核心安装：`pip install qitos`
- 仓库源码安装：`pip install -r requirements.txt`
- 完整开发安装：`pip install -r requirements-dev.txt`
- 可选扩展：`qitos[models]`、`qitos[benchmarks]`、`qitos[cookiecutter]`、`qitos[all]`
- 安装说明： [安装](https://qitor.mintlify.app/zh/installation)

## 参与贡献

欢迎贡献方法模板、benchmark adapters、memory/history 工作流、qita UX 与核心框架能力。产品级 agent 应优先进入 `qitos-zoo`。开发环境、方法模板贡献、文档贡献流程详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT，见 [LICENSE](LICENSE)。
