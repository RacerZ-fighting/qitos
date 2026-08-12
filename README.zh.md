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

- **长工具输出可继续读取**：配置 `FileArtifactStore` 后，Engine 会先保存完整的超长
  结果，再生成有界模型预览。Reducer 与 trace 仍能读取 canonical
  `ToolResult.output`，checkpoint 保存模型实际看到的 replacement，Agent 可通过现有
  `read_file` 按 workspace 相对路径继续分页读取。
- **可恢复的异步 checkpoint**：Engine 会在安全边界等待唯一的 `CheckpointStore`，
  包括第一次 Provider 请求前保存的初始化输入快照，并保存原始任务、完整模型历史前缀、
  状态与 fork 链路。SQLite
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
- **真实且类型化的模型流**：Chat、Responses 与 Anthropic 流现在通过同一个
  `ModelStreamChunk` 契约保留供应商终止原因、reasoning 与工具调用分片、完整工具调用
  及 usage。不完整的流会明确失败，不再伪造完成；发生错误后 Engine handler 也不会
  再收到正常 `on_end`。
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
- **OpenAI-compatible 事务式流重试**：Engine 由 QitOS 统一持有模型传输重试
  预算，并关闭 SDK 内部重试。可重试的中途断流会丢弃本轮部分文本与工具调用后
  在默认 300 秒恢复窗口内继续重试；事件空闲超时能识别卡住的流，又不会截断持续有输出的长响应。
- **qita 轨迹分析工作台**：run 页面现在默认进入失败诊断视图，用 Focus Navigator、Agent Behavior Story 和右侧 Inspector 引导用户先看关键证据。每步按照 `Input -> Thought -> Action Calls -> Environment Observation` 展示；每个 action 都和自己的完整参数、状态、耗时及 model-visible result 成对出现，canonical raw 与无法配对的证据仍可在 Inspector 审计。异常调用默认展开、成功调用默认折叠，长正文自动折行且绝不会只有截断预览。CyberGym 的预算耗尽和 `submit_poc` 验证失败会被提升为重点复盘信号；Light/Dark 主题覆盖 board、run、replay 与 compare 页面。
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
- **Tracing 集成**：W&B (`WandbTraceProcessor`) 和 MLflow (`MlflowTraceProcessor`) 实验追踪。
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
- 可选扩展：`qitos[wandb]`、`qitos[mlflow]`、`qitos[cookiecutter]`、`qitos[all]`
- 安装说明： [安装](https://qitor.mintlify.app/zh/installation)

## 参与贡献

欢迎贡献方法模板、benchmark adapters、memory/history 工作流、qita UX 与核心框架能力。产品级 agent 应优先进入 `qitos-zoo`。开发环境、方法模板贡献、文档贡献流程详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT，见 [LICENSE](LICENSE)。
