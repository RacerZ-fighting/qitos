"""
@Copyright (c) 2026, Qitor Research. All rights reserved.
QitOS public API surface.


"""

__version__ = "0.6.0"

from .core.budget import (
    BudgetLedger,
    BudgetSnapshot,
    StepBudgetExhaustedError,
    StepPurpose,
)
from .core.env import Env, EnvSpec
from .core.memory import Memory
from .core.model_response import ModelResponse, ModelTiming
from .core.model_request import ModelContinuation, ModelRequest
from .core.model_stream import ModelStreamEvent, ModelStreamEventType
from .core.thinking import ThinkingLevel, clamp_thinking_level
from .core.runtime_input import RuntimeInput
from .core.subagent import (
    DEFAULT_SUBAGENT_MAX_STEPS,
    AgentConclusion,
    SubagentHandle,
    SubagentInvocationCancelled,
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentPersistenceError,
    SubagentRunLimitError,
    SubagentResult,
    SubagentRuntimeContext,
    SubagentStatus,
)
from .core.process import (
    ProcessError,
    ProcessHandle,
    ProcessNotFoundError,
    ProcessOutput,
    ProcessPersistenceError,
    ProcessSnapshot,
    ProcessStatus,
    ProcessTerminalNotifier,
)
from .core.tool_result import ToolResult, ToolResultStatus
from .core.artifact import ArtifactRef, ArtifactStore, ArtifactStoreError
from .core.plan import (
    Plan,
    PlanContractError,
    PlanNode,
    PlanStatus,
    PlanUpdate,
    parse_plan_update,
    plan_from_dict,
    plan_to_dict,
    reduce_plan,
    render_plan_markdown,
    validate_plan_transition,
)
from .core.spec import BenchmarkRunResult, ExperimentSpec, RunSpec
from .core.task import (
    Task,
    TaskBlocker,
    TaskBudget,
    TaskLifecycle,
    TaskReference,
    TaskStatus,
)
from .core.tool import (
    BaseTool,
    ToolPermissionContext,
    ToolPermissionDecision,
    ToolPermissionRule,
    ToolValidationResult,
    tool,
)
from .core.tool_registry import ToolExposure, ToolRegistry
from .core.agent import Agent, AgentBusyError, AgentRunRejected, QueueMode
from .core.agent_events import AgentEvent
from .core.agent_loop import (
    AgentContext,
    AgentLoopConfig,
    AgentLoopResult,
    AgentRunStatus,
    RunFinalizationDiagnostic,
    RunFinalizationDiagnosticCode,
    RunFinalizer,
    TurnTransactionBoundary,
    agent_loop,
    agent_loop_continue,
)
from .core.cancellation import CancelMode, CancelSignalView, CancelToken
from .core.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .core.tool_executor import (
    ToolBatchExecutor,
    ToolExecutionConfig,
    ToolTransactionBoundary,
)

__all__ = [
    "BudgetLedger",
    "BudgetSnapshot",
    "StepBudgetExhaustedError",
    "StepPurpose",
    "RuntimeInput",
    "DEFAULT_SUBAGENT_MAX_STEPS",
    "AgentConclusion",
    "SubagentHandle",
    "SubagentInvocationCancelled",
    "SubagentLaunchContext",
    "SubagentLaunchRequest",
    "SubagentPersistenceError",
    "SubagentRunLimitError",
    "SubagentResult",
    "SubagentRuntimeContext",
    "SubagentStatus",
    "ProcessError",
    "ProcessHandle",
    "ProcessNotFoundError",
    "ProcessOutput",
    "ProcessPersistenceError",
    "ProcessSnapshot",
    "ProcessStatus",
    "ProcessTerminalNotifier",
    "Task",
    "TaskBlocker",
    "TaskBudget",
    "TaskLifecycle",
    "TaskReference",
    "TaskStatus",
    "Plan",
    "PlanContractError",
    "PlanNode",
    "PlanStatus",
    "PlanUpdate",
    "parse_plan_update",
    "plan_from_dict",
    "plan_to_dict",
    "reduce_plan",
    "render_plan_markdown",
    "validate_plan_transition",
    "Memory",
    "ModelResponse",
    "ModelContinuation",
    "ModelRequest",
    "ModelStreamEvent",
    "ModelStreamEventType",
    "ModelTiming",
    "ThinkingLevel",
    "clamp_thinking_level",
    "ToolResult",
    "ToolResultStatus",
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactStoreError",
    "RunSpec",
    "ExperimentSpec",
    "BenchmarkRunResult",
    "Env",
    "EnvSpec",
    "BaseTool",
    "tool",
    "ToolPermissionContext",
    "ToolPermissionDecision",
    "ToolPermissionRule",
    "ToolValidationResult",
    "ToolRegistry",
    "ToolExposure",
    "Agent",
    "AgentBusyError",
    "AgentRunRejected",
    "QueueMode",
    "AgentEvent",
    "AgentContext",
    "AgentLoopConfig",
    "AgentLoopResult",
    "AgentRunStatus",
    "RunFinalizationDiagnostic",
    "RunFinalizationDiagnosticCode",
    "RunFinalizer",
    "TurnTransactionBoundary",
    "agent_loop",
    "agent_loop_continue",
    "CancelMode",
    "CancelSignalView",
    "CancelToken",
    "AssistantMessage",
    "Message",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "ToolBatchExecutor",
    "ToolExecutionConfig",
    "ToolTransactionBoundary",
]
