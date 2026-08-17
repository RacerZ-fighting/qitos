"""
@Copyright (c) 2026, Qitor Research. All rights reserved.
QitOS public API surface.


"""

__version__ = "0.6.0"

from .core.budget import BudgetLedger, BudgetSnapshot
from .core.env import Env, EnvSpec
from .core.memory import Memory
from .core.model_response import ModelResponse, ModelTiming
from .core.model_request import ModelContinuation, ModelRequest
from .core.model_stream import ModelStreamEvent, ModelStreamEventType
from .core.runtime_input import RuntimeInput
from .core.child import (
    DEFAULT_CHILD_MAX_STEPS,
    AgentConclusion,
    ChildHandle,
    ChildInvocationCancelled,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPersistenceError,
    ChildRunLimitError,
    ChildResult,
    ChildRuntimeContext,
    ChildStatus,
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
from .core.history import HistorySnapshot
from .core.tool_result import ToolResult, ToolResultStatus
from .core.artifact import ArtifactRef, ArtifactStore, ArtifactStoreError
from .core.work_plan import (
    WorkPlanContractError,
    WorkPlanItem,
    WorkPlanState,
    WorkPlanStatus,
    WorkPlanUpdate,
    parse_work_plan_update,
    reduce_work_plan,
    render_work_plan_markdown,
    work_plan_state_from_dict,
    work_plan_state_to_dict,
)
from .core.spec import BenchmarkRunResult, ExperimentSpec, RunSpec
from .core.task import (
    Task,
    TaskBudget,
    TaskResource,
    TaskResult,
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
    TurnTransactionBoundary,
    agent_loop,
    agent_loop_continue,
)
from .core.cancellation import CancelMode, CancelToken
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
    "RuntimeInput",
    "DEFAULT_CHILD_MAX_STEPS",
    "AgentConclusion",
    "ChildHandle",
    "ChildInvocationCancelled",
    "ChildLaunchContext",
    "ChildLaunchRequest",
    "ChildPersistenceError",
    "ChildRunLimitError",
    "ChildResult",
    "ChildRuntimeContext",
    "ChildStatus",
    "ProcessError",
    "ProcessHandle",
    "ProcessNotFoundError",
    "ProcessOutput",
    "ProcessPersistenceError",
    "ProcessSnapshot",
    "ProcessStatus",
    "ProcessTerminalNotifier",
    "Task",
    "TaskResource",
    "TaskBudget",
    "TaskResult",
    "WorkPlanContractError",
    "WorkPlanItem",
    "WorkPlanState",
    "WorkPlanStatus",
    "WorkPlanUpdate",
    "parse_work_plan_update",
    "reduce_work_plan",
    "render_work_plan_markdown",
    "work_plan_state_from_dict",
    "work_plan_state_to_dict",
    "Memory",
    "ModelResponse",
    "ModelContinuation",
    "ModelRequest",
    "ModelStreamEvent",
    "ModelStreamEventType",
    "ModelTiming",
    "HistorySnapshot",
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
    "TurnTransactionBoundary",
    "agent_loop",
    "agent_loop_continue",
    "CancelMode",
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
