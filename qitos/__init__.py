"""
@Copyright (c) 2026, Qitor Research. All rights reserved.
QitOS public API surface.


"""

__version__ = "0.6.0"

from .core.agent_module import AgentModule
from .core.budget import BudgetLedger, BudgetSnapshot
from .core.action import Action
from .core.decision import Decision
from .core.env import Env, EnvSpec
from .core.errors import QitosRuntimeError, StopReason
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
from .core.history import History, HistoryPolicy, HistorySnapshot
from .core.message_builder import ContextSnapshot, ContextSnapshotConflictError
from .core.observation import Observation
from .core.tool_result import ToolResult, ToolResultStatus
from .core.artifact import ArtifactRef, ArtifactStore, ArtifactStoreError
from .core.state import StateSchema
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
from .core.agent_spec import (
    AgentSpec,
    AgentRegistry,
    ContextStrategy,
    HandoffContext,
    StateAdapter,
)
from .engine.engine import Engine, EngineResult, StepSummary
from .engine.events import EngineEvent, EngineEventType, EventStream
from .engine.states import ContextConfig

__all__ = [
    "AgentModule",
    "BudgetLedger",
    "BudgetSnapshot",
    "Engine",
    "EngineEvent",
    "EngineEventType",
    "EventStream",
    "EngineResult",
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
    "StepSummary",
    "ContextConfig",
    "Task",
    "TaskResource",
    "TaskBudget",
    "TaskResult",
    "StateSchema",
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
    "Decision",
    "Action",
    "Memory",
    "ModelResponse",
    "ModelContinuation",
    "ModelRequest",
    "ModelStreamEvent",
    "ModelStreamEventType",
    "ModelTiming",
    "History",
    "HistoryPolicy",
    "HistorySnapshot",
    "ContextSnapshot",
    "ContextSnapshotConflictError",
    "Observation",
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
    "AgentSpec",
    "AgentRegistry",
    "ContextStrategy",
    "HandoffContext",
    "StateAdapter",
    "StopReason",
    "QitosRuntimeError",
]
