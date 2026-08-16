"""Stable engine exports."""

from .critic_decorator import critic
from ..core.cancellation import CancelMode, CancelToken
from .engine import Engine, EngineResult, StepSummary
from .events import EngineEvent, EngineEventType, EventStream
from .hooks import EngineHook, HookContext, ToolHookContext
from ..core.runtime_input import RuntimeInput
from ._loop_detector import ToolCallLoopDetector
from .states import (
    ContextConfig,
    ContextTelemetry,
    CriticTrace,
    EngineConfig,
    HandoffTrace,
    RuntimeBudget,
    RuntimeEvent,
    RuntimePhase,
    StepRecord,
)

__all__ = [
    "CancelMode",
    "CancelToken",
    "critic",
    "CriticTrace",
    "Engine",
    "EngineConfig",
    "EngineHook",
    "EngineResult",
    "EngineEvent",
    "EngineEventType",
    "EventStream",
    "HandoffTrace",
    "HookContext",
    "ToolHookContext",
    "ToolCallLoopDetector",
    "StepSummary",
    "ContextConfig",
    "ContextTelemetry",
    "RuntimeBudget",
    "RuntimeInput",
    "RuntimeEvent",
    "RuntimePhase",
    "StepRecord",
]
