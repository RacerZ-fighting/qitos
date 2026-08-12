"""Private action execution helpers for Engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Generic, List, TypeVar, cast

from ..core.action import Action
from ..core.artifact import ArtifactRef, ArtifactStoreError
from ..core.decision import Decision
from ..core.tool_result import ToolResult
from .states import RuntimePhase, StepRecord

if TYPE_CHECKING:
    from .engine import Engine


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class _ActionRuntime(Generic[StateT, ActionT]):
    def __init__(self, engine: Engine[Any, Any, Any]):
        self.engine = engine

    def run_act(
        self, state: StateT, decision: Decision[ActionT], record: StepRecord
    ) -> List[Any]:
        engine = self.engine
        engine._dispatch_hook(
            "on_before_act",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.ACT,
                state=state,
                decision=decision,
                record=record,
            ),
        )
        engine._emit(record.step_id, RuntimePhase.ACT, payload={"stage": "start"})

        if decision.mode != "act":
            engine._emit(
                record.step_id,
                RuntimePhase.ACT,
                payload={"stage": "skipped", "reason": "decision_not_act"},
            )
            return []
        if engine.executor is None:
            raise RuntimeError("No tool registry configured for action execution")

        actions: List[Action] = []
        for action in decision.actions:
            if isinstance(action, Action):
                # Check for handoff tool interception
                handoff = engine._intercept_handoff_action(action)
                if handoff is not None:
                    return handoff
                actions.append(action)
                continue
            payload = (
                action if isinstance(action, dict) else cast(Dict[str, Any], action)
            )
            normalized = Action.from_dict(payload)
            # Check for handoff tool interception
            handoff = engine._intercept_handoff_action(normalized)
            if handoff is not None:
                return handoff
            actions.append(normalized)
        # Pre-flight checks: collect blocked/loop-blocked actions, execute the rest
        blocked_indices: set[int] = set()
        blocked_results: List[tuple[int, ToolResult]] = []
        blocked_invocations: List[tuple[int, Dict[str, Any]]] = []
        for i, normalized_action in enumerate(actions):
            engine._memory_append("action", normalized_action, record.step_id)
            block_reason = self._action_block_reason(state, normalized_action)
            if block_reason:
                blocked_result = ToolResult(
                    status="denied",
                    output={
                        "message": block_reason,
                        "tool_name": normalized_action.name,
                    },
                    error="action_blocked",
                    metadata={
                        "tool_name": normalized_action.name,
                        "error_category": "action_blocked",
                    },
                )
                blocked_indices.add(i)
                blocked_results.append((i, blocked_result))
                blocked_invocations.append(
                    (
                        i,
                        {
                            "tool_name": normalized_action.name,
                            "action_id": normalized_action.action_id,
                            "args": dict(normalized_action.args or {}),
                            "toolset_name": None,
                            "toolset_version": None,
                            "source": "agent_action_gate",
                            "attempts": 0,
                            "latency_ms": 0,
                            "status": "denied",
                            "error_category": "action_blocked",
                            "error": "action_blocked",
                        },
                    )
                )
                if not self._history_tool_calls_enabled(record):
                    # When a custom MessageBuilder is active, avoid injecting
                    # synthetic user messages for blocked actions.  The
                    # block_reason is already carried in the ToolResult.
                    _has_custom_builder = (
                        getattr(getattr(engine, "agent", None), "message_builder", None)
                        is not None
                    )
                    if not _has_custom_builder:
                        engine._history_append(
                            "user",
                            block_reason,
                            record.step_id,
                            metadata={
                                "source": "action_gate",
                                "tool_name": normalized_action.name,
                            },
                        )
                engine._emit(
                    record.step_id,
                    RuntimePhase.ACT,
                    payload={
                        "stage": "action_blocked",
                        "tool_name": normalized_action.name,
                        "reason": block_reason,
                        "action_results": [
                            blocked_result.to_model_dict()
                        ],
                    },
                )
                continue
            loop_detector = engine._tool_loop_detector
            if loop_detector is not None:
                loop_result = loop_detector.check_detailed(
                    normalized_action.name, normalized_action.args
                )
                if loop_result.level == "block":
                    loop_tool_result = ToolResult(
                        status="denied",
                        output={
                            "message": loop_result.message,
                            "tool_name": normalized_action.name,
                        },
                        error="tool_call_loop_detected",
                        metadata={
                            "tool_name": normalized_action.name,
                            "reason": loop_result.message,
                        },
                    )
                    blocked_indices.add(i)
                    blocked_results.append((i, loop_tool_result))
                    blocked_invocations.append(
                        (
                            i,
                            {
                                "tool_name": normalized_action.name,
                                "action_id": normalized_action.action_id,
                                "args": dict(normalized_action.args or {}),
                                "toolset_name": None,
                                "toolset_version": None,
                                "source": "loop_detector",
                                "attempts": 0,
                                "latency_ms": 0,
                                "status": "denied",
                                "error_category": "tool_call_loop_detected",
                                "error": "tool_call_loop_detected",
                            },
                        )
                    )
                    engine._emit(
                        record.step_id,
                        RuntimePhase.ACT,
                        payload={
                            "stage": "tool_call_loop_detected",
                            "tool_name": normalized_action.name,
                            "recovery_message": loop_result.message,
                        },
                    )
                    continue
                if loop_result.level == "warn":
                    # Do not inject a synthetic user turn between a tool call and
                    # its result. The event remains visible to tracing/TUI only.
                    engine._emit(
                        record.step_id,
                        RuntimePhase.ACT,
                        payload={
                            "stage": "tool_call_loop_warning",
                            "tool_name": normalized_action.name,
                            "recovery_message": loop_result.message,
                        },
                    )

        # If all actions were blocked, return immediately
        if len(blocked_indices) == len(actions):
            blocked_only_results = [
                result
                for _, result in sorted(blocked_results, key=lambda pair: pair[0])
            ]
            blocked_only_invocations = [
                invocation
                for _, invocation in sorted(
                    blocked_invocations,
                    key=lambda pair: pair[0],
                )
            ]
            self._finalize_model_outputs(
                actions,
                blocked_only_results,
                step_id=record.step_id,
            )
            record.action_results = blocked_only_results
            record.tool_invocations = blocked_only_invocations
            for blocked_item in blocked_only_results:
                engine._memory_append("action_result", blocked_item, record.step_id)
            self._commit_tool_result_history(actions, blocked_only_results, record)
            engine._dispatch_hook(
                "on_after_act",
                engine._hook_context(
                    step_id=record.step_id,
                    phase=RuntimePhase.ACT,
                    state=state,
                    decision=decision,
                    action_results=[r.to_model_dict() for r in blocked_only_results],
                    record=record,
                ),
            )
            return [r.to_dict() for r in blocked_only_results]

        # Execute non-blocked actions
        executable_actions = [
            a for i, a in enumerate(actions) if i not in blocked_indices
        ]
        executable_indices = [
            i for i in range(len(actions)) if i not in blocked_indices
        ]
        execution = engine.executor.execute(
            executable_actions, env=engine.env, state=state
        )
        exec_stats = dict(getattr(engine.executor, "last_execution_stats", {}) or {})
        # Build tool_invocations from execution results (executable only)
        exec_invocations = [
            {
                "tool_name": item.name,
                "action_id": (
                    executable_actions[index].action_id
                    if index < len(executable_actions)
                    else ""
                ),
                "args": (
                    dict(executable_actions[index].args or {})
                    if index < len(executable_actions)
                    else {}
                ),
                "toolset_name": item.metadata.get("toolset_name"),
                "toolset_version": item.metadata.get("toolset_version"),
                "source": item.metadata.get("source"),
                "attempts": item.attempts,
                "latency_ms": item.latency_ms,
                "status": item.status.value,
                "error_category": item.metadata.get("error_category"),
                "error": item.error,
                # Issue #35: observable action lifecycle — ordering, terminal
                # state and cancellation source.
                "segment_index": item.metadata.get("segment_index"),
                "started": item.metadata.get("started", True),
                "started_at": item.metadata.get("started_at"),
                "ended_at": item.metadata.get("ended_at"),
                "cancel_source": item.metadata.get("cancel_source"),
                "timeout_s": item.metadata.get("timeout_s"),
                "timeout_source": item.metadata.get("timeout_source"),
            }
            for index, item in enumerate(execution)
        ]
        if exec_stats:
            record.action_execution = exec_stats
        results: List[ToolResult] = []
        for item in execution:
            result_metadata = {
                "tool_name": item.name,
                "latency_ms": item.latency_ms,
                "attempts": item.attempts,
            }
            results.append(
                ToolResult(
                    status=item.status.value,
                    output=item.output,
                    error=item.error,
                    metadata=result_metadata,
                    artifacts=item.artifacts,
                    model_output=item.model_output,
                )
            )

        # Merge blocked results and execution results back into original action order
        if blocked_indices:
            # Map execution result indices to original action indices
            exec_result_by_orig_idx: Dict[int, ToolResult] = {}
            for exec_i, orig_i in enumerate(executable_indices):
                if exec_i < len(results):
                    exec_result_by_orig_idx[orig_i] = results[exec_i]
            blocked_result_by_orig_idx: Dict[int, ToolResult] = {
                idx: r for idx, r in blocked_results
            }
            blocked_inv_by_orig_idx: Dict[int, Dict[str, Any]] = {
                idx: inv for idx, inv in blocked_invocations
            }
            exec_inv_by_orig_idx: Dict[int, Dict[str, Any]] = {}
            for exec_i, orig_i in enumerate(executable_indices):
                if exec_i < len(exec_invocations):
                    exec_inv_by_orig_idx[orig_i] = exec_invocations[exec_i]

            merged_results: List[ToolResult] = []
            merged_invocations: List[Dict[str, Any]] = []
            for i in range(len(actions)):
                if i in blocked_indices:
                    merged_results.append(
                        blocked_result_by_orig_idx.get(
                            i,
                            ToolResult(
                                status="error", output=None, error="action_blocked"
                            ),
                        )
                    )
                    merged_invocations.append(blocked_inv_by_orig_idx.get(i, {}))
                else:
                    merged_results.append(
                        exec_result_by_orig_idx.get(
                            i,
                            ToolResult(
                                status="error", output=None, error="execution_failed"
                            ),
                        )
                    )
                    merged_invocations.append(exec_inv_by_orig_idx.get(i, {}))
            results = merged_results
            record.tool_invocations = merged_invocations
        else:
            record.tool_invocations = exec_invocations

        self._finalize_model_outputs(actions, results, step_id=record.step_id)

        # Optional agent-owned pre-history commit for model-visible state
        # receipts.  This is intentionally generic: an agent may canonicalize
        # a state-tool result before history/TUI serialization while the
        # normal reduce pass remains responsible for trace projection.  It is
        # executed once in original tool-call order.
        commit_results = getattr(
            getattr(engine, "agent", None), "commit_action_results", None
        )
        if callable(commit_results):
            commit_results(state, actions, results, step_id=record.step_id)

        if engine.env is not None:
            env_result = engine._run_env_step(
                decision=decision,
                action_results=[item.to_dict() for item in results],
            )
            if env_result is not None:
                results.append(
                    ToolResult(
                        status="success",
                        output={"env": engine._env_step_result_to_dict(env_result)},
                        metadata={"source": "env"},
                    )
                )
        record.action_results = results
        for result_item in results:
            engine._memory_append("action_result", result_item, record.step_id)
        if engine._tool_loop_detector is not None:
            for normalized_action in executable_actions:
                engine._tool_loop_detector.record(
                    normalized_action.name, dict(normalized_action.args or {})
                )

        self._commit_tool_result_history(actions, results, record)
        engine._emit(
            record.step_id,
            RuntimePhase.ACT,
            payload={
                "stage": "action_results",
                "tool_invocations": record.tool_invocations,
                "action_results": [
                    item.to_model_dict() for item in results
                ],
            },
        )
        engine._dispatch_hook(
            "on_after_act",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.ACT,
                state=state,
                decision=decision,
                action_results=[item.to_model_dict() for item in results],
                record=record,
            ),
        )
        return [item.to_dict() for item in results]

    def _commit_tool_result_history(
        self,
        actions: List[Action],
        results: List[ToolResult],
        record: StepRecord,
    ) -> None:
        """Commit one ordered terminal result for every model tool call."""

        if not self._history_tool_calls_enabled(record):
            return
        engine = self.engine
        for idx, result in enumerate(results):
            payload = result.output
            if isinstance(payload, dict) and set(payload.keys()) == {"env"}:
                continue
            tool_name = actions[idx].name if idx < len(actions) else ""
            tool_call_id = actions[idx].action_id if idx < len(actions) else None
            if not tool_call_id:
                tool_call_id = f"call_{record.step_id}_{idx}"
            model_payload = result.model_visible_output
            serialized = self._serialize_for_tool_message(
                model_payload,
                result.error,
                result.status,
            )
            engine._history_append(
                "tool",
                serialized,
                record.step_id,
                metadata={"source": "engine", "tool_name": tool_name},
                tool_call_id=tool_call_id,
                name=(tool_name or None),
            )

    def _serialize_for_tool_message(
        self,
        output: Any,
        error: str | None,
        status: str,
    ) -> str:
        if isinstance(output, str):
            card = output.strip()
            if card:
                if status not in {"success", "error"}:
                    return f"[TOOL:{status}]\n\n{card}"
                return card
            if error not in (None, ""):
                return f"[TOOL:error]\n\nError: {error}"
            return ""

        payload = (
            output
            if status == "success" and error in (None, "")
            else {"status": status, "error": error, "output": output}
        )
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError, OverflowError):
            return str(payload)

    @staticmethod
    def _history_tool_calls_enabled(record: Any) -> bool:
        return bool(
            getattr(record, "native_tool_call_used", False)
            or getattr(record, "history_tool_calls_pending", False)
        )

    def _action_block_reason(self, state: StateT, action: Action) -> str:
        blocker = getattr(self.engine.agent, "block_action", None)
        if blocker is None:
            return ""
        try:
            reason = blocker(state, action)
        except TypeError:
            reason = blocker(action)
        except Exception:
            return ""
        return str(reason or "").strip()

    def _finalize_model_outputs(
        self,
        actions: List[Action],
        results: List[ToolResult],
        *,
        step_id: int,
    ) -> None:
        config = self.engine.context_config
        max_chars = int(getattr(config, "tool_result_max_chars", 0) or 0)
        if max_chars <= 0:
            return
        per_message_max = int(
            getattr(config, "tool_result_per_message_max_chars", 0) or 0
        )
        message_chars = 0
        for index, result in enumerate(results):
            if isinstance(result.output, dict) and set(result.output) == {"env"}:
                continue
            serialized = self._serialize_for_tool_message(
                result.model_visible_output,
                result.error,
                result.status,
            )
            effective_max = max_chars
            if (
                per_message_max > 0
                and message_chars + len(serialized) > per_message_max
            ):
                effective_max = min(
                    max_chars,
                    max(256, per_message_max - message_chars),
                )
            if len(serialized) > effective_max:
                action = actions[index] if index < len(actions) else None
                artifact = self._persist_tool_output(
                    result,
                    tool_name=(action.name if action is not None else "tool"),
                    tool_call_id=(
                        action.action_id
                        if action is not None and action.action_id
                        else f"call_{step_id}_{index}"
                    ),
                    step_id=step_id,
                )
                if artifact is not None:
                    result.artifacts = (*result.artifacts, artifact)
                result.model_output = self._render_bounded_output(
                    serialized,
                    effective_max,
                    artifact,
                )
                serialized = self._serialize_for_tool_message(
                    result.model_output,
                    result.error,
                    result.status,
                )
            message_chars += len(serialized)

    def _persist_tool_output(
        self,
        result: ToolResult,
        *,
        tool_name: str,
        tool_call_id: str,
        step_id: int,
    ) -> ArtifactRef | None:
        store = self.engine.artifact_store
        if store is None:
            return None
        media_type = (
            "text/markdown" if isinstance(result.output, str) else "application/json"
        )
        artifact_id = ":".join(
            (
                self.engine.active_run_id or "run",
                str(step_id),
                tool_name,
                tool_call_id,
            )
        )
        try:
            return store.write_text(
                artifact_id=artifact_id,
                content=result.text,
                media_type=media_type,
            )
        except ArtifactStoreError as exc:
            self.engine._emit(
                step_id,
                RuntimePhase.ACT,
                payload={
                    "stage": "artifact_persist_failed",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "error": str(exc),
                },
            )
            return None

    @staticmethod
    def _render_bounded_output(
        content: str,
        max_chars: int,
        artifact: ArtifactRef | None,
    ) -> str:
        if artifact is None:
            header = (
                f"Tool output exceeded the model budget ({len(content)} characters). "
                "The full output could not be persisted."
            )
        else:
            header = (
                f"Tool output exceeded the model budget ({len(content)} characters).\n"
                f"Full output: {artifact.path}\n"
                f"Artifact bytes: {artifact.size_bytes}"
            )
        prefix = f"{header}\n\nPreview:\n"
        available = max(0, max_chars - len(prefix))
        if available == 0:
            return header
        if len(content) <= available:
            return prefix + content
        marker = "\n…"
        preview = content[: max(0, available - len(marker))].rstrip()
        return prefix + preview + marker
