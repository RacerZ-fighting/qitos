"""Decision interpretation and parser compatibility outside the model transport."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Generic, List, TypeVar

from ..core.decision import Decision
from ..core.errors import (
    ErrorCategory,
    ParseExecutionError,
    RuntimeErrorInfo,
)
from ..core.model_response import ModelResponse
from ..core.state import StateSchema
from ..core.turn import TurnSnapshot
from ..protocols import get_protocol, resolve_protocol_chain
from .parser import (
    build_parser_diagnostics,
    normalize_parser_diagnostics,
    parser_contract,
    parser_name,
)
from .states import RuntimePhase, StepRecord

if TYPE_CHECKING:
    from ._model_runtime import _ModelRuntime
    from .engine import Engine


StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class _DecisionRuntime(Generic[StateT, ObservationT, ActionT]):
    """Normalize one completed provider response into a canonical Decision."""

    def __init__(
        self,
        engine: Engine[StateT, ObservationT, ActionT],
        model_runtime: _ModelRuntime[StateT, ObservationT, ActionT],
    ) -> None:
        self.engine = engine
        self.model_runtime = model_runtime

    def select_branch(
        self,
        state: StateT,
        observation: ObservationT,
        branch_decision: Decision[ActionT],
    ) -> Decision[ActionT]:
        engine = self.engine
        if engine.search is not None:
            candidates = engine.search.expand(
                state, observation, branch_decision
            ) or list(branch_decision.candidates)
            scores = engine.search.score(state, observation, candidates)
            candidates = engine.search.prune(candidates, scores)
            if not candidates:
                new_state = engine.search.backtrack(state)
                if new_state is not state:
                    state.__dict__.update(new_state.__dict__)
                return Decision.wait(rationale="search backtrack")
            scores = engine.search.score(state, observation, candidates)
            selected = engine.search.select(candidates, scores)
            mark_selected = getattr(engine.search, "mark_selected", None)
            if callable(mark_selected):
                mark_selected(state, selected)
        else:
            selected = engine.branch_selector.select(
                branch_decision.candidates, state, observation
            )
        selected.validate()
        return selected

    def normalize_decision(
        self,
        raw_decision: Any,
        step: int,
        record: StepRecord | None = None,
        turn: TurnSnapshot | None = None,
    ) -> Decision[ActionT]:
        if isinstance(raw_decision, Decision):
            if record is not None and not record.decision_source:
                record.decision_source = "agent"
            return raw_decision

        response = raw_decision if isinstance(raw_decision, ModelResponse) else None
        native_decision = self.model_runtime._decision_from_native_tool_calls(
            response=response,
            step=step,
            record=record,
            llm=turn.model if turn is not None else None,
            protocol=turn.protocol if turn is not None else None,
        )
        if native_decision is not None:
            return native_decision
        parser_input = response.text if response is not None else raw_decision

        # When native tool calling is preferred and the model returned plain
        # text without tool_calls, treat it as a final answer — the model is
        # done acting and is giving its summary/conclusion in natural language.
        # Parsers (especially json_decision_v1) will misinterpret natural
        # language as invalid JSON and return wait(), which causes the agent
        # to loop forever without ever producing a final result.
        is_native_text_response = (
            response is not None
            and self.model_runtime._native_tool_call_preferred(
                turn.model if turn is not None else None,
                turn.protocol if turn is not None else None,
            )
            and not (isinstance(response.tool_calls, list) and response.tool_calls)
            and str(response.text or "").strip()
        )

        if is_native_text_response and response is not None:
            # Still try the parser chain first — if the model happened to
            # produce valid structured output (JSON with a final_answer, or
            # ReAct "Final Answer:" label), let the parser extract it.
            parse_outcome = self._parse_with_protocol_chain(
                parser_input=parser_input,
                step=step,
                record=record,
            )
            if parse_outcome is not None:
                if parse_outcome.mode != "wait":
                    return parse_outcome

                # Some protocol parsers use an unmarked wait as an early-step
                # heuristic for plain text. Preserve the historical final
                # fallback unless parsing actually failed on action-shaped text.
                parser_error = bool(parse_outcome.meta.get("parser_error"))
                if not parser_error and self._looks_like_explicit_wait(parser_input):
                    return parse_outcome
                rejection_reason: str | None = None
                if parser_error:
                    if self._looks_like_structured_action_intent(parser_input):
                        rejection_reason = "structured_action_parse_error"
                    elif self._looks_like_structured_final_intent(parser_input):
                        rejection_reason = "structured_final_parse_error"
                if rejection_reason is not None:
                    self.engine._emit(
                        step,
                        RuntimePhase.DECIDE,
                        payload={
                            "stage": "native_text_final_rejected",
                            "reason": rejection_reason,
                            "parser_diagnostics": parse_outcome.meta.get(
                                "parser_diagnostics"
                            ),
                        },
                    )
                    return parse_outcome

            if record is not None:
                record.decision_source = "native_text_final"
            return Decision.final(
                answer=str(response.text).strip(),
                meta={"decision_source": "native_text_final"},
            )

        parse_outcome = self._parse_with_protocol_chain(
            parser_input=parser_input,
            step=step,
            record=record,
        )
        if parse_outcome is not None:
            return parse_outcome

        raise ValueError(
            "Agent.decide must return Decision when no parser is configured"
        )

    @staticmethod
    def _looks_like_explicit_wait(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False
        return bool(
            re.search(
                r"(?i)(?:[\"']mode[\"']|(?:^|[\{,])\s*mode)\s*[:=]\s*[\"']?wait[\"']?",
                source,
            )
            or re.search(r"(?i)<[^>]+\bmode\s*=\s*[\"']wait[\"']", source)
        )

    @staticmethod
    def _looks_like_structured_action_intent(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False

        if re.search(
            r"(?i)(?:"
            r"<\s*(?:minimax:)?tool_call\b|"
            r"<\s*(?:tool_use|tool_name|invoke)\b|"
            r"<\|tool_calls?_section_begin\|>|"
            r"<\|tool_call_(?:begin|argument_begin)\|>"
            r")",
            source,
        ):
            return True

        if re.search(
            r"(?im)^\s*(?:[-*]\s*)?action(?:s)?(?:\s*:|\s+[A-Za-z_][\w.-]*\s*\()",
            source,
        ):
            return True

        field_pattern = (
            r"actions?|tools?|tool[_-]?calls?|calls?|commands?|name|args|arguments"
        )
        json_carrier_pattern = r"actions?|tools?|tool[_-]?calls?|calls?|commands?"
        if re.search(
            rf"(?i)[\"']({json_carrier_pattern})[\"']\s*:", source
        ) or re.search(rf"(?i)[{{,]\s*({json_carrier_pattern})\s*:", source):
            return True

        structured_fields: set[str] = set()
        for pattern in (
            rf"(?im)^\s*(?:[-*]\s*)?[\"']?({field_pattern})[\"']?\s*(?::|=)",
            rf"(?i)[\"']({field_pattern})[\"']\s*:",
            rf"(?i)(?:^|[{{,])\s*({field_pattern})\s*:",
            rf"(?i)<\s*({field_pattern})\b",
        ):
            structured_fields.update(re.findall(pattern, source))

        normalized_fields = {
            re.sub(r"[_-]+", "", field).lower() for field in structured_fields
        }
        unambiguous_carriers = {
            "action",
            "actions",
            "toolcall",
            "toolcalls",
        }
        if normalized_fields & unambiguous_carriers:
            return True

        argument_fields = {"args", "arguments"}
        if "name" in normalized_fields and normalized_fields & argument_fields:
            return True

        ambiguous_carriers = {
            "tool",
            "tools",
            "call",
            "calls",
            "command",
            "commands",
        }
        return bool(
            normalized_fields & ambiguous_carriers
            and normalized_fields & ({"name"} | argument_fields)
        )

    @staticmethod
    def _looks_like_structured_final_intent(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False
        return bool(
            re.search(
                r"(?i)(?:[\"']mode[\"']|(?:^|[\{,])\s*mode)" r"\s*[:=]\s*[\"']?final\b",
                source,
            )
            or re.search(
                r"(?i)(?:[\"']final[_-]?answer[\"']\s*:|"
                r"(?:^|[\{,])\s*final[_-]?answer\s*:)",
                source,
            )
            or re.search(r"(?i)<\s*(?:final_answer|final)\b", source)
            or re.search(r"(?im)^\s*final\s+answer\s*:", source)
        )

    def _parse_with_protocol_chain(
        self,
        *,
        parser_input: Any,
        step: int,
        record: StepRecord | None,
    ) -> Decision[ActionT] | None:
        parser_attempts: List[Dict[str, Any]] = []
        last_exception: Exception | None = None
        last_diagnostics: Dict[str, Any] | None = None
        candidates = self._candidate_parsers()
        for candidate in candidates:
            parser = candidate["parser"]
            protocol = candidate.get("protocol")
            fallback_used = bool(candidate.get("fallback_used"))
            try:
                decision = parser.parse(
                    parser_input,
                    context={"step": step, "protocol": getattr(protocol, "id", None)},
                )
                normalized = normalize_parser_diagnostics(
                    getattr(decision, "meta", None),
                    parser=parser,
                    raw_output=parser_input,
                    step_id=step,
                )
                if normalized is not None:
                    normalized = dict(normalized)
                    normalized.setdefault("protocol", getattr(protocol, "id", None))
                    normalized.setdefault("selected_parser", parser_name(parser))
                    normalized.setdefault("fallback_used", fallback_used)
                    normalized.setdefault("parser_attempts", list(parser_attempts))
                parser_attempts.append(
                    {
                        "parser": parser_name(parser),
                        "contract": parser_contract(parser),
                        "protocol": getattr(protocol, "id", None),
                        "result": (
                            "success"
                            if normalized is None
                            or normalized.get("severity") != "error"
                            else "error"
                        ),
                        "fallback_used": fallback_used,
                    }
                )
                if (
                    normalized is not None
                    and normalized.get("severity") == "error"
                    and candidate.get("allow_fallback", True)
                ):
                    last_diagnostics = dict(normalized)
                    continue
                self._record_parser_observability(
                    step=step,
                    raw_output=parser_input,
                    decision=decision,
                    record=record,
                    parser=parser,
                    diagnostics=normalized,
                    protocol=protocol,
                    parser_attempts=parser_attempts,
                    fallback_used=fallback_used,
                )
                return decision
            except Exception as exc:
                last_exception = exc
                parser_attempts.append(
                    {
                        "parser": parser_name(parser),
                        "contract": parser_contract(parser),
                        "protocol": getattr(protocol, "id", None),
                        "result": "exception",
                        "fallback_used": fallback_used,
                    }
                )
                last_diagnostics = build_parser_diagnostics(
                    parser=parser,
                    severity="error",
                    code="unexpected_parser_exception",
                    summary="Parser raised an unexpected exception.",
                    raw_output=parser_input,
                    details=str(exc),
                    repair_instruction="The parser failed internally before producing structured repair feedback.",
                    expected_shape="See the configured parser contract for the expected output format.",
                    step_id=step,
                )
                last_diagnostics["protocol"] = getattr(protocol, "id", None)
                last_diagnostics["selected_parser"] = parser_name(parser)
                last_diagnostics["fallback_used"] = fallback_used
                last_diagnostics["parser_attempts"] = list(parser_attempts)
                continue
        if last_diagnostics is not None:
            selected_parser = (
                parser_name(candidates[-1]["parser"])
                if candidates
                else "unknown_parser"
            )
            last_diagnostics.setdefault("selected_parser", selected_parser)
            last_diagnostics.setdefault(
                "fallback_used",
                any(item.get("fallback_used") for item in parser_attempts),
            )
            last_diagnostics.setdefault("parser_attempts", parser_attempts)
            self._record_parser_observability(
                step=step,
                raw_output=parser_input,
                decision=None,
                record=record,
                parser=candidates[-1]["parser"] if candidates else "unknown_parser",
                diagnostics=last_diagnostics,
                protocol=candidates[-1].get("protocol") if candidates else None,
                parser_attempts=parser_attempts,
                fallback_used=any(
                    item.get("fallback_used") for item in parser_attempts
                ),
            )
            if last_exception is not None:
                info = RuntimeErrorInfo(
                    category=ErrorCategory.PARSE,
                    message=str(last_exception),
                    phase="decide",
                    step_id=step,
                    recoverable=True,
                    details={"parser_diagnostics": last_diagnostics},
                )
                raise ParseExecutionError(info) from last_exception
            return Decision.wait(
                rationale=str(last_diagnostics.get("summary") or "Parser error."),
                meta={
                    "parser_error": True,
                    "parser_feedback": str(
                        last_diagnostics.get("repair_instruction")
                        or last_diagnostics.get("summary")
                        or ""
                    ),
                    "parser_diagnostics": last_diagnostics,
                },
            )
        return None

    def _candidate_parsers(self) -> List[Dict[str, Any]]:
        engine = self.engine
        if engine.parser is not None:
            return [
                {
                    "parser": engine.parser,
                    "protocol": get_protocol(engine.protocol),
                    "fallback_used": False,
                    "allow_fallback": False,
                }
            ]
        protocol = engine.resolve_protocol()
        candidates: List[Dict[str, Any]] = []
        agent_parser = getattr(engine.agent, "model_parser", None)
        if agent_parser is not None:
            candidates.append(
                {
                    "parser": agent_parser,
                    "protocol": protocol,
                    "fallback_used": False,
                    "allow_fallback": True,
                }
            )
        for index, item in enumerate(resolve_protocol_chain(protocol)):
            try:
                parser = item.parser_factory()
            except Exception:
                continue
            if agent_parser is not None and parser.__class__ is agent_parser.__class__:
                continue
            candidates.append(
                {
                    "parser": parser,
                    "protocol": item,
                    "fallback_used": bool(agent_parser) or index > 0,
                    "allow_fallback": True,
                }
            )
        return candidates

    def _interpret_model_response(
        self,
        *,
        state: StateT,
        observation: ObservationT,
        response: ModelResponse,
        record: StepRecord,
    ) -> Decision[ActionT] | None:
        interpret = getattr(self.engine.agent, "interpret_model_response", None)
        if not callable(interpret):
            return None
        decision = interpret(state, observation, response)
        if decision is None:
            return None
        if not isinstance(decision, Decision):
            raise ValueError(
                "Agent.interpret_model_response must return Decision or None"
            )
        self.engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "model_response_interpreted",
                "mode": decision.mode,
                "model_response": dict(record.model_response),
            },
        )
        record.decision_source = "agent_interpretation"
        return decision

    def _record_parser_observability(
        self,
        *,
        step: int,
        raw_output: Any,
        decision: Decision[ActionT] | None,
        record: StepRecord | None,
        parser: Any,
        diagnostics: Dict[str, Any] | None = None,
        protocol: Any = None,
        parser_attempts: List[Dict[str, Any]] | None = None,
        fallback_used: bool = False,
    ) -> None:
        engine = self.engine
        contract = parser_contract(parser)
        normalized = diagnostics or normalize_parser_diagnostics(
            getattr(decision, "meta", None),
            parser=parser,
            raw_output=raw_output,
            step_id=step,
        )
        protocol_id = getattr(protocol, "id", None) if protocol is not None else None
        attempts = list(parser_attempts or [])
        if normalized is not None:
            normalized.setdefault("protocol", protocol_id)
            normalized.setdefault("selected_parser", parser_name(parser))
            normalized.setdefault("fallback_used", bool(fallback_used))
            normalized.setdefault("parser_attempts", attempts)
        if (
            decision is not None
            and isinstance(decision.meta, dict)
            and normalized is not None
        ):
            decision.meta["parser_diagnostics"] = normalized
            if normalized.get("severity") == "error":
                decision.meta.setdefault("parser_error", True)
                decision.meta.setdefault(
                    "parser_feedback",
                    normalized.get("repair_instruction")
                    or normalized.get("summary")
                    or "",
                )
            else:
                decision.meta.setdefault(
                    "parser_warning",
                    normalized.get("salvage_summary")
                    or normalized.get("summary")
                    or "",
                )
        parsed_mode = getattr(decision, "mode", None) if decision is not None else None
        result_payload = {
            "stage": "parser_result",
            "parser": parser_name(parser),
            "contract": contract,
            "protocol": protocol_id,
            "selected_parser": parser_name(parser),
            "parsed_mode": parsed_mode,
            "has_diagnostics": normalized is not None,
            "salvage_applied": bool((normalized or {}).get("salvage_applied")),
            "severity": (normalized or {}).get("severity"),
            "fallback_used": bool(fallback_used),
            "parser_attempts": attempts,
        }
        engine._emit(step, RuntimePhase.DECIDE, payload=result_payload)
        if normalized is not None:
            engine._emit(
                step,
                RuntimePhase.DECIDE,
                payload={"stage": "parser_diagnostics", "diagnostics": normalized},
            )
            engine._trace_runtime.record_parser_diagnostics(normalized)
        if record is not None:
            record.protocol_id = protocol_id
            record.parser_selected = parser_name(parser)
            record.parser_fallback_used = bool(fallback_used)
            record.parser_attempts = attempts
            record.parser_contract = contract
            record.parser_diagnostics = dict(normalized or {})
            record.parser_salvage_applied = bool(
                (normalized or {}).get("salvage_applied")
            )
            record.decision_source = "parser"
