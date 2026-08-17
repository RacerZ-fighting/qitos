"""Tests for trace, render, and benchmark redaction."""

from __future__ import annotations

import json

from qitos.benchmark.common import write_benchmark_results
from qitos.core.spec import BenchmarkRunResult
from qitos.trace.events import TraceEvent
from qitos.trace.redaction import REDACTED_FIELDS, REDACTED_MARKER, redact_mapping
from qitos.trace.writer import TraceWriter


def test_redacted_fields_includes_sensitive_names():
    """The default field set includes common sensitive names."""
    expected = {
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "access_token",
        "refresh_token",
        "private_key",
        "credentials",
    }
    assert expected.issubset(REDACTED_FIELDS)


def test_redact_mapping_masks_sensitive_fields():
    """Sensitive field values are replaced with the redaction marker."""
    data = {
        "tool_args": {"command": "ls"},
        "api_key": "sk-12345",
        "authorization": "Bearer abc",
        "safe_field": "visible",
    }
    result = redact_mapping(data)
    assert result["tool_args"] == REDACTED_MARKER
    assert result["api_key"] == REDACTED_MARKER
    assert result["authorization"] == REDACTED_MARKER
    assert result["safe_field"] == "visible"


def test_redact_mapping_handles_nested_dicts():
    """Nested mappings are redacted recursively."""
    data = {
        "outer": {
            "password": "secret123",
            "name": "test",
        }
    }
    result = redact_mapping(data)
    assert result["outer"]["password"] == REDACTED_MARKER
    assert result["outer"]["name"] == "test"


def test_redact_mapping_handles_lists_of_dicts():
    """Mappings inside lists are redacted recursively."""
    data = {
        "items": [
            {"token": "abc", "value": 1},
            {"token": "def", "value": 2},
            [{"secret": "nested", "value": 3}],
        ]
    }
    result = redact_mapping(data)
    assert result["items"][0]["token"] == REDACTED_MARKER
    assert result["items"][1]["token"] == REDACTED_MARKER
    assert result["items"][0]["value"] == 1
    assert result["items"][2][0]["secret"] == REDACTED_MARKER


def test_trace_writer_redacts_sensitive_manifest_and_events(tmp_path):
    """TraceWriter should not persist raw secrets from run metadata or events."""
    writer = TraceWriter(
        output_dir=str(tmp_path),
        run_id="redaction-demo",
        metadata={
            "run_spec": {
                "environment": {
                    "api_key": "sk-raw-secret",
                    "nested": {"token": "raw-token"},
                }
            }
        },
        strict_validate=False,
    )
    writer.write_event(
        TraceEvent(
            run_id="redaction-demo",
            step_id=0,
            phase="setup",
            ok=True,
            payload={"authorization": "Bearer raw-secret", "safe": "visible"},
            error=None,
            ts="2026-06-03T00:00:00+00:00",
        )
    )
    writer.finalize(status="failed", summary={})

    manifest_text = (tmp_path / "redaction-demo" / "manifest.json").read_text()
    event_text = (tmp_path / "redaction-demo" / "events.jsonl").read_text()
    manifest = json.loads(manifest_text)
    event = json.loads(event_text)

    assert "sk-raw-secret" not in manifest_text
    assert "raw-token" not in manifest_text
    assert "Bearer raw-secret" not in event_text
    assert manifest["run_spec"]["environment"]["api_key"] == REDACTED_MARKER
    assert manifest["run_spec"]["environment"]["nested"]["token"] == REDACTED_MARKER
    assert event["payload"]["authorization"] == REDACTED_MARKER
    assert event["payload"]["safe"] == "visible"


def test_trace_writer_retains_model_transaction_facts(tmp_path):
    """Trace persistence keeps model identity and usage while redacting secrets."""
    writer = TraceWriter(
        output_dir=str(tmp_path),
        run_id="model-transaction",
        strict_validate=False,
    )
    writer.write_event(
        TraceEvent(
            run_id="model-transaction",
            step_id=0,
            phase="DECIDE",
            payload={
                "stage": "model_output",
                "model_response": {
                    "provider": "demo-provider",
                    "model_name": "demo-model",
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 7},
                    "usage_source": "provider",
                    "metadata": {"api_key": "raw-secret"},
                },
            },
        )
    )
    writer.finalize(status="failed", summary={})

    event = json.loads((tmp_path / "model-transaction" / "events.jsonl").read_text())
    response = event["payload"]["model_response"]
    assert response["provider"] == "demo-provider"
    assert response["model_name"] == "demo-model"
    assert response["finish_reason"] == "stop"
    assert response["usage"] == {"total_tokens": 7}
    assert response["usage_source"] == "provider"
    assert response["metadata"]["api_key"] == REDACTED_MARKER


def test_benchmark_result_writer_redacts_sensitive_metadata(tmp_path):
    """Benchmark jsonl output should not persist raw secrets in row metadata."""
    row = BenchmarkRunResult(
        task_id="task-1",
        benchmark="demo",
        split="test",
        prediction=None,
        success=False,
        stop_reason="failed",
        steps=1,
        latency_seconds=0.1,
        token_usage=0,
        cost=0.0,
        trace_run_dir=None,
        run_spec_ref=None,
        metadata={
            "execution": {
                "run_spec": {
                    "environment": {
                        "api_key": "sk-row-secret",
                        "nested": {"token": "row-token"},
                    }
                }
            }
        },
    )

    path = write_benchmark_results(tmp_path / "rows.jsonl", [row])
    text = path.read_text()
    payload = json.loads(text)

    assert "sk-row-secret" not in text
    assert "row-token" not in text
    assert (
        payload["metadata"]["execution"]["run_spec"]["environment"]["api_key"]
        == REDACTED_MARKER
    )
    assert (
        payload["metadata"]["execution"]["run_spec"]["environment"]["nested"]["token"]
        == REDACTED_MARKER
    )
