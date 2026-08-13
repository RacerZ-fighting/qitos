from __future__ import annotations

import pytest

from qitos.core.state_delta import (
    StateDeltaError,
    apply_state_delta,
    build_state_delta,
    state_digest,
)


def test_state_delta_updates_nested_sequences_without_replacing_parent() -> None:
    before = {
        "plan": [{"step": "inspect", "status": "pending"}],
        "findings": [{"id": "f-1", "evidence": ["e-1"]}],
    }
    after = {
        "plan": [{"step": "inspect", "status": "completed"}],
        "findings": [
            {"id": "f-1", "evidence": ["e-1", "e-2"]},
            {"id": "f-2", "evidence": ["e-3"]},
        ],
    }

    delta = build_state_delta(before, after)

    assert apply_state_delta(before, delta) == after
    assert all(operation["path"] not in {"/plan", "/findings"} for operation in delta)
    assert state_digest(apply_state_delta(before, delta)) == state_digest(after)


def test_state_delta_escapes_json_pointer_and_removes_values() -> None:
    before = {"a/b": {"~key": [1, 2, 3]}, "removed": True}
    after = {"a/b": {"~key": [1, 4]}}

    delta = build_state_delta(before, after)

    assert apply_state_delta(before, delta) == after
    assert any(operation["path"].startswith("/a~1b/~0key") for operation in delta)


def test_state_delta_rejects_invalid_paths() -> None:
    with pytest.raises(StateDeltaError):
        apply_state_delta({}, [{"op": "replace", "path": "/missing", "value": 1}])
