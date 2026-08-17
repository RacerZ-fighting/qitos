from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qitos.core import (
    JournalCorruptionError,
    JournalPosition,
    JournalRecord,
    JournalRecordType,
    RunHandle,
    RunNotFoundError,
    RunStatus,
)
from qitos.kit.journal import JsonlRunCatalog, JsonlSessionJournal


def _commit_payload(turn: int = 0) -> dict[str, object]:
    return {
        "turn": turn,
        "transcript_record_ids": [],
        "tool_terminal_record_ids": [],
    }


def _inherit(
    local_run_id: str,
    local_seq: int,
    record: JournalRecord,
) -> JournalRecord:
    return JournalRecord.create(
        seq=local_seq,
        record_id=f"{local_run_id}:inherited:{record.record_id}",
        type=JournalRecordType.INHERITED,
        run_id=local_run_id,
        payload={
            "origin_run_id": record.run_id,
            "origin_seq": record.seq,
            "origin_record_id": record.record_id,
            "record": record.to_dict(),
        },
    )


def _write_records(root: Path, run_id: str, records: list[JournalRecord]) -> None:
    run_directory = root / run_id
    run_directory.mkdir()
    (run_directory / "journal.jsonl").write_text(
        "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_catalog_inspects_active_writer_without_taking_ownership(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("active", {"agent": "worker"})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["active:turn:0:transcript:0"]},
        record_id="active:input",
    )
    committed = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="active:turn:0:committed",
    )

    handle = await JsonlRunCatalog(tmp_path).inspect_run("active")

    assert handle.status is RunStatus.RESUMABLE
    assert handle.can_resume is True
    assert handle.can_fork is True
    assert handle.can_continue is True
    assert handle.lineage_id == "active"
    assert handle.committed_position == committed
    assert handle.continuation_position == committed
    assert handle.agent_name == "worker"
    # The task text arrives with the goal-bearing Task (S3); until then the
    # handle carries no task.
    assert handle.task == ""
    await journal.append(
        JournalRecordType.RUN_INTERRUPTED,
        {"status": "aborted", "error": "cancelled"},
        record_id="active:run:terminal",
    )
    await journal.close()


@pytest.mark.asyncio
async def test_catalog_read_never_repairs_or_rebuilds_artifacts(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run", {})
    path = journal.path
    index_path = journal.index_path
    await journal.close()
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"seq":2')
    index_path.write_bytes(b"unreadable projection")
    journal_before = path.read_bytes()
    index_before = index_path.read_bytes()

    handle = await JsonlRunCatalog(tmp_path).inspect_run("run")

    assert handle.record_count == 1
    assert path.read_bytes() == journal_before
    assert index_path.read_bytes() == index_before


@pytest.mark.asyncio
async def test_catalog_reads_canonical_jsonl_when_projection_looks_current(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("indexed", {"agent": "worker"})
    await journal.close()
    source_stat = journal.path.stat()
    index_before = journal.index_path.read_bytes()
    canonical = journal.path.read_bytes()
    changed = canonical.replace(b'"worker"', b'"review"')
    assert len(changed) == len(canonical)
    journal.path.write_bytes(changed)
    os.utime(
        journal.path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
    )

    handle = await JsonlRunCatalog(tmp_path).inspect_run("indexed")

    assert handle.agent_name == "review"
    assert journal.path.read_bytes() == changed
    assert journal.index_path.read_bytes() == index_before


@pytest.mark.asyncio
async def test_catalog_rejects_non_final_corruption_and_missing_runs(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("broken", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["broken:turn:0:transcript:0"]},
        record_id="broken:input",
    )
    await journal.close()
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    lines[0] = "not-json"
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    catalog = JsonlRunCatalog(tmp_path)
    with pytest.raises(JournalCorruptionError, match="line 1"):
        await catalog.inspect_run("broken")
    with pytest.raises(RunNotFoundError):
        await catalog.inspect_run("missing")


@pytest.mark.asyncio
async def test_catalog_rejects_invalid_declared_lineage_id(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("broken-lineage", {"lineage_id": " "})
    await journal.close()

    with pytest.raises(JournalCorruptionError, match="lineage_id is invalid"):
        await JsonlRunCatalog(tmp_path).inspect_run("broken-lineage")


@pytest.mark.asyncio
async def test_catalog_lists_and_projects_terminal_lifecycle(tmp_path: Path) -> None:
    first = JsonlSessionJournal(tmp_path)
    await first.create("first", {"agent": "agent-a"})
    await first.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["first:turn:0:transcript:0"]},
        record_id="first:input",
    )
    await first.append(
        JournalRecordType.RUN_INTERRUPTED,
        {"status": "aborted", "error": "cancelled"},
        record_id="first:run:terminal",
    )
    await first.close()

    second = JsonlSessionJournal(tmp_path)
    await second.create("second", {"agent": "agent-b"})
    await second.append(
        JournalRecordType.RUN_COMPLETED,
        {"status": "completed", "error": None},
        record_id="second:run:terminal",
    )
    await second.close()

    handles = await JsonlRunCatalog(tmp_path).list_runs()
    by_id = {handle.run_id: handle for handle in handles}

    assert {handle.run_id for handle in handles} == {"first", "second"}
    assert list(handles) == sorted(
        handles,
        key=lambda handle: (handle.updated_at, handle.run_id),
        reverse=True,
    )
    assert by_id["first"].interruption_reason == "cancelled"
    assert by_id["first"].is_terminal is False
    assert by_id["second"].is_terminal is True
    assert by_id["second"].stop_reason == "completed"


@pytest.mark.asyncio
async def test_catalog_validates_lineage_children_and_nested_forks(
    tmp_path: Path,
) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {"agent": "worker"})
    await parent.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["parent:turn:0:transcript:0"]},
        record_id="parent:input",
    )
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="parent:turn:0:committed",
    )
    child = await parent.fork(boundary, "child")
    child_handle = await JsonlRunCatalog(tmp_path).inspect_run("child")
    assert child_handle.committed_position is not None
    assert child_handle.lineage_id == "parent"
    assert child_handle.parent_run_id == "parent"

    grandchild = await child.fork(child_handle.committed_position, "grandchild")
    await grandchild.close()
    await child.close()
    await parent.close()

    catalog = JsonlRunCatalog(tmp_path)
    lineage = await catalog.lineage("grandchild")
    children = await catalog.children("parent")

    assert [handle.run_id for handle in lineage] == [
        "parent",
        "child",
        "grandchild",
    ]
    assert {handle.lineage_id for handle in lineage} == {"parent"}
    assert [handle.run_id for handle in children] == ["child"]
    assert lineage[-1].task == ""
    assert lineage[-1].agent_name == "worker"

    (tmp_path / "parent" / "journal.jsonl").unlink()
    independent = JsonlSessionJournal(tmp_path)
    await independent.open("grandchild")
    effective = [
        record
        for record in await independent.replay()
        if record.type is JournalRecordType.INHERITED
    ]
    assert effective
    await independent.close()


@pytest.mark.asyncio
async def test_catalog_exposes_declared_lineage_and_terminal_fork_boundary(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create(
        "parent",
        {"agent": "worker", "lineage_id": "session-stable"},
    )
    initial = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="parent:turn:0:committed",
    )
    terminal_commit = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(turn=1),
        record_id="parent:turn:1:committed",
    )
    await journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"status": "completed", "error": None},
        record_id="parent:run:terminal",
    )

    parent = await JsonlRunCatalog(tmp_path).inspect_run("parent")

    assert parent.lineage_id == "session-stable"
    assert parent.committed_position == terminal_commit
    assert parent.continuation_position == terminal_commit
    assert parent.continuation_position != initial
    assert parent.can_resume is False
    assert parent.can_continue is True
    assert parent.has_terminal_state is True
    assert parent.stop_reason == "completed"

    child = await journal.fork(parent.continuation_position, "child")
    await child.close()
    await journal.close()
    child_handle = await JsonlRunCatalog(tmp_path).inspect_run("child")

    assert child_handle.lineage_id == parent.lineage_id
    assert child_handle.parent_run_id == parent.run_id
    assert child_handle.status is RunStatus.RESUMABLE
    assert child_handle.continuation_position is not None


@pytest.mark.asyncio
async def test_catalog_continues_a_forked_journal_at_its_fork_boundary(
    tmp_path: Path,
) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {"agent": "worker"})
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="parent:turn:0:committed",
    )
    child = await parent.fork(boundary, "child")
    await child.close()
    await parent.close()

    handle = await JsonlRunCatalog(tmp_path).inspect_run("child")

    assert handle.status is RunStatus.RESUMABLE
    assert handle.can_resume is True
    assert handle.committed_position is not None
    assert handle.continuation_position is not None
    # Without an own commit, the continuation point is the fork boundary:
    # the last inherited record of the embedded prefix.
    assert handle.continuation_position.record_id.endswith(
        "parent:turn:0:committed"
    )
    assert handle.has_terminal_state is False


@pytest.mark.asyncio
async def test_catalog_fails_closed_on_old_payload_shapes(tmp_path: Path) -> None:
    # Retired commit shapes are rejected at the shared index boundary as soon
    # as they are appended...
    legacy_commit = JsonlSessionJournal(tmp_path / "legacy-commit")
    await legacy_commit.create("run", {})
    with pytest.raises(JournalCorruptionError):
        await legacy_commit.append(
            JournalRecordType.STEP_COMMITTED,
            {"step_id": 0, "terminal_record_ids": []},
            record_id="run:turn:0:committed",
        )
    await legacy_commit.close()

    # ...and a legacy journal found on disk fails closed at catalog read.
    legacy_dir = tmp_path / "legacy-on-disk" / "run"
    legacy_dir.mkdir(parents=True)
    start = JournalRecord.create(
        seq=1,
        record_id="run:start",
        type=JournalRecordType.RUN_STARTED,
        run_id="run",
        payload={},
    )
    legacy_step = JournalRecord.create(
        seq=2,
        record_id="run:turn:0:committed",
        type=JournalRecordType.STEP_COMMITTED,
        run_id="run",
        payload={},
    ).to_dict()
    legacy_step["payload"] = {"step_id": 0, "terminal_record_ids": []}
    (legacy_dir / "journal.jsonl").write_text(
        json.dumps(start.to_dict()) + "\n" + json.dumps(legacy_step) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(JournalCorruptionError):
        await JsonlRunCatalog(tmp_path / "legacy-on-disk").inspect_run("run")

    legacy_terminal = JsonlSessionJournal(tmp_path / "legacy-terminal")
    await legacy_terminal.create("run", {})
    await legacy_terminal.append(
        JournalRecordType.RUN_COMPLETED,
        {"stop_reason": "completed"},
        record_id="run:run:terminal",
    )
    await legacy_terminal.close()
    with pytest.raises(JournalCorruptionError):
        await JsonlRunCatalog(tmp_path / "legacy-terminal").inspect_run("run")

    legacy_input = JsonlSessionJournal(tmp_path / "legacy-input")
    await legacy_input.create("run", {})
    await legacy_input.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"task": "inspect target"},
        record_id="run:input",
    )
    await legacy_input.close()
    with pytest.raises(JournalCorruptionError):
        await JsonlRunCatalog(tmp_path / "legacy-input").inspect_run("run")

    snapshot_run = tmp_path / "legacy-snapshot" / "run"
    snapshot_run.mkdir(parents=True)
    record = JournalRecord.create(
        seq=1,
        record_id="run:start",
        type=JournalRecordType.RUN_STARTED,
        run_id="run",
        payload={},
    )
    snapshot = JournalRecord.create(
        seq=2,
        record_id="run:snapshot",
        type=JournalRecordType.RUN_STARTED,
        run_id="run",
        payload={},
    ).to_dict()
    snapshot["type"] = "state.snapshot"
    (snapshot_run / "journal.jsonl").write_text(
        json.dumps(record.to_dict()) + "\n" + json.dumps(snapshot) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(JournalCorruptionError):
        await JsonlRunCatalog(tmp_path / "legacy-snapshot").inspect_run("run")


@pytest.mark.asyncio
async def test_catalog_rejects_missing_parent_and_lineage_cycle(tmp_path: Path) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {})
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="parent:turn:0:committed",
    )
    child = await parent.fork(boundary, "child")
    await child.close()
    await parent.close()
    (tmp_path / "parent" / "journal.jsonl").unlink()

    with pytest.raises(RunNotFoundError):
        await JsonlRunCatalog(tmp_path).lineage("child")

    cycle_root = tmp_path / "cycle"
    cycle_root.mkdir()
    a_stub = [
        JournalRecord.create(
            seq=1,
            record_id="a:start",
            type=JournalRecordType.RUN_STARTED,
            run_id="a",
            payload={},
        ),
        JournalRecord.create(
            seq=2,
            record_id="a:fork",
            type=JournalRecordType.RUN_FORKED,
            run_id="a",
            payload={
                "parent_run_id": "b",
                "parent_seq": 5,
                "parent_record_id": "b:inherited:a:boundary",
            },
        ),
        JournalRecord.create(
            seq=3,
            record_id="a:boundary",
            type=JournalRecordType.STEP_COMMITTED,
            run_id="a",
            payload={
                "turn": 0,
                "transcript_record_ids": [],
                "tool_terminal_record_ids": [],
            },
        ),
    ]
    b_records = [
        JournalRecord.create(
            seq=1,
            record_id="b:start",
            type=JournalRecordType.RUN_STARTED,
            run_id="b",
            payload={},
        ),
        JournalRecord.create(
            seq=2,
            record_id="b:fork",
            type=JournalRecordType.RUN_FORKED,
            run_id="b",
            payload={
                "parent_run_id": "a",
                "parent_seq": 3,
                "parent_record_id": "a:boundary",
            },
        ),
        *[_inherit("b", index, record) for index, record in enumerate(a_stub, 3)],
    ]
    a_records = [
        a_stub[0],
        a_stub[1],
        *[_inherit("a", index, record) for index, record in enumerate(b_records, 3)],
    ]
    _write_records(cycle_root, "a", a_records)
    _write_records(cycle_root, "b", b_records)

    with pytest.raises(JournalCorruptionError, match="cycle"):
        await JsonlRunCatalog(cycle_root).lineage("a")


def test_run_handle_is_immutable_and_json_safe() -> None:
    now = datetime.now(timezone.utc)
    position = JournalPosition("run", 1, "start")
    handle = RunHandle(
        run_id="run",
        lineage_id="session",
        status=RunStatus.RESUMABLE,
        created_at=now,
        updated_at=now,
        latest_position=position,
        committed_position=None,
        continuation_position=None,
        forked_from=None,
        record_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        handle.task = "changed"  # type: ignore[misc]
    assert json.loads(json.dumps(handle.to_dict()))["latest_position"] == {
        "run_id": "run",
        "seq": 1,
        "record_id": "start",
    }
    assert handle.to_dict()["lineage_id"] == "session"
    assert handle.to_dict()["has_terminal_state"] is False
