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


def _commit_payload(step_id: int = 0) -> dict[str, object]:
    return {
        "step_id": step_id,
        "transaction_id": f"transaction-{step_id}",
        "terminal_record_ids": [],
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
        {"task": "inspect target"},
        record_id="input",
    )
    committed = await journal.append(
        JournalRecordType.STATE_SNAPSHOT,
        {"state": {}, "state_digest": "digest", "reason": "initial"},
        record_id="snapshot",
    )

    handle = await JsonlRunCatalog(tmp_path).inspect_run("active")

    assert handle.status is RunStatus.RESUMABLE
    assert handle.can_resume is True
    assert handle.can_fork is True
    assert handle.committed_position == committed
    assert handle.agent_name == "worker"
    assert handle.task == "inspect target"
    await journal.append(
        JournalRecordType.RUN_INTERRUPTED,
        {"reason": "cancelled"},
        record_id="interrupted",
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
        {"task": "one"},
        record_id="input",
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
async def test_catalog_lists_and_projects_terminal_lifecycle(tmp_path: Path) -> None:
    first = JsonlSessionJournal(tmp_path)
    await first.create("first", {"agent": "agent-a"})
    await first.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"task": "first task"},
        record_id="first-input",
    )
    await first.append(
        JournalRecordType.RUN_INTERRUPTED,
        {"reason": "cancelled"},
        record_id="first-interrupted",
    )
    await first.close()

    second = JsonlSessionJournal(tmp_path)
    await second.create("second", {"agent": "agent-b"})
    await second.append(
        JournalRecordType.RUN_COMPLETED,
        {"stop_reason": "completed"},
        record_id="second-complete",
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
        {"task": "trace lineage"},
        record_id="parent-input",
    )
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        _commit_payload(),
        record_id="parent-commit",
    )
    child = await parent.fork(boundary, "child")
    child_handle = await JsonlRunCatalog(tmp_path).inspect_run("child")
    assert child_handle.committed_position is not None

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
    assert [handle.run_id for handle in children] == ["child"]
    assert lineage[-1].task == "trace lineage"
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
async def test_catalog_rejects_missing_parent_and_lineage_cycle(tmp_path: Path) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {})
    boundary = await parent.append(
        JournalRecordType.STATE_SNAPSHOT,
        {"state": {}, "state_digest": "digest", "reason": "initial"},
        record_id="boundary",
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
            type=JournalRecordType.STATE_SNAPSHOT,
            run_id="a",
            payload={},
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
        *[
            _inherit("a", index, record)
            for index, record in enumerate(b_records, 3)
        ],
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
        status=RunStatus.RESUMABLE,
        created_at=now,
        updated_at=now,
        latest_position=position,
        committed_position=None,
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
