from __future__ import annotations

import json
from pathlib import Path

import pytest

from qitos.core.journal import JournalCorruptionError, JournalRecordType
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.journal import jsonl as jsonl_module


@pytest.mark.asyncio
async def test_journal_round_trips_durable_records(tmp_path: Path) -> None:
    synced: list[tuple[int, str]] = []
    journal = JsonlSessionJournal(
        tmp_path,
        sync_file=lambda descriptor, mode: synced.append((descriptor, mode)),
        sync_directory=lambda path: synced.append((-1, str(path))),
    )

    await journal.create("run-1", {"task": "inspect"})
    first = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "inspect target"},
        record_id="input-1",
    )
    second = await journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"reason": "final"},
        record_id="complete-1",
    )

    assert (first.seq, second.seq) == (2, 3)
    assert all(record.run_id == "run-1" for record in await journal.replay())
    assert [record.record_id for record in await journal.replay()] == [
        "run-1:start",
        "input-1",
        "complete-1",
    ]
    assert any(mode in {"full", "data", "file"} for _, mode in synced)
    assert any(descriptor == -1 for descriptor, _ in synced)


@pytest.mark.asyncio
async def test_darwin_records_request_full_filesystem_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes: list[str] = []
    monkeypatch.setattr(jsonl_module.sys, "platform", "darwin")
    journal = JsonlSessionJournal(
        tmp_path,
        sync_file=lambda descriptor, mode: modes.append(mode),
        sync_directory=lambda path: None,
    )

    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "durable"},
        record_id="input-1",
    )

    assert modes == ["full", "full"]


@pytest.mark.asyncio
async def test_append_settles_duplicate_record_id_without_duplicate_line(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})

    first = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "one"},
        record_id="stable",
    )
    settled = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "one"},
        record_id="stable",
    )

    assert settled == first
    assert [record.record_id for record in await journal.replay()].count("stable") == 1


@pytest.mark.asyncio
async def test_replay_discards_only_a_torn_final_line(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "complete"},
        record_id="input-1",
    )
    path = journal.path
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"seq":3')

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")

    assert [record.seq for record in await reopened.replay()] == [1, 2]
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.asyncio
async def test_replay_rejects_corruption_before_the_final_line(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "one"},
        record_id="input-1",
    )
    await journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"reason": "final"},
        record_id="complete-1",
    )
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    lines[1] = "not-json"
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalCorruptionError, match="line 2"):
        await JsonlSessionJournal(tmp_path).open("run-1")


@pytest.mark.asyncio
async def test_fork_creates_an_independently_replayable_journal(tmp_path: Path) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {"task": "inspect"})
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "step_id": 0,
            "consumed_terminal_ids": [],
            "state_delta": [],
            "before_digest": "same",
            "after_digest": "same",
        },
        record_id="parent-step",
    )

    child = await parent.fork(position, "child")
    parent.path.unlink()
    inherited = await child.replay()

    assert child.run_id == "child"
    assert [record.type for record in inherited[:2]] == [
        JournalRecordType.RUN_STARTED,
        JournalRecordType.RUN_FORKED,
    ]
    assert inherited[-1].payload["origin_record_id"] == "parent-step"
    assert inherited[-1].payload["record"]["type"] == "step.committed"
    assert json.loads(child.path.read_text(encoding="utf-8").splitlines()[0])[
        "run_id"
    ] == "child"
