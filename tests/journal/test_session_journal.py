from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from qitos.core.journal import (
    JournalClosedError,
    JournalCorruptionError,
    JournalOwnershipError,
    JournalRecordRef,
    JournalRecordType,
)
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.journal import jsonl as jsonl_module
from qitos.kit.journal._sqlite_index import JournalIndexError, SqliteJournalIndex


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
async def test_append_and_replay_isolate_nested_payloads(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    payload = {"nested": {"items": ["canonical"]}}

    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        payload,
        record_id="input-1",
    )
    payload["nested"]["items"].append("caller mutation")
    replayed = await journal.replay()
    replayed[1].payload["nested"]["items"].append("result mutation")

    reread = await journal.replay()
    assert reread[1].payload == {"nested": {"items": ["canonical"]}}


@pytest.mark.asyncio
async def test_writer_ownership_requires_close_before_reopen(tmp_path: Path) -> None:
    owner = JsonlSessionJournal(tmp_path)
    await owner.create("run-1", {})

    with pytest.raises(JournalOwnershipError, match="active Journal writer"):
        await JsonlSessionJournal(tmp_path).open("run-1")

    await owner.close()
    successor = JsonlSessionJournal(tmp_path)
    await successor.open("run-1")
    await successor.close()

    with pytest.raises(JournalClosedError, match="closed"):
        await owner.append(
            JournalRecordType.INPUT_ACCEPTED,
            {},
            record_id="late",
        )
    assert [record.type for record in await owner.replay()] == [
        JournalRecordType.RUN_STARTED
    ]


@pytest.mark.asyncio
async def test_process_exit_releases_writer_ownership(tmp_path: Path) -> None:
    seed = JsonlSessionJournal(tmp_path)
    await seed.create("run-1", {})
    await seed.close()
    script = textwrap.dedent(
        """
        import asyncio
        import sys

        from qitos.kit.journal import JsonlSessionJournal

        async def main() -> None:
            journal = JsonlSessionJournal(sys.argv[1])
            await journal.open("run-1")
            print("ready", flush=True)
            await asyncio.Event().wait()

        asyncio.run(main())
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready = await asyncio.wait_for(
            asyncio.to_thread(process.stdout.readline),
            timeout=5,
        )
        assert ready.strip() == "ready"
        with pytest.raises(JournalOwnershipError):
            await JsonlSessionJournal(tmp_path).open("run-1")
    finally:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)

    successor = JsonlSessionJournal(tmp_path)
    await successor.open("run-1")
    await successor.close()


@pytest.mark.asyncio
async def test_context_manager_releases_writer_ownership(tmp_path: Path) -> None:
    async with JsonlSessionJournal(tmp_path) as journal:
        await journal.create("run-1", {})

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")
    await reopened.close()


@pytest.mark.asyncio
async def test_projection_close_failure_still_releases_writer_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    close_projection = SqliteJournalIndex.close

    def close_then_fail(index: SqliteJournalIndex) -> None:
        close_projection(index)
        raise JournalIndexError("injected close failure")

    with monkeypatch.context() as patch:
        patch.setattr(SqliteJournalIndex, "close", close_then_fail)
        with pytest.raises(JournalIndexError, match="injected close failure"):
            await journal.close()

    successor = JsonlSessionJournal(tmp_path)
    await successor.open("run-1")
    await successor.close()


@pytest.mark.asyncio
async def test_current_sqlite_projection_remains_available_after_reopen(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "indexed"},
        record_id="input-1",
    )
    index_path = journal.index_path
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")

    assert index_path.is_file()
    assert [record.record_id for record in await reopened.replay()] == [
        "run-1:start",
        "input-1",
    ]
    await reopened.close()


@pytest.mark.asyncio
async def test_replay_uses_jsonl_when_current_projection_content_diverges(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "one"},
        record_id="input-1",
    )
    path = journal.path
    await journal.close()
    source_stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b'"one"', b'"two"'))
    os.utime(
        path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
    )

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")

    assert (await reopened.replay())[-1].payload == {"content": "two"}
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    ["invalid_database", "unsupported_schema", "record_checksum"],
)
async def test_broken_sqlite_projection_rebuilds_from_jsonl(
    tmp_path: Path,
    damage: str,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    index_path = journal.index_path
    await journal.close()
    if damage == "invalid_database":
        index_path.write_bytes(b"not a sqlite database")
    elif damage == "unsupported_schema":
        with sqlite3.connect(index_path) as connection:
            connection.execute("PRAGMA user_version = 999")
    else:
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "UPDATE journal_record SET record_sha256 = ? WHERE seq = 1",
                ("not-the-canonical-digest",),
            )

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")

    assert [record.record_id for record in await reopened.replay()] == ["run-1:start"]
    await reopened.close()


@pytest.mark.asyncio
async def test_projection_failure_does_not_change_canonical_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})

    def fail_projection(*_: object, **__: object) -> None:
        raise JournalIndexError("injected projection failure")

    monkeypatch.setattr(SqliteJournalIndex, "append", fail_projection)
    position = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "canonical"},
        record_id="input-1",
    )
    await journal.close()

    assert position.record_id == "input-1"
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")
    assert (await reopened.replay())[-1].payload == {"content": "canonical"}
    await reopened.close()


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
    await journal.close()
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
    await journal.close()
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    lines[1] = "not-json"
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalCorruptionError, match="line 2"):
        await JsonlSessionJournal(tmp_path).open("run-1")


@pytest.mark.asyncio
async def test_replay_rejects_unsupported_canonical_schema(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.close()
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["schema_version"] = 2
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    invalid = "\n".join(lines).encode("utf-8")
    journal.path.write_bytes(invalid)

    with pytest.raises(JournalCorruptionError, match="schema version"):
        await JsonlSessionJournal(tmp_path).open("run-1")
    assert journal.path.read_bytes() == invalid


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


@pytest.mark.asyncio
async def test_committed_tool_transaction_lookup_rebuilds_and_isolated(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    terminal_id = "transaction-1:tool:0:terminal"
    reference = JournalRecordRef("run-1", terminal_id)
    await journal.append(
        JournalRecordType.TOOL_TERMINAL,
        {
            "step_id": 3,
            "transaction_id": "transaction-1",
            "action_index": 0,
            "action": {
                "name": "inspect",
                "args": {"target": "service"},
                "action_id": "call-1",
                "metadata": {},
            },
            "result": {
                "status": "success",
                "output": {"reachable": True},
                "error": None,
                "metadata": {"evidence_id": "evidence-1"},
                "model_output": "service is reachable",
            },
        },
        record_id=terminal_id,
    )

    assert journal.find_tool_transaction(reference) is None

    committed = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "step_id": 3,
            "transaction_id": "transaction-1",
            "terminal_record_ids": [terminal_id],
        },
        record_id="transaction-1:committed",
    )
    transaction = journal.find_tool_transaction(reference)

    assert transaction is not None
    assert transaction.terminal == reference
    assert transaction.committed_at == committed
    assert transaction.step_id == 3
    assert transaction.action_index == 0
    assert transaction.action.name == "inspect"
    assert transaction.result.model_visible_output == "service is reachable"

    transaction.action.args["target"] = "mutated"
    transaction.result.metadata["evidence_id"] = "mutated"
    reread = journal.find_tool_transaction(reference)
    assert reread is not None
    assert reread.action.args == {"target": "service"}
    assert reread.result.metadata == {"evidence_id": "evidence-1"}

    await journal.close()
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")
    restored = reopened.find_tool_transaction(reference)
    assert restored is not None
    assert restored.result.model_visible_output == "service is reachable"


@pytest.mark.asyncio
async def test_fork_resolves_inherited_committed_tool_origin(
    tmp_path: Path,
) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {})
    terminal_id = "transaction-1:tool:0:terminal"
    reference = JournalRecordRef("parent", terminal_id)
    await parent.append(
        JournalRecordType.TOOL_TERMINAL,
        {
            "step_id": 0,
            "transaction_id": "transaction-1",
            "action_index": 0,
            "action": {
                "name": "inspect",
                "args": {},
                "action_id": "call-1",
                "metadata": {},
            },
            "result": {
                "status": "success",
                "output": "canonical",
                "error": None,
                "metadata": {},
            },
        },
        record_id=terminal_id,
    )
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "step_id": 0,
            "transaction_id": "transaction-1",
            "terminal_record_ids": [terminal_id],
        },
        record_id="transaction-1:committed",
    )

    child = await parent.fork(position, "child")
    parent.path.unlink()

    inherited = child.find_tool_transaction(reference)
    assert inherited is not None
    assert inherited.terminal.run_id == "parent"
    assert inherited.result.model_visible_output == "canonical"
    assert (
        child.find_tool_transaction(JournalRecordRef("child", terminal_id)) is None
    )

    await child.close()
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("child")
    restored = reopened.find_tool_transaction(reference)
    assert restored is not None
    assert restored.result.model_visible_output == "canonical"
