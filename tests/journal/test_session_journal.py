from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from qitos.core.journal import (
    JournalAppendCancelled,
    JournalClosedError,
    JournalCommitError,
    JournalCommitState,
    JournalCorruptionError,
    JournalError,
    JournalOwnershipError,
    JournalRecordRef,
    JournalRecordType,
    JournalUnsupportedVersionError,
)
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.journal import jsonl as jsonl_module
from qitos.kit.journal._sqlite_index import JournalIndexError, SqliteJournalIndex
from qitos.kit.journal._writer_lease import JournalWriterLease


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
async def test_append_settles_canonical_write_before_propagating_cancellation(
    tmp_path: Path,
) -> None:
    commit_started = asyncio.Event()
    release_commit = threading.Event()
    loop = asyncio.get_running_loop()
    block_next_sync = False

    def sync_file(_descriptor: int, _mode: str) -> None:
        if not block_next_sync:
            return
        loop.call_soon_threadsafe(commit_started.set)
        if not release_commit.wait(timeout=5):
            raise RuntimeError("test did not release Journal commit")

    journal = JsonlSessionJournal(tmp_path, sync_file=sync_file)
    await journal.create("run-1", {})
    block_next_sync = True
    append = asyncio.create_task(
        journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {"content": "durable"},
            record_id="input-1",
        )
    )
    await asyncio.wait_for(commit_started.wait(), timeout=1)

    append.cancel()
    release_commit.set()
    if sys.version_info >= (3, 11):
        with pytest.raises(JournalAppendCancelled) as cancelled:
            await append
        assert cancelled.value.committed_position is not None
        assert cancelled.value.committed_position.record_id == "input-1"
    else:
        # Python 3.10 normalizes a CancelledError subclass when it crosses the
        # boundary of a Task that was cancelled directly. The durable Journal
        # state remains authoritative and is verified below.
        with pytest.raises(asyncio.CancelledError):
            await append
    assert [record.record_id for record in await journal.replay()] == [
        "run-1:start",
        "input-1",
    ]


@pytest.mark.asyncio
async def test_cancelled_append_with_failed_rollback_poisoned_until_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit_started = asyncio.Event()
    release_commit = threading.Event()
    loop = asyncio.get_running_loop()
    fail_next_sync = False

    def sync_file(_descriptor: int, _mode: str) -> None:
        nonlocal fail_next_sync
        if not fail_next_sync:
            return
        fail_next_sync = False
        loop.call_soon_threadsafe(commit_started.set)
        if not release_commit.wait(timeout=5):
            raise RuntimeError("test did not release Journal commit")
        raise OSError("injected sync failure")

    journal = JsonlSessionJournal(tmp_path, sync_file=sync_file)
    await journal.create("run-1", {})

    def fail_rollback(_descriptor: int, _length: int) -> None:
        raise OSError("injected rollback failure")

    fail_next_sync = True
    with monkeypatch.context() as patch:
        patch.setattr(jsonl_module.os, "ftruncate", fail_rollback)
        append = asyncio.create_task(
            journal.append(
                JournalRecordType.INPUT_ACCEPTED,
                {"content": "possibly durable"},
                record_id="input-1",
            )
        )
        await asyncio.wait_for(commit_started.wait(), timeout=1)
        append.cancel()
        release_commit.set()
        if sys.version_info >= (3, 11):
            with pytest.raises(JournalAppendCancelled) as cancelled:
                await append
            assert cancelled.value.commit_state is JournalCommitState.UNKNOWN
            assert cancelled.value.committed_position is None
            assert cancelled.value.pending_position is not None
            assert cancelled.value.pending_position.record_id == "input-1"
            assert isinstance(cancelled.value.commit_error, JournalCommitError)
        else:
            with pytest.raises(asyncio.CancelledError):
                await append

    with pytest.raises(JournalError, match="close and reopen"):
        await journal.replay()
    with pytest.raises(JournalError, match="close and reopen"):
        await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {"content": "must not reuse seq"},
            record_id="input-2",
        )

    await journal.close()
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")
    assert [record.record_id for record in await reopened.replay()] == [
        "run-1:start",
        "input-1",
    ]
    await reopened.close()


@pytest.mark.asyncio
async def test_failed_append_rolls_back_before_allowing_next_record(
    tmp_path: Path,
) -> None:
    fail_next_sync = False

    def sync_file(_descriptor: int, _mode: str) -> None:
        nonlocal fail_next_sync
        if fail_next_sync:
            fail_next_sync = False
            raise OSError("injected sync failure")

    journal = JsonlSessionJournal(tmp_path, sync_file=sync_file)
    await journal.create("run-1", {})
    fail_next_sync = True

    with pytest.raises(JournalCommitError) as failed:
        await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {"content": "rolled back"},
            record_id="input-1",
        )

    assert failed.value.commit_state is JournalCommitState.NOT_COMMITTED
    position = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"content": "committed"},
        record_id="input-2",
    )
    assert position.seq == 2
    assert [record.record_id for record in await journal.replay()] == [
        "run-1:start",
        "input-2",
    ]


@pytest.mark.asyncio
async def test_cancelled_create_settles_and_removes_unstarted_run(
    tmp_path: Path,
) -> None:
    directory_sync_started = asyncio.Event()
    release_directory_sync = threading.Event()
    loop = asyncio.get_running_loop()

    def sync_directory(path: Path) -> None:
        if path != tmp_path:
            return
        loop.call_soon_threadsafe(directory_sync_started.set)
        if not release_directory_sync.wait(timeout=5):
            raise RuntimeError("test did not release directory sync")

    journal = JsonlSessionJournal(tmp_path, sync_directory=sync_directory)
    creating = asyncio.create_task(journal.create("run-1", {}))
    await asyncio.wait_for(directory_sync_started.wait(), timeout=1)

    creating.cancel()
    release_directory_sync.set()
    with pytest.raises(asyncio.CancelledError):
        await creating

    replacement = JsonlSessionJournal(tmp_path)
    await replacement.create("run-1", {})
    await replacement.close()


@pytest.mark.asyncio
async def test_failed_create_removes_files_created_before_directory_sync_error(
    tmp_path: Path,
) -> None:
    def fail_parent_sync(path: Path) -> None:
        if path == tmp_path:
            raise OSError("injected directory sync failure")

    journal = JsonlSessionJournal(tmp_path, sync_directory=fail_parent_sync)

    with pytest.raises(OSError, match="directory sync"):
        await journal.create("run-1", {})

    replacement = JsonlSessionJournal(tmp_path)
    await replacement.create("run-1", {})
    await replacement.close()


@pytest.mark.asyncio
async def test_cancelled_open_releases_acquired_writer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = JsonlSessionJournal(tmp_path)
    await seed.create("run-1", {})
    await seed.close()
    lease_acquired = asyncio.Event()
    release_acquire = threading.Event()
    loop = asyncio.get_running_loop()
    acquire = JournalWriterLease.acquire

    def acquire_then_block(run_directory: Path, run_id: str) -> JournalWriterLease:
        lease = acquire(run_directory, run_id)
        loop.call_soon_threadsafe(lease_acquired.set)
        if not release_acquire.wait(timeout=5):
            lease.release()
            raise RuntimeError("test did not release writer acquisition")
        return lease

    monkeypatch.setattr(
        JournalWriterLease,
        "acquire",
        staticmethod(acquire_then_block),
    )
    opening = asyncio.create_task(JsonlSessionJournal(tmp_path).open("run-1"))
    await asyncio.wait_for(lease_acquired.wait(), timeout=1)

    opening.cancel()
    release_acquire.set()
    with pytest.raises(asyncio.CancelledError):
        await opening

    monkeypatch.setattr(JournalWriterLease, "acquire", acquire)
    successor = JsonlSessionJournal(tmp_path)
    await successor.open("run-1")
    await successor.close()


@pytest.mark.asyncio
async def test_cancelled_open_closes_loaded_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = JsonlSessionJournal(tmp_path)
    await seed.create("run-1", {})
    await seed.close()
    load_completed = asyncio.Event()
    release_load = threading.Event()
    projection_closed = threading.Event()
    loop = asyncio.get_running_loop()
    opener = JsonlSessionJournal(tmp_path)
    load_records = opener._load_records
    close_projection = SqliteJournalIndex.close

    def load_then_block(path: Path, run_id: str):
        result = load_records(path, run_id)
        assert result[1] is not None
        loop.call_soon_threadsafe(load_completed.set)
        if not release_load.wait(timeout=5):
            result[1].close()
            raise RuntimeError("test did not release Journal load")
        return result

    def close_and_record(index: SqliteJournalIndex) -> None:
        projection_closed.set()
        close_projection(index)

    monkeypatch.setattr(opener, "_load_records", load_then_block)
    monkeypatch.setattr(SqliteJournalIndex, "close", close_and_record)
    opening = asyncio.create_task(opener.open("run-1"))
    await asyncio.wait_for(load_completed.wait(), timeout=1)

    opening.cancel()
    release_load.set()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert projection_closed.is_set()


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
async def test_payload_is_canonical_before_append_and_stable_after_reopen(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})

    first = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"items": ("one", "two")},
        record_id="stable",
    )

    assert (await journal.replay())[-1].payload == {"items": ["one", "two"]}
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-1")
    settled = await reopened.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"items": ("one", "two")},
        record_id="stable",
    )

    assert settled == first
    assert [record.record_id for record in await reopened.replay()].count("stable") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"nested": {1: "non-string key"}},
        {"number": float("nan")},
        {"value": b"bytes"},
        {"value": {"set"}},
        {"value": object()},
    ],
)
async def test_payload_rejects_non_json_values_before_writing(
    tmp_path: Path,
    invalid_payload: dict[str, object],
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    before = journal.path.read_bytes()

    with pytest.raises(JournalError, match="strict JSON value"):
        await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            invalid_payload,
            record_id="invalid",
        )

    assert journal.path.read_bytes() == before


@pytest.mark.asyncio
async def test_payload_rejects_cycles_before_writing(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    before = journal.path.read_bytes()

    with pytest.raises(JournalError, match="strict JSON value"):
        await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            cyclic,
            record_id="invalid",
        )

    assert journal.path.read_bytes() == before


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
    script = textwrap.dedent("""
        import asyncio
        import sys

        from qitos.kit.journal import JsonlSessionJournal

        async def main() -> None:
            journal = JsonlSessionJournal(sys.argv[1])
            await journal.open("run-1")
            print("ready", flush=True)
            await asyncio.Event().wait()

        asyncio.run(main())
        """)
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
@pytest.mark.parametrize("schema_version", [0, 2])
async def test_replay_distinguishes_unsupported_schema_from_corruption(
    tmp_path: Path,
    schema_version: int,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    await journal.close()
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["schema_version"] = schema_version
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    invalid = "\n".join(lines).encode("utf-8")
    journal.path.write_bytes(invalid)

    with pytest.raises(
        JournalUnsupportedVersionError,
        match=rf"found {schema_version}.*supports 1",
    ):
        await JsonlSessionJournal(tmp_path).open("run-1")
    assert journal.path.read_bytes() == invalid


@pytest.mark.asyncio
async def test_fork_creates_an_independently_replayable_journal(tmp_path: Path) -> None:
    parent = JsonlSessionJournal(tmp_path)
    await parent.create("parent", {"task": "inspect"})
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "turn": 0,
            "transcript_record_ids": [],
            "tool_terminal_record_ids": [],
        },
        record_id="parent:turn:0:committed",
    )

    child = await parent.fork(position, "child")
    parent.path.unlink()
    inherited = await child.replay()

    assert child.run_id == "child"
    assert [record.type for record in inherited[:2]] == [
        JournalRecordType.RUN_STARTED,
        JournalRecordType.RUN_FORKED,
    ]
    assert inherited[-1].payload["origin_record_id"] == "parent:turn:0:committed"
    assert inherited[-1].payload["record"]["type"] == "step.committed"
    assert (
        json.loads(child.path.read_text(encoding="utf-8").splitlines()[0])["run_id"]
        == "child"
    )


@pytest.mark.asyncio
async def test_committed_tool_transaction_lookup_rebuilds_and_isolated(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-1", {})
    terminal_id = "run-1:turn:3:tool:call-1:terminal"
    transcript_id = "run-1:turn:3:transcript:0"
    reference = JournalRecordRef("run-1", terminal_id)
    await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        {
            "message": {
                "role": "tool",
                "tool_call_id": "call-1",
                "tool_name": "inspect",
                "result": {
                    "status": "timed_out",
                    "output": {"process_status": "running"},
                    "error": "",
                    "metadata": {"evidence_id": "evidence-1"},
                    "model_output": "service is reachable",
                },
                "timestamp": 1.0,
            }
        },
        record_id=transcript_id,
    )
    await journal.append(
        JournalRecordType.TOOL_TERMINAL,
        {
            "turn": 3,
            "call_id": "call-1",
            "call": {
                "id": "call-1",
                "name": "inspect",
                "arguments": {"target": "service"},
            },
            "message_record_id": transcript_id,
        },
        record_id=terminal_id,
    )

    assert journal.find_tool_transaction(reference) is None

    committed = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "turn": 3,
            "transcript_record_ids": [transcript_id],
            "tool_terminal_record_ids": [terminal_id],
        },
        record_id="run-1:turn:3:committed",
    )
    transaction = journal.find_tool_transaction(reference)

    assert transaction is not None
    assert transaction.terminal == reference
    assert transaction.committed_at == committed
    assert transaction.action.id == "call-1"
    assert transaction.action.name == "inspect"
    assert transaction.result.status == "timed_out"
    assert transaction.result.error == ""
    assert transaction.result.model_visible_output == "service is reachable"

    reread = journal.find_tool_transaction(reference)
    assert reread is not None
    assert reread.action.arguments == {"target": "service"}
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
    terminal_id = "parent:turn:0:tool:call-1:terminal"
    transcript_id = "parent:turn:0:transcript:0"
    reference = JournalRecordRef("parent", terminal_id)
    await parent.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        {
            "message": {
                "role": "tool",
                "tool_call_id": "call-1",
                "tool_name": "inspect",
                "result": {
                    "status": "success",
                    "output": "canonical",
                    "error": None,
                    "metadata": {},
                },
                "timestamp": 1.0,
            }
        },
        record_id=transcript_id,
    )
    await parent.append(
        JournalRecordType.TOOL_TERMINAL,
        {
            "turn": 0,
            "call_id": "call-1",
            "call": {
                "id": "call-1",
                "name": "inspect",
                "arguments": {},
            },
            "message_record_id": transcript_id,
        },
        record_id=terminal_id,
    )
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "turn": 0,
            "transcript_record_ids": [transcript_id],
            "tool_terminal_record_ids": [terminal_id],
        },
        record_id="parent:turn:0:committed",
    )

    child = await parent.fork(position, "child")
    parent.path.unlink()

    inherited = child.find_tool_transaction(reference)
    assert inherited is not None
    assert inherited.terminal.run_id == "parent"
    assert inherited.result.model_visible_output == "canonical"
    assert child.find_tool_transaction(JournalRecordRef("child", terminal_id)) is None

    await child.close()
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("child")
    restored = reopened.find_tool_transaction(reference)
    assert restored is not None
    assert restored.result.model_visible_output == "canonical"
