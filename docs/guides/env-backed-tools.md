# Environment-backed tools

QitOS tools can depend on named environment capability groups without knowing where
those operations run. This keeps tool schemas, permissions, action lifecycle, and trace
behavior reusable across a local workspace, a container, or a remote runner.

## Capability composition

`CapabilityEnv` maps a group name to one provider object:

```python
from qitos.core.env import RuntimeCapabilitySnapshot
from qitos.kit.env import CapabilityEnv

env = CapabilityEnv(
    {
        "file": my_file_provider,
        "process": my_process_provider,
    },
    name="remote_attempt",
    snapshot=RuntimeCapabilitySnapshot(
        backend="remote-attempt",
        working_directory="/workspace",
        operation_groups=("file", "process"),
        facilities=("process.foreground",),
    ),
)
```

The snapshot is the Runtime fact used by Tool exposure, per-turn model metadata, and
execution. Product profiles may select or narrow that Runtime, but they do not replace
the backend snapshot. A Tool whose required operation group is absent is removed from
the next turn's exposure. Mode-specific facilities can also fail with the stable
`capability_unavailable` category; for example, foreground-only process providers do
not pretend to support managed background work or PTY input.

Providers retain their own lifecycle. The application that creates the container,
remote session, or host resource must close it after the Engine finishes; composing a
`CapabilityEnv` does not transfer ownership.

Engine awaits `Env.ainitialize()`, `Env.ahealth_check()`, and `Env.ateardown()`.
Synchronous legacy hooks are moved off the event loop by the default implementations;
backends with native async allocation can override them without a sync/async bridge.

The stable filesystem and process contracts live in `qitos.core.env` and
`qitos.core.process`.
`FileSystemCapability` supplies root-scoped metadata, bounded text and binary reads,
atomic text replacement, directory operations, and listings. A bounded text read
includes the SHA-256 revision of the complete file. `write_text_atomic()` can require
that revision before replacing the target and returns both the previous and committed
revision. Implementations serialize mutations to one canonical path within the same
environment instance. `CommandCapability.run_argv()` executes fixed arguments without
shell interpolation, while `run()` remains the explicit shell command compatibility
path. Runtime code uses `arun()` and `arun_argv()` so subprocess I/O, deadlines, and
cancellation stay on the caller's event loop.

Host background commands use one Run-owned supervisor. `astart()` returns an opaque
`ProcessHandle`; `apoll()` and `aread()` return immutable snapshots; `awrite()`,
`await_process()`, and `aterminate()` provide the interaction lifecycle. The in-memory
view is byte-bounded and reports omitted output explicitly. The complete normalized
UTF-8 transcript is written under `.qitos/processes/` in the workspace.

```python
import asyncio

from qitos.kit.env.host_env import HostCommandCapability


async def main() -> None:
    commands = HostCommandCapability("/workspace")
    started = await commands.astart(
        "python -u server.py",
        owner_run_id="run-example",
    )
    update = await commands.aread(
        started.handle,
        cursor=started.output.next_cursor,
        wait_seconds=5,
    )
    print(update.output.content)
    await commands.aterminate(started.handle)
    await commands.aclose()


asyncio.run(main())
```

When a Session Journal is supplied to `astart()`, QitOS writes
`process.started` only after the OS process exists and writes one terminal record after
output collection settles. A failed started-record write terminates the process and
returns no usable handle. `Engine.arun()` awaits environment teardown before closing
the Journal, so no process reader or watcher remains detached from the Run. On resume,
`arecover()` converts a start without a terminal record into an immutable `lost`
snapshot; it never reattaches to or replays the command. Fork Journal records are
historical context only and grant no process ownership to the child Run.

`CommandCapability.astart()` also accepts an optional async terminal notifier. The Host
watcher invokes it only after output collection and the `process.terminal` append have
completed. `CodingToolSet` binds that notifier to the active Engine's durable
`RuntimeInput` mailbox and posts one stable `process.completed` event containing the
bounded terminal snapshot. The watcher never calls the model or mutates Agent state;
delivery happens at the next turn safe point. A closed or failed mailbox leaves the
terminal snapshot queryable through the process controls. If the Run stops after
`process.terminal` but before mailbox acceptance, resume derives the same bounded event
from that terminal fact; a completed model transaction that already consumed its stable
event id prevents duplicate delivery.

Register `CodingToolSet(profile="shell")` to expose the shared managed lifecycle to a
model. `run_command(..., run_in_background=True)` returns a process id, and
`process_list`, `process_read`, `process_write`, `process_wait`, and
`process_terminate` operate only on handles owned by the active Run. Reads and waits
honor the current Tool deadline. A backend such as Docker that has not implemented the
managed async contract can still run foreground commands, but reports background mode
as unavailable instead of creating an untracked process.

Host-backed command tools inherit the current process environment by default. An
application that owns a stricter execution boundary can instead pass one complete,
pre-filtered environment snapshot:

```python
from qitos.kit.tool.shell import RunCommand

shell = RunCommand(
    workspace_root="/workspace",
    process_env={
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HTTPS_PROXY": "http://proxy.internal:8080",
    },
)
```

The same snapshot is used for shell commands, fixed-argument subprocesses, and managed
background starts. QitOS does not merge, discover, or filter credentials in this
mapping; the application that constructs the runtime owns that policy.

## Declaring tool requirements

Use `environment_ops` for a tool that can consume a selected Env provider while
retaining an application-owned fallback:

```python
@function_tool(name="read_record", read_only=True, environment_ops=["file"])
def read_record(path: str, runtime_context=None):
    file_ops = select_runtime_ops(runtime_context, "file", local_file_ops)
    return file_ops.read_text(path)
```

Use `required_ops` when the tool has no valid fallback. QitOS preflights required groups
before the run and injects the exact providers into the tool runtime context. If an Env
is selected, declared `environment_ops` must also be present; QitOS does not silently
switch to another backend after environment selection.

Applications that require fail-closed remote execution can use
`CodingToolSet(allow_local_fallback=False)`. Its tool specs promote the selected
environment groups to hard requirements, so direct calls and Engine runs both fail
instead of touching the Controller filesystem.

## Standard workspace profile

`CodingToolSet(profile="workspace")` exposes a compact general-purpose surface:

- `read_file`, `write_file`, and exact `edit_file`;
- fixed-argv `glob` and `grep` through `rg`;
- bounded `hex_view` for binary inspection; and
- `list_files`, `list_tree`, and `make_directory`.

Reads are line- and byte-bounded and return `content_sha256`. `write_file` accepts that
value as `expected_sha256`. Exact edits fail on missing or ambiguous text unless
`replace_all` is explicit, and they automatically use the revision read before the
edit as a compare-and-swap guard. A concurrent change therefore returns
`file_revision_conflict` without replacing the newer file. Search distinguishes no
matches from process errors. Tools that return QitOS's structured
`{ "status": "error", ... }` contract produce an `ActionStatus.ERROR`; an error payload
is never recorded as a successful action.

This profile deliberately excludes controller-local notebook, LSP, worktree, cron,
browser, and arbitrary HTTP helpers. Applications may compose those separately when
their execution and security boundaries are defined.
