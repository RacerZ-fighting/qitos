# Environment-backed tools

QitOS tools can depend on named environment capability groups without knowing where
those operations run. This keeps tool schemas, permissions, action lifecycle, and trace
behavior reusable across a local workspace, a container, or a remote runner.

## Capability composition

`CapabilityEnv` maps a group name to one provider object:

```python
from qitos.kit.env import CapabilityEnv

env = CapabilityEnv(
    {
        "file": my_file_provider,
        "process": my_process_provider,
    },
    name="remote_attempt",
    attestation={"workspace": "/workspace"},
)
```

Providers retain their own lifecycle. The application that creates the container,
remote session, or host resource must close it after the Engine finishes; composing a
`CapabilityEnv` does not transfer ownership.

The stable filesystem and process contracts live in `qitos.core.env` and
`qitos.core.process`.
`FileSystemCapability` supplies root-scoped metadata, bounded text and binary reads,
writes, directory operations, and listings. `CommandCapability.run_argv()` executes
fixed arguments without shell interpolation, while `run()` remains the explicit shell
command compatibility path. Runtime code uses `arun()` and `arun_argv()` so subprocess
I/O, deadlines, and cancellation stay on the caller's event loop.

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

Reads are line- and byte-bounded, edits fail on missing or ambiguous text unless
`replace_all` is explicit, and search distinguishes no matches from process errors.
Tools that return QitOS's structured `{ "status": "error", ... }` contract now produce
an `ActionStatus.ERROR`; an error payload is never recorded as a successful action.

This profile deliberately excludes controller-local notebook, LSP, worktree, cron,
browser, and arbitrary HTTP helpers. Applications may compose those separately when
their execution and security boundaries are defined.
