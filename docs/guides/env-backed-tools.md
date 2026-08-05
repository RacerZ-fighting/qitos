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

The stable filesystem and process contracts live in `qitos.core.env`.
`FileSystemCapability` supplies root-scoped metadata, bounded text and binary reads,
writes, directory operations, and listings. `CommandCapability.run_argv()` executes
fixed arguments without shell interpolation, while `run()` remains the explicit shell
command path.

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
