"""Fork and time-travel support for checkpoints.

Borrowed from LangGraph's fork mechanism:
- references/langgraph/libs/langgraph/langgraph/pregel/_loop.py

Fork creates a new checkpoint branch from an existing checkpoint,
without overwriting the original history.  Time-travel restores
state from an earlier checkpoint within the same thread.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional
from uuid import uuid4

from .store import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointMetadata,
    CheckpointStore,
    CheckpointTuple,
)


async def fork_checkpoint(
    store: CheckpointStore,
    config: CheckpointConfig,
    new_thread_id: Optional[str] = None,
) -> CheckpointConfig:
    """Create a new branch from an existing checkpoint.

    Args:
        store: The checkpoint store.
        config: Points to the source checkpoint to fork from.
        new_thread_id: If provided, the fork goes to a different thread
            (true branch).  If ``None``, the fork shares the same thread
            (time-travel).

    Returns:
        Config pointing to the newly created checkpoint.
    """
    tuple_ = await store.get_tuple(config)
    if tuple_ is None:
        raise ValueError(f"Checkpoint not found for config: {config}")

    source = tuple_.checkpoint
    dest_thread = new_thread_id or source.thread_id

    # Create a new checkpoint with same state but new id
    forked = Checkpoint(
        id=CheckpointId(uuid4().hex),
        thread_id=dest_thread,
        step=source.step,
        state_data=deepcopy(source.state_data),
        task_text=source.task_text,
        task_data=deepcopy(source.task_data),
        history=deepcopy(source.history),
        parent_id=source.id,
        parent_thread_id=source.thread_id,
        schema_version=source.schema_version,
    )

    metadata: CheckpointMetadata = {
        "source": "fork",
        "step": source.step,
        "run_id": tuple_.metadata.get("run_id", ""),
    }
    if new_thread_id is not None:
        metadata["parents"] = {dest_thread: source.id}

    dest_config = CheckpointConfig(thread_id=dest_thread)
    return await store.put(dest_config, forked, metadata)


async def list_fork_history(
    store: CheckpointStore,
    config: CheckpointConfig,
    max_depth: int = 100,
) -> List[CheckpointTuple]:
    """Walk the parent chain backwards from a checkpoint.

    Returns checkpoints from the given checkpoint back to the root,
    in reverse chronological order (newest first).
    """
    results: List[CheckpointTuple] = []
    visited: set[str] = set()

    current_config = config
    for _ in range(max_depth):
        tuple_ = await store.get_tuple(current_config)
        if tuple_ is None:
            break
        cp_id = tuple_.checkpoint.id
        if cp_id in visited:
            break
        visited.add(cp_id)
        results.append(tuple_)

        parent_config = tuple_.parent_config
        if parent_config is None:
            break
        current_config = parent_config

    return results


__all__ = ["fork_checkpoint", "list_fork_history"]
