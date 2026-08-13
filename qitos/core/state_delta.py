"""Deterministic JSON state deltas used by the canonical journal."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


class StateDeltaError(ValueError):
    """Raised when a state delta cannot be applied exactly."""


def state_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_state_delta(before: Any, after: Any) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    _build(before, after, "", operations)
    return operations


def _build(before: Any, after: Any, path: str, output: list[dict[str, Any]]) -> None:
    if before == after:
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) - set(after)):
            output.append({"op": "remove", "path": _child_path(path, key)})
        for key in sorted(set(after) - set(before)):
            output.append(
                {"op": "add", "path": _child_path(path, key), "value": deepcopy(after[key])}
            )
        for key in sorted(set(before) & set(after)):
            _build(before[key], after[key], _child_path(path, key), output)
        return
    if isinstance(before, list) and isinstance(after, list):
        common = min(len(before), len(after))
        for index in range(common):
            _build(before[index], after[index], f"{path}/{index}", output)
        for index in range(len(before) - 1, len(after) - 1, -1):
            output.append({"op": "remove", "path": f"{path}/{index}"})
        for index in range(common, len(after)):
            output.append(
                {"op": "add", "path": f"{path}/-", "value": deepcopy(after[index])}
            )
        return
    output.append({"op": "replace", "path": path, "value": deepcopy(after)})


def _child_path(path: str, key: object) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def apply_state_delta(value: Any, delta: Sequence[Mapping[str, Any]]) -> Any:
    owned = deepcopy(value)
    for raw_operation in delta:
        operation = dict(raw_operation)
        op = operation.get("op")
        path = operation.get("path")
        allowed = {"op", "path"} if op == "remove" else {"op", "path", "value"}
        if set(operation) != allowed or op not in {"add", "remove", "replace"}:
            raise StateDeltaError("state delta operation is invalid")
        if not isinstance(path, str):
            raise StateDeltaError("state delta path must be text")
        if path == "":
            if op == "remove":
                raise StateDeltaError("root state cannot be removed")
            owned = deepcopy(operation["value"])
            continue
        parent, token = _resolve_parent(owned, path)
        if isinstance(parent, dict):
            exists = token in parent
            if op == "add":
                if exists:
                    raise StateDeltaError("state delta add target already exists")
                parent[token] = deepcopy(operation["value"])
            elif op == "replace":
                if not exists:
                    raise StateDeltaError("state delta replace target is missing")
                parent[token] = deepcopy(operation["value"])
            else:
                if not exists:
                    raise StateDeltaError("state delta remove target is missing")
                del parent[token]
            continue
        if not isinstance(parent, list):
            raise StateDeltaError("state delta parent is not a container")
        if token == "-":
            if op != "add":
                raise StateDeltaError("array '-' path only supports add")
            parent.append(deepcopy(operation["value"]))
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise StateDeltaError("state delta array index is invalid") from exc
        if index < 0 or index >= len(parent):
            raise StateDeltaError("state delta array index is out of range")
        if op == "add":
            parent.insert(index, deepcopy(operation["value"]))
        elif op == "replace":
            parent[index] = deepcopy(operation["value"])
        else:
            del parent[index]
    return owned


def _resolve_parent(value: Any, path: str) -> tuple[Any, str]:
    if not path.startswith("/"):
        raise StateDeltaError("state delta path is invalid")
    tokens = [_unescape(token) for token in path[1:].split("/")]
    current = value
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise StateDeltaError("state delta path is missing")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise StateDeltaError("state delta array index is invalid") from exc
            if index < 0 or index >= len(current):
                raise StateDeltaError("state delta array index is out of range")
            current = current[index]
        else:
            raise StateDeltaError("state delta path crosses a scalar")
    return current, tokens[-1]


def _unescape(token: str) -> str:
    output = ""
    index = 0
    while index < len(token):
        if token[index] != "~":
            output += token[index]
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise StateDeltaError("state delta JSON pointer escape is invalid")
        output += "~" if token[index + 1] == "0" else "/"
        index += 2
    return output


__all__ = [
    "StateDeltaError",
    "apply_state_delta",
    "build_state_delta",
    "state_digest",
]
