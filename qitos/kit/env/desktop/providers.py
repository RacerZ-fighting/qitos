"""Desktop environment providers inspired by OSWorld, with a container-first default."""

from __future__ import annotations

import time
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import httpx

from qitos.core.multimodal import guess_mime_type

from .actions import action_result_payload, normalize_gui_action, to_osworld_action


class DesktopProvider(ABC):
    """Abstract desktop runtime provider."""

    name: str = "desktop_provider"
    version: str = "0.1"

    @abstractmethod
    def start(self) -> None:
        """Start or attach to the runtime."""

    @abstractmethod
    def reset(self, task: Any = None, workspace: Optional[str] = None) -> None:
        """Reset provider state for one task run."""

    @abstractmethod
    def stop(self) -> None:
        """Stop or release provider resources."""

    @abstractmethod
    def capture_state(self) -> Dict[str, Any]:
        """Return screenshot/a11y/terminal/instruction-like state."""

    @abstractmethod
    def execute_action(
        self, action: Mapping[str, Any], state: Any = None
    ) -> Dict[str, Any]:
        """Execute one normalized GUI action."""

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "provider": self.name, "version": self.version}


class MockDesktopProvider(DesktopProvider):
    """Deterministic in-memory desktop provider for smoke tests and local examples."""

    name = "mock_desktop"
    version = "0.1"

    def __init__(
        self,
        *,
        screenshot_path: str,
        instruction: str = "",
        accessibility_tree: Any = None,
        terminal: str = "",
        dom: Any = None,
        ocr: Optional[List[Dict[str, Any]]] = None,
        ui_candidates: Optional[List[Dict[str, Any]]] = None,
        screen_size: tuple[int, int] = (1920, 1080),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.screenshot_path = str(Path(screenshot_path).expanduser().resolve())
        self.instruction = str(instruction or "")
        self.accessibility_tree = accessibility_tree
        self.terminal = str(terminal or "")
        self.dom = dom
        self.ocr = list(ocr or [])
        self.ui_candidates = list(ui_candidates or [])
        self.screen_size = tuple(screen_size)
        self.metadata = dict(metadata or {})
        self.actions: List[Dict[str, Any]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def reset(self, task: Any = None, workspace: Optional[str] = None) -> None:
        _ = task
        _ = workspace
        self.started = True
        self.actions = []

    def stop(self) -> None:
        self.started = False

    def capture_state(self) -> Dict[str, Any]:
        return {
            "screenshot": {
                "path": self.screenshot_path,
                "mime_type": guess_mime_type(self.screenshot_path),
                "detail": "original",
            },
            "accessibility_tree": self.accessibility_tree,
            "terminal": self.terminal,
            "dom": self.dom,
            "ocr": list(self.ocr),
            "ui_candidates": list(self.ui_candidates),
            "instruction": self.instruction,
            "screen_size": {
                "width": int(self.screen_size[0]),
                "height": int(self.screen_size[1]),
            },
            "metadata": dict(self.metadata),
            "action_history": list(self.actions),
        }

    def execute_action(
        self, action: Mapping[str, Any], state: Any = None
    ) -> Dict[str, Any]:
        _ = state
        normalized = normalize_gui_action(action)
        self.actions.append(normalized)
        message = f"Executed {normalized['action_type']} in mock desktop runtime."
        return action_result_payload(
            action=normalized,
            status="success",
            execution_state="executed",
            message=message,
            provider=self.name,
            metadata={"screen_size": list(self.screen_size)},
        )


class ContainerDesktopProvider(DesktopProvider):
    """Container-first desktop provider with deterministic screenshot observation.

    This provider is intentionally lightweight in v0.5 phase 1: it verifies a Docker
    container, captures screenshot-backed state from configured assets, and records
    normalized GUI actions for a harness/controller to execute. It keeps the provider
    boundary aligned with OSWorld's desktop_env without hard-wiring provider-native
    model protocols into QitOS.
    """

    name = "container_desktop"
    version = "0.1"

    def __init__(
        self,
        *,
        container: str,
        screenshot_path: str,
        workspace_root: str = "/workspace",
        controller_endpoint: str = "",
        instruction: str = "",
        accessibility_tree: Any = None,
        terminal: str = "",
        dom: Any = None,
        ocr: Optional[List[Dict[str, Any]]] = None,
        ui_candidates: Optional[List[Dict[str, Any]]] = None,
        screen_size: tuple[int, int] = (1920, 1080),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.container = str(container or "").strip()
        self.workspace_root = str(workspace_root or "/workspace")
        self.screenshot_path = str(Path(screenshot_path).expanduser().resolve())
        self.controller_endpoint = str(controller_endpoint or "").strip()
        self.instruction = str(instruction or "")
        self.accessibility_tree = accessibility_tree
        self.terminal = str(terminal or "")
        self.dom = dom
        self.ocr = list(ocr or [])
        self.ui_candidates = list(ui_candidates or [])
        self.screen_size = tuple(screen_size)
        self.metadata = dict(metadata or {})
        self.actions: List[Dict[str, Any]] = []
        self.started = False

    def start(self) -> None:
        self._ensure_runtime_available()
        self.started = True

    def reset(self, task: Any = None, workspace: Optional[str] = None) -> None:
        _ = task
        _ = workspace
        self._ensure_runtime_available()
        self.actions = []
        self.started = True

    def stop(self) -> None:
        self.started = False

    def capture_state(self) -> Dict[str, Any]:
        remote_state = self._capture_remote_state()
        return {
            "screenshot": {
                "path": self.screenshot_path,
                "mime_type": guess_mime_type(self.screenshot_path),
                "detail": "original",
            },
            "accessibility_tree": remote_state.get(
                "accessibility_tree", self.accessibility_tree
            ),
            "terminal": str(remote_state.get("terminal") or self.terminal),
            "dom": remote_state.get("dom", self.dom),
            "ocr": list(remote_state.get("ocr") or self.ocr),
            "ui_candidates": list(
                remote_state.get("ui_candidates") or self.ui_candidates
            ),
            "instruction": self.instruction,
            "screen_size": {
                "width": int(self.screen_size[0]),
                "height": int(self.screen_size[1]),
            },
            "metadata": {
                "container": self.container,
                "controller_endpoint": self._effective_controller_endpoint(),
                "workspace_root": self.workspace_root,
                **dict(self.metadata),
            },
            "action_history": list(self.actions),
        }

    def execute_action(
        self, action: Mapping[str, Any], state: Any = None
    ) -> Dict[str, Any]:
        _ = state
        normalized = normalize_gui_action(action)
        self.actions.append(normalized)
        endpoint = self._effective_controller_endpoint()
        if endpoint:
            try:
                resp = httpx.post(
                    f"{endpoint.rstrip('/')}/execute",
                    json=to_osworld_action(normalized),
                    timeout=60,
                    follow_redirects=True,
                )
                success = int(resp.status_code) == 200
                return action_result_payload(
                    action=normalized,
                    status="success" if success else "error",
                    execution_state="executed" if success else "failed",
                    message=(resp.text or "")[:600],
                    provider=self.name,
                    metadata={
                        "container": self.container,
                        "workspace_root": self.workspace_root,
                        "controller_endpoint": endpoint,
                        "status_code": int(resp.status_code),
                    },
                )
            except httpx.HTTPError as exc:
                return action_result_payload(
                    action=normalized,
                    status="error",
                    execution_state="failed",
                    message=str(exc),
                    provider=self.name,
                    metadata={
                        "container": self.container,
                        "workspace_root": self.workspace_root,
                        "controller_endpoint": endpoint,
                    },
                )
        return action_result_payload(
            action=normalized,
            status="accepted",
            execution_state="accepted",
            message=f"Queued {normalized['action_type']} for container desktop runtime.",
            provider=self.name,
            metadata={
                "container": self.container,
                "workspace_root": self.workspace_root,
                "controller_endpoint": endpoint,
            },
        )

    def health_check(self) -> Dict[str, Any]:
        endpoint = self._effective_controller_endpoint()
        if not self.container and not endpoint:
            return {
                "ok": False,
                "message": "container and controller endpoint are both empty",
                "provider": self.name,
            }
        if self.container:
            try:
                proc = subprocess.run(
                    ["docker", "inspect", self.container],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return {"ok": False, "message": str(exc), "provider": self.name}
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "message": "docker inspect failed",
                    "stderr": proc.stderr,
                    "provider": self.name,
                    "container": self.container,
                }
        if endpoint:
            try:
                screenshot = httpx.get(
                    f"{endpoint.rstrip('/')}/screenshot",
                    timeout=15,
                    follow_redirects=True,
                )
                endpoint_ok = bool(screenshot.status_code == 200)
            except httpx.HTTPError as exc:
                return {
                    "ok": False,
                    "message": str(exc),
                    "provider": self.name,
                    "container": self.container,
                    "controller_endpoint": endpoint,
                }
            if not endpoint_ok:
                return {
                    "ok": False,
                    "message": "controller screenshot probe failed",
                    "provider": self.name,
                    "container": self.container,
                    "controller_endpoint": endpoint,
                    "status_code": int(screenshot.status_code),
                }
        return {
            "ok": True,
            "provider": self.name,
            "container": self.container,
            "controller_endpoint": endpoint,
            "workspace_root": self.workspace_root,
        }

    def _ensure_runtime_available(self) -> None:
        health = self.health_check()
        if not bool(health.get("ok", False)):
            raise RuntimeError(
                str(health.get("message", "container desktop provider unavailable"))
            )

    def _effective_controller_endpoint(self) -> str:
        endpoint = self.controller_endpoint
        if endpoint:
            return endpoint
        return str(self.metadata.get("controller_endpoint") or "").strip()

    def _capture_remote_state(self) -> Dict[str, Any]:
        endpoint = self._effective_controller_endpoint()
        if not endpoint:
            return {}

        state: Dict[str, Any] = {}
        screenshot_url = f"{endpoint.rstrip('/')}/screenshot"
        try:
            resp = httpx.get(screenshot_url, timeout=20, follow_redirects=True)
            if resp.status_code == 200 and resp.content:
                target = Path(self.screenshot_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resp.content)
                state["screenshot_bytes"] = len(resp.content)
        except (httpx.HTTPError, OSError) as exc:
            state["screenshot_error"] = str(exc)

        for key, route in (
            ("accessibility_tree", "/accessibility"),
            ("terminal", "/terminal"),
        ):
            try:
                resp = httpx.get(
                    f"{endpoint.rstrip('/')}{route}",
                    timeout=20,
                    follow_redirects=True,
                )
                if resp.status_code != 200:
                    continue
                payload = (
                    resp.json()
                    if "application/json" in str(resp.headers.get("content-type") or "")
                    else {}
                )
                if not isinstance(payload, Mapping):
                    state[key] = payload
                elif key == "accessibility_tree":
                    state[key] = (
                        payload.get("AT")
                        or payload.get("accessibility_tree")
                        or payload
                    )
                else:
                    state[key] = (
                        payload.get("output") or payload.get("terminal") or payload
                    )
            except (httpx.HTTPError, ValueError) as exc:
                state[f"{key}_error"] = str(exc)

        try:
            resp = httpx.get(
                f"{endpoint.rstrip('/')}/observation",
                timeout=20,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                payload = (
                    resp.json()
                    if "application/json" in str(resp.headers.get("content-type") or "")
                    else {}
                )
                if isinstance(payload, Mapping):
                    for key in ("dom", "ocr", "ui_candidates", "screen_size"):
                        if key in payload:
                            state[key] = payload.get(key)
        except (httpx.HTTPError, ValueError) as exc:
            state["observation_error"] = str(exc)

        state["captured_at"] = time.time()
        return state


__all__ = [
    "ContainerDesktopProvider",
    "DesktopProvider",
    "MockDesktopProvider",
]
