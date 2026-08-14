"""OSWorld benchmark runtime helpers and lifecycle hooks."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib.parse import urlparse

import httpx

from qitos.core import ExperimentSpec, RunSpec, Task
from qitos.core.task import TaskBudget

from ..contracts import BenchmarkRuntimeHook, PreparedBenchmarkTask

OSWORLD_DEFAULT_BOOT_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
OSWORLD_DEFAULT_IMAGE = "happysixd/osworld-docker"
OSWORLD_DEFAULT_READY_TIMEOUT_SEC = 240.0
OSWORLD_FIRST_BOOT_READY_TIMEOUT_SEC = 1800.0
OSWORLD_DEFAULT_PORT_RETRY_ATTEMPTS = 3
OSWORLD_VISUAL_READY_TIMEOUT_SEC = 120.0
OSWORLD_VISUAL_READY_TIMEOUT_NO_KVM_SEC = 900.0
OSWORLD_VISUAL_READY_TIMEOUT_FIRST_BOOT_SEC = 300.0
OSWORLD_VISUAL_READY_MIN_SCREENSHOT_BYTES = 10_000
OSWORLD_VISUAL_READY_POLL_SEC = 2.0
OSWORLD_DEFAULT_PORTS: tuple[int, ...] = (5000, 9222, 8006, 8080)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    text = text.strip("._")
    return text or "unknown"


def _first_existing(paths: Sequence[str | Path]) -> str | None:
    for item in paths:
        token = str(item or "").strip()
        if not token:
            continue
        path = Path(token).expanduser().resolve()
        if path.exists():
            return str(path)
    return None


def _run_setup_step(
    *, step_type: str, parameters: Mapping[str, Any], endpoint: str
) -> dict[str, Any]:
    step = str(step_type or "").strip().lower()
    params = dict(parameters or {})
    if step == "sleep":
        seconds = float(params.get("seconds", 1.0))
        time.sleep(max(0.0, seconds))
        return {
            "status_code": 200,
            "step_type": step,
            "slept_seconds": seconds,
            "ok": True,
        }

    path_overrides = {
        "open": "/setup/open_file",
        "execute": "/setup/execute",
        "execute_with_verification": "/setup/execute_with_verification",
        "launch": "/setup/launch",
        "activate_window": "/setup/activate_window",
        "close_window": "/setup/close_window",
        "change_wallpaper": "/setup/change_wallpaper",
        "download": "/setup/download_file",
        "upload_file": "/setup/upload",
    }
    route = path_overrides.get(step, f"/setup/{step}")
    url = f"{endpoint.rstrip('/')}{route}"
    resp = httpx.post(url, json=params, timeout=120, follow_redirects=True)
    return {
        "status_code": int(resp.status_code),
        "step_type": step,
        "route": route,
        "ok": bool(resp.status_code == 200),
        "body": (resp.text or "")[:600],
    }


def run_setup_config(
    *, endpoint: str, setup_config: Sequence[Any]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, item in enumerate(setup_config, start=1):
        if not isinstance(item, Mapping):
            events.append(
                {"index": index, "ok": False, "error": f"invalid setup item: {item!r}"}
            )
            continue
        step_type = str(item.get("type") or "").strip()
        params = item.get("parameters")
        if not step_type or not isinstance(params, Mapping):
            events.append(
                {
                    "index": index,
                    "ok": False,
                    "error": f"invalid setup schema: {item!r}",
                }
            )
            continue
        out = _run_setup_step(
            step_type=step_type, parameters=dict(params), endpoint=endpoint
        )
        out["index"] = index
        events.append(out)
        if not out.get("ok", False):
            break
    return events


@dataclass
class OSWorldPrepareResult:
    env_config: dict[str, Any]
    metadata: dict[str, Any]


class OSWorldContainerLauncher:
    """Benchmark-layer bootstrap helper for OSWorld sessions."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root or _repo_root()).resolve()
        self._emit = emit if callable(emit) else None
        self._settings = dict(settings or {})

    def _setting(self, key: str, default: Any = None) -> Any:
        if key in self._settings:
            return self._settings.get(key)
        return default

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self._emit is None:
            return
        try:
            self._emit(dict(event))
        except Exception:
            return

    def _vm_cache_dir(self) -> Path:
        raw = str(self._setting("vm_cache_dir", "")).strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return self._repo_root / "references" / "OSWorld" / "docker_vm_data"

    def _vm_name_from_url(self, url: str) -> str:
        name = Path(urlparse(url).path).name or "Ubuntu.qcow2.zip"
        if name.endswith(".zip"):
            name = name[:-4]
        if not name.lower().endswith(".qcow2"):
            name = f"{name}.qcow2"
        return name

    def _download_to_file(self, *, url: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(dst.suffix + ".part")
        if part.exists():
            part.unlink()
        with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        part.replace(dst)

    def _ensure_default_vm(self) -> tuple[Path, bool]:
        cache_dir = self._vm_cache_dir()
        vm_url = str(self._setting("vm_url", OSWORLD_DEFAULT_BOOT_URL)).strip()
        vm_name = self._vm_name_from_url(vm_url)
        vm_path = cache_dir / vm_name
        if vm_path.exists():
            return vm_path, False
        artifact_name = Path(urlparse(vm_url).path).name or f"{vm_name}.zip"
        artifact_path = cache_dir / artifact_name
        if not artifact_path.exists():
            self._download_to_file(url=vm_url, dst=artifact_path)
        if artifact_path.suffix.lower() == ".zip":
            part = vm_path.with_suffix(vm_path.suffix + ".part")
            with zipfile.ZipFile(artifact_path, "r") as zf:
                members = [
                    x for x in zf.infolist() if x.filename.lower().endswith(".qcow2")
                ]
                if not members:
                    raise RuntimeError(
                        f"No qcow2 image found in archive: {artifact_path}"
                    )
                member = members[0]
                with zf.open(member, "r") as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            part.replace(vm_path)
        else:
            artifact_path.replace(vm_path)
        return vm_path, True

    def _probe_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except Exception:
            return False

    def _wait_for_endpoint(self, endpoint: str, *, timeout_sec: float) -> bool:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 80)
        started = time.monotonic()
        while (time.monotonic() - started) <= float(timeout_sec):
            if self._probe_port(host, port):
                return True
            time.sleep(1.0)
        return False

    def _wait_for_visual_ready(self, endpoint: str, *, timeout_sec: float) -> bool:
        started = time.monotonic()
        while (time.monotonic() - started) <= float(timeout_sec):
            try:
                resp = httpx.get(
                    f"{endpoint.rstrip('/')}/screenshot",
                    timeout=20,
                    follow_redirects=True,
                )
                if int(resp.status_code) == 200 and len(resp.content or b"") >= int(
                    OSWORLD_VISUAL_READY_MIN_SCREENSHOT_BYTES
                ):
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(OSWORLD_VISUAL_READY_POLL_SEC)
        return False

    def _resolve_boot_inputs(
        self,
    ) -> tuple[dict[str, str], dict[str, str], bool, dict[str, Any]]:
        explicit_vm_path = str(self._setting("vm_path", "")).strip()
        explicit_boot_url = str(self._setting("boot", "")).strip()
        container_env: dict[str, str] = {}
        volumes: dict[str, str] = {}
        boot_config: dict[str, Any] = {
            "boot_url_set": False,
            "vm_path_set": False,
            "source": "auto",
            "downloaded": False,
        }
        first_boot = False

        if explicit_vm_path:
            vm_file = Path(explicit_vm_path).expanduser().resolve()
            if not vm_file.exists():
                raise RuntimeError(f"osworld.vm_path not found: {vm_file}")
            volumes[str(vm_file)] = "/System.qcow2:ro"
            boot_config.update(
                {"vm_path_set": True, "source": "env_vm_path", "vm_path": str(vm_file)}
            )
        elif explicit_boot_url:
            container_env["BOOT"] = explicit_boot_url
            first_boot = True
            boot_config.update({"boot_url_set": True, "source": "env_boot_url"})
        else:
            vm_file, downloaded = self._ensure_default_vm()
            volumes[str(vm_file)] = "/System.qcow2:ro"
            first_boot = bool(downloaded)
            boot_config.update(
                {
                    "vm_path_set": True,
                    "source": "auto_cached_vm",
                    "vm_path": str(vm_file),
                    "downloaded": bool(downloaded),
                }
            )

        for setting_key, mapped in (
            ("disk_size", "DISK_SIZE"),
            ("ram_size", "RAM_SIZE"),
            ("cpu_cores", "CPU_CORES"),
            ("kvm", "KVM"),
        ):
            value = str(self._setting(setting_key, "")).strip()
            if value:
                container_env[mapped] = value

        defaults = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}
        for key, value in defaults.items():
            container_env.setdefault(key, value)

        return container_env, volumes, first_boot, boot_config

    def _pick_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _resolve_ports(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        explicit = {
            5000: self._setting("server_port"),
            9222: self._setting("chromium_port"),
            8006: self._setting("vnc_port"),
            8080: self._setting("vlc_port"),
        }
        attempts = max(
            1,
            int(
                self._setting(
                    "port_retry_attempts", OSWORLD_DEFAULT_PORT_RETRY_ATTEMPTS
                )
            ),
        )
        for port in OSWORLD_DEFAULT_PORTS:
            raw = explicit.get(port)
            if raw:
                mapping[port] = int(raw)
                continue
            selected = port
            for _ in range(attempts):
                if not self._probe_port("127.0.0.1", selected):
                    break
                selected = self._pick_free_port()
            mapping[port] = int(selected)
        return mapping

    def _launch_container(
        self,
        *,
        task: Task,
        task_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        image = (
            str(self._setting("image", OSWORLD_DEFAULT_IMAGE)).strip()
            or OSWORLD_DEFAULT_IMAGE
        )
        container_env, volumes, first_boot, boot_config = self._resolve_boot_inputs()
        ports = self._resolve_ports()
        task_token = _safe_name(str(task.id))
        container_name = (
            str(self._setting("container", "")).strip() or f"qitos-osworld-{task_token}"
        )
        command = ["docker", "run", "-d", "--rm", "--name", container_name]
        for host_port, container_port in sorted(
            (value, key) for key, value in ports.items()
        ):
            command.extend(["-p", f"{host_port}:{container_port}"])
        if not (
            os.path.exists("/dev/kvm")
            and str(container_env.get("KVM", "")).upper() != "N"
        ):
            container_env["KVM"] = "N"
        for key, value in sorted(container_env.items()):
            command.extend(["-e", f"{key}={value}"])
        for host_path, container_path in sorted(volumes.items()):
            command.extend(["-v", f"{host_path}:{container_path}"])
        cap_add = list(self._setting("cap_add", []) or [])
        for cap in cap_add:
            token = str(cap or "").strip()
            if token:
                command.extend(["--cap-add", token])
        if (
            os.path.exists("/dev/kvm")
            and str(container_env.get("KVM", "")).upper() != "N"
        ):
            command.extend(["--device", "/dev/kvm"])
        command.append(image)

        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"OSWorld container launch failed: {(proc.stderr or proc.stdout or '').strip()}"
            )
        container_id = str(
            (proc.stdout or "").strip().splitlines()[0] or container_name
        )
        controller_endpoint = f"http://127.0.0.1:{int(ports[5000])}"
        ready_timeout = (
            OSWORLD_FIRST_BOOT_READY_TIMEOUT_SEC
            if first_boot
            else float(
                self._setting("ready_timeout_sec", OSWORLD_DEFAULT_READY_TIMEOUT_SEC)
            )
        )
        controller_ready = self._wait_for_endpoint(
            controller_endpoint, timeout_sec=ready_timeout
        )
        visual_timeout = (
            OSWORLD_VISUAL_READY_TIMEOUT_FIRST_BOOT_SEC
            if first_boot
            else (
                OSWORLD_VISUAL_READY_TIMEOUT_SEC
                if os.path.exists("/dev/kvm")
                else OSWORLD_VISUAL_READY_TIMEOUT_NO_KVM_SEC
            )
        )
        visual_ready = controller_ready and self._wait_for_visual_ready(
            controller_endpoint,
            timeout_sec=float(
                self._setting("visual_ready_timeout_sec", visual_timeout)
            ),
        )
        return {
            "container": container_name,
            "container_id": container_id,
            "controller_endpoint": controller_endpoint,
            "controller_ready": bool(controller_ready),
            "visual_ready": bool(visual_ready),
            "ports": {str(key): int(value) for key, value in ports.items()},
            "boot": boot_config,
            "first_boot": bool(first_boot),
            "image": image,
            "launched_container": True,
            "runtime_container": dict(task_metadata.get("runtime_container") or {}),
            "docker_command": command,
        }

    def prepare(
        self, *, task: Task, task_metadata: Mapping[str, Any]
    ) -> OSWorldPrepareResult:
        runtime_container = dict(task_metadata.get("runtime_container") or {})
        startup = dict(runtime_container.get("startup") or {})
        global_mock_mode = _truthy(self._setting("mock_mode"), default=False)
        screenshot_candidates = [
            startup.get("screenshot_path"),
            (task.env_spec.config if task.env_spec else {}).get("screenshot_path"),
            self._setting("screenshot_path", ""),
        ]
        screenshot_path = _first_existing(screenshot_candidates)
        if screenshot_path is None:
            screenshot_path = str(
                (
                    self._repo_root
                    / ".qitos"
                    / "osworld_screenshots"
                    / f"{_safe_name(task.id)}.png"
                ).resolve()
            )
        controller_endpoint = str(
            startup.get("controller_endpoint")
            or self._setting("controller_endpoint", "")
        ).strip()
        container_name = str(
            startup.get("container") or self._setting("container", "")
        ).strip()
        mock_mode = _truthy(startup.get("mock_mode"), default=global_mock_mode)
        mode = "mock" if mock_mode else "container"
        launch_metadata: dict[str, Any] = {
            "container": container_name,
            "controller_endpoint": controller_endpoint,
            "controller_ready": False,
            "visual_ready": False,
            "launched_container": False,
            "ports": {},
        }

        boot_config: dict[str, Any] = {
            "source": "explicit_settings",
            "downloaded": False,
        }
        if not mock_mode:
            launch_settings = {
                **self._settings,
                **startup,
            }
            launcher = OSWorldContainerLauncher(
                repo_root=self._repo_root,
                emit=self._emit,
                settings=launch_settings,
            )
            if controller_endpoint:
                ready_timeout = float(
                    startup.get("ready_timeout_sec")
                    or OSWORLD_DEFAULT_READY_TIMEOUT_SEC
                )
                launch_metadata["controller_ready"] = launcher._wait_for_endpoint(
                    controller_endpoint,
                    timeout_sec=ready_timeout,
                )
                launch_metadata["visual_ready"] = launcher._wait_for_visual_ready(
                    controller_endpoint,
                    timeout_sec=float(
                        startup.get("visual_ready_timeout_sec")
                        or OSWORLD_VISUAL_READY_TIMEOUT_SEC
                    ),
                )
            else:
                launch_metadata = launcher._launch_container(
                    task=task, task_metadata=task_metadata
                )
                controller_endpoint = str(
                    launch_metadata.get("controller_endpoint") or ""
                )
                container_name = str(launch_metadata.get("container") or "")
                boot_config = dict(launch_metadata.get("boot") or boot_config)

        env_config = dict((task.env_spec.config if task.env_spec else {}) or {})
        env_config.update(
            {
                "provider": (
                    "container" if mode == "container" and container_name else "mock"
                ),
                "screenshot_path": screenshot_path,
                "instruction": task.objective,
                "container": container_name,
                "controller_endpoint": controller_endpoint,
                "workspace_root": str(
                    startup.get("workspace_root")
                    or env_config.get("workspace_root")
                    or "/workspace"
                ),
                "metadata": {
                    **dict(env_config.get("metadata") or {}),
                    "benchmark": "osworld",
                    "controller_endpoint": controller_endpoint,
                    "controller_ready": bool(
                        launch_metadata.get("controller_ready", False)
                    ),
                    "visual_ready": bool(launch_metadata.get("visual_ready", False)),
                    "ports": dict(launch_metadata.get("ports") or {}),
                },
            }
        )
        if mode == "container" and controller_endpoint:
            env_config["provider"] = "container"
        return OSWorldPrepareResult(
            env_config=env_config,
            metadata={
                "container": container_name,
                "controller_endpoint": controller_endpoint,
                "controller_ready": bool(
                    launch_metadata.get("controller_ready", False)
                ),
                "visual_ready": bool(launch_metadata.get("visual_ready", False)),
                "launched_container": bool(
                    launch_metadata.get("launched_container", False)
                ),
                "container_id": str(launch_metadata.get("container_id") or ""),
                "boot": boot_config,
                "mode": mode,
                "ports": dict(launch_metadata.get("ports") or {}),
                "screenshot_path": screenshot_path,
            },
        )


class OSWorldRuntimeHook(BenchmarkRuntimeHook):
    """Prepare and finalize OSWorld runs without leaking benchmark logic into DesktopEnv."""

    def __init__(
        self,
        *,
        repo_root: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = (
            Path(repo_root).expanduser().resolve() if repo_root else _repo_root()
        )
        self.settings = dict(settings or {})

    def prepare(
        self, *, task: Task, run_spec: RunSpec, experiment_spec: ExperimentSpec
    ) -> PreparedBenchmarkTask:
        launcher = OSWorldContainerLauncher(
            repo_root=self.repo_root, settings=self.settings
        )
        launch = launcher.prepare(task=task, task_metadata=task.metadata)
        prepared_task = Task.from_dict(task.to_dict())
        if prepared_task.env_spec is None:
            prepared_task.env_spec = task.env_spec
        if prepared_task.env_spec is None:
            raise RuntimeError("OSWorld task is missing env_spec.")
        prepared_task.env_spec.config = launch.env_config
        prepared_task.budget = TaskBudget(
            max_steps=int(
                (prepared_task.budget.max_steps or 0)
                or int((self.settings.get("max_steps") or 15))
            )
        )
        runtime_metadata = {
            "benchmark": "osworld",
            "runtime_prepare": launch.metadata,
            "run_spec_fingerprint": run_spec.fingerprint(),
            "experiment_name": experiment_spec.name,
            "sample_identity": {
                "task_id": prepared_task.id,
                "domain": (prepared_task.metadata or {}).get("domain"),
                "example_id": (prepared_task.metadata or {}).get("example_id"),
            },
        }
        setup_config = list((prepared_task.metadata or {}).get("config") or [])
        controller_endpoint = str(launch.metadata.get("controller_endpoint") or "")
        if controller_endpoint and setup_config:
            runtime_metadata["setup_events"] = run_setup_config(
                endpoint=controller_endpoint,
                setup_config=setup_config,
            )
        return PreparedBenchmarkTask(
            task=prepared_task, runtime_metadata=runtime_metadata
        )

    def finalize(
        self,
        *,
        prepared: PreparedBenchmarkTask,
        run_spec: RunSpec,
        experiment_spec: ExperimentSpec,
        execution: Any = None,
        error: Exception | None = None,
    ) -> Dict[str, Any]:
        _ = (run_spec, experiment_spec, execution)
        cleanup_policy = str(
            ((prepared.task.metadata or {}).get("runtime_container") or {}).get(
                "cleanup_policy", "destroy_on_release"
            )
        )
        finalize: Dict[str, Any] = {
            "cleanup_policy": cleanup_policy,
            "error": str(error) if error is not None else None,
            "finalized": True,
        }
        runtime_prepare = dict(
            (prepared.runtime_metadata or {}).get("runtime_prepare") or {}
        )
        if cleanup_policy == "destroy_on_release" and runtime_prepare.get(
            "launched_container"
        ):
            container_name = str(runtime_prepare.get("container") or "").strip()
            if container_name:
                proc = subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                finalize["cleanup"] = {
                    "container": container_name,
                    "exit_code": int(proc.returncode),
                    "stdout": str(proc.stdout or "").strip(),
                    "stderr": str(proc.stderr or "").strip(),
                }
        return finalize


__all__ = [
    "OSWORLD_DEFAULT_BOOT_URL",
    "OSWORLD_DEFAULT_PORTS",
    "OSWorldContainerLauncher",
    "OSWorldPrepareResult",
    "OSWorldRuntimeHook",
    "run_setup_config",
]
