from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from qitos.benchmark import read_benchmark_results
from qitos.benchmark.osworld import (
    OSWorldBenchmarkAdapter,
    OSWorldContainerLauncher,
    OSWorldRuntimeHook,
    evaluate_task,
    run_setup_config,
)
from qitos.cli import main as qit_main
from qitos.core import ExperimentSpec, RunSpec
from qitos.kit.env import DesktopEnv


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn2gbcAAAAASUVORK5CYII="
    )
    path.write_bytes(payload)


def _write_osworld_dataset(root: Path, screenshot_path: Path) -> Path:
    dataset = root / "evaluation_examples"
    example_dir = dataset / "examples" / "browser"
    example_dir.mkdir(parents=True, exist_ok=True)
    (dataset / "test_all.json").write_text(
        json.dumps({"browser": ["001"]}, ensure_ascii=False), encoding="utf-8"
    )
    example = {
        "id": "001",
        "instruction": "Click the Continue button in the desktop starter task.",
        "split": "test",
        "related_apps": ["browser"],
        "config": [],
        "trajectory": [],
        "osworld_settings": {
            "mock_mode": True,
            "screenshot_path": str(screenshot_path),
        },
    }
    (example_dir / "001.json").write_text(
        json.dumps(example, ensure_ascii=False), encoding="utf-8"
    )
    return dataset


def _write_reference_root(root: Path) -> Path:
    metrics_dir = root / "desktop_env" / "evaluators" / "metrics"
    getters_dir = root / "desktop_env" / "evaluators" / "getters"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    getters_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "test_metric.py").write_text(
        "def test_metric(value, context):\n    return float(value)\n",
        encoding="utf-8",
    )
    (getters_dir / "dummy.py").write_text(
        "def get_dummy(context):\n    return 1.0\n",
        encoding="utf-8",
    )
    return root


def test_osworld_adapter_loads_temp_dataset(tmp_path: Path) -> None:
    screenshot = tmp_path / "desktop.png"
    _write_tiny_png(screenshot)
    dataset = _write_osworld_dataset(tmp_path, screenshot)
    adapter = OSWorldBenchmarkAdapter(dataset_path=str(dataset))
    rows = adapter.load_records(split="test")
    tasks = adapter.to_tasks(rows, split="test")
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "osworld-browser-001"
    assert task.metadata["benchmark"] == "osworld"
    assert task.metadata["osworld_settings"]["mock_mode"] is True


def test_osworld_runtime_hook_prepares_mock_env(tmp_path: Path) -> None:
    screenshot = tmp_path / "desktop.png"
    _write_tiny_png(screenshot)
    dataset = _write_osworld_dataset(tmp_path, screenshot)
    task = OSWorldBenchmarkAdapter(dataset_path=str(dataset)).to_tasks(
        OSWorldBenchmarkAdapter(dataset_path=str(dataset)).load_records(split="test"),
        split="test",
    )[0]
    prepared = OSWorldRuntimeHook(settings={"mock_mode": True}).prepare(
        task=task,
        run_spec=RunSpec(benchmark_name="osworld", benchmark_split="test"),
        experiment_spec=ExperimentSpec(
            name="osworld:test", benchmark_name="osworld", benchmark_split="test"
        ),
    )
    assert prepared.task.env_spec is not None
    assert prepared.task.env_spec.config["provider"] == "mock"
    assert prepared.runtime_metadata["benchmark"] == "osworld"


def test_osworld_setup_and_eval_bridges(tmp_path: Path) -> None:
    reference_root = _write_reference_root(tmp_path / "references" / "OSWorld")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            _ = (format, args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        setup = run_setup_config(
            endpoint=endpoint,
            setup_config=[{"type": "execute", "parameters": {"command": "echo hi"}}],
        )
        assert setup and setup[0]["ok"] is True
        evaluation = evaluate_task(
            endpoint=endpoint,
            evaluator={"func": "test_metric", "result_getter": "dummy"},
            action_history=[],
            sample_id="001",
            task_id="osworld-browser-001",
            reference_root=reference_root,
        )
        assert evaluation["score"] == 1.0
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_osworld_runtime_and_desktop_env_use_external_controller(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "desktop.png"
    _write_tiny_png(screenshot)
    dataset = _write_osworld_dataset(tmp_path, screenshot)
    recorded_actions: list[dict[str, object]] = []
    screenshot_bytes = b"\x89PNG" + b"x" * 12032

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/screenshot":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(screenshot_bytes)
                return
            if self.path == "/accessibility":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"AT": {"role": "window", "name": "OSWorld"}}).encode(
                        "utf-8"
                    )
                )
                return
            if self.path == "/terminal":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"output": "$ echo osworld\nosworld\n"}).encode("utf-8")
                )
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if self.path == "/execute":
                recorded_actions.append(dict(payload))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            _ = (format, args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        adapter = OSWorldBenchmarkAdapter(dataset_path=str(dataset))
        task = adapter.to_tasks(adapter.load_records(split="test"), split="test")[0]
        task.metadata["osworld_settings"] = {
            "mock_mode": False,
            "controller_endpoint": endpoint,
            "screenshot_path": str(screenshot),
            "visual_ready_timeout_sec": 1,
        }
        task.metadata["runtime_container"]["startup"] = dict(
            task.metadata["osworld_settings"]
        )
        prepared = OSWorldRuntimeHook().prepare(
            task=task,
            run_spec=RunSpec(benchmark_name="osworld", benchmark_split="test"),
            experiment_spec=ExperimentSpec(
                name="osworld:test", benchmark_name="osworld", benchmark_split="test"
            ),
        )
        assert prepared.task.env_spec is not None
        cfg = prepared.task.env_spec.config
        env = DesktopEnv.from_container(
            container=str(cfg.get("container") or ""),
            screenshot_path=str(cfg["screenshot_path"]),
            workspace_root=str(cfg.get("workspace_root") or "/workspace"),
            controller_endpoint=str(cfg.get("controller_endpoint") or ""),
            metadata=dict(cfg.get("metadata") or {}),
        )
        env.setup()
        observation = env.reset()
        assert observation.data["desktop"]["provider"] == "container_desktop"
        assert Path(str(cfg["screenshot_path"])).exists()
        step = env.step(
            {
                "decision_mode": "act",
                "actions": [{"name": "click", "args": {"x": 320, "y": 240}}],
            }
        )
        assert step.info["performed_actions"][0]["execution_state"] == "executed"
        assert recorded_actions
        assert recorded_actions[0]["action_type"] == "CLICK"
        env.teardown()
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_osworld_launcher_streams_redirected_vm_download(tmp_path: Path) -> None:
    payload = b"vm-image" * 700_000

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/image.bin")
                self.end_headers()
                return
            if self.path == "/image.bin":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            _ = (format, args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        launcher = OSWorldContainerLauncher(
            repo_root=tmp_path,
            settings={
                "vm_cache_dir": str(tmp_path / "cache"),
                "vm_url": f"{endpoint}/redirect",
            },
        )
        vm_path, downloaded = launcher._ensure_default_vm()

        assert downloaded is True
        assert vm_path.read_bytes() == payload
        assert not vm_path.with_suffix(vm_path.suffix + ".part").exists()
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_osworld_benchmark_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    screenshot = tmp_path / "desktop.png"
    _write_tiny_png(screenshot)
    dataset = _write_osworld_dataset(tmp_path, screenshot)
    output = tmp_path / "osworld.jsonl"
    monkeypatch.chdir(tmp_path)
    rc = qit_main(
        [
            "bench",
            "run",
            "--benchmark",
            "osworld",
            "--split",
            "test",
            "--root",
            str(dataset),
            "--strategy",
            "osworld_smoke",
            "--output",
            str(output),
            "--trace-logdir",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 0
    rows = read_benchmark_results(output)
    assert rows
    assert rows[0].benchmark == "osworld"
    assert "benchmark_runtime" in rows[0].metadata
