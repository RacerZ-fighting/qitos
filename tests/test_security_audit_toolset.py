from __future__ import annotations

from pathlib import Path

from qitos.kit.tool.experimental.security_research import SecurityAuditToolSet


def _write_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        """
from flask import Flask, request, redirect
import subprocess
import requests
import pickle

app = Flask(__name__)
DEBUG = True
SECRET_KEY = "prod-secret-value-123456"

@app.route("/run", methods=["POST"])
def run():
    cmd = request.args.get("cmd")
    subprocess.run(cmd, shell=True)
    data = requests.get(request.args.get("target"), verify=False)
    payload = pickle.loads(request.data)
    return redirect(request.args.get("next"))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "cli.py").write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config")

if __name__ == "__main__":
    parser.parse_args()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "notes.py").write_text(
        "# TODO: harden auth flow\n# nosec: temporary suppression for legacy code\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text(
        "flask\nrequests==2.31.0\n", encoding="utf-8"
    )
    (root / "package.json").write_text(
        '{"dependencies":{"express":"latest","left-pad":"file:../left-pad"},"devDependencies":{"vite":"*"}}',
        encoding="utf-8",
    )
    (root / "docker-compose.yml").write_text(
        "services:\n  web:\n    privileged: true\n    environment:\n      - DEBUG=true\n",
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "API_KEY=changeme\nREAL_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "deploy.yml").write_text(
        "jobs:\n  deploy:\n    steps:\n      - run: echo deploying\n",
        encoding="utf-8",
    )


def test_security_audit_inventory_and_entrypoints(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    toolset = SecurityAuditToolSet(workspace_root=str(tmp_path))

    inventory = toolset.audit_inventory()
    assert inventory["status"] == "success"
    assert "python" in inventory["data"]["languages"]
    assert any(
        item["path"] == "requirements.txt" for item in inventory["data"]["manifests"]
    )
    assert any(
        item["kind"] == "http_route"
        for item in inventory["data"]["entrypoint_candidates"]
    )

    entrypoints = toolset.audit_entrypoints()
    assert entrypoints["status"] == "success"
    kinds = {item["kind"] for item in entrypoints["data"]["entrypoints"]}
    assert "http_route" in kinds
    assert "cli_entrypoint" in kinds


def test_security_audit_sink_secret_config_and_notes(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    toolset = SecurityAuditToolSet(workspace_root=str(tmp_path))

    sinks = toolset.audit_sink_scan(category="all")
    categories = {item["category"] for item in sinks["data"]["findings"]}
    assert "command_exec" in categories
    assert "deserialization" in categories
    assert "redirect_header" in categories

    secrets = toolset.audit_secret_scan()
    assert any(item["title"] == "GitHub token" for item in secrets["data"]["findings"])
    assert not any(
        "changeme" in item["evidence"].lower() for item in secrets["data"]["findings"]
    )

    configs = toolset.audit_config_scan()
    config_categories = {item["category"] for item in configs["data"]["findings"]}
    assert "debug_config" in config_categories
    assert "tls_verification" in config_categories
    assert "container_privilege" in config_categories

    notes = toolset.audit_notes_scan()
    assert notes["data"]["count"] >= 2


def test_security_dependency_inventory_and_hotspots(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    toolset = SecurityAuditToolSet(workspace_root=str(tmp_path))

    inventory = toolset.audit_dependency_inventory()
    manifests = {item["path"]: item for item in inventory["data"]["manifests"]}
    assert manifests["requirements.txt"]["direct_dependency_count"] == 2
    assert manifests["package.json"]["git_or_path_dependencies"] == [
        "left-pad:file:../left-pad"
    ]
    assert manifests["package.json"]["outdated_clues"]

    toolset.audit_sink_scan(category="all")
    toolset.audit_secret_scan()
    toolset.audit_config_scan()
    hotspots = toolset.audit_hotspots()
    assert hotspots["status"] == "success"
    assert hotspots["data"]["hotspots"]
    assert hotspots["data"]["summary"]["finding_count"] >= 3


def test_security_dependency_audit_handles_unavailable_and_external_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    _write_repo(tmp_path)
    toolset = SecurityAuditToolSet(workspace_root=str(tmp_path), include_external=True)

    monkeypatch.setattr("shutil.which", lambda name: None)
    unavailable = toolset.audit_dependency_audit()
    assert unavailable["status"] == "error"
    assert unavailable["error_category"] == "unavailable"

    def _which(name: str):
        return f"/usr/bin/{name}"

    def _run_command(cmd, cwd=None):
        _ = cwd
        if cmd[0] == "pip-audit":
            return {
                "status": "success",
                "stdout": '{"dependencies":[{"name":"flask","version":"2.0.0","vulns":[{"id":"PYSEC-123"}]}]}',
                "stderr": "",
                "returncode": 1,
            }
        if cmd[0] == "npm":
            return {
                "status": "success",
                "stdout": '{"vulnerabilities":{"lodash":{"severity":"high","via":[{"title":"Prototype pollution"}]}}}',
                "stderr": "",
                "returncode": 1,
            }
        return {
            "status": "success",
            "stdout": '{"results":[{"source":{"path":"package.json"},"packages":[{"package":{"name":"axios"},"vulnerabilities":[{"id":"OSV-1"}]}]}]}',
            "stderr": "",
            "returncode": 1,
        }

    monkeypatch.setattr("shutil.which", _which)
    monkeypatch.setattr(toolset, "_run_command", _run_command)
    audited = toolset.audit_dependency_audit()
    assert audited["status"] == "success"
    assert len(audited["data"]["findings"]) >= 3
    assert any(
        item["category"] == "vulnerable_dependency"
        for item in audited["data"]["findings"]
    )
