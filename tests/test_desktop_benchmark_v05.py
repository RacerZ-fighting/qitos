from __future__ import annotations

import json
from pathlib import Path

from qitos.benchmark import evaluate_benchmark_results, read_benchmark_results
from qitos.cli import main as qit_main


def test_desktop_benchmark_eval_json(tmp_path: Path, capsys) -> None:
    row = {
        "task_id": "desktop_continue_button",
        "benchmark": "desktop-starter",
        "split": "starter",
        "prediction": "Clicked Continue and completed the desktop starter workflow.",
        "success": True,
        "stop_reason": "final",
        "steps": 2,
        "latency_seconds": 0.2,
        "token_usage": 123,
        "cost": 0.0,
        "trace_run_dir": None,
        "run_spec_ref": "abc",
        "metadata": {"failure_tags": []},
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rc = qit_main(["bench", "eval", "--input", str(path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"benchmark": "desktop-starter"' in out
    assert '"failure_tag_distribution"' in out
