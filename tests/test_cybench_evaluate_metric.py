from __future__ import annotations

from qitos.kit.metric import (
    CyBenchGuidedFinalScoreMetric,
    CyBenchGuidedSubtaskScoreMetric,
    CyBenchPartialMatchRateMetric,
    CyBenchUnguidedSuccessMetric,
)
from qitos.metric import MetricInput, MetricRegistry


def test_cybench_metrics():
    rows = [
        MetricInput(
            task_id="t1",
            payload={
                "unguided_success": True,
                "guided_subtask_score": 1.0,
                "guided_final_score": 1.0,
                "partial_matches": [True],
            },
        ),
        MetricInput(
            task_id="t2",
            payload={
                "unguided_success": False,
                "guided_subtask_score": 0.5,
                "guided_final_score": 0.0,
                "partial_matches": [False, True],
            },
        ),
    ]
    reports = {
        r.name: r
        for r in MetricRegistry(
            [
                CyBenchUnguidedSuccessMetric(),
                CyBenchGuidedSubtaskScoreMetric(),
                CyBenchGuidedFinalScoreMetric(),
                CyBenchPartialMatchRateMetric(),
            ]
        ).compute_all(rows)
    }
    assert reports["cybench_unguided_success_rate"].value == 0.5
    assert reports["cybench_guided_subtask_score"].value == 0.75
    assert reports["cybench_guided_final_score"].value == 0.5
    assert abs(reports["cybench_partial_match_rate"].value - (2 / 3)) < 1e-9
