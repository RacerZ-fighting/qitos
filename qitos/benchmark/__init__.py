"""Benchmark result contracts and file utilities for QitOS.

This package keeps only the engine-free core: the adapter/scorer protocols,
the normalized benchmark result row, and result-file read/write/aggregation.
Benchmark execution adapters were removed with the recipe layer; run
inspection stays with `qita`.
"""

from .base import BenchmarkAdapter, BenchmarkSource
from .common import (
    build_experiment_spec,
    evaluate_benchmark_results,
    read_benchmark_results,
    write_benchmark_results,
)
from .contracts import (
    BenchmarkEvaluator,
    BenchmarkRuntimeHook,
    BenchmarkScorer,
    PreparedBenchmarkTask,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkSource",
    "BenchmarkRuntimeHook",
    "BenchmarkEvaluator",
    "BenchmarkScorer",
    "PreparedBenchmarkTask",
    "build_experiment_spec",
    "write_benchmark_results",
    "read_benchmark_results",
    "evaluate_benchmark_results",
]
