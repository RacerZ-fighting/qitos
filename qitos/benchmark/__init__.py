"""Benchmark result contracts and file utilities for QitOS.

This package keeps only the engine-free normalized result row plus result-file
read/write/aggregation. Benchmark execution adapters and extension hooks were
removed with the recipe layer; run inspection stays with `qita`.
"""

from .common import (
    build_experiment_spec,
    evaluate_benchmark_results,
    read_benchmark_results,
    write_benchmark_results,
)

__all__ = [
    "build_experiment_spec",
    "write_benchmark_results",
    "read_benchmark_results",
    "evaluate_benchmark_results",
]
