"""Benchmarks and adapters for UrduEval."""

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.benchmarks.custom import CustomBenchmark
from urdu_eval.benchmarks.registry import (
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)

__all__ = [
    "Benchmark",
    "CustomBenchmark",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
]
