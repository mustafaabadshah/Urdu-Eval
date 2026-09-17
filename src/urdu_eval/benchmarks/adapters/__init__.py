"""External benchmark adapters for public datasets."""

from urdu_eval.benchmarks.adapters.urdu_bench import UrduBenchAdapter
from urdu_eval.benchmarks.adapters.urdu_mmlu import UrduMMLUAdapter

__all__ = ["UrduMMLUAdapter", "UrduBenchAdapter"]
