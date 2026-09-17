#!/usr/bin/env python3
"""Public compatibility facade for the TypeAR prototype.

The implementation is split by responsibility; imports from this module remain
stable for existing callers, and ``python typear.py`` still launches the demo.
"""

from typear_benchmark import benchmark_prefix_cache
from typear_cli import example_schema, main
from typear_runtime import (
    Choice,
    TypeARClient,
    candidate_softmax,
    run_schema,
    run_sequential_decisions,
)
from typear_schema import (
    SCORE_LEVELS,
    Decision,
    SchemaError,
    compile_json_schema,
)
from typear_sglang import SGLangClient, SGLangError, extract_candidate_logprobs


__all__ = [
    "Choice",
    "Decision",
    "SCORE_LEVELS",
    "SGLangClient",
    "SGLangError",
    "SchemaError",
    "TypeARClient",
    "benchmark_prefix_cache",
    "candidate_softmax",
    "compile_json_schema",
    "example_schema",
    "run_schema",
    "run_sequential_decisions",
]


if __name__ == "__main__":
    main()
