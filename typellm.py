#!/usr/bin/env python3
"""Public compatibility facade for the TypeLLM prototype.

The implementation is split by responsibility; imports from this module remain
stable for existing callers, and ``python typellm.py`` still launches the demo.
"""

from typellm_benchmark import benchmark_prefix_cache
from typellm_cli import example_schema, main
from typellm_runtime import (
    Choice,
    TypeLLMClient,
    candidate_softmax,
    run_schema,
    run_sequential_decisions,
)
from typellm_schema import (
    MAX_ENUM_CHOICES,
    Decision,
    SchemaError,
    compile_json_schema,
)
from typellm_sglang import SGLangClient, SGLangError, extract_candidate_logprobs


__all__ = [
    "Choice",
    "Decision",
    "MAX_ENUM_CHOICES",
    "SGLangClient",
    "SGLangError",
    "SchemaError",
    "TypeLLMClient",
    "benchmark_prefix_cache",
    "candidate_softmax",
    "compile_json_schema",
    "example_schema",
    "run_schema",
    "run_sequential_decisions",
]


if __name__ == "__main__":
    main()
