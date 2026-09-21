"""Public compatibility facade for the TypeLLM prototype.

The implementation is split by responsibility; imports from this module remain
stable for existing callers, and ``python -m typellm`` launches the demo.
"""

from .benchmark import benchmark_prefix_cache
from .runtime import (
    Choice,
    TypeLLMClient,
    candidate_softmax,
    run_schema,
    run_sequential_decisions,
)
from .schema import (
    MAX_ENUM_CHOICES,
    Decision,
    SchemaError,
    compile_json_schema,
)
from .sglang import SGLangClient, SGLangError, extract_candidate_logprobs


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


def __getattr__(name: str):
    # Leave the CLI unloaded until needed so `python -m typellm.cli` works.
    if name in {"example_schema", "main"}:
        from .cli import example_schema, main

        globals().update(example_schema=example_schema, main=main)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"example_schema", "main"})
