"""Prefix-cache benchmark utilities for TypeLLM."""

from __future__ import annotations

import json
import math
import statistics
import uuid
from typing import Any, Sequence

from typellm_runtime import Choice
from typellm_sglang import SGLangClient


def benchmark_prefix_cache(
    context: str,
    decisions: Sequence[Choice],
    *,
    base_url: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Compare growing shared prefixes with equal-sized unrelated prefixes."""
    client = SGLangClient(base_url, model)
    resolved = [
        {label: client.single_token(label) for label in decision.choices}
        for decision in decisions
    ]

    def run(shared: bool) -> dict[str, Any]:
        client.flush_cache()
        growing = context.rstrip() + "\n\n"
        latencies: list[float] = []
        cached_tokens: list[int | None] = []
        for decision, label_tokens in zip(decisions, resolved):
            if shared:
                prompt = growing + decision.opening_text()
            else:
                # A nonce at byte zero prevents a shared radix-tree prefix while
                # keeping the rest of the prompt and its length comparable.
                nonce = uuid.uuid4().hex
                prompt = (
                    f"unrelated-{nonce}\n"
                    + context.rstrip()
                    + "\n\n"
                    + decision.opening_text()
                )
            ids = [token_id for token_id, _ in label_tokens.values()]
            scores, meta, elapsed = client.score_candidates(prompt, ids)
            selected_id = max(scores, key=scores.__getitem__)
            selected_label = next(
                label
                for label, (token_id, _) in label_tokens.items()
                if token_id == selected_id
            )
            selected_text = label_tokens[selected_label][1]
            latencies.append(elapsed)
            cached = meta.get("cached_tokens")
            cached_tokens.append(int(cached) if cached is not None else None)
            if shared:
                semantic_value = decision.choices[selected_label]
                growing = (
                    prompt
                    + selected_text
                    + '",\n  value='
                    + json.dumps(semantic_value, ensure_ascii=False)
                    + "\n)\n\n"
                )
        return {
            "latencies_seconds": latencies,
            "mean_seconds": statistics.fmean(latencies),
            "total_seconds": sum(latencies),
            "cached_tokens": cached_tokens,
        }

    result = {"shared_growing_prefix": run(True), "unrelated_prefixes": run(False)}
    shared = result["shared_growing_prefix"]["total_seconds"]
    unrelated = result["unrelated_prefixes"]["total_seconds"]
    result["total_speedup"] = unrelated / shared if shared else math.inf
    return result
