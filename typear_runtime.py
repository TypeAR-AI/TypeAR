"""Schema binding and sequential constrained decision execution."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from string import ascii_uppercase, digits
from typing import Any, Mapping, Sequence

from typear_schema import SCORE_LEVELS, SchemaError, compile_json_schema
from typear_sglang import SGLangClient


LOG = logging.getLogger("typear")


@dataclass(frozen=True)
class Choice:
    """A runtime decision whose control labels have been bound to values."""

    question: str
    choices: Mapping[str, Any]
    name: str | None = None
    syntax: str = "Choice"

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("Choice.choices must not be empty")
        if any(not label for label in self.choices):
            raise ValueError("Choice labels must be non-empty strings")

    def opening_text(self) -> str:
        question = json.dumps(self.question, ensure_ascii=False)
        choices = json.dumps(
            dict(self.choices), ensure_ascii=False, separators=(",", ":")
        )
        lines = [f"{self.syntax}("]
        if self.name is not None:
            lines.append(f"  name={json.dumps(self.name, ensure_ascii=False)},")
        lines.extend(
            [
                f"  question={question},",
                f"  choices={choices},",
                '  answer="',
            ]
        )
        return "\n".join(lines)


class TypeARClient:
    """Compile ordered schemas into constrained single-token decisions."""

    DEFAULT_LABEL_POOL = tuple(ascii_uppercase + digits)

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: str | None = None,
        *,
        mode: str = "argmax",
        execution: str = "sequential",
        temperature: float = 1.0,
        seed: int | None = None,
        timeout: float = 120.0,
        label_pool: Sequence[str] | None = None,
    ) -> None:
        _validate_decoding(mode, temperature)
        _validate_execution(execution)
        self.sglang = SGLangClient(base_url, model, timeout)
        self.mode = mode
        self.execution = execution
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label_pool = tuple(label_pool or self.DEFAULT_LABEL_POOL)
        self.label_token_map: dict[str, int] = {}
        self.last_prompt: str | None = None
        self.last_prompts: list[str] = []

    def _control_labels(self, count: int) -> list[str]:
        labels: list[str] = []
        token_ids: set[int] = set()
        for label in self.label_pool:
            try:
                token_id, _ = self.sglang.single_token(label)
            except ValueError:
                continue
            if token_id in token_ids:
                continue
            labels.append(label)
            token_ids.add(token_id)
            self.label_token_map[label] = token_id
            if len(labels) == count:
                LOG.info("runtime_label_token_map=%s", self.label_token_map)
                return labels
        raise SchemaError(
            f"Schema needs {count} control labels, but only {len(labels)} unique "
            "single-token labels were found for the current tokenizer"
        )

    def compile_schema(self, schema: Mapping[str, Any]) -> list[Choice]:
        """Compile standard JSON Schema or the original ordered-list format."""
        if not isinstance(schema, Mapping):
            raise SchemaError("schema must be a mapping")
        if schema.get("type", "object") != "object":
            raise SchemaError("the top-level schema type must be 'object'")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            decisions = compile_json_schema(schema)
            labels = self._control_labels(max(len(item.choices) for item in decisions))
            return [
                Choice(
                    question=item.question,
                    choices=dict(zip(labels, item.choices)),
                    name=item.name,
                    syntax=item.syntax,
                )
                for item in decisions
            ]
        if not isinstance(properties, list):
            raise SchemaError(
                "schema.properties must be either an ordered JSON Schema object "
                "or a legacy TypeAR list"
            )
        if not properties:
            raise SchemaError("schema.properties must not be empty")

        normalized: list[tuple[str, str, str, list[Any]]] = []
        names: set[str] = set()
        max_choices = 0
        for index, prop in enumerate(properties):
            if not isinstance(prop, Mapping):
                raise SchemaError(f"properties[{index}] must be a mapping")
            name = prop.get("name")
            question = prop.get("question")
            kind = prop.get("type")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"properties[{index}].name must be a non-empty string")
            if name in names:
                raise SchemaError(f"duplicate property name: {name!r}")
            names.add(name)
            if not isinstance(question, str) or not question:
                raise SchemaError(
                    f"properties[{index}].question must be a non-empty string"
                )
            if kind == "choice":
                values = prop.get("choices")
                if not isinstance(values, list) or not values:
                    raise SchemaError(
                        f"choice property {name!r} requires a non-empty choices list"
                    )
                if any(not isinstance(value, str) or not value for value in values):
                    raise SchemaError(
                        f"all choices for {name!r} must be non-empty strings"
                    )
                if len(set(values)) != len(values):
                    raise SchemaError(f"choices for {name!r} must be unique")
            elif kind == "bool":
                if "choices" in prop:
                    raise SchemaError(
                        f"bool property {name!r} must not define choices; "
                        "it automatically uses true/false"
                    )
                values = [True, False]
            elif kind == "score":
                if "choices" in prop:
                    raise SchemaError(
                        f"score property {name!r} must not define choices; "
                        "it automatically uses 0.0 through 1.0"
                    )
                values = list(SCORE_LEVELS)
            else:
                raise SchemaError(
                    f"properties[{index}].type must be 'choice', 'bool', or "
                    f"'score', got {kind!r}"
                )
            max_choices = max(max_choices, len(values))
            normalized.append((name, kind, question, values))

        labels = self._control_labels(max_choices)
        return [
            Choice(
                question=question,
                choices=dict(zip(labels, values)),
                name=name,
                syntax=(
                    "Bool"
                    if kind == "bool"
                    else "Score"
                    if kind == "score"
                    else "Choice"
                ),
            )
            for name, kind, question, values in normalized
        ]

    def generate(
        self,
        *,
        context: str,
        schema: Mapping[str, Any],
        mode: str | None = None,
        execution: str | None = None,
        temperature: float | None = None,
        return_probabilities: bool = False,
        print_final_prompt: bool = False,
    ) -> dict[str, Any]:
        active_mode = self.mode if mode is None else mode
        active_execution = self.execution if execution is None else execution
        active_temperature = self.temperature if temperature is None else temperature
        _validate_decoding(active_mode, active_temperature)
        _validate_execution(active_execution)
        decisions = self.compile_schema(schema)
        if active_execution == "sequential":
            rows, self.last_prompt = _execute_decisions(
                self.sglang,
                context,
                decisions,
                active_mode,
                active_temperature,
                self.rng,
            )
            self.last_prompts = [self.last_prompt]
        else:
            rows, self.last_prompts = _execute_batch_decisions(
                self.sglang,
                context,
                decisions,
                active_mode,
                active_temperature,
                self.rng,
            )
            self.last_prompt = None

        output: dict[str, Any] = {}
        for decision, row in zip(decisions, rows):
            assert decision.name is not None
            value = row["value"]
            if return_probabilities:
                probabilities = {
                    decision.choices[label]: probability
                    for label, probability in row["probabilities"].items()
                }
                output[decision.name] = {
                    "value": value,
                    "probabilities": probabilities,
                }
            else:
                output[decision.name] = value

        if print_final_prompt:
            if active_execution == "sequential":
                assert self.last_prompt is not None
                _print_final_prompt(self.last_prompt)
            else:
                _print_final_prompts(self.last_prompts)
        return output


def candidate_softmax(
    logprobs: Mapping[str, float], temperature: float = 1.0
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    scaled = {label: value / temperature for label, value in logprobs.items()}
    pivot = max(scaled.values())
    weights = {label: math.exp(value - pivot) for label, value in scaled.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _sample(probs: Mapping[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last = ""
    for label, probability in probs.items():
        last = label
        cumulative += probability
        if threshold <= cumulative:
            return label
    return last


def _validate_decoding(mode: str, temperature: float) -> None:
    if mode not in {"argmax", "sample"}:
        raise ValueError("mode must be 'argmax' or 'sample'")
    if mode == "sample" and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and > 0 in sample mode")


def _validate_execution(execution: str) -> None:
    if execution not in {"sequential", "batch"}:
        raise ValueError("execution must be 'sequential' or 'batch'")


def _execute_decisions(
    client: SGLangClient,
    context: str,
    decisions: Sequence[Choice],
    mode: str,
    temperature: float,
    rng: random.Random,
) -> tuple[list[dict], str]:
    prefix = context.rstrip() + "\n\n"
    results: list[dict] = []

    for index, decision in enumerate(decisions):
        prefix += decision.opening_text()
        label_tokens = {label: client.single_token(label) for label in decision.choices}
        ids = [token_id for token_id, _ in label_tokens.values()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Decision {index} has labels with duplicate token IDs: {ids}")
        LOG.info(
            "decision=%d name=%s candidate_token_ids=%s",
            index,
            decision.name,
            {label: token_id for label, (token_id, _) in label_tokens.items()},
        )

        by_id, meta, elapsed = client.score_candidates(prefix, ids)
        raw = {
            label: by_id[token_id]
            for label, (token_id, _) in label_tokens.items()
        }
        probability_temperature = temperature if mode == "sample" else 1.0
        probabilities = candidate_softmax(raw, probability_temperature)
        selected = (
            max(raw, key=raw.__getitem__)
            if mode == "argmax"
            else _sample(probabilities, rng)
        )
        selected_text = label_tokens[selected][1]
        semantic_value = decision.choices[selected]

        LOG.info("decision=%d raw_candidate_logprobs=%s", index, raw)
        LOG.info("decision=%d renormalized_probabilities=%s", index, probabilities)
        LOG.info(
            "decision=%d selected=%s value=%s elapsed=%.4fs cached_tokens=%s",
            index,
            selected,
            semantic_value,
            elapsed,
            meta.get("cached_tokens"),
        )

        # Send the growing full prefix again. SGLang—not this client—owns and
        # recovers all KV tensors through its RadixAttention prefix cache.
        prefix += (
            selected_text
            + '",\n  value='
            + json.dumps(semantic_value, ensure_ascii=False, separators=(",", ":"))
            + "\n)"
        )
        if index + 1 < len(decisions):
            prefix += "\n\n"
        results.append(
            {
                "name": decision.name,
                "question": decision.question,
                "label": selected,
                "value": semantic_value,
                "probabilities": probabilities,
            }
        )
    return results, prefix


def _execute_batch_decisions(
    client: SGLangClient,
    context: str,
    decisions: Sequence[Choice],
    mode: str,
    temperature: float,
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    shared_prefix = context.rstrip() + "\n\n"
    label_tokens_by_decision: list[dict[str, tuple[int, str]]] = []
    candidate_ids: list[list[int]] = []
    prompts: list[str] = []

    for index, decision in enumerate(decisions):
        label_tokens = {
            label: client.single_token(label) for label in decision.choices
        }
        ids = [token_id for token_id, _ in label_tokens.values()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Decision {index} has labels with duplicate token IDs: {ids}")
        label_tokens_by_decision.append(label_tokens)
        candidate_ids.append(ids)
        prompts.append(shared_prefix + decision.opening_text())
        LOG.info(
            "batch_decision=%d name=%s candidate_token_ids=%s",
            index,
            decision.name,
            {label: token_id for label, (token_id, _) in label_tokens.items()},
        )

    # Warm the exact common prefix once, then let SGLang fork the cached state
    # across the K batched prompts. TypeAR never reads or moves KV tensors.
    cache_meta = client.cache_prefix(shared_prefix)
    LOG.info("batch_shared_prefix_cached_tokens=%s", cache_meta.get("cached_tokens"))
    scored, elapsed = client.score_candidates_batch(prompts, candidate_ids)

    results: list[dict] = []
    completed_prompts: list[str] = []
    for index, (decision, label_tokens, (by_id, meta)) in enumerate(
        zip(decisions, label_tokens_by_decision, scored)
    ):
        raw = {
            label: by_id[token_id]
            for label, (token_id, _) in label_tokens.items()
        }
        probability_temperature = temperature if mode == "sample" else 1.0
        probabilities = candidate_softmax(raw, probability_temperature)
        selected = (
            max(raw, key=raw.__getitem__)
            if mode == "argmax"
            else _sample(probabilities, rng)
        )
        selected_text = label_tokens[selected][1]
        semantic_value = decision.choices[selected]
        completed_prompts.append(
            prompts[index]
            + selected_text
            + '",\n  value='
            + json.dumps(semantic_value, ensure_ascii=False, separators=(",", ":"))
            + "\n)"
        )
        LOG.info("batch_decision=%d raw_candidate_logprobs=%s", index, raw)
        LOG.info(
            "batch_decision=%d renormalized_probabilities=%s", index, probabilities
        )
        LOG.info(
            "batch_decision=%d selected=%s value=%s batch_elapsed=%.4fs "
            "cached_tokens=%s",
            index,
            selected,
            semantic_value,
            elapsed,
            meta.get("cached_tokens"),
        )
        results.append(
            {
                "name": decision.name,
                "question": decision.question,
                "label": selected,
                "value": semantic_value,
                "probabilities": probabilities,
            }
        )
    return results, completed_prompts


def _print_final_prompt(prefix: str) -> None:
    print("\n===== FINAL ACCUMULATED PROMPT =====")
    print(prefix)
    print("===== END FINAL PROMPT =====")


def _print_final_prompts(prefixes: Sequence[str]) -> None:
    for index, prefix in enumerate(prefixes):
        print(f"\n===== FINAL BATCH PROMPT {index} =====")
        print(prefix)
        print(f"===== END BATCH PROMPT {index} =====")


def run_sequential_decisions(
    context: str,
    decisions: list[Choice],
    mode: str = "argmax",
    temperature: float = 1.0,
    *,
    base_url: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    print_final_prompt: bool = True,
) -> list[dict]:
    _validate_decoding(mode, temperature)
    client = SGLangClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
    )
    results, prefix = _execute_decisions(
        client, context, decisions, mode, temperature, random.Random(seed)
    )
    if print_final_prompt:
        _print_final_prompt(prefix)
    return results


def run_schema(
    context: str,
    schema: Mapping[str, Any],
    mode: str = "argmax",
    temperature: float = 1.0,
    *,
    execution: str = "sequential",
    base_url: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    return_probabilities: bool = False,
    print_final_prompt: bool = False,
) -> dict[str, Any]:
    client = TypeARClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
        mode=mode,
        execution=execution,
        temperature=temperature,
        seed=seed,
    )
    return client.generate(
        context=context,
        schema=schema,
        return_probabilities=return_probabilities,
        print_final_prompt=print_final_prompt,
    )
