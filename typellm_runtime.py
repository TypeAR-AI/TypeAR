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

from typellm_schema import (
    MAX_ENUM_CHOICES,
    SchemaError,
    compile_json_schema,
)
from typellm_sglang import SGLangClient


LOG = logging.getLogger("typellm")


@dataclass(frozen=True)
class Choice:
    """A runtime decision whose control labels have been bound to values."""

    question: str
    choices: Mapping[str, Any]
    name: str | None = None
    syntax: str = "Choice"
    numeric_type: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    text_type: bool = False
    max_length: int | None = None
    return_probabilities: bool = False

    def __post_init__(self) -> None:
        if not self.choices and self.numeric_type is None and not self.text_type:
            raise ValueError("Choice.choices must not be empty")
        if self.text_type and (self.choices or self.numeric_type is not None):
            raise ValueError("Text choices cannot have enum values or a numeric type")
        if self.max_length is not None and (type(self.max_length) is not int or self.max_length < 0):
            raise ValueError("max_length must be a non-negative integer")
        if self.numeric_type not in {None, "integer", "number"}:
            raise ValueError("numeric_type must be None, 'integer', or 'number'")
        if self.numeric_type is not None and self.choices:
            raise ValueError("Open numeric choices must be empty")
        if len(self.choices) > MAX_ENUM_CHOICES:
            raise ValueError(
                f"Choice has {len(self.choices)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        if any(not label for label in self.choices):
            raise ValueError("Choice labels must be non-empty strings")

    def opening_text(self) -> str:
        if self.text_type:
            limit = "" if self.max_length is None else f" Maximum {self.max_length} characters."
            return f"Text(name={json.dumps(self.name)})\nReturn only a JSON string.{limit}\nInstructions: {self.question}"
        question = json.dumps(self.question, ensure_ascii=False)
        if self.numeric_type is not None:
            attributes = []
            if self.name is not None:
                attributes.append(f"name={json.dumps(self.name, ensure_ascii=False)}")
            if self.minimum is not None:
                attributes.append(f"minimum={json.dumps(self.minimum)}")
            if self.maximum is not None:
                attributes.append(f"maximum={json.dumps(self.maximum)}")
            metadata = f"{self.syntax}({', '.join(attributes)})"
            instruction = (
                "Return only a JSON number without a decimal point or exponent notation."
                if self.numeric_type == "integer"
                else "Return only a JSON number without exponent notation."
            )
            return "\n".join(
                [
                    metadata,
                    instruction,
                    f"Question: {self.question}",
                ]
            )

        lines = [f"{self.syntax}("]
        if self.name is not None:
            lines.append(f"  name={json.dumps(self.name, ensure_ascii=False)},")
        lines.append(f"  question={question},")
        choices = json.dumps(
            dict(self.choices), ensure_ascii=False, separators=(",", ":")
        )
        lines.extend(
            [
                f"  choices={choices},",
                '  instruction="Answer the question using only the best label.",',
                ")",
            ]
        )
        return "\n".join(lines)


class TypeLLMClient:
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
        numeric_max_digits: int = 32,
        tokenizer: str | None = None,
        numeric_cache_dir: str | os.PathLike[str] | None = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
        text_max_tokens: int = 512,
    ) -> None:
        _validate_decoding(mode, temperature)
        _validate_execution(execution)
        if type(numeric_max_digits) is not int or numeric_max_digits <= 0:
            raise ValueError("numeric_max_digits must be a positive integer")
        self.sglang = SGLangClient(
            base_url,
            model,
            timeout,
            tokenizer=tokenizer,
            numeric_cache_dir=numeric_cache_dir,
            thinking=thinking,
            thinking_budget=thinking_budget,
            text_max_tokens=text_max_tokens,
        )
        self.mode = mode
        self.execution = execution
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label_pool = tuple(label_pool or self.DEFAULT_LABEL_POOL)
        self.numeric_max_digits = numeric_max_digits
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
            finite_sizes = [len(item.choices) for item in decisions if item.choices]
            labels = self._control_labels(max(finite_sizes)) if finite_sizes else []
            return [
                Choice(
                    question=item.question,
                    choices=dict(zip(labels, item.choices)),
                    name=item.name,
                    syntax=item.syntax,
                    numeric_type=item.numeric_type,
                    minimum=item.minimum,
                    maximum=item.maximum,
                    text_type=item.text_type,
                    max_length=item.max_length,
                    return_probabilities=item.return_probabilities,
                )
                for item in decisions
            ]
        if not isinstance(properties, list):
            raise SchemaError(
                "schema.properties must be either an ordered JSON Schema object "
                "or a legacy TypeLLM list"
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
            for old_key in ("question", "x-question"):
                if old_key in prop:
                    raise SchemaError(f"{old_key} is no longer supported; use instructions")
            question = prop.get("instructions")
            kind = prop.get("type")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"properties[{index}].name must be a non-empty string")
            if name in names:
                raise SchemaError(f"duplicate property name: {name!r}")
            names.add(name)
            if not isinstance(question, str) or not question:
                raise SchemaError(
                    f"properties[{index}].instructions must be a non-empty string"
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
                if len(values) > MAX_ENUM_CHOICES:
                    raise SchemaError(
                        f"choices for {name!r} has {len(values)} values; "
                        f"the maximum is {MAX_ENUM_CHOICES}"
                    )
            elif kind == "bool":
                if "choices" in prop:
                    raise SchemaError(
                        f"bool property {name!r} must not define choices; "
                        "it automatically uses true/false"
                    )
                values = [True, False]
            else:
                raise SchemaError(
                    f"properties[{index}].type must be 'choice' or 'bool', "
                    f"got {kind!r}"
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
                    else "Choice"
                ),
            )
            for name, kind, question, values in normalized
        ]

    def generate(
        self,
        *,
        context: str | None = None,
        state: str | None = None,
        schema: Mapping[str, Any] | None = None,
        questions: Mapping[str, Any] | None = None,
        mode: str | None = None,
        execution: str | None = None,
        temperature: float | None = None,
        print_final_prompt: bool = False,
    ) -> dict[str, Any]:
        if (context is None) == (state is None):
            raise ValueError("provide exactly one of context or state")
        context = state if state is not None else context
        if not isinstance(context, str):
            raise ValueError("context or state must be a string")
        if (schema is None) == (questions is None):
            raise SchemaError("provide exactly one of questions or schema")
        if questions is not None:
            if not isinstance(questions, Mapping):
                raise SchemaError("questions must be a mapping of field names to definitions")
            schema = {"type": "object", "properties": questions}
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
                self.numeric_max_digits,
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
                self.numeric_max_digits,
            )
            self.last_prompt = None

        output: dict[str, Any] = {}
        for decision, row in zip(decisions, rows):
            assert decision.name is not None
            value = row["value"]
            if decision.return_probabilities:
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


def _numeric_candidates(
    text: str, numeric_type: str, max_digits: int
) -> tuple[str, ...]:
    """Return the next characters admitted by a small JSON-number state machine."""
    digit_count = sum(char.isdigit() for char in text)
    if digit_count >= max_digits:
        return ()
    if text in {"", "-"}:
        return tuple(("-" if not text else "") + digits)

    unsigned = text[1:] if text.startswith("-") else text
    if numeric_type == "integer":
        if unsigned == "0":
            return ()
        return tuple(digits)

    if "." in unsigned:
        return tuple(digits)
    if unsigned == "0":
        return (".",)
    return tuple(digits + ".")


def _numeric_text_is_complete(text: str, numeric_type: str) -> bool:
    if not text or text == "-":
        return False
    unsigned = text[1:] if text.startswith("-") else text
    if not unsigned or (
        len(unsigned) > 1
        and unsigned.startswith("0")
        and not unsigned.startswith("0.")
    ):
        return False
    if numeric_type == "integer":
        return unsigned.isdigit()
    if "." not in unsigned:
        return unsigned.isdigit()
    integer, fraction = unsigned.split(".", 1)
    return integer.isdigit() and bool(fraction) and fraction.isdigit()


def _numeric_text_is_prefix(text: str, numeric_type: str) -> bool:
    """Whether text can still be extended into a supported JSON-style number."""
    if text in {"", "-"}:
        return True
    unsigned = text[1:] if text.startswith("-") else text
    if not unsigned or unsigned.count(".") > 1:
        return False
    integer, separator, fraction = unsigned.partition(".")
    if not integer.isdigit():
        return False
    if len(integer) > 1 and integer.startswith("0"):
        return False
    if numeric_type == "integer":
        return not separator
    return not separator or not fraction or fraction.isdigit()


def _numeric_transition(
    text: str, piece: str, numeric_type: str, max_digits: int
) -> tuple[str, bool] | None:
    """Apply one tokenizer piece, rejecting grammar-invalid continuations."""
    if '"' in piece:
        return None
    value_text = text + piece
    if sum(char.isdigit() for char in value_text) > max_digits:
        return None
    return (
        (value_text, False)
        if _numeric_text_is_prefix(value_text, numeric_type)
        else None
    )


def _numeric_token_candidates(
    client: SGLangClient, text: str, numeric_type: str, max_digits: int
) -> dict[int, tuple[str, str, bool]]:
    """Map valid next token IDs to (decoded piece, next text, is finished)."""
    if hasattr(client, "numeric_token_pieces"):
        pieces = client.numeric_token_pieces()
    else:
        # Compatibility for small custom clients written against the first
        # prototype. Native SGLangClient always uses its model-derived table.
        pieces = [
            client.single_token(piece)
            for piece in _numeric_candidates(text, numeric_type, max_digits)
        ]
    candidates: dict[int, tuple[str, str, bool]] = {}
    for token_id, piece in pieces:
        transition = _numeric_transition(text, piece, numeric_type, max_digits)
        if transition is not None:
            next_text, finished = transition
            candidates[int(token_id)] = (piece, next_text, finished)
    return candidates


def _parse_numeric_value(text: str, decision: Choice) -> int | float:
    if not _numeric_text_is_complete(text, decision.numeric_type or ""):
        raise ValueError(f"Generated invalid {decision.numeric_type}: {text!r}")
    value: int | float = (
        int(text) if decision.numeric_type == "integer" else float(text)
    )
    if not math.isfinite(float(value)):
        raise ValueError(f"Generated non-finite number for {decision.name!r}")
    if decision.minimum is not None and value < decision.minimum:
        raise ValueError(
            f"Generated value {value} is below minimum {decision.minimum} "
            f"for {decision.name!r}"
        )
    if decision.maximum is not None and value > decision.maximum:
        raise ValueError(
            f"Generated value {value} is above maximum {decision.maximum} "
            f"for {decision.name!r}"
        )
    return value


def _decode_numeric(
    client: SGLangClient,
    prefix: str,
    decision: Choice,
    mode: str,
    temperature: float,
    rng: random.Random,
    max_digits: int,
) -> tuple[int | float, str, str]:
    end_token_id, end_token_text = client.end_of_message_token()
    text = ""
    step = 0
    while True:
        candidates = _numeric_token_candidates(
            client, text, decision.numeric_type or "", max_digits
        )
        if _numeric_text_is_complete(text, decision.numeric_type or ""):
            candidates[end_token_id] = (end_token_text, text, True)
        if not candidates:
            raise ValueError(
                f"Could not complete numeric field {decision.name!r} within "
                f"{max_digits} digits"
            )
        ids = list(candidates)
        by_id, meta, elapsed = client.score_candidates(prefix + text, ids)
        raw = {str(token_id): by_id[token_id] for token_id in ids}
        probs = candidate_softmax(
            raw, temperature if mode == "sample" else 1.0
        )
        selected_key = (
            max(raw, key=raw.__getitem__)
            if mode == "argmax"
            else _sample(probs, rng)
        )
        selected_id = int(selected_key)
        selected_piece, next_text, finished = candidates[selected_id]
        ranked = sorted(ids, key=by_id.__getitem__, reverse=True)[:20]
        LOG.info(
            "numeric name=%s step=%d candidates=%d top_candidates=%s "
            "selected_id=%d selected=%r elapsed=%.4fs cached_tokens=%s",
            decision.name,
            step,
            len(ids),
            [
                {
                    "id": token_id,
                    "text": candidates[token_id][0],
                    "logprob": by_id[token_id],
                    "probability": probs[str(token_id)],
                }
                for token_id in ranked
            ],
            selected_id,
            selected_piece,
            elapsed,
            meta.get("cached_tokens"),
        )
        LOG.debug("numeric candidate_logprobs=%s probabilities=%s", raw, probs)
        text = next_text
        if finished:
            value = _parse_numeric_value(text, decision)
            completed = prefix + text + end_token_text
            return value, completed, text
        step += 1


def _execute_decisions(
    client: SGLangClient,
    context: str,
    decisions: Sequence[Choice],
    mode: str,
    temperature: float,
    rng: random.Random,
    numeric_max_digits: int = 32,
) -> tuple[list[dict], str]:
    messages: list[dict[str, str]] = []
    prefix = ""
    results: list[dict] = []

    for index, decision in enumerate(decisions):
        user_content = decision.opening_text()
        if index == 0:
            user_content = context.rstrip() + "\n\n" + user_content
        messages.append({"role": "user", "content": user_content})
        prefix = client.render_chat(messages, add_generation_prompt=True)
        if decision.text_type:
            value = client.generate_texts([prefix], [decision.max_length],
                temperature=0 if mode == "argmax" else temperature,
                seed=rng.randrange(2**31))[0]
            messages.append({"role": "assistant", "content": json.dumps(value, ensure_ascii=False)})
            prefix = client.render_chat(messages, add_generation_prompt=False)
            results.append({"name": decision.name, "question": decision.question,
                            "label": None, "value": value, "probabilities": None})
            continue
        if decision.numeric_type is not None:
            semantic_value, prefix, generated_text = _decode_numeric(
                client,
                prefix,
                decision,
                mode,
                temperature,
                rng,
                numeric_max_digits,
            )
            messages.append({"role": "assistant", "content": generated_text})
            results.append(
                {
                    "name": decision.name,
                    "question": decision.question,
                    "label": None,
                    "value": semantic_value,
                    "probabilities": None,
                }
            )
            continue
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
        messages.append({"role": "assistant", "content": selected_text})
        prefix = client.render_chat(messages, add_generation_prompt=False)
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
    numeric_max_digits: int = 32,
) -> tuple[list[dict], list[str]]:
    shared_messages = [{"role": "user", "content": context.rstrip()}]
    shared_prefix = client.render_chat(
        shared_messages, add_generation_prompt=False
    )
    label_tokens_by_decision: list[dict[str, tuple[int, str]]] = []
    candidate_ids: list[list[int]] = []
    prompts: list[str] = []

    finite_indexes: list[int] = []
    open_results: dict[int, tuple[dict, str]] = {}
    text_pending = []
    for index, decision in enumerate(decisions):
        messages = shared_messages + [
            {"role": "user", "content": decision.opening_text()}
        ]
        prompt = client.render_chat(messages, add_generation_prompt=True)
        if decision.text_type:
            text_pending.append((index, decision, prompt, messages))
            continue
        if decision.numeric_type is not None:
            value, _completed, generated_text = _decode_numeric(
                client,
                prompt,
                decision,
                mode,
                temperature,
                rng,
                numeric_max_digits,
            )
            open_results[index] = (
                {
                    "name": decision.name,
                    "question": decision.question,
                    "label": None,
                    "value": value,
                    "probabilities": None,
                },
                client.render_chat(
                    messages
                    + [{"role": "assistant", "content": generated_text}],
                    add_generation_prompt=False,
                ),
            )
            continue
        label_tokens = {
            label: client.single_token(label) for label in decision.choices
        }
        ids = [token_id for token_id, _ in label_tokens.values()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Decision {index} has labels with duplicate token IDs: {ids}")
        label_tokens_by_decision.append(label_tokens)
        candidate_ids.append(ids)
        prompts.append(prompt)
        finite_indexes.append(index)
        LOG.info(
            "batch_decision=%d name=%s candidate_token_ids=%s",
            index,
            decision.name,
            {label: token_id for label, (token_id, _) in label_tokens.items()},
        )

    if text_pending:
        values = client.generate_texts(
            [item[2] for item in text_pending],
            [item[1].max_length for item in text_pending],
            temperature=0 if mode == "argmax" else temperature,
            seed=rng.randrange(2**31),
        )
        for (index, decision, prompt, messages), value in zip(text_pending, values):
            completed = client.render_chat(messages + [{"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}], add_generation_prompt=False)
            open_results[index] = ({"name": decision.name, "question": decision.question,
                                       "label": None, "value": value, "probabilities": None}, completed)

    # Warm the exact common prefix once, then let SGLang fork the cached state
    # across the K batched prompts. TypeLLM never reads or moves KV tensors.
    if prompts:
        cache_meta = client.cache_prefix(shared_prefix)
        LOG.info("batch_shared_prefix_cached_tokens=%s", cache_meta.get("cached_tokens"))
        scored, elapsed = client.score_candidates_batch(prompts, candidate_ids)
    else:
        scored, elapsed = [], 0.0

    results: list[dict] = []
    completed_prompts: list[str] = []
    for finite_index, (decision_index, label_tokens, (by_id, meta)) in enumerate(
        zip(finite_indexes, label_tokens_by_decision, scored)
    ):
        index = decision_index
        decision = decisions[index]
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
            client.render_chat(
                shared_messages
                + [{"role": "user", "content": decision.opening_text()}]
                + [{"role": "assistant", "content": selected_text}],
                add_generation_prompt=False,
            )
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

    if open_results:
        finite_rows = iter(zip(results, completed_prompts))
        merged_results: list[dict] = []
        merged_prompts: list[str] = []
        for index in range(len(decisions)):
            if index in open_results:
                row, prompt = open_results[index]
            else:
                row, prompt = next(finite_rows)
            merged_results.append(row)
            merged_prompts.append(prompt)
        results, completed_prompts = merged_results, merged_prompts
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
    numeric_max_digits: int = 32,
    tokenizer: str | None = None,
    numeric_cache_dir: str | os.PathLike[str] | None = None,
    thinking: bool = False,
    thinking_budget: int = 1024,
    print_final_prompt: bool = True,
) -> list[dict]:
    _validate_decoding(mode, temperature)
    if type(numeric_max_digits) is not int or numeric_max_digits <= 0:
        raise ValueError("numeric_max_digits must be a positive integer")
    client = SGLangClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
        tokenizer=tokenizer,
        numeric_cache_dir=numeric_cache_dir,
        thinking=thinking,
        thinking_budget=thinking_budget,
    )
    results, prefix = _execute_decisions(
        client,
        context,
        decisions,
        mode,
        temperature,
        random.Random(seed),
        numeric_max_digits,
    )
    if print_final_prompt:
        _print_final_prompt(prefix)
    return results


def run_schema(
    context: str | None = None,
    schema: Mapping[str, Any] | None = None,
    mode: str = "argmax",
    temperature: float = 1.0,
    *,
    state: str | None = None,
    questions: Mapping[str, Any] | None = None,
    execution: str = "sequential",
    base_url: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    numeric_max_digits: int = 32,
    tokenizer: str | None = None,
    numeric_cache_dir: str | os.PathLike[str] | None = None,
    thinking: bool = False,
    thinking_budget: int = 1024,
    text_max_tokens: int = 512,
    print_final_prompt: bool = False,
) -> dict[str, Any]:
    client = TypeLLMClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
        mode=mode,
        execution=execution,
        temperature=temperature,
        seed=seed,
        numeric_max_digits=numeric_max_digits,
        tokenizer=tokenizer,
        numeric_cache_dir=numeric_cache_dir,
        thinking=thinking,
        thinking_budget=thinking_budget,
        text_max_tokens=text_max_tokens,
    )
    return client.generate(
        context=context,
        state=state,
        schema=schema,
        questions=questions,
        print_final_prompt=print_final_prompt,
    )
