"""Schema binding and constrained decision execution."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import permutations as all_permutations
from string import ascii_uppercase, digits
from typing import Any, Mapping, Sequence

from .schema import (
    MAX_ENUM_CHOICES,
    SchemaError,
    compile_json_schema,
    dependency_layers,
)
from .images import encode_images
from .sglang import SGLangClient, Usage, call_scope


LOG = logging.getLogger("typellm")


def _closed_answer(decision: "Choice", value_json: str) -> str:
    """The full answer an open field leaves in history, e.g. {"total": 174600}."""
    return decision.answer_prefill + " " + value_json + "}" if decision.answer_prefill else value_json


def _closed_label(decision: "Choice", label: str) -> str:
    """A choice answer as history keeps it, e.g. {"category": "A"}."""
    return decision.label_prefill + label + '"}' if decision.label_prefill else label


def _logsumexp(values: Sequence[float]) -> float:
    pivot = max(values)
    return pivot + math.log(math.fsum(math.exp(v - pivot) for v in values))


def _choose(probs: Mapping[str, float], mode: str, rng: random.Random) -> str:
    return max(probs, key=probs.__getitem__) if mode == "argmax" else _sample(probs, rng)


def _decide_nulls(client, items, mode, temperature, rng) -> list[bool]:
    """Decide null or string for nullable string fields in one batched request.

    At the prefilled key, the tokens that start null compete with the tokens that
    start a string, as this tokenizer splits {"name": null} and {"name": "text"}.
    """
    starts = client.json_value_starts()
    if not starts["null"] or not starts["string"]:
        raise ValueError("Nullable string fields need a tokenizer that starts null and strings "
                         'with single tokens after \'{"name":\'')
    null_ids = [token for token, _ in starts["null"]]
    ids = null_ids + [token for token, _ in starts["string"]]
    if len(items) == 1:
        by_id, meta, _ = client.score_candidates(items[0][0], ids)
        scored = [(by_id, meta)]
    else:
        scored, _ = client.score_candidates_batch([prompt for prompt, _ in items], [ids] * len(items))
    nulls = []
    for (_, decision), (by_id, _) in zip(items, scored):
        probs = candidate_softmax(
            {"null": _logsumexp([by_id[i] for i in null_ids]),
             "value": _logsumexp([by_id[i] for i in ids if i not in null_ids])},
            temperature if mode == "sample" else 1.0)
        choice = _choose(probs, mode, rng)
        LOG.info("null_decision name=%s p_null=%.4f selected=%s", decision.name, probs["null"], choice)
        nulls.append(choice == "null")
    return nulls


def _user_content(text: str, image_count: int) -> str | list[dict[str, str]]:
    """Put images ahead of the text in the first user turn."""
    if not image_count:
        return text
    return [{"type": "image"}] * image_count + [{"type": "text", "text": text}]


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
    permutations: int | str = 1
    return_probabilities: bool = False
    depends_on: tuple[str, ...] | None = None
    nullable: bool = False

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

    @property
    def answer_prefill(self) -> str:
        """Start of the answer for open fields; the model continues with the value.

        Chat models tend to answer {"name": value}, so the value is decoded right
        where they would write it.
        """
        if self.name is None or not (self.text_type or self.numeric_type is not None):
            return ""
        # No trailing space: in {"name": 12} the space belongs to the value's first token.
        return "{" + json.dumps(self.name, ensure_ascii=False) + ":"

    @property
    def label_prefill(self) -> str:
        """Start of a choice answer, {"name": "; the next token is the label."""
        if self.name is None or self.text_type or self.numeric_type is not None:
            return ""
        return "{" + json.dumps(self.name, ensure_ascii=False) + ': "'

    def opening_text(self) -> str:
        """The field's prompt: the same Field / Type / Instructions / Answer lines for every type."""
        lines = []
        if self.name is not None:
            lines.append(f"Field: {json.dumps(self.name, ensure_ascii=False)}")
        # A nullable field says "or null" in both its type and its answer: a bare
        # <number> reads as "a number is required" and pulls absent values to 0.
        or_null = " or null" if self.nullable else ""
        if self.text_type:
            kind = f"string{or_null}" + ("" if self.max_length is None else f", at most {self.max_length} characters")
        elif self.numeric_type is not None:
            # Plain decimals: json.dumps writes 1e-05, which the answer line forbids.
            bounds = [f"{word} {Decimal(json.dumps(value)):f}" for word, value in
                      (("minimum", self.minimum), ("maximum", self.maximum)) if value is not None]
            kind = ", ".join([self.numeric_type + or_null, *bounds])
        else:
            kind = "boolean" if self.syntax == "Bool" else "choice"
        lines.append(f"Type: {kind}")
        lines.append(f"Instructions: {self.question}")
        if self.text_type or self.numeric_type is not None:
            placeholder = "<string" + or_null + ">" if self.text_type else f"<{self.numeric_type}{or_null}>"
            answer = (f"Answer as {{{json.dumps(self.name, ensure_ascii=False)}: {placeholder}}}."
                      if self.name is not None else f"Answer with a JSON {placeholder[1:-1]} only.")
            if self.numeric_type == "number":
                answer += " Do not use exponent notation."
            if self.nullable:
                answer += " Return null only if there is no value."
        else:
            lines.append(f"Choices: {json.dumps(dict(self.choices), ensure_ascii=False)}")
            answer = (f'Answer as {{{json.dumps(self.name, ensure_ascii=False)}: "<label>"}}.'
                      if self.name is not None else "Answer with the best label only.")
        lines.append(answer)
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
        temperature: float = 1.0,
        seed: int | None = None,
        timeout: float = 120.0,
        label_pool: Sequence[str] | None = None,
        numeric_max_digits: int = 32,
        tokenizer: str | None = None,
        numeric_cache_dir: str | os.PathLike[str] | None = None,
        thinking: bool = False,
        thinking_budget: int | None = None,
        text_max_tokens: int = 512,
    ) -> None:
        _validate_decoding(mode, temperature)
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
            answer_reserve_tokens=numeric_max_digits + 3,
        )
        self.mode = mode
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label_pool = tuple(label_pool or self.DEFAULT_LABEL_POOL)
        self.numeric_max_digits = numeric_max_digits
        self.label_token_map: dict[str, int] = {}
        # Per-thread/task, so concurrent generate() calls never see each other's prompts.
        self._last_prompts: ContextVar[list[str]] = ContextVar(
            f"typellm_last_prompts_{id(self)}", default=[]
        )
        self._last_usage: ContextVar[Usage | None] = ContextVar(
            f"typellm_last_usage_{id(self)}", default=None
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_last_prompts"]
        del state["_last_usage"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._last_prompts = ContextVar(f"typellm_last_prompts_{id(self)}", default=[])
        self._last_usage = ContextVar(f"typellm_last_usage_{id(self)}", default=None)

    @property
    def last_prompts(self) -> list[str]:
        """Final prompts of the last generate() call made in this thread or task."""
        return self._last_prompts.get()

    @property
    def last_usage(self) -> Usage | None:
        """Requests and tokens of the last generate() call made in this thread or task.

        Set even when the call raised, so partial work is still counted.
        """
        return self._last_usage.get()

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
                    permutations=item.permutations,
                    return_probabilities=item.return_probabilities,
                    depends_on=item.depends_on,
                    nullable=item.nullable,
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
            if "depends_on" in prop:
                raise SchemaError("depends_on requires questions or object-form schema.properties")
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
        images: Sequence[Any] | None = None,
        mode: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        print_final_prompt: bool = False,
    ) -> dict[str, Any]:
        """Answer every field; fields run in parallel unless they declare depends_on.

        A seed makes this call reproducible on its own; without one, calls share
        the client's random stream.
        """
        self._last_usage.set(None)
        if (context is None) == (state is None):
            raise ValueError("provide exactly one of context or state")
        context = state if state is not None else context
        if not isinstance(context, str):
            raise ValueError("context or state must be a string")
        encoded_images = encode_images(images) if images is not None else ()
        if (schema is None) == (questions is None):
            raise SchemaError("provide exactly one of questions or schema")
        if questions is not None:
            if not isinstance(questions, Mapping):
                raise SchemaError("questions must be a mapping of field names to definitions")
            schema = {"type": "object", "properties": questions}
        active_mode = self.mode if mode is None else mode
        active_temperature = self.temperature if temperature is None else temperature
        _validate_decoding(active_mode, active_temperature)
        rng = self.rng if seed is None else random.Random(seed)
        decisions = self.compile_schema(schema)
        # Independent fields run together; depends_on turns the fields into a
        # graph whose layers run in order.
        run = (_execute_dependency_decisions if any(d.depends_on is not None for d in decisions)
               else _execute_batch_decisions)
        attach = self.sglang.images(encoded_images) if encoded_images else nullcontext()
        with call_scope() as scope, attach:
            try:
                rows, prompts = run(
                    self.sglang, context, decisions, active_mode,
                    active_temperature, rng, self.numeric_max_digits,
                    image_count=len(encoded_images),
                )
            finally:
                self._last_usage.set(scope.usage)
        self._last_prompts.set(prompts)

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
            _print_final_prompts(prompts)
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
    digit_count = sum(char.isdigit() for char in value_text)
    # A trailing "." at the digit limit could never be completed.
    if digit_count > max_digits or (
        digit_count == max_digits and value_text.endswith(".")
    ):
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
    if isinstance(value, float) and not math.isfinite(value):
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


def _decode_numeric_batch(
    client: SGLangClient,
    items: Sequence[tuple[str, Choice]],
    mode: str,
    temperature: float,
    rng: random.Random,
    max_digits: int,
) -> list[tuple[int | float | None, str, str]]:
    """Decode numeric fields in lockstep: one batched request per step.

    A field prefilled with {"name": first takes a sign step at the key: the model
    picks among the tokens this tokenizer starts a positive or negative number
    with (and null, when nullable), then its digits follow.
    """
    end_token_id, end_token_text = client.end_of_message_token()
    prefilled = any(d.answer_prefill for _, d in items)
    # After a prefilled {"name": the number ends where the object closes.
    close_id, close_text = client.single_token("}") if prefilled else (None, "")
    starts = client.json_value_starts() if prefilled and hasattr(client, "json_value_starts") else None
    prefixes, signing = [], []
    for prompt, decision in items:
        sign_step = bool(decision.answer_prefill and starts and starts["positive"])
        if decision.nullable and not (sign_step and starts["null"]):
            raise ValueError(f"Nullable number {decision.name!r} needs a tokenizer that starts null "
                             'and numbers with single tokens after \'{"name":\'')
        prefixes.append(prompt if sign_step or not decision.answer_prefill else prompt + " ")
        signing.append(sign_step)
    texts = [""] * len(items)
    unsigned = [False] * len(items)  # sign already chosen: no leading "-" in the digits
    outputs: list[tuple[int | float | None, str, str] | None] = [None] * len(items)
    step = 0
    while any(output is None for output in outputs):
        active = []
        for i, (_, decision) in enumerate(items):
            if outputs[i] is not None:
                continue
            if signing[i]:
                # A digit or "-" written right after the colon ({"k":7}) is valid
                # JSON too; rare, but kept so no way of starting a number is lost.
                candidates = {token: ("direct", piece, next_text) for token, (piece, next_text, _) in
                              _numeric_token_candidates(client, "", decision.numeric_type or "", max_digits).items()}
                candidates.update({token: ("positive", piece, "") for token, piece in starts["positive"]})
                candidates.update({token: ("negative", piece, "-") for token, piece in starts["negative"]})
                if decision.nullable:
                    candidates.update({token: ("null", piece, "null") for token, piece in starts["null"]})
                active.append((i, candidates, list(candidates)))
                continue
            candidates = _numeric_token_candidates(
                client, texts[i], decision.numeric_type or "", max_digits
            )
            if unsigned[i] and texts[i] == "":
                candidates = {k: v for k, v in candidates.items() if not v[1].startswith("-")}
            if _numeric_text_is_complete(texts[i], decision.numeric_type or ""):
                candidates[end_token_id] = (end_token_text, texts[i], True)
                if decision.answer_prefill:
                    candidates[close_id] = (close_text, texts[i], True)
            if not candidates:
                raise ValueError(
                    f"Could not complete numeric field {decision.name!r} within "
                    f"{max_digits} digits"
                )
            active.append((i, candidates, list(candidates)))
        if len(active) == 1:
            i, _, ids = active[0]
            by_id, meta, elapsed = client.score_candidates(prefixes[i] + texts[i], ids)
            scored = [(by_id, meta)]
        else:
            scored, elapsed = client.score_candidates_batch(
                [prefixes[i] + texts[i] for i, _, _ in active], [ids for _, _, ids in active]
            )
        for (i, candidates, ids), (by_id, meta) in zip(active, scored):
            decision = items[i][1]
            raw = {str(token_id): by_id[token_id] for token_id in ids}
            probs = candidate_softmax(
                raw, temperature if mode == "sample" else 1.0
            )
            if signing[i]:
                signing[i] = False
                null_keys = [k for k in probs if candidates[int(k)][0] == "null"]
                p_null = math.fsum(probs[k] for k in null_keys)
                # Null asks "is there a value?": compare it with all the ways a value
                # can start, not with the single most likely start. The starts are
                # added up before temperature applies, as for strings.
                if decision.nullable:
                    grouped = candidate_softmax(
                        {"null": _logsumexp([raw[k] for k in null_keys]),
                         "value": _logsumexp([v for k, v in raw.items() if k not in null_keys])},
                        temperature if mode == "sample" else 1.0)
                    choice = _choose(grouped, mode, rng)
                    LOG.info("numeric name=%s start=%s p_null=%.4f", decision.name, choice, grouped["null"])
                    if choice == "null":
                        # Either null token counts; the prompt keeps the canonical {"name": null}.
                        outputs[i] = (None, prefixes[i] + " null}", "null")
                        continue
                value_probs = {k: p / (1 - p_null) for k, p in probs.items() if k not in null_keys}
                kind, piece, next_text = candidates[int(_choose(value_probs, mode, rng))]
                # Text, not tokens, goes to the server: ' -' + digits reads as the
                # prompt ' ' + '-' + digits, which the server tokenizes naturally.
                if kind == "negative":
                    prefixes[i] += piece[:-1]
                elif kind == "positive":
                    prefixes[i] += piece
                    unsigned[i] = bool(starts["negative"])
                texts[i] = next_text
                continue
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
            texts[i] = next_text
            if finished:
                outputs[i] = (_parse_numeric_value(next_text, decision),
                              prefixes[i] + next_text + selected_piece, next_text)
        step += 1
    return outputs  # type: ignore[return-value]


def _balanced_orders(count: int) -> list[tuple[int, ...]]:
    """A balanced Latin square (Williams design) on positions 0..count-1.

    Every item takes every position equally often and follows every other item
    equally often: count orders when count is even, 2*count when it is odd.
    """
    first = [0]
    low, high = 1, count - 1
    while len(first) < count:
        first.append(low)
        low += 1
        if len(first) < count:
            first.append(high)
            high -= 1
    rows = [tuple((item + shift) % count for item in first) for shift in range(count)]
    if count % 2:
        rows += [row[::-1] for row in rows]
    return list(dict.fromkeys(rows))


def _choice_orderings(decision, rng):
    """Rebind values to fixed control labels, sampling ranks without enumeration."""
    labels = list(decision.choices)
    values = list(decision.choices.values())
    if decision.permutations == "auto" and len(values) > 1:
        # Start from a canonical order so the result does not depend on the
        # order the enum was written in, then balance positions and neighbours.
        canonical = sorted(range(len(values)), key=lambda i: json.dumps(values[i]))
        orders = [tuple(canonical[i] for i in row) for row in _balanced_orders(len(values))]
        return [(replace(decision, choices=dict(zip(labels, (values[i] for i in order))),
                         permutations=1), order) for order in orders]
    total = math.factorial(len(values))
    count = total if decision.permutations in ("all", "auto") else min(decision.permutations, total)
    if count == 1:
        return [(decision, tuple(range(len(values))))]
    if count == total:
        orders = all_permutations(range(len(values)))
    else:
        # Rejection sampling of integer ranks avoids materializing K! orders,
        # including when K! exceeds the platform's range length limit.
        ranks = set()
        ordered_ranks = []
        while len(ranks) < count:
            rank = rng.randrange(total)
            if rank not in ranks:
                ranks.add(rank)
                ordered_ranks.append(rank)
        orders = []
        for rank in ordered_ranks:
            available = list(range(len(values)))
            order = []
            while available:
                index, rank = divmod(rank, math.factorial(len(available) - 1))
                order.append(available.pop(index))
            orders.append(tuple(order))
    return [(replace(decision, choices=dict(zip(labels, (values[i] for i in order))),
                     permutations=1), order) for order in orders]


def _mean_order_probabilities(scored, orders, label_tokens, temperature):
    labels = list(label_tokens)
    aligned = {label: [] for label in labels}
    for (by_id, _meta), (_variant, order) in zip(scored, orders):
        probs = candidate_softmax({label: by_id[token_id]
                                  for label, (token_id, _) in label_tokens.items()}, temperature)
        for position, original_index in enumerate(order):
            aligned[labels[original_index]].append(probs[labels[position]])
    return {label: math.fsum(values) / len(scored) for label, values in aligned.items()}


def _execute_dependency_decisions(
    client, context, decisions, mode, temperature, rng, numeric_max_digits,
    image_count=0,
):
    rows_by_name = {}
    prompts_by_name = {}
    ancestors = {}
    for layer in dependency_layers(decisions):
        dependency_values = {}
        parent_prefixes = {}
        for decision in layer:
            visible = set(decision.depends_on or ())
            for name in decision.depends_on or ():
                visible.update(ancestors[name])
            ancestors[decision.name] = visible
            parents = decision.depends_on or ()
            if parents:
                # Longest serialized prefix is a deterministic heuristic; KV
                # from distinct branches cannot be concatenated.
                parent = max(parents, key=lambda name: len(prompts_by_name[name]))
                parent_prefixes[decision.name] = prompts_by_name[parent]
            dependency_values[decision.name] = {
                d.name: rows_by_name[d.name]["value"]
                for d in decisions if d.name in visible
            }
        rows, prompts = _execute_batch_decisions(
            client, context, layer, mode, temperature, rng, numeric_max_digits,
            dependency_values=dependency_values,
            parent_prefixes=parent_prefixes,
            image_count=image_count,
        )
        for decision, row, prompt in zip(layer, rows, prompts):
            rows_by_name[decision.name] = row
            prompts_by_name[decision.name] = prompt
    return ([rows_by_name[d.name] for d in decisions],
            [prompts_by_name[d.name] for d in decisions])


def _execute_batch_decisions(
    client: SGLangClient,
    context: str,
    decisions: Sequence[Choice],
    mode: str,
    temperature: float,
    rng: random.Random,
    numeric_max_digits: int = 32,
    dependency_values: Mapping[str, Mapping[str, Any]] | None = None,
    parent_prefixes: Mapping[str, str] | None = None,
    image_count: int = 0,
) -> tuple[list[dict], list[str]]:
    def question_content(decision):
        content = decision.opening_text()
        values = (dependency_values or {}).get(decision.name, {})
        if values:
            content = "Dependency results (JSON):\n" + json.dumps(values, ensure_ascii=False) + "\n\n" + content
        return content

    incremental = parent_prefixes is not None

    def complete(prompt, messages, answer):
        if incremental:
            return client.complete_chat_prefix(prompt, answer)
        return client.render_chat(
            messages + [{"role": "assistant", "content": answer}],
            add_generation_prompt=False,
        )

    shared_messages = [{"role": "user", "content": _user_content(context.rstrip(), image_count)}]
    shared_prefix = client.render_chat(
        shared_messages, add_generation_prompt=False
    )
    label_tokens_by_decision: list[dict[str, tuple[int, str]]] = []
    candidate_ids: list[list[int]] = []
    prompts: list[str] = []
    scoring_prompts = []
    scoring_ids = []
    ordering_groups = []

    # Warm each distinct parent once before siblings, including thinking/text
    # requests. Root context is warmed only in the first DAG layer.
    if incremental:
        prefixes = list(dict.fromkeys(parent_prefixes.values()))
        if any(d.name not in parent_prefixes for d in decisions):
            prefixes.insert(0, shared_prefix)
        for prefix in prefixes:
            client.cache_prefix(prefix)

    finite_indexes: list[int] = []
    open_results: dict[int, tuple[dict, str]] = {}
    text_pending = []
    numeric_pending = []
    # Clients that can batch thinking get every prompt of the layer, including
    # permutation variants, before any reasoning runs.
    defer = callable(getattr(client, "prepare_answer_prefixes", None))
    raw_prompts: list[str] = []

    def generation_prompt(parent, content, messages):
        if parent is not None:
            prompt = (client.extend_chat_prefix(parent, content, finish_thinking=False)
                      if defer else client.extend_chat_prefix(parent, content))
        else:
            prompt = (client.render_chat(messages, add_generation_prompt=True, finish_thinking=False)
                      if defer else client.render_chat(messages, add_generation_prompt=True))
        raw_prompts.append(prompt)
        return len(raw_prompts) - 1

    decision_slots = {}
    scoring_slots = []
    scoring_prefills = []
    for index, decision in enumerate(decisions):
        messages = shared_messages + [
            {"role": "user", "content": question_content(decision)}
        ]
        parent = (parent_prefixes or {}).get(decision.name)
        decision_slots[index] = generation_prompt(parent, question_content(decision), messages)
        if decision.text_type:
            text_pending.append((index, decision, messages))
            continue
        if decision.numeric_type is not None:
            numeric_pending.append((index, decision, messages))
            continue
        label_tokens = {
            label: client.single_token(label) for label in decision.choices
        }
        ids = [token_id for token_id, _ in label_tokens.values()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Decision {index} has labels with duplicate token IDs: {ids}")
        label_tokens_by_decision.append(label_tokens)
        candidate_ids.append(ids)
        orders = _choice_orderings(decision, rng)
        ordering_groups.append(orders)
        for variant, _order in orders:
            variant_content = question_content(variant)
            scoring_slots.append(
                decision_slots[index] if variant.choices == decision.choices else
                generation_prompt(parent, variant_content,
                                  shared_messages + [{"role": "user", "content": variant_content}]))
            scoring_ids.append(ids)
            scoring_prefills.append(decision.label_prefill)
        finite_indexes.append(index)
        LOG.info(
            "batch_decision=%d name=%s candidate_token_ids=%s",
            index,
            decision.name,
            {label: token_id for label, (token_id, _) in label_tokens.items()},
        )

    ready = client.prepare_answer_prefixes(raw_prompts) if defer else raw_prompts
    prompts = [ready[decision_slots[index]] for index in finite_indexes]
    scoring_prompts = [ready[slot] + prefill for slot, prefill in zip(scoring_slots, scoring_prefills)]
    # Open fields continue from {"name": ; history gets the closed object.
    def open_row(decision, value):
        return {"name": decision.name, "question": decision.question,
                "label": None, "value": value, "probabilities": None}

    open_pending = [(index, decision, messages, ready[decision_slots[index]])
                    for index, decision, messages in numeric_pending + text_pending]
    nullable = [item for item in open_pending if item[1].nullable and item[1].text_type]
    if nullable:
        nulls = _decide_nulls(
            client, [(prompt + decision.answer_prefill, decision) for _, decision, _, prompt in nullable],
            mode, temperature, rng)
        for (index, decision, messages, prompt), is_null in zip(nullable, nulls):
            if is_null:
                open_results[index] = (open_row(decision, None),
                                       complete(prompt, messages, _closed_answer(decision, "null")))
    open_pending = [item for item in open_pending if item[0] not in open_results]
    numeric_pending = [item for item in open_pending if item[1].numeric_type is not None]
    text_pending = [item for item in open_pending if item[1].text_type]

    if numeric_pending:
        decoded = _decode_numeric_batch(
            client,
            [(prompt + decision.answer_prefill, decision) for _, decision, _, prompt in numeric_pending],
            mode, temperature, rng, numeric_max_digits,
        )
        for (index, decision, messages, prompt), (value, _completed, generated_text) in zip(numeric_pending, decoded):
            open_results[index] = (open_row(decision, value),
                                   complete(prompt, messages, _closed_answer(decision, generated_text)))

    if text_pending:
        values = client.generate_texts(
            # From {"name": the model picks the string's first token, quote included,
            # among the same starts the null decision weighed.
            [prompt + decision.answer_prefill for _, decision, _, prompt in text_pending],
            [decision.max_length for _, decision, _, _ in text_pending],
            temperature=0 if mode == "argmax" else temperature,
            seed=rng.randrange(2**31),
            after_key=all(decision.answer_prefill for _, decision, _, _ in text_pending),
        )
        for (index, decision, messages, prompt), value in zip(text_pending, values):
            completed = complete(prompt, messages, _closed_answer(decision, json.dumps(value, ensure_ascii=False)))
            open_results[index] = (open_row(decision, value), completed)

    # Warm the exact common prefix once, then let SGLang fork the cached state
    # across the K batched prompts. TypeLLM never reads or moves KV tensors.
    if prompts:
        if not incremental:
            cache_meta = client.cache_prefix(shared_prefix)
            LOG.info("batch_shared_prefix_cached_tokens=%s", cache_meta.get("cached_tokens"))
        scored, elapsed = client.score_candidates_batch(scoring_prompts, scoring_ids)
        grouped_scores = []
        offset = 0
        for orders in ordering_groups:
            grouped_scores.append(scored[offset:offset + len(orders)])
            offset += len(orders)
        scored = [group[0] for group in grouped_scores]
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
        probabilities = _mean_order_probabilities(
            grouped_scores[finite_index], ordering_groups[finite_index],
            label_tokens, probability_temperature)
        selected = (
            max(probabilities, key=probabilities.__getitem__)
            if mode == "argmax"
            else _sample(probabilities, rng)
        )
        selected_text = label_tokens[selected][1]
        semantic_value = decision.choices[selected]
        completed_prompts.append(
            complete(
                prompts[finite_index],
                shared_messages + [{"role": "user", "content": question_content(decision)}],
                _closed_label(decision, selected_text),
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


def _print_final_prompts(prefixes: Sequence[str]) -> None:
    for index, prefix in enumerate(prefixes):
        print(f"\n===== FINAL BATCH PROMPT {index} =====")
        print(prefix)
        print(f"===== END BATCH PROMPT {index} =====")


def run_schema(
    context: str | None = None,
    schema: Mapping[str, Any] | None = None,
    mode: str = "argmax",
    temperature: float = 1.0,
    *,
    state: str | None = None,
    questions: Mapping[str, Any] | None = None,
    images: Sequence[Any] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    numeric_max_digits: int = 32,
    tokenizer: str | None = None,
    numeric_cache_dir: str | os.PathLike[str] | None = None,
    thinking: bool = False,
    thinking_budget: int | None = None,
    text_max_tokens: int = 512,
    print_final_prompt: bool = False,
) -> dict[str, Any]:
    client = TypeLLMClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
        mode=mode,
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
        images=images,
        print_final_prompt=print_final_prompt,
    )
