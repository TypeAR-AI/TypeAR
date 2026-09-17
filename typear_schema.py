"""Compilation and validation for TypeAR's finite JSON Schema subset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCORE_LEVELS = tuple(round(index / 10, 1) for index in range(11))


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    """Tokenizer-independent decision compiled from ordinary JSON Schema."""

    name: str
    question: str
    choices: tuple[Any, ...]
    syntax: str = "Choice"


def _has_duplicates(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        for previous in values[:index]:
            # JSON booleans are distinct from numbers, while 1 and 1.0 denote
            # the same JSON numeric value.
            both_numbers = (
                type(value) in {int, float} and type(previous) in {int, float}
            )
            if (both_numbers and value == previous) or (
                type(value) is type(previous) and value == previous
            ):
                return True
    return False


def compile_json_schema(schema: Mapping[str, Any]) -> list[Decision]:
    """Compile an ordered JSON Schema object into finite TypeAR decisions."""
    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a mapping")
    if schema.get("type") != "object":
        raise SchemaError("root JSON Schema type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SchemaError("JSON Schema properties must be a non-empty object")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise SchemaError("JSON Schema required must be a list of field names")
    if len(set(required)) != len(required):
        raise SchemaError("JSON Schema required contains duplicate field names")
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise SchemaError(
            f"required fields are missing from properties: {unknown_required!r}"
        )

    decisions: list[Decision] = []
    for name, field in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("property names must be non-empty strings")
        if not isinstance(field, Mapping):
            raise SchemaError(f"property {name!r} must be a schema object")

        explicit_question = field.get("x-question")
        if "x-question" in field and not isinstance(explicit_question, str):
            raise SchemaError(f"x-question for {name!r} must be a string")
        description = field.get("description")
        if "description" in field and not isinstance(description, str):
            raise SchemaError(f"description for {name!r} must be a string")
        question = (
            explicit_question
            if explicit_question is not None
            else description
            if description is not None
            else f'Choose the value for "{name}".'
        )

        field_type = field.get("type")
        enum = field.get("enum")
        score = field.get("x-score", False)
        if "x-score" in field and type(score) is not bool:
            raise SchemaError(f"x-score for {name!r} must be a boolean")
        if score:
            if field_type != "number":
                raise SchemaError(f"x-score for {name!r} requires type 'number'")
            if enum is not None:
                raise SchemaError(f"x-score for {name!r} must not also define enum")
            values = list(SCORE_LEVELS)
            syntax = "Score"
        elif field_type == "boolean":
            values = [True, False] if enum is None else enum
            if not isinstance(values, list) or not values:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if any(type(value) is not bool for value in values):
                raise SchemaError(f"boolean enum for {name!r} may contain only booleans")
            syntax = "Bool"
        elif field_type in {"string", "integer", "number"}:
            if enum is None:
                raise NotImplementedError(
                    f"property {name!r} has type {field_type!r} without a finite enum"
                )
            if not isinstance(enum, list) or not enum:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            values = enum
            if field_type == "string":
                valid = all(isinstance(value, str) for value in values)
            elif field_type == "integer":
                valid = all(type(value) is int for value in values)
            else:
                valid = all(
                    type(value) in {int, float} and math.isfinite(float(value))
                    for value in values
                )
            if not valid:
                raise SchemaError(
                    f"enum values for {name!r} do not match type {field_type!r}"
                )
            syntax = "Choice"
        else:
            raise NotImplementedError(
                f"property {name!r} has unsupported JSON Schema type {field_type!r}"
            )

        if _has_duplicates(values):
            raise SchemaError(f"enum for {name!r} contains duplicate values")
        decisions.append(Decision(name, question, tuple(values), syntax))

    return decisions
