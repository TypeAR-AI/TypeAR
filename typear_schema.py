"""Compilation for TypeAR's schema subset and bounded open values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MAX_ENUM_CHOICES = 16


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    """Tokenizer-independent decision compiled from ordinary JSON Schema."""

    name: str
    question: str
    choices: tuple[Any, ...]
    syntax: str = "Choice"
    numeric_type: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None


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
    """Compile an ordered JSON Schema object into TypeAR decisions."""
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

        instructions = field.get("instructions")
        if "instructions" in field and not isinstance(instructions, str):
            raise SchemaError(f"instructions for {name!r} must be a string")
        for old_key in ("question", "x-question"):
            if old_key in field:
                raise SchemaError(f"{old_key} for {name!r} is no longer supported; use instructions")
        description = field.get("description")
        if "description" in field and not isinstance(description, str):
            raise SchemaError(f"description for {name!r} must be a string")
        question = (
            instructions
            if instructions is not None
            else description
            if description is not None
            else f'Choose the value for "{name}".'
        )

        field_type = field.get("type")
        enum = field.get("enum")
        if "x-score" in field:
            raise SchemaError(
                f"x-score for {name!r} is not supported; use a number enum"
            )
        if "x-other" in field:
            raise SchemaError(
                f"x-other for {name!r} is not supported; use a closed enum"
            )
        if field_type == "boolean":
            values = [True, False] if enum is None else enum
            if not isinstance(values, list) or not values:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if any(type(value) is not bool for value in values):
                raise SchemaError(f"boolean enum for {name!r} may contain only booleans")
            syntax = "Bool"
        elif field_type in {"integer", "number"} and enum is None:
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            for keyword, bound in (("minimum", minimum), ("maximum", maximum)):
                if bound is not None and not (
                    type(bound) in {int, float} and math.isfinite(float(bound))
                ):
                    raise SchemaError(
                        f"{keyword} for {name!r} must be a finite number"
                    )
            if minimum is not None and maximum is not None and minimum > maximum:
                raise SchemaError(
                    f"minimum for {name!r} must not exceed maximum"
                )
            decisions.append(
                Decision(
                    name=name,
                    question=question,
                    choices=(),
                    syntax="Integer" if field_type == "integer" else "Number",
                    numeric_type=field_type,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
            continue
        elif field_type in {"string", "integer", "number"}:
            if enum is None:
                raise NotImplementedError(
                    f"property {name!r} has type {field_type!r} without a finite enum"
                )
            if not isinstance(enum, list) or not enum:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if len(enum) > MAX_ENUM_CHOICES:
                raise SchemaError(
                    f"enum for {name!r} has {len(enum)} values; "
                    f"the maximum is {MAX_ENUM_CHOICES}"
                )
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

        if len(values) > MAX_ENUM_CHOICES:
            raise SchemaError(
                f"enum for {name!r} has {len(values)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        if _has_duplicates(values):
            raise SchemaError(f"enum for {name!r} contains duplicate values")
        decisions.append(
            Decision(name, question, tuple(values), syntax)
        )

    return decisions
