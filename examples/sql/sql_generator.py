"""Single-table, single-column SELECT generation through typed decisions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typellm import MAX_ENUM_CHOICES, TypeLLMClient

OPERATORS = {"eq": "=", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
NULL_OPERATORS = {"is_null": "IS NULL", "is_not_null": "IS NOT NULL"}


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_schema(connection: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """Only expose columns with explicitly supported declared SQLite types."""
    schema = {}
    for (table,) in connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ):
        columns = {}
        for row in connection.execute(f"PRAGMA table_info({quote_identifier(table)})"):
            name, declared = row[1], row[2].upper().strip()
            if declared in {"INTEGER", "INT", "BIGINT", "SMALLINT"}:
                columns[name] = "integer"
            elif declared in {"REAL", "FLOAT", "DOUBLE"}:
                columns[name] = "number"
            elif declared == "TEXT":
                columns[name] = "string"
        if columns:
            schema[table] = columns
    if not schema:
        raise ValueError("No supported columns found; use INTEGER, REAL, or TEXT declarations.")
    if len(schema) > MAX_ENUM_CHOICES or any(len(c) > MAX_ENUM_CHOICES for c in schema.values()):
        raise ValueError(f"This minimal example supports at most {MAX_ENUM_CHOICES} tables/columns per enum.")
    return schema


def choice(values, instructions):
    return {"type": "string", "enum": list(values), "instructions": instructions}


def operators(kind: str) -> list[str]:
    return (["eq", "ne"] if kind == "string" else list(OPERATORS)) + list(NULL_OPERATORS)


def compile_query(schema: dict, plan: dict, limit: int = 100) -> tuple[str, list[Any]]:
    """Validate independently of the model, then quote names and bind values."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    table, selected = plan.get("table"), plan.get("select_column")
    if not isinstance(table, str) or table not in schema:
        raise ValueError("Unknown table")
    columns = schema[table]
    if not isinstance(selected, str) or selected not in columns:
        raise ValueError("Unknown selected column")
    sql = f"SELECT {quote_identifier(selected)} FROM {quote_identifier(table)}"
    params = []
    predicate = plan.get("filter")
    if predicate is not None:
        column, operator = predicate.get("column"), predicate.get("operator")
        if not isinstance(column, str) or column not in columns:
            raise ValueError("Unknown filter column")
        kind = columns[column]
        if not isinstance(operator, str) or operator not in operators(kind):
            raise ValueError("Unsupported operator for this column type")
        if operator in NULL_OPERATORS:
            sql += f" WHERE {quote_identifier(column)} {NULL_OPERATORS[operator]}"
        else:
            value = predicate.get("value")
            valid = (
                (kind == "integer" and type(value) is int and -(2**63) <= value < 2**63)
                or (kind == "number" and type(value) in (int, float) and math.isfinite(value)
                    and (type(value) is not int or -(2**63) <= value < 2**63))
                or (kind == "string" and type(value) is str)
            )
            if not valid:
                raise ValueError(f"Invalid value for {kind} column")
            sql += f" WHERE {quote_identifier(column)} {OPERATORS[operator]} ?"
            params.append(value)
    sql += " LIMIT ?"
    params.append(limit)
    return sql, params


def generate_query(client, schema: dict, request: str, limit: int = 100) -> dict:
    trace = []
    def ask(stage, questions, previous):
        context = json.dumps({"request": request, "database_schema": schema,
            "selected": previous, "scope": "One table, one output column, at most one WHERE comparison. No joins, aggregates, sorting, or subqueries."}, ensure_ascii=False)
        result = client.generate(context=context, questions=questions)
        trace.append({"stage": stage, "questions": questions, "result": result})
        return result

    table = ask("table", {"table": choice(schema, "Select the table needed for the request.")}, {})["table"]
    if table not in schema:
        raise ValueError("Unknown table")
    fields = ask("columns", {
        "select_column": choice(schema[table], "Select the column to return."),
        "has_filter": {"type": "boolean", "instructions": "Does the request require a WHERE condition?"},
    }, {"table": table})
    if type(fields["has_filter"]) is not bool:
        raise ValueError("has_filter must be boolean")
    plan = {"table": table, "select_column": fields["select_column"], "filter": None}
    if fields["has_filter"]:
        column = ask("filter_column", {"column": choice(schema[table], "Select the column used in the WHERE condition.")}, plan)["column"]
        if column not in schema[table]:
            raise ValueError("Unknown filter column")
        kind = schema[table][column]
        plan["filter"] = {"column": column}
        operator = ask("operator", {"operator": choice(operators(kind), "Choose the comparison: gt means >, gte means >=, eq means =, ne means !=, lt means <, lte means <=; use is_null/is_not_null for missing values.")}, plan)["operator"]
        if operator not in operators(kind):
            raise ValueError("Unsupported operator")
        plan["filter"]["operator"] = operator
        if operator not in NULL_OPERATORS:
            plan["filter"]["value"] = ask("value", {"value": {"type": kind,
                "instructions": "Extract the literal comparison value from the request. Return only the value, without SQL syntax or surrounding SQL quotes."}}, plan)["value"]
    sql, params = compile_query(schema, plan, limit)
    return {"plan": plan, "sql": sql, "params": params, "trace": trace}


class DemoClient:
    """Fixed fixture for the bundled question; does NOT run a language model."""
    def __init__(self):
        self.answers = iter([{"table": "users"}, {"select_column": "name", "has_filter": True},
            {"column": "age"}, {"operator": "gt"}, {"value": 18}])

    def generate(self, **kwargs):
        return next(self.answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="Existing SQLite database, opened read-only; omitted: in-memory sample")
    parser.add_argument("--question", default="Find the names of users older than 18.")
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=None, help="Served model name; omitted: discover from SGLang")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Execute the compiled SELECT")
    parser.add_argument("--demo", action="store_true", help="Offline fixed-response demo; no GPU or model")
    args = parser.parse_args()
    if args.demo and (args.db or args.question != parser.get_default("question")):
        parser.error("--demo only supports the bundled database and question")
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) if args.db else sqlite3.connect(":memory:")
    try:
        if not args.db:
            connection.executescript("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER); INSERT INTO users VALUES (1, 'Alice', 25), (2, 'Bob', 17), (3, 'Carol', 31);")
        connection.execute("PRAGMA query_only = ON")
        schema = read_schema(connection)
        client = DemoClient() if args.demo else TypeLLMClient(args.url, model=args.model, execution="batch", thinking=args.thinking)
        result = generate_query(client, schema, args.question, args.limit)
        result = {"mode": "offline fixture (no LLM)" if args.demo else "live model", "database_schema": schema, **result}
        if args.execute:
            # Bound SQLite execution work as well as output size.
            connection.set_progress_handler(lambda: 1, 1_000_000)
            result["rows"] = connection.execute(result["sql"], result["params"]).fetchall()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
