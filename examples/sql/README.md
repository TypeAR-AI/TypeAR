# Minimal typed SQL generation

Read a SQLite database schema, turn its table/column names into TypeLLM enums,
then generate a query plan and compile it to a parameterized SELECT.

## Run without a GPU

From the repository root:

```bash
python examples/sql/sql_generator.py --demo --execute
```

This uses **fixed fixture responses, not a model**. It exercises the full schema
conversion/compiler path against an in-memory database with Alice (25), Bob (17),
and Carol (31). It prints the database schema, each generated TypeLLM question
schema and result, the plan, SQL, parameters, and returned rows:

```sql
SELECT "name" FROM "users" WHERE "age" > ? LIMIT ?
```

```json
{"params": [18, 100], "rows": [["Alice"], ["Carol"]]}
```

## Run with TypeLLM

Start a compatible SGLang server and install the repository dependencies first.
Omit `--demo` to make real model calls:

```bash
python examples/sql/sql_generator.py \
  --model Qwen/Qwen3.8-27B \
  --question "Find the names of users older than 18." \
  --execute
```

To use an existing database:

```bash
python examples/sql/sql_generator.py \
  --db ./app.sqlite \
  --url http://127.0.0.1:30000 \
  --question "Find the names of users older than 18."
```

The database is opened read-only. SQL is only executed when `--execute` is passed.
`--limit` defaults to 100 and accepts 1–1000. `--thinking` enables reasoning.
Omitting `--model` lets the client discover the served name. Model calls receive
the user question and schema metadata, not database rows.

## How schema conversion works

1. Table names → `table: {type: string, enum: [...]}`.
2. Selected table's supported columns → `select_column` enum; `has_filter` boolean.
3. If needed, selected table's columns → `filter_column` enum.
4. Selected column type → permitted operator enum (text: equality/inequality;
   numeric: also ordered comparisons; both support explicit NULL checks).
5. Selected column type → comparison `value` type: integer, number, or string.
   NULL checks need no value generation.
6. Validate the plan; quote whitelisted identifiers; map operators to fixed SQL;
   pass literal values and the limit separately as SQLite parameters.

Dependent stages run sequentially; independent fields within a stage use
TypeLLM's `execution="batch"`. All dynamic question schemas are visible in `trace`.
The compiled schema is a snapshot: regenerate if the database schema changes.

## Scope and limits

- SQLite only; one table, one output column, zero or one WHERE condition.
- Supported declarations: INTEGER/INT/BIGINT/SMALLINT → integer;
  REAL/FLOAT/DOUBLE → number; TEXT → string. Other columns are excluded.
- No joins, aggregates, sorting, multiple conditions, date/decimal handling,
  wildcard matching, views, or arbitrary SQL fragments. Output order is unspecified.
- Table and column candidate counts must fit TypeLLM's enum limit.
- SQLite can store values inconsistent with declared types. This example validates
  query construction and bound values, not the types of existing database rows.
- Unsupported natural-language requests are not reliably detected: a model may
  produce a legal plan that answers a different question. Review the plan before
  execution; type/domain constraints do not guarantee semantic correctness.
- `--execute` also bounds SQLite VM work; large queries may be interrupted.

Local regression tests (no model calls):

```bash
python -m unittest discover -s examples/sql -p 'test_*.py'
```
