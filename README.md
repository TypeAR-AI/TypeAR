<div align="center">

<img width="1500" alt="typellm-banner" src="https://github.com/user-attachments/assets/b1f2dbc6-21b7-4222-a0fb-dacfe1650797" />

# TypeLLM: LLMs with type-safe generation

<h4 align="center">
  <a href="https://typellm.ai/">Homepage</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/blog">Blog</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/docs">Docs</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/early-access">Early Access</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/contact">Contact</a>&nbsp;
</h4>
</div>

## Updates

- **[2026/09/24]** Added [image input](#image-input) for vision-language models, tested with Qwen3.8-27B.
- **[2026/09/23]** Added [JevBench results](https://github.com/TypeLLM/TypeLLM/blob/main/evals/jevbench/README.md): TypeLLM scored 195/231 without thinking and 228/231 with thinking.
- **[2026/09/23]** Added [permutation averaging](#per-question-permutation-averaging) to improve the predictive distribution. See the [blog post](https://typellm.ai/blog/fair-die).
- **[2026/09/22]** Added `depends_on` dependency graphs with incremental prefix reuse. See the [blog post](https://typellm.ai/blog/type-safe-workflow).
- **[2026/09/19]** Added optional [thinking mode](#thinking-mode) with a per-field budget.
- **[2026/09/18]** Added constrained `integer` and `number` outputs.

## Introduction

TypeLLM brings type-safe generation to existing autoregressive LLMs without changing their architecture or weights. Inspired by [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), it lets models retain their native thinking and free-form generation while producing schema-guaranteed outputs through JSON Schema. Built on [SGLang](https://github.com/sgl-project/sglang), TypeLLM also supports richer interaction patterns beyond independent typed decisions.


### Supported output types

**String · Integer · Number · Boolean · Enum choice** — See [schemas and examples](#output-types).

### Features

1. **No out-of-schema hallucinations** — Choices stay within the allowed values.
2. **Negligible output-token cost** — Single-token categorical selection and bounded numeric decoding; optional thinking adds tokens.
3. **Shared-prefix reuse** — KV caching avoids reprocessing shared context.
4. **Dependency-aware execution** — Independent fields run together; declare `depends_on` to form a dependency graph.
5. **Made for open autoregressive LLMs** — Use compatible models you already serve with SGLang.
6. **Supports thinking mode** — Enable reasoning before the final constrained answer.
7. **Image input** — Pass images to vision-language models alongside the text context. See [Image input](#image-input).
8. **Permutation averaging** — Reduce option-order bias on explicit enum questions with sampled or exhaustive orderings. See the [docs](https://typellm.ai/docs/probabilities#permutation-averaging).

### JevBench results

Evaluated on 231 public [JevBench](https://github.com/fstandhartinger/jevbench) tasks.

![Accuracy Benchmark — 231 public tasks from JevBench](https://raw.githubusercontent.com/TypeLLM/TypeLLM/main/evals/jevbench/assets/accuracy-promo-svg.png)

[Full results and all per-task answers](https://github.com/TypeLLM/TypeLLM/blob/main/evals/jevbench/README.md) · [Method and configuration](https://github.com/TypeLLM/TypeLLM/blob/main/evals/jevbench/METHOD.md)

## Quick start

### 1. Serve a model with SGLang

Use [SGLang](https://github.com/sgl-project/sglang) to configure and serve a
compatible autoregressive model on your local GPU server. This example uses
Qwen3.8-27B; follow the
[Qwen3.8-27B SGLang deployment guide](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
to start it with prefix caching enabled.

See [Supported models](#supported-models) for tested checkpoints and thinking behavior.

### 2. Run TypeLLM

```bash
pip install -U typellm
```

Point `TypeLLMClient` at the SGLang server's HTTP endpoint:

```python
from typellm import TypeLLMClient

client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",
)
```

Example request:

```python
result = client.generate(
    context="""
    Receipt from Hilton London
    Total: £324.50
    Employee travelled to London for a client meeting.
    """,
    questions={
        "merchant": {
            "type": "string",
            "instructions": "Return only the merchant name.",
        },
        "total": {
            "type": "number",
            "instructions": "Extract the total amount in GBP.",
        },
        "expense_type": {
            "type": "string",
            "enum": ["meal", "travel", "equipment"],
            "instructions": "What type of expense is this?",
        },
        "reimbursable": {
            "type": "boolean",
            "instructions": "Should this expense be reimbursed?",
        },
        "confidence": {
            "type": "number",
            "enum": [0.0, 0.25, 0.5, 0.75, 1.0],
            "instructions": "How confident are you?",
        },
    },
)

print(result)
```

Example output:

```python
{
    "merchant": "Hilton London",
    "total": 324.5,
    "expense_type": "travel",
    "reimbursable": True,
    "confidence": 0.75,
}
```

## Output types

TypeLLM supports finite decisions, numeric fields, and free text:

| Field | Schema | Returned value |
|---|---|---|
| Text | `{"type": "string"}` | `str` |
| Integer | `{"type": "integer"}` | `int` |
| Number | `{"type": "number"}` | `float` |
| Boolean | `{"type": "boolean"}` | `bool` |
| Enum choice | `{"type": "string", "enum": ["meal", "travel"]}` | Candidate type: `str`, `int`, or `float` |

Enum choices support `string`, `integer`, and `number` types, with at most 24 values. The declared `type` validates the candidate values.

A string without `enum` generates free text:

```python
result = client.generate(
    context="The train ticket is for a client meeting.",
    questions={
        "summary": {"type": "string", "instructions": "Summarize in one sentence."},
    },
)
```

Ask for a numeric answer without enumerating every possible value:

```python
result = client.generate(
    context="Calculate the requested value accurately.",
    questions={
        "answer": {
            "type": "number",
            "instructions": "What is 17.5 multiplied by 4?",
        },
    },
)

print(result)
# {"answer": 70.0}
```

Numeric answers use plain decimal notation with at most 32 digits by default;
set `TypeLLMClient(numeric_max_digits=...)` to adjust this limit.

Use `instructions` to tell the model what decision to make:

```python
{
    "type": "string",
    "enum": ["billing", "technical", "account"],
    "instructions": "Which team should handle this ticket?",
}
```

If `instructions` is omitted, TypeLLM uses `description` or an instruction
generated from the field name.

### Nullable fields

Add `"null"` to the type to allow a missing value. The field returns `None`
when the input has no value for it:

```python
result = client.generate(
    context="Read the attached receipt.",
    images=["receipt.jpg"],
    questions={
        "tip": {"type": ["number", "null"], "instructions": "Tip amount."},
        "table": {"type": ["string", "null"], "instructions": "Table number."},
        "paid_in_cash": {"type": ["boolean", "null"], "instructions": "Was the bill paid in cash?"},
        "card": {"type": ["string", "null"], "enum": ["VISA", "MASTERCARD", None],
                 "instructions": "Card network, if paid by card."},
    },
)
# {"tip": None, "table": "7A", "paid_in_cash": False, "card": None}
```

- `type` takes one type plus `"null"`. A nullable boolean adds `null` as a third
  choice. As in JSON Schema, a nullable enum returns `null` only if its `enum`
  lists `None`.
- For numbers and strings, TypeLLM weighs the probability of `null` against
  the probability of starting a value, then decodes the value. The tokens it
  compares are read from the served model's tokenizer.
- `return_probabilities` works for nullable booleans and enums, and its
  probabilities include `None`.

## Thinking mode

Thinking is off by default. Enable it when constructing the client:

```python
client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",
    thinking=True,
)
result = client.generate(context=context, questions=questions)
```

Set `thinking_budget=2048` to cap reasoning per field; no explicit budget is set
by default. If reasoning reaches its limit or ends early at a recognized turn
terminator, TypeLLM closes a nonempty thinking block and proceeds to the typed
answer. Empty unfinished reasoning and unrecognized stops raise an error.

Models with always-on thinking still reason with `thinking=False`;
`thinking_budget` applies to them too. See [Supported models](#supported-models).

## Image input

Pass images with `images=` alongside the text context. The served model must be
a vision-language model, such as `Qwen/Qwen3.8-27B`.

```python
result = client.generate(
    context="The customer says this receipt was charged twice.",
    images=["receipt.png"],
    questions={
        "total": {"type": "number", "instructions": "What is the receipt total?"},
        "paid": {"type": "boolean", "instructions": "Is the receipt marked as paid?"},
    },
)
```

Each image can be a local file path, an http(s) URL, a `data:` URI, raw bytes,
or a PIL image. PIL images are sent as PNG; modes PNG cannot store, such as
CMYK, are converted to RGB, and float (`F`) images are rejected. Local files
are read by the client, so the SGLang server does not need access to your
filesystem. Images come before the text in the first
user turn, and every request in the call carries them, for independent and
dependent fields, permutations and thinking.

TypeLLM reads the image placeholder from the model's chat template. If the
template does not render image content, `generate()` raises an error before
sending any request.

## Dependency-aware execution

Fields without `depends_on` run together, and each sees only the original
context. When a field needs earlier results, declare `depends_on`; TypeLLM then
runs the fields as a dependency graph.

Independent fields share the cached context across branches. Configure SGLang's
`--max-running-requests` for the desired concurrency.

### Dependency execution (`depends_on`)

Declare which earlier results a field needs. Forward references are allowed:
fields do not need to be declared in execution order.

```python
result = client.generate(
    context="The payments service is returning errors after a deployment.",
    questions={
        "system": {
            "type": "string",
            "enum": ["payments", "accounts", "search"],
            "instructions": "Which system is affected?",
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "instructions": "Assess severity for the affected system.",
            "depends_on": ["system"],
        },
        "deployment_related": {
            "type": "boolean",
            "instructions": "Is the incident related to a deployment?",
            "depends_on": ["system"],
        },
        "rollback": {
            "type": "boolean",
            "instructions": "Based on the incident assessments, should we roll back?",
            "depends_on": ["severity", "deployment_related"],
        },
    },
)
```

This runs `system`, then `severity` and `deployment_related`, then `rollback`.
Each layer finishes before the next starts; unrelated branches remain separate.

- `depends_on` lists unique field names. Missing or empty lists mark independent roots.
- Fields receive their direct and transitive dependency results. Probability-returning
  dependencies contribute only their selected value.
- Unknown names, self-dependencies, duplicates, and cycles raise `SchemaError`.
- Both `questions` and object-form `schema.properties` support dependencies.
  Returned keys follow field declaration order.

All fields execute. Dependencies do not change enums, substitute values into
instructions, or conditionally skip fields. A failed layer stops subsequent layers.

### Incremental prefix reuse along dependencies

TypeLLM extends parent prompts along dependency paths, retaining previous answers
and reasoning for KV cache reuse. In a chain `A → B → C`, each step builds on the
previous prefix; independent branches share their common prefix.

When a field depends on multiple parents, TypeLLM reuses one parent prefix and
includes all dependency results. SGLang manages the cache; KV tensors from
different branches are not merged.


## Probabilities and sampling

Set `return_probabilities` on individual enum or boolean fields:

```python
result = client.generate(
    context=context,
    questions={
        "expense_type": {
            "type": "string",
            "enum": ["meal", "travel", "equipment"],
            "return_probabilities": True,
        },
    },
)
```

```python
{
    "expense_type": {
        "value": "travel",
        "probabilities": {
            "meal": 0.04,
            "travel": 0.93,
            "equipment": 0.03,
        },
    }
}
```

Only opted-in fields return `value` and `probabilities`; other fields return plain values.
The option is not supported on open Numeric or Text fields.

Argmax is the default. To enable sampling:

```python
client = TypeLLMClient(
    "http://127.0.0.1:30000",
    mode="sample",
    temperature=0.8,
    seed=42,
)
```

Sampling applies to finite candidates for Choice fields and to token generation
for Numeric and Text fields. `temperature` controls sampling in each case.

For a one-off request, use the convenience function:

```python
from typellm import run_schema

result = run_schema(
    context=context,
    questions=questions,
    base_url="http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",
)
```

### Per-question permutation averaging

Add `permutations` to an `enum` question to reduce option-order bias. TypeLLM averages the probabilities and keeps the same return format.

```python
result = client.generate(
    context="A single roll of a fair die.",
    questions={"roll": {
        "type": "string",
        "enum": ["one", "two", "three", "four", "five", "six"],
        "instructions": "What number will come up on this roll?",
        "permutations": 8,
        "return_probabilities": True,
    }},
)
```

Use `8` for eight distinct orderings or `"all"` for every ordering (up to 720). Omit it or use `1` to keep the original behavior. Only explicit `enum` fields support this option.

[Docs](https://typellm.ai/docs/probabilities#permutation-averaging) · [Read the blog](https://typellm.ai/blog/fair-die)

## Cost analysis

Enum and boolean fields use one output token each. Numeric, text, and optional
thinking outputs use multiple tokens.

For a chain of `D` dependent fields, `C` original context tokens, and
roughly `S` new tokens per field, input prefill counts are:

```text
without prefix reuse: O(D*C + D^2*S)
with prefix reuse:    O(C + D*S)
```

For independent fields, the shared context is prefilled once, followed by
each field's question. These counts describe input token positions, not GPU
compute or latency: new tokens still attend to the cached prefix, and reuse
depends on cache availability. Hosted billing depends on the provider's
cached-input pricing.

## Supported models

The following models have been tested with TypeLLM on a live SGLang GPU
server.

| Model / checkpoint | Thinking support |
| --- | --- |
| `Qwen/Qwen3.8-27B` | On / off |
| `Qwen/Qwen3.5-0.8B/4B/9B` | On / off |
| `openbmb/MiniCPM5-1B` | On / off |
| `inclusionAI/Ling-mini-2.0` | Off only |
| `inclusionAI/Ring-mini-2.0` | Always on |

Other sizes in the Qwen3.5 and Qwen3.8 families are expected to be compatible.

[Image input](#image-input) has been tested with `Qwen/Qwen3.8-27B`.

The MiniCPM5, Ling and Ring runs used an RTX PRO 6000 Blackwell and
SGLang 0.5.19 on 2026-09-22.

Use the checkpoint ID as `model=`. If the server's tokenizer path is unavailable
locally, set `tokenizer=` to its matching Hugging Face ID or local directory.
The tokenizer must load from standard artifacts without custom model code.

## Comparison with Jev-style models

| Feature | TypeLLM | [Jev](https://docs.typesafe.ai/introduction) | [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | [system-one-open](https://github.com/mithalouni/system-one-open) | [OpenJev DeBERTa](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) |
| --- | --- | --- | --- | --- | --- |
| Enum selection | ✓ | ✓ | ✓ | ✓ | ✓ |
| Boolean decisions | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rubric scoring | Numeric enum; no dedicated Score API | Score | Score | Score | Score |
| integer/decimal type | ✓ | — | — | — | — |
| string type | ✓ | — | — | — | — |
| Enable Thinking | ✓ | — | — | — | — |
| Image input | ✓ | Not documented | — | Not documented | — |
| Multi-field execution | Batch, DAG | Batch | Batch | Batch | Batch |
| Built-in field dependency graph | ✓ | — | — | — | — |
| KV prefix reuse | Shared context + dependency paths | Not disclosed | Shared context | Not documented | Not applicable |

---
© 2026 TypeLLM
