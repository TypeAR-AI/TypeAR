

<div align="center">

<img width="1500" alt="typellm-banner" src="https://github.com/user-attachments/assets/b1f2dbc6-21b7-4222-a0fb-dacfe1650797" />

# TypeLLM: LLMs with type-safe generation

<h4 align="center">
  <a href="https://typellm.ai/">Homepage</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/blog/introducing-typellm">Blog</a>&nbsp; • &nbsp;
  <a href="https://typellm.ai/contact">Contact</a>&nbsp;
</h4>
</div>





### Updates

- **[2026/09/19]** Added optional [thinking mode](#thinking-mode) with
  `thinking=True/False` and a configurable per-field thinking budget, followed
  by type-safe constrained decoding. Thinking is off by default.
- [2026/09/18] Added integer and float outputs through tokenizer-native
  constrained decoding for JSON Schema `integer` and `number` fields.

## Introduction

TypeLLM extends autoregressive LLMs with type-safe generation. Models can still think and generate freely when needed, while producing guaranteed typed outputs when structure matters. Define the output with a JSON Schema, and TypeLLM returns values your software can use directly.

### Supported output types

- **Text** — Free text (`string`).
- **Integer** — Whole numbers (`integer`).
- **Number** — Numeric values (`number`).
- **Boolean** — `true` or `false`.
- **Enum choice** — One of your allowed string or numeric values.

Enum and boolean fields select from finite candidates; numeric and text fields
without `enum` generate values token by token. See [schemas and examples](#output-types).

### Features

1. **No out-of-schema hallucinations** — Choices stay within the allowed values.
2. **Negligible output-token cost** — Single-token categorical selection and bounded numeric decoding; optional thinking adds tokens.
3. **Linear input computation cost** — Prefix caching avoids reprocessing shared context.
4. **Batch or sequential execution** — Run independent decisions together or condition on earlier results.
5. **Made for open autoregressive LLMs** — Use compatible models you already serve with SGLang.
6. **Supports thinking mode** — Enable reasoning before the final constrained answer.

## Quick start

### 1. Serve a model with SGLang

Use [SGLang](https://github.com/sgl-project/sglang) 0.5.6 or newer to configure and
serve a compatible autoregressive model on your local GPU server. This example uses
Qwen3.8-27B; follow the
[Qwen3.8-27B SGLang deployment guide](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
to start it with prefix caching enabled.

**Qwen3.5-4B and Qwen3.5-9B are also supported and GPU-tested, including thinking mode.**

Serve the chosen checkpoint with SGLang and use the same model ID in the client:

```python
from typellm import TypeLLMClient

client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",  # or "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"
)
```

Install TypeLLM (lightweight client-side tokenizer dependencies):

```bash
pip install typellm
```

Or, if you cloned this repo:

```bash
pip install -r requirements.txt
```

### 2. Run TypeLLM

Point `TypeLLMClient` at the SGLang server's HTTP endpoint:

```python
from typellm import TypeLLMClient

client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",
)

result = client.generate(
    context="""
    Receipt from Hilton London
    Total: £324
    Employee travelled to London for a client meeting.
    """,
    questions={
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
# {
#     "expense_type": "travel",
#     "reimbursable": True,
#     "confidence": 0.75,
# }
```

`questions` maps output field names to their definitions. Every field is answered.
The existing `schema=` JSON Schema interface is also supported; pass only one.
`state=` is an alias for `context=`; pass only one of them.

Python dictionary insertion order determines the decision order. Each later
field is conditioned on the original context and the values selected for all
earlier fields.

## Thinking mode

Thinking is off by default. Enable it when constructing the client:

```python
client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",
    thinking=True,          # False disables thinking (the default)
)
result = client.generate(context=context, questions=questions)
```

No thinking-token budget is set by default. Optionally pass `thinking_budget=2048`
to cap reasoning per field. TypeLLM reserves context space for the final answer;
if thinking reaches its length limit, it keeps the partial reasoning, closes the
thinking block, and proceeds with constrained decoding. Server and network errors
still propagate.

Models whose chat template always opens a `<think>` block reason before every
field even with `thinking=False`; `thinking_budget` still caps it.

## Testing

To run the 128-case numeric regression, first serve `Qwen/Qwen3.8-27B` with SGLang
at `http://127.0.0.1:30000`, then run:

```bash
python3 evals/numeric_eval.py
```

The evaluation covers integer and number extraction, arithmetic, negative
integers, and sequential dependencies. Its deterministic test cases are stored
in `evals/numeric_eval_cases.jsonl`; the script writes detailed results to
`evals/numeric_eval_formal_results.jsonl` and prints an aggregate summary.

## Output types

TypeLLM supports finite decisions, numeric fields, and free text:

| Field | Schema | Returned value |
|---|---|---|
| Text | `{"type": "string"}` | `str` |
| Integer | `{"type": "integer"}` | `int` |
| Number | `{"type": "number"}` | `float` |
| Boolean | `{"type": "boolean"}` | `bool` |
| Enum choice | `{"type": "string", "enum": ["meal", "travel"]}` | Candidate type: `str`, `int`, or `float` |

Enum choices support `string`, `integer`, and `number` types, with at most 16 values. The declared `type` validates the candidate values.

Generation works in three ways:

- **Choice** — Selects from finite candidates for enum and boolean fields.
- **Numeric** — Generates integers or numbers without `enum` token by token under numeric constraints.
- **Text** — Generates a JSON string for strings without `enum`, then decodes it to `str`.

A string without `enum` generates free text:

```python
result = client.generate(
    context="The train ticket is for a client meeting.",
    questions={
        "summary": {"type": "string", "instructions": "Summarize in one sentence."},
    },
)
```

`maxLength` is optional: add `"maxLength": 100` to limit Unicode character count.
Omitting it adds no character limit. Set `TypeLLMClient(text_max_tokens=512)` to
control the separate per-field generation budget (default 512 tokens).
Incomplete, invalid, or over-length text raises `SGLangError`.
Sequential fields can use earlier text; batch text fields generate independently.
Text generation uses
multiple tokens; type safety does not guarantee factual accuracy. Text fields
currently support `maxLength`, but not `minLength`, `pattern`, or `format`.


For example, ask for a numeric answer without enumerating every possible value:

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

Numeric fields without `enum` accept optional `minimum` and `maximum`. The bounds
are shown to the model and the generated value is validated against them; an out-of-range
value raises `ValueError` instead of being returned. These fields generate plain
decimal notation with at most 32 digits by default; set
`TypeLLMClient(numeric_max_digits=...)` to adjust this limit.

Use `instructions` to tell the model what decision to make:

```python
{
    "type": "string",
    "enum": ["billing", "technical", "account"],
    "instructions": "Which team should handle this ticket?",
}
```

If `instructions` is omitted, TypeLLM uses `description` or an instruction
generated from the field name. Rename old `question` / `x-question` fields
to `instructions`.

## Sequential and batch execution

Sequential execution is the default:

Questions can express a complete decision workflow. For example, incident
triage might select, in order:

1. the affected system;
2. the severity, conditioned on that system;
3. whether to roll back, conditioned on both earlier decisions;
4. a confidence score.

Each field becomes a new user turn, and the assistant directly emits its
single-token label or constrained numeric value. The completed turn is appended
before the next question, so later decisions see the complete decision history.
The final accumulated prompt is available as `client.last_prompt`, or can be
printed with `print_final_prompt=True`.

When the fields are independent, run them as one native SGLang batch:

```python
result = client.generate(
    context=context,
    questions=questions,
    execution="batch",
)
```

Batch execution prefills the shared context once, then forks it into one branch
per field. Each branch appends only its own question and generates one token.
The completed branch prompts are available as `client.last_prompts`.

Use `sequential` when later decisions depend on earlier values. Use `batch`
only when every field may be decided independently from the shared context.
For actual concurrent execution, configure the SGLang server with
`--max-running-requests` at least as large as the desired number of branches.

### Batch performance

Batch execution supports any number of independent fields, subject to the
SGLang server's concurrency and memory limits. The shared context is prefilled
once, and every field becomes a branch containing only its own question and
one-token answer.

As one illustrative measurement, a local run used Qwen3.8-27B NVFP4 on one
NVIDIA RTX PRO 6000 Blackwell GPU, a roughly 1,100-token shared context, and
`K=16` one-token Boolean decisions. SGLang was configured with
`--max-running-requests 16`.

| Execution | End-to-end latency | Latency per decision | Relative throughput |
|---|---:|---:|---:|
| Sequential | 9.35 s | 0.584 s | 1.0x |
| Batch | 1.61 s | 0.101 s | 5.8x |

In this run, every branch reused 1,088 cached tokens, for 17,408 reused token
positions in total. This is a single example of the general batch method, not
a fixed-width design or a portable hardware benchmark. Latency depends on
`K`, the model, questions, context length, GPU, and server configuration.

The two modes intentionally compute different conditionals. Sequential mode
includes all earlier selected values in every later prompt. Batch mode gives
each branch only the common context and its own question, which enables
parallelism but removes cross-decision dependencies.

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

## Cost analysis

**Closed decisions generate one token per field, and prefix reuse makes the
newly processed input grow approximately linearly with the unique context
added across the workflow.** An open number requires one constrained step per
generated tokenizer token. The long original context is normally prefilled once
rather than recomputed for every decision.

For `D` decisions, an original context of `C` tokens, and roughly `S` newly
appended tokens per decision:

```text
without prefix reuse: O(D*C + D^2*S)
with prefix reuse:    O(C + D*S)
output generation:    O(D + N)
```

Here `N` is the total number of generated tokenizer tokens in open numeric
fields, including their end-of-message tokens, and is zero for a fully finite
schema.

For `K` independent closed batch decisions with question lengths
`Q_1, ..., Q_K`, the corresponding prefill count is approximately
`C + sum(Q_k)`, followed by one batched decode step that produces `K` output
tokens. Open numeric fields require additional constrained token steps.

These are prefill token-position counts, not exact GPU FLOPs. New tokens still
attend to the cached prefix, and real latency also depends on cache alignment,
context length, batching, memory bandwidth, and cache eviction.

This describes self-hosted compute. A hosted provider may still bill the full
submitted input unless it offers cached-input pricing.

## License

[Apache License 2.0](LICENSE).
