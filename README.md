# TypeAR: Type-Safe Decoding for Autoregressive LLMs

### Updates

- [2026/09/18] Added integer and float outputs through tokenizer-native
  constrained decoding for JSON Schema `integer` and `number` fields.

## Introduction

[TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
highlights a useful idea: software needs decisions, not more strings to parse.
TypeAR brings the same typed-decision interface to the open-source
autoregressive models you already run—without a proprietary model API, model
retraining, structured-output library, or manual KV-tensor management.

1. **No out-of-schema choices.** Every categorical decision stays inside its
   declared domain.
2. **Negligible output-token cost by default.** Categorical decisions generate
   one token; numbers use bounded, tokenizer-native constrained decoding.
3. **Linear input cost.** Prefix-cache reuse makes newly processed input grow
   approximately linearly with the unique context added across the workflow.
4. **Sequential dependencies when needed.** In sequential mode, each later
   decision is conditioned on all previous values; batch mode runs independent
   decisions concurrently.
5. **Made for open autoregressive LLMs.** TypeAR works with any compatible,
   pretrained open-source model served by SGLang.

**Open question.** Can a pretrained autoregressive model outperform Jev on
dependent decision workflows by combining its general-purpose reasoning
capabilities with explicit conditioning on every decision made so far?

## Quick start

### 1. Serve a model with SGLang

Use [SGLang](https://github.com/sgl-project/sglang) to configure and serve a
compatible autoregressive model on your local GPU server. This example uses
Qwen3.8-27B; follow the
[Qwen3.8-27B SGLang deployment guide](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
to start it with prefix caching enabled.

Install the lightweight client-side tokenizer dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run TypeAR

Point `TypeARClient` at the SGLang server's HTTP endpoint:

```python
from typear import TypeARClient

client = TypeARClient(
    "http://127.0.0.1:30000",
    model="qwen3.8-27b",
)

result = client.generate(
    context="""
    Receipt from Hilton London
    Total: £324
    Employee travelled to London for a client meeting.
    """,
    schema={
        "type": "object",
        "properties": {
            "expense_type": {
                "type": "string",
                "enum": ["meal", "travel", "equipment"],
                "question": "What type of expense is this?",
            },
            "reimbursable": {
                "type": "boolean",
                "question": "Should this expense be reimbursed?",
            },
            "confidence": {
                "type": "number",
                "enum": [0.0, 0.25, 0.5, 0.75, 1.0],
                "question": "How confident are you?",
            },
        },
        "required": ["expense_type", "reimbursable", "confidence"],
    },
)

print(result)
# {
#     "expense_type": "travel",
#     "reimbursable": True,
#     "confidence": 0.75,
# }
```

Python dictionary insertion order determines the decision order. Each later
field is conditioned on the original context and the values selected for all
earlier fields.

## Testing

To run the 128-case numeric regression, first serve `qwen3.8-27b` with SGLang
at `http://127.0.0.1:30000`, then run:

```bash
python3 _numeric_eval.py
```

The evaluation covers integer and number extraction, arithmetic, negative
integers, and sequential dependencies. Its deterministic test cases are stored
in `evals/numeric_eval_cases.jsonl`; the script writes detailed results to
`evals/numeric_eval_formal_results.jsonl` and prints an aggregate summary.

## Supported schema

TypeAR supports both finite decisions and grammar-constrained numeric fields:

| Field | Schema | Returned value |
|---|---|---|
| String choice | `{"type": "string", "enum": ["meal", "travel"]}` | `str` |
| Integer choice | `{"type": "integer", "enum": [1, 2, 3]}` | `int` |
| Number choice | `{"type": "number", "enum": [0.1, 0.5, 1.0]}` | `int` or `float` |
| Integer | `{"type": "integer"}` | `int` |
| Number | `{"type": "number"}` | `float` |
| Boolean | `{"type": "boolean"}` | `bool` |

Finite enums may contain at most 16 values.

Use `question` to tell the model what decision to make:

```python
{
    "type": "string",
    "enum": ["billing", "technical", "account"],
    "question": "Which team should handle this ticket?",
}
```

If `question` is absent, TypeAR uses the standard JSON Schema `description`,
then falls back to an instruction generated from the field name. The older
`x-question` spelling remains accepted for compatibility.

## Numeric domains

Ordinary TypeAR decisions come from finite sets because single-token control
labels can cover an enum or a Boolean. For integer and number fields without an
enum, TypeAR scans the served model's tokenizer once and caches every token that
can participate in a number. At each step it scores all tokenizer-native pieces
that legally extend the current numeric prefix. A model may therefore emit
`"5"`, `"54"`, or `"5461"` in one step without losing probability assigned to
multi-character numeric tokens.

```python
"answer": {
    "type": "number",
    "question": "What is 17.5 multiplied by 4?",
}
```

The assistant emits the number directly and may select the model's native
end-of-message token only after a valid number exists. Returned values are real
Python `int` and `float` objects, not strings. `minimum` and `maximum` are
optional standard JSON Schema constraints; when supplied, TypeAR validates the
completed value against them.
`numeric_max_digits` defaults to 32. Scientific notation is not currently
accepted. When `return_probabilities=True`, an open numeric field returns
`"probabilities": None`, because it has no finite final-value domain.

The numeric-token table is derived from the tokenizer reported by SGLang and
stored under `~/.cache/typear/numeric_tokens`. Its cache key is the tokenizer
content hash, so it is reused across questions and rebuilt after a tokenizer
change. TypeAR also uses the tokenizer's native chat template with thinking
disabled for constrained decisions. If the server reports a path that exists
only on the remote host, pass the equivalent local path or Hugging Face ID as
`TypeARClient(tokenizer="...")`.

## Sequential and batch execution

Sequential execution is the default:

A schema can express a complete decision workflow. For example, an incident
triage schema might select, in order:

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
    schema=schema,
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

Return probabilities over the allowed semantic values:

```python
result = client.generate(
    context=context,
    schema=schema,
    return_probabilities=True,
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

Argmax is the default. To sample only among the allowed values:

```python
client = TypeARClient(
    "http://127.0.0.1:30000",
    mode="sample",
    temperature=0.8,
    seed=42,
)
```

Temperature is applied to constrained candidate scores, including each step of
numeric decoding.

For a one-off request, use the convenience function:

```python
from typear import run_schema

result = run_schema(
    context=context,
    schema=schema,
    base_url="http://127.0.0.1:30000",
    model="qwen3.8-27b",
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
