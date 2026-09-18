# TypeAR: Type-Safe Decoding for Autoregressive LLMs

[TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
highlights a useful idea: software needs decisions, not more strings to parse.
TypeAR brings the same typed-decision interface to the open-source
autoregressive models you already run—without a proprietary model API, model
retraining, structured-output library, or manual KV-tensor management.

1. **No undeclared escape.** Every decision stays inside its declared domain
   unless the schema explicitly enables an open `x-other` branch.
2. **Negligible output-token cost by default.** Closed decisions generate one
   token; open text is generated only when `x-other` is selected.
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
                "x-question": "What type of expense is this?",
            },
            "reimbursable": {
                "type": "boolean",
                "x-question": "Should this expense be reimbursed?",
            },
            "confidence": {
                "type": "number",
                "x-score": True,
                "x-question": "How confident are you?",
            },
        },
        "required": ["expense_type", "reimbursable", "confidence"],
    },
)

print(result)
# {
#     "expense_type": "travel",
#     "reimbursable": True,
#     "confidence": 0.8,
# }
```

Python dictionary insertion order determines the decision order. Each later
field is conditioned on the original context and the values selected for all
earlier fields.

## Supported schema

TypeAR currently supports finite decision spaces:

| Field | Schema | Returned value |
|---|---|---|
| String choice | `{"type": "string", "enum": ["meal", "travel"]}` | `str` |
| Integer choice | `{"type": "integer", "enum": [1, 2, 3]}` | `int` |
| Number choice | `{"type": "number", "enum": [0.1, 0.5, 1.0]}` | `int` or `float` |
| Boolean | `{"type": "boolean"}` | `bool` |
| Score | `{"type": "number", "x-score": true}` | one of `0.0, 0.1, ..., 1.0` |
| Open string choice | `{"type": "string", "enum": ["meal"], "x-other": true}` | enum value or bounded free text |

Use `x-question` to tell the model what decision to make:

```python
{
    "type": "string",
    "enum": ["billing", "technical", "account"],
    "x-question": "Which team should handle this ticket?",
}
```

If `x-question` is absent, TypeAR uses `description`, then falls back to an
instruction generated from the field name.

## Beyond finite domains

Ordinary TypeAR decisions come from finite sets because single-token control
labels can cover an enum, a Boolean, or the eleven score levels from `0.0` to
`1.0`. A typed field without a finite domain is rejected rather than guessed
at.

For a string enum, `x-other` can explicitly declare one open branch:

```python
"expense_type": {
    "type": "string",
    "enum": ["meal", "travel", "equipment"],
    "x-other": True,
    "x-question": "What type of expense is this?",
}
```

The first stage remains a constrained decision. TypeAR compiles the field to:

```text
Choice(
  name="expense_type",
  question="What type of expense is this?",
  choices={"A":"meal","B":"travel","C":"equipment","D":"other"},
  answer="
```

Selecting `A`, `B`, or `C` writes the declared value back exactly as before.
Selecting `D` opens a string value and continues generation until a closing
quote, with `other_max_new_tokens` as a safety cap:

```text
Choice(
  name="expense_type",
  question="What type of expense is this?",
  choices={"A":"meal","B":"travel","C":"equipment","D":"other"},
  answer="D",
  value="conference registration"
)
```

The escape is itself a declared single-token choice: free text is reachable
only when the model selects `other`. The generated string is then written into
the running prefix, so every later sequential decision can condition on it.
In batch mode, multiple selected `other` branches are generated as a second
native SGLang batch.

The default open generation is deterministic and capped at 64 tokens. It can
be configured independently from constrained decision sampling:

```python
client = TypeARClient(
    "http://127.0.0.1:30000",
    other_max_new_tokens=32,
    other_temperature=0.2,
)
```

When probabilities are requested, they remain probabilities over the declared
first-stage branches, including `other`; TypeAR does not assign a fabricated
probability to the subsequently generated string. The caller can also add a
generated value to a later enum, turning it into a declared one-token choice
in the next decision; TypeAR does not mutate schemas automatically.

## Sequential and batch execution

Sequential execution is the default:

A schema can express a complete decision workflow. For example, an incident
triage schema might select, in order:

1. the affected system;
2. the severity, conditioned on that system;
3. whether to roll back, conditioned on both earlier decisions;
4. a confidence score.

TypeAR appends each selected value to the running prefix before asking the next
question. The final accumulated prompt is available as `client.last_prompt`, or
can be printed with `print_final_prompt=True`.

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

Temperature is applied to the candidate scores, not to unrestricted model
generation.

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
added across the workflow.** An `x-other` branch adds free-text output cost only
when selected. The long original context is normally prefilled once rather
than recomputed for every decision.

For `D` decisions, an original context of `C` tokens, and roughly `S` newly
appended tokens per decision:

```text
without prefix reuse: O(D*C + D^2*S)
with prefix reuse:    O(C + D*S)
output generation:    O(D + E)
```

Here `E` is the total number of free-text tokens produced by selected
`x-other` branches and is zero for a fully closed schema.

For `K` independent closed batch decisions with question lengths
`Q_1, ..., Q_K`, the corresponding prefill count is approximately
`C + sum(Q_k)`, followed by one batched decode step that produces `K` output
tokens. Selected open branches add their `E` free-text tokens in a second
batched generation step.

These are prefill token-position counts, not exact GPU FLOPs. New tokens still
attend to the cached prefix, and real latency also depends on cache alignment,
context length, batching, memory bandwidth, and cache eviction.

This describes self-hosted compute. A hosted provider may still bill the full
submitted input unless it offers cached-input pricing.
