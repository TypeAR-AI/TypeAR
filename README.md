

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

- **[2026/09/22]** Added `depends_on` dependency-graph execution with incremental
  parent-prefix reuse.

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
4. **Dependency-aware execution** — Run decisions sequentially, batch independent fields, or declare `depends_on` to form a dependency graph.
5. **Made for open autoregressive LLMs** — Use compatible models you already serve with SGLang.
6. **Supports thinking mode** — Enable reasoning before the final constrained answer.

## Quick start

### 1. Serve a model with SGLang

Use [SGLang](https://github.com/sgl-project/sglang) to configure and serve a
compatible autoregressive model on your local GPU server. This example uses
Qwen3.8-27B; follow the
[Qwen3.8-27B SGLang deployment guide](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
to start it with prefix caching enabled.

See [Supported models](#supported-models) for tested checkpoints and thinking behavior.

Serve the chosen checkpoint with SGLang and use the same model ID in the client:

```python
from typellm import TypeLLMClient

client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="Qwen/Qwen3.8-27B",  # See the Supported models section.
)
```

Install the lightweight client-side tokenizer dependencies:

```bash
pip install -r requirements.txt
```

Or simply:

```bash
pip install typellm
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

Without `depends_on`, fields run independently in batch by default. With
`depends_on`, dependencies determine execution order. Returned keys follow
Python dictionary insertion order in either mode. To condition each field on
all earlier results, explicitly set `execution="sequential"`.

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
thinking block, and proceeds with constrained decoding. The same recovery applies
when nonempty reasoning ends at a recognized native EOS/turn terminator before
the thinking-close marker: TypeLLM removes the trailing terminator if present,
closes the thinking block, and continues typed decoding without rerunning parents.
Forced closure is logged at INFO level; it does not guarantee answer accuracy.
Empty unfinished reasoning, unknown stops, and server/network errors still fail.

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

Enum choices support `string`, `integer`, and `number` types, with at most 24 values. The declared `type` validates the candidate values.
The tokenizer must provide enough distinct single-token control labels; the
default pool uses A–Z and 0–9, with 24 candidates normally mapped to A–X.

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

## Dependency-aware execution

The default `execution="auto"` uses batch execution when no field declares
`depends_on`, and dependency execution otherwise. You can also set the mode on
the client or pass it to `run_schema`.

To use sequential execution, set `execution="sequential"`. It works as follows:

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
per field. Each branch appends only its own question. Enum and Boolean fields select one
token; open numeric and text fields can generate multiple tokens.
The completed branch prompts are available as `client.last_prompts`.

Use `sequential` when later decisions depend on earlier values. Use `batch`
only when every field may be decided independently from the shared context.
For actual concurrent execution, configure the SGLang server with
`--max-running-requests` at least as large as the desired number of branches.

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

This runs `system`, then `severity` and `deployment_related` in one layer,
then `rollback`. Each layer finishes before the next starts. Enum/Boolean fields
use native batch scoring; text fields use batched text generation. Open numeric
fields currently decode individually within the layer.

- `depends_on` is a list of unique field names. Missing or empty lists denote
  independent roots once dependency execution is active.
- Each field sees the original context and its direct/transitive dependencies.
  It inherits one parent branch's conversation and receives dependency values as
  JSON. Parent thinking is retained only when the model protocol permits it. Unrelated branches are
  excluded. Probability-returning dependencies contribute their selected value,
  not their probability distribution.
- Unknown names, self-dependencies, duplicate dependencies, and cycles raise
  `SchemaError` before tokenizer binding or inference.
- Any explicit `depends_on`, including `[]`, activates dependency execution in
  `auto` mode. Use `execution="dag"` to request it explicitly; with no edges,
  all fields are independent roots.
- Explicit `execution="sequential"` or `"batch"` with `depends_on` raises
  `SchemaError`, so dependency declarations are never silently ignored.
- Both `questions=` and object-form `schema.properties` support this TypeLLM
  extension. The legacy list-form schema does not support it.
- Results and `client.last_prompts` follow field declaration order;
  `client.last_prompt` is `None`. `print_final_prompt=True` prints each completed
  branch prompt.

Dependencies specify ordering and visible results. They do not substitute values
into instructions, change candidate enums, or conditionally skip fields. All
fields execute; a failed layer propagates the error without starting later layers.

### Incremental prefix reuse along dependencies

Dependency execution appends the next user turn to the selected parent's
history. The completed prompt, including thinking, is preserved verbatim.
For a chain `A → B → C`, the prompt for
`B` starts with the completed prompt for `A`, and `C` extends `B`. Siblings fork
from the same parent prefix. Chat turn delimiters come from the model's tokenizer
template; unsupported append-only templates raise `SGLangError`.

At a join, TypeLLM selects the direct parent with the longest serialized prompt
(character count; ties follow `depends_on` order). It extends that prefix and
includes dependency values as JSON. KV tensors from different branches are not
merged. This is a deterministic reuse heuristic, not a token-optimal planner.

Before each layer, each distinct selected parent prefix is warmed once; the
original context is warmed for the root layer. This also prefills the chosen
answer and turn terminator if the earlier request did not cache them. Text values
are reserialized as JSON, so their answer suffix may need fresh prefill. Thinking
content already present in the chosen prefix is retained.

SGLang owns the KV cache. Actual reuse depends on token-prefix matches, cache
configuration, token/page boundaries, and eviction. Local regression tests check
exact string-prefix preservation and branch isolation; GPU cache-hit rates and
latency for this dependency path have not yet been measured. Thinking requests
and open numeric decoding still execute individually within a layer.

### Batch performance

Batch execution supports any number of independent fields, subject to the
SGLang server's concurrency and memory limits. The shared context is prefilled
once, and every field becomes a branch containing only its own question and
answer (one token for enum/Boolean fields).

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

The MiniCPM5, Ling and Ring runs used an RTX PRO 6000 Blackwell and
SGLang 0.5.19 on 2026-09-22.

### Protocol compatibility

Use the same client API with a compatible SGLang server:

```python
client = TypeLLMClient(
    "http://127.0.0.1:30000",
    model="openbmb/MiniCPM5-1B",
    thinking=True,
    thinking_budget=512,
)
```

If the server's tokenizer path is not available on the client, pass its matching
Hugging Face ID or local tokenizer directory as `tokenizer=`. The loader uses
standard tokenizer artifacts without executing custom model code.
Models without a compatible standard tokenizer remain unsupported.

## Comparison with Jev-style models

| Feature | TypeLLM | [Jev](https://docs.typesafe.ai/introduction) | [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | [system-one-open](https://github.com/mithalouni/system-one-open) | [OpenJev DeBERTa](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) |
| --- | --- | --- | --- | --- | --- |
| Enum selection | ✓ | ✓ | ✓ | ✓ | ✓ |
| Boolean decisions | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rubric scoring | Numeric enum; no dedicated Score API | Score | Score | Score | Score |
| integer/decimal type | ✓ | — | — | — | — |
| string type | ✓ | — | — | — | — |
| Enable Thinking | ✓ | — | — | — | — |
| Multi-field execution | Batch, sequential, DAG | Batch | Batch | Batch | Batch |
| Built-in field dependency graph | ✓ | — | — | — | — |
| KV prefix reuse | Shared context + dependency paths | Not disclosed | Shared context | Not documented | Not applicable |

## Acknowledgements

TypeLLM was inspired by [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), while pursuing a different goal: extending autoregressive LLMs with typed outputs and richer interaction patterns without changing their architecture or weights. It builds on their existing generation and reasoning capabilities, adding multiple output types, batch and sequential execution, and dependency-driven interactions. Thanks to the [SGLang](https://github.com/sgl-project/sglang) team for the inference infrastructure, and to the open-model community for making these models available to build on.
