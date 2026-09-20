# Forced thinking closure: GPU regression

Run date: 2026-09-20 UTC. Uses the same RTX PRO 6000, SGLang 0.5.19,
BF16 models, and 8,192-token server context as the parent report.

## Results

| Model | Requests | Returned / type-valid | Exact matches | Errors | Forced / natural closures |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 20 | 20 / 20 | 13 | 0 | 26 / 0 |
| Qwen3.5-4B | 20 | 20 / 20 | 20 | 0 | 25 / 1 |
| Qwen3.5-9B | 20 | 20 / 20 | 20 | 0 | 26 / 0 |

All 60 requests returned type-valid results; 53 matched the expected answers.
All 78 field-level context-reservation checks passed. Mixed requests contain
four fields, so closure counts exceed request counts. No implementation change
was needed during this GPU run.

The six default-budget requests also returned successfully. Five exhausted the
computed reasoning allowance (7,966 or 7,970 tokens), then used forced closure.
Their resulting prompts were 8,048 tokens: 128 answer tokens plus 16 spare tokens
remained in the 8,192-token context. One 4B request closed naturally after 2,483
reasoning tokens. The default-budget 9B requests each took approximately 95 s;
forced closure does not make long reasoning fast.

0.8B produced seven semantically wrong answers despite legal types, including
classifying a train ticket as `meal`, choosing `0.0` instead of `0.5`, and returning
`apple` instead of `apples`. Forced closure addresses incomplete reasoning, not
model accuracy. The previous baseline has different reasoning limits and test
coverage, so these counts are not a controlled accuracy comparison.

`4B-startup-not-ready.jsonl` records an initial attempt before the server was
listening (20 connection errors). It is excluded from the table. The formal run
waited for a successful health check before testing.

## Method

Each model runs 20 requests with thinking enabled:

- 18 requests: the nine parent-suite cases in sequential and batch execution,
  with an explicit 32-token reasoning budget to deliberately trigger truncation.
- Two requests: the expense enum case in both execution modes with
  `thinking_budget=None`, using the available context instead of a fixed budget.
- Independent requests run serially. Probability output, finite numeric values,
  enum domains, Python value types, text lengths, and mixed outputs are checked.
- Every completed reasoning prompt is checked against the discovered server
  context limit with 128 tokens reserved for the final answer.

The 32-token setting is a regression-test input, not a changed library default.
No retry or non-thinking fallback is used. Natural closing remains supported.
Answers are checked separately from type validity; an enum can be legal and wrong.
These simple synthetic cases are not a general accuracy benchmark, and the two
default-budget requests do not cover every output type at the context boundary.

Run from the repository root against a ready matching model server:

```bash
python evals/qwen35_small/forced_closure.py \
  --model Qwen/Qwen3.5-4B --output results/4B.jsonl
```

Raw JSONL records include request settings, reasoning finish reasons, closure
counts, reserved-context checks, returned values, and per-request validation.
Numeric probability keys become strings in JSON; type/domain checks run on the
original Python objects before serialization.
