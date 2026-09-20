# Qwen3.5 small-model GPU smoke tests

This suite tests TypeLLM's existing SGLang integration with the official
`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`, and `Qwen/Qwen3.5-9B` checkpoints.
It is a compatibility smoke test, not a general model-quality benchmark.

## Results (2026-09-20 UTC)

Primary runs use no explicit thinking-token budget and serial independent requests.

| Model | Thinking | Requests | Returned / type-valid | Exact matches | Errors |
|---|---|---:|---:|---:|---:|
| 0.8B | Off | 18 | 18 / 18 | 11 | 0 |
| 0.8B | On | 18 | 0 / 0 | 0 | 18 |
| 4B | Off | 18 | 18 / 18 | 18 | 0 |
| 4B | On | 18 | 7 / 7 | 7 | 11 |
| 9B | Off | 18 | 18 / 18 | 18 | 0 |
| 9B | On | 18 | 10 / 10 | 9 | 8 |

All 71 returned results across 108 primary requests satisfied the tested type/domain constraints. The 37 failures were incomplete thinking; no result was returned. Exact matches count whole requests (all four mixed fields must match). This small synthetic suite does not establish general accuracy.

The server was configured with an 8,192-token context window. Recorded 0.8B/9B reasoning metadata confirms failed thinking ended with `finish_reason: length` near that limit. These results do not establish failure at the models' native maximum context length. Successful 9B thinking also exceeded the old 2,048-token budget (for example, 2,902 and 3,488 tokens).

One 9B thinking text response was a string containing `{"answer": "blue"}` instead of `blue`: valid string type, failed exact-answer check. The 0.8B non-thinking run also made enum and text mistakes despite valid output types.

The old error message in raw logs mentions a budget even when `thinking_budget` is null; the local implementation now says thinking ended without a closing marker.

The failed concurrent runs (`9B.jsonl`, `0.8B-default.jsonl`) are excluded from this table. Their scheduler crash is an environment limitation, not an accuracy result. Per-request batch mode was still tested in the primary serial-request runs.

These GPU runs predate forced thinking closure: no automatic continuation,
forced closure, or non-thinking fallback was enabled. They are a baseline, not
GPU validation of the subsequent forced-closure implementation. That change
reserves final-answer space and closes length-truncated thinking. See the
separate [forced-closure GPU regression](forced-results/README.md) for its results.

## Method

Nine requests cover string enum, boolean, numeric enum, integer extraction,
negative integers, decimal extraction, text, text with `maxLength`, and a mixed
four-field request. Each runs in sequential and batch execution, with thinking
off and on: 36 requests per model. Choice fields also exercise field-level
probability output. Mixed requests combine wrapped and plain results.

Independent test requests run serially. An attempted four-client run on 9B and
0.8B crashed SGLang 0.5.19 in `batch_result_processor._normalize_decode_outputs`
with `AttributeError: 'list' object has no attribute 'tolist'`. Those runs are
retained for diagnosis, not used for model-accuracy claims. The replacements
are `9B-default-serial.jsonl` and `0.8B-default-serial.jsonl`.

Validation checks output keys, Python value types, finite numbers, enum
membership, text length, probability candidates and normalization. Exact-answer
checks are recorded separately. Exceptions are failures to produce a result,
not returned values violating the schema. No automatic retry is used.

The baseline comparison used `thinking_budget=None`, sent as explicit
`max_new_tokens: null` to SGLang. Omitting that transport key would instead use
SGLang's 128-token default. The current implementation instead computes the
remaining thinking space from the server context window, reserving room for
closure and the final answer. The earlier
`0.8B.jsonl` run used a 2,048-token reasoning budget; its replacement comparison
is `0.8B-default-serial.jsonl`. Do not combine the two settings.

## Environment

- NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately 96 GB VRAM.
- Existing SGLang 0.5.19 Docker environment; native BF16 checkpoints.
- Context length 8,192, maximum running requests 16, memory fraction 0.65.
- FlashInfer attention, FP32 Mamba state, chunked prefill size 2,048.
- Default TypeLLM argmax selection; text budget 128 tokens; HTTP timeout 180 s.
- Thinking uses the existing TypeLLM temperature 0.6, top-p 0.95, top-k 20.
  Thinking is stochastic; `seed=42` does not seed that reasoning request.
- Only text inputs are tested here, not image inputs.

## Reproduce

Serve one checkpoint at a time, then run from the repository root:

```bash
python evals/qwen35_small/run.py \
  --model Qwen/Qwen3.5-4B \
  --output evals/qwen35_small/4B.jsonl
```

Use `--workers 1` to run requests serially, and `--thinking-budget 2048` only
to reproduce an explicitly budgeted run.
The script writes each request as JSONL and a summary next to it. Recorded
request times include client initialization/cache effects and are not a
controlled throughput benchmark.
