# Dependency execution: live GPU verification

Date: 2026-09-22 UTC. Source commit: `fb9ccc6c8e2f8bace1e3a1edc1b5a066fb141c08`.
Seven remote source hashes match the local TypeLLM modules (see manifests).

## Environment

- Existing `sglang-qwen38-27b-g4` VM, us-central1-b.
- One NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.178.04.
- Model: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`.
- SGLang 0.5.19; client Transformers 5.17.0.
- Existing server: FlashInfer, float32 Mamba state, max running requests 16,
  chunked prefill 2048, static memory fraction 0.80.
- Thinking cases use a 512-token per-field budget. Thinking remains stochastic.
- GPU was started for this test and stopped successfully after downloading all
  results. No model code or serving configuration was changed.

## Results

Remote unit tests: **61/61 passed**.

| Live case | Thinking off | Thinking on |
|---|---|---|
| Diamond, 24 candidates at each numeric node | FAIL | FAIL |
| Integer → text → Boolean chain | PASS | PASS |
| Select 24th candidate → Boolean | PASS | PASS |
| Default independent batch | PASS | PASS |
| Diagnostic diamond, two candidates per numeric node | PASS | PASS |
| Diagnostic standalone 24-candidate selection of 9 | PASS | PASS |
| Diagnostic standalone 24-candidate selection of 17 | PASS | PASS |

Initial matrix: **6/8** correct. Additional diagnostics: **6/6** correct.
All 18 inspected parent/child edges preserved the complete parent token prefix,
with no boundary-token loss observed. Unrelated sentinel branch content was
absent from all inspected diamond dependency paths. No runtime exceptions.

## Actual cache evidence

For the initial diamond without thinking:

| Request | Prompt tokens | Cached tokens |
|---|---:|---:|
| Root decision | 1169 | 960 |
| Left child | 1344 | 1152 |
| Right child | 1344 | 1152 |
| Join/final decision | 1533 | 1344 |

The original context was 1012 tokens. Cache reuse at child and join requests
therefore extends beyond the original context. Shared-token prefixes at the
application level were 1172 tokens for root→child and 1347 for parent→join;
server cache hits need not include the final partial cache page.

With thinking, the initial diamond's child reasoning requests each reused 1216
tokens and the join reasoning request reused 1472. This confirms parent-prefix
reuse also occurs before generating child reasoning, not merely during final
candidate scoring.

This is a warm-server smoke run, not a cache-disabled comparison or a latency
benchmark. Cache was not flushed between cases. No speedup is claimed.

## Failures and interpretation

Without thinking, the 24-candidate diamond selected root=7, left=8, right=9,
then returned final=16 instead of 17. The final prompt contained all the correct
dependency values.

With thinking, the right node returned 8 instead of 9. Its generated reasoning
computed 9 but associated it with label I, while the actual mapping was I→8,
J→9. The join correctly summed the returned values 8+8 to 16, but the workflow
answer was wrong. The dependency JSON faithfully contained the selected values.

These observations point to model answer/label-mapping errors in this setting;
they do not demonstrate a dependency transport or branch-isolation failure.
The smaller-enum diamond passed both modes, and standalone 24-candidate tests
passed. This small sample does not establish general semantic reliability or
rule out every server-side numerical/cache issue.

## Reproduction and evidence

```bash
python evals/dependency_gpu/run.py --output evals/dependency_gpu/results
python evals/dependency_gpu/run.py --diagnostics --output evals/dependency_gpu/diagnostics
```

Each output directory includes `manifest.json`, `results.jsonl`, and
`summary.json`. JSONL preserves results, exact completed prompts, token-prefix
comparisons, timings, and raw server request metadata. The test returns a
nonzero exit status for any failed case. Original failures are retained.
