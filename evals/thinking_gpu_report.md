# Thinking + TypeLLM: GPU smoke evaluation

Run: 2026-09-19 UTC. This is an exploratory 20-case comparison, not a statistical accuracy benchmark or a proof covering all schemas.

## Environment

- Existing GCP instance: `sglang-qwen38-27b-g4`, `us-central1-b`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB; driver 580.178.04.
- Model: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, served as `qwen3.8-27b`.
- SGLang: 0.5.19; existing Docker service, maximum running requests 16.
- No production TypeLLM code or default configuration changed. The experiment uses a `ThinkingClient` adapter in `_thinking_eval.py`.

## Method

Each case runs with thinking off and on, alternating order. Thinking uses the model's native `enable_thinking=True` template, temperature 0.6, top-p 0.95, top-k 20, with a 1,024-token budget per field. No fixed sampling seed: this server rejects `seed` in native sampling parameters. A preliminary failed smoke request exposed this incompatibility; it was fixed before the full run.

Thinking stops at `</think>`. Only after observing a nonempty, closed thinking block does the adapter invoke the existing categorical candidate scoring or numeric constrained decoder. Incomplete thinking raises an error and returns no typed result. Subsequent fields retain only final selected labels/numbers, as in the current runtime, not earlier reasoning. The adapter assumes a native template with an open `<think>` block and explicitly rejects incompatible templates.

Coverage: 12 cases sampled from the existing deterministic numeric suite (two each of integer extraction, integer arithmetic, number extraction, number arithmetic, negative integers, sequential dependency); four categorical/Boolean/numeric-enum cases; three adversarial prompts; one mixed batch request. Each arm contains 20 requests and up to 24 output fields.

Type validity checks use exact Python types (Boolean is not accepted as integer), enum membership, finite numeric values, and exact output keys. These cases do not cover every supported schema constraint, such as numeric minimum/maximum boundaries. The adversarial enum case has no unique correct answer and is excluded from semantic accuracy. Errors count as semantic failures for cases with an expected answer, but are reported separately from returned-value type violations.

## Results

| Metric | Thinking off | Thinking on |
| --- | ---: | ---: |
| Requests | 20 | 20 |
| Requests returning a result | 20 | 19 |
| Returned fields | 24 | 23 |
| Type/domain violations among returned fields | 0 | 0 |
| Request errors | 0 | 1 |
| Semantically correct scored requests | 18/19 | 17/19 |
| Mean end-to-end request seconds, including errors | 0.239 | 3.943 |
| Thinking completion tokens, including closing markers | 0 | 4,678 |

The thinking arm made 24 reasoning calls: 23 closed normally with nonempty content, and one exhausted its budget. This confirms the experiment actually generated reasoning instead of merely flipping a template flag.

### Notable outcomes

- Adversarial enum: a prompt requested `hotel` outside `[meal, travel, equipment]`. Both arms returned the allowed value `meal`.
- Adversarial number: a prompt requested NaN/Infinity while the field asked for `2.5 × 4`. Without thinking, the result was `0.0` (type valid, semantically incorrect). With thinking, the model exhausted 1,024 tokens without closing the block; the adapter raised `RuntimeError` and returned no value. This is a completion failure, not an illegal typed output.
- Mixed batch: without thinking, the returned total was correctly `24.5`; with thinking, it was `-24.5`. Both are valid finite numbers, demonstrating that type safety does not guarantee semantic correctness.

## Interpretation and limits

No returned value violated its tested type/domain when free thinking was followed by the unchanged constrained decision stage. Thinking can therefore coexist with this mechanism in the tested setup, but it added latency and introduced a budget failure; this small run did not show an accuracy improvement. The safety mechanism is the restricted final decoder and validation, not the `<think>` tags.

Latency is descriptive of this single run, including initialization/cache effects and errors. In this experimental adapter, the batch case's reasoning calls run serially; these numbers are not an optimized thinking-batch throughput benchmark. Thinking is stochastic, cases are small, and order alternation does not completely eliminate cache effects.

## Reproduce

Run on the GPU host with the existing SGLang service listening locally:

```bash
cd ~/typellm-thinking-eval-20260919
HF_HUB_OFFLINE=1 ~/typellm-venv/bin/python -u _thinking_eval.py \
  --model qwen3.8-27b \
  --tokenizer RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  --output evals/thinking_gpu_results.jsonl
```

Files:

- `thinking_gpu_results.jsonl`: all 40 requests, schema, typed results, errors, timing and thinking completion metadata.
- `thinking_gpu_results.summary.json`: aggregate metrics.
- `thinking_gpu_environment.txt`: GPU and live model/version details.

Reasoning text is not saved; only completion/termination metadata is retained. The GPU instance was left running after this evaluation, following the request to power it on.

## Follow-up diagnosis

The invoice sign error was subsequently reproduced and traced to a prompt/grammar mismatch: `signed number` strongly encouraged `+24.5`, while the numeric grammar excluded `+`, leaving `-` above `2` in candidate scores. An explicit JSON-number instruction passed 10 targeted positive/negative/zero probes. See [thinking_diagnosis_report.md](thinking_diagnosis_report.md) for token-level evidence and limitations; the original run's measurements above remain unchanged.
