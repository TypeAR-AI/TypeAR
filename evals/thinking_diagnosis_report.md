# Diagnosis: why thinking produced -24.5 instead of 24.5

Date: 2026-09-19. Same running GPU/model/service as `thinking_gpu_report.md`.

## Finding

The reproduced sign error comes from a mismatch between the instruction `Return only the signed number answer.` and the final JSON-style numeric grammar. The model strongly prefers an explicit positive sign (`+24.5`), but the allowed first-token set excludes `+`. Among the remaining extremely low-probability tokens, `-` can outrank `2`. Greedy constrained selection then appends `-`, and subsequent decoding produces `24.5`, yielding a valid but semantically wrong negative number.

This is a prompt/decoder interface problem, not evidence that the model's unconstrained answer was negative. The original failed run did not save its reasoning or logits, so its exact trajectory cannot be reconstructed. We reproduced the same failure under the same invoice prompt and original two-newline boundary in 3 of 6 new reasoning samples, with direct token-score evidence.

## Evidence

Six new thinking samples were generated for the numeric field from the mixed batch case, preserving its original two user messages and compiled question. For each fixed thinking sample, compare free greedy generation with the unchanged constrained numeric decoder. Three suffix variants were tested after `</think>`:

| Boundary | Free answer after trimming whitespace | Correct constrained values | Incorrect -24.5 values |
| --- | --- | ---: | ---: |
| Two newlines (original adapter) | +24.5 in 6/6 | 3/6 | 3/6 |
| One newline | +24.5 in 6/6 | 4/6 | 2/6 |
| No newline | +24.5 in 6/6 | 0/6 | 6/6 |

Changing whitespace changes the tail probabilities but does not remove the underlying mismatch. The no-newline free continuations first generate two newlines, which the numeric decoder also excludes.

In trial 1 with the original two-newline suffix, the exact next-token log probabilities were:

| Token | Allowed by final numeric decoder? | Log probability |
| --- | --- | ---: |
| + | No | -0.0000026226 |
| - | Yes | -18.7500 |
| 2 | Yes | -20.8750 |

The unrestricted continuation was `+24.5`. The constrained continuation was `-24.5`. The unrestricted probability of `+` was about 99.99974%; the absolute probabilities of both remaining sign/digit candidates were tiny. Conditional candidate selection does not assess whether the permitted set collectively has meaningful probability mass; it picks its highest-scoring member. For argmax, softmax does not change that ordering.

Relevant implementation:

- `typellm_runtime.py`, `Choice.opening_text`: asks for a **signed** integer/number.
- `typellm_numeric.py`, `NUMERIC_CHARACTERS`: does not include `+`.
- `typellm_runtime.py`, `_numeric_text_is_prefix` / `_numeric_transition`: accept digits and an optional leading minus, not plus.
- `typellm_sglang.py`, `score_candidates`: ignores the unrestricted generated token and reads requested candidate scores.
- `typellm_runtime.py`, `_decode_numeric`: greedily selects among grammar-valid candidates.

## Diagnostic intervention

Without changing production code, replace the numeric instruction for the probe with:

> Return only a JSON number. For zero or positive values, start with a digit and never use '+'. Use '-' only for negative values. Do not use markdown.

Keep thinking enabled and retain the original two-newline boundary and constrained decoder. Results:

- Positive 24.5: 6/6 correct; free answers now start with `24.5`, not `+24.5`.
- Negative -24.5: 2/2 correct.
- Zero: 1/1 correct.
- Negative -85.25: 1/1 correct.

All 10 free and constrained outputs agree numerically with the expected value. This supports the prompt/grammar mismatch diagnosis. It is a targeted validation, not a broad regression or a guarantee of universal accuracy. It changes the model's output-format instruction, not the grammar's safety constraints.

## Other failure in the original run

The adversarial numeric case explicitly requested NaN/Infinity while the question asked for `2.5 × 4`. With thinking disabled it returned 0.0: type valid but wrong. With thinking enabled it used all 1,024 reasoning tokens without closing `</think>` and returned an error. The tokenizer's native template also inserts an xhigh reasoning-effort instruction. These observations explain the measured termination condition; the original reasoning was not saved, so the precise reason for its long deliberation is unverified.

## Recommended change

Align integer and number instructions with the exact supported output grammar: no leading plus, minus only for genuinely negative values, no prose/markdown, and appropriate integer/decimal syntax. Keep the final constrained decoder and fail-closed budget handling. Before adopting it broadly, rerun the existing numeric suite with positive, negative and zero values, in thinking and non-thinking modes.

A separate future safeguard could detect when all allowed token probabilities are very low and retry or fail rather than return a forced low-confidence value. Such a threshold requires calibration and is not implemented here. Simply changing newline count is not a robust fix.

Only diagnostic scripts and evaluation artifacts were added; production prompts and runtime were not edited.

## Artifacts

- `_thinking_diagnose.py`: repeated matched-prefix free/constrained probe.
- `_thinking_prompt_probe.py`: explicit-JSON-format intervention.
- `thinking_diagnosis.jsonl`: captured prompts and token-level observations.
- `thinking_diagnosis.summary.json`: boundary comparisons and representative scores.
- `thinking_prompt_probe.jsonl`: all ten intervention outcomes.
