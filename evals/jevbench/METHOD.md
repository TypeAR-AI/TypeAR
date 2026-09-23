# Method and configuration

## Model and runtime

- Model: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, using NVFP4 weights and a BF16 language-model head.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Serving runtime: SGLang 0.5.19.
- No benchmark-specific training. Both runs use identical TypeLLM and compatibility-mapping source hashes.
- Source hashes and client dependencies: [no-thinking manifest](no_thinking/manifest.json) and [thinking manifest](thinking/manifest.json).

## Tasks and inputs

The evaluation covers 231 public tasks: 72 Original, 48 Easy and 111 Hard.
There are 139 Choice, 74 Noul and 18 Score questions. No full-534 composite
or official leaderboard score is claimed.

- Upstream revision: [`f8ce71361165846101d02ebc83ad44e47ae44fc3`](https://github.com/fstandhartinger/jevbench/tree/f8ce71361165846101d02ebc83ad44e47ae44fc3).
- Dataset hash: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Native criteria order is preserved; no permutation averaging is applied.
- Choice maps to a string enum, Noul to Boolean, and Score to an integer enum with rubric descriptions.
- State and question fields are model inputs. Reference answers are used only for scoring.

Task bodies and benchmark code remain in the pinned upstream repository.
The answer files link to the exact source lines for each public task.

## Decoding

Classification uses argmax with probability temperature 1.0. Returned probabilities
are normalized constrained label-token probabilities, rather than generated
probability text or scores from a separately trained decision head.

Thinking uses `thinking=True, thinking_budget=None`, temperature 0.6, top-p 0.95
and top-k 20. Available model context still bounds generation. All 231 scored
thinking responses closed naturally; none stopped at a length limit.

Thinking is sampled: the local TypeLLM seed of 42 does not seed server-side
reasoning. These results are one run per setting, not a repeated-run variance
estimate or a claim of statistical significance.

## Scoring and answer fields

Accuracy uses the highest-probability label, with ties resolved to the
lexicographically smallest label under the upstream scorer. Probability vectors
must contain exactly the task's labels. All 231 vectors in each run passed the
strict sum tolerance of 0.001 without renormalization.

For Score questions, the typed answer is the probability-weighted expected
level; accuracy uses the most probable level. Ordinal MAE evaluates the expected
level separately. Choice/Score confidence is the selected-label probability.

Answer records include the reference label, selected label, typed answer, complete
probability vector, validity, correctness, latency, thinking/answer token counts,
generation termination metadata and a source-raw-artifact hash.

| Metric | No thinking | Thinking |
|---|---:|---:|
| Strict-valid probability vectors | 231/231 | 231/231 |
| Brier, lower is better | 0.240853 | 0.029395 |
| Top-label ECE, lower is better | 0.052494 | 0.016979 |
| Score ordinal MAE | 0.272135 | 0.000002 |

## Timing and token accounting

Tasks run serially. Timing includes local client work and network round trips
through an SSH/IAP tunnel, so it is not a measure of GPU inference alone.
The no-thinking run had one unrelated refund connectivity example before
evaluation. The thinking run reused the running service and cache; this is not
a matched cold-cache latency experiment.

Token totals cover the scored responses: 212,255 thinking tokens and 231 answer
tokens with thinking, versus 231 answer tokens without thinking. They are not a
measurement of total GPU consumption or billed cost. Self-hosted cost is unmetered.

## Verification and provenance

From the repository root:

```bash
python evals/jevbench/verify.py
# Also compare labels and reference answers with the pinned upstream checkout:
python evals/jevbench/verify.py --benchmark /path/to/jevbench
```

The verifier checks answer shapes, probability sums, exact-label argmax scoring,
aggregate accuracy/Brier, tier scores, paired transitions, token totals and file
checksums without model calls.

[No-thinking audit](no_thinking/local_audit.json) and
[thinking audit](thinking/local_audit.json) document local replay against the
original raw artifacts. Raw transport details and reasoning traces are not
bundled. Their hashes provide provenance but do not by themselves let readers
reproduce the raw-artifact audit. Final answers and probability-based scores can
be inspected and recomputed from the public export.

`export_results.py` regenerates the result files from local artifacts and a
benchmark checkout. Checksums describe the published snapshot; they are not an
independent third-party audit.
