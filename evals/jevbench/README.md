# JevBench public subset: TypeLLM

TypeLLM results on **231 public JevBench decisions**, evaluated on 2026-09-23.

## Comparison

| Evaluation / metric | Open-Jev 27B v1.1 | Jev 1.13.0 | GPT-5.6 Luna (none) | GPT-6 Astra (low) | TypeLLM + Qwen3.8-27B (no thinking) | TypeLLM + Qwen3.8-27B (thinking) |
|---|---:|---:|---:|---:|---:|---:|
| **JevBench public · correct / 231** | **197/231 · 85.28%** | **200/231 · 86.58%** | **206/231 · 89.18%** | **231/231 · 100.00%** | **195/231 · 84.42%** | **228/231 · 98.70%** |
| JevBench original · correct / 72 | 69/72 | 71/72 | 69/72 | 72/72 | 71/72 | 72/72 |
| JevBench easy · correct / 48 | 48/48 | 48/48 | 48/48 | 48/48 | 48/48 | 47/48 |
| JevBench hard · correct / 111 | 80/111 | 81/111 | 89/111 | 111/111 | 76/111 | 109/111 |

Both TypeLLM runs use **Qwen3.8-27B**, checkpoint
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (NVFP4 weights, BF16 language-model head).

External results are reported by [Open-Jev](https://zefan-cai.github.io/open-jev/).
TypeLLM results are from the runs documented here; model and inference settings
differ across systems.

## Answers

- [All 231 paired answers](ANSWERS.md)
- Full answers and probabilities: [no thinking](no_thinking/answers.jsonl) · [thinking](thinking/answers.jsonl)
- [Paired comparison](comparison.jsonl) · [Errors and changed outcomes](mismatches.jsonl)
- Detailed metrics: [no thinking](no_thinking/summary.json) · [thinking](thinking/summary.json)

Each answer includes its reference label, probabilities, latency and token counts.

## Configuration and performance

Both runs use the same model and task order, with argmax decisions and no
permutation averaging. Thinking has **no explicit token budget**. This is one
run per setting on the public subset, not the full 534-task benchmark.

| Metric | No thinking | Thinking |
|---|---:|---:|
| P50 request seconds | 1.296 | 6.677 |
| P95 request seconds | 2.150 | 61.083 |
| Thinking tokens in scored responses | 0 | 212,255 |
| Answer tokens in scored responses | 231 | 231 |

Thinking averaged 918.9 tokens per task. Timings include client and network
overhead. See [method and configuration](METHOD.md) for model details, scoring
and calibration metrics.

## Verify

From the repository root:

```bash
python evals/jevbench/verify.py
```

Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench) by Florian
Standhartinger and contributors · [License](https://github.com/fstandhartinger/jevbench/blob/f8ce71361165846101d02ebc83ad44e47ae44fc3/LICENSE)
· [Third-party notices](https://github.com/fstandhartinger/jevbench/blob/f8ce71361165846101d02ebc83ad44e47ae44fc3/THIRD-PARTY.md).
