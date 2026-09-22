# Model protocol CPU validation

Historical results: Gemma 4 support has since been removed. Its results below
are retained as test evidence, not current support claims.

Date: 2026-09-22. No model weights loaded and no GPU started.

Official tokenizers and templates are real; reasoning HTTP responses are simulated.
This verifies client formatting and token compatibility, not model accuracy, SGLang
deployment compatibility, JSON-constrained generation, or physical KV cache hits.

## Environment

transformers 5.17.0, tokenizers 0.23.2, jinja2 3.1.6

## Results

| Checkpoint | Pinned revision | Exact single-token labels | Numeric tokens | Modes |
|---|---|---:|---:|---|
| `openbmb/MiniCPM5-1B` | `87179e5c1f455ef22e6223592d2d61351b525bfc` | 36 | 1113 | thinking off: pass; thinking on: pass |
| `inclusionAI/Ling-mini-2.0` | `920c3fd9916e3d5e543fc4f609e827cad8a32983` | 36 | 13 | thinking off: pass; thinking on: rejected as unsupported |
| `inclusionAI/Ring-mini-2.0` | `9b973db6445b542a4844a7230addd19605044419` | 36 | 13 | thinking off: pass; thinking on: pass |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 36 | 26 | thinking off: pass; thinking on: pass |

All seven supported mode combinations passed parent/child token-prefix checks.
The unsupported Ling thinking request was rejected explicitly. Ring reasons in
both modes because its native template always opens a thinking block.

MiniCPM5 ends dialogue turns with `<|im_end|>`, not
its configured EOS `</s>`. Gemma uses `<turn|>`, not `<eos>`.

Gemma histories discard only the known generated thought block. Parent-prefix
checks compare the cleaned history with the child input, not the full raw
reasoning request. Re-prefill of the changed suffix is expected.

## Reproduce

```bash
python evals/model_protocols/probe.py --output evals/model_protocols/results.jsonl
python -m unittest discover -s tests
```

`results.jsonl` records model revisions, template hashes, labels, endings and
prefix checks; `results.manifest.json` records client source hashes and versions.

## Official references

- [MiniCPM5 template](https://huggingface.co/openbmb/MiniCPM5-1B/blob/main/chat_template.jinja)
- [Ling tokenizer](https://huggingface.co/inclusionAI/Ling-mini-2.0/blob/main/tokenizer_config.json)
- [Ring tokenizer](https://huggingface.co/inclusionAI/Ring-mini-2.0/blob/main/tokenizer_config.json)
- [Gemma 4 formatting and history policy](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
