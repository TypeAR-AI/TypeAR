# Latest API GPU verification

Commit: `a6f97b9e576113f84b038b1b24157c2e393ae521`.

Tested an isolated git-archive snapshot of the pushed commit on the existing Qwen3.8-27B NVFP4 / RTX PRO 6000 Blackwell / SGLang service. Returned source hashes match local source. No experimental ThinkingClient adapter was used.

- Remote unit tests: 38/38 passed.
- Live API matrix: 16/16 passed across thinking on/off, sequential/batch, state/context, questions/schema. Each request included number, Boolean and enum fields.
- Full numeric suite via `generate(state=..., questions=...)` with production `thinking=True`: 128/128 correct, 144 valid output fields; no request errors or type violations. Total request time 213.87 seconds, mean 1.671 seconds per case.
- question/x-question rejection: 2/2 passed.
- Real one-token thinking-budget failure: rejected with SGLangError as expected, no output returned.
- `run_schema(state=..., questions=..., thinking=True)`: passed.

All 148 recorded checks passed. Invalid-field checks validate the client-side rejection path on the GPU host; they intentionally do not invoke model inference. The normal model requests and budget test used the live GPU. This single run establishes regression coverage, not a guarantee of semantic accuracy for all inputs.

`results.jsonl`, `summary.json`, and `manifest.json` contain the evidence. GPU remains running.
