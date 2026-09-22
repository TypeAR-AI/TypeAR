# Live GPU protocol validation — 2026-09-22

Historical results: Gemma 4 support has since been removed. Its results below
are retained as test evidence, not current support claims.

Subsequent client change: recognized native turn/EOS stops with nonempty unfinished
reasoning now force-close the thinking block and continue typed decoding. This
recovery has local regression coverage but has not been rerun on GPU. The results
below retain the original behavior and must not be read as GPU validation of this
later change.

All four checkpoints were served and exercised on one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), driver 580.178.04, with SGLang 0.5.19. These are small deterministic-task smoke tests, not a general model-accuracy benchmark. MiniCPM4 was excluded.

## Final results

Completed means no client exception. Type-valid includes enum membership, bounded text length and probability normalization. Exact means every field matches the fixture; a type-valid response can still be wrong.

| Model | Cases | Completed / type-valid | Exact | DAG exact | Parent-prefix checks |
|---|---:|---:|---:|---:|---:|
| `openbmb/MiniCPM5-1B` | 42 | 41/42 | 26/42 | 3/6 | 9/9 |
| `inclusionAI/Ling-mini-2.0` | 21 | 21/21 | 19/21 | 2/3 | 6/6 |
| `inclusionAI/Ring-mini-2.0` | 42 | 42/42 | 39/42 | 6/6 | 12/12 |
| `google/gemma-4-E2B-it` | 42 | 42/42 | 24/42 | 3/6 | 12/12 |

## Mode breakdown

`auto` below includes nine independent batch cases plus three dependency-graph cases. Sequential has nine cases. Ring reasons in both flag settings; Ling only supports thinking off.

| Model | Thinking flag | Execution | Exact |
|---|---|---|---:|
| minicpm5 | False | auto | 3/12 |
| minicpm5 | False | sequential | 7/9 |
| minicpm5 | True | auto | 8/12 |
| minicpm5 | True | sequential | 8/9 |
| ling | False | auto | 10/12 |
| ling | False | sequential | 9/9 |
| ring | False | auto | 10/12 |
| ring | False | sequential | 9/9 |
| ring | True | auto | 11/12 |
| ring | True | sequential | 9/9 |
| gemma4 | False | auto | 3/12 |
| gemma4 | False | sequential | 3/9 |
| gemma4 | True | auto | 11/12 |
| gemma4 | True | sequential | 7/9 |

## Findings and limitations

- **MiniCPM5:** 41/42 completed. One thinking diamond case ended with the model turn terminator before `</think>` (83 generated tokens, finish matched token 130073), so the client correctly refused a typed result. Other failures include wrong classification/numbers and unwanted text formatting. No-thinking batch quality was particularly weak.
- **Ling:** both misses were in auto mode: mixed text returned `{ ` instead of `apples`, and the 24-choice task selected 8 instead of 23. The dependent boolean correctly reflected that selected value.
- **Ring:** all six DAG cases passed. Three auto text cases returned JSON-wrapped strings instead of the requested bare word; all structured fields still met their types.
- **Gemma:** before the fix, 20 thinking cases raised a missing `<channel|>` error even though server metadata reported a stop matched on that marker. `no_stop_trim=True` did not prevent special-token filtering. Gemma reasoning requests now explicitly set `skip_special_tokens=False`; all 42 cases completed after the fix. Exact results were 6/21 without thinking and 18/21 with thinking. Remaining thinking misses: integer 32 instead of 42, text `answer` instead of `blue`, and a false boolean after the correct dependency text `seven`. Treat this checkpoint as experimental.
- The 24-choice DAG passed for MiniCPM5 and Gemma only with thinking, for Ring in both flag settings, and failed for Ling. This establishes transport/label compatibility, not reliable 24-choice selection for every model.
- Every completed diamond case kept the unrelated sibling sentinel out of the dependent histories. All 39 recorded parent-prefix checks passed. The failed MiniCPM5 diamond has no completed prompt checks and is excluded from that count.
- Prefix checks here compare serialized strings, not GPU memory identities. Server metadata records cached tokens, but warm earlier cases also contribute. There is no cold-cache control or cache-on/off benchmark. Gemma strips generated thoughts before reusing parent history; these checks concern cleaned history, not raw-thinking KV reuse. See [Google’s history guidance](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).

## Scope and reproduction

`run.py` exercises enum/probabilities, boolean, numeric enum, positive/negative integer, float, text, bounded text and mixed fields, plus a forward-reference diamond with sibling isolation, an integer→text→boolean chain, and a 24-enum→boolean chain. Fixtures/validators are imported from `../qwen35_small/run.py`. Exact expected values are in those files. Thinking budget: 512 tokens; final text limit: 128; seed: 42; context: 8192; static memory fraction: 0.75; max running requests: 8. Reasoning sampling remains temperature 0.6, top-p 0.95, top-k 20.

Start a compatible server, then run:

```sh
python evals/model_gpu/run.py --model inclusionAI/Ring-mini-2.0 --url http://127.0.0.1:30001 --output /tmp/ring-results
```

`remote.sh` records the original four-model orchestration. `typellm-gemma-retest.sh` records the Gemma-only rerun. The shell scripts use this test VM’s paths and remove only their dedicated temporary weight caches. Do not run them blindly on another host. The initial and retest logs record the actual execution times.

## Evidence

- `output/{minicpm5,ling,ring,gemma4}/results.jsonl`: results, successful prompts, per-request sampling and server metadata.
- Each model directory also has `summary.json`, `manifest.json` (module SHA-256 hashes and served model info), `server-command.json`, `server.log` and `client.log`.
- `output/gemma4-before-fix/`: retained original 22/42 completed, 6/42 exact run.
- `output/gpu.txt` and `output/image.txt`: device and immutable image digest.
- Initial remote unit suite: 89 passing tests. Local final suite: 90 passing tests, including a regression that models server filtering of the special channel marker.
- All initial model runs used one source snapshot; Gemma alone was rerun after the narrow decoder fix. Other protocols keep the default `skip_special_tokens=True` behavior. Module hashes reflect these two snapshots.

GPU model revisions match the pinned tokenizer revisions in [the CPU protocol report](../model_protocols/REPORT.md); server logs record the loaded snapshot paths. The runs completed without startup failures or GPU out-of-memory errors.

The test VM was stopped after collecting results; GCE reported `TERMINATED`.
