# A fair die with permutation averaging

Ask Qwen the same fair-die question with the native per-field `permutations` API.
The expected probability of each outcome is 1/6. The example compares one
ordering, eight sampled distinct orderings, and all 720 orderings.

```python
from typellm import TypeLLMClient

client = TypeLLMClient("http://127.0.0.1:30000", seed=42)
result = client.generate(
    context="A single roll of a fair die.",
    questions={"roll": {
        "type": "string",
        "enum": ["one", "two", "three", "four", "five", "six"],
        "instructions": "what number will come up on a single roll of a fair six-sided die?",
        "permutations": "all",
        "return_probabilities": True,
    }},
)
print(result["roll"])
```

Only explicit enum questions support `permutations`. The client handles
reordering, batch scoring, probability alignment, and averaging; the return
format remains `{value, probabilities}`. `1` preserves the original ordering,
while `8` samples eight distinct permutations.

## Run

Install TypeLLM from this checkout (`pip install -e .`) and serve
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` through SGLang with the served name
`qwen3.8-27b`. From the repository root:

```bash
python examples/fair_die/fair_die.py --url http://127.0.0.1:30000
```

Use `--model`, `--tokenizer`, and `--output` to override the defaults.
The script writes `result.json`, including probabilities, KL(P || Q), model
configuration, and elapsed time. P is uniform; lower KL is better. A `null` KL
means the prediction contains a zero probability, giving infinite divergence.

The 8-permutation result is one seeded sample, not an average over many subsets.
This example tests one known distribution; it does not establish calibration
across tasks. Each run uses one `generate` call, which can issue multiple backend
requests.

## Recorded GPU result

Run on September 23, 2026, using the model and settings above.

| Permutations | one | two | three | four | five | six | KL(P ∥ Q) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75.97% | 1.79% | 13.20% | 6.24% | 1.58% | 1.23% | 1.14973 |
| 8 | 41.36% | 20.07% | 17.89% | 3.97% | 2.25% | 14.47% | 0.40247 |
| 720 (all) | 27.08% | 13.83% | 14.76% | 13.83% | 13.24% | 17.27% | 0.03406 |
| Ground truth | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 0 |

See [result.json](result.json) for full precision. This run uses a single field named `roll` for every ordering. Earlier experiments used separate `order_N` fields, so their prompts and numerical results differ. The 720-order mean is much closer to uniform, but still favors `one`.
