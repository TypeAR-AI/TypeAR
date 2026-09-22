# TypeLLM × Doom

Game screenshot → Qwen VL → TypeLLM constrained actions → ViZDoom.

From the TypeLLM repository, use Python 3.10+:

```bash
python -m venv .venv
.venv/bin/pip install -r examples/doom/requirements.txt

# Preview the game without a model (random actions, explicitly labeled).
.venv/bin/python examples/doom/play.py --preview --window

# Play using an existing SGLang Qwen VL server.
.venv/bin/python examples/doom/play.py \
  --base-url http://127.0.0.1:30000 \
  --model YOUR_SERVED_VISION_MODEL \
  --tokenizer YOUR_QWEN_VL_TOKENIZER --window

# A harder arena: turn and shoot at approaching enemies.
.venv/bin/python examples/doom/play.py \
  --scenario defend_the_center \
  --base-url http://127.0.0.1:30000 \
  --model YOUR_SERVED_VISION_MODEL \
  --tokenizer YOUR_QWEN_VL_TOKENIZER --window
```

Each run writes `replay.html`, `decisions.jsonl`, and `summary.json` under
`outputs/doom/<timestamp>/`. Open the standalone replay in a browser to inspect
screenshots and actions. Omit `--window` for headless runs.

`basic` uses strafe left/right/stay plus shoot. `defend_the_center` uses turn
left/right/stay plus shoot. Both use ViZDoom's bundled scenarios and assets.
The model also selects `duration` from `[1, 2, 3, 4, 5, 6, 7, 8]` on each decision, allowing
short aiming corrections and longer searches. The duration prompt uses target
offset and recent turn effects rather than preferring the longest duration. Duration appears in the replay
and action history. For fixed-duration comparisons, pass `--frame-skip 8`.

Every decision receives the latest 10 actions in `context` and the
latest three screenshots as separate original-resolution images, with the current
frame explicitly labeled. Use `--history-frames N` to change the image window
(set it to `--steps` or higher to retain every screenshot). Use `--history-actions N` to adjust the action window (0 disables it). History resets at
the start of each episode. The exact text context and image step numbers are
saved in `decisions.jsonl`; engine rewards and hidden positions are not inputs.

The example-local adapter attaches the same ordered images to every field request within
a decision, renders the Qwen VL image placeholder through the model's native
chat template, and reuses TypeLLM's categorical scoring. It defaults to batch execution with thinking off: direction, shooting and
duration are independent decisions on the same screenshots and history.
Use `--execution sequential` to compare sequential execution. No enemy coordinates or engine state are sent to
the model. It does not fall back to random actions if inference fails.

This is synchronous gameplay: the world pauses during inference and advances
the selected `duration` tics per decision (or fewer if the episode ends). Replay playback is illustrative, not a claim
of real-time inference speed. Preview runs have no model latency measurements.
The original single-frame version was validated on SGLang 0.5.19 with `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`
on an RTX PRO 6000 GPU. The screenshot-driven arena run (seed 42) completed
60 decisions, ended in player death with reward 5, and measured 124 ms median
per decision. This is one smoke run, not a benchmark or a real-time result.
Other model/deployment combinations still need validation; text-only models
cannot run this demo.

The separate-three-image version was also run on the same deployment and seed:
49 decisions, final reward 3, ending in player death. Each input screenshot was
sent separately at its original 640×480 resolution, without client-side resizing
or compositing. The model server may apply its own vision preprocessing.

References: [ViZDoom](https://github.com/Farama-Foundation/ViZDoom),
[bundled scenarios](https://vizdoom.farama.org/environments/default/).
