"""Screenshot → TypeLLM → Doom. Run with --preview for an engine-only smoke test."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import vizdoom as vzd
from PIL import Image
from typellm import TypeLLMClient
from vision import DoomVisionClient


SHORT_THINKING = "Keep reasoning to at most two short sentences. Decide directly from the images and instructions; do not speculate about the game or benchmark source."

def questions(scenario, fixed_duration=None):
    result = {
        "move": {"type": "string", "enum": ["left", "right", "stay"],
                 "instructions": "Strafe toward the enemy to center it. Stay when aligned; search right if none is visible."},
    }
    if scenario == "defend_the_center":
        result = {
            "turn": {"type": "string", "enum": ["left", "right", "stay"],
                     "instructions": "Use the CURRENT image. Enemy left of center: turn left. Enemy right of center: turn right. Stay only when the enemy is horizontally centered on the crosshair. No visible enemy: turn right to search."},
        }
    result["shoot"] = {"type": "boolean", "instructions": "Use the CURRENT image. Shoot only when a clearly visible enemy is horizontally centered on the crosshair. If the enemy is left or right of center, no enemy is visible, or alignment is uncertain, return false."}
    if fixed_duration is None:
        result["duration"] = {"type": "integer", "enum": list(range(1, 9)),
                              "instructions": "Choose 1-8 tics (29 ms each). No visible enemy: 8 to search. Otherwise use 1-2 for fine aim, 3-5 for medium turns, 6-8 for large turns. Shorten if the previous turn overshot."}
    return result


def buttons(action, scenario):
    field = "move" if scenario == "basic" else "turn"
    if set(action) != {field, "shoot", "duration"} or action[field] not in {"left", "right", "stay"} or type(action["shoot"]) is not bool:
        raise ValueError(f"Invalid action: {action!r}")
    if type(action["duration"]) is not int or action["duration"] not in range(1, 9):
        raise ValueError(f"Invalid duration: {action['duration']!r}")
    return [int(action[field] == "left"), int(action[field] == "right"), int(action["shoot"])]


def png_data(frame):
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decision_context(history, step, frame_skip, image_steps, history_actions=10):
    actions = [{"step": row["step"], "action": row["action"]} for row in (history[-history_actions:] if history_actions else [])]
    return (
        "Play Doom. Images are oldest to newest; the last is CURRENT. "
        "Act on CURRENT; use earlier images and actions to track motion. "
        f"Step: {step}. Image steps: {image_steps}.\n"
        "Recent actions (duration in tics):\n"
        + json.dumps(actions, separators=(",", ":"))
    )


def history_image(history, current, step, count):
    past = history[-(count - 1):] if count > 1 else []
    frames = [(row["step"], row["image"]) for row in past] + [(step, current)]
    return [encoded for _, encoded in frames], [number for number, _ in frames]



def save_replay(output, records, summary):
    template = Path(__file__).with_name("replay.html").read_text()
    data = json.dumps({"records": records, "summary": summary}).replace("<", "\\u003c")
    (output / "replay.html").write_text(template.replace("/*DATA*/null", data))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--thinking", action="store_true", help="Enable model reasoning before constrained decisions")
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--execution", choices=["batch", "sequential"], default="batch")
    parser.add_argument("--model", help="Served model name")
    parser.add_argument("--tokenizer", help="Local tokenizer directory or Hugging Face ID for the served Qwen VL model")
    parser.add_argument("--scenario", choices=["basic", "defend_the_center"], default="basic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80, help="Maximum decisions per episode")
    parser.add_argument("--frame-skip", type=int, choices=list(range(1, 9)), default=None,
                        help="Override AI duration with a fixed number of tics for comparison")
    parser.add_argument("--history-actions", type=int, default=10, help="Recent actions to include; 0 disables action history")
    parser.add_argument("--history-frames", type=int, default=3, help="Recent screenshots included with the recent action history")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Random actions; no model calls, NOT an AI result")
    parser.add_argument("--output", type=Path, default=Path("outputs/doom") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.history_frames) <= 0:
        parser.error("episodes, steps, frame-skip and history-frames must be positive")
    if args.history_actions < 0:
        parser.error("history-actions must be non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    client = None
    if not args.preview:
        client = TypeLLMClient(args.base_url, model=args.model, tokenizer=args.tokenizer, execution=args.execution, thinking=args.thinking, thinking_budget=args.thinking_budget)
        client.sglang = DoomVisionClient(args.base_url, model=args.model, tokenizer=args.tokenizer, thinking=args.thinking, thinking_budget=args.thinking_budget)
    game = vzd.DoomGame()
    records, episodes = [], []
    error = None
    try:
        game.load_config(str(Path(vzd.scenarios_path) / f"{args.scenario}.cfg"))
        game.set_window_visible(args.window)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_render_crosshair(True)
        game.set_mode(vzd.Mode.PLAYER)
        lateral = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT] if args.scenario == "basic" else [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT]
        game.set_available_buttons([*lateral, vzd.Button.ATTACK])
        game.set_seed(args.seed)
        game.init()
        with (args.output / "decisions.jsonl").open("w") as log:
            for episode in range(args.episodes):
                game.new_episode()
                history = []
                for step in range(args.steps):
                    state = game.get_state()
                    if state is None:
                        break
                    screenshot = png_data(state.screen_buffer)
                    started = time.perf_counter()
                    if args.preview:
                        field = "move" if args.scenario == "basic" else "turn"
                        action = {field: rng.choice(["left", "right", "stay"]), "shoot": rng.choice([True, False]), "duration": args.frame_skip or rng.choice(list(range(1, 9)))}
                    else:
                        client.sglang.images, image_steps = history_image(history, screenshot, step + 1, args.history_frames)
                        context = decision_context(history, step + 1, args.frame_skip, image_steps, args.history_actions)
                        if args.thinking:
                            context += "\n" + SHORT_THINKING
                        action = client.generate(
                            context=context,
                            questions=questions(args.scenario, args.frame_skip),
                        )
                        if args.frame_skip is not None:
                            action["duration"] = args.frame_skip
                    elapsed = (time.perf_counter() - started) * 1000
                    reward = game.make_action(buttons(action, args.scenario), action["duration"])
                    row = {"episode": episode + 1, "step": step + 1, "action": action,
                           "latency_ms": elapsed if client else None, "reward": reward,
                           "total_reward": game.get_total_reward(), "game_tics": game.get_episode_time()}
                    if client:
                        row["context"] = context
                        row["image_steps"] = image_steps
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    records.append({**row, "image": screenshot})
                    history.append(records[-1])
                    print(json.dumps({k: v for k, v in row.items() if k != "context"}), flush=True)
                    if game.is_episode_finished():
                        break
                episodes.append({"episode": episode + 1, "reward": game.get_total_reward(),
                                 "finished": game.is_episode_finished(), "dead": game.is_player_dead()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        game.close()
        latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
        summary = {"controller": "RANDOM PREVIEW — NOT AI" if args.preview else "TypeLLM / vision",
                   "scenario": args.scenario, "seed": args.seed, "frame_skip": args.frame_skip,
                   "execution": args.execution, "duration_mode": "fixed" if args.frame_skip else "model-selected",
                   "history_frames": args.history_frames, "history_actions": args.history_actions,
                   "thinking": args.thinking, "thinking_budget": args.thinking_budget,
                   "model": args.model, "tokenizer": args.tokenizer, "decisions": len(records),
                   "p50_ms": statistics.median(latencies) if latencies else None,
                   "p95_ms": sorted(latencies)[max(0, __import__('math').ceil(len(latencies)*.95)-1)] if latencies else None,
                   "episodes": episodes, "error": error,
                   "timing": "Synchronous game steps; replay timing is illustrative, not real-time inference."}
        save_replay(args.output, records, summary)
        print(f"Replay: {args.output.resolve() / 'replay.html'}")


if __name__ == "__main__":
    main()
