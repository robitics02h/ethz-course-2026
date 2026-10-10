"""Evaluate single-cube policies with release checks and failure diagnostics.

This supplements, and does not modify, the official evaluation harness.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hw3.eval_utils import (
    apply_action, check_cube_out_of_bounds, check_success,
    infer_action_chunk, load_checkpoint,
)
from hw3.sim_env import SO100SimEnv
from so101_gym.constants import ASSETS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--adversarial-obstacle", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, norm, _, sk, ak = load_checkpoint(args.checkpoint, device)
    env = SO100SimEnv(ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=args.seed,
                      obstacle_mode="adversarial" if args.adversarial_obstacle else "train")
    results = []
    started = time.monotonic()
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            initial_cube = obs["cube"][:3].tolist()
            obstacle = obs["obstacle"].tolist()
            queue, first_entry, stable, max_height, max_y = [], None, 0, 0., 0.
            for step in range(800):
                if not queue:
                    queue.extend(infer_action_chunk(model, norm, obs, sk, device))
                apply_action(env, queue.pop(0), ak)
                obs = env.step()
                inside = check_success(env)
                if inside and first_entry is None:
                    first_entry = step + 1
                released = bool(inside and obs["gripper"][0] > .8
                                and np.linalg.norm(obs["ee_pos"] - obs["cube"][:3]) > .08)
                stable = stable + 1 if released else 0
                max_height = max(max_height, float(obs["cube"][2]))
                max_y = max(max_y, float(obs["cube"][1]))
                if stable >= 10 or check_cube_out_of_bounds(env):
                    break
            if stable >= 10:
                failure = None
            elif first_entry is not None:
                failure = "not_stably_released"
            elif max_height < .065:
                failure = "no_lift"
            elif max_y < .60:
                failure = "transport_before_bin"
            else:
                failure = "placement_or_release"
            results.append({"episode": ep + 1, "bin_entry": first_entry is not None,
                            "released_and_stable": stable >= 10, "failure": failure,
                            "steps": step + 1, "max_cube_z": max_height,
                            "max_cube_y": max_y, "initial_cube": initial_cube,
                            "obstacle": obstacle, "final_cube": obs["cube"][:3].tolist()})
            if (ep + 1) % 10 == 0:
                print(f"{ep + 1}: bin={sum(r['bin_entry'] for r in results)}, "
                      f"released={sum(r['released_and_stable'] for r in results)}", flush=True)
    finally:
        env.renderer.close()
    report = {"checkpoint": str(args.checkpoint), "seed": args.seed,
              "obstacle_mode": "adversarial" if args.adversarial_obstacle else "train",
              "episodes": args.episodes, "elapsed_seconds": time.monotonic() - started,
              "bin_entry_successes": sum(r["bin_entry"] for r in results),
              "released_and_stable_successes": sum(r["released_and_stable"] for r in results),
              "failure_counts": {name: sum(r["failure"] == name for r in results)
                                 for name in sorted({r["failure"] for r in results if r["failure"]})},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
