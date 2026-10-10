"""Record a learned single-cube policy and report both bin entry and release."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from hw3.eval_utils import apply_action, check_success, check_wrong_cube_in_bin, infer_action_chunk, load_checkpoint
from hw3.sim_env import SO100SimEnv, SO100MulticubeSimEnv
from so101_gym.constants import ASSETS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--multicube", action="store_true")
    parser.add_argument("--goal-cube", choices=["red", "green", "blue"], default="red")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Zero-based reset index for reproducing a diagnostic scene")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, normalizer, _, state_keys, action_keys = load_checkpoint(args.checkpoint, device)
    if args.multicube:
        env = SO100MulticubeSimEnv(ASSETS_DIR / "so100_multicube_ee.xml", seed=args.seed,
                                   goal_cube=args.goal_cube)
    else:
        env = SO100SimEnv(ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=args.seed)
    if args.scene_index < 0:
        parser.error("--scene-index must be nonnegative")
    for _ in range(args.scene_index + 1):
        obs = env.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    if not video.isOpened():
        env.renderer.close()
        raise RuntimeError("Cannot open video writer")
    queue = []
    first_success = None
    stable_released = 0
    wrong_cube = False
    try:
        for step in range(800):
            if not queue:
                queue.extend(infer_action_chunk(model, normalizer, obs, state_keys, device))
            apply_action(env, queue.pop(0), action_keys)
            obs = env.step()
            in_bin = check_success(env)
            if args.multicube:
                wrong_cube = wrong_cube or check_wrong_cube_in_bin(env) is not None
            if in_bin and first_success is None:
                first_success = step + 1
            released = bool(in_bin and obs["gripper"][0] > 0.8
                            and np.linalg.norm(obs["ee_pos"] - obs["cube"][:3]) > 0.08)
            stable_released = stable_released + 1 if released else 0
            frame = env.render()
            label = f"goal={args.goal_cube} | " if args.multicube else ""
            cv2.putText(frame, f"Learned policy | {label}step {step + 1} | bin={in_bin}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            video.write(frame)
            if stable_released >= 10:
                break
        cv2.imwrite(str(args.output.with_suffix(".png")), frame)
        report = {"checkpoint": str(args.checkpoint), "seed": args.seed,
                  "scene_index": args.scene_index,
                  "steps": step + 1, "first_bin_entry_step": first_success,
                  "released_and_stable": stable_released >= 10,
                  "multicube": args.multicube, "goal_cube": args.goal_cube if args.multicube else None,
                  "wrong_cube_in_bin": wrong_cube,
                  "final_cube": obs["cube"][:3].tolist()}
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        video.release()
        env.renderer.close()


if __name__ == "__main__":
    main()
