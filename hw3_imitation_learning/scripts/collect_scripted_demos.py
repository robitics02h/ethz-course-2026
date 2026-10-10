"""Collect successful Exercise 1 demonstrations with a scripted MuJoCo expert.

Uses the unmodified scene, reset distribution, and teleoperation mocap control.
Only successful physical pick/place rollouts are saved; no object teleportation.
Run from hw3_imitation_learning with its Python environment.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hw3.eval_utils import check_success
from hw3.sim_env import JOINT_NAMES, SO100SimEnv
from hw3.teleop_utils import ZarrEpisodeWriter
from so101_gym.constants import ASSETS_DIR


def rollout(env: SO100SimEnv, speed: float, preview: bool, approach_grip: float = 1.0, waypoint_hold: int = 5, side_x: float | None = None):
    rows, frames = [], []
    min_cube_z = float("inf")
    max_cube_z = float("-inf")
    previous_target = env.data.mocap_pos[0].copy()

    def step():
        nonlocal min_cube_z, max_cube_z, previous_target
        obs = env.get_obs()
        rows.append((obs["joints"], obs["ee"], obs["cube"], obs["gripper"],
                     np.array([env.data.ctrl[env.act_ids[-1]]]), obs["obstacle"],
                     env.data.mocap_pos[0].copy() - previous_target))
        previous_target = env.data.mocap_pos[0].copy()
        min_cube_z = min(min_cube_z, float(obs["cube"][2]))
        max_cube_z = max(max_cube_z, float(obs["cube"][2]))
        if preview:
            frames.append(env.render())
        env.step()

    def move(target, grip=1.0, hold=None):
        hold = waypoint_hold if hold is None else hold
        target = np.asarray(target, dtype=float)
        start = env.data.mocap_pos[0].copy()
        count = max(1, int(np.ceil(np.linalg.norm(target - start) / speed)))
        env.set_gripper(grip)
        for i in range(count):
            env.set_mocap_pos(start + (target - start) * ((i + 1) / count))
            step()
        for _ in range(hold):
            step()

    cube = env.get_cube_state()[:3].copy()
    obstacle = env.get_obstacle_pos()
    goal = env.get_goal_pos()
    side_x = obstacle[0] - 0.125 if side_x is None else side_x
    move([side_x, 0.50, 0.16], approach_grip)
    move([side_x, cube[1] - 0.04, 0.16], approach_grip)
    move([cube[0], cube[1], 0.12])
    grasp = [cube[0] - 0.03, cube[1], 0.065]
    move(grasp)
    move(grasp, -0.174, hold=15)
    move([cube[0], cube[1], 0.16], -0.174)
    lifted_at_waypoint = bool(env.get_cube_state()[2] > 0.065)
    move([side_x, cube[1], 0.16], -0.174)
    move([side_x, 0.55, 0.16], -0.174)
    move([goal[0] - 0.02, goal[1], 0.16], -0.174)
    move([goal[0] - 0.02, goal[1], 0.12], -0.174)
    move([goal[0] - 0.02, goal[1], 0.12], 1.0, hold=15)
    move([goal[0] - 0.02, goal[1], 0.20], 1.0, hold=15)
    stable = True
    for _ in range(10):
        stable = stable and check_success(env)
        step()
    separated = np.linalg.norm(env.get_ee_pos() - env.get_cube_state()[:3]) > 0.09
    lifted = max_cube_z > 0.065
    accepted = bool(lifted and stable and separated and min_cube_z > 0.005)
    return rows, frames, {
        "accepted": accepted, "lifted": lifted, "stable_in_bin": bool(stable),
        "lifted_at_waypoint": lifted_at_waypoint, "max_cube_z": max_cube_z,
        "released": bool(separated and env.get_gripper_angle() > 0.8),
        "min_cube_z": min_cube_z, "steps": len(rows),
        "initial_cube": cube.tolist(), "obstacle": obstacle.tolist(),
        "final_cube": env.get_cube_state()[:3].tolist(), "speed": speed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--approach-grip", type=float, default=-0.174,
                        help="Keep fingers closed while passing the obstacle; original expert used 1.0")
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--obstacle-pos-std", type=float, default=0.01)
    parser.add_argument("--waypoint-hold", type=int, default=5)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_attempts < args.episodes:
        parser.error("Require 1 <= episodes <= max-attempts")
    out = args.output_dir or Path("datasets/raw/single_cube/scripted") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=False)
    xml = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
    env = SO100SimEnv(xml, seed=args.seed, render_w=640, render_h=480,
                      cube_pos_std=args.cube_pos_std, obstacle_pos_std=args.obstacle_pos_std)
    writer = ZarrEpisodeWriter(out / "so100_transfer_cube_scripted.zarr", flush_every=1024)
    writer.set_attrs(xml=str(xml), joint_names=list(JOINT_NAMES), control_hz=10.0,
                     source="scripted_expert", seed=args.seed, approach_grip=args.approach_grip,
                     cube_pos_std=args.cube_pos_std, obstacle_pos_std=args.obstacle_pos_std,
                     waypoint_hold=args.waypoint_hold,
                     description="Physical MuJoCo rollouts; randomization settings in metadata; success filtered")
    rng = np.random.default_rng(args.seed)
    report = {"waypoint_hold": args.waypoint_hold, "approach_grip": args.approach_grip, "cube_pos_std": args.cube_pos_std,
              "obstacle_pos_std": args.obstacle_pos_std, "seed": args.seed, "requested_episodes": args.episodes, "attempts": []}
    command_actions = []
    try:
        for attempt in range(args.max_attempts):
            if attempt:
                env.reset()
            rows, frames, result = rollout(env, float(rng.uniform(0.0035, 0.0045)), writer.num_episodes == 0, args.approach_grip, args.waypoint_hold)
            result["attempt"] = attempt
            report["attempts"].append(result)
            if result["accepted"] and result["released"]:
                for row in rows:
                    writer.append(*row[:6])
                    command_actions.append(row[6])
                writer.end_episode()
                if writer.num_episodes == 1:
                    video = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
                    if not video.isOpened():
                        raise RuntimeError("Cannot open preview video writer")
                    for frame in frames:
                        video.write(frame)
                    video.release()
                    cv2.imwrite(str(out / "final_frame.png"), env.render())
            print(f"Attempt {attempt + 1}: accepted={result['accepted']}, saved={writer.num_episodes}/{args.episodes}, steps={len(rows)}", flush=True)
            report["saved_episodes"] = writer.num_episodes
            report["saved_steps"] = writer.num_steps_total
            (out / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n")
            if writer.num_episodes >= args.episodes:
                break
    finally:
        writer.flush()
        if command_actions:
            writer.root["data"].create_array("action_mocap_delta", data=np.asarray(command_actions, dtype=np.float32))
        env.renderer.close()
    print(f"Saved {writer.num_episodes} episodes to {out}", flush=True)
    if writer.num_episodes < args.episodes:
        raise SystemExit("Insufficient successful rollouts; inspect collection_report.json")


if __name__ == "__main__":
    main()
