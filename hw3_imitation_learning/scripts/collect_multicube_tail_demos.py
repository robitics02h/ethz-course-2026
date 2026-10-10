"""Collect complete demonstrations on tail geometries from ordinary scene resets.

Reject initial layouts outside a fixed standardized displacement interval.
Never alter object positions after reset. Balance color and ordered target/bin
slot pairs, and retain every skipped reset in the provenance report for replay.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from collect_multicube_demos import rollout
from record_teleop_demos import MulticubeZarrWriter
from hw3.sim_env import SO100MulticubeSimEnv, CUBE_COLORS, JOINT_NAMES
from so101_gym.constants import ASSETS_DIR


def layout_geometry(env):
    slots = env._cube_slot_qpos_templates[:, :2]
    cubes = env.get_all_cubes_xyz().reshape(3, 3)
    points = np.vstack((cubes[:, :2], env.get_goal_pos()[:2]))
    assignment = np.linalg.norm(points[:, None, :] - slots[None, :, :], axis=2).argmin(axis=1)
    if len(set(assignment.tolist())) != 4:
        raise ValueError('Initial objects cannot be assigned uniquely to the four slots')
    target = CUBE_COLORS.index(env.goal_cube)
    residual = (points - slots[assignment]) / env.cube_pos_std
    target_bin_residual = np.r_[residual[target], residual[-1]]
    return dict(target_slot=int(assignment[target]), bin_slot=int(assignment[-1]),
                residual_sigma=target_bin_residual.tolist(),
                tail_score_sigma=float(np.abs(target_bin_residual).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20282000)
    p.add_argument('--episodes-per-cell', type=int, default=6)
    p.add_argument('--min-sigma', type=float, default=2.)
    p.add_argument('--max-sigma', type=float, default=3.5)
    p.add_argument('--max-attempts', type=int, default=20000)
    args = p.parse_args()
    if args.episodes_per_cell < 1 or args.max_attempts < 36 * args.episodes_per_cell:
        p.error('Invalid episode count or insufficient reset budget')
    if not np.isfinite([args.min_sigma, args.max_sigma]).all() or not 0 <= args.min_sigma < args.max_sigma:
        p.error('Require finite 0 <= min-sigma < max-sigma')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml', seed=args.seed)
    writer = MulticubeZarrWriter(args.output_dir/'multicube_scripted.zarr', cube_dim=0, flush_every=1024)
    writer.set_attrs(source='scripted_multicube_expert', seed=args.seed, control_hz=10.,
        joint_names=list(JOINT_NAMES), cube_colors=list(CUBE_COLORS), shuffle_cubes=True,
        selection='tail_geometry', min_sigma=args.min_sigma, max_sigma=args.max_sigma,
        episodes_per_cell=args.episodes_per_cell,
        description='Complete physical teacher rollouts from default resets, stratified by tail geometry; actual mocap command labels')
    cell_counts = {f'{color}:{target}:{bin_slot}':0 for color in CUBE_COLORS
                   for target in range(4) for bin_slot in range(4) if target != bin_slot}
    counts = dict.fromkeys(CUBE_COLORS, 0)
    commands = []
    reasons = Counter()
    report = dict(seed=args.seed, source='scripted_multicube_expert', selection='tail_geometry',
        min_sigma=args.min_sigma, max_sigma=args.max_sigma, cube_pos_std=env.cube_pos_std,
        episodes_per_cell=args.episodes_per_cell, desired_episodes=36*args.episodes_per_cell,
        slot_xy=env._cube_slot_qpos_templates[:, :2].tolist(),
        attempts=[], counts=counts, cell_counts=cell_counts)

    def save_report():
        report.update(saved_episodes=writer.num_episodes, raw_steps=writer.num_steps_total,
                      reasons=dict(reasons))
        (args.output_dir/'collection_report.json').write_text(json.dumps(report, indent=2)+'\n')

    try:
        for attempt in range(args.max_attempts):
            if min(cell_counts.values()) >= args.episodes_per_cell:break
            color = CUBE_COLORS[attempt % 3]
            env.set_goal(color)
            env.reset()
            geometry = layout_geometry(env)
            cell = f'{color}:{geometry["target_slot"]}:{geometry["bin_slot"]}'
            result = dict(attempt=attempt, goal_color=color, saved_episode=None,
                          geometry=geometry, cell=cell, accepted=False, rolled_out=False)
            if not args.min_sigma <= geometry['tail_score_sigma'] <= args.max_sigma:
                result['reason'] = 'outside_tail_interval'
            elif cell_counts[cell] >= args.episodes_per_cell:
                result['reason'] = 'cell_full'
            else:
                rows, _, outcome = rollout(env)
                result.update(outcome, rolled_out=True)
                result['reason'] = 'saved' if outcome['accepted'] else 'teacher_failed'
                if outcome['accepted']:
                    result['saved_episode'] = writer.num_episodes
                    for row in rows:
                        writer.append_with_goal(*row[:8])
                        commands.append(row[8])
                    writer.end_episode()
                    counts[color] += 1
                    cell_counts[cell] += 1
                    print(f'Reset {attempt}: saved {writer.num_episodes}/{report["desired_episodes"]}; '
                          f'{cell}, tail={geometry["tail_score_sigma"]:.2f} sigma', flush=True)
            report['attempts'].append(result)
            reasons[result['reason']] += 1
            if result['rolled_out'] or (attempt+1) % 500 == 0:save_report()
    finally:
        writer.flush()
        if commands:
            writer.root['data'].create_array('action_mocap_delta', data=np.asarray(commands, dtype=np.float32))
        save_report()
        env.renderer.close()
    if min(cell_counts.values()) < args.episodes_per_cell:
        raise SystemExit('Collection incomplete: inspect cell counts and rejection report')
    print(f'Saved {writer.num_episodes} complete demonstrations. Rejections: {dict(reasons)}', flush=True)


if __name__ == '__main__':
    main()
