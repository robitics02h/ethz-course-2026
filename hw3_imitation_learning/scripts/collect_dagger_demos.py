"""Collect automated expert takeovers from policy-visited Ex2 states.

This is an automated variant of the course's human-intervention workflow.
Policy prefixes are retained separately for provenance, not as expert labels.
The simulator is never reset between a policy prefix and expert takeover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from collect_scripted_demos import rollout
from hw3.eval_utils import apply_action, infer_action_chunk, load_checkpoint
from hw3.sim_env import JOINT_NAMES, SO100SimEnv
from hw3.teleop_utils import ZarrEpisodeWriter
from so101_gym.constants import ASSETS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=24)
    parser.add_argument('--max-attempts', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--left-route-x', type=float, default=-.12)
    args = parser.parse_args()
    if args.episodes < 3 or args.max_attempts < args.episodes:
        parser.error('Require 3 <= episodes <= max-attempts')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model, norm, _, sk, ak = load_checkpoint(args.checkpoint, device)
    if ak != ['action_ee_xyz', 'action_gripper']:
        raise ValueError('This collector requires XYZ mocap increments and absolute gripper commands')
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    xml = ASSETS_DIR / 'so100_transfer_cube_obstacle_ee.xml'
    env = SO100SimEnv(xml, seed=args.seed, obstacle_mode='adversarial')
    writer = ZarrEpisodeWriter(args.output_dir / 'automated_dagger.zarr', flush_every=1024)
    writer.set_attrs(xml=str(xml), joint_names=list(JOINT_NAMES), control_hz=10.,
                     source='automated_expert_takeover', seed=args.seed,
                     policy_checkpoint=str(args.checkpoint), policy_sha256=sha,
                     obstacle_mode='adversarial', left_route_x=args.left_route_x)
    # Ensure the hard left/right zones cannot vanish under success filtering.
    quotas = {'left': args.episodes // 3, 'center': args.episodes // 3,
              'right': args.episodes - 2 * (args.episodes // 3)}
    counts = dict.fromkeys(quotas, 0)
    report = {'checkpoint': str(args.checkpoint), 'sha256': sha, 'seed': args.seed,
              'state_keys': sk, 'action_keys': ak,
              'source': 'automated_expert_takeover', 'quotas': quotas,
              'prefix_steps_schedule': [4, 12, 24, 40], 'attempts': []}
    commands = []
    try:
        for attempt in range(args.max_attempts):
            obs = env.reset()
            x = float(obs['obstacle'][0])
            zone = 'left' if x < -.265 else 'right' if x > -.175 else 'center'
            if counts[zone] >= quotas[zone]:
                continue
            prefix, queue = [], []
            # Vary takeover timing separately within each zone.
            n_prefix = [4, 12, 24, 40][sum(r['zone'] == zone for r in report['attempts']) % 4]
            for _ in range(n_prefix):
                if not queue:
                    queue.extend(infer_action_chunk(model, norm, obs, sk, device))
                action = queue.pop(0)
                prefix.append({'state': {k: np.asarray(v).tolist() for k, v in obs.items()},
                               'mocap_target': env.data.mocap_pos[0].tolist(),
                               'action': np.asarray(action).tolist()})
                apply_action(env, action, ak)
                obs = env.step()
            takeover = {k: np.asarray(v).tolist() for k, v in obs.items()}
            side = args.left_route_x if zone == 'left' else x - .125
            rows, _, result = rollout(env, .004, False, -.174, side_x=side)
            result.update(attempt=attempt, zone=zone, prefix_steps=n_prefix,
                          policy_prefix=prefix, takeover_state=takeover, side_x=side,
                          saved_episode=None)
            if result['accepted'] and result['released']:
                result['saved_episode'] = writer.num_episodes
                for row in rows:
                    writer.append(*row[:6])
                    commands.append(row[6])
                writer.end_episode()
                counts[zone] += 1
            report['attempts'].append(result)
            report.update(saved_episodes=writer.num_episodes, counts=counts.copy())
            (args.output_dir / 'collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f"Attempt {attempt}: {zone}, prefix={n_prefix}, accepted={result['accepted']}, saved={counts}", flush=True)
            if writer.num_episodes == args.episodes:
                break
    finally:
        writer.flush()
        if commands:
            writer.root['data'].create_array('action_mocap_delta', data=np.asarray(commands, dtype=np.float32))
        env.renderer.close()
    if writer.num_episodes != args.episodes:
        raise SystemExit('Incomplete zone quotas; inspect collection_report.json')
    print(f'Saved {writer.num_episodes} expert takeover episodes to {args.output_dir}')


if __name__ == '__main__':
    main()
