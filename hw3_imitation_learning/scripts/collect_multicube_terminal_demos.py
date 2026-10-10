"""Save only open/retreat/hold suffixes, reached through physical expert prefixes."""
import argparse
import json
from pathlib import Path
import numpy as np
from collect_multicube_demos import rollout
from record_teleop_demos import MulticubeZarrWriter
from hw3.sim_env import SO100MulticubeSimEnv, CUBE_COLORS, JOINT_NAMES
from so101_gym.constants import ASSETS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--episodes-per-color', type=int, default=20)
    p.add_argument('--seed', type=int, default=20276000)
    p.add_argument('--max-attempts', type=int, default=120)
    args = p.parse_args()
    if args.episodes_per_color < 1 or args.max_attempts < 3*args.episodes_per_color:
        p.error('Invalid episode count or attempt budget')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml', seed=args.seed)
    perturb = np.random.default_rng(args.seed+1)
    writer = MulticubeZarrWriter(args.output_dir/'multicube_terminal.zarr', cube_dim=0, flush_every=1024)
    writer.set_attrs(source='scripted_multicube_terminal_suffix', seed=args.seed,
        control_hz=10., joint_names=list(JOINT_NAMES), cube_colors=list(CUBE_COLORS),
        description='Open, retreat and hold only; expert prefix stored separately for physical replay; no reset at suffix start')
    counts = dict.fromkeys(CUBE_COLORS, 0)
    report = dict(source='scripted_multicube_terminal_suffix', seed=args.seed,
        counts=counts, attempts=[], saved_episodes=0)
    commands = []
    try:
        for attempt in range(args.max_attempts):
            pending = [c for c in CUBE_COLORS if counts[c] < args.episodes_per_color]
            if not pending:break
            color = pending[attempt % len(pending)]
            env.set_goal(color);env.reset()
            variation = dict(place_offset=perturb.uniform([-.006,-.006,-.012],[.006,.006,.012]).tolist(),
                retreat_offset=perturb.uniform([-.008,-.008,-.01],[.008,.008,.01]).tolist(),
                hold_steps=int(perturb.integers(90,111)))
            rows,_,result = rollout(env,terminal_variation=variation)
            start = result['terminal_start']
            assert start is not None and start > 0
            suffix = rows[start:]
            assert all(abs(float(row[4][0])-1.) < 1e-6 for row in suffix)
            result.update(attempt=attempt, saved_episode=None, variation=variation,
                prefix_actions=[np.r_[row[8],row[4]].tolist() for row in rows[:start]],
                prefix_steps=start, suffix_steps=len(suffix))
            if result['accepted']:
                result['saved_episode'] = writer.num_episodes
                for row in suffix:
                    writer.append_with_goal(*row[:8]);commands.append(row[8])
                writer.end_episode();counts[color] += 1
            report['attempts'].append(result)
            report.update(saved_episodes=writer.num_episodes,raw_steps=writer.num_steps_total)
            (args.output_dir/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'Attempt {attempt}: {color}, accepted={result["accepted"]}, suffix={len(suffix)}, saved={counts}',flush=True)
    finally:
        writer.flush()
        if commands:
            writer.root['data'].create_array('action_mocap_delta',data=np.asarray(commands,dtype=np.float32))
        env.renderer.close()
    if min(counts.values()) < args.episodes_per_color:
        raise SystemExit('Collection incomplete')
    print(f'Saved {writer.num_episodes} terminal suffixes.',flush=True)


if __name__=='__main__':
    main()
