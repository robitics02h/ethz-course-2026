"""Replay saved policy prefixes and expert commands to audit DAgger labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import zarr
from hw3.eval_utils import apply_action, check_success
from hw3.sim_env import SO100SimEnv
from so101_gym.constants import ASSETS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.dataset_dir / 'collection_report.json').read_text())
    root = zarr.open_group(str(args.dataset_dir / 'automated_dagger.zarr'), mode='r')
    ends, data = root['meta/episode_ends'][:], root['data']
    env = SO100SimEnv(ASSETS_DIR / 'so100_transfer_cube_obstacle_ee.xml',
                      seed=report['seed'], obstacle_mode='adversarial')
    attempts = {a['attempt']: a for a in report['attempts']}
    checked = []
    try:
        for attempt in range(max(attempts) + 1):
            env.reset()
            item = attempts.get(attempt)
            if item is None or item['saved_episode'] is None:
                continue
            for row in item['policy_prefix']:
                apply_action(env, np.asarray(row['action']), ['action_ee_xyz', 'action_gripper'])
                env.step()
            episode = item['saved_episode']
            start = 0 if episode == 0 else ends[episode - 1]
            end = ends[episode]
            ee, cube = data['state_ee'][start:end], data['state_cube'][start:end]
            commands = data['action_mocap_delta'][start:end]
            grip = data['action_gripper'][start:end]
            position_error = quaternion_error = 0.
            for j in range(end - start):
                current_cube = env.get_cube_state()
                position_error = max(position_error,
                    float(np.max(np.abs(env.get_ee_pos() - ee[j, :3]))),
                    float(np.max(np.abs(current_cube[:3] - cube[j, :3]))))
                quaternion_error = max(quaternion_error,
                    float(np.max(np.abs(current_cube[3:] - cube[j, 3:]))))
                apply_action(env, np.r_[commands[j], grip[j]], ['action_ee_xyz', 'action_gripper'])
                env.step()
            checked.append({'episode': episode, 'max_position_error_m': position_error,
                            'max_cube_quaternion_component_error': quaternion_error,
                            'bin_success': check_success(env)})
    finally:
        env.renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checked, indent=2) + '\n')
    # Float32 command storage can create small divergence at contact.
    assert len(checked) == report['saved_episodes']
    assert all(a['bin_success'] and a['max_position_error_m'] < .0005 for a in checked)
    print(f"Replayed {len(checked)} episodes; all successful. Maximum position error "
          f"{max(a['max_position_error_m'] for a in checked):.3g} m")


if __name__ == '__main__':
    main()
