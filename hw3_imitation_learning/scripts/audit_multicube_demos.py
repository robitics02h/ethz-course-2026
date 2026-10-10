"""Validate and physically replay saved automated multicube demonstrations."""
import argparse,json
from pathlib import Path
import numpy as np
import zarr
from hw3.eval_utils import apply_action,check_success,check_wrong_cube_in_bin
from hw3.sim_env import SO100MulticubeSimEnv
from so101_gym.constants import ASSETS_DIR


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.dataset_dir/'collection_report.json').read_text())
    corrections = report.get('source') == 'automated_multicube_policy_takeover'
    terminal = report.get('source') == 'scripted_multicube_terminal_suffix'
    name = 'multicube_terminal.zarr' if terminal else 'multicube_corrections.zarr' if corrections else 'multicube_scripted.zarr'
    z=zarr.open_group(str(args.dataset_dir/name),mode='r')
    data=z['data'];ends=z['meta/episode_ends'][:]
    assert len(ends)==report['saved_episodes'] and np.all(np.diff(np.r_[0,ends])>0)
    for key,arr in data.arrays():
        assert arr.shape[0]==ends[-1],key
        assert np.isfinite(arr[:]).all(),key
    env=SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml',seed=report['seed'])
    results=[]
    try:
        if corrections:
            by_scene = {a['scan_episode']: a for a in report['attempts']}
            attempts = [by_scene.get(i) for i in range(max(by_scene)+1)]
        else:
            attempts = report['attempts']
        for attempt in attempts:
            if attempt is None:
                env.reset()
                continue
            env.set_goal(attempt['goal_color']);env.reset()
            ep=attempt['saved_episode']
            if ep is None:continue
            for row in attempt.get('policy_prefix', []):
                apply_action(env,np.asarray(row['action']),['action_ee_xyz','action_gripper'])
                env.step()
            for action in attempt.get('prefix_actions', []):
                apply_action(env,np.asarray(action),['action_ee_xyz','action_gripper'])
                env.step()
            start=0 if ep==0 else ends[ep-1];end=ends[ep]
            commands=data['action_mocap_delta'][start:end];grip=data['action_gripper'][start:end]
            ee=data['state_ee'][start:end]
            goals=data['state_goal'][start:end]
            assert np.all(goals==env.get_goal_onehot())
            assert np.allclose(data['goal_pos'][start:end],env.get_goal_pos())
            error=0.;stable=0;wrong=False
            for j in range(end-start):
                error=max(error,float(np.max(np.abs(env.get_ee_pos()-ee[j,:3]))))
                apply_action(env,np.r_[commands[j],grip[j]],['action_ee_xyz','action_gripper']);env.step()
                wrong=wrong or check_wrong_cube_in_bin(env) is not None
                released=check_success(env) and env.get_gripper_angle()>.8 and np.linalg.norm(env.get_ee_pos()-env.get_target_cube_state()[:3])>.08
                stable=stable+1 if released else 0
            results.append(dict(episode=int(ep),color=attempt['goal_color'],stable_release=bool(stable>=10),wrong_cube=wrong,max_ee_error_m=error))
    finally:env.renderer.close()
    summary=dict(episodes=len(results),stable_release_successes=sum(r['stable_release'] and not r['wrong_cube'] for r in results),max_ee_error_m=max(r['max_ee_error_m'] for r in results),results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['stable_release_successes']==len(ends)
    assert summary['max_ee_error_m']<.0005
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))

if __name__=='__main__':main()
