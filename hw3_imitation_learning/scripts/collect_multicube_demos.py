"""Collect success-filtered scripted Ex3 demonstrations in unmodified physics.

Records true mocap increments, all cube states, target color and bin position.
This is automated expert data, not human teleoperation or a learned policy.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from hw3.sim_env import SO100MulticubeSimEnv, CUBE_COLORS, JOINT_NAMES
from hw3.eval_utils import check_success, check_wrong_cube_in_bin
from so101_gym.constants import ASSETS_DIR
from record_teleop_demos import MulticubeZarrWriter


def rollout(env, speed=.004, preview=False, resume=False, terminal_variation=None):
    rows, frames = [], []
    previous = env.data.mocap_pos[0].copy()
    initial = env.get_all_cubes_xyz().reshape(3,3).copy()
    goal = env.get_goal_pos().copy()
    cube = env.get_target_cube_state()[:3].copy()
    max_height = float(cube[2])
    already_in_bin = check_success(env)
    holding = bool(resume and cube[2] > .065 and env.get_gripper_angle() < .6
                   and np.linalg.norm(env.get_ee_pos()-cube) < .085)
    wrong = False
    terminal_start = None
    def step():
        nonlocal previous, max_height, wrong
        obs = env.get_obs()
        rows.append((obs['joints'],obs['ee'],obs['cubes'],obs['gripper'],
                     np.array([env.data.ctrl[env.act_ids[-1]]]),obs['obstacle'],
                     obs['goal'],obs['goal_pos'],env.data.mocap_pos[0].copy()-previous))
        previous = env.data.mocap_pos[0].copy()
        if preview: frames.append(env.render())
        env.step()
        max_height = max(max_height,float(env.get_target_cube_state()[2]))
        wrong = wrong or check_wrong_cube_in_bin(env) is not None
    def move(target, grip, hold=5):
        target=np.asarray(target);start=env.data.mocap_pos[0].copy()
        count=max(1,int(np.ceil(np.linalg.norm(target-start)/speed)))
        env.set_gripper(grip)
        for i in range(count):
            env.set_mocap_pos(start+(target-start)*(i+1)/count);step()
        for _ in range(hold):step()
    # Raise first, approach the selected cube, then lift before transport.
    start=env.data.mocap_pos[0].copy()
    if resume and already_in_bin:
        terminal_start = len(rows)
        move(start,1.,15)
        move([start[0],start[1],.23],1.,15)
    else:
        if resume and not holding:
            # A partly closed grasp can pick up and then drop the cube during
            # retraction. Open first and let it settle before re-observing.
            move(start,1.,15)
            move([start[0],start[1],.20],1.,15)
        else:
            move([start[0],start[1],.20],-.174)
        if not holding:
            # Re-observe after retracting: a failed grasp may have moved the cube.
            if resume: cube = env.get_target_cube_state()[:3].copy()
            move([cube[0],cube[1],.16],1.)
            move([cube[0]-.03,cube[1],.065],1.)
            move([cube[0]-.03,cube[1],.065],-.174,15)
            move([cube[0],cube[1],.20],-.174,15)
        move([goal[0]-.02,goal[1],.20],-.174)
        place = np.array([goal[0]-.02,goal[1],.12])
        retreat = np.array([goal[0]-.02,goal[1],.23])
        hold_steps = 15
        if terminal_variation is not None:
            place += np.asarray(terminal_variation['place_offset'])
            retreat += np.asarray(terminal_variation['retreat_offset'])
            hold_steps = int(terminal_variation['hold_steps'])
        move(place,-.174)
        terminal_start = len(rows)
        move(place,1.,15)
        move(retreat,1.,hold_steps)
    stable=True
    for _ in range(10):
        step()
        stable = stable and check_success(env) and env.get_gripper_angle()>.8 and np.linalg.norm(env.get_ee_pos()-env.get_target_cube_state()[:3])>.08
    return rows,frames,dict(accepted=bool(stable and (max_height>.08 or (resume and already_in_bin)) and not wrong),
        takeover_holding=holding, takeover_in_bin=already_in_bin,
        stable_release=bool(stable),wrong_cube_in_bin=bool(wrong),max_target_height=max_height,
        goal_color=env.goal_cube,initial_cubes=initial.tolist(),bin_position=goal.tolist(),
        final_cubes=env.get_all_cubes_xyz().reshape(3,3).tolist(),steps=len(rows),terminal_start=terminal_start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--episodes-per-color',type=int,default=20)
    p.add_argument('--max-attempts',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261030)
    args=p.parse_args()
    if args.episodes_per_color<1 or args.max_attempts<3*args.episodes_per_color:p.error('Insufficient attempt budget or invalid episode count')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    env=SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml',seed=args.seed)
    writer=MulticubeZarrWriter(args.output_dir/'multicube_scripted.zarr',cube_dim=0,flush_every=1024)
    writer.set_attrs(source='scripted_multicube_expert',seed=args.seed,control_hz=10.,
        joint_names=list(JOINT_NAMES),cube_colors=list(CUBE_COLORS),shuffle_cubes=True,
        description='Success-filtered physical rollouts with target color and randomized cube/bin layout; actual mocap command labels')
    counts=dict.fromkeys(CUBE_COLORS,0);commands=[]
    report=dict(seed=args.seed,source='scripted_multicube_expert',attempts=[],counts=counts)
    try:
        for attempt in range(args.max_attempts):
            pending=[c for c in CUBE_COLORS if counts[c]<args.episodes_per_color]
            if not pending:break
            color=pending[attempt%len(pending)]
            env.set_goal(color);env.reset()
            rows,frames,result=rollout(env,preview=counts[color]==0)
            result.update(attempt=attempt,saved_episode=None)
            if result['accepted']:
                result['saved_episode']=writer.num_episodes
                for row in rows:
                    writer.append_with_goal(*row[:8]);commands.append(row[8])
                writer.end_episode();counts[color]+=1
                if counts[color]==1:
                    video=cv2.VideoWriter(str(args.output_dir/f'preview_{color}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
                    if not video.isOpened():raise RuntimeError('Could not open preview writer')
                    for frame in frames:video.write(frame)
                    video.release()
            report['attempts'].append(result)
            report.update(saved_episodes=writer.num_episodes,raw_steps=writer.num_steps_total)
            (args.output_dir/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f"Attempt {attempt}: {color}, success={result['accepted']}, saved={counts}",flush=True)
    finally:
        writer.flush()
        if commands:writer.root['data'].create_array('action_mocap_delta',data=np.asarray(commands,dtype=np.float32))
        env.renderer.close()
    if min(counts.values())<args.episodes_per_color:raise SystemExit('Collection incomplete; inspect failed layouts in report')
    print(f'Saved {writer.num_episodes} successful demonstrations.')

if __name__=='__main__':main()
