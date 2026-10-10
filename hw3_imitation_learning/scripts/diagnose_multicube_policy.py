"""Evaluate goal-conditioned policies with release and failure diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hw3.eval_utils import load_checkpoint,infer_action_chunk,apply_action,check_success,check_wrong_cube_in_bin,check_cube_out_of_bounds
from hw3.sim_env import SO100MulticubeSimEnv,CUBE_COLORS
from so101_gym.constants import ASSETS_DIR


def diagnostic_action(env, action, action_keys, target_lag_limit=0.):
 """Optional privileged diagnostic; never used by the official evaluator.

 Bound the next accumulated target relative to the current measured hand pose.
 This changes only the XYZ command, using the existing action interface. It does
 not teleport the robot or cube, or change physics or the success criterion.
 """
 xyz_action = action_keys == ['action_ee_xyz', 'action_gripper']
 if not xyz_action:
  if target_lag_limit:raise ValueError('Target lag limiting requires XYZ + gripper actions')
  return action, None, 0.
 current = env.data.mocap_pos[env.mocap_id].copy()
 hand = env.get_ee_pos()
 offset = current + action[:3] - hand
 lag = float(np.linalg.norm(offset))
 correction = 0.
 applied = action
 if target_lag_limit and lag > target_lag_limit:
  bounded_target = hand + offset * (target_lag_limit / lag)
  applied = np.asarray(action, dtype=np.float64).copy()
  applied[:3] = bounded_target - current
  correction = float(np.linalg.norm(applied[:3] - action[:3]))
 return applied, lag, correction


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--episodes',type=int,default=30);p.add_argument('--seed',type=int,default=4000)
 p.add_argument('--scene-indices',type=int,nargs='+',help='Evaluate only these zero-based scene indices while preserving the reset sequence')
 p.add_argument('--execute-steps',type=int,choices=[1,2,4],default=4,help='Diagnostic only: replan after executing this many predicted actions')
 p.add_argument('--trace-dir',type=Path,help='Save per-step robot/cube/control traces for inspected episodes')
 p.add_argument('--post-success-steps',type=int,default=0,help='Diagnostic only: continue this many steps after first stable success and require stability throughout')
 p.add_argument('--target-lag-limit',type=float,default=0.,help='Privileged diagnostic only: bound the proposed mocap target this many metres from the measured hand; zero disables')
 args=p.parse_args();torch.set_num_threads(1);device=torch.device('cpu')
 if args.post_success_steps < 0:p.error('--post-success-steps must be nonnegative')
 if not np.isfinite(args.target_lag_limit) or args.target_lag_limit < 0:p.error('--target-lag-limit must be finite and nonnegative')
 m,n,_,sk,ak=load_checkpoint(args.checkpoint,device)
 env=SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml',seed=args.seed)
 results=[]
 try:
  for ep in range(args.episodes):
   color=CUBE_COLORS[ep%3];env.set_goal(color);obs=env.reset()
   if args.scene_indices is not None and ep not in args.scene_indices:continue
   initial=env.get_all_cubes_xyz().reshape(3,3).copy();goal=env.get_goal_pos().copy()
   queue=[];stable=0;entry=False;wrong=False;maxz=0.;closest=float('inf');maxcommand=0.;maxinput=0.;other_lift=False
   trace=[];requested_actions=[]
   max_requested_lag=0.;max_measured_lag=0.;limited_steps=0;first_limited_step=None;total_correction=0.;max_correction=0.
   first_success_step=None;post_success_violation=False
   for step in range(800+args.post_success_steps):
    if not queue:
     from hw3.eval_utils import obs_to_state
     maxinput=max(maxinput,float(np.abs(n.normalize_state(obs_to_state(obs,sk))).max()))
     queue.extend(infer_action_chunk(m,n,obs,sk,device)[:args.execute_steps])
    a=queue.pop(0);maxcommand=max(maxcommand,float(np.abs(a[:3]).max()))
    applied,requested_lag,correction=diagnostic_action(env,a,ak,args.target_lag_limit)
    if requested_lag is not None:max_requested_lag=max(max_requested_lag,requested_lag)
    if correction:
     limited_steps+=1
     if first_limited_step is None:first_limited_step=step+1
     total_correction+=correction;max_correction=max(max_correction,correction)
    apply_action(env,applied,ak);obs=env.step()
    max_measured_lag=max(max_measured_lag,float(np.linalg.norm(env.data.mocap_pos[env.mocap_id]-obs['ee_pos'])))
    inside=check_success(env);entry=entry or inside;wrong=wrong or check_wrong_cube_in_bin(env) is not None
    pos=env.get_all_cubes_xyz().reshape(3,3);target=pos[ep%3]
    other_lift=other_lift or any(pos[i,2]>.08 for i in range(3) if i!=ep%3)
    maxz=max(maxz,float(target[2]));closest=min(closest,float(np.linalg.norm(obs['ee_pos']-target)))
    released=inside and obs['gripper'][0]>.8 and np.linalg.norm(obs['ee_pos']-target)>.08
    stable=stable+1 if released else 0
    if first_success_step is None and stable>=10:first_success_step=step+1
    if first_success_step is not None and step+1>first_success_step and not released:post_success_violation=True
    if args.trace_dir:
     trace.append(np.r_[obs['ee_pos'],pos.reshape(-1),obs['gripper'],env.data.mocap_pos[env.mocap_id],applied])
     requested_actions.append(a.copy())
    if check_cube_out_of_bounds(env):break
    if first_success_step is None and step+1>=800:break
    if first_success_step is not None and step+1>=first_success_step+args.post_success_steps:break
   success=stable>=10 and not wrong and not post_success_violation
   failure=None if success else 'wrong_cube' if wrong else 'lost_stability_after_success' if post_success_violation else 'not_released' if entry else 'wrong_cube_lift' if other_lift and maxz<.065 else 'no_lift' if maxz<.065 else 'transport_or_place'
   results.append(dict(episode=ep,color=color,success=success,bin_entry=bool(entry),failure=failure,steps=step+1,max_cube_z=maxz,other_cube_lift=bool(other_lift),min_hand_target_distance=closest,max_xyz_command=maxcommand,max_normalized_input=maxinput,initial_cubes=initial.tolist(),bin_position=goal.tolist(),final_cubes=pos.tolist(),first_success_step=first_success_step,post_success_violation=post_success_violation,max_requested_target_lag=max_requested_lag,max_target_hand_distance=max_measured_lag,lag_limited_steps=limited_steps,first_limited_step=first_limited_step,total_xyz_correction=total_correction,max_xyz_correction=max_correction))
   if args.trace_dir:
    args.trace_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.trace_dir/f'scene_{ep}.npz',trace=np.asarray(trace),requested_actions=np.asarray(requested_actions),goal=goal,initial_cubes=initial,color=color,target_lag_limit=args.target_lag_limit,columns='ee_xyz,cubes_xyz(9),gripper,mocap_xyz,applied_action(4)')
   if (ep+1)%10==0:print(f"{ep+1}: stable successes={sum(r['success'] for r in results)}",flush=True)
 finally:env.renderer.close()
 report=dict(checkpoint=str(args.checkpoint),seed=args.seed,execute_steps=args.execute_steps,post_success_steps=args.post_success_steps,target_lag_limit=args.target_lag_limit,privileged_intervention=bool(args.target_lag_limit),episodes=len(results),stable_successes=sum(r['success'] for r in results),bin_entries=sum(r['bin_entry'] for r in results),failures={k:sum(r['failure']==k for r in results if r['failure']) for k in sorted({r['failure'] for r in results if r['failure']})},results=results)
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()
