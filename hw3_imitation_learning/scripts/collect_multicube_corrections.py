"""Collect expert suffixes on failed policy scenes, without resetting at takeover.

Scan scenes become development/training scenes, never final holdout scenes.
One correction per failed scene prevents duplicate-prefix episode leakage.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from collect_multicube_demos import rollout
from record_teleop_demos import MulticubeZarrWriter
from hw3.eval_utils import load_checkpoint,infer_action_chunk,apply_action,check_cube_out_of_bounds,check_wrong_cube_in_bin
from hw3.sim_env import SO100MulticubeSimEnv,CUBE_COLORS,JOINT_NAMES
from so101_gym.constants import ASSETS_DIR


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--scan',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--prefix-scale',type=float,default=1.,help='Scale failure-type takeover times; less than one intervenes earlier')
 p.add_argument('--max-episodes',type=int,default=None,help='Stop after this many accepted corrections')
 args=p.parse_args()
 if not 0 < args.prefix_scale <= 1: p.error('--prefix-scale must be in (0, 1]')
 if args.max_episodes is not None and args.max_episodes < 2: p.error('--max-episodes must be at least 2')
 scan=json.loads(args.scan.read_text());args.output_dir.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);device=torch.device('cpu');model,norm,_,sk,ak=load_checkpoint(args.checkpoint,device)
 if ak!=['action_ee_xyz','action_gripper']:raise ValueError('Requires mocap XYZ and absolute gripper actions')
 env=SO100MulticubeSimEnv(ASSETS_DIR/'so100_multicube_ee.xml',seed=scan['seed'])
 writer=MulticubeZarrWriter(args.output_dir/'multicube_corrections.zarr',cube_dim=0,flush_every=1024)
 sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
 writer.set_attrs(source='automated_multicube_policy_takeover',seed=scan['seed'],control_hz=10.,joint_names=list(JOINT_NAMES),cube_colors=list(CUBE_COLORS),checkpoint_sha256=sha)
 report=dict(source='automated_multicube_policy_takeover',checkpoint=str(args.checkpoint),checkpoint_sha256=sha,scan=str(args.scan),seed=scan['seed'],attempts=[],saved_episodes=0)
 report['prefix_scale']=args.prefix_scale
 commands=[]
 try:
  for item in scan['results']:
   env.set_goal(item['color']);obs=env.reset()
   assert np.allclose(env.get_all_cubes_xyz().reshape(3,3),item['initial_cubes'])
   if item['success']:continue
   planned={'no_lift':140,'wrong_cube_lift':100,'not_released':240,'transport_or_place':180,'wrong_cube':100}[item['failure']]
   n_prefix=min(max(4,int(planned*args.prefix_scale)),max(4,item['steps']-10));queue=[];prefix=[];invalid=False
   for t in range(n_prefix):
    if not queue:queue.extend(infer_action_chunk(model,norm,obs,sk,device))
    action=queue.pop(0)
    prefix.append(dict(state={k:np.asarray(v).tolist() for k,v in obs.items()},mocap_target=env.data.mocap_pos[0].tolist(),action=np.asarray(action).tolist()))
    apply_action(env,action,ak);obs=env.step()
    if check_cube_out_of_bounds(env) or check_wrong_cube_in_bin(env) is not None:invalid=True;break
   takeover={k:np.asarray(v).tolist() for k,v in obs.items()}
   if invalid:
    rows=[];result=dict(accepted=False,reason='irrecoverable_prefix')
   else:
    rows,_,result=rollout(env,resume=True)
   result.update(scan_episode=item['episode'],goal_color=item['color'],baseline_failure=item['failure'],policy_prefix=prefix,prefix_steps=len(prefix),takeover_state=takeover,saved_episode=None)
   if result['accepted']:
    result['saved_episode']=writer.num_episodes
    for row in rows:writer.append_with_goal(*row[:8]);commands.append(row[8])
    writer.end_episode()
   report['attempts'].append(result);report.update(saved_episodes=writer.num_episodes,raw_steps=writer.num_steps_total)
   (args.output_dir/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
   print(f"Scene {item['episode']}: {item['failure']}, prefix={len(prefix)}, accepted={result['accepted']}, saved={writer.num_episodes}",flush=True)
   if args.max_episodes is not None and writer.num_episodes>=args.max_episodes:break
 finally:
  writer.flush()
  if commands:writer.root['data'].create_array('action_mocap_delta',data=np.asarray(commands,dtype=np.float32))
  env.renderer.close()
 if writer.num_episodes<2:raise SystemExit('Too few successful corrections')
 print(f'Saved {writer.num_episodes} policy-state expert corrections.')

if __name__=='__main__':main()
