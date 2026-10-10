from pathlib import Path
import json,torch,numpy as np
from hw3.eval_utils import load_checkpoint,infer_action_chunk,apply_action,check_success,check_cube_out_of_bounds
from hw3.sim_env import SO100SimEnv
from so101_gym.constants import ASSETS_DIR
torch.set_num_threads(1)
m,n,h,sk,ak=load_checkpoint(Path('ex1.pt'),torch.device('cpu'))
e=SO100SimEnv(ASSETS_DIR/'so100_transfer_cube_obstacle_ee.xml',seed=42)
results=[]
for ep in range(100):
 o=e.reset();queue=[];entry=None;stable=0
 for t in range(800):
  if not queue:queue.extend(infer_action_chunk(m,n,o,sk,torch.device('cpu')))
  apply_action(e,queue.pop(0),ak);o=e.step()
  inside=check_success(e)
  if inside and entry is None:entry=t+1
  released=inside and o['gripper'][0]>.8 and np.linalg.norm(o['ee_pos']-o['cube'][:3])>.08
  stable=stable+1 if released else 0
  if stable>=10 or check_cube_out_of_bounds(e):break
 results.append({'episode':ep+1,'bin_entry':entry is not None,'released_and_stable':stable>=10,'steps':t+1})
 if (ep+1)%10==0: print(ep+1,'entry',sum(r['bin_entry'] for r in results),'released',sum(r['released_and_stable'] for r in results),flush=True)
e.renderer.close()
report={'seed':42,'episodes':100,'bin_entry_successes':sum(r['bin_entry'] for r in results),'released_and_stable_successes':sum(r['released_and_stable'] for r in results),'release_criterion':'10 consecutive control steps in bin, gripper > 0.8 rad, EE-cube distance > 0.08 m','results':results}
Path('artifacts/ex1/release_validation.json').write_text(json.dumps(report,indent=2)+'\n')
