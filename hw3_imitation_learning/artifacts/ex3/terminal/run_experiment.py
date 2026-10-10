"""Terminal-suffix ablation with a matched zero-suffix continuation control."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

P=Path('artifacts/ex3/terminal')
RAW='datasets/raw/multi_cube/scripted/ex3_terminal60'
DATA='datasets/processed/multi_cube/ex3_terminal60.zarr'
INIT='checkpoints/multi_cube/ex3_coverage300_warm.pt'
MODELS={'control':'checkpoints/multi_cube/ex3_terminal_control.pt',
        'candidate':'checkpoints/multi_cube/ex3_terminal5.pt'}

def run(name,args):
    print(name,flush=True)
    with (P/f'{name}.log').open('w') as log:
        subprocess.run(['.venv/bin/python','-u',*args],stdout=log,stderr=subprocess.STDOUT,check=True)

manifest=dict(status='running',initial_checkpoint=INIT,
    intervention='60 physical open-retreat-hold suffixes, 5% sampling; all original 300 demonstrations retained',
    generation=dict(seed=20276000,episodes=60,place_offset_m=[.006,.006,.012],retreat_offset_m=[.008,.008,.01],hold_steps=[90,110],control_hz=10),
    training=dict(epochs=20,lr=1e-5,seed=42,normalizer='frozen',validation='original six episode IDs',selection='lowest validation action MSE',control='same update count, seed, replacement sampling; new-data fraction zero'),
    development=dict(seed=20277000,episodes=500,hold_episodes=200,post_success_steps=100),
    confirmation=dict(seed=20278000,episodes=5000,gate='5% candidate must be no worse than both incumbent and zero-suffix control on regular and hold development tests'),
    promotion_rule='Independent confirmation must improve incumbent with exact paired two-sided p<0.05 and official score >=99; otherwise preserve current submission',
    hashes={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in ['ex3.pt','ex3_result.hwresult',INIT]})
(P/'run_info.json').write_text(json.dumps(manifest,indent=2)+'\n')
run('audit',['scripts/audit_multicube_demos.py','--dataset-dir',RAW,'--output',str(P/'audit.json')])
run('process',['scripts/compute_actions.py','--action-space','ee','--ee-action-source','command','--datasets-dir',RAW,'--output',DATA])
common=['scripts/train.py','--policy','multitask','--goal-relative','--init-checkpoint',INIT,
    '--zarr','datasets/processed/multi_cube/ex3_initial60.zarr',
    '--extra-zarr','datasets/processed/multi_cube/ex3_coverage240.zarr',DATA,
    '--state-keys','state_ee_xyz','state_gripper','state_joints[:5]',
    'original_pos_cube_red[:3]','original_pos_cube_green[:3]','original_pos_cube_blue[:3]',
    'state_goal','goal_pos','--action-keys','action_ee_xyz','action_gripper',
    '--chunk-size','4','--d-model','512','--depth','3','--threads','2',
    '--epochs','20','--lr','0.00001']
def train(name):
    run('train_'+name,common+['--new-data-fraction','0' if name=='control' else '.05','--output',MODELS[name]])
with ThreadPoolExecutor(2) as pool:list(pool.map(train,MODELS))
def evaluate(name,checkpoint,episodes,seed=20277000,hold=0):
    run(name,['scripts/evaluate_multicube_parallel.py','--checkpoint',checkpoint,
        '--seed',str(seed),'--episodes',str(episodes),'--workers','4',
        '--post-success-steps',str(hold),'--output',str(P/f'{name}.json')])
for name,ck in MODELS.items():
    evaluate('dev_'+name,ck,500)
    evaluate('hold_'+name,ck,200,hold=100)
deadline=time.monotonic()+1200
while not all((P/f'{kind}_baseline.json').exists() for kind in ['dev','hold']):
    if time.monotonic()>deadline:raise TimeoutError('Baseline diagnostics incomplete')
    time.sleep(5)
scores={kind:{name:json.loads((P/f'{kind}_{name}.json').read_text())['stable_successes']
    for name in ['baseline','control','candidate']} for kind in ['dev','hold']}
gate=all(s['candidate']>=max(s['baseline'],s['control']) for s in scores.values())
(P/'gate.json').write_text(json.dumps(dict(scores=scores,passed=gate),indent=2)+'\n')
run('official',['student_eval/run_eval.py','--exercise','3','--checkpoint',MODELS['candidate'],'--output',str(P/'candidate.hwresult')])
if gate:
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(evaluate,'confirm_baseline','ex3.pt',5000,20278000),
                 pool.submit(evaluate,'confirm_candidate',MODELS['candidate'],5000,20278000)]
        for future in futures:future.result()
print('Experiment complete; submission files unchanged.',flush=True)
