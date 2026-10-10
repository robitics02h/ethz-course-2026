"""Ordinary-demonstration coverage experiment; never overwrite submission files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zarr

P = Path('artifacts/ex3/coverage')
RAW = Path('datasets/raw/multi_cube/scripted/ex3_coverage240')
DATA = 'datasets/processed/multi_cube/ex3_coverage240.zarr'
BASE = 'datasets/processed/multi_cube/ex3_initial60.zarr'
MODELS = {'warm': 'checkpoints/multi_cube/ex3_coverage300_warm.pt',
          'scratch': 'checkpoints/multi_cube/ex3_coverage300_scratch.pt'}

def run(name, args):
    print(name, flush=True)
    with (P/f'{name}.log').open('w') as log:
        subprocess.run(['.venv/bin/python', '-u', *args], stdout=log, stderr=subprocess.STDOUT, check=True)

manifest = dict(status='collecting',new_episodes=240,total_episodes=300,
    collection_seed=20273000,source='Ordinary successful scripted physical demonstrations; no recovery suffixes',
    development=dict(seed=20274000,episodes=200,selection='Higher strict success; prefer warm if tied'),
    confirmation=dict(seed=20275000,episodes=1000,used_for_selection=False),
    promotion_rule='Selected model strictly beats incumbent with exact paired two-sided p<0.05, is not worse than the saved 99-point reference on confirmation, and official score >=99',
    variants={'warm':dict(initial='checkpoints/multi_cube/ex3_relative60.pt',epochs=40,lr=3e-5,validation='original fixed six episodes',normalizer='frozen'),
              'scratch':dict(initial=None,epochs=100,lr=.001,validation='30 held-out episodes selected by seed42',normalizer='fit on training split')},
    limitations=['Variants differ in initialization, normalization and validation split; this is practical model selection, not an isolated causal test.'],
    hashes={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in ['ex3.pt','ex3_result.hwresult','checkpoints/multi_cube/ex3_recovery_weight0.pt']})
(P/'run_info.json').write_text(json.dumps(manifest,indent=2)+'\n')
deadline = time.monotonic()+3600
while True:
    try:
        report = json.loads((RAW/'collection_report.json').read_text())
        data = zarr.open_group(str(RAW/'multicube_scripted.zarr'),mode='r')['data']
        if report['saved_episodes']==240 and 'action_mocap_delta' in data and data['action_mocap_delta'].shape[0]==report['raw_steps']:
            break
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    if time.monotonic()>deadline:raise TimeoutError('Collection did not complete')
    time.sleep(5)
run('audit',['scripts/audit_multicube_demos.py','--dataset-dir',str(RAW),'--output',str(P/'audit.json')])
run('process',['scripts/compute_actions.py','--action-space','ee','--ee-action-source','command',
    '--datasets-dir',str(RAW),'--output',DATA])
common = ['scripts/train.py','--policy','multitask','--goal-relative','--zarr',BASE,
    '--extra-zarr',DATA,'--state-keys','state_ee_xyz','state_gripper','state_joints[:5]',
    'original_pos_cube_red[:3]','original_pos_cube_green[:3]','original_pos_cube_blue[:3]',
    'state_goal','goal_pos','--action-keys','action_ee_xyz','action_gripper',
    '--chunk-size','4','--d-model','512','--depth','3','--threads','2']
def train(name):
    options = ['--init-checkpoint','checkpoints/multi_cube/ex3_relative60.pt','--epochs','40','--lr','0.00003'] if name=='warm' else ['--epochs','100','--lr','0.001']
    run('train_'+name,common+options+['--output',MODELS[name]])
with ThreadPoolExecutor(2) as pool:
    list(pool.map(train,MODELS))
for name,checkpoint in MODELS.items():
    run('dev_'+name,['scripts/evaluate_multicube_parallel.py','--checkpoint',checkpoint,
        '--seed','20274000','--episodes','200','--workers','4','--output',str(P/f'dev_{name}.json')])
scores = {name:json.loads((P/f'dev_{name}.json').read_text())['stable_successes'] for name in MODELS}
selected = max(MODELS,key=lambda name:(scores[name],name=='warm'))
(P/'selection.json').write_text(json.dumps(dict(scores=scores,selected=selected,checkpoint=MODELS[selected]),indent=2)+'\n')
run('candidate1000',['scripts/evaluate_multicube_parallel.py','--checkpoint',MODELS[selected],
    '--seed','20275000','--episodes','1000','--workers','4','--output',str(P/'candidate1000.json')])
run('official',['student_eval/run_eval.py','--exercise','3','--checkpoint',MODELS[selected],
    '--output',str(P/'candidate.hwresult')])
print('Experiment complete; submission files unchanged.',flush=True)
