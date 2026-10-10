"""Run the predeclared early-takeover experiment; preserve submission files."""
import json
import subprocess
import time
from pathlib import Path

PY = '.venv/bin/python'
OUT = Path('artifacts/ex3/early_recovery')
KNOWN = 'datasets/raw/multi_cube/dagger/ex3_early_known_verified'
FRESH = 'datasets/raw/multi_cube/dagger/ex3_early_fresh'
KNOWN_PROCESSED = 'datasets/processed/multi_cube/ex3_early_known.zarr'
FRESH_PROCESSED = 'datasets/processed/multi_cube/ex3_early_fresh.zarr'
CHECKPOINT = 'checkpoints/multi_cube/ex3_early_recovery.pt'

def run(name, args):
    print(name, flush=True)
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run([PY, '-u', *args], stdout=log, stderr=subprocess.STDOUT, check=True)

run('known_revised_audit', ['scripts/audit_multicube_demos.py', '--dataset-dir', KNOWN,
    '--output', str(OUT / 'known_revised_audit.json')])
deadline = time.monotonic() + 7200
while not (OUT / 'scan900.json').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Failure scan did not complete')
    time.sleep(10)
run('fresh_collection', ['scripts/collect_multicube_corrections.py', '--checkpoint', 'ex3.pt',
    '--scan', str(OUT / 'scan900.json'), '--prefix-scale', '0.6', '--max-episodes', '36',
    '--output-dir', FRESH])
run('fresh_audit', ['scripts/audit_multicube_demos.py', '--dataset-dir', FRESH,
    '--output', str(OUT / 'fresh_audit.json')])
for name, raw, processed in [('known', KNOWN, KNOWN_PROCESSED), ('fresh', FRESH, FRESH_PROCESSED)]:
    run(name + '_process', ['scripts/compute_actions.py', '--action-space', 'ee',
        '--ee-action-source', 'command', '--datasets-dir', raw, '--output', processed])
run('train', ['scripts/train.py', '--policy', 'multitask', '--goal-relative',
    '--init-checkpoint', 'checkpoints/multi_cube/ex3_relative60.pt',
    '--zarr', 'datasets/processed/multi_cube/ex3_initial60.zarr',
    '--extra-zarr', KNOWN_PROCESSED, FRESH_PROCESSED,
    '--state-keys', 'state_ee_xyz', 'state_gripper', 'state_joints[:5]',
    'original_pos_cube_red[:3]', 'original_pos_cube_green[:3]', 'original_pos_cube_blue[:3]',
    'state_goal', 'goal_pos', '--action-keys', 'action_ee_xyz', 'action_gripper',
    '--chunk-size', '4', '--d-model', '512', '--depth', '3', '--threads', '2',
    '--epochs', '40', '--lr', '0.00003', '--output', CHECKPOINT])
run('candidate500', ['scripts/evaluate_multicube_parallel.py', '--checkpoint', CHECKPOINT,
    '--seed', '20268000', '--episodes', '500', '--workers', '4', '--output', str(OUT / 'candidate500.json')])
run('official', ['student_eval/run_eval.py', '--exercise', '3', '--checkpoint', CHECKPOINT,
    '--output', str(OUT / 'candidate.hwresult')])
print('Experiment complete; submission files unchanged.', flush=True)
