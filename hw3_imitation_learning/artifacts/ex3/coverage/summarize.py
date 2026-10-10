from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
import torch

p = Path('artifacts/ex3/coverage')
r = json.loads((p/'run_info.json').read_text())
selection = json.loads((p/'selection.json').read_text())
reports = {n:json.loads((p/f'{n}.json').read_text()) for n in ['incumbent1000','reference99_1000','candidate1000']}
candidate = reports['candidate1000']
comparisons = {}
for name in ['incumbent1000','reference99_1000']:
    base = reports[name]
    assert base['seed']==candidate['seed']==20275000
    pairs = list(zip(base['results'],candidate['results']))
    assert len(pairs)==1000
    for x,y in pairs:
        assert x['episode']==y['episode'] and x['color']==y['color']
        assert x['initial_cubes']==y['initial_cubes'] and x['bin_position']==y['bin_position']
    fixed = [x['episode'] for x,y in pairs if not x['success'] and y['success']]
    regressed = [x['episode'] for x,y in pairs if x['success'] and not y['success']]
    total = len(fixed)+len(regressed)
    pv = min(1.,2*sum(math.comb(total,i) for i in range(min(len(fixed),len(regressed))+1))/2**total) if total else 1.
    draws = np.random.default_rng(20275001).multinomial(1000,[len(fixed)/1000,len(regressed)/1000,1-total/1000],size=20000)
    interval = np.quantile((draws[:,0]-draws[:,1])/10,[.025,.975])
    comparisons[name] = dict(baseline_successes=base['stable_successes'],candidate_successes=candidate['stable_successes'],
        improved=fixed,regressed=regressed,exact_two_sided_mcnemar_p=pv,
        paired_bootstrap_95percent_difference_percentage_points=interval.tolist())
official = int(re.search(r'Success rate:\s*(\d+)/100',(p/'official.log').read_text()).group(1))
eligible = (candidate['stable_successes']>reports['incumbent1000']['stable_successes']
    and comparisons['incumbent1000']['exact_two_sided_mcnemar_p']<.05
    and candidate['stable_successes']>=reports['reference99_1000']['stable_successes'] and official>=99)
training = {}
for name in ['warm','scratch']:
    ck=torch.load(f'checkpoints/multi_cube/ex3_coverage300_{name}.pt',map_location='cpu',weights_only=False)
    training[name] = {k:ck[k] for k in ['epoch','val_loss','validation_episodes','data_paths','learning_rate','chunk_size','goal_relative']}
for path,sha in r['hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
r.update(status='evaluated',selection=selection,training_results=training,
    audit=json.loads((p/'audit.json').read_text())|{'results': 'See audit.json for per-episode replay checks'},
    confirmation_results={n:{k:d[k] for k in ['episodes','stable_successes','failures']} for n,d in reports.items()},
    comparisons=comparisons,official_successes=official,official_episodes=100,
    promotion_supported=eligible,selected_checkpoint_sha256=hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest(),
    integrity_verified=True)
(p/'run_info.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(selection=selection,results=r['confirmation_results'],official=official,promotion_supported=eligible,
    comparisons={n:{k:v for k,v in d.items() if k not in ['improved','regressed']}|{'fixed':len(d['improved']),'regressed':len(d['regressed'])} for n,d in comparisons.items()}),indent=2))
