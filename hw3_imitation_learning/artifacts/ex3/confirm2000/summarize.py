import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

p = Path('artifacts/ex3/confirm2000')
report = json.loads((p/'run_info.json').read_text())
rounds = []
all_pairs = []
for suffix in ['a','b']:
    base = json.loads((p/f'base_{suffix}.json').read_text())
    candidate = json.loads((p/f'candidate_{suffix}.json').read_text())
    assert base['seed'] == candidate['seed'] and len(base['results']) == len(candidate['results']) == 1000
    pairs = list(zip(base['results'], candidate['results']))
    for x,y in pairs:
        assert x['episode'] == y['episode'] and x['color'] == y['color']
        assert x['initial_cubes'] == y['initial_cubes'] and x['bin_position'] == y['bin_position']
    rounds.append(dict(seed=base['seed'],baseline_successes=base['stable_successes'],
        candidate_successes=candidate['stable_successes'],
        improved=[x['episode'] for x,y in pairs if not x['success'] and y['success']],
        regressed=[x['episode'] for x,y in pairs if x['success'] and not y['success']]))
    all_pairs.extend(pairs)
n = len(all_pairs)
fixed = sum(not x['success'] and y['success'] for x,y in all_pairs)
regressed = sum(x['success'] and not y['success'] for x,y in all_pairs)
discordant = fixed + regressed
pvalue = min(1., 2*sum(math.comb(discordant,i) for i in range(min(fixed,regressed)+1))/2**discordant) if discordant else 1.
rng = np.random.default_rng(20271002)
draws = rng.multinomial(n, [fixed/n,regressed/n,1-discordant/n],size=20000)
ci = np.quantile((draws[:,0]-draws[:,1])/n,[.025,.975])
totals = dict(episodes=n,baseline_successes=sum(x['success'] for x,y in all_pairs),
    candidate_successes=sum(y['success'] for x,y in all_pairs),improved=fixed,regressed=regressed,
    net_improvement_percentage_points=100*(fixed-regressed)/n,
    paired_bootstrap_95percent_difference_percentage_points=(100*ci).tolist(),
    exact_two_sided_mcnemar_p=pvalue)
by_color = {c:dict(episodes=sum(x['color']==c for x,y in all_pairs),
    baseline_successes=sum(x['success'] for x,y in all_pairs if x['color']==c),
    candidate_successes=sum(y['success'] for x,y in all_pairs if x['color']==c)) for c in ['red','green','blue']}
eligible = fixed > regressed and pvalue < .05 and all(r['candidate_successes']>r['baseline_successes'] for r in rounds)
report.update(status='evaluated',rounds=rounds,totals=totals,by_color=by_color,
    failures={name:dict(Counter(pair[i]['failure'] for pair in all_pairs if pair[i]['failure'])) for i,name in enumerate(['baseline','candidate'])},
    promotion_supported=eligible,layouts_verified_identical=True,
    statistical_limit='Simulation reset distribution only; one trained candidate. Bootstrap resamples paired episodes, not training runs. Previously inspected test sets excluded.')
(p/'run_info.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(rounds=rounds,totals=totals,by_color=by_color,promotion_supported=eligible),indent=2))
