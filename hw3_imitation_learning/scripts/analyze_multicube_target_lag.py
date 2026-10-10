"""Reconstruct accumulated targets from ordinary demonstrations without replay.

Only full episodes that start at reset are supported. At reset the mocap target
equals the measured hand position. Each row contains the pre-step hand position
and the command increment applied for that step.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import zarr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zarr', type=Path, nargs='+', required=True)
    p.add_argument('--exclude-episodes', type=int, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    episodes = []
    lags = []
    episode_offset = 0
    for path in args.raw_zarr:
        root = zarr.open_group(str(path), mode='r')
        if root.attrs.get('source') != 'scripted_multicube_expert':
            raise ValueError('Only complete ordinary scripted demonstrations are supported')
        ends = root['meta/episode_ends'][:]
        hand = root['data/state_ee'][:, :3].astype(np.float64)
        delta = root['data/action_mocap_delta'][:].astype(np.float64)
        assert len(hand) == len(delta) == ends[-1]
        start = 0
        for index, end in enumerate(ends):
            target = hand[start] + np.cumsum(delta[start:end], axis=0)
            distance = np.linalg.norm(target - hand[start:end], axis=1)
            excluded = index + episode_offset in args.exclude_episodes
            episodes.append(dict(path=str(path), episode=index + episode_offset,
                excluded=excluded, steps=len(distance), max_lag_m=float(distance.max())))
            if not excluded:lags.extend(distance)
            start = end
        episode_offset += len(ends)
    if not lags:raise ValueError('No training episodes remain')
    maximum = float(np.max(lags))
    bound = round(math.ceil(1.1 * maximum / .01) * .01, 2)
    report = dict(episodes=len(episodes), included_episodes=sum(not e['excluded'] for e in episodes),
        included_steps=len(lags), exclude_episodes=args.exclude_episodes,
        timing='proposed target after command versus pre-physics hand pose',
        reconstruction='recorded reset hand position + cumulative recorded mocap increments',
        float_precision='raw float32 recordings accumulated in float64; original replay audit errors are far smaller than the bound margin',
        percentiles_m={str(q):float(np.percentile(lags, q)) for q in [50, 90, 95, 99, 99.9, 100]},
        diagnostic_limit_m=bound,
        limit_rule='ceil(1.1 * maximum training-demo lag / 0.01) * 0.01; fixed before diagnostic rollouts',
        results=episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
