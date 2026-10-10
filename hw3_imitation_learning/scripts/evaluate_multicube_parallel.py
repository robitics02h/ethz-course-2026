"""Shard strict diagnostics by scene index, preserving the reset RNG sequence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--post-success-steps', type=int, default=0)
    p.add_argument('--target-lag-limit', type=float, default=0.,
                   help='Privileged diagnostic only; zero preserves original action execution')
    p.add_argument('--trace-dir', type=Path)
    args = p.parse_args()
    if args.episodes < 1 or args.workers < 1:
        p.error('episodes and workers must be positive')
    if args.post_success_steps < 0:p.error('post-success steps must be nonnegative')
    if not math.isfinite(args.target_lag_limit) or args.target_lag_limit < 0:
        p.error('target-lag-limit must be finite and nonnegative')
    workers = min(args.workers, args.episodes)
    directory = args.output.parent / (args.output.stem + '_shards')
    directory.mkdir(parents=True, exist_ok=True)

    def run(worker):
        start = worker * args.episodes // workers
        end = (worker + 1) * args.episodes // workers
        output = directory / f'{worker}.json'
        command = [sys.executable, '-u', str(Path(__file__).with_name('diagnose_multicube_policy.py')),
            '--checkpoint', str(args.checkpoint), '--seed', str(args.seed),
            '--episodes', str(end), '--scene-indices', *map(str, range(start, end)),
            '--output', str(output), '--post-success-steps', str(args.post_success_steps),
            '--target-lag-limit', str(args.target_lag_limit)]
        if args.trace_dir:command.extend(['--trace-dir', str(args.trace_dir)])
        with (directory / f'{worker}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.read_text())
        assert [r['episode'] for r in report['results']] == list(range(start, end))
        print(f'Completed scenes {start}–{end-1}: {report["stable_successes"]}/{end-start}', flush=True)
        return report['results']

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [item for shard in pool.map(run, range(workers)) for item in shard]
    assert [r['episode'] for r in results] == list(range(args.episodes))
    report = dict(checkpoint=str(args.checkpoint), seed=args.seed, execute_steps=4, post_success_steps=args.post_success_steps,
        target_lag_limit=args.target_lag_limit, privileged_intervention=bool(args.target_lag_limit),
        episodes=len(results), stable_successes=sum(r['success'] for r in results),
        bin_entries=sum(r['bin_entry'] for r in results),
        failures=dict(Counter(r['failure'] for r in results if r['failure'])), results=results,
        workers=workers, sharding='Each worker replays all preceding resets; only assigned scenes execute actions')
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
