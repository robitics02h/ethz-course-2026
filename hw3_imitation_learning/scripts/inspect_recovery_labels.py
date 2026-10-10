"""Compare nearby encoded states across episodes; label disagreement is diagnostic, not causal proof."""
import json
from pathlib import Path
import numpy as np
import torch
from hw3.dataset import load_zarr, load_and_merge_zarrs
from hw3.eval_utils import load_checkpoint


def main():
    torch.set_num_threads(2)
    checkpoint = torch.load('ex3.pt', map_location='cpu', weights_only=False)
    model, norm, _, sk, ak = load_checkpoint(Path('ex3.pt'), torch.device('cpu'))
    base, actions, ends = load_zarr(Path('datasets/processed/multi_cube/ex3_initial60.zarr'), sk, ak)
    extra, extra_actions, extra_ends = load_and_merge_zarrs([
        Path('datasets/processed/multi_cube/ex3_early_known.zarr'),
        Path('datasets/processed/multi_cube/ex3_early_fresh.zarr')], sk, ak)
    ids = np.searchsorted(ends, np.arange(len(base)), side='right')
    keep = ~np.isin(ids, checkpoint['validation_episodes'])
    base, actions, ids = base[keep], actions[keep], ids[keep]
    with torch.no_grad():
        encoded = model.encode_state(torch.from_numpy(norm.normalize_state(base).astype(np.float32)))
        extra_encoded = model.encode_state(torch.from_numpy(norm.normalize_state(extra).astype(np.float32)))
    def compare(query, query_actions, query_ids=None):
        nearest, distances = [], []
        for start in range(0, len(query), 256):
            distance = torch.cdist(query[start:start+256], encoded) / np.sqrt(encoded.shape[1])
            if query_ids is not None:
                distance[torch.as_tensor(query_ids[start:start+256, None] == ids[None, :])] = float('inf')
            values, indices = distance.min(dim=1)
            nearest.extend(indices.tolist()); distances.extend(values.tolist())
        nearest = np.asarray(nearest); distances = np.asarray(distances)
        disagreement = np.abs(query_actions[:, 3] - actions[nearest, 3]) > .8
        summaries = []
        for threshold in [.05, .1, .2, .4]:
            close = distances < threshold
            summaries.append(dict(max_encoded_rms_distance=threshold,nearby_queries=int(close.sum()),
                conflicting_gripper_queries=int((close & disagreement).sum())))
        examples = np.flatnonzero(disagreement)[np.argsort(distances[disagreement])][:10]
        return dict(queries=len(query),thresholds=summaries,closest_disagreements=[dict(
            query_index=int(i),distance=float(distances[i]),query_action=query_actions[i].tolist(),
            nearest_base_action=actions[nearest[i]].tolist(),base_episode=int(ids[nearest[i]])) for i in examples])
    q = np.arange(0, len(extra), 20)
    b = np.arange(0, len(base), 20)
    report = dict(metric='RMS Euclidean distance in deployed goal-relative encoded features',
        gripper_disagreement='Absolute command difference > 0.8',
        sampling='Every twentieth transition; original validation episodes excluded from base reference',
        recovery_vs_base=compare(extra_encoded[q], extra_actions[q]),
        base_vs_other_base_episode=compare(encoded[b], actions[b], ids[b]),
        limitations='Nearby states need not be behaviorally equivalent. This detects potential ambiguity, not proof of its causal effect on failures.')
    path = Path('artifacts/ex3/recovery_weight/label_inspection.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v['thresholds'] for k,v in report.items() if isinstance(v,dict)},indent=2))


if __name__ == '__main__':
    main()
