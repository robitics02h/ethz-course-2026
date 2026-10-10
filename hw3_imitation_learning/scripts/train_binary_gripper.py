"""Train a binary gripper head while freezing the incumbent's entire XYZ network.

Uses exactly the incumbent's datasets, normalization and validation episodes.
Writes a threshold-only control and a head trained with binary cross entropy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hw3.dataset import Normalizer, build_valid_indices, load_and_merge_zarrs
from hw3.model import build_policy
from train import TensorBatchLoader, train_one_epoch


@torch.no_grad()
def validation_metrics(model, loader):
    model.eval()
    loss_sum = errors = count = transition_errors = transition_count = 0
    for state, action in loader:
        logits = model.gripper_logits(state)
        target = action[:, :, 3] > model.gripper_endpoints.mean()
        loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), reduction="sum").item()
        wrong = (logits >= 0) != target
        transition = (target != target[:, :1]).any(dim=1)
        errors += wrong.sum().item()
        count += target.numel()
        transition_errors += wrong[transition].sum().item()
        transition_count += target[transition].numel()
    return dict(bce=loss_sum / count, error_rate=errors / count,
                transition_error_rate=transition_errors / max(transition_count, 1),
                label_count=count, transition_label_count=transition_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--state-noise', type=float, default=.01)
    args = p.parse_args()
    if len({v.resolve() for v in (args.init_checkpoint, args.output, args.control_output)}) != 3:
        p.error('Initial checkpoint, output and control must have different paths')
    if min(args.epochs, args.batch_size, args.threads) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.state_noise) or args.state_noise < 0:
        p.error('Invalid training settings')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    if initial['policy_type'] != 'multitask' or initial['action_keys'] != ['action_ee_xyz', 'action_gripper']:
        p.error('Requires a multicube XYZ + gripper checkpoint')
    if initial['model_state_dict'].get('binary_gripper', False):
        p.error('Initialize from a regression checkpoint')
    states, actions, ends = load_and_merge_zarrs([Path(v) for v in initial['data_paths']],
        state_keys=initial['state_keys'], action_keys=initial['action_keys'])
    endpoints = np.unique(actions[:, 3])
    if len(endpoints) != 2 or not np.allclose(endpoints, [-.174, 1.], atol=1e-6, rtol=0):
        raise ValueError(f'Expected exactly two demonstrated gripper commands, got {endpoints}')
    normalizer = Normalizer(**{k:np.asarray(v, dtype=np.float32) for k,v in initial['normalizer'].items()})
    starts = build_valid_indices(ends, initial['chunk_size'])
    val_ids = np.asarray(initial['validation_episodes'])
    if not len(val_ids) or np.any(val_ids < 0) or np.any(val_ids >= len(ends)):
        raise ValueError('Invalid original validation episodes')
    val_mask = np.isin(np.searchsorted(ends, starts, side='right'), val_ids)
    if not val_mask.any() or val_mask.all():
        raise ValueError('Both training and validation chunks are required')
    normalized_states = torch.from_numpy(normalizer.normalize_state(states).astype(np.float32))
    normalized_actions = torch.from_numpy(normalizer.normalize_action(actions).astype(np.float32))
    train_loader = TensorBatchLoader(normalized_states, normalized_actions, starts[~val_mask],
        initial['chunk_size'], args.batch_size, shuffle=True)
    val_loader = TensorBatchLoader(normalized_states, normalized_actions, starts[val_mask],
        initial['chunk_size'], args.batch_size, shuffle=False)
    model = build_policy('multitask', **{k:initial[k] for k in ('state_dim', 'action_dim', 'chunk_size', 'd_model', 'depth')})
    model.load_state_dict(initial['model_state_dict'])
    model.configure_binary_gripper(normalizer.action_mean, normalizer.action_std)
    frozen = {k:v.clone() for k,v in model.net.state_dict().items()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == initial['chunk_size'] * (initial['d_model'] + 1)
    print(f'Data: {len(ends)} episodes, {len(starts[~val_mask])} train / {val_mask.sum()} validation chunks', flush=True)
    print(f'Frozen original network; trainable head: {sum(p.numel() for p in trainable)} parameters', flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * .02)
    metadata_keys = ('normalizer', 'chunk_size', 'd_model', 'depth', 'goal_relative',
                     'policy_type', 'state_keys', 'action_keys', 'state_dim', 'action_dim',
                     'data_paths', 'validation_episodes', 'ee_action_source')
    metadata = {k:initial[k] for k in metadata_keys}
    metadata.update(init_checkpoint=str(args.init_checkpoint),
        init_checkpoint_sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        planned_epochs=args.epochs, learning_rate=args.lr, seed=args.seed,
        state_noise=args.state_noise, loss_type='binary_gripper_bce',
        frozen_xyz_network=True, closed_command=-.174, open_command=1.,
        initial_logit_temperature=.15, original_regression_val_loss=initial['val_loss'])

    def save(path, epoch, metrics, control=False):
        assert all(torch.equal(v, frozen[k]) for k,v in model.net.state_dict().items()), 'Frozen XYZ network changed'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(metadata, model_state_dict=model.state_dict(),
            optimizer_state_dict=None if control else optimizer.state_dict(),
            epoch=epoch, val_loss=metrics['bce'], validation_metrics=metrics,
            gripper_training='threshold_only' if control else 'frozen_feature_bce'), path)

    metrics = validation_metrics(model, val_loader)
    save(args.control_output, 0, metrics, control=True)
    save(args.output, 0, metrics)
    best = metrics['bce']
    history = [dict(epoch=0, **metrics)]
    print(f'Threshold control: {metrics}', flush=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, torch.device('cpu'), args.state_noise)
        metrics = validation_metrics(model, val_loader)
        scheduler.step()
        improved = metrics['bce'] < best
        if improved:
            best = metrics['bce']
            save(args.output, epoch, metrics)
        history.append(dict(epoch=epoch, train_bce=loss, **metrics))
        print(f'Epoch {epoch}/{args.epochs}: train BCE={loss:.6f}; val={metrics}; saved={improved}', flush=True)
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2) + '\n')
    print(f'Best validation BCE: {best:.6f}; checkpoint: {args.output}', flush=True)


if __name__ == '__main__':
    main()
