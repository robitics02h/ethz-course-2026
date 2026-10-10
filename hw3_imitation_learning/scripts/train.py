"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

from torch.utils.data import DataLoader, Subset

EPOCHS = 220
BATCH_SIZE = 512
LR = 1e-3
VAL_SPLIT = 0.1


class TensorBatchLoader:
    """Gather whole batches from normalized tensors without per-sample Python work."""

    def __init__(self, states, actions, starts, chunk_size, batch_size, shuffle, sample_weights=None):
        self.states, self.actions = states, actions
        self.starts = torch.as_tensor(starts, dtype=torch.long)
        self.offsets = torch.arange(chunk_size)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.sample_weights = sample_weights

    def __iter__(self):
        starts = self.starts
        if self.sample_weights is not None:
            starts = starts[torch.multinomial(self.sample_weights, len(starts), replacement=True)]
        elif self.shuffle:
            starts = starts[torch.randperm(len(starts))]
        for indices in starts.split(self.batch_size):
            yield self.states[indices], self.actions[indices[:, None] + self.offsets]

    def __len__(self):
        return (len(self.starts) + self.batch_size - 1) // self.batch_size


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    state_noise: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states, action_chunks = states.to(device), action_chunks.to(device)
        if state_noise:
            states = states + torch.randn_like(states) * state_noise
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(states)
        n_batches += len(states)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states, action_chunks = states.to(device), action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item() * len(states)
        n_batches += len(states)

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Action chunk horizon H (default: 4).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--extra-zarr", type=Path, nargs="+", default=[])
    parser.add_argument("--extra-data-fraction", type=float,
                        help="Expected fraction of training samples drawn from --extra-zarr; uses weighted sampling with replacement")
    parser.add_argument("--new-data-fraction", type=float,
                        help="Fine-tuning only: sample this fraction from datasets appended after every data_path in the initial checkpoint")
    parser.add_argument("--transport-sample-multiplier", type=float, default=1.,
                        help="Goal-relative policy only: multiply sampling weight for lifted, held target states away from the bin")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Fine-tune weights with frozen normalization and original validation episodes; append new data via --extra-zarr")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--goal-relative", action="store_true",
                        help="Canonical target selection and relative geometry for the 24D multicube state")
    parser.add_argument("--state-noise", type=float, default=0.01,
                        help="Gaussian training noise in normalized state units")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop early without changing the planned cosine schedule")
    args = parser.parse_args()
    if not np.isfinite(args.transport_sample_multiplier) or args.transport_sample_multiplier <= 0:
        parser.error("--transport-sample-multiplier must be finite and positive")
    if args.transport_sample_multiplier != 1 and not args.goal_relative:
        parser.error("Transport sampling requires the documented goal-relative state layout")
    if args.extra_data_fraction is not None and (not args.extra_zarr or not 0 <= args.extra_data_fraction < 1):
        parser.error("--extra-data-fraction requires --extra-zarr and a value in [0, 1)")
    if args.new_data_fraction is not None:
        if not args.init_checkpoint or not 0 <= args.new_data_fraction < 1:
            parser.error("--new-data-fraction requires --init-checkpoint and a value in [0, 1)")
        if args.extra_data_fraction is not None:
            parser.error("Choose only one of --new-data-fraction and --extra-data-fraction")
    if min(args.epochs, args.batch_size, args.chunk_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, chunk size, threads and learning rate must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.state_noise < 0 or min(args.d_model, args.depth) < 1:
        parser.error("Noise must be nonnegative and model dimensions must be positive")
    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root_attrs = zarr_lib.open_group(str(args.zarr), mode="r").attrs
    args.state_keys = args.state_keys or [root_attrs.get("state_key", "state")]
    args.action_keys = args.action_keys or [root_attrs.get("action_key", "action")]
    if args.goal_relative:
        expected = ["state_ee_xyz", "state_gripper", "state_joints[:5]",
                    "original_pos_cube_red[:3]", "original_pos_cube_green[:3]",
                    "original_pos_cube_blue[:3]", "state_goal", "goal_pos"]
        if args.policy != "multitask" or args.state_keys != expected:
            parser.error("--goal-relative requires multitask policy and the documented 24D state-key order")
    if args.policy == "multitask":
        required = {"original_pos_cube_red", "original_pos_cube_green",
                    "original_pos_cube_blue", "state_goal", "goal_pos"}
        missing = required - {key.split("[")[0] for key in args.state_keys}
        if missing:
            parser.error(f"Multitask policy requires goal-conditioning inputs: {sorted(missing)}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    initial_checkpoint = None
    initial_sha256 = None
    if args.init_checkpoint:
        if args.output and args.output.resolve() == args.init_checkpoint.resolve():
            parser.error("Fine-tuning output must differ from the initial checkpoint")
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        initial_sha256 = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
        expected_config = {"policy_type": args.policy, "state_keys": args.state_keys,
                           "action_keys": args.action_keys, "chunk_size": args.chunk_size,
                           "d_model": args.d_model, "depth": args.depth,
                           "goal_relative": args.goal_relative}
        for key, value in expected_config.items():
            if initial_checkpoint.get(key, False if key == "goal_relative" else None) != value:
                parser.error(f"Initial checkpoint configuration mismatch: {key}")
        original_paths = [Path(p).resolve() for p in initial_checkpoint["data_paths"]]
        if [p.resolve() for p in zarr_paths[:len(original_paths)]] != original_paths:
            parser.error("Preserving validation episode IDs requires original datasets first, in the same order")
        if args.new_data_fraction is not None and len(zarr_paths) <= len(original_paths):
            parser.error("--new-data-fraction requires at least one appended dataset")

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    # Hold out whole episodes, preventing overlap between train/validation chunks.
    if len(ep_ends) < 2:
        raise ValueError("Need at least two episodes for held-out validation")
    rng = np.random.default_rng(args.seed)
    if initial_checkpoint is not None:
        val_episodes = np.asarray(initial_checkpoint["validation_episodes"], dtype=np.int64)
        if not len(val_episodes) or np.any(val_episodes < 0) or np.any(val_episodes >= len(ep_ends)):
            raise ValueError("Initial checkpoint validation episodes are invalid for this dataset")
    else:
        val_episodes = rng.permutation(len(ep_ends))[:max(1, round(len(ep_ends) * VAL_SPLIT))]
    episode_ids = np.searchsorted(ep_ends, np.arange(len(states)), side="right")
    val_mask = np.isin(episode_ids, val_episodes)
    if initial_checkpoint is not None:
        normalizer = Normalizer(**{k: np.asarray(v, dtype=np.float32)
                                   for k, v in initial_checkpoint["normalizer"].items()})
    else:
        normalizer = Normalizer.from_data(states[~val_mask], actions[~val_mask])

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    chunk_is_val = val_mask[dataset.indices]
    train_ds = Subset(dataset, np.flatnonzero(~chunk_is_val).tolist())
    val_ds = Subset(dataset, np.flatnonzero(chunk_is_val).tolist())
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Training and validation episodes must contain full action chunks")
    print(f"Held-out episodes: {val_episodes.tolist()}")

    normalized_states = torch.from_numpy(normalizer.normalize_state(states).astype(np.float32))
    normalized_actions = torch.from_numpy(normalizer.normalize_action(actions).astype(np.float32))
    train_starts = dataset.indices[~chunk_is_val]
    sample_weights = None
    sampling_fraction = args.new_data_fraction if args.new_data_fraction is not None else args.extra_data_fraction
    sampling_boundary = None
    if sampling_fraction is not None:
        base_paths = zarr_paths[:len(original_paths)] if args.new_data_fraction is not None else [args.zarr]
        base_end = sum(int(zarr_lib.open_group(str(path), mode="r")["meta/episode_ends"][-1]) for path in base_paths)
        sampling_boundary = base_end
        extra = torch.as_tensor(train_starts >= base_end)
        if not extra.any() or extra.all():
            parser.error("Both base and extra datasets must contain training chunks")
        sample_weights = torch.where(extra,
            sampling_fraction / extra.sum().double(),
            (1 - sampling_fraction) / (~extra).sum().double())
        print(f"New/extra-data sampling fraction: {sampling_fraction:.1%}; boundary={base_end} (with replacement)")
    transport_sampling_fraction = None
    if args.transport_sample_multiplier != 1:
        raw = states[train_starts]
        target = raw[:, 9:18].reshape(-1, 3, 3)[np.arange(len(raw)), raw[:, 18:21].argmax(axis=1)]
        transport = torch.as_tensor((target[:, 2] > .08) & (raw[:, 3] < .6)
            & (np.linalg.norm(target - raw[:, :3], axis=1) < .085)
            & (np.linalg.norm(target[:, :2] - raw[:, 21:23], axis=1) > .04))
        if sample_weights is None:
            sample_weights = torch.ones(len(train_starts), dtype=torch.float64)
        prior_fraction = float(sample_weights[transport].sum() / sample_weights.sum())
        sample_weights = sample_weights * torch.where(transport, args.transport_sample_multiplier, 1.)
        transport_sampling_fraction = float(sample_weights[transport].sum() / sample_weights.sum())
        print(f"Transport sampling: {prior_fraction:.1%} -> {transport_sampling_fraction:.1%}")
    train_loader = TensorBatchLoader(normalized_states, normalized_actions,
                                    train_starts, args.chunk_size,
                                    args.batch_size, shuffle=True, sample_weights=sample_weights)
    val_loader = TensorBatchLoader(normalized_states, normalized_actions,
                                  dataset.indices[chunk_is_val], args.chunk_size,
                                  args.batch_size, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model, depth=args.depth,
    ).to(device)

    if args.goal_relative:
        model.configure_goal_features(normalizer.state_mean, normalizer.state_std)
    initial_val_loss = None
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        initial_val_loss = evaluate(model, val_loader, device)
        print(f"Fine-tuning {args.init_checkpoint}; fixed normalization/validation; initial val {initial_val_loss:.6f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.02)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = args.output or ckpt_dir / save_name
    if args.init_checkpoint and save_path.resolve() == args.init_checkpoint.resolve():
        parser.error("Fine-tuning output must differ from the initial checkpoint")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    stop_epoch = min(args.epochs, args.stop_after or args.epochs)
    for epoch in range(1, stop_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.state_noise)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "planned_epochs": args.epochs,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                    "init_checkpoint_sha256": initial_sha256,
                    "initial_val_loss": initial_val_loss,
                    "learning_rate": args.lr,
                    "extra_data_fraction": args.extra_data_fraction,
                    "new_data_fraction": args.new_data_fraction,
                    "sampling_boundary": sampling_boundary,
                    "transport_sample_multiplier": args.transport_sample_multiplier,
                    "transport_sampling_fraction": transport_sampling_fraction,
                    "seed": args.seed,
                    "validation_episodes": val_episodes.tolist(),
                    "data_paths": [str(p) for p in zarr_paths],
                    "ee_action_source": root_attrs.get("ee_action_source", "state-delta"),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "state_noise": args.state_noise,
                    "goal_relative": args.goal_relative,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    save_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
