"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
import math
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class ObstaclePolicy(BasePolicy):
    """MLP behavior cloning policy predicting normalized action chunks."""

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int = 4,
        d_model: int = 512, depth: int = 3,
    ):
        super().__init__(state_dim, action_dim, chunk_size)
        if min(state_dim, action_dim, chunk_size, d_model, depth) < 1:
            raise ValueError("Policy dimensions and depth must be positive")
        layers = [nn.Linear(state_dim, d_model), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)


class MultiTaskPolicy(ObstaclePolicy):
    """Action-chunk MLP with optional color-invariant target geometry.

    Legacy checkpoints use the flat 24D input. Goal-relative checkpoints select
    the requested cube and encode its geometry relative to the robot and bin;
    all motion and gripper commands are still predicted by the learned network.
    Checkpoint buffers hold the representation, input normalization, and an
    optional binary gripper mode with normalized closed/open output commands.
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int = 4,
                 d_model: int = 512, depth: int = 3):
        super().__init__(state_dim, action_dim, chunk_size, d_model, depth)
        self.register_buffer("goal_features", torch.tensor(False))
        self.register_buffer("input_mean", torch.zeros(state_dim))
        self.register_buffer("input_std", torch.ones(state_dim))
        # Zero initialization leaves the legacy model's RNG sequence unchanged.
        self.register_buffer("binary_gripper", torch.tensor(False))
        self.register_buffer("gripper_endpoints", torch.tensor([-1., 1.]))
        self.gripper_weight = nn.Parameter(torch.zeros(chunk_size, d_model), requires_grad=False)
        self.gripper_bias = nn.Parameter(torch.zeros(chunk_size), requires_grad=False)

    def configure_goal_features(self, state_mean, state_std):
        if self.state_dim != 24:
            raise ValueError("Goal-relative representation requires the documented 24D state order")
        self.input_mean.copy_(torch.as_tensor(state_mean, device=self.input_mean.device))
        self.input_std.copy_(torch.as_tensor(state_std, device=self.input_std.device))
        self.goal_features.fill_(True)

    @torch.no_grad()
    def configure_binary_gripper(self, action_mean, action_std,
                                 closed: float = -.174, opened: float = 1.,
                                 temperature: float = .15):
        """Initialize a frozen-feature classifier at the old regressor's midpoint.

        Endpoints are normalized, because the existing evaluator denormalizes all
        four action coordinates. Only the new classifier parameters are trainable.
        """
        if self.action_dim != 4 or not all(math.isfinite(v) for v in (closed, opened, temperature)) or not closed < opened or temperature <= 0:
            raise ValueError("Binary gripper requires XYZ + gripper actions and ordered endpoints")
        mean = torch.as_tensor(action_mean, device=self.gripper_bias.device, dtype=self.gripper_bias.dtype)
        std = torch.as_tensor(action_std, device=self.gripper_bias.device, dtype=self.gripper_bias.dtype)
        if mean.shape != (4,) or std.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Action normalization must contain four finite means and positive standard deviations")
        self.gripper_endpoints.copy_((mean.new_tensor([closed, opened]) - mean[-1]) / std[-1])
        old_weight = self.net[-1].weight.reshape(self.chunk_size, 4, -1)[:, 3]
        old_bias = self.net[-1].bias.reshape(self.chunk_size, 4)[:, 3]
        self.gripper_weight.copy_(old_weight * std[-1] / temperature)
        self.gripper_bias.copy_((old_bias * std[-1] + mean[-1] - (closed + opened) / 2) / temperature)
        self.net.requires_grad_(False)
        self.gripper_weight.requires_grad_(True)
        self.gripper_bias.requires_grad_(True)
        self.binary_gripper.fill_(True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Legacy flat MLP checkpoints predate these buffers; preserve their behavior.
        names = ("goal_features", "input_mean", "input_std")
        if not any(prefix + name in state_dict for name in names):
            state_dict[prefix + "goal_features"] = torch.zeros_like(self.goal_features)
            state_dict[prefix + "input_mean"] = torch.zeros_like(self.input_mean)
            state_dict[prefix + "input_std"] = torch.ones_like(self.input_std)
        binary_names = ("binary_gripper", "gripper_endpoints", "gripper_weight", "gripper_bias")
        if not any(prefix + name in state_dict for name in binary_names):
            state_dict[prefix + "binary_gripper"] = torch.zeros_like(self.binary_gripper)
            state_dict[prefix + "gripper_endpoints"] = self.gripper_endpoints.new_tensor([-1., 1.])
            state_dict[prefix + "gripper_weight"] = torch.zeros_like(self.gripper_weight)
            state_dict[prefix + "gripper_bias"] = torch.zeros_like(self.gripper_bias)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Also restore which parameters can be trained when loading a checkpoint.
        binary = bool(self.binary_gripper)
        self.net.requires_grad_(not binary)
        self.gripper_weight.requires_grad_(binary)
        self.gripper_bias.requires_grad_(binary)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.goal_features:
            return state
        raw = state * self.input_std + self.input_mean
        ee, cubes, goal = raw[:, :3], raw[:, 9:18].reshape(-1, 3, 3), raw[:, 21:24]
        # Select the requested object before learning motion: color permutations
        # with the same target location produce exactly the same features.
        index = raw[:, 18:21].argmax(dim=1)
        target = cubes[torch.arange(len(state), device=state.device), index]
        center = raw.new_tensor([-.2, .55, .08])
        scale = raw.new_tensor([.05, .2, .1])
        return torch.cat((state[:, :9], (target-ee)/.1, (goal-ee)/.1,
                          (target-goal)/.1, (target-center)/scale,
                          (goal-center)/scale), dim=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_state(state)
        if not self.binary_gripper:
            return super().forward(encoded)
        features = self.net[:-1](encoded)
        actions = self.net[-1](features).reshape(-1, self.chunk_size, self.action_dim)
        logits = nn.functional.linear(features, self.gripper_weight, self.gripper_bias)
        gripper = torch.where(logits >= 0, self.gripper_endpoints[1], self.gripper_endpoints[0])
        return torch.cat((actions[:, :, :3], gripper.unsqueeze(-1)), dim=-1)

    def gripper_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Only the classification head receives gradients during head training."""
        with torch.no_grad():
            features = self.net[:-1](self.encode_state(state))
        return nn.functional.linear(features, self.gripper_weight, self.gripper_bias)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        if not self.binary_gripper:
            return super().compute_loss(state, action_chunk)
        target = (action_chunk[:, :, 3] > self.gripper_endpoints.mean()).to(state.dtype)
        return nn.functional.binary_cross_entropy_with_logits(self.gripper_logits(state), target)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 4,
    d_model: int = 512,
    depth: int = 3,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
