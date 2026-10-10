"""Check legacy predictions, frozen XYZ outputs, binary units and strict loading."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from hw3.eval_utils import load_checkpoint
from hw3.model import MultiTaskPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incumbent', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True,
                   help='Predictions captured before modifying hw3/model.py')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    reference = np.load(args.reference)
    state = torch.from_numpy(reference['state'])
    expected = torch.from_numpy(reference['prediction'])
    base, norm, _, _, _ = load_checkpoint(args.incumbent, torch.device('cpu'))
    with torch.no_grad():
        old = base.sample_actions(state)
    assert torch.equal(old, expected), 'Legacy predictions changed'
    checks = dict(reference_states=len(state), legacy_predictions_bitwise_unchanged=True, models={})
    for name, path in [('control', args.control), ('candidate', args.candidate)]:
        model, model_norm, _, _, _ = load_checkpoint(path, torch.device('cpu'))
        assert model.binary_gripper
        assert not any(p.requires_grad for p in model.net.parameters())
        assert model.gripper_weight.requires_grad and model.gripper_bias.requires_grad
        for key in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            assert np.array_equal(getattr(norm, key), getattr(model_norm, key)), key
        for key, value in base.net.state_dict().items():
            assert torch.equal(value, model.net.state_dict()[key]), key
        for key in ('goal_features', 'input_mean', 'input_std'):
            assert torch.equal(getattr(base, key), getattr(model, key)), key
        with torch.no_grad():
            prediction = model.sample_actions(state)
        assert torch.equal(prediction[:, :, :3], old[:, :, :3]), 'XYZ predictions changed'
        grip = np.unique(norm.denormalize_action(prediction.numpy())[:, :, 3])
        assert len(grip) == 2 and np.allclose(grip, [-.174, 1.], atol=1e-7, rtol=0), grip
        clone = copy.deepcopy(base)
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(clone.sample_actions(state), prediction), 'Round trip changed actions'
        clone.compute_loss(state[:32], old[:32]).backward()
        assert all(p.grad is None for p in clone.net.parameters()), 'Training reaches original XYZ network'
        assert clone.gripper_weight.grad is not None and clone.gripper_weight.grad.abs().sum() > 0
        # Reusing an instance for an old checkpoint must also disable binary mode.
        original = torch.load(args.incumbent, map_location='cpu', weights_only=False)
        clone.load_state_dict(original['model_state_dict'], strict=True)
        with torch.no_grad():
            assert torch.equal(clone.sample_actions(state), expected), 'Legacy reload kept binary state'
        broken = dict(model.state_dict())
        del broken['gripper_endpoints']
        try:
            clone.load_state_dict(broken, strict=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Partial binary checkpoint was silently accepted')
        checks['models'][name] = dict(xyz_weights_bitwise_unchanged=True,
            xyz_predictions_bitwise_unchanged=True, normalization_unchanged=True,
            physical_gripper_commands=grip.tolist(), strict_round_trip=True,
            legacy_reload=True, partial_checkpoint_rejected=True,
            gradient_restricted_to_classifier=True)

    # Historical flat checkpoints contain only net.* keys.
    flat_state = {k:v for k,v in base.state_dict().items() if k.startswith('net.')}
    flat = copy.deepcopy(base)
    flat.load_state_dict(flat_state, strict=True)
    with torch.no_grad():
        assert torch.equal(flat.sample_actions(state), base.net(state).reshape(-1, base.chunk_size, base.action_dim))
    checks['flat_legacy_supported'] = True
    # Adding a disabled classifier must not change randomness for legacy training.
    torch.manual_seed(314)
    from hw3.model import ObstaclePolicy
    ObstaclePolicy(24, 4)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(314)
    MultiTaskPolicy(24, 4)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    checks['legacy_initialization_rng_unchanged'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
