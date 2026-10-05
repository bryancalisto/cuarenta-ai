"""
Smoke tests for CuarentaNet: shapes, masking, gradients, and padding equivalence.
No training, no ISMCTS integration.

Run:  python test_network.py
"""
import numpy as np
import torch

from network import CuarentaNet
from history import OBS_FLAT_DIM, ACT_FEAT_DIM
from env import CuarentaEnv

N_ACTIONS = CuarentaEnv.MAX_ACTIONS
ACTION_FEAT_DIM = CuarentaEnv.ACTION_FEAT_DIM


def test_shapes():
    net = CuarentaNet()
    B, T = 2, 7
    obs = torch.randn(B, T, OBS_FLAT_DIM)
    act = torch.rand(B, T, ACT_FEAT_DIM)
    am = torch.ones(B, T, dtype=torch.bool)
    am[:, -1] = False                     # last step: pending, no action yet
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool)
    for b in range(B):
        mask[b, :3] = True                # 3 legal slots per batch item

    logits, value = net(obs, act, am, feats, mask)

    assert logits.shape == (B, N_ACTIONS), logits.shape
    assert value.shape == (B,), value.shape

    # Illegal slots are -1e9 (or very negative); legal slots are finite.
    for b in range(B):
        assert (logits[b, 3:] <= -1e8).all()
        assert torch.isfinite(logits[b, :3]).all()

    # Value in [-1, 1]
    assert (value >= -1).all() and (value <= 1).all()

    print("test_shapes: OK")


def test_masking_is_respected_by_softmax():
    """Softmax over masked logits should assign zero probability to illegal slots."""
    net = CuarentaNet()
    B, T = 1, 3
    obs = torch.randn(B, T, OBS_FLAT_DIM)
    act = torch.rand(B, T, ACT_FEAT_DIM)
    am = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, [0, 5, 7]] = True

    logits, _ = net(obs, act, am, feats, mask)
    probs = torch.softmax(logits[0], dim=-1)

    nonzero = (probs > 1e-6).nonzero(as_tuple=True)[0].tolist()
    assert set(nonzero) == {0, 5, 7}, nonzero
    assert abs(probs.sum().item() - 1.0) < 1e-5
    print("test_masking_is_respected_by_softmax: OK")


def test_gradients_flow():
    """A single backward pass should produce non-None grads on every param."""
    net = CuarentaNet()
    B, T = 4, 9
    obs = torch.randn(B, T, OBS_FLAT_DIM)
    act = torch.rand(B, T, ACT_FEAT_DIM)
    am = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, :5] = True
    lengths = torch.tensor([T, T - 1, 3, 1])

    logits, value = net(obs, act, am, feats, mask, lengths=lengths)

    policy_target = torch.zeros(B, N_ACTIONS)
    policy_target[:, 2] = 1.0
    value_target = torch.tensor([1.0, -1.0, 0.5, -0.5])

    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(policy_target * logp).sum(-1).mean()
    value_loss = ((value - value_target) ** 2).mean()
    loss = policy_loss + value_loss
    loss.backward()

    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"params with no grad: {missing}"

    for n, p in net.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"

    print(f"test_gradients_flow: OK  loss={loss.item():.4f}")


def test_lengths_mask_padding():
    """
    `lengths` must make the LSTM ignore left padding.

    Same real sequence, once alone and once behind 5 padding steps: the last-step
    embedding (hence logits and value) must be identical. If this fails, the net is
    reading zeros as "a real state where nothing exists", and short histories are
    being contaminated — which is what the dataset produces for every decision in
    the first few plies of a game.
    """
    torch.manual_seed(0)
    net = CuarentaNet().eval()

    T_real = 6
    obs = torch.randn(1, T_real, OBS_FLAT_DIM)
    act = torch.rand(1, T_real, ACT_FEAT_DIM)
    am = torch.ones(1, T_real, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(1, N_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool); mask[:, :4] = True

    with torch.no_grad():
        logits_a, value_a = net(obs, act, am, feats, mask,
                                lengths=torch.tensor([T_real]))

        P = 5
        obs_p = torch.cat([torch.zeros(1, P, OBS_FLAT_DIM), obs], dim=1)
        act_p = torch.cat([torch.zeros(1, P, ACT_FEAT_DIM), act], dim=1)
        am_p = torch.cat([torch.zeros(1, P, dtype=torch.bool), am], dim=1)
        logits_b, value_b = net(obs_p, act_p, am_p, feats, mask,
                                lengths=torch.tensor([T_real]))

    assert torch.allclose(logits_a, logits_b, atol=1e-5), (
        f"padding changed the policy: {(logits_a - logits_b).abs().max().item()}"
    )
    assert torch.allclose(value_a, value_b, atol=1e-5), (
        f"padding changed the value: {value_a.item()} vs {value_b.item()}"
    )
    print("test_lengths_mask_padding: OK")


def test_last_obs_drives_value():
    """Replacing the last observation must change the value (the LSTM must see it)."""
    torch.manual_seed(0)
    net = CuarentaNet().eval()
    B, T = 1, 8
    obs = torch.randn(B, T, OBS_FLAT_DIM)
    act = torch.rand(B, T, ACT_FEAT_DIM)
    am = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, :4] = True

    with torch.no_grad():
        _, v_a = net(obs, act, am, feats, mask)
        _, v_ctrl = net(obs, act, am, feats, mask)
        assert (v_a - v_ctrl).abs().item() == 0.0, "deterministic forward failed"

        obs2 = obs.clone(); obs2[0, -1, :] = 1.0
        _, v_b = net(obs2, act, am, feats, mask)

    diff = (v_a - v_b).abs().item()
    assert diff > 1e-6, "LSTM not sensitive to the last (pending) observation"
    print(f"test_last_obs_drives_value: OK (diff={diff:.3e})")


if __name__ == "__main__":
    test_shapes()
    test_masking_is_respected_by_softmax()
    test_gradients_flow()
    test_lengths_mask_padding()
    test_last_obs_drives_value()
    print("\nAll network tests passed.")
