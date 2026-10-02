"""
Smoke tests for CuarentaNet. No training, no ISMCTS integration.
Run:  python test_network.py
"""
import numpy as np
import torch

from network import CuarentaNet, OBS_FLAT_DIM
from history import build_net_inputs
from env import CuarentaEnv

N_ACTIONS = CuarentaEnv.MAX_ACTIONS
ACTION_FEAT_DIM = CuarentaEnv.ACTION_FEAT_DIM

def test_shapes():
    net = CuarentaNet()
    B, T = 2, 7
    obs   = torch.randn(B, T, OBS_FLAT_DIM)
    act   = torch.zeros(B, T, N_ACTIONS)
    # pretend an action was taken at every step except the last
    for b in range(B):
        for t in range(T - 1):
            act[b, t, np.random.randint(N_ACTIONS)] = 1
    am    = torch.ones(B, T, dtype=torch.bool)
    am[:, -1] = False                     # last step: no action yet
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask  = torch.zeros(B, N_ACTIONS, dtype=torch.bool)
    # make 3 slots legal per batch item
    for b in range(B):
        mask[b, :3] = True

    logits, value, (h, c) = net(obs, act, am, feats, mask)

    assert logits.shape == (B, N_ACTIONS), logits.shape
    assert value.shape == (B,), value.shape
    assert h.shape == (net.lstm.num_layers, B, net.hidden), h.shape
    assert c.shape == h.shape

    # Illegal slots should be -1e9 (or very negative)
    for b in range(B):
        assert (logits[b, 3:] <= -1e8).all()
        # Legal slots should be finite
        assert torch.isfinite(logits[b, :3]).all()

    # Value in [-1, 1]
    assert (value >= -1).all() and (value <= 1).all()

    print("test_shapes: OK")


def test_masking_is_respected_by_softmax():
    """Softmax over masked logits should assign zero probability to illegal slots."""
    net = CuarentaNet()
    B, T = 1, 3
    obs   = torch.randn(B, T, OBS_FLAT_DIM)
    act   = torch.zeros(B, T, N_ACTIONS)
    am    = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask  = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, [0, 5, 7]] = True

    logits, _, _ = net(obs, act, am, feats, mask)
    probs = torch.softmax(logits[0], dim=-1)

    assert probs.shape == (N_ACTIONS,)
    # Only slots 0, 5, 7 should have nonzero prob
    nonzero = (probs > 1e-6).nonzero(as_tuple=True)[0].tolist()
    assert set(nonzero) == {0, 5, 7}, nonzero
    assert abs(probs.sum().item() - 1.0) < 1e-5
    print("test_masking_is_respected_by_softmax: OK")


def test_gradients_flow():
    """A single backward pass should produce non-None grads on every param."""
    net = CuarentaNet()
    B, T = 4, 9
    obs   = torch.randn(B, T, OBS_FLAT_DIM, requires_grad=False)
    act   = torch.zeros(B, T, N_ACTIONS)
    for b in range(B):
        for t in range(T - 1):
            act[b, t, np.random.randint(N_ACTIONS)] = 1
    am    = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask  = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, :5] = True

    logits, value, _ = net(obs, act, am, feats, mask)

    # Fake targets for both heads
    policy_target = torch.zeros(B, N_ACTIONS)
    policy_target[:, 2] = 1.0              # pretend slot 2 was the "correct" action
    value_target = torch.tensor([1.0, -1.0, 0.5, -0.5])

    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(policy_target * logp).sum(-1).mean()
    value_loss  = ((value - value_target) ** 2).mean()
    loss = policy_loss + value_loss
    loss.backward()

    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"params with no grad: {missing}"

    # Sanity: grads are finite
    for n, p in net.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"

    print(f"test_gradients_flow: OK  loss={loss.item():.4f}")


def test_build_net_inputs_shapes():
    """Ensure history.build_net_inputs produces the shapes the net wants."""
    from env import CuarentaEnv
    from history import HistoryTracker

    env = CuarentaEnv(seed=0)
    env.reset(seed=0)
    tracker = HistoryTracker(env)

    import random
    rng = random.Random(0)
    for _ in range(10):
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        a = rng.randrange(len(legal))
        tracker.record(p, obs, a)
        _, _, done, _, _ = env.step(a)
        if done:
            break

    p = env.state.current_player
    obs_seq, act_seq, act_mask, mask_seq = tracker.sequence_for(p, max_len=32)

    # Current decision inputs
    obs_now = env.observe(p)
    feats_now = obs_now["action_features"]      # [32, 84]
    mask_now = obs_now["action_mask"]           # [32]

    obs_t, act_t, am_t, feats_t, mask_t = build_net_inputs(
        obs_seq, act_seq, act_mask, feats_now, mask_now
    )

    assert obs_t.shape   == (1, obs_seq.shape[0], OBS_FLAT_DIM), obs_t.shape
    assert act_t.shape   == (1, act_seq.shape[0], N_ACTIONS), act_t.shape
    assert am_t.shape    == (1, act_seq.shape[0]), am_t.shape
    assert feats_t.shape == (1, N_ACTIONS, ACTION_FEAT_DIM), feats_t.shape
    assert mask_t.shape  == (1, N_ACTIONS), mask_t.shape

    net = CuarentaNet()
    logits, value, _ = net(obs_t, act_t, am_t, feats_t, mask_t)
    assert logits.shape == (1, N_ACTIONS)
    assert value.shape == (1,)

    # Legal slots in the CURRENT decision should be finite
    legal_idx = mask_t[0].nonzero(as_tuple=True)[0].tolist()
    assert torch.isfinite(logits[0, legal_idx]).all()

    print("test_build_net_inputs_shapes: OK")


if __name__ == "__main__":
    test_shapes()
    test_masking_is_respected_by_softmax()
    test_gradients_flow()
    test_build_net_inputs_shapes()


    # Check with inputs
    net = CuarentaNet().eval()
    B, T = 1, 8
    obs   = torch.randn(B, T, OBS_FLAT_DIM)
    act   = torch.zeros(B, T, N_ACTIONS)
    am    = torch.ones(B, T, dtype=torch.bool); am[:, -1] = False
    feats = torch.rand(B, N_ACTIONS, ACTION_FEAT_DIM)
    mask  = torch.zeros(B, N_ACTIONS, dtype=torch.bool); mask[:, :4] = True

    with torch.no_grad():
        _, v_a, _ = net(obs, act, am, feats, mask)

        # Control: identical inputs -> identical output
        _, v_ctrl, _ = net(obs, act, am, feats, mask)
        assert (v_a - v_ctrl).abs().item() == 0.0, "deterministic forward failed"

        # Real perturbation: set the whole last obs to ones
        obs2 = obs.clone(); obs2[0, -1, :] = 1.0
        _, v_b, _ = net(obs2, act, am, feats, mask)

        # This should be clearly different. Use a generous threshold; if a full
        # obs replacement doesn't change the value, the LSTM isn't seeing it.
        diff = (v_a - v_b).abs().item()
        print(f"value diff: {diff:.6e}")
        assert diff > 1e-6, "LSTM not sensitive to last obs"

    
    print("\nAll network tests passed.")