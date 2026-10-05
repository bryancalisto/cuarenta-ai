
"""
Tests for history.py / HistoryTracker.

The important one is `test_pending_step_convention`: the network's state embedding
comes from the LAST LSTM step, so the last step of the inputs must be the observation
of the state being decided in. The old code fed only *completed* decisions, so the net
always decided from a 4-ply-old observation.

Run:  python test_history.py
"""

import random

import numpy as np
import torch

from env import CuarentaEnv, action_features, flat_obs_for
from history import (
    HistoryTracker, flatten_obs, stack_history,
    build_net_inputs, OBS_FLAT_DIM, ACT_FEAT_DIM,
)

N_ACTIONS = CuarentaEnv.MAX_ACTIONS


def _play(env, tracker, n_steps, rng):
    """Play n_steps of a game with random legal actions, recording into tracker."""
    done = False
    for _ in range(n_steps):
        if env.state.done:
            done = True
            break
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        a = rng.randrange(len(legal))
        tracker.record(p, obs, a)
        _, _, done, _, _ = env.step(a)
        if done:
            break
    return done


def test_flatten_obs():
    env = CuarentaEnv(seed=0)
    env.reset(seed=0)
    flat = flatten_obs(env.observe(0))
    assert flat.shape == (OBS_FLAT_DIM,), flat.shape
    assert flat.dtype == np.float32
    print("test_flatten_obs: OK")


def test_pending_step_convention():
    """
    inputs_for(player, obs_now) must end with obs_now and act_mask=False, and must
    have exactly history_length + 1 steps.
    """
    env = CuarentaEnv(seed=3)
    env.reset(seed=3)
    tracker = HistoryTracker(env)
    rng = random.Random(0)
    _play(env, tracker, 15, rng)

    p = env.state.current_player
    obs_now = env.observe(p)

    obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs_now, max_len=32)

    expected_T = tracker.history_length(p) + 1
    assert obs_seq.shape == (expected_T, OBS_FLAT_DIM), obs_seq.shape
    assert act_seq.shape == (expected_T, ACT_FEAT_DIM), act_seq.shape
    assert act_mask.shape == (expected_T,), act_mask.shape

    # last step = the CURRENT observation, and it carries no action yet
    assert act_mask[-1] == False, "pending step must have act_mask=False"
    assert np.allclose(act_seq[-1], 0.0), "pending step must have zero action features"
    assert np.array_equal(obs_seq[-1], flatten_obs(obs_now)), (
        "the last step must be the observation of the state being decided in"
    )
    # every earlier step carries a completed action
    assert act_mask[:-1].all(), "history steps must all have act_mask=True"
    print("test_pending_step_convention: OK")


def test_action_history_is_feature_based():
    """
    act_seq[t] must be the 84-dim feature vector of the action taken at step t
    (not a one-hot over slot indices, which is meaningless across states).
    """
    env = CuarentaEnv(seed=7)
    env.reset(seed=7)
    tracker = HistoryTracker(env)

    # hand-play two decisions for player 0 and remember the features we expect
    expected = []
    while len(expected) < 2 and not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        slot = 0
        if p == 0:
            expected.append(obs["action_features"][slot].copy())
        tracker.record(p, obs, slot)
        env.step(slot)

    p = 0
    obs_now = env.observe(p) if env.state.current_player == p else None
    if obs_now is None:
        # game moved on; just test whatever history exists for player 0
        obs_seq, act_seq, act_mask = tracker.inputs_for(0, env.observe(0), max_len=32)
    else:
        obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs_now, max_len=32)

    for t, exp in enumerate(expected):
        assert np.allclose(act_seq[t], exp), f"step {t} action features differ"
    # and it is certainly not a 64-dim one-hot
    assert act_seq.shape[-1] == ACT_FEAT_DIM != N_ACTIONS
    print("test_action_history_is_feature_based: OK")


def test_max_len_truncation_keeps_pending_last():
    env = CuarentaEnv(seed=11)
    env.reset(seed=11)
    tracker = HistoryTracker(env)
    rng = random.Random(1)
    _play(env, tracker, 30, rng)

    p = env.state.current_player
    obs_now = env.observe(p)
    obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs_now, max_len=8)

    assert obs_seq.shape[0] == 8
    assert act_mask[-1] == False
    assert np.array_equal(obs_seq[-1], flatten_obs(obs_now))
    print("test_max_len_truncation_keeps_pending_last: OK")


def test_stack_history_shapes():
    env = CuarentaEnv(seed=0)
    env.reset(seed=0)
    obs = flatten_obs(env.observe(0))
    obs_seq, act_seq, act_mask = stack_history([obs, obs], [np.zeros(ACT_FEAT_DIM), None])
    assert obs_seq.shape == (2, OBS_FLAT_DIM)
    assert act_seq.shape == (2, ACT_FEAT_DIM)
    assert act_mask.tolist() == [True, False]
    print("test_stack_history_shapes: OK")


def test_build_net_inputs_shapes():
    env = CuarentaEnv(seed=0)
    env.reset(seed=0)
    tracker = HistoryTracker(env)
    rng = random.Random(0)
    _play(env, tracker, 10, rng)

    p = env.state.current_player
    obs_now = env.observe(p)
    obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs_now, max_len=32)

    obs_t, act_t, am_t, feats_t, mask_t = build_net_inputs(
        obs_seq, act_seq, act_mask, obs_now["action_features"], obs_now["action_mask"]
    )
    assert obs_t.shape == (1, obs_seq.shape[0], OBS_FLAT_DIM)
    assert act_t.shape == (1, act_seq.shape[0], ACT_FEAT_DIM)
    assert am_t.shape == (1, act_seq.shape[0])
    assert feats_t.shape == (1, N_ACTIONS, 84)
    assert mask_t.shape == (1, N_ACTIONS)

    from network import CuarentaNet
    logits, value = CuarentaNet()(obs_t, act_t, am_t, feats_t, mask_t)
    assert logits.shape == (1, N_ACTIONS)
    assert value.shape == (1,)
    print("test_build_net_inputs_shapes: OK")


if __name__ == "__main__":
    test_flatten_obs()
    test_pending_step_convention()
    test_action_history_is_feature_based()
    test_max_len_truncation_keeps_pending_last()
    test_stack_history_shapes()
    test_build_net_inputs_shapes()
    print("\nAll history tests passed.")
