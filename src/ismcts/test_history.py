
import numpy as np
from env import CuarentaEnv
from history import (
    HistoryTracker, flatten_obs, stack_history,
    build_net_inputs, OBS_FLAT_DIM,
)

env = CuarentaEnv(seed=0)
env.reset(seed=0)
tracker = HistoryTracker(env)

# Flatten check
obs = env.observe(0)
flat = flatten_obs(obs)
assert flat.shape == (OBS_FLAT_DIM,), (flat.shape, OBS_FLAT_DIM)

# Play a few steps, recording
import random
rng = random.Random(0)
for _ in range(15):
    p = env.state.current_player
    o = env.observe(p)
    legal = env.legal_actions()
    a = rng.randrange(len(legal))
    tracker.record(p, o, a)
    _, _, done, _, _ = env.step(a)
    if done:
        break

# Per-player sequence
for p in range(4):
    obs_seq, act_seq, act_mask, mask_seq = tracker.sequence_for(p, max_len=32)
    T = obs_seq.shape[0]
    assert obs_seq.shape  == (T, OBS_FLAT_DIM)
    assert act_seq.shape  == (T, 32)
    assert act_mask.shape == (T,)
    assert mask_seq.shape == (T, 32)

# Current-decision inputs
p = env.state.current_player
o = env.observe(p)

obs_seq, act_seq, act_mask, mask_seq = tracker.sequence_for(p, max_len=32)

obs_t, act_t, am_t, feats_t, mask_t = build_net_inputs(
    obs_seq,
    act_seq,
    act_mask,
    action_feats=o["action_features"],
    action_mask=o["action_mask"],
)
assert obs_t.shape   == (1, obs_t.shape[1], OBS_FLAT_DIM)
assert feats_t.shape == (1, 32, 84)
print("history.py OK")