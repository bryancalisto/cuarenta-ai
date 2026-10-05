"""
History utilities for the Cuarenta RL pipeline.

Three responsibilities:

1. "flatten_obs(obs_dict)" -> 1-D float32 vector
   Collapses the env's dict observation into a single vector for the network.

2. "stack_history(obs_list, act_feat_list)" -> arrays for the LSTM
   Turns a chronological list of (flat obs, action-features-taken) into arrays,
   with an optional sliding-window truncation. The caller (the Dataset) pads to a
   fixed T; `lengths` tells the LSTM which steps are real.

3. "HistoryTracker"
   Maintains, per player, the sequence of observations that player has seen and
   the actions they took. This is what the network consumes during training and
   during ISMCTS.


THE PENDING-STEP CONVENTION (this was a real bug — read this before using it)
----------------------------------------------------------------------------
The network reads its "state embedding" from the LAST LSTM step:

    s_emb = h_seq[:, -1, :]        (see network.CuarentaNet.forward)

so the last step of a training/inference sequence must be the observation of the
state we are deciding IN. There are two kinds of step:

    completed step: (obs at decision k,  action features taken at decision k)   act_mask=True
    pending   step: (obs at the decision we are making NOW,        no action)    act_mask=False

`record()` stores completed steps (they only exist once an action has been
played). `inputs_for(player, obs_now)` appends the pending step for the current
decision. The old code called `sequence_for(player)` — completed steps only — and
fed that straight to the network, so the network was always deciding from the
observation of that player's PREVIOUS turn (4 plies earlier!): it never saw the
current table, hand or score directly, and it could not even tell whether a
capture available now was a caída, since the current `last_placed` was missing
from its input.

The actor's own history is the only history available at the root. Inside the
search, opponents' histories obviously cannot be reconstructed from a sampled
determinization, so interior-node evaluations are history-free single pending
steps (see network_ismcts_1). That mismatch is why training applies history
dropout (dataset.CuarentaDataset) — the net must be good at both.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import ACTION_FEAT_DIM


# ----------------------------------------------------------------------------
# Observation layout
# ----------------------------------------------------------------------------
#
# These are the obs keys we concatenate into a single per-timestep vector.
# "action_features" and "action_mask" are deliberately excluded: they go on a
# separate path in the network (the per-action scoring branch).
#
# IMPORTANT: OBS_FLAT_DIM must match "network.OBS_FLAT_DIM". If updating any, update both there and here
#
_OBS_KEYS_BITVEC = [
    "my_hand",              # 40
    "table",                # 40
    "my_team_captures",     # 40
    "opp_team_captures",    # 40
    "last_placed",          # 40
]
_OBS_KEYS_SCALAR = [
    "my_team_score",        # 1
    "opp_team_score",       # 1
    "current_player",       # 1
    "deck_remaining",       # 1
]
_OBS_KEYS_VEC4 = [
    "plays_this_deal",      # 4
]

OBS_FLAT_DIM = (
    40 * len(_OBS_KEYS_BITVEC)
    + sum(1 for _ in _OBS_KEYS_SCALAR)
    + 4 * len(_OBS_KEYS_VEC4)
)   # = 40*5 + 4 + 4 = 208

# One step of the *action history* is the feature vector of the action that was
# taken (see env.CuarentaEnv._action_features), NOT the slot index.
#
# Why not a one-hot over slots like the old code did? Because a slot index only
# means something relative to that state's legal-action list: "slot 3" is a
# different card/capture in every state, so a slot one-hot tells the LSTM nothing
# about what actually happened. Action features (card one-hot, captured cards,
# is_capture/is_caida/is_limpia) are state independent and are exactly what the
# policy head consumes, so both branches of the network now speak the same
# language.
ACT_FEAT_DIM = ACTION_FEAT_DIM   # 84


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Collapse the env's dict observation into a single 1-D float32 vector.
    Excludes "action_features" and "action_mask" (handled separately).
    """
    parts: List[np.ndarray] = []
    for k in _OBS_KEYS_BITVEC:
        parts.append(obs[k].astype(np.float32).ravel())
    for k in _OBS_KEYS_SCALAR:
        parts.append(obs[k].astype(np.float32).ravel())
    for k in _OBS_KEYS_VEC4:
        parts.append(obs[k].astype(np.float32).ravel())
    return np.concatenate(parts, axis=0)


def action_features_of(obs: Dict[str, np.ndarray], action_slot: int) -> np.ndarray:
    """Feature vector of the action sitting in `action_slot` of this observation."""
    return np.asarray(obs["action_features"][action_slot], dtype=np.float32).copy()


# ----------------------------------------------------------------------------
# Sequence stacking
# ----------------------------------------------------------------------------

def stack_history(
    obs_flat_list: List[np.ndarray],
    act_feat_list: List[np.ndarray],
    *,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a per-player trajectory into arrays for the LSTM.

    Arguments
    ---------
    obs_flat_list : list of T flat observations [OBS_FLAT_DIM], chronological.
    act_feat_list : list of T action-feature vectors [ACT_FEAT_DIM]. `t` is the
                    action taken *after* seeing `obs_flat_list[t]`. For the
                    pending (current) decision the entry is all zeros.
    max_len       : if not None, keep only the most recent `max_len` steps
                    (sliding window, older steps dropped from the FRONT).

    Returns
    -------
    obs_seq  : [T, OBS_FLAT_DIM]  float32
    act_seq  : [T, ACT_FEAT_DIM]  float32 (zeros where no action was taken)
    act_mask : [T]                bool, True where an action exists
    """
    T = len(obs_flat_list)
    obs_seq = np.zeros((T, OBS_FLAT_DIM), dtype=np.float32)
    act_seq = np.zeros((T, ACT_FEAT_DIM), dtype=np.float32)
    act_mask = np.zeros((T,), dtype=bool)

    for t in range(T):
        obs_seq[t] = obs_flat_list[t]
        if t < len(act_feat_list) and act_feat_list[t] is not None:
            act_seq[t] = act_feat_list[t]
            act_mask[t] = True

    if max_len is not None:
        obs_seq = obs_seq[-max_len:]
        act_seq = act_seq[-max_len:]
        act_mask = act_mask[-max_len:]

    return obs_seq, act_seq, act_mask


# ----------------------------------------------------------------------------
# Per-player tracker
# ----------------------------------------------------------------------------

class HistoryTracker:
    """
    Maintains, for each player, the sequence of (observation, action taken) pairs
    that player personally observed.

    This is the correct thing to feed an LSTM: each player's history contains
    only information visible to that player.

    Usage (inside the game loop, BEFORE calling env.step):

        tracker = HistoryTracker(env)
        p = env.state.current_player
        obs = env.observe(p)                     # this is the CURRENT decision
        slot = agent.pick(obs)

        # Build net inputs: history so far + the current obs as a PENDING step.
        obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs, max_len=32)

        tracker.record(p, obs, slot)             # mark the decision as completed
        env.step(slot)

    `record` pairs the observation with the action that is *about to be played*:
    "saw this, then did that". `inputs_for` appends the observation of the
    decision being made right now with act_mask=False, because that action does
    not exist yet.
    """

    def __init__(self, env):
        self.env = env
        self._obs_flat: Dict[int, List[np.ndarray]] = {p: [] for p in range(4)}
        self._act_feat: Dict[int, List[np.ndarray]] = {p: [] for p in range(4)}

    def reset(self) -> None:
        self._obs_flat = {p: [] for p in range(4)}
        self._act_feat = {p: [] for p in range(4)}

    def record(self, player: int, obs: Dict[str, np.ndarray], action_slot: int) -> None:
        """
        Mark a decision as completed: remember what `player` saw and the features
        of the action it took. `action_slot` indexes into obs["action_features"].
        """
        self._obs_flat[player].append(flatten_obs(obs))
        self._act_feat[player].append(action_features_of(obs, action_slot))

    # (No alias for `record`: the previous API encouraged calling a
    # "sequence_for(player)" that silently omitted the current decision, which is the
    # bug described at the top of this file. Use inputs_for() when feeding the net.)

    def inputs_for(
        self,
        player: int,
        obs_now: Dict[str, np.ndarray],
        *,
        max_len: Optional[int] = 32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Net inputs for the decision `player` is making RIGHT NOW.

        Returns (obs_seq, act_seq, act_mask) of length T = history steps + 1, where
        the final step is `obs_now` with act_mask=False (the pending step). T is
        capped at `max_len` (most recent steps win).

        NOTE: `obs_now` must be an observation of the current state from
        `player`'s point of view, i.e. env.observe(player) (or, inside the search,
        an observation built from a sampled determinization).
        """
        obs_seq, act_seq, act_mask = stack_history(
            self._obs_flat[player], self._act_feat[player], max_len=None
        )
        # Append the pending step. Its action features are zeros and act_mask is
        # False, which network.forward uses to zero out the (biased) projection of
        # the zero vector — otherwise a length-1 linear layer would inject its bias
        # and look like "some action was taken".
        obs_pending = np.asarray(flatten_obs(obs_now), dtype=np.float32)[None, :]
        obs_seq = np.concatenate([obs_seq, obs_pending], axis=0)
        act_seq = np.concatenate(
            [act_seq, np.zeros((1, ACT_FEAT_DIM), dtype=np.float32)], axis=0
        )
        act_mask = np.concatenate([act_mask, np.zeros((1,), dtype=bool)], axis=0)

        if max_len is not None:
            obs_seq = obs_seq[-max_len:]
            act_seq = act_seq[-max_len:]
            act_mask = act_mask[-max_len:]
        return obs_seq, act_seq, act_mask

    def history_length(self, player: int) -> int:
        return len(self._obs_flat[player])


# ----------------------------------------------------------------------------
# Torch conversion helpers
# ----------------------------------------------------------------------------

def build_net_inputs(
    obs_seq: np.ndarray,       # [T, OBS_FLAT_DIM]
    act_seq: np.ndarray,       # [T, ACT_FEAT_DIM]
    act_mask: np.ndarray,      # [T]
    action_feats: np.ndarray,  # [MAX_ACTIONS, ACTION_FEAT_DIM]
    action_mask: np.ndarray,   # [MAX_ACTIONS]
    *,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add a batch dim and convert to torch tensors for the network (batch of 1).
    `lengths` is implied to be the full T.
    """
    obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
    act = torch.as_tensor(act_seq, dtype=torch.float32, device=device).unsqueeze(0)
    am = torch.as_tensor(act_mask, dtype=torch.bool, device=device).unsqueeze(0)
    feats = torch.as_tensor(action_feats, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    return obs, act, am, feats, mask
