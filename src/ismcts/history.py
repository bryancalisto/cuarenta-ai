"""
History utilities for the Cuarenta RL pipeline.

Three responsibilities:

1. "flatten_obs(obs_dict)" -> 1-D float32 vector
   Collapses the env's dict observation into a single vector for the network.

2. "stack_history(obs_list, action_list)" -> padded numpy arrays
   Turns a chronological list of (obs, action) pairs into fixed-shape arrays
   suitable for the LSTM. Handles the "last obs has no action yet" case and
   optional sliding-window truncation.

3. "HistoryTracker"
   Maintains, per player, the sequence of observations that player has seen and
   the actions they took. This is what the network consumes during training and
   during ISMCTS.

Plus a small helper "build_net_inputs" that converts numpy arrays into the
torch tensors the network expects (adds a batch dimension).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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


# ----------------------------------------------------------------------------
# Sequence stacking
# ----------------------------------------------------------------------------

def stack_history(
    obs_list: List[Dict[str, np.ndarray]],
    action_list: List[int],
    *,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a per-player trajectory into padded tensors for the LSTM.

    Arguments
    ---------
    obs_list : list of T obs dicts, in chronological order. Each is the
               observation as seen by that player at that step.
    action_list : list of A action indices, A <= T. `action_list[t]` is the
               action the player took *after* seeing `obs_list[t]`. The last
               obs may be unpaired (the current decision point: obs seen,
               action not yet taken).
    max_len : if not None, keep only the most recent `max_len` steps (sliding
               window). Truncation is from the FRONT (older steps dropped).

    Returns
    -------
    obs_seq  : [T, OBS_FLAT_DIM]   float32
    act_seq  : [T, N_ACTIONS]      int64, one-hot of actions (all zeros where
                                   no action was taken)
    act_mask : [T]                 bool, True where an action exists
    mask_seq : [T, N_ACTIONS]      bool, action_mask (legality) at each step
    """
    from env import CuarentaEnv  # local import to avoid circular dependency
    N_ACTIONS = CuarentaEnv.MAX_ACTIONS

    T = len(obs_list)
    obs_seq  = np.zeros((T, OBS_FLAT_DIM), dtype=np.float32)
    act_seq  = np.zeros((T, N_ACTIONS), dtype=np.int64)
    act_mask = np.zeros((T,), dtype=bool)
    mask_seq = np.zeros((T, N_ACTIONS), dtype=bool)

    for t, obs in enumerate(obs_list):
        obs_seq[t]  = flatten_obs(obs)
        mask_seq[t] = obs["action_mask"].astype(bool)

    # Fill in the actions, up to min(len(obs_list), len(action_list)).
    for t, a in enumerate(action_list):
        if t >= T:
            break
        act_seq[t, a] = 1
        act_mask[t]   = True

    if max_len is not None:
        obs_seq  = obs_seq[-max_len:]
        act_seq  = act_seq[-max_len:]
        act_mask = act_mask[-max_len:]
        mask_seq = mask_seq[-max_len:]

    return obs_seq, act_seq, act_mask, mask_seq


# ----------------------------------------------------------------------------
# Per-player tracker
# ----------------------------------------------------------------------------

class HistoryTracker:
    """
    Maintains, for each player, the sequence of (obs, action) pairs they have
    personally observed.

    This is the correct thing to feed an LSTM: each player's history contains
    only information visible to that player.

    Usage:
        tracker = HistoryTracker(env)
        tracker.reset()

        # Inside your rollout loop, BEFORE calling env.step:
        p = env.state.current_player
        obs = env.observe(p)
        a = agent.act(obs)
        tracker.record(p, obs, a)
        env.step(a)

    Note on ordering: "record" stores the observation and the action that is
    *about to be played*. So "obs_list[t]" and "action_list[t]" are paired as
    "saw this, then did that".

    For the current decision point (obs seen but no action taken yet), the obs
    is recorded but no action is recorded. "sequence_for" then returns a
    trailing obs with "act_mask[t] = False".
    """

    def __init__(self, env):
        self.env = env
        self.max_actions = type(env).MAX_ACTIONS
        self._obs:    Dict[int, List[Dict[str, np.ndarray]]] = {p: [] for p in range(4)}
        self._action: Dict[int, List[int]] = {p: [] for p in range(4)}

    def reset(self) -> None:
        self._obs    = {p: [] for p in range(4)}
        self._action = {p: [] for p in range(4)}

    def record(self, player: int, obs: Dict[str, np.ndarray], action: int) -> None:
        self._obs[player].append(obs)
        self._action[player].append(int(action))

    def sequence_for(
        self,
        player: int,
        *,
        max_len: Optional[int] = 32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (obs_seq, act_seq, act_mask, mask_seq) for `player`."""
        return stack_history(
            self._obs[player], self._action[player], max_len=max_len
        )

    def pending_obs_for(self, player: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns the latest obs for "player" if it hasn't been paired with an
        action yet (useful when the network needs to act at the root without
        recording a fictitious action).
        """
        obs = self._obs[player]
        act = self._action[player]
        if len(obs) > len(act):
            return obs[-1]
        return None

    def history_length(self, player: int) -> int:
        return len(self._obs[player])


# ----------------------------------------------------------------------------
# Torch conversion helpers
# ----------------------------------------------------------------------------

def build_net_inputs(
    obs_seq: np.ndarray,       # [T, OBS_FLAT_DIM]
    act_seq: np.ndarray,       # [T, N_ACTIONS]
    act_mask: np.ndarray,      # [T]
    action_feats: np.ndarray,  # [N_ACTIONS, ACTION_FEAT_DIM]
    action_mask: np.ndarray,   # [N_ACTIONS]
    *,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add a batch dim and convert to torch tensors for the network.

    All inputs get a leading batch dim of 1. To batch multiple trajectories
    together you would stack along dim 0 here (requires padding to common T).
    """
    obs   = torch.as_tensor(obs_seq,      dtype=torch.float32, device=device).unsqueeze(0)
    act   = torch.as_tensor(act_seq,      dtype=torch.float32, device=device).unsqueeze(0)
    am    = torch.as_tensor(act_mask,     dtype=torch.bool,    device=device).unsqueeze(0)
    feats = torch.as_tensor(action_feats, dtype=torch.float32, device=device).unsqueeze(0)
    mask  = torch.as_tensor(action_mask,  dtype=torch.bool,    device=device).unsqueeze(0)
    return obs, act, am, feats, mask


def to_torch_sequence(
    obs_seq: np.ndarray,
    act_seq: np.ndarray,
    act_mask: np.ndarray,
    mask_seq: np.ndarray,
    *,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as build_net_inputs, but for cases where you only need the sequence
    tensors (no current-decision action features / mask). Useful for tests
    and for preprocessing offline data.
    """
    obs  = torch.as_tensor(obs_seq,  dtype=torch.float32, device=device)
    act  = torch.as_tensor(act_seq,  dtype=torch.float32, device=device)
    am   = torch.as_tensor(act_mask, dtype=torch.bool,    device=device)
    mask = torch.as_tensor(mask_seq, dtype=torch.bool,    device=device)
    return obs, act, am, mask


