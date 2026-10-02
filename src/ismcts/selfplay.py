"""
Self-play data generation for the Cuarenta net.

Each game is played with all four seats controlled by the same net (parameter
sharing, as in AlphaZero). At each decision we record:

  - obs_seq:    the root player's observation history  [T, OBS_FLAT_DIM]
  - act_seq:    one-hot of actions taken in that history  [T, N_ACTIONS]
  - act_mask:   whether an action was taken at each step  [T]
  - feats:      current decision's per-slot action features  [N_ACTIONS, ACTION_FEAT_DIM]
  - mask:       current decision's legal-action mask  [N_ACTIONS]
  - visits:     visit counts from the search over slots  [N_ACTIONS]
  - team:       the root player's team (0 or 1) for outcome assignment
  - player:     the root player's seat (0..3) — useful for debugging

After the game ends, we set `outcome` on every recorded decision: +1 if the
recorded team won, -1 otherwise. This is the value-head target from the
recorded player's team POV.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from env import CuarentaEnv, TEAMS
from history import HistoryTracker
from network import CuarentaNet, N_ACTIONS
from network_ismcts_1 import NetworkISMCTS


@dataclass
class Decision:
    """One decision point (one search) in one game."""
    obs_seq: np.ndarray       # [T, OBS_FLAT_DIM]
    act_seq: np.ndarray       # [T, N_ACTIONS]
    act_mask: np.ndarray      # [T]
    feats: np.ndarray         # [N_ACTIONS, ACTION_FEAT_DIM]
    mask: np.ndarray          # [N_ACTIONS]
    visits: np.ndarray        # [N_ACTIONS]  (int)
    team: int                 # 0 or 1
    player: int               # 0..3
    outcome: float = 0.0      # filled in after the game ends


def play_one_game(
    net: CuarentaNet,
    *,
    n_iter: int = 100,
    max_rollout_len: int = 200,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    max_history_len: int = 32,
    device: str = "cpu",
) -> List[Decision]:
    """
    Play a full self-play game, return one Decision per search performed.
    All 4 seats use `net`.

    `temperature`: 1.0 = sample from visits^(1/T) (exploratory),
                   0.0 = argmax visits (greedy).
                   Standard AlphaZero uses temp=1 for the first N moves, then 0.
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, net, tracker,
        n_iter=n_iter,
        max_rollout_len=max_rollout_len,
        max_history_len=max_history_len,
        rng=random.Random(seed),
        # Dirichlet exploration is on by default (0.25); keep it for self-play.
        dirichlet_eps=0.25,
        device=device,
    )

    decisions: List[Decision] = []

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)

        # Search + get visit counts
        visits_dict = searcher.search(root_player=p)   # {slot_idx: count}

        # Convert {slot: count} to a full [N_ACTIONS] array
        visits = np.zeros(N_ACTIONS, dtype=np.int64)
        for slot, count in visits_dict.items():
            visits[slot] = count

        # Snapshot inputs for training
        obs_seq, act_seq, act_mask, _ = tracker.sequence_for(p, max_len=max_history_len)
        if obs_seq.shape[0] == 0:
            # First decision of the game (no history): single-step input
            from history import flatten_obs
            obs_seq = flatten_obs(obs)[None, :]
            act_seq = np.zeros((1, N_ACTIONS), dtype=np.int64)
            act_mask = np.zeros((1,), dtype=bool)

        decisions.append(Decision(
            obs_seq=obs_seq.astype(np.float32),
            act_seq=act_seq.astype(np.float32),
            act_mask=act_mask.astype(bool),
            feats=obs["action_features"].astype(np.float32),
            mask=obs["action_mask"].astype(bool),
            visits=visits,
            team=TEAMS[p],
            player=p,
        ))

        # --- Choose action to actually play ---
        legal_slots = np.flatnonzero(obs["action_mask"])
        if sum(visits) == 0:
            # No signal — random fallback
            a = int(np.random.choice(legal_slots))
        elif temperature <= 1e-6:
            a = int(visits.argmax())
        else:
            # Sample from visits^(1/temp), restricted to legal actions.
            v = visits[legal_slots].astype(np.float64)
            v = v ** (1.0 / temperature)
            if v.sum() <= 0:
                probs = np.ones_like(v) / len(v)
            else:
                probs = v / v.sum()
            a = int(np.random.choice(legal_slots, p=probs))

        # Record BEFORE stepping so the tracker is consistent
        tracker.record(p, obs, a)
        _, _, done, _, info = env.step(a)

    # --- Backfill outcomes ---
    winner_team = env.state.winner
    for d in decisions:
        d.outcome = 1.0 if d.team == winner_team else -1.0

    return decisions