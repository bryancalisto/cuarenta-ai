"""
Self-play data generation for the Cuarenta net (expert iteration).

Each game is played with all four seats controlled by the same net (parameter
sharing, as in AlphaZero). At each decision we record:

  - obs_seq:    the acting player's observation history, ENDING WITH the current
                decision's observation as a "pending" step (act_mask False), so the
                network's state embedding is the state we are deciding in.
  - act_seq:    per-step feature vector (84) of the action actually taken at that
                step (zeros for the pending step). Feature-based, not slot-based:
                a slot index means a different action in every state.
  - act_mask:   whether an action was taken at each step.
  - feats:      current decision's per-slot action features  [MAX_ACTIONS, 84]
  - mask:       current decision's legal-action mask         [MAX_ACTIONS]
  - visits:     visit counts from the search over slots      [MAX_ACTIONS]
  - team:       the acting player's team (0/1) for outcome assignment
  - player:     the acting player's seat (0..3) — useful for debugging
  - ply:        global move number (for phase analysis / logging)
  - root_value: the search's root value estimate (root team's POV), for debugging

After the game ends we set `outcome` on every recorded decision: +1 if the recorded
team won, -1 otherwise. That is the value-head target.

MOVE SELECTION (temperature)
----------------------------
The action actually played is sampled from the visit distribution with a
temperature schedule: tau = 1.0 for the first `temperature_cutoff` decisions of the
game (exploration, and it keeps the opening diverse), then tau -> 0 (greedy, i.e.
play the most-visited action). AlphaZero does the same; keeping tau = 1 for the
whole game (what the old code did) means the net keeps playing weak sampled moves
deep into the game, and the recorded visit targets then describe a noisy policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from env import CuarentaEnv, TEAMS
from history import HistoryTracker, ACT_FEAT_DIM
from network import CuarentaNet, N_ACTIONS
from network_ismcts_1 import NetworkISMCTS


@dataclass
class Decision:
    """One decision point (one search) in one game."""
    obs_seq: np.ndarray       # [T, OBS_FLAT_DIM]  (last step = pending current obs)
    act_seq: np.ndarray       # [T, ACT_FEAT_DIM]  (action features taken per step)
    act_mask: np.ndarray      # [T]
    feats: np.ndarray         # [MAX_ACTIONS, ACTION_FEAT_DIM]
    mask: np.ndarray          # [MAX_ACTIONS]
    visits: np.ndarray        # [MAX_ACTIONS]  (int)
    team: int                 # 0 or 1
    player: int               # 0..3
    ply: int = 0              # move number in the game
    root_value: float = 0.0   # search root value (root team's POV)
    outcome: float = 0.0      # filled in after the game ends


def play_one_game(
    net: CuarentaNet,
    *,
    n_iter: int = 200,
    max_history_len: int = 32,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    temperature_cutoff: int = 24,
    dirichlet_eps: float = 0.25,
    leaf_value_mode: str = "net",
    max_rollout_len: int = 512,
    device: str = "cpu",
) -> List[Decision]:
    """
    Play a full self-play game, return one Decision per search performed.
    All 4 seats use `net`.

    `temperature` applies for the first `temperature_cutoff` decisions, after which
    the agent plays greedily (argmax over visits). Set `temperature_cutoff=0` for a
    fully greedy game (useful for evaluation).
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, net, tracker,
        n_iter=n_iter,
        max_history_len=max_history_len,
        leaf_value_mode=leaf_value_mode,
        max_rollout_len=max_rollout_len,
        rng=random.Random(seed),
        dirichlet_eps=dirichlet_eps,
        device=device,
    )

    decisions: List[Decision] = []
    ply = 0

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)

        # --- Search ---
        # The search reads the player's history + this pending observation itself
        # (see NetworkISMCTS._eval_root), so we do NOT need to pass obs_seq here.
        visits_dict = searcher.search(root_player=p)   # {slot_idx: count}

        visits = np.zeros(N_ACTIONS, dtype=np.int64)
        for slot, count in visits_dict.items():
            visits[slot] = count

        # --- Snapshot the net's inputs for training ---
        # inputs_for appends `obs` as the pending step, which is exactly what the
        # search used. Kept consistent on purpose: train == inference.
        obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs, max_len=max_history_len)

        decisions.append(Decision(
            obs_seq=obs_seq.astype(np.float32),
            act_seq=act_seq.astype(np.float32),
            act_mask=act_mask.astype(bool),
            feats=obs["action_features"].astype(np.float32),
            mask=obs["action_mask"].astype(bool),
            visits=visits,
            team=TEAMS[p],
            player=p,
            ply=ply,
            root_value=float(searcher.last_root_value),
        ))

        # --- Choose the action to actually play ---
        legal_slots = np.flatnonzero(obs["action_mask"])
        tau = temperature if ply < temperature_cutoff else 0.0

        if sum(visits) == 0:
            a = int(np.random.choice(legal_slots))            # no signal: random
        elif tau <= 1e-6:
            a = int(visits.argmax())                          # greedy
        else:
            # Sample from visits^(1/tau) restricted to legal actions.
            v = visits[legal_slots].astype(np.float64) ** (1.0 / tau)
            probs = v / v.sum() if v.sum() > 0 else np.ones(len(v)) / len(v)
            a = int(np.random.choice(legal_slots, p=probs))

        # Record BEFORE stepping so the tracker is consistent ("saw this, did that").
        tracker.record(p, obs, a)
        _, _, done, _, info = env.step(a)
        ply += 1

    # --- Backfill outcomes ---
    winner_team = env.state.winner
    for d in decisions:
        d.outcome = 1.0 if d.team == winner_team else -1.0

    return decisions
