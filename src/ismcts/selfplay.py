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

GAME RESULT (returns and rewards)
---------------------------------
`play_one_game` also returns a `GameResult`: the final scores, the number of plies,
and — per team — the episode RETURN (the sum of the env's step rewards) together
with its components. The env pays:

    +2 for a caída, +2 for a limpia, +50 for winning the game

so for each team `returns == 2*caidas + 2*limpias + 50*won` exactly (there are no
other reward terms). Those are the numbers worth watching over training: a net that
stops taking caídas shows it immediately even when its win rate looks flat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import CuarentaEnv, TEAMS
from history import HistoryTracker, ACT_FEAT_DIM
from network import CuarentaNet, N_ACTIONS
from network_ismcts_1 import NetworkISMCTS
from utils import PointsReasons, has_reason


@dataclass
class Decision:
    """One decision point (one search) in one game."""
    obs_seq: np.ndarray       # [T, OBS_FLAT_DIM]  (last step = pending current obs)
    act_seq: np.ndarray       # [T, ACT_FEAT_DIM]  (action features taken per step)
    act_mask: np.ndarray      # [T]
    feats: np.ndarray         # [MAX_ACTIONS, ACTION_FEAT_DIM]
    mask: np.ndarray          # [MAX_ACTIONS]
    visits: np.ndarray        # [MAX_ACTIONS]  (int) -> policy target
    team: int                 # 0 or 1
    player: int               # 0..3
    ply: int = 0              # move number in the game
    root_value: float = 0.0   # search root value (root team's POV)
    priors: Optional[np.ndarray] = None  # clean net prior over the legal slots
    outcome: float = 0.0      # filled in after the game ends
    # True  = self-play data: the policy target comes from the search, and the dataset
    #         may soften it (see dataset.target_temperature).
    # False = imitation data from the heuristic teacher: its one-hot target must stay
    #         one-hot, and the trainer can give it a different policy weight. We mix
    #         these in mainly to keep the VALUE head calibrated on non-self-play games.
    selfplay: bool = True


@dataclass
class GameResult:
    """
    Summary of one finished game, for logging/analysis. Everything is per TEAM
    (index 0/1), not per seat.

    `returns` is the episode RETURN: the sum of the env's step rewards for that
    team's players over the whole game. The env pays +2 per caída, +2 per limpia and
    +50 for winning, so `returns == 2*caidas + 2*limpias + 50*won` exactly.
    """
    winner: int                       # winning team (0 or 1)
    scores: Tuple[int, int]           # final points per team (the game's 0..40 scale)
    plies: int                        # number of moves played
    returns: Tuple[float, float]      # summed step rewards
    caidas: Tuple[int, int]           # count of caída bonuses earned
    limpias: Tuple[int, int]          # count of limpia bonuses earned
    mean_root_value: Tuple[float, float] = (0.0, 0.0)   # search's own estimate

    def as_metrics(self, prefix: str = "game") -> Dict[str, float]:
        """Flatten into wandb-friendly scalars (team 0 / team 1 / both)."""
        won0 = 1.0 if self.winner == 0 else 0.0
        return {
            f"{prefix}/plies": float(self.plies),
            f"{prefix}/score_team0": float(self.scores[0]),
            f"{prefix}/score_team1": float(self.scores[1]),
            f"{prefix}/score_diff": float(self.scores[0] - self.scores[1]),
            f"{prefix}/return_team0": float(self.returns[0]),
            f"{prefix}/return_team1": float(self.returns[1]),
            f"{prefix}/return_mean": float(np.mean(self.returns)),
            f"{prefix}/caidas_team0": float(self.caidas[0]),
            f"{prefix}/caidas_team1": float(self.caidas[1]),
            f"{prefix}/limpias_team0": float(self.limpias[0]),
            f"{prefix}/limpias_team1": float(self.limpias[1]),
            f"{prefix}/win_bonus_team0": 50.0 * won0,
            f"{prefix}/win_bonus_team1": 50.0 * (1.0 - won0),
            f"{prefix}/mean_root_value_team0": float(self.mean_root_value[0]),
            f"{prefix}/mean_root_value_team1": float(self.mean_root_value[1]),
        }


def play_one_game(
    net: CuarentaNet,
    *,
    n_iter: int = 200,
    max_history_len: int = 32,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    temperature_cutoff: int = 24,
    # `dirichlet_eps` is smaller than the AlphaZero default of 0.25 on purpose. The
    # noise is mixed into the root priors BEFORE the search, so it ends up inside the
    # visit distribution we train on. With a small sim budget and a value head that is
    # only slightly better than the policy, a large eps makes a big share of the
    # training target pure noise (the net can only memorise it). 0.1 still breaks
    # ties/mode collapse without dominating a 400-sim search.
    dirichlet_eps: float = 0.10,
    leaf_value_mode: str = "net",
    max_rollout_len: int = 512,
    device: str = "cpu",
) -> Tuple[List[Decision], GameResult]:
    """
    Play a full self-play game. Returns (decisions, result): one Decision per search
    performed, plus a GameResult with the scores, plies and per-team returns.
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

    # --- reward bookkeeping (per team) ---
    # The env's step reward is from the acting player's TEAM point of view, and it
    # only ever pays caída/limpia bonuses and the final win bonus, so these counters
    # fully explain the episode return.
    returns = [0.0, 0.0]
    caidas = [0, 0]
    limpias = [0, 0]

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
            # Clean prior over the legal slots, for diagnostics (target vs prior
            # entropy, how far the search moved from the policy).
            priors=np.asarray(searcher.last_root_priors, dtype=np.float32),
            selfplay=True,
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
        _, reward, done, _, info = env.step(a)

        # Accumulate the reward for the team that just acted, and its components.
        t = TEAMS[p]
        returns[t] += float(reward)
        reasons = info.get("reasons") or []
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[t] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[t] += 1

        ply += 1

    # --- Backfill outcomes ---
    winner_team = env.state.winner
    for d in decisions:
        d.outcome = 1.0 if d.team == winner_team else -1.0

    # Mean of the search's own value estimate, per team — a sanity check against the
    # actual result (this is what the value head predicted DURING play).
    mr = {}
    for team in (0, 1):
        vals = [d.root_value for d in decisions if d.team == team]
        mr[team] = float(np.mean(vals)) if vals else 0.0

    result = GameResult(
        winner=int(winner_team),
        scores=(int(env.state.scores[0]), int(env.state.scores[1])),
        plies=ply,
        returns=(returns[0], returns[1]),
        caidas=(caidas[0], caidas[1]),
        limpias=(limpias[0], limpias[1]),
        mean_root_value=(mr[0], mr[1]),
    )
    return decisions, result
