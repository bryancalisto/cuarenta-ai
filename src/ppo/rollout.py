"""
Rollout storage, advantage estimation and action selection.

WHAT IS STORED, AND WHY IT IS SHAPED THIS WAY
---------------------------------------------
PPO normally shuffles individual (state, action, reward) transitions. That is wrong
here when the policy is recurrent: the LSTM's hidden state depends on the whole
history, so a transition means nothing on its own. So each environment contributes a
`Rollout` holding:

  * `full_obs` / `full_dones` — EVERY turn of the match (all four seats), because the
    learner's hidden state is advanced on every turn and therefore has to be replayed
    on every turn, and
  * the per-learning-decision data (`actions`, `log_probs`, `rewards`, `values`,
    `dones`, plus the per-action FEATURES and legality mask of that decision) and
    `learn_positions`, the indices into `full_obs` where those decisions happened.

During the update, `replay_state_embeddings` re-encodes `full_obs` in order from
`hidden_start` (the hidden state saved at rollout time — the R2D2/"stored state"
approach) and resets at every episode boundary; only the `learn_positions` rows are
turned into policy logits and losses. Action features are stored only for the learning
decisions (~1 turn in 4), which keeps the buffered data small.

ACTIONS
-------
An action is a SLOT: an index into that position's `env.legal_actions()` list, at most
`MAX_ACTIONS` of them. The policy scores each slot from its 84-dim FEATURES
(`env.action_features_matrix`), exactly like the ISMCTS pipeline, so the network never
depends on what a slot index happens to mean in a given position.

REWARD ATTRIBUTION
------------------
The env pays rewards per MOVE from the acting team's point of view. A learning seat
only acts every fourth turn, so its rewards are gathered and flushed onto its most
recent decision when its turn comes around again (`pending_reward` in
`trainer._collect_rollout`). That way a caída by an opponent, or a capture by the
learner's team mate, both shape the decision that led to it.

GAE, AND WHY THE VALUE HEAD IS NORMALISED
-----------------------------------------
`compute_gae` is the standard formulation, with `next_done` zeroing the bootstrap
across episode boundaries. Cuarenta's returns span a wide and drifting range (the
engineerable score is 0..40, and the terminal reward is ±50), so value targets are
regression-normalised with a `RunningMeanStd`: GAE and bootstrapping happen in real
reward space and only the network's regression target is normalised.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import distributions as torch_dist

LSTMState = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Small tensor/state helpers
# ---------------------------------------------------------------------------

def masked_legal_logits(logits_1d: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Logits of the legal slots only (the mask marks the first `n` of `MAX_ACTIONS`)."""
    return logits_1d[action_mask]


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach and clone a state dict so it can be stored in the opponent pool."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def clone_lstm_state(state: Optional[LSTMState]) -> Optional[LSTMState]:
    """Detach + clone an (h, c) pair so it can be safely stashed for later reuse."""
    if state is None:
        return None
    h, c = state
    return (h.detach().cpu().clone(), c.detach().cpu().clone())


def lstm_state_to_device(state: Optional[LSTMState], device: torch.device) -> Optional[LSTMState]:
    if state is None:
        return None
    h, c = state
    return (h.to(device), c.to(device))


def shaped_reward(raw_reward: float, acting_team: int, learner_team: int,
                  symmetrize: bool = True) -> float:
    """
    Re-express one env reward (from the ACTING team's point of view) from the learning
    team's point of view.

    Cuarenta is zero-sum between the two teams, so an opponent's gain is the learner's
    loss:

        acting team == learner team  ->  +raw_reward
        otherwise, symmetrize        ->  -raw_reward
        otherwise (legacy behaviour) ->   0.0   (opponent rewards dropped entirely)

    The last case is what the old code did, and it is why the critic could not
    represent losing: the learner's episode return was never negative, and the ±50
    terminal bonus was discarded whenever the opponent won the game.
    """
    if acting_team == learner_team:
        return float(raw_reward)
    return -float(raw_reward) if symmetrize else 0.0


def select_legal_action(
    logits_1d: torch.Tensor,
    action_mask: torch.Tensor,
    deterministic: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Sample (or argmax) an action SLOT among the legal ones.

    The network produces one logit per slot; illegal slots are already -1e9 (masked in
    `ActorCriticNetwork.forward`), so a softmax over the legal prefix is the policy's
    distribution over the actions actually available in this position. Returns:

        slot:      index into `env.legal_actions()`
        log_prob:  log probability of the chosen slot
        entropy:   entropy of the legal-action distribution
    """
    legal_logits = masked_legal_logits(logits_1d, action_mask)
    dist = torch_dist.Categorical(logits=legal_logits)
    if deterministic:
        slot = int(torch.argmax(legal_logits).item())
    else:
        slot = int(dist.sample().item())
    choice = torch.tensor(slot, device=logits_1d.device)
    return slot, dist.log_prob(choice), dist.entropy()


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """One rollout window for a single environment (see the module docstring)."""

    # Full per-turn trajectory (every seat's turn) — needed so the LSTM can be
    # replayed through the whole match, not just the learner's own decisions.
    full_obs: List[np.ndarray] = field(default_factory=list)
    full_dones: List[bool] = field(default_factory=list)
    learn_positions: List[int] = field(default_factory=list)  # index into full_obs

    # Per-learning-transition data, aligned with learn_positions.
    actions: List[int] = field(default_factory=list)          # slot index
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # The candidate actions of that decision: [MAX_ACTIONS, ACTION_FEAT_DIM] features
    # and the [MAX_ACTIONS] legality mask. Stored only for learning decisions.
    action_feats: List[np.ndarray] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)

    hidden_start: Optional[LSTMState] = None

    def clear(self):
        self.full_obs.clear()
        self.full_dones.clear()
        self.learn_positions.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_feats.clear()
        self.action_masks.clear()
        self.hidden_start = None


def replay_state_embeddings(
    net,
    obs_seq: torch.Tensor,      # (T, obs_dim) for ONE environment, time-ordered
    dones: torch.Tensor,        # (T,) float32, 1.0 marks the last step of an episode
    init_state: Optional[LSTMState],
) -> torch.Tensor:
    """
    Re-encode a full time-ordered sequence and return the state embedding of EVERY
    timestep, as (T, F).

    The hidden state is carried across steps exactly as during rollout collection and
    reset to zero right after any step flagged `done`, mirroring the
    `lstm_states[env_idx] = None` reset on episode end. That symmetry is what makes
    recurrent PPO correct: the buffered data reproduces the hidden states the actions
    and values were actually generated with.

    The caller then builds logits only where it needs them, from the stored per-action
    features (see `ActorCriticNetwork.logits_from`).
    """
    T = obs_seq.shape[0]

    # Split into contiguous segments; a segment ends right after a `done`.
    segments: List[Tuple[int, int]] = []
    seg_start = 0
    for t in range(T):
        if dones[t].item() > 0.5:
            segments.append((seg_start, t + 1))
            seg_start = t + 1
    if seg_start < T:
        segments.append((seg_start, T))

    state = init_state
    chunks = []
    for s, e in segments:
        x = net.encode_sequence(obs_seq[s:e].unsqueeze(0), state)   # (1, seg_len, F)
        chunks.append(x.squeeze(0))
        state = None      # a `done` ended this segment; the next one starts fresh

    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation. Returns (advantages, returns), each shaped (T,).

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        return_t = A_t + V(s_t)

    `dones[t] == 1` marks the last transition of an episode: the bootstrap term and
    the GAE recursion both stop there, and the value target for that step is just the
    terminal reward.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae = delta + gamma * lam * (1.0 - next_done) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class RunningMeanStd:
    """
    Running mean/variance of a scalar stream (Welford's parallel algorithm).

    Tracks the scale of the value function's regression target (returns). Cuarenta
    rewards span a wide, drifting range (captures, caídas, end-of-deck bonuses, and
    the ±50 terminal), so the value head is trained in a normalised space that stays
    roughly unit-scale even as the achievable return changes with play quality.
    """

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps  # small non-zero prior so the first update is stable

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = int(x.size)

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: Dict[str, float]):
        self.mean = float(sd["mean"])
        self.var = float(sd["var"])
        self.count = float(sd["count"])


# ---------------------------------------------------------------------------
# Episode statistics (for logging returns / rewards, mirroring the ISMCTS pipeline)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """
    Summary of one finished game, per TEAM (index 0/1).

    `returns` is the episode RETURN: the sum of the env's step rewards for that team's
    players. The env's reward is dense -- +1 per card captured, +2 per caída, +2 per
    limpia, the cartón bonus at the end of a deal, and `env.WIN_BONUS` for winning the
    game -- so `returns == cards + 2*caidas + 2*limpias + carton + WIN_BONUS*won` (the
    cards term is not tracked here because a cartón payment starts a new round and resets
    the capture piles, which makes it unrecoverable from outside the env).
    """
    winner: int                     # winning team (0 or 1)
    scores: Tuple[int, int]         # final game points per team (0..40 scale)
    plies: int                      # moves played
    returns: Tuple[float, float]    # summed step rewards per team
    caidas: Tuple[int, int]         # caída bonuses earned per team
    limpias: Tuple[int, int]        # limpia bonuses earned per team
    carton: Tuple[int, int] = (0, 0)  # cartón points paid per team
    learner_team: int = 0
    # The learner's SHAPED return (its own team's rewards, opponents' negated): this
    # is the quantity PPO actually optimises, and it is what the value head estimates.
    # It differs from `returns[learner_team]` (raw rewards) when reward symmetrisation
    # is on, and from the old code where opponent rewards were simply dropped.
    shaped_return: float = 0.0
    opponent_is_latest: bool = False

    def as_metrics(self, prefix: str = "selfplay") -> Dict[str, float]:
        """Flatten into wandb-friendly scalars."""
        learn_won = 1.0 if self.winner == self.learner_team else 0.0
        lt = self.learner_team
        ot = 1 - lt
        return {
            f"{prefix}/win_rate": learn_won,
            f"{prefix}/plies": float(self.plies),
            f"{prefix}/score_for": float(self.scores[lt]),
            f"{prefix}/score_against": float(self.scores[ot]),
            f"{prefix}/score_diff": float(self.scores[lt] - self.scores[ot]),
            f"{prefix}/return_for": float(self.returns[lt]),
            f"{prefix}/return_against": float(self.returns[ot]),
            f"{prefix}/return_mean": float(np.mean(self.returns)),
            f"{prefix}/shaped_return": float(self.shaped_return),
            f"{prefix}/shaped_return_vs_raw": float(self.shaped_return - self.returns[lt]),
            f"{prefix}/caidas_for": float(self.caidas[lt]),
            f"{prefix}/caidas_against": float(self.caidas[ot]),
            f"{prefix}/limpias_for": float(self.limpias[lt]),
            f"{prefix}/limpias_against": float(self.limpias[ot]),
            f"{prefix}/carton_for": float(self.carton[lt]),
            f"{prefix}/carton_against": float(self.carton[ot]),
        }


def mean_episode_metrics(stats: List[EpisodeStats], prefix: str = "selfplay") -> Dict[str, float]:
    """Average a batch of `EpisodeStats` into flat scalars for logging."""
    if not stats:
        return {}
    per_game = [s.as_metrics(prefix) for s in stats]
    keys = per_game[0].keys()
    out = {k: float(np.mean([m[k] for m in per_game])) for k in keys}
    out[f"{prefix}/n_games"] = float(len(stats))
    return out


class MetricWindow:
    """
    Rolling window of the last N episodes, so every logged number is a real average
    over finished games rather than a single sample.
    """

    def __init__(self, size: int, prefix: str):
        self.size = size
        self.prefix = prefix
        self.stats: collections.deque = collections.deque(maxlen=size)

    def add(self, stat: EpisodeStats) -> None:
        self.stats.append(stat)

    def metrics(self) -> Dict[str, float]:
        return mean_episode_metrics(list(self.stats), self.prefix)

    def __len__(self) -> int:
        return len(self.stats)
