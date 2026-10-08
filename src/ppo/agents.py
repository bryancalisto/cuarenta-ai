"""
Opponents and evaluation/play agents.

Three kinds of thing live here:

  * game-playing bots used as opponents and as evaluation yardsticks
    (`RandomAgent`, `HeuristicAgent`),
  * thin wrappers that turn a checkpoint into an agent usable in the env loop
    (`PolicyNetAgent` for legacy DAgger MLPs, `RecurrentNetAgent` for PPO
    actor-critics), and
  * `load_agent_from_checkpoint`, which picks the right wrapper for a file.

THE HEURISTIC OPPONENT WAS BROKEN AND IS FIXED HERE
---------------------------------------------------
The old version decided captures from the observation vectors instead of the game
rules: it looked for `last_placed` inside the capture set (caída) and otherwise took
"the capture with the most cards" as a PROXY for a limpia. So it never actually
played for a limpia. It also could not be seeded (it used the global `random`), so
evaluations were not repeatable.

It now scores every legal action with the same `score_play` the env uses, which is
the only way to read a position the way the game does:

    1. caída (2 points)  -> take it
    2. limpia (2 points) -> take it
    3. any capture       -> the one taking the most cards
    4. otherwise         -> place a card at random

Two consequences worth remembering: this opponent is *stronger* than the one earlier
runs were measured against, so PPO win rates are not comparable with old logs; and
`load_agent_from_checkpoint` used to be defined twice in this file (the second
definition silently shadowed the first), which is now a single implementation that
dispatches on the checkpoint contents.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from captures import score_play
from env import Action, MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS, TEAMS
from networks import (ActorCriticNetwork, LSTMState, PolicyNetwork,
                      load_actor_critic, load_checkpoint_file)
from utils import (OBS_KEYS_LEGACY, PointsReasons, flatten_obs, has_reason)


# ---------------------------------------------------------------------------
# Bots
# ---------------------------------------------------------------------------

# Every bot implements the same tiny interface so they are interchangeable as
# opponents and evaluation yardsticks:
#     act(obs, legal, state=None, team=None) -> Action
#     reset() -> None                        (called between games)
#
class RandomAgent:
    """Plays a uniformly random legal action."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal: List[Action], state=None, team: Optional[int] = None) -> Action:
        return self.rng.choice(legal)

    def reset(self) -> None:
        """No per-game state; present so every agent shares one interface."""


class HeuristicAgent:
    """
    Rule-based agent, in priority order:

      1. Caída  — capture that takes the card placed last (2 points).
      2. Limpia — capture that clears the whole table (2 points).
      3. Largest capture by card count.
      4. Otherwise place a random card.

    Pass the env `state` (and optionally the acting `team`) to get the score-aware
    limpia rule: the limpia bonus only counts while a team is below 38 points.
    Without a state the agent falls back to observation-only reasoning, which detects
    both bonuses but cannot know the scores.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs, legal: List[Action], state=None, team: Optional[int] = None) -> Action:
        if state is not None and team is None:
            team = TEAMS[state.current_player]

        non_caida_allowed = True
        if state is not None and team is not None:
            non_caida_allowed = (
                state.scores[team] < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS
            )

        scored = []
        for a in legal:
            if state is not None:
                pts, reasons = score_play(
                    played=a.card, capture_set=a.capture, table=state.table,
                    last_placed=state.last_placed, non_caida_allowed=non_caida_allowed,
                )
            else:
                pts, reasons = self._score_from_obs(obs, a)
            scored.append((a, pts, reasons))

        caidas = [a for a, _, r in scored if has_reason(r, PointsReasons.CAIDA)]
        if caidas:
            return self.rng.choice(caidas)

        limpias = [a for a, _, r in scored if has_reason(r, PointsReasons.LIMPIA)]
        if limpias:
            return self.rng.choice(limpias)

        captures = [(a, len(a.capture)) for a, _, _ in scored if a.capture is not None]
        if captures:
            best_n = max(n for _, n in captures)
            return self.rng.choice([a for a, n in captures if n == best_n])

        places = [a for a, _, _ in scored if a.capture is None]
        if places:
            return self.rng.choice(places)

        return self.rng.choice(legal)

    @staticmethod
    def _score_from_obs(obs, action: Action):
        """
        Fallback reasoning with no state: a caída takes exactly the last-placed card,
        a limpia takes every card on the table. Scores are unknown here, so the
        38-point limpia restriction cannot be applied.
        """
        if action.capture is None:
            return 0, []
        last_placed = obs["last_placed"]
        lp_idx = int(np.argmax(last_placed)) if float(np.sum(last_placed)) > 0 else -1
        if lp_idx != -1 and len(action.capture) == 1 and next(iter(action.capture)).index == lp_idx:
            return 2, [PointsReasons.CAIDA]
        if float(len(action.capture)) == float(np.sum(obs["table"])):
            return 2, [PointsReasons.LIMPIA]
        return 0, []


# ---------------------------------------------------------------------------
# Checkpoint-backed agents
# ---------------------------------------------------------------------------

def encode_legal_action(action: Action, legal: List[Action]) -> int:
    """
    Encode an action into the flat 40*41 action-id space.

    `capture_option_index` is positional inside the per-card capture list, so `legal`
    must be the env's own list for the current state (see `env.ActionEncoder`).
    """
    same_card = [a for a in legal if a.card == action.card]
    card_idx = action.card.index
    if action.capture is None:
        capture_idx = 0
    else:
        capture_idx = same_card.index(action) + 1
    return card_idx * 41 + capture_idx


class PolicyNetAgent:
    """Wraps a legacy `PolicyNetwork` (DAgger checkpoint) for play/evaluation."""

    def __init__(self, net: PolicyNetwork, device: torch.device, deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.net.eval()

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs, legal: List[Action]) -> Action:
        # The legacy DAgger/ONNX networks were trained on the 244-dim alphabetical
        # layout, so they must keep consuming exactly that.
        obs_t = torch.tensor(flatten_obs(obs, OBS_KEYS_LEGACY), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        logits = self.net(obs_t)
        legal_indices = [encode_legal_action(a, legal) for a in legal]
        legal_logits = logits[0, legal_indices]
        if self.deterministic:
            choice = int(torch.argmax(legal_logits).item())
        else:
            dist = torch.distributions.Categorical(logits=legal_logits)
            choice = int(dist.sample().item())
        return legal[choice]


class PPOPolicyAgent:
    """
    Wraps a PPO `ActorCriticNetwork` (observation embedding + per-action features) for
    interactive play and evaluation.

    Interface, shared with the bots:
        act(obs, legal, state=None, team=None) -> Action   the agent's own turn
        advance(obs, legal, state=None)        -> None     someone else's turn
        reset()                                -> None     new game

    `advance` exists because the network is recurrent and is advanced on EVERY turn of
    a match during training (see src/ppo/trainer.py). A play loop must call it for the
    seats the agent does not control, otherwise its LSTM sees a different history than
    it was trained on. For a feedforward network `advance` is a no-op.

    `state` is required by `act`: the candidate-action features (caída/limpia flags)
    are computed from the game state, not from the observation alone.
    """

    def __init__(self, net: ActorCriticNetwork, device: torch.device,
                 deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.lstm_state = None
        self.net.eval()

    def reset(self) -> None:
        self.lstm_state = None

    def _forward(self, obs, legal, state):
        from env import action_features_matrix
        from utils import flatten_obs
        obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        feats_np, mask_np = action_features_matrix(legal, state)
        feats_t = torch.as_tensor(feats_np, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_np, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _, new_state = self.net(obs_t, feats_t, mask_t, self.lstm_state)
            if self.net.use_lstm:
                self.lstm_state = new_state
        return logits, mask_t

    @torch.no_grad()
    def act(self, obs, legal: List[Action], state=None, team: Optional[int] = None) -> Action:
        if state is None:
            raise ValueError(
                "PPOPolicyAgent.act needs the env state: the action features encode "
                "caída/limpia bonuses, which depend on the scores and the last placed "
                "card. Call agent.act(obs, legal, state=env.state)."
            )
        logits, mask_t = self._forward(obs, legal, state)
        if self.deterministic:
            slot = int(torch.argmax(logits[0][mask_t[0].nonzero(as_tuple=True)[0]]).item())
            slot = int(mask_t[0].nonzero(as_tuple=True)[0][slot].item())
        else:
            legal_logits = logits[0][mask_t[0]]
            slot = int(torch.distributions.Categorical(logits=legal_logits).sample().item())
        return legal[slot]

    @torch.no_grad()
    def advance(self, obs, legal: List[Action], state=None) -> None:
        """Advance the recurrent state on a turn the agent does not act on."""
        if not self.net.use_lstm or state is None:
            return
        self._forward(obs, legal, state)


def load_agent_from_checkpoint(
    path: str,
    device: torch.device,
    deterministic: bool = True,
    use_lstm: Optional[bool] = None,
):
    """
    Load any policy checkpoint and return an agent ready for `act(obs, legal)`.

    Dispatches on the checkpoint contents:
      * `model_state` + `action_feat_dim` -> PPO actor-critic -> `PPOPolicyAgent`
      * `model_state` (old flat head)     -> rejected with an explanation
      * `policy_state`                    -> legacy DAgger MLP -> `PolicyNetAgent`
    """
    state = load_checkpoint_file(path, device)

    if "model_state" in state and "action_feat_dim" in state:
        # Current PPO checkpoint: observation embedding + per-action features.
        net = load_actor_critic(path, device, use_lstm=use_lstm)
        return PPOPolicyAgent(net, device, deterministic=deterministic)

    if "model_state" in state:
        net = load_actor_critic(path, device, use_lstm=use_lstm)
        return RecurrentNetAgent(net, device, deterministic=deterministic)

    if "policy_state" in state:
        policy_state = state["policy_state"]
        obs_size = int(state.get("obs_size", policy_state["fc1.weight"].shape[1]))
        action_size = int(state.get("action_size", policy_state["fc3.weight"].shape[0]))
        net = PolicyNetwork(obs_size=obs_size, action_size=action_size).to(device)
        net.load_state_dict(policy_state)
        net.eval()
        return PolicyNetAgent(net, device, deterministic=deterministic)

    raise KeyError(f"Unsupported checkpoint format: {path}")
