from __future__ import annotations
from networks import ActorCriticNetwork 

import random
from typing import List

import numpy as np
import torch

from env import Action
from networks import PolicyNetwork


def flatten_obs(obs) -> np.ndarray:
    """Flatten an observation dict using the same sorted-key order as training."""
    keys = sorted(obs.keys())
    return np.concatenate([obs[k].reshape(-1) for k in keys]).astype(np.float32)


def encode_legal_action(action: Action, legal: List[Action]) -> int:
    """
    Encode an action into the flat 40 * 41 action space used by the imitation
    policy and the PPO warm-start model.
    """
    same_card = [a for a in legal if a.card == action.card]
    card_idx = action.card.index
    if action.capture is None:
        capture_idx = 0
    else:
        capture_idx = same_card.index(action) + 1
    return card_idx * 41 + capture_idx


class RandomAgent:
    """Plays a uniformly random legal action"""
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal: List[Action]) -> Action:
        return self.rng.choice(legal)


class HeuristicAgent:
    """
    Simple rule-based agent:
      1. If a caída is available -> take it.
      2. If a limpia is available -> take it.
      3. If any capture -> take the one capturing the most cards.
      4. Otherwise place a random card.
    """

    def act(self, obs, legal: List[Action]) -> Action:
        captures = [a for a in legal if a.capture]
        if captures:
            lp_vec = obs["last_placed"]
            lp_idx = int(lp_vec.argmax()) if lp_vec.sum() > 0 else -1

            if lp_idx != -1:
                for a in captures:
                    if len(a.capture) == 1 and next(iter(a.capture)).index == lp_idx:
                        return a  # caída

            # Prefer the largest capture (proxy for limpia / best value)
            best_cap = max(captures, key=lambda a: len(a.capture))
            return best_cap

        places = [a for a in legal if a.capture is None]
        return random.choice(places)


class PolicyNetAgent:
    """Wraps a PolicyNetwork checkpoint for evaluation."""

    def __init__(self, net: PolicyNetwork, device: torch.device, deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.net.eval()

    @torch.no_grad()
    def act(self, obs, legal: List[Action]) -> Action:
        obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.net(obs_t)
        legal_indices = [encode_legal_action(a, legal) for a in legal]
        legal_logits = logits[0, legal_indices]
        if self.deterministic:
            choice = int(torch.argmax(legal_logits).item())
        else:
            dist = torch.distributions.Categorical(logits=legal_logits)
            choice = int(dist.sample().item())
        return legal[choice]


def load_agent_from_checkpoint(
    path: str,
    device: torch.device,
    deterministic: bool = True,
):
    state = torch.load(path, map_location=device, weights_only=False)

    policy_state = state["policy_state"]
    obs_size = int(state.get("obs_size", policy_state["fc1.weight"].shape[1]))
    action_size = int(state.get("action_size", policy_state["fc3.weight"].shape[0]))
    net = PolicyNetwork(obs_size=obs_size, action_size=action_size).to(device)
    net.load_state_dict(policy_state)
    net.eval()
    return PolicyNetAgent(net, device, deterministic=deterministic)


class RecurrentNetAgent:
    """
    Wraps a ActorCriticNetwork checkpoint (feedforward or recurrent) for
    evaluation / play. Maintains its own persistent LSTM hidden state across
    turns within a single game, mirroring how the opponent net is used
    during training rollouts.
    """
    def __init__(self, net: ActorCriticNetwork, device: torch.device, deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.lstm_state = None
        self.net.eval()

    def reset(self):
        """Call this at the start of every new game/hand."""
        self.lstm_state = None

    @torch.no_grad()
    def act(self, obs, legal: List[Action]) -> Action:
        obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _, new_state = self.net(obs_t, self.lstm_state if self.net.use_lstm else None)
        if self.net.use_lstm:
            self.lstm_state = new_state

        legal_indices = [encode_legal_action(a, legal) for a in legal]
        legal_logits = logits[0, legal_indices]
        if self.deterministic:
            choice = int(torch.argmax(legal_logits).item())
        else:
            dist = torch.distributions.Categorical(logits=legal_logits)
            choice = int(dist.sample().item())
        return legal[choice]


def load_agent_from_checkpoint(
    path: str,
    device: torch.device,
    deterministic: bool = True,
) -> RecurrentNetAgent:
    state = torch.load(path, map_location=device, weights_only=False)
    obs_size = int(state["obs_size"])
    action_size = int(state["action_size"])
    use_lstm = bool(state.get("use_lstm", False))
    lstm_hidden = int(state.get("lstm_hidden", 128))

    net = ActorCriticNetwork(obs_size, action_size, use_lstm=use_lstm, lstm_hidden=lstm_hidden).to(device)

    if "model_state" in state:
        net.load_state_dict(state["model_state"])
    elif "policy_state" in state:
        # Legacy DAgger-style checkpoint: feedforward trunk only, no LSTM/value weights.
        net.policy.load_state_dict(state["policy_state"])
    else:
        raise KeyError(f"Unsupported checkpoint format: {path}")

    net.eval()
    return RecurrentNetAgent(net, device, deterministic=deterministic)