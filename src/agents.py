from __future__ import annotations

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
    """Plays a uniformly random legal action."""
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal: List[Action]) -> Action:
        return self.rng.choice(legal)


class FlatNetAgent:
    """Wraps a flat PolicyNetwork checkpoint for evaluation."""

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
    return FlatNetAgent(net, device, deterministic=deterministic)
