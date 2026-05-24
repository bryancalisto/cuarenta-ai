from __future__ import annotations

import random
from typing import List

import numpy as np
import torch

from model import CuarentaNet, ActionMasks
from env import Action
from utils import obs_to_tensors
from imitation_learning.imitation_learning import PolicyNetwork


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


class HeuristicAgent:
    """
    Simple rule-based agent:
      1. If a caída is available → take it.
      2. If a limpia is available → take it.
      3. If any capture → take the one capturing the most cards.
      4. Otherwise place the lowest-value card.
    """

    def act(self, obs, legal: List[Action]) -> Action:
        captures = [a for a in legal if a.capture]

        if captures:
            # Check for caída
            if obs.get("last_placed") is not None:
                lp_vec = obs["last_placed"]
                lp_idx = int(lp_vec.argmax()) if lp_vec.sum() > 0 else -1
                for a in captures:
                    if a.capture:
                        cap_indices = {c.index for c in a.capture}
                        if lp_idx in cap_indices and a.card.index == lp_idx:
                            return a  # caída

            # Prefer limpia (capture all table cards)
            best_cap = max(captures, key=lambda a: len(a.capture))
            return best_cap

        # No capture: play lowest-value card
        places = [a for a in legal if a.capture is None]
        return min(places, key=lambda a: a.card.value)


class NetAgent:
    """Wraps a CuarentaNet for evaluation."""

    def __init__(self, net: CuarentaNet, device: torch.device, deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.net.eval()

    @torch.no_grad()
    def act(self, obs, legal: List[Action]) -> Action:
        obs_t = obs_to_tensors([obs], self.device)
        masks = ActionMasks.from_env_actions([legal])
        action_idx, _, _ = self.net.act(obs_t, masks, deterministic=self.deterministic)
        return legal[action_idx]


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
        logits = self.net(obs_t)[0]
        legal_indices = [encode_legal_action(a, legal) for a in legal]
        legal_logits = logits[0, legal_indices]
        if self.deterministic:
            choice = int(torch.argmax(legal_logits).item())
        else:
            dist = torch.distributions.Categorical(logits=legal_logits)
            choice = int(dist.sample().item())
        return legal[choice]
