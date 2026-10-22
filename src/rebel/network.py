"""
The ReBeL network: a policy and a value conditioned on the public belief state.

WHAT IS DIFFERENT FROM THE OTHER PIPELINES
------------------------------------------
`src/ismcts` and `src/ppo` both condition the value on the acting player's own
observation and carry the game's history in an LSTM. ReBeL's value is a function of the
PUBLIC BELIEF STATE: it has to tell you how good the position is given what everyone
could be holding, not just what you are holding. Two consequences:

  * the input is the shared 208-dim observation (the acting seat's view: own hand, table,
    captures, scores, turn counts) PLUS 160 belief features -- the posterior probability
    that each of the 40 cards sits in each of the 4 hands (see `belief.ownership()`);
  * the trunk is FEEDFORWARD. There is no history LSTM: the belief already summarises
    everything that has happened, and the ISMCTS pipeline's LSTM exists to reconstruct
    that summary from a sequence of observations.

THE SIMPLIFICATION, STATED PLAINLY
----------------------------------
The belief features are the per-card ownership MARGINALS of the public belief state. They
throw away the correlations between hands that the particle set actually represents (for
example "if seat 1 has no sevens it probably has the king"), so the network sees a lossy
view of the PBS. Full ReBeL feeds the belief itself. Feeding marginals is the smallest
change that keeps the network belief-conditioned, and it is the obvious thing to improve
later -- either by widening the summary (pairwise co-occurrence features) or by having the
network consume the particles directly (a set encoder).

The policy head is the same action-feature head the other pipelines use: each candidate
action is scored from its own 84-dim feature vector, so slot `i` never has to mean the
same thing in two positions. Logits are the dot product of the state embedding with the
action embedding, scaled by 1/sqrt(H) -- the ISMCTS trick, kept here so the two pipelines
stay comparable and so the head generalises across positions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from core import ACTION_FEAT_DIM, BELIEF_DIM, N_SLOTS, OBS_DIM

INPUT_DIM = OBS_DIM + BELIEF_DIM       # 208 + 160 = 368


class RebelNet(nn.Module):
    """Feedforward actor-critic over (observation, belief marginals, action features)."""

    def __init__(self, obs_dim: int = INPUT_DIM, action_feat_dim: int = ACTION_FEAT_DIM,
                 n_slots: int = N_SLOTS, hidden: int = 256, value_hidden: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_feat_dim = int(action_feat_dim)
        self.n_slots = int(n_slots)
        self.hidden = int(hidden)
        self.value_hidden = int(value_hidden)

        # One dropout mask per forward, on the embedding BOTH heads read, so the policy
        # and the value never see different perturbations of the same belief state.
        # Identity at p=0 and in eval mode, so the search, evaluation and exports see the
        # deterministic net.
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self.obs_trunk = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_feat_dim, self.hidden), nn.ReLU(),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(self.hidden, self.value_hidden), nn.ReLU(),
            nn.Linear(self.value_hidden, 1), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, feats: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs   [B, 368]        observation + belief marginals
        feats [B, S, 84]      candidate-action features, one row per slot
        mask  [B, S] bool     which slots are legal (optional)

        Returns (logits [B, S], value [B]) with the value from the acting team's
        perspective, in [-1, 1] (+1 = that team wins).
        """
        state_emb = self.dropout(self.obs_trunk(obs))         # [B, H], one mask
        action_emb = self.action_mlp(feats)                   # [B, S, H]
        logits = (action_emb * state_emb.unsqueeze(1)).sum(-1) / math.sqrt(self.hidden)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        value = self.value_mlp(state_emb).squeeze(-1)         # [B]
        return logits, value

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def policy_probs(self, obs: np.ndarray, feats: np.ndarray, n_legal: int) -> np.ndarray:
        """
        Softmax over the legal slots, as a numpy array of length `n_legal`.

        Used as the belief's likelihood model: "how likely was this player to play this
        card, given this hand and this public state". The legal slots are assumed packed
        first, which is how `action_features_matrix` builds them.
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        feats_t = torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
        logits, _ = self.forward(obs_t, feats_t, None)
        probs = torch.softmax(logits[0, :n_legal], dim=0)
        return probs.cpu().numpy()

    @torch.no_grad()
    def value_of(self, obs: np.ndarray, feats: np.ndarray) -> float:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        feats_t = torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
        _, value = self.forward(obs_t, feats_t, None)
        return float(value.item())

    def architecture_config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_feat_dim": self.action_feat_dim,
            "n_slots": self.n_slots,
            "hidden": self.hidden,
            "value_hidden": self.value_hidden,
            "dropout": self.dropout_p,
        }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_rebel_checkpoint(path, net: RebelNet, *, step: int = 0, config: Optional[dict] = None,
                          optimizer: Optional[torch.optim.Optimizer] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "step": int(step),
        "architecture": "rebel_belief_actor_critic",
        **net.architecture_config(),
    }
    if config is not None:
        payload["config"] = dict(config)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_rebel_checkpoint(path, device: torch.device = torch.device("cpu"),
                          **overrides) -> RebelNet:
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in state:
        raise KeyError(
            f"{path} is not a ReBeL checkpoint (no 'model_state'). The other pipelines "
            "save their own formats: src/ppo writes 'model_state' too but with an LSTM "
            "and a different input width, and src/ismcts uses CuarentaNet."
        )
    dims = {
        "obs_dim": int(state.get("obs_dim", INPUT_DIM)),
        "action_feat_dim": int(state.get("action_feat_dim", ACTION_FEAT_DIM)),
        "n_slots": int(state.get("n_slots", N_SLOTS)),
        "hidden": int(state.get("hidden", 256)),
        "value_hidden": int(state.get("value_hidden", 128)),
        "dropout": float(state.get("dropout", 0.0)),
    }
    dims.update({k: v for k, v in overrides.items() if v is not None})
    net = RebelNet(**dims).to(device)
    net.load_state_dict(state["model_state"])
    net.eval()
    return net
