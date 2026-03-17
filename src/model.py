"""
model.py — PyTorch neural network for Cuarenta PPO agent.

Architecture:
  ObsEncoder   : flat obs dict → 256-dim embedding (Transformer over card slots)
  PolicyHead   : embedding → (card logits, capture logits), masked
  ValueHead    : embedding → scalar V(s)
  CuarentaNet  : full actor-critic combining the above
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np


# ─── Dimensions ──────────────────────────────────────────────────────────────

CARD_DIM   = 40     # one-hot card vectors
OBS_SCALAR = 5      # my_team_score, opp_team_score, current_player, deck_remaining, last_placed_scalar
EMBED_DIM  = 256
NUM_HEADS  = 4
FF_DIM     = 512
N_LAYERS   = 2


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _mlp(dims: List[int], activation=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


# ─── Observation Encoder ─────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """
    Encodes the full observation dict into a fixed-size embedding.

    Input channels (all float32 numpy arrays, converted to tensors):
        my_hand           (40,)  binary
        table             (40,)  binary
        played_this_deal  (40,)  binary
        last_placed       (40,)  binary
        my_team_score     (1,)
        opp_team_score    (1,)
        current_player    (1,)
        deck_remaining    (1,)

    We stack the 4 card bit-vectors into a (4, 40) card matrix,
    project each of the 40 card slots across 4 channels → card embeddings,
    run a small Transformer over the 40 slots,
    then pool + concat scalar features → final embedding.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Project 4-channel card features to embed_dim per card slot
        self.card_proj = nn.Linear(4, embed_dim)

        # Positional embedding for the 40 card slots
        self.pos_embed = nn.Embedding(CARD_DIM, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Scalar feature projection
        self.scalar_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict of tensors, each (B, dim)
        Returns:
            embedding : (B, embed_dim)
            card_slots : (B, 40, embed_dim)  — used by policy head for action masking
        """
        B = obs["my_hand"].shape[0]

        # Stack card channels: (B, 4, 40) → transpose → (B, 40, 4)
        card_matrix = torch.stack([
            obs["my_hand"],
            obs["table"],
            obs["played_this_deal"],
            obs["last_placed"],
        ], dim=1).transpose(1, 2)   # (B, 40, 4)

        # Project each card slot: (B, 40, embed_dim)
        card_slots = self.card_proj(card_matrix)

        # Add positional embeddings
        positions = torch.arange(CARD_DIM, device=card_slots.device)
        card_slots = card_slots + self.pos_embed(positions).unsqueeze(0)

        # Transformer: (B, 40, embed_dim)
        card_slots = self.transformer(card_slots)

        # Pool over card dimension
        pooled = card_slots.mean(dim=1)   # (B, embed_dim)

        # Scalar features: normalise scores to [0,1]
        scalars = torch.cat([
            obs["my_team_score"] / 40.0,
            obs["opp_team_score"] / 40.0,
            obs["current_player"] / 3.0,
            obs["deck_remaining"] / 40.0,
        ], dim=-1)   # (B, 4)
        scalar_emb = self.scalar_proj(scalars)   # (B, 64)

        # Fuse
        fused = self.fusion(torch.cat([pooled, scalar_emb], dim=-1))   # (B, embed_dim)

        return fused, card_slots


# ─── Policy Head ─────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    Two-headed output:
      1. card_logits    (B, 40)  — which card to play
      2. capture_logits (B, 40)  — which table cards to include in capture

    At inference, we build a flat action index over legal actions and
    sample from a joint distribution. During training we use the
    log-probs of the chosen (card, capture_set) pair.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Card head: attend from global embedding to each card slot
        self.card_query = nn.Linear(embed_dim, embed_dim)
        self.card_key   = nn.Linear(embed_dim, embed_dim)
        self.card_out   = nn.Linear(embed_dim, 1)

        # Capture head: per-slot binary logit
        self.capture_mlp = _mlp([embed_dim * 2, 128, 1])

    def forward(
        self,
        embedding: torch.Tensor,      # (B, embed_dim)
        card_slots: torch.Tensor,     # (B, 40, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            card_logits    (B, 40)
            capture_logits (B, 40)
        """
        B = embedding.shape[0]

        # Card logits via dot-product attention
        q = self.card_query(embedding).unsqueeze(1)       # (B, 1, E)
        k = self.card_key(card_slots)                     # (B, 40, E)
        card_logits = (q * k).sum(-1) / (EMBED_DIM ** 0.5)  # (B, 40)

        # Capture logits: concat global embedding with each slot
        global_expanded = embedding.unsqueeze(1).expand(-1, CARD_DIM, -1)  # (B, 40, E)
        capture_input   = torch.cat([global_expanded, card_slots], dim=-1)  # (B, 40, 2E)
        capture_logits  = self.capture_mlp(capture_input).squeeze(-1)       # (B, 40)

        return card_logits, capture_logits


# ─── Value Head ──────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = _mlp([embed_dim, 128, 1])

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)   # (B,)


# ─── Full Actor-Critic ────────────────────────────────────────────────────────

class CuarentaNet(nn.Module):
    """
    Full actor-critic network.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = ObsEncoder(embed_dim)
        self.policy  = PolicyHead(embed_dim)
        self.value   = ValueHead(embed_dim)

    def forward(self, obs: Dict[str, torch.Tensor]):
        embedding, card_slots = self.encoder(obs)
        card_logits, capture_logits = self.policy(embedding, card_slots)
        value = self.value(embedding)
        return card_logits, capture_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding, _ = self.encoder(obs)
        return self.value(embedding)

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        legal_action_masks: "ActionMasks",
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action index from the legal action distribution.

        Returns:
            action_idx   : int index into legal_action_masks.actions
            log_prob     : scalar tensor
            entropy      : scalar tensor
        """
        card_logits, capture_logits, value = self.forward(obs)
        dist = ActionDistribution(card_logits, capture_logits, legal_action_masks)
        if deterministic:
            action_idx = dist.mode()
        else:
            action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy  = dist.entropy()
        return action_idx.item(), log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        action_indices: torch.Tensor,
        legal_action_masks: "ActionMasks",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update: compute log_probs, entropy, values for a batch.

        Returns:
            log_probs (B,)
            entropy   (B,)
            values    (B,)
        """
        card_logits, capture_logits, values = self.forward(obs)
        dist = ActionDistribution(card_logits, capture_logits, legal_action_masks)
        log_probs = dist.log_prob(action_indices)
        entropy   = dist.entropy()
        return log_probs, entropy, values


# ─── Action Masking & Distribution ───────────────────────────────────────────

class ActionMasks:
    """
    Encodes the legal actions for a batch of states as sparse masks.

    legal_actions is a list (len=B) of lists of (card_idx, capture_bitvec) tuples,
    where card_idx ∈ [0,39] and capture_bitvec is a 40-bit int.
    """

    def __init__(self, legal_actions: List[List[Tuple[int, int]]]):
        self.legal_actions = legal_actions   # raw list of (card_idx, cap_bitvec) per batch item

    @classmethod
    def from_env_actions(cls, env_actions_batch) -> "ActionMasks":
        """Convert list of lists of env.Action objects to ActionMasks."""
        result = []
        for actions in env_actions_batch:
            encoded = []
            for a in actions:
                card_idx = a.card.index
                cap_bitvec = 0
                if a.capture:
                    for c in a.capture:
                        cap_bitvec |= (1 << c.index)
                encoded.append((card_idx, cap_bitvec))
            result.append(encoded)
        return cls(result)


class ActionDistribution:
    """
    Joint distribution over (card, capture_set) using masked softmax.

    We enumerate all legal (card_idx, cap_bitvec) pairs and assign each
    a score = card_logit[card_idx] + sum(capture_logit[i] for i in cap_bits).
    Then softmax over legal actions only.
    """

    def __init__(
        self,
        card_logits: torch.Tensor,      # (B, 40)
        capture_logits: torch.Tensor,   # (B, 40)
        masks: ActionMasks,
    ):
        self.card_logits    = card_logits
        self.capture_logits = capture_logits
        self.masks          = masks
        self.B              = card_logits.shape[0]
        self.device         = card_logits.device

        # Pre-compute per-batch legal action scores
        self._scores_list, self._max_actions = self._compute_scores()

    def _compute_scores(self):
        scores_list = []
        max_n = 0
        for b in range(self.B):
            actions = self.masks.legal_actions[b]
            scores = []
            for (card_idx, cap_bitvec) in actions:
                s = self.card_logits[b, card_idx]
                # Sum capture logits for each bit set
                bits = cap_bitvec
                while bits:
                    lsb = bits & (-bits)
                    bit_idx = lsb.bit_length() - 1
                    s = s + self.capture_logits[b, bit_idx]
                    bits ^= lsb
                scores.append(s)
            scores_list.append(torch.stack(scores) if scores else torch.zeros(1, device=self.device))
            max_n = max(max_n, len(scores))
        return scores_list, max_n

    def _probs(self, b: int) -> torch.Tensor:
        return F.softmax(self._scores_list[b], dim=0)

    def sample(self) -> torch.Tensor:
        """Returns (B,) tensor of sampled action indices."""
        indices = []
        for b in range(self.B):
            probs = self._probs(b)
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)
        return torch.tensor(indices, device=self.device)

    def mode(self) -> torch.Tensor:
        indices = []
        for b in range(self.B):
            idx = self._scores_list[b].argmax().item()
            indices.append(idx)
        return torch.tensor(indices, device=self.device)

    def log_prob(self, action_indices: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for b in range(self.B):
            probs = self._probs(b)
            idx = action_indices[b].item()
            # Clamp for numerical safety
            lp = torch.log(probs[idx].clamp(min=1e-8))
            log_probs.append(lp)
        return torch.stack(log_probs)

    def entropy(self) -> torch.Tensor:
        entropies = []
        for b in range(self.B):
            probs = self._probs(b)
            ent = -(probs * torch.log(probs.clamp(min=1e-8))).sum()
            entropies.append(ent)
        return torch.stack(entropies)