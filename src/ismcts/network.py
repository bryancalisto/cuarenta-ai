"""
Cuarenta policy/value network.

Architecture:

  history:  [T, OBS_FLAT]      per-step observation
  actions:  [T, MAX_ACTIONS]            per-step action taken (one-hot)
  masks:    [T, MAX_ACTIONS]            per-step legality mask

        (per-step MLP)
  obs ────────────────► h_obs ──┐
                                ├── concat ──► LSTM ──► h_seq ──► h_last (state embedding)
  act ──► (linear) ─► h_act ────┘

  For the current step:
    state embedding:  s_emb = h_last                    [H]
    action features:  feats[MAX_ACTIONS, 84]  (per-slot features)
    action embedding: a_emb = action_mlp(feats)         [MAX_ACTIONS, H]
    policy logits:    logits_i = s_emb · a_emb_i        [MAX_ACTIONS]

  Value head:
    value = tanh( value_mlp(h_last) )                   scalar in [-1, 1]

  The policy over "slot i" is computed from the features of the action in that
  slot (its card, its captures, whether it's a caida, etc.), not from the slot
  index. This means the net doesn't have to memorize "slot 3 is good" — it
  generalizes to any position of any legal action. Crucially, in ISMCTS the
  legal action set differs between nodes, so slot-meaning-invariance is
  essential for the prior to transfer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from env import CuarentaEnv


OBS_FLAT_DIM = 40 * 5 + 4 + 4   # 5 bitvecs (my_hand, table, caps×2, last_placed) + 4 scalars + plays_this_deal(4)
N_ACTIONS = CuarentaEnv.MAX_ACTIONS
ACTION_FEAT_DIM = CuarentaEnv.ACTION_FEAT_DIM

class CuarentaNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_FLAT_DIM,
        n_actions: int = N_ACTIONS,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden: int = 256,
        lstm_layers: int = 2,
        action_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.action_feat_dim = action_feat_dim
        self.hidden = hidden

        # ---- per-step encoder for the observation ----
        # Two-layer MLP. Not strictly necessary (LSTM could take raw obs), but
        # helps the LSTM see a more abstract representation and reduces the
        # input dim it has to handle.
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ---- per-step encoder for the action taken (one-hot over 32 slots) ----
        # this is the "action that was taken", at previous timesteps. It's
        # a different thing from the action-scoring branch below (which scores
        # candidate actions at the current step). Keeping the names distinct:
        #   - `act_hist_proj` : projects a one-hot of the previously taken action
        #   - `action_mlp`    : embeds per-slot action features for scoring
        self.act_hist_proj = nn.Linear(n_actions, hidden)

        # ---- LSTM over the sequence of (obs, action) pairs ----
        self.lstm = nn.LSTM(
            input_size=hidden * 2,          # concat(h_obs, h_act)
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,               # [B, T, *] convention
        )

        # ---- action scoring branch ----
        # Each candidate action (there are up to "n_actions" of them) is
        # described by an ACTION_FEAT_DIM feature vector. We embed each into R^{action_embed_dim}
        # and then dot with the state embedding.
        #
        # `action_embed_dim` must equal `hidden` so the dot product works.
        if action_embed_dim is None:
            action_embed_dim = hidden
        assert action_embed_dim == hidden, (
            f"action_embed_dim ({action_embed_dim}) must equal hidden ({hidden}) "
            "so s_emb · a_emb is a valid dot product."
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(action_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # ---- value head ----
        # Predicts a scalar in [-1, 1] from the acting player's team POV.
        # tanh keeps it bounded; targets are +1/-1 for win/loss.
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )


    def forward(
        self,
        obs_seq: torch.Tensor,          # [B, T, obs_dim]
        act_seq: torch.Tensor,          # [B, T, n_actions]   one-hot per step
        act_mask_seq: torch.Tensor,     # [B, T]              bool: was there an action at step t?
        action_feats: torch.Tensor,     # [B, n_actions, action_feat_dim]
        action_mask: torch.Tensor,      # [B, n_actions]      bool: slot is legal NOW
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          policy_logits:  [B, n_actions]   masked to -inf on illegal slots
          value:          [B]              in [-1, 1]
          new_hidden:     LSTM (h, c) for potential incremental use
        """
        B, T, _ = obs_seq.shape
        H = self.hidden

        # ---- encode per-step inputs ----
        h_obs = self.obs_mlp(obs_seq)                   # [B, T, H]
        h_act = self.act_hist_proj(act_seq)             # [B, T, H]

        # If an action was not taken at step t (e.g. the current step for
        # which we're about to predict), we don't want the projected action
        # to pollute the LSTM input. Zero it out.
        #
        # act_mask_seq is [B, T] bool. Broadcast to [B, T, 1], then multiply.
        #
        # The LSTM input at each timestep is concat(h_obs, h_act) — the encoded observation and the encoded action taken at that step.
        #
        # But consider the last timestep in the sequence. That's the current decision point: we've observed the current state, but no action has been taken yet. So act_seq[t_last] is all zeros (no one-hot).
        # If we just ran act_hist_proj on that zero vector, we'd get:
        #
        # act_hist_proj(0) = W @ 0 + b = b
        #
        # where b is the bias vector of the linear layer. Not zero. The linear layer has a bias by default, so projecting a zero input gives you b, a learned but constant vector — a spurious "action was taken" signal.
        #
        # That's bad because:
        # - The LSTM would see a fake action at the current step. It would treat "no action taken yet" as equivalent to some action — some specific phantom action whose embedding happens to be b.
        # - The same problem happens at every padded timestep if you pad shorter histories to a common length (e.g., at the start of an episode when T < max_len).
        # - During training vs. inference inconsistency: at training time, some timesteps have actions and some don't. If the LSTM mixes them up, the value head is trained on polluted inputs and gives wrong estimates.
        #
        # Let's break it down:
        # - act_mask_seq is [B, T] bool. True = "an action was taken at this step", False = "no action yet (current step, or padding)".
        # - act_mask_seq.unsqueeze(-1) → [B, T, 1]. Adds a trailing dim so we can broadcast against [B, T, H].
        # - .float() → converts bool True/False to 1.0/0.0.
        # - Multiplying h_act * mask: for each (b, t), if mask[b, t] == False, the whole h_act[b, t, :] vector becomes zero.
        #
        # After this operation:
        # - Timesteps with an action: h_act[b, t, :] = act_hist_proj(one_hot) — the real learned embedding.
        # - Timesteps without an action: h_act[b, t, :] = 0 — genuinely zero, not "bias vector from a zero input".
        # Now when we concat(h_obs, h_act) and feed the LSTM, the "no action" timesteps carry an unambiguous signal: h_act = 0 means "no action happened here".
        h_act = h_act * act_mask_seq.unsqueeze(-1).float()   # [B, T, H]

        lstm_in = torch.cat([h_obs, h_act], dim=-1)     # [B, T, 2H]

        h_seq, new_hidden = self.lstm(lstm_in, hidden)  # [B, T, H]

        # The "state embedding" for the current decision is the LSTM output
        # at the last timestep. If you later want to act mid-sequence, just
        # slice h_seq differently; for now, always take the last step.
        s_emb = h_seq[:, -1, :]                         # [B, H]

        # ---- policy: score each candidate action slot ----
        # action_feats: [B, n_actions, action_feat_dim] -> [B, n_actions, H]
        a_emb = self.action_mlp(action_feats)           # [B, n_actions, H]

        # Logits via dot product: [B, n_actions]
        #
        # Optionally we could also normalize s_emb and a_emb; scaling by sqrt(H) keeps
        # initial logit magnitudes reasonable, similar to attention.
        logits = (a_emb * s_emb.unsqueeze(1)).sum(-1) / (H ** 0.5)   # [B, n_actions]

        # Mask illegal actions so softmax / cross-entropy never sees them.
        # action_mask is [B, n_actions] bool. "masked_fill" with a large
        # negative number on illegal slots.
        logits = logits.masked_fill(~action_mask, -1e9) # Softmax of -1e9 underflows to exactly 0 in fp32

        # ---- value ----
        value = self.value_mlp(s_emb).squeeze(-1)       # [B]

        return logits, value, new_hidden

    # ---------------------------------------------------------------- helpers

    @torch.no_grad()
    def act(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        act_mask_seq: torch.Tensor,
        action_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """
        Convenience for inference at a single decision point.

        Accepts tensors without a batch dim ([T, ...] and [n_actions, ...]),
        adds one, and returns (chosen_action_index, value).
        """
        # Add batch dim
        obs_seq = obs_seq.unsqueeze(0)
        act_seq = act_seq.unsqueeze(0)
        act_mask_seq = act_mask_seq.unsqueeze(0)
        action_feats = action_feats.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)

        logits, value, _ = self.forward(
            obs_seq, act_seq, act_mask_seq, action_feats, action_mask
        )
        # Greedy action (argmax). ISMCTS will not use `act` directly; this is
        # just for quick smoke tests / eval.
        a = int(logits[0].argmax().item())
        return a, value[0]