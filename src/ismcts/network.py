"""
Cuarenta policy/value network.

Architecture:

  history:  [B, T, OBS_FLAT_DIM]     per-step observation (208)
  act_seq:  [B, T, ACT_FEAT_DIM]     per-step action that was taken (84 features)
  masks:    [B, T]                   was an action taken at step t? (act_mask)
  lengths:  [B]                      how many of the T steps are real (not padding)

        (per-step MLP)
  obs ────────────────► h_obs ──┐
                                ├── concat ──► LSTM ──► h_seq ──► h_last (state embedding)
  act ──► (linear) ─► h_act ────┘

  For the current step:
    state embedding:  s_emb = h_seq[:, -1]              [B, H]
    action features:  feats[MAX_ACTIONS, ACT_FEAT_DIM]  (per-slot features)
    action embedding: a_emb = action_mlp(feats)         [B, MAX_ACTIONS, H]
    policy logits:    logits_i = s_emb · a_emb_i / sqrt(H)
    value:            tanh(value_mlp(s_emb))            scalar in [-1, 1]

The policy over "slot i" is computed from the features of the action in that
slot (its card, its captures, whether it's a caida, etc.), not from the slot
index. This means the net doesn't have to memorize "slot 3 is good" — it
generalizes to any position of any legal action. Crucially, in ISMCTS the
legal action set differs between nodes, so slot-meaning-invariance is
essential for the prior to transfer.

Three things here are deliberate and were bugs/weaknesses before:

1. THE LAST STEP IS THE CURRENT DECISION. s_emb comes from the final LSTM step,
   and callers guarantee that step is the observation of the state being decided
   in (history.HistoryTracker.inputs_for appends it as the "pending" step with
   act_mask=False). Previously the sequence contained only *completed* decisions,
   so the net decided from a 4-ply-old observation.

2. THE ACTION HISTORY IS FEATURE-BASED, NOT SLOT-BASED. `act_hist_proj` maps the
   84-dim feature vector of the action that was taken. A slot one-hot would be
   meaningless across states (slot 3 is a different action everywhere), so the
   LSTM could not tell what its past moves actually were.

3. PADDING IS MASKED OUT (`lengths`). Sequences shorter than T are left-padded by
   the Dataset; packing tells the LSTM to ignore those steps instead of feeding it
   all-zero "no cards anywhere" observations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from env import CuarentaEnv
from history import OBS_FLAT_DIM, ACT_FEAT_DIM


N_ACTIONS = CuarentaEnv.MAX_ACTIONS          # 64 slots (max legal actions)
ACTION_FEAT_DIM = CuarentaEnv.ACTION_FEAT_DIM  # 84 features per candidate action


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
        # Two-layer MLP. Not strictly necessary (the LSTM could take raw obs), but
        # it gives the LSTM a more abstract per-step representation.
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ---- per-step encoder for the action taken at that step ----
        # Input is the 84-dim feature vector of the action (see history.ACT_FEAT_DIM),
        # NOT a one-hot over slots. Distinct from `action_mlp` below, which embeds
        # the *candidate* actions of the current decision:
        #   - `act_hist_proj` : "what did I do back then"          (history channel)
        #   - `action_mlp`    : "how good is each candidate now"   (policy channel)
        self.act_hist_proj = nn.Linear(ACT_FEAT_DIM, hidden)

        # ---- LSTM over the sequence of (obs, action) pairs ----
        self.lstm = nn.LSTM(
            input_size=hidden * 2,          # concat(h_obs, h_act)
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,               # [B, T, *] convention
        )

        # ---- action scoring branch ----
        # Each candidate action is described by an ACTION_FEAT_DIM feature vector.
        # We embed each into R^{action_embed_dim} and dot it with the state embedding.
        # `action_embed_dim` must equal `hidden` for the dot product to work.
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
        # Predicts a scalar in [-1, 1]: the expected game result for the team of
        # the player whose observation is in the last step. Trained against
        # +/-1 outcomes.
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,          # [B, T, obs_dim]
        act_seq: torch.Tensor,          # [B, T, ACT_FEAT_DIM]
        act_mask_seq: torch.Tensor,     # [B, T]      bool: was an action taken at step t?
        action_feats: torch.Tensor,     # [B, n_actions, action_feat_dim]
        action_mask: torch.Tensor,      # [B, n_actions] bool: slot is legal NOW
        lengths: Optional[torch.Tensor] = None,   # [B] int64: real (non-padding) steps
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          policy_logits:  [B, n_actions]   masked to -1e9 on illegal slots
          value:          [B]              in [-1, 1]
        """
        B, T, _ = obs_seq.shape
        H = self.hidden

        # ---- encode per-step inputs ----
        h_obs = self.obs_mlp(obs_seq)                   # [B, T, H]
        h_act = self.act_hist_proj(act_seq)             # [B, T, H]

        # If an action was not taken at step t (the pending current step, or a
        # padded step), don't let the projected action pollute the LSTM input.
        #
        # Why this matters: `act_hist_proj` is a linear layer WITH a bias, so
        # projecting a zero vector returns the bias b, not zero:
        #     act_hist_proj(0) = W @ 0 + b = b
        # The LSTM would then see a spurious, constant "an action was taken" signal
        # at every step without an action, and worse, training and inference would
        # disagree about what that phantom action means.
        #
        # act_mask_seq [B,T] -> [B,T,1], multiply: mask False => h_act[b,t,:] = 0.
        h_act = h_act * act_mask_seq.unsqueeze(-1).float()   # [B, T, H]

        lstm_in = torch.cat([h_obs, h_act], dim=-1)     # [B, T, 2H]

        if lengths is None:
            h_seq, _ = self.lstm(lstm_in)               # [B, T, H]
        else:
            # Left-padded sequences: the real steps are the LAST `lengths[b]` ones.
            # Packaging lets cuDNN skip the padding entirely. Because packing always
            # consumes the *first* `length` steps, we reverse time, pack, run, and
            # reverse back.
            lengths_cpu = lengths.detach().to("cpu", dtype=torch.int64).clamp(min=1)
            rev = lstm_in.flip(dims=[1])
            packed = pack_padded_sequence(
                rev, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            rev_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
            h_seq = rev_out.flip(dims=[1])              # [B, T, H]

        # The "state embedding" for the current decision is the LSTM output at the
        # last timestep, which (by the pending-step convention) is the observation
        # of the state we are deciding in. A length-1 sequence is legal here: the
        # LSTM then just runs its gates on the encoded pending step with zero
        # initial state. The search relies on that for interior nodes.
        s_emb = h_seq[:, -1, :]                         # [B, H]

        # ---- policy: score each candidate action slot ----
        a_emb = self.action_mlp(action_feats)           # [B, n_actions, H]

        # Logits via dot product, scaled by sqrt(H) to keep initial magnitudes
        # reasonable (same trick as attention).
        logits = (a_emb * s_emb.unsqueeze(1)).sum(-1) / (H ** 0.5)   # [B, n_actions]

        # Mask illegal actions so softmax / cross-entropy never sees them.
        logits = logits.masked_fill(~action_mask, -1e9)  # softmax(-1e9) underflows to 0 in fp32

        # ---- value ----
        value = self.value_mlp(s_emb).squeeze(-1)       # [B]

        return logits, value

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def load_checkpoint(net, path: str, map_location: str = "cpu"):
        """
        Load weights into `net` with a helpful message if the checkpoint is from an
        older version of the network.

        Checkpoints trained before the history/action-feature fixes are NOT loadable:
        `act_hist_proj` used to take a 64-dim slot one-hot and now takes 84 action
        features, so its weight has a different shape. Retrain the bootstrap instead
        of trying to convert (the input semantics changed, not just the layout).
        """
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        try:
            net.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not load checkpoint {path}: {e}\n"
                "If the mismatch is on 'act_hist_proj.weight', this checkpoint "
                "predates the history fixes (action history went from a 64-slot "
                "one-hot to 84 action features). Retrain with:\n"
                "  uv run src/ismcts/bootstrap.py --games 2000 --epochs 12"
            ) from e
        return net

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
        obs_seq = obs_seq.unsqueeze(0)
        act_seq = act_seq.unsqueeze(0)
        act_mask_seq = act_mask_seq.unsqueeze(0)
        action_feats = action_feats.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)

        logits, value = self.forward(
            obs_seq, act_seq, act_mask_seq, action_feats, action_mask
        )
        # Greedy action (argmax). ISMCTS does not use `act` directly; this is for
        # quick smoke tests / evaluation of the raw policy head.
        a = int(logits[0].argmax().item())
        return a, value[0]
