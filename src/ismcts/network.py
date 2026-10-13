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
    value:            value_mlp(s_emb)                  scalar, see VALUE TARGET below

VALUE TARGET (`value_target`, default "win")
--------------------------------------------
The value head is the last layer of the search's leaf evaluation, so its UNITS are part
of the search's contract, not just a training detail. Two targets are supported:

  * "win"    — the ±1 game result, head squashed by `tanh` into (-1, 1). This is the
               AlphaZero target and the default: it is exactly what the search backs up
               at a terminal node, and winning is ZERO-SUM, which is what makes the
               search's "negate the value when the mover is the opponent" backup valid.
  * "return" — the env's dense reward as a return-to-go, divided by
               `RETURN_VALUE_SCALE`. The head is LINEAR here (no tanh): a game return is
               ~10-150, and squashing that into (-1, 1) would make the head unable to
               separate a good position from a winning one.
               The target for a decision is the SHAPED return from that ply on:
                   G_t = sum_{t' >= t} sign * r_t' ,  sign = +1 if the actor at t' is on
                                                        this decision's team, else -1
               i.e. exactly the quantity `NetworkISMCTS` backs up in the same mode. The
               sign flip is what lets ONE value head serve both teams.
               `--value-lambda < 1` computes that return with TD(lambda) bootstrapping
               instead of summing to the end of the game (SAME expected value, less
               variance; see `selfplay.backfill_value_targets`, which also explains why it
               only pays off once the head is any good).

WHY THE MODE LIVES ON THE NET: `NetworkISMCTS` reads `net.value_target` and mirrors it
(terminal value, whether per-step rewards are accumulated, and how an unvisited child is
scored). A return-mode checkpoint searched by a win-mode search — or the reverse — is
silently wrong rather than loud, so build the net with `CuarentaNet(value_target=...)`
wherever a checkpoint is loaded (`train.py: _load_net` does it from `--value-target`).
Changing the mode of an existing run invalidates its weights: every checkpoint trained so
far is win-mode.

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

# What the value head predicts (see the VALUE TARGET section in the module docstring).
VALUE_TARGETS = ("win", "return")

# One warning per process when a checkpoint has no provenance sidecar (see
# `CuarentaNet.resolve_value_target`); a per-call warning would drown the console.
_WARNED_MISSING_VALUE_TARGET = False

# Divisor that maps the env's return scale onto the UNIT-VARIANCE scale the ±1 win target
# already lives on. Dividing every target and every search value by the same number cannot
# change a ranking or a visit count on its own, so this is "only" a conditioning constant —
# but it decides three things that matter:
#
#   * the SIZE of the value loss, i.e. how much of `policy_loss + value_loss` is the value
#     head's gradient. At the wrong scale the value head gets a negligible share and never
#     learns (measured at 100: value 0.02 vs policy 0.92 = 2% of the loss, versus ~20-30%
#     in win mode where the targets have variance ~1),
#   * the size of Q's advantages next to PUCT's exploration term
#     `c_puct * P * sqrt(N) / (1 + N)`, which `c_puct = 1.5` assumes are O(1) — the same
#     assumption win mode satisfies,
#   * whether `mse > 1.0` / `mse < 0.05` style thresholds mean anything: with targets of
#     unit variance, "1.0" is exactly "no better than predicting the mean" in BOTH modes.
#
# MEASURED (this is why it is 30, not 100): the shaped return-to-go target has mean ~0 and
# std ~31-36 points — 35.8 over 30 heuristic games, and 31.0 on the trainer's probe games
# (`eval/probe_target_var` 0.0962 at scale 100). So 30 puts the targets at std ~1.1.
#
# A return-mode checkpoint's head is expressed in these units, and the scale is NOT in the
# state_dict, so changing this constant silently invalidates existing return-mode
# checkpoints (their head would read ~3x too small). Start such a run fresh.
RETURN_VALUE_SCALE = 30.0


class CuarentaNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_FLAT_DIM,
        n_actions: int = N_ACTIONS,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden: int = 256,
        lstm_layers: int = 2,
        action_embed_dim: Optional[int] = None,
        value_target: str = "win",
        dropout: float = 0.0,
    ):
        super().__init__()
        if value_target not in VALUE_TARGETS:
            raise ValueError(
                f"value_target must be one of {VALUE_TARGETS}, got {value_target!r}")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.action_feat_dim = action_feat_dim
        self.hidden = hidden
        # Read by `NetworkISMCTS` (which mirrors the units) and by `_load_net`-style
        # helpers so a checkpoint is always searched with the semantics it was trained on.
        self.value_target = value_target
        self.return_value_scale = RETURN_VALUE_SCALE

        # ---- dropout on the SHARED state embedding ----
        # `dropout` is a TRAINING-time regulariser for the small-replay-buffer regime: the
        # buffer is replayed for `EPOCHS_PER_ITER x REPLAY_ITERS` passes, and this net has
        # 1.6M parameters for ~20k decisions, so both heads can fit the buffer instead of
        # the game (measured in return mode: in-sample value R^2 0.79 while held-out R^2
        # was ~0.05).
        #
        # Placement matters: ONE mask is drawn per forward and applied to `s_emb`, the
        # embedding BOTH heads read. Dropping inside each head separately would give the
        # policy and the value different views of the same state, and dropping inside the
        # LSTM is not available in PyTorch except between stacked layers (`nn.LSTM(dropout=)`
        # is ignored for `num_layers=1` and applies to the inter-layer output otherwise) --
        # the recurrent state carries the entire action history, and corrupting it every
        # step would fight the very thing the LSTM exists for.
        #
        # It is IDENTITY in eval mode, so the search, the evaluators and the ONNX export are
        # unaffected and they do not need to know about it.
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

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
        # The output ACTIVATION is what makes the two value targets different:
        #
        #   win    -> tanh, so the output is a win probability-like score in (-1, 1) and
        #             can never exceed the terminal values the search backs up.
        #   return -> identity, because the target is a scaled episode return (~0.1-1.5)
        #             that is NOT bounded by 1 and is mostly positive: `tanh` would
        #             saturate and destroy the ordering the search needs.
        #
        # Both keep the same parameter shapes, so a state_dict loads into either mode
        # WITHOUT an error. That is exactly why the mode is stored on the module and read
        # back by the search and by `_load_net` instead of being guessed.
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Identity() if value_target == "return" else nn.Tanh(),
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

        # One dropout mask on the shared embedding (identity unless training and
        # dropout > 0), so both heads read the SAME perturbed view of the state.
        s_emb = self.dropout(s_emb)

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
    def read_sidecar(path: str) -> dict:
        """
        The provenance `src/ismcts/train.py` writes next to every checkpoint it saves, or {}
        for a checkpoint that has none (all of them predating the sidecar).
        """
        import json
        try:
            with open(str(path) + ".json") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    @classmethod
    def resolve_value_target(cls, path: str, explicit: str | None = None) -> str:
        """
        Which value target a checkpoint was trained with: the explicit argument if given,
        else its sidecar, else "win" WITH A LOUD ONE-TIME WARNING.

        This is the trap this whole mechanism exists for. The win head ends in `tanh` and
        the return head is linear, but that is an ACTIVATION: both load the same weights
        without an error, and if the mode is wrong then `NetworkISMCTS` ignores the dense
        reward and backs up ±1 game results — i.e. it searches as a DIFFERENT AGENT than the
        one that was trained, and a 100-game match silently reports the wrong strength.
        MEASURED: evaluating a return-mode checkpoint with a win-mode search gave 64% against
        the heuristic where the trainer's own (correctly built) evaluation gave 81.7%.
        """
        if explicit:
            return explicit
        recorded = cls.read_sidecar(path).get("value_target")
        if recorded in VALUE_TARGETS:
            return recorded
        global _WARNED_MISSING_VALUE_TARGET
        if not _WARNED_MISSING_VALUE_TARGET:
            _WARNED_MISSING_VALUE_TARGET = True
            import sys
            print(f"[net] WARNING: {path} has no provenance sidecar, so its value target is "
                  f"unknown; assuming 'win'. If it was trained with --value-target return, "
                  f"pass --value-target return (or rebuild the checkpoint with the current "
                  f"trainer): the search would otherwise ignore the dense reward and "
                  f"evaluate a different agent than the one you trained.",
                  file=sys.stderr)
        return "win"

    @classmethod
    def from_checkpoint(cls, path: str, *, value_target: str | None = None, **kwargs):
        """
        Build a net for `path` with the head its weights were trained with, and load them.

        Every standalone entry point (evaluators, the Tk UI, the cross-pipeline script)
        should use this instead of `CuarentaNet()` + `load_checkpoint`: the mode is what
        makes the search and the head agree, and it cannot be read off the weights.
        """
        net = cls(value_target=cls.resolve_value_target(path, value_target), **kwargs)
        return cls.load_checkpoint(net, path)

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
