"""
Networks used by the learning pipelines.

  * `PolicyNetwork` — the plain MLP policy used by DAgger imitation learning and by
    the deployed ONNX model. Kept byte-compatible with `scripts/pth2onnx.py`, which
    imports it by name.
  * `ActorCriticNetwork` — the PPO actor-critic: the same trunk, an optional LSTM,
    a policy head and a value head.

The legacy `PolicyNetwork` is still here because the DAgger imitation pipeline and the
ONNX exporter use it; the PPO pipeline uses `ActorCriticNetwork`.

TWO CONVENTIONS WORTH KNOWING BEFORE READING THE CODE
-----------------------------------------------------
1. THE FLAT 1640-WAY HEAD BELONGS TO `PolicyNetwork` ONLY. Its `action_size` is the
   size of the flat action space (`env.ActionEncoder.max_action_idx` = 40*41 = 1640),
   not the number of legal actions. Legality is applied at selection/loss time by
   gathering only the legal ids, never inside the network. That is why the old
   checkpoints — DAgger's included — carry a head far wider than any position needs.

   `ActorCriticNetwork` (the PPO policy) does NOT work that way: its head scores each
   legal action from its own 84-dim feature vector (at most 64 slots), so it is told
   the shape of a move instead of memorising an action id, and it generalises across
   positions. The ONNX export follows the PPO network — 208-dim observation in, 64
   action scores out — and the sidecar JSON beside the exported model records that.

2. THE LSTM STATE IS CARRIED BY THE CALLER, NOT THE NETWORK. The network is
   stateless: it takes an optional `(h, c)` and returns the new one. During rollout
   collection the trainer keeps one `(h, c)` per environment and advances it on EVERY
   turn of the match — for the learning seat AND for the opponent net, which only acts
   on three of the four seats. During the update the same sequence is replayed in
   order from the stored starting state (see `src/ppo/rollout.py:
   run_recurrent_forward`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]


class PolicyNetwork(nn.Module):
    """Flat MLP policy: observation -> logits over the whole action space."""

    def __init__(self, obs_size: int = 121, action_size: int = 200, hidden_size: int = 256):
        """
        obs_size: total observation vector size
        action_size: max number of possible action encodings
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. obs shape: (batch, obs_size). Output: (batch, action_size)."""
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticNetwork(nn.Module):
    """
    PPO actor-critic with an ACTION-FEATURE policy head — the same design the ISMCTS
    pipeline uses (`src/ismcts/network.py: CuarentaNet`), so both pipelines speak the
    same observation and action representation.

        obs [B, T, 208] -> trunk(208→H→H) -> LSTM (optional) -> s_emb [B, T, H]
        feats [B, n_slots, 84] -> action_mlp(84→H→H) -> a_emb [B, n_slots, H]
        logits = (a_emb · s_emb) / sqrt(H)          (masked to legal slots by the caller)
        value  = MLP(s_emb) -> linear, trained on normalised returns

    Why features instead of a flat logit per action id (the old 1640-wide head):
    the meaning of a slot changes from position to position, so a flat head has to
    learn "action id 617 is good" instead of "capturing the card that was just placed
    is good". Feature scoring is position-invariant, generalises across positions, and
    keeps the head small.

    The LSTM is stateless from the network's point of view: the caller owns (h, c) —
    see `src/ppo/rollout.py` for the replay contract.
    """

    def __init__(
        self,
        obs_dim: int = 208,
        action_feat_dim: int = 84,
        n_slots: int = 64,
        hidden: int = 256,
        use_lstm: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        value_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_feat_dim = action_feat_dim
        self.n_slots = n_slots
        self.hidden = hidden
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.value_hidden = value_hidden

        # One dropout mask per forward, on the embedding BOTH heads read, so the policy
        # and the value never see different perturbations of the same state. Identity at
        # p=0 and in eval mode, so acting, evaluation and the ONNX export are unaffected.
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if use_lstm:
            self.lstm = nn.LSTM(input_size=hidden, hidden_size=lstm_hidden,
                                num_layers=lstm_layers, batch_first=True)
            feature_size = lstm_hidden
        else:
            feature_size = hidden

        # The policy logits are a dot product between the state embedding and each
        # action embedding, so both branches must end in the same width: `feature_size`
        # (the LSTM width when recurrent, the trunk width otherwise).
        self.action_mlp = nn.Sequential(
            nn.Linear(action_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )
        self.feature_size = feature_size

    # ------------------------------------------------------------------ pieces

    def embed_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Trunk only: observation -> per-step feature (before the LSTM)."""
        return self.obs_mlp(obs)

    def embed_actions(self, action_feats: torch.Tensor) -> torch.Tensor:
        """Per-candidate-action embedding: [.., n_slots, action_feat_dim] -> [.., n_slots, H]."""
        return self.action_mlp(action_feats)

    def logits_from(self, s_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        Policy logits by dot product, scaled by sqrt(H).

        `s_emb` [.., H] and `a_emb` [.., n_slots, H] broadcast: the result is
        [.., n_slots]. Legality is NOT applied here — the caller masks.
        """
        return (a_emb * s_emb.unsqueeze(-2)).sum(-1) / (self.feature_size ** 0.5)

    def encode_sequence(self, obs: torch.Tensor,
                        lstm_state: Optional[LSTMState] = None) -> torch.Tensor:
        """
        Per-timestep state embeddings for a sequence [B, T, obs_dim] -> [B, T, F].

        Used for the recurrent replay: the caller splits the sequence at episode
        boundaries and reads the rows it needs (see rollout.replay_state_embeddings).
        """
        x = self.embed_obs(obs)
        if self.use_lstm:
            x, _ = self.lstm(x, lstm_state)
        return self.dropout(x)          # one mask over the sequence (train mode only)

    def forward(
        self,
        obs: torch.Tensor,                       # [B, T, obs_dim] or [B, obs_dim]
        action_feats: torch.Tensor,              # [B, n_slots, action_feat_dim]
        action_mask: Optional[torch.Tensor] = None,   # [B, n_slots] bool
        lstm_state: Optional[LSTMState] = None,
    ):
        """
        Returns (logits [B, n_slots], value [B], new_lstm_state).

        With a sequence input the heads read the LAST timestep (the decision point) and
        `new_lstm_state` is the state after the whole sequence.
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = self.embed_obs(obs)
        new_state = None
        if self.use_lstm:
            x, new_state = self.lstm(x, lstm_state)
        x = self.dropout(x)                       # one mask, shared by both heads
        s_emb = x[:, -1, :]                       # [B, F] — the decision point

        logits = self.logits_from(s_emb, self.embed_actions(action_feats))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(s_emb).squeeze(-1)
        return logits, value, new_state

    def get_value(self, obs: torch.Tensor, lstm_state: Optional[LSTMState] = None) -> torch.Tensor:
        """Value only — used for bootstrapping at the end of a rollout window."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = self.encode_sequence(obs, lstm_state)
        return self.value_head(x[:, -1, :]).squeeze(-1)

    def architecture_config(self) -> Dict[str, object]:
        """The constructor arguments, for embedding in checkpoints."""
        return {
            "obs_dim": self.obs_dim,
            "action_feat_dim": self.action_feat_dim,
            "n_slots": self.n_slots,
            "hidden": self.hidden,
            "use_lstm": self.use_lstm,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "value_hidden": self.value_hidden,
            "dropout": self.dropout_p,
        }


def build_recurrent_state(batch_size: int, net: "ActorCriticNetwork", device) -> Optional[LSTMState]:
    """Zero (h, c) for a batch, or None when the network is feedforward."""
    if not net.use_lstm:
        return None
    h = torch.zeros(net.lstm_layers, batch_size, net.lstm_hidden, device=device)
    return (h, h.clone())


def load_checkpoint_file(path: str, map_location):
    """
    Load a checkpoint file, preferring torch's safe loader.

    PPO snapshots contain only tensors and plain values, so `weights_only=True`
    works and cannot execute code. The DAgger imitation checkpoints, however, embed
    live Python objects (an `env.ActionEncoder`), which need `weights_only=False` —
    that unpickles arbitrary objects, so only load checkpoint files you trust. The
    fallback prints a note when it happens.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        print(f"[net] {path} contains non-tensor objects (likely a DAgger checkpoint); "
              f"loading with weights_only=False")
        return torch.load(path, map_location=map_location, weights_only=False)


def load_actor_critic(checkpoint_path: str, device: torch.device,
                      use_lstm: Optional[bool] = None) -> ActorCriticNetwork:
    """
    Build an `ActorCriticNetwork` from a PPO checkpoint file and load its weights.

    `use_lstm` overrides what the checkpoint says (useful for switching a feedforward
    checkpoint to a recurrent one, keeping the trunk).

    Legacy checkpoints are rejected with an explanation: the old PPO networks had a
    flat 1640-wide action head over the 244-dim observation, which is a different
    architecture (and different observation layout), so their weights cannot be
    transplanted. Train a new bootstrap (`src/ppo/bootstrap.py`) instead.
    """
    state = load_checkpoint_file(checkpoint_path, device)

    if "model_state" not in state:
        raise KeyError(
            f"{checkpoint_path} is not a PPO checkpoint (no 'model_state'). "
            "The DAgger imitation policies use a different architecture: use "
            "agents.load_agent_from_checkpoint for those."
        )
    if int(state.get("obs_dim", state.get("obs_size", 0))) == 244:
        raise RuntimeError(
            f"{checkpoint_path} is an OLD PPO checkpoint: it has the legacy 244-dim "
            "observation and a flat 1640-way action head. The pipeline now uses the "
            "ISMCTS observation layout (208) and per-action features, so this file "
            "cannot be loaded. Train a fresh bootstrap: "
            "python3 src/ppo/bootstrap.py"
        )

    net = ActorCriticNetwork(
        obs_dim=int(state.get("obs_dim", 208)),
        action_feat_dim=int(state.get("action_feat_dim", 84)),
        n_slots=int(state.get("n_slots", 64)),
        hidden=int(state.get("hidden", 256)),
        use_lstm=bool(state.get("use_lstm", True)) if use_lstm is None else use_lstm,
        lstm_hidden=int(state.get("lstm_hidden", 128)),
        lstm_layers=int(state.get("lstm_layers", 1)),
        value_hidden=int(state.get("value_hidden", 128)),
        dropout=float(state.get("dropout", 0.0)),
    ).to(device)
    net.load_state_dict(state["model_state"])
    net.eval()
    return net
