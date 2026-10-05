"""
PyTorch dataset for Cuarenta training.

Each sample is a Decision (see selfplay.py). Histories have variable length T,
so we left-pad/truncate to a fixed T_max in __getitem__ and also return `length`
(the number of real steps) so network.CuarentaNet can mask the padding out with
pack_padded_sequence. Padding content therefore does not matter; we use zeros.

LEFT padding (zeros at the beginning, real history at the end) keeps the last
step — the pending current decision — at index T_max-1, which is where the
network reads its state embedding from.

HISTORY DROPOUT (augmentation)
------------------------------
The network is used in two regimes:

  * at the search ROOT, with the player's real history (up to max_history_len
    steps of that player's own decisions), and
  * at INTERIOR nodes, where only a single "pending" step exists, because
    opponents' histories cannot be reconstructed from a sampled determinization.

Training only on full histories would leave the second regime out of
distribution, so with probability `history_dropout` we randomly truncate the
window to a prefix of random length (length 1 == "pending step only", the exact
interior-node regime). This is domain randomization for the history length: the
net learns to answer from the current observation alone, and gets better when
history is available. It also discourages the net from memorizing whole games.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from history import OBS_FLAT_DIM, ACT_FEAT_DIM
from network import N_ACTIONS, ACTION_FEAT_DIM


class CuarentaDataset(Dataset):
    def __init__(self, decisions, *, t_max: int = 32, history_dropout: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self.decisions = decisions
        self.t_max = t_max
        self.history_dropout = float(history_dropout)
        self.rng = rng or np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, i):
        d = self.decisions[i]

        # --- how many of the most recent steps do we keep? ---
        T = d.obs_seq.shape[0]
        keep = min(T, self.t_max)
        if self.history_dropout > 0 and self.rng.random() < self.history_dropout:
            # Keep a random number of the most recent steps, 1..keep. 1 = the
            # pending step only = what an interior search node looks like.
            keep = int(self.rng.integers(1, keep + 1))

        length = keep          # number of REAL (non-padding) steps
        pad = self.t_max - keep

        obs = np.zeros((self.t_max, OBS_FLAT_DIM), dtype=np.float32)
        act = np.zeros((self.t_max, ACT_FEAT_DIM), dtype=np.float32)
        am = np.zeros((self.t_max,), dtype=bool)

        # Take the last `keep` real steps, and place them at the END of the window
        # (left padding). The final step must stay last: it is the pending step.
        obs[pad:] = d.obs_seq[T - keep:]
        act[pad:] = d.act_seq[T - keep:]
        am[pad:] = d.act_mask[T - keep:]

        # Policy target: normalize the search's visit counts into a distribution
        # over the MAX_ACTIONS slots. Slot semantics are per-state here, which is
        # fine: the policy head scores the *features* of each slot.
        visits = d.visits.astype(np.float32)
        s = visits.sum()
        pi = visits / s if s > 0 else np.zeros_like(visits)

        return (
            torch.from_numpy(obs),                 # [T, OBS_FLAT_DIM]
            torch.from_numpy(act),                 # [T, ACT_FEAT_DIM]
            torch.from_numpy(am),                  # [T]
            torch.from_numpy(d.feats),             # [MAX_ACTIONS, ACTION_FEAT_DIM]
            torch.from_numpy(d.mask),              # [MAX_ACTIONS]
            torch.from_numpy(pi),                  # [MAX_ACTIONS]
            torch.tensor(d.outcome, dtype=torch.float32),
            torch.tensor(length, dtype=torch.int64),
        )
