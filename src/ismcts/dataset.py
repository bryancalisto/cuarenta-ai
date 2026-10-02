"""
PyTorch dataset for Cuarenta training.

Each sample is a Decision (see selfplay.py). Because histories have variable
length T, we pad/truncate to a fixed T_max in the collate function.

We pad LEFT (zeros at the beginning, real history at the end) so that the
last step (the current decision) is always at index T_max-1. This matters
because the network reads s_emb from the last LSTM step.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from history import OBS_FLAT_DIM
from network import N_ACTIONS, ACTION_FEAT_DIM


class CuarentaDataset(Dataset):
    def __init__(self, decisions, *, t_max: int = 32):
        self.decisions = decisions
        self.t_max = t_max

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, i):
        d = self.decisions[i]

        # Left-pad or truncate the history to T = t_max
        T = d.obs_seq.shape[0]
        if T >= self.t_max:
            # Keep the LAST t_max steps (recent history matters most)
            obs = d.obs_seq[-self.t_max:]
            act = d.act_seq[-self.t_max:]
            am  = d.act_mask[-self.t_max:]
        else:
            pad = self.t_max - T
            obs = np.zeros((self.t_max, OBS_FLAT_DIM), dtype=np.float32)
            act = np.zeros((self.t_max, N_ACTIONS), dtype=np.float32)
            am  = np.zeros((self.t_max,), dtype=bool)
            obs[pad:] = d.obs_seq
            act[pad:] = d.act_seq
            am[pad:]  = d.act_mask

        # Policy target: normalize visits into a probability distribution.
        visits = d.visits.astype(np.float32)
        s = visits.sum()
        pi = visits / s if s > 0 else np.zeros_like(visits)

        return (
            torch.from_numpy(obs),                 # [T, OBS_FLAT_DIM]
            torch.from_numpy(act),                 # [T, N_ACTIONS]
            torch.from_numpy(am),                  # [T]
            torch.from_numpy(d.feats),             # [N_ACTIONS, ACTION_FEAT_DIM]
            torch.from_numpy(d.mask),              # [N_ACTIONS]
            torch.from_numpy(pi),                  # [N_ACTIONS]
            torch.tensor(d.outcome, dtype=torch.float32),
        )