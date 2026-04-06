from __future__ import annotations

from typing import List, Dict

from matplotlib.table import table
import numpy as np
import torch

from card import Card

def obs_to_tensors(obs_list: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack a list of obs dicts into batched tensors."""
    keys = obs_list[0].keys()
    return {
        k: torch.tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32).to(device)
        for k in keys
    }

def sort_cards_by_rank(table: List[Card]) -> List[Card]:
    return sorted(table, key=lambda c: c.rank.value)