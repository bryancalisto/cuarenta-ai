from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import List, Dict

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


class PointsReasons(StrEnum):
    CAIDA='caida'
    LIMPIA='limpia'


def has_reason(reasons, reason: "PointsReasons") -> bool:
    return reason in (reasons or [])


class OpponentMode(StrEnum):
    HEURISTIC="heuristic"
    SELFPLAY="selfplay"
    RANDOM="random"
