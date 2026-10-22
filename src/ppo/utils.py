"""
Small shared helpers for the Cuarenta code base.

Kept deliberately tiny: the game rules live in `captures.py` / `env.py`, the PPO
learning code lives in `src/ppo/`, and the search-based code lives in `src/ismcts/`.
This module is the shared corner of both pipelines (the `has_reason` helper below
exists because both of them tripped over the same bug).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Dict, List, Tuple

import numpy as np
import torch

from card import Card


# ---------------------------------------------------------------------------
# Observation layouts (the networks' input contract)
# ---------------------------------------------------------------------------
#
# `CuarentaEnv.observe(player)` returns a dict of numpy arrays. Two flat layouts are
# declared here, because two generations of models consume different subsets:
#
#   OBS_KEYS_ISMCTS (208) — the layout used by the ISMCTS pipeline
#       (src/ismcts/history.py) and by PPO. Order matters and is shared, so a
#       network trained by one pipeline reads exactly the same tensor in the other.
#
#   OBS_KEYS_LEGACY (244) — the historical alphabetical order used by the DAgger
#       imitation policy and by the deployed ONNX model. It additionally carries
#       `played_this_deal` (the union of both capture piles, i.e. redundant public
#       information) and lacks `plays_this_deal`.
#
# Keeping both declared means the imitation/ONNX line keeps working unchanged while
# PPO moves to the ISMCTS representation.
OBS_KEYS_ISMCTS: Tuple[str, ...] = (
    "my_hand",              # 40
    "table",                # 40
    "my_team_captures",     # 40
    "opp_team_captures",    # 40
    "last_placed",          # 40  (card bitvec; the caída reference card)
    "my_team_score",        # 1
    "opp_team_score",       # 1
    "current_player",       # 1
    "deck_remaining",       # 1
    "plays_this_deal",      # 4
)

OBS_KEYS_LEGACY: Tuple[str, ...] = (
    "current_player",       # 1
    "deck_remaining",       # 1
    "last_placed",          # 40
    "my_hand",              # 40
    "my_team_captures",     # 40
    "my_team_score",        # 1
    "opp_team_captures",    # 40
    "opp_team_score",       # 1
    "played_this_deal",     # 40
    "table",                # 40
)

OBS_SIZE_ISMCTS = 40 * 5 + 4 + 4      # 208
OBS_SIZE_LEGACY = 40 * 6 + 4          # 244

# Backwards-compatible aliases: the PPO-facing default is the ISMCTS layout.
OBS_KEYS = OBS_KEYS_ISMCTS
OBS_SIZE = OBS_SIZE_ISMCTS


def flatten_obs(obs: Dict[str, np.ndarray], keys: Tuple[str, ...] = OBS_KEYS_ISMCTS) -> np.ndarray:
    """Flatten an observation dict into the requested layout's order."""
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys]
    ).astype(np.float32)


def flatten_obs_batch(obs_list: List[Dict[str, np.ndarray]],
                      keys: Tuple[str, ...] = OBS_KEYS_ISMCTS) -> np.ndarray:
    """Stack a list of observation dicts into a [B, len(layout)] array."""
    return np.stack([flatten_obs(o, keys) for o in obs_list])


def check_obs_layout(obs: Dict[str, np.ndarray], keys: Tuple[str, ...] = OBS_KEYS_ISMCTS) -> None:
    """
    Assert an observation dict contains everything a layout needs (it may contain
    more: the env is a superset so that both layouts stay valid).

    Called by the trainers at startup and by the tests, so a change to the env's
    observation keys cannot silently invalidate trained checkpoints (or the ONNX
    model, which pins the legacy layout).
    """
    missing = [k for k in keys if k not in obs]
    if missing:
        raise ValueError(
            f"env observation is missing keys required by this layout: {missing}. "
            "If the env changed deliberately, update the layouts in utils.py, "
            "retrain, and re-export the ONNX model."
        )
    expected = OBS_SIZE_ISMCTS if keys is OBS_KEYS_ISMCTS else OBS_SIZE_LEGACY
    flat_size = int(sum(np.asarray(obs[k]).size for k in keys))
    if flat_size != expected:
        raise ValueError(f"observation size changed: {flat_size} != {expected}")


def obs_to_tensors(obs_list: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack a list of obs dicts into batched tensors (one tensor per obs key)."""
    keys = obs_list[0].keys()
    return {
        k: torch.tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32).to(device)
        for k in keys
    }


def sort_cards_by_rank(table: List[Card]) -> List[Card]:
    return sorted(table, key=lambda c: c.rank.value)


class PointsReasons(StrEnum):
    CAIDA = 'caida'
    LIMPIA = 'limpia'


class OpponentMode(StrEnum):
    HEURISTIC = "heuristic"
    SELFPLAY = "selfplay"
    RANDOM = "random"


def has_reason(reasons, reason: "PointsReasons") -> bool:
    return reason in (reasons or [])
