"""
The observation-flattening contract, and nothing else.

`src/ismcts/history.py` is mostly an encoder that stacks a game into a padded sequence for
its history LSTM. ReBeL does not have one -- its network reads a single position (the
observation plus the belief marginals), because the belief state already summarises
everything that happened -- so this file only keeps the part the game core needs:
`env.flat_obs_for()` imports `flatten_obs` from here to build the 208-dim vector from a
state.

The key order is the shared one (`src/ismcts/history.py` and `src/ppo/utils.py` declare
the same list). It is duplicated here rather than imported so that `src/rebel` stays
self-contained like the other pipelines, and because this is a compatibility surface:
changing it would invalidate every trained checkpoint in the repository.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

_OBS_KEYS_BITVEC = [
    "my_hand",              # 40
    "table",                # 40
    "my_team_captures",     # 40
    "opp_team_captures",    # 40
    "last_placed",          # 40
]
_OBS_KEYS_SCALAR = [
    "my_team_score",        # 1
    "opp_team_score",       # 1
    "current_player",       # 1
    "deck_remaining",       # 1
]
_OBS_KEYS_VEC4 = [
    "plays_this_deal",      # 4
]

OBS_FLAT_DIM = (40 * len(_OBS_KEYS_BITVEC) + len(_OBS_KEYS_SCALAR)
                + 4 * len(_OBS_KEYS_VEC4))       # = 208
ACT_FEAT_DIM = 84


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Collapse the env's dict observation into one 1-D float32 vector.

    "action_features" and "action_mask" are excluded: they travel on the per-action path
    of the network, not in the state encoding.
    """
    parts: List[np.ndarray] = []
    for key in _OBS_KEYS_BITVEC:
        parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
    for key in _OBS_KEYS_SCALAR:
        parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
    for key in _OBS_KEYS_VEC4:
        parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
    return np.concatenate(parts, axis=0)
