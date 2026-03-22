from __future__ import annotations

from typing import List, Dict

import numpy as np
import torch

def obs_to_tensors(obs_list: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack a list of obs dicts into batched tensors."""
    keys = obs_list[0].keys()
    return {
        k: torch.tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32).to(device)
        for k in keys
    }