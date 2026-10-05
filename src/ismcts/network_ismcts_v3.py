"""
DEPRECATED SHIM — kept only so older scripts keep importing.

`network_ismcts_v3` was "net value at the leaf, length-1 observation from the ROOT
player's POV". That behaviour is now the default mode of the canonical search in
network_ismcts_1.NetworkISMCTS, with two changes:

  * leaf observations are built for the PLAYER TO MOVE (not the root player), so the
    value head is asked the question it was trained on ("chance that the mover's
    team wins") and the search converts it to root POV with a sign flip, and
  * priors are computed at every expanded node, not just the root.

There is no reason to use this module. Import NetworkISMCTS from network_ismcts_1.
"""

from network_ismcts_1 import NetworkISMCTS  # noqa: F401

__all__ = ["NetworkISMCTS"]
