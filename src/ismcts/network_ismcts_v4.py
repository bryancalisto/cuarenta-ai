"""
DEPRECATED SHIM — kept only so older scripts keep importing.

`network_ismcts_v4` threaded the LSTM hidden state down the tree, feeding
`(obs at that ply, action taken)` for every ply. It was an attempt to keep the
network in-distribution inside the tree, but:

  * the sequence it built mixed point-of-views (each ply's obs was built for the
    root player while the actual actor changed every ply), which the net never saw
    in training, and
  * it re-evaluated interior nodes on every simulation (`node.evaluated` was
    written but never read), silently discarding the cache it thought it had.

The canonical search (network_ismcts_1.NetworkISMCTS) instead evaluates each node
ONCE with a single "pending step" for the player to move, caches priors + value on
the node, and relies on history dropout during training so that history-free
evaluations are in-distribution. Import NetworkISMCTS from network_ismcts_1.
"""

from network_ismcts_1 import NetworkISMCTS  # noqa: F401

__all__ = ["NetworkISMCTS"]
