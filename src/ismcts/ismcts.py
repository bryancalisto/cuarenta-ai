from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import Action, CuarentaState, NUM_PLAYERS, TEAMS
from rules import clone_state, legal_actions_from_state, step_state
from determinize import Determinizer


class Node:
    """
    A search-tree node for the *network-free* ISMCTS.

    N         visit count
    W         sum of values, ALWAYS from the ROOT TEAM's point of view
    children  Action -> Node
    player    whose turn it is at this node
    team      TEAMS[player], cached
    """
    __slots__ = ("N", "W", "children", "player", "team")

    def __init__(self, player):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# helpers

def _value_for_root_team(s: CuarentaState, root_team: int) -> float:
    """
    Terminal value from the root team's POV: +1 win, -1 loss.

    If the state is NOT terminal we return 0.0, which literally means "this playout
    did not decide anything". Be careful with that number: 0.0 is a *bootstrap* value
    of "unknown", not an estimate that the game is a draw (Cuarenta has no draws).
    See `ISMCTS._rollout` for why silently returning 0.0 for a truncated rollout is
    one of the things that broke the net-guided version of this search.
    """
    if not s.done:
        return 0.0
    return 1.0 if s.winner == root_team else -1.0


def _rollout_policy(
    state: CuarentaState,
    rng: random.Random,
    *,
    policy: str = "greedy",
) -> Action:
    """
    Out-of-tree move policy used inside rollouts.
    - "random": uniform over legal actions
    - "greedy": pick the action with max immediate points, breaking ties randomly
    """
    legal = legal_actions_from_state(state)
    if policy == "random":
        return legal[rng.randrange(len(legal))]

    # greedy
    from captures import score_play
    from utils import PointsReasons, has_reason
    from env import MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS

    team = TEAMS[state.current_player]
    best_score = -1
    best: List[Action] = []
    for a in legal:
        pts, reasons = score_play(
            played=a.card,
            capture_set=a.capture,
            table=state.table,
            last_placed=state.last_placed,
            non_caida_allowed=state.scores[team] < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
        )
        score = pts
        # `reasons` is a list -> always test membership (see utils.has_reason).
        if has_reason(reasons, PointsReasons.CAIDA):
            score += 5   # value caída highly
        elif has_reason(reasons, PointsReasons.LIMPIA):
            score += 5
        if score > best_score:
            best_score = score
            best = [a]
        elif score == best_score:
            best.append(a)
    return best[rng.randrange(len(best))]


# search

class ISMCTS:
    """
    Plain Information Set MCTS: uniform determinization sampling, uniform priors,
    and *greedy rollouts* for leaf values. No neural network anywhere.

    This is kept as the classic baseline (and as an evaluation opponent for the
    net-guided search). The net-guided version lives in network_ismcts_1.py.

    Usage:
        searcher = ISMCTS(env, n_iter=400, rollout_policy="greedy")
        visits = searcher.search(root_player=env.state.current_player)
        best_action_idx = max(visits, key=visits.get)   # index into legal_actions()

    The search returns a dict {action_index_in_legal_actions: visit_count}.
    """

    def __init__(
        self,
        env,
        *,
        n_iter: int = 400,
        c_puct: float = 1.4,
        rollout_policy: str = "greedy",
        max_rollout_len: int = 512,
        rng: Optional[random.Random] = None,
    ):
        self.env = env
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy
        # 512 is comfortably longer than any real game (games run ~130-215 plies),
        # so rollouts should reach a terminal state. An old default of 200 could
        # truncate, and a truncated rollout used to be scored as 0.0.
        self.max_rollout_len = max_rollout_len
        self.rng = rng or random.Random()
        # Diagnostics: how many rollouts hit the length cap instead of terminating.
        self.truncated_rollouts = 0
        self.total_rollouts = 0

    # public

    def search(self, root_player: Optional[int] = None) -> Dict[int, int]:
        """
        Run ISMCTS from the env's current state, from `root_player`'s
        information set. Returns visit counts keyed by index into
        env.legal_actions().
        """
        if root_player is None:
            root_player = self.env.state.current_player
        root_team = TEAMS[root_player]

        # Sanity: the root player must be the one to move.
        assert self.env.state.current_player == root_player

        legal = legal_actions_from_state(self.env.state)
        root = Node(player=root_player)

        determinizer = Determinizer(self.env, root_player)
        for _ in range(self.n_iter):
            # 1. Sample a determinization consistent with root's info set
            sampled = determinizer.sample(self.rng)

            # 2. Run one simulation
            self._simulate(root, sampled, root_team)

        # Return visits over root actions (in the same order as legal_actions)
        visits: Dict[int, int] = {}
        for i, a in enumerate(legal):
            child = root.children.get(a)
            visits[i] = child.N if child else 0
        return visits

    # internals

    def _simulate(self, root: Node, state: CuarentaState, root_team: int) -> float:
        """
        One iteration: select → expand → rollout → backup.
        Returns the value (from root_team POV) that was backed up.
        """
        node = root
        path: List[Tuple[Node, Action]] = []

        # SELECTION + EXPANSION
        #
        # Note on tree sharing across determinizations: children are keyed by Action,
        # and an action that was legal in one sampled world may be illegal in the next
        # (the mover may hold different cards). Such "phantom" children are kept in the
        # tree (their stats are still useful at the information-set level) but they are
        # never traversed in a world where the action is illegal: `legal_set` filters
        # them out in `_uct_select`.
        while not state.done:
            legal = legal_actions_from_state(state)
            if not legal:
                break
            legal_set = set(legal)

            # Any action with no child yet in THIS determinization -> expand one.
            unexplored = [a for a in legal if a not in node.children]
            if unexplored:
                action = unexplored[self.rng.randrange(len(unexplored))]
                next_state, _, done, _ = step_state(state, action, self.rng)
                child = Node(player=next_state.current_player)
                node.children[action] = child
                path.append((node, action))
                state = next_state
                node = child
                break

            # All legal actions have children — UCT-select among them
            action, child = self._uct_select(node, legal_set, root_team)
            next_state, _, _, _ = step_state(state, action, self.rng)
            path.append((node, action))
            state = next_state
            node = child

        # ROLLOUT
        value = self._rollout(state, root_team)

        # BACKUP
        #
        # `node.N` = number of simulations that passed through that node, so the
        # backup only increments the CHILD of each edge. The usual `parent.N += 1`
        # looks harmless but double counts every intermediate node (once as a child,
        # once as a parent), which inflates its visit count and halves `W / N`
        # because W is added only once per simulation. Measured before the fix:
        # 300 simulations produced visit counts summing to ~595.
        for parent, action in path:
            child = parent.children[action]
            child.N += 1
            # W is stored in ROOT-TEAM POV for every node, whatever team acted there,
            # so `_uct_select` must flip the sign when the node's mover is on the
            # other team.
            child.W += value
        root.N += 1

        return value

    def _uct_select(self, node: Node, legal_set, root_team) -> Tuple[Action, Node]:
        """Select child with highest UCT score. Explores unvisited children."""
        log_n = math.log(node.N + 1)
        best_u, best = -1e18, None
        for action, child in node.children.items():
            if action not in legal_set:
                continue

            if child.N == 0:
                u = 1e17  # force exploration of unvisited children
            else:
                mean = child.W / child.N
                # Q from the point of view of whoever moves at this node
                node_utility = mean if node.team == root_team else -mean
                u = node_utility + self.c_puct * math.sqrt(log_n / child.N)
            if u > best_u:
                best_u, best = u, (action, child)

        if best is None:
            # No legal children — can happen if every stored child comes from a
            # determinization where the current player held different cards.
            raise RuntimeError("_uct_select: no legal children (should be unreachable)")

        return best

    def _rollout(self, state: CuarentaState, root_team: int) -> float:
        """
        Play out to terminal with the rollout policy.

        We keep a truncation counter because a truncated rollout is scored 0.0, which
        is a *neutral* value. If most rollouts truncate, every leaf looks equally good
        and the search has no signal at all — that was exactly what happened when the
        net-guided search used max_rollout_len=80 on games of 130+ plies.
        """
        s = clone_state(state)
        steps = 0
        while not s.done and steps < self.max_rollout_len:
            legal = legal_actions_from_state(s)
            if not legal:
                break
            action = _rollout_policy(s, self.rng, policy=self.rollout_policy)
            s, _, _, _ = step_state(s, action, self.rng)
            steps += 1

        self.total_rollouts += 1
        if not s.done:
            self.truncated_rollouts += 1
        return _value_for_root_team(s, root_team)
