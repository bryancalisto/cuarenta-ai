from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import Action, CuarentaState, NUM_PLAYERS, TEAMS
from rules import clone_state, legal_actions_from_state, step_state
from determinize import Determinizer



class Node:
    __slots__ = ("N", "W", "children", "player", "team")
    def __init__(self, player):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player]

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# helpers 

def _value_for_root_team(s: CuarentaState, root_team: int) -> float:
    """
    Terminal value from root team's POV. +1 win, -1 loss, 0 otherwise.
    You can tune this: e.g. use score difference for a smoother signal.
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
    In-tree / out-of-tree move policy for rollouts.
    - "random": uniform over legal actions
    - "greedy": pick the action with max immediate points
                (fall back to random among ties)
    """
    legal = legal_actions_from_state(state)
    if policy == "random":
        return legal[rng.randrange(len(legal))]

    # greedy
    from captures import score_play
    from utils import PointsReasons
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
        if reasons == PointsReasons.CAIDA:
            score += 5   # value caída highly
        elif reasons == PointsReasons.LIMPIA:
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
    Information Set MCTS with uniform determinization and rollout-based values.

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
        max_rollout_len: int = 200,
        rng: Optional[random.Random] = None,
    ):
        self.env = env
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy
        self.max_rollout_len = max_rollout_len
        self.rng = rng or random.Random()

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
        # (If you later allow searching at non-root nodes, relax this.)
        assert self.env.state.current_player == root_player

        legal = legal_actions_from_state(self.env.state)
        # print(f"[search] legal has {len(legal)} actions, current_player={self.env.state.current_player}")
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

        # SELECTION / EXPANSION
        while not state.done:
            # if node.untried:
            #     # Expand one untried action
            #     idx = self.rng.randrange(len(node.untried))
            #     action = node.untried.pop(idx)
            #     next_state, _, done, _ = step_state(state, action, self.rng)
            #     child = Node(
            #         player=next_state.current_player,
            #         untried=[] if done else legal_actions_from_state(next_state),
            #     )
            #     node.children[action] = child
            #     path.append((node, action))
            #     state = next_state
            #     node = child
            #     break
            legal = legal_actions_from_state(state)
            if not legal:
                break

            # Find children that are legal in THIS determinization
            legal_set = set(legal)

            # Any child not legal here is a "phantom" from another determinization.
            # We keep it in the tree (its stats are still useful at info-set level)
            # but we don't traverse it in this simulation.
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
        for parent, action in path:
            child = parent.children[action]
            child.N += 1
            # Value is from root team's POV, but each node's W is tracked
            # relative to the root team regardless of who acted there, which
            # is fine because we only ever use UCT at the root's team-agnostic
            # level... actually no — see note below.
            child.W += value
            parent.N += 1

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
                if node.team == root_team:
                    node_utility = mean
                else:
                    node_utility = -mean
                u = node_utility + self.c_puct * math.sqrt(log_n / child.N)
            if u > best_u:
                best_u, best = u, (action, child)
        
        if best is None:
            # No legal children — this can happen if all stored children come from
            # determinizations where the current player held different cards.
            # Treat as a leaf: pick a random legal action not yet expanded.
            # (Handled by the caller via `unexplored`; if we get here, fall back.)
            raise RuntimeError("_uct_select: no legal children (should be unreachable)")
    
        return best

    def _rollout(self, state: CuarentaState, root_team: int) -> float:
        """Play out to terminal with the rollout policy."""
        s = clone_state(state)
        steps = 0
        while not s.done and steps < self.max_rollout_len:
            legal = legal_actions_from_state(s)
            if not legal:
                break
            action = _rollout_policy(s, self.rng, policy=self.rollout_policy)
            s, _, _, _ = step_state(s, action, self.rng)
            steps += 1
        return _value_for_root_team(s, root_team)

