"""
Network-guided ISMCTS v3: net value at leaves (no rollout).

Same tree structure, determinization, and PUCT as network_ismcts_1. The ONLY
change is the leaf value: instead of a rollout, we ask the net's value head.

How the leaf value is computed
------------------------------
At a leaf state, we build a length-1 observation from the *root player's*
POV at that state, run it through the LSTM, and read the value head.

Why root player's POV? The value head was trained on histories from the
acting player's POV, but the search is rooted at a specific player. Using
the root player's POV throughout keeps the value semantically consistent
(always "expected outcome for root team").

There IS a distribution shift: the net was trained on full histories, but at
a leaf we feed only the leaf's observation (no history). This is the
simplification that makes v3 cheap. It may cost some accuracy.

POV flip
--------
The net's value head outputs "value from the acting player's team POV".
The acting player at a leaf may be on the other team from root_player.
We flip the sign accordingly.

Terminal states
---------------
If the leaf is terminal (game over), we use the true winner, not the net.
That's free signal and exactly right.

API
---
    searcher = NetworkISMCTS(env, net, tracker, n_iter=100)
    visits = searcher.search()   # {slot_index: count}

    # Optional: pick leaf_value_mode to fall back to rollout for debugging.
    searcher = NetworkISMCTS(..., leaf_value_mode="rollout")
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import Action, CuarentaState, NUM_PLAYERS, TEAMS, CuarentaEnv
from rules import clone_state, legal_actions_from_state, step_state
from determinize import Determinizer
from history import HistoryTracker, flatten_obs, OBS_FLAT_DIM
from network import CuarentaNet, N_ACTIONS, ACTION_FEAT_DIM

# Import the greedy policy so we can fall back to it if leaf_value_mode="rollout".
from captures import score_play
from utils import PointsReasons
from env import MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("N", "W", "children", "player", "team", "prior")

    def __init__(self, player: Optional[int] = None, prior: float = 0.0):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player] if player is not None else None
        self.prior = prior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_terminal(state: CuarentaState, root_team: int) -> float:
    return 1.0 if state.winner == root_team else -1.0


def _flatten_state_for(player: int, state: CuarentaState) -> np.ndarray:
    """
    Flat observation for `player` at `state`.
    Must mirror history._OBS_KEYS_* exactly.
    """
    def bitvec(cards) -> np.ndarray:
        v = np.zeros(40, dtype=np.float32)
        for c in cards:
            v[c.index] = 1.0
        return v

    my_team = TEAMS[player]
    opp_team = 1 - my_team
    parts = [
        bitvec(state.hands[player]),
        bitvec(state.table),
        bitvec(state.cards_captured[my_team]),
        bitvec(state.cards_captured[opp_team]),
        bitvec([state.last_placed] if state.last_placed else []),
        np.array([state.scores[my_team]], dtype=np.float32),
        np.array([state.scores[opp_team]], dtype=np.float32),
        np.array([state.current_player], dtype=np.float32),
        np.array([len(state.deck)], dtype=np.float32),
        np.array(state.plays_this_deal, dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def _rollout_policy(state: CuarentaState, rng: random.Random) -> Action:
    """Greedy rollout policy — for the rollout fallback mode."""
    legal = legal_actions_from_state(state)
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
        s = pts + (5 if reasons in (PointsReasons.CAIDA, PointsReasons.LIMPIA) else 0)
        if s > best_score:
            best_score = s
            best = [a]
        elif s == best_score:
            best.append(a)
    return best[rng.randrange(len(best))]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class NetworkISMCTS:
    def __init__(
        self,
        env: CuarentaEnv,
        net: CuarentaNet,
        tracker: HistoryTracker,
        *,
        n_iter: int = 100,
        c_puct: float = 1.5,
        max_history_len: int = 32,
        max_rollout_len: int = 200,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        leaf_value_mode: str = "net",   # "net" or "rollout"
        rng: Optional[random.Random] = None,
        device: str = "cpu",
    ):
        assert leaf_value_mode in ("net", "rollout")
        self.env = env
        self.net = net
        self.tracker = tracker
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.max_history_len = max_history_len
        self.max_rollout_len = max_rollout_len
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.leaf_value_mode = leaf_value_mode
        self.rng = rng or random.Random()
        self.device = device
        self.net.to(device)
        self.net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_player: Optional[int] = None) -> Dict[int, int]:
        if root_player is None:
            root_player = self.env.state.current_player
        root_team = TEAMS[root_player]
        assert self.env.state.current_player == root_player

        legal = legal_actions_from_state(self.env.state)

        # Root priors from the net (same as network_ismcts_1)
        priors = self._root_priors(root_player, legal)

        # Dirichlet noise at the root
        if self.dirichlet_eps > 0 and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        root = Node(player=root_player)
        root_priors_map: Dict[Action, float] = {a: float(p) for a, p in zip(legal, priors)}

        determinizer = Determinizer(self.env, root_player)
        for _ in range(self.n_iter):
            sampled = determinizer.sample(self.rng)
            self._simulate(root, sampled, root_team, root_player, root_priors_map)

        visits: Dict[int, int] = {}
        for i, a in enumerate(legal):
            child = root.children.get(a)
            visits[i] = child.N if child else 0
        return visits

    # ------------------------------------------------------------------
    # Root prior computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _root_priors(self, root_player: int, legal: List[Action]) -> np.ndarray:
        obs_seq, act_seq, act_mask, _ = self.tracker.sequence_for(
            root_player, max_len=self.max_history_len
        )
        if obs_seq.shape[0] == 0:
            obs = self.env.observe(root_player)
            obs_seq = flatten_obs(obs)[None, :]
            act_seq = np.zeros((1, N_ACTIONS), dtype=np.int64)
            act_mask = np.zeros((1,), dtype=bool)

        obs_now = self.env.observe(root_player)
        feats_now = obs_now["action_features"]
        mask_now = obs_now["action_mask"]

        obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        am_t = torch.as_tensor(act_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        feats_t = torch.as_tensor(feats_now, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_now, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, _, _ = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        logits = logits[0, :len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Leaf value
    # ------------------------------------------------------------------

    def _leaf_value(self, state: CuarentaState, root_player: int, root_team: int) -> float:
        if state.done:
            return _value_terminal(state, root_team)
        if self.leaf_value_mode == "net":
            return self._net_value(state, root_player, root_team)
        else:
            return self._rollout_value(state, root_team)

    @torch.no_grad()
    def _net_value(self, state: CuarentaState, root_player: int, root_team: int) -> float:
        """
        Net value at a leaf, from root_team's POV.

        Feeds a length-1 sequence containing the root player's POV of the
        leaf state. This is the v3 simplification (no history threading).
        """
        obs_flat = _flatten_state_for(root_player, state)          # [OBS_FLAT_DIM]
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32,
                                device=self.device).view(1, 1, -1)
        act_t = torch.zeros(1, 1, N_ACTIONS, dtype=torch.float32, device=self.device)
        am_t = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        # The value head doesn't use action features; pass zeros.
        feats_t = torch.zeros(1, N_ACTIONS, ACTION_FEAT_DIM,
                              dtype=torch.float32, device=self.device)
        mask_t = torch.ones(1, N_ACTIONS, dtype=torch.bool, device=self.device)

        _, value, _ = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        raw = float(value[0])

        # Value head is from the acting player's team POV (trained that way).
        # Flip if the leaf's actor is on the other team.
        if TEAMS[state.current_player] != root_team:
            raw = -raw
        return raw

    def _rollout_value(self, state: CuarentaState, root_team: int) -> float:
        """Fallback: greedy rollout to a terminal or truncation."""
        s = clone_state(state)
        steps = 0
        while not s.done and steps < self.max_rollout_len:
            legal = legal_actions_from_state(s)
            if not legal:
                break
            a = _rollout_policy(s, self.rng)
            s, _, _, _ = step_state(s, a, self.rng)
            steps += 1
        if s.done:
            return 1.0 if s.winner == root_team else -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: Node,
        state: CuarentaState,
        root_team: int,
        root_player: int,
        root_priors_map: Dict[Action, float],
    ) -> float:
        node = root
        path: List[Tuple[Node, Action]] = []

        # Descend, expand at most one node, then stop.
        while not state.done:
            legal = legal_actions_from_state(state)
            if not legal:
                break
            legal_set = set(legal)

            unexplored = [a for a in legal if a not in node.children]
            if unexplored:
                action = unexplored[self.rng.randrange(len(unexplored))]
                next_state, _, _, _ = step_state(state, action, self.rng)

                prior = root_priors_map.get(action, 0.0) if node is root else 0.0
                child = Node(player=next_state.current_player, prior=prior)
                node.children[action] = child

                path.append((node, action))
                state = next_state
                node = child
                break

            action, child = self._puct_select(node, legal_set, root_team)
            next_state, _, _, _ = step_state(state, action, self.rng)
            path.append((node, action))
            state = next_state
            node = child

        # Leaf value
        value = self._leaf_value(state, root_player, root_team)

        # Backup
        for parent, action in path:
            c = parent.children[action]
            c.N += 1
            c.W += value
            parent.N += 1
        return value

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

    def _puct_select(self, node: Node, legal_set, root_team: int) -> Tuple[Action, Node]:
        assert node.team is not None
        sqrt_n = math.sqrt(max(node.N, 1))
        best_u, best = -1e18, None

        for action, child in node.children.items():
            if action not in legal_set:
                continue
            if child.N == 0:
                q = 0.0
            else:
                mean_root_pov = child.W / child.N
                q = mean_root_pov if node.team == root_team else -mean_root_pov
            p = child.prior
            u = q + self.c_puct * p * sqrt_n / (1 + child.N)
            if u > best_u:
                best_u, best = u, (action, child)

        if best is None:
            raise RuntimeError("_puct_select: no legal children")
        return best