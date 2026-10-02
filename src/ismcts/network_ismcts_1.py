"""
Network-guided ISMCTS (B1).

Same tree structure as the rollout ISMCTS, but the PUCT prior P(s,a) comes
from CuarentaNet's policy head (evaluated once at the root from the root
player's real history), instead of being implicitly uniform.

Value still comes from a greedy rollout. In a later step (B2) we'll replace
the rollout with a leaf-level LSTM evaluation, at which point the network's
value head gets used.

Usage:
    net = CuarentaNet()               # or load a checkpoint
    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(env, net, tracker, n_iter=100)
    visits = searcher.search()
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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A search-tree node.

    N         visit count
    W         sum of values (from ROOT TEAM's POV)
    children  map: Action -> Node
    player    who is to move at this node
    team      TEAMS[player]  (cached)
    prior     P(s,a) — the network's prior for this action, at the parent
    """
    __slots__ = ("N", "W", "children", "player", "team", "prior")

    def __init__(self, player: Optional[int] = None, prior: float = 1.0):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player] if player is not None else None
        self.prior = prior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_for_root_team(s: CuarentaState, root_team: int) -> float:
    """Terminal value from root team's POV. +1 win, -1 loss, 0 otherwise."""
    if not s.done:
        return 0.0
    return 1.0 if s.winner == root_team else -1.0


def _rollout_policy(state: CuarentaState, rng: random.Random, policy: str = "greedy") -> Action:
    """Move-selection policy for rollouts. See rollout ISMCTS for details."""
    legal = legal_actions_from_state(state)
    if policy == "random":
        return legal[rng.randrange(len(legal))]

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
            score += 5
        elif reasons == PointsReasons.LIMPIA:
            score += 5
        if score > best_score:
            best_score = score
            best = [a]
        elif score == best_score:
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
        rollout_policy: str = "greedy",
        max_rollout_len: int = 200,
        max_history_len: int = 32,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        rng: Optional[random.Random] = None,
        device: str = "cpu",
    ):
        self.env = env
        self.net = net
        self.tracker = tracker
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy
        self.max_rollout_len = max_rollout_len
        self.max_history_len = max_history_len
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or random.Random()
        self.device = device
        self.net.to(device)
        self.net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_player: Optional[int] = None) -> Dict[int, int]:
        """
        Returns visit counts keyed by index into env.legal_actions().
        """
        if root_player is None:
            root_player = self.env.state.current_player
        root_team = TEAMS[root_player]
        assert self.env.state.current_player == root_player, (
            "search must be called when it is root_player's turn"
        )

        # Legal actions at the root — the current player's actual options.
        legal = legal_actions_from_state(self.env.state)

        # ---- Compute root policy priors with one net forward ----
        priors = self._root_priors(root_player, legal)   # np.array [len(legal)]

        # ---- Add Dirichlet noise for exploration during self-play ----
        # (During eval, set dirichlet_eps=0 to disable.)
        if self.dirichlet_eps > 0 and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        # ---- Build root ----
        # We do NOT pre-populate root.children with priors. Instead, priors are
        # attached at expansion time inside _simulate. Reason: PUCT with N=0
        # children picks purely by prior, and if the prior is untrained/bad,
        # the search locks onto the wrong action. Instead, the first len(legal)
        # sims expand root actions uniformly at random (each with its correct
        # prior attached), then PUCT takes over.
        root = Node(player=root_player)
        root_priors: Dict[Action, float] = {a: float(p) for a, p in zip(legal, priors)}

        # ---- Run simulations ----
        determinizer = Determinizer(self.env, root_player)
        for _ in range(self.n_iter):
            sampled = determinizer.sample(self.rng)
            self._simulate(root, sampled, root_team, root_priors)

        # ---- Return visits (indices into `legal`) ----
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
        """
        One LSTM forward pass over the root player's real history -> priors
        over the root's legal actions.
        """
        # History sequence (numpy) for root player
        obs_seq, act_seq, act_mask, _ = self.tracker.sequence_for(
            root_player, max_len=self.max_history_len
        )
        if obs_seq.shape[0] == 0:
            # No history yet (start of game, first decision). Build a single-step
            # input from the *current* observation with no action taken.
            obs = self.env.observe(root_player)
            obs_seq = flatten_obs(obs)[None, :]           # [1, OBS_FLAT_DIM]
            act_seq = np.zeros((1, N_ACTIONS), dtype=np.int64)
            act_mask = np.zeros((1,), dtype=bool)

        # Current action features from the *real* state.
        obs_now = self.env.observe(root_player)
        feats_now = obs_now["action_features"]            # [32, 84]
        mask_now = obs_now["action_mask"]                 # [32]

        # To torch, add batch dim
        obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        am_t = torch.as_tensor(act_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        feats_t = torch.as_tensor(feats_now, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_now, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, _value, _ = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        # logits is [1, 32]. Slice out the legal slots (first len(legal)).
        logits = logits[0, :len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: Node,
        state: CuarentaState,
        root_team: int,
        root_priors: Dict[Action, float],
    ) -> float:
        node = root
        path: List[Tuple[Node, Action]] = []

        # --- Selection / expansion ---
        while not state.done:
            legal = legal_actions_from_state(state)
            if not legal:
                break
            legal_set = set(legal)

            # 1. Are there legal actions without a child yet? If so, expand one
            #    uniformly at random. Priors come from `root_priors` if this is
            #    the root, else 0.0 (interior nodes have no net prior for now).
            unexplored = [a for a in legal if a not in node.children]
            if unexplored:
                action = unexplored[self.rng.randrange(len(unexplored))]
                next_state, _, _, _ = step_state(state, action, self.rng)

                prior = root_priors.get(action, 0.0) if node is root else 0.0
                child = Node(player=next_state.current_player, prior=prior)
                node.children[action] = child

                path.append((node, action))
                state = next_state
                node = child
                break   # expand at most one node per sim, then roll out

            # 2. Every legal action has a child: PUCT-select one.
            action, child = self._puct_select(node, legal_set, root_team)
            next_state, _, _, _ = step_state(state, action, self.rng)
            path.append((node, action))
            state = next_state
            node = child

        # --- Leaf value via short rollout ---
        value = self._rollout(state, root_team)

        # --- Backup ---
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
        """
        PUCT selection among children whose actions are in `legal_set`.

        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

        Q is stored from ROOT TEAM's POV (same convention as rollout ISMCTS),
        so we invert it when the node's acting player is on the other team.
        """
        sqrt_n = math.sqrt(max(node.N, 1))
        best_u, best = -1e18, None

        for action, child in node.children.items():
            if action not in legal_set:
                continue
            # Q from node's acting-player POV
            if child.N == 0:
                q = 0.0
            else:
                mean_root_pov = child.W / child.N
                q = mean_root_pov if node.team == root_team else -mean_root_pov

            # PUCT exploration term uses the child's stored prior
            p = child.prior
            u = q + self.c_puct * p * sqrt_n / (1 + child.N)
            if u > best_u:
                best_u, best = u, (action, child)

        if best is None:
            # No legal children in this determinization; caller should treat
            # as leaf. Raise to surface the issue during development.
            raise RuntimeError("_puct_select: no legal children for current determinization")
        return best

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, state: CuarentaState, root_team: int) -> float:
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