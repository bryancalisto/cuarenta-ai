"""
Network-guided ISMCTS (B2-simple).

Same PUCT structure as rollout ISMCTS, but:
  - The policy prior at the root comes from CuarentaNet (one forward pass
    over the root player's real history).
  - The leaf value comes from CuarentaNet's value head, evaluated on a
    length-1 sequence built from the leaf's observation. No rollout.

What's intentionally simplified:
  - Interior nodes have no policy prior (only the root does).
  - The LSTM is not threaded through the tree — each leaf evaluation only
    sees the leaf's own observation, not the sim's action sequence.
  Both are follow-ups for B2-full. For now we want a fast, trainable search.

Usage:
    net = CuarentaNet()
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
    W         sum of values (from ROOT TEAM's POV, always)
    children  map: Action -> Node
    player    who is to move at this node (None until first expansion)
    team      TEAMS[player] if player is known, else None
    prior     P(s,a) — the network's prior for the action leading to this node
    """
    __slots__ = ("N", "W", "children", "player", "team", "prior")

    def __init__(self, player: Optional[int] = None, prior: float = 0.0):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player] if player is not None else None
        self.prior = prior


# ---------------------------------------------------------------------------
# State -> flat observation (for leaf value evaluation)
# ---------------------------------------------------------------------------

def flatten_state_for(player: int, state: CuarentaState) -> np.ndarray:
    """
    Build the OBS_FLAT_DIM vector for `player` from an arbitrary state.

    Mirrors env.observe() but skips action_features / action_mask, because
    those aren't needed for the value head. The ordering of fields here MUST
    match history.OBS_KEYS_{BITVEC,SCALAR,VEC4} exactly.
    """
    def bitvec(cards) -> np.ndarray:
        v = np.zeros(40, dtype=np.float32)
        for c in cards:
            v[c.index] = 1.0
        return v

    my_team = TEAMS[player]
    opp_team = 1 - my_team

    parts: List[np.ndarray] = [
        # bitvecs
        bitvec(state.hands[player]),
        bitvec(state.table),
        bitvec(state.cards_captured[my_team]),
        bitvec(state.cards_captured[opp_team]),
        bitvec([state.last_placed] if state.last_placed else []),
        # scalars
        np.array([state.scores[my_team]], dtype=np.float32),
        np.array([state.scores[opp_team]], dtype=np.float32),
        np.array([state.current_player], dtype=np.float32),
        np.array([len(state.deck)], dtype=np.float32),
        # vec4
        np.array(state.plays_this_deal, dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


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
        max_rollout_len: int = 200, # temporary for testing
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
        self.max_history_len = max_history_len
        self.max_rollout_len = max_rollout_len
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

        legal = legal_actions_from_state(self.env.state)

        # ---- Root priors from the net ----
        priors = self._root_priors(root_player, legal)

        # ---- Dirichlet noise for exploration during self-play ----
        if self.dirichlet_eps > 0 and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        # ---- Build root ----
        root = Node(player=root_player)
        for a, p in zip(legal, priors):
            root.children[a] = Node(player=None, prior=float(p))

        # ---- Build the determinizer ONCE ----
        determinizer = Determinizer(self.env, root_player)

        # ---- Run simulations ----
        for _ in range(self.n_iter):
            sampled = determinizer.sample(self.rng)
            self._simulate(root, sampled, root_team, root_player)

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
        obs_seq, act_seq, act_mask, _ = self.tracker.sequence_for(
            root_player, max_len=self.max_history_len
        )
        if obs_seq.shape[0] == 0:
            # No history yet: single-step input from the current observation.
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

    @torch.no_grad()
    def _net_value(self, state: CuarentaState, root_player: int, root_team: int) -> float:
        """
        Value at a leaf, from root_team's POV.

        If the leaf is terminal, use the true winner. Otherwise ask the net.
        The net outputs value from the *acting player's team* POV (that's how
        it's trained), so we invert it if the leaf's acting player is on the
        other team from root_player.
        """
        if state.done:
            return 1.0 if state.winner == root_team else -1.0

        # Build a length-1 sequence: just the leaf observation, no action.
        obs_flat = flatten_state_for(root_player, state)   # [OBS_FLAT_DIM]
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32,
                                device=self.device).view(1, 1, -1)
        act_t = torch.zeros(1, 1, N_ACTIONS, dtype=torch.float32, device=self.device)
        am_t  = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        # Value head doesn't use action features; pass zeros. Provide an
        # all-ones mask so the softmax path doesn't see -1e9 everywhere (not
        # that it affects the value, but it avoids any risk of NaN edge cases).
        feats_t = torch.zeros(1, N_ACTIONS, ACTION_FEAT_DIM,
                              dtype=torch.float32, device=self.device)
        mask_t  = torch.ones(1, N_ACTIONS, dtype=torch.bool, device=self.device)

        _, value, _ = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        raw = float(value[0])

        # Value head is trained from the acting player's team POV. Convert to
        # root_team POV.
        if TEAMS[state.current_player] != root_team:
            raw = -raw
        return raw
    

    def _leaf_value(self, state: CuarentaState, root_team: int) -> float:
        if state.done:
            return 1.0 if state.winner == root_team else -1.0
        return self._rollout(state, root_team)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: Node,
        state: CuarentaState,
        root_team: int,
        root_player: int,
    ) -> float:
        node = root
        path: List[Tuple[Node, Action]] = []

        # --- Selection / expansion ---
        while not state.done:
            legal = legal_actions_from_state(state)
            if not legal:
                break
            legal_set = set(legal)

            unexplored = [a for a in legal if a not in node.children]
            if unexplored:
                action = unexplored[self.rng.randrange(len(unexplored))]
                next_state, _, _, _ = step_state(state, action, self.rng)

                if action in node.children:
                    child = node.children[action]
                    child.player = next_state.current_player
                    child.team = TEAMS[child.player]
                else:
                    child = Node(player=next_state.current_player, prior=0.0)
                    node.children[action] = child

                path.append((node, action))
                state = next_state
                node = child
                break   # expand at most one node per sim

            action, child = self._puct_select(node, legal_set, root_team)
            next_state, _, _, _ = step_state(state, action, self.rng)

            # A pre-populated root child may not have its identity set yet — set it now.
            if child.player is None:
                child.player = next_state.current_player
                child.team = TEAMS[child.player]

            path.append((node, action))
            state = next_state
            node = child

        # --- Leaf value (net or terminal) ---
        # value = self._net_value(state, root_player, root_team)
        value = self._leaf_value(state, root_team)

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
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

        Q is stored from ROOT TEAM's POV. Flip for nodes where the acting
        player is on the other team.
        """
        assert node.team is not None, "PUCT called on unexpanded node"
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
            raise RuntimeError("_puct_select: no legal children in this determinization")
        return best
    
    def _rollout_policy(self, state: CuarentaState, rng: random.Random) -> Action:
        """Greedy rollout policy — same as the standalone rollout ISMCTS."""
        legal = legal_actions_from_state(state)
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

    def _rollout(self, state: CuarentaState, root_team: int) -> float:
        """Short rollout for leaf value."""
        s = clone_state(state)
        steps = 0
        while not s.done and steps < self.max_rollout_len:
            legal = legal_actions_from_state(s)
            if not legal:
                break
            a = self._rollout_policy(s, self.rng)
            s, _, _, _ = step_state(s, a, self.rng)
            steps += 1
        if s.done:
            return 1.0 if s.winner == root_team else -1.0
        return 0.0