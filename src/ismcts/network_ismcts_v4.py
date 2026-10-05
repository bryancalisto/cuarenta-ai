"""
Network-guided ISMCTS v4: LSTM-threaded search with net value at leaves.

Design
------
Root:
  - One net forward over the root player's real history.
  - Produces: root priors + the initial LSTM hidden state.

Descend:
  - At each ply, we advance the LSTM one step using
    (obs_at_this_state_from_root_pov, action_taken).
  - When we arrive at an unevaluated node (a leaf), we run one more step
    consuming (obs_at_leaf_from_root_pov, no_action) and read policy+value.

Cache:
  - Each node stores `evaluated` and `value_root_pov`.
  - Once evaluated, its children exist (with priors) and value is cached.

Theoretical note
----------------
The net was trained on per-decision POV sequences. v4 uses root_player's POV
throughout the search, which is a coherent, single-channel observation
sequence (much closer to training than v3-simple's length-1 input). It's not
perfectly in-distribution, but it's the least-OOD choice compatible with
ISMCTS.

If empirically this isn't enough, the fix is to change the training format
to dense, single-POV sequences — a separate change.

API
---
    searcher = NetworkISMCTS(env, net, tracker, n_iter=100)
    visits = searcher.search()   # {slot_index: count}

    # Optional fallback for comparison / debugging.
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

from captures import score_play
from utils import PointsReasons
from env import MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("N", "W", "children", "player", "team", "prior",
                 "evaluated", "value_root_pov")

    def __init__(self, player: Optional[int] = None, prior: float = 0.0):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player] if player is not None else None
        self.prior = prior
        self.evaluated = False
        self.value_root_pov = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_from_state(player: int, state: CuarentaState) -> np.ndarray:
    """Flat observation for `player` at `state`. Must mirror history layout."""
    def bitvec(cards):
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


def _action_features_from_state(action: Action, state: CuarentaState) -> np.ndarray:
    """Per-slot action features for a single action, without needing the env."""
    f = np.zeros(ACTION_FEAT_DIM, dtype=np.float32)
    f[action.card.index] = 1.0
    if action.capture is not None:
        for c in action.capture:
            f[40 + c.index] = 1.0
        f[80] = 1.0
    _, reasons = score_play(
        played=action.card,
        capture_set=action.capture,
        table=state.table,
        last_placed=state.last_placed,
        non_caida_allowed=state.scores[TEAMS[state.current_player]]
                        < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
    )
    f[81] = 1.0 if reasons == PointsReasons.CAIDA else 0.0
    f[82] = 1.0 if reasons == PointsReasons.LIMPIA else 0.0
    return f


def _build_feats_and_mask(legal: List[Action], state: CuarentaState):
    """Build the [N_ACTIONS, ACTION_FEAT_DIM] tensor and [N_ACTIONS] mask."""
    feats = np.zeros((N_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((N_ACTIONS,), dtype=bool)
    for i, a in enumerate(legal):
        feats[i] = _action_features_from_state(a, state)
        mask[i] = True
    return feats, mask


def _rollout_policy(state: CuarentaState, rng: random.Random) -> Action:
    """Greedy rollout policy (only used in rollout fallback mode)."""
    legal = legal_actions_from_state(state)
    team = TEAMS[state.current_player]
    best_score = -1
    best: List[Action] = []
    for a in legal:
        pts, reasons = score_play(
            played=a.card, capture_set=a.capture,
            table=state.table, last_placed=state.last_placed,
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
        leaf_value_mode: str = "net",    # "net" | "rollout"
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

        # Root: net forward on the real history.
        root_priors, root_hidden = self._eval_root(root_player, legal)

        # Dirichlet noise for exploration during self-play.
        if self.dirichlet_eps > 0 and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            root_priors = (1 - self.dirichlet_eps) * root_priors + self.dirichlet_eps * noise

        root = Node(player=root_player)
        root.evaluated = True
        for a, p in zip(legal, root_priors):
            root.children[a] = Node(player=None, prior=float(p))

        determinizer = Determinizer(self.env, root_player)
        for _ in range(self.n_iter):
            sampled = determinizer.sample(self.rng)
            self._simulate(root, sampled, root_team, root_player, root_hidden)

        visits: Dict[int, int] = {}
        for i, a in enumerate(legal):
            child = root.children.get(a)
            visits[i] = child.N if child else 0
        return visits

    # ------------------------------------------------------------------
    # Net plumbing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_root(self, root_player: int, legal: List[Action]):
        """
        One full-history forward. Returns (priors over legal, initial hidden).
        """
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

        logits, _v, hidden = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        logits = logits[0, :len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, hidden

    @torch.no_grad()
    def _lstm_step(self, hidden, root_player: int, prev_state: CuarentaState, action: Action):
        """
        Advance the LSTM one step using (obs at prev_state from root POV, action).
        Returns new hidden state only.
        """
        obs_flat = _obs_from_state(root_player, prev_state)
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32,
                                device=self.device).view(1, 1, -1)

        legal = legal_actions_from_state(prev_state)
        slot = legal.index(action)
        act_one_hot = torch.zeros(1, 1, N_ACTIONS, dtype=torch.float32, device=self.device)
        act_one_hot[0, 0, slot] = 1.0
        am_t = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        dummy_feats = torch.zeros(1, N_ACTIONS, ACTION_FEAT_DIM,
                                  dtype=torch.float32, device=self.device)
        dummy_mask = torch.ones(1, N_ACTIONS, dtype=torch.bool, device=self.device)

        _, _v, new_hidden = self.net(obs_t, act_one_hot, am_t, dummy_feats, dummy_mask,
                                     hidden=hidden)
        return new_hidden

    @torch.no_grad()
    def _eval_at_node(self, hidden, root_player: int, state: CuarentaState, legal: List[Action]):
        """
        Run one LSTM step consuming (obs at state, no action), then read
        policy priors and value. Returns (value_acting, priors, new_hidden).

        `value_acting` is from the acting player's team POV.
        """
        obs_flat = _obs_from_state(root_player, state)
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32,
                                device=self.device).view(1, 1, -1)
        act_t = torch.zeros(1, 1, N_ACTIONS, dtype=torch.float32, device=self.device)
        am_t = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        feats, mask = _build_feats_and_mask(legal, state)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, value, new_hidden = self.net(obs_t, act_t, am_t, feats_t, mask_t,
                                             hidden=hidden)
        logits = logits[0, :len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return float(value[0]), probs, new_hidden

    def _rollout_value(self, state: CuarentaState, root_team: int) -> float:
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
        root_hidden,
    ) -> float:
        node = root
        hidden = root_hidden
        path: List[Tuple[Node, Action]] = []
        value_root_pov: Optional[float] = None

        while not state.done:
            legal = legal_actions_from_state(state)
            if not legal:
                break

            # Is there any legal action in this determinization that we haven't
            # yet created a child for? If so, evaluate this node (once) and stop.
            missing = [a for a in legal if a not in node.children]
            if missing:
                if self.leaf_value_mode == "net":
                    value_acting, priors, _ = self._eval_at_node(hidden, root_player, state, legal)
                    value_root_pov = (
                        value_acting
                        if TEAMS[state.current_player] == root_team
                        else -value_acting
                    )
                else:
                    value_root_pov = self._rollout_value(state, root_team)
                    _, priors, _ = self._eval_at_node(hidden, root_player, state, legal)

                # Attach priors only to actions we don't already have children for.
                for a, p in zip(legal, priors):
                    if a not in node.children:
                        node.children[a] = Node(player=None, prior=float(p))
                break  # stop the sim here

            # Every legal action already has a child: PUCT-select one and descend.
            action, _ = self._puct_select(node, set(legal), root_team)
            hidden = self._lstm_step(hidden, root_player, state, action)
            next_state, _, _, _ = step_state(state, action, self.rng)

            path.append((node, action))
            state = next_state
            node = node.children[action]
            node.player = next_state.current_player
            node.team = TEAMS[node.player]

        # Terminal (or explicit leaf break without value_root_pov set).
        if value_root_pov is None:
            value_root_pov = 1.0 if state.winner == root_team else -1.0

        for parent, action in path:
            c = parent.children[action]
            c.N += 1
            c.W += value_root_pov
            parent.N += 1
        return value_root_pov

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

    def _puct_select(self, node: Node, legal_set, root_team: int) -> Tuple[Action, Node]:
        assert node.team is not None, "PUCT on unexpanded node"
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
            u = q + self.c_puct * child.prior * sqrt_n / (1 + child.N)
            if u > best_u:
                best_u, best = u, (action, child)

        if best is None:
            raise RuntimeError("_puct_select: no legal children")
        return best