"""
Network-guided ISMCTS — THE canonical search used by self-play, training and eval.

History note: this file was "network_ismcts_1" (the first net-guided variant, which
used *rollouts* for leaf values and the net only for root priors). v3 and v4 were
later experiments; both are now thin shims that re-export this class. The bugs this
version fixes, and why they mattered:

 1. LEAF VALUES WERE ZERO FOR MOST OF THE GAME.
    The old rollout mode valued leaves with a greedy rollout capped at
    `max_rollout_len=80` while games run 130-215 plies, and a truncated rollout was
    scored 0.0. So for the first half of every game *every* simulation backed up 0.0,
    the tree had no value signal at all, and the visit distribution was decided by
    expansion order + Dirichlet noise instead of by play quality. Fine-tuning on those
    visit targets is what wrecked the net against the heuristic.
    Now the value head evaluates the leaf (one net forward), which is also far
    cheaper than an 80-ply rollout, so we can afford far more simulations.

 2. INTERIOR NODES HAD NO PRIORS.
    v1 attached net priors only at the root (`prior = root_priors.get(action, 0.0)
    if node is root else 0.0`), so below the root the PUCT policy term was always
    zero and selection degenerated. Now every expanded node gets an evaluation, so
    its children carry real priors.

    The price is that interior evaluations cannot use real histories: in a sampled
    determinization we know an opponent's hand, but not what that opponent has seen.
    So interior nodes are evaluated from a single "pending step" (the current state
    from the mover's point of view, no action history), while the ROOT uses the real
    history. Training compensates with history dropout (dataset.py) so the net is
    good in both regimes.

 3. VALUES ARE NOW CONSISTENT WITH TRAINING.
    Evaluations are always built for the player to move: `my_hand`, scores, etc. are
    that player's POV, and the value head's output means "chance that the mover's
    team wins" — exactly what it was trained on. The search converts to root-team
    POV with a sign flip. For the mover that is not the root, the observation comes
    from the sampled determinization (a possible world), which is the standard
    ISMCTS trade: sampled worlds average out to the information-set value.

 4. TRUNCATED ROLLOUTS ARE NOT SILENTLY TREATED AS DRAWS.
    With `leaf_value_mode="rollout"` (a debugging/baseline mode) a rollout that hits
    the length cap is bootstrapped with the net's value instead of 0.0.

VALUE TARGET: the search MIRRORS the net's units
------------------------------------------------
`value_target` is read from `net.value_target` (see network.py for the two targets and why
they exist), because the search's terminal value, its leaf value and what the value head
predicts must all be the same quantity. Two modes:

  "win" (default)
      Every backup is a game result: a terminal node is worth +/-1 and nothing else, and
      per-step rewards are IGNORED. The search maximizes P(win), and Q is symmetric
      around 0 ("unvisited child = 0 = even"), which is what `c_puct` is calibrated
      against. This is plain AlphaZero.

  "return"
      The search accumulates the env's DENSE reward down every simulation path and adds
      the net's value at the leaf, i.e. it estimates the shaped return-to-go

          G = sum over plies of sign(actor team) * reward / RETURN_VALUE_SCALE

      with sign + for the root's team and - for the opponent's. Two consequences deserve
      to be stated out loud:
        * The sign flip is borrowed from the zero-sum case as a SHAPING choice. The env's
          reward is not zero-sum (both teams are paid for captures), so this optimizes a
          surrogate of the payoff. PPO does the same to its critic
          (`rollout.shaped_reward`); here it steers the POLICY, so it is a stronger
          assumption and `c_puct`/sim budget may want retuning.
        * Q is then a return: always positive and far from 0, so an unvisited child must
          NOT be scored as 0 (that reads as "terrible" and kills exploration).
          `_puct_select` centers Q on the node's own value, which makes a fresh child
          exactly neutral -- the same thing 0 means in win mode.

Sign / bookkeeping conventions (used everywhere below)
-----------------------------------------------------
  * `node.W` is the SUM of values in ROOT-TEAM POV, for every node, whoever moved
    there. `_puct_select` converts it to MOVER POV before comparing children.
  * `node.N` counts simulations that passed through the edge into that node, i.e.
    it is N(s,a) of the parent's action.
  * `node.value_mover_pov` caches the net's value for the player to move at that
    node (state value, not action value).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import (
    Action, CuarentaState, TEAMS, CuarentaEnv,
    action_features_matrix, flat_obs_for, MAX_ACTIONS,
    MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
)
from rules import clone_state, legal_actions_from_state, step_state
from determinize import Determinizer
from history import HistoryTracker, ACT_FEAT_DIM
from network import CuarentaNet, N_ACTIONS, ACTION_FEAT_DIM, RETURN_VALUE_SCALE


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A search-tree node.

    N                visit count = N(parent_action)
    W                sum of values in ROOT-TEAM POV
    children         Action -> Node. Keyed by the Action itself, so the same node
                     can be shared across determinizations.
    player           whose turn it is here (None until first descended into; see
                     the deal-boundary note in `_simulate`)
    team             TEAMS[player]
    prior            P(s,a) copied from the parent's evaluation when this child was
                     created (the network's prior for the action that leads here)
    priors           this node's OWN evaluation: {action: P(a)} for the actions that
                     were legal in the world used to evaluate it
    value_mover_pov  cached net value for the player to move here
    evaluated        has `priors` / `value_mover_pov` been filled?
    """
    __slots__ = ("N", "W", "children", "player", "team", "prior",
                 "priors", "value_mover_pov", "evaluated")

    def __init__(self, player: Optional[int] = None, prior: float = 0.0):
        self.N = 0
        self.W = 0.0
        self.children: Dict[Action, "Node"] = {}
        self.player = player
        self.team = TEAMS[player] if player is not None else None
        self.prior = prior
        self.priors: Optional[Dict[Action, float]] = None
        self.value_mover_pov: float = 0.0
        self.evaluated = False

    def prior_for(self, action: Action) -> float:
        """
        Prior for `action` at this node.

        A node is evaluated once, in one sampled world. A later world may make an
        action legal that was illegal at evaluation time (the mover's hand is
        sampled!). Those actions have no network prior; give them the mean of the
        known priors so they are neither favoured nor starved. With uniform-ish
        fresh priors PUCT will try them because their Q is unknown (0.0).
        """
        if not self.priors:
            return 1.0
        p = self.priors.get(action)
        if p is not None:
            return p
        return sum(self.priors.values()) / max(len(self.priors), 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terminal_value_root_pov(state: CuarentaState, root_team: int) -> float:
    """
    Value of a finished game in WIN mode: +1 if the root team won, -1 otherwise.
    Cuarenta has no draws.

    (In return mode a finished game is worth 0: there are no future rewards left, and the
    reward of the final ply is counted as that edge's reward. See
    `NetworkISMCTS._terminal_value`.)
    """
    return 1.0 if state.winner == root_team else -1.0


def _rollout_policy(state: CuarentaState, rng: random.Random) -> Action:
    """Greedy rollout policy, used only in `leaf_value_mode="rollout"`."""
    from captures import score_play
    from utils import PointsReasons, has_reason

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
        # `reasons` is a list: membership test, never `==`. (This exact `==` bug is
        # what made the old rollout policy ignore caídas/limpias.)
        s = pts + (5 if (has_reason(reasons, PointsReasons.CAIDA)
                         or has_reason(reasons, PointsReasons.LIMPIA)) else 0)
        if s > best_score:
            best_score, best = s, [a]
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
        n_iter: int = 200,
        c_puct: float = 1.5,
        max_history_len: int = 32,
        leaf_value_mode: str = "net",      # "net" (default) | "rollout" (debug)
        max_rollout_len: int = 512,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,       # 0.0 disables exploration noise (eval)
        rng: Optional[random.Random] = None,
        device: str = "cpu",
        value_target: Optional[str] = None,   # None = take it from the net (see below)
    ):
        assert leaf_value_mode in ("net", "rollout")
        self.env = env
        self.net = net
        self.tracker = tracker
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.max_history_len = max_history_len
        self.leaf_value_mode = leaf_value_mode
        self.max_rollout_len = max_rollout_len
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or random.Random()
        self.device = device
        # The search's units follow the value head's (see the VALUE TARGET section in the
        # module docstring). Taken from the net by default so that a return-mode
        # checkpoint is automatically searched with return-mode semantics, in every
        # evaluator, without threading a flag through each of them.
        self.value_target = value_target or getattr(net, "value_target", "win")
        assert self.value_target in ("win", "return"), self.value_target
        self.value_scale = (float(getattr(net, "return_value_scale", RETURN_VALUE_SCALE))
                            if self.value_target == "return" else 1.0)
        self.net.to(device)
        self.net.eval()

        # ---- diagnostics, useful when learning/debugging ----
        self.stats = {"sims": 0, "evals": 0, "rollouts": 0, "truncated_rollouts": 0}
        self.last_root_priors: Optional[np.ndarray] = None        # CLEAN net prior
        self.last_root_priors_noisy: Optional[np.ndarray] = None  # after Dirichlet
        self.last_root_value: float = 0.0                         # root team's POV

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_player: Optional[int] = None) -> Dict[int, int]:
        """
        Run `n_iter` simulations from the env's current state and return
        {index into env.legal_actions(): visit count}.

        A fresh tree is built for every decision (no tree reuse between plies).
        Reusing a tree in ISMCTS would require carrying information sets across
        moves, which does not compose with fresh determinizations; a fresh tree is
        the simple, correct choice.
        """
        if root_player is None:
            root_player = self.env.state.current_player
        root_team = TEAMS[root_player]
        assert self.env.state.current_player == root_player, (
            "search must be called when it is root_player's turn"
        )

        legal = legal_actions_from_state(self.env.state)
        if not legal:
            return {}

        # ---- ROOT EVALUATION: the one place with a real history ----
        # `inputs_for` appends the current observation as a "pending" step, so the
        # LSTM's last step is the state we are actually deciding in.
        root_priors, root_value = self._eval_root(root_player, legal)

        # Dirichlet noise for exploration. During self-play this is what makes the
        # collected data cover more than the current favourite move; set eps=0 at
        # evaluation time so the search is deterministic given the sim count.
        #
        # We keep the CLEAN prior for diagnostics (how far did the search move away
        # from what the net believed?) and the noisy one for the actual search.
        self.last_root_priors = root_priors
        if self.dirichlet_eps > 0 and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            root_priors = (1 - self.dirichlet_eps) * root_priors + self.dirichlet_eps * noise
        self.last_root_priors_noisy = root_priors
        self.last_root_value = root_value

        # ---- Build the root ----
        # All legal actions get a child immediately (AlphaZero style). This is safe
        # with PUCT: an unvisited child has Q=0 and an exploration term
        # c*P*sqrt(N)/(1+0), so even a low-prior action gets visited, and the first
        # simulations spread over actions roughly in proportion to their priors.
        # (`player` is left None here: the next player is NOT always (p+1)%4 — at a
        # deal boundary `_new_round` advances the starter an extra seat — so the
        # child's player is set when we actually step into it.)
        root = Node(player=root_player)
        root.evaluated = True
        root.priors = {a: float(p) for a, p in zip(legal, root_priors)}
        root.value_mover_pov = root_value
        for a in legal:
            root.children[a] = Node(player=None, prior=root.priors[a])

        determinizer = Determinizer(self.env, root_player)
        for _ in range(self.n_iter):
            sampled = determinizer.sample(self.rng)
            self._simulate(root, sampled, root_team)

        return {i: root.children[a].N for i, a in enumerate(legal)}

    # ------------------------------------------------------------------
    # Network plumbing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, obs_flat: np.ndarray, feats: np.ndarray, mask: np.ndarray,
                 obs_seq: Optional[np.ndarray] = None,
                 act_seq: Optional[np.ndarray] = None,
                 act_mask: Optional[np.ndarray] = None):
        """
        One network call. If `obs_seq` is given we feed a whole sequence whose last
        step is the pending step; otherwise a single-step sequence is built from
        `obs_flat` (the interior-node case, with zero LSTM state).
        """
        if obs_seq is None:
            obs_seq = np.asarray(obs_flat, dtype=np.float32)[None, :]
            act_seq = np.zeros((1, ACT_FEAT_DIM), dtype=np.float32)
            act_mask = np.zeros((1,), dtype=bool)

        obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        am_t = torch.as_tensor(act_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        self.stats["evals"] += 1
        logits, value = self.net(obs_t, act_t, am_t, feats_t, mask_t)
        return logits[0], float(value[0])

    def _eval_root(self, root_player: int, legal: List[Action]) -> Tuple[np.ndarray, float]:
        """
        Root evaluation: the real history + the pending current observation.
        Returns (priors over `legal`, value in ROOT-TEAM POV).
        """
        obs_now = self.env.observe(root_player)
        obs_seq, act_seq, act_mask = self.tracker.inputs_for(
            root_player, obs_now, max_len=self.max_history_len
        )
        logits, value = self._forward(
            obs_flat=None,
            feats=obs_now["action_features"],
            mask=obs_now["action_mask"],
            obs_seq=obs_seq, act_seq=act_seq, act_mask=act_mask,
        )
        # Illegal slots are already -1e9; slice the legal prefix before softmax.
        logits = logits[:len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, value

    def _eval_node(self, state: CuarentaState, legal: List[Action]) -> Tuple[np.ndarray, float]:
        """
        Interior/leaf evaluation from a single pending step, built for the player to
        move in `state`. Returns (priors over `legal`, value in MOVER-TEAM POV).

        The mover may be an opponent of the root; their hand comes from the sampled
        determinization and their score fields are their own POV, which is exactly
        the layout the value head was trained on ("chance *I* win").
        """
        mover = state.current_player
        feats, mask = action_features_matrix(legal, state)
        logits, value = self._forward(
            obs_flat=flat_obs_for(mover, state),
            feats=feats, mask=mask,
        )
        logits = logits[:len(legal)]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, value

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(self, root: Node, state: CuarentaState, root_team: int) -> float:
        """
        One simulation: descend by PUCT, evaluate/expand the leaf, back up.

        Returns the value (root-team POV) that was backed up: in win mode that is just the
        leaf/terminal value, in return mode it is the leaf/terminal value PLUS the rewards
        collected on the way down (see the VALUE TARGET note in the module docstring).
        """
        node = root
        path: List[Tuple[Node, Action]] = []
        value_root_pov: Optional[float] = None
        # Rewards collected along this simulation's path, already in root POV and in the
        # value head's units. Stays 0.0 in win mode, where rewards are deliberately not
        # part of the value (the search then optimizes P(win), not the shaped return).
        collected = 0.0

        while True:
            # --- terminal ---
            if state.done:
                # win: the true result, free and exact. return: 0 future reward (the reward
                # of the ply that ended the game was counted as that edge's reward below).
                value_root_pov = self._terminal_value(state, root_team)
                break

            legal = legal_actions_from_state(state)
            if not legal:
                value_root_pov = 0.0   # unreachable in Cuarenta; keep the sim alive
                break
            legal_set = set(legal)

            # --- is this node expanded in THIS world? ---
            # Actions are shared across determinizations, but legality is not: a
            # sampled opponent hand changes which captures exist. So "expanded" is
            # per-world, while the node (and its stats) are shared.
            missing = [a for a in legal if a not in node.children]
            if missing:
                if not node.evaluated:
                    priors, value_mover_pov = self._eval_node(state, legal)
                    node.priors = {a: float(p) for a, p in zip(legal, priors)}
                    node.value_mover_pov = value_mover_pov
                    node.evaluated = True

                for a in missing:
                    node.children[a] = Node(player=None, prior=node.prior_for(a))

                self.stats["sims"] += 1

                # The net's value for the player to move here, flipped into root-team POV.
                # In BOTH modes this is "the value from this state onward", which is exactly
                # what the value head was trained on (win probability, or return-to-go).
                net_value_root_pov = self._to_root_pov(
                    node.value_mover_pov, node.team, root_team
                )

                if self.leaf_value_mode == "rollout":
                    # Baseline/debug mode: value this leaf with a greedy playout.
                    rollout_v, terminated = self._rollout_value(state, root_team)
                    if terminated:
                        value_root_pov = rollout_v
                    else:
                        # A truncated playout means UNKNOWN, not a draw. Bootstrap the rest
                        # with the net's value of this state instead of scoring 0.0 (that
                        # silent zero is what made v1's search blind). In return mode the
                        # playout's own rewards are kept and the net values what is left.
                        self.stats["truncated_rollouts"] += 1
                        value_root_pov = (rollout_v + net_value_root_pov
                                          if self.value_target == "return"
                                          else net_value_root_pov)
                else:
                    value_root_pov = net_value_root_pov
                break

            # --- fully expanded in this world: select and descend ---
            action, _ = self._puct_select(node, legal_set, root_team)
            # The reward of this edge is paid to the team of the player moving in `state`,
            # read from the state itself: a node's cached `player`/`team` may have been set
            # by an earlier determinization, and deal boundaries depend on the (sampled)
            # hands, so the cached mover is not authoritative for THIS world.
            actor_team = TEAMS[state.current_player]
            next_state, reward, _, _ = step_state(state, action, self.rng)
            if self.value_target == "return":
                collected += self._shaped_value(reward, actor_team, root_team)
            path.append((node, action))
            state = next_state
            node = node.children[action]
            # The mover here is a deterministic function of the parent's mover plus
            # this step (including the extra seat shift at a deal boundary), so we
            # can only know it after stepping — which is what we just did.
            node.player = state.current_player
            node.team = TEAMS[node.player]

        # --- BACKUP: every edge on the path, plus the leaf edge ---
        #
        # VISIT-COUNT CONVENTION (this used to be double counted — see below):
        # `node.N` is the number of simulations that have passed THROUGH that node,
        # i.e. the number of times the edge leading into it was traversed. So the
        # only thing the backup does is increment the CHILD of each edge once.
        #
        # The tempting extra `parent.N += 1` is a bug: an intermediate node is the
        # child of one edge and the parent of the next, so it would be incremented
        # twice per simulation. Two consequences, both silent:
        #   * visit counts are inflated for any action whose subtree got explored
        #     deeper, which biases the self-play policy target toward whatever the
        #     search happened to dig into, and
        #   * `child.W / child.N` halves Q for those same nodes, because W is only
        #     added once per simulation.
        # The root has no incoming edge, so its visit count is incremented here once
        # per simulation.
        backup = value_root_pov + collected
        for parent, action in path:
            child = parent.children[action]
            child.N += 1
            child.W += backup
        root.N += 1

        return backup

    def _terminal_value(self, state: CuarentaState, root_team: int) -> float:
        """
        Value of a FINISHED game in the value head's units (root-team POV).

          win    -> +/-1. The payoff is winner-take-all, and winning is zero-sum, which is
                    exactly what lets one value be negated for the opposing team.
          return -> 0.0, because no rewards remain to be collected. The reward of the ply
                    that ended the game is already on that edge, so the simulation's
                    backed-up value still ends up equal to the game's shaped return.
        """
        if self.value_target == "return":
            return 0.0
        return _terminal_value_root_pov(state, root_team)

    def _shaped_value(self, reward: float, actor_team: Optional[int],
                      root_team: int) -> float:
        """
        One step's reward as a contribution to the root team's value: positive when the
        root's team was paid, negative when the opponent was, in the value head's units.

        This mirror image of `ppo/rollout.shaped_reward` is a SHAPING choice, not a fact
        about the game: the env's reward is not zero-sum (both teams earn for captures), so
        negating the opponent's reward invents a rivalry the reward does not have. It is
        the only choice that keeps a single scalar Q usable by PUCT, and it is documented
        as such in the VALUE TARGET note above.
        """
        if actor_team is None:
            return 0.0
        sign = 1.0 if actor_team == root_team else -1.0
        return sign * float(reward) / self.value_scale

    @staticmethod
    def _to_root_pov(value_mover_pov: float, mover_team: Optional[int], root_team: int) -> float:
        """Convert a value expressed for the mover's team into root-team POV."""
        if mover_team is None:
            return value_mover_pov
        return value_mover_pov if mover_team == root_team else -value_mover_pov

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

    def _puct_select(self, node: Node, legal_set, root_team: int) -> Tuple[Action, Node]:
        """
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        `N(s)` is the node's own visit count and `N(s,a)` is the child's (see the
        visit-count convention in `_simulate`).

        Q is expressed in the MOVER's POV (converted from root POV), so the mover
        maximizes it. Unvisited children are NEUTRAL: in win mode that is exactly Q = 0.0
        (a value symmetric around 0 makes "even" the natural unknown), and in return mode
        it is the node's OWN value, because a return is large and positive — scoring a
        fresh child 0.0 there would look catastrophic and the search would never try it.
        Both give `u = c_puct * P * sqrt(N) / (1 + N)` for a fresh child, i.e. the standard
        AlphaZero treatment (no "unvisited gets +infinity" hack, which would ignore the
        priors).
        """
        assert node.team is not None, "PUCT on a node whose mover is unknown"
        sqrt_n = math.sqrt(max(node.N, 1))

        # Return mode works on ADVANTAGES (Q minus this node's own value), which is what
        # makes the exploration bonus comparable to the differences that matter, and what
        # makes a fresh child exactly neutral. Win mode is unchanged: there 0.0 is already
        # the neutral point, so centering would be a no-op at best.
        center = node.value_mover_pov if self.value_target == "return" else 0.0

        best_u, best = -1e18, None
        for action, child in node.children.items():
            if action not in legal_set:
                continue   # phantom action from another determinization
            if child.N == 0:
                q = center
            else:
                mean_root_pov = child.W / child.N
                q = self._to_root_pov(mean_root_pov, node.team, root_team)
            u = (q - center) + self.c_puct * child.prior * sqrt_n / (1 + child.N)
            if u > best_u:
                best_u, best = u, (action, child)

        if best is None:
            # Should be unreachable: `missing` was empty, so every legal action has
            # a child. Kept as a loud failure rather than a silent wrong move.
            raise RuntimeError("_puct_select: no legal children for this determinization")
        return best

    # ------------------------------------------------------------------
    # Rollout fallback (leaf_value_mode="rollout")
    # ------------------------------------------------------------------

    def _rollout_value(self, state: CuarentaState, root_team: int) -> Tuple[float, bool]:
        """
        Greedy playout to a terminal state.

        Returns (value in root-team POV, terminated?), where the value is the value of
        everything from `state` ONWARD: in win mode the terminal +/-1, in return mode the
        shaped reward the playout collected. `terminated=False` means the playout hit
        `max_rollout_len` without the game ending, in which case the value is incomplete
        and the caller must add the net's bootstrap for the remainder.
        """
        s = clone_state(state)
        steps = 0
        collected = 0.0
        while not s.done and steps < self.max_rollout_len:
            legal = legal_actions_from_state(s)
            if not legal:
                break
            actor_team = TEAMS[s.current_player]
            a = _rollout_policy(s, self.rng)
            s, reward, _, _ = step_state(s, a, self.rng)
            if self.value_target == "return":
                collected += self._shaped_value(reward, actor_team, root_team)
            steps += 1
        self.stats["rollouts"] += 1
        if s.done:
            # win: `collected` is 0.0, so this is the old +/-1 exactly.
            # return: the terminal state adds no further reward.
            return collected + self._terminal_value(s, root_team), True
        return collected, False
