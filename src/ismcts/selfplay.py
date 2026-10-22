"""
Self-play data generation for the Cuarenta net (expert iteration).

Each game is played with all four seats controlled by the same net (parameter
sharing, as in AlphaZero). At each decision we record:

  - obs_seq:    the acting player's observation history, ENDING WITH the current
                decision's observation as a "pending" step (act_mask False), so the
                network's state embedding is the state we are deciding in.
  - act_seq:    per-step feature vector (84) of the action actually taken at that
                step (zeros for the pending step). Feature-based, not slot-based:
                a slot index means a different action in every state.
  - act_mask:   whether an action was taken at each step.
  - feats:      current decision's per-slot action features  [MAX_ACTIONS, 84]
  - mask:       current decision's legal-action mask         [MAX_ACTIONS]
  - visits:     visit counts from the search over slots      [MAX_ACTIONS]
  - team:       the acting player's team (0/1) for outcome assignment
  - player:     the acting player's seat (0..3) — useful for debugging
  - ply:        global move number (for phase analysis / logging)
  - root_value: the search's root value estimate (root team's POV), for debugging

After the game ends we set `outcome` on every recorded decision: +1 if the recorded
team won, -1 otherwise. That is the value-head target in the DEFAULT mode.

VALUE TARGET: +1 / -1 by default, the env reward on request
-----------------------------------------------------------
`--value-target` (see the VALUE TARGET note in network.py) picks what the value head
learns AND what the search backs up; `backfill_value_targets` below writes whichever one
applies. Why ±1 is the default:

  * ZERO-SUM, which the backup needs. The search keeps ONE value per node and negates it
    when that node's mover is on the other team (`network_ismcts_1._to_root_pov`,
    `_puct_select`). That negamax convention is only meaningful for a zero-sum quantity,
    and winning is exactly that: +1 for one team IS -1 for the other. The env reward is
    NOT zero-sum — both teams collect +1 per captured card — so negating it is a shaping
    choice rather than a fact, and Q ends up on a different scale (which is why return
    mode also has to center Q on the node's value; see `_puct_select`).
  * RANGE. The value head ends in `tanh` -> (-1, 1) and terminal values are ±1. A game
    return is a different order of magnitude (tens to ~150 points) — so return mode uses
    a LINEAR head and divides by `RETURN_VALUE_SCALE`, and "unvisited child has Q = 0"
    stops meaning "unknown/even" unless Q is centered.
  * ALIGNMENT. Points in Cuarenta come from caídas, limpias and cartón, while captured
    cards only pay off through the cartón, so `WIN_BONUS` is a small share of a game's
    return and a team that farms cards and loses still earns most of it. Return-maximising
    is therefore NOT winning — which is fine if that is what you want to measure, and is
    exactly why the ±1 target stays the default.

So in win mode the env reward is LOGGING-ONLY here (it is what the `selfplay/*` curves in
train.py plot), and PPO is the pipeline that optimises it. In return mode the search
optimises it too: `_simulate` accumulates the per-step rewards along each simulation path
and the head is trained on the same shaped return-to-go (`--value-target return`).

MOVE SELECTION (temperature)
----------------------------
The action actually played is sampled from the visit distribution with a
temperature schedule: tau = 1.0 for the first `temperature_cutoff` decisions of the
game (exploration, and it keeps the opening diverse), then tau -> 0 (greedy, i.e.
play the most-visited action). AlphaZero does the same; keeping tau = 1 for the
whole game (what the old code did) means the net keeps playing weak sampled moves
deep into the game, and the recorded visit targets then describe a noisy policy.

GAME RESULT (returns and rewards)
---------------------------------
`play_one_game` also returns a `GameResult`: the final scores, the number of plies,
and — per team — the episode RETURN (the sum of the env's step rewards) together
with every component that produces it. The env pays, per turn and from the acting
team's point of view:

    +1 per captured card, +CAIDA_POINTS (+2) for a caída,
    +LIMPIA_POINTS (+2) for a limpia, the cartón points when a deal ends,
    +WIN_BONUS (+20) for winning the game

So for each team

    returns == captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won

holds EXACTLY, and `GameResult.accounted_return` computes that right-hand side. Those
numbers are worth watching over training: a net that stops taking caídas shows it
immediately even when its win rate looks flat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import CuarentaEnv, TEAMS, WIN_BONUS
from captures import CAIDA_POINTS, LIMPIA_POINTS
from history import HistoryTracker, ACT_FEAT_DIM
from network import CuarentaNet, N_ACTIONS, RETURN_VALUE_SCALE
from network_ismcts_1 import NetworkISMCTS
from utils import PointsReasons, has_reason


def backfill_value_targets(
    decisions: List["Decision"],
    rewards: List[float],
    actor_teams: List[int],
    winner_team: int,
    value_target: str,
    value_lambda: float = 1.0,
    bootstrap_values: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Give every Decision its value target, according to `value_target`.

    Arguments are the game as a flat list of plies:

        rewards[t]          the env's reward at ply t, from the ACTING PLAYER's team POV
        actor_teams[t]      the team of the player who moved at ply t
        bootstrap_values[t] (return mode, lambda < 1 only) the net's value OF THE STATE at
                            ply t, from the ply-t actor's team POV, in the same units as
                            `rewards` (i.e. raw game points). `None` or zeros mean "no
                            estimate available", and then the Monte-Carlo target is used.

    `win` writes the game result (+1 / -1) into every decision. `return` writes the SHAPED
    return from that ply on, scaled by `RETURN_VALUE_SCALE`:

        G_t = sum_{t' >= t} sign * reward[t'] ,  sign = +1 if actor_teams[t'] == team

    where `team` is the decision's own team — the same quantity `NetworkISMCTS` backs up
    when it runs in return mode, which is the point: target and search must agree.

    WHY `value_lambda` EXISTS (return mode only)
    --------------------------------------------
    The Monte-Carlo target above is a SUM OVER THE WHOLE REST OF THE GAME, and most of its
    spread is captures in later rounds that the current state does not determine, so a value
    head asked to predict it ends up memorising the replay buffer instead of the game —
    which is fatal here because the search values its leaves with that head.

    `value_lambda` (1.0 = pure Monte-Carlo, the old behaviour) trades that noise for the
    head's own current estimate, using the standard TD(lambda) recursion with no discounting:

        G_t^lambda = sign * reward[t] + (1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda

    with `G_T = 0` after the last ply (in return mode a finished game is worth 0 future
    reward; the reward of the winning ply is already in `reward[T-1]`). Both `V(s_{t+1})`
    and the tail are expressed in the DECISION's team POV, so `bootstrap_values`, which are
    recorded from each ply's own actor, get the same sign flip the search uses.

        lambda = 1.0  -> exactly the Monte-Carlo target (asserted by the tests)
        lambda = 0.0  -> one-step TD: my reward now plus the net's estimate of the rest
        0 < lambda < 1 mixes the two, with weight lambda^(k) on the k-ply-later future

    At lambda < 1 the targets inherit the head's error (that is what bootstrapping means),
    but they are far less noisy, and the noise was the thing being memorised.

    Lives here (rather than inside the game loop) because the imitation teacher's games in
    `bootstrap.py` need exactly the same targets: mixing ±1 samples into a return-target
    dataset would be a silent scale error in the value loss. Those teacher games are
    generated WITHOUT a net, so they have no bootstrap values and keep Monte-Carlo targets
    (`bootstrap_values=None`); both are estimates of the same quantity, so mixing them only
    changes each sample's variance.
    """
    if value_target == "win":
        for d in decisions:
            d.outcome = 1.0 if d.team == winner_team else -1.0
        return {}

    n = len(rewards)
    assert len(actor_teams) == n, (len(actor_teams), n)
    assert all(0 <= d.ply < n for d in decisions), "a decision has no ply in this game"
    if value_lambda < 0.0 or value_lambda > 1.0:
        raise ValueError(f"value_lambda must be in [0, 1], got {value_lambda}")

    use_bootstrap = (
        value_lambda < 1.0
        and bootstrap_values is not None
        and len(bootstrap_values) == n
        and any(v != 0.0 for v in bootstrap_values)
    )

    # return_to_go[team][t] = the shaped return from ply t on, from `team`'s point of view.
    # Computed for both teams because the acting side alternates (and shifts by a seat at
    # a deal boundary), so "whose reward is this" cannot be derived from the ply parity.
    # The two teams' values are exact mirror images (the immediate reward, the bootstrap and
    # the tail all carry the same sign flip), which the tests assert.
    # Monte-Carlo table, always computed: it is the lambda = 1 target AND the baseline the
    # lambda target is compared against in the logs.
    mc = {0: [0.0] * (n + 1), 1: [0.0] * (n + 1)}
    for t in range(n - 1, -1, -1):
        for team in (0, 1):
            sign = 1.0 if actor_teams[t] == team else -1.0
            mc[team][t] = mc[team][t + 1] + sign * float(rewards[t])

    ret = mc
    if use_bootstrap:
        lam = float(value_lambda)
        ret = {0: [0.0] * (n + 1), 1: [0.0] * (n + 1)}
        for t in range(n - 1, -1, -1):
            # The head's estimate of the state AFTER this ply. It was recorded by the ply
            # `t + 1` decision (whose actor may be either team), so it is sign-flipped into
            # this team's POV. Past the last ply there is no future reward at all.
            if t + 1 < n:
                v_next_actor_pov = float(bootstrap_values[t + 1])
                next_actor = actor_teams[t + 1]
            else:
                v_next_actor_pov = 0.0
                next_actor = None
            for team in (0, 1):
                sign = 1.0 if actor_teams[t] == team else -1.0
                if next_actor is None:
                    v_next = 0.0
                else:
                    v_next = v_next_actor_pov if next_actor == team else -v_next_actor_pov
                ret[team][t] = (sign * float(rewards[t])
                                + (1.0 - lam) * v_next
                                + lam * ret[team][t + 1])

    for d in decisions:
        d.outcome = ret[d.team][d.ply] / RETURN_VALUE_SCALE

    # Spread of the labels actually trained on vs the Monte-Carlo ones, in GAME POINTS (the
    # tables are still in raw points here; only `d.outcome` is divided by the scale). A run
    # logs both: lambda is buying something only while `target_std` < `mc_std`, and it is
    # actively harmful (it feeds V's error into every label) when it is the other way round.
    def _std(table) -> float:
        vals = [table[d.team][d.ply] for d in decisions]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    return {"target_std": _std(ret), "mc_std": _std(mc)}


@dataclass
class Decision:
    """One decision point (one search) in one game."""
    obs_seq: np.ndarray       # [T, OBS_FLAT_DIM]  (last step = pending current obs)
    act_seq: np.ndarray       # [T, ACT_FEAT_DIM]  (action features taken per step)
    act_mask: np.ndarray      # [T]
    feats: np.ndarray         # [MAX_ACTIONS, ACTION_FEAT_DIM]
    mask: np.ndarray          # [MAX_ACTIONS]
    visits: np.ndarray        # [MAX_ACTIONS]  (int) -> policy target
    team: int                 # 0 or 1
    player: int               # 0..3
    ply: int = 0              # move number in the game
    root_value: float = 0.0   # search root value (root team's POV, see below)
    priors: Optional[np.ndarray] = None  # clean net prior over the legal slots
    # The VALUE TARGET, filled in after the game ends (see `backfill_value_targets`):
    # either ±1 (win mode) or the scaled shaped return-to-go (return mode). Which one is
    # decided by `--value-target`, and the name is kept from the AlphaZero formulation.
    outcome: float = 0.0
    # True  = self-play data: the policy target comes from the search, and the dataset
    #         may soften it (see dataset.target_temperature).
    # False = imitation data from the heuristic teacher: its one-hot target must stay
    #         one-hot, and the trainer can give it a different policy weight. We mix
    #         these in mainly to keep the VALUE head calibrated on non-self-play games.
    selfplay: bool = True


@dataclass
class GameResult:
    """
    Summary of one finished game, for logging/analysis. Everything is per TEAM
    (index 0/1), not per seat.

    `returns` is the episode RETURN: the sum of the env's step rewards for that team's
    players over the whole game. Under the env's current reward it equals

        captures + 2*caidas + 2*limpias + WIN_BONUS*won

    EXACTLY (see `accounted_return`), which is why the components are all fields here:
    they are the reward curves you actually want to read during training.
    """
    winner: int                       # winning team (0 or 1)
    scores: Tuple[int, int]           # final points per team (the game's 0..40 scale)
    plies: int                        # number of moves played
    returns: Tuple[float, float]      # summed step rewards
    captures: Tuple[int, int]         # cards captured (env pays +1 per captured card)
    caidas: Tuple[int, int]           # count of caída bonuses earned
    limpias: Tuple[int, int]          # count of limpia bonuses earned
    carton: Tuple[int, int] = (0, 0)  # cartón points paid (they ARE rewarded again)
    mean_root_value: Tuple[float, float] = (0.0, 0.0)   # search's own estimate
    # Spread of the value targets of this game, in game points: what the head was actually
    # trained on (`value_target_std`) vs the Monte-Carlo alternative (`value_mc_std`). In
    # return mode with lambda < 1 the first should be the smaller one; if it is not, the
    # bootstrapped label is carrying more of the head's error than it removes of the
    # future's noise (see `backfill_value_targets`).
    value_target_std: float = 0.0
    value_mc_std: float = 0.0

    def accounted_return(self, team: int) -> float:
        """
        The return the env's reward terms MUST add up to for `team`.

        Lives next to the fields it summarises so that a reward change and the identity
        describing it are edited in one place; `test_reward_parity.py` asserts
        `returns[team] == accounted_return(team)` on real games, which is what keeps
        this docstring honest.
        """
        won = 1.0 if self.winner == team else 0.0
        return (float(self.captures[team])
                + CAIDA_POINTS * self.caidas[team]
                + LIMPIA_POINTS * self.limpias[team]
                + float(self.carton[team])
                + WIN_BONUS * won)

    def as_metrics(self, prefix: str = "game") -> Dict[str, float]:
        """Flatten into wandb-friendly scalars (team 0 / team 1 / both)."""
        won0 = 1.0 if self.winner == 0 else 0.0
        return {
            f"{prefix}/plies": float(self.plies),
            f"{prefix}/score_team0": float(self.scores[0]),
            f"{prefix}/score_team1": float(self.scores[1]),
            f"{prefix}/score_diff": float(self.scores[0] - self.scores[1]),
            f"{prefix}/return_team0": float(self.returns[0]),
            f"{prefix}/return_team1": float(self.returns[1]),
            f"{prefix}/return_mean": float(np.mean(self.returns)),
            f"{prefix}/captures_team0": float(self.captures[0]),
            f"{prefix}/captures_team1": float(self.captures[1]),
            f"{prefix}/capture_diff": float(self.captures[0] - self.captures[1]),
            f"{prefix}/caidas_team0": float(self.caidas[0]),
            f"{prefix}/caidas_team1": float(self.caidas[1]),
            f"{prefix}/limpias_team0": float(self.limpias[0]),
            f"{prefix}/limpias_team1": float(self.limpias[1]),
            f"{prefix}/carton_team0": float(self.carton[0]),
            f"{prefix}/carton_team1": float(self.carton[1]),
            f"{prefix}/win_bonus_team0": WIN_BONUS * won0,
            f"{prefix}/win_bonus_team1": WIN_BONUS * (1.0 - won0),
            f"{prefix}/mean_root_value_team0": float(self.mean_root_value[0]),
            f"{prefix}/mean_root_value_team1": float(self.mean_root_value[1]),
            f"{prefix}/value_target_std": float(self.value_target_std),
            f"{prefix}/value_mc_std": float(self.value_mc_std),
        }


def play_one_game(
    net: CuarentaNet,
    *,
    n_iter: int = 200,
    max_history_len: int = 32,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    temperature_cutoff: int = 24,
    # `dirichlet_eps` is smaller than the AlphaZero default of 0.25 on purpose. The
    # noise is mixed into the root priors BEFORE the search, so it ends up inside the
    # visit distribution we train on. With a small sim budget and a value head that is
    # only slightly better than the policy, a large eps makes a big share of the
    # training target pure noise (the net can only memorise it). 0.1 still breaks
    # ties/mode collapse without dominating a 400-sim search.
    dirichlet_eps: float = 0.10,
    leaf_value_mode: str = "net",
    max_rollout_len: int = 512,
    device: str = "cpu",
    value_lambda: float = 1.0,
) -> Tuple[List[Decision], GameResult]:
    """
    Play a full self-play game. Returns (decisions, result): one Decision per search
    performed, plus a GameResult with the scores, plies and per-team returns.
    All 4 seats use `net`.

    `temperature` applies for the first `temperature_cutoff` decisions, after which
    the agent plays greedily (argmax over visits). Set `temperature_cutoff=0` for a
    fully greedy game (useful for evaluation).

    The value target written into the returned decisions follows the SEARCH, which follows
    the net (`net.value_target`): ±1 in win mode, the scaled shaped return in return mode.
    Taking it from the search rather than from a separate argument is deliberate — the
    target must be the quantity the search itself backs up.

    `value_lambda` (return mode only; 1.0 = pure Monte-Carlo) is passed to
    `backfill_value_targets`, together with the net's value at every decision point — which
    this function already has, because the search evaluates each root. So bootstrapping
    costs nothing extra here.
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, net, tracker,
        n_iter=n_iter,
        max_history_len=max_history_len,
        leaf_value_mode=leaf_value_mode,
        max_rollout_len=max_rollout_len,
        rng=random.Random(seed),
        dirichlet_eps=dirichlet_eps,
        device=device,
    )
    value_target = searcher.value_target
    value_scale = searcher.value_scale

    decisions: List[Decision] = []
    ply = 0

    # --- reward bookkeeping (per team) ---
    # The env's step reward is from the acting player's TEAM point of view and is fully
    # explained by the counters below: +1 per captured card, +2 per caída, +2 per limpia,
    # the cartón points and +WIN_BONUS for winning (see the REWARD note in env.py). These
    # are kept HERE rather than read off the final state because `_new_round` resets
    # `cards_captured` at every cartón, so the end-of-game capture piles are only the last
    # deal's. `GameResult.accounted_return` recomputes the sum from them.
    returns = [0.0, 0.0]
    captures = [0, 0]
    caidas = [0, 0]
    limpias = [0, 0]
    carton = [0, 0]
    # The per-ply reward stream and who was paid, which is what the return-to-go value
    # target in return mode is built from (see `backfill_value_targets`).
    ply_rewards: List[float] = []
    ply_actor_teams: List[int] = []

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)

        # --- Search ---
        # The search reads the player's history + this pending observation itself
        # (see NetworkISMCTS._eval_root), so we do NOT need to pass obs_seq here.
        visits_dict = searcher.search(root_player=p)   # {slot_idx: count}

        visits = np.zeros(N_ACTIONS, dtype=np.int64)
        for slot, count in visits_dict.items():
            visits[slot] = count

        # --- Snapshot the net's inputs for training ---
        # inputs_for appends `obs` as the pending step, which is exactly what the
        # search used. Kept consistent on purpose: train == inference.
        obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs, max_len=max_history_len)

        decisions.append(Decision(
            obs_seq=obs_seq.astype(np.float32),
            act_seq=act_seq.astype(np.float32),
            act_mask=act_mask.astype(bool),
            feats=obs["action_features"].astype(np.float32),
            mask=obs["action_mask"].astype(bool),
            visits=visits,
            team=TEAMS[p],
            player=p,
            ply=ply,
            # Raw units on purpose: in return mode the net's output is a SCALED return, so
            # multiply it back to compare it with the game's actual return in the logs.
            root_value=float(searcher.last_root_value) * value_scale,
            # Clean prior over the legal slots, for diagnostics (target vs prior
            # entropy, how far the search moved from the policy).
            priors=np.asarray(searcher.last_root_priors, dtype=np.float32),
            selfplay=True,
        ))

        # --- Choose the action to actually play ---
        legal_slots = np.flatnonzero(obs["action_mask"])
        tau = temperature if ply < temperature_cutoff else 0.0

        if sum(visits) == 0:
            a = int(np.random.choice(legal_slots))            # no signal: random
        elif tau <= 1e-6:
            a = int(visits.argmax())                          # greedy
        else:
            # Sample from visits^(1/tau) restricted to legal actions.
            v = visits[legal_slots].astype(np.float64) ** (1.0 / tau)
            probs = v / v.sum() if v.sum() > 0 else np.ones(len(v)) / len(v)
            a = int(np.random.choice(legal_slots, p=probs))

        # The action object is needed for the capture count, and `legal_actions()` is
        # cached, so reading it here is free — but it must be read BEFORE `step`, which
        # invalidates the cache.
        action = env.legal_actions()[a]

        # Record BEFORE stepping so the tracker is consistent ("saw this, did that").
        tracker.record(p, obs, a)
        _, reward, done, _, info = env.step(a)

        # Accumulate the reward for the team that just acted, and its components.
        t = TEAMS[p]
        returns[t] += float(reward)
        ply_rewards.append(float(reward))
        ply_actor_teams.append(t)
        if action.capture:
            captures[t] += len(action.capture)   # the env's +1-per-card term
        reasons = info.get("reasons") or []
        # Caída and limpia are mutually exclusive in `score_play` (elif), so an `elif`
        # here mirrors it exactly instead of double counting.
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[t] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[t] += 1
        # `carton_points` is only present on the ply that ENDS a deal, and it is the ACTING
        # team's cartón — the same number the env just added to this step's reward.
        carton[t] += int(info.get("carton_points") or 0)

        ply += 1

    # --- Backfill the value targets (see `backfill_value_targets`) ---
    winner_team = env.state.winner
    # `Decision.root_value` is the net's own value at that decision's state (raw units, in
    # the acting team's POV), i.e. exactly the bootstrap value a TD(lambda) target needs.
    # In win mode the bootstrap is unused, so passing it is free.
    bootstrap = [d.root_value for d in decisions]
    target_stats = backfill_value_targets(decisions, ply_rewards, ply_actor_teams,
                                          winner_team, value_target,
                                          value_lambda=value_lambda,
                                          bootstrap_values=bootstrap)

    # Mean of the search's own value estimate, per team — a sanity check against the
    # actual result (this is what the value head predicted DURING play).
    mr = {}
    for team in (0, 1):
        vals = [d.root_value for d in decisions if d.team == team]
        mr[team] = float(np.mean(vals)) if vals else 0.0

    result = GameResult(
        winner=int(winner_team),
        scores=(int(env.state.scores[0]), int(env.state.scores[1])),
        plies=ply,
        returns=(returns[0], returns[1]),
        captures=(captures[0], captures[1]),
        caidas=(caidas[0], caidas[1]),
        limpias=(limpias[0], limpias[1]),
        carton=(carton[0], carton[1]),
        mean_root_value=(mr[0], mr[1]),
        value_target_std=float(target_stats.get("target_std", 0.0)),
        value_mc_std=float(target_stats.get("mc_std", 0.0)),
    )
    return decisions, result
