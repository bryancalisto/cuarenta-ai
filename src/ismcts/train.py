"""
Training loop for the Cuarenta net (expert iteration / AlphaZero style).

  for iteration in range(N_ITERATIONS):
      1. Generate self-play games with the current net (search = this net itself).
      2. Mix in a few imitation games from the heuristic teacher (value calibration).
      3. Train the net to match (a) the search's visit distribution and (b) the value
        target (±1 game outcomes by default, --value-target return for the dense reward).
      4. Evaluate: vs the reference checkpoint (gating), vs the heuristic, vs the rest
         of the evaluation LEAGUE (initial / prev_checkpoint / random — the same
         members the PPO pipeline logs), plus two health metrics that predict failure
         BEFORE the win rate moves.
      5. Save a checkpoint, and update `best.pt` when the reference match is won.

This is not PPO. The policy loss is cross-entropy against the visit distribution
(flattened a little, see dataset.target_temperature) and the value loss is MSE
against the game outcome.

SHARED METRIC CONTRACT WITH THE OTHER PIPELINES
-----------------------------------------------
Every match is logged as `eval/<member><suffix>` with the member names PPO also uses
(`reference`, `heuristic`, `initial`, `prev_checkpoint`, `random`) and the suffixes
`_stderr`, `_score_for/_against`, `_return_for/_against`, `_caidas_for/_against`,
`_limpias_for/_against`, `_plies`, `_n_games` (see `src/ppo/eval_harness.py:
match_metrics`). the same names PPO and ReBeL log (the old `vs_*` aliases were removed)
`reference` / `heuristic` with identical values. The x-axis metric is `iteration`,
the same name PPO and ReBeL log (`iter` is still logged with the same value).

WHY THERE ARE FOUR NUMBERS ON THE EVAL LINE
-------------------------------------------
Self-play can silently destroy a good net while the win rate vs the heuristic still
looks flat, so two health metrics are printed every evaluation next to the two
matches. They are what says "stop" before the win rate does:

  * `ref`      — win rate vs the REFERENCE checkpoint (the best net so far), both
                 sides searching. This is the AlphaZero gating criterion and the
                 primary signal: it is a same-distribution, same-opponent comparison
                 (identical search, identical weights except for training), so it
                 moves for the right reason.
  * `heuristic`— win rate vs the fixed HeuristicAgent. Kept because it is the
                 project's external reference, but it is NOT the training target:
                 self-play legitimately moves the net away from the heuristic's
                 distribution, so this number can fall while the net gets stronger.
  * `edge`     — win rate of the net WITH search against the SAME net's raw policy
                 (argmax of the prior, no search). This is the "teacher edge": the
                 whole point of expert iteration. Once it collapses, the self-play
                 targets are mostly self-distillation, so training has nothing to
                 teach.
  * `v_mse`    — held-out value MSE on a FIXED set of heuristic games (seeds
                 PROBE_SEEDS, never used for training). 1.0 means "worse than always
                 predicting 0". The search VALUES LEAVES WITH THIS HEAD, so drift here
                 is not a side metric: it degrades the search directly, while the raw
                 policy's strength can stay flat.

Under the hood:
  * SELF-PLAY and EVAL use the same search settings (same n_iter, same leaf value),
    so the numbers describe the agent you are actually training.
  * Value anchor: see train_one_epoch. It keeps the value head near the starting
    checkpoint's calibration, because the search depends on it.
"""

from __future__ import annotations

import atexit
import datetime
import json

import os
import argparse
import copy
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import dotenv

from env import CuarentaEnv, TEAMS
from network import CuarentaNet, VALUE_TARGETS, RETURN_VALUE_SCALE
from selfplay import play_one_game, Decision, GameResult
from dataset import CuarentaDataset
from network_ismcts_1 import NetworkISMCTS
from bootstrap import play_one_heuristic_game

dotenv.load_dotenv() # to load wandb token

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ITERATIONS = 30             # outer loop iterations
GAMES_PER_ITER = 40           # self-play games per iteration (more = better; too few
                              # games leave the value head so few distinct outcomes
                              # that it can only memorise them)
IMITATION_GAMES_PER_ITER = 5  # heuristic games mixed in for value calibration
IMITATION_POLICY_WEIGHT = 0.0 # 0 = use them for the VALUE loss only
SELFPLAY_WORKERS = 0          # processes for the self-play games (0 = one per CPU, 1 = serial)
N_SEARCH_ITER = 400           # ISMCTS sims per decision (self-play AND eval)
DIRICHLET_EPS = 0.10          # root exploration noise (it lands in the target too)
TEMPERATURE_CUTOFF = 24       # plies with tau=1 before switching to greedy play
TARGET_TEMPERATURE = 1.5      # soften the search's visit distribution (1.0 = raw)
EPOCHS_PER_ITER = 4           # training epochs per iteration
BATCH_SIZE = 64
LR = 1e-3
T_MAX = 32
HISTORY_DROPOUT = 0.3         # see dataset.py: makes history-free evals in-distribution
# REPLAY_ITERS = 5              # keep this many iterations of data in the buffer
REPLAY_ITERS = 3              # keep this many iterations of data in the buffer
EVAL_GAMES = 60               # games per match (60 => ~±6% standard error)
EVAL_RAW_GAMES = 40           # games for the teacher-edge probe
EVAL_EVERY = 2                # run evaluations every N iterations
# Every evaluation this trainer can run, and the set it runs by default.
#
#   reference       the gated best-so-far -- REQUIRED, it is what the gate reads
#   heuristic       the rule-based bot: the fixed external yardstick
#   initial         the frozen starting weights: "has this run gained anything?"
#   prev_checkpoint the previous snapshot: same net, small noisy difference
#   random          the sanity floor (needs no weights, so it is always cheap)
#   teacher_edge    search vs this net's own raw policy -- the expert-iteration health
#                   check, i.e. "does the search still beat the policy?"
#
# `prev_checkpoint` is NOT in the default: it is the same net a few iterations earlier, so
# the difference is small and noisy, and `reference` + `initial` already bracket the
# trajectory. It costs `--league-games` games, so it is opt-in.
ALL_MEMBERS = ("reference", "heuristic", "initial", "prev_checkpoint", "random",
               "teacher_edge")
DEFAULT_MEMBERS = ("reference", "heuristic", "initial", "random", "teacher_edge")
NO_LEAGUE_MEMBERS = ("reference", "heuristic", "random", "teacher_edge")

LEAGUE_GAMES = 30             # games per EXTRA league member (initial,
                              # prev_checkpoint, random): fewer than EVAL_GAMES
                              # because only `reference` gates acceptance
ACCEPT_THRESHOLD = 0.55       # beat the reference by this much to be accepted
LEAF_VALUE_MODE = "net"       # "net" (default) | "rollout"
MAX_ROLLOUT_LEN = 512         # only used by leaf_value_mode="rollout"
# What the value head predicts, and therefore what the SEARCH backs up: "win" = ±1 game
# results (AlphaZero, zero-sum, plays to maximize win probability) or "return" = the env's
# dense reward as a scaled return-to-go (the search then accumulates rewards down every
# simulation; see the VALUE TARGET note in network.py). Changing it changes the head's
# activation, so existing checkpoints are win-mode and cannot be reused in return mode.
VALUE_TARGET = "win"
VALUE_LAMBDA = 0.9            # RETURN mode only: TD(lambda) bootstrapping of the value
                              # target (1.0 = pure Monte-Carlo returns, the old behaviour).
                              # The MC target's spread is mostly captures in later rounds
                              # that the state does not determine, and a head trained on it
                              # memorises the buffer instead of the game. See
                              # selfplay.backfill_value_targets for the recursion.
PROBE_EVERY = 1               # run the held-out head probe every N iterations. It is
                              # cheap (no search) and it is the earliest warning that the
                              # value head has stopped generalising; 0 = only with the
                              # full evaluation (--eval-every).
DROPOUT = 0.0                 # dropout on the shared embedding (0 = off). See network.py
                              # for the placement and why it is not inside the LSTM.
VALUE_ANCHOR_WEIGHT = 2.0     # keep the value head near the starting checkpoint's
                              # calibration (0 = off, 5+ = effectively frozen; too small
                              # a value has no effect on it, which is the failure the anchor
                              # exists for)
CKPT_DIR = "checkpoints"
# Fixed, never-trained-on seeds for the value/policy probe.
PROBE_SEEDS = list(range(900_000, 900_020))


def _pick_device(requested: str = "auto") -> str:
    """
    Device selection.

    The search does ONE batch-1 network forward per evaluated node, and MPS per-op
    launch overhead dominates at that size, so CPU is substantially FASTER than MPS
    for this workload.

    So "auto" prefers CUDA when present and CPU otherwise; MPS is available with
    --device mps but usually a bad trade here.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Chosen at import time, overridden by --device in main(). Deliberately NOT printed
# here: worker processes re-import this module and would each print the same line.
DEVICE = _pick_device()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

# Config every self-play worker needs. The search parameters are module constants (a
# spawned worker only sees module defaults, never the parent's CLI), so the parent's
# resolved values travel with the job as one small dict.
def _selfplay_cfg() -> dict:
    return {
        "n_iter": N_SEARCH_ITER,
        "max_history_len": T_MAX,
        "temperature": 1.0,
        "temperature_cutoff": TEMPERATURE_CUTOFF,
        "dirichlet_eps": DIRICHLET_EPS,
        "leaf_value_mode": LEAF_VALUE_MODE,
        "max_rollout_len": MAX_ROLLOUT_LEN,
        "value_lambda": VALUE_LAMBDA,
    }


def _selfplay_kwargs(seed: int, cfg: dict, device: str) -> dict:
    """The keyword arguments `play_one_game` needs, from the cfg above."""
    return dict(
        n_iter=cfg["n_iter"],
        max_history_len=cfg["max_history_len"],
        seed=seed,
        temperature=cfg["temperature"],
        temperature_cutoff=cfg["temperature_cutoff"],
        dirichlet_eps=cfg["dirichlet_eps"],
        leaf_value_mode=cfg["leaf_value_mode"],
        max_rollout_len=cfg["max_rollout_len"],
        device=device,
        value_lambda=cfg["value_lambda"],
    )


# The pool, the file its workers reload weights from, and the run's search config: built
# once per process (creating it per iteration would pay torch's import in every child on
# every iteration -- seconds against a ~10 s batch of games).
_SELFPLAY: dict = {}
atexit.register(lambda: close_selfplay_pool())


def _selfplay_worker_init(weights_path: str, cfg: dict, training: bool,
                          value_target: str, dropout: float) -> None:
    """
    Prepare a worker: one torch thread, and the settings a net needs that are NOT in its
    weights.

    `training` mirrors the parent's module mode, because that decides whether dropout
    fires; a worker that quietly evaluated would generate different games from the
    trainer's own (with `--dropout > 0`) while looking perfectly normal. `value_target`
    and `dropout` are the other two settings that live outside the state_dict.
    """
    import torch as _torch

    _torch.set_num_threads(1)
    _SELFPLAY.update(weights_path=weights_path, cfg=cfg, training=training,
                     value_target=value_target, dropout=dropout,
                     net=None, weights_mtime=None)


def _selfplay_worker(seed: int):
    """
    One self-play game in a worker process.

    The action sampler in `play_one_game` draws from numpy's GLOBAL generator, and the
    search's root noise from the same one, so the per-game seed is installed here: each
    game then depends only on its seed, which is what makes a batch independent of how
    many workers played it, or of the order in which they finished.

    The weights are reloaded when the parent has written a new file, not shipped with
    every task: one 6 MB read per worker per iteration beats 6 MB through the pipe per
    game. Every worker starts on CPU, so a CUDA trainer still collects on the CPU -- the
    search is Python-bound there anyway, and the state dict crosses as CPU tensors.
    """
    np.random.seed(seed % (2 ** 32))
    mtime = os.stat(_SELFPLAY["weights_path"]).st_mtime_ns
    if _SELFPLAY["net"] is None or _SELFPLAY["weights_mtime"] != mtime:
        state = torch.load(_SELFPLAY["weights_path"], map_location="cpu", weights_only=True)
        net = _load_net(state, _SELFPLAY["value_target"], dropout=_SELFPLAY["dropout"])
        net.train(_SELFPLAY["training"])
        _SELFPLAY.update(net=net, weights_mtime=mtime)
    return play_one_game(_SELFPLAY["net"], **_selfplay_kwargs(seed, _SELFPLAY["cfg"], "cpu"))


def _ensure_selfplay_pool(workers: int, state_dict, cfg: dict, training: bool,
                          value_target: str, dropout: float):
    """Publish `state_dict` for the workers and return the (created-once) pool."""
    import tempfile

    if _SELFPLAY.get("pool") is None:
        from multiprocessing import get_context

        # `spawn`, not `fork`: the workers import torch and build a net, which is exactly
        # the combination fork is known to break.
        ctx = get_context("spawn")
        path = os.path.join(tempfile.gettempdir(), f"cuarenta_selfplay_{os.getpid()}.pt")
        _SELFPLAY.update(
            path=path,
            pool=ctx.Pool(
                processes=workers,
                initializer=_selfplay_worker_init,
                initargs=(path, cfg, training, value_target, dropout),
            ),
        )
    # Written on every call (the weights change every iteration); the workers notice the
    # new mtime on their next game and reload.
    torch.save(state_dict, _SELFPLAY["path"])
    return _SELFPLAY["pool"]


def close_selfplay_pool() -> None:
    """Shut the workers down and delete the weights file they were reading."""
    pool = _SELFPLAY.pop("pool", None)
    path = _SELFPLAY.pop("path", None)
    if pool is not None:
        pool.terminate()
        pool.join()
    if path and os.path.exists(path):
        os.remove(path)


def collect_games(net: CuarentaNet, n_games: int, base_seed: int, workers: int = 1):
    """
    Self-play games with the current net at all four seats.

    Returns (decisions, metrics) where `metrics` averages the per-game returns,
    scores, plies and reward components (caídas/limpias) over the batch of games —
    i.e. how well the agent is actually doing, in reward terms, this iteration.

    The search is pure-Python work around batch-1 network forwards, so it holds the GIL:
    extra THREADS would only interleave, never run in parallel. Processes do, which is
    what `workers > 1` uses — one game per worker, over a pool that lives for the whole
    run and reloads the weights from a file the parent rewrites each iteration. Games are
    seeded individually (`base_seed + i`), so the batch is the same whatever the worker
    count: `workers=1` is the serial path, not a different experiment. With
    `--dropout > 0` the dropout masks still come from each process's own torch RNG, so a
    run is reproducible up to that noise.
    """
    cfg = _selfplay_cfg()
    if workers > 1 and n_games > 1:
        pool = _ensure_selfplay_pool(min(workers, n_games), _sd(net), cfg, net.training,
                                     net.value_target, net.dropout_p)
        games = list(tqdm(
            pool.imap(_selfplay_worker, [base_seed + i for i in range(n_games)],
                      chunksize=1),
            total=n_games, desc="self-play games", leave=False))
    else:
        games = []
        for i in tqdm(range(n_games), desc="self-play games", leave=False):
            # Same per-game seeding as the workers, so the two paths generate the same
            # games for the same seeds.
            np.random.seed((base_seed + i) % (2 ** 32))
            games.append(play_one_game(net, **_selfplay_kwargs(base_seed + i, cfg, DEVICE)))

    all_decisions: list[Decision] = [d for decisions, _ in games for d in decisions]
    game_results: list[GameResult] = [result for _, result in games]
    return all_decisions, mean_game_metrics(game_results, prefix="selfplay")


def _imitation_worker(payload) -> list[Decision]:
    """One teacher game. `payload` is (seed, value_target): a spawned worker gets module
    defaults, so the run's value target has to travel with the job."""
    seed, value_target = payload
    import torch
    torch.set_num_threads(1)
    return play_one_heuristic_game(seed, value_target)


def collect_imitation_games(n_games: int, base_seed: int,
                            value_target: str = "win") -> list[Decision]:
    """
    A few heuristic-vs-heuristic games per iteration, marked as NOT self-play.

    Why: self-play makes the value head an estimator of "who wins when both sides
    play like me". That estimator is exactly what the search relies on, so as soon as
    it drifts away from the value it had against other opponent types, the search's
    decisions get worse on those opponents. Mixing in teacher games keeps the head
    calibrated on the wider distribution, so self-play cannot quietly narrow what the
    search knows about other opponent types.

    These samples are given policy weight 0 by default (IMITATION_POLICY_WEIGHT), so
    they do NOT drag the policy back toward the teacher; only the value loss uses them.

    `value_target` must be the run's, because these samples carry a VALUE TARGET of their
    own: teacher games labelled with ±1 in a return-target run would be a silent scale
    error in the value loss (they are a small share of the batch, so the loss would just
    look slightly worse than it is).
    """
    from multiprocessing import Pool
    seeds = [base_seed + i for i in range(n_games)]
    with Pool() as pool:
        games = list(tqdm(pool.imap(_imitation_worker,
                                    [(s, value_target) for s in seeds]),
                          total=n_games, desc="imitation games", leave=False))
    decisions = [d for g in games for d in g]
    for d in decisions:
        d.selfplay = False
    return decisions


def mean_game_metrics(results: list[GameResult], prefix: str) -> dict:
    """
    Average a batch of GameResults into flat scalars for logging.

    This is where the "how is the agent actually doing" curves come from:
    returns (summed env rewards), scores, game length, and the reward components
    (caídas/limpias), plus the search's own mean value estimate so you can compare
    what the net believed with what actually happened.
    """
    if not results:
        return {}
    per_game = [r.as_metrics(prefix) for r in results]
    keys = per_game[0].keys()
    out = {k: float(np.mean([m[k] for m in per_game])) for k in keys}
    out[f"{prefix}/win_rate_team0"] = (
        sum(1 for r in results if r.winner == 0) / len(results)
    )
    out[f"{prefix}/n_games"] = float(len(results))
    return out


def buffer_stats(decisions: list[Decision]) -> dict:
    """
    Cheap diagnostics over the replay buffer: how sharp are the targets, how sharp is
    the policy, and do the two heads of the trainer want different things?
    """
    t_ent, p_ent, t_max, p_max, n_legal, n_sp = [], [], [], [], [], 0
    for d in decisions:
        v = d.visits.astype(np.float64)
        s = v.sum()
        if s <= 0:
            continue
        p = v / s
        nz = p[p > 0]
        t_ent.append(-(nz * np.log(nz)).sum())
        t_max.append(nz.max())
        n_legal.append(int(d.mask.sum()))
        if d.selfplay:
            n_sp += 1
        if d.priors is not None and len(d.priors) > 0:
            q = np.asarray(d.priors, dtype=np.float64)
            q = q / max(q.sum(), 1e-9)
            qnz = q[q > 0]
            p_ent.append(-(qnz * np.log(qnz)).sum())
            p_max.append(qnz.max())
    return {
        "n": len(decisions),
        "selfplay_frac": n_sp / max(len(decisions), 1),
        "target_entropy": float(np.mean(t_ent)) if t_ent else 0.0,
        "target_top_prob": float(np.mean(t_max)) if t_max else 0.0,
        "prior_entropy": float(np.mean(p_ent)) if p_ent else 0.0,
        "prior_top_prob": float(np.mean(p_max)) if p_max else 0.0,
        "legal_actions": float(np.mean(n_legal)) if n_legal else 0.0,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(net, loader, opt, device, anchor_net=None, anchor_weight=0.0):
    """
    One pass over the replay buffer.

    `anchor_net` (optional) is a frozen copy of the net as it was BEFORE self-play
    started (i.e. the imitation checkpoint), and `anchor_weight` says how strongly to
    keep the current value head close to it:

        loss = policy_loss + value_loss + anchor_weight * (v(s) - v_anchor(s))^2

    Why anchor the value head at all? Because the search VALUES LEAVES WITH IT
    (`leaf_value_mode="net"`), so a value head that drifts becomes a search that
    plays badly. The failure is one-sided: the policy can hold its strength while the
    held-out value MSE goes past 1.0 ("worse than always predicting 0"), and nothing on
    the win-rate line says why. The anchor keeps the calibrated bootstrap value while
    still letting the head adapt:

        anchor_weight = 0.0  -> train the value head freely (AlphaZero-style)
        anchor_weight = 2.0  -> strong anchor (default)
        anchor_weight 5+     -> effectively freeze the value head at the bootstrap

    The per-sample policy weights come from the dataset: samples with weight 0
    (imitation games mixed in for value calibration) do not move the policy.
    """
    net.train()
    total_loss = total_p = total_v = total_a = total_ent = 0.0
    n_batches = 0
    # Moments of the value TARGET over this pass. They turn the raw value loss into an
    # R^2 ("how much of the target's variance does the head explain IN SAMPLE?"), which is
    # the only form of that number that means the same thing in both value targets: the
    # ±1 target has variance ~1, the return target has variance ~1 too once divided by
    # RETURN_VALUE_SCALE, but the win bonus / reward shape can move it, and a hard-coded
    # `value_loss < 0.05` (the old warning) silently meant different things in each mode.
    v_sum = v_sq = 0.0
    v_n = 0

    use_anchor = anchor_net is not None and anchor_weight > 0

    for (obs, act, am, feats, mask, pi_target, v_target, length, pol_weight) in loader:
        obs = obs.to(device)
        act = act.to(device)
        am = am.to(device)
        feats = feats.to(device)
        mask = mask.to(device)
        pi_target = pi_target.to(device)
        v_target = v_target.to(device)
        length = length.to(device)
        pol_weight = pol_weight.to(device)

        logits, value = net(obs, act, am, feats, mask, lengths=length)

        # Policy loss: weighted cross-entropy between softmax(logits) and the
        # search's (possibly flattened) visit distribution.
        logp = torch.log_softmax(logits, dim=-1)
        ce = -(pi_target * logp).sum(dim=-1)
        w = pol_weight.clamp(min=0.0)
        policy_loss = (ce * w).sum() / w.sum().clamp(min=1e-6)

        # Value loss: MSE against the value target from the acting team's POV — ±1 game
        # results in win mode, the scaled shaped return-to-go in return mode (the same
        # quantity `NetworkISMCTS` backs up; see --value-target / the note in network.py).
        value_loss = ((value - v_target) ** 2).mean()

        loss = policy_loss + value_loss

        anchor_loss = torch.zeros((), device=device)
        if use_anchor:
            with torch.no_grad():
                _anchor_logits, anchor_value = anchor_net(
                    obs, act, am, feats, mask, lengths=length
                )
            anchor_loss = ((value - anchor_value) ** 2).mean()
            loss = loss + anchor_weight * anchor_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            # Target entropy is the floor of the cross-entropy: the part above it is
            # what the net actually fails to fit, so `excess_ce` is the honest
            # measure of "is the policy learning anything from these targets?".
            #
            # It must be weighted the SAME way as `policy_loss` (samples with policy
            # weight 0 — the imitation games added for value calibration — contribute
            # to neither), otherwise the two numbers are not comparable and
            # excess_ce is biased by however much imitation data is in the batch.
            ent = -(pi_target * torch.log(pi_target + 1e-9)).sum(dim=-1)
            total_ent += ((ent * w).sum() / w.sum().clamp(min=1e-6)).item()

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        total_a += anchor_loss.item()
        n_batches += 1
        with torch.no_grad():
            vt = v_target.to(torch.float64)
            v_sum += vt.sum().item()
            v_sq += (vt ** 2).sum().item()
            v_n += vt.numel()

    n = max(n_batches, 1)
    v_mean = v_sum / max(v_n, 1)
    v_var = max(v_sq / max(v_n, 1) - v_mean ** 2, 0.0)
    # R^2 = 1 - mse/variance. 0.0 means "explains nothing" (predicting the mean),
    # negative means worse than the mean, and the caller compares it against the
    # HELD-OUT version from the probe to spot memorisation.
    v_r2 = 1.0 - (total_v / n) / v_var if v_var > 0 else 0.0
    return (total_loss / n, total_p / n, total_v / n, total_a / n, total_ent / n,
            v_r2, v_var)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
#
# Every evaluator plays a game in a worker process and returns the winning team.
# Team 0 is always the net under test, so `win rate == fraction of games where the
# winner was team 0`.
#
# Configuration is passed explicitly instead of being read from module globals:
# evaluation runs in worker PROCESSES, and a spawned worker only sees the module
# defaults, not the CLI overrides.

def _make_search(env, net, tracker, seed, *, n_iter, leaf_value_mode, max_rollout_len,
                 value_target=None):
    """
    The exact search used for self-play, minus exploration noise.

    `value_target` is normally left None: the search takes it from the net, which is the
    only way the search and the head it reads can stay in the same units. It is accepted
    here because `search_cfg` carries the run's setting and passing it is a cheap way to
    make an eval worker's configuration explicit.
    """
    return NetworkISMCTS(
        env, net, tracker,
        n_iter=n_iter,
        max_history_len=T_MAX,
        leaf_value_mode=leaf_value_mode,
        max_rollout_len=max_rollout_len,
        rng=random.Random(seed),
        dirichlet_eps=0.0,          # no noise at eval: deterministic given n_iter
        device="cpu",               # workers run on CPU
        value_target=value_target,
    )


def _load_net(state_dict, value_target="win", dropout: float = 0.0):
    """
    Build a net and load `state_dict` into it.

    `value_target` picks the head's activation, which is NOT part of the state_dict, so
    every evaluation worker has to be told it (they only see module defaults, never the
    parent's CLI values — that is what `search_cfg` is for). `dropout` is the same kind
    of setting, and only matters where the net acts in train mode (self-play workers):
    evaluation builds the net in eval mode, where dropout is the identity.
    """
    net = CuarentaNet(value_target=value_target, dropout=dropout, hidden=128)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _pick_visit_action(searcher, p, legal):
    """Argmax over visit counts, with a random fallback if the search had no budget."""
    visits = searcher.search(root_player=p)
    if sum(visits.values()) == 0:
        return random.randrange(len(legal))
    return max(visits, key=visits.get)


def _play_match(env, actor, trackers):
    """
    Play one game to the end and return its scored summary (a dict).

    `actor(player, obs, legal)` returns the action SLOT to play; `trackers` is the
    list of HistoryTrackers that must observe every decision (one per searcher — each
    search keeps its own view of the game).

    The summary carries everything worth logging about a match: winner, final scores,
    length, and per-team RETURNS (= summed env rewards) with their components.
    """
    from utils import PointsReasons, has_reason

    returns = [0.0, 0.0]
    caidas = [0, 0]
    limpias = [0, 0]
    plies = 0

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        a = actor(p, obs, legal)
        for tr in trackers:
            tr.record(p, obs, a)
        _obs, reward, _done, _trunc, info = env.step(a)

        t = TEAMS[p]
        returns[t] += float(reward)
        reasons = info.get("reasons") or []
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[t] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[t] += 1
        plies += 1

    return {
        "winner": env.state.winner,
        "score0": int(env.state.scores[0]),
        "score1": int(env.state.scores[1]),
        "plies": plies,
        "return0": returns[0],
        "return1": returns[1],
        "caidas0": caidas[0],
        "caidas1": caidas[1],
        "limpias0": limpias[0],
        "limpias1": limpias[1],
    }


def _eval_reference_worker(args):
    """
    One game: current net (team 0) vs REFERENCE checkpoint (team 1), BOTH searching.

    This is the AlphaZero gating match: same search, same rules, only the weights
    differ, so beating the reference means the training actually improved the agent
    (as opposed to merely moving away from a fixed opponent).
    """
    seed, cur_sd, ref_sd, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    cur = _load_net(cur_sd, cfg.get("value_target", "win"))
    ref = _load_net(ref_sd, cfg.get("value_target", "win"))
    tracker_cur = HistoryTracker(env)
    tracker_ref = HistoryTracker(env)
    searcher_cur = _make_search(env, cur, tracker_cur, seed, **cfg)
    searcher_ref = _make_search(env, ref, tracker_ref, seed, **cfg)

    def actor(p, obs, legal):
        if TEAMS[p] == 0:
            return _pick_visit_action(searcher_cur, p, legal)
        return _pick_visit_action(searcher_ref, p, legal)

    return _play_match(env, actor, [tracker_cur, tracker_ref])


def _eval_heuristic_worker(args):
    """One game: net (team 0) vs HeuristicAgent (team 1)."""
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from heuristic_agent import HeuristicAgent
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict, cfg.get("value_target", "win"))
    tracker = HistoryTracker(env)
    searcher = _make_search(env, net, tracker, seed, **cfg)
    heuristic = HeuristicAgent(seed=seed * 2 + 2)

    def actor(p, obs, legal):
        if TEAMS[p] == 0:
            return _pick_visit_action(searcher, p, legal)
        return legal.index(heuristic.act(env, p))

    return _play_match(env, actor, [tracker])


def _eval_random_worker(args):
    """
    One game: net WITH search (team 0) vs a UNIFORM-RANDOM legal move (team 1).

    The sanity floor of the evaluation league (the same member PPO logs as `random`).
    Nothing should ever lose to it: if this win rate is not high, a bug in the eval
    plumbing — not the net — is the first suspect.
    """
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict, cfg.get("value_target", "win"))
    tracker = HistoryTracker(env)
    searcher = _make_search(env, net, tracker, seed, **cfg)
    # Seeded per game, so a given seed always produces the same opponent moves.
    rng = random.Random(seed * 2 + 3)

    def actor(p, obs, legal):
        if TEAMS[p] == 0:
            return _pick_visit_action(searcher, p, legal)
        return rng.randrange(len(legal))

    return _play_match(env, actor, [tracker])


def requested_members(args) -> tuple:
    """
    Which evaluation members this run asked for.

    `--members a,b,c` names them; `--no-league` takes `NO_LEAGUE_MEMBERS`; neither takes
    `DEFAULT_MEMBERS`. `reference` is mandatory because the acceptance gate reads it --
    everything else can be dropped, which is the point: every member costs its own games.
    """
    resolved = getattr(args, "members_resolved", None)
    if resolved is not None:
        return resolved
    explicit = getattr(args, "members", None)
    if explicit and getattr(args, "no_league", False):
        raise SystemExit("--members and --no-league are mutually exclusive: name the "
                         "members you want, or take the --no-league set.")
    if explicit:
        names = tuple(n.strip() for n in explicit.split(",") if n.strip())
        if not names:
            raise SystemExit("--members was given but is empty")
        unknown = [n for n in names if n not in ALL_MEMBERS]
        if unknown:
            raise SystemExit(f"unknown --members {unknown}; valid names are: "
                             f"{', '.join(ALL_MEMBERS)}")
        if "reference" not in names:
            raise SystemExit("`reference` must be evaluated: it is the member the "
                             "acceptance gate reads. Everything else can be dropped.")
        return names
    return NO_LEAGUE_MEMBERS if getattr(args, "no_league", False) else DEFAULT_MEMBERS


def league_matches(args, cur_state, initial_state, prev_checkpoint_state, cfg):
    """
    The REST of the evaluation league: everything except the two gate matches.

    These three members are the ones PPO also logs, under the same names and with the
    same suffixes, so runs from both pipelines can be compared on one wandb project:

      * `initial`         — the frozen weights this run STARTED from: "did training
                            actually improve on what we began with?". Taken from the
                            in-memory copy `main` makes at startup, so no checkpoint
                            file is read again.
      * `prev_checkpoint` — the snapshot written by the PREVIOUS save event (i.e. the
                            weights from before this iteration's training), which is
                            what PPO passes as `prev_checkpoint`. `None` on the first
                            evaluation of a run, and then the member is SKIPPED
                            entirely instead of being logged with stale numbers.
      * `random`          — uniform-random legal moves.

    Each plays `--league-games` games (default 30) instead of `--eval-games`, because
    only `reference` drives the acceptance gate and needs the tighter standard error.
    `--no-league` drops `initial` and `prev_checkpoint`; `random` is ALWAYS
    evaluated, because it is the sanity floor every pipeline reports and it needs no
    weights at all.

    Returns a list of `(name, worker, payloads)`; every payload is picklable, so the
    caller can hand it straight to a worker pool.
    """
    n = args.league_games if args.league_games > 0 else LEAGUE_GAMES
    wanted = set(requested_members(args))
    if not ({"initial", "prev_checkpoint", "random"} & wanted):
        return []
    specs = []
    if "initial" in wanted and initial_state is not None:
        specs.append(("initial", _eval_reference_worker, initial_state, 81000))
    if "prev_checkpoint" in wanted and prev_checkpoint_state is not None:
        specs.append(("prev_checkpoint", _eval_reference_worker,
                      prev_checkpoint_state, 82000))
    # Always available: it needs no weights at all. `random` is why --no-league still
    # reports a floor -- it is the member every pipeline shares and it costs nothing extra.
    if "random" in wanted:
        specs.append(("random", _eval_random_worker, None, 83000))

    out = []
    for name, worker, state, seed_base in specs:
        # `state is None` means the opponent is not a checkpoint at all (random), so the
        # worker takes a 3-tuple payload instead of a 4-tuple.
        if state is None:
            payloads = [(seed_base + i, cur_state, cfg) for i in range(n)]
        else:
            payloads = [(seed_base + i, cur_state, state, cfg) for i in range(n)]
        out.append((name, worker, payloads))
    return out


def _eval_raw_worker(args):
    """
    One game: net WITH search (team 0) vs the SAME net's raw policy (team 1).

    The raw side plays argmax of its own prior (`_eval_root`, no search), so this
    measures the teacher edge — how much the search actually adds over the policy it
    is supposed to be teaching. If this drifts toward 50%, the self-play targets are
    self-distillation and the net cannot improve from them.
    """
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict, cfg.get("value_target", "win"))
    tracker = HistoryTracker(env)
    searcher = _make_search(env, net, tracker, seed, **cfg)

    def actor(p, obs, legal):
        if TEAMS[p] == 0:
            return _pick_visit_action(searcher, p, legal)
        priors, _v = searcher._eval_root(p, legal)
        return int(np.argmax(priors))

    return _play_match(env, actor, [tracker])


def _probe_worker(args):
    """
    Metrics on ONE fixed heuristic game (never used for training): policy top-1
    agreement with the teacher, value MSE/sign accuracy, and mean |value|.

    Runs the net directly (no search, no self-play), so it is a cheap, deterministic
    probe of the two heads that is comparable across iterations. `value_mse` should
    stay at or below ~1.0; `mean_abs_value` flags overconfidence (a calibrated
    estimate of a mostly-undecided game should not sit near ±1).
    """
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from torch.utils.data import DataLoader
    from dataset import CuarentaDataset

    decisions = play_one_heuristic_game(seed, cfg.get("value_target", "win"))
    net = _load_net(state_dict, cfg.get("value_target", "win"))
    loader = DataLoader(CuarentaDataset(decisions, t_max=T_MAX, history_dropout=0.0),
                        batch_size=256)
    correct = v_se = v_sign = abs_v = 0.0
    v_sum = v_sq = 0.0      # moments of the TARGET, for the scale-free MSE baseline
    n = 0
    with _torch.no_grad():
        for obs, act, am, feats, mask, pi, v, length, _pw in loader:
            logits, value = net(obs, act, am, feats, mask, lengths=length)
            correct += (logits.argmax(-1) == pi.argmax(-1)).sum().item()
            v = v.to(_torch.float64)
            v_se += ((value - v) ** 2).sum().item()
            v_sign += ((value > 0) == (v > 0)).sum().item()
            abs_v += value.abs().to(_torch.float64).sum().item()
            v_sum += v.sum().item()
            v_sq += (v ** 2).sum().item()
            n += v.shape[0]
    return {"correct": correct, "v_se": v_se, "v_sign": v_sign, "abs_v": abs_v,
            "v_sum": v_sum, "v_sq": v_sq, "n": n}


def _pooled(worker, args, desc):
    """Run worker(args) for every args entry in parallel. Returns the result list."""
    from multiprocessing import Pool
    with Pool() as pool:
        return list(tqdm(pool.imap(worker, args), total=len(args), desc=desc,
                         leave=False))


def _sd(net):
    """CPU copy of a net's weights, safe to hand to worker processes."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# Canonical league name -> the legacy name this trainer used to log as well. The aliases
# are gone (see `_with_aliases`), so the mapping is empty and the call sites log one
# number per member.
LEGACY_ALIASES: dict = {}


def match_metrics(results: list[dict], prefix: str) -> dict:
    """
    Average a batch of match summaries into flat scalars.

    Team 0 is always the net under test, so the naming is from its point of view:
    `_for` = the net's team, `_against` = the opponent. Logged for every matchup, so
    you can watch the RETURNS (summed env rewards), the actual game SCORES, the game
    length and the caída/limpia counts — not just who won.

    CONVENTION WARNING (differs from PPO): the net under test always plays TEAM 0 and
    the opponent always plays TEAM 1, so any first-mover/seat edge is baked into these
    numbers (which is exactly why gating is calibrated against a mirror baseline, see
    `main`). `src/ppo/eval_harness.py` instead ROTATES the policy over the four seats
    and reports the same key names. The names therefore compare, but a mirror match is
    ~50% in expectation for PPO; here it is whatever the seat edge makes it. This
    convention is deliberately unchanged: changing it would silently change what the
    existing `vs_*` curves mean.

    `_stderr` (binomial standard error of the win rate) and `_n_games` are part of the
    contract with the PPO pipeline — a win rate without its `n` cannot be compared
    across runs, because a 30-game league match and a 60-game gate match have very
    different error bars.
    """
    if not results:
        return {}
    n = len(results)

    def m(key):
        return float(np.mean([r[key] for r in results]))

    win_rate = sum(1 for r in results if r["winner"] == 0) / n
    stderr = float(np.sqrt(max(win_rate * (1.0 - win_rate), 1e-9) / n))

    # Keys are unprefixed here (the gating code and the printout use them directly);
    # run_evaluation adds the "eval/" prefix when logging to wandb.
    return {
        prefix: win_rate,
        f"{prefix}_stderr": stderr,
        f"{prefix}_score_for": m("score0"),
        f"{prefix}_score_against": m("score1"),
        f"{prefix}_plies": m("plies"),
        f"{prefix}_return_for": m("return0"),
        f"{prefix}_return_against": m("return1"),
        f"{prefix}_caidas_for": m("caidas0"),
        f"{prefix}_caidas_against": m("caidas1"),
        f"{prefix}_limpias_for": m("limpias0"),
        f"{prefix}_limpias_against": m("limpias1"),
        f"{prefix}_n_games": float(n),
    }


def _with_aliases(metrics: dict, canonical: str, legacy: str | None) -> dict:
    """
    Kept for its call sites; the legacy `vs_*` aliases were REMOVED.

    The canonical member names (`reference`, `heuristic`, `initial`, `prev_checkpoint`,
    `random`) are the ones every pipeline logs (see `src/ppo/eval_harness.py:
    match_metrics` and `src/rebel/eval_rebel.py`), so runs overlay on one wandb project.
    The `vs_reference` / `vs_heuristic` duplicates this trainer used to emit alongside
    them were dropped: with the canonical names in place they were only noise in the
    console and a second name for every series. Nothing reads them any more — the
    printed summary and the acceptance gate use the canonical keys.
    """
    return metrics


def head_probe_metrics(state_dict, cfg, desc: str = "eval head probe") -> dict:
    """
    Held-out head metrics on fixed, never-trained-on games (see `_probe_worker`).

    Runs on its own so it can be called EVERY iteration, not only when the full league is
    evaluated. That matters: the failure this measures (a value head that memorises the
    replay buffer and is then worse than useless on fresh games, while the search — which
    uses that head to value its leaves — quietly degrades) shows up in the held-out numbers
    several iterations before it shows up in the win rate. The probe costs ~20 games of
    teacher play and no search, so it is cheap next to a full evaluation.

    Keys (all under `probe_*`): policy top-1 vs the teacher, value MSE/sign, mean |value|,
    and the moments of the value TARGET, from which `probe_value_r2` is computed — the
    scale-free statement of "does this head know anything on fresh games?" (0 = no better
    than predicting the mean, negative = worse).
    """
    cur = state_dict
    results = _pooled(_probe_worker, [(s, cur, cfg) for s in PROBE_SEEDS], desc)
    n = sum(r["n"] for r in results)
    v_mean = sum(r["v_sum"] for r in results) / max(n, 1)
    v_var = max(sum(r["v_sq"] for r in results) / max(n, 1) - v_mean ** 2, 0.0)
    mse = sum(r["v_se"] for r in results) / max(n, 1)
    abs_v = sum(r["abs_v"] for r in results) / max(n, 1)
    return {
        "probe_policy_acc": sum(r["correct"] for r in results) / max(n, 1),
        "probe_value_mse": mse,
        "probe_value_sign": sum(r["v_sign"] for r in results) / max(n, 1),
        "probe_mean_abs_value": abs_v,
        "probe_target_mean": v_mean,
        "probe_target_var": v_var,
        # Held-out R^2 of the value head: the single number to watch. Scale-free, so it
        # reads the same in both value targets (see network.RETURN_VALUE_SCALE).
        "probe_value_r2": 1.0 - mse / v_var if v_var > 0 else 0.0,
        # How LOUD the head is relative to the spread of what it is predicting. A head
        # with no held-out skill that predicts at ~1x the target's spread is claiming a
        # confidence the data does not support, and in return mode the search trusts it
        # completely (a capture is worth 1/30 of a unit; the leaf value is worth ~1).
        "probe_value_confidence": abs_v / (v_var ** 0.5) if v_var > 0 else 0.0,
    }


def save_checkpoint(net, path: str, *, iteration: int, value_target: str,
                    return_value_scale: float, extra: dict | None = None) -> None:
    """
    Save `net` to `path` and write the provenance sidecar `path + ".json"` next to it.

    WHY A SIDECAR: the value target and its scale are NOT in the weights — the head differs
    only by an activation, so a return-mode checkpoint loads into a win-mode net (and vice
    versa) without an error, and the head's units are then silently wrong. Everything
    downstream that reads a checkpoint has to know: the trainer's own `--init-checkpoint`
    (warm-starting across modes wastes iterations), and above all the ONNX export, which has
    to build the same head the weights were trained with or the backend serves a squashed
    value. Recording it here is what makes those paths able to check instead of guess.
    """
    torch.save(net.state_dict(), path)
    meta = {
        "architecture": f"cuarenta_net(value_target={value_target})",
        "value_target": value_target,
        "return_value_scale": float(return_value_scale),
        "hidden": net.hidden,
        "obs_dim": net.obs_dim,
        "n_actions": net.n_actions,
        "action_feat_dim": net.action_feat_dim,
        "dropout": float(getattr(net, "dropout_p", 0.0)),
        "iteration": int(iteration),
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    with open(path + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)


def read_checkpoint_meta(path: str) -> dict:
    """
    Provenance of a checkpoint, or {} when it has none (every checkpoint trained before the
    sidecar existed). Callers must treat an empty dict as "unknown", never as "win mode".
    """
    try:
        with open(path + ".json") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def run_evaluation(net, reference_state, args, cfg, iteration,
                   initial_state=None, prev_checkpoint_state=None):
    """
    All the evaluations for one iteration. Returns a dict of flat metrics.

    `initial_state` / `prev_checkpoint_state` are the extra league members (see
    `league_matches`); both are optional so existing callers keep working, and a
    missing one simply removes that member from the league.
    """
    cur = _sd(net)
    wanted = set(requested_members(args))
    metrics = {}

    # 1. Gating match vs the reference checkpoint (primary criterion). Logged under
    # the canonical name `reference` — the name PPO uses — plus the legacy
    # canonical `reference` name the gating code and the dashboards read.
    results = _pooled(
        _eval_reference_worker,
        [(50000 + i, cur, reference_state, cfg) for i in range(args.eval_games)],
        "eval vs reference",
    )
    metrics.update(_with_aliases(match_metrics(results, "reference"),
                                 "reference", None))

    # 2. External reference: the heuristic agent (canonical `heuristic`)
    if "heuristic" in wanted:
        results = _pooled(
            _eval_heuristic_worker,
            [(50000 + i, cur, cfg) for i in range(args.eval_games)],
            "eval vs heuristic",
        )
        metrics.update(_with_aliases(match_metrics(results, "heuristic"),
                                     "heuristic", None))

    # 2b. The rest of the league: `initial`, `prev_checkpoint`, `random` (each
    # `--league-games` games; `--no-league` skips them). They are cheap fillers for
    # cross-pipeline comparison, not gating criteria.
    league_names = []
    for name, worker, payloads in league_matches(args, cur, initial_state,
                                                 prev_checkpoint_state, cfg):
        results = _pooled(worker, payloads, f"eval vs {name}")
        metrics.update(match_metrics(results, name))
        league_names.append(name)

    # 3. Health: teacher edge (search vs this net's own raw policy). On by default: it is
    # the expert-iteration health check, and the earliest warning that the search has
    # stopped improving on the policy.
    if "teacher_edge" in wanted:
        results = _pooled(
            _eval_raw_worker,
            [(70000 + i, cur, cfg) for i in range(args.eval_raw_games)],
            "eval teacher edge",
        )
        metrics.update(match_metrics(results, "teacher_edge"))

    # 4. Health: held-out head metrics on fixed, never-trained-on games
    metrics.update(head_probe_metrics(cur, cfg))

    # Only the members this run asked for are in `metrics`, so the summary names the ones
    # that ran rather than assuming the default set.
    shown = [f"reference={metrics['reference']:.1%}"]
    for _name in ("heuristic", "teacher_edge"):
        if _name in metrics:
            shown.append(f"{_name}={metrics[_name]:.1%}")
    print("  eval: " + "  ".join(shown))
    if league_names:
        print("        league: " + "  ".join(
            f"{name}={metrics[name]:.1%} (n={int(metrics[f'{name}_n_games'])})"
            for name in league_names))
    # The detail line follows the same rule as the win-rate line above: it prints what the
    # run actually evaluated. `heuristic` is optional (a trimmed `--members` set may not
    # contain it), so its block is included only when those keys exist, and the reference
    # numbers are then shown in its place instead of a KeyError.
    has_heur = "heuristic" in metrics
    detail_member = "heuristic" if has_heur else "reference"
    label = "heur" if has_heur else "ref"
    print(f"        scores: vs_ref {metrics['reference_score_for']:.1f}-"
          f"{metrics['reference_score_against']:.1f}  "
          f"{label} {metrics[f'{detail_member}_score_for']:.1f}-"
          f"{metrics[f'{detail_member}_score_against']:.1f}  "
          f"returns: vs_{label} {metrics[f'{detail_member}_return_for']:.1f}-"
          f"{metrics[f'{detail_member}_return_against']:.1f}  "
          f"caídas {metrics[f'{detail_member}_caidas_for']:.2f}-"
          f"{metrics[f'{detail_member}_caidas_against']:.2f}  "
          f"plies {metrics[f'{detail_member}_plies']:.0f}  "
          f"n {int(metrics['reference_n_games'])}"
          + (f"/{int(metrics['heuristic_n_games'])}" if has_heur else ""))
    probe_mode = cfg.get("value_target", "win")
    probe_scale = (RETURN_VALUE_SCALE if probe_mode == "return" else 1.0)
    print(f"        probe: policy_acc={metrics['probe_policy_acc']:.3f}  "
          f"value_mse={metrics['probe_value_mse']:.3f}  "
          f"value_sign={metrics['probe_value_sign']:.3f}  "
          f"mean|v|={metrics['probe_mean_abs_value']:.3f}"
          + (f"  [return mode, game-return units: mse="
             f"{metrics['probe_value_mse'] * probe_scale ** 2:.1f}, "
             f"mean_pred={metrics['probe_target_mean'] * probe_scale:.1f}]"
             if probe_mode == "return" else ""))

    # "Worse than always predicting the target's mean", stated scale-free. With ±1
    # targets this IS the old hard-coded `mse > 1.0` (their variance is ~1.0); with return
    # targets the variance is ~0.05-0.2, so a fixed threshold would never fire.
    if metrics["probe_value_mse"] > max(metrics["probe_target_var"], 1e-6):
        print("        WARNING: held-out value MSE is above the target's own variance — "
              "the value head is worse than always predicting the mean, and the search "
              "uses it to value leaves. Raise --value-anchor-weight, cut "
              "--epochs-per-iter, or add games.")
    if metrics.get("teacher_edge", 1.0) < 0.55:
        print("        WARNING: the search barely beats this net's own raw policy — "
              "the self-play targets are close to self-distillation, so expect the "
              "policy to stop improving.")
    if probe_mode == "win" and metrics["probe_value_mse"] > 0 \
            and metrics["probe_mean_abs_value"] > 0.7:
        print("        NOTE: mean |value| is high — the value head is confident. "
              "Compare it with the actual win rates; overconfidence shows up as a "
              "probe value MSE that creeps toward 1.0.")
    elif probe_mode == "return" and metrics["probe_target_mean"] * probe_scale > probe_scale:
        print("        NOTE: the value head predicts an expected return above "
              "RETURN_VALUE_SCALE — higher than any game it has been shown. Compare "
              "mean_pred above with the self-play returns.")

    if not args.no_wandb:
        # `iteration` is the x-axis shared by all three pipelines (PPO and ReBeL log
        # the same metric); `iter` is kept alongside it, with the same value, so
        # dashboards built on the old name keep working.
        wandb.log(dict({f"eval/{k}": v for k, v in metrics.items()},
                       iteration=iteration, iter=iteration))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # `global` must come before any use of these names in this function.
    global N_SEARCH_ITER, LEAF_VALUE_MODE, DIRICHLET_EPS, TARGET_TEMPERATURE, DEVICE
    global VALUE_TARGET, REPLAY_ITERS, VALUE_LAMBDA

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Checkpoint to start from (e.g. checkpoints/bootstrap.pt). "
                             "Checkpoints saved before the history/feature fixes are "
                             "incompatible and will fail to load.")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=GAMES_PER_ITER)
    parser.add_argument("--games-workers", type=int, default=SELFPLAY_WORKERS,
                        help="processes that play the self-play games (0 = one per CPU, "
                             "capped at --games-per-iter; 1 = serial). The games are "
                             "seeded individually, so this changes the speed, not the "
                             "batch")
    parser.add_argument("--imitation-games", type=int, default=IMITATION_GAMES_PER_ITER,
                        help="Heuristic games per iteration, mixed in for value "
                             "calibration (0 disables)")
    parser.add_argument("--imitation-policy-weight", type=float,
                        default=IMITATION_POLICY_WEIGHT,
                        help="Policy-loss weight of imitation samples "
                             "(0 = value loss only, 1 = also imitate the teacher)")
    parser.add_argument("--n-search-iter", type=int, default=N_SEARCH_ITER)
    parser.add_argument("--dirichlet-eps", type=float, default=DIRICHLET_EPS)
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE,
                        help="Flatten the search's visit distribution before training "
                             "on it (1.0 = raw)")
    parser.add_argument("--epochs-per-iter", type=int, default=EPOCHS_PER_ITER,
                        help="training epochs per iteration over the replay buffer. This "
                             "× --replay-iters is how many times each sample is replayed, "
                             "and THAT is what lets a 1.6M-parameter net memorise ~20k "
                             "decisions. Cut it before adding regularisation.")
    parser.add_argument("--replay-iters", type=int, default=REPLAY_ITERS,
                        help="how many iterations of data to keep in the replay buffer "
                             "(default %(default)s)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--leaf-value-mode", type=str, default=LEAF_VALUE_MODE,
                        choices=["net", "rollout"])
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES,
                        help="Games for the two GATE matches (reference, heuristic) — "
                             "these drive acceptance, so they want a tight stderr")
    parser.add_argument("--eval-raw-games", type=int, default=EVAL_RAW_GAMES)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--league-games", type=int, default=LEAGUE_GAMES,
                        help="Games for each of the extra league members "
                             "(initial, prev_checkpoint, random). Cheaper than "
                             "--eval-games because they are for comparison, not gating")
    parser.add_argument("--members", type=str, default=None,
                        help="comma-separated evaluation members to run; any of "
                             + ",".join(ALL_MEMBERS) + ". Default: "
                             + ",".join(DEFAULT_MEMBERS) + ". `reference` is required "
                             "(the gate reads it). Each member costs its own games, so "
                             "this is the knob for cutting evaluation time. Mutually "
                             "exclusive with --no-league.")
    parser.add_argument("--no-league", action="store_true",
                        help="Skip the extra league members entirely (initial, "
                             "prev_checkpoint, random) and only log reference/heuristic")
    parser.add_argument("--accept-threshold", type=float, default=ACCEPT_THRESHOLD,
                        help="Win rate vs the reference required to accept a new net")
    parser.add_argument("--value-target", type=str, default=VALUE_TARGET,
                        choices=list(VALUE_TARGETS),
                        help="what the value head learns and what the search optimizes: "
                             "`win` (default) = the ±1 game result, the AlphaZero "
                             "contract where a terminal node is worth ±1 and per-step "
                             "rewards are ignored; `return` = the env's dense reward as a "
                             "scaled shaped return-to-go, which makes the search optimize "
                             "the reward instead of the win rate. `return` changes the "
                             "head's activation, and existing checkpoints are win-mode: "
                             "do not warm-start it from one.")
    parser.add_argument("--value-anchor-weight", type=float, default=VALUE_ANCHOR_WEIGHT,
                        help="Keep the value head close to the starting checkpoint's "
                             "value output (0 = off, 5+ = almost frozen). Protects "
                             "the leaf values the search depends on.")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="dropout on the shared state embedding of the training net "
                             "(default %(default)s = off). Regularises the two heads "
                             "against the replay buffer, which is replayed many times per "
                             "iteration (see the value-R^2 warning). It is identity in "
                             "eval mode, so the search, the evaluators and the ONNX "
                             "export are unaffected.")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR,
                        help="Where to write checkpoints (relative to the CWD)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="auto = cuda if available else cpu (MPS is slower here)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--value-lambda", type=float, default=VALUE_LAMBDA,
                        help="RETURN mode only: blend the Monte-Carlo value target with "
                             "the net's own estimate using TD(lambda) "
                             "(1.0 = pure Monte-Carlo, 0.0 = one-step TD). Lower values "
                             "trade the target's noise for the head's bias, which is what "
                             "stops the head from memorising the replay buffer. Ignored in "
                             "win mode.")
    parser.add_argument("--probe-every", type=int, default=PROBE_EVERY,
                        help="Run the held-out head probe every N iterations "
                             "(default %(default)s: every iteration). The probe is ~20 "
                             "teacher games, no search, and it is the EARLIEST signal "
                             "that the value head has stopped generalising — by the time "
                             "the win rate moves it is late. Set to 0 to only run it "
                             "inside the full evaluation (--eval-every).")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the initial baseline evaluation")
    args = parser.parse_args()

    # Self-play reads these globals; the evaluators get them explicitly via `cfg`
    # (worker processes would otherwise only see the module defaults).
    N_SEARCH_ITER = args.n_search_iter
    LEAF_VALUE_MODE = args.leaf_value_mode
    VALUE_TARGET = args.value_target
    REPLAY_ITERS = args.replay_iters
    VALUE_LAMBDA = args.value_lambda
    DIRICHLET_EPS = args.dirichlet_eps
    TARGET_TEMPERATURE = args.target_temperature
    DEVICE = _pick_device(args.device)
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Self-play device: {DEVICE} (eval workers always use CPU)")
    print(f"Checkpoints -> {os.path.abspath(ckpt_dir)}")

    search_cfg = {
        "n_iter": N_SEARCH_ITER,
        "leaf_value_mode": LEAF_VALUE_MODE,
        "max_rollout_len": MAX_ROLLOUT_LEN,
        "value_target": VALUE_TARGET,
    }

    if not args.no_wandb:
        wandb.init(
            project="cuarenta-ai",
            name="ismcts",
            config={
                "n_iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "imitation_games": args.imitation_games,
                "imitation_policy_weight": args.imitation_policy_weight,
                "n_search_iter": N_SEARCH_ITER,
                "dirichlet_eps": DIRICHLET_EPS,
                "target_temperature": TARGET_TEMPERATURE,
                "leaf_value_mode": LEAF_VALUE_MODE,
                "max_rollout_len": MAX_ROLLOUT_LEN,
                "value_target": VALUE_TARGET,
                "temperature_cutoff": TEMPERATURE_CUTOFF,
                "epochs_per_iter": args.epochs_per_iter,
                "replay_iters": args.replay_iters,
                "batch_size": BATCH_SIZE,
                "lr": args.lr,
                "t_max": T_MAX,
                "history_dropout": HISTORY_DROPOUT,
                "eval_games": args.eval_games,
                "eval_members": ",".join(requested_members(args)),
                "league_games": args.league_games,
                "no_league": args.no_league,
                "eval_raw_games": args.eval_raw_games,
                "probe_every": args.probe_every,
                "accept_threshold": args.accept_threshold,
                "value_anchor_weight": args.value_anchor_weight,
                "init_checkpoint": args.init_checkpoint,
                "value_target": VALUE_TARGET,
                "value_lambda": VALUE_LAMBDA,
                "return_value_scale": RETURN_VALUE_SCALE,
                "dropout": args.dropout,
                "device": DEVICE,
            },
        )
        # X-AXIS: all three pipelines (ISMCTS, PPO, ReBeL) log `iteration` and declare it
        # here, so runs from different pipelines can be overlaid on ONE chart. The old
        # `iter` key is still logged with the same value by every log() call below, so
        # dashboards built on the previous name keep their x-axis.
        wandb.define_metric("iteration")
        wandb.define_metric("iter")
        for group in ("train", "buffer", "selfplay", "eval", "time"):
            wandb.define_metric(f"{group}/*", step_metric="iteration")

    # net = CuarentaNet(value_target=VALUE_TARGET, dropout=args.dropout).to(DEVICE)
    net = CuarentaNet(value_target=VALUE_TARGET, dropout=args.dropout, hidden=128).to(DEVICE)

    if args.init_checkpoint:
        # Loads with a clear error if the checkpoint predates the history fixes.
        CuarentaNet.load_checkpoint(net, args.init_checkpoint)
        print(f"Initialized net from {args.init_checkpoint}")

        # The value target and its scale are NOT in the weights: the head differs only by
        # an activation, so a checkpoint from the other mode loads without complaint and
        # its output is then in the wrong units for the search. The sidecar this trainer
        # writes next to every checkpoint is the only way to check, so use it when present
        # and say plainly when it is not.
        _meta = read_checkpoint_meta(args.init_checkpoint)
        if not _meta:
            print("  NOTE: no provenance sidecar beside this checkpoint (it predates them), "
                  "so its value target and scale cannot be verified. If it was not trained "
                  "with the SAME mode and scale as this run, the value head is in the wrong "
                  "units and the first iterations will be wasted relearning it.")
        elif (_meta.get("value_target") != VALUE_TARGET
              or abs(float(_meta.get("return_value_scale", RETURN_VALUE_SCALE))
                     - RETURN_VALUE_SCALE) > 1e-9):
            print(f"  WARNING: {args.init_checkpoint} was trained with value_target="
                  f"{_meta.get('value_target')!r} / scale={_meta.get('return_value_scale')!r}"
                  f", this run uses {VALUE_TARGET!r} / {RETURN_VALUE_SCALE:g}. The weights "
                  "load anyway (only the head's activation differs) and the value output is "
                  "in the WRONG UNITS. Start fresh, or match both settings.")
        else:
            print(f"  provenance OK: value_target={_meta.get('value_target')!r} "
                  f"scale={_meta.get('return_value_scale')!r} "
                  f"(saved at iteration {_meta.get('iteration')})")
    else:
        print("Starting from random initialization")
    print(f"Value target: {VALUE_TARGET} "
          + (f"(search accumulates the dense reward; linear head; "
             f"RETURN_VALUE_SCALE={RETURN_VALUE_SCALE:g} puts the targets at unit variance, "
             f"so value loss is comparable to win mode)"
             if VALUE_TARGET == "return" else
             "(±1 game results; search ignores per-step rewards)"))
    if VALUE_TARGET == "return":
        print(f"Value targets: TD(lambda={VALUE_LAMBDA:g})"
              + ("  [pure Monte-Carlo returns]" if VALUE_LAMBDA >= 1.0 else
                 "  [bootstrapped by the net; less noise, inherits the head's bias]"))
    elif VALUE_LAMBDA != 1.0:
        print(f"--value-lambda {VALUE_LAMBDA:g} ignored: it only applies to "
              "--value-target return")
    if args.dropout > 0:
        print(f"Dropout on the shared embedding: {args.dropout:g} (training only)")
    print(f"Replay: {args.replay_iters} iterations x {args.epochs_per_iter} epochs = "
          f"{args.replay_iters * args.epochs_per_iter} passes over each sample")

    # ---- Value anchor ----
    # Frozen copy of the starting net. Its value output is the calibration reference
    # for the anchor term in the loss; see train_one_epoch for why the head is anchored
    # at all. Only created when we actually start from a checkpoint: anchoring to
    # a random net's value would be anchoring to noise.
    anchor_net = None
    anchor_weight = args.value_anchor_weight
    if anchor_weight > 0:
        if args.init_checkpoint:
            anchor_net = copy.deepcopy(net).eval()
            for p in anchor_net.parameters():
                p.requires_grad_(False)
            print(f"Value anchor: ON (weight={anchor_weight}) vs {args.init_checkpoint}")
        else:
            anchor_weight = 0.0
            print("Value anchor requested but no --init-checkpoint: disabling it")

    # ---- The reference we gate on ----
    # AlphaZero compares each new net against the best net so far. We keep those
    # weights in memory so the comparison is always "did training actually improve the
    # agent that would ship?".
    reference_state = _sd(net)
    # The FROZEN starting weights, for the `initial` league member ("did training
    # improve on what we began with?"). It is a second in-memory copy of the same net
    # the mirror calibration above uses — nothing is re-read from disk, and unlike
    # `reference_state` it is never replaced when a net is accepted.
    initial_state = _sd(net)
    # Weights of the last checkpoint SAVE event, for the `prev_checkpoint` league
    # member. Updated after each save, so at evaluation time it holds the snapshot from
    # before this iteration's training (what PPO passes as `prev_checkpoint`) rather
    # than the current weights, which would just be another mirror match. `None` until
    # a checkpoint exists, and then the member is skipped.
    last_saved_state = None
    # The last probe numbers, so the training-time value warning can quote the HELD-OUT
    # value MSE next to the in-sample R^2 (the gap between the two is what identifies
    # memorisation; an absolute value loss cannot).
    last_probe: dict = {}
    best_heuristic = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_heur_path = os.path.join(ckpt_dir, "best_heuristic.pt")

    # Never silently clobber a "best" file from an earlier run: stash it once as
    # `*_prev.pt`. (A checkpoint is the only copy of a trained net, and it is easy to
    # start a second run in the same directory by accident.)
    for path in (best_path, best_heur_path):
        if os.path.exists(path):
            prev = path.replace(".pt", "_prev.pt")
            if not os.path.exists(prev):
                os.replace(path, prev)
                print(f"Kept the earlier {os.path.basename(path)} as "
                      f"{os.path.basename(prev)}")
            else:
                # A backup from an even earlier run already exists; drop the stale file.
                os.remove(path)
                print(f"Replaced the stale {os.path.basename(path)} "
                      f"(backup kept as {os.path.basename(prev)})")

    if args.init_checkpoint:
        # The reference IS the starting net until something beats it, so make sure it
        # exists on disk before training can touch the weights.
        save_checkpoint(net, best_path, iteration=-1, value_target=VALUE_TARGET,
                        return_value_scale=RETURN_VALUE_SCALE,
                        extra={"role": "starting weights (reference until beaten)"})
        print(f"Saved starting weights as {best_path} (the reference until beaten)")

    # Mild weight decay: with only a few hundred self-play games in the buffer, the
    # value head can memorise outcomes (see the warning in the epoch loop below).
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- Baseline evaluation ----
    # Do this BEFORE any self-play so the run has a reference point measured with the
    # exact same search settings. Without it you cannot tell whether self-play helped
    # or hurt.
    #
    # This baseline match is the net against ITSELF (`reference_state` is a copy), so
    # it doubles as a MIRROR CALIBRATION: whatever team 0 wins here is the
    # first-mover/seat asymmetry of the setup, not skill. Gating must beat that
    # number, not a naive 50%, or a net that only looks good because it moves first
    # would be accepted as progress.
    mirror_baseline = None
    accept_needed = args.accept_threshold
    if not args.no_eval:
        print("\n=== Baseline evaluation (before any self-play) ===")
        # No checkpoint has been saved yet, so the league has no `prev_checkpoint`
        # member for the baseline: that member is SKIPPED rather than logged stale.
        metrics = run_evaluation(net, reference_state, args, search_cfg, iteration=-1,
                                 initial_state=initial_state,
                                 prev_checkpoint_state=last_saved_state)
        last_probe = {k: metrics[k] for k in
                      ("probe_value_mse", "probe_target_var") if k in metrics}
        mirror_baseline = metrics["reference"]
        # `heuristic` can be dropped with --members; without it there is nothing to keep a
        # "best vs the bot" checkpoint against, so tracking is disabled rather than reading
        # a key that is not there.
        best_heuristic = metrics.get("heuristic")
        accept_needed = max(args.accept_threshold, mirror_baseline + 0.05)
        print(f"  mirror baseline (net vs itself) = {mirror_baseline:.1%}; "
              f"a new net must reach {accept_needed:.1%} vs the reference to be "
              f"accepted")

    start = time.perf_counter()
    decisions_buffer: list[Decision] = []

    # One game per process: the search is Python work around batch-1 forwards, so extra
    # threads would take turns rather than run in parallel.
    # Auto leaves a little headroom rather than taking every core: a worker carries a
    # torch runtime, and a training run is not the only thing a laptop is doing.
    auto_workers = min(os.cpu_count() or 1, 8)
    games_workers = args.games_workers or min(auto_workers, max(1, args.games_per_iter))
    games_workers = max(1, min(games_workers, max(1, args.games_per_iter)))
    print(f"Self-play workers: {games_workers}"
          + ("  (serial; pass --games-workers N for N processes)" if games_workers == 1
             else f"  ({args.games_per_iter} games split over {games_workers} processes)"))

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration} ===")
        t_iter = time.perf_counter()

        # 1. Self-play with the current net
        t0 = time.perf_counter()
        decisions, sp_metrics = collect_games(net, args.games_per_iter,
                                              base_seed=iteration * 1000,
                                              workers=games_workers)
        t_collect = time.perf_counter() - t0
        print(f"  collected {len(decisions)} self-play decisions "
              f"in {timedelta(seconds=t_collect)}")
        # How the agent is doing in REWARD terms this iteration: episode returns,
        # final scores, game length and caída/limpia counts.
        print(f"  self-play: win_rate_t0={sp_metrics['selfplay/win_rate_team0']:.2f}  "
              f"scores={sp_metrics['selfplay/score_team0']:.1f}-"
              f"{sp_metrics['selfplay/score_team1']:.1f}  "
              f"returns={sp_metrics['selfplay/return_team0']:.1f}-"
              f"{sp_metrics['selfplay/return_team1']:.1f}  "
              f"caídas={sp_metrics['selfplay/caidas_team0']:.2f}+"
              f"{sp_metrics['selfplay/caidas_team1']:.2f}  "
              f"limpias={sp_metrics['selfplay/limpias_team0']:.2f}+"
              f"{sp_metrics['selfplay/limpias_team1']:.2f}  "
              f"plies={sp_metrics['selfplay/plies']:.0f}  "
              f"root|v|={sp_metrics['selfplay/mean_root_value_team0']:+.2f}/"
              f"{sp_metrics['selfplay/mean_root_value_team1']:+.2f}  "
              # The value LABELS' spread, in game points, next to what pure Monte-Carlo
              # would have given. In return mode with lambda < 1 the first number is what
              # the head actually trains on: it should be the SMALLER one, or the
              # bootstrapped label is carrying more of the head's error than it removes of
              # the far future's noise (see selfplay.backfill_value_targets).
              f"value_std={sp_metrics['selfplay/value_target_std']:.1f}"
              f"(mc {sp_metrics['selfplay/value_mc_std']:.1f})")
        if not args.no_wandb:
            # `iteration` is the shared x-axis (see wandb.define_metric above);
            # `iter` carries the same value under its old name.
            wandb.log(dict(sp_metrics, iteration=iteration, iter=iteration))

        # 2. A few teacher games, to keep the value head calibrated off-distribution.
        t0 = time.perf_counter()
        if args.imitation_games > 0:
            imitation = collect_imitation_games(args.imitation_games,
                                                base_seed=500_000 + iteration * 1000,
                                                value_target=VALUE_TARGET)
            print(f"  collected {len(imitation)} imitation decisions "
                  f"(policy weight {args.imitation_policy_weight})")
            decisions = decisions + imitation
        t_imitation = time.perf_counter() - t0

        # Replay buffer: keep the most recent REPLAY_ITERS iterations of data.
        decisions_buffer.extend(decisions)
        max_buffer = int(REPLAY_ITERS * len(decisions) * 1.5)
        if len(decisions_buffer) > max_buffer:
            decisions_buffer = decisions_buffer[-max_buffer:]

        stats = buffer_stats(decisions_buffer)
        print(f"  buffer: n={stats['n']} selfplay_frac={stats['selfplay_frac']:.2f} "
              f"target_ent={stats['target_entropy']:.3f} "
              f"target_top={stats['target_top_prob']:.3f} "
              f"prior_ent={stats['prior_entropy']:.3f} "
              f"legal={stats['legal_actions']:.1f}")
        if not args.no_wandb:
            wandb.log(dict({f"buffer/{k}": v for k, v in stats.items()},
                           iteration=iteration, iter=iteration))

        # 3. Train on the whole buffer
        t0 = time.perf_counter()
        ds = CuarentaDataset(
            decisions_buffer, t_max=T_MAX, history_dropout=HISTORY_DROPOUT,
            target_temperature=TARGET_TEMPERATURE,
            imitation_policy_weight=args.imitation_policy_weight,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        for epoch in range(args.epochs_per_iter):
            loss, pl, vl, al, tent, v_r2, v_var = train_one_epoch(
                net, loader, opt, DEVICE,
                anchor_net=anchor_net, anchor_weight=anchor_weight,
            )
            excess = pl - tent
            print(f"  epoch {epoch}: loss={loss:.4f} policy={pl:.4f} value={vl:.4f}"
                  + (f" anchor={al:.4f}" if anchor_weight > 0 else "")
                  + f" | value_r2={v_r2:+.2f}"
                  + f" target_ent={tent:.3f} excess_ce={excess:.3f}")
            if not args.no_wandb:
                wandb.log({"train/loss": loss, "train/policy_loss": pl,
                           "train/value_loss": vl, "train/anchor_loss": al,
                           "train/target_entropy": tent,
                           "train/excess_ce": excess,
                           "train/value_r2": v_r2,
                           "train/value_target_var": v_var,
                           "iteration": iteration, "iter": iteration})
        t_train = time.perf_counter() - t0

        # Watch the value loss: every decision of a game shares one ±1 label, so a
        # small buffer can be MEMORISED, and a value loss near 0 on the training
        # buffer does NOT mean the value head is good on fresh games. Since the search
        # values leaves with this head, a memorised value head quietly poisons the
        # search. The probe in run_evaluation is the trustworthy version of this check.
        if vl < 0.05:
            print(f"  WARNING: value loss {vl:.4f} is suspiciously low — the value "
                  f"head may be memorising this buffer's games (check probe_value_mse)")

        # 4. Save
        ckpt_path = os.path.join(ckpt_dir, f"cuarenta_iter{iteration:03d}.pt")
        # The `prev_checkpoint` league member is the snapshot from the PREVIOUS save
        # event (`last_saved_state`), i.e. the weights from before this iteration's
        # training — NOT the file we are about to write, which is just the current net.
        prev_ckpt_state = last_saved_state
        save_checkpoint(net, ckpt_path, iteration=iteration, value_target=VALUE_TARGET,
                        return_value_scale=RETURN_VALUE_SCALE, extra={"role": "periodic"})
        last_saved_state = _sd(net)
        print(f"  saved {ckpt_path}")

        # 5. Held-out head probe — EVERY iteration by default, and deliberately BEFORE
        # the (expensive) league evaluation so the training-time warning below always has a
        # fresh held-out number to quote. It costs ~20 teacher games and no search.
        will_evaluate = (iteration + 1) % args.eval_every == 0
        if args.probe_every and (iteration + 1) % args.probe_every == 0 and not will_evaluate:
            probe = head_probe_metrics(_sd(net), search_cfg)
            last_probe = {k: probe[k] for k in
                          ("probe_value_mse", "probe_target_var") if k in probe}
            print(f"  probe: value_r2={probe['probe_value_r2']:+.2f}  "
                  f"mse={probe['probe_value_mse']:.3f}  "
                  f"conf={probe['probe_value_confidence']:.2f}  "
                  f"policy_acc={probe['probe_policy_acc']:.3f}")
            if probe["probe_value_r2"] < 0.0:
                print("        WARNING: the value head has NO held-out skill (r2 < 0 means "
                      "worse than predicting the target's mean) while the search values "
                      "its leaves with it. Stop and fix the value head (fewer epochs, "
                      "λ-returns, an anchor) before spending more iterations.")
            if not args.no_wandb:
                wandb.log(dict({f"eval/{k}": v for k, v in probe.items()},
                               iteration=iteration, iter=iteration))

        # 6. Full evaluation (league + gate + probe + the rest of the health metrics)
        if will_evaluate:
            metrics = run_evaluation(net, reference_state, args, search_cfg, iteration,
                                     initial_state=initial_state,
                                     prev_checkpoint_state=prev_ckpt_state)
            last_probe = {k: metrics[k] for k in
                          ("probe_value_mse", "probe_target_var") if k in metrics}

            # Gating on the REFERENCE match (AlphaZero style). The bar is the mirror
            # calibration plus a margin, because a single 60-game match has a standard
            # error of ~6%: demanding +5% over the mirror makes accidental acceptance
            # unlikely while still moving fast enough to see progress.
            if metrics["reference"] >= accept_needed:
                reference_state = _sd(net)
                save_checkpoint(net, best_path, iteration=iteration,
                                value_target=VALUE_TARGET,
                                return_value_scale=RETURN_VALUE_SCALE,
                                extra={"role": "reference (gated best-so-far)",
                                       "reference_win_rate": float(metrics["reference"])})
                print(f"  ACCEPTED: beats the reference {metrics['reference']:.1%} "
                      f"(bar {accept_needed:.1%}) -> {best_path} (new reference)")
            else:
                print(f"  not accepted: reference={metrics['reference']:.1%} "
                      f"< bar {accept_needed:.1%} (reference unchanged)")

            # Also keep the best-vs-heuristic net: it is what you would deploy today,
            # even though it is not the training criterion.
            if (best_heuristic is not None and "heuristic" in metrics
                    and metrics["heuristic"] > best_heuristic):
                best_heuristic = metrics["heuristic"]
                save_checkpoint(net, best_heur_path, iteration=iteration,
                                value_target=VALUE_TARGET,
                                return_value_scale=RETURN_VALUE_SCALE,
                                extra={"role": "best vs heuristic",
                                       "heuristic_win_rate": float(best_heuristic)})
                print(f"  new best vs heuristic ({best_heuristic:.1%}) -> "
                      f"{best_heur_path}")

        # 7. Timing: how long each phase took, so you can see where a run's hours go.
        t_iter = time.perf_counter() - t_iter
        timing = {
            "time/iteration_s": t_iter,
            "time/selfplay_s": t_collect,
            "time/imitation_s": t_imitation,
            "time/train_s": t_train,
        }
        print(f"  time: iter={t_iter:.0f}s "
              f"(selfplay {t_collect:.0f}s, imitation {t_imitation:.0f}s, "
              f"train {t_train:.0f}s)")
        if not args.no_wandb:
            wandb.log(dict(timing, iteration=iteration, iter=iteration))

    print(f"\nTotal time: {timedelta(seconds=time.perf_counter() - start)}")
    # `best_heuristic` stays None when `heuristic` is not one of the requested members,
    # so this is a `is not None` test rather than a numeric one.
    if best_heuristic is not None:
        print(f"Best winrate vs heuristic: {best_heuristic:.2%}")
    elif "heuristic" not in requested_members(args):
        print("Best winrate vs heuristic: n/a (heuristic not in --members)")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
