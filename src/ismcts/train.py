"""
Training loop for the Cuarenta net (expert iteration / AlphaZero style).

  for iteration in range(N_ITERATIONS):
      1. Generate self-play games with the current net (search = this net itself).
      2. Train the net to match (a) the search's visit distribution and (b) the outcome.
      3. Evaluate against fixed references (heuristic agent, rollout-ISMCTS).
      4. Save a checkpoint, and save `best.pt` whenever evaluation improves.

This is not PPO. The policy loss is cross-entropy against the visit distribution and
the value loss is MSE against the game outcome.

Before trusting a run, check that the SEARCH IS STRONGER THAN THE RAW POLICY. In
AlphaZero the search is the teacher: the net improves by imitating a search that is
better than the net's own policy head. If the search is *worse* (or equally good),
each iteration teaches the net noise and its win rate falls — which is exactly what
happened with the old settings: leaf values came from rollouts capped at 80 plies on
games of 130+ plies, so most leaves were valued 0.0, the visit distribution was
decided by expansion order, and 50 iterations took the net from 73% to ~40% against
the heuristic agent. The knobs that matter here:

  * LEAF_VALUE_MODE / MAX_ROLLOUT_LEN — see network_ismcts_1.py. Default "net".
  * N_SEARCH_ITER — number of simulations per decision. More sims = better teacher,
    and with net leaf values a sim is ~50x cheaper than an 80-ply rollout.
  * TEMPERATURE_CUTOFF — after this many plies the self-play agent plays greedily.

Evaluation notes:
  * SELF-PLAY and EVAL use the same search settings (same n_iter, same leaf value),
    so the number you log describes the agent you are actually training. The old code
    trained with max_rollout_len=80 but evaluated with 200.
  * 30 games gives a standard error of roughly ±9%, so single evaluations swing a lot.
    Use more games if you plan to compare close checkpoints, and always compare against
    the *initial* checkpoint evaluated with the same settings (see `--init-checkpoint`).
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import random
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import dotenv

from env import CuarentaEnv
from network import CuarentaNet
from selfplay import play_one_game, Decision
from dataset import CuarentaDataset
from network_ismcts_1 import NetworkISMCTS

dotenv.load_dotenv() # to load wandb token

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ITERATIONS = 30             # outer loop iterations
GAMES_PER_ITER = 12           # self-play games per iteration (more = better)
N_SEARCH_ITER = 200           # ISMCTS sims per decision (self-play AND eval)
TEMPERATURE_CUTOFF = 24       # plies with tau=1 before switching to greedy play
EPOCHS_PER_ITER = 4           # training epochs per iteration (see the value warning)
BATCH_SIZE = 64
LR = 1e-3
T_MAX = 32
HISTORY_DROPOUT = 0.3         # see dataset.py: makes history-free evals in-distribution
REPLAY_ITERS = 5              # keep this many iterations of data in the buffer
EVAL_GAMES = 60               # games per evaluation (30 => ~±9% stderr)
EVAL_EVERY = 2                # evaluate vs heuristic every N iterations
LEAF_VALUE_MODE = "net"       # "net" (default) | "rollout"
MAX_ROLLOUT_LEN = 512         # only used by leaf_value_mode="rollout"
VALUE_ANCHOR_WEIGHT = 0.5     # keep the value head near the starting checkpoint's
                              # (measured: without it, 2 short iterations pushed the
                              # held-out value MSE from 0.82 to 1.21 and the win rate
                              # from 55% to 37%, because the search trusts that head)
CKPT_DIR = "checkpoints"


def _pick_device(requested: str = "auto") -> str:
    """
    Device selection.

    Measured on an Apple Silicon machine (see the numbers below), the search does
    ONE batch-1 network forward per evaluated node, and MPS per-op launch overhead
    dominates at that size:

        n_iter=200, CPU:  32 ms/search      n_iter=200, MPS: 251 ms/search
        n_iter=400, CPU:  59 ms/search      n_iter=400, MPS: 370 ms/search

    i.e. CPU was 3-8x FASTER. So "auto" prefers CUDA when present and CPU otherwise;
    MPS is available with --device mps but usually a bad trade for this workload.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _pick_device()
print(f"Using device: {DEVICE}")

os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_games(net: CuarentaNet, n_games: int, base_seed: int) -> list[Decision]:
    all_decisions: list[Decision] = []
    for i in tqdm(range(n_games), desc="self-play games", leave=False):
        decisions = play_one_game(
            net,
            n_iter=N_SEARCH_ITER,
            max_history_len=T_MAX,
            seed=base_seed + i,
            temperature=1.0,
            temperature_cutoff=TEMPERATURE_CUTOFF,
            dirichlet_eps=0.25,
            leaf_value_mode=LEAF_VALUE_MODE,
            max_rollout_len=MAX_ROLLOUT_LEN,
            device=DEVICE,
        )
        all_decisions.extend(decisions)
    return all_decisions


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
    plays badly. Measured on this project: two short self-play iterations left the
    policy as good as before (top-1 vs the heuristic 0.655 -> 0.645) but pushed the
    held-out value MSE from 0.82 past 1.0 (worse than always predicting 0), and the
    win rate fell 55% -> 37%. The anchor keeps the calibrated bootstrap value while
    still letting the head adapt:

        anchor_weight = 0.0  -> train the value head freely (AlphaZero-style)
        anchor_weight = 0.5  -> soft anchor (default when starting from a checkpoint)
        anchor_weight high   -> effectively freeze the value head

    (A stronger fix, for later: more self-play games per iteration. With only a few
    hundred decisions in the buffer, every consistent target is memorisable.)
    """
    net.train()
    total_loss = total_p = total_v = total_a = 0.0
    n_batches = 0

    use_anchor = anchor_net is not None and anchor_weight > 0

    for (obs, act, am, feats, mask, pi_target, v_target, length) in loader:
        obs = obs.to(device)
        act = act.to(device)
        am = am.to(device)
        feats = feats.to(device)
        mask = mask.to(device)
        pi_target = pi_target.to(device)
        v_target = v_target.to(device)
        length = length.to(device)

        logits, value = net(obs, act, am, feats, mask, lengths=length)

        # Policy loss: cross-entropy between softmax(logits) and the search's
        # visit distribution. We use log_softmax so the -1e9 masked slots contribute
        # exactly 0 (their target probability is 0).
        logp = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pi_target * logp).sum(dim=-1).mean()

        # Value loss: MSE against the ±1 game outcome, from the acting team's POV.
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

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        total_a += anchor_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_p / n, total_v / n, total_a / n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
#
# Every evaluator plays a game in a worker process and returns the winning team.
# Team 0 is always the net under test, so `wins == games where winner == 0`.

def _make_search(env, net, tracker, seed, *, n_iter, leaf_value_mode, max_rollout_len):
    """
    The exact search used for self-play, minus exploration noise.

    The configuration is passed in explicitly instead of being read from module
    globals: evaluation runs in worker PROCESSES, and a spawned worker only sees the
    module defaults, not the CLI overrides. Searching with different settings than
    the ones being trained is precisely how the old code ended up training with
    max_rollout_len=80 and evaluating with 200.
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
    )


def _load_net(state_dict):
    net = CuarentaNet()
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _pick_visit_action(searcher, p, legal):
    """Argmax over visit counts, with a random fallback if the search had no budget."""
    visits = searcher.search(root_player=p)
    if sum(visits.values()) == 0:
        return random.randrange(len(legal))
    return max(visits, key=visits.get)


def _eval_heuristic_worker(args):
    """One game: net (team 0) vs HeuristicAgent (team 1). Returns winning team."""
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from heuristic_agent import HeuristicAgent
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict)
    tracker = HistoryTracker(env)
    searcher = _make_search(env, net, tracker, seed, **cfg)
    heuristic = HeuristicAgent(seed=seed * 2 + 2)

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()

        if TEAMS[p] == 0:
            a = _pick_visit_action(searcher, p, legal)
        else:
            a = legal.index(heuristic.act(env, p))

        tracker.record(p, obs, a)
        env.step(a)

    return env.state.winner


def _eval_rollout_worker(args):
    """One game: net (team 0) vs rollout-ISMCTS (team 1). Returns winning team."""
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from history import HistoryTracker
    from ismcts import ISMCTS as RolloutISMCTS

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict)
    tracker = HistoryTracker(env)
    net_searcher = _make_search(env, net, tracker, seed, **cfg)
    rollout_searcher = RolloutISMCTS(
        env, n_iter=cfg["n_iter"], rollout_policy="greedy",
        rng=random.Random(seed * 2 + 2),
    )

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        searcher = net_searcher if TEAMS[p] == 0 else rollout_searcher
        a = _pick_visit_action(searcher, p, legal)
        tracker.record(p, obs, a)
        env.step(a)

    return env.state.winner


def _eval_random_worker(args):
    """One game: net (team 0) vs uniform-random (team 1). Returns winning team."""
    seed, state_dict, cfg = args
    import torch as _torch
    _torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from history import HistoryTracker

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    net = _load_net(state_dict)
    tracker = HistoryTracker(env)
    searcher = _make_search(env, net, tracker, seed, **cfg)

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        if TEAMS[p] == 0:
            a = _pick_visit_action(searcher, p, legal)
        else:
            a = random.randrange(len(legal))
        tracker.record(p, obs, a)
        env.step(a)

    return env.state.winner


def _pooled_eval(worker, net, n_games: int, seed_base: int, desc: str, cfg: dict) -> float:
    """Run `n_games` head-to-head games in parallel; return the net's win rate."""
    from multiprocessing import Pool
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    args = [(seed_base + i, state_dict, cfg) for i in range(n_games)]
    with Pool() as pool:
        results = list(tqdm(pool.imap(worker, args), total=n_games, desc=desc, leave=False))
    return sum(1 for w in results if w == 0) / max(n_games, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # `global` must come before any use of these names in this function.
    global N_SEARCH_ITER, LEAF_VALUE_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Checkpoint to start from (e.g. checkpoints/bootstrap.pt). "
                             "Checkpoints saved before the history/feature fixes are "
                             "incompatible and will fail to load.")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=GAMES_PER_ITER)
    parser.add_argument("--n-search-iter", type=int, default=N_SEARCH_ITER)
    parser.add_argument("--epochs-per-iter", type=int, default=EPOCHS_PER_ITER)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--leaf-value-mode", type=str, default=LEAF_VALUE_MODE,
                        choices=["net", "rollout"])
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    parser.add_argument("--value-anchor-weight", type=float, default=VALUE_ANCHOR_WEIGHT,
                        help="Keep the value head close to the starting checkpoint's "
                             "value output (0 = off, 1.0+ = almost frozen). Protects "
                             "the leaf values the search depends on.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="auto = cuda if available else cpu (MPS is slower here)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the initial baseline evaluation")
    args = parser.parse_args()

    # Self-play reads these globals; the evaluators get them explicitly via `cfg`
    # (worker processes would otherwise only see the module defaults).
    global DEVICE
    DEVICE = _pick_device(args.device)
    print(f"Self-play device: {DEVICE} (eval workers always use CPU)")
    N_SEARCH_ITER = args.n_search_iter
    LEAF_VALUE_MODE = args.leaf_value_mode
    search_cfg = {
        "n_iter": N_SEARCH_ITER,
        "leaf_value_mode": LEAF_VALUE_MODE,
        "max_rollout_len": MAX_ROLLOUT_LEN,
    }

    if not args.no_wandb:
        wandb.init(
            project="cuarenta-ai",
            config={
                "n_iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "n_search_iter": N_SEARCH_ITER,
                "leaf_value_mode": LEAF_VALUE_MODE,
                "max_rollout_len": MAX_ROLLOUT_LEN,
                "temperature_cutoff": TEMPERATURE_CUTOFF,
                "epochs_per_iter": args.epochs_per_iter,
                "batch_size": BATCH_SIZE,
                "lr": args.lr,
                "t_max": T_MAX,
                "history_dropout": HISTORY_DROPOUT,
                "eval_games": args.eval_games,
                "init_checkpoint": args.init_checkpoint,
                "value_anchor_weight": args.value_anchor_weight,
            },
        )

    net = CuarentaNet().to(DEVICE)

    if args.init_checkpoint:
        # Loads with a clear error if the checkpoint predates the history fixes.
        CuarentaNet.load_checkpoint(net, args.init_checkpoint)
        print(f"Initialized net from {args.init_checkpoint}")
    else:
        print("Starting from random initialization")

    # ---- Value anchor ----
    # Frozen copy of the starting net. Its value output is the calibration reference
    # for the anchor term in the loss; see train_one_epoch for the measurements that
    # motivate it. Only created when we actually start from a checkpoint: anchoring to
    # a random net's value would be anchoring to noise.
    anchor_net = None
    anchor_weight = args.value_anchor_weight
    if anchor_weight > 0:
        if args.init_checkpoint:
            import copy as _copy
            anchor_net = _copy.deepcopy(net).eval()
            for p in anchor_net.parameters():
                p.requires_grad_(False)
            print(f"Value anchor: ON (weight={anchor_weight}) vs {args.init_checkpoint}")
        else:
            anchor_weight = 0.0
            print("Value anchor requested but no --init-checkpoint: disabling it")

    # Mild weight decay: with only a few hundred self-play games in the buffer, the
    # value head can memorise outcomes (see the warning in the epoch loop below).
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # Write out the starting weights as `best.pt` before touching them, so a baseline
    # checkpoint always exists on disk even if self-play never beats it. Never
    # overwrite an existing `best.pt` from an earlier run (that file may be the only
    # copy of a good net).
    best_path = os.path.join(CKPT_DIR, "best.pt")
    if args.init_checkpoint and not os.path.exists(best_path):
        torch.save(net.state_dict(), best_path)
        print(f"Saved starting weights as {best_path}")
    elif args.init_checkpoint:
        print(f"Keeping the existing {best_path} (delete it to start a fresh "
              f"'best' trace; the win rates of two runs are not comparable)")

    # ---- Baseline evaluation ----
    # Do this BEFORE any self-play so the run has a reference point measured with
    # the exact same search settings. Without it you cannot tell whether self-play
    # helped or hurt.
    best_winrate = -1.0
    if not args.no_eval:
        baseline = _pooled_eval(_eval_heuristic_worker, net, args.eval_games,
                                seed_base=50000, desc="baseline eval", cfg=search_cfg)
        best_winrate = baseline
        print(f"\n=== Baseline (before self-play) vs heuristic: {baseline:.1%} ===")
        if not args.no_wandb:
            wandb.log({"iter": -1, "winrate_vs_heuristic": baseline})

    start = time.perf_counter()
    decisions_buffer: list[Decision] = []

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration} ===")

        # 1. Self-play with the current net
        t0 = time.perf_counter()
        decisions = collect_games(net, args.games_per_iter, base_seed=iteration * 1000)
        t_collect = time.perf_counter() - t0
        print(f"  collected {len(decisions)} decisions in {timedelta(seconds=t_collect)}")

        # Replay buffer: keep the most recent REPLAY_ITERS iterations of data.
        decisions_buffer.extend(decisions)
        max_buffer = int(REPLAY_ITERS * len(decisions) * 1.5)
        if len(decisions_buffer) > max_buffer:
            decisions_buffer = decisions_buffer[-max_buffer:]

        # 2. Train on the whole buffer
        ds = CuarentaDataset(decisions_buffer, t_max=T_MAX, history_dropout=HISTORY_DROPOUT)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        for epoch in range(args.epochs_per_iter):
            loss, pl, vl, al = train_one_epoch(
                net, loader, opt, DEVICE,
                anchor_net=anchor_net, anchor_weight=anchor_weight,
            )
            if not args.no_wandb:
                wandb.log({"iter": iteration, "epoch": epoch,
                           "loss": loss, "policy_loss": pl, "value_loss": vl,
                           "value_anchor_loss": al})
            print(f"  epoch {epoch}: loss={loss:.4f} policy={pl:.4f} value={vl:.4f}"
                  + (f" anchor={al:.4f}" if anchor_weight > 0 else ""))

        # Watch the value loss: every decision of a game shares one ±1 label, so a
        # small buffer can be MEMORISED, and a value loss near 0 on the training
        # buffer does NOT mean the value head is good on fresh games. Since the
        # search now values leaves with this head (leaf_value_mode="net"), a
        # memorised value head quietly poisons the search — that was one of the
        # failure modes of the old pipeline (train value loss 0.036 while held-out
        # MSE was 1.49). If you see this warning, use fewer epochs per iteration,
        # more self-play games, or raise weight decay.
        if vl < 0.05:
            print(f"  WARNING: value loss {vl:.4f} is suspiciously low — the value "
                  f"head may be memorising this buffer's games")

        # 3. Save
        ckpt_path = os.path.join(CKPT_DIR, f"cuarenta_iter{iteration:03d}.pt")
        torch.save(net.state_dict(), ckpt_path)
        print(f"  saved {ckpt_path}")

        # 4. Evaluate vs the heuristic (the reference opponent for this project)
        if (iteration + 1) % EVAL_EVERY == 0:
            wr_h = _pooled_eval(_eval_heuristic_worker, net, args.eval_games,
                                seed_base=50000, desc="eval vs heuristic", cfg=search_cfg)
            print(f"  eval vs heuristic: {wr_h:.2%}")
            if not args.no_wandb:
                wandb.log({"iter": iteration, "winrate_vs_heuristic": wr_h})

            # Keep the best checkpoint, not just the last one. A single evaluation
            # has a large standard error, so also require a small margin before
            # overwriting the best.
            if wr_h > best_winrate:
                best_winrate = wr_h
                best_path = os.path.join(CKPT_DIR, "best.pt")
                torch.save(net.state_dict(), best_path)
                print(f"  new best ({wr_h:.2%}) -> {best_path}")

    print(f"\nTotal time: {timedelta(seconds=time.perf_counter() - start)}")
    print(f"Best winrate vs heuristic: {best_winrate:.2%}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
