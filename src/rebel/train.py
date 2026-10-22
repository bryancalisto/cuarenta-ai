"""
Training loop for the ReBeL pipeline: self-play games in, supervised updates out.

THE LOOP, AND WHY IT LOOKS LIKE THIS
------------------------------------
One iteration is exactly AlphaZero's expert iteration, with CFR in place of MCTS:

   1. generate `--games` self-play games in which the search plays every seat and the
      action is sampled from the search's root strategy (exploration);
   2. treat each decision as a supervised example: the search's root strategy is the
      policy target and the final win/loss is the value target;
   3. take `--epochs` passes over that buffer with Adam;
   4. every `--eval-every` iterations, play the whole league -- the gated `reference`
      (best.pt so far), the `heuristic` bot, the `initial` checkpoint the run started from,
      the `prev_checkpoint` snapshot and a `random` policy -- and keep `best.pt` when the
      win rate against the heuristic improves. Every member is logged under the exact
      `eval/*` key family the PPO and ISMCTS pipelines use, so the three pipelines' runs
      can be overlaid on one chart; a member whose checkpoint does not exist yet is skipped
      and produces no keys at all.

The buffer is thrown away after each iteration (no replay window), which is the plain
expert-iteration choice: `src/ismcts/train.py` keeps a window because PUCT visits are
noisy; here every target comes from a CFR solve, so the data of iteration N is already
stale once the network changes.

THE LOSSES
----------
    policy  cross-entropy between the search's strategy and the network's log-softmax
            over the first `n_legal` slots -- the search's target is already normalised
            over the legal actions and the legal slots are packed first, so nothing has
            to be renormalised or masked here;
    value   MSE between the value head and `z` (+1/-1 for the acting team), scaled by
            `--value-coef`, so the two heads can be balanced without touching the targets.

METRICS
-------
Same wandb project and the same metric families as the other two pipelines
(`train/*`, `selfplay/*`, `eval/*`, `time/*`), with the iteration on the x-axis, so the
ReBeL, ISMCTS and PPO curves can be read off one chart. Logging is strictly cosmetic:
every wandb call is wrapped, and a failure there never stops training.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence
import dotenv

import numpy as np
import torch
import torch.nn.functional as F

dotenv.load_dotenv()

# Scripts are run as `python3 src/rebel/<script>.py` from the repository root; this makes
# the bare sibling imports below work from anywhere (see selfplay.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_rebel import (PRIMARY_MEMBERS, evaluate_league,              # noqa: E402
                        league_members, league_summary_lines)
from network import (RebelNet, load_rebel_checkpoint,                  # noqa: E402
                     save_rebel_checkpoint)
from selfplay import (Decision, GameRecord, SelfPlayConfig, play_games,  # noqa: E402
                      selfplay_metrics)

DEFAULT_SAVE_DIR = "checkpoints-rebel"
WANDB_PROJECT = "cuarenta-ai"

# Seeds: self-play games and evaluation games come from disjoint ranges so an evaluation
# can never replay a training deal. The evaluation base is constant across iterations,
# which makes successive evaluations paired comparisons of the same deals.
SELFPLAY_SEED_OFFSET = 1_000_000
EVAL_SEED_OFFSET = 2_000_000

_WANDB = None          # the wandb module while a run is live, otherwise None
_WANDB_WARNED = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ReBeL self-play training for Cuarenta (CFR search + belief network)")
    # loop shape
    parser.add_argument("--iterations", type=int, default=10,
                        help="self-play -> train cycles")
    parser.add_argument("--games", type=int, default=4, help="self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=2, help="passes over the buffer")
    # search
    parser.add_argument("--n-iter", type=int, default=32, help="CFR iterations per decision")
    parser.add_argument("--depth", type=int, default=3,
                        help="plies searched before the network values a leaf")
    parser.add_argument("--particles", type=int, default=16,
                        help="belief particles (the search's chance node)")
    parser.add_argument("--belief-history", type=int, default=12,
                        help="recent actions the belief's Bayes update scores")
    # optimisation
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="weight of the value loss relative to the policy loss")
    parser.add_argument("--dropout", type=float, default=None,
                        help="dropout on the shared state embedding, training only. "
                             "Omit it to inherit an --init-checkpoint's value")
    # checkpoints / evaluation
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="checkpoint to warm start from")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="where policy_<iteration>.pt / best.pt go")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="iterations between evaluations (0 disables; it must be no "
                             "larger than --iterations for best.pt to appear)")
    parser.add_argument("--eval-games", type=int, default=8,
                        help="games per evaluation for the reference and heuristic "
                             "matches (8 games is ±18%% standard error)")
    parser.add_argument("--league-games", type=int, default=30,
                        help="games per evaluation for the league members initial, "
                             "prev_checkpoint and random")
    parser.add_argument("--no-league", dest="league", action="store_false", default=True,
                        help="skip the league members initial, prev_checkpoint and random "
                             "and evaluate only reference and heuristic")
    parser.add_argument("--eval-workers", type=int, default=1,
                        help="processes used for the evaluation games")
    # plumbing
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                        help="log to Weights & Biases (default)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--run-name", type=str, default="rebel")
    return parser


# ---------------------------------------------------------------------------
# wandb: best effort, never fatal
# ---------------------------------------------------------------------------

def _wandb_setup(args, cfg: SelfPlayConfig) -> bool:
    """
    Start the wandb run. Returns whether logging is live.

    The whole thing is inside a try/except on purpose: a run that cannot reach the network
    must still train. Everything below goes through `_wandb_log`, which is equally
    forgiving, so a wandb outage costs the charts and nothing else.
    """
    global _WANDB
    if not args.wandb:
        print("[rebel] wandb disabled (--no-wandb): metrics are printed only")
        return False
    try:
        import wandb

        wandb.init(project=WANDB_PROJECT, name=args.run_name,
                   config={**vars(args), "search": cfg.__dict__})
        # The iteration number is the x-axis for every family, so the charts read
        # "value of X at iteration N" rather than "log call number N".
        wandb.define_metric("iteration")
        for group in ("train", "selfplay", "eval", "time"):
            wandb.define_metric(f"{group}/*", step_metric="iteration")
        _WANDB = wandb
        return True
    except Exception as exc:
        print(f"[rebel] wandb unavailable, continuing without it: {exc!r}")
        _WANDB = None
        return False


def _wandb_log(iteration: int, metrics: Dict[str, float]) -> None:
    global _WANDB_WARNED
    if _WANDB is None or not metrics:
        return
    try:
        _WANDB.log(dict(metrics, iteration=iteration))
    except Exception as exc:
        if not _WANDB_WARNED:
            print(f"[rebel] wandb logging failed, continuing without it: {exc!r}")
            _WANDB_WARNED = True


# ---------------------------------------------------------------------------
# The supervised update on one buffer
# ---------------------------------------------------------------------------

def buffer_tensors(decisions: Sequence[Decision], device: torch.device):
    """
    Stack a buffer of decisions into tensors.

    `targets` stays a list of per-decision vectors because `n_legal` varies: the targets
    are already truncated to the legal actions (as the search produced them), so padding
    them into a [B, 64] block would only create slots that must be ignored everywhere.
    """
    obs = torch.as_tensor(np.stack([d.obs_input for d in decisions]),
                          dtype=torch.float32, device=device)
    feats = torch.as_tensor(np.stack([d.feats for d in decisions]),
                            dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.stack([d.mask for d in decisions]),
                           dtype=torch.bool, device=device)
    z = torch.as_tensor([d.z for d in decisions], dtype=torch.float32, device=device)
    targets = [torch.as_tensor(d.target_probs, dtype=torch.float32, device=device)
               for d in decisions]
    return obs, feats, mask, z, targets


def train_on_buffer(net, optimizer, decisions: Sequence[Decision], *, epochs: int,
                    batch_size: int, value_coef: float, device: torch.device,
                    rng: random.Random) -> Dict[str, float]:
    """
    `epochs` passes of Adam over one iteration's buffer; returns the mean `train/*` metrics.

    The policy loss is computed per sample over `logits[:, :n_legal]`, because the number
    of legal actions changes from decision to decision -- a single softmax over all 64
    slots would mix in the padding slots, which the network only ever sees filled with
    zeros and which the search never scored. The legal slots are packed first (that is the
    shared action-feature contract), so slicing is enough.
    """
    obs, feats, mask, z, targets = buffer_tensors(decisions, device)
    n_legal = [int(d.n_legal) for d in decisions]
    n = len(decisions)

    history: Dict[str, List[float]] = {"loss_pi": [], "loss_vf": [], "entropy": []}
    order = list(range(n))
    net.train()
    for _epoch in range(epochs):
        rng.shuffle(order)                       # minibatches, not mini-epochs
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            logits, values = net(obs[idx_t], feats[idx_t], mask[idx_t])

            loss_pi = logits.new_zeros(())
            entropy = logits.new_zeros(())
            for row, i in enumerate(idx):
                logp = F.log_softmax(logits[row, :n_legal[i]], dim=-1)
                # The search's target is a probability over the legal actions; the
                # cross-entropy against it is the policy gradient ReBeL prescribes.
                loss_pi = loss_pi - (targets[i] * logp).sum()
                # Entropy of the NETWORK's own distribution (not the target's): it is the
                # "is the policy collapsing?" signal during training.
                entropy = entropy - (logp.exp() * logp).sum()
            loss_pi = loss_pi / len(idx)
            entropy = entropy / len(idx)

            # Value: the acting team's win/loss, which is what the value head predicts
            # (+1 = the team to move wins). `tanh` bounds it, so an untrained head starts
            # near 0 and the loss is at most 4.
            loss_vf = F.mse_loss(values, z[idx_t])
            loss = loss_pi + value_coef * loss_vf

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            history["loss_pi"].append(float(loss_pi.detach().cpu()))
            history["loss_vf"].append(float(loss_vf.detach().cpu()))
            history["entropy"].append(float(entropy.detach().cpu()))

    net.eval()
    return {f"train/{k}": float(np.mean(v)) for k, v in history.items() if v}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = SelfPlayConfig(games=args.games, n_particles=args.particles,
                         belief_history=args.belief_history, n_iter=args.n_iter,
                         depth=args.depth, seed=args.seed, explore=True,
                         device=args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # An explicit --dropout wins; omitting it keeps the warm start's setting.
    dropout = args.dropout if args.dropout is not None else 0.0
    net = RebelNet(dropout=dropout).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if init_path.exists():
            net = load_rebel_checkpoint(init_path, device, dropout=args.dropout)
            print(f"[rebel] warm start from {init_path}")
        else:
            print(f"[rebel] init checkpoint not found, starting from scratch: {init_path}")
    net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    wandb_live = _wandb_setup(args, cfg)

    print(f"[rebel] iterations={args.iterations} games={args.games} "
          f"n_iter={args.n_iter} depth={args.depth} particles={args.particles} "
          f"lr={args.lr} dropout={net.dropout_p} device={args.device} save_dir={save_dir} "
          f"eval_games={args.eval_games} league_games={args.league_games} "
          f"league={'on' if args.league else 'off'} "
          f"wandb={'on' if wandb_live else 'off'}")

    batch_rng = random.Random(args.seed + 7)
    eval_seed_base = EVAL_SEED_OFFSET + args.seed
    best_win = -1.0
    start = time.perf_counter()

    def game_line(record: GameRecord, i: int) -> None:
        winner = "unfinished" if record.winner is None else f"team {record.winner}"
        print(f"  game {i + 1}/{args.games}: {winner} "
              f"{record.scores[0]}-{record.scores[1]} in {record.plies} plies "
              f"({len(record.decisions)} decisions)")

    for iteration in range(1, args.iterations + 1):
        t_iter = time.perf_counter()

        # ---- 1. self-play --------------------------------------------------
        t0 = time.perf_counter()
        records = play_games(net, cfg, base_seed=SELFPLAY_SEED_OFFSET + 1000 * iteration
                             + args.seed, progress=game_line)
        t_selfplay = time.perf_counter() - t0

        sp = selfplay_metrics(records)
        decisions = [d for r in records for d in r.decisions]
        if not decisions:
            raise RuntimeError("self-play produced no decisions; nothing to train on")

        # ---- 2. update -----------------------------------------------------
        t0 = time.perf_counter()
        train_metrics = train_on_buffer(
            net, optimizer, decisions, epochs=args.epochs, batch_size=args.batch_size,
            value_coef=args.value_coef, device=device, rng=batch_rng)
        t_train = time.perf_counter() - t0

        target_value_mean = float(np.mean([d.z for d in decisions]))
        metrics: Dict[str, float] = {
            **train_metrics,
            "train/target_value_mean": target_value_mean,
            "train/n_decisions": float(len(decisions)),
            **sp,
        }

        # ---- 3. evaluation + checkpoints -----------------------------------
        t_eval = 0.0
        eval_note = ""
        eval_report: List[str] = []
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            t0 = time.perf_counter()
            # The league: `reference` (best.pt so far), `heuristic`, `initial` (the frozen
            # warm-start checkpoint), `prev_checkpoint` (the last snapshot saved before this
            # evaluation) and `random`. Members whose checkpoint does not exist yet are
            # absent from `members` and therefore produce NO `eval/*` keys -- a member that
            # could not be played must not look like a measurement. The primary members run
            # `--eval-games` games, the rest `--league-games`.
            members, eval_skipped = league_members(
                save_dir=save_dir, init_checkpoint=args.init_checkpoint,
                iteration=iteration, include_league=args.league)
            games_per_member = {
                name: (args.eval_games if name in PRIMARY_MEMBERS else args.league_games)
                for name in members
            }
            eval_metrics = evaluate_league(net, members, games_per_member, eval_seed_base,
                                           cfg, workers=args.eval_workers)
            t_eval = time.perf_counter() - t0
            metrics.update(eval_metrics)
            eval_report = league_summary_lines(eval_metrics, list(members))
            eval_report += [f"[skip] {name}: {reason}"
                            for name, reason in eval_skipped.items()]
            win = eval_metrics["eval/heuristic"]
            if win > best_win:
                best_win = win
                save_rebel_checkpoint(save_dir / "best.pt", net, step=iteration,
                                      config={**vars(args), "search": cfg.__dict__},
                                      optimizer=optimizer)
                eval_note = (f" | eval {win:.1%} ± {eval_metrics['eval/heuristic_stderr']:.1%} "
                             f"({int(eval_metrics['eval/heuristic_n_games'])} games) "
                             f"-> new best, saved best.pt")
            else:
                eval_note = (f" | eval {win:.1%} ± {eval_metrics['eval/heuristic_stderr']:.1%} "
                             f"(best so far {best_win:.1%})")

        # Always keep the per-iteration snapshot: the trajectory of the run is often more
        # informative than its best single number.
        save_rebel_checkpoint(save_dir / f"policy_{iteration}.pt", net, step=iteration,
                              config={**vars(args), "search": cfg.__dict__},
                              optimizer=optimizer)

        # ---- 4. timing + logging -------------------------------------------
        t_iter = time.perf_counter() - t_iter
        metrics.update({
            "time/iteration_s": t_iter,
            "time/selfplay_s": t_selfplay,
            "time/train_s": t_train,
            "time/eval_s": t_eval,
            "time/seconds_per_game": t_selfplay / max(len(records), 1),
            "time/total_s": time.perf_counter() - start,
        })

        _wandb_log(iteration, metrics)

        print(f"iter {iteration:3d}/{args.iterations} | "
              f"pi {train_metrics.get('train/loss_pi', float('nan')):.4f}  "
              f"vf {train_metrics.get('train/loss_vf', float('nan')):.4f}  "
              f"ent {train_metrics.get('train/entropy', float('nan')):.2f} | "
              f"selfplay win_t0 {sp['selfplay/win_rate_team0']:.2f}  "
              f"plies {sp['selfplay/plies']:.0f}  "
              f"depleted {sp['selfplay/belief_depleted']:.0f} | "
              f"{t_iter:.0f}s "
              f"(play {t_selfplay:.0f}s, train {t_train:.0f}s, eval {t_eval:.0f}s)"
              f"{eval_note}")
        for line in eval_report:
            print(f"       eval  {line}")

    total = time.perf_counter() - start
    print(f"\n[rebel] done in {total:.0f}s; checkpoints in {save_dir}/")
    if best_win >= 0.0:
        print(f"[rebel] best win rate vs heuristic: {best_win:.1%}")
    if _WANDB is not None:
        try:
            _WANDB.finish()
        except Exception as exc:                                # noqa: BLE001
            print(f"[rebel] wandb finish failed (ignored): {exc!r}")


if __name__ == "__main__":
    main()
