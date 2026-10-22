"""
Warm start the ReBeL network by imitating the heuristic bot.

WHY THIS STAGE EXISTS
---------------------
The value head is the search's only source of information beyond the horizon it can play
out, and a randomly initialised head carries none: a search that is valued by noise picks
its moves on noise. Even a hand-written value (tanh of the score difference) makes the
same search play far better. Self-play starting from a random value therefore has almost
nothing to learn from, so the first stage is supervised: teach the network to imitate a
known-decent player, then let self-play improve on it. `src/ppo/bootstrap.py` does the
same thing for the same reason.

WHAT IS LEARNED
---------------
Both heads, from games where the HEURISTIC plays all four seats:

  * policy -- cross-entropy against the heuristic's move. The target is a one-hot over the
    legal-action slots with label smoothing, which keeps the prior from becoming so sharp
    that the search has nothing to explore with (the search's priors ARE this head).
  * value  -- the eventual winner from the acting team's point of view (+1 / -1). This is
    the signal the search consumes at its leaves, and it is exactly the same target the
    self-play stage uses, so the two stages cannot disagree about what value means.

HOW THE INPUTS ARE BUILT (the part that is easy to get wrong)
------------------------------------------------------------
Every training example is the 368-float input the search will feed the network at
inference: the 208-dim observation PLUS the 160 belief marginals, both produced by
`selfplay.build_input` -- the same function the search and the evaluator use. During
collection the belief's Bayes update runs WITHOUT a policy likelihood (`policy_fn=None`),
so its weights are the uniform prior conditioned on the acting seat's hand. Self-play then
switches the likelihood to the network, which is a small mismatch early on and shrinks as
the network learns; it is deliberately not papered over here.

WHY IT RUNS IN CHUNKS
---------------------
One decision costs 368 + 64*84 + 64 floats, so a few thousand games would not fit in
memory (200k decisions would be ~4.6 GB). Each iteration generates `--games` games,
trains `--epochs` passes over that chunk with a held-out validation split, and discards
it. The best epoch is kept by validation metrics, so the checkpoint is not simply
"whatever the last epoch produced" -- a value head can overfit inside one chunk, and the
last epoch is then the worst one.

  * `val/value_mse` below 1.0 means better than always predicting 0 (that is the reference
    the metric is read against, as in `src/ppo/bootstrap.py`),
  * `val/value_sign_acc` above 0.5 means it beats a coin flip at "who is winning",
  * `val/policy_acc` is how often the net's top move is the heuristic's move.

Usage:
    python3 src/rebel/bootstrap.py                      # writes checkpoints-rebel/bootstrap.pt
    python3 src/rebel/bootstrap.py --iterations 4 --games 60 --no-wandb   # quick
    python3 src/rebel/train.py --init-checkpoint checkpoints-rebel/bootstrap.pt
"""

from __future__ import annotations

import argparse
import dotenv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core import (CuarentaEnv, HeuristicAgent, TEAMS, action_features_matrix,
                  clone_state, legal_actions_from_state)
from belief import PublicBelief, action_index_of
from network import RebelNet, load_rebel_checkpoint, save_rebel_checkpoint
from selfplay import Decision, GameRecord, build_input
from train import train_on_buffer

dotenv.load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "checkpoints-rebel" / "bootstrap.pt"


@dataclass
class BootstrapConfig:
    """Everything the data generator and the loop need (a spawned worker sees only this)."""
    particles: int = 8
    belief_history: int = 8
    label_smoothing: float = 0.1
    val_fraction: float = 0.15
    value_target: str = "mix"
    seed: int = 0


# ---------------------------------------------------------------------------
# Data generation: the heuristic plays all four seats
# ---------------------------------------------------------------------------

def imitation_target(index: int, n_legal: int, smoothing: float) -> np.ndarray:
    """
    One-hot over the legal slots, smoothed.

    `smoothing` is mass moved off the chosen action and spread uniformly over the others.
    A perfectly sharp target would make the policy head maximally confident, and since the
    search uses that head as its PRIOR, a maximally confident prior is a prior with nothing
    left to explore around. 0.1 is the usual ballpark.
    """
    target = np.full(n_legal, smoothing / max(1, n_legal - 1), dtype=np.float32)
    target[index] = 1.0 - smoothing
    return target / target.sum()


def heuristic_game(seed: int, cfg: BootstrapConfig, rng: random.Random) -> GameRecord:
    """
    One game with the heuristic on all four seats, recorded as imitation examples.

    The heuristic is used rather than random play because its decisions are the thing
    being imitated -- and because a random policy's positions are unrepresentative of the
    ones the search will actually face.
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    heuristic = HeuristicAgent(seed=seed)
    belief = PublicBelief(n_particles=cfg.particles, rng=rng, history=cfg.belief_history)

    decisions: List[Decision] = []
    plies = 0
    while not env.state.done:
        state = env.state
        seat = state.current_player
        legal = legal_actions_from_state(state)

        # Condition the belief on the acting seat's hand, exactly as the search does, then
        # build the network input from it.
        belief.refresh(state, seat)
        obs_input = build_input(seat, state, belief)
        feats, mask = action_features_matrix(legal, state)

        action = heuristic.act(env, seat)
        index = action_index_of(
            legal, action.card.index,
            tuple(sorted(c.index for c in action.capture)) if action.capture else ())
        if index is None:
            raise RuntimeError(
                f"the heuristic returned an action that is not legal for seat {seat}: "
                f"{action}. Its move list and `legal_actions_from_state` disagree.")

        decisions.append(Decision(
            acting_seat=seat,
            acting_team=TEAMS[seat],
            obs_input=obs_input,
            feats=feats,
            mask=mask,
            n_legal=len(legal),
            target_probs=imitation_target(index, len(legal), cfg.label_smoothing),
            search_value=0.0,          # no search happened; unused by this stage
        ))

        belief.record_action(clone_state(state), seat, action, strategy=None)
        env.step(index)
        plies += 1

    winner = env.state.winner
    scores = (env.state.scores[0], env.state.scores[1])

    # The value target. The search consumes this at its leaves, so what goes in here
    # decides what the search is willing to play for.
    #
    # The pure win/loss target is sparse, which is why there is a choice at all: the head
    # learns "who is winning" without learning that scoring is HOW you win, and a search
    # optimising it declines caidas and limpias.
    #
    #   win   -- +1 / -1. Matches the search's terminal utility, but sparse: it says
    #            nothing about a +2 caida until the game is nearly decided.
    #   score -- tanh(final margin / 10). Dense and directly sensitive to scoring plays,
    #            at the cost of being a relaxation of "win" (a bigger margin is not
    #            strictly better than a win).
    #   mix   -- half of each, keeping both signals.
    #
    # The score and mix targets do not cure that on their own: the head's difficulty is
    # not the scale of the target, so the default stays on the one that matches the
    # search's terminal utility.
    for decision in decisions:
        won = 1.0 if winner == decision.acting_team else -1.0
        margin = scores[decision.acting_team] - scores[1 - decision.acting_team]
        scored = math.tanh(margin / 10.0)
        decision.z = {"win": won, "score": scored, "mix": 0.5 * won + 0.5 * scored}[
            cfg.value_target]

    return GameRecord(decisions=decisions, winner=winner,
                      scores=(env.state.scores[0], env.state.scores[1]),
                      plies=plies, seed=seed,
                      belief_depleted=belief.depleted_updates,
                      belief_resamples=belief.resamples)


def generate_games(n: int, base_seed: int, cfg: BootstrapConfig,
                   rng: random.Random) -> List[GameRecord]:
    return [heuristic_game(base_seed + i, cfg, rng) for i in range(n)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(net, decisions: Sequence[Decision], device: torch.device,
             batch_size: int = 256) -> Dict[str, float]:
    """
    Policy accuracy and value quality on held-out games.

    `value_mse` is read against 1.0 = "always predict 0" (the value head's untrained
    behaviour), and `value_sign_acc` against 0.5. Both are reported because a value head
    can be well calibrated in MSE while being useless for ranking positions by sign.
    """
    if not decisions:
        return {}
    net.eval()
    correct = total = 0
    value_sq = 0.0
    sign_correct = 0
    sign_total = 0
    for start in range(0, len(decisions), batch_size):
        chunk = decisions[start:start + batch_size]
        obs = torch.as_tensor(np.stack([d.obs_input for d in chunk]),
                              dtype=torch.float32, device=device)
        feats = torch.as_tensor(np.stack([d.feats for d in chunk]),
                                dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.stack([d.mask for d in chunk]),
                               dtype=torch.bool, device=device)
        logits, values = net(obs, feats, mask)
        for row, decision in enumerate(chunk):
            n = decision.n_legal
            predicted = int(torch.argmax(logits[row, :n]).item())
            target_index = int(np.argmax(decision.target_probs))
            correct += int(predicted == target_index)
            total += 1
        z = torch.as_tensor([d.z for d in chunk], dtype=torch.float32, device=device)
        value_sq += float(torch.sum((values - z) ** 2).item())
        sign_correct += int(torch.sum(torch.sign(values) == torch.sign(z)).item())
        sign_total += len(chunk)

    return {
        "val/policy_acc": correct / max(1, total),
        "val/value_mse": value_sq / max(1, total),
        "val/value_sign_acc": sign_correct / max(1, sign_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm-start the ReBeL network by imitating the heuristic bot.")
    parser.add_argument("--iterations", type=int, default=8,
                        help="chunks of generated games (default 8)")
    parser.add_argument("--games", type=int, default=100,
                        help="games generated per chunk (default 100; memory scales with "
                             "this, because each decision carries its 64x84 action block)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="training passes over each chunk (default 3)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="weight of the value loss (default 0.5)")
    parser.add_argument("--particles", type=int, default=8,
                        help="belief particles used to build the belief features")
    parser.add_argument("--belief-history", type=int, default=8,
                        help="how many recent actions feed the belief's Bayes update")
    parser.add_argument("--value-target", type=str, default="win",
                        choices=["win", "score", "mix"],
                        help="what the value head predicts: the game result (win, default), "
                             "the final score margin (score), or half of each (mix). "
                             "MEASURED: all three rank scoring captures about the same, so "
                             "the default stays with the target that matches the search's "
                             "terminal utility. See `heuristic_game` for the data.")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="mass moved off the imitated move (default 0.1)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="fraction of each chunk held out, split by GAME (default 0.15)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=None,
                        help="dropout on the shared state embedding, training only. "
                             "Omit it to inherit an --init-checkpoint's value")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="continue from an existing ReBeL checkpoint")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="where to write the checkpoint (default checkpoints-rebel/bootstrap.pt)")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="optionally write the per-epoch metrics as JSON")
    parser.add_argument("--seed-base", type=int, default=700_000,
                        help="game i uses seed seed_base + i (default 700000)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cuarenta-ai")
    parser.add_argument("--run-name", type=str, default="rebel-bootstrap")
    parser.add_argument("--log-every", type=int, default=25,
                        help="print progress every N games (default 25)")
    return parser


def _wandb_setup(args) -> Optional[object]:
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name,
                         config=vars(args), reinit=True)
        # The repo's convention: an explicit step metric so the x-axis is the iteration.
        wandb.define_metric("iteration")
        for family in ("train", "val", "data", "time"):
            wandb.define_metric(f"{family}/*", step_metric="iteration")
        return run
    except Exception as exc:                     # logging must never kill a run
        print(f"[bootstrap] wandb unavailable, continuing without it: {exc!r}")
        return None


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed_base % (2 ** 31))
    rng = random.Random(args.seed_base)

    cfg = BootstrapConfig(particles=args.particles, belief_history=args.belief_history,
                          label_smoothing=args.label_smoothing,
                          val_fraction=args.val_fraction,
                          value_target=args.value_target, seed=args.seed_base)

    if args.init_checkpoint:
        net = load_rebel_checkpoint(args.init_checkpoint, device, dropout=args.dropout).to(device)
        print(f"[bootstrap] continuing from {args.init_checkpoint}")
    else:
        net = RebelNet(hidden=args.hidden, value_hidden=args.value_hidden,
                       dropout=args.dropout or 0.0).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    run = _wandb_setup(args)

    print(f"[bootstrap] imitating the heuristic for {args.iterations} chunks x "
          f"{args.games} games (epochs={args.epochs}, lr={args.lr}, "
          f"particles={args.particles}, device={args.device})")

    best_key = None
    best_state = None
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        games = generate_games(args.games, args.seed_base + (iteration - 1) * args.games,
                               cfg, rng)
        # Split by GAME, never by decision: decisions inside one game are strongly
        # correlated, so a decision-level split would leak the validation set.
        n_val = max(1, int(round(len(games) * args.val_fraction)))
        val_games = games[:n_val]
        train_games = games[n_val:]
        train_decisions = [d for g in train_games for d in g.decisions]
        val_decisions = [d for g in val_games for d in g.decisions]
        play_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        epoch_metrics: List[Dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_on_buffer(
                net, optimizer, train_decisions, epochs=1, batch_size=args.batch_size,
                value_coef=args.value_coef, device=device, rng=rng)
            val_metrics = validate(net, val_decisions, device)
            merged = {**train_metrics, **val_metrics, "epoch": float(epoch)}
            epoch_metrics.append(merged)
            print(f"  iter {iteration} epoch {epoch}/{args.epochs} | "
                  f"pi {merged.get('train/loss_pi', float('nan')):.4f} "
                  f"vf {merged.get('train/loss_vf', float('nan')):.4f} | "
                  f"val acc {val_metrics.get('val/policy_acc', float('nan')):.3f} "
                  f"mse {val_metrics.get('val/value_mse', float('nan')):.3f} "
                  f"sign {val_metrics.get('val/value_sign_acc', float('nan')):.3f}")
        train_s = time.perf_counter() - t0

        # Best epoch by the same rule as the PPO bootstrap: first require the value head to
        # be better than the "always predict 0" reference, then maximise policy accuracy.
        for merged in epoch_metrics:
            key = (float(merged.get("val/value_mse", 9.9) <= 1.0),
                   float(merged.get("val/policy_acc", 0.0)))
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = dict(merged)
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        record = {
            "iteration": float(iteration),
            "data/train_decisions": float(len(train_decisions)),
            "data/val_decisions": float(len(val_decisions)),
            "data/mean_plies": float(np.mean([g.plies for g in games])),
            "data/belief_depleted": float(sum(g.belief_depleted for g in games)),
            "time/play_s": play_s,
            "time/train_s": train_s,
            **{k: v for k, v in epoch_metrics[-1].items()},
            **{f"best/{k.split('/')[-1]}": v for k, v in best_metrics.items()},
        }
        history.append(record)
        if run is not None:
            import wandb
            wandb.log(record, step=iteration)
        print(f"iter {iteration}/{args.iterations} | {len(train_decisions)} train / "
              f"{len(val_decisions)} val decisions | mean {record['data/mean_plies']:.0f} plies | "
              f"play {play_s:.1f}s train {train_s:.1f}s | best so far: "
              f"acc {best_metrics.get('val/policy_acc', float('nan')):.3f} "
              f"mse {best_metrics.get('val/value_mse', float('nan')):.3f}")

    if best_state is not None:
        net.load_state_dict(best_state)
    out = Path(args.out)
    save_rebel_checkpoint(
        out, net, step=args.iterations,
        config={"stage": "bootstrap", "iterations": args.iterations, "games": args.games,
                "epochs": args.epochs, "lr": args.lr, "label_smoothing": args.label_smoothing,
                "particles": args.particles, "belief_history": args.belief_history,
                "seed_base": args.seed_base, "value_target": args.value_target,
                "val_metrics": {k: float(v) for k, v in best_metrics.items()}})
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(history, indent=2))
    if run is not None:
        import wandb
        wandb.finish()

    elapsed = time.perf_counter() - started
    print()
    print("=" * 68)
    print(f"bootstrap written to {out}  ({elapsed:.0f}s total)")
    print(f"  best epoch  policy acc {best_metrics.get('val/policy_acc', float('nan')):.3f}"
          f"   value mse {best_metrics.get('val/value_mse', float('nan')):.3f}"
          f" (1.0 = always predict 0)"
          f"   value sign acc {best_metrics.get('val/value_sign_acc', float('nan')):.3f}")
    print("  run self-play from it with:")
    print(f"    python3 src/rebel/train.py --init-checkpoint {out}")
    print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
