"""
Bootstrap CuarentaNet via imitation learning from the HeuristicAgent.

Phase 1 of the training plan:
  1. Generate N games where all four seats play HeuristicAgent.
  2. Record every decision as a training sample:
       (obs_seq, act_seq, act_mask, action_features, action_mask, one_hot_action, outcome)
  3. Train CuarentaNet to imitate the heuristic (cross-entropy) and predict
      game outcome (MSE).

This gives the net a competent starting policy. Subsequent self-play starts
from a strong baseline instead of random weights, which makes the whole
training loop dramatically more productive.

Two things that matter for trusting the result:

 * TRAIN/VAL SPLIT. Games are split by seed into a training set and a held-out
   validation set, and every epoch reports validation numbers. Without this you
   cannot tell learning from memorisation: an earlier version of this script
   reported `value=0.036` because the value head had memorised the outcome of the
   few hundred games it was trained on (one ±1 label per game is trivially
   memorisable). On genuinely unseen games that same checkpoint had value MSE
   1.49 — worse than predicting 0 (MSE 1.0 for ±1 targets).
 * The teacher is only as good as `heuristic_agent.py`. It used to never play for
   a caída or a limpia at all (a `list == str` bug), so the bootstrap policy
   inherited that blind spot.

Usage:
    uv run src/ismcts/bootstrap.py --games 2000 --epochs 20
    uv run src/ismcts/bootstrap.py --games 300 --epochs 5 --no-wandb     # quick
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import List, Tuple
import dotenv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from env import CuarentaEnv, TEAMS, NUM_PLAYERS
from heuristic_agent import HeuristicAgent
from history import HistoryTracker, flatten_obs, ACT_FEAT_DIM
from network import CuarentaNet, N_ACTIONS, ACTION_FEAT_DIM, VALUE_TARGETS
from selfplay import backfill_value_targets
from dataset import CuarentaDataset
from selfplay import Decision


dotenv.load_dotenv()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T_MAX = 32
DEFAULT_GAMES = 2000
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
# Fraction of games held out for validation.
DEFAULT_VAL_FRACTION = 0.1
# Randomly shorten histories during training so the net also works with no history
# at all — which is what interior search nodes have (see dataset.py).
HISTORY_DROPOUT = 0.3
# Held-out value MSE above this is unusable: for ±1 targets it is exactly the MSE of
# always predicting 0 ("no idea who wins"). See `evaluate` for what it means in return
# mode (it errs toward discarding value heads, which is the safe direction).
VALUE_MSE_BAR = 1.0


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def play_one_heuristic_game(seed: int, value_target: str = "win") -> List[Decision]:
    """
    Play a full game with HeuristicAgent at all four seats.
    Return one Decision per decision point (from the acting player's POV).

    `value_target` decides what the value head is taught from this game, and it must match
    the net's head (see network.py): ±1 for the game result, or the scaled shaped
    return-to-go. The teacher's POLICY target is the same in both cases (its one-hot move);
    only the value target changes.

    The return target here is Monte-Carlo (lambda = 1) whatever the training run's
    `--value-lambda` is: this function plays no search, so it has no value estimates to
    bootstrap from (`bootstrap_values=None`). Both are estimates of the same quantity, so
    the only difference is each label's variance.
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    agent = HeuristicAgent(seed=seed)
    tracker = HistoryTracker(env)
    decisions: List[Decision] = []
    ply = 0
    ply_rewards: List[float] = []
    ply_actor_teams: List[int] = []

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()

        # Heuristic chooses an Action object; we work with its slot index.
        action = agent.act(env, p)
        slot = legal.index(action)

        # Net inputs: history + the CURRENT observation as a pending step.
        obs_seq, act_seq, act_mask = tracker.inputs_for(p, obs, max_len=T_MAX)

        # Imitation target: one-hot over the slot the teacher picked.
        visits = np.zeros(N_ACTIONS, dtype=np.int64)
        visits[slot] = 1

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
            # Teacher data, not self-play data: the one-hot target is the ground
            # truth we are imitating, so the dataset must not soften it.
            selfplay=False,
        ))

        tracker.record(p, obs, slot)
        # env.step is (obs, reward, done, truncated, info): the reward is what the
        # return-to-go target is built from, not just a byproduct to discard.
        _, reward, _, _, _ = env.step(slot)
        ply_rewards.append(float(reward))
        ply_actor_teams.append(TEAMS[p])
        ply += 1

    # Value target from the acting player's team POV — the SAME function self-play uses,
    # so a return-target dataset cannot silently mix in ±1 samples.
    backfill_value_targets(decisions, ply_rewards, ply_actor_teams,
                           env.state.winner, value_target)

    return decisions


def _collect_worker(payload) -> List[Decision]:
    """One game in a worker process. `payload` is (seed, value_target) — a tuple, because
    a spawned worker only sees module defaults, never the parent's CLI values."""
    seed, value_target = payload
    import torch
    torch.set_num_threads(1)
    return play_one_heuristic_game(seed, value_target)


def collect_heuristic_games(seeds: List[int], value_target: str = "win") -> List[List[Decision]]:
    """Parallel data collection. Returns one list of Decisions per game."""
    n_proc = os.cpu_count() or 1
    print(f"Collecting {len(seeds)} heuristic games with {n_proc} workers "
          f"(value target: {value_target})...")

    t0 = time.perf_counter()
    with Pool(n_proc) as pool:
        results = list(tqdm(
            pool.imap(_collect_worker, [(s, value_target) for s in seeds]),
            total=len(seeds),
            desc="games",
        ))

    n_dec = sum(len(g) for g in results)
    dt = time.perf_counter() - t0
    print(f"Collected {n_dec} decisions in {timedelta(seconds=dt)}")
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _unpack(batch, device):
    obs, act, am, feats, mask, pi_target, v_target, length, pol_weight = batch
    return (
        obs.to(device), act.to(device), am.to(device), feats.to(device),
        mask.to(device), pi_target.to(device), v_target.to(device), length.to(device),
        pol_weight.to(device),
    )


def train_one_epoch(net, loader, opt, device):
    net.train()
    total_loss = total_p = total_v = total_ent = 0.0
    total_correct = total_samples = 0

    for batch in loader:
        obs, act, am, feats, mask, pi_target, v_target, length, pol_weight = _unpack(batch, device)

        logits, value = net(obs, act, am, feats, mask, lengths=length)

        # Per-sample cross-entropy, then a weighted mean: samples with policy
        # weight 0 (imitation data added just for the value head) do not push the
        # policy at all. In bootstrap every sample is a teacher one-hot, so the
        # weights are all 1.0.
        logp = torch.log_softmax(logits, dim=-1)
        ce = -(pi_target * logp).sum(dim=-1)
        w = pol_weight.clamp(min=0.0)
        policy_loss = (ce * w).sum() / w.sum().clamp(min=1e-6)

        # Value loss: MSE against the value target -- ±1 game results in win mode, the
        # scaled return-to-go in return mode (see `backfill_value_targets`).
        value_loss = ((value - v_target) ** 2).mean()

        loss = policy_loss + value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Diagnostic: the entropy of the target distribution is the FLOOR of the
        # cross-entropy (you cannot do better than predicting the target exactly).
        # `policy_loss - target_entropy` is the part the net actually fails to fit.
        # Weighted the same way as `policy_loss` so the two are comparable.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            target = pi_target.argmax(dim=-1)
            total_correct += (pred == target).sum().item()
            total_samples += obs.size(0)
            ent = -(pi_target * torch.log(pi_target + 1e-9)).sum(dim=-1)
            total_ent += ((ent * w).sum() / w.sum().clamp(min=1e-6)).item()

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()

    n = max(len(loader), 1)
    return (total_loss / n, total_p / n, total_v / n,
            total_correct / max(total_samples, 1), total_ent / n)


@torch.no_grad()
def evaluate(net, loader, device):
    """
    Held-out metrics. `value_mse` is compared against `VALUE_MSE_BAR` (1.0), which is the
    MSE of always predicting 0 for ±1 targets, i.e. "I have no idea who wins". A model
    above that is worse than admitting ignorance (overconfident and wrong).

    The bar means the same thing in both modes, which is the point of
    `RETURN_VALUE_SCALE`: for ±1 targets the variance is ~1 and for scaled return targets
    it is ~1 by construction (the shaped return-to-go has mean ~0 and std ~35 points, and
    the targets are divided by 30), so `mse > VALUE_MSE_BAR` is "worse than predicting the
    mean" either way. Predicting exactly 0 is also the near-optimal constant in both modes,
    since neither target has a large mean.
    """
    net.eval()
    total = 0
    correct = 0
    v_se = 0.0
    v_correct = 0
    for batch in loader:
        obs, act, am, feats, mask, pi_target, v_target, length, _pw = _unpack(batch, device)
        logits, value = net(obs, act, am, feats, mask, lengths=length)
        correct += (logits.argmax(-1) == pi_target.argmax(-1)).sum().item()
        v_se += ((value - v_target) ** 2).sum().item()
        v_correct += ((value > 0) == (v_target > 0)).sum().item()
        total += obs.size(0)
    return {
        "policy_acc": correct / max(total, 1),
        "value_mse": v_se / max(total, 1),
        "value_sign_acc": v_correct / max(total, 1),
        "n": total,
    }


def train(
    train_decisions: List[Decision],
    val_decisions: List[Decision],
    *,
    out_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
    use_wandb: bool = True,
    patience: int = 3,
    value_target: str = "win",
):
    print(f"\nTraining on {len(train_decisions)} samples "
          f"(val {len(val_decisions)}) for {epochs} epochs on {device}")

    train_ds = CuarentaDataset(train_decisions, t_max=T_MAX, history_dropout=HISTORY_DROPOUT)
    val_ds = CuarentaDataset(val_decisions, t_max=T_MAX, history_dropout=0.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # The head must match the target the data carries: `tanh` (win) or linear (return).
    net = CuarentaNet(value_target=value_target).to(device)
    # Mild weight decay: the value head overfits quickly here (in win mode there is one
    # ±1 label per game, and all decisions in a game share it).
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_key = None
    best_mse = float("inf")
    best_epoch = -1
    best_acc = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        loss, pl, vl, acc, tent = train_one_epoch(net, train_loader, opt, device)
        val = evaluate(net, val_loader, device)
        dt = time.perf_counter() - t0

        print(f"  epoch {epoch}: loss={loss:.4f} policy={pl:.4f} value={vl:.4f} "
              f"train_acc={acc:.3f} | val_acc={val['policy_acc']:.3f} "
              f"val_value_mse={val['value_mse']:.3f} "
              f"val_value_sign={val['value_sign_acc']:.3f} "
              f"| target_ent={tent:.3f} excess_ce={pl - tent:.3f} ({dt:.1f}s)")

        if use_wandb:
            wandb.log({
                "bootstrap/epoch": epoch,
                "bootstrap/loss": loss,
                "bootstrap/policy_loss": pl,
                "bootstrap/value_loss": vl,
                "bootstrap/action_accuracy": acc,
                "bootstrap/val_action_accuracy": val["policy_acc"],
                "bootstrap/val_value_mse": val["value_mse"],
                "bootstrap/val_value_sign_acc": val["value_sign_acc"],
                "bootstrap/target_entropy": tent,
                "bootstrap/excess_ce": pl - tent,
            })

        # ---- model selection on held-out metrics ----
        #
        # The two heads want different stopping points: the policy keeps improving
        # with more epochs, while the value head starts overfitting (its train loss
        # falls — in win mode one ±1 label per game is easy to memorise — while held-out
        # MSE RISES, eventually above the VALUE_MSE_BAR = worse than predicting 0).
        #
        # Criterion, in order:
        #   1. prefer epochs whose held-out value MSE is at most VALUE_MSE_BAR (usable
        #      head — the search uses it to value leaves),
        #   2. among those, the highest held-out policy accuracy,
        #   3. tie-break on the lower value MSE.
        # Packed into a tuple because Python compares tuples left to right.
        key = (val["value_mse"] <= VALUE_MSE_BAR, val["policy_acc"],
               -val["value_mse"])
        if best_key is None or key > best_key:
            best_key = key
            best_mse = val["value_mse"]
            best_acc = val["policy_acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif patience > 0 and epoch - best_epoch >= patience:
            print(f"  early stop: no held-out improvement for {patience} epochs")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nRestored weights from epoch {best_epoch} "
              f"(val_acc={best_acc:.3f}, val_value_mse={best_mse:.3f})")
    if best_mse > VALUE_MSE_BAR:
        print(f"WARNING: held-out value MSE is above {VALUE_MSE_BAR} — the value head is "
              "worse than always predicting the target's mean (overconfident). Generate "
              "more imitation games (or regularise) before trusting it in the search.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f"\nSaved {out_path}")

    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="Number of heuristic games to generate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--out", type=str, default="checkpoints/bootstrap.pt")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="Fraction of GAMES held out for validation")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop if held-out value MSE does not improve for this "
                             "many epochs (0 = train the full budget)")
    parser.add_argument("--value-target", type=str, default="win",
                        choices=list(VALUE_TARGETS),
                        help="what the value head predicts: the ±1 game result (win, "
                             "default) or the env's dense reward as a scaled "
                             "return-to-go (return). Must be the same mode the training "
                             "run uses, and it changes the head's activation, so it is "
                             "NOT interchangeable with an existing checkpoint. See the "
                             "VALUE TARGET note in network.py.")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | mps | cuda. Defaults to auto.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint to initialize the net from. "
             "NOTE: checkpoints saved before the history/feature fixes are "
             "incompatible (act_hist_proj input changed from 64 slots to 84 "
             "features); such a load will fail with a shape error.",
    )

    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(
            project="cuarenta-ai",
            name="bootstrap",
            config={
                "games": args.games,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "t_max": T_MAX,
                "val_fraction": args.val_fraction,
                "history_dropout": HISTORY_DROPOUT,
                "device": device,
                "value_target": args.value_target,
            },
        )

    # Split by GAME (not by decision): decisions inside one game share an outcome
    # label, so a decision-level split would leak the label into validation.
    seeds = list(range(args.seed_base, args.seed_base + args.games))
    rng = random.Random(args.seed_base)
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * args.val_fraction))
    val_seeds, train_seeds = seeds[:n_val], seeds[n_val:]

    games = collect_heuristic_games(train_seeds, args.value_target)
    val_games = collect_heuristic_games(val_seeds, args.value_target)

    train_decisions = [d for g in games for d in g]
    val_decisions = [d for g in val_games for d in g]

    train(
        train_decisions,
        val_decisions,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_wandb=not args.no_wandb,
        patience=args.patience,
        value_target=args.value_target,
    )

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
