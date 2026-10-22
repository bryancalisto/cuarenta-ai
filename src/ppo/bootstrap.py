"""
Supervised warm start for PPO: imitate the heuristic agent.

Phase 1 of the pipeline, mirroring `src/ismcts/bootstrap.py`:

  1. Play N games with `HeuristicAgent` at all four seats, recording every decision
     as (observation sequence, candidate-action features + mask, the teacher's slot,
     the final game outcome).
  2. Train the PPO actor-critic to imitate the teacher (cross-entropy over the legal
     slots) and to predict the outcome (MSE against ±1).
  3. Save to `checkpoints-ppo/bootstrap.pt`, which `train.py` warm-starts from.

WHY THIS EXISTS
---------------
Self-play from random weights on a 40-card game wastes a lot of samples rediscovering
"capture when you can". A few thousand supervised games give the policy a competent
starting point — and, just as important, a *value head* that already knows roughly who
is winning, which is what self-play bootstraps from.

WHY SEQUENCES AND NOT SINGLE STEPS
----------------------------------
The policy is recurrent (its LSTM is advanced on every turn of a match), so each game
is stored as one trajectory: the observation of every turn, plus the candidate-action
features of the turns we train on. Training replays the whole trajectory through the
trunk + LSTM (the same contract the PPO update uses, see `rollout.replay_state_embeddings`),
so the hidden state the policy learns from is the one it will actually have in play.

Usage:
    python3 src/ppo/bootstrap.py --games 2000 --epochs 8
    python3 src/ppo/bootstrap.py --games 300 --epochs 4 --no-wandb     # quick
    python3 src/ppo/bootstrap.py --use-lstm false --out checkpoints-ppo/bootstrap_ff.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from env import ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv, TEAMS, action_features_matrix
from utils import OBS_SIZE_ISMCTS, flatten_obs
from agents import HeuristicAgent
from networks import ActorCriticNetwork
from rollout import replay_state_embeddings


@dataclass
class GameTrajectory:
    """
    One imitation game, as a trajectory the recurrent network can replay.

    `obs`            [T, 208]   observation of every turn of the game
    `feats`          [T, S, 84] candidate-action features of every turn
    `masks`          [T, S]     legality mask of every turn
    `actions`        [T]        the slot the teacher chose
    `values`         [T]        the game outcome from the acting team's point of view
    `team`           [T]        the acting team (for reporting)
    """
    obs: np.ndarray
    feats: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    teams: np.ndarray
    winner: int
    plies: int


def play_one_heuristic_game(seed: int) -> GameTrajectory:
    """Play one game with the heuristic at all four seats and record every decision."""
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    agent = HeuristicAgent(seed=seed)

    obs_list, feat_list, mask_list, action_list, team_list = [], [], [], [], []

    while not env.state.done:
        player = env.state.current_player
        legal = env.legal_actions()
        action = agent.act(env.observe(player), legal, state=env.state, team=TEAMS[player])
        slot = legal.index(action)

        feats, mask = action_features_matrix(legal, env.state)
        obs_list.append(flatten_obs(env.observe(player)))
        feat_list.append(feats)
        mask_list.append(mask)
        action_list.append(slot)
        team_list.append(TEAMS[player])

        env.step(action)

    winner = int(env.state.winner)
    teams = np.asarray(team_list, dtype=np.int64)
    # Value target: the game outcome from the acting player's team point of view.
    values = np.where(teams == winner, 1.0, -1.0).astype(np.float32)

    return GameTrajectory(
        obs=np.stack(obs_list).astype(np.float32),
        feats=np.stack(feat_list).astype(np.float32),
        masks=np.stack(mask_list).astype(bool),
        actions=np.asarray(action_list, dtype=np.int64),
        values=values,
        teams=teams,
        winner=winner,
        plies=len(obs_list),
    )


def _collect_worker(seed: int) -> GameTrajectory:
    torch.set_num_threads(1)
    return play_one_heuristic_game(seed)


def collect_games(seeds: List[int]) -> List[GameTrajectory]:
    """Play heuristic games in parallel."""
    with Pool() as pool:
        return list(tqdm(pool.imap(_collect_worker, seeds), total=len(seeds),
                         desc="heuristic games", leave=False))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _game_loss(net: ActorCriticNetwork, game: GameTrajectory, device,
               vf_coef: float, advs: Optional[np.ndarray] = None,
               normalize_advs: bool = True) -> Tuple[torch.Tensor, float, float]:
    """
    Cross-entropy + value MSE for one game (one trajectory).

    `advs` optionally weights the policy term by the game outcome (advantage-weighted
    imitation), so the teacher's moves in games it WON count more than the moves in
    games it lost. Returns (loss, policy_loss, value_loss).
    """
    obs = torch.as_tensor(game.obs, device=device)                    # [T, 208]
    feats = torch.as_tensor(game.feats, device=device)                # [T, S, 84]
    masks = torch.as_tensor(game.masks, device=device)                # [T, S]
    actions = torch.as_tensor(game.actions, device=device)            # [T]
    values_t = torch.as_tensor(game.values, device=device)            # [T]

    # Replay the trajectory through trunk (+LSTM) and score each turn's candidates.
    s_emb = replay_state_embeddings(net, obs, torch.zeros(obs.shape[0], device=device), None)
    a_emb = net.embed_actions(feats)                                  # [T, S, F]
    logits = net.logits_from(s_emb, a_emb).masked_fill(~masks, -1e9)   # [T, S]

    log_probs = F.log_softmax(logits, dim=-1)
    per_turn_ce = -(log_probs.gather(1, actions.unsqueeze(1)).squeeze(1))

    if advs is not None:
        w = torch.as_tensor(advs, device=device, dtype=torch.float32)
        if normalize_advs and float(w.std()) > 1e-6:
            w = (w - w.mean()) / (w.std() + 1e-8)
        w = torch.clamp(w, min=0.0)      # only reinforce moves from games that went well
        weights = 1.0 + w
        policy_loss = (per_turn_ce * weights).sum() / weights.sum().clamp(min=1e-6)
    else:
        policy_loss = per_turn_ce.mean()

    value_loss = F.mse_loss(net.value_head(s_emb).squeeze(-1), values_t)
    return policy_loss + vf_coef * value_loss, float(policy_loss.item()), float(value_loss.item())


@torch.no_grad()
def evaluate(net: ActorCriticNetwork, games: List[GameTrajectory], device,
             vf_coef: float) -> Dict[str, float]:
    net.eval()
    correct = total = 0
    v_se = 0.0
    for game in games:
        obs = torch.as_tensor(game.obs, device=device)
        feats = torch.as_tensor(game.feats, device=device)
        masks = torch.as_tensor(game.masks, device=device)
        s_emb = replay_state_embeddings(net, obs, torch.zeros(obs.shape[0], device=device), None)
        logits = net.logits_from(s_emb, net.embed_actions(feats)).masked_fill(~masks, -1e9)
        correct += int((logits.argmax(-1).cpu().numpy() == game.actions).sum())
        total += len(game.actions)
        v = net.value_head(s_emb).squeeze(-1).cpu().numpy()
        v_se += float(((v - game.values) ** 2).sum())
    net.train()
    return {"policy_acc": correct / max(total, 1), "value_mse": v_se / max(total, 1),
            "n": total}


def train(
    train_games: List[GameTrajectory],
    val_games: List[GameTrajectory],
    *,
    out_path: str,
    epochs: int,
    lr: float,
    use_lstm: bool,
    hidden: int,
    lstm_hidden: int,
    lstm_layers: int,
    value_hidden: int,
    dropout: float,
    vf_coef: float,
    advantage_weighted: bool,
    batch_games: int,
    device: str,
    patience: int,
    use_wandb: bool,
):
    print(f"\nTraining on {len(train_games)} games ({sum(g.plies for g in train_games)} "
          f"decisions); validation on {len(val_games)} games, {device}")

    net = ActorCriticNetwork(
        obs_dim=OBS_SIZE_ISMCTS, action_feat_dim=ACTION_FEAT_DIM, n_slots=MAX_ACTIONS,
        hidden=hidden, use_lstm=use_lstm, lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers, value_hidden=value_hidden, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    best_state, best_key, best_epoch = None, None, -1

    for epoch in range(epochs):
        t0 = time.perf_counter()
        net.train()
        order = list(range(len(train_games)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_games):
            batch = [train_games[i] for i in order[start:start + batch_games]]
            loss_sum = torch.zeros((), device=device)
            for game in batch:
                advs = game.values if advantage_weighted else None
                loss, _pl, _vl = _game_loss(net, game, device, vf_coef, advs=advs)
                loss_sum = loss_sum + loss
            loss = loss_sum / len(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        val = evaluate(net, val_games, device, vf_coef)
        dt = time.perf_counter() - t0
        print(f"  epoch {epoch}: loss={np.mean(losses):.4f} | "
              f"val_acc={val['policy_acc']:.3f} val_value_mse={val['value_mse']:.3f} "
              f"({dt:.1f}s)")
        if use_wandb:
            import wandb
            wandb.log({"bootstrap/epoch": epoch, "bootstrap/loss": float(np.mean(losses)),
                       "bootstrap/val_policy_acc": val["policy_acc"],
                       "bootstrap/val_value_mse": val["value_mse"]})

        # Selection: prefer a usable value head (MSE <= 1.0), then the best policy.
        key = (val["value_mse"] <= 1.0, val["policy_acc"], -val["value_mse"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif patience > 0 and epoch - best_epoch >= patience:
            print(f"  early stop: no held-out improvement for {patience} epochs")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"Restored weights from epoch {best_epoch}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "step": 0,
        "model_state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "ret_rms": {"mean": 0.0, "var": 1.0, "count": 1e-4},
        "config": {"source": "bootstrap", "games": len(train_games), "epochs": epochs,
                   "advantage_weighted": advantage_weighted},
        **net.architecture_config(),
    }, out_path)
    print(f"\nSaved {out_path}")
    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Supervised warm start for PPO")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-games", type=int, default=8)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--advantage-weighted", action="store_true",
                   help="weight the teacher's moves by whether the game was won")
    p.add_argument("--use-lstm", type=str, nargs="?", const="true", default="true",
                   choices=["true", "false"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout on the shared state embedding, training only")
    p.add_argument("--out", type=str, default="checkpoints-ppo/bootstrap.pt")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    use_lstm = args.use_lstm == "true"
    print(f"Using device: {device}  (lstm={use_lstm})")

    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        try:
            import dotenv
            dotenv.load_dotenv()
            import wandb
            wandb.init(project="cuarenta-ai",
                       name=f"ppo-bootstrap-{'lstm' if use_lstm else 'ff'}",
                       group="ppo", config=vars(args))
        except Exception as exc:          # noqa: BLE001 - logging must not be fatal
            print(f"[bootstrap] wandb unavailable, continuing without logging: {exc}")
            wandb_enabled = False

    seeds = list(range(args.seed_base, args.seed_base + args.games))
    rng = random.Random(args.seed_base)
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * args.val_fraction))
    val_seeds, train_seeds = seeds[:n_val], seeds[n_val:]

    t0 = time.perf_counter()
    val_games = collect_games(val_seeds)
    train_games = collect_games(train_seeds)
    print(f"Collected {len(train_games) + len(val_games)} games "
          f"({sum(g.plies for g in train_games)} train decisions) "
          f"in {timedelta(seconds=time.perf_counter() - t0)}")

    train(train_games, val_games, out_path=args.out, epochs=args.epochs, lr=args.lr,
          use_lstm=use_lstm, hidden=args.hidden, lstm_hidden=args.lstm_hidden,
          lstm_layers=args.lstm_layers, value_hidden=args.value_hidden,
          dropout=args.dropout,
          vf_coef=args.vf_coef, advantage_weighted=args.advantage_weighted,
          batch_games=args.batch_games, device=device, patience=args.patience,
          use_wandb=wandb_enabled)

    if wandb_enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
