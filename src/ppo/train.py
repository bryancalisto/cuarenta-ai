"""
Command-line entry point for self-play PPO.

Examples (run from the repository root):

    # 1) supervised warm start (recommended first step, a few minutes)
    python3 src/ppo/bootstrap.py --games 2000 --epochs 8

    # 2) self-play PPO from that checkpoint
    python3 src/ppo/train.py --use-lstm true --n-envs 10 --steps 2_000_000

    # quick smoke test (seconds, no wandb)
    python3 src/ppo/train.py --steps 8192 --n-envs 2 --n-steps 64 --no-wandb \
        --save-dir /tmp/ppo_smoke --init-policy none

Metrics go to Weights & Biases (project `cuarenta-ai`, the same one the ISMCTS runs
use) under four families, all plotted against the training step:

    train/*     losses, entropy, approx_kl, clip_frac              (PPO health)
    selfplay/*  win rate, scores, returns, caídas, game length     (the games we make)
    eval/*      win rate + scores + returns vs the reference, the heuristic bot, the
                starting policy, the previous checkpoint and a random bot
    time/*      steps per second and where the seconds went
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_POLICY_INIT, DEFAULT_SAVE_DIR, PPOConfig
from trainer import PPOTrainer


def str2bool(value):
    """argparse helper: --use-lstm true / false (and bare --use-lstm means true)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-play PPO for Cuarenta")
    # rollout / length
    p.add_argument("--steps", type=int, default=5_000_000, help="total environment steps")
    p.add_argument("--n-envs", type=int, default=8, help="parallel games")
    p.add_argument("--n-steps", type=int, default=512, help="turns per env per update")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--envs-per-minibatch", type=int, default=4,
                   help="minibatch size in ENVIRONMENTS when --use-lstm true")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # model
    p.add_argument("--use-lstm", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--hidden", type=int, default=256, help="trunk and action-embedding width")
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--value-hidden", type=int, default=128)
    # opponents
    p.add_argument("--opponent-mode", type=str, default="selfplay",
                   choices=["heuristic", "selfplay", "random"])
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--opponent-latest", type=float, default=0.8,
                   help="probability of facing the newest snapshot instead of an older one")
    # reward shaping
    p.add_argument("--no-symmetrize-rewards", dest="symmetrize_rewards",
                   action="store_false", default=True,
                   help="revert to dropping opponent rewards (A/B only: it makes the "
                        "critic blind to how the learner can lose)")
    # league / evaluation
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-games", type=int, default=60,
                   help="games per match (60 gives ~6%% standard error)")
    p.add_argument("--accept-threshold", type=float, default=0.55,
                   help="win rate vs the reference needed to move it forward")
    # checkpoints / logging
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                   help="where checkpoints/plots/npz go (default: checkpoints-ppo)")
    p.add_argument("--init-policy", type=str, default=DEFAULT_POLICY_INIT,
                   help="checkpoint to warm start from; 'none' trains from scratch")
    p.add_argument("--resume", type=str, default=None, help="snapshot to resume from")
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=1, help="rollouts between progress updates")
    p.add_argument("--run-name", type=str, default="ppo", help="wandb run name/group")
    p.add_argument("--wandb-project", type=str, default="cuarenta-ai")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", default=True)
    p.add_argument("--no-plots", dest="save_plots", action="store_false", default=True)
    # misc
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument("--seed", type=int, default=0, help="seed for python/numpy/torch RNGs")
    return p


def config_from_args(args) -> PPOConfig:
    init_policy = args.init_policy
    if init_policy and init_policy.lower() == "none":
        init_policy = None
    return PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        envs_per_minibatch=args.envs_per_minibatch,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        hidden=args.hidden,
        use_lstm=args.use_lstm,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        value_hidden=args.value_hidden,
        symmetrize_rewards=args.symmetrize_rewards,
        pool_size=args.pool_size,
        opponent_latest=args.opponent_latest,
        opponent_mode=args.opponent_mode,
        eval_games=args.eval_games,
        eval_every=args.eval_every,
        accept_threshold=args.accept_threshold,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        save_dir=args.save_dir,
        init_policy=init_policy,
        total_steps=args.steps,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        wandb=args.wandb,
        save_plots=args.save_plots,
    )


def main() -> None:
    args = build_parser().parse_args()

    # One seed for all three RNGs: self-play is stochastic (sampled actions and
    # minibatch order), so without this a run cannot be reproduced.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = config_from_args(args)
    trainer = PPOTrainer(cfg, device=args.device)

    if cfg.init_policy:
        init_path = Path(cfg.init_policy)
        if init_path.exists():
            trainer.load_policy(str(init_path))
        else:
            print(f"[PPO] init policy not found, training from scratch: {init_path}\n"
                  f"      (run `python3 src/ppo/bootstrap.py` to create one)")

    if args.resume:
        trainer.load_policy(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
