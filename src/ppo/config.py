"""
PPO configuration.

Everything the trainer needs lives in one dataclass, and `train.py` builds it from the
command line. Grouped by concern, with the ISMCTS pipeline's defaults where the two
overlap (eval cadence, gating threshold, checkpoint directory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils import OpponentMode

# Warm start produced by `src/ppo/bootstrap.py` (imitation of the heuristic agent).
DEFAULT_POLICY_INIT = "checkpoints-ppo/bootstrap.pt"
# Where checkpoints, plots and metrics land (paths are relative to the CWD, like the
# ISMCTS pipeline's `checkpoints/`).
DEFAULT_SAVE_DIR = "checkpoints-ppo"


@dataclass
class PPOConfig:
    # ---- rollout collection ----
    n_envs: int = 8                  # parallel games
    n_steps: int = 512               # turns collected per env per update

    # ---- PPO update ----
    n_epochs: int = 4
    minibatch_size: int = 128        # used only when use_lstm=False
    envs_per_minibatch: int = 4      # used only when use_lstm=True (whole env seqs)
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95

    # ---- optimiser ----
    lr: float = 3e-4

    # ---- model (ISMCTS-style: observation embedding + per-action feature scoring) ----
    hidden: int = 256                # trunk width and action-embedding width
    use_lstm: bool = True
    lstm_hidden: int = 128
    lstm_layers: int = 1
    value_hidden: int = 128
    # Training-time dropout on the shared state embedding (0 = off). Identity in eval
    # mode, so acting, evaluation and the ONNX export are unaffected.
    dropout: float = 0.0

    # ---- reward shaping ----
    # True (default): an opponent's reward counts as a NEGATIVE reward for the learner,
    # which is what a zero-sum game implies. This matters most for the terminal ±50:
    # without it the learner's episode return is never negative and the critic cannot
    # learn "we are losing".
    symmetrize_rewards: bool = True

    # ---- self-play opponents ----
    pool_size: int = 20              # past checkpoints kept as opponents
    opponent_latest: float = 0.8     # probability of facing the newest snapshot
    opponent_mode: str = OpponentMode.SELFPLAY

    # ---- evaluation league (see trainer.evaluate) ----
    eval_games: int = 60             # games per match (60 => ~±6% standard error)
    eval_every: int = 100_000        # evaluate every N environment steps
    accept_threshold: float = 0.55   # win rate vs the reference needed to be accepted

    # ---- checkpointing / logging / length ----
    checkpoint_every: int = 100_000
    log_every: int = 1
    save_dir: str = DEFAULT_SAVE_DIR
    init_policy: Optional[str] = DEFAULT_POLICY_INIT
    total_steps: int = 5_000_000
    skip_baseline_eval: bool = False  # tests set this True to keep runs short
    run_name: str = "ppo"
    wandb_project: str = "cuarenta-ai"   # same project as the ISMCTS runs
    wandb: bool = True
    save_plots: bool = True
