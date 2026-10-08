#!/usr/bin/env bash
# PPO training: supervised warm start (optional) + self-play, in one command.
#
# Usage (from anywhere; the script cd's to the repository root):
#   ./src/ppo/run_training.sh                     # bootstrap (if needed) + 2M steps
#   ./src/ppo/run_training.sh --smoke             # tiny run, no wandb (seconds)
#   ./src/ppo/run_training.sh --skip-bootstrap    # reuse checkpoints-ppo/bootstrap.pt
#   ./src/ppo/run_training.sh --bootstrap-only
#   ./src/ppo/run_training.sh --feedforward --steps 500000
#
# Outputs land in checkpoints-ppo/ (policy_<step>.pt snapshots, best.pt = the net that
# beat the reference, best_heuristic.pt = the best against the rule-based bot, plus a
# training_metrics_*.png summary). Metrics go to wandb project cuarenta-ai.

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-checkpoints-ppo}"
BOOTSTRAP_CKPT="${BOOTSTRAP_CKPT:-$SAVE_DIR/bootstrap.pt}"
STEPS="${STEPS:-2000000}"
N_ENVS="${N_ENVS:-10}"
N_STEPS="${N_STEPS:-512}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-cpu}"
USE_LSTM="${USE_LSTM:-true}"
BOOTSTRAP_GAMES="${BOOTSTRAP_GAMES:-2000}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-8}"
BOOTSTRAP_ONLY=0
SKIP_BOOTSTRAP=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-dir)      SAVE_DIR="$2"; BOOTSTRAP_CKPT="$SAVE_DIR/bootstrap.pt"; shift 2 ;;
        --steps)         STEPS="$2"; shift 2 ;;
        --n-envs)        N_ENVS="$2"; shift 2 ;;
        --n-steps)       N_STEPS="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --bootstrap-games)  BOOTSTRAP_GAMES="$2"; shift 2 ;;
        --bootstrap-epochs) BOOTSTRAP_EPOCHS="$2"; shift 2 ;;
        --feedforward)   USE_LSTM="false"; shift ;;
        --bootstrap-only) BOOTSTRAP_ONLY=1; shift ;;
        --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
        --smoke)
            STEPS=4096; N_ENVS=2; N_STEPS=64; SAVE_DIR="/tmp/ppo_smoke"
            BOOTSTRAP_CKPT="$SAVE_DIR/bootstrap.pt"
            BOOTSTRAP_GAMES=40; BOOTSTRAP_EPOCHS=2
            EXTRA_ARGS+=(--no-wandb --no-plots --eval-games 6 --eval-every 2048 --log-every 50)
            shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "PPO training config:"
echo "  steps:            $STEPS  (n_envs=$N_ENVS x n_steps=$N_STEPS)"
echo "  lr:               $LR"
echo "  use_lstm:         $USE_LSTM"
echo "  save dir:         $SAVE_DIR"
echo "  bootstrap ckpt:   $BOOTSTRAP_CKPT"
echo "  device:           $DEVICE"
echo

# Phase 1: supervised warm start. Mirrors src/ismcts/run_training.sh: imitate the
# heuristic agent first, so self-play starts from a competent policy AND a value head
# that already knows roughly who is winning.
if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
    if [[ -f "$BOOTSTRAP_CKPT" ]]; then
        echo "Bootstrap checkpoint already exists, keeping it: $BOOTSTRAP_CKPT"
    else
        BOOTSTRAP_WANDB_ARGS=()
        for a in "${EXTRA_ARGS[@]}"; do
            [[ "$a" == "--no-wandb" ]] && BOOTSTRAP_WANDB_ARGS+=(--no-wandb)
        done
        python3 src/ppo/bootstrap.py \
            --games "$BOOTSTRAP_GAMES" \
            --epochs "$BOOTSTRAP_EPOCHS" \
            --use-lstm "$USE_LSTM" \
            --out "$BOOTSTRAP_CKPT" \
            "${BOOTSTRAP_WANDB_ARGS[@]}"
    fi
fi

if [[ "$BOOTSTRAP_ONLY" -eq 1 ]]; then
    exit 0
fi

# Phase 2: self-play PPO, warm-started from the bootstrap.
python3 src/ppo/train.py \
    --steps "$STEPS" \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --lr "$LR" \
    --use-lstm "$USE_LSTM" \
    --device "$DEVICE" \
    --save-dir "$SAVE_DIR" \
    --init-policy "$BOOTSTRAP_CKPT" \
    "${EXTRA_ARGS[@]}"
