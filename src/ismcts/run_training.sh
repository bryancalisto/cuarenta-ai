#!/usr/bin/env bash
# Local training run: bootstrap (imitation) + expert-iteration self-play.
#
# Usage:
#   ./run_training.sh                                  # defaults
#   ./run_training.sh --iters 2 --games-per-iter 2     # smoke test
#   ./run_training.sh --bootstrap-only                 # just the imitation phase
#   ./run_training.sh --iters 30 --games-per-iter 20 --n-search-iter 400
#
# IMPORTANT: checkpoints saved before the history/feature fixes are INCOMPATIBLE
# (the LSTM action-history input changed from a 64-slot one-hot to 84 action
# features), so this script retrains the bootstrap from scratch by default. Pass
# BOOTSTRAP_CKPT=... to reuse a checkpoint trained with the current code.

set -e

# Defaults (overridable by env vars OR CLI flags)
BOOTSTRAP_CKPT="${BOOTSTRAP_CKPT:-checkpoints/bootstrap.pt}"
GAMES="${GAMES:-2000}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-12}"
ITERS="${ITERS:-30}"
GAMES_PER_ITER="${GAMES_PER_ITER:-12}"
N_SEARCH_ITER="${N_SEARCH_ITER:-200}"
EVAL_GAMES="${EVAL_GAMES:-60}"
BOOTSTRAP_ONLY=0
SKIP_BOOTSTRAP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap-ckpt)   BOOTSTRAP_CKPT="$2"; shift 2 ;;
        --games)            GAMES="$2"; shift 2 ;;
        --bootstrap-epochs) BOOTSTRAP_EPOCHS="$2"; shift 2 ;;
        --iters)            ITERS="$2"; shift 2 ;;
        --games-per-iter)   GAMES_PER_ITER="$2"; shift 2 ;;
        --n-search-iter)    N_SEARCH_ITER="$2"; shift 2 ;;
        --eval-games)       EVAL_GAMES="$2"; shift 2 ;;
        --bootstrap-only)   BOOTSTRAP_ONLY=1; shift ;;
        --skip-bootstrap)   SKIP_BOOTSTRAP=1; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --bootstrap-ckpt PATH    (default: $BOOTSTRAP_CKPT)"
            echo "  --games N                imitation games (default: $GAMES)"
            echo "  --bootstrap-epochs N     (default: $BOOTSTRAP_EPOCHS)"
            echo "  --iters N                self-play iterations (default: $ITERS)"
            echo "  --games-per-iter N       (default: $GAMES_PER_ITER)"
            echo "  --n-search-iter N        ISMCTS sims per move (default: $N_SEARCH_ITER)"
            echo "  --eval-games N           games per evaluation (default: $EVAL_GAMES)"
            echo "  --bootstrap-only         only run the imitation phase"
            echo "  --skip-bootstrap         only run self-play (needs --bootstrap-ckpt)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "Training config:"
echo "  bootstrap ckpt:   $BOOTSTRAP_CKPT"
echo "  imitation games:  $GAMES (epochs $BOOTSTRAP_EPOCHS)"
echo "  iterations:       $ITERS"
echo "  games/iter:       $GAMES_PER_ITER"
echo "  n_search_iter:    $N_SEARCH_ITER"
echo "  eval games:       $EVAL_GAMES"
echo

# Phase 1: imitate the heuristic agent to get a competent starting policy.
# Its validation metrics (val_acc, val_value_mse) are printed every epoch; a value
# MSE above 1.0 means the value head is worse than always predicting 0.
if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
    uv run src/ismcts/bootstrap.py \
        --games "$GAMES" \
        --epochs "$BOOTSTRAP_EPOCHS" \
        --out "$BOOTSTRAP_CKPT"
fi

if [[ "$BOOTSTRAP_ONLY" -eq 1 ]]; then
    exit 0
fi

# Phase 2: expert iteration. The script evaluates the starting checkpoint FIRST so
# the run always has a same-settings baseline to compare against.
uv run src/ismcts/train.py \
    --init-checkpoint "$BOOTSTRAP_CKPT" \
    --iterations "$ITERS" \
    --games-per-iter "$GAMES_PER_ITER" \
    --n-search-iter "$N_SEARCH_ITER" \
    --eval-games "$EVAL_GAMES"
