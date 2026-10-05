#!/usr/bin/env bash
# Local training run with CLI flags.
#
# Usage:
#   ./run_training.sh                              # defaults
#   ./run_training.sh --iters 3 --games-per-iter 8 # smoke test
#   ./run_training.sh --iters 30 --games-per-iter 20

set -e

# Defaults (overridable by env vars OR CLI flags)
INIT_CKPT="${INIT_CKPT:-checkpoints/best-bootstrap.pt}"
ITERS="${ITERS:-30}"
GAMES_PER_ITER="${GAMES_PER_ITER:-20}"
N_SEARCH_ITER="${N_SEARCH_ITER:-100}"

# Parse CLI flags (override env vars)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --init-checkpoint) INIT_CKPT="$2"; shift 2 ;;
        --iters)           ITERS="$2"; shift 2 ;;
        --games-per-iter)  GAMES_PER_ITER="$2"; shift 2 ;;
        --n-search-iter)   N_SEARCH_ITER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --init-checkpoint PATH   (default: $INIT_CKPT)"
            echo "  --iters N                (default: $ITERS)"
            echo "  --games-per-iter N       (default: $GAMES_PER_ITER)"
            echo "  --n-search-iter N        (default: $N_SEARCH_ITER)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "Training config:"
echo "  init checkpoint:  $INIT_CKPT"
echo "  iterations:       $ITERS"
echo "  games/iter:       $GAMES_PER_ITER"
echo "  n_search_iter:    $N_SEARCH_ITER"
echo

uv run src/ismcts/train.py \
    --init-checkpoint "$INIT_CKPT" \
    --iterations "$ITERS" \
    --games-per-iter "$GAMES_PER_ITER" \
    --n-search-iter "$N_SEARCH_ITER"