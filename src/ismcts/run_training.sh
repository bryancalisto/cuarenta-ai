#!/usr/bin/env bash
# Local training run: bootstrap (imitation) + expert-iteration self-play.
#
# Usage:
#   ./run_training.sh                                     # defaults (long: ~4h)
#   ./run_training.sh --iters 2 --games-per-iter 2        # smoke test
#   ./run_training.sh --bootstrap-only                    # just the imitation phase
#   ./run_training.sh --iters 30 --games-per-iter 20 --n-search-iter 400
#
# VALUE TARGET: the net's value head (and therefore what the search optimizes) is the ±1
# game result by default. `--value-target return` switches both to the env's dense reward
# as a scaled return-to-go, so the search chases caídas/captures instead of win
# probability. It changes the head's activation, so a return-mode run cannot warm-start
# from a win-mode checkpoint (the script warns): pass --value-target return HERE and it
# will also train the bootstrap in that mode.
#
# IMPORTANT: checkpoints saved before the history/feature fixes are INCOMPATIBLE
# (the LSTM action-history input changed from a 64-slot one-hot to 84 action
# features), so this script retrains the bootstrap from scratch by default. Pass
# BOOTSTRAP_CKPT=... to reuse a checkpoint trained with the current code.
#
# The self-play phase prints four evaluation numbers per evaluation:
#   vs_reference   win rate against the best net so far  <- the gating criterion
#   vs_heuristic   win rate against the fixed heuristic  <- external reference
#   teacher_edge   search vs this net's own raw policy   <- must stay well above 50%
#   probe_*        held-out policy accuracy / value MSE  <- value MSE must stay <= 1.0

set -e

# Defaults (overridable by env vars OR CLI flags)
BOOTSTRAP_CKPT="${BOOTSTRAP_CKPT:-checkpoints/bootstrap.pt}"
GAMES="${GAMES:-2000}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-12}"
ITERS="${ITERS:-30}"
GAMES_PER_ITER="${GAMES_PER_ITER:-40}"
IMITATION_GAMES="${IMITATION_GAMES:-5}"
N_SEARCH_ITER="${N_SEARCH_ITER:-400}"
EVAL_GAMES="${EVAL_GAMES:-60}"
VALUE_TARGET="${VALUE_TARGET:-win}"
BOOTSTRAP_ONLY=0
SKIP_BOOTSTRAP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap-ckpt)   BOOTSTRAP_CKPT="$2"; shift 2 ;;
        --games)            GAMES="$2"; shift 2 ;;
        --bootstrap-epochs) BOOTSTRAP_EPOCHS="$2"; shift 2 ;;
        --iters)            ITERS="$2"; shift 2 ;;
        --games-per-iter)   GAMES_PER_ITER="$2"; shift 2 ;;
        --imitation-games)  IMITATION_GAMES="$2"; shift 2 ;;
        --n-search-iter)    N_SEARCH_ITER="$2"; shift 2 ;;
        --eval-games)       EVAL_GAMES="$2"; shift 2 ;;
        --value-target)     VALUE_TARGET="$2"; shift 2 ;;
        --bootstrap-only)   BOOTSTRAP_ONLY=1; shift ;;
        --skip-bootstrap)   SKIP_BOOTSTRAP=1; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --bootstrap-ckpt PATH    (default: $BOOTSTRAP_CKPT)"
            echo "  --games N                imitation games (default: $GAMES)"
            echo "  --bootstrap-epochs N     (default: $BOOTSTRAP_EPOCHS)"
            echo "  --iters N                self-play iterations (default: $ITERS)"
            echo "  --games-per-iter N       (default: $GAMES_PER_ITER)"
            echo "  --imitation-games N      teacher games mixed in (default: $IMITATION_GAMES)"
            echo "  --n-search-iter N        ISMCTS sims per move (default: $N_SEARCH_ITER)"
            echo "  --eval-games N           games per match (default: $EVAL_GAMES)"
            echo "  --value-target win|return  what the value head learns and the search"
            echo "                           optimizes (default: $VALUE_TARGET)"
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
echo "  games/iter:       $GAMES_PER_ITER self-play + $IMITATION_GAMES teacher"
echo "  n_search_iter:    $N_SEARCH_ITER"
echo "  eval games:       $EVAL_GAMES per match"
echo "  value target:     $VALUE_TARGET"
echo

# Phase 1: imitate the heuristic agent to get a competent starting policy.
# Its validation metrics (val_acc, val_value_mse) are printed every epoch; a value
# MSE above 1.0 means the value head is worse than always predicting 0. The saved
# weights are the best epoch by (usable value MSE, then policy accuracy).
if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
    uv run src/ismcts/bootstrap.py \
        --games "$GAMES" \
        --epochs "$BOOTSTRAP_EPOCHS" \
        --value-target "$VALUE_TARGET" \
        --out "$BOOTSTRAP_CKPT"
fi

if [[ "$BOOTSTRAP_ONLY" -eq 1 ]]; then
    exit 0
fi

# Phase 2: expert iteration. The script evaluates the starting checkpoint FIRST so
# the run always has a same-settings baseline to compare against, and uses that
# self-match as the "mirror" calibration for gating (team 0 may have a seat edge).
uv run src/ismcts/train.py \
    --init-checkpoint "$BOOTSTRAP_CKPT" \
    --iterations "$ITERS" \
    --games-per-iter "$GAMES_PER_ITER" \
    --imitation-games "$IMITATION_GAMES" \
    --n-search-iter "$N_SEARCH_ITER" \
    --value-target "$VALUE_TARGET" \
    --eval-games "$EVAL_GAMES"
