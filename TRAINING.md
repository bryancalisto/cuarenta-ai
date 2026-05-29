# Training Guide for Cuarenta AI

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run tests:**
```bash
python src/tests.py
```
All core tests pass. Some random agent games may timeout (edge case in game logic), but this doesn't affect training.

## Training

Start PPO training with default hyperparameters:
```bash
python src/ppo.py
```

At the end of a run, PPO now saves `training_metrics_*.png` and `training_metrics_*.npz` in `data/policies/`.

Default — warm start from `data/policies/cuarenta_policy_final.pt` and train in self-play

```
python src/ppo.py
```

To train against the heuristic instead, override the opponent mode:

```
python src/ppo.py --opponent-mode heuristic
```

Sanity check baseline

```
python src/ppo.py --opponent-mode random
```

### Custom hyperparameters:

```bash
python src/ppo.py \
  --steps 5_000_000 \
  --n-envs 8 \
  --n-steps 512 \
  --lr 3e-4 \
  --save-dir data/policies
```

**Key options:**
- `--steps`: Total training steps (default: 5M)
- `--n-envs`: Parallel environments (default: 8)
- `--n-steps`: Steps per rollout (default: 512)
- `--lr`: Learning rate (default: 3e-4)
- `--n-epochs`: Gradient epochs per rollout
- `--caida-exposure-penalty`: Reward penalty for leaving a card exposed to an immediate caida by the next player
- `--save-dir`: Policy snapshot directory (default: `data/policies/`)
- `--init-policy`: Warm-start policy path (default: `data/policies/cuarenta_policy_final.pt`)

### Resume training:
```bash
python src/ppo.py --resume data/policies/cuarenta_policy_iter1.pt
```

## Game Environment

The environment (`src/env.py`):
- **Players**: 4-player game with 2 teams (0,2) vs (1,3)
- **Cards**: 40-card Spanish deck (ranks 1-7, 11, 12, 13)
- **Win condition**: First team to 40 points wins
- **Observation**: 40-dim card vectors + scalars (scores, player, deck)
- **Actions**: Play a card with optional capture set

## Architecture

- **Model** (`src/model.py`): Transformer-based encoder + policy/value heads
- **Training** (`src/ppo.py`): PPO with self-play against a policy snapshot pool
- **Captures** (`src/captures.py`): Legal move generation and scoring logic
