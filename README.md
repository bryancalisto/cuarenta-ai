# 40-ai


# Setup
conda install torch numpy

## Run tests first to verify the env
python tests.py

## Train
python src/ppo.py

## Evaluate after training
python evaluate.py data/policies/cuarenta_policy_iter1.pt --mode vs-heuristic
python evaluate.py data/policies/cuarenta_policy_iter1.pt --mode watch

# Cuarenta RL Agent

PPO + self-play agent for the Ecuadorian card game Cuarenta, implemented in PyTorch.

## Files

| File | Description |
|---|---|
| `card.py` | 40-card Spanish deck — suits, ranks, card index |
| `captures.py` | Legal capture logic — match, sum, caída, ronda |
| `env.py` | 4-player game environment (PettingZoo-style) |
| `model.py` | PyTorch actor-critic — Transformer encoder + two-headed policy |
| `ppo.py` | PPO trainer with GAE, self-play checkpoint pool, optional caida-exposure shaping |
| `evaluate.py` | Evaluation — vs random, vs heuristic, watch mode |
| `tests.py` | Full test suite for the environment |

## Setup

```bash
pip install torch numpy
```

## Train

```bash
# Basic training (warm starts from data/policies/cuarenta_policy_final.pt)
python src/ppo.py

# Custom config
python src/ppo.py --steps 10000000 --n-envs 16 --lr 1e-4 --save-dir data/policies

# Resume from checkpoint
python src/ppo.py --resume data/policies/cuarenta_policy_iter1.pt
```

Training prints a log line every 10 rollouts:
```
step=   81920 | sps=  4231 | win_rate=0.523 | loss_pi=0.0312 | loss_vf=0.4821 | entropy=2.341 | ep_rew=12.4
```

## Evaluate

```bash
# Win rate vs random agent (200 games)
python evaluate.py data/policies/cuarenta_policy_iter1.pt --mode vs-random

# Win rate vs rule-based heuristic
python evaluate.py data/policies/cuarenta_policy_iter1.pt --mode vs-heuristic

# Watch a game played against the heuristic
python evaluate.py data/policies/cuarenta_policy_iter1.pt --mode watch
```

## Play 2 humans team vs 2 agents team 

```bash
python src/imitation_learning/play_vs_agent.py --checkpoint data/policies/cuarenta_policy_iter1.pt --human-players 0,2
```

## Architecture

```
Observation (per player, partial info)
  ├── my_hand          (40,) binary
  ├── table            (40,) binary
  ├── played_this_deal (40,) binary
  ├── last_placed      (40,) binary
  └── scalars: scores, current_player, deck_remaining

ObsEncoder
  ├── Card matrix (B, 40, 4) → Linear → (B, 40, 256)
  ├── Positional embedding (40, 256)
  ├── Transformer (2 layers, 4 heads)
  ├── Mean pool → (B, 256)
  ├── Scalar MLP → (B, 64)
  └── Fusion → (B, 256)

PolicyHead (two-headed)
  ├── card_logits    (B, 40) — which card to play
  └── capture_logits (B, 40) — which table cards to take

ValueHead
  └── (B, 256) → MLP → scalar V(s)
```

Actions are masked to legal moves only. Joint (card, capture_set) score
= `card_logit[card_idx] + Σ capture_logit[i for i in capture_bits]`,
then softmax over legal actions.

## Self-Play

- Agent plays as **team 0** (seats 0 & 2), opponent plays as **team 1**
- Opponent is sampled from a checkpoint pool (default: 80% latest, 20% random older)
- New checkpoint saved to pool every 50k steps
- Pool size capped at 20 checkpoints

## Hyperparameter Guide

| Param | Default | Notes |
|---|---|---|
| `n_envs` | 8 | Increase for faster data collection (CPU bound) |
| `n_steps` | 512 | Steps per rollout per env |
| `clip_eps` | 0.2 | PPO clip range |
| `lr` | 3e-4 | Reduce to 1e-4 if training is unstable |
| `ent_coef` | 0.01 | Increase to encourage exploration early |
| `gamma` | 0.99 | Discount — keep high (long episodes) |
| `gae_lambda` | 0.95 | GAE lambda |
| `total_steps` | 5M | Start here; scale to 20M+ for strong play |

## Known Limitations / TODOs

- **Game bug**: `_new_round` doesn't check if scores already ≥ 40 after a round ends
  without a decisive capture. Fix: add a win check at top of `step()` after `_new_round()`.
- **No LSTM**: Adding a recurrent layer over the turn sequence would help with partial info.
- **No opponent modelling**: A dedicated opponent-modelling head could infer hidden hands.
- **Single-process**: For serious training, parallelise envs with multiprocessing or use RLlib.

## Generating ONNX model
```bash
python3 -m scripts.pth2onnx <path to .pt file>
```

# More info about Cuarenta
https://40caidaylimpia.com/reglas-para-jugar-40/
