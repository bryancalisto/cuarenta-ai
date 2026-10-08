"""
PPO package: self-play PPO for Cuarenta (no search).

Layout (mirrors `src/ismcts/`: one script directory, bare sibling imports, a run
script, wandb logging by default and checkpoints under a single directory):

    config.py        every hyper-parameter in one dataclass
    bootstrap.py     supervised warm start: imitate the heuristic agent
    rollout.py       rollout buffer, GAE, return normalisation, action selection,
                     the recurrent replay contract, reward shaping, episode stats
    selfplay.py      the opponent pool (past checkpoints) and its sampling rules
    eval_harness.py  deterministic, parallel evaluation matches (win rate + scores +
                     returns + caídas, with standard errors)
    trainer.py       PPOTrainer: collect -> update -> log -> evaluate -> checkpoint
    train.py         command-line entry point
    run_training.sh  bootstrap + self-play in one command
    test_ppo.py      the invariants the trainer depends on

Representation: PPO uses the SAME observation layout and action encoding as the ISMCTS
pipeline — a 208-dim observation (`utils.OBS_KEYS_ISMCTS`) plus 84-dim FEATURES for each
of up to 64 legal actions (`env.action_features_matrix`). The policy head scores actions
from their features, so it is position-invariant and much smaller than the old flat
1640-way head. (The legacy 244-dim layout is still declared for the DAgger policy and
the deployed ONNX model.)

Outputs: `checkpoints-ppo/` holds `bootstrap.pt`, periodic `policy_<step>.pt` snapshots,
`best.pt` (passed the reference-gated match), `best_heuristic.pt`, a matplotlib summary
and the raw metrics as .npz. Metrics go to wandb project `cuarenta-ai`, the same project
the ISMCTS runs use.

The game core and the shared pieces live here too, so the package is self-contained
(like `src/ismcts/`): `card.py`, `captures.py`, `env.py`, `utils.py`, `observation
layout`, `networks.py`, `agents.py`, `ui.py`. The DAgger imitation pipeline in
`src/imitation_learning/` imports them from this directory.

Run it:

    ./src/ppo/run_training.sh                 # bootstrap + 2M steps
    python3 src/ppo/train.py --help           # every knob
    python3 src/ppo/test_ppo.py               # tests
"""
