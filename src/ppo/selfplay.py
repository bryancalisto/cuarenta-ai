"""
Self-play opponent pool.

The learner's opponents are snapshots of its own past self. Keeping a POOL (rather
than only the newest snapshot) matters: a fixed opponent makes the policy overfit to
one specific style and the value function estimates outcomes against that one
opponent; a pool keeps the training distribution broad and keeps the critic honest.

Two knobs:

  * `pool_size`        — how many past snapshots to keep (oldest are dropped),
  * `opponent_latest`  — probability of facing the NEWEST snapshot. Sampling happens
                         per environment at every episode reset, so different games
                         in flight face different opponents, and the split between
                         "vs latest" and "vs older pool" is tracked in the logs
                         (`selfplay/vs_latest_frac`).

The pool is seeded with the weights the run STARTS FROM. That is `PPOTrainer`'s doing,
not this class's: the constructor runs before the warm-start checkpoint is loaded, so it
seeds the pool with the random init and `load_policy` then calls `reset_to` to replace it
(see the note there — skipping that reset makes every self-play game a game against
random weights). With the pool seeded correctly, `is_latest` is well defined from the
first game and the first evaluation has a `prev_checkpoint` to compare against: the
policy the run started from.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch

from networks import ActorCriticNetwork
from rollout import clone_state_dict_to_cpu


class OpponentPool:
    """Holds past checkpoint weights and assigns one to each environment."""

    def __init__(
        self,
        opponent_nets: List[ActorCriticNetwork],
        pool_size: int,
        latest_probability: float,
        use_lstm: bool,
        latest_states: List[bool],
    ):
        self.nets = opponent_nets                  # one net per env, reused
        self.pool_size = pool_size
        self.latest_probability = latest_probability
        self.use_lstm = use_lstm
        self.is_latest = latest_states             # per env: facing the newest snapshot?
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        self.steps: List[int] = []

    # ------------------------------------------------------------------ updates

    def add_snapshot(self, net: ActorCriticNetwork, step: int) -> None:
        """Register the current weights as a new pool entry (oldest dropped)."""
        self.snapshots.append(clone_state_dict_to_cpu(net.state_dict()))
        self.steps.append(int(step))
        if len(self.snapshots) > self.pool_size:
            self.snapshots.pop(0)
            self.steps.pop(0)

    def reset_to(self, net: ActorCriticNetwork, step: int) -> None:
        """
        Throw away every snapshot and re-seed the pool with `net`.

        Call this AFTER loading a checkpoint. The pool is constructed in `PPOTrainer.__init__`
        and seeded there, which is BEFORE the warm-start checkpoint is loaded -- so that
        seed is the randomly initialised network. `sample_for_env` loads `snapshots[-1]`
        into the opponent net at every episode reset, so without this reset the learner
        would spend the whole run playing against random weights instead of against itself.

        Training from scratch needs no reset: there the net really is the pool seed.
        """
        self.snapshots.clear()
        self.steps.clear()
        self.add_snapshot(net, step)

    def sync_all_to(self, net: ActorCriticNetwork) -> None:
        """Point every opponent net at `net` (used at startup and after loading)."""
        for opp in self.nets:
            opp.load_state_dict(net.state_dict())
            opp.eval()

    # ------------------------------------------------------------------ sampling

    def sample_for_env(self, env_idx: int, lstm_states: List = None) -> None:
        """
        Assign this env a (possibly new) opponent snapshot for its next episode.

        Called at every episode reset rather than at checkpoint time, so a game in
        flight keeps one opponent from start to finish while newly starting games can
        pick up fresher weights.
        """
        if not self.snapshots:
            # Still the initial (self) weights; nothing to sample yet.
            self.is_latest[env_idx] = True
            return

        if random.random() < self.latest_probability:
            snapshot = self.snapshots[-1]
            self.is_latest[env_idx] = True
        else:
            snapshot = random.choice(self.snapshots)
            self.is_latest[env_idx] = False

        self.nets[env_idx].load_state_dict(snapshot)
        self.nets[env_idx].eval()
        if self.use_lstm and lstm_states is not None:
            lstm_states[env_idx] = None

    # ------------------------------------------------------------------ info

    def summary(self) -> str:
        if not self.snapshots:
            return "pool: empty (playing the current weights)"
        return (f"pool: {len(self.snapshots)} snapshots "
                f"(steps {self.steps[0]}..{self.steps[-1]})")
