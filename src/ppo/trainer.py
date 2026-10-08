"""
PPO trainer: self-play rollout collection, PPO updates, logging and checkpointing.

THE LOOP
--------
    for each rollout window:
        1. collect `n_envs x n_steps` turns of self-play           (_collect_rollout)
        2. compute GAE, normalise, run `n_epochs` PPO minibatches  (_ppo_update)
        3. log metrics, occasionally evaluate and checkpoint

DESIGN DECISIONS THAT ARE EASY TO GET WRONG
-------------------------------------------
* ONE LEARNING SEAT PER ENV. Exactly one of the four seats generates gradient
  transitions per game; the other three are played by an opponent from the pool (or
  the heuristic/random bot). Which seat learns is re-randomised every episode, so the
  policy does not specialise to one seat. `EpisodeStats.learner_team` records it, and
  every logged reward is reported from the learner's point of view (`_for` / `_against`).

* REWARD ATTRIBUTION. The env rewards each MOVE from the acting team's point of view,
  but the learner only acts every fourth turn, so its rewards are accumulated
  (`pending_reward`) and flushed onto its most recent decision when its turn comes
  round again. An opponent's caída therefore shapes the learner decision that led to it.

* ZERO-SUM REWARDS. `cfg.symmetrize_rewards` (default ON) counts an opponent's reward
  as a NEGATIVE reward for the learner. The old behaviour dropped it entirely, which
  meant the learner's episode return was never negative and the critic literally could
  not learn "we are losing" — the ±50 terminal reward for the winning side was simply
  discarded when the winner was the opponent.

* RECURRENCE. With `use_lstm`, both the learner's and the opponent's hidden state are
  advanced on EVERY turn (not only on the learner's turns), because the observation
  already describes the whole position and the memory should cover the match. The
  hidden state at the start of each window is stored in the rollout and replayed
  during the update (see `rollout.replay_state_embeddings`).

* NORMALISED VALUE TARGETS. GAE and bootstrapping happen in real reward space; the
  value head regresses normalised returns tracked by `RunningMeanStd`.
"""

from __future__ import annotations

import collections
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from env import (ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv, TEAMS,
                 action_features_matrix)
from utils import (OBS_SIZE_ISMCTS, OpponentMode, PointsReasons, check_obs_layout,
                   flatten_obs, has_reason)
from agents import HeuristicAgent, RandomAgent
from networks import ActorCriticNetwork
from config import PPOConfig
from rollout import (
    EpisodeStats, MetricWindow, Rollout, RunningMeanStd, clone_lstm_state,
    clone_state_dict_to_cpu, compute_gae, lstm_state_to_device,
    replay_state_embeddings, select_legal_action, shaped_reward,
)
from selfplay import OpponentPool
from eval_harness import OpponentSpec, evaluate


class PPOTrainer:
    """Trains a Cuarenta policy with self-play PPO (no search)."""

    def __init__(self, cfg: PPOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[PPO] device={self.device}  lstm={cfg.use_lstm}  hidden={cfg.hidden}  "
              f"value={cfg.value_hidden}"
              + (f"  lstm_hidden={cfg.lstm_hidden}" if cfg.use_lstm else ""))

        # ---- environments ----
        self.envs = [CuarentaEnv(seed=i) for i in range(cfg.n_envs)]
        self.env_obs: List[Optional[np.ndarray]] = [None] * cfg.n_envs
        self.env_done: List[bool] = [True] * cfg.n_envs
        self.lstm_states: Optional[List] = [None] * cfg.n_envs if cfg.use_lstm else None
        self.opponent_lstm_states: Optional[List] = [None] * cfg.n_envs if cfg.use_lstm else None

        # ---- sizes (inferred from the env, and checked against the declared layout) ----
        sample_obs, _ = self.envs[0].reset()
        check_obs_layout(sample_obs)                 # ISMCTS layout (208)
        self.obs_size = int(flatten_obs(sample_obs).shape[0])
        self.n_slots = MAX_ACTIONS
        print(f"[PPO] obs layout: ISMCTS ({self.obs_size} dims)  "
              f"action slots: {self.n_slots} x {ACTION_FEAT_DIM} features")

        # ---- learner + optimiser ----
        self.net = ActorCriticNetwork(
            obs_dim=self.obs_size, action_feat_dim=ACTION_FEAT_DIM, n_slots=self.n_slots,
            hidden=cfg.hidden, use_lstm=cfg.use_lstm, lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers, value_hidden=cfg.value_hidden,
        ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.ret_rms = RunningMeanStd()

        # ---- opponents: one net per env, all sharing the learner's architecture ----
        opponent_nets = [
            ActorCriticNetwork(**self.net.architecture_config()).to(self.device)
            for _ in range(cfg.n_envs)
        ]
        self.env_opponent_is_latest = [True] * cfg.n_envs
        self.pool = OpponentPool(
            opponent_nets=opponent_nets,
            pool_size=cfg.pool_size,
            latest_probability=cfg.opponent_latest,
            use_lstm=cfg.use_lstm,
            latest_states=self.env_opponent_is_latest,
        )
        self.pool.sync_all_to(self.net)
        # ---- evaluation league ----
        # Three references, like the ISMCTS pipeline keeps (best so far, the fixed
        # external bot, and the starting policy):
        #   reference : the best-so-far weights, replaced when the current policy
        #               beats them by cfg.accept_threshold (AlphaZero-style gating)
        #   initial   : the policy this run started from (frozen for the whole run)
        #   previous  : the most recently saved snapshot
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reference_state = clone_state_dict_to_cpu(self.net.state_dict())
        self.initial_state = clone_state_dict_to_cpu(self.net.state_dict())
        self.best_heuristic_winrate = -1.0
        # Acceptance bar for the reference match. Calibrated at startup against a MIRROR
        # match (the policy against a copy of itself), because ">55%" only means progress
        # if a mirror is 50%; with finite games it is not exactly 50%.
        self.accept_bar = float(cfg.accept_threshold)
        self.best_path = self.save_dir / "best.pt"                 # beats the reference
        self.best_heur_path = self.save_dir / "best_heuristic.pt"  # best vs the bot

        # Seed the pool so it is never empty: `sample_for_env` would otherwise leave the
        # opponent nets on whatever weights they happen to hold, and the first evaluation
        # would lose its `prev_checkpoint` match (the league skips an empty pool).
        #
        # NOTE: `__init__` runs BEFORE a checkpoint is loaded, so this seed is the freshly
        # built RANDOM network, not the warm start. `load_policy` replaces it with
        # `pool.reset_to(...)` once the real weights are in. Without that reset the learner
        # plays the entire run against random weights -- which is exactly what happened
        # before, and it is invisible because a random opponent is still a legal opponent.
        self.pool.add_snapshot(self.net, 0)

        # ---- bookkeeping ----
        self.global_step = 0
        self.rollout_num = 0
        self.heuristic_agent = HeuristicAgent(seed=0)
        self.episode_metrics = MetricWindow(100, prefix="selfplay")
        self.training_history: Dict[str, List[float]] = collections.defaultdict(list)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def wandb_off_eval_disabled(self) -> bool:
        """The baseline evaluation is skipped only for tiny/CI-style runs."""
        return bool(getattr(self.cfg, "skip_baseline_eval", False))

    def _setup_wandb(self) -> None:
        """
        Start a W&B run (default on, like the ISMCTS pipeline; project `cuarenta-ai`).

        Never fatal: if there is no API key / no network, the run continues and only
        prints that logging is disabled, so a long training run cannot be lost to a
        logging problem.
        """
        self.wandb = None
        if not self.cfg.wandb:
            return
        try:
            import dotenv
            dotenv.load_dotenv()          # pick up WANDB_API_KEY from .env if present
            import wandb
            wandb.init(project=self.cfg.wandb_project,
                       name=f"{self.cfg.run_name}-{'lstm' if self.cfg.use_lstm else 'ff'}",
                       group=self.cfg.run_name,
                       config=asdict(self.cfg))
            self.wandb = wandb
        except Exception as exc:          # noqa: BLE001 - logging must not be fatal
            print(f"[PPO] wandb unavailable, continuing without logging: {exc}")
        # ONE x-axis name across every pipeline. PPO, ISMCTS and ReBeL log `eval/*` with the
        # same member and suffix names, and for those curves to overlay in the wandb console
        # they must share the step metric's NAME too -- `iteration`. PPO's own progress
        # measure is the environment step, which is still logged as a metric; it is simply
        # no longer the axis.
        wandb.define_metric("iteration")
        for group in ("train", "selfplay", "eval", "time"):
            wandb.define_metric(f"{group}/*", step_metric="iteration")

    def _log(self, metrics: Dict[str, float]) -> None:
        if self.wandb is not None:
            # `iteration` = rollouts completed, the same quantity ISMCTS and ReBeL log under
            # that name and the axis every eval/* family is defined against.
            self.wandb.log(dict(metrics, iteration=self.rollout_num,
                                step=self.global_step))

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def load_policy(self, path: str) -> None:
        """
        Load a checkpoint. Two flavours are supported:

          * a PPO snapshot (`model_state`)  -> full resume: weights, optimiser,
            return statistics and the step counter,
          * a DAgger imitation policy (`policy_state`) -> warm start the trunk only
            (the value head keeps its initial weights), and the run starts at step 0.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state" in state:
            self.net.load_state_dict(state["model_state"])
            if "optimizer_state" in state:
                self.opt.load_state_dict(state["optimizer_state"])
            if "ret_rms" in state:
                self.ret_rms.load_state_dict(state["ret_rms"])
            self.global_step = int(state.get("step", 0))
            print(f"[PPO] resumed from {path} (step={self.global_step})")
        elif "policy_state" in state:
            self.net.policy.load_state_dict(state["policy_state"])
            print(f"[PPO] warm start from imitation policy {path} "
                  f"(trunk only; value head keeps its initial weights)")
        else:
            raise KeyError(f"Unsupported checkpoint format: {path}")

        self.pool.sync_all_to(self.net)
        # Re-seed the pool with what we just loaded. `__init__` seeded it with the freshly
        # built (random) network, and `sample_for_env` serves `snapshots[-1]` to the
        # opponent net at every episode reset -- so without this, the learner would play
        # the entire run against random weights instead of against its own past self.
        self.pool.reset_to(self.net, self.global_step)
        # The league starts from whatever we just loaded (the "initial" reference is
        # the policy this run began with, frozen for the whole run).
        self.reference_state = clone_state_dict_to_cpu(self.net.state_dict())
        self.initial_state = clone_state_dict_to_cpu(self.net.state_dict())
        if self.cfg.use_lstm:
            self.lstm_states = [None] * self.cfg.n_envs
            self.opponent_lstm_states = [None] * self.cfg.n_envs

    def _save_checkpoint(self, path: Path) -> Path:
        """Write weights (plus optimiser/config metadata) to `path`."""
        torch.save({
            "step": self.global_step,
            "model_state": clone_state_dict_to_cpu(self.net.state_dict()),
            "optimizer_state": self.opt.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
            "config": asdict(self.cfg),
            **self.net.architecture_config(),
        }, path)
        return path

    def _save_policy_snapshot(self) -> Path:
        """Periodic numbered snapshot, and a new entry in the opponent pool."""
        path = self._save_checkpoint(self.save_dir / f"policy_{self.global_step}.pt")
        print(f"[PPO] checkpoint → {path}")
        self.pool.add_snapshot(self.net, self.global_step)
        return path

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> List[Rollout]:
        """
        Collect one window of experience for a single learning seat per env.

        See the module docstring for reward attribution and the recurrence contract.
        """
        cfg = self.cfg
        rollouts = [Rollout() for _ in range(cfg.n_envs)]
        ep_reward = [0.0] * cfg.n_envs          # learner's shaped return this episode
        ep_len = [0] * cfg.n_envs
        learning_player = [random.randrange(4) for _ in range(cfg.n_envs)]
        pending_reward = [0.0] * cfg.n_envs     # awaiting attribution to the last move

        # Per-episode reward bookkeeping for logging (raw, per team).
        returns: List[List[float]] = [[0.0, 0.0] for _ in range(cfg.n_envs)]
        caidas: List[List[int]] = [[0, 0] for _ in range(cfg.n_envs)]
        carton: List[List[int]] = [[0, 0] for _ in range(cfg.n_envs)]
        limpias: List[List[int]] = [[0, 0] for _ in range(cfg.n_envs)]

        # Snapshot the hidden state each env starts this window with, so the update
        # can replay from the exact same point.
        if cfg.use_lstm:
            for env_idx in range(cfg.n_envs):
                rollouts[env_idx].hidden_start = (
                    None if self.env_done[env_idx] else clone_lstm_state(self.lstm_states[env_idx])
                )

        self.net.eval()
        with torch.no_grad():
            for _ in range(cfg.n_steps):
                for env_idx, env in enumerate(self.envs):
                    if self.env_done[env_idx]:
                        obs, _ = env.reset()
                        self.env_obs[env_idx] = flatten_obs(obs)
                        self.env_done[env_idx] = False
                        ep_reward[env_idx] = 0.0
                        ep_len[env_idx] = 0
                        pending_reward[env_idx] = 0.0
                        returns[env_idx] = [0.0, 0.0]
                        caidas[env_idx] = [0, 0]
                        carton[env_idx] = [0, 0]
                        limpias[env_idx] = [0, 0]
                        learning_player[env_idx] = random.randrange(4)
                        if cfg.use_lstm:
                            self.lstm_states[env_idx] = None
                            self.opponent_lstm_states[env_idx] = None
                        if cfg.opponent_mode == OpponentMode.SELFPLAY:
                            self.pool.sample_for_env(env_idx, self.opponent_lstm_states)

                    learner_seat = learning_player[env_idx]
                    learner_team = TEAMS[learner_seat]
                    current_player = env.state.current_player
                    current_team = TEAMS[current_player]
                    is_learning = current_player == learner_seat
                    obs_flat = self.env_obs[env_idx]
                    legal_actions = env.legal_actions()
                    # Candidate actions as features + legality mask (ISMCTS-style).
                    feat_np, mask_np = action_features_matrix(legal_actions, env.state)
                    rl = rollouts[env_idx]

                    obs_t = torch.tensor(obs_flat, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                    feats_t = torch.as_tensor(feat_np, device=self.device).unsqueeze(0)
                    mask_t = torch.as_tensor(mask_np, device=self.device).unsqueeze(0)

                    need_policy = is_learning or cfg.use_lstm
                    if need_policy:
                        # With an LSTM the state is advanced on EVERY turn, not just the
                        # learner's own, so the memory covers the whole match.
                        state_in = self.lstm_states[env_idx] if cfg.use_lstm else None
                        logits, value, new_state = self.net(obs_t, feats_t, mask_t, state_in)
                        if cfg.use_lstm:
                            self.lstm_states[env_idx] = new_state
                    else:
                        logits, value = None, None

                    # The SAME contract applies to the opponent net, which `_opponent_action`
                    # only ever gets to advance on its own turns. Advance it here, on the
                    # learner's turn, so both memories span the whole match; otherwise the
                    # opponent plays from a history its weights never saw and self-play
                    # measures the learner against a handicapped copy of itself.
                    if is_learning and cfg.use_lstm and cfg.opponent_mode == OpponentMode.SELFPLAY:
                        self._advance_opponent_state(env_idx, feats_t, mask_t, obs_t)

                    rl.full_obs.append(obs_flat.copy())
                    rl.full_dones.append(False)   # patched to True below if the episode ends

                    if is_learning:
                        # Flush reward accrued since the previous learning move.
                        if rl.rewards:
                            rl.rewards[-1] += pending_reward[env_idx]
                        pending_reward[env_idx] = 0.0

                        slot, log_prob, _ = select_legal_action(logits[0], mask_t[0])
                        action = legal_actions[slot]

                        rl.learn_positions.append(len(rl.full_obs) - 1)
                        rl.actions.append(slot)
                        rl.log_probs.append(log_prob.item())
                        rl.rewards.append(0.0)   # filled by accumulation
                        rl.values.append(value.item())
                        rl.dones.append(False)
                        rl.action_feats.append(feat_np)
                        rl.action_masks.append(mask_np)
                    else:
                        action = self._opponent_action(env_idx, env, current_player,
                                                       legal_actions, feats_t, mask_t, obs_t)

                    next_obs, reward, done, info = env.step(action)

                    # ---- reward shaping -------------------------------------------------
                    # `reward` is from the acting team's point of view. From the
                    # learner's team point of view it is +reward for its own players
                    # and -reward for the opponents (zero-sum, see cfg.symmetrize_rewards).
                    shaped = shaped_reward(reward, current_team, learner_team,
                                           cfg.symmetrize_rewards)
                    pending_reward[env_idx] += shaped
                    ep_reward[env_idx] += shaped
                    ep_len[env_idx] += 1

                    # Raw per-team rewards, for the logging curves.
                    returns[env_idx][current_team] += float(reward)
                    reasons = info.get("reasons") or []
                    if has_reason(reasons, PointsReasons.CAIDA):
                        caidas[env_idx][current_team] += 1
                    elif has_reason(reasons, PointsReasons.LIMPIA):
                        limpias[env_idx][current_team] += 1
                    carton[env_idx][current_team] += int(info.get("carton_points") or 0)

                    self.env_obs[env_idx] = flatten_obs(next_obs)

                    if done:
                        rl.full_dones[-1] = True
                        if rl.rewards:
                            # Attribute the terminal outcome to the last learning move.
                            rl.rewards[-1] += pending_reward[env_idx]
                            rl.dones[-1] = True
                        pending_reward[env_idx] = 0.0
                        self.env_done[env_idx] = True

                        self.episode_metrics.add(EpisodeStats(
                            winner=int(info.get("winner", -1)),
                            scores=(int(env.state.scores[0]), int(env.state.scores[1])),
                            plies=ep_len[env_idx],
                            returns=(returns[env_idx][0], returns[env_idx][1]),
                            caidas=(caidas[env_idx][0], caidas[env_idx][1]),
                            limpias=(limpias[env_idx][0], limpias[env_idx][1]),
                            carton=(carton[env_idx][0], carton[env_idx][1]),
                            learner_team=learner_team,
                            shaped_return=ep_reward[env_idx],
                            opponent_is_latest=self.env_opponent_is_latest[env_idx],
                        ))

        self.net.train()
        return rollouts

    def _opponent_action(self, env_idx: int, env, current_player: int,
                         legal_actions, feats_t, mask_t, obs_t):
        """Pick the opponent's action for one turn (heuristic / random / pooled net)."""
        mode = self.cfg.opponent_mode
        if mode == OpponentMode.HEURISTIC:
            return self.heuristic_agent.act(env.observe(current_player), legal_actions,
                                            state=env.state, team=TEAMS[current_player])
        if mode == OpponentMode.RANDOM:
            return random.choice(legal_actions)

        opp_net = self.pool.nets[env_idx]
        opp_state = self.opponent_lstm_states[env_idx] if self.cfg.use_lstm else None
        opp_logits, _, new_opp_state = opp_net(obs_t, feats_t, mask_t, opp_state)
        if self.cfg.use_lstm:
            self.opponent_lstm_states[env_idx] = new_opp_state
        slot, _, _ = select_legal_action(opp_logits[0], mask_t[0])
        return legal_actions[slot]

    def _advance_opponent_state(self, env_idx: int, feats_t, mask_t, obs_t) -> None:
        """
        Advance the opponent net's memory on a turn it does NOT act on.

        The opponent is a snapshot of a policy that was itself trained with its memory
        advanced on EVERY turn (see `_collect_rollout`), so feeding it a state that
        skipped the learner's turns hands it a hidden state it was never trained on.
        That is not a cosmetic difference: the replayed history is what the LSTM
        conditions on, so a desynchronised opponent plays measurably worse than its
        weights imply, and self-play would be scoring the learner against a weakened
        copy of itself rather than against itself. Its logits are discarded here.
        """
        opp_net = self.pool.nets[env_idx]
        _, _, new_opp_state = opp_net(obs_t, feats_t, mask_t,
                                      self.opponent_lstm_states[env_idx])
        self.opponent_lstm_states[env_idx] = new_opp_state

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, rollouts: List[Rollout]) -> Dict[str, float]:
        """Compute GAE, normalise advantages/returns, dispatch to the right update."""
        cfg = self.cfg
        env_records: List[Dict] = []
        all_advs_flat: List[float] = []
        all_rets_flat: List[float] = []

        # Stored values are in NORMALISED space; unnormalise to real reward space
        # before GAE/bootstrapping.
        ret_mean, ret_std = self.ret_rms.mean, self.ret_rms.std

        for env_idx, rl in enumerate(rollouts):
            if not rl.rewards:
                continue

            if self.env_done[env_idx]:
                last_val = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(self.env_obs[env_idx], dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                    lstm_state = self.lstm_states[env_idx] if cfg.use_lstm else None
                    last_val = float(self.net.get_value(obs_t, lstm_state).item()) * ret_std + ret_mean

            rewards = np.array(rl.rewards, dtype=np.float32)
            values = np.array(rl.values, dtype=np.float32) * ret_std + ret_mean
            dones = np.array(rl.dones, dtype=np.float32)
            advs, rets = compute_gae(rewards, values, dones, last_val, cfg.gamma, cfg.gae_lambda)

            if cfg.use_lstm:
                record = {
                    "obs": torch.tensor(np.stack(rl.full_obs), dtype=torch.float32, device=self.device),
                    "seq_dones": torch.tensor(rl.full_dones, dtype=torch.float32, device=self.device),
                    "learn_positions": torch.tensor(rl.learn_positions, dtype=torch.long, device=self.device),
                    "hidden_start": lstm_state_to_device(rl.hidden_start, self.device),
                }
            else:
                record = {
                    "obs": torch.tensor(
                        np.stack([rl.full_obs[i] for i in rl.learn_positions]),
                        dtype=torch.float32, device=self.device,
                    ),
                }

            record.update({
                "actions": torch.tensor(rl.actions, dtype=torch.long, device=self.device),
                "old_log_probs": torch.tensor(rl.log_probs, dtype=torch.float32, device=self.device),
                "advs": advs,
                "rets": rets,
                # Candidate actions of each learning decision: [n_learn, MAX_ACTIONS, 84]
                # features and the matching legality mask.
                "feats": torch.tensor(np.stack(rl.action_feats), dtype=torch.float32,
                                      device=self.device),
                "masks": torch.tensor(np.stack(rl.action_masks), dtype=torch.bool,
                                      device=self.device),
            })
            env_records.append(record)
            all_advs_flat.extend(advs.tolist())
            all_rets_flat.extend(rets.tolist())

        if not env_records:
            return {"loss_pi": 0.0, "loss_vf": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0}

        adv_mean = float(np.mean(all_advs_flat))
        adv_std = float(np.std(all_advs_flat) + 1e-8)

        # Fold this rollout's returns into the running stats, then normalise the value
        # targets with the refreshed stats so the MSE regresses in unit scale.
        self.ret_rms.update(np.array(all_rets_flat, dtype=np.float32))
        ret_mean, ret_std = self.ret_rms.mean, self.ret_rms.std
        for rec in env_records:
            rec["advs"] = torch.tensor((rec["advs"] - adv_mean) / adv_std,
                                       dtype=torch.float32, device=self.device)
            rec["rets"] = torch.tensor((rec["rets"] - ret_mean) / ret_std,
                                       dtype=torch.float32, device=self.device)

        if not cfg.use_lstm:
            return self._ppo_update_feedforward(env_records)
        return self._ppo_update_recurrent(env_records)

    @staticmethod
    def _ppo_losses(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                    advs: torch.Tensor, values: torch.Tensor, rets: torch.Tensor,
                    entropies: torch.Tensor, cfg: PPOConfig):
        """
        The clipped PPO objective, plus two diagnostics worth watching:

          * `approx_kl`  = mean(old_logp - new_logp): how far this update moves the
            policy. A spike means the minibatch was too large a step; if it keeps
            climbing, lower the learning rate or the number of epochs.
          * `clip_frac`  = fraction of samples whose probability ratio is outside
            [1-eps, 1+eps] and therefore gets no gradient. When most of the batch is
            clipped, the update has stopped learning from that data.
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advs
        loss_pi = -torch.min(surr1, surr2).mean()
        loss_vf = F.mse_loss(values, rets)
        entropy = entropies.mean()
        loss = loss_pi + cfg.vf_coef * loss_vf - cfg.ent_coef * entropy

        with torch.no_grad():
            approx_kl = float((old_log_probs - new_log_probs).mean().item())
            clip_frac = float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())
        return loss, loss_pi, loss_vf, entropy, approx_kl, clip_frac

    def _ppo_update_recurrent(self, env_records: List[Dict]) -> Dict[str, float]:
        """
        PPO update that preserves per-environment temporal order. A minibatch is a
        group of ENVIRONMENTS (not timesteps), because each environment's sequence has
        to be replayed through the LSTM in order from its own stored hidden state.
        """
        cfg = self.cfg
        stats = collections.defaultdict(list)
        envs_per_mb = max(1, min(cfg.envs_per_minibatch, len(env_records)))

        for _ in range(cfg.n_epochs):
            order = list(range(len(env_records)))
            random.shuffle(order)

            for start in range(0, len(order), envs_per_mb):
                mb_idxs = order[start: start + envs_per_mb]

                pi_losses, vf_losses, entropies, kls, clips = [], [], [], [], []
                for idx in mb_idxs:
                    rec = env_records[idx]
                    # 1) Replay the whole match through the trunk (+LSTM) to reproduce
                    #    the state embeddings the actions were sampled from …
                    emb_seq = replay_state_embeddings(
                        self.net, rec["obs"], rec["seq_dones"], rec["hidden_start"]
                    )
                    lp = rec["learn_positions"]
                    s_emb = emb_seq[lp]                       # [n_learn, F]
                    values_learn = self.net.value_head(s_emb).squeeze(-1)

                    # 2) … then score each decision's candidate actions from their
                    #    stored features and build the masked distributions.
                    a_emb = self.net.embed_actions(rec["feats"])        # [n_learn, S, F]
                    logits_learn = self.net.logits_from(s_emb, a_emb)   # [n_learn, S]
                    logits_learn = logits_learn.masked_fill(~rec["masks"], -1e9)

                    dist = torch.distributions.Categorical(logits=logits_learn)
                    log_probs_t = dist.log_prob(rec["actions"])
                    entropy_t = dist.entropy()

                    loss, loss_pi, loss_vf, entropy, kl, clip = self._ppo_losses(
                        log_probs_t, rec["old_log_probs"], rec["advs"],
                        values_learn, rec["rets"], entropy_t, cfg,
                    )
                    pi_losses.append(loss_pi)
                    vf_losses.append(loss_vf)
                    entropies.append(entropy)
                    kls.append(kl)
                    clips.append(clip)
                    loss_total = loss

                loss = (torch.stack(pi_losses).mean()
                        + cfg.vf_coef * torch.stack(vf_losses).mean()
                        - cfg.ent_coef * torch.stack(entropies).mean())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"].append(float(torch.stack(pi_losses).mean().item()))
                stats["loss_vf"].append(float(torch.stack(vf_losses).mean().item()))
                stats["entropy"].append(float(torch.stack(entropies).mean().item()))
                stats["approx_kl"].append(float(np.mean(kls)))
                stats["clip_frac"].append(float(np.mean(clips)))

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _ppo_update_feedforward(self, env_records: List[Dict]) -> Dict[str, float]:
        """Flat/shuffled PPO update — valid when there is no recurrence to preserve."""
        cfg = self.cfg
        obs_tensor = torch.cat([r["obs"] for r in env_records], dim=0)
        actions_tensor = torch.cat([r["actions"] for r in env_records], dim=0)
        old_lps_tensor = torch.cat([r["old_log_probs"] for r in env_records], dim=0)
        advs_tensor = torch.cat([r["advs"] for r in env_records], dim=0)
        rets_tensor = torch.cat([r["rets"] for r in env_records], dim=0)
        feats_tensor = torch.cat([r["feats"] for r in env_records], dim=0)
        masks_tensor = torch.cat([r["masks"] for r in env_records], dim=0)

        stats = collections.defaultdict(list)
        N = obs_tensor.shape[0]

        for _ in range(cfg.n_epochs):
            idxs = torch.randperm(N, device=self.device)
            for start in range(0, N, cfg.minibatch_size):
                mb_idx = idxs[start: start + cfg.minibatch_size]
                logits, values, _ = self.net(
                    obs_tensor[mb_idx], feats_tensor[mb_idx], masks_tensor[mb_idx]
                )
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_t = dist.log_prob(actions_tensor[mb_idx])
                entropy_t = dist.entropy()

                loss, loss_pi, loss_vf, _entropy, kl, clip = self._ppo_losses(
                    log_probs_t, old_lps_tensor[mb_idx], advs_tensor[mb_idx],
                    values, rets_tensor[mb_idx], entropy_t, cfg,
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"].append(loss_pi.item())
                stats["loss_vf"].append(loss_vf.item())
                stats["entropy"].append(entropy_t.mean().item())
                stats["approx_kl"].append(kl)
                stats["clip_frac"].append(clip)

        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_games: Optional[int] = None, seed_base: int = 500_000,
                 include_random: bool = True) -> Dict[str, float]:
        """
        Run the evaluation league for the current weights.

        Matches, all with fixed seeds and rotated seats:

          * `reference`  — the best-so-far policy. This is the gating match: the run
                           accepts a new best only by beating it (cfg.accept_threshold).
          * `heuristic`  — the fixed rule-based bot, i.e. the project's external
                           yardstick. Watch its `_score_*`/`_caidas_*` splits, not just
                           the win rate: a 60-game rate has a ±6% standard error.
          * `initial`    — the policy this run started from (the bootstrap), frozen for
                           the whole run: "did training actually improve on what we
                           began with?".
          * `prev_checkpoint` — the most recently saved snapshot.
          * `random`     — a sanity floor.
        """
        n_games = n_games or self.cfg.eval_games
        opponents = {
            "reference": OpponentSpec(kind="checkpoint", state_dict=self.reference_state),
            "heuristic": OpponentSpec(kind="heuristic"),
        }
        if self.initial_state is not None:
            opponents["initial"] = OpponentSpec(kind="checkpoint",
                                                state_dict=self.initial_state)
        if self.pool.snapshots:
            opponents["prev_checkpoint"] = OpponentSpec(kind="checkpoint",
                                                        state_dict=self.pool.snapshots[-1])
        if include_random:
            opponents["random"] = OpponentSpec(kind="random")

        return evaluate(self.net, opponents, n_games=n_games, seed_base=seed_base)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        self._setup_wandb()

        remaining = cfg.total_steps - self.global_step
        print(f"[PPO] training {remaining} more steps (total_steps={cfg.total_steps})")

        # ---- baseline: the same league, before any self-play ----
        # At this point the `reference` match is the policy against a copy of ITSELF,
        # so it doubles as the mirror calibration: whatever win rate it shows is the
        # noise/seat asymmetry of the setup, not skill. Acceptance must clear that bar
        # (plus a margin), or a lucky evaluation would be mistaken for progress.
        if not self.wandb_off_eval_disabled():
            print("\n=== Baseline evaluation (before any self-play) ===")
            baseline = self.evaluate()
            baseline.pop("_results", None)
            self.accept_bar = max(float(cfg.accept_threshold),
                                  float(baseline["eval/reference"]) + 0.05)
            self.best_heuristic_winrate = float(baseline["eval/heuristic"])
            print(f"[PPO] mirror baseline (policy vs itself) = "
                  f"{baseline['eval/reference']:.1%}; a new policy must reach "
                  f"{self.accept_bar:.1%} vs the reference to be accepted")
            self._log(dict(baseline, **{"eval/is_baseline": 1.0}))
            for key, value in baseline.items():
                if isinstance(value, (int, float)):
                    self.training_history.setdefault(f"eval_{key.split('/')[-1]}", []).append(float(value))
            self.training_history.setdefault("eval_step", []).append(float(self.global_step))

        pbar = tqdm(total=remaining, desc="Training", unit="step")
        next_eval_step = self.global_step + cfg.eval_every
        next_ckpt_step = self.global_step + cfg.checkpoint_every

        while self.global_step < cfg.total_steps:
            t_rollout = time.time()
            rollouts = self._collect_rollout()
            t_update = time.time()
            stats = self._ppo_update(rollouts)
            t_done = time.time()

            self.global_step += cfg.n_envs * cfg.n_steps
            self.rollout_num += 1
            rollout_elapsed = t_done - t_rollout

            metrics = self._rollout_metrics(stats, rollout_elapsed,
                                            t_update - t_rollout, t_done - t_update)
            self._record_history(metrics)
            self._log(metrics)

            if self.rollout_num % cfg.log_every == 0:
                sp = self.episode_metrics.metrics()
                pbar.set_postfix_str(
                    f"pi={stats['loss_pi']:.3f} vf={stats['loss_vf']:.3f} "
                    f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                    f"rew={sp.get('selfplay/shaped_return', float('nan')):.1f} "
                    f"wr={sp.get('selfplay/win_rate', float('nan')):.2f}"
                )
            pbar.update(cfg.n_envs * cfg.n_steps)

            if self.global_step >= next_eval_step:
                next_eval_step += cfg.eval_every
                self._evaluate_and_log()
            # The two schedules are independent (`next_ckpt_step` is advanced only here).
            # An earlier version also pushed `next_ckpt_step` forward from the eval branch
            # above; with eval_every == checkpoint_every that made this condition
            # unreachable, so NO periodic snapshot was ever written and the opponent pool
            # never gained a real opponent.
            if self.global_step >= next_ckpt_step:
                next_ckpt_step += cfg.checkpoint_every
                self._save_policy_snapshot()

        pbar.close()
        final_path = self._save_policy_snapshot()
        if cfg.save_plots:
            plot_path = self._save_training_plots()
            print(f"[PPO] training plot → {plot_path}")
        print(f"\n[PPO] Training complete after {self.global_step} steps. "
              f"Final checkpoint → {final_path}")
        if self.wandb is not None:
            self.wandb.finish()

    def _rollout_metrics(self, stats: Dict[str, float], elapsed: float,
                         collect_s: float, update_s: float) -> Dict[str, float]:
        """Assemble the per-rollout metric dict (train / rollout / selfplay / time)."""
        out = {
            "train/loss_pi": stats.get("loss_pi", 0.0),
            "train/loss_vf": stats.get("loss_vf", 0.0),
            "train/entropy": stats.get("entropy", 0.0),
            "train/approx_kl": stats.get("approx_kl", 0.0),
            "train/clip_frac": stats.get("clip_frac", 0.0),
            "train/ret_mean": self.ret_rms.mean,
            "train/ret_std": self.ret_rms.std,
            "time/rollout_s": elapsed,
            "time/collect_s": collect_s,
            "time/update_s": update_s,
            "time/sps": (self.cfg.n_envs * self.cfg.n_steps) / max(elapsed, 1e-8),
        }
        # Episode-level numbers over the last 100 finished games.
        out.update(self.episode_metrics.metrics())
        return out

    def _record_history(self, metrics: Dict[str, float]) -> None:
        """Keep the same numbers in memory so the matplotlib summary can be drawn."""
        for key, value in metrics.items():
            short = key.split("/", 1)[-1]
            self.training_history[short].append(float(value))
        self.training_history["step"].append(float(self.global_step))

    def _evaluate_and_log(self) -> None:
        """
        Evaluate, log the league, and gate the checkpoints.

        Gating (like the ISMCTS pipeline): the current weights become the new
        `reference` only if they beat the old reference by `cfg.accept_threshold`.
        The seats rotate across games, so no seat-bias correction is needed — a mirror
        match is 50% by construction.
        """
        metrics = self.evaluate()
        results = metrics.pop("_results", None)

        names = [k.split("/")[1] for k in metrics if k.startswith("eval/")
                 and not k.endswith(("_stderr", "_n_games"))
                 and "_score_" not in k and "_return_" not in k
                 and "_caidas_" not in k and "_limpias_" not in k and "_plies" not in k]
        print("[PPO] eval @ " + str(self.global_step) + ": "
              + "  ".join(f"{n}={metrics[f'eval/{n}']:.1%}" for n in names))
        for n in names:
            print(f"        {n:>15}: score {metrics[f'eval/{n}_score_for']:.1f}"
                  f"-{metrics[f'eval/{n}_score_against']:.1f}  "
                  f"returns {metrics[f'eval/{n}_return_for']:.1f}"
                  f"-{metrics[f'eval/{n}_return_against']:.1f}  "
                  f"caídas {metrics[f'eval/{n}_caidas_for']:.2f}"
                  f"-{metrics[f'eval/{n}_caidas_against']:.2f}  "
                  f"plies {metrics[f'eval/{n}_plies']:.0f}")

        if metrics["eval/reference"] >= self.accept_bar:
            self.reference_state = clone_state_dict_to_cpu(self.net.state_dict())
            self._save_checkpoint(self.best_path)
            print(f"        ACCEPTED: beats the reference "
                  f"{metrics['eval/reference']:.1%} "
                  f"(bar {self.accept_bar:.1%}) → {self.best_path}")
        else:
            print(f"        not accepted: reference={metrics['eval/reference']:.1%} "
                  f"< bar {self.accept_bar:.1%}")

        if metrics["eval/heuristic"] > self.best_heuristic_winrate:
            self.best_heuristic_winrate = metrics["eval/heuristic"]
            self._save_checkpoint(self.best_heur_path)
            print(f"        new best vs heuristic ({self.best_heuristic_winrate:.1%}) "
                  f"→ {self.best_heur_path}")

        # The value-probe style health check: the reference match says "are we
        # improving", the score splits say "how", and low entropy/high KL say "the
        # update is getting too big".
        if metrics.get("eval/heuristic_stderr", 0) > 0:
            print(f"        (±{metrics['eval/heuristic_stderr']:.1%} se on the "
                  f"heuristic match: treat single-evaluation swings as noise)")

        self._log(metrics)
        self.training_history.setdefault("eval_step", []).append(float(self.global_step))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.training_history.setdefault(f"eval_{key.split('/')[-1]}", []).append(float(value))

    # ------------------------------------------------------------------
    # Plots (optional, matplotlib)
    # ------------------------------------------------------------------

    def _save_training_plots(self) -> Path:
        """Write a matplotlib summary of the recorded metrics next to the checkpoints."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = self.training_history
        if not hist.get("step"):
            return self.save_dir / "training_metrics_empty.png"

        steps = np.array(hist["step"], dtype=np.float32)
        fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)

        panels = [
            (0, 0, "loss_pi", "Policy loss"),
            (0, 1, "loss_vf", "Value loss"),
            (1, 0, "entropy", "Policy entropy"),
            (1, 1, "approx_kl", "Approx KL"),
            (2, 0, "shaped_return", "Episode return (learner, shaped)"),
            (2, 1, "score_diff", "Mean score difference (self-play)"),
        ]
        for row, col, key, title in panels:
            ax = axes[row, col]
            values = np.array(hist.get(key, []), dtype=np.float32)
            if len(values):
                ax.plot(steps[:len(values)], values, linewidth=1.8, label=title)
                if len(values) >= 10:
                    w = 10
                    smooth = np.convolve(values, np.ones(w) / w, mode="valid")
                    ax.plot(steps[w - 1: w - 1 + len(smooth)], smooth, linewidth=2.0,
                            label=f"{w}-rollout mean")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        # Win rate panels: self-play rolling window and the held-out evaluations.
        ax = axes[3, 0]
        values = np.array(hist.get("win_rate", []), dtype=np.float32)
        if len(values):
            ax.plot(steps[:len(values)], values, label="self-play win rate (last 100)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Win rate")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        ax = axes[3, 1]
        eval_steps = np.array(hist.get("eval_step", []), dtype=np.float32)
        for key, label in (("eval_reference", "vs reference (gating)"),
                           ("eval_heuristic", "vs heuristic"),
                           ("eval_initial", "vs starting policy"),
                           ("eval_prev_checkpoint", "vs previous checkpoint"),
                           ("eval_random", "vs random")):
            values = np.array(hist.get(key, []), dtype=np.float32)
            if len(values) and len(eval_steps):
                n = min(len(values), len(eval_steps))
                ax.plot(eval_steps[:n], values[:n], marker="o", linewidth=1.8, label=label)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Evaluation win rates")
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"Cuarenta PPO training (steps={int(self.global_step)})", fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        plot_path = self.save_dir / f"training_metrics_{int(self.global_step):08d}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        np.savez(self.save_dir / f"training_metrics_{int(self.global_step):08d}.npz",
                 **{k: np.array(v, dtype=np.float32) for k, v in hist.items()})
        return plot_path
