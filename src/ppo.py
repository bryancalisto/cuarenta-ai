"""
ppo.py — PPO trainer for Cuarenta.

Trains a single CuarentaNet that plays all 4 seats (parameter sharing).
Self-play: the opponent pool holds past checkpoints; the current agent
plays against a mix of them.

Hyperparameters are grouped in PPOConfig for easy tuning.
"""

from __future__ import annotations

import time
import random
import collections
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import CuarentaNet, ActionMasks
from env import CuarentaEnv, TEAMS
from agents import HeuristicAgent
from utils import obs_to_tensors


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Rollout
    n_envs:           int   = 8          # parallel environments
    n_steps:          int   = 512        # steps per env per rollout
    # PPO
    n_epochs:         int   = 4          # gradient epochs per rollout
    minibatch_size:   int   = 128
    clip_eps:         float = 0.2
    vf_coef:          float = 0.5
    ent_coef:         float = 0.01
    max_grad_norm:    float = 0.5
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    # Optimiser
    lr:               float = 3e-4
    # Self-play
    checkpoint_every: int   = 50_000    # steps between checkpoint saves
    pool_size:        int   = 20        # max old checkpoints to keep
    opponent_latest:  float = 0.8       # fraction of games vs latest ckpt
    opponent_mode:    str   = "heuristic"  # "heuristic" | "selfplay" | "random"
    # Logging
    log_every:        int   = 10        # rollouts between log prints
    save_dir:         str   = "checkpoints"
    # Training length
    total_steps:      int   = 5_000_000


# ─── Rollout Buffer ───────────────────────────────────────────────────────────

@dataclass
class Rollout:
    """Stores one rollout's worth of data for a single agent seat."""
    obs:          List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions:      List[int]                   = field(default_factory=list)
    log_probs:    List[float]                 = field(default_factory=list)
    rewards:      List[float]                 = field(default_factory=list)
    values:       List[float]                 = field(default_factory=list)
    dones:        List[bool]                  = field(default_factory=list)
    # ActionMasks stored per step for recomputing log_probs during update
    masks:        List[List[Tuple[int,int]]]  = field(default_factory=list)

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()
        self.masks.clear()


def compute_gae(
    rewards: np.ndarray,   # (T,)
    values: np.ndarray,    # (T,)
    dones: np.ndarray,     # (T,)
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (advantages, returns) both (T,)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val  = last_value if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae   = delta + gamma * lam * (1.0 - next_done) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ─── PPO Trainer ──────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Trains a CuarentaNet with PPO + self-play.

    All 4 seats share the same network (parameter sharing across seats).
    Opponents are sampled from a checkpoint pool.
    """

    def __init__(self, cfg: PPOConfig, device: Optional[str] = None):
        self.cfg    = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[PPO] Using device: {self.device}")

        # Networks
        self.net = CuarentaNet().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

        # Opponent
        self.heuristic_agent = HeuristicAgent()
        self.opponent_net = CuarentaNet().to(self.device)
        self.opponent_net.load_state_dict(self.net.state_dict())
        self.opponent_net.eval()

        # Checkpoint pool for self-play opponents
        self.ckpt_pool: List[Dict] = []
        self.ckpt_step: List[int]  = []

        # Envs
        self.envs = [CuarentaEnv(seed=i) for i in range(cfg.n_envs)]
        self.env_obs  = [None] * cfg.n_envs   # current obs per env
        self.env_done = [True]  * cfg.n_envs  # reset if True

        # Stats
        self.global_step    = 0
        self.episode_rewards = collections.deque(maxlen=100)
        self.episode_lengths = collections.deque(maxlen=100)
        self.elo_score       = collections.deque(maxlen=500)  # 1=win, 0=loss

        # Save dir
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        rollout_num = 0
        t0 = time.time()

        pbar = tqdm(total=cfg.total_steps, desc="Training", unit="step")

        while self.global_step < cfg.total_steps:
            rollouts = self._collect_rollout()
            stats = self._ppo_update(rollouts)

            prev_step = self.global_step
            self.global_step += cfg.n_envs * cfg.n_steps
            rollout_num += 1

            if self.global_step % cfg.checkpoint_every < cfg.n_envs * cfg.n_steps:
                self._save_checkpoint()

            # ── Logging ──────────────────────────────────────────────────────
            if rollout_num % cfg.log_every == 0:
                elapsed = time.time() - t0
                sps = self.global_step / elapsed
                wr = np.mean(self.elo_score) if self.elo_score else float("nan")
                pbar.set_postfix_str(
                    f"sps={sps:>6.0f} | "
                    f"win_rate={wr:.3f} | "
                    f"loss_pi={stats['loss_pi']:.4f} | "
                    f"loss_vf={stats['loss_vf']:.4f} | "
                    f"entropy={stats['entropy']:.4f} | "
                    f"ep_rew={np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}"
                )

            pbar.update(self.global_step - prev_step)

        pbar.close()
        self._save_checkpoint(final=True)
        print(f"[PPO] Training complete. {self.global_step} steps.")

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(self) -> List[Rollout]:
        """
        Run n_steps in each env. Collect experience for the LEARNING agent
        (team 0: players 0 and 2). Opponents (team 1) use the opponent_net.
        """
        cfg   = self.cfg
        net   = self.net
        onet  = self.opponent_net

        # Per-env rollout buffers (one per env)
        rollouts = [Rollout() for _ in range(cfg.n_envs)]
        ep_reward = [0.0] * cfg.n_envs
        ep_len    = [0]   * cfg.n_envs

        net.eval()

        with torch.no_grad():
            for _ in range(cfg.n_steps):
                for env_idx, env in enumerate(self.envs):
                    # Reset if needed
                    if self.env_done[env_idx]:
                        obs, _ = env.reset()
                        self.env_obs[env_idx] = obs
                        self.env_done[env_idx] = False
                        ep_reward[env_idx] = 0.0
                        ep_len[env_idx]    = 0

                    current_player = env.state.current_player
                    is_learning    = (TEAMS[current_player] == 0)
                    obs            = self.env_obs[env_idx]
                    legal          = env.legal_actions()

                    if is_learning:
                        obs_t  = obs_to_tensors([obs], self.device)
                        masks  = ActionMasks.from_env_actions([legal])
                        action_idx, log_prob, _ = net.act(obs_t, masks)
                        value = net.get_value(obs_t).item()
                        action = legal[action_idx]
                    else:
                        # Opponent action depends on mode
                        if cfg.opponent_mode == "heuristic":
                            action = self.heuristic_agent.act(obs, legal)
                        elif cfg.opponent_mode == "random":
                            action = random.choice(legal)
                        else:  # selfplay
                            obs_t  = obs_to_tensors([obs], self.device)
                            masks  = ActionMasks.from_env_actions([legal])
                            action_idx, _, _ = onet.act(obs_t, masks)
                            action = legal[action_idx]

                    next_obs, reward, done, info = env.step(action)
                    ep_reward[env_idx] += reward
                    ep_len[env_idx]    += 1

                    if is_learning:
                        rollouts[env_idx].obs.append(obs)
                        rollouts[env_idx].actions.append(action_idx)
                        rollouts[env_idx].log_probs.append(log_prob.item())
                        rollouts[env_idx].rewards.append(reward)
                        rollouts[env_idx].values.append(value)
                        rollouts[env_idx].dones.append(done)
                        rollouts[env_idx].masks.append(masks.legal_actions[0])

                    self.env_obs[env_idx] = next_obs
                    if done:
                        self.env_done[env_idx] = True
                        self.episode_rewards.append(ep_reward[env_idx])
                        self.episode_lengths.append(ep_len[env_idx])
                        if "winner" in info:
                            self.elo_score.append(1.0 if info["winner"] == 0 else 0.0)

        net.train()
        return rollouts

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self, rollouts: List[Rollout]) -> Dict[str, float]:
        cfg = self.cfg

        # Compute GAE for each rollout, then flatten
        all_obs, all_actions, all_old_lps, all_advs, all_rets, all_masks = \
            [], [], [], [], [], []

        for env_idx, rl in enumerate(rollouts):
            if not rl.rewards:
                continue
            T = len(rl.rewards)

            # Bootstrap value for last state
            if self.env_done[env_idx]:
                last_val = 0.0
            else:
                with torch.no_grad():
                    obs_t    = obs_to_tensors([self.env_obs[env_idx]], self.device)
                    last_val = self.net.get_value(obs_t).item()

            rewards  = np.array(rl.rewards,   dtype=np.float32)
            values   = np.array(rl.values,    dtype=np.float32)
            dones    = np.array(rl.dones,     dtype=np.float32)

            advs, rets = compute_gae(rewards, values, dones, last_val, cfg.gamma, cfg.gae_lambda)

            all_obs.extend(rl.obs)
            all_actions.extend(rl.actions)
            all_old_lps.extend(rl.log_probs)
            all_advs.extend(advs.tolist())
            all_rets.extend(rets.tolist())
            all_masks.extend(rl.masks)

        if not all_obs:
            return {"loss_pi": 0.0, "loss_vf": 0.0, "entropy": 0.0}

        N = len(all_obs)
        all_actions = torch.tensor(all_actions, dtype=torch.long,  device=self.device)
        all_old_lps = torch.tensor(all_old_lps, dtype=torch.float32, device=self.device)
        all_advs    = torch.tensor(all_advs,    dtype=torch.float32, device=self.device)
        all_rets    = torch.tensor(all_rets,    dtype=torch.float32, device=self.device)

        # Normalise advantages
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8)

        stats = collections.defaultdict(list)

        for _ in range(cfg.n_epochs):
            idxs = torch.randperm(N)
            for start in range(0, N, cfg.minibatch_size):
                mb_idx  = idxs[start:start + cfg.minibatch_size]
                mb_obs  = obs_to_tensors([all_obs[i] for i in mb_idx.tolist()], self.device)
                mb_acts = all_actions[mb_idx]
                mb_olps = all_old_lps[mb_idx]
                mb_advs = all_advs[mb_idx]
                mb_rets = all_rets[mb_idx]
                mb_masks = ActionMasks([all_masks[i] for i in mb_idx.tolist()])

                log_probs, entropy, values = self.net.evaluate_actions(mb_obs, mb_acts, mb_masks)

                # Policy loss (clipped)
                ratio    = torch.exp(log_probs - mb_olps)
                surr1    = ratio * mb_advs
                surr2    = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advs
                loss_pi  = -torch.min(surr1, surr2).mean()

                # Value loss
                loss_vf  = F.mse_loss(values, mb_rets)

                # Entropy bonus
                loss_ent = -entropy.mean()

                loss = loss_pi + cfg.vf_coef * loss_vf + cfg.ent_coef * loss_ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"].append(loss_pi.item())
                stats["loss_vf"].append(loss_vf.item())
                stats["entropy"].append((-loss_ent).item())

        return {k: np.mean(v) for k, v in stats.items()}

    # ── Checkpointing & self-play ─────────────────────────────────────────────

    def _save_checkpoint(self, final: bool = False):
        cfg  = self.cfg
        step = self.global_step
        tag  = "final" if final else f"{step:08d}"
        path = Path(cfg.save_dir) / f"ckpt_{tag}.pt"

        state = {
            "step":       step,
            "net":        self.net.state_dict(),
            "opt":        self.opt.state_dict(),
            "elo_score":  list(self.elo_score),
        }
        torch.save(state, path)
        print(f"[PPO] Saved checkpoint → {path}")

        # Add to pool
        self.ckpt_pool.append(self.net.state_dict())
        self.ckpt_step.append(step)

        # Trim pool
        if len(self.ckpt_pool) > cfg.pool_size:
            self.ckpt_pool.pop(0)
            self.ckpt_step.pop(0)

        # Update opponent network from pool
        self._refresh_opponent()

    def _refresh_opponent(self):
        """Sample from checkpoint pool and load into opponent_net."""
        cfg = self.cfg
        if not self.ckpt_pool:
            return

        if random.random() < cfg.opponent_latest:
            # Play against latest checkpoint
            sd = self.ckpt_pool[-1]
        else:
            # Play against random older checkpoint
            sd = random.choice(self.ckpt_pool)

        self.opponent_net.load_state_dict(sd)
        self.opponent_net.eval()

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.global_step = state["step"]
        print(f"[PPO] Loaded checkpoint from {path} (step={self.global_step})")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int,   default=5_000_000)
    parser.add_argument("--n-envs",    type=int,   default=8)
    parser.add_argument("--n-steps",   type=int,   default=512)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--save-dir",  type=str,   default="checkpoints")
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--opponent-mode", type=str, default="heuristic",
                        choices=["heuristic", "selfplay", "random"])
    parser.add_argument("--device",    type=str,   default=None)
    args = parser.parse_args()

    cfg = PPOConfig(
        total_steps   = args.steps,
        n_envs        = args.n_envs,
        n_steps       = args.n_steps,
        lr            = args.lr,
        save_dir      = args.save_dir,
        opponent_mode = args.opponent_mode,
    )

    trainer = PPOTrainer(cfg, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()