"""
ppo.py - PPO trainer for Cuarenta.

This trainer starts from the imitation-learned policy in
`data/policies/cuarenta_policy_final.pt`, then improves it with PPO in
self-play. Policy snapshots are written as
`data/policies/cuarenta_policy_iterN.pt`, where N is the next available
integer in that directory.
"""

from __future__ import annotations

import collections
import copy
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agents import HeuristicAgent
from env import CuarentaEnv, TEAMS
from imitation_learning.imitation_learning import ActionEncoder, PolicyNetwork


DEFAULT_POLICY_INIT = "data/policies/cuarenta_policy_final.pt"


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    # Rollout
    n_envs: int = 8
    n_steps: int = 512
    # PPO
    n_epochs: int = 4
    minibatch_size: int = 128
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Optimiser
    lr: float = 3e-4
    # Self-play / checkpointing
    checkpoint_every: int = 50_000
    pool_size: int = 20
    opponent_latest: float = 0.8
    opponent_mode: str = "selfplay"  # "heuristic" | "selfplay" | "random"
    save_dir: str = "data/policies"
    init_policy: str = DEFAULT_POLICY_INIT
    # Logging / training length
    log_every: int = 10
    total_steps: int = 5_000_000


# ─── Helpers ──────────────────────────────────────────────────────────────────


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten an observation dict in the same order used by imitation learning."""
    keys = sorted(obs.keys())
    return np.concatenate([obs[k].reshape(-1) for k in keys]).astype(np.float32)


def batch_obs_to_tensor(obs_batch: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Stack flattened observations into a batch tensor."""
    return torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach and clone a state dict so it can be stored in the opponent pool."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def next_policy_iter(save_dir: Path) -> int:
    """Return the next available iteration number for `cuarenta_policy_iterN.pt`."""
    pattern = re.compile(r"^cuarenta_policy_iter(\d+)\.pt$")
    max_iter = 0
    if save_dir.exists():
        for path in save_dir.glob("cuarenta_policy_iter*.pt"):
            match = pattern.match(path.name)
            if match:
                max_iter = max(max_iter, int(match.group(1)))
    return max_iter + 1


def masked_legal_logits(
    logits_1d: torch.Tensor,
    legal_action_ids: List[int],
) -> torch.Tensor:
    """Gather logits for only the legal action ids in the provided order."""
    return logits_1d[legal_action_ids]


# ─── Model ───────────────────────────────────────────────────────────────────


class FlatActorCritic(nn.Module):
    """
    Actor-critic built around the imitation-learning policy network.

    The policy submodule keeps the original `PolicyNetwork` layer names so the
    imitation checkpoint can be loaded directly into it.
    """

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.policy.relu(self.policy.fc1(obs))
        x = self.policy.relu(self.policy.fc2(x))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(obs)
        logits = self.policy.fc3(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(obs)
        return self.value_head(hidden).squeeze(-1)


# ─── Rollout Buffer ───────────────────────────────────────────────────────────


@dataclass
class Rollout:
    """Stores one rollout's worth of data for a single learning seat."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)  # index within the legal action list
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    legal_action_ids: List[List[int]] = field(default_factory=list)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.legal_action_ids.clear()


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (advantages, returns) both shaped (T,)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae = delta + gamma * lam * (1.0 - next_done) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def select_legal_action(
    logits_1d: torch.Tensor,
    legal_action_ids: List[int],
    deterministic: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Sample or select the best action among the provided legal action ids.

    Returns:
        legal_index: index into the legal action list
        log_prob:    log probability of the selected action within that legal set
        entropy:     entropy of the legal-action distribution
    """
    legal_logits = masked_legal_logits(logits_1d, legal_action_ids)
    dist = torch.distributions.Categorical(logits=legal_logits)
    if deterministic:
        legal_index = int(torch.argmax(legal_logits).item())
    else:
        legal_index = int(dist.sample().item())
    choice = torch.tensor(legal_index, device=logits_1d.device)
    log_prob = dist.log_prob(choice)
    entropy = dist.entropy()
    return legal_index, log_prob, entropy


# ─── PPO Trainer ──────────────────────────────────────────────────────────────


class PPOTrainer:
    """
    Trains a Cuarenta policy with PPO and self-play.

    The agent learns as team 0 (players 0 and 2). Opponents are either a
    heuristic policy, random policy, or a sampled checkpoint from the
    current self-play pool.
    """

    def __init__(self, cfg: PPOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[PPO] Using device: {self.device}")

        # Environments
        self.envs = [CuarentaEnv(seed=i) for i in range(cfg.n_envs)]
        self.env_obs = [None] * cfg.n_envs
        self.env_done = [True] * cfg.n_envs
        self.action_encoders = [ActionEncoder(env) for env in self.envs]

        # Infer observation/action sizes from the real environment.
        sample_obs, _ = self.envs[0].reset()
        self.obs_size = int(flatten_obs(sample_obs).shape[0])
        self.action_size = int(self.action_encoders[0].max_action_idx)

        # Networks
        self.net = FlatActorCritic(self.obs_size, self.action_size).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

        self.opponent_net = PolicyNetwork(obs_size=self.obs_size, action_size=self.action_size).to(self.device)
        self.opponent_net.load_state_dict(self.net.policy.state_dict())
        self.opponent_net.eval()

        # Opponent pool for self-play
        self.ckpt_pool: List[Dict[str, torch.Tensor]] = []
        self.ckpt_step: List[int] = []

        # Stats
        self.global_step = 0
        self.episode_rewards = collections.deque(maxlen=100)
        self.episode_lengths = collections.deque(maxlen=100)
        self.elo_score = collections.deque(maxlen=500)  # 1 = win, 0 = loss
        self.training_history: Dict[str, List[float]] = collections.defaultdict(list)

        # Misc
        self.heuristic_agent = HeuristicAgent()
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.next_policy_iter = next_policy_iter(self.save_dir)

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        rollout_num = 0
        t0 = time.time()
        pbar = tqdm(total=cfg.total_steps, desc="Training", unit="step")

        while self.global_step < cfg.total_steps:
            rollout_start = time.time()
            rollouts = self._collect_rollout()
            stats = self._ppo_update(rollouts)

            prev_step = self.global_step
            self.global_step += cfg.n_envs * cfg.n_steps
            rollout_num += 1

            rollout_elapsed = time.time() - rollout_start
            self._record_training_metrics(rollout_num, stats, rollout_elapsed)

            if self.global_step % cfg.checkpoint_every < cfg.n_envs * cfg.n_steps:
                self._save_policy_snapshot()

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
        final_path = self._save_policy_snapshot()
        plot_path = self._save_training_plots()
        print(f"[PPO] Training complete. {self.global_step} steps.")
        print(f"[PPO] Final policy snapshot → {final_path}")
        print(f"[PPO] Training plot → {plot_path}")

    # ── Rollout collection ───────────────────────────────────────────────────

    def _collect_rollout(self) -> List[Rollout]:
        """
        Run n_steps in each env and collect experience for the learning team.

        Team 0 (players 0 and 2) is trained. Team 1 uses the configured opponent.
        """
        cfg = self.cfg
        rollouts = [Rollout() for _ in range(cfg.n_envs)]
        ep_reward = [0.0] * cfg.n_envs
        ep_len = [0] * cfg.n_envs

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

                    current_player = env.state.current_player
                    is_learning = TEAMS[current_player] == 0
                    obs_flat = self.env_obs[env_idx]
                    legal_actions = env.legal_actions()
                    legal_action_ids = [self.action_encoders[env_idx].action_to_idx(a) for a in legal_actions]

                    if is_learning:
                        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
                        logits, value = self.net(obs_t)
                        action_pos, log_prob, _ = select_legal_action(logits[0], legal_action_ids)
                        action = legal_actions[action_pos]
                    else:
                        if cfg.opponent_mode == "heuristic":
                            action = self.heuristic_agent.act(env.observe(current_player), legal_actions)
                        elif cfg.opponent_mode == "random":
                            action = random.choice(legal_actions)
                        else:
                            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
                            logits = self.opponent_net(obs_t)
                            action_pos, _, _ = select_legal_action(logits[0], legal_action_ids)
                            action = legal_actions[action_pos]

                    next_obs, reward, done, info = env.step(action)
                    ep_reward[env_idx] += reward
                    ep_len[env_idx] += 1

                    if is_learning:
                        rollouts[env_idx].obs.append(obs_flat.copy())
                        rollouts[env_idx].actions.append(action_pos)
                        rollouts[env_idx].log_probs.append(log_prob.item())
                        rollouts[env_idx].rewards.append(reward)
                        rollouts[env_idx].values.append(value.item())
                        rollouts[env_idx].dones.append(done)
                        rollouts[env_idx].legal_action_ids.append(legal_action_ids)

                    self.env_obs[env_idx] = flatten_obs(next_obs)
                    if done:
                        self.env_done[env_idx] = True
                        self.episode_rewards.append(ep_reward[env_idx])
                        self.episode_lengths.append(ep_len[env_idx])
                        if "winner" in info:
                            self.elo_score.append(1.0 if info["winner"] == 0 else 0.0)

        self.net.train()
        return rollouts

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self, rollouts: List[Rollout]) -> Dict[str, float]:
        cfg = self.cfg

        all_obs: List[np.ndarray] = []
        all_actions: List[int] = []
        all_old_lps: List[float] = []
        all_advs: List[float] = []
        all_rets: List[float] = []
        all_legal_ids: List[List[int]] = []

        for env_idx, rl in enumerate(rollouts):
            if not rl.rewards:
                continue

            if self.env_done[env_idx]:
                last_val = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(self.env_obs[env_idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                    last_val = float(self.net.get_value(obs_t).item())

            rewards = np.array(rl.rewards, dtype=np.float32)
            values = np.array(rl.values, dtype=np.float32)
            dones = np.array(rl.dones, dtype=np.float32)

            advs, rets = compute_gae(rewards, values, dones, last_val, cfg.gamma, cfg.gae_lambda)

            all_obs.extend(rl.obs)
            all_actions.extend(rl.actions)
            all_old_lps.extend(rl.log_probs)
            all_advs.extend(advs.tolist())
            all_rets.extend(rets.tolist())
            all_legal_ids.extend(rl.legal_action_ids)

        if not all_obs:
            return {"loss_pi": 0.0, "loss_vf": 0.0, "entropy": 0.0}

        obs_tensor = batch_obs_to_tensor(all_obs, self.device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_lps_tensor = torch.tensor(all_old_lps, dtype=torch.float32, device=self.device)
        advs_tensor = torch.tensor(all_advs, dtype=torch.float32, device=self.device)
        rets_tensor = torch.tensor(all_rets, dtype=torch.float32, device=self.device)

        advs_tensor = (advs_tensor - advs_tensor.mean()) / (advs_tensor.std() + 1e-8)

        stats = collections.defaultdict(list)
        N = len(all_obs)

        for _ in range(cfg.n_epochs):
            idxs = torch.randperm(N, device=self.device)
            for start in range(0, N, cfg.minibatch_size):
                mb_idx = idxs[start : start + cfg.minibatch_size]
                mb_obs = obs_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_lps = old_lps_tensor[mb_idx]
                mb_advs = advs_tensor[mb_idx]
                mb_rets = rets_tensor[mb_idx]
                mb_legal_ids = [all_legal_ids[i] for i in mb_idx.tolist()]

                logits, values = self.net(mb_obs)

                log_probs = []
                entropies = []
                for row, legal_ids in enumerate(mb_legal_ids):
                    legal_logits = logits[row, legal_ids]
                    dist = torch.distributions.Categorical(logits=legal_logits)
                    choice = mb_actions[row]
                    log_probs.append(dist.log_prob(choice))
                    entropies.append(dist.entropy())

                log_probs_t = torch.stack(log_probs)
                entropy_t = torch.stack(entropies)

                ratio = torch.exp(log_probs_t - mb_old_lps)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advs
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_vf = F.mse_loss(values, mb_rets)
                loss_ent = -entropy_t.mean()

                loss = loss_pi + cfg.vf_coef * loss_vf + cfg.ent_coef * loss_ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"].append(loss_pi.item())
                stats["loss_vf"].append(loss_vf.item())
                stats["entropy"].append((-loss_ent).item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _record_training_metrics(self, rollout_num: int, stats: Dict[str, float], rollout_elapsed: float):
        """Store metrics for end-of-training plotting."""
        self.training_history["rollout"].append(float(rollout_num))
        self.training_history["step"].append(float(self.global_step))
        self.training_history["loss_pi"].append(float(stats.get("loss_pi", 0.0)))
        self.training_history["loss_vf"].append(float(stats.get("loss_vf", 0.0)))
        self.training_history["entropy"].append(float(stats.get("entropy", 0.0)))
        self.training_history["win_rate"].append(float(np.mean(self.elo_score)) if self.elo_score else float("nan"))
        self.training_history["episode_reward"].append(float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0)
        self.training_history["episode_length"].append(float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0)
        self.training_history["sps"].append(float((self.cfg.n_envs * self.cfg.n_steps) / max(rollout_elapsed, 1e-8)))

    def _save_training_plots(self) -> Path:
        """Write a matplotlib summary of the recorded training metrics."""
        if not self.training_history["rollout"]:
            return self.save_dir / "training_metrics_empty.png"

        steps = np.array(self.training_history["step"], dtype=np.float32)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
        axes = axes.reshape(3, 2)

        plots = [
            (0, 0, "loss_pi", "Policy Loss"),
            (0, 1, "loss_vf", "Value Loss"),
            (1, 0, "entropy", "Entropy"),
            (1, 1, "win_rate", "Win Rate"),
            (2, 0, "episode_reward", "Episode Reward"),
            (2, 1, "episode_length", "Episode Length"),
        ]

        for row, col, key, title in plots:
            ax = axes[row, col]
            values = np.array(self.training_history[key], dtype=np.float32)
            ax.plot(steps, values, label=title, color="#1f77b4", linewidth=1.8)
            if len(values) >= 5 and key in {"loss_pi", "loss_vf", "entropy", "episode_reward", "episode_length"}:
                window = min(10, len(values))
                smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                smooth_steps = steps[window - 1 :]
                ax.plot(smooth_steps, smooth, color="#ff7f0e", linewidth=2.0, label=f"{window}-roll mean")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        axes[2, 0].set_xlabel("Training step")
        axes[2, 1].set_xlabel("Training step")
        axes[1, 0].set_ylabel("Metric value")
        axes[2, 0].set_ylabel("Metric value")

        fig.suptitle(
            f"Cuarenta PPO Training Metrics (steps={int(self.global_step)}, iter={self.next_policy_iter - 1})",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])

        plot_path = self.save_dir / f"training_metrics_{int(self.global_step):08d}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        metrics_path = self.save_dir / f"training_metrics_{int(self.global_step):08d}.npz"
        np.savez(
            metrics_path,
            **{k: np.array(v, dtype=np.float32) for k, v in self.training_history.items()},
        )
        return plot_path

    # ── Checkpointing & self-play ─────────────────────────────────────────────

    def _save_policy_snapshot(self) -> Path:
        path = self.save_dir / f"cuarenta_policy_iter{self.next_policy_iter}.pt"
        self.next_policy_iter += 1

        state = {
            "step": self.global_step,
            "policy_state": clone_state_dict_to_cpu(self.net.policy.state_dict()),
            "value_state": clone_state_dict_to_cpu(self.net.value_head.state_dict()),
            "optimizer_state": self.opt.state_dict(),
            "obs_size": self.obs_size,
            "action_size": self.action_size,
        }
        torch.save(state, path)
        print(f"[PPO] Saved policy snapshot → {path}")

        self.ckpt_pool.append(clone_state_dict_to_cpu(self.net.policy.state_dict()))
        self.ckpt_step.append(self.global_step)

        if len(self.ckpt_pool) > self.cfg.pool_size:
            self.ckpt_pool.pop(0)
            self.ckpt_step.pop(0)

        self._refresh_opponent()
        return path

    def _refresh_opponent(self):
        """Sample from the checkpoint pool and load into the opponent network."""
        cfg = self.cfg
        if not self.ckpt_pool:
            return

        if random.random() < cfg.opponent_latest:
            sd = self.ckpt_pool[-1]
        else:
            sd = random.choice(self.ckpt_pool)

        self.opponent_net.load_state_dict(sd)
        self.opponent_net.eval()

    def load_policy(self, path: str):
        """Warm-start from a policy checkpoint or an imitation-learning policy file."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        if "net" in state:
            self.net.load_state_dict(state["net"])
            if "opt" in state:
                self.opt.load_state_dict(state["opt"])
            self.global_step = int(state.get("step", 0))
        else:
            if "policy_state" not in state:
                raise KeyError(f"Unsupported checkpoint format: {path}")
            self.net.policy.load_state_dict(state["policy_state"])
            if "value_state" in state:
                self.net.value_head.load_state_dict(state["value_state"])
            if "optimizer_state" in state:
                self.opt.load_state_dict(state["optimizer_state"])
            self.global_step = int(state.get("step", 0))

        self.opponent_net.load_state_dict(self.net.policy.state_dict())
        self.opponent_net.eval()
        print(f"[PPO] Loaded policy from {path} (step={self.global_step})")


# ─── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="data/policies")
    parser.add_argument("--init-policy", type=str, default=DEFAULT_POLICY_INIT)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--opponent-mode",
        type=str,
        default="selfplay",
        choices=["heuristic", "selfplay", "random"],
    )
    parser.add_argument("--device", type=str, default='mps')
    args = parser.parse_args()

    cfg = PPOConfig(
        total_steps=args.steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        lr=args.lr,
        save_dir=args.save_dir,
        init_policy=args.init_policy,
        opponent_mode=args.opponent_mode,
    )

    trainer = PPOTrainer(cfg, device=args.device)

    if args.init_policy:
        init_path = Path(args.init_policy)
        if init_path.exists():
            trainer.load_policy(str(init_path))
        else:
            print(f"[PPO] Init policy not found, training from scratch: {init_path}")

    if args.resume:
        trainer.load_policy(args.resume)

    trainer.train()
