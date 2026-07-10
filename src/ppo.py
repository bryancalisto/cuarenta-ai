from __future__ import annotations

import collections
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from agents import HeuristicAgent, RandomAgent

import matplotlib

from networks import FlatActorCritic
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from env import ActionEncoder, CuarentaEnv, TEAMS


DEFAULT_POLICY_INIT = "data/policies/cuarenta_dagger_policy_final.pt"

LSTMState = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class PPOConfig:
    # Rollout
    n_envs: int = 8
    n_steps: int = 512
    # PPO
    n_epochs: int = 4
    minibatch_size: int = 128          # used only when use_lstm=False
    envs_per_minibatch: int = 4        # used only when use_lstm=True
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    # optimiser
    lr: float = 3e-4
    # model
    use_lstm: bool = True
    lstm_hidden: int = 128
    trunk_hidden: int = 256
    value_hidden: int = 128
    # checkpointing
    checkpoint_every: int = 100_000
    save_dir: str = "data/policies"
    init_policy: str = DEFAULT_POLICY_INIT
    # self-play
    pool_size: int = 20
    opponent_latest: float = 0.8
    opponent_mode: str = "selfplay"  # "heuristic" | "selfplay" | "random"
    # Logging / training length
    log_every: int = 1
    total_steps: int = 5_000_000


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten an observation dict"""
    keys = sorted(obs.keys())
    return np.concatenate([obs[k].reshape(-1) for k in keys]).astype(np.float32)


def batch_obs_to_tensor(obs_batch: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Stack flattened observations into a batch tensor."""
    return torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach and clone a state dict so it can be stored in the opponent pool."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def clone_lstm_state(state: Optional[LSTMState]) -> Optional[LSTMState]:
    """Detach + clone an (h, c) LSTM state so it can be safely stashed for later reuse."""
    if state is None:
        return None
    h, c = state
    return (h.detach().cpu().clone(), c.detach().cpu().clone())


def lstm_state_to_device(state: Optional[LSTMState], device: torch.device) -> Optional[LSTMState]:
    if state is None:
        return None
    h, c = state
    return (h.to(device), c.to(device))


def masked_legal_logits(
    logits_1d: torch.Tensor,
    legal_action_ids: List[int],
) -> torch.Tensor:
    """Gather logits for only the legal action ids in the provided order."""
    return logits_1d[legal_action_ids]


def run_recurrent_forward(
    net: FlatActorCritic,
    obs_seq: torch.Tensor,      # (T, obs_size), time-ordered for ONE environment
    dones: torch.Tensor,        # (T,) float32, 1.0 marks the last step of an episode
    init_state: Optional[LSTMState],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replay a full, time-ordered sequence of transitions from a single
    environment through the recurrent actor-critic, returning logits and
    values for EVERY timestep (unlike FlatActorCritic.forward, which only
    returns the last step).

    The hidden state is carried across steps exactly as it was during
    rollout collection, and is reset to zero immediately after any step
    flagged `done` — mirroring `self.lstm_states[env_idx] = None` on episode
    reset in `_collect_rollout`. This is what makes recurrent training
    correct: episode boundaries in the buffered data produce the same
    resets here as they did when the data was generated.
    """
    T = obs_seq.shape[0]

    # Split into contiguous segments; a segment ends right after a `done`.
    segments: List[Tuple[int, int]] = []
    seg_start = 0
    for t in range(T):
        if dones[t].item() > 0.5:
            segments.append((seg_start, t + 1))
            seg_start = t + 1
    if seg_start < T:
        segments.append((seg_start, T))

    state = init_state
    logits_chunks, values_chunks = [], []
    for s, e in segments:
        seq = obs_seq[s:e].unsqueeze(0)  # (1, seg_len, obs_size)
        x = net.policy.relu(net.policy.fc1(seq))
        x = net.policy.relu(net.policy.fc2(x))
        x, state = net.lstm(x, state)
        logits_chunks.append(net.fc3(x).squeeze(0))
        values_chunks.append(net.value_head(x).squeeze(0).squeeze(-1))
        state = None  # a `done` ended this segment; the next one starts fresh

    return torch.cat(logits_chunks, dim=0), torch.cat(values_chunks, dim=0)


@dataclass
class Rollout:
    """Stores one rollout's worth of data for a single learning seat, in order."""

    # Full per-turn trajectory (every seat's turn) — needed so the LSTM can be
    # replayed through the whole match, not just the learner's own decisions.
    full_obs: List[np.ndarray] = field(default_factory=list)
    full_dones: List[bool] = field(default_factory=list)
    learn_positions: List[int] = field(default_factory=list)  # index into full_obs for each learning transition

    # Per-learning-transition data, aligned with learn_positions.
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    legal_action_ids: List[List[int]] = field(default_factory=list)

    hidden_start: Optional[LSTMState] = None

    def clear(self):
        self.full_obs.clear()
        self.full_dones.clear()
        self.learn_positions.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.legal_action_ids.clear()
        self.hidden_start = None


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


class RunningMeanStd:
    """
    Running mean/variance of a scalar stream via Welford's parallel algorithm.

    Used to track the scale of the value function's regression target (returns).
    Raw Cuarenta point rewards span a wide, drifting range (captures, caídas,
    end-of-hand bonuses), so the value head is trained in a normalized space
    that stays ~unit-scale regardless of how the reward magnitude shifts as play
    quality improves. GAE and bootstrapping happen in real reward space, so
    network value outputs are unnormalized on the way in and targets normalized
    on the way out.
    """

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps  # small non-zero prior so the first update is stable

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = int(x.size)

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: Dict[str, float]):
        self.mean = float(sd["mean"])
        self.var = float(sd["var"])
        self.count = float(sd["count"])


class PPOTrainer:
    """
    Trains a Cuarenta policy with PPO and self-play.

    Learns to play optimally from any team via randomized team assignment.
    Opponents sampled from checkpoint pool (self-play).

    Recurrence design note: when `cfg.use_lstm` is True, both the learner and
    the opponent maintain a persistent per-environment LSTM hidden state that
    is only advanced on that seat's own turns (the observation already
    encodes everything that happened on other seats' turns). During the PPO
    update, each environment's buffered transitions are replayed through the
    network IN ORDER starting from the hidden state recorded at rollout time
    (a "stored state" approach, as in R2D2 / sample-factory), rather than
    being shuffled into independent single-step samples. This is what makes
    the recurrence actually learn something — shuffling timesteps (the
    original approach) trains the LSTM as if it were a stateless layer.
    """

    def __init__(self, cfg: PPOConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[PPO] Using device: {self.device}")
        print(f"[PPO] LSTM enabled: {cfg.use_lstm}")
        print(f"[PPO] Trunk hidden: {cfg.trunk_hidden}")
        if cfg.use_lstm:
            print(f"[PPO] LSTM hidden: {cfg.lstm_hidden}")
        print(f"[PPO] Trunk hidden: {cfg.value_hidden}")

        # Environments
        self.envs = [
            CuarentaEnv(seed=i)
            for i in range(cfg.n_envs)
        ]
        self.env_obs = [None] * cfg.n_envs
        self.env_done = [True] * cfg.n_envs
        self.action_encoders = [ActionEncoder(env) for env in self.envs]
        self.lstm_states: Optional[List[Optional[LSTMState]]] = [None] * cfg.n_envs if cfg.use_lstm else None
        self.opponent_lstm_states: Optional[List[Optional[LSTMState]]] = [None] * cfg.n_envs if cfg.use_lstm else None

        # Infer observation/action sizes from the real environment.
        sample_obs, _ = self.envs[0].reset()
        self.obs_size = int(flatten_obs(sample_obs).shape[0])
        self.action_size = int(self.action_encoders[0].max_action_idx)

        self.net = FlatActorCritic(
            self.obs_size, self.action_size,
            use_lstm=cfg.use_lstm, lstm_hidden=cfg.lstm_hidden,
            trunk_hidden=cfg.trunk_hidden, value_hidden=cfg.value_hidden,
        ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

        # Running stats of returns; value head trains in this normalized space.
        self.ret_rms = RunningMeanStd()

        # Opponent shares the SAME architecture as the learner (including the
        # LSTM), so checkpoints in the pool can be loaded into it directly.
        self.opponent_nets: List[FlatActorCritic] = [
            FlatActorCritic(
                self.obs_size, self.action_size,
                use_lstm=cfg.use_lstm, lstm_hidden=cfg.lstm_hidden,
                trunk_hidden=cfg.trunk_hidden, value_hidden=cfg.value_hidden,
            ).to(self.device)
            for _ in range(cfg.n_envs)
        ]
        for net in self.opponent_nets:
            net.load_state_dict(self.net.state_dict())
            net.eval()
        self.env_opponent_is_latest: List[bool] = [True] * cfg.n_envs

        self.ckpt_pool: List[Dict[str, torch.Tensor]] = []
        self.ckpt_step: List[int] = []

        self.global_step = 0
        self.episode_rewards = collections.deque(maxlen=100)
        self.episode_lengths = collections.deque(maxlen=100)
        self.elo_score = collections.deque(maxlen=500)  # 1 = win, 0 = loss
        self.winrate_vs_latest_deque = collections.deque(maxlen=200)
        self.winrate_vs_pool_deque = collections.deque(maxlen=200)
        self.team_0_wins = 0
        self.team_1_wins = 0
        self.total_games = 0
        self.team_0_wins_vs_latest = 0
        self.team_1_wins_vs_latest = 0
        self.games_vs_latest = 0
        self.training_history: Dict[str, List[float]] = collections.defaultdict(list)

        # Misc
        self.heuristic_agent = HeuristicAgent()
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _evaluate_vs(self, opponent_agent, n_games: int = 50) -> float:
        """Play n_games full games with the current net vs a fixed opponent,
        net always deterministic, opponent controlling 3 of 4 seats (rotated
        randomly like in training). Returns win rate for the net's team."""
        self.net.eval()
        wins = 0
        for _ in range(n_games):
            env = CuarentaEnv(seed=random.randrange(1_000_000))
            encoder = ActionEncoder(env)
            obs, _ = env.reset()
            learner = random.randrange(4)
            lstm_state = None
            if hasattr(opponent_agent, "reset"):
                opponent_agent.reset()

            done, info = False, {}
            while not done:
                player = env.state.current_player
                legal = env.legal_actions()
                obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)

                if self.cfg.use_lstm:
                    logits, _, lstm_state = self.net(obs_t, lstm_state)
                elif player == learner:
                    logits, _, _ = self.net(obs_t)

                if player == learner:
                    legal_ids = [encoder.action_to_idx(a) for a in legal]
                    action_pos, _, _ = select_legal_action(logits[0], legal_ids, deterministic=True)
                    action = legal[action_pos]
                else:
                    action = opponent_agent.act(env.observe(player), legal)

                obs, _, done, info = env.step(action)

            if info.get("winner") == TEAMS[learner]:
                wins += 1
        self.net.train()
        return wins / n_games

    def train(self):
        cfg = self.cfg
        rollout_num = 0
        remaining_steps = cfg.total_steps - self.global_step

        if self.global_step > 0:
            print(f"[PPO] Resuming from step {self.global_step}. Running {remaining_steps} more steps to reach {cfg.total_steps}.")

        pbar = tqdm(total=remaining_steps, desc="Training", unit="step")

        while self.global_step < cfg.total_steps:
            rollout_start = time.time()
            rollouts = self._collect_rollout()
            stats = self._ppo_update(rollouts)

            self.global_step += cfg.n_envs * cfg.n_steps
            rollout_num += 1

            rollout_elapsed = time.time() - rollout_start
            self._record_training_metrics(rollout_num, stats, rollout_elapsed)

            if self.global_step % cfg.checkpoint_every < cfg.n_envs * cfg.n_steps:
                self._save_policy_snapshot()
                heuristic_wr = self._evaluate_vs(self.heuristic_agent, n_games=50)
                random_wr = self._evaluate_vs(RandomAgent(seed=0), n_games=50)
                self.training_history["eval_step"].append(float(self.global_step))
                self.training_history["win_rate_vs_heuristic"].append(heuristic_wr)
                self.training_history["win_rate_vs_random"].append(random_wr)
                print(f"[PPO] Eval @ {self.global_step}: vs heuristic={heuristic_wr:.2%}  vs random={random_wr:.2%}")

            if rollout_num % cfg.log_every == 0:
                rew = np.mean(self.episode_rewards) if self.episode_rewards else 0
                ep_dur = np.mean(self.episode_lengths) if self.episode_lengths else 0
                pbar.set_postfix_str(
                    f"loss_pi={stats['loss_pi']:.4f} "
                    f"loss_vf={stats['loss_vf']:.4f} "
                    f"rew={rew:.2f} "
                    f"ep_dur={ep_dur:.1f} "
                    f"entropy={stats['entropy']:.4f}"
                )

            pbar.update(cfg.n_envs * cfg.n_steps)

        pbar.close()
        final_path = self._save_policy_snapshot()
        plot_path = self._save_training_plots()

        print(f"\n{'='*60}")
        print(f"[PPO] Training complete. {self.global_step} steps.")
        print(f"{'='*60}")
        print(f"Total games:          {self.total_games}")
        print(f"Team 0 wins (all):    {self.team_0_wins} ({100*self.team_0_wins/max(self.total_games,1):.1f}%)")
        print(f"Team 1 wins (all):    {self.team_1_wins} ({100*self.team_1_wins/max(self.total_games,1):.1f}%)")
        print(f"Games vs latest ckpt: {self.games_vs_latest}")
        print(f"Team 0 wins (latest): {self.team_0_wins_vs_latest} ({100*self.team_0_wins_vs_latest/max(self.games_vs_latest,1):.1f}%)")
        print(f"Team 1 wins (latest): {self.team_1_wins_vs_latest} ({100*self.team_1_wins_vs_latest/max(self.games_vs_latest,1):.1f}%)")
        print(f"Mean episode reward:  {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}")
        print(f"Mean episode length:  {np.mean(self.episode_lengths) if self.episode_lengths else 0:.1f}")
        print(f"Mean policy loss:     {np.mean(self.training_history['loss_pi']) if self.training_history['loss_pi'] else 0:.4f}")
        print(f"Mean value loss:      {np.mean(self.training_history['loss_vf']) if self.training_history['loss_vf'] else 0:.4f}")
        print(f"Mean win rate:        {np.mean(self.elo_score) if self.elo_score else 0:.3f}")
        print(f"Final policy snapshot → {final_path}")
        print(f"Training plot         → {plot_path}")
        print(f"{'='*60}\n")

    def _collect_rollout(self) -> List[Rollout]:
        """
        Collect experience for a SINGLE learning player per env via self-play.

        Exactly one seat (`learning_player`) generates gradient transitions.
        The other three seats are played by the opponent policy (or random) and
        never produce training data. Their effect on the game is only felt
        through the reward the learning seat receives.

        Reward attribution: every env step yields a reward from the acting
        player's team perspective. We re-express it from the LEARNING player's
        team perspective (sign-flipped when an opponent-team seat acted) and
        accumulate it onto the learning seat's most recent transition. This way
        an opponent's caída against us, or a teammate's capture for us, both
        shape the learning seat's move that led to it.
        """
        cfg = self.cfg
        rollouts = [Rollout() for _ in range(cfg.n_envs)]
        ep_reward = [0.0] * cfg.n_envs
        ep_len = [0] * cfg.n_envs
        learning_player = [random.randrange(4) for _ in range(cfg.n_envs)]
        pending_reward = [0.0] * cfg.n_envs  # reward awaiting attribution to last learning move

        # Snapshot the hidden state each env is starting this window with, so
        # the update step can replay from the exact same starting point.
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
                        learning_player[env_idx] = random.randrange(4)
                        if cfg.use_lstm:
                            self.lstm_states[env_idx] = None
                            self.opponent_lstm_states[env_idx] = None
                        if cfg.opponent_mode == "selfplay":
                            self._sample_opponent_for_env(env_idx)

                    learner = learning_player[env_idx]
                    current_player = env.state.current_player
                    is_learning = current_player == learner
                    obs_flat = self.env_obs[env_idx]
                    legal_actions = env.legal_actions()
                    legal_action_ids = [self.action_encoders[env_idx].action_to_idx(a) for a in legal_actions]
                    rl = rollouts[env_idx]

                    obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if cfg.use_lstm:
                        # Advance the learner's hidden state on EVERY turn, not just
                        # its own, so it accumulates memory of the whole match —
                        # including opponents' moves — not only its own decisions.
                        logits, value, lstm_state = self.net(obs_t, self.lstm_states[env_idx])
                        self.lstm_states[env_idx] = lstm_state
                    elif is_learning:
                        logits, value, _ = self.net(obs_t)
                    else:
                        logits, value = None, None

                    rl.full_obs.append(obs_flat.copy())
                    rl.full_dones.append(False)  # patched to True below if this turn ends the episode

                    if is_learning:
                        # Flush reward accrued since the previous learning move
                        # (its own outcome + intervening opponents) onto it.
                        if rl.rewards:
                            rl.rewards[-1] += pending_reward[env_idx]
                        pending_reward[env_idx] = 0.0

                        action_pos, log_prob, _ = select_legal_action(logits[0], legal_action_ids)
                        action = legal_actions[action_pos]

                        rl.learn_positions.append(len(rl.full_obs) - 1)
                        rl.actions.append(action_pos)
                        rl.log_probs.append(log_prob.item())
                        rl.rewards.append(0.0)  # filled by accumulation
                        rl.values.append(value.item())
                        rl.dones.append(False)
                        rl.legal_action_ids.append(legal_action_ids)
                    else:
                        if cfg.opponent_mode == "heuristic":
                            action = self.heuristic_agent.act(env.observe(current_player), legal_actions)
                        elif cfg.opponent_mode == "random":
                            action = random.choice(legal_actions)
                        else:
                            opp_net = self.opponent_nets[env_idx]
                            opp_state = self.opponent_lstm_states[env_idx] if cfg.use_lstm else None
                            opp_logits, _, new_opp_state = opp_net(obs_t, opp_state)
                            if cfg.use_lstm:
                                self.opponent_lstm_states[env_idx] = new_opp_state
                            action_pos, _, _ = select_legal_action(opp_logits[0], legal_action_ids)
                            action = legal_actions[action_pos]

                    next_obs, reward, done, info = env.step(action)

                    # Re-express reward from the learning seat's team perspective.
                    rew = reward if TEAMS[current_player] == TEAMS[learner] else 0
                    pending_reward[env_idx] += rew
                    ep_reward[env_idx] += rew
                    ep_len[env_idx] += 1

                    self.env_obs[env_idx] = flatten_obs(next_obs)
                    if done:
                        rl.full_dones[-1] = True
                        # Attribute the terminal outcome to the last learning move.
                        if rl.rewards:
                            rl.rewards[-1] += pending_reward[env_idx]
                            rl.dones[-1] = True
                        pending_reward[env_idx] = 0.0

                        self.env_done[env_idx] = True
                        self.episode_rewards.append(ep_reward[env_idx])
                        self.episode_lengths.append(ep_len[env_idx])
                        if "winner" in info:
                            self.total_games += 1
                            if info["winner"] == 0:
                                self.team_0_wins += 1
                            else:
                                self.team_1_wins += 1
                            if self.env_opponent_is_latest[env_idx]:
                                self.games_vs_latest += 1
                                if info["winner"] == 0:
                                    self.team_0_wins_vs_latest += 1
                                else:
                                    self.team_1_wins_vs_latest += 1
                            self.elo_score.append(1.0 if info["winner"] == TEAMS[learner] else 0.0)

                            # split win rate by opponent staleness
                            won = 1.0 if info["winner"] == TEAMS[learner] else 0.0
                            if self.env_opponent_is_latest[env_idx]:
                                self.winrate_vs_latest_deque.append(won)
                            else:
                                self.winrate_vs_pool_deque.append(won)


        self.net.train()
        return rollouts

    def _ppo_update(self, rollouts: List[Rollout]) -> Dict[str, float]:
        cfg = self.cfg

        env_records: List[Dict] = []
        all_advs_flat: List[float] = []
        all_rets_flat: List[float] = []

        # Stats from prior updates: the network's stored value predictions are in
        # normalized space, so unnormalize them back to real reward space before
        # GAE/bootstrapping. (Identity on the first update: mean=0, std≈1.)
        ret_mean, ret_std = self.ret_rms.mean, self.ret_rms.std

        for env_idx, rl in enumerate(rollouts):
            if not rl.rewards:
                continue

            if self.env_done[env_idx]:
                last_val = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(self.env_obs[env_idx], dtype=torch.float32, device=self.device).unsqueeze(0)
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
                "legal_ids": rl.legal_action_ids,
            })

            env_records.append(record)
            all_advs_flat.extend(advs.tolist())
            all_rets_flat.extend(rets.tolist())

        if not env_records:
            return {"loss_pi": 0.0, "loss_vf": 0.0, "entropy": 0.0}

        adv_mean = float(np.mean(all_advs_flat))
        adv_std = float(np.std(all_advs_flat) + 1e-8)

        # Fold this rollout's returns into the running stats, then normalize the
        # value targets with the refreshed stats so MSE regresses in unit scale.
        self.ret_rms.update(np.array(all_rets_flat, dtype=np.float32))
        ret_mean, ret_std = self.ret_rms.mean, self.ret_rms.std
        for rec in env_records:
            rec["advs"] = torch.tensor((rec["advs"] - adv_mean) / adv_std, dtype=torch.float32, device=self.device)
            rec["rets"] = torch.tensor((rec["rets"] - ret_mean) / ret_std, dtype=torch.float32, device=self.device)

        if not cfg.use_lstm:
            return self._ppo_update_feedforward(env_records)
        return self._ppo_update_recurrent(env_records)

    def _ppo_update_recurrent(self, env_records: List[Dict]) -> Dict[str, float]:
        """
        PPO update that preserves per-environment temporal order. Minibatches
        are groups of ENVIRONMENTS (not individual timesteps), since each
        environment's sequence must be replayed through the LSTM in order,
        starting from its own stored initial hidden state.
        """
        cfg = self.cfg
        stats = collections.defaultdict(list)
        envs_per_mb = max(1, min(cfg.envs_per_minibatch, len(env_records)))

        for _ in range(cfg.n_epochs):
            order = list(range(len(env_records)))
            random.shuffle(order)

            for start in range(0, len(order), envs_per_mb):
                mb_idxs = order[start : start + envs_per_mb]

                pi_losses, vf_losses, entropies = [], [], []
                for idx in mb_idxs:
                    rec = env_records[idx]
                    logits_seq, values_seq = run_recurrent_forward(
                        self.net, rec["obs"], rec["seq_dones"], rec["hidden_start"]
                    )
                    lp = rec["learn_positions"]
                    logits_learn = logits_seq[lp]   # (n_learn, action_size)
                    values_learn = values_seq[lp]   # (n_learn,)

                    log_probs, ent = [], []
                    for t, legal_ids in enumerate(rec["legal_ids"]):
                        legal_logits = logits_learn[t, legal_ids]
                        dist = torch.distributions.Categorical(logits=legal_logits)
                        log_probs.append(dist.log_prob(rec["actions"][t]))
                        ent.append(dist.entropy())
                    log_probs_t = torch.stack(log_probs)
                    entropy_t = torch.stack(ent)

                    ratio = torch.exp(log_probs_t - rec["old_log_probs"])
                    surr1 = ratio * rec["advs"]
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * rec["advs"]

                    pi_losses.append(-torch.min(surr1, surr2).mean())
                    vf_losses.append(F.mse_loss(values_learn, rec["rets"]))
                    entropies.append(entropy_t.mean())

                loss_pi = torch.stack(pi_losses).mean()
                loss_vf = torch.stack(vf_losses).mean()
                entropy = torch.stack(entropies).mean()
                loss = loss_pi + cfg.vf_coef * loss_vf - cfg.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"].append(loss_pi.item())
                stats["loss_vf"].append(loss_vf.item())
                stats["entropy"].append(entropy.item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _ppo_update_feedforward(self, env_records: List[Dict]) -> Dict[str, float]:
        """Original flat/shuffled PPO update, valid when there's no recurrence to preserve."""
        cfg = self.cfg
        obs_tensor = torch.cat([r["obs"] for r in env_records], dim=0)
        actions_tensor = torch.cat([r["actions"] for r in env_records], dim=0)
        old_lps_tensor = torch.cat([r["old_log_probs"] for r in env_records], dim=0)
        advs_tensor = torch.cat([r["advs"] for r in env_records], dim=0)
        rets_tensor = torch.cat([r["rets"] for r in env_records], dim=0)
        all_legal_ids: List[List[int]] = []
        for r in env_records:
            all_legal_ids.extend(r["legal_ids"])

        stats = collections.defaultdict(list)
        N = obs_tensor.shape[0]

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

                logits, values, _ = self.net(mb_obs)

                log_probs, entropies = [], []
                for row, legal_ids in enumerate(mb_legal_ids):
                    legal_logits = logits[row, legal_ids]
                    dist = torch.distributions.Categorical(logits=legal_logits)
                    log_probs.append(dist.log_prob(mb_actions[row]))
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
        self.training_history["win_rate_vs_latest_opp"].append(
            float(np.mean(self.winrate_vs_latest_deque)) if self.winrate_vs_latest_deque else float("nan")
        )
        self.training_history["win_rate_vs_pool_opp"].append(
            float(np.mean(self.winrate_vs_pool_deque)) if self.winrate_vs_pool_deque else float("nan")
        )
        self.training_history["episode_reward"].append(float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0)
        self.training_history["episode_length"].append(float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0)
        self.training_history["team_0_wins"].append(float(self.team_0_wins))
        self.training_history["team_1_wins"].append(float(self.team_1_wins))
        self.training_history["team_0_wins_vs_latest"].append(float(self.team_0_wins_vs_latest))
        self.training_history["team_1_wins_vs_latest"].append(float(self.team_1_wins_vs_latest))
        self.training_history["sps"].append(float((self.cfg.n_envs * self.cfg.n_steps) / max(rollout_elapsed, 1e-8)))

    def _save_training_plots(self) -> Path:
        """Write a matplotlib summary of the recorded training metrics."""
        if not self.training_history["rollout"]:
            return self.save_dir / "training_metrics_empty.png"

        steps = np.array(self.training_history["step"], dtype=np.float32)
        eval_steps = np.array(self.training_history.get("eval_step", []), dtype=np.float32)

        fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
        axes = axes.reshape(4, 2)

        rollout_plots = [
            (0, 0, "loss_pi", "Policy Loss"),
            (0, 1, "loss_vf", "Value Loss"),
            (1, 0, "entropy", "Entropy"),
            (1, 1, "win_rate", "Win Rate (self-play)"),
            (2, 0, "episode_reward", "Episode Reward"),
            (2, 1, "episode_length", "Episode Length"),
        ]

        for row, col, key, title in rollout_plots:
            ax = axes[row, col]
            values = np.array(self.training_history.get(key, []), dtype=np.float32)
            if len(values) > 0:
                ax.plot(steps[:len(values)], values, label=title, color="#1f77b4", linewidth=1.8)
                if len(values) >= 5 and key in {"loss_pi", "loss_vf", "entropy", "episode_reward", "episode_length"}:
                    window = min(10, len(values))
                    smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                    smooth_steps = steps[window - 1 : window - 1 + len(smooth)]
                    ax.plot(smooth_steps, smooth, color="#ff7f0e", linewidth=2.0, label=f"{window}-roll mean")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        
        # overlay opponent-staleness split on the self-play win-rate panel
        ax_wr = axes[1, 1]
        latest_vals = np.array(self.training_history.get("win_rate_vs_latest_opp", []), dtype=np.float32)
        pool_vals = np.array(self.training_history.get("win_rate_vs_pool_opp", []), dtype=np.float32)
        if len(latest_vals) > 0:
            ax_wr.plot(steps[:len(latest_vals)], latest_vals, color="#2ca02c", linewidth=1.5,
                    label="vs latest ckpt", alpha=0.85)
        if len(pool_vals) > 0:
            ax_wr.plot(steps[:len(pool_vals)], pool_vals, color="#9467bd", linewidth=1.5,
                    label="vs older pool ckpt", alpha=0.85)
        ax_wr.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_wr.legend(loc="best", fontsize=8)

        # Held-out eval curves: sparser cadence, own x-axis (eval_step), plotted
        # with markers since there are far fewer points than the rollout metrics.
        eval_plots = [
            (3, 0, "win_rate_vs_heuristic", "Win Rate vs Heuristic Bot", "#2ca02c"),
            (3, 1, "win_rate_vs_random", "Win Rate vs Random Bot", "#d62728"),
        ]
        for row, col, key, title, color in eval_plots:
            ax = axes[row, col]
            values = np.array(self.training_history.get(key, []), dtype=np.float32)
            if len(values) > 0 and len(eval_steps) > 0:
                n = min(len(values), len(eval_steps))
                ax.plot(eval_steps[:n], values[:n], marker="o", color=color, linewidth=1.8, label=title)
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="50% baseline")
                ax.set_ylim(-0.05, 1.05)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        axes[3, 0].set_xlabel("Training step")
        axes[3, 1].set_xlabel("Training step")
        axes[1, 0].set_ylabel("Metric value")
        axes[2, 0].set_ylabel("Metric value")
        axes[3, 0].set_ylabel("Win rate")

        fig.suptitle(
            f"Cuarenta PPO Training Metrics (steps={int(self.global_step)})",
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

    def _save_policy_snapshot(self) -> Path:
        path = self.save_dir / f"policy_{self.global_step}.pt"

        model_state = clone_state_dict_to_cpu(self.net.state_dict())
        state = {
            "step": self.global_step,
            "model_state": model_state,
            "optimizer_state": self.opt.state_dict(),
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "use_lstm": self.cfg.use_lstm,
            "lstm_hidden": self.cfg.lstm_hidden,
            "trunk_hidden": self.cfg.trunk_hidden,
            "value_hidden": self.cfg.value_hidden,
            "ret_rms": self.ret_rms.state_dict(),
        }
        torch.save(state, path)
        print(f"[PPO] Checkpoint → {path}")

        self.ckpt_pool.append(model_state)
        self.ckpt_step.append(self.global_step)

        if len(self.ckpt_pool) > self.cfg.pool_size:
            self.ckpt_pool.pop(0)
            self.ckpt_step.pop(0)

        return path

    
    def _sample_opponent_for_env(self, env_idx: int):
        """Assign this env a (possibly new) opponent snapshot for its next episode.
        Called at episode reset, not at checkpoint time, so each env can be facing
        a different, independently-refreshed opponent every ~100-250 steps instead
        of all envs sharing one frozen snapshot for an entire 100k-step window."""
        cfg = self.cfg
        if not self.ckpt_pool:
            return  # still using initial (self) weights, nothing to sample yet

        if random.random() < cfg.opponent_latest:
            sd = self.ckpt_pool[-1]
            self.env_opponent_is_latest[env_idx] = True
        else:
            sd = random.choice(self.ckpt_pool)
            self.env_opponent_is_latest[env_idx] = False

        self.opponent_nets[env_idx].load_state_dict(sd)
        self.opponent_nets[env_idx].eval()
        if cfg.use_lstm:
            self.opponent_lstm_states[env_idx] = None


    def load_policy(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state" in state:
            self.net.load_state_dict(state["model_state"])
            if "optimizer_state" in state:
                self.opt.load_state_dict(state["optimizer_state"])
        elif "policy_state" in state:
            # Legacy / pretrained checkpoint (e.g. DAgger): feedforward trunk only.
            self.net.policy.load_state_dict(state["policy_state"])
        else:
            raise KeyError(f"Unsupported checkpoint format: {path}")

        if "ret_rms" in state:
            self.ret_rms.load_state_dict(state["ret_rms"])

        self.global_step = int(state.get("step", 0))

        for net in self.opponent_nets:
            net.load_state_dict(self.net.state_dict())
            net.eval()
        if self.cfg.use_lstm:
            self.lstm_states = [None] * self.cfg.n_envs
            self.opponent_lstm_states = [None] * self.cfg.n_envs
        print(f"[PPO] Loaded policy from {path} (step={self.global_step})")


if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() == 'true':
            return True
        elif v.lower() == 'false':
            return False
        else:
            raise argparse.ArgumentTypeError('true or false expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="output/policies")
    parser.add_argument("--init-policy", type=str, default=DEFAULT_POLICY_INIT)
    parser.add_argument("--caida-exposure-penalty", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--opponent-mode",
        type=str,
        default="selfplay",
        choices=["heuristic", "selfplay", "random"],
    )
    parser.add_argument("--use-lstm", type=str2bool, nargs='?', const=True, default=False, help="Use LSTM in the architecture")
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--trunk-hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--envs-per-minibatch", type=int, default=4)
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
        use_lstm=args.use_lstm,
        lstm_hidden=args.lstm_hidden,
        trunk_hidden=args.trunk_hidden,
        value_hidden=args.value_hidden,
        envs_per_minibatch=args.envs_per_minibatch,
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