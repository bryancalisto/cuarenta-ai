"""
DAgger-based imitation learning for Cuarenta

Collects expert trajectories via UI, trains policy with DAgger.
Policy is saved for later RL finetuning.
"""

import sys
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from datetime import datetime

from networks import PolicyNetwork

sys.path.insert(0, str(Path(__file__).parent.parent))

from env import ActionEncoder, CuarentaEnv, Action


class ExpertPolicy:
    """Wraps UI to provide expert demonstrations."""

    def __init__(self, ui_callback: Callable[[], Action]):
        """ui_callback: function that returns next Action from UI."""
        self.ui_callback = ui_callback

    def get_action(self) -> Action:
        """Get expert action from UI."""
        return self.ui_callback()


class CuarentaPolicyLearner:
    """DAgger learner for Cuarenta policy."""

    def __init__(
        self,
        env: CuarentaEnv,
        expert_policy: ExpertPolicy,
        policy_net: Optional[PolicyNetwork] = None,
        device: str = "cpu",
    ):
        self.env = env
        self.expert_policy = expert_policy
        self.device = torch.device(device)

        # Initialize policy
        if policy_net is None:
            self.policy = PolicyNetwork().to(self.device)
        else:
            self.policy = policy_net.to(self.device)

        self.action_encoder = ActionEncoder(env)

        # Training state
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        # Collected data
        self.trajectory_buffer = []  # List of (obs_dict, action)

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """Flatten observation dict to single tensor."""
        # Concatenate all obs arrays in order
        keys = sorted(obs_dict.keys())
        vectors = [obs_dict[k].flatten() for k in keys]
        combined = np.concatenate(vectors)
        return torch.FloatTensor(combined).to(self.device)

    # default to 60 steps per trajectory because that's 3 full rounds of play (4 players, 5 cards each, for 3 rounds). Game will often end before that, but this gives enough room to capture full games and some extra steps for longer games.
    def collect_trajectory(self, num_steps: int = 60) -> float:
        """
        Collect one trajectory of expert demonstrations via UI.
        Returns average reward.
        """
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0
        print("Collecting trajectory with expert...")

        while steps < num_steps:
            # Get expert action
            print("Getting expert action...")
            action = self.expert_policy.get_action()
            obs, reward, done, _ = self.env.step(action)
            print(f'ACTION EXECUTED 1: {action}, REWARD: {reward}, DONE: {done}')

            # Store trajectory
            action_idx = self.action_encoder.action_to_idx(action)
            self.trajectory_buffer.append((obs.copy(), action_idx))

            total_reward += reward
            steps += 1

            if done:
                break

        return total_reward / max(steps, 1)

    def dagger_iteration(self, num_trajectories: int = 5, batch_size: int = 32, epochs: int = 5) -> float:
        """
        One iteration of DAgger:
        1. Collect trajectories with expert
        2. Train policy on all collected data
        Returns average loss.
        """
        print(f"Collecting {num_trajectories} trajectories...")
        for i in range(num_trajectories):
            avg_reward = self.collect_trajectory()
            print(f"  Trajectory {i+1}: avg_reward={avg_reward:.2f}, buffer_size={len(self.trajectory_buffer)}")

        # Prepare data
        print(f"Training on {len(self.trajectory_buffer)} samples...")
        obs_list = []
        action_list = []

        for obs_dict, action_idx in self.trajectory_buffer:
            obs_tensor = self._obs_to_tensor(obs_dict)
            obs_list.append(obs_tensor)
            action_list.append(action_idx)

        if not obs_list:
            print("No data collected!")
            return 0.0

        obs_tensor = torch.stack(obs_list)
        action_tensor = torch.LongTensor(action_list).to(self.device)

        dataset = TensorDataset(obs_tensor, action_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for obs_batch, action_batch in loader:
                self.optimizer.zero_grad()
                logits = self.policy(obs_batch)
                loss = self.criterion(logits, action_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss

        return total_loss / epochs

    def save_policy(self, path: str):
        """Save policy weights and metadata."""
        save_data = {
            "policy_state": self.policy.state_dict(),
            "action_encoder": self.action_encoder,
        }
        torch.save(save_data, path)
        print(f"Policy saved to {path}")

    def save_trajectories(self, output_dir: str = "data"):
        """Save collected trajectories to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_file = output_path / f"trajectories_{timestamp}.pkl"

        with open(traj_file, "wb") as f:
            pickle.dump(self.trajectory_buffer, f)

        print(f"Saved {len(self.trajectory_buffer)} trajectory steps to {traj_file}")

    def load_policy(self, path: str):
        """Load policy weights and metadata."""
        save_data = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(save_data["policy_state"])
        if "action_encoder" in save_data:
            self.action_encoder = save_data["action_encoder"]
        print(f"Policy loaded from {path}")

    def predict_action(self, obs_dict: dict) -> Optional[Action]:
        """
        Use policy to predict best action given observation.
        Returns Action or None if no legal action found.
        """
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs_dict).unsqueeze(0)  # (1, obs_size)
            logits = self.policy(obs_tensor)  # (1, action_size)
            probs = torch.softmax(logits, dim=1)[0]  # (action_size,)

            # Get legal actions
            legal_actions = self.env.legal_actions()
            legal_indices = [self.action_encoder.action_to_idx(a) for a in legal_actions]

            # Find best legal action
            best_idx = None
            best_prob = -1.0
            for idx in legal_indices:
                if idx < len(probs) and probs[idx] > best_prob:
                    best_prob = probs[idx]
                    best_idx = idx

            if best_idx is not None:
                action = self.action_encoder.idx_to_action(best_idx)
                return action

        return None


def create_dagger_trainer(
    env: CuarentaEnv,
    ui_callback: Callable[[], Action],
    device: str = "cpu",
) -> CuarentaPolicyLearner:
    """Factory function to create DAgger trainer with expert policy."""
    # Compute obs_size from actual observation
    obs = env.observe(0)
    keys = sorted(obs.keys())
    vectors = [obs[k].flatten() for k in keys]
    obs_size = sum(len(v) for v in vectors)

    # Action encoding: card_idx (0-39) * 41 + capture_idx (0-40) = 0-1639
    action_size = 40 * 41  # 1640

    expert = ExpertPolicy(ui_callback)
    policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)
    return CuarentaPolicyLearner(env, expert, policy_net=policy, device=device)
