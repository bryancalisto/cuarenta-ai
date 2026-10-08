"""
Main training script for DAgger-based imitation learning.

Run as: python train.py [--resume <path>]
"""

import sys
from pathlib import Path
import tkinter as tk
import argparse

# The game core, networks, agents and UI live in src/ppo (shared with the
# PPO pipeline); put that directory on sys.path so the imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ppo"))

from env import CuarentaEnv
from ui import CuarentaUI
from imitation_learning.imitation_learning import create_dagger_trainer


def main():
    """Run DAgger training with UI expert."""
    parser = argparse.ArgumentParser(description="Train Cuarenta policy with DAgger")
    parser.add_argument("--resume", type=str, default=None, help="Path to policy checkpoint to resume from")
    args = parser.parse_args()

    # Initialize environment
    env = CuarentaEnv()
    env.reset()

    # Create Tkinter root
    root = tk.Tk()

    # Create UI
    ui = CuarentaUI(root, env)
    ui.refresh_display()

    # Create expert policy callback
    def expert_callback():
        """Called by trainer to get next expert action."""
        print("Waiting for expert action from UI...")
        return ui.wait_for_action()

    # Create DAgger trainer
    trainer = create_dagger_trainer(env, expert_callback, device="mps")

    # Load checkpoint if provided
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        trainer.load_policy(args.resume)

    # Training loop
    num_dagger_iterations = 5
    trajectories_per_iteration = 1

    for iteration in range(num_dagger_iterations):
        print(f"\n{'='*60}")
        print(f"DAgger Iteration {iteration+1}/{num_dagger_iterations}")
        print(f"{'='*60}")

        loss = trainer.dagger_iteration(
            num_trajectories=trajectories_per_iteration,
            batch_size=16,
            epochs=5,
        )
        print(f"Iteration {iteration+1} complete. Avg loss: {loss:.4f}")

        # Save trajectories after each iteration
        trainer.save_trajectories("data/trajectories")

        # Periodically save
        if (iteration + 1) % 2 == 0:
            save_path = f"data/policies/cuarenta_policy_iter{iteration+1}.pt"
            trainer.save_policy(save_path)

    # Final save
    final_path = "data/policies/cuarenta_policy_final.pt"
    trainer.save_policy(final_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Policy saved to: {final_path}")
    print(f"{'='*60}")

    root.destroy()


if __name__ == "__main__":
    main()
