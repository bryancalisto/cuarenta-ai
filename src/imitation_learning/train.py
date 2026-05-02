"""
Main training script for DAgger-based imitation learning.

Run as: python train.py
"""

import sys
from pathlib import Path
import tkinter as tk

sys.path.insert(0, str(Path(__file__).parent.parent))

from env import CuarentaEnv
from imitation_learning.ui import CuarentaUI
from imitation_learning.imitation_learning import create_dagger_trainer


def main():
    """Run DAgger training with UI expert."""
    # Initialize environment
    env = CuarentaEnv()
    obs, _ = env.reset()

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
    trainer = create_dagger_trainer(env, expert_callback, device="cpu")

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
        trainer.save_trajectories("data")

        # Periodically save
        if (iteration + 1) % 2 == 0:
            save_path = f"/tmp/cuarenta_policy_iter{iteration+1}.pt"
            trainer.save_policy(save_path)

    # Final save
    final_path = "/tmp/cuarenta_policy_final.pt"
    trainer.save_policy(final_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Policy saved to: {final_path}")
    print(f"{'='*60}")

    root.destroy()


if __name__ == "__main__":
    main()
