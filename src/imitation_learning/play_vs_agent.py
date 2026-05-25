"""
Play Cuarenta against a trained policy using the existing Tk UI.

This reuses the imitation-learning UI for rendering and human input, while an
agent loaded from a checkpoint plays the non-human seats.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set

import tkinter as tk
from tkinter import messagebox

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from env import CuarentaEnv
from agents import load_agent_from_checkpoint
from imitation_learning.ui import CuarentaUI


def parse_human_players(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run_game(
    checkpoint: str,
    human_players: Set[int],
    device: str | None = None,
    seed: int | None = None,
    delay: float = 1.0,
):
    env = CuarentaEnv(seed=seed)
    env.reset()

    root = tk.Tk()
    ui = CuarentaUI(root, env, human_players=sorted(human_players))
    root.title("Cuarenta vs Agent")
    torch_device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    agent = load_agent_from_checkpoint(checkpoint, torch_device, deterministic=True)

    try:
        while not env.state.done:
            current_player = env.state.current_player
            legal = env.legal_actions()

            ui.refresh_display()
            if current_player in human_players:
                ui.action_label.config(text=f"Your turn: P{current_player}")
                action = ui.wait_for_action()
            else:
                ui.action_label.config(text=f"AI thinking: P{current_player}")
                root.update_idletasks()
                root.update()
                action = agent.act(env.observe(current_player), legal)
                ui.record_action(action)

            _, reward, done, info = env.step(action)
            ui.show_thrown_card(action.card)
            ui.refresh_display()
            root.update_idletasks()
            root.update()

            if delay > 0:
                time.sleep(delay)

        ui.refresh_display()
        winner = env.state.winner
        team0 = env.state.scores[0]
        team1 = env.state.scores[1]
        messagebox.showinfo(
            "Game Over",
            f"Winner: Team {winner}\nFinal score: Team 0 {team0} - Team 1 {team1}",
        )
        root.update_idletasks()
        root.update()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Play Cuarenta against a trained policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a PPO policy snapshot or PPO checkpoint",
    )
    parser.add_argument(
        "--human-players",
        type=str,
        default="0",
        help="Comma-separated player ids controlled by the human, e.g. 0 or 0,2",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    run_game(
        checkpoint=args.checkpoint,
        human_players=set(parse_human_players(args.human_players)),
        device=args.device,
        seed=args.seed,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
