"""
Play Cuarenta against a trained policy using the existing Tkinter UI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import tkinter as tk
from tkinter import messagebox

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from card import build_deck
from env import CuarentaEnv
from agents import load_agent_from_checkpoint
from ui import CuarentaUI
from env import Action


def parse_human_players(value: str) -> List[int]:
    """
    value format:
    """
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def decode_action_index(action_idx: int, legal: List[Action]) -> Action:
    """Mirror the 40x41 action encoding used by the PPO checkpoint."""
    action_idx = int(np.asarray(action_idx).item())
    card_idx = action_idx // 41
    capture_idx = action_idx % 41
    card = build_deck()[card_idx]

    matching = [a for a in legal if a.card == card]
    if not matching:
        return legal[action_idx % len(legal)]

    if capture_idx == 0:
        for legal_action in matching:
            if legal_action.capture is None:
                return legal_action
        return legal[action_idx % len(legal)]

    capture_actions = [a for a in matching if a.capture is not None]
    capture_offset = capture_idx - 1
    if capture_offset < len(capture_actions):
        return capture_actions[capture_offset]

    return legal[action_idx % len(legal)]


def run_game(
    checkpoint: str,
    human_players: Set[int],
    device: str | None = None,
    seed: int | None = None,
    show_agent_hand: bool = False,
    delay: float = 1.0,
):
    env = CuarentaEnv(seed=seed)
    env.reset()

    root = tk.Tk()
    ui = CuarentaUI(root, env, human_players=sorted(human_players), show_agent_hand=show_agent_hand)
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
                selected_action = ui.wait_for_action()
                action_for_env = selected_action
            else:
                ui.action_label.config(text=f"AI thinking: P{current_player}")
                root.update_idletasks()
                root.update()
                chosen = agent.act(env.observe(current_player), legal)
                if isinstance(chosen, Action):
                    selected_action = chosen
                    action_for_env = chosen
                else:
                    selected_action = decode_action_index(chosen, legal)
                    action_for_env = selected_action
                ui.record_action(selected_action)

            _, reward, done, info = env.step(action_for_env)
            ui.show_thrown_card(selected_action.card)
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
        help="Path to a Torch checkpoint (.pt) or Stable-Baselines3 PPO archive (.zip)",
    )

    parser.add_argument(
        "--show-agent-hand",
        type=bool,
        default=False,
        help="Shows hand of agent (usually used for debugging)",
    )

    parser.add_argument(
        "--human-players",
        type=str,
        default="0",
        help="Comma-separated player indexes (in play order, starting at the top one) controlled by the human, e.g. 0 or 0,2",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    run_game(
        checkpoint=args.checkpoint,
        human_players=set(parse_human_players(args.human_players)),
        show_agent_hand=args.show_agent_hand,
        device=args.device,
        seed=args.seed,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
