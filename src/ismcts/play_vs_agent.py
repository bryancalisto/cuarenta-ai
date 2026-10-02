"""
Play Cuarenta against a trained NetworkISMCTS agent, using the Tkinter UI.

Usage:
    # Play as P0 (your turn at the top), agent plays the other 3 seats:
    uv run play_vs_agent.py --checkpoint checkpoints/cuarenta_iter012.pt --human-players 0

    # Play as P0 and P2 (the whole team 0):
    uv run play_vs_agent.py --checkpoint checkpoints/cuarenta_iter012.pt --human-players 0,2

    # Watch the agent play against itself:
    uv run play_vs_agent.py --checkpoint checkpoints/cuarenta_iter012.pt --human-players ""

    # Show the agent's hand (debug):
    uv run play_vs_agent.py --checkpoint checkpoints/cuarenta_iter012.pt --human-players 0 --show-agent-hand
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import tkinter as tk
from tkinter import messagebox

import torch

# sys.path.insert(0, str(Path(__file__).parent.parent))

from env import CuarentaEnv, Action
from history import HistoryTracker
from network import CuarentaNet
from network_ismcts_1 import NetworkISMCTS   # <- swap to _v3 if you switch later
from ui import CuarentaUI


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_human_players(value: str) -> List[int]:
    """Parse comma-separated player indices. Empty string -> no humans."""
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

class Agent:
    """
    Wraps a CuarentaNet + NetworkISMCTS search into a simple `act(env, tracker)`
    interface. Returns the actual Action object to play.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        n_iter: int = 200,
        max_rollout_len: int = 200,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.device = device
        self.net = CuarentaNet()
        state = torch.load(checkpoint, map_location="cpu")
        # Handle both raw state_dicts and wrapped dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.net.load_state_dict(state)
        self.net.to(device)
        self.net.eval()

        self.n_iter = n_iter
        self.max_rollout_len = max_rollout_len
        self.rng = random.Random(seed)

    def act(self, env: CuarentaEnv, tracker: HistoryTracker) -> Action:
        """Run one search from the current player's POV and return the best Action."""
        # Build a fresh searcher bound to the current env/tracker.
        # (We could cache it, but NetworkISMCTS reads env.state each call,
        # so a fresh instance is clean and cheap.)
        searcher = NetworkISMCTS(
            env,
            self.net,
            tracker,
            n_iter=self.n_iter,
            max_rollout_len=self.max_rollout_len,
            rng=self.rng,
            dirichlet_eps=0.0,   # no exploration noise during interactive play
            device=self.device,
        )
        visits = searcher.search()

        legal = env.legal_actions()
        if sum(visits.values()) == 0:
            # No search signal; play random (shouldn't happen with n_iter > 0)
            return legal[self.rng.randrange(len(legal))]

        best_slot = max(visits, key=visits.get)
        return legal[best_slot]


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def run_game(
    checkpoint: str,
    human_players: Set[int],
    device: Optional[str] = None,
    seed: Optional[int] = None,
    show_agent_hand: bool = False,
    delay: float = 1.0,
    n_iter: int = 200,
):
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    root = tk.Tk()
    ui = CuarentaUI(
        root, env,
        human_players=sorted(human_players),
        show_agent_hand=show_agent_hand,
    )
    root.title("Cuarenta vs Agent")

    torch_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint on {torch_device}: {checkpoint}")
    agent = Agent(checkpoint, n_iter=n_iter, device=torch_device, seed=seed)

    tracker = HistoryTracker(env)

    try:
        while not env.state.done:
            current_player = env.state.current_player
            obs = env.observe(current_player)      # for tracker + UI

            ui.refresh_display()

            if current_player in human_players:
                # Human turn: wait for click
                ui.action_label.config(text=f"Your turn: P{current_player}")
                action = ui.wait_for_action()
            else:
                # Agent turn
                ui.action_label.config(text=f"AI thinking: P{current_player}")
                root.update_idletasks()
                root.update()

                t0 = time.perf_counter()
                action = agent.act(env, tracker)
                elapsed = time.perf_counter() - t0
                print(f"P{current_player} played {action} in {elapsed*1000:.0f} ms")

                ui.record_action(action)
                ui.refresh_display()
                root.update_idletasks()
                root.update()

            # IMPORTANT: record BEFORE stepping.
            # The tracker must reflect what the player saw and chose.
            # For the agent, its internal search already used the tracker as-is
            # (which is fine — the tracker didn't yet include this decision).
            # Recording here keeps things consistent for both humans and agents.
            tracker.record(current_player, obs, env.legal_actions().index(action))

            _, reward, done, truncated, info = env.step(env.legal_actions().index(action))
            ui.show_thrown_card(action.card)
            ui.refresh_display()
            root.update_idletasks()
            root.update()

            if delay > 0:
                time.sleep(delay)

        # Game over
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
    parser = argparse.ArgumentParser(description="Play Cuarenta against a trained NetworkISMCTS agent")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a CuarentaNet .pt checkpoint (e.g. checkpoints/cuarenta_iter012.pt)",
    )
    parser.add_argument(
        "--human-players", type=str, default="0",
        help='Comma-separated player indices controlled by the human. E.g. "0" or "0,2". '
             'Empty string means the agent plays all four seats.',
    )
    parser.add_argument("--show-agent-hand", action="store_true",
                        help="Show the agent's hand for debugging.")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | mps | cuda. Defaults to mps if available.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between moves (visual pacing).")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="ISMCTS simulations per move. 200 is strong; 100 is fine.")
    args = parser.parse_args()

    run_game(
        checkpoint=args.checkpoint,
        human_players=set(parse_human_players(args.human_players)),
        show_agent_hand=args.show_agent_hand,
        device=args.device,
        seed=args.seed,
        delay=args.delay,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()