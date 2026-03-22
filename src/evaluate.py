"""
evaluate.py — Evaluate a trained CuarentaNet checkpoint.

Modes:
  --mode vs-random  : win-rate vs random agent
  --mode vs-heuristic : win-rate vs rule-based heuristic agent
  --mode watch      : print a single game to stdout
"""

from __future__ import annotations

import argparse

import torch

from agents import HeuristicAgent, NetAgent, RandomAgent
from model import CuarentaNet
from env import CuarentaEnv


# ─── Match runner ─────────────────────────────────────────────────────────────

def run_match(
    team0_agents,  # [agent_for_seat0, agent_for_seat2]
    team1_agents,  # [agent_for_seat1, agent_for_seat3]
    seed: int = 0,
    verbose: bool = False,
) -> int:
    """Returns winning team (0 or 1)."""
    agents = {
        0: team0_agents[0],
        1: team1_agents[0],
        2: team0_agents[1],
        3: team1_agents[1],
    }
    env = CuarentaEnv(seed=seed)
    obs, _ = env.reset()

    while True:
        p     = env.state.current_player
        legal = env.legal_actions()
        obs_p = env.observe(p)
        action = agents[p].act(obs_p, legal)
        obs, reward, done, info = env.step(action)

        if verbose:
            print(env.render())

        if done:
            return info.get("winner", -1)


def run_tournament(
    agent_a,
    agent_b,
    n_games: int = 200,
    seed_offset: int = 0,
) -> dict:
    """
    Play n_games between agent_a (team 0) and agent_b (team 1),
    then swap sides for another n_games.

    Returns win rates and counts.
    """
    wins_a = 0
    wins_b = 0

    # A as team 0
    for i in range(n_games // 2):
        winner = run_match(
            [agent_a, agent_a],
            [agent_b, agent_b],
            seed=seed_offset + i,
        )
        if winner == 0:
            wins_a += 1
        elif winner == 1:
            wins_b += 1

    # A as team 1 (swap)
    for i in range(n_games // 2):
        winner = run_match(
            [agent_b, agent_b],
            [agent_a, agent_a],
            seed=seed_offset + n_games // 2 + i,
        )
        if winner == 1:   # A is now team 1
            wins_a += 1
        elif winner == 0:
            wins_b += 1

    total = wins_a + wins_b
    return {
        "wins_a":   wins_a,
        "wins_b":   wins_b,
        "draws":    n_games - total,
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
    }

# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_net(path: str, device: torch.device) -> CuarentaNet:
    net = CuarentaNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state["net"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--mode",     type=str, default="vs-random",
                        choices=["vs-random", "vs-heuristic", "watch"])
    parser.add_argument("--n-games",  type=int, default=200)
    parser.add_argument("--device",   type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("mps" if torch.mps.is_available() else "cpu"))
    net    = load_net(args.checkpoint, device)
    agent  = NetAgent(net, device, deterministic=True)

    if args.mode == "vs-random":
        opp = RandomAgent(seed=999)
        results = run_tournament(agent, opp, n_games=args.n_games)
        print(f"\n── vs Random ({args.n_games} games) ──────────────────")
        print(f"  Net win rate : {results['win_rate_a']:.1%}")
        print(f"  Random win rate : {results['win_rate_b']:.1%}")
        print(f"  Draws        : {results['draws']}")

    elif args.mode == "vs-heuristic":
        opp = HeuristicAgent()
        results = run_tournament(agent, opp, n_games=args.n_games)
        print(f"\n── vs Heuristic ({args.n_games} games) ────────────────")
        print(f"  Net win rate       : {results['win_rate_a']:.1%}")
        print(f"  Heuristic win rate : {results['win_rate_b']:.1%}")
        print(f"  Draws              : {results['draws']}")

    elif args.mode == "watch":
        print("── Watching one game ───────────────────────────────")
        opp = HeuristicAgent()
        winner = run_match([agent, agent], [opp, opp], seed=42, verbose=True)
        print(f"\nWinner: Team {winner}")


if __name__ == "__main__":
    main()