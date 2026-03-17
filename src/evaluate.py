"""
evaluate.py — Evaluate a trained CuarentaNet checkpoint.

Modes:
  --mode elo        : compute Elo vs a pool of checkpoints
  --mode vs-random  : win-rate vs random agent
  --mode vs-heuristic : win-rate vs rule-based heuristic agent
  --mode watch      : print a single game to stdout
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

import numpy as np
import torch

from model import CuarentaNet, ActionMasks
from env import CuarentaEnv, Action, TEAMS, NUM_PLAYERS
from ppo import obs_to_tensors


# ─── Agents ──────────────────────────────────────────────────────────────────

class RandomAgent:
    """Plays a uniformly random legal action."""
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal: List[Action]) -> Action:
        return self.rng.choice(legal)


class HeuristicAgent:
    """
    Simple rule-based agent:
      1. If a caída is available → take it.
      2. If a ronda is available → take it.
      3. If any capture → take the one capturing the most cards.
      4. Otherwise place the lowest-value card.
    """

    def act(self, obs, legal: List[Action]) -> Action:
        captures = [a for a in legal if a.capture]

        if captures:
            # Check for caída
            if obs.get("last_placed") is not None:
                lp_vec = obs["last_placed"]
                lp_idx = int(lp_vec.argmax()) if lp_vec.sum() > 0 else -1
                for a in captures:
                    if a.capture:
                        cap_indices = {c.index for c in a.capture}
                        if lp_idx in cap_indices and a.card.index == lp_idx:
                            return a  # caída

            # Prefer ronda (capture all table cards)
            best_cap = max(captures, key=lambda a: len(a.capture))
            return best_cap

        # No capture: play lowest-value card
        places = [a for a in legal if a.capture is None]
        return min(places, key=lambda a: a.card.value)


class NetAgent:
    """Wraps a CuarentaNet for evaluation."""

    def __init__(self, net: CuarentaNet, device: torch.device, deterministic: bool = True):
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.net.eval()

    @torch.no_grad()
    def act(self, obs, legal: List[Action]) -> Action:
        obs_t = obs_to_tensors([obs], self.device)
        masks = ActionMasks.from_env_actions([legal])
        action_idx, _, _ = self.net.act(obs_t, masks, deterministic=self.deterministic)
        return legal[action_idx]


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


# ─── Elo ──────────────────────────────────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float, score_a: float, k: float = 32.0):
    """score_a: 1=win, 0.5=draw, 0=loss."""
    exp_a = expected_score(elo_a, elo_b)
    new_a = elo_a + k * (score_a - exp_a)
    new_b = elo_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_net(path: str, device: torch.device) -> CuarentaNet:
    net = CuarentaNet().to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state["net"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--mode",     type=str, default="vs-random",
                        choices=["vs-random", "vs-heuristic", "watch", "elo"])
    parser.add_argument("--n-games",  type=int, default=200)
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--elo-pool", type=str, nargs="*", default=[],
                        help="Other checkpoints for elo mode")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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

    elif args.mode == "elo":
        if not args.elo_pool:
            print("Provide --elo-pool <ckpt1.pt> <ckpt2.pt> ...")
            sys.exit(1)

        elo_scores = {"target": 1000.0}
        agents     = {"target": agent}

        for path in args.elo_pool:
            key = path
            n   = load_net(path, device)
            agents[key]     = NetAgent(n, device)
            elo_scores[key] = 1000.0

        print(f"\n── Elo Tournament ({args.n_games} games per pair) ──────")
        keys = list(agents.keys())
        for i, ka in enumerate(keys):
            for kb in keys[i+1:]:
                results = run_tournament(agents[ka], agents[kb], n_games=args.n_games)
                score_a = results["wins_a"] + 0.5 * results["draws"]
                for _ in range(results["wins_a"]):
                    elo_scores[ka], elo_scores[kb] = update_elo(elo_scores[ka], elo_scores[kb], 1.0)
                for _ in range(results["wins_b"]):
                    elo_scores[ka], elo_scores[kb] = update_elo(elo_scores[ka], elo_scores[kb], 0.0)

        print("\nFinal Elo ratings:")
        for k, v in sorted(elo_scores.items(), key=lambda x: -x[1]):
            print(f"  {k:40s} : {v:.1f}")


if __name__ == "__main__":
    main()