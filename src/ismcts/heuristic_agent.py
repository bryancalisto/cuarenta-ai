"""
Standalone heuristic agent for Cuarenta.

Rules, in priority order:
  1. If a caída is available (capture set == {last_placed}) -> take it.
  2. If a limpia is available (capture would clear the table) -> take it.
  3. If any captures are available -> take the one capturing the most cards.
  4. Otherwise -> place a random card.

The agent is deterministic given a seed (uses its own random.Random).
"""

from __future__ import annotations

import random
from typing import Optional

from env import Action
from captures import score_play
from utils import PointsReasons
from env import MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS, TEAMS


class HeuristicAgent:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, env, player: int) -> Action:
        """
        Choose an action for `player` given the env's current state.

        Prefers:
          - Caída (2 pts + captures the last-placed card)
          - Limpia (2 pts + clears the table)
          - Largest capture by card count
          - Otherwise: place a random card.
        """
        legal = env.legal_actions()
        team = TEAMS[player]
        state = env.state

        # Evaluate every action's points/reason via the same score_play the env uses.
        scored = []
        for a in legal:
            pts, reason = score_play(
                played=a.card,
                capture_set=a.capture,
                table=state.table,
                last_placed=state.last_placed,
                non_caida_allowed=state.scores[team] < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
            )
            scored.append((a, pts, reason))

        # 1. Caída
        caidas = [a for a, _, r in scored if r == PointsReasons.CAIDA]
        if caidas:
            return caidas[0]

        # 2. Limpia
        limpias = [a for a, _, r in scored if r == PointsReasons.LIMPIA]
        if limpias:
            return limpias[0]

        # 3. Largest capture by card count
        captures = [(a, len(a.capture)) for a, _, _ in scored if a.capture is not None]
        if captures:
            max_n = max(n for _, n in captures)
            best = [a for a, n in captures if n == max_n]
            return self.rng.choice(best)

        # 4. Random placement
        places = [a for a, _, _ in scored if a.capture is None]
        if places:
            return self.rng.choice(places)

        # Fallback: shouldn't happen
        return self.rng.choice(legal)