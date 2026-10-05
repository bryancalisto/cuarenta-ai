"""
Standalone heuristic agent for Cuarenta.

Rules, in priority order:
  1. If a caída is available (the capture takes the card that was placed last, and
     it has the same value as the card we play) -> take it (worth 2 points).
  2. If a limpia is available (the capture clears the whole table) -> take it
     (worth 2 points).
  3. If any captures are available -> take the one capturing the most cards
     (captures themselves are worth 0 points, but cards feed the cartón/majority
     bonus at the end of the deck, so more cards is still better).
  4. Otherwise -> place a random card.

The agent is deterministic given a seed (uses its own random.Random).

IMPORTANT, and this is what made it a poor teacher before: `captures.score_play`
returns `reasons` as a LIST, so tests must be `has_reason(reasons, ...)` or
`reason in reasons`. The old code wrote `r == PointsReasons.CAIDA`, which is
`list == str` and therefore ALWAYS False. As a result this agent never played for
a caída or a limpia at all: it fell through to "largest capture by card count",
which systematically prefers a bigger 0-point capture over a 1-card caída worth 2.
Since this agent generates the imitation-learning data for CuarentaNet's bootstrap
and is also the opponent used for evaluation, that bug poisoned both.

Deliberately NOT modelled here (leave for a later iteration, they are what a strong
Cuarenta player actually does):
  - not handing the opponent a caída (i.e. avoiding placements whose card value
    equals the last placed card),
  - counting cartón arithmetic to decide between two equal immediate captures,
  - keeping a "safe" card to answer the opponent's next move.
A stronger teacher = a stronger starting policy for self-play, so these are good
next candidates once the pipeline itself is trustworthy.
"""

from __future__ import annotations

import random
from typing import Optional

from env import Action
from captures import score_play
from utils import PointsReasons, has_reason
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
          - Otherwise: place a random card
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

        # 1. Caída. Several caídas can be available in one position, so break the
        #    tie by how many cards the capture takes (all caídas pay the same 2).
        caidas = [(a, len(a.capture)) for a, _, r in scored if has_reason(r, PointsReasons.CAIDA)]
        if caidas:
            max_n = max(n for _, n in caidas)
            return self.rng.choice([a for a, n in caidas if n == max_n])

        # 2. Limpia: clears the table (also worth 2). Same tie-break by capture size.
        limpias = [(a, len(a.capture)) for a, _, r in scored if has_reason(r, PointsReasons.LIMPIA)]
        if limpias:
            max_n = max(n for _, n in limpias)
            return self.rng.choice([a for a, n in limpias if n == max_n])

        # 3. Largest capture by card count.
        captures = [(a, len(a.capture)) for a, _, _ in scored if a.capture is not None]
        if captures:
            max_n = max(n for _, n in captures)
            best = [a for a, n in captures if n == max_n]
            return self.rng.choice(best)

        # 4. Random placement.
        places = [a for a, _, _ in scored if a.capture is None]
        if places:
            return self.rng.choice(places)

        # Fallback: shouldn't happen
        return self.rng.choice(legal)
