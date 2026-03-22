"""
captures.py — Legal capture logic for Cuarenta.

Capture rules:
  1. MATCH: Play a card that matches exactly one card on the table (same rank value).
  2. SUM:   Play a card whose value equals the SUM of 2+ table cards.
  3. CAÍDA: If the played card value matches the LAST card placed AND you capture it,
            it's a caída (scores 2 pts). A caída can chain if the card before that
            also matched, etc.
  4. RONDA: If you capture ALL remaining table cards → scores 2 pts (mesa limpia).

Priority rule: If a match OR sum capture is possible, the player MUST capture
(cannot voluntarily leave cards). However, the player CHOOSES *which* valid capture
to take if multiple exist.

NOTE on figure captures (Jack/Caballo/Rey = 11/12/13):
  - Figures can only capture by EXACT MATCH (same rank), NOT by sum.
  - A number card can sum to 11, 12, or 13 and capture a figure.
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Set, FrozenSet, Tuple
from card import Card, Rank

# Ranks that can only be captured by exact match (not as part of a sum from numbers)
FIGURE_RANKS = {Rank.JACK, Rank.QUEEN, Rank.KING}


def _is_figure(card: Card) -> bool:
    return card.rank in FIGURE_RANKS


def valid_captures(
    played: Card,
    table: List[Card],
) -> List[FrozenSet[Card]]:
    """
    Return all distinct legal capture sets for `played` card against `table`.

    Each capture set is a frozenset of table cards that `played` can take.
    An empty list means NO capture is possible (player must just place the card).

    Rules enforced:
      - Figures on table can only be captured by exact rank match.
      - A played figure can only capture by exact rank match (not sum).
      - A played number card can match exactly OR sum multiple table cards.
      - A single played card may ONLY take one capture set per turn
        (player picks one option from the returned list).
    """
    if not table:
        return []

    captures: List[FrozenSet[Card]] = []
    played_value = played.value
    played_is_figure = _is_figure(played)

    # --- Exact match captures ---
    # A card can always capture table cards of the same rank (1-for-1).
    matches = [c for c in table if c.rank == played.rank]
    for m in matches:
        captures.append(frozenset([m]))

    # --- Sum captures (only for non-figure played cards) ---
    if not played_is_figure:
        # Separate table into figures (cannot be part of a sum) and numbers
        number_table = [c for c in table if not _is_figure(c)]
        figure_matches = [c for c in table if _is_figure(c) and c.rank == played.rank]

        # Try all subsets of size ≥ 2 from number cards
        for size in range(2, len(number_table) + 1):
            for subset in combinations(number_table, size):
                if sum(c.value for c in subset) == played_value:
                    cap = frozenset(subset)
                    # Deduplicate
                    if cap not in captures:
                        captures.append(cap)

        # A sum capture can also include a figure IF the figure has same rank as played
        # (e.g., playing a 10 can capture: [figure_10] already handled above,
        #  or [num_3, num_7], etc.)
        # Edge case: can a number sum capture include a figure? No — figures are excluded
        # from sums in standard Cuarenta rules.

    # Remove duplicate sets (shouldn't happen but be safe)
    seen: Set[FrozenSet[Card]] = set()
    unique: List[FrozenSet[Card]] = []
    for cap in captures:
        if cap not in seen:
            seen.add(cap)
            unique.append(cap)

    return unique


def must_capture(played: Card, table: List[Card]) -> bool:
    """Return True if the player is OBLIGATED to capture (any legal capture exists)."""
    return len(valid_captures(played, table)) > 0


def detect_caida(
    played: Card,
    capture_set: FrozenSet[Card],
    table: List[Card],
    last_placed: Card | None,
) -> bool:
    """
    Detect if this capture is a CAÍDA (cascade match).

    A caída occurs when:
      - The played card captures a table card
      - AND the LAST card placed on the table (by any player) has the same value
        as the played card AND is being captured in this move.

    In practice: last_placed must be in capture_set and
    last_placed.value == played.value.
    """
    if last_placed is None:
        return False
    if last_placed not in capture_set:
        return False
    return last_placed.value == played.value


def detect_ronda(
    capture_set: FrozenSet[Card],
    table: List[Card],
) -> bool:
    """
    Detect if this capture clears the entire table (RONDA / mesa limpia).
    """
    return set(capture_set) == set(table)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

CAIDA_POINTS = 2
LIMPIA_POINTS = 2


def score_play(
    played: Card,
    capture_set: FrozenSet[Card] | None,
    table: List[Card],
    last_placed: Card | None,
) -> Tuple[int, List[str]]:
    """
    Compute points earned for a single play.

    Returns (points, reasons) where reasons is a list of human-readable strings.

    Points:
      - +2 for a caída
      - +2 for a ronda (clearing the table)

    Actually in Cuarenta the scoring is:
      - Each CAPTURE of ≥1 card = 1 point per captured card (the pip cards
        themselves are worth 1 pt each towards the 40).
      - Caída = 2 bonus points (on top of the capture points).
      - Ronda = 2 bonus points (on top of capture points).
    """
    points = 0
    reasons = []

    if capture_set:
        # Caída bonus
        if detect_caida(played, capture_set, table, last_placed):
            points += CAIDA_POINTS
            reasons.append("caída (+2)")

        # Limpia bonus
        if detect_ronda(capture_set, table):
            points += LIMPIA_POINTS
            reasons.append("limpia (+2)")

    return points, reasons