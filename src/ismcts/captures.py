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
from functools import lru_cache
from typing import List, Set, FrozenSet, Tuple
from card import Card, Rank

# Bound: the key space is (card, table) and a long training run sees many deals, so the
# memo must evict instead of growing without limit. 64k entries is a few MB.
CACHE_SIZE = 1 << 16
from utils import sort_cards_by_rank, PointsReasons

# Ranks that can only be captured by exact match (not as part of a sum from numbers)
FIGURE_RANKS = {Rank.JACK, Rank.QUEEN, Rank.KING, Rank.AS}


def _is_figure(card: Card) -> bool:
    return card.rank in FIGURE_RANKS

"""
Captures all same-rank captures for this played card.
"""
def get_same_rank_captures(played: Card, table: List[Card]) -> List[FrozenSet[Card]]:
    return [frozenset([c]) for c in table if c.rank == played.rank]


"""
Captures all the 2-card sums that add up to the played card value.
Example:
    - If you play a 7, you can capture a 4 and 3.
    - If you play a 6, you can capture a 2 and 4 or a 5 and AS.
"""
def get_sum_captures(played: Card, table: List[Card]) -> List[FrozenSet[Card]]:
    captures: List[FrozenSet[Card]] = []
    sorted_table = sort_cards_by_rank(table)
    played_value = played.value
    complement_map = {}

    # Two-sum O(n) algorithm
    for card_i in range(len(sorted_table)):
        card = sorted_table[card_i]
        complement = played_value - card.value

        # Check if the complement value exists in the complement_map
        if complement in complement_map:
            for complement_card in complement_map[complement]:
                captures.append(frozenset([card, complement_card]))

        # Add the current card to the complement_map
        if card.value not in complement_map:
            complement_map[card.value] = []
        complement_map[card.value].append(card)

    return captures


"""
Captures all the played card matches and each's sequence of consecutive cards on the table (stairs).
Example:
    - If you play a 3 corazones rojos, you can capture a 3, 4, 5, 6, J (but not a 2 or a 7) of any suit.
    - If you play a J, you can capture a J, Q, K (but not a 7 or an A).
    - If you play a 7, you can capture a 7, J, Q, K as 8, 9, 10 are not in the game.
"""
def get_stairs_captures(played: Card, table: List[Card]) -> List[FrozenSet[Card]]:
    captures: List[FrozenSet[Card]] = []
    sorted_table = sort_cards_by_rank(table)
    played_rank = played.rank

    for card_i in range(len(sorted_table)):
        if sorted_table[card_i].rank == played_rank:
            # There's a rank match. Take that match and look for a possible sequence in the remaining cards
            matched_card = sorted_table[card_i]
            current_capture = [matched_card]

            for card_i in range(card_i + 1, len(sorted_table)):
                # Skip duplicates
                if sorted_table[card_i].rank == matched_card.rank:
                    continue

                # Check if the remaining card is exactly 1 rank higher than the last captured card or if 7 captures J
                if sorted_table[card_i].rank.value == matched_card.rank.value + 1 or (matched_card.rank == Rank.SIETE and sorted_table[card_i].rank == Rank.JACK):
                    current_capture.append(sorted_table[card_i])
                    matched_card = sorted_table[card_i]
                else:
                    break
            
            if len(current_capture) > 1:
                # We found a valid stairs capture
                captures.append(frozenset(current_capture))
    
    return captures


def _valid_captures_uncached(
    played: Card,
    table: Tuple[Card, ...],
) -> Tuple[FrozenSet[Card], ...]:
    """
    Return all distinct legal capture sets for `played` card against `table`.

    Each capture set is a frozenset of table cards that `played` can take.
    An empty list means NO capture is possible (player must just place the card).
    A single played card may only take one capture set per turn (player picks one option from the returned list).

    Valid capture criteria:
        - A same rank card: A played card can only capture by exact rank match (e.g. capture a 7 corazones with 7 de corazones negros).
        - Sum of cards: A played card can sum multiple table cards. (e.g. capture a 4 trebol and 3 corazones negros with a 7 brillos)
        - Stairs: A played card can capture in a sequence (e.g. capture 3, 4, 5, 6, J - no matter the figure - with a 3).

    """
    if not table:
        return ()

    captures: List[FrozenSet[Card]] = []
    played_is_figure = _is_figure(played)
    sum_captures: List[FrozenSet[Card]] = []

    # Same rank card capture
    same_rank_captures = get_same_rank_captures(played, table)
    captures.extend(same_rank_captures)

    # Sum of cards captures (only for non-figure - numeric - played cards)
    if not played_is_figure:
        sum_captures = get_sum_captures(played, table)
        captures.extend(sum_captures)
    
    # Stairs captures
    stairs_captures = get_stairs_captures(played, table)
    captures.extend(stairs_captures)

    # Chained captures: if the played card can start a stair and there is also
    # a valid sum capture, allow taking both groups in a single play.
    #
    # Example:
    #   table = [2, 3, 5, 6, 7], played = 5
    #   sum capture   -> [2, 3]
    #   stairs capture -> [5, 6, 7]
    #   chained union  -> [2, 3, 5, 6, 7]
    if sum_captures and stairs_captures:
        for sum_cap in sum_captures:
            for stair_cap in stairs_captures:
                captures.append(sum_cap | stair_cap)

    # Remove duplicate sets (shouldn't happen but be safe)
    seen: Set[FrozenSet[Card]] = set()
    unique: List[FrozenSet[Card]] = []
    for cap in captures:
        if cap not in seen:
            seen.add(cap)
            unique.append(cap)

    return tuple(unique)


@lru_cache(maxsize=CACHE_SIZE)
def _valid_captures_cached(played: Card, table: Tuple[Card, ...]) -> Tuple[FrozenSet[Card], ...]:
    return _valid_captures_uncached(played, table)


def valid_captures(played: Card, table: List[Card]) -> Tuple[FrozenSet[Card], ...]:
    """
    All legal capture sets for `played` against `table` (see `_valid_captures_uncached`).

    MEMOISED, because this is the innermost cost of the search: the sum and stairs rules
    walk subsets of the table, and a single ISMCTS decision calls this ~1000 times (once
    per card in every simulated hand, for every simulation). The answer depends ONLY on
    the played card and the cards on the table, and both repeat constantly inside one
    decision -- thousands of calls collapse onto a few dozen distinct pairs.

    The key is the table IN ITS GIVEN ORDER (`tuple(table)`): the returned list's order can
    depend on iteration order, so canonicalising by sorting would change results for
    equal-but-reordered tables. Cache entries are immutable tuples, and callers only
    iterate/sort/index them.
    """
    if not table:
        return ()
    return _valid_captures_cached(played, tuple(table))


def valid_captures_cache_info():
    """(hits, misses, ...) for the memo above -- used by the tests and the profiler."""
    return _valid_captures_cached.cache_info()


def must_capture(played: Card, table: List[Card]) -> bool:
    """Return True if the player is OBLIGATED to capture (any legal capture exists)."""
    return len(valid_captures(played, table)) > 0


def detect_caida(
    played: Card,
    capture_set: FrozenSet[Card],
    last_placed: Card | None,
) -> bool:
    """
    Detect if this capture is a CAÍDA (cascade match).

    A caída occurs when the played card captures a table card AND the LAST card placed on the table (by a opposite team player) has the same value as the played card AND is being captured in this move.

    In practice: last_placed must be in capture_set and
    last_placed.value == played.value.
    """
    if last_placed is None:
        return False
    if last_placed not in capture_set:
        return False
    return last_placed.value == played.value


def detect_limpia(
    capture_set: FrozenSet[Card],
    table: List[Card],
) -> bool:
    """
    Detect if this capture clears the entire table (limpia).
    """
    return set(capture_set) == set(table)


# Scoring helpers

CAIDA_POINTS = 2
LIMPIA_POINTS = 2


def score_play(
    played: Card,
    capture_set: FrozenSet[Card] | None,
    table: List[Card],
    last_placed: Card | None,
    non_caida_allowed: bool = True,
) -> Tuple[int, List[str]]:
    """
    Compute points earned for a single play.

    Returns (points, reasons) where reasons is a list of human-readable strings.

    Points:
      - +2 for a caída
      - +2 for a limpia (clearing the table)

    Actually in Cuarenta the scoring is:
      - Each CAPTURE of ≥1 card = 1 point per captured card (the pip cards
        themselves are worth 1 pt each towards the 40).
      - Caída = 2 bonus points (on top of the capture points).
      - Limpia = 2 bonus points (on top of capture points).
    """
    points = 0
    reasons = []

    if capture_set:
        # Caída bonus
        if detect_caida(played, capture_set, last_placed):
            points += CAIDA_POINTS
            reasons.append(PointsReasons.CAIDA)

        # Limpia bonus
        elif detect_limpia(capture_set, table) and non_caida_allowed:
            points += LIMPIA_POINTS
            reasons.append(PointsReasons.LIMPIA)

    return points, reasons
