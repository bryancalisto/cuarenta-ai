"""
Cuarenta environment (single-process, no search).

This is the game core shared by the PPO pipeline (`src/ppo/`) and the imitation
learning pipeline (`src/imitation_learning/`). The search-based (ISMCTS) pipeline
has its own copy under `src/ismcts/`, because it additionally needs stateless
transitions and determinization.

CONTRACT
--------
* 4 players, fixed teams {0,2} vs {1,3}, 5 cards each, game to 40 points.
* `reset()` -> (obs, info) for player 0. `step(action: Action)` -> (obs, reward,
  done, info) where `obs` belongs to the player who is about to move and `reward`
  is from the acting player's TEAM point of view.
* `legal_actions()` returns the `Action` objects for the current player. A player
  able to capture MUST capture (that is expressed by never offering a
  place-only action for a card that can capture).

ACTION ENCODING (used by the PPO policy head and by the Go/ONNX backend)
------------------------------------------------------------------------
    action_id = card_index * 41 + capture_option_index
    capture_option_index = 0            -> "place this card on the table"
    capture_option_index = 1..n         -> the n-th legal capture for that card

`capture_option_index` is POSITIONAL inside the per-card capture list, so the
ORDER of `legal_actions()` matters: it defines what index 1, 2, 3, ... mean.
The order is "same-rank captures, then sum captures, then stairs captures, in the
order `captures.valid_captures` produces them", with each capture option sorted by
card index. Do not sort this list differently without re-exporting every model and
updating the backend.

OBSERVATION LAYOUTS
-------------------
`observe()` returns 11 arrays; each consumer flattens a subset (both layouts are
declared in `utils.py`):

  * `OBS_KEYS_ISMCTS` (208) — what the ISMCTS pipeline and PPO use:
    my_hand, table, my_team_captures, opp_team_captures, last_placed, my_team_score,
    opp_team_score, current_player, deck_remaining, plays_this_deal.
  * `OBS_KEYS_LEGACY` (244) — the alphabetical order the DAgger policy and the OLD
    exported ONNX model were built on; it swaps `plays_this_deal` for
    `played_this_deal` (40 dims that are just both capture piles, i.e. redundant).
    The model the backend serves today uses the 208-dim contract above, not this one.

ACTIONS
-------
Two representations exist:

  * a flat id `card_index * 41 + capture_option_index` (`ActionEncoder`) — what the OLD
    ONNX export emitted and what the DAgger pipeline still trains on. The Go backend no
    longer uses it: it speaks the per-action-feature contract below; and
  * per-action FEATURES (`action_features`, 84 dims per legal action, at most
    `MAX_ACTIONS` of them) — what PPO's policy head scores, exactly like ISMCTS. The
    policy never sees slot indices, so it generalises across positions.

`played_this_deal` looks like it could leak hidden information (it is computed as
"every card that is in no hand, not on the table and not in the deck"), but it is
exactly the union of both teams' capture piles, which are public anyway.

REWARD
------
    +1 per captured card, +2 for a caída, +2 for a limpia, WIN_BONUS for winning the game
so a team's episode return is `cards_captured + 2*caidas + 2*limpias + WIN_BONUS*won`.
The cartón is a reward term too (see the REWARD note in src/ismcts/env.py, where the same
reward lives).

This is a DENSE reward: taking cards pays immediately instead of only through the cartón
majority at the end of the last deck.

WHAT THIS MEANS FOR THE eval/ METRICS: the evaluation league's `_return_for` /
`_return_against` report this same quantity, so their scale is tied to this formula. All
three pipelines (PPO, ISMCTS, ReBeL) pay the same dense reward now, so `eval/*_return_*`
IS comparable across pipelines as long as their `WIN_BONUS` values match — check that
before reading a return chart across two runs. Win rates and scores are comparable
unconditionally.
"""

from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, FrozenSet

from card import Card, build_deck, shuffled_deck
from captures import CAIDA_POINTS, LIMPIA_POINTS, valid_captures, score_play
from utils import PointsReasons, has_reason

NUM_PLAYERS = 4
HAND_SIZE = 5
# Policy head geometry, shared with the ISMCTS pipeline: the network scores at most
# MAX_ACTIONS legal actions per decision, each described by ACTION_FEAT_DIM features.
MAX_ACTIONS = 64
ACTION_FEAT_DIM = 84
WIN_SCORE = 40

# Reward for winning the game: +1 per captured card, +2 caída, +2 limpia, WIN_BONUS for
# the win. Measured with the trained net, the win bonus is
# only ~15% of a game's return, so raise this if you want a win-first agent. It is a
# single knob on purpose; see the REWARD note in src/ismcts/env.py for the full rationale.
WIN_BONUS = 20
MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS = 38
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}   # player: team index
TEAM_PLAYERS = {0: [0, 2], 1: [1, 3]}

# A fresh 40-card deck, hoisted to module level: `observe()` needs it on every step
# (to work out which cards have left play) and `Card` is a frozen dataclass, so a
# shared set of them is safe to reuse.
ALL_CARDS: FrozenSet[Card] = frozenset(build_deck())


@dataclass
class Action:
    card: Card
    capture: Optional[FrozenSet[Card]]  # None = place without capturing

    def __str__(self) -> str:
        if self.capture:
            caps = ",".join(str(c) for c in sorted(self.capture))
            return f"play {self.card} → capture [{caps}]"
        return f"play {self.card} → place"


# ---------------------------------------------------------------------------
# Action features — the network's per-action input (same layout as src/ismcts/env.py)
# ---------------------------------------------------------------------------
#
# ACTION_FEAT_DIM = 84:
#   [0  : 40]  played card one-hot
#   [40 : 80]  captured cards one-hot
#   [80]       is_capture
#   [81]       is_caida   (worth 2 points)
#   [82]       is_limpia  (worth 2 points)
#   [83]       unused (reserved)
#
# The policy head scores each legal action by these features rather than by a slot
# index, so "slot 3" never has to mean the same thing in two positions. That is what
# makes one network usable in any position (and, later, shareable with the search).
def action_features(action: "Action", state: "CuarentaState") -> np.ndarray:
    f = np.zeros(ACTION_FEAT_DIM, dtype=np.float32)

    f[action.card.index] = 1.0
    if action.capture is not None:
        for c in action.capture:
            f[40 + c.index] = 1.0
        f[80] = 1.0

    # `score_play` returns reasons as a LIST: test membership, never `==` against a
    # string (see utils.has_reason — this exact mistake once made the caída features
    # permanently 0 and the env's reward never pay the bonus).
    _, reasons = score_play(
        played=action.card,
        capture_set=action.capture,
        table=state.table,
        last_placed=state.last_placed,
        non_caida_allowed=state.scores[TEAMS[state.current_player]]
                        < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
    )
    f[81] = 1.0 if has_reason(reasons, PointsReasons.CAIDA) else 0.0
    f[82] = 1.0 if has_reason(reasons, PointsReasons.LIMPIA) else 0.0
    return f


def action_features_matrix(legal: List["Action"], state: "CuarentaState"):
    """
    Build the [MAX_ACTIONS, ACTION_FEAT_DIM] feature block and the [MAX_ACTIONS] bool
    mask for a list of legal actions (packed into the first len(legal) slots).
    """
    feats = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((MAX_ACTIONS,), dtype=bool)
    for i, a in enumerate(legal):
        feats[i] = action_features(a, state)
        mask[i] = True
    return feats, mask


@dataclass
class CuarentaState:
    hands: Dict[int, List[Card]]          # player_id → cards in hand
    table: List[Card]                     # face-up cards on the table (ordered)
    deck: List[Card]                      # remaining undealt cards
    n_decks_consumed: int                 # how many deals have been played out
    scores: List[int]                     # [team0_score, team1_score]
    cards_captured: List[List[Card]]      # [team0_cards, team1_cards]
    current_player: int                   # whose turn it is
    last_placed: Optional[Card]           # last card placed on table (for caída detection)
    plays_this_deal: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
                                          # turns taken per seat in the current deal
    done: bool = False
    winner: Optional[int] = None          # winning team (0 or 1), if done

    def team(self, player: int) -> int:
        return TEAMS[player]


class CuarentaEnv:
    """
    Turn-based Cuarenta environment.

    Determinism: all randomness comes from `self.random` (a `random.Random`), so a
    seeded env replays exactly. `reset(seed=...)` re-seeds it.
    """

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.state: Optional[CuarentaState] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Start a new game. Returns (observation for player 0, info)."""
        if seed is not None:
            self.random = random.Random(seed)
        deck = shuffled_deck(self.random)
        hands, deck = self._deal(deck)
        self.state = CuarentaState(
            hands=hands,
            table=[],
            deck=deck,
            scores=[0, 0],
            cards_captured=[[], []],
            current_player=0,
            n_decks_consumed=0,
            last_placed=None,
            plays_this_deal=[0, 0, 0, 0],
            done=False,
            winner=None,
        )
        return self.observe(0), {}

    def _count_carton_points(self, cards_captured: List[Card]) -> int:
        """
        Majority bonus ("cartón") for one team's capture pile.

        points = len(captured) - 19; if that is positive add 5, then round up to the
        next even number. So 20 cards -> 6, 21 cards -> 8, 22 cards -> 8, 23 -> 10 ...
        Capped by the caller at MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS.
        """
        carton_points = len(cards_captured) - 19

        if carton_points > 0:
            carton_points += 5
            if carton_points % 2 != 0:
                carton_points += 1
        else:
            carton_points = 0

        return carton_points

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        """
        Apply `action` for the current player.

        Returns (observation, reward, done, info); see the module docstring for the
        reward terms. `reward` is from the CURRENT PLAYER'S TEAM point of view.
        """
        s = self.state
        assert not s.done, "Game is already over. Call reset()."
        player = s.current_player
        reward = 0

        # validate action
        assert action.card in s.hands[player], (
            f"Player {player} does not hold {action.card}. Hand: {s.hands[player]}"
        )
        legal = self.legal_actions()
        assert any(
            a.card == action.card and a.capture == action.capture for a in legal
        ), f"Action {action} is not legal. Legal: {legal}"

        # remove card from hand
        s.hands[player].remove(action.card)
        s.plays_this_deal[player] += 1

        team = TEAMS[player]

        # compute score and update table
        points, reasons = score_play(
            played=action.card,
            capture_set=action.capture,
            table=s.table,
            last_placed=s.last_placed,
            non_caida_allowed=s.scores[team] < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS
        )

        info = {
            "player": player,
            "team": team,
            "action": str(action),
            "points": points,
            "reasons": reasons,
        }

        if action.capture:
            # Move captured cards from the table into the team's capture pile.
            for card in action.capture:
                s.table.remove(card)
                s.cards_captured[team].append(card)

            # add the thrown card to cards captured
            s.cards_captured[team].append(action.card)
            s.last_placed = None   # table was modified; caída chain resets
            reward += len(action.capture)

            if has_reason(reasons, PointsReasons.CAIDA):
                reward += CAIDA_POINTS

            if has_reason(reasons, PointsReasons.LIMPIA):
                reward += LIMPIA_POINTS
        else:
            # Place card on table
            s.table.append(action.card)
            s.last_placed = action.card

        s.scores[team] += points

        # check win
        if s.scores[team] >= WIN_SCORE:
            s.done = True
            s.winner = team
            reward += WIN_BONUS   # terminal bonus
            info["winner"] = team
            return self.observe(player), reward, True, info

        # advance turn
        s.current_player = (player + 1) % NUM_PLAYERS

        # If every hand is empty the deal is over: either redeal from the same deck,
        # or (when the deck cannot fill another deal) pay the cartón and start a new
        # round with a freshly shuffled deck. Scores persist across rounds.
        if all(len(s.hands[p]) == 0 for p in range(NUM_PLAYERS)):
            if len(s.deck) >= NUM_PLAYERS * HAND_SIZE:
                # There's still deck to play
                self._redeal()
            else:
                # Count carton for each team and update scores
                team_carton = {}
                for team_i in range(2):
                    carton_points = self._count_carton_points(s.cards_captured[team_i])
                    # Carton points don't count if points are 38 or higher
                    if s.scores[team_i] >= MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS:
                        carton_points = 0
                    team_carton[team_i] = carton_points
                    s.scores[team_i] = min(
                        s.scores[team_i] + carton_points,
                        MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
                    )

                info["carton_points"] = team_carton[TEAMS[player]]
                # Cartón is PAID to the acting team, in sync with src/ismcts/env.py (see the
                # REWARD note there) — all three pipelines pay the same dense reward so their
                # `eval/*_return_*` curves stay comparable.
                reward += team_carton[TEAMS[player]]
                info["new_round"] = True   # the deal that just ended is complete
                self._new_round()

        return self.observe(s.current_player), reward, False, info

    def legal_actions(self) -> List[Action]:
        """
        All legal `Action`s for the current player, in the order that defines the
        per-card capture indices of the action encoding (see the module docstring).

        A card that can capture is never offered as a place-only action: capturing
        is mandatory when it is possible.
        """
        s = self.state
        player = s.current_player
        hand = s.hands[player]
        actions: List[Action] = []

        for card in hand:
            captures = valid_captures(card, s.table)
            if captures:
                for cap in captures:
                    actions.append(Action(card=card, capture=cap))
            else:
                actions.append(Action(card=card, capture=None))

        return actions

    def observe(self, player: int) -> Dict:
        """
        Observation for `player` as a dict of numpy arrays.

        Hidden information (other players' hands, the deck order) is not included.
        """
        s = self.state

        def cards_to_bitvec(cards: List[Card]) -> np.ndarray:
            vec = np.zeros(40, dtype=np.float32)
            for c in cards:
                vec[c.index] = 1.0
            return vec

        # Cards that have left play for good: in nobody's hand, not on the table and
        # not in the deck. That is exactly the union of both capture piles (public
        # information), so this leaks nothing.
        known = (
            set(s.hands[player])
            | set(s.table)
            | set(s.deck)
            | {c for p in range(NUM_PLAYERS) if p != player for c in s.hands[p]}
        )
        played_this_deal = ALL_CARDS - known

        obs = {
            "my_hand":            cards_to_bitvec(s.hands[player]),
            "table":              cards_to_bitvec(s.table),
            "played_this_deal":   cards_to_bitvec(list(played_this_deal)),
            "my_team_score":      np.array([s.scores[TEAMS[player]]], dtype=np.float32),
            "opp_team_score":     np.array([s.scores[1 - TEAMS[player]]], dtype=np.float32),
            "my_team_captures":   cards_to_bitvec(s.cards_captured[TEAMS[player]]),
            "opp_team_captures":  cards_to_bitvec(s.cards_captured[1 - TEAMS[player]]),
            "current_player":     np.array([s.current_player], dtype=np.float32),
            "last_placed":        cards_to_bitvec([s.last_placed] if s.last_placed else []),
            "deck_remaining":     np.array([len(s.deck)], dtype=np.float32),
            # How many turns each seat has taken in the current deal. The ISMCTS
            # layout uses this instead of `played_this_deal` (40 dims, and redundant
            # with the two capture bitvecs); the legacy/ONNX layout still uses the
            # latter, so the env provides both and each layout picks its subset.
            "plays_this_deal":    np.array(s.plays_this_deal, dtype=np.float32),
        }
        return obs

    # helpers

    @staticmethod
    def _deal(deck: List[Card]) -> Tuple[Dict[int, List[Card]], List[Card]]:
        """Deal HAND_SIZE cards to each player. Returns (hands, remaining_deck)."""
        hands = {}
        for p in range(NUM_PLAYERS):
            hands[p] = deck[:HAND_SIZE]
            deck = deck[HAND_SIZE:]
        return hands, deck

    def _redeal(self):
        """Deal the next HAND_SIZE cards to each player from the remaining deck."""
        s = self.state
        for p in range(NUM_PLAYERS):
            s.hands[p] = s.deck[:HAND_SIZE]
            s.deck = s.deck[HAND_SIZE:]
        s.plays_this_deal = [0, 0, 0, 0]

    def _new_round(self):
        """Shuffle a new deck and deal fresh hands (scores persist)."""
        s = self.state
        new_deck = shuffled_deck(self.random)
        s.hands, s.deck = self._deal(new_deck)
        s.cards_captured = [[], []]
        s.table = []
        s.last_placed = None
        s.plays_this_deal = [0, 0, 0, 0]
        s.n_decks_consumed += 1

        # next player starts in new round
        s.current_player = (s.current_player + 1) % NUM_PLAYERS

    def render(self) -> str:
        """Human-readable game state string."""
        s = self.state
        lines = [
            "=" * 50,
            f"  CUARENTA  |  Team 0 (P0,P2): {s.scores[0]}  vs  Team 1 (P1,P3): {s.scores[1]}",
            f"  Current player: {s.current_player}  |  Deck remaining: {len(s.deck)}",
            f"  Table: {' '.join(str(c) for c in s.table) or '(empty)'}",
            f"  Last placed: {s.last_placed}",
        ]
        for p in range(NUM_PLAYERS):
            marker = " ◄" if p == s.current_player else ""
            lines.append(f"  P{p} (team {TEAMS[p]}): {' '.join(str(c) for c in sorted(s.hands[p]))}{marker}")
        lines.append("=" * 50)
        return "\n".join(lines)


class ActionEncoder:
    def __init__(self, env: CuarentaEnv):
        self.env = env
        self.all_cards = build_deck()  # 40 cards
        self.max_action_idx = 40 * 41  # 40 cards × 41 capture options (0=no capture, 1-40 for captures)

    def action_to_idx(self, action: Action) -> int:
        """Encode action as integer."""
        card_idx = self.all_cards.index(action.card)
        if action.capture is None:
            capture_idx = 0
        else:
            # Get legal actions for this card and find which one matches
            legal_actions = [a for a in self.env.legal_actions() if a.card == action.card]
            try:
                capture_idx = legal_actions.index(action) + 1
            except ValueError:
                capture_idx = 0  # Fallback
        return card_idx * 41 + capture_idx

    def idx_to_action(self, idx: int) -> Optional[Action]:
        """Decode integer to action (if legal). Returns None if illegal."""
        card_idx = idx // 41
        capture_idx = idx % 41

        if card_idx >= len(self.all_cards):
            return None

        card = self.all_cards[card_idx]

        # Check if card is in current player's hand
        player = self.env.state.current_player
        if card not in self.env.state.hands[player]:
            return None

        # Get legal actions for this card
        legal_actions = [a for a in self.env.legal_actions() if a.card == card]
        if not legal_actions:
            return None

        if capture_idx == 0:
            # Looking for place-only action
            for action in legal_actions:
                if action.capture is None:
                    return action
            # No place-only action; return first legal action with this card
            return legal_actions[0] if legal_actions else None
        else:
            # Looking for specific capture
            if capture_idx - 1 < len(legal_actions):
                return legal_actions[capture_idx - 1]

        return None
