from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, FrozenSet

from card import Card, build_deck, shuffled_deck
from captures import valid_captures, score_play

NUM_PLAYERS = 4
HAND_SIZE = 5
WIN_SCORE = 40
MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS = 38
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}   # player: team index
TEAM_PLAYERS = {0: [0, 2], 1: [1, 3]}


# Action encoding
#
# An action is encoded as a single integer for RL compatibility.
# action = card_index * 41 + capture_option_index
# capture_option_index = 0 means "no capture" (place only)
# capture_option_index 1..40 index into the sorted list of valid captures
# in practice we build the legal action list dynamically each turn.

@dataclass
class Action:
    card: Card
    capture: Optional[FrozenSet[Card]]  # None = place without capturing

    def __str__(self) -> str:
        if self.capture:
            caps = ",".join(str(c) for c in sorted(self.capture))
            return f"play {self.card} → capture [{caps}]"
        return f"play {self.card} → place"

# State

@dataclass
class CuarentaState:
    hands: Dict[int, List[Card]]          # player_id → cards in hand
    table: List[Card]                     # face-up cards on the table (ordered)
    deck: List[Card]                      # remaining undealt cards
    n_decks_consumed: int                 # amount of decks that have been consumed so far
    scores: List[int]                     # [team0_score, team1_score]
    cards_captured: List[List[Card]]      # [team0_cards, team1_cards]
    current_player: int                   # whose turn it is
    last_placed: Optional[Card]           # last card placed on table (for caída detection)
    done: bool = False
    winner: Optional[int] = None          # winning team (0 or 1), if done

    def team(self, player: int) -> int:
        return TEAMS[player]


class CuarentaEnv:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.state: Optional[CuarentaState] = None


    def reset(self) -> Tuple[Dict, Dict]:
        """Start a new game. Returns (observation, info)."""
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
            done=False,
            winner=None,
        )
        return self.observe(0), {}

    def _count_carton_points(self, cards_captured: List[int]) -> List[int]:
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
        Apply action for the current player.
        Returns (observation, reward, done, info).

        Reward is from the perspective of the CURRENT PLAYER'S TEAM.
        """
        s = self.state
        assert not s.done, "Game is already over. Call reset()."
        player = s.current_player
        reward = 0

        # validate action
        # print(f'PLAYER: {player}, ACTION: {action}')
        assert action.card in s.hands[player], (
            f"Player {player} does not hold {action.card}. Hand: {s.hands[player]}"
        )
        legal = self.legal_actions()
        assert any(
            a.card == action.card and a.capture == action.capture for a in legal
        ), f"Action {action} is not legal. Legal: {legal}"

        # remove card from hand
        s.hands[player].remove(action.card)

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
            # Remove captured cards from table and put them in the team's captured cards
            for card in action.capture:
                s.table.remove(card)
                s.cards_captured[team].append(card)
            s.last_placed = None   # table was modified; caída chain resets
            # Reduced capture reward. If score >= 38, don't reward captures to force caida
            if s.scores[team] < 38:
                # reward += len(action.capture) * 0.25
                reward += len(action.capture) * 0
        else:
            # Place card on table
            s.table.append(action.card)
            s.last_placed = action.card

        # apply score (but don't reward it if score >= 38, to force caida)
        s.scores[team] += points
        if s.scores[team] < 38:
            reward += points

        # check win
        if s.scores[team] >= WIN_SCORE:
            s.done = True
            s.winner = team
            reward += 100.0   # terminal bonus
            info["winner"] = team
            return self.observe(player), reward, True, info

        # advance turn
        s.current_player = (player + 1) % NUM_PLAYERS

        # if all hands empty, count cartón and check end game. if no end game, then redeal
        # Cartón means the amount of captured cards counted in this way: points = len(captured) - 19; if points > 0 then round(6 + points) to the next integer that's divisible by two
        if all(len(s.hands[p]) == 0 for p in range(NUM_PLAYERS)):
            if len(s.deck) >= NUM_PLAYERS * HAND_SIZE:
                # There's still deck to play
                self._redeal()
            else:
                # Count carton for each team and update scores
                team_carton = {}
                for team in range(2):
                    team_carton[team] = 0

                    carton_points = self._count_carton_points(s.cards_captured[team])
                    # Reduce carton reward if score >= 38 to force caida
                    if s.scores[team] >= 38:
                        carton_points = 0

                    team_carton[team] = carton_points
                    s.scores[team] = min(s.scores[team] + carton_points, MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS)

                # Reward current player for their team's carton only
                current_team = TEAMS[player]
                info["carton_points"] = team_carton[current_team]

                self._new_round()

        return self.observe(s.current_player), reward, False, info


    def legal_actions(self) -> List[Action]:
        """Return all legal Action objects for the current player."""
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
        Return the observation for `player` as a dict of numpy arrays.
        Hidden info (opponent hands, opponent captures) is NOT included.
        """
        s = self.state

        def cards_to_bitvec(cards: List[Card]) -> np.ndarray:
            vec = np.zeros(40, dtype=np.float32)
            for c in cards:
                vec[c.index] = 1.0
            return vec

        # Cards already played this deal (not in any hand or table or deck)
        all_cards = set(build_deck())
        known = (
            set(s.hands[player])
            | set(s.table)
            | set(s.deck)
            | {c for p in range(NUM_PLAYERS) if p != player for c in s.hands[p]}
        )
        played_this_deal = all_cards - known

        obs = {
            "my_hand":            cards_to_bitvec(s.hands[player]),
            "table":              cards_to_bitvec(s.table),
            "played_this_deal":   cards_to_bitvec(played_this_deal),
            "my_team_score":      np.array([s.scores[TEAMS[player]]], dtype=np.float32),
            "opp_team_score":     np.array([s.scores[1 - TEAMS[player]]], dtype=np.float32),
            "my_team_captures":   cards_to_bitvec(s.cards_captured[TEAMS[player]]),
            "opp_team_captures":  cards_to_bitvec(s.cards_captured[1 - TEAMS[player]]),
            "current_player":     np.array([s.current_player], dtype=np.float32),
            "last_placed":        cards_to_bitvec([s.last_placed] if s.last_placed else []),
            "deck_remaining":     np.array([len(s.deck)], dtype=np.float32),
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
        """Deal next HAND_SIZE cards to each player from remaining deck."""
        s = self.state
        for p in range(NUM_PLAYERS):
            s.hands[p] = s.deck[:HAND_SIZE]
            s.deck = s.deck[HAND_SIZE:]

    def _redeal_partial(self):
        """Deal remaining deck cards as evenly as possible."""
        s = self.state
        idx = 0
        for p in range(NUM_PLAYERS):
            if idx < len(s.deck):
                s.hands[p] = [s.deck[idx]]
                idx += 1
            else:
                s.hands[p] = []
        s.deck = []

    def _new_round(self):
        """Shuffle a new deck and deal fresh hands (scores persist)."""
        s = self.state
        new_deck = shuffled_deck(self.random)
        s.hands, s.deck = self._deal(new_deck)
        s.cards_captured = [[], []]
        s.table = []
        s.last_placed = None

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
    """Convert between Action objects and integer indices."""

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