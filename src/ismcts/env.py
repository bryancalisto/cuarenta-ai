from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, FrozenSet

from card import Card, build_deck, shuffled_deck
from captures import valid_captures, score_play
from utils import PointsReasons

NUM_PLAYERS = 4
HAND_SIZE = 5
WIN_SCORE = 40
MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS = 38
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}   # player: team index
TEAM_PLAYERS = {0: [0, 2], 1: [1, 3]}


@dataclass(frozen=True)
class Action:
    card: Card
    capture: Optional[FrozenSet[Card]]  # None = place without capturing

    def __str__(self) -> str:
        if self.capture:
            caps = ",".join(str(c) for c in sorted(self.capture))
            return f"play {self.card} → capture [{caps}]"
        return f"play {self.card} → place"


# Action encoding
#
# An action is encoded as a single integer for RL compatibility.
# action = card_index * 41 + capture_option_index
# capture_option_index = 0 means "no capture" (place only)
# capture_option_index 1..40 index into the sorted list of valid captures.
# In practice we build the legal action list dynamically each turn.
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


@dataclass
class CuarentaState:
    hands: Dict[int, List[Card]]          # player_id → cards in hand
    table: List[Card]                     # face-up cards on the table (ordered)
    deck: List[Card]                      # remaining undealt cards
    scores: List[int]                     # [team0_score, team1_score]
    cards_captured: List[List[Card]]      # [team0_cards, team1_cards]
    current_player: int                   # whose turn it is
    plays_this_deal: List[int]            # [p0, p1, p2, p3] turns taken this deal
    last_placed: Optional[Card]           # last card placed on table (for caída detection)
    done: bool = False
    winner: Optional[int] = None          # winning team (0 or 1), if done

    def team(self, player: int) -> int:
        return TEAMS[player]


# Gymnasium action space helper. Implemented this wrapper to avoid 'Environment has issues: Illegal action index 18; 5 legal actions.' error
class MaskedDiscrete(spaces.Discrete):
    """A Discrete space whose sample() respects a per-state mask."""

    def __init__(self, n: int, mask_fn):
        super().__init__(n)
        self._mask_fn = mask_fn


    def sample(self, mask=None):
        if mask is None:
            mask = self._mask_fn()
        mask = np.asarray(mask, dtype=bool)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("MaskedDiscrete.sample(): no legal actions")
        return int(self.np_random.choice(legal))


    def contains(self, x) -> bool:
        # still accepts any integer in [0, n)
        return super().contains(x)


class CuarentaEnv(gym.Env):
    MAX_ACTIONS = 64
    ACTION_FEAT_DIM = 84

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.random = random.Random(seed)
        self.state: Optional[CuarentaState] = None

        """
        Action Space:
        offset 0: card to play bitvec (40, one-hot)
        offset 40: capture bitvec (40, one-hot)
        offset 80: is capture (1, 0 or 1)
        offset 81: is caida (1, 0 or 1)
        offset 82: is limpia (1, 0 or 1)
        offset 83: ACTION_FEAT_DIM = 84
        """
        
        self.action_space = MaskedDiscrete(self.MAX_ACTIONS, self.action_mask)
        
        self.observation_space = spaces.Dict({
            "my_hand":            spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32),
            "table":              spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32),
            "my_team_score":      spaces.Box(low=0, high=40, shape=(1,), dtype=np.float32),
            "opp_team_score":     spaces.Box(low=0, high=40, shape=(1,), dtype=np.float32),
            "my_team_captures":   spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32),
            "opp_team_captures":  spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32),
            "current_player":     spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),
            "plays_this_deal":     spaces.Box(low=0, high=HAND_SIZE, shape=(4,), dtype=np.float32),
            "last_placed":        spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32),
            "deck_remaining":     spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "action_features":    spaces.Box(0, 1, (self.MAX_ACTIONS, self.ACTION_FEAT_DIM), np.float32),
            "action_mask":        spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.bool),
        })
    
    def _action_features(self, action: Action) -> np.ndarray:
        f = np.zeros(self.ACTION_FEAT_DIM, dtype=np.float32)

        # played card
        f[action.card.index] = 1.0

        # captured cards + is_capture
        if action.capture is not None:
            for c in action.capture:
                f[40 + c.index] = 1.0
            f[80] = 1.0
        
        # caída / limpia flags
        _, reasons = score_play(
            played=action.card,
            capture_set=action.capture,
            table=self.state.table,
            last_placed=self.state.last_placed,
            non_caida_allowed=self.state.scores[TEAMS[self.state.current_player]]
                            < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
        )
        f[81] = 1.0 if reasons == PointsReasons.CAIDA else 0.0
        f[82] = 1.0 if reasons == PointsReasons.LIMPIA else 0.0
            
        return f

    def action_mask(self) -> np.ndarray:
        if self.state is None:
            # no state yet — report all slots legal so sample() doesn't crash
            return np.ones(self.MAX_ACTIONS, dtype=np.bool)

        legal = self.legal_actions()
        mask = np.zeros(self.MAX_ACTIONS, dtype=np.bool)
        mask[:len(legal)] = True
        return mask

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Start a new game. Returns (observation, info)."""
        super().reset(seed=seed)

        # Derive a fresh, seed-reproducible Python RNG for shuffling.
        # np_random is fully determined by the reset seed, so this is deterministic.
        seed_int = int(self.np_random.integers(0, 2**31 - 1))
        self.random = random.Random(seed_int)

        self._legal_cache = None
        deck = shuffled_deck(self.random)
        hands, deck = self._deal(deck)
        current_player = 0
        self.state = CuarentaState(
            hands=hands,
            table=[],
            deck=deck,
            scores=[0, 0],
            cards_captured=[[], []],
            current_player=current_player,
            plays_this_deal=[0, 0, 0, 0],
            last_placed=None,
            done=False,
            winner=None,
        )

        return self.observe(current_player), {}

    def _count_carton_points(self, cards_captured: List[Card]) -> int:
        carton_points = len(cards_captured) - 19

        if carton_points > 0:
            carton_points += 5
            if carton_points % 2 != 0:
                carton_points += 1
        
        else:
            carton_points = 0        

        return carton_points

    def step(self, action_idx: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Apply action for the current player.
        Returns (observation, reward, done, truncated info).

        Reward is from the perspective of the CURRENT PLAYER'S TEAM.
        """
        s = self.state
        assert not s.done, "Game is already over. Call reset()."
        self._legal_cache = None # invalidate before mutation
        legal = self.legal_actions() # rebuild (this also repopulates cache)

        assert 0 <= action_idx < len(legal), (
            f"Illegal action index {action_idx}; {len(legal)} legal actions."
        )
        action = legal[action_idx]

        player = s.current_player
        reward = 0
        truncated = False # don't care about "truncated" in this env

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

        # Increase player's plays count for the determinizer to use
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
            # print(f'captured: {action.capture}')
            # Remove captured cards from table and put them in the team's captured cards
            for card in action.capture:
                s.table.remove(card)
                s.cards_captured[team].append(card)
            
            # add the thrown card to cards captured
            s.cards_captured[team].append(action.card)
            s.last_placed = None   # table was modified; caída chain resets

            if reasons == PointsReasons.CAIDA:
                reward += 2 
            elif reasons == PointsReasons.LIMPIA:
                reward += 2
        else:
            # Place card on table
            s.table.append(action.card)
            s.last_placed = action.card

        s.scores[team] += points

        # check win
        if s.scores[team] >= WIN_SCORE:
            s.done = True
            s.winner = team
            reward += 50   # terminal bonus
            info["winner"] = team
            self._legal_cache = None
            return self.observe(player), reward, True, truncated, info

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
                    # Carton points don't count if points are 38 or higher
                    if s.scores[team] >= MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS:
                        carton_points = 0

                    team_carton[team] = carton_points
                    s.scores[team] = min(s.scores[team] + carton_points, MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS)

                current_team = TEAMS[player]
                info["carton_points"] = team_carton[current_team]

                self._new_round()

        self._legal_cache = None
        return self.observe(s.current_player), reward, False, truncated, info


    def legal_actions(self) -> List[Action]:
        """Return all legal Action objects for the current player."""
        if self._legal_cache is not None:
            return self._legal_cache

        s = self.state
        player = s.current_player
        hand = s.hands[player]
        actions: List[Action] = []

        for card in hand:
            captures = sorted(
                valid_captures(card, s.table),
                key=lambda cap: tuple(sorted(c.index for c in cap)),
            )

            if captures:
                for cap in captures:
                    actions.append(Action(card=card, capture=cap))
            else:
                actions.append(Action(card=card, capture=None))
        
        # sort overall so the order is deterministic across cards too
        actions.sort(key=lambda a: (a.card.index, sorted(c.index for c in (a.capture or []))))

        if len(actions) > self.MAX_ACTIONS:
            raise RuntimeError(
                f"More than {self.MAX_ACTIONS} legal actions "
                f"({len(actions)}). Increase MAX_ACTIONS."
            )

        self._legal_cache = actions
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

        # action features + mask
        legal = self.legal_actions()
        feats = np.zeros((self.MAX_ACTIONS, self.ACTION_FEAT_DIM), dtype=np.float32)
        mask = np.zeros((self.MAX_ACTIONS,), dtype=np.float32) 
        for i, a in enumerate(legal):
            feats[i] = self._action_features(a)
            mask[i] = 1

        obs = {
            "my_hand":            cards_to_bitvec(s.hands[player]),
            "table":              cards_to_bitvec(s.table),
            "my_team_score":      np.array([s.scores[TEAMS[player]]], dtype=np.float32),
            "opp_team_score":     np.array([s.scores[1 - TEAMS[player]]], dtype=np.float32),
            "my_team_captures":   cards_to_bitvec(s.cards_captured[TEAMS[player]]),
            "opp_team_captures":  cards_to_bitvec(s.cards_captured[1 - TEAMS[player]]),
            "current_player":     np.array([s.current_player], dtype=np.float32),
            "plays_this_deal":    np.array(s.plays_this_deal, dtype=np.float32),
            "last_placed":        cards_to_bitvec([s.last_placed] if s.last_placed else []),
            "deck_remaining":     np.array([len(s.deck)], dtype=np.float32),
            "action_features":    feats,
            "action_mask":        mask,
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
        new_hands = {p: s.deck[p * HAND_SIZE:(p + 1) * HAND_SIZE] for p in range(NUM_PLAYERS)}
        new_deck = s.deck[NUM_PLAYERS * HAND_SIZE:]
        self._install_hands(new_hands, new_deck)
        
    
    def _install_hands(self, hands: Dict[int, List[Card]], deck: List[Card]) -> None:
        """Set hands+deck and reset the per-deal play counter. Single source of truth."""
        s = self.state
        s.hands = hands
        s.deck = deck
        s.plays_this_deal = [0, 0, 0, 0]

    def _new_round(self):
        """Shuffle a new deck and deal fresh hands (scores persist)."""
        s = self.state
        new_deck = shuffled_deck(self.random)
        hands, deck = self._deal(new_deck)
        self._install_hands(hands, deck)
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
            f"  Table: {' | '.join(str(c) for c in s.table) or '(empty)'}",
            f"  Last placed: {s.last_placed}",
        ]
        for p in range(NUM_PLAYERS):
            marker = " ◄" if p == s.current_player else ""
            lines.append(f"  P{p} (team {TEAMS[p]}): {' | '.join(str(c) for c in sorted(s.hands[p]))} | {marker}")
        lines.append("=" * 50)
        string = "\n".join(lines)
        print(string)
        return string



# Register the environment with Gymnasium
gym.register(
    id="gymnasium_env/Cuarenta-v0",
    entry_point=CuarentaEnv,
    max_episode_steps=512,  # Prevent infinite episodes
)

debug = False
if debug:
    # Some validations. Using this while developing this env
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(gym.make("gymnasium_env/Cuarenta-v0").unwrapped)
        print("Environment check: 👍")
    except Exception as e:
        print(f"Environment has issues: {e}")