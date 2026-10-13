# rules.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy

from card import Card, build_deck, shuffled_deck
from captures import CAIDA_POINTS, LIMPIA_POINTS, valid_captures, score_play
from utils import PointsReasons, has_reason
from env import (
    Action, CuarentaState,
    NUM_PLAYERS, HAND_SIZE, WIN_BONUS, WIN_SCORE,
    MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
    TEAMS,
)

# cloning
def clone_state(s: CuarentaState) -> CuarentaState:
    """
    Cheap structural clone. Cards are immutable (they hash by value in your
    codebase), so we don't need deepcopy on the Card objects themselves — only
    on the containers.
    """
    return CuarentaState(
        hands={p: list(h) for p, h in s.hands.items()},
        table=list(s.table),
        deck=list(s.deck),
        scores=list(s.scores),
        cards_captured=[list(s.cards_captured[0]), list(s.cards_captured[1])],
        current_player=s.current_player,
        plays_this_deal=list(s.plays_this_deal),
        last_placed=s.last_placed,
        done=s.done,
        winner=s.winner,
    )


# legality
def legal_actions_from_state(s: CuarentaState) -> List[Action]:
    """Same logic as CuarentaEnv.legal_actions, but stateless."""
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

    actions.sort(key=lambda a: (a.card.index, sorted(c.index for c in (a.capture or []))))
    return actions

# transitions

def _count_carton_points(cards_captured: List[Card]) -> int:
    pts = len(cards_captured) - 19
    if pts > 0:
        pts += 5
        if pts % 2 != 0:
            pts += 1
    else:
        pts = 0
    return pts


def _deal(deck: List[Card]) -> Tuple[Dict[int, List[Card]], List[Card]]:
    hands = {}
    for p in range(NUM_PLAYERS):
        hands[p] = deck[:HAND_SIZE]
        deck = deck[HAND_SIZE:]
    return hands, deck


def _redeal(s: CuarentaState) -> None:
    for p in range(NUM_PLAYERS):
        s.hands[p] = s.deck[:HAND_SIZE]
        s.deck = s.deck[HAND_SIZE:]
    s.plays_this_deal = [0, 0, 0, 0]


def _new_round(s: CuarentaState, rng) -> None:
    # rng here is a random.Random; if you want full determinism in search,
    # you should pass a seeded rng in. For rollouts, an unseeded rng is fine.
    new_deck = shuffled_deck(rng)
    s.hands, s.deck = _deal(new_deck)
    s.cards_captured = [[], []]
    s.table = []
    s.last_placed = None
    s.plays_this_deal = [0, 0, 0, 0]
    s.current_player = (s.current_player + 1) % NUM_PLAYERS


def step_state(
    s: CuarentaState,
    action: Action,
    rng,
) -> Tuple[CuarentaState, float, bool, int]:
    """
    Pure(ish) transition. Returns (new_state, reward, done, player_who_acted).

    - Does NOT mutate `s`.
    - `rng` is used only when a redeal/new-round occurs (deck shuffling).
    - reward is from the acting player's TEAM perspective.
    """
    assert not s.done, "Cannot step a finished state."
    player = s.current_player
    assert action.card in s.hands[player], (
        f"P{player} doesn't hold {action.card}. Hand: {s.hands[player]}"
    )

    # We clone eagerly to avoid mutating the caller's state.
    ns = clone_state(s)
    team = TEAMS[player]
    reward = 0.0

    ns.hands[player].remove(action.card)
    ns.plays_this_deal[player] += 1

    points, reasons = score_play(
        played=action.card,
        capture_set=action.capture,
        table=ns.table,
        last_placed=ns.last_placed,
        non_caida_allowed=ns.scores[team] < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS,
    )

    if action.capture:
        for card in action.capture:
            ns.table.remove(card)
            ns.cards_captured[team].append(card)
        ns.cards_captured[team].append(action.card)
        ns.last_placed = None
        # +1 per captured card, the dense term (capturing builds the carton and denies
        # the opponent; see the REWARD note in env.py).
        reward += len(action.capture)

        if has_reason(reasons, PointsReasons.CAIDA):
            reward += CAIDA_POINTS
        if has_reason(reasons, PointsReasons.LIMPIA):
            reward += LIMPIA_POINTS
    else:
        ns.table.append(action.card)
        ns.last_placed = action.card

    ns.scores[team] += points

    if ns.scores[team] >= WIN_SCORE:
        ns.done = True
        ns.winner = team
        reward += WIN_BONUS
        return ns, reward, True, player

    ns.current_player = (player + 1) % NUM_PLAYERS

    # Handle end-of-deal
    if all(len(ns.hands[p]) == 0 for p in range(NUM_PLAYERS)):
        if len(ns.deck) >= NUM_PLAYERS * HAND_SIZE:
            _redeal(ns)
        else:
            for t in range(2):
                carton = _count_carton_points(ns.cards_captured[t])
                if ns.scores[t] >= MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS:
                    carton = 0
                ns.scores[t] = min(
                    ns.scores[t] + carton, MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS
                )
                # Cartón IS rewarded, matching env.py (see the REWARD note in
                # src/ismcts/env.py). The ReBeL solver's utility is the game outcome, so this
                # term only matters where rewards are accumulated (its own eval harness
                # checks the return identity), but the two copies must still agree.
                if t == team:
                    reward += carton
            _new_round(ns, rng)

    return ns, reward, False, player


debug = False
if debug:
    import numpy as np, random
    from env import CuarentaEnv
    from rules import step_state, legal_actions_from_state, clone_state

    env = CuarentaEnv(seed=1)
    env.reset(seed=1)
    rng = random.Random(1)

    for i in range(200):
        obs_before = env.observe(env.state.current_player)
        state_before = clone_state(env.state)
        legal = legal_actions_from_state(env.state)
        a = int(np.flatnonzero(obs_before["action_mask"])[0])  # first legal

        # env path
        _, r_env, done_env, _, info_env = env.step(a)

        # rules path
        ns, r_rules, done_rules, player = step_state(state_before, legal[a], rng)

        assert r_env == r_rules, (r_env, r_rules)
        assert done_env == done_rules, (done_env, done_rules)
        assert ns.scores == env.state.scores, (ns.scores, env.state.scores)
        assert [c.index for c in ns.table] == [c.index for c in env.state.table]
        assert ns.current_player == env.state.current_player
        assert ns.plays_this_deal == env.state.plays_this_deal
        if done_env:
            break

    print("step_state matches env.step")