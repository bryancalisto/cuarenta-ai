"""
Tests for determinize.py / Determinizer.

The determinizer is where ISMCTS turns one information set into concrete worlds: the root
player's own hand is fixed, everything they cannot see (the other three hands and the deck)
is filled in from the cards that are left. Two things can go wrong there, and both fail
SILENTLY:

  * a LEAK -- a card the root player cannot see is placed somewhere the search can read, or
    the root's own hand is disturbed, so the search reasons about a world that is
    inconsistent with what the player actually knows;
  * a BOOKKEEPING slip -- 39 cards, a duplicated card, or a hand one card too big. The
    search still runs; it just explores positions that cannot exist, and the only symptom
    is that it plays a little worse.

Neither would raise anywhere, so they are pinned here, over whole games (which exercises
redeals and new rounds) and for every seat.

Run:  python test_determinizer.py
"""

import random

from card import build_deck
from env import HAND_SIZE, NUM_PLAYERS, CuarentaEnv, CuarentaState
from determinize import Determinizer

ALL_CARDS = build_deck()
CARD_INDEX = {card.index for card in ALL_CARDS}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _known_to_root(state: CuarentaState, root: int) -> set:
    """
    Every card the root player can account for: their own hand, the face-up table, and
    both capture piles (captured cards are public).

    This is the complement of what the determinizer is allowed to invent.
    """
    return (set(state.hands[root]) | set(state.table)
            | set(state.cards_captured[0]) | set(state.cards_captured[1]))


def _assert_consistent_with_info_set(sample: CuarentaState, truth: CuarentaState, root: int):
    """The full contract of one determinization."""
    # 1. public fields are copied verbatim -- the sample must describe the SAME position
    assert sample.table == truth.table, "table changed"
    assert sample.cards_captured[0] == truth.cards_captured[0], "team 0 captures changed"
    assert sample.cards_captured[1] == truth.cards_captured[1], "team 1 captures changed"
    assert list(sample.scores) == list(truth.scores), "scores changed"
    assert sample.current_player == truth.current_player, "current player changed"
    assert sample.last_placed == truth.last_placed, "last_placed changed"
    assert list(sample.plays_this_deal) == list(truth.plays_this_deal), "turn counts changed"
    assert len(sample.deck) == len(truth.deck), "deck size changed"
    assert sample.done == truth.done and sample.winner == truth.winner

    # 2. the root's own hand is exactly the real one, in the same order
    assert sample.hands[root] == truth.hands[root], "the root player's hand was disturbed"

    # 3. every other hand has the size the turn counters imply
    for p in range(NUM_PLAYERS):
        if p == root:
            continue
        expected = HAND_SIZE - truth.plays_this_deal[p]
        assert len(sample.hands[p]) == expected, (
            f"seat {p}: sampled hand has {len(sample.hands[p])} cards, "
            f"HAND_SIZE - plays_this_deal = {expected}")

    # 4. card conservation: 40 cards, each exactly once
    seen = []
    for p in range(NUM_PLAYERS):
        seen += list(sample.hands[p])
    seen += list(sample.table) + list(sample.deck)
    seen += list(sample.cards_captured[0]) + list(sample.cards_captured[1])
    assert len(seen) == 40, f"{len(seen)} cards in the sampled state, expected 40"
    assert len({c.index for c in seen}) == 40, "a card appears twice in the sampled state"
    assert {c.index for c in seen} == CARD_INDEX, "the sampled state is not a full deck"

    # 5. nothing the root can see was handed to somebody else
    known = _known_to_root(truth, root)
    for p in range(NUM_PLAYERS):
        if p == root:
            continue
        overlap = known & set(sample.hands[p])
        assert not overlap, (
            f"seat {p} was dealt cards the root player can see: "
            f"{sorted(c.index for c in overlap)}")

    # 6. the hidden cards are exactly the ones that were filled in
    hidden = set(ALL_CARDS) - known
    filled = set()
    for p in range(NUM_PLAYERS):
        if p != root:
            filled |= set(sample.hands[p])
    filled |= set(sample.deck)
    assert filled == hidden, (
        "the cards filled in are not exactly the cards the root player cannot see "
        f"(missing {sorted(c.index for c in hidden - filled)}, "
        f"invented {sorted(c.index for c in filled - hidden)})")


def _fresh(seed: int):
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_public_state_and_conservation():
    """One position, all four seats, many samples each."""
    env = _fresh(11)
    rng = random.Random(0)
    state = env.state
    for root in range(NUM_PLAYERS):
        determinizer = Determinizer(env, root)
        for _ in range(25):
            _assert_consistent_with_info_set(determinizer.sample(rng), state, root)
    print("test_public_state_and_conservation: OK (4 seats x 25 samples)")


def test_root_hand_is_fixed_and_the_rest_varies():
    """
    The root's hand must never move; the hidden hands must actually be resampled.

    A determinizer that returned the true state (or always the same world) would pass the
    consistency checks while giving the search no distribution to reason over.
    """
    env = _fresh(29)
    rng = random.Random(1)
    state = env.state
    root = 1
    determinizer = Determinizer(env, root)

    samples = [determinizer.sample(rng) for _ in range(30)]
    for sample in samples:
        assert sample.hands[root] == state.hands[root]

    # the three hidden hands, as tuples, should not all be identical across 30 draws
    worlds = {tuple(sorted(c.index for p in range(NUM_PLAYERS)
                           if p != root for c in sample.hands[p]))
              for sample in samples}
    assert len(worlds) > 1, (
        "30 determinizations produced a single assignment; the determinizer is not "
        "sampling the hidden cards")
    print(f"test_root_hand_is_fixed_and_the_rest_varies: OK "
          f"({len(worlds)} distinct hidden-card assignments in 30 draws)")


def test_determinism_for_a_seeded_rng():
    """Same seed -> same world. The search relies on being reproducible."""
    a = Determinizer(_fresh(37), 2)
    b = Determinizer(_fresh(37), 2)
    first = a.sample(random.Random(1234))
    second = b.sample(random.Random(1234))
    for p in range(NUM_PLAYERS):
        assert first.hands[p] == second.hands[p], f"seat {p} differs between two identical runs"
    assert first.deck == second.deck
    print("test_determinism_for_a_seeded_rng: OK")


def test_invariants_hold_across_whole_games():
    """
    Every ply of several full games, including the redeal and new-round boundaries.

    This is the test that matters: hand sizes and the turn counters are only consistent
    within a deal, and the determinizer asserts that relationship, so a boundary handled
    wrongly shows up here rather than as a slightly weaker search.
    """
    rng = random.Random(7)
    checked = 0
    redeals = 0
    for game in range(3):
        env = _fresh(1000 + game)
        previous_deck = len(env.state.deck)
        plies = 0
        while not env.state.done and plies < 600:
            root = env.state.current_player
            legal = env.legal_actions()
            determinizer = Determinizer(env, root)
            truth = env.state
            for _ in range(3):
                _assert_consistent_with_info_set(determinizer.sample(rng), truth, root)
            checked += 1
            if len(env.state.deck) > previous_deck:
                redeals += 1
            previous_deck = len(env.state.deck)
            env.step(rng.randrange(len(legal)))
            plies += 1
    assert checked > 150, f"only {checked} positions checked; the games ended too early"
    assert redeals > 0, "no redeal was exercised; the test is not covering the boundary"
    print(f"test_invariants_hold_across_whole_games: OK "
          f"({checked} positions, {redeals} redeals)")


def test_sampled_states_are_playable():
    """
    A sampled world must be a legal position for the rules engine, not just consistent on
    paper: it should offer legal actions and accept a step without asserting.
    """
    rng = random.Random(3)
    env = _fresh(43)
    root = env.state.current_player
    determinizer = Determinizer(env, root)
    for _ in range(10):
        sampled = determinizer.sample(rng)
        from rules import legal_actions_from_state, step_state
        legal = legal_actions_from_state(sampled)
        assert legal, "a sampled world offered no legal action"
        step_state(sampled, legal[rng.randrange(len(legal))], rng)
    print("test_sampled_states_are_playable: OK")


def test_end_of_deal_with_nothing_hidden():
    """
    Edge case: the deck is empty and the other hands are empty, so the root player can
    account for every card and there is nothing left to sample.

    The determinizer must accept it (sizes 0, empty unknown pool) rather than dividing by
    zero or asserting.
    """
    deck = build_deck()
    root = 0
    root_hand = deck[:5]
    table = [deck[5]]
    captured = deck[6:]
    assert len(root_hand) + len(table) + len(captured) == 40
    state = CuarentaState(
        hands={0: list(root_hand), 1: [], 2: [], 3: []},
        table=list(table),
        deck=[],
        scores=[10, 12],
        cards_captured=[list(captured), []],
        current_player=root,
        last_placed=table[0],
        plays_this_deal=[0, HAND_SIZE, HAND_SIZE, HAND_SIZE],
        done=False,
        winner=None,
    )
    env = CuarentaEnv(seed=0)
    env.state = state
    determinizer = Determinizer(env, root)
    sample = determinizer.sample(random.Random(0))
    _assert_consistent_with_info_set(sample, state, root)
    print("test_end_of_deal_with_nothing_hidden: OK (empty unknown pool)")


def test_a_desynced_turn_counter_is_rejected():
    """
    The determinizer sizes hands from `plays_this_deal`. If those disagree with the real
    hands the sample would be silently wrong, so it asserts instead - this pins that
    guard, because the alternative is a search over impossible worlds.
    """
    env = _fresh(53)
    root = env.state.current_player
    other = (root + 1) % NUM_PLAYERS
    env.state.plays_this_deal[other] = HAND_SIZE - len(env.state.hands[other]) + 1
    try:
        Determinizer(env, root)
    except AssertionError:
        print("test_a_desynced_turn_counter_is_rejected: OK (asserted)")
        return
    raise AssertionError("a desynced plays_this_deal was accepted")


if __name__ == "__main__":
    test_public_state_and_conservation()
    test_root_hand_is_fixed_and_the_rest_varies()
    test_determinism_for_a_seeded_rng()
    test_invariants_hold_across_whole_games()
    test_sampled_states_are_playable()
    test_end_of_deal_with_nothing_hidden()
    test_a_desynced_turn_counter_is_rejected()
    print("\nAll determinizer tests passed.")
