"""
tests.py — Exhaustive tests for Cuarenta card.py, captures.py, and env.py.

Run with:  python tests.py
"""

import sys
import random
import traceback
from typing import List
from itertools import combinations

# ─── Helpers ────────────────────────────────────────────────────────────────

passed = 0
failed = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
        failed += 1

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ─── Imports ─────────────────────────────────────────────────────────────────

from card import Card, Rank, Suit, ALL_RANKS, build_deck, shuffled_deck
from captures import valid_captures, must_capture, detect_caida, detect_ronda, score_play
from env import CuarentaEnv, Action, TEAMS, NUM_PLAYERS, HAND_SIZE, WIN_SCORE

# ─── card.py tests ───────────────────────────────────────────────────────────

section("card.py — Deck structure")

deck = build_deck()
test("Deck has exactly 40 cards", len(deck) == 40)
test("All cards are unique", len(set(deck)) == 40)
test("No 8s/9s/10s in deck (only ranks 1-7, 11, 12, 13)", all(c.value not in {8, 9} for c in deck))

# Check ranks are correct
all_values = sorted(c.value for c in deck)
expected_values = sorted([r for r in [1,2,3,4,5,6,7,11,12,13]] * 4)
test("Rank values are 1-7, 11, 12, 13 only", all_values == expected_values)

# Suit counts
for suit in Suit:
    count = sum(1 for c in deck if c.suit == suit)
    test(f"Suit {suit.name} has 10 cards", count == 10)

# Card index uniqueness
indices = [c.index for c in deck]
test("All card indices are unique (0–39)", sorted(indices) == list(range(40)))

section("card.py — Card properties")
as_brillos = Card(Rank.AS, Suit.BRILLOS)
rey_treboles = Card(Rank.REY, Suit.TREBOLES)
test("As de Brillos value = 1", as_brillos.value == 1)
test("Rey de Treboles value = 13", rey_treboles.value == 13)
test("Jack value = 11", Card(Rank.JACK, Suit.CORAZONES_ROJOS).value == 11)
test("Card str formatting works", str(as_brillos) == "AO")
test("Deck is shuffleable", len(shuffled_deck()) == 40)

# ─── captures.py tests ───────────────────────────────────────────────────────

section("captures.py — Basic matching")

def c(rank_val, suit_val=0):
    """Quick card constructor by value."""
    rank_map = {r.value: r for r in Rank}
    return Card(rank_map[rank_val], Suit(suit_val))

# Exact match: play 3, table has 3 → capture
played = c(3)
table = [c(3, 1), c(5, 2)]
caps = valid_captures(played, table)
test("Match capture: play 3 vs table [3,5] → 1 option", len(caps) == 1)
test("Match capture takes the 3", c(3,1) in list(caps[0]))

# No match: play 4, table has [1, 6]
played = c(4)
table = [c(1), c(6, 1)]
caps = valid_captures(played, table)
test("No match: play 4 vs [1,6] → no captures", len(caps) == 0)

section("captures.py — Sum captures")

# Sum: play 7, table has [3, 4] → sum=7
played = c(7)
table = [c(3), c(4, 1)]
caps = valid_captures(played, table)
test("Sum capture: play 7 vs [3,4] → 1 option", len(caps) == 1)
test("Sum capture takes both cards", frozenset([c(3), c(4,1)]) in caps)

# Sum: play 6, table has [1, 2, 3] → can sum [1,2,3] or [1,5]? [1,2,3]
played = c(6)
table = [c(1), c(2, 1), c(3, 2)]
caps = valid_captures(played, table)
test("Sum capture: play 6 vs [1,2,3] → 1+2+3=6 is valid", any(len(cap)==3 for cap in caps))

# Multiple capture options: play 5, table [1,4] and [2,3]
played = c(5)
table = [c(1), c(4, 1), c(2, 2), c(3, 3)]
caps = valid_captures(played, table)
test("Multiple sums: play 5 vs [1,4,2,3] → 2 capture options", len(caps) == 2,
     f"got {len(caps)}")

# Match + sum coexist: play 3, table has [3, 1, 2]
played = c(3)
table = [c(3, 1), c(1, 2), c(2, 3)]
caps = valid_captures(played, table)
test("Match AND sum: play 3 vs [3,1,2] → 2 options (match OR sum)", len(caps) == 2,
     f"got {len(caps)}: {caps}")

section("captures.py — Figure rules")

# Figura can only match, not be in a sum
played = c(11)  # Jack
table = [c(3), c(7, 1), c(11, 2)]
caps = valid_captures(played, table)
# Should get: exact match (10), NOT [3,7] sum — wait, played is a figure!
# Figures can't sum → only match capture
test("Figure played: only match capture, not sum", 
     all(len(cap) == 1 for cap in caps),
     f"captures: {caps}")
test("Figure played: captures matching figure", any(c(11,2) in cap for cap in caps))

# Figure on table: can only be captured by exact match
played = c(7)   # number
table = [c(11, 1), c(3, 2), c(4, 3)]   # Jack on table
caps = valid_captures(played, table)
# 7 = 3+4 (number sum), but cannot take Jack (11) by sum
cap_cards = set().union(*caps) if caps else set()
test("Figure on table: number cannot sum-capture a figure",
     c(11,1) not in cap_cards,
     f"captured figure: {c(11,1) in cap_cards}, caps={caps}")

# Figure on table, figure played: exact match
played = c(11)   # Jack
table = [c(11, 1), c(5, 2), c(6, 3)]
caps = valid_captures(played, table)
test("Figure matches figure exactly", any(c(11,1) in cap for cap in caps))

section("captures.py — Caída detection")

# Caída: last card placed was a 5, played a 5, captures that 5
played = c(5)
table = [c(5, 1), c(3, 2)]
cap = frozenset([c(5, 1)])
last_placed = c(5, 1)
test("Caída detected when last placed matches played and captured",
     detect_caida(played, cap, table, last_placed))

# No caída if last placed not in capture set
cap2 = frozenset([c(3, 2)])
test("No caída when last placed not in capture", 
     not detect_caida(played, cap2, table, last_placed))

# No caída if last placed is None
test("No caída when no last placed",
     not detect_caida(played, cap, table, None))

section("captures.py — Ronda detection")

table_full = [c(3), c(5, 1)]
cap_all = frozenset(table_full)
cap_partial = frozenset([c(3)])
test("Ronda detected when all table cards captured", detect_ronda(cap_all, table_full))
test("No ronda when partial capture", not detect_ronda(cap_partial, table_full))

section("captures.py — Score computation")

# Plain capture
# Use a table with 3 cards so capturing 2 does NOT trigger ronda
played = c(5)
table = [c(2), c(3, 1), c(6, 2)]   # 3 cards on table
cap = frozenset([c(2), c(3, 1)])    # capture 2 of them (no ronda)
pts, reasons = score_play(played, cap, table, None)
test("Score 2 for capturing 2 cards (no ronda)", pts == 2, f"got {pts}")

# Caída bonus
table2 = [c(5, 1), c(2, 2)]
cap2 = frozenset([c(5, 1)])
last2 = c(5, 1)
pts2, reasons2 = score_play(c(5), cap2, table2, last2)
test("Caída adds 2 bonus points (1 card + 2 bonus = 3)", pts2 == 3, f"got {pts2}")
test("Caída reason in reasons", any("caída" in r for r in reasons2))

# Ronda bonus
table3 = [c(3), c(4, 1)]
cap3 = frozenset(table3)
pts3, reasons3 = score_play(c(7), cap3, table3, None)
test("Ronda adds 2 bonus points (2 cards + 2 bonus = 4)", pts3 == 4, f"got {pts3}")

# No capture
pts4, _ = score_play(c(6), None, [c(1)], None)
test("No points for placing (no capture)", pts4 == 0)

# ─── env.py tests ─────────────────────────────────────────────────────────────

section("env.py — Initialization & deal")

env = CuarentaEnv(seed=42)
obs, info = env.reset()
s = env.state

test("Each player has 5 cards", all(len(s.hands[p]) == HAND_SIZE for p in range(NUM_PLAYERS)))
test("Table starts empty", len(s.table) == 0)
test("Deck has 20 remaining cards", len(s.deck) == 20, f"got {len(s.deck)}")
test("Scores start at 0", s.scores == [0, 0])
test("Player 0 starts", s.current_player == 0)
test("Game not done", not s.done)

section("env.py — Observation structure")

test("obs has my_hand key", "my_hand" in obs)
test("my_hand is 40-dim", obs["my_hand"].shape == (40,))
test("table is 40-dim", obs["table"].shape == (40,))
test("my_hand has exactly 5 ones", obs["my_hand"].sum() == HAND_SIZE, 
     f"sum={obs['my_hand'].sum()}")
test("table is all zeros (empty)", obs["table"].sum() == 0)

section("env.py — Legal actions")

legal = env.legal_actions()
test("Legal actions non-empty", len(legal) > 0)
test("Legal actions only include cards in hand", 
     all(a.card in s.hands[0] for a in legal))

# Table is empty → no captures possible → must all be place actions
test("Empty table → all actions are place (no capture)", 
     all(a.capture is None for a in legal))

section("env.py — Step: placing cards")

# Play first legal action (place on empty table)
action0 = legal[0]
obs2, reward, done, info2 = env.step(action0)
test("Step returns done=False early game", not done)
test("Reward 0 for placing", reward == 0.0, f"got {reward}")
test("Table has 1 card after place", len(env.state.table) == 1)
test("Player 0 hand has 4 cards now", len(env.state.hands[0]) == 4)
test("Current player is now 1", env.state.current_player == 1)

section("env.py — Step: capture mechanics")

# Manually set up a capture scenario
env2 = CuarentaEnv(seed=0)
env2.reset()
s2 = env2.state

# Force a known state: put a 5 on table, give player 0 a 5
s2.current_player = 0
five_brillos = Card(Rank.CINCO, Suit.BRILLOS)
five_corazones_rojos = Card(Rank.CINCO, Suit.CORAZONES_ROJOS)
s2.table = [five_corazones_rojos]
s2.hands[0] = [five_brillos, Card(Rank.DOS, Suit.BRILLOS)]

legal2 = env2.legal_actions()
# Should have a capture action for the 5
capture_actions = [a for a in legal2 if a.capture is not None]
test("Capture action available for matching card", len(capture_actions) > 0)

# Take the capture
cap_action = capture_actions[0]
obs3, reward3, done3, info3 = env2.step(cap_action)
test("Reward > 0 for capture", reward3 > 0, f"got {reward3}")
test("Table empty after capturing only card", len(env2.state.table) == 0)

section("env.py — Caída scoring in step")

env3 = CuarentaEnv(seed=1)
env3.reset()
s3 = env3.state
s3.current_player = 0

# Set up: 7 on table as last placed + another card, player plays 7 → caída but NOT ronda
seven_e = Card(Rank.SIETE, Suit.CORAZONES_NEGROS)
seven_b = Card(Rank.SIETE, Suit.TREBOLES)
extra   = Card(Rank.DOS, Suit.BRILLOS)
s3.table = [seven_e, extra]   # 2 cards so capturing just the 7 is NOT a ronda
s3.last_placed = seven_e
s3.hands[0] = [seven_b]

legal3 = env3.legal_actions()
capture3 = [a for a in legal3 if a.capture is not None]
test("Caída: capture action exists", len(capture3) > 0)

obs4, reward4, done4, info4 = env3.step(capture3[0])
test("Caída: reward = 1 (card) + 2 (caída) = 3", reward4 == 3.0, f"got {reward4}")
test("Caída reason reported", any("caída" in r for r in info4["reasons"]))

section("env.py — Ronda scoring")

env4 = CuarentaEnv(seed=2)
env4.reset()
s4 = env4.state
s4.current_player = 0

# Set up: table [3, 4], player has 7 → sum captures both → ronda
three = Card(Rank.TRES, Suit.BRILLOS)
four = Card(Rank.CUATRO, Suit.CORAZONES_ROJOS)
seven = Card(Rank.SIETE, Suit.CORAZONES_NEGROS)
s4.table = [three, four]
s4.hands[0] = [seven]

legal4 = env4.legal_actions()
capture4 = [a for a in legal4 if a.capture is not None]
test("Ronda: capture action exists", len(capture4) > 0)

obs5, reward5, done5, info5 = env4.step(capture4[0])
test("Ronda: reward = 2 (cards) + 2 (ronda) = 4", reward5 == 4.0, f"got {reward5}")
test("Ronda reason reported", any("ronda" in r for r in info5["reasons"]))
test("Table empty after ronda", len(env4.state.table) == 0)

section("env.py — Must-capture rule")

env5 = CuarentaEnv(seed=3)
env5.reset()
s5 = env5.state
s5.current_player = 0

# Give player 0 a card that CAN capture — place action should NOT be in legal
match_card = Card(Rank.SEIS, Suit.BRILLOS)
same_on_table = Card(Rank.SEIS, Suit.CORAZONES_ROJOS)
s5.hands[0] = [match_card, Card(Rank.REY, Suit.TREBOLES)]
s5.table = [same_on_table]

legal5 = env5.legal_actions()
# match_card must capture, so no place action for it
place_actions_for_6 = [a for a in legal5 if a.card == match_card and a.capture is None]
test("Must-capture: no place action for card that can capture", 
     len(place_actions_for_6) == 0)
# Rey (king) can't capture anything → only place
place_actions_for_rey = [a for a in legal5 if a.card.rank == Rank.REY and a.capture is None]
test("Card that cannot capture: place action available", len(place_actions_for_rey) == 1)

section("env.py — Full game simulation (random agents)")

def random_agent_game(seed: int) -> int:
    """Play a full game with random agents. Returns winning team."""
    rng = random.Random(seed)
    env = CuarentaEnv(seed=seed)
    env.reset()
    max_steps = 10_000
    for _ in range(max_steps):
        legal = env.legal_actions()
        chosen = legal[rng.randint(0, len(legal) - 1)]
        _, _, done, info = env.step(chosen)
        if done:
            return info.get("winner", -1)
    return -1   # timeout

winners = [random_agent_game(i) for i in range(20)]
test("20 random games all complete", all(w in {0, 1} for w in winners),
     f"winners: {winners}")
test("Both teams win at least once in 20 games", 
     0 in winners and 1 in winners,
     f"winners: {winners}")

section("env.py — Render")
env_r = CuarentaEnv(seed=7)
env_r.reset()
rendered = env_r.render()
test("Render returns non-empty string", len(rendered) > 0)
test("Render contains score info", "Team 0" in rendered)
test("Render contains player hands", "P0" in rendered)

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'═'*55}")
print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
print(f"{'═'*55}\n")

if failed > 0:
    sys.exit(1)