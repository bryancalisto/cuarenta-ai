#!/usr/bin/env python3
"""
Write a JSON fixture of real Cuarenta positions plus the exact policy inputs Python
derives from them, so the Go backend can be checked against the reference implementation
instead of only against itself.

Why this exists: `internal/game` builds the observation, the candidate-action features,
the legality mask and the legal-action ORDER that the ONNX model consumes. Its own tests
pin those layouts against Go constants, so they catch a Go-side regression -- but they
cannot catch a disagreement with Python, which is what actually trained the model. A
different card-index convention, a hand iterated in another order, `last_placed`
normalised differently or a rescaled score would leave every Go test passing while the
AI quietly played a worse game: the model always receives a well-formed vector, just not
the one it was trained on.

So this script dumps, for a spread of positions taken from real games:

  * the full game state, in a form `game.State` can be rebuilt from field by field
    (cards as [suit_name, rank], so each side maps them through its OWN enum), and
  * what `src/ppo/env.py` computes for that state: the 208-dim observation, the legal
    actions IN ORDER, the [64, 84] action-feature block and the [64] mask.

`backend/internal/game/state_parity_test.go` rebuilds each position in Go and compares.

Beyond sampling positions evenly across each game, the generator scans the whole game and
forces the fixture to contain the cases that matter: a limpia (capture the whole table
below 38), a caida, a multi-card capture and an empty table. Those are the branches where
the two implementations are most likely to disagree, and a pure even sample can miss them
(they are a few percent of decisions).

Usage:
    python3 scripts/state_parity_fixture.py
    python3 scripts/state_parity_fixture.py --games 3 --per-game 8 --out <path>
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "ppo"))

from card import Card                                            # noqa: E402
from env import (ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv,      # noqa: E402
                 action_features_matrix)
from utils import OBS_KEYS_ISMCTS, OBS_SIZE_ISMCTS, flatten_obs  # noqa: E402

# Feature-block indices (see the ACTION_FEAT_DIM comment in src/ppo/env.py).
FLAG_IS_CAPTURE = 80
FLAG_IS_CAIDA = 81
FLAG_IS_LIMPIA = 82

# Branches the fixture must exercise, cheapest-to-check first.
INTERESTING = ("limpia", "caida", "multi_capture", "empty_table")


def card_json(card: Card):
    """Cards travel as (suit name, rank) so neither side depends on the other's order."""
    return [card.suit.name, int(card.rank)]


def cards_json(cards):
    return [card_json(c) for c in cards]


def state_json(state):
    return {
        "hands": [cards_json(state.hands[p]) for p in range(4)],
        "table": cards_json(state.table),
        "deck": cards_json(state.deck),
        "scores": [int(s) for s in state.scores],
        "cards_captured": [cards_json(state.cards_captured[0]),
                           cards_json(state.cards_captured[1])],
        "current_player": int(state.current_player),
        "last_placed": card_json(state.last_placed) if state.last_placed else None,
        "plays_this_deal": [int(x) for x in state.plays_this_deal],
        "decks_consumed": int(state.n_decks_consumed),
        "done": bool(state.done),
    }


def legal_json(legal):
    return [
        {
            "card": card_json(action.card),
            # Sorted by card index: a Frozenset has no order of its own, so both sides
            # must be told which order to compare in.
            "capture": cards_json(sorted(action.capture)) if action.capture else None,
        }
        for action in legal
    ]


def features_of(env):
    """The authoritative per-action features, exactly as the training code builds them."""
    legal = env.legal_actions()
    feats, mask = action_features_matrix(legal, env.state)
    feats = np.asarray(feats, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    return legal, feats, mask


def properties_of(env, legal, feats, mask) -> set:
    """Which interesting branches this decision offers."""
    props = set()
    n_legal = int(mask.sum())
    for i in range(n_legal):
        row = feats[i]
        if row[FLAG_IS_LIMPIA] > 0.5:
            props.add("limpia")
        if row[FLAG_IS_CAIDA] > 0.5:
            props.add("caida")
        capture = legal[i].capture
        if capture and len(capture) > 1:
            props.add("multi_capture")
    if not env.state.table:
        props.add("empty_table")
    return props


def make_case(env, step: int, game: int) -> dict:
    player = env.state.current_player
    legal, feats, mask = features_of(env)
    obs = flatten_obs(env.observe(player))
    if len(obs) != OBS_SIZE_ISMCTS:
        raise SystemExit(
            f"{env.state.current_player}: flatten_obs produced {len(obs)} values, "
            f"expected OBS_SIZE_ISMCTS={OBS_SIZE_ISMCTS}"
        )
    return {
        "name": f"g{game}_step{step}_p{player}",
        "game": game,
        "step": step,
        "player": player,
        "state": state_json(env.state),
        "expected_obs": [float(x) for x in obs],
        "expected_legal": legal_json(legal),
        "expected_action_features": [float(x) for x in feats.reshape(-1)],
        "expected_action_mask": [float(x) for x in mask.reshape(-1)],
    }


def replay_game(seed: int, on_step) -> int:
    """
    Play one deterministic game (same seed -> same deals AND same random action choices),
    calling `on_step(step, env, legal)` at every decision. Returns the decision count.

    Every pass over a game uses this, so the passes see the identical game and a position
    captured in a later pass is reachable from the state it describes.
    """
    env = CuarentaEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)
    step = 0
    while not env.state.done:
        legal = env.legal_actions()
        if not legal:
            break
        on_step(step, env, legal)
        env.step(legal[rng.randrange(len(legal))])
        step += 1
    return step


def case_properties(case) -> set:
    """Which interesting branches a generated case exercises."""
    block = np.asarray(case["expected_action_features"],
                       dtype=np.float32).reshape(MAX_ACTIONS, ACTION_FEAT_DIM)
    props = set()
    if block[:, FLAG_IS_LIMPIA].max() > 0.5:
        props.add("limpia")
    if block[:, FLAG_IS_CAIDA].max() > 0.5:
        props.add("caida")
    if any(a["capture"] and len(a["capture"]) > 1 for a in case["expected_legal"]):
        props.add("multi_capture")
    if not case["state"]["table"]:
        props.add("empty_table")
    return props


def build_cases(games: int, per_game: int, seed: int):
    cases = []

    for game in range(games):
        game_seed = seed + game

        # Pass 1: how long is this game? Every pass is deterministic, so the later passes
        # replay the very same game.
        total_steps = replay_game(game_seed, lambda step, env, legal: None)
        if total_steps <= 0:
            continue

        # Pass 2: sample `per_game` decisions spread over the WHOLE game -- later deals
        # and end-game scores are where the observation layout actually differs, so
        # sampling only the opening would miss exactly the bugs this fixture is for.
        count = min(per_game, total_steps)
        targets = ({0} if count == 1
                   else {int(round(i * (total_steps - 1) / (count - 1))) for i in range(count)})

        def spread(step, env, legal, _game=game, _targets=targets):
            if step in _targets:
                cases.append(make_case(env, step, _game))

        replay_game(game_seed, spread)

        # Pass 3: the even sample above can miss the rare-but-important branches (a limpia
        # is only a few percent of decisions), so walk the game and pin the first decision
        # offering each branch the fixture does not have yet.
        covered = set()
        for case in cases:
            covered |= case_properties(case)

        missing = {p for p in INTERESTING if p not in covered}
        if missing:
            found = set()

            def scan(step, env, legal, _game=game, _missing=missing, _found=found):
                _, feats, mask = features_of(env)
                props = properties_of(env, legal, feats, mask)
                newly = (props & _missing) - _found
                if newly:
                    _found |= newly
                    cases.append(make_case(env, step, _game))

            replay_game(game_seed, scan)

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--per-game", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20240)
    ap.add_argument("--out", default=str(REPO / "backend" / "internal" / "game"
                                        / "testdata" / "state_fixture.json"))
    args = ap.parse_args()

    cases = build_cases(args.games, args.per_game, args.seed)
    if len(cases) < 4:
        raise SystemExit(f"only built {len(cases)} cases; lower --per-game")

    payload = {
        "generator": "scripts/state_parity_fixture.py",
        "obs_keys": list(OBS_KEYS_ISMCTS),
        # The flattened width, NOT len(obs_keys): there are 10 keys but 208 floats.
        "obs_dim": OBS_SIZE_ISMCTS,
        "action_feat_dim": ACTION_FEAT_DIM,
        "n_slots": MAX_ACTIONS,
        "cases": cases,
    }

    # Refuse to write a fixture that cannot catch the interesting disagreements.
    def any_flag(index):
        return [np.asarray(c["expected_action_features"], dtype=np.float32)
                .reshape(MAX_ACTIONS, ACTION_FEAT_DIM)[:, index].max() for c in cases]

    if max(any_flag(FLAG_IS_CAPTURE)) <= 0.5:
        raise SystemExit("degenerate fixture: no case offers a capture")
    if max(any_flag(FLAG_IS_LIMPIA)) <= 0.5:
        raise SystemExit("degenerate fixture: no case offers a limpia; raise --games")
    if max(any_flag(FLAG_IS_CAIDA)) <= 0.5:
        raise SystemExit("degenerate fixture: no case offers a caida; raise --games")
    if not any(len(c["expected_legal"]) < MAX_ACTIONS for c in cases):
        raise SystemExit("degenerate fixture: every case fills all slots")
    if not any(c["state"]["table"] for c in cases):
        raise SystemExit("degenerate fixture: no case has cards on the table")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

    counts = {p: sum(1 for c in cases if p in case_properties(c)) for p in INTERESTING}
    print(f"[fixture] wrote {out} ({len(cases)} cases)")
    print("[fixture] coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for c in cases:
        print(f"[fixture]   {c['name']:16s} n_legal={len(c['expected_legal']):2d} "
              f"table={len(c['state']['table'])} scores={c['state']['scores']} "
              f"plays_this_deal={c['state']['plays_this_deal']}")


if __name__ == "__main__":
    main()
