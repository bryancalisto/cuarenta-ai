"""
The env's reward, and the copy of it that lives in the search.

`CuarentaEnv.step` (the real environment) and `rules.step_state` (the pure transition
the search steps through) both pay the reward, and they MUST agree: if they drift, then

  * anything that bootstraps from `step_state`'s reward disagrees with anything that
    reads the env's (this has happened TWICE, both times on the cartón term: once paid in
    `step_state` but not in `env.step`, and once re-enabled in `env.step` but not in
    `step_state`), and
  * `GameResult.accounted_return` — the identity behind the `selfplay/return_*` curves —
    stops being checkable.

In the DEFAULT (`--value-target win`) the search DISCARDS the reward -- it backs up ±1
game outcomes, see the VALUE TARGET note in selfplay.py -- which is exactly why a drift
here is silent. In `--value-target return` the same reward stream is accumulated into the
backup, so there a drift is not a logging detail: the search would be blind to whatever
term the env pays and `step_state` does not. These tests
are what make it loud.

Run: python test_reward_parity.py
"""

import random

import numpy as np

from captures import CAIDA_POINTS, LIMPIA_POINTS
from env import WIN_BONUS, CuarentaEnv, TEAMS
from rules import clone_state, step_state
from utils import PointsReasons, has_reason


# Fixed reward constants of the CURRENT design. If a reward change is intentional,
# these lines (and `GameResult.accounted_return`) are what you update.
CAIDA_REWARD = CAIDA_POINTS
LIMPIA_REWARD = LIMPIA_POINTS


def _expected_step_reward(env, action, info, done, team):
    """
    The reward the docs promise for one step: +1 per captured card, +2 caída, +2 limpia,
    the cartón points when a deal ends, and +WIN_BONUS on the winning move (see the REWARD
    note in env.py).
    """
    reward = 0.0
    if action.capture:
        reward += len(action.capture)
        reasons = info.get("reasons") or []
        if has_reason(reasons, PointsReasons.CAIDA):
            reward += CAIDA_REWARD
        if has_reason(reasons, PointsReasons.LIMPIA):
            reward += LIMPIA_REWARD
    # `carton_points` is only set on the ply that ends a deal, and it is the acting team's.
    reward += float(info.get("carton_points") or 0.0)
    if done and env.state.winner == team:
        reward += WIN_BONUS
    return reward


def _same_state(a, b) -> bool:
    """Structural equality of two CuarentaStates (all fields the search can read)."""
    return (
        {p: sorted(h) for p, h in a.hands.items()} == {p: sorted(h) for p, h in b.hands.items()}
        and sorted(a.table) == sorted(b.table)
        and sorted(a.deck) == sorted(b.deck)
        and a.scores == b.scores
        and [sorted(c) for c in a.cards_captured] == [sorted(c) for c in b.cards_captured]
        and a.current_player == b.current_player
        and a.plays_this_deal == b.plays_this_deal
        and a.last_placed == b.last_placed
        and a.done == b.done
        and a.winner == b.winner
    )


def test_step_state_reward_matches_env_step():
    """
    Drive the SAME game twice — once through `env.step`, once through `step_state` — and
    require the reward, the done flag, the acting player and the resulting state to match
    at every single ply, including deal boundaries and cartón payments.

    Both RNGs are seeded identically so the reshuffles at a new deal / new round consume
    the same stream; the rules transition must therefore be a drop-in replacement for the
    env (which is what the search assumes).
    """
    for seed, rng_seed in ((970_000, 41), (970_001, 42)):
        env = CuarentaEnv(seed=seed)
        env.reset(seed=seed)
        # Take control of the env's deck RNG so it is reproducible and identical to the
        # one handed to `step_state`.
        env.random = random.Random(rng_seed)
        state = clone_state(env.state)
        rng = random.Random(rng_seed)

        plies = 0
        carton_steps = 0
        while not env.state.done:
            slot = 0                                  # any legal slot will do
            action = env.legal_actions()[slot]
            player = env.state.current_player
            team = TEAMS[player]

            obs, reward, done, _trunc, info = env.step(slot)
            state, r_reward, r_done, r_player = step_state(state, action, rng)

            expected = _expected_step_reward(env, action, info, done, team)
            assert abs(reward - expected) < 1e-9, (
                f"seed={seed} ply={plies}: env paid {reward}, the documented reward is "
                f"{expected} (action={action}, info={info})"
            )
            assert abs(reward - r_reward) < 1e-9, (
                f"seed={seed} ply={plies}: env.step paid {reward} but rules.step_state "
                f"paid {r_reward} — the search's transition and the env have drifted "
                f"(action={action}, info={info})"
            )
            assert done == r_done and player == r_player, (seed, plies, done, r_done)
            assert _same_state(env.state, state), (
                f"seed={seed} ply={plies}: state diverged between env.step and "
                f"step_state\n  env:   {env.state}\n  rules: {state}"
            )

            if "carton_points" in info:
                carton_steps += 1
                assert info["carton_points"] >= 0
                # The cartón IS paid as reward, by BOTH implementations (this is the exact
                # term that has drifted between them twice: once when it was removed from
                # `env.step` only, once when it was added back to `env.step` only).

            plies += 1
            assert plies < 1000, "game did not terminate"

        assert plies > 40, f"seed={seed}: only {plies} plies, the game looks truncated"
        assert carton_steps >= 1, (
            f"seed={seed}: no cartón step in the whole game, so this test never exercised "
            "the end-of-deal branch where the two rewards disagreed"
        )
        print(f"test_step_state_reward_matches_env_step: OK (seed {seed}, {plies} plies, "
              f"{carton_steps} cartón steps)")


def test_carton_is_scored_and_rewarded():
    """
    A cartón must both move the SCORE and pay the acting team the same number of reward
    points — checked directly on a crafted deal-ending position, so the two cannot silently
    diverge (one of them is what the charts show, the other is what the agent optimises).
    """
    from card import Card, Rank, Suit, build_deck
    from env import NUM_PLAYERS

    env = CuarentaEnv(seed=980_000)
    env.reset(seed=980_000)

    # Craft: only one card left in one hand, an exhausted deck, and 20 captured cards
    # (>= 20 is the cartón threshold) for team 0.
    s = env.state
    team0 = [c for c in build_deck()][:20]
    s.cards_captured = [list(team0), []]
    s.deck = []
    s.scores = [0, 0]
    last = Card(Rank.CINCO, Suit.CORAZONES_ROJOS)
    s.hands = {p: [] for p in range(NUM_PLAYERS)}
    s.hands[s.current_player] = [last]
    s.table = []
    s.last_placed = None
    env._legal_cache = None

    carton_before = s.scores[0]
    _obs, reward, done, _trunc, info = env.step(0)
    del done
    assert "carton_points" in info, f"expected a cartón at the end of this deal: {info}"
    assert info["carton_points"] > 0, info
    assert s.scores[0] == carton_before + info["carton_points"], (
        f"cartón must be added to the score: {s.scores[0]} vs {carton_before} + "
        f"{info['carton_points']}"
    )
    # Nothing was captured and no bonus was earned, so the reward must be exactly the
    # cartón it paid into the score.
    assert reward == info["carton_points"] > 0, (
        f"the cartón must be paid as reward exactly like it is scored: reward={reward}, "
        f"carton_points={info['carton_points']}"
    )
    print("test_carton_is_scored_and_rewarded: OK "
          f"(cartón={info['carton_points']} score points AND reward points)")


def test_value_target_is_the_outcome_not_the_return():
    """
    In the DEFAULT mode the value head learns ±1 game outcomes, NOT the env return (see the
    VALUE TARGET note in selfplay.py: the search's negamax backup needs a zero-sum value).

    `--value-target return` is the opt-in alternative and is tested in test_value_target.py;
    what this test pins is that the DEFAULT did not drift, and that the return really is a
    different quantity (large, positive for BOTH teams — which is why it cannot be the
    value of a negamax search without reshaping Q).
    """
    import torch
    from network import CuarentaNet
    from selfplay import play_one_game

    net = CuarentaNet().eval()
    decisions, result = play_one_game(net, n_iter=8, seed=990_000, device="cpu")
    for d in decisions:
        expected = 1.0 if d.team == result.winner else -1.0
        assert d.outcome == expected, (d.outcome, expected)
        assert d.outcome in (-1.0, 1.0), "the value target must stay ±1"

    # And the two quantities really are different: the return is large and positive for
    # BOTH teams (the reward is not zero-sum), which is precisely why it cannot be the
    # value target of a negamax search.
    assert all(r > 0 for r in result.returns), result.returns
    assert max(result.returns) > 1.0, result.returns
    assert abs(result.accounted_return(0) - result.returns[0]) < 1e-6
    assert abs(result.accounted_return(1) - result.returns[1]) < 1e-6
    print("test_value_target_is_the_outcome_not_the_return: OK "
          f"(returns {result.returns[0]:.0f}/{result.returns[1]:.0f}, outcomes ±1)")


if __name__ == "__main__":
    test_step_state_reward_matches_env_step()
    test_carton_is_scored_and_rewarded()
    test_value_target_is_the_outcome_not_the_return()
    print("All reward-parity tests passed.")
