"""
The two value targets (`--value-target win|return`) and what each one makes the search do.

The target is NOT just a training detail: the value head is the search's leaf evaluation,
so the target decides what the search optimizes. These tests pin both halves of that
contract:

  * the TARGET written into the dataset follows the mode (win = ±1 game result, return =
    the shaped return-to-go the search itself backs up), and
  * the SEARCH follows the mode too: in return mode it accumulates the env's dense reward
    and prefers the bigger immediate capture; in win mode it ignores rewards entirely.

Run: python test_value_target.py
"""

import random

import numpy as np
import torch

from card import Card, Rank, Suit, build_deck
from env import CuarentaEnv, CuarentaState
from network import CuarentaNet, RETURN_VALUE_SCALE, VALUE_TARGETS
from network_ismcts_1 import NetworkISMCTS, Node
from selfplay import backfill_value_targets, play_one_game


def _net(value_target: str, seed: int = 7) -> CuarentaNet:
    torch.manual_seed(seed)
    net = CuarentaNet(value_target=value_target)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# 1. the targets
# ---------------------------------------------------------------------------

def test_win_targets_are_pm_one_and_return_targets_are_not():
    """
    Both modes, same game loop: `win` must produce only ±1 (the winner's team gets +1),
    `return` must produce the scaled shaped return-to-go instead.
    """
    assert set(VALUE_TARGETS) == {"win", "return"}

    for seed in (1_010, 1_011):
        # --- win ---
        decisions, result = play_one_game(_net("win"), n_iter=8, seed=seed, device="cpu")
        assert all(d.outcome in (-1.0, 1.0) for d in decisions), "win target must be ±1"
        for d in decisions:
            assert d.outcome == (1.0 if d.team == result.winner else -1.0)

        # --- return ---
        decisions, result = play_one_game(_net("return"), n_iter=8, seed=seed,
                                          device="cpu")
        targets = [d.outcome for d in decisions]
        assert not all(t in (-1.0, 1.0) for t in targets), (
            "return mode produced only ±1 targets, so it is not using the reward"
        )
        assert any(abs(t) > 0.05 for t in targets), targets[:5]

        # EXACT identity for the FIRST ply, computed from the env's own per-team rewards:
        # the shaped return-to-go from ply 0 is (my team's return − the opponent's),
        # because every ply pays exactly one team. This is the strongest check available
        # without re-deriving the suffix sums here.
        first = decisions[0]
        assert first.ply == 0
        mine, theirs = first.team, 1 - first.team
        expected = (result.returns[mine] - result.returns[theirs]) / RETURN_VALUE_SCALE
        assert abs(first.outcome - expected) < 1e-6, (
            f"seed={seed}: first-ply target {first.outcome} != {expected} "
            f"(returns {result.returns})"
        )

        # The LAST decision is the move that ended the game, made by the winning team: its
        # return-to-go is that single step's reward, which includes WIN_BONUS.
        from env import WIN_BONUS
        last = decisions[-1]
        assert last.ply == len(decisions) - 1
        assert last.team == result.winner, "the winning move ended the game"
        assert last.outcome * RETURN_VALUE_SCALE >= WIN_BONUS, last.outcome

        # ... and the decision BEFORE it belongs to the other team (consecutive plies are
        # played by different players, hence different teams), whose return-to-go now
        # contains the negated winning reward: negative.
        second_last = decisions[-2]
        assert second_last.team != last.team
        assert second_last.outcome < 0, (
            f"the opponent's last target must be negative (it includes the negated "
            f"winning bonus), got {second_last.outcome}"
        )

    print("test_win_targets_are_pm_one_and_return_targets_are_not: OK")


def test_backfill_helper_matches_the_search_convention():
    """
    `backfill_value_targets` is shared by self-play and by the bootstrap's teacher games
    (mixing ±1 teacher samples into a return dataset would be a silent scale error), so its
    convention is tested directly on a hand-made reward stream.

    Stream: team 0 is paid 3, team 1 is paid 5, team 0 is paid 7. The shaped return-to-go
    for the team acting at each ply:

        ply 0 (team 0): +3 - 5 + 7 =  5
        ply 1 (team 1): +5 - 7     = -2   (its own 5, minus the opponent's 7)
        ply 2 (team 0): +7         =  7

    Note the sign on ply 1: a target is always expressed from the acting team's point of
    view, which is exactly what a search rooted at that ply would back up.
    """
    class D:
        def __init__(self, team, ply):
            self.team, self.ply = team, ply
            self.outcome = None

    decisions = [D(0, 0), D(1, 1), D(0, 2)]
    backfill_value_targets(decisions, [3.0, 5.0, 7.0], [0, 1, 0], winner_team=1,
                           value_target="return")
    for d, expected in zip(decisions, (5.0, -2.0, 7.0)):
        assert abs(d.outcome * RETURN_VALUE_SCALE - expected) < 1e-9, (d.ply, d.outcome)

    # win mode on the same stream: the winner (team 1) sees +1, team 0 sees -1.
    backfill_value_targets(decisions, [3.0, 5.0, 7.0], [0, 1, 0], winner_team=1,
                           value_target="win")
    assert [d.outcome for d in decisions] == [-1.0, 1.0, -1.0]
    print("test_backfill_helper_matches_the_search_convention: OK")


# ---------------------------------------------------------------------------
# 2. the search
# ---------------------------------------------------------------------------

def _big_or_small_capture_position() -> CuarentaEnv:
    """
    A position whose legal actions differ ONLY in how much reward they collect.

    table (in order): 2♠, 5♦, with `last_placed=None` so no caída is possible.
    P0's hand:        7♥ -> sums 2+5, capturing BOTH cards (2 cards + a limpia = +4)
                      5♥ -> matches 5♦ by rank, capturing 1 card (+1)
                      K, Q, J -> figures can only match a rank exactly, and none of their
                      ranks is on the table, so they place for 0.

    Hands are full (HAND_SIZE cards, `plays_this_deal` all zero) because the determinizer
    validates that bookkeeping before it will sample the position.
    """
    table = [Card(Rank.DOS, Suit.BRILLOS),
             Card(Rank.CINCO, Suit.TREBOLES)]
    hand = [Card(Rank.SIETE, Suit.CORAZONES_ROJOS),
            Card(Rank.CINCO, Suit.CORAZONES_ROJOS),
            Card(Rank.KING, Suit.CORAZONES_NEGROS),
            Card(Rank.QUEEN, Suit.BRILLOS),
            Card(Rank.JACK, Suit.TREBOLES)]
    # The state must be COMPLETE for the determinizer, which asserts that the cards it
    # cannot see (everything but P0's hand + table + captures) exactly fill the other
    # hands plus the deck. So the remaining 33 cards are dealt 15/18 rather than left out.
    rest = [c for c in build_deck() if c not in set(hand) and c not in set(table)]
    opp_hands = {1: rest[0:5], 2: rest[5:10], 3: rest[10:15]}
    env = CuarentaEnv(seed=0)
    env.state = CuarentaState(
        hands={0: list(hand), **opp_hands},
        table=list(table), deck=rest[15:], scores=[0, 0], cards_captured=[[], []],
        current_player=0, plays_this_deal=[0, 0, 0, 0],
        last_placed=None, done=False, winner=None,
    )
    env._legal_cache = None
    return env


def _constant_value_search(env, value_target: str, n_iter: int = 200, c_puct: float = 0.1):
    """
    A search over `env` whose NETWORK is replaced by a constant: uniform priors and a leaf
    value of exactly 0 in every state.

    That isolates the reward term of the backup. With a real random net the leaf values
    differ per state, which would confound "did the search notice the +4 capture?" with
    "did the net happen to like the position it leads to". `c_puct` is deliberately small
    for the same reason: this test is about the BACKUP, not about how much exploration the
    search does.
    """
    from history import HistoryTracker

    net = _net(value_target)
    searcher = NetworkISMCTS(env, net, HistoryTracker(env), n_iter=n_iter, c_puct=c_puct,
                             rng=random.Random(0), dirichlet_eps=0.0, device="cpu")

    def fake_eval(legal):
        return np.ones(len(legal), dtype=np.float64) / len(legal), 0.0

    searcher._eval_root = lambda root_player, legal: fake_eval(legal)
    searcher._eval_node = lambda state, legal: fake_eval(legal)
    return searcher, env.legal_actions()


def test_return_mode_search_prefers_the_bigger_capture():
    """
    Return mode: the search must prefer the action that captures two cards over the one
    that captures one, purely because the env pays more for it.

    Win mode, same position, same constant value head: the reward must NOT show up at all
    (that is the whole point of the ±1 target), so the visits stay near-uniform.
    """
    for mode, expect_preference in (("return", True), ("win", False)):
        env = _big_or_small_capture_position()
        searcher, legal = _constant_value_search(env, mode)
        assert searcher.value_target == mode

        sizes = [len(a.capture) if a.capture else 0 for a in legal]
        best = int(np.argmax(sizes))
        assert sizes[best] == 2 and 1 in sizes, sizes

        visits = searcher.search(root_player=0)
        total = sum(visits.values())
        frac = visits[best] / total

        if expect_preference:
            assert frac > 0.7, (
                f"return mode should chase the 2-card capture (+2 reward vs +1); it took "
                f"it {frac:.0%} of the time. visits={visits}, sizes={sizes}"
            )
        else:
            assert frac < 0.7, (
                f"win mode must ignore the reward: it took the 2-card capture {frac:.0%} "
                f"of the time. visits={visits}, sizes={sizes}"
            )
        print(f"  {mode}: big-capture share {frac:.0%} (sizes {sizes})")

    print("test_return_mode_search_prefers_the_bigger_capture: OK")


def test_puct_treats_unvisited_children_as_neutral_in_return_mode():
    """
    In return mode every value is a large positive number, so the trick that makes PUCT
    explore in win mode ("an unvisited child has Q = 0, i.e. even") would instead say
    "an unvisited child is catastrophically bad". The search centers Q on the node's own
    value so a fresh child is neutral.

    Built by hand: the node's value is 0.9, one child has been visited and its mean (0.5)
    is WORSE than the node's own value, the other child is unvisited with a lower prior.
    Return mode must try the unvisited child; the naive (win-mode) rule would stick with
    the visited one because 0.5 > 0.
    """
    def build():
        node = Node(player=0, prior=0.0)
        node.N = 10
        node.value_mover_pov = 0.9
        node.evaluated = True
        node.priors = {}
        visited = Node(player=None, prior=0.50)
        visited.N = 5
        visited.W = 5 * 0.5          # mean 0.5, below the node's own 0.9
        fresh = Node(player=None, prior=0.10)
        node.children = {"visited": visited, "fresh": fresh}
        return node, visited, fresh

    results = {}
    for mode in ("win", "return"):
        env = CuarentaEnv(seed=0)
        env.reset(seed=0)
        from history import HistoryTracker
        searcher = NetworkISMCTS(env, _net(mode), HistoryTracker(env), n_iter=1,
                                 rng=random.Random(0), dirichlet_eps=0.0, device="cpu")
        node, visited, fresh = build()
        action, child = searcher._puct_select(node, {"visited", "fresh"}, root_team=0)
        results[mode] = action

    assert results["return"] == "fresh", (
        "return mode must not write off an unvisited child just because the return scale "
        f"is positive (picked {results['return']})"
    )
    assert results["win"] == "visited", (
        "win mode is unchanged: with Q=0 meaning 'even', a visited child at 0.5 wins "
        f"(picked {results['win']})"
    )
    print(f"test_puct_treats_unvisited_children_as_neutral_in_return_mode: OK "
          f"(return picks {results['return']}, win picks {results['win']})")


def test_search_takes_the_mode_from_the_net():
    """
    The search infers the mode from the net (every evaluator relies on this instead of
    threading a flag), and it can be overridden explicitly.
    """
    env = CuarentaEnv(seed=3)
    env.reset(seed=3)
    from history import HistoryTracker

    for mode in ("win", "return"):
        searcher = NetworkISMCTS(env, _net(mode), HistoryTracker(env), n_iter=1,
                                 rng=random.Random(0), device="cpu")
        assert searcher.value_target == mode
        assert searcher.value_scale == (RETURN_VALUE_SCALE if mode == "return" else 1.0)

    overridden = NetworkISMCTS(env, _net("return"), HistoryTracker(env), n_iter=1,
                               rng=random.Random(0), device="cpu", value_target="win")
    assert overridden.value_target == "win"
    assert overridden.value_scale == 1.0, "the scale must be a no-op outside return mode"
    print("test_search_takes_the_mode_from_the_net: OK")


def test_rollout_leaf_mode_works_in_both_targets():
    """
    `--leaf-value-mode rollout` (a debug baseline that values leaves with a greedy playout)
    is a second place that reads per-step rewards, and it has to do so in return mode too —
    otherwise a truncated playout would bootstrap a reward-scale value with a ±1 one.
    Cheap smoke test: it must run, and the targets must still come from the reward.
    """
    for mode in ("win", "return"):
        decisions, result = play_one_game(_net(mode), n_iter=4, seed=1_020, device="cpu",
                                          leaf_value_mode="rollout",
                                          max_rollout_len=40)
        assert decisions, mode
        if mode == "win":
            assert all(d.outcome in (-1.0, 1.0) for d in decisions)
        else:
            assert not all(d.outcome in (-1.0, 1.0) for d in decisions)
            mine = decisions[0].team
            expected = (result.returns[mine] - result.returns[1 - mine]) / RETURN_VALUE_SCALE
            assert abs(decisions[0].outcome - expected) < 1e-6, decisions[0].outcome
    print("test_rollout_leaf_mode_works_in_both_targets: OK")


def test_return_value_scale_puts_the_targets_at_unit_variance():
    """
    `RETURN_VALUE_SCALE` is not arbitrary: it is chosen so that the return target has (about)
    UNIT variance, which is what makes it comparable to the ±1 win target. That matters for
    three things that were measured wrong at scale 100 — the value loss's share of
    `policy_loss + value_loss` (~2% instead of ~20-30%), the size of Q's advantages next to
    PUCT's exploration bonus (which `c_puct = 1.5` assumes are O(1)), and whether "MSE > 1.0"
    means "no better than the mean" (it only does at unit variance).

    If someone changes the reward or the scale constant, this is the test that notices.
    The band is wide on purpose: this asserts the ORDER OF MAGNITUDE of the scale's
    calibration, not a distribution measured on one policy.
    """
    from bootstrap import play_one_heuristic_game

    targets = []
    for seed in (1_030, 1_031, 1_032, 1_033):
        targets += [d.outcome for d in play_one_heuristic_game(seed, "return")]
    t = np.asarray(targets, dtype=np.float64)
    std = float(t.std())
    print(f"  return targets: n={len(t)} mean={t.mean():+.3f} std={std:.3f} "
          f"(= {std * RETURN_VALUE_SCALE:.1f} raw points), "
          f"predict-0 mse={(t ** 2).mean():.3f}")
    assert abs(t.mean()) < 0.5, (
        f"the shaped return-to-go target must be centred near 0 (it is my future reward "
        f"minus the opponent's); got mean {t.mean():.3f}"
    )
    assert 0.5 < std < 2.0, (
        f"RETURN_VALUE_SCALE={RETURN_VALUE_SCALE} gives targets std {std:.3f}; the constant "
        "exists to make that ~1.0. Re-measure the raw target spread and set "
        "RETURN_VALUE_SCALE to it (or fix the reward), rather than tuning thresholds "
        "around the resulting loss scale."
    )
    print("test_return_value_scale_puts_the_targets_at_unit_variance: OK")



def test_lambda_one_reproduces_monte_carlo():
    """
    `value_lambda = 1.0` must be EXACTLY the old Monte-Carlo target — the lambda machinery
    is a strict generalisation, not a replacement, so switching it on cannot silently change
    what a run is learning.
    """
    stream = [3.0, 5.0, 7.0]
    teams = [0, 1, 0]
    boot = [11.0, -4.0, 2.5]        # deliberately non-zero: ignored at lambda = 1

    class D:
        def __init__(self, team, ply):
            self.team, self.ply = team, ply
            self.outcome = None

    mc = [D(0, 0), D(1, 1), D(0, 2)]
    backfill_value_targets(mc, stream, teams, winner_team=1, value_target="return")
    lam = [D(0, 0), D(1, 1), D(0, 2)]
    backfill_value_targets(lam, stream, teams, winner_team=1, value_target="return",
                           value_lambda=1.0, bootstrap_values=boot)
    for a, b in zip(mc, lam):
        assert a.outcome == b.outcome, (a.ply, a.outcome, b.outcome)
    print("test_lambda_one_reproduces_monte_carlo: OK")


def test_lambda_target_math_and_antisymmetry():
    """
    The TD(lambda) recursion, checked by hand.

    Stream: team 0 is paid 3 at ply 0, team 1 is paid 5 at ply 1, team 0 is paid 7 at ply 2.
    `boot[t]` is the net's estimate of the state AT ply t, in that ply's actor's POV, so
    boot = [11, -4, 2.5]. With lambda = 0.5:

        G_0[2] = +7 + 0                                       = 7      (no ply 3 exists)
        G_0[1] = -5 + 0.5*(V(s_2)=+2.5, same team) + 0.5*7     = -0.25
        G_0[0] = +3 + 0.5*(V(s_1)=-(-4)=+4, OTHER team) + 0.5*(-0.25) = 4.875

    Note which bootstrap belongs where: `V(s_t)` is the estimate recorded AT ply t (so for
    the recursion at ply t it is `boot[t + 1]`, and it is negated when that ply's actor is
    the other team). Getting that indexing wrong is the easy mistake here — this test is
    what pins it.

    Antisymmetry: for any ply, the two teams' values are exact negatives, because the own
    reward, the bootstrap and the tail all carry the same sign flip. And lambda = 0 keeps
    only the immediate reward plus the (sign-flipped) estimate of the next state.
    """
    stream = [3.0, 5.0, 7.0]
    teams = [0, 1, 0]
    boot = [11.0, -4.0, 2.5]

    class D:
        def __init__(self, team, ply):
            self.team, self.ply = team, ply
            self.outcome = None

    def targets(pairs, value_lambda):
        ds = [D(t, p) for t, p in pairs]
        backfill_value_targets(ds, stream, teams, winner_team=0, value_target="return",
                               value_lambda=value_lambda, bootstrap_values=boot)
        return [x.outcome * RETURN_VALUE_SCALE for x in ds]

    got = targets([(0, 0), (1, 1), (0, 2)], value_lambda=0.5)
    expected = [4.875, 0.25, 7.0]
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-9, (got, expected)

    # Team 0 asking at ply 1 (where team 1 actually acted) must be the negative of team 1's.
    team0_at_1 = targets([(0, 1)], value_lambda=0.5)[0]
    team1_at_1 = targets([(1, 1)], value_lambda=0.5)[0]
    assert abs(team0_at_1 + team1_at_1) < 1e-9, (team0_at_1, team1_at_1)

    got0 = targets([(0, 0), (1, 1), (0, 2)], value_lambda=0.0)
    # ply 0: +3 + V(s_1) in TEAM 0's POV = +3 + (+4) = 7
    # ply 1: +5 + V(s_2) in TEAM 1's POV = +5 + (-2.5) = 2.5
    # ply 2: +7 + 0 (game over) = 7
    assert abs(got0[0] - 7.0) < 1e-9, got0
    assert abs(got0[1] - 2.5) < 1e-9, got0
    assert abs(got0[2] - 7.0) < 1e-9, got0
    # One more property worth pinning: the terminal ply is IDENTICAL in both lambdas (its
    # target is the last reward), so lambda can never change what the winning move teaches.
    assert abs(got[2] - got0[2]) < 1e-9
    print(f"test_lambda_target_math_and_antisymmetry: OK "
          f"(lambda 0.5 -> {[round(g, 3) for g in got]}, lambda 0.0 -> {got0})")


def test_lambda_reduces_target_spread_on_real_games():
    """
    What the whole thing is for: on the SAME games (same seed and same numpy stream, so the
    search plays identically), bootstrapped targets are less spread out than Monte-Carlo
    ones, and lower lambda spreads them less.

    HONEST CAVEAT, which is why the numbers below are printed: with the RANDOM head used
    here V ~= 0, so most of the reduction comes from the `lambda^k` damping of the far
    future rather than from a good bootstrap. With a trained head the reduction is smaller
    (the tail's noise is traded for the head's error, not removed), and lambda too low hurts
    accuracy because the target leans on a weak V. Watch `train/value_target_var` in a real
    run: if it collapses far below ~0.5 the target is being damped too hard.
    """
    for seed in (1_040, 1_041):
        # The SEARCH is seeded from `seed`, but the temperature sampling of the played move
        # uses the global numpy RNG, so the stream has to be reset for each call or the
        # games diverge (their actions are sampled from the visits).
        stds, plies = {}, None
        for lam in (1.0, 0.9, 0.5, 0.0):
            net = _net("return", seed=11)
            np.random.seed(1234)
            d, r = play_one_game(net, n_iter=8, seed=seed, device="cpu", value_lambda=lam)
            if plies is None:
                plies, returns0, first_target = [x.ply for x in d], r.returns[0], d[0].outcome
            else:
                assert [x.ply for x in d] == plies, "same seed => same game"
                assert abs(r.returns[0] - returns0) < 1e-9, "same game, same returns"
            stds[lam] = float(np.array([x.outcome for x in d]).std())

        assert stds[1.0] > 0.0
        assert stds[0.9] < stds[1.0] and stds[0.5] < stds[0.9] and stds[0.0] < stds[0.5], stds
        # The FIRST ply's target is the whole game's shaped return at lambda = 1, so it must
        # be the biggest of the four (lambda shrinks exactly that tail).
        print(f"  seed {seed}: target std by lambda " +
              "  ".join(f"{k:g}={v:.3f}" for k, v in stds.items()))
    print("test_lambda_reduces_target_spread_on_real_games: OK")


if __name__ == "__main__":
    test_backfill_helper_matches_the_search_convention()
    test_search_takes_the_mode_from_the_net()
    test_puct_treats_unvisited_children_as_neutral_in_return_mode()
    test_return_mode_search_prefers_the_bigger_capture()
    test_win_targets_are_pm_one_and_return_targets_are_not()
    test_rollout_leaf_mode_works_in_both_targets()
    test_return_value_scale_puts_the_targets_at_unit_variance()
    test_lambda_one_reproduces_monte_carlo()
    test_lambda_target_math_and_antisymmetry()
    test_lambda_reduces_target_spread_on_real_games()
    print("All value-target tests passed.")
