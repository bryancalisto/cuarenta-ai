"""
Tests for the ReBeL pipeline.

The algorithmic core (belief, solver, CFR) was verified before the harness around it
existed, so these tests are about the two things that can silently break the pipeline:

  1. the CONTRACTS between the pieces -- the belief's posterior really does track the
     hidden cards as a game goes on, the search returns a distribution over the legal
     actions, the network round-trips through a checkpoint;
  2. the PROPERTIES the training targets depend on -- the search finds a winning move
     when there is one, and the belief's Bayes update actually uses the observed play.

Nothing here is weakened to make it pass: if an assertion trips, the pipeline is wrong
somewhere and the failure is the message.

Run: python3 src/rebel/test_rebel.py   (from the repository root)
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Scripts are run as `python3 src/rebel/<script>.py` from the repository root, so this
# file's own directory must be on sys.path for the bare sibling imports (see selfplay.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from belief import PublicBelief                                       # noqa: E402
from card import Card, Rank, Suit, build_deck                         # noqa: E402
from core import (flat_obs_for, CuarentaEnv, CuarentaState, TEAMS,                 # noqa: E402
                  action_features_matrix, clone_state, legal_actions_from_state,
                  score_play, step_state)
from eval_rebel import (Opponent, evaluate_league, league_members)     # noqa: E402
from heuristic_agent import HeuristicAgent                            # noqa: E402
from network import (INPUT_DIM, RebelNet, load_rebel_checkpoint,      # noqa: E402
                     save_rebel_checkpoint)
from selfplay import SelfPlayConfig, belief_policy_fn                 # noqa: E402
from solver import ReBeLSolver, kuhn_poker_solve                      # noqa: E402


# ---------------------------------------------------------------------------
# 1. the CFR core, measured against a game with a known answer
# ---------------------------------------------------------------------------

def test_kuhn_cfr_converges():
    """
    The same regret-matching MCCFR the Cuarenta solver runs, on Kuhn poker, where the
    right answers are known: the game value is -1/18 exactly, and player 0's equilibrium
    is a family of strategies parameterised by its bluff frequency `b` -- it bets a king
    three times as often as it bluffs a jack (`bet_with_king = 3 * bluff_with_jack`).

    That ratio is the sharp check. The bluff frequency alone is NOT: every member of the
    equilibrium family has the same value, so a mismatch there would not prove anything.
    (Reference measurement on this code: value -0.05530, bluff 0.0934, bet 0.2823.)
    """
    stats = kuhn_poker_solve(n_iter=20_000, seed=0)
    value = stats["value_to_player0"]
    bluff = stats["bluff_frequency_with_jack"]
    bet = stats["bet_frequency_with_king"]

    assert abs(value + 1 / 18) < 0.01, f"game value {value} != -1/18"
    assert 0.0 <= bluff <= 1 / 3 + 0.05, f"bluff frequency out of range: {bluff}"
    assert abs(bet - 3 * bluff) < 0.6, (
        f"bet_with_king {bet} is not ~3x bluff_with_jack {bluff} "
        f"(3x = {3 * bluff}); the regret updates are not converging to the "
        "equilibrium family"
    )
    print(f"test_kuhn_cfr_converges: OK (value {value:+.5f} vs -1/18, "
          f"bluff {bluff:.4f}, bet {bet:.4f}, 3*bluff {3 * bluff:.4f})")


# ---------------------------------------------------------------------------
# 2. the belief stays sane for a whole game
# ---------------------------------------------------------------------------

def test_belief_stays_consistent_over_a_game():
    """
    Drive a full game (heuristic play -- the belief's job is the same whatever the action
    source) and after EVERY ply check the invariants the search relies on:

      * the acting seat's own cards are certain (probability > 0.999): the root of a
        search must be ONE information set, and that is exactly what conditioning the
        particles on the actor's hand buys;
      * the total posterior mass equals the number of cards actually in hands (the
        ownership marginals are a distribution over cards, so their sum counts them);
      * every probability is a probability;
      * `depleted_updates` stays at 0 -- the Bayes update never had to fall back to
        uniform, i.e. no refresh disagreed with the observed action history.

    A `policy_fn` backed by an untrained network is attached on purpose: that is how the
    likelihood model is wired in self-play, and it is the path that would blow up if the
    belief asked it for probabilities the wrong way round.
    """
    torch.manual_seed(0)
    net = RebelNet()
    net.eval()

    env = CuarentaEnv(seed=5)
    env.reset(seed=5)
    rng = random.Random(0)
    # A short history window and few particles keep this test a few seconds long; the
    # invariants do not depend on either.
    belief = PublicBelief(n_particles=16, rng=rng, history=6)
    belief.policy_fn = belief_policy_fn(net, belief)
    heuristic = HeuristicAgent(seed=0)

    plies = 0
    while not env.state.done:
        state = env.state
        seat = int(state.current_player)
        legal = env.legal_actions()
        cards_in_hands = sum(len(state.hands[p]) for p in range(4))

        belief.refresh(state, seat)
        ownership = belief.ownership()

        assert belief.depleted_updates == 0, (
            f"ply {plies}: the Bayes update fell back to uniform {belief.depleted_updates} "
            f"time(s) -- the observed actions disagree with the particle sample"
        )
        for card in state.hands[seat]:
            assert ownership[seat, card.index] > 0.999, (
                f"ply {plies} seat {seat}: own card {card} is not certain "
                f"({ownership[seat, card.index]:.4f})"
            )
        assert np.all(ownership >= 0.0) and np.all(ownership <= 1.0), (
            f"ply {plies}: ownership left [0,1]: "
            f"min {ownership.min()}, max {ownership.max()}"
        )
        mass = float(ownership.sum())
        assert abs(mass - cards_in_hands) <= 1.0, (
            f"ply {plies}: ownership mass {mass:.3f} != {cards_in_hands} cards in hands"
        )

        pick = legal.index(heuristic.act(env, seat))
        belief.record_action(clone_state(state), seat, legal[pick], strategy=None)
        env.step(pick)
        plies += 1

    assert plies > 20, f"only {plies} plies; the game did not exercise the belief"
    print(f"test_belief_stays_consistent_over_a_game: OK ({plies} plies, "
          f"0 depleted updates, {len(belief.records)} recorded actions)")


# ---------------------------------------------------------------------------
# 3. the Bayes update reacts to a surprising play
# ---------------------------------------------------------------------------

def _uniform_ownership(belief: PublicBelief) -> np.ndarray:
    """Posterior marginals over the SAME particles but with flat weights (the prior)."""
    out = np.zeros((4, 40), dtype=np.float64)
    n = len(belief.particles)
    for particle in belief.particles:
        for seat in range(4):
            for card in particle.hands[seat]:
                out[seat, card.index] += 1.0 / n
    return out


def _bayes_probe(policy_fn) -> dict:
    """
    One controlled Bayes update.

    Seat 0 plays its LOWEST card while a policy that almost always plays its lowest card
    (`policy_fn`) is attached to the belief; the belief is then refreshed for seat 1, so
    seat 0 is an OPPONENT from the belief's point of view and its hand is genuinely
    uncertain. What the update should learn is that seat 0's remaining cards are high:
    particles that gave seat 0 a card below the played one are unlikely under the policy.

    Why the observer has to change for this to mean anything: when the belief is refreshed
    for the ACTOR itself, every particle already has the actor's true hand, so its own play
    is explained and the weights correctly stay uniform. And the likelihood must depend on
    the HAND, not just on a slot index -- a policy that always prefers slot 0 gives every
    particle the same probability and (again correctly) changes nothing.
    """
    rng = random.Random(0)
    env = CuarentaEnv(seed=11)
    env.reset(seed=11)

    belief = PublicBelief(n_particles=128, rng=rng, history=12)
    belief.policy_fn = policy_fn(belief)
    belief.refresh(env.state, 0)                 # observer = the actor: weights uniform

    legal = env.legal_actions()
    pick = min(range(len(legal)), key=lambda i: legal[i].card.index)
    played = legal[pick]

    belief.record_action(clone_state(env.state), 0, played, strategy=None)
    env.step(pick)
    belief.refresh(env.state, 1)                 # now seat 0 is an opponent

    weights = np.asarray(belief.weights, dtype=np.float64)
    posterior = belief.ownership()
    prior = _uniform_ownership(belief)
    # "seat 0 holds a card no higher than the one it just played" -- the belief should
    # believe this much less after the update than before it.
    cutoff = played.card.index + 1
    return {
        "ess": belief.effective_sample_size(),
        "spread": float(weights.max() / max(weights.min(), 1e-300)),
        "mass_below_before": float(prior[0, :cutoff].sum()),
        "mass_below_after": float(posterior[0, :cutoff].sum()),
        "n_particles": belief.n_particles,
        "depleted": belief.depleted_updates,
    }


def test_belief_bayes_update_reacts_to_a_surprising_play():
    """
    The belief must LEARN from an observed play, and must learn nothing when it has no
    likelihood model -- both directions are asserted, because "nothing changed" and "the
    weights moved" are equally easy to get wrong and only one of them is correct.
    """
    def lowest_card_policy(belief):
        def policy_fn(seat, state, legal):
            probs = np.full(len(legal), 0.03 / max(len(legal) - 1, 1))
            best = min(range(len(legal)), key=lambda i: legal[i].card.index)
            probs[best] = 0.97
            return probs
        return policy_fn

    informed = _bayes_probe(lowest_card_policy)
    uninformed = _bayes_probe(lambda belief: None)

    assert informed["depleted"] == 0
    assert informed["ess"] < informed["n_particles"] * 0.5, (
        f"the Bayes update left the weights nearly uniform (ESS {informed['ess']:.1f} of "
        f"{informed['n_particles']}): the observed play was not used"
    )
    assert informed["spread"] > 10.0, (
        f"weight spread {informed['spread']:.2f}x is too flat for a play the policy "
        "considered that unlikely"
    )
    assert informed["mass_below_after"] < informed["mass_below_before"], (
        f"P(seat 0 holds a card <= the played one) went from "
        f"{informed['mass_below_before']:.3f} to {informed['mass_below_after']:.3f}: the "
        "update did not infer anything about the rest of that hand"
    )

    # Without a likelihood model every action is uninformative: the weights must stay
    # exactly uniform rather than drifting for some other reason.
    assert uninformed["ess"] == uninformed["n_particles"], (
        f"with policy_fn=None the belief is not uniform (ESS {uninformed['ess']})"
    )
    assert abs(uninformed["mass_below_after"] - uninformed["mass_below_before"]) < 1e-12

    print(f"test_belief_bayes_update_reacts_to_a_surprising_play: OK "
          f"(ESS {informed['n_particles']} -> {informed['ess']:.1f}, "
          f"spread {informed['spread']:.1f}x, "
          f"P(low card held by seat 0) {informed['mass_below_before']:.3f} -> "
          f"{informed['mass_below_after']:.3f}; policy_fn=None keeps ESS "
          f"{uninformed['ess']:.0f})")


# ---------------------------------------------------------------------------
# 4. the solver returns a distribution over the legal actions
# ---------------------------------------------------------------------------

def test_solver_returns_a_distribution():
    """
    `root_probs` is the training target for the policy head, so it must be a genuine
    probability distribution over exactly the root's legal actions -- not over the 64
    slots, not unnormalised, not full of negative regrets.
    """
    torch.manual_seed(0)
    net = RebelNet()
    net.eval()

    env = CuarentaEnv(seed=3)
    env.reset(seed=3)
    rng = random.Random(0)
    belief = PublicBelief(n_particles=8, rng=rng)
    belief.policy_fn = belief_policy_fn(net, belief)

    seat = int(env.state.current_player)
    legal = env.legal_actions()
    belief.refresh(env.state, seat)

    solver = ReBeLSolver(net=net, n_iter=8, depth=3, rng=rng)
    result = solver.solve(env.state, seat, belief)
    probs = np.asarray(result.root_probs, dtype=np.float64)

    assert probs.shape == (len(legal),), (probs.shape, len(legal))
    assert np.all(probs >= 0.0), probs
    assert abs(probs.sum() - 1.0) < 1e-9, probs.sum()
    assert 0 <= int(np.argmax(probs)) < len(legal)
    assert result.strategy, "the solver returned an empty strategy table"
    for key, table in result.strategy.items():
        assert abs(float(table.sum()) - 1.0) < 1e-9, (key, table)
    assert result.iterations == 8 and result.depth == 3
    assert result.leaf_evals > 0, "the search never valued a leaf with the network"
    assert -1.0 <= result.root_value <= 1.0, result.root_value
    print(f"test_solver_returns_a_distribution: OK ({len(legal)} legal actions, "
          f"argmax {int(np.argmax(probs))}, {len(result.strategy)} info sets, "
          f"{result.leaf_evals} leaf evals, value {result.root_value:+.3f})")


# ---------------------------------------------------------------------------
# 5. the search finds a winning capture
# ---------------------------------------------------------------------------

def test_search_prefers_the_winning_capture():
    """
    IMPLEMENTED CHECK: the strong one -- the search's argmax must be the capture that
    wins the game. The position below is constructed so that the winning capture is
    reachable within the search's depth (the terminal is 1 ply away, and depth=2), so it
    passes with an UNTRAINED network: the value comes from the real terminal, not from
    the network's opinion. That makes it a test of the search, not of training.

    Team 0 sits on 38 points; seat 0 holds the three of treboles, which captures the whole
    table (a limpia, worth 2 points -> 40, the winning score) because the three of brillos
    is on the table and it is the last card placed.
    """
    torch.manual_seed(0)

    deck = build_deck()
    three_b = Card(Rank.TRES, Suit.BRILLOS)     # on the table, placed last
    three_t = Card(Rank.TRES, Suit.TREBOLES)    # in seat 0's hand: captures the whole table
    used = {three_b, three_t}
    rest = [c for c in deck if c not in used]
    hands = {0: [three_t] + rest[0:4], 1: rest[4:9], 2: rest[9:14], 3: rest[14:19]}
    state = CuarentaState(
        hands=hands,
        table=[three_b],
        deck=rest[19:],                  # 40 cards accounted for (the determinizer checks)
        scores=[38, 10],
        cards_captured=[[], []],
        current_player=0,
        last_placed=three_b,
        plays_this_deal=[0, 0, 0, 0],
        done=False,
        winner=None,
    )

    rng = random.Random(0)
    belief = PublicBelief(n_particles=8, rng=rng)
    belief.refresh(state, 0)

    solver = ReBeLSolver(net=RebelNet(), n_iter=64, depth=2, rng=rng)
    result = solver.solve(state, 0, belief)
    legal = legal_actions_from_state(state)
    chosen = legal[int(np.argmax(result.root_probs))]

    assert chosen.capture is not None and three_b in chosen.capture, (
        f"the search chose {chosen}, not the capture of the last-placed card"
    )
    assert result.root_value > 0.8, (
        f"the search values a won position at {result.root_value:+.3f}; the terminal is "
        "inside its depth, so a low value means the backup is not seeing it"
    )

    # And it really does win: apply it and check the game is over.
    child, _reward, done, _actor = step_state(state, chosen, random.Random(0))
    assert done and child.winner == 0, (done, child.winner)
    assert child.scores[0] >= 40, child.scores
    print(f"test_search_prefers_the_winning_capture: OK (played {chosen}, "
          f"p={result.root_probs[int(np.argmax(result.root_probs))]:.3f}, "
          f"value {result.root_value:+.3f} -> scores {child.scores} done={done})")


# ---------------------------------------------------------------------------
# 6. network shapes and the checkpoint round-trip
# ---------------------------------------------------------------------------

def test_value_head_prefers_scoring_captures():
    """
    The value head must rank a caida/limpia capture above a plain placement.

    This is the property the whole search rests on: `ReBeLSolver` optimises the value, so a
    head that does not know scoring is good makes the search decline caidas -- which is
    exactly the failure that was observed in play (the search picking worse actions than
    the policy head it searches from).

    It is a REGRESSION test for the value head, and it needs a trained checkpoint: an
    untrained head's outputs are near-constant, so it has no ranking to speak of. Skipped
    when `checkpoints-rebel/bootstrap.pt` is absent.

    Comparison is between the MEAN value of the capturing actions and the MEAN value of the
    placing actions. An earlier version compared the best of each set, which is biased
    whenever the two sets have different sizes -- it reported the opposite of the truth
    (86% "captures rated lower" against a real 98% "captures rated higher").
    """
    from bootstrap import DEFAULT_OUT
    if not Path(DEFAULT_OUT).exists():
        print("test_value_head_prefers_scoring_captures: SKIP (no bootstrap checkpoint)")
        return
    net = load_rebel_checkpoint(str(DEFAULT_OUT))
    net.eval()
    rng = random.Random(0)
    preferred = total = 0
    for game in range(12):
        env = CuarentaEnv(seed=2_000_000 + game)
        env.reset(seed=2_000_000 + game)
        belief = PublicBelief(n_particles=8, rng=rng)
        steps = 0
        while not env.state.done and steps < 400:
            state = env.state
            actor = state.current_player
            legal = legal_actions_from_state(state)
            caps = [a for a in legal if a.capture]
            places = [a for a in legal if not a.capture]
            if caps and places:
                best_points = max(score_play(
                    played=a.card, capture_set=a.capture, table=state.table,
                    last_placed=state.last_placed, non_caida_allowed=True)[0] for a in caps)
                if best_points > 0:                      # a caida/limpia is on offer
                    belief.refresh(state, actor)
                    features = belief.ownership().reshape(-1)
                    team = TEAMS[actor]

                    def my_value(action):
                        nxt, _r, done, _p = step_state(clone_state(state), action, rng)
                        if done:
                            return 1.0 if nxt.winner == team else -1.0
                        obs = np.concatenate([flat_obs_for(nxt.current_player, nxt), features])
                        feats, _ = action_features_matrix(
                            legal_actions_from_state(nxt), nxt)
                        # the next seat is always an opponent (partners sit two apart)
                        return -net.value_of(obs, feats)

                    capture_mean = float(np.mean([my_value(a) for a in caps]))
                    place_mean = float(np.mean([my_value(a) for a in places]))
                    total += 1
                    preferred += int(capture_mean > place_mean)
            env.step(rng.randrange(len(legal)))
            steps += 1

    assert total > 50, f"only found {total} caida/limpia positions to test"
    rate = preferred / total
    assert rate > 0.8, (
        f"the value head prefers a capture in only {rate:.1%} of positions where capturing "
        f"earns points ({preferred}/{total}). The search optimises this head, so it would "
        "decline caidas; the value targets are broken or the head is badly undertrained.")
    print(f"test_value_head_prefers_scoring_captures: OK ({preferred}/{total} = {rate:.1%})")


def test_net_shapes_and_checkpoint_roundtrip():
    """
    The output contract the trainer reads: a logit per slot, one value per row in [-1, 1]
    (it is a tanh head), illegal slots pushed to ~-1e9 so they can never be selected.
    Then a save/load round-trip must reproduce the outputs exactly -- a checkpoint that
    loads "successfully" into differently-shaped weights would invalidate every
    evaluation silently.
    """
    torch.manual_seed(0)
    net = RebelNet()
    net.eval()

    obs = torch.randn(3, INPUT_DIM)
    feats = torch.randn(3, 64, 84)
    mask = torch.zeros(3, 64, dtype=torch.bool)
    mask[:, :5] = True                     # 5 legal slots, packed first

    with torch.no_grad():
        logits, value = net(obs, feats, mask)

    assert logits.shape == (3, 64), logits.shape
    assert value.shape == (3,), value.shape
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0), value
    assert torch.all(logits[~mask] < -1e8), logits[~mask]
    assert torch.isfinite(logits[mask]).all()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rebel_test.pt"
        save_rebel_checkpoint(path, net, step=3, config={"test": True})
        assert path.exists()
        loaded = load_rebel_checkpoint(path, torch.device("cpu"))
        assert loaded.architecture_config() == net.architecture_config(), (
            loaded.architecture_config(), net.architecture_config())
        with torch.no_grad():
            logits2, value2 = loaded(obs, feats, mask)
        assert torch.allclose(logits, logits2, atol=1e-6), (logits - logits2).abs().max()
        assert torch.allclose(value, value2, atol=1e-6), (value - value2).abs().max()

    print(f"test_net_shapes_and_checkpoint_roundtrip: OK (logits {tuple(logits.shape)}, "
          f"value {tuple(value.shape)} in [-1,1], illegal slots "
          f"{logits[~mask].max():.3g}, round-trip exact)")


# ---------------------------------------------------------------------------
# 7. the league evaluator logs the canonical `eval/*` key family
# ---------------------------------------------------------------------------

# The contract with `src/ppo/eval_harness.py` and `src/ismcts/train.py`, spelled out as
# LITERAL STRINGS on purpose: any drift in a suffix (or a member name) is a different chart
# in the wandb console, so this test must not be able to pass by matching a substring or by
# rebuilding the names from the module under test.
CANONICAL_EVAL_SUFFIXES = (
    "",
    "_stderr",
    "_score_for",
    "_score_against",
    "_return_for",
    "_return_against",
    "_caidas_for",
    "_caidas_against",
    "_limpias_for",
    "_limpias_against",
    "_plies",
    "_n_games",
)


def _canonical_eval_keys(member: str) -> set:
    return {f"eval/{member}{suffix}" for suffix in CANONICAL_EVAL_SUFFIXES}


def test_league_eval_logs_the_canonical_key_set():
    """
    Every league member must be logged under the EXACT key family the PPO pipeline defines,
    and under no other spelling: the point of the league is that a ReBeL run can be overlaid
    on a PPO/ISMCTS chart in the wandb console.

    Two games per member is enough to check the CONTRACT (which keys exist and with what
    shape) -- it is not, and is not meant to be, a measurement of strength. The members
    exercised here cover all three opponent kinds: the rule-based bot, a checkpoint loaded
    from a temp file (`initial`) and the uniform-random policy.
    """
    torch.manual_seed(0)
    net = RebelNet()
    net.eval()

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "rebel_league_test.pt"
        save_rebel_checkpoint(checkpoint, net, step=0)

        members = {
            "heuristic": Opponent("heuristic", "heuristic"),
            "initial": Opponent("initial", "checkpoint", str(checkpoint)),
            "random": Opponent("random", "random"),
        }
        games = {name: 2 for name in members}
        # A deliberately tiny search: this test is about the metric contract, and the
        # belief/search configuration changes neither the names nor the counts.
        cfg = SelfPlayConfig(games=1, n_particles=2, belief_history=4, n_iter=2, depth=1,
                             seed=0, explore=False, device="cpu")

        metrics = evaluate_league(net, members, games, seed_base=900_000, cfg=cfg,
                                  workers=1)

    expected = set()
    for name in ("heuristic", "initial", "random"):
        keys = _canonical_eval_keys(name)
        expected |= keys
        # Exact key equality per member: every canonical key present, nothing else.
        assert {k for k in metrics if k.startswith(f"eval/{name}")} == keys, (
            f"{name}: {sorted({k for k in metrics if k.startswith('eval/' + name)})}"
        )
        # `_for` means the NET's team and `_against` the opponent's: both sides must exist
        # for every quantity, or a chart silently shows half the match.
        for quantity in ("score", "return", "caidas", "limpias"):
            assert f"eval/{name}_{quantity}_for" in metrics
            assert f"eval/{name}_{quantity}_against" in metrics
        assert metrics[f"eval/{name}_n_games"] == 2.0, metrics[f"eval/{name}_n_games"]
        win = metrics[f"eval/{name}"]
        assert 0.0 <= win <= 1.0, win
        assert metrics[f"eval/{name}_plies"] > 0.0
        # `sqrt(p(1-p)/n)` over the requested 2 games, exactly as the PPO harness computes it.
        assert abs(metrics[f"eval/{name}_stderr"]
                   - math.sqrt(max(win * (1.0 - win), 1e-9) / 2)) < 1e-12
    assert set(metrics) == expected, sorted(set(metrics) ^ expected)

    print(f"test_league_eval_logs_the_canonical_key_set: OK "
          f"({len(expected)} keys, 3 members x 2 games: "
          + ", ".join(f"{n} {metrics[f'eval/{n}']:.0%}" for n in members) + ")")


def test_league_skips_members_without_a_checkpoint():
    """
    A member whose checkpoint does not exist yet must be ABSENT, not logged as 0 or as a
    stale number: a zero win rate against a checkpoint that was never played is a lie the
    charts cannot detect. Both directions are asserted -- present members are present, and
    the missing ones are skipped with a reason.
    """
    torch.manual_seed(0)
    net = RebelNet()
    net.eval()

    with tempfile.TemporaryDirectory() as tmp:
        # Nothing saved yet: no best.pt, no policy_*.pt, no --init-checkpoint.
        members, skipped = league_members(save_dir=tmp, init_checkpoint=None, iteration=1,
                                          include_league=True)
        assert set(members) == {"heuristic", "random"}, sorted(members)
        assert set(skipped) == {"reference", "initial", "prev_checkpoint"}, sorted(skipped)

        # A `policy_<n>.pt` written before the evaluation's iteration is prev_checkpoint.
        save_rebel_checkpoint(Path(tmp) / "policy_1.pt", net, step=1)
        init_path = Path(tmp) / "init.pt"
        save_rebel_checkpoint(init_path, net, step=0)
        members, skipped = league_members(save_dir=tmp, init_checkpoint=str(init_path),
                                          iteration=2, include_league=True)
        assert set(members) == {"heuristic", "initial", "prev_checkpoint", "random"}, \
            sorted(members)
        assert members["prev_checkpoint"].checkpoint == str(Path(tmp) / "policy_1.pt")
        assert members["initial"].checkpoint == str(init_path)

        # The iteration's own snapshot is SAVED AFTER its evaluation, so a policy_2.pt must
        # not be picked as `prev_checkpoint` while evaluating iteration 2.
        save_rebel_checkpoint(Path(tmp) / "policy_2.pt", net, step=2)
        members, _ = league_members(save_dir=tmp, init_checkpoint=None, iteration=2)
        assert members["prev_checkpoint"].checkpoint == str(Path(tmp) / "policy_1.pt")

        # best.pt is the `reference` member as soon as it exists.
        save_rebel_checkpoint(Path(tmp) / "best.pt", net, step=1)
        members, skipped = league_members(save_dir=tmp, init_checkpoint=str(init_path),
                                          iteration=3, include_league=True)
        assert set(members) == {"reference", "heuristic", "initial", "prev_checkpoint",
                                "random"}, sorted(members)
        assert members["reference"].checkpoint == str(Path(tmp) / "best.pt")
        assert skipped == {}

        # --no-league keeps the primary pair and skips the extension members.
        members, skipped = league_members(save_dir=tmp, init_checkpoint=str(init_path),
                                          iteration=3, include_league=False)
        assert set(members) == {"reference", "heuristic"}, sorted(members)
        assert set(skipped) == {"initial", "prev_checkpoint", "random"}, sorted(skipped)
        assert all("--no-league" in reason for reason in skipped.values())

    print("test_league_skips_members_without_a_checkpoint: OK (missing checkpoints are "
          "absent from the keys, with a reason; --no-league keeps reference+heuristic)")


if __name__ == "__main__":
    start = time.perf_counter()
    test_kuhn_cfr_converges()
    test_belief_stays_consistent_over_a_game()
    test_belief_bayes_update_reacts_to_a_surprising_play()
    test_solver_returns_a_distribution()
    test_search_prefers_the_winning_capture()
    test_value_head_prefers_scoring_captures()
    test_net_shapes_and_checkpoint_roundtrip()
    test_league_eval_logs_the_canonical_key_set()
    test_league_skips_members_without_a_checkpoint()
    print(f"\nAll ReBeL tests passed in {time.perf_counter() - start:.1f}s.")
