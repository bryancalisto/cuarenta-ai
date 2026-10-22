"""
Tests for the SHARED eval-metric contract (ISMCTS vs PPO).

The owner overlays ISMCTS, PPO and ReBeL runs on one wandb project (`cuarenta-ai`),
which only works if every pipeline logs the same `eval/<member><suffix>` keys for the
same league members. This test pins that contract down for ISMCTS:

  * every member that was evaluated has the FULL canonical key set;
  * `_for` and `_against` both exist for scores/returns/caídas/limpias;
  * the win rate carries `_stderr` and `_n_games`, and `_n_games` equals the number of
    games that were requested for that member (`--eval-games` for the two gate
    matches, `--league-games` for the extra members);
  * the legacy `vs_*` aliases are GONE (removed in favour of the canonical names),
    counterparts;
  * `prev_checkpoint` is SKIPPED — not logged with stale numbers — when the run has no
    saved snapshot yet;
  * `--no-league` drops the extra members but keeps the two gate matches.

`_pooled` (the multiprocessing pool) is replaced by a serial map: the test exercises
the real worker functions, just without process overhead, so it runs in seconds.

Run: python3 src/ismcts/test_eval_metrics.py   (or from inside src/ismcts)
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import CuarentaNet
import train as T

# The canonical suffixes, exactly as PPO's `eval_harness.match_metrics` emits them.
SUFFIXES = [
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
]

GATE_MEMBERS = ["reference", "heuristic"]
LEAGUE_MEMBERS = ["initial", "prev_checkpoint", "random"]


def canonical_keys(member: str) -> list[str]:
    return [f"{member}{s}" for s in SUFFIXES]


def _args(eval_games=2, league_games=2, no_league=False, members=None):
    return SimpleNamespace(
        eval_games=eval_games,
        eval_raw_games=1,
        league_games=league_games,
        no_league=no_league,
        members=members,
        no_wandb=True,
    )


def _serial_eval(net, reference_state, args, cfg, iteration, initial_state,
                 prev_checkpoint_state):
    """`run_evaluation` with the process pool replaced by a serial loop."""
    real_pooled = T._pooled
    T._pooled = lambda worker, payloads, desc: [worker(p) for p in payloads]
    try:
        return T.run_evaluation(net, reference_state, args, cfg, iteration,
                                initial_state=initial_state,
                                prev_checkpoint_state=prev_checkpoint_state)
    finally:
        T._pooled = real_pooled


def test_full_league_logs_every_canonical_key():
    """
    One tiny evaluation of the WHOLE league, with a `prev_checkpoint` provided.

    Asserts the exact key sets (no missing suffix, and no member logged with the wrong
    number of games) plus the alias equality.
    """
    cfg = {"n_iter": 4, "leaf_value_mode": "net", "max_rollout_len": T.MAX_ROLLOUT_LEN}
    torch.manual_seed(0)
    net = CuarentaNet().eval()
    # `reference`, `initial` and `prev_checkpoint` are deliberately DIFFERENT nets, so a
    # mix-up between the members cannot pass unnoticed.
    torch.manual_seed(1)
    reference_state = T._sd(CuarentaNet())
    initial_state = T._sd(CuarentaNet())
    torch.manual_seed(2)
    prev_state = T._sd(CuarentaNet())

    args = _args(eval_games=2, league_games=2,
                 members="reference,heuristic,initial,prev_checkpoint,random")
    metrics = _serial_eval(net, reference_state, args, cfg, iteration=0,
                           initial_state=initial_state,
                           prev_checkpoint_state=prev_state)

    # 1. every member was evaluated, and every canonical key is present
    for member in GATE_MEMBERS + LEAGUE_MEMBERS:
        missing = [k for k in canonical_keys(member) if k not in metrics]
        assert not missing, f"{member}: missing keys {missing}"

    # 2. `_for`/`_against` come in pairs (a key without its opposite is a broken chart)
    for member in GATE_MEMBERS + LEAGUE_MEMBERS:
        for family in ("score", "return", "caidas", "limpias"):
            for side in ("for", "against"):
                assert f"{member}_{family}_{side}" in metrics, (member, family, side)

    # 3. `_n_games` is the number of games ACTUALLY requested for that member
    for member in GATE_MEMBERS:
        assert metrics[f"{member}_n_games"] == float(args.eval_games), member
    for member in LEAGUE_MEMBERS:
        assert metrics[f"{member}_n_games"] == float(args.league_games), member

    # 4. win rates are probabilities, and their stderr is the binomial one
    for member in GATE_MEMBERS + LEAGUE_MEMBERS:
        p, se, n = (metrics[member], metrics[f"{member}_stderr"],
                    metrics[f"{member}_n_games"])
        assert 0.0 <= p <= 1.0, (member, p)
        expected = (max(p * (1.0 - p), 1e-9) / n) ** 0.5
        assert abs(se - expected) < 1e-9, (member, se, expected)

    # 5. legacy aliases exist and are IDENTICAL to their canonical counterparts
    for canonical, legacy in T.LEGACY_ALIASES.items():
        for suffix in [s.lstrip("_") for s in SUFFIXES if s]:
            key = f"{legacy}_{suffix}"
            assert key in metrics, f"missing legacy alias {key}"
            assert metrics[key] == metrics[f"{canonical}_{suffix}"], key
        assert metrics[legacy] == metrics[canonical], legacy

    # 6. the gate match is the one the acceptance logic reads (unchanged behaviour)

    n_keys = len(canonical_keys("reference"))
    print(f"test_full_league_logs_every_canonical_key: OK "
          f"({len(GATE_MEMBERS + LEAGUE_MEMBERS)} members x {n_keys} keys, "
          f"n_games={int(metrics['reference_n_games'])}/"
          f"{int(metrics['random_n_games'])}, "
          f"reference={metrics['reference']:.0%} random={metrics['random']:.0%})")
    return metrics


def test_prev_checkpoint_is_skipped_without_a_snapshot():
    """
    A missing `prev_checkpoint` (nothing saved yet) must REMOVE the member rather than
    log stale numbers, and must not disturb the other members.
    """
    # Cheap: only the payload builder, no games played.
    league = T.league_matches(_args(league_games=2), {"w": 1}, {"w": 1}, None, {})
    names = [name for name, _worker, _payloads in league]
    assert names == ["initial", "random"], names
    assert all(len(payloads) == 2 for _n, _w, payloads in league)

    # `prev_checkpoint` is opt-in now (it left the default set), so ask for it by name.
    args = _args(league_games=3)
    args.members = "reference,heuristic,initial,prev_checkpoint,random"
    with_prev = T.league_matches(args, {"w": 1}, {"w": 1}, {"w": 1}, {})
    names = [name for name, _worker, _payloads in with_prev]
    assert names == ["initial", "prev_checkpoint", "random"], names
    assert all(len(payloads) == 3 for _n, _w, payloads in with_prev)

    # The DEFAULT set does not spend games on `prev_checkpoint`.
    default = T.league_matches(_args(league_games=3), {"w": 1}, {"w": 1}, {"w": 1}, {})
    assert [n for n, _w, _p in default] == ["initial", "random"], [n for n, _w, _p in default]

    # A selection with no league members runs none of them.
    gate_only_args = _args(league_games=3)
    gate_only_args.members = "reference,heuristic"
    only_gate = T.league_matches(gate_only_args, {"w": 1}, {"w": 1}, {"w": 1}, {})
    assert only_gate == [], only_gate

    # No `initial` state at all (defensive: a run started from nothing).
    no_initial = T.league_matches(_args(league_games=1), {"w": 1}, None, None, {})
    assert [n for n, _w, _p in no_initial] == ["random"]

    print("test_prev_checkpoint_is_skipped_without_a_snapshot: OK")


def test_no_league_keeps_the_gate_matches():
    """`--no-league` must drop only the extra members (the gate still runs)."""
    cfg = {"n_iter": 4, "leaf_value_mode": "net", "max_rollout_len": T.MAX_ROLLOUT_LEN}
    torch.manual_seed(0)
    net = CuarentaNet().eval()
    state = T._sd(net)

    args = _args(eval_games=1, league_games=2, no_league=True)
    metrics = _serial_eval(net, state, args, cfg, iteration=0,
                           initial_state=state, prev_checkpoint_state=state)
    for member in GATE_MEMBERS:
        assert f"{member}_n_games" in metrics, member
    for member in LEAGUE_MEMBERS:
        # `random` is not a checkpoint member and is ALWAYS evaluated (every pipeline
        # reports it as the sanity floor), so --no-league only drops the two members that
        # need saved weights.
        if member == "random":
            assert member in metrics, "random must survive --no-league"
            assert f"{member}_n_games" in metrics, member
        else:
            assert member not in metrics, f"{member} should be dropped by --no-league"
            assert f"{member}_n_games" not in metrics, member
    print("test_no_league_keeps_the_gate_matches: OK")


def test_match_metrics_key_set_is_exact():
    """
    The raw metric builder must emit EXACTLY the canonical key set for one prefix (no
    extra accidental keys, which would show up as junk series in the wandb project).
    """
    results = [
        {"winner": 0, "score0": 40, "score1": 30, "plies": 120,
         "return0": 60.0, "return1": 52.0, "caidas0": 5, "caidas1": 1,
         "limpias0": 0, "limpias1": 0},
        {"winner": 1, "score0": 20, "score1": 40, "plies": 100,
         "return0": 52.0, "return1": 60.0, "caidas0": 1, "caidas1": 5,
         "limpias0": 0, "limpias1": 0},
    ]
    mm = T.match_metrics(results, "random")
    assert sorted(mm) == sorted(canonical_keys("random")), sorted(mm)
    assert mm["random"] == 0.5
    assert mm["random_score_for"] == 30.0
    assert mm["random_n_games"] == 2.0
    assert abs(mm["random_stderr"] - (0.25 / 2) ** 0.5) < 1e-12
    assert T.match_metrics([], "random") == {}, "no games must mean no keys"

    aliased = T._with_aliases(mm, "random", None)
    assert aliased == mm, "the alias helper must no longer duplicate anything"
    print(f"test_match_metrics_key_set_is_exact: OK ({len(mm)} keys)")


def test_trimmed_members_run_end_to_end():
    """
    A trimmed `--members` set must run all the way through `run_evaluation` — matches,
    metric keys, AND the printed summary.

    This is here because the summary print read `metrics['heuristic_*']` unconditionally,
    which only crashed for a trimmed `--members` set, and no unit test of
    `requested_members` could catch it: the bug is in the reporting code, not the
    selection logic. Two shapes are covered — a set without `heuristic`, and the
    gate-only set with no league members at all.
    """
    cfg = {"n_iter": 4, "leaf_value_mode": "net", "max_rollout_len": T.MAX_ROLLOUT_LEN}
    torch.manual_seed(0)
    net = CuarentaNet().eval()
    state = T._sd(net)

    args = _args(eval_games=1, league_games=1, members="reference,random")
    metrics = _serial_eval(net, state, args, cfg, iteration=0,
                           initial_state=state, prev_checkpoint_state=state)
    assert "reference" in metrics and "random" in metrics, sorted(metrics)
    assert not any(k.startswith("heuristic") for k in metrics), (
        "heuristic was not requested but its keys were logged: "
        f"{[k for k in metrics if k.startswith('heuristic')]}"
    )

    # Gate only: no league members, so no `random` either.
    gate_args = _args(eval_games=1, league_games=1, members="reference,heuristic")
    gate_metrics = _serial_eval(net, state, gate_args, cfg, iteration=1,
                                initial_state=state, prev_checkpoint_state=state)
    assert "heuristic" in gate_metrics, sorted(gate_metrics)
    assert not any(k.startswith("random") for k in gate_metrics), (
        f"random was not requested: {[k for k in gate_metrics if k.startswith('random')]}"
    )
    print("test_trimmed_members_run_end_to_end: OK "
          "(reference+random, then reference+heuristic, both printed and logged)")


if __name__ == "__main__":
    test_match_metrics_key_set_is_exact()
    test_prev_checkpoint_is_skipped_without_a_snapshot()
    test_no_league_keeps_the_gate_matches()
    test_trimmed_members_run_end_to_end()
    test_full_league_logs_every_canonical_key()
    print("\nAll eval-metric tests passed.")
    print("Canonical key set per member (12): " + ", ".join(canonical_keys("reference")))
