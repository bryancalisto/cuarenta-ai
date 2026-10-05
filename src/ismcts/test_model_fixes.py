"""
Regression tests for the bugs fixed in the model pipeline.

Each test fails on the pre-fix code. Run: python test_model_fixes.py

  1. caída/limpia: `score_play` returns a LIST of reasons; testing it with `==`
     against a string is always False. That single mistake disabled the
     `is_caida`/`is_limpia` action features, the env's +2 step reward, and made the
     HeuristicAgent ignore caídas/limpias completely (it was the imitation teacher
     AND the evaluation opponent).
  2. the search must value leaves with the network's value head by default (not with
     rollouts that truncate and silently score 0.0), evaluate each node once, and
     produce a visit distribution that sums to the simulation budget.
  3. the dataset must report real sequence lengths so padding is ignored, and its
     history dropout must produce the single-pending-step inputs used at interior
     search nodes.
"""

import random

import numpy as np
import torch

from card import Card, Rank, Suit
from env import CuarentaEnv, CuarentaState, TEAMS
from captures import score_play
from utils import PointsReasons, has_reason
from heuristic_agent import HeuristicAgent


def _caida_position():
    """
    A position with a 2-point caída available AND a bigger 0-point capture.

    table: 4♣, 3♠ (3♠ was placed last), 2♥
    hand:  3♥ (captures 3♠ = caída, 2 pts) and 7♥ (captures 3♠+4♣ = 0 pts)
    """
    table = [Card(Rank.CUATRO, Suit.TREBOLES),
             Card(Rank.TRES, Suit.BRILLOS),
             Card(Rank.DOS, Suit.CORAZONES_ROJOS)]
    hand = [Card(Rank.TRES, Suit.CORAZONES_ROJOS),
            Card(Rank.SIETE, Suit.CORAZONES_ROJOS)]
    env = CuarentaEnv(seed=0)
    env.state = CuarentaState(
        hands={0: list(hand), 1: [], 2: [], 3: []},
        table=list(table), deck=[], scores=[0, 0], cards_captured=[[], []],
        current_player=0, plays_this_deal=[0, 0, 0, 0],
        last_placed=Card(Rank.TRES, Suit.BRILLOS), done=False, winner=None,
    )
    env._legal_cache = None
    return env


def test_reasons_are_a_list():
    """Guard against reintroducing the `list == str` comparison."""
    env = _caida_position()
    legal = env.legal_actions()
    caida_slots = []
    for i, a in enumerate(legal):
        pts, reasons = score_play(played=a.card, capture_set=a.capture,
                                  table=env.state.table,
                                  last_placed=env.state.last_placed,
                                  non_caida_allowed=True)
        if pts == 2:
            caida_slots.append(i)
            assert isinstance(reasons, list)
            assert has_reason(reasons, PointsReasons.CAIDA)
            assert not (reasons == PointsReasons.CAIDA), (
                "reasons is a list: `==` is always False — use has_reason()/`in`"
            )
    assert caida_slots, "the crafted position should offer a caída"
    print("test_reasons_are_a_list: OK")


def test_caida_feature_is_set():
    """features[81] / features[82] must actually fire (they used to be always 0)."""
    env = _caida_position()
    obs = env.observe(0)
    legal = env.legal_actions()
    is_caida = [int(obs["action_features"][i][81]) for i in range(len(legal))]
    assert any(is_caida), "is_caida feature never fires"
    print(f"test_caida_feature_is_set: OK (features={is_caida})")


def test_heuristic_plays_for_caida():
    """
    The teacher must prefer a 2-point caída over a bigger 0-point capture.
    Before the fix it chose the 2-card, 0-point capture.
    """
    env = _caida_position()
    legal = env.legal_actions()
    chosen = HeuristicAgent(seed=0).act(env, 0)
    slot = legal.index(chosen)
    pts, reasons = score_play(played=chosen.card, capture_set=chosen.capture,
                              table=env.state.table,
                              last_placed=env.state.last_placed,
                              non_caida_allowed=True)
    assert has_reason(reasons, PointsReasons.CAIDA), (
        f"heuristic picked slot {slot} ({chosen}) with pts={pts}, not the caída"
    )
    print(f"test_heuristic_plays_for_caida: OK (picked slot {slot}, pts={pts})")


def test_env_reward_pays_caida_bonus():
    """step() must add +2 reward for a caída/limpia (it never did before)."""
    env = _caida_position()
    legal = env.legal_actions()
    caida_slot = next(
        i for i, a in enumerate(legal)
        if has_reason(score_play(played=a.card, capture_set=a.capture,
                                 table=env.state.table,
                                 last_placed=env.state.last_placed,
                                 non_caida_allowed=True)[1],
                      PointsReasons.CAIDA)
    )
    _obs, reward, done, _trunc, info = env.step(caida_slot)
    assert reward >= 2, f"caída reward missing (reward={reward}, info={info})"
    print(f"test_env_reward_pays_caida_bonus: OK (reward={reward})")


def test_search_uses_net_values_not_rollouts():
    """
    Default leaf_value_mode='net': the value head values the leaves, so no playouts
    run at all, and each expanded node is evaluated exactly once (cached).
    """
    from network import CuarentaNet
    from history import HistoryTracker
    from network_ismcts_1 import NetworkISMCTS

    env = CuarentaEnv(seed=5)
    env.reset(seed=5)
    net = CuarentaNet().eval()
    tracker = HistoryTracker(env)
    n_iter = 64
    searcher = NetworkISMCTS(env, net, tracker, n_iter=n_iter,
                             rng=random.Random(0), dirichlet_eps=0.0)

    legal = env.legal_actions()
    visits = searcher.search(root_player=env.state.current_player)

    assert len(visits) == len(legal)
    assert sum(visits.values()) == n_iter, (
        f"every simulation should land on exactly one root action: {sum(visits.values())}"
    )
    assert searcher.stats["rollouts"] == 0, "net mode must not run rollouts"
    # one evaluation for the root + one per expanded node
    assert 0 < searcher.stats["evals"] <= n_iter + 1, searcher.stats
    # priors are a probability distribution
    assert abs(float(searcher.last_root_priors.sum()) - 1.0) < 1e-4
    print(f"test_search_uses_net_values_not_rollouts: OK {searcher.stats}")


def test_search_is_deterministic_given_seed():
    """Same seed, same state, no Dirichlet noise => identical visit counts."""
    from network import CuarentaNet
    from history import HistoryTracker
    from network_ismcts_1 import NetworkISMCTS

    torch.manual_seed(0)
    net = CuarentaNet().eval()
    runs = []
    for _ in range(2):
        env = CuarentaEnv(seed=21)
        env.reset(seed=21)
        s = NetworkISMCTS(env, net, HistoryTracker(env), n_iter=32,
                          rng=random.Random(1234), dirichlet_eps=0.0)
        runs.append(s.search(root_player=env.state.current_player))
    assert runs[0] == runs[1], f"search is not reproducible: {runs}"
    print("test_search_is_deterministic_given_seed: OK")


def test_dataset_lengths_and_dropout():
    """
    - length must equal min(T, t_max) with dropout off (so padding is ignored)
    - the padded region must be zeros and the pending step must stay last
    - with dropout on, length can be as small as 1 (interior-node regime)
    """
    from selfplay import Decision
    from dataset import CuarentaDataset
    from history import OBS_FLAT_DIM, ACT_FEAT_DIM
    from network import N_ACTIONS, ACTION_FEAT_DIM

    def make_decision(T):
        obs = np.random.rand(T, OBS_FLAT_DIM).astype(np.float32)
        act = np.random.rand(T, ACT_FEAT_DIM).astype(np.float32)
        am = np.ones(T, dtype=bool); am[-1] = False
        feat = np.zeros((N_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
        mask = np.zeros((N_ACTIONS,), dtype=bool); mask[:3] = True
        vis = np.zeros(N_ACTIONS, dtype=np.int64); vis[1] = 5
        return Decision(obs_seq=obs, act_seq=act, act_mask=am, feats=feat,
                        mask=mask, visits=vis, team=0, player=0)

    ds = CuarentaDataset([make_decision(12), make_decision(40)], t_max=32, history_dropout=0.0)
    obs, act, am, feats, mask, pi, v, length = ds[0]
    assert obs.shape == (32, OBS_FLAT_DIM)
    assert int(length) == 12
    pad = 32 - 12
    assert torch.all(obs[:pad] == 0), "padding should be zeros"
    assert not am[-1], "the last step is the pending step"
    assert am[pad:-1].all(), "all completed steps carry an action"

    obs, act, am, _, _, _, _, length = ds[1]
    assert int(length) == 32, "clipped to t_max"

    # dropout: lengths must be 1..T and never 0
    ds_drop = CuarentaDataset([make_decision(20) for _ in range(200)],
                              t_max=32, history_dropout=0.9)
    lengths = [int(ds_drop[i][-1]) for i in range(200)]
    assert min(lengths) >= 1 and max(lengths) <= 20, (min(lengths), max(lengths))
    assert min(lengths) == 1, "dropout should sometimes leave the pending step alone"
    print(f"test_dataset_lengths_and_dropout: OK (lengths {min(lengths)}..{max(lengths)})")


if __name__ == "__main__":
    test_reasons_are_a_list()
    test_caida_feature_is_set()
    test_heuristic_plays_for_caida()
    test_env_reward_pays_caida_bonus()
    test_search_uses_net_values_not_rollouts()
    test_search_is_deterministic_given_seed()
    test_dataset_lengths_and_dropout()
    print("\nAll model-fix tests passed.")
