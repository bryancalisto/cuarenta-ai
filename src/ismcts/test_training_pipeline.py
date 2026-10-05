"""
Tests for the training-loop additions:

  1. mixed data: self-play samples keep full policy weight, imitation samples can be
     value-only, and a value-only-only batch produces zero policy loss;
  2. target temperature flattens SELF-PLAY targets but must NOT touch the teacher's
     one-hot imitation targets;
  3. buffer diagnostics (target/prior entropy, top probability) are sane;
  4. the head probe used for the health metrics runs and returns a full sample count;
  5. the reference-match evaluator runs a game end to end with two different nets.

Run: python test_training_pipeline.py
"""

import numpy as np
import torch

from dataset import CuarentaDataset
from network import N_ACTIONS, CuarentaNet
from network_ismcts_1 import NetworkISMCTS
from selfplay import Decision, play_one_game
from train import (MAX_ROLLOUT_LEN, T_MAX, _eval_reference_worker, _load_net,
                   _probe_worker, _sd, buffer_stats)
from history import OBS_FLAT_DIM, ACT_FEAT_DIM, HistoryTracker
from env import CuarentaEnv


def _decision(T=6, visits=None, selfplay=True, priors=None):
    obs = np.random.rand(T, OBS_FLAT_DIM).astype(np.float32)
    act = np.random.rand(T, ACT_FEAT_DIM).astype(np.float32)
    am = np.ones(T, dtype=bool); am[-1] = False
    feat = np.zeros((N_ACTIONS, 84), dtype=np.float32)
    mask = np.zeros((N_ACTIONS,), dtype=bool); mask[:3] = True
    if visits is None:
        visits = np.zeros(N_ACTIONS, dtype=np.int64); visits[[0, 1]] = [7, 3]
    return Decision(obs_seq=obs, act_seq=act, act_mask=am, feats=feat, mask=mask,
                    visits=visits, team=0, player=0, priors=priors, selfplay=selfplay)


def test_target_temperature_flattens_selfplay_only():
    d_sp = _decision(selfplay=True)                       # visits 7:3
    onehot_visits = np.zeros(N_ACTIONS, dtype=np.int64)
    onehot_visits[1] = 1                                  # teacher one-hot
    d_im = _decision(visits=onehot_visits, selfplay=False)

    raw = CuarentaDataset([d_sp], t_max=8, target_temperature=1.0)[0][5].numpy()
    flat = CuarentaDataset([d_sp], t_max=8, target_temperature=2.0)[0][5].numpy()
    onehot = CuarentaDataset([d_im], t_max=8, target_temperature=2.0)[0][5].numpy()

    assert abs(raw.sum() - 1.0) < 1e-5
    # flattening must reduce the peak
    assert flat.max() < raw.max(), (flat.max(), raw.max())
    assert abs(flat.sum() - 1.0) < 1e-5
    # the imitation one-hot must stay one-hot
    assert abs(onehot.max() - 1.0) < 1e-5, onehot.max()
    print(f"test_target_temperature_flattens_selfplay_only: OK "
          f"(raw_top={raw.max():.3f} flat_top={flat.max():.3f})")


def test_policy_weight_and_value_only_batches():
    ds = CuarentaDataset([_decision(), _decision(selfplay=False)],
                         t_max=8, imitation_policy_weight=0.0)
    assert float(ds[0][-1]) == 1.0
    assert float(ds[1][-1]) == 0.0

    # A batch of only value-only samples must contribute no policy gradient.
    ds_only_im = CuarentaDataset([_decision(selfplay=False) for _ in range(4)],
                                 t_max=8, imitation_policy_weight=0.0)
    from torch.utils.data import DataLoader
    from train import train_one_epoch

    net = CuarentaNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss, pl, vl, al, tent = train_one_epoch(net, DataLoader(ds_only_im, batch_size=4),
                                             opt, "cpu")
    assert pl == 0.0, f"policy loss should be exactly 0 for value-only data, got {pl}"
    print(f"test_policy_weight_and_value_only_batches: OK (policy={pl} value={vl:.3f})")


def test_buffer_stats():
    d_sp = _decision(priors=np.array([0.5, 0.3, 0.2], dtype=np.float32))
    d_im = _decision(selfplay=False, visits=np.zeros(N_ACTIONS, dtype=np.int64))
    d_im.visits[1] = 1
    stats = buffer_stats([d_sp, d_im])
    assert stats["n"] == 2
    assert abs(stats["selfplay_frac"] - 0.5) < 1e-6
    assert 0.0 < stats["target_entropy"] < np.log(3) + 1e-6
    assert stats["prior_entropy"] > 0
    assert 0.0 < stats["target_top_prob"] <= 1.0
    print(f"test_buffer_stats: OK {stats}")


def test_probe_worker_runs():
    net = CuarentaNet().eval()
    cfg = {"n_iter": 4, "leaf_value_mode": "net", "max_rollout_len": MAX_ROLLOUT_LEN}
    res = _probe_worker((910_000, _sd(net), cfg))
    assert res["n"] > 0, "probe produced no decisions"
    assert 0.0 <= res["correct"] / res["n"] <= 1.0
    assert 0.0 <= res["v_se"] / res["n"]
    print(f"test_probe_worker_runs: OK (n={res['n']}, "
          f"acc={res['correct']/res['n']:.3f}, mse={res['v_se']/res['n']:.3f})")


def test_reference_match_runs():
    """Two DIFFERENT nets must be able to play a gating match to completion."""
    torch.manual_seed(0)
    a = CuarentaNet().eval()
    torch.manual_seed(1)
    b = CuarentaNet().eval()
    cfg = {"n_iter": 8, "leaf_value_mode": "net", "max_rollout_len": MAX_ROLLOUT_LEN}
    res = _eval_reference_worker((920_000, _sd(a), _sd(b), cfg))
    assert res["winner"] in (0, 1), res
    # the match summary must carry the reward bookkeeping used for the return curves
    for key in ("score0", "score1", "plies", "return0", "return1",
                "caidas0", "caidas1", "limpias0", "limpias1"):
        assert key in res, f"missing {key} in match summary"
    print(f"test_reference_match_runs: OK (winner team {res['winner']}, "
          f"scores {res['score0']}-{res['score1']}, {res['plies']} plies, "
          f"returns {res['return0']:.0f}-{res['return1']:.0f})")


def test_selfplay_marks_data_as_selfplay_and_records_priors():
    net = CuarentaNet().eval()
    decisions, result = play_one_game(net, n_iter=8, seed=930_000, device="cpu")
    assert decisions, "no decisions produced"
    assert all(d.selfplay for d in decisions)
    assert decisions[0].priors is not None and len(decisions[0].priors) >= 1
    assert abs(float(np.sum(decisions[0].priors)) - 1.0) < 1e-4, "priors must be a dist"
    print("test_selfplay_marks_data_as_selfplay_and_records_priors: OK")


def test_game_result_return_identity():
    """
    The env only pays +2 per caída, +2 per limpia and +50 for winning, so each team's
    episode return must equal 2*caidas + 2*limpias + 50*won exactly. This is the
    identity the return curves rely on, so it is worth pinning down.
    """
    net = CuarentaNet().eval()
    for seed in (950_000, 950_001, 950_002):
        decisions, r = play_one_game(net, n_iter=8, seed=seed, device="cpu")
        for team in (0, 1):
            won = 1.0 if r.winner == team else 0.0
            expected = 2.0 * r.caidas[team] + 2.0 * r.limpias[team] + 50.0 * won
            assert abs(r.returns[team] - expected) < 1e-6, (
                f"seed={seed} team={team}: return {r.returns[team]} != {expected}"
            )
        assert r.plies == len(decisions), (r.plies, len(decisions))
        assert sum(r.scores) > 0, "scores should be non-zero at the end of a game"
        assert r.winner in (0, 1)
    print("test_game_result_return_identity: OK")


def test_match_metrics_and_selfplay_metrics():
    from train import match_metrics, mean_game_metrics
    net = CuarentaNet().eval()
    decisions, result = play_one_game(net, n_iter=8, seed=960_000, device="cpu")
    sp = mean_game_metrics([result], prefix="selfplay")
    assert "selfplay/win_rate_team0" in sp and "selfplay/return_team0" in sp
    assert "selfplay/caidas_team1" in sp and "selfplay/score_diff" in sp

    match = [{"winner": 0, "score0": 40, "score1": 30, "plies": 120,
              "return0": 60.0, "return1": 52.0, "caidas0": 5, "caidas1": 1,
              "limpias0": 0, "limpias1": 0}]
    mm = match_metrics(match, "vs_heuristic")
    assert mm["vs_heuristic"] == 1.0
    assert mm["vs_heuristic_score_for"] == 40.0
    assert mm["vs_heuristic_return_against"] == 52.0
    print(f"test_match_metrics_and_selfplay_metrics: OK ({len(sp)} + {len(mm)} keys)")


if __name__ == "__main__":
    test_target_temperature_flattens_selfplay_only()
    test_policy_weight_and_value_only_batches()
    test_buffer_stats()
    test_probe_worker_runs()
    test_reference_match_runs()
    test_selfplay_marks_data_as_selfplay_and_records_priors()
    test_game_result_return_identity()
    test_match_metrics_and_selfplay_metrics()
    print("\nAll training-pipeline tests passed.")
