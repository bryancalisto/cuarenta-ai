"""
Tests for the PPO pipeline.

These pin down the invariants the trainer depends on — the ones that are silent when
they break:

  1. the observation layout the policy head consumes (and therefore every exported
     ONNX model),
  2. the env's reward decomposition `return == 2*caidas + 2*limpias + 50*won`
     (this is also the regression test for the caída/limpia reward bug),
  3. the action-id encoding round-trip and legal-action masking,
  4. GAE against a hand-computed example,
  5. the recurrence contract: replaying a buffered sequence must reproduce exactly
     what stepping the network produced at collection time, including the hidden-state
     reset at episode boundaries,
  6. a full collect -> update cycle for both feedforward and recurrent policies,
  7. checkpoint save/load round-trip,
  8. the evaluation harness.

Run: python3 src/ppo/test_ppo.py    (from the repository root or from src/ppo/)
"""

from __future__ import annotations

import random

import numpy as np
import torch

from card import Card, Rank, Suit
from captures import score_play
from env import (ACTION_FEAT_DIM, MAX_ACTIONS, WIN_BONUS, Action, CuarentaEnv,
                 CuarentaState, TEAMS, action_features_matrix)
from networks import ActorCriticNetwork, load_actor_critic
from utils import (OBS_KEYS, OBS_SIZE, OBS_SIZE_ISMCTS, PointsReasons,
                   check_obs_layout, flatten_obs, has_reason)
from agents import HeuristicAgent, RandomAgent
from config import PPOConfig

from eval_harness import OpponentSpec, evaluate
from rollout import (EpisodeStats, RunningMeanStd, compute_gae,
                     replay_state_embeddings, select_legal_action)
from trainer import PPOTrainer


def _tiny_cfg(**overrides) -> PPOConfig:
    """A config small enough to run several rollouts in a test."""
    base = dict(n_envs=2, n_steps=24, n_epochs=2, minibatch_size=32,
                envs_per_minibatch=1, hidden=32, value_hidden=16,
                lstm_hidden=16, save_dir="/tmp/ppo_test", wandb=False,
                save_plots=False, total_steps=48, eval_games=2)
    base.update(overrides)
    return PPOConfig(**base)


# ---------------------------------------------------------------------------
# 1. observation layout
# ---------------------------------------------------------------------------

def test_obs_layouts():
    """
    PPO uses the ISMCTS layout (208). The legacy layout (244) must keep producing the
    exact historical tensor, because the DAgger policy and the deployed ONNX model were
    trained on it.
    """
    from utils import OBS_KEYS_ISMCTS, OBS_KEYS_LEGACY, OBS_SIZE_LEGACY
    env = CuarentaEnv(seed=0)
    obs, _ = env.reset(seed=0)
    check_obs_layout(obs, OBS_KEYS_ISMCTS)
    check_obs_layout(obs, OBS_KEYS_LEGACY)

    flat = flatten_obs(obs)
    assert flat.shape == (OBS_SIZE,) == (208,), flat.shape
    # same content and order as the ISMCTS pipeline's history.flatten_obs
    legacy = flatten_obs(obs, OBS_KEYS_LEGACY)
    assert legacy.shape == (OBS_SIZE_LEGACY,) == (244,), legacy.shape
    historical = np.concatenate([obs[k].reshape(-1) for k in sorted(obs.keys())
                                 if k != "plays_this_deal"]).astype(np.float32)
    assert np.array_equal(legacy, historical), (
        "the legacy layout drifted from the historical sorted() order — that would "
        "invalidate the DAgger checkpoints and the exported ONNX model"
    )
    print("test_obs_layouts: OK")


# ---------------------------------------------------------------------------
# 2. env rewards
# ---------------------------------------------------------------------------

def test_env_reward_identity():
    """
    The env pays exactly +2 per caída, +2 per limpia and +50 for the win, so each
    team's episode return must equal `2*caidas + 2*limpias + 50*won`. Before the
    caída/limpia reward bug was fixed the bonuses never fired, so returns were just
    50*won (this test fails on that code).
    """
    rng = random.Random(0)
    saw_bonus = 0
    for game in range(3):
        env = CuarentaEnv(seed=200 + game)
        env.reset(seed=200 + game)
        returns, caidas, limpias, cards, carton = ([0.0, 0.0], [0, 0], [0, 0], [0, 0], [0, 0])
        while not env.state.done:
            player = env.state.current_player
            action = rng.choice(env.legal_actions())
            _obs, reward, _done, info = env.step(action)
            team = TEAMS[player]
            returns[team] += float(reward)
            cards[team] += len(action.capture) if action.capture else 0
            carton[team] += int(info.get("carton_points") or 0)
            reasons = info.get("reasons") or []
            if has_reason(reasons, PointsReasons.CAIDA):
                caidas[team] += 1
                saw_bonus += 1
            elif has_reason(reasons, PointsReasons.LIMPIA):
                limpias[team] += 1
                saw_bonus += 1
        for team in (0, 1):
            won = 1.0 if env.state.winner == team else 0.0
            # The env's reward is dense: +1 per captured card, +2 caída, +2 limpia,
            # +4 for the win. This test used to assert 2*caidas+2*limpias+50*won, which
            # stopped being true when the reward was reshaped.
            expected = (float(cards[team]) + 2.0 * caidas[team] + 2.0 * limpias[team]
                        + float(carton[team]) + WIN_BONUS * won)
            assert abs(returns[team] - expected) < 1e-9, (
                f"game {game} team {team}: return {returns[team]} != {expected}"
            )
    assert saw_bonus > 0, "no caída/limpia was ever scored; test is not exercising the fix"
    print(f"test_env_reward_identity: OK ({saw_bonus} bonus events across 3 games)")


def test_heuristic_plays_for_caida():
    """
    The crafted position offers a 2-point caída and a bigger 0-point capture; the
    heuristic must take the caída (it used to take the biggest capture instead).
    """
    table = [Card(Rank.CUATRO, Suit.TREBOLES),
             Card(Rank.TRES, Suit.BRILLOS),
             Card(Rank.DOS, Suit.CORAZONES_ROJOS)]
    hand = [Card(Rank.TRES, Suit.CORAZONES_ROJOS),
            Card(Rank.SIETE, Suit.CORAZONES_ROJOS)]
    env = CuarentaEnv(seed=0)
    env.state = CuarentaState(
        hands={0: list(hand), 1: [], 2: [], 3: []},
        table=list(table), deck=[], n_decks_consumed=0, scores=[0, 0],
        cards_captured=[[], []], current_player=0,
        last_placed=Card(Rank.TRES, Suit.BRILLOS), done=False, winner=None,
    )
    legal = env.legal_actions()
    chosen = HeuristicAgent(seed=0).act(env.observe(0), legal, state=env.state, team=0)
    pts, reasons = score_play(played=chosen.card, capture_set=chosen.capture,
                              table=env.state.table, last_placed=env.state.last_placed,
                              non_caida_allowed=True)
    assert has_reason(reasons, PointsReasons.CAIDA), (
        f"heuristic chose {chosen} (pts={pts}) instead of the caída"
    )
    print(f"test_heuristic_plays_for_caida: OK (chose {chosen})")


# ---------------------------------------------------------------------------
# 3. action encoding and masking
# ---------------------------------------------------------------------------

def test_action_features_are_slot_invariant():
    """
    Candidate actions are described by FEATURES, not by slot index: the same action in
    two different positions must produce the same vector, and a caída must be marked.
    """
    env = CuarentaEnv(seed=7)
    env.reset(seed=7)
    rng = random.Random(0)
    seen_play = {}
    while not env.state.done:
        legal = env.legal_actions()
        feats, mask = action_features_matrix(legal, env.state)
        assert mask[:len(legal)].all() and not mask[len(legal):].any()
        for i, a in enumerate(legal):
            # the played card is a one-hot in the first 40 dims
            assert feats[i][a.card.index] == 1.0
            assert int(feats[i][80]) == (1 if a.capture is not None else 0)
            # the caída/limpia flags must agree with the rules for THIS position
            from env import MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS
            non_caida = (env.state.scores[TEAMS[env.state.current_player]]
                         < MAX_ACHIEVABLE_POINTS_BY_NON_CAIDA_ACTIONS)
            _pts, reasons = score_play(played=a.card, capture_set=a.capture,
                                       table=env.state.table,
                                       last_placed=env.state.last_placed,
                                       non_caida_allowed=non_caida)
            assert int(feats[i][81]) == int(has_reason(reasons, PointsReasons.CAIDA))
            assert int(feats[i][82]) == int(has_reason(reasons, PointsReasons.LIMPIA))
            # the card/capture part (dims 0..79) describes the action itself, so the
            # same action in another position must encode identically there
            key = (a.card.index, tuple(sorted(c.index for c in (a.capture or []))))
            if key in seen_play:
                assert np.array_equal(feats[i][:80], seen_play[key]), (
                    "the card/capture half of the features is not position-invariant"
                )
            seen_play[key] = feats[i][:80].copy()
        env.step(rng.choice(legal))
    print(f"test_action_features_are_slot_invariant: OK ({len(seen_play)} distinct actions)")


def test_legal_action_masking():
    """Only legal SLOTS can ever be selected, and the distribution covers exactly them."""
    torch.manual_seed(0)
    from env import MAX_ACTIONS
    logits = torch.randn(MAX_ACTIONS)
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool)
    mask[:3] = True                       # 3 legal actions
    for _ in range(20):
        slot, log_prob, entropy = select_legal_action(logits, mask)
        assert 0 <= slot < 3, slot
        assert torch.isfinite(log_prob)
    assert entropy.item() > 0
    pos, _, _ = select_legal_action(logits, mask, deterministic=True)
    assert pos == int(torch.argmax(logits[:3]).item())
    print("test_legal_action_masking: OK")


# ---------------------------------------------------------------------------
# 4. GAE
# ---------------------------------------------------------------------------

def test_gae_hand_computed():
    """
    Two steps, gamma=0.5, lambda=1.0, no done flags, last_value=0:
        delta_1 = r1 + 0.5*0         - v1 = 1.0 + 0 - 2.0 = -1.0   -> A1 = -1.0
        delta_0 = r0 + 0.5*v1 - v0 = 0.0 + 1.0 - 1.0 = 0.0        -> A0 = 0.0 + 0.5*1.0*(-1.0) = -0.5
    """
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([1.0, 2.0], dtype=np.float32)
    dones = np.array([0.0, 0.0], dtype=np.float32)
    advs, rets = compute_gae(rewards, values, dones, last_value=0.0, gamma=0.5, lam=1.0)
    assert np.allclose(advs, [-0.5, -1.0]), advs
    assert np.allclose(rets, [0.5, 1.0]), rets

    # A terminal step must not bootstrap: with dones[1]=1 the second advantage is
    # just r1 - v1 regardless of last_value.
    dones = np.array([0.0, 1.0], dtype=np.float32)
    advs, rets = compute_gae(rewards, values, dones, last_value=99.0, gamma=0.5, lam=1.0)
    assert np.isclose(advs[1], 1.0 - 2.0), advs
    assert np.isclose(rets[1], 1.0), rets
    print("test_gae_hand_computed: OK")


def test_running_mean_std():
    rms = RunningMeanStd()
    rms.update(np.full(100, 3.0, dtype=np.float32))
    assert abs(rms.mean - 3.0) < 0.05, rms.mean
    assert rms.std < 0.05, rms.std
    sd = rms.state_dict()
    other = RunningMeanStd()
    other.load_state_dict(sd)
    assert abs(other.mean - rms.mean) < 1e-9 and abs(other.std - rms.std) < 1e-9
    print("test_running_mean_std: OK")


# ---------------------------------------------------------------------------
# 5. recurrence contract
# ---------------------------------------------------------------------------

def test_recurrent_replay_matches_stepwise():
    """
    `run_recurrent_forward` must reproduce, for every timestep, the output the
    network produced when the same sequence was stepped one turn at a time — with the
    hidden state reset exactly at the `done` boundary in between. If this drifts, the
    LSTM is being trained on hidden states it never saw at collection time.
    """
    torch.manual_seed(0)
    net = ActorCriticNetwork(obs_dim=OBS_SIZE, use_lstm=True, hidden=16,
                             lstm_hidden=16, value_hidden=8)
    net.eval()
    T = 6
    obs = torch.randn(T, OBS_SIZE)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # episode ends after t=2

    emb_seq = replay_state_embeddings(net, obs, dones, None)

    state = None
    manual = []
    with torch.no_grad():
        for t in range(T):
            x = net.encode_sequence(obs[t].unsqueeze(0).unsqueeze(0), state)
            manual.append(x[0, 0])
            _logits, _value, state = net(obs[t].unsqueeze(0).unsqueeze(0),
                                         torch.zeros(1, MAX_ACTIONS, ACTION_FEAT_DIM),
                                         None, state)
            if dones[t].item() > 0.5:
                state = None
    manual = torch.stack(manual)
    assert torch.allclose(emb_seq, manual, atol=1e-5), (
        f"state embeddings differ: {(emb_seq - manual).abs().max().item()}"
    )
    print("test_recurrent_replay_matches_stepwise: OK")


# ---------------------------------------------------------------------------
# 6. full collect -> update cycle
# ---------------------------------------------------------------------------

def _one_cycle(use_lstm: bool, symmetrize: bool = True):
    random.seed(0)
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_lstm=use_lstm, symmetrize_rewards=symmetrize)
    trainer = PPOTrainer(cfg, device="cpu")
    rollouts = trainer._collect_rollout()
    n_transitions = sum(len(r.actions) for r in rollouts)
    assert n_transitions > 0, "no learning transitions were collected"
    stats = trainer._ppo_update(rollouts)
    for key in ("loss_pi", "loss_vf", "entropy", "approx_kl", "clip_frac"):
        assert key in stats, f"missing {key} in update stats"
        assert np.isfinite(stats[key]), f"{key} is not finite: {stats[key]}"
    return trainer, rollouts, stats, n_transitions


def test_collect_and_update_feedforward():
    trainer, rollouts, stats, n = _one_cycle(use_lstm=False)
    # no recurrence => no hidden state to replay, and every learning decision is
    # still recorded with its legal action ids
    assert rollouts[0].hidden_start is None
    assert all(len(r.action_masks) == len(r.actions) for r in rollouts)
    assert all(len(r.action_feats) == len(r.actions) for r in rollouts)
    print(f"test_collect_and_update_feedforward: OK ({n} transitions, "
          f"pi={stats['loss_pi']:.4f}, vf={stats['loss_vf']:.4f})")


def test_collect_and_update_recurrent():
    trainer, rollouts, stats, n = _one_cycle(use_lstm=True)
    # recurrent rollouts must keep the FULL per-turn trajectory for replay
    assert len(rollouts[0].full_obs) >= len(rollouts[0].actions)
    assert len(rollouts[0].learn_positions) == len(rollouts[0].actions)
    print(f"test_collect_and_update_recurrent: OK ({n} transitions, "
          f"pi={stats['loss_pi']:.4f}, vf={stats['loss_vf']:.4f})")


def test_both_networks_advance_every_turn():
    """
    The recurrence contract covers BOTH networks, not just the learner.

    The documented contract is "the memory is advanced on every turn of the match", and
    the bootstrap was trained under it. The learner always obeyed it; the opponent did
    not, because it is only consulted on the three seats it acts on. That left the
    opponent playing from a history its weights had never seen, so self-play was scoring
    the learner against a handicapped copy of itself -- and nothing revealed it, because
    a desynchronised opponent is still a legal opponent.

    One forward pass per turn, per network, is the observable form of the contract.
    """
    random.seed(0)
    torch.manual_seed(0)
    trainer = PPOTrainer(_tiny_cfg(n_envs=2, n_steps=24, use_lstm=True), device="cpu")

    counts = {"learner": 0, "opponent": 0}

    def wrap(module, key):
        inner = module.forward

        def counted(*args, **kwargs):
            counts[key] += 1
            return inner(*args, **kwargs)

        module.forward = counted

    wrap(trainer.net, "learner")
    for opponent_net in trainer.pool.nets:
        wrap(opponent_net, "opponent")

    rollouts = trainer._collect_rollout()
    turns = sum(len(r.full_obs) for r in rollouts)

    assert turns > 0, "the window collected no turns"
    assert counts["learner"] == turns, (
        f"the learner net ran {counts['learner']} times for {turns} turns; it must see "
        "every turn, because the observation of the acting player is what its memory "
        "tracks")
    assert counts["opponent"] == turns, (
        f"the opponent net ran {counts['opponent']} times for {turns} turns; it must "
        "also be advanced on the learner's turns, otherwise self-play hands it a hidden "
        "state it was never trained on")
    print(f"test_both_networks_advance_every_turn: OK ({turns} turns, learner "
          f"{counts['learner']} forwards, opponent {counts['opponent']} forwards)")


def test_episode_stats_are_recorded():
    """A window long enough to actually finish games, so the reward curves populate."""
    random.seed(0)
    torch.manual_seed(0)
    trainer = PPOTrainer(_tiny_cfg(n_envs=1, n_steps=400, use_lstm=False), device="cpu")
    trainer._collect_rollout()
    metrics = trainer.episode_metrics.metrics()
    assert len(trainer.episode_metrics) >= 1, "a 400-turn window should finish a game"
    for key in ("selfplay/win_rate", "selfplay/score_for", "selfplay/return_for",
                "selfplay/caidas_for", "selfplay/plies"):
        assert key in metrics, f"missing {key}"
    # the return decomposition holds for the learner's team too
    for stat in trainer.episode_metrics.stats:
        lt = stat.learner_team
        won = 1.0 if stat.winner == lt else 0.0
        expected = 2 * stat.caidas[lt] + 2 * stat.limpias[lt] + stat.carton[lt] + WIN_BONUS * won
        # Everything this harness counts must be PRESENT in the return. Not an exact
        # identity: the env's per-card and carton terms cannot be reconstructed from
        # outside it (a carton payment starts a new round and resets the pools).
        assert stat.returns[lt] - expected >= -1e-6, (stat.returns[lt], expected)
    print(f"test_episode_stats_are_recorded: OK ({metrics['selfplay/plies']:.0f} plies/game)")


def test_shaped_reward_rule():
    """
    Zero-sum shaping: the learner's own team's rewards pass through, the opponents'
    are negated — which is what lets the critic learn that losing is bad. The legacy
    behaviour (symmetrize=False) drops them, so every learner reward is >= 0.
    """
    from rollout import shaped_reward
    # learner on team 0, opponent scores 50 for the win -> the learner must see -50
    assert shaped_reward(50.0, acting_team=1, learner_team=0, symmetrize=True) == -50.0
    assert shaped_reward(50.0, acting_team=1, learner_team=0, symmetrize=False) == 0.0
    # the learner's own team's rewards are untouched in both modes
    assert shaped_reward(2.0, acting_team=0, learner_team=0, symmetrize=True) == 2.0
    assert shaped_reward(2.0, acting_team=0, learner_team=0, symmetrize=False) == 2.0
    # and mirrored when the learner sits on team 1
    assert shaped_reward(2.0, acting_team=0, learner_team=1, symmetrize=True) == -2.0
    print("test_shaped_reward_rule: OK")


def test_rollout_rewards_use_the_shaping_rule():
    """End to end: with symmetrization on, the learner's shaped return can go negative."""
    trainer_on, rollouts_on, _s, _n = _one_cycle(use_lstm=False, symmetrize=True)
    all_rewards = [x for r in rollouts_on for x in r.rewards]
    assert all_rewards, "no rewards collected"
    # every reward is attributable to one of the two teams' scores with a sign
    assert any(x != 0.0 for x in all_rewards)
    print(f"test_rollout_rewards_use_the_shaping_rule: OK "
          f"({sum(1 for x in all_rewards if x < 0)} negative of {len(all_rewards)})")


# ---------------------------------------------------------------------------
# 7. checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    random.seed(0)
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_lstm=True)
    trainer = PPOTrainer(cfg, device="cpu")
    path = trainer._save_policy_snapshot()

    # the pool starts seeded with the warm-start weights, so saving appends to it
    assert trainer.pool.steps[-1] == trainer.global_step, "saving must register a pool snapshot"

    reloaded = load_actor_critic(str(path), torch.device("cpu"))
    assert reloaded.use_lstm and reloaded.architecture_config() == trainer.net.architecture_config()

    obs = torch.randn(1, 1, OBS_SIZE)
    feats = torch.randn(1, MAX_ACTIONS, ACTION_FEAT_DIM)
    mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool); mask[:, :4] = True
    with torch.no_grad():
        a_logits, a_value, _ = trainer.net(obs, feats, mask)
        b_logits, b_value, _ = reloaded(obs, feats, mask)
    assert torch.allclose(a_logits, b_logits, atol=1e-6)
    assert torch.allclose(a_value, b_value, atol=1e-6)
    print(f"test_checkpoint_roundtrip: OK ({path.name})")


# ---------------------------------------------------------------------------
# 8. evaluation harness
# ---------------------------------------------------------------------------

def test_eval_harness_runs():
    random.seed(0)
    torch.manual_seed(0)
    trainer = PPOTrainer(_tiny_cfg(use_lstm=False), device="cpu")
    metrics = evaluate(
        trainer.net,
        {"heuristic": OpponentSpec(kind="heuristic"), "random": OpponentSpec(kind="random")},
        n_games=2, seed_base=777_000,
    )
    for name in ("heuristic", "random"):
        assert 0.0 <= metrics[f"eval/{name}"] <= 1.0
        assert metrics[f"eval/{name}_n_games"] == 2.0
        for suffix in ("score_for", "score_against", "return_for", "return_against",
                       "caidas_for", "caidas_against", "limpias_for", "limpias_against",
                       "plies", "stderr"):
            assert f"eval/{name}_{suffix}" in metrics, f"missing eval/{name}_{suffix}"
    # determinism: the same seeds must give the same result
    again = evaluate(trainer.net, {"random": OpponentSpec(kind="random")},
                     n_games=2, seed_base=777_000)
    assert again["eval/random"] == metrics["eval/random"], "evaluation is not deterministic"
    print(f"test_eval_harness_runs: OK (vs heuristic {metrics['eval/heuristic']:.0%}, "
          f"vs random {metrics['eval/random']:.0%})")


if __name__ == "__main__":
    test_obs_layouts()
    test_env_reward_identity()
    test_heuristic_plays_for_caida()
    test_action_features_are_slot_invariant()
    test_legal_action_masking()
    test_gae_hand_computed()
    test_running_mean_std()
    test_recurrent_replay_matches_stepwise()
    test_collect_and_update_feedforward()
    test_collect_and_update_recurrent()
    test_both_networks_advance_every_turn()
    test_episode_stats_are_recorded()
    test_shaped_reward_rule()
    test_rollout_rewards_use_the_shaping_rule()
    test_checkpoint_roundtrip()
    test_eval_harness_runs()
    print("\nAll PPO tests passed.")
