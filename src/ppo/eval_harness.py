"""
Deterministic, parallel evaluation (the PPO counterpart of the ISMCTS harness).

Same match summary, same metric names, so both pipelines produce comparable curves:
`eval/<name>`, `eval/<name>_score_for/_against`, `_return_for/_against`,
`_caidas_for/_against`, `_limpias_for/_against`, `_plies`, `_stderr`.

WHAT A MATCH IS
---------------
The policy under test plays ONE seat, rotated deterministically over the four seats
across games so every seat is exercised and no seat advantage is baked in. The
opponent plays the other three. Every turn is fed to the policy network (so its LSTM
state evolves exactly as during collection); only its own turns produce actions.
Actions are greedy, env seeds are fixed and the opponent is seeded per game, so the
same (checkpoint, seed) always gives the same result.

Because the seats rotate, a mirror match (policy vs itself) is 50% IN EXPECTATION, so no
systematic seat bias is baked in — unlike the ISMCTS harness, where the policy always sat
on team 0 and the bar had to be calibrated for that offset. Expectation is not a
guarantee, though: over a finite number of games the measured mirror wanders (one real
run measured 41.7% over 60 games), which is why the trainer calibrates its acceptance bar
against a measured mirror baseline rather than assuming exactly 50%.

Reported per match (`_for` = the policy's team, `_against` = the opponent), and
`return` is the env's dense reward (`cards + 2*caidas + 2*limpias + carton + WIN_BONUS*won`), reported for both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from env import ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv, TEAMS, action_features_matrix
from utils import PointsReasons, flatten_obs, has_reason
from agents import HeuristicAgent, RandomAgent
from networks import ActorCriticNetwork
from rollout import select_legal_action


@dataclass
class OpponentSpec:
    """
    Who the policy plays against.

    kind: "heuristic" | "random" | "checkpoint".
    For "checkpoint", `state_dict` holds a PPO snapshot (same architecture as the
    policy under test) and `use_lstm` says whether it is recurrent.
    """
    kind: str = "heuristic"
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    use_lstm: bool = False


# ---------------------------------------------------------------------------
# One match
# ---------------------------------------------------------------------------

def _play_match(policy: ActorCriticNetwork, opponent: OpponentSpec,
                game_idx: int, seed: int, device: torch.device) -> Dict:
    """Play a single evaluation game; returns the match summary dict."""
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    policy_seat = game_idx % 4                 # rotate seats: 0,1,2,3,0,...
    policy_team = TEAMS[policy_seat]

    if opponent.kind == "heuristic":
        opp_agent = HeuristicAgent(seed=seed * 2 + 1)
        opp_net = None
    elif opponent.kind == "random":
        opp_agent = RandomAgent(seed=seed * 2 + 1)
        opp_net = None
    elif opponent.kind == "checkpoint":
        opp_agent = None
        opp_net = ActorCriticNetwork(**policy.architecture_config()).to(device)
        opp_net.load_state_dict(opponent.state_dict)
        opp_net.eval()
    else:
        raise ValueError(f"unknown opponent kind: {opponent.kind}")

    policy_state = None
    opponent_state = None
    returns = [0.0, 0.0]
    caidas = [0, 0]
    limpias = [0, 0]
    plies = 0

    with torch.no_grad():
        while not env.state.done:
            player = env.state.current_player
            legal = env.legal_actions()
            obs_t = torch.tensor(flatten_obs(env.observe(player)), dtype=torch.float32,
                                 device=device).unsqueeze(0)
            feats_np, mask_np = action_features_matrix(legal, env.state)
            feats_t = torch.as_tensor(feats_np, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask_np, device=device).unsqueeze(0)

            # Both networks are advanced on EVERY turn — the same contract as training
            # (`trainer._collect_rollout`), where the recurrent state accumulates the
            # whole match rather than only the player's own turns. A network that only
            # saw its own seats here would be evaluated out of distribution.
            policy_logits, _, policy_state = policy(obs_t, feats_t, mask_t, policy_state)

            if player == policy_seat:
                slot, _, _ = select_legal_action(policy_logits[0], mask_t[0],
                                                 deterministic=True)
                action = legal[slot]
            elif opp_net is not None:
                opp_logits, _, opponent_state = opp_net(obs_t, feats_t, mask_t, opponent_state)
                slot, _, _ = select_legal_action(opp_logits[0], mask_t[0],
                                                 deterministic=True)
                action = legal[slot]
            else:
                action = opp_agent.act(env.observe(player), legal,
                                       state=env.state, team=TEAMS[player])

            _obs, reward, _done, info = env.step(action)

            team = TEAMS[player]
            returns[team] += float(reward)
            reasons = info.get("reasons") or []
            if has_reason(reasons, PointsReasons.CAIDA):
                caidas[team] += 1
            elif has_reason(reasons, PointsReasons.LIMPIA):
                limpias[team] += 1
            plies += 1

    opp_team = 1 - policy_team
    return {
        "winner": env.state.winner,
        "policy_seat": policy_seat,
        "policy_team": policy_team,
        "won": 1 if env.state.winner == policy_team else 0,
        "score_for": int(env.state.scores[policy_team]),
        "score_against": int(env.state.scores[opp_team]),
        "return_for": returns[policy_team],
        "return_against": returns[opp_team],
        "caidas_for": caidas[policy_team],
        "caidas_against": caidas[opp_team],
        "limpias_for": limpias[policy_team],
        "limpias_against": limpias[opp_team],
        "plies": plies,
    }


def _worker(payload):
    """Pool worker: run one evaluation game."""
    seed, game_idx, policy_state, arch, opp_kind, opp_state = payload
    device = torch.device("cpu")
    torch.set_num_threads(1)

    policy = ActorCriticNetwork(**arch).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()

    opponent = OpponentSpec(kind=opp_kind, state_dict=opp_state)
    return _play_match(policy, opponent, game_idx, seed, device)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def match_metrics(results: List[Dict], prefix: str) -> Dict[str, float]:
    """Average a batch of match summaries into flat scalars (same names as ISMCTS)."""
    if not results:
        return {}
    n = len(results)

    def m(key):
        return float(np.mean([r[key] for r in results]))

    win_rate = m("won")
    stderr = float(np.sqrt(max(win_rate * (1 - win_rate), 1e-9) / n))
    return {
        f"eval/{prefix}": win_rate,
        f"eval/{prefix}_stderr": stderr,
        f"eval/{prefix}_score_for": m("score_for"),
        f"eval/{prefix}_score_against": m("score_against"),
        f"eval/{prefix}_return_for": m("return_for"),
        f"eval/{prefix}_return_against": m("return_against"),
        f"eval/{prefix}_caidas_for": m("caidas_for"),
        f"eval/{prefix}_caidas_against": m("caidas_against"),
        f"eval/{prefix}_limpias_for": m("limpias_for"),
        f"eval/{prefix}_limpias_against": m("limpias_against"),
        f"eval/{prefix}_plies": m("plies"),
        f"eval/{prefix}_n_games": float(n),
    }


def evaluate(
    policy: ActorCriticNetwork,
    opponents: Dict[str, OpponentSpec],
    *,
    n_games: int,
    seed_base: int,
    workers: Optional[int] = None,
    keep_results_for: Optional[str] = "heuristic",
) -> Dict[str, float]:
    """
    Evaluate `policy` against each named opponent, in parallel worker processes.

    Uses the same seeds for every evaluation (game `i` gets `seed_base + i`), which
    turns successive evaluations into paired comparisons — the cheapest way to resolve
    small improvements.
    """
    from multiprocessing import Pool

    policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    arch = policy.architecture_config()

    metrics: Dict[str, float] = {}
    with Pool(workers) as pool:
        for name, spec in opponents.items():
            payloads = [
                (seed_base + i, i, policy_state, arch, spec.kind,
                 spec.state_dict if spec.kind == "checkpoint" else None)
                for i in range(n_games)
            ]
            results = list(pool.imap(_worker, payloads))
            metrics.update(match_metrics(results, name))
            if keep_results_for == name:
                metrics["_results"] = results
    return metrics
