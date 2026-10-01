# test_search_finds_caida.py
import random
from env import CuarentaEnv, TEAMS
from ismcts import ISMCTS
from rules import legal_actions_from_state

# Construct a position where the current player has an obvious caída.
# Play random until P0 or P2 is on their turn with a caída available.
env = CuarentaEnv(seed=0)
env.reset(seed=0)

for step in range(200):
    if env.state.done:
        print("game ended early")
        break
    p = env.state.current_player
    if p in (0, 2):
        # Check whether any legal action gives +2 (caida/limpia)
        legal = env.legal_actions()
        from captures import score_play
        from utils import PointsReasons
        has_bonus = any(
            score_play(
                played=a.card, capture_set=a.capture, table=env.state.table,
                last_placed=env.state.last_placed, non_caida_allowed=True,
            )[1] in (PointsReasons.CAIDA, PointsReasons.LIMPIA)
            for a in legal
        )
        if has_bonus:
            searcher = ISMCTS(env, n_iter=300, rollout_policy="greedy",
                              rng=random.Random(step))
            visits = searcher.search()
            best_idx = max(visits, key=visits.get)
            best_action = legal[best_idx]
            pts, reason = score_play(
                played=best_action.card, capture_set=best_action.capture,
                table=env.state.table, last_placed=env.state.last_placed,
                non_caida_allowed=True,
            )
            print(f"Step {step}, P{p}, has caida/limpia. Search chose: {best_action}")
            print(f"  -> points={pts}, reason={reason}")
            print(f"  visits={visits}")
            break
    # random play
    legal = env.legal_actions()
    a = random.randrange(len(legal))
    env.step(a)