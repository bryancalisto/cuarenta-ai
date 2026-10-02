# profile_search.py
import cProfile, pstats, random
from env import CuarentaEnv
from ismcts import ISMCTS

env = CuarentaEnv(seed=0)
env.reset(seed=0)
searcher = ISMCTS(env, n_iter=100, rollout_policy="greedy", rng=random.Random(0), max_rollout_len=40)

cProfile.run("for _ in range(20): searcher.search()", "prof.out")
p = pstats.Stats("prof.out")
p.sort_stats("cumulative").print_stats(20)