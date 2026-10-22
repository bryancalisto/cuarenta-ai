import random, time
from env import CuarentaEnv
from ismcts import ISMCTS

env = CuarentaEnv(seed=0)
env.reset(seed=0)
searcher = ISMCTS(env, n_iter=100, rollout_policy="greedy", rng=random.Random(0))

# warmup
for _ in range(5):
    searcher.search()

t0 = time.time()
N = 50
for _ in range(N):
    searcher.search()
dt = (time.time() - t0) / N
print(f"search(n_iter=100): {dt*1000:.1f} ms")
print(f"throughput: {N/(time.time()-t0):.1f} searches/sec")
