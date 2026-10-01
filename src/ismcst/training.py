import numpy as np
from ismcts import ISMCTS
import gymnasium as gym

env = gym.make("gymnasium_env/Cuarenta-v0").unwrapped
searcher = ISMCTS(env, n_iter=200, rollout_policy="greedy")

env.reset(seed=0)
done = False
while not done:
    visits = searcher.search()
    legal = env.legal_actions()
    if sum(visits.values()) == 0:
        # no info — pick random (shouldn't happen with n_iter>0)
        a = int(np.flatnonzero(env.observe(env.state.current_player)["action_mask"])[0])
    else:
        a = max(visits, key=visits.get)
    _, _, done, _, info = env.step(a)
    env.render()