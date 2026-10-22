import random
import time
from datetime import timedelta
from env import CuarentaEnv
from ismcts import ISMCTS
from rules import legal_actions_from_state
from tqdm import tqdm
from multiprocessing import Pool


def random_action(env):
    legal = env.legal_actions()
    return random.randrange(len(legal))


def ismcts_action(env, searcher):
    visits = searcher.search()
    if sum(visits.values()) == 0:
        return random.randrange(len(env.legal_actions()))
    return max(visits, key=visits.get)


def play_game(ismcts_seats, n_iter=100, rollout="greedy", seed=None):
    """ismcts_seats: set of player indices that use ISMCTS. Others are random."""
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    searcher = ISMCTS(env, n_iter=n_iter, rollout_policy=rollout, rng=random.Random(seed))
    while not env.state.done:
        p = env.state.current_player
        if p in ismcts_seats:
            a = ismcts_action(env, searcher)
            # legal_now = env.legal_actions()
            # print(f"[loop] p={p}, chosen a={a}, legal_now={len(legal_now)}")
        else:
            a = random_action(env)
        _, _, done, _, info = env.step(a)
    return info["winner"]

def worker_task(g):
    """
    A helper function that accepts a single argument (the seed/game index)
    since multiprocessing.Pool.imap works best with single-argument functions.
    """
    return play_game({0, 2}, n_iter=100, seed=g)

if __name__ == "__main__":
    # ISMCTS controls team 0 (players 0 and 2)
    start_time = time.perf_counter()
    wins = {0: 0, 1: 0}
    N = 100

    with Pool() as pool:
        results = tqdm(
            pool.map(worker_task, range(N)), 
            total=N, 
            desc='Games'
        )
        
        for winner in results:
            wins[winner] += 1

    print(f"ISMCTS (team 0) wins: {wins[0]}/{N} = {wins[0]/N:.1%}")
    end_time = time.perf_counter()
    execution_time_seconds = end_time - start_time
    readable_time = timedelta(seconds=execution_time_seconds)
    print(f"Execution time: {readable_time}")
    