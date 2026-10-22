"""
Head-to-head: NetworkISMCTS (untrained net) vs rollout ISMCTS.
Network plays team 0 (seats 0 and 2); rollout plays team 1 (seats 1 and 3).

Uses multiprocessing.Pool so each game runs in its own process.
"""
import random
import time
from datetime import timedelta
from multiprocessing import Pool

from tqdm import tqdm

from env import CuarentaEnv
from ismcts import ISMCTS as RolloutISMCTS
from network_ismcts_1 import NetworkISMCTS
from network import CuarentaNet
from history import HistoryTracker


# ---------------------------------------------------------------------------
# Single-game worker
# ---------------------------------------------------------------------------

def play_game(n_iter: int = 100, seed: int = 0, dirichlet_eps: float = 0.0) -> int:
    """
    Runs one game. Network plays seats {0, 2}, rollout plays seats {1, 3}.
    Returns the winning team index (0 or 1).
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    # Fresh network per worker. Do NOT share a net across processes.
    # Using torch.set_num_threads(1) avoids oversubscription — the OS is
    # already parallelizing across games, so intra-op threads just fight.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # Deliberately a FRESH random net: this script is a baseline ("what does an untrained
    # net with a search do against rollout-ISMCTS?"), so there is no checkpoint and no
    # trained value target to resolve.
    net = CuarentaNet()
    net.eval()
    tracker = HistoryTracker(env)

    searcher_net = NetworkISMCTS(
        env, net, tracker,
        n_iter=n_iter,
        rng=random.Random(seed * 2 + 1),
        dirichlet_eps=dirichlet_eps,
        max_rollout_len=200,
    )
    searcher_rollout = RolloutISMCTS(
        env,
        n_iter=n_iter,
        rollout_policy="greedy",
        rng=random.Random(seed * 2 + 2),
    )

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)

        if p in (0, 2):
            visits = searcher_net.search()
        else:
            visits = searcher_rollout.search()

        legal = env.legal_actions()
        if sum(visits.values()) == 0:
            a = random.randrange(len(legal))
        else:
            a = max(visits, key=visits.get)

        # Record BEFORE stepping so the tracker reflects what the player saw
        # and chose. NetworkISMCTS relies on the tracker at the root.
        tracker.record(p, obs, a)
        _, _, done, _, info = env.step(a)

    return info["winner"]


def worker_task(g: int) -> int:
    """Single-argument wrapper for Pool.map."""
    return play_game(n_iter=100, seed=g)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 100
    wins = {0: 0, 1: 0}

    start = time.perf_counter()
    with Pool() as pool:
        results = tqdm(
            pool.map(worker_task, range(N)),
            total=N,
            desc="Games",
        )
        for w in results:
            wins[w] += 1

    elapsed = time.perf_counter() - start
    print(f"\nNetworkISMCTS (untrained) wins: {wins[0]}/{N} = {wins[0]/N:.1%}")
    print(f"RolloutISMCTS wins:              {wins[1]}/{N} = {wins[1]/N:.1%}")
    print(f"Execution time: {timedelta(seconds=elapsed)}")
    