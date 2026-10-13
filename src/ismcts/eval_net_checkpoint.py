"""
Load a checkpoint and play it against rollout-ISMCTS.

Usage:
    python eval_net_checkpoint.py checkpoints/cuarenta_iter000.pt 50
"""
import sys, random
from multiprocessing import Pool
from tqdm import tqdm
import torch
from env import CuarentaEnv
from network import CuarentaNet
from history import HistoryTracker
from network_ismcts_1 import NetworkISMCTS
from ismcts import ISMCTS as RolloutISMCTS

# Optional `--value-target win|return` anywhere on the command line; otherwise the
# checkpoint's sidecar decides (and `resolve_value_target` warns if there is neither).
VALUE_TARGET = None
if "--value-target" in sys.argv:
    VALUE_TARGET = sys.argv[sys.argv.index("--value-target") + 1]
    del sys.argv[sys.argv.index("--value-target"):sys.argv.index("--value-target") + 2]


def play_one(seed: int, ckpt_path: str, n_iter: int = 100) -> int:
    import torch
    torch.set_num_threads(1)
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)
    # `from_checkpoint` builds the head the weights were trained with (sidecar, or an
    # explicit --value-target) and loads them; the mode decides how the search behaves, so
    # getting it wrong evaluates a different agent (see CuarentaNet.resolve_value_target).
    net = CuarentaNet.from_checkpoint(ckpt_path, value_target=VALUE_TARGET)
    net.eval()
    tracker = HistoryTracker(env)

    net_searcher = NetworkISMCTS(
        env, net, tracker, n_iter=n_iter, max_rollout_len=200,
        rng=random.Random(seed * 2 + 1), dirichlet_eps=0.0,
    )
    rollout_searcher = RolloutISMCTS(
        env, n_iter=n_iter, rollout_policy="greedy",
        rng=random.Random(seed * 2 + 2),
    )

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        if p in (0, 2):
            visits = net_searcher.search()
        else:
            visits = rollout_searcher.search()
        legal = env.legal_actions()
        if sum(visits.values()) == 0:
            a = random.randrange(len(legal))
        else:
            a = max(visits, key=visits.get)
        tracker.record(p, obs, a)
        _, _, _, _, info = env.step(a)
    return info["winner"]


def worker(arg):
    seed, ckpt = arg
    return play_one(seed, ckpt)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    with Pool() as pool:
        results = list(tqdm(pool.imap(worker, [(s, ckpt) for s in range(N)]), total=N))
    wins = sum(1 for w in results if w == 0)
    print(f"{ckpt}: net-ISMCTS (team 0) wins {wins}/{N} = {wins/N:.1%} vs rollout-ISMCTS")