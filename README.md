## Generating ONNX model
```bash
python3 -m scripts.pth2onnx checkpoints-ppo/bootstrap.pt --out backend/models/ppo_bootstrap.onnx
```
One ONNX call is one decision. The model takes the 208-wide observation shared with the
ISMCTS pipeline (see `OBS_KEYS_ISMCTS` in `src/ppo/utils.py`), a `[64, 84]` matrix of
candidate-action features plus a `[64]` legality mask, and the LSTM state
(`lstm_h`/`lstm_c`); it returns `action_logits [64]`, a scalar `value`, and the next LSTM
state. Slot `i` is *your* `i`-th legal action, so the backend plays `legal[argmax(logits[:n_legal])]`.
A sidecar `<name>.json` is written next to the model recording the architecture, the
observation keys and the sha256 of the source checkpoint.

Re-exporting is only half the job: the Go service must also agree with the PyTorch
network it was exported from. `scripts/onnx_parity_fixture.py` captures PyTorch's own
slot and value for real decisions, and `backend/internal/inference/policy_parity_test.go`
replays them through ONNX and compares the two, refusing to run if the fixture and the
exported model came from different checkpoints:

```bash
python3 scripts/onnx_parity_fixture.py                 # after every re-export
cd backend && go test ./internal/inference/ -run TestONNXMatchesPyTorchPolicy -v
```

That checks the export, but not the Go code that *builds* the model's inputs. The second
half of the guard is `scripts/state_parity_fixture.py`: it dumps real positions together
with the observation, the ordered legal actions, the action-feature block and the mask
that `src/ppo/env.py` computes from them, so the Go game core is compared against Python
rather than against its own constants. A wrong card-index convention or a differently
ordered hand would otherwise leave every Go test green while the model received a
well-formed but wrong vector.

```bash
python3 scripts/state_parity_fixture.py                # after changing either game core
cd backend && go test ./internal/game/ -v
```

# Training code

Three independent reinforcement-learning pipelines live in this repository, plus the
imitation-learning stage they warm-start from:

| path | what it is | how to run it |
|---|---|---|
| `src/imitation_learning/` | DAgger imitation learning from human play (Tkinter UI, imports the game core from `src/ppo/`) | `make train_with_imitation` |
| `src/ppo/` | **self-play PPO**, no search | `python3 src/ppo/train.py --help`, or `src/ppo/run_training.sh` |
| `src/ismcts/` | AlphaZero-style **information-set MCTS + network** | `./src/ismcts/run_training.sh` |
| `src/rebel/` | **ReBeL**: CFR search over a public belief state, belief-conditioned network | `./src/rebel/run_training.sh` |

`src/ppo/` is self-contained: the game core (`env.py`, `captures.py`, `card.py`), the
shared helpers (`utils.py`), the networks (`networks.py`), the agents (`agents.py`) and
the Tkinter UI (`ui.py`) all live in it, next to the RL code. The DAgger imitation
pipeline in `src/imitation_learning/` imports them from there.

## PPO

```bash
./src/ppo/run_training.sh            # supervised warm start + self-play (2M steps)
./src/ppo/run_training.sh --smoke    # seconds, no wandb

# or step by step
python3 src/ppo/bootstrap.py --games 2000 --epochs 8   # -> checkpoints-ppo/bootstrap.pt
python3 src/ppo/train.py --use-lstm true --n-envs 10 --steps 2_000_000
python3 src/ppo/test_ppo.py                            # obs layout, rewards, GAE, LSTM replay, eval
```

PPO uses the same representation as the ISMCTS pipeline: a 208-dim observation plus
84-dim feature vectors for each legal action (at most 64 slots), with a policy head that
scores actions from their features rather than from a flat action id.

Because that representation is shared, the two pipelines can be played against each
other, which is the only comparison that says anything about which one actually plays
better:

```bash
python3 src/eval_ppo_only_vs_ismcts.py --games 50 --workers 4   # or: make eval_ppo_vs_ismcts
python3 src/eval_ppo_only_vs_ismcts.py --n-iter 400 --ppo-checkpoint checkpoints-ppo/best.pt
python3 src/eval_ppo_only_vs_ismcts.py --games 100 --sweep 1,8,32,128 --workers 4
```

The `--sweep` run is usually the most informative: at 1 simulation ISMCTS's search is a
no-op, so that row compares the two *policies*, and each further row shows what search
adds on top of the ISMCTS network.

The Go backend serves players from `backend/models/ppo_bootstrap.onnx` — this policy
exported (see [Generating ONNX model](#generating-onnx-model)) — located through
`ONNX_MODEL_PATH` in `backend/.env`. After training a better checkpoint, export it and
regenerate the parity fixture, then point `ONNX_MODEL_PATH` at the new file.

**Serving an ISMCTS checkpoint.** The backend runs the ISMCTS search in Go
(`backend/internal/search/`) as well as the policy, because the policy alone is worth about a
coin flip against the heuristic (~49%) while the same weights with the search win ~77%:

```
AI_SEARCH_ITERATIONS=128    # in backend/.env; 0 = policy only (the old behaviour)
AI_SEARCH_CPUCT=1.5
```

Measured cost/benefit per AI turn, against the heuristic (100 games each, ±~5%): 0 sims
49.0%, 32 sims 57.0%, 64 sims 59.0%, 128 sims 72.0%, **400 sims 77.0%**.

The Go search is fast enough to afford the full budget — MEASURED in Go on the same export
(only the simulations that expand a node call the model):

| budget | per AI turn |
|---|---|
| 128 sims | ~10 ms (750 decisions in 7.6 s) |
| 400 sims | ~31 ms (1231 decisions in 38.9 s) |

so there is no latency reason to serve the policy-only path: 400 sims costs ~31 ms and buys
the full 77%.

And it plays like the original. The Go search against the BARE POLICY (the Go equivalent of
Python's `teacher_edge`) measured **10-2 / 83%** over 12 games at 128 sims and
**14-6 / 70%** over 20 games at 400 sims, against Python's 56% at 64 sims and 67-85% at 400.
Run it yourself:

```bash
ISMCTS_MATCH_GAMES=20 ISMCTS_MATCH_ITER=400 \
  go test ./internal/inference/ -run TestSearchBeatsTheBarePolicy -v
```

The Go search mirrors `src/ismcts/network_ismcts_1.py` (PUCT over sampled determinizations,
the dense-reward backup in return mode, the FPU convention for unvisited children) and takes
the value target from the ONNX sidecar, so a return-mode checkpoint is searched in return
mode. It refuses to run for the PPO model (which carries a recurrent state tied to the live
game, so sampled positions cannot be evaluated) and falls back to the bare policy.

The export path for an ISMCTS checkpoint is separate from PPO's:

```bash
make export_ismcts_onnx ISMCTS_POLICY=checkpoints-return/best.pt RVALUE=return
make test_onnx_ismcts                       # PyTorch-vs-ONNX parity through the Go service
ONNX_MODEL_PATH=backend/models/ismcts_return.onnx   # in backend/.env
```

`scripts/pth2onnx_ismcts.py` refuses to export when it cannot tell which value target the
weights were trained with (the two heads differ only by an activation, so the weights cannot
say), takes it from the `--value-target` flag or the checkpoint's `.json` sidecar written by
the trainer, verifies torch-vs-ONNX parity itself, and asserts the exported graph's head
activation matches that claim. The Go side accepts both signatures, deciding from the graph
which one a model uses, and `backend/models/*.json` records `value_target`/`value_semantics`
because the `value` output means a win estimate in win mode and a shaped return in game
points in return mode.

Everything a run produces lands in `checkpoints-ppo/`: `bootstrap.pt`, periodic
`policy_<step>.pt` snapshots, `best.pt` (the net that won the reference-gated match),
`best_heuristic.pt`, a `training_metrics_*.png` summary and the raw metrics as `.npz`.
Metrics go to Weights & Biases (project `cuarenta-ai`, the same one the ISMCTS runs use)
with the training step as the x-axis, under four families:

* `train/*`  — policy/value loss, entropy, `approx_kl`, `clip_frac` (PPO health),
* `selfplay/*` — win rate, game scores, episode returns, caídas, limpías and game length
  of the games being generated,
* `eval/*`   — win rate, scores, returns and bonus counts for every member of the
  evaluation league: `reference` (best-so-far, used for gating), `heuristic` (the
  rule-based bot), `initial` (the policy the run started from), `prev_checkpoint` and
  `random`, each with a standard error,
* `time/*`   — steps per second and where the seconds went.

## ISMCTS

```bash
./src/ismcts/run_training.sh                                  # imitation + self-play (~4h)
./src/ismcts/run_training.sh --iters 2 --games-per-iter 2     # smoke test
./src/ismcts/run_training.sh --value-target return            # optimise the dense reward
python3 src/ismcts/test_value_target.py                       # the two value targets
```

AlphaZero-style: PUCT over sampled determinizations, the policy target is the search's
visit distribution, the value target is the game result. `--value-target` decides what the
run actually optimises, and it changes BOTH the head and the search, because the head is
the search's leaf evaluation:

* `win` (default) — the head predicts ±1 game results and a terminal node is worth ±1;
  per-step rewards are ignored, so the search maximizes win probability and Q is symmetric
  around 0.
* `return` — the head is linear and predicts the env's dense reward as a scaled
  return-to-go (`RETURN_VALUE_SCALE`); the search accumulates rewards down every simulation
  path, `+` for the root's team and `−` for the opponent's, and scores an unvisited child
  as neutral rather than 0. It then chases caídas/captures instead of win probability —
  not the same objective, since the win bonus is only ~15% of a game's return.

The mode is read off the net, so every evaluator searches with the semantics its checkpoint
was trained with; a return-mode run must not warm-start from a win-mode checkpoint (the
trainer notes this, because neither the mode nor `RETURN_VALUE_SCALE` can be detected from
the weights).

**How to know the value head is failing before the win rate says so.** In return mode the
search's Q *is* the value head: a capture is worth `1/30` of a unit while the leaf value is
worth ~1, so a head that is confidently wrong destroys the search. `probe_value_r2` (logged
every iteration, held-out games that are never trained on) is the number to watch —
0 means "as good as predicting the mean", negative means worse. Measured on a real run:
in-sample R² 0.77 while held-out R² was **−0.27**, with the head's output magnitude growing
70× (the search then followed noise), and the net fell from 73% to 42% against the heuristic
within 5 iterations. The causes were a 1.6M-parameter net replayed 20× over a ~25k buffer,
and a Monte-Carlo return target whose std is ~35 points and mostly unpredictable from the
state. Knobs, in the order worth trying: `--epochs-per-iter`/`--replay-iters` (passes),
`--dropout`, `--init-checkpoint` from a return-mode bootstrap plus
`--value-anchor-weight 2`, and `--value-lambda` (below).

`--value-lambda` (return mode; default 0.9, `1.0` = pure Monte-Carlo) trains the value head
on TD(λ) returns — the net's own value bootstraps the tail instead of the realised future:

    G_t = reward_t + (1 - λ)·V(s_{t+1}) + λ·G_{t+1},   G_T = 0

That is the same expected value as the Monte-Carlo target (so the objective is unchanged),
estimated with less variance. **It only helps while V is any good**: measured on the same
games, a head with no skill cut the label spread from 34.7 to 9.5 points (λ=0.95), while an
already-overfitted head *raised* it from 24.6 to 33.5 (λ=0.95) / 39.1 (λ=0.9), because the
`(1 − λ)` weight injects V's error into every label. Watch `selfplay/value_target_std` vs
`selfplay/value_mc_std` (also printed on the self-play line): λ is buying something only
while the first is the smaller number. That is why the recipe starts from a return-mode
bootstrap: warm-start V first, then bootstrap from it.

`--members a,b,c` trims the evaluation league (`reference` is required — the gate reads it);
the default set is `reference,heuristic,initial,random,teacher_edge`. Every member costs its
own games, so it is the knob for cutting evaluation time.

## ReBeL

```bash
python3 src/rebel/bootstrap.py         # warm start: imitate the heuristic -> bootstrap.pt
./src/rebel/run_training.sh            # self-play + training (default: 20 iterations)
./src/rebel/run_training.sh --smoke    # seconds, no wandb
python3 src/rebel/test_rebel.py        # CFR on Kuhn poker, belief plumbing, search, checkpoints
python3 src/rebel/eval_rebel.py --checkpoint checkpoints-rebel/best.pt --games 20
```

ReBeL (Recursive Belief-based Learning) replaces MCTS with a CFR search over the **public
belief state**: instead of averaging searches on sampled determinizations (which is
unsound with hidden information), it solves the subgame as a game — the network values the
leaves, and the belief, a weighted particle set over the hidden hands that is updated by
Bayes whenever an action is observed, defines the chance node. Because the value must
answer "how good is this given what everyone *could* be holding", the network reads the
208-dim observation plus 160 belief marginals (4 seats × 40 cards). Training is the usual
self-play loop: the search's root strategy is the policy target, the game's win/loss is the
value target. Like the other pipelines, `src/rebel/` is self-contained (its own copy of
the game core) and reuses the shared 208/84 representation; checkpoints, metrics and the
`best.pt` gating against the heuristic bot mirror `src/ppo/`.

## More info about Cuarenta
https://40caidaylimpia.com/reglas-para-jugar-40/

## Comparing pipelines: the shared `eval/*` metrics

All three pipelines (PPO, ISMCTS, ReBeL) log the same evaluation league under the same key
names, and define every family against the same x-axis (`iteration`), so their curves can be
overlaid on one chart in the Weights & Biases console (project `cuarenta-ai`).

Members — the evaluated network plays a TEAM, the opponent plays the other:

| member | opponent |
|---|---|
| `reference` | the gated best-so-far checkpoint |
| `heuristic` | the rule-based bot (always evaluated) |
| `initial` | the checkpoint the run started from |
| `prev_checkpoint` | the snapshot before this evaluation |
| `random` | uniform-random legal moves |

Per member, 12 keys, with `_for` = the net's team and `_against` = the opponent:

```
eval/<member>                  eval/<member>_caidas_for      eval/<member>_plies
eval/<member>_stderr           eval/<member>_caidas_against  eval/<member>_n_games
eval/<member>_score_for        eval/<member>_limpias_for
eval/<member>_score_against    eval/<member>_limpias_against
eval/<member>_return_for       (stderr = sqrt(p(1-p)/n))
eval/<member>_return_against
```

`return` is the environment's dense reward: `cards_captured + 2*caidas + 2*limpias +
carton + WIN_BONUS*won` — the cartón points are paid too (see the
`REWARD` note in `src/ismcts/env.py`, which all three pipelines follow). `WIN_BONUS` is a
single constant in each `env.py`; check that it matches before comparing return charts
across pipelines. Members are skipped — with no keys logged at all — when their checkpoint
does not exist yet, rather than reporting stale numbers.

Known differences to keep in mind when reading across pipelines:

* **Seat convention.** ISMCTS always puts the net on team 0 (a seat edge is baked in, which
  is why its gate is calibrated against a mirror baseline). PPO rotates the policy seat over
  all four seats; ReBeL alternates the net's team game by game.
* **Checkpoint opponents.** PPO and ReBeL have a checkpoint opponent play the argmax of its
  own policy; ISMCTS has it search as well. So `eval/reference` is a strict mirror in ISMCTS
  and "search vs a raw policy" in the other two.
* **Win rates, scores, bonus counts and plies** are comparable across pipelines;
  **`_return_*` is comparable only because all three now share one reward formula** — it is
  not comparable with runs from before that change.
