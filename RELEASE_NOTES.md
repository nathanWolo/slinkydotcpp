# slinky v0.1.0

The first release on `main` (2026-10-15). The package version in
`pyproject.toml` is 0.1.0.

slinky is a Battlesnake simulator written in JAX that follows the official
rules engine turn for turn, together with the agents, trainers and evaluation
tools built on it. The trainers cover PPO, REINFORCE, DQN, Rainbow DQN and
AlphaZero. There are single-mode agents for the 11×11 duel and multi-mode
agents: one network that plays the four leaderboard modes (duel, standard,
royale and constrictor) and is told which mode it is playing.

The repository's first two commits (2023) are a small C++ prototype, kept in
`legacy/` in v0.1.0 and removed after it (it stays in the history). Everything else was written in October 2026 and reached `main`
through this release.

## What is in the release

### The environment (`src/slinky/`)

- `reset` and `step` are pure JAX functions, so they can be `jit`ted,
  `vmap`ped over thousands of games and `scan`ned over turns. Each snake is
  stored as a countdown grid, so a step is a fixed number of dense array
  operations.
- Rulesets: standard, solo, wrapped, constrictor, wrapped constrictor and
  royale. Maps: standard, empty and royale (the shrinking safe zone).
- Per-agent egocentric or allocentric observations (13 channels), rewards
  (win/loss, or `placement_reward` for games with more than two snakes) and
  action masks. `step_autoreset` restarts finished games and keeps the final
  observation for bootstrapping.
- The turn resolution is checked against the official Go rules engine
  (`tools/oracle/`, used only by the tests). The tests replay games the engine
  generated (1v1 and 4-, 8- and 12-player games, boards from 7×7 to 19×19,
  wrapped, constrictor, solo, royale and three hazard maps), send states from
  slinky's own rollouts to the engine, and fuzz states that games rarely
  reach. Start positions and food spawns follow the engine's distributions,
  not its random-number stream.
- `slinky.modes` defines the four leaderboard modes in one place, with the
  side inputs a multi-mode network needs (mode one-hot, turn, royale shrink
  phase, share of opponents alive).
- `slinky.engine_json` converts between slinky states and the engine's
  board-state JSON; the server converts Battlesnake API requests to it.

### Reference agents

- `random_legal`: uniform among the moves that do not certainly die next turn.
- `heuristic` (`slinky.heuristic`): a hand-written snake with time-aware flood
  fills, Voronoi territory and a one-ply simultaneous-move search over the
  exact rules.
- `mcts-<n>` (`slinky.mcts`): simultaneous-move MCTS (decoupled UCT, the
  heuristic's evaluation at the leaves). The same search, with PUCT, network
  priors and network leaf values, is AlphaZero's.

### Trainers (`baselines/`)

Single files, one per algorithm, all in JAX (optax for the optimisers).

- **Duel (11×11, 2 snakes):** `dqn.py` (Double DQN), `rainbow.py` (all six
  Rainbow extensions, a distributional head on [−1, 1]), `ppo.py`,
  `reinforce.py` (with a value baseline, or none) and `az.py` (AlphaZero:
  self-play through `slinky.mcts`'s PUCT search, all in JAX). `az_loop.py`
  repeats `az.py` in generations and keeps a new network only if it beats the
  current best in a gated 4,096-game match.
- **Multi-mode:** `ppo_modes.py` (self-contained) and `reinforce_modes.py`,
  `dqn_modes.py`, `rainbow_modes.py` and `az_modes.py`, which share
  `slinky.unified`. Each trains one network on all four modes at once, with
  one group of games per mode, the placement reward and the mode side inputs.
  The `unified` trainers take `--size S|M|L` (three network sizes), and a
  `STOP` file in the run directory makes them checkpoint and exit.
  `--cooldown` then anneals the learning rate to 0. `az_modes.py` searches
  all four modes in one lockstep search per turn.
- Every trainer writes `config.json`, `metrics.jsonl`, `params.npz` and a
  resumable `runner.npz` under `runs/`.

### Agents registry (`slinky.agents`)

Every agent above, and every trained network, has a name that the replay
tool, the benchmarks and the server all accept: `random_legal`, `heuristic`,
`mcts-256`, `dqn`, `ppo`, `ppo-sample`, `rainbow`, `reinforce`, `az-64`,
`ppo_modes`, `az_modes:<run dir>:simulations=256`, and so on. A name can
point at a run directory and override settings with `:field=value` items.
A multi-mode agent is told the mode of the game it plays, or another mode with
`:game_mode=`.

### Evaluation and benchmarks

- `slinky.evaluate.play_match` (two policies, seat rotation, 95% intervals)
  and `play_table` (any number of policies at a 4-snake table, scored by share
  of the win and mean placement). Game `g` depends only on `(key, g)`, so
  results do not depend on the batch size.
- `benchmarks/strength.py` (resumable all-pairs matchups, `--workers` for one
  pinned process per core), `benchmarks/modes.py` (the four modes at their own
  tables, generalist against specialists, paired game-by-game comparisons),
  `benchmarks/mcts_latency.py` (the MCTS budget that fits a time limit),
  `benchmarks/throughput.py`, and the charts `plot_strength.py` and
  `plot_az_ladder.py`.
- `benchmarks/plateau.py` watches a multi-mode run's snapshots, plays each one
  against the heuristic in every mode and stops the run when a trend rule finds
  the strength flat. `benchmarks/compare_unified.py` plays the final panel and
  round robin of finished multi-mode runs.

### Playing and watching games

- **Ladder server** (`python -m slinky.server`): serves registry agents over
  the official Battlesnake API, one URL path per snake, several snakes per
  server. It reads the game config from each request. A multi-mode agent plays
  every 11×11 game that has a mode, including royale (with the zone's shrink
  sides read from the request) and constrictor on the empty map. The
  single-mode networks play the 11×11 duel only. Other games get the
  `--fallback` agent (the heuristic by default). `--move-log` writes one JSON
  line per answer and flags answers the engine did not apply (a lower bound on
  timeouts).
- **AWS Lambda snake** (`deploy/lambda/`): the search-free trained networks
  ported to plain NumPy behind one Function URL, with build and deploy
  scripts. `tests/test_snake_numpy.py` checks the port against the JAX agents.
- **Replays** (`python -m slinky.replay`): records games between named agents
  and writes a self-contained HTML viewer.
- **Training dashboard** (`python baselines/dashboard.py`): a live page over
  every run under `runs/` and `baselines/checkpoints/`, for single-mode and
  multi-mode runs, with per-mode charts, a strength-against-search chart, a
  games tab with the replay viewer, and a light/dark theme. It needs only the
  standard library.

### PufferLib port (`puffer/`)

The game ported to C as a PufferLib 3.0 environment (rules, maps, rewards,
observations, checked against slinky and the engine), `ppo.py`'s network in
PyTorch, and training with PufferLib's PPO trainer. A trained network exports
as a `ppo.py` checkpoint, so `ppo:<run dir>` plays it. It needs its own
virtual environment; see `puffer/README.md`.

### Checkpoints (`baselines/checkpoints/`)

The duel networks (`dqn-duel-seed0`, `ppo-duel-seed0`, `rainbow-duel-seed0`,
`rainbow-long-seed0`, `reinforce-duel-seed0`, `az-duel-seed0`), the multi-mode
PPO generalist (`ppo-modes-seed0`) with its four specialists, two more duel
specialist seeds and three 16-minibatch controls, and the two PufferLib
networks (`puffer-duel-seed0`, `puffer-duel-matched-seed0`).

## Results

Scores are win + ½ draw (or share of the win in 4-snake games), from the
first-named agent's side, with seats rotated. Intervals are 95%. Unless
stated otherwise each result is one training seed. The full tables are in
`baselines/README.md`, `benchmarks/README.md` and `puffer/README.md`.

### The duel

| agent | training | vs `random_legal` | vs heuristic |
|---|---|---|---|
| Double DQN | 1.28M env steps, 31 min, CPU | 0.998 | 0.053 |
| Rainbow DQN | 1.28M env steps, 81 min, CPU | 0.983 | 0.084 |
| Rainbow DQN, trained on | 4.61M env steps, 5.9 h, CPU | 0.986 | 0.158 |
| REINFORCE | 5.25M env steps, 73 min, CPU | 0.944 | 0.013 |
| PPO | 5.24M env steps, 67 min, 3 CPU cores | 0.998 | 0.296 |
| PPO in PufferLib | 19.7M env steps, 57 min of training, 4 CPU cores | 0.996 | 0.434 |
| AlphaZero raw network (`az-0`) | 8.39M env steps, 9.7 h, 4 CPU cores | 0.997 | 0.484 |
| AlphaZero, 64 simulations (`az-64`) | same network | 0.999 | 0.853 |

Game counts: the DQN, PPO and PufferLib rows are from the strength benchmark
(1,024 games per opponent), the AlphaZero rows too (512 games). Rainbow and
REINFORCE are from the baselines comparison (5,000 games against
`random_legal`, 1,000 against the heuristic).

- **Head to head** (1,000 games each, one harness): PPO beats the DQN
  0.873 ± 0.021 and Rainbow 0.753 ± 0.027; Rainbow and the DQN are level
  (0.504 ± 0.031); REINFORCE loses to PPO 0.055 ± 0.014. `az-0` beats PPO
  0.581 [0.538, 0.623] over 512 games. The PufferLib network beats PPO 0.620
  and is level with `az-0` (0.501), over 1,024 games each.
- **MCTS** against the heuristic (512 games per point unless noted): 0.528 at
  32 simulations, 0.777 at 1,024, 0.846 at 2,048 (256 games). At 24,000
  simulations, the most that fit in Battlesnake's 500 ms per move on one core
  at p90, it scores 0.875 [0.807, 0.922] against the heuristic and 0.949
  [0.896, 0.976] against PPO (128 games each).
- **AlphaZero's search** is worth more per simulation than plain MCTS: `az-64`
  is level with MCTS-2048 (0.489) and MCTS-4096 (0.501), 512 games each.
  `az-256` scores 0.586 [0.499, 0.668] against MCTS at 500 ms per move over
  128 games, at a seventh of its time per move.

### Multi-mode PPO (committed checkpoint)

One PPO network trained on all four modes, 5.24M env steps per mode (21M in
all), 9.75 h on 2 CPU cores, against a specialist trained on each mode alone
with the same per-mode budget. Head to head, 1,024 games per cell:

| mode | generalist vs specialist, greedy | sampled |
|---|---|---|
| standard | 0.813 [0.788, 0.836] | 0.811 [0.786, 0.834] |
| duel | 0.624 [0.594, 0.653] | 0.678 [0.648, 0.706] |
| royale | 0.705 [0.676, 0.732] | 0.698 [0.669, 0.726] |
| constrictor | 0.406 (only 12 distinct games) | 0.494 [0.463, 0.524] |

The generalist gains nothing overall in constrictor, where greedy play repeats
a few deterministic games; `docs/research/multimode_constrictor.md` has the
investigation. Specialists retrained with the generalist's 16 minibatches
(standard, duel, royale) are no stronger head to head. In the duel the
generalist scores 0.388 against the heuristic.

### Multi-mode training campaign (GPU)

After the committed checkpoints, every multi-mode trainer was trained on one
GPU until a trend rule (`benchmarks/plateau.py`) found its strength flat, then
the learning rate was cooled down to 0. The measure is the share of the win
of one seat of the agent against three heuristics (one in the duel), averaged
over the four modes. Parity is 0.25 in the 4-snake modes and 0.5 in the duel.
Each check played 1,024 games per mode against the heuristic. Seed 0 only.
These networks are **not** in `baselines/checkpoints/`.

| run | env steps per mode at the end | best share vs heuristic | final |
|---|---|---|---|
| AlphaZero M | 16.8M (flat at 14.7M) | 0.439 at 15.7M | 0.430 |
| PPO M | 22.4M (flat at 19.9M) | 0.282 at 17.8M | 0.280 |
| PPO S | 17.5M (flat at 14.7M) | 0.223 at 14.7M | 0.223 |
| DQN M | 35.7M (step cap, older stop rule) | 0.188 at 26.2M | 0.153 |
| Rainbow DQN M | 10.7M (flat at 8.4M) | 0.105 at 4.2M | 0.085 |
| REINFORCE M | flat at 8.4M | 0.085 at 9.46M | 0.06–0.085 from 3M on |
| REINFORCE S | 9.9M (flat at 7.4M) | 0.071 at 3.2M | 0.058 |

S and M are network sizes (`slinky.unified.SIZES`; M is `ppo_modes.py`'s
network). AlphaZero M's final per-mode shares were duel 0.605, royale 0.566,
standard 0.386 and constrictor 0.163. REINFORCE's policies collapsed to low
entropy at both sizes and stopped improving after about 3M env steps.

**AlphaZero M with search** (final network, share against the heuristic,
4,096 games per point, 1,024 at 1,024 simulations; 0 simulations is the raw
network):

| simulations | duel | standard | royale | constrictor | mean |
|--:|--:|--:|--:|--:|--:|
| 0 | 0.625 ± 0.013 | 0.420 ± 0.015 | 0.578 ± 0.015 | 0.195 ± 0.012 | 0.455 |
| 4 | 0.662 ± 0.013 | 0.382 ± 0.015 | 0.584 ± 0.015 | 0.353 ± 0.014 | 0.495 |
| 16 | 0.783 ± 0.010 | 0.561 ± 0.015 | 0.734 ± 0.013 | 0.462 ± 0.015 | 0.635 |
| 64 | 0.849 ± 0.008 | 0.702 ± 0.014 | 0.817 ± 0.012 | 0.537 ± 0.015 | 0.726 |
| 256 | 0.873 ± 0.007 | 0.802 ± 0.012 | 0.863 ± 0.010 | 0.491 ± 0.015 | 0.757 |
| 1024 | 0.894 ± 0.013 | 0.858 ± 0.021 | 0.899 ± 0.018 | 0.454 ± 0.030 | 0.776 |

Search helps every mode, and 64 simulations (the training setting) gets most
of the gain. Constrictor peaks at 64 simulations and falls with deeper search.
In the duel, AlphaZero M at 256 simulations against the duel-only `az-256`
scored 0.527 ± 0.027 over 1,024 games (420 / 240 / 364), so the multi-mode
network is at least level with the duel specialist there.

### PufferLib against slinky's JAX trainer

On a 4-core CPU (`puffer/README.md`), the C environment steps a duel with
observations at 407k game-steps/s on one thread, 3.7× slinky on four cores.
PufferLib's trainer at `ppo.py`'s settings trained 1.5–1.9× faster than
`ppo.py`, but its network was weaker (0.198 against the heuristic, 0.396
against `ppo.py`'s network, one seed each).

On a GPU the order reverses. End-to-end duel PPO training ran at 20,404 game
steps/s with `ppo.py` against 4,319 with PufferLib at matched settings, and
8,553 at PufferLib's defaults. In equal wall time (about 340 s each), over
4,096 games, `ppo.py`'s network scored 0.29 against the heuristic and the
PufferLib network 0.11, and the PufferLib network scored 0.16 against
`ppo.py`'s head to head. Most of PufferLib's time went to the PyTorch update.

## Install, run and test

Python 3.10 or newer. From a checkout:

```bash
uv venv && uv pip install -e . --group dev   # jax, numpy, optax, pytest, ruff
# or: pip install -e ".[rl]"
```

For a GPU, install the JAX build for your accelerator as the JAX
documentation describes.

```bash
python baselines/ppo.py                          # train the duel PPO (about 70 min on 3 cores)
python baselines/az_modes.py --size M --run-dir runs/az-modes-M   # multi-mode AlphaZero
python baselines/dashboard.py                    # live dashboard on http://127.0.0.1:8050
python benchmarks/strength.py --help             # matchups between registry agents
python -m slinky.replay --a heuristic --b ppo --games 8 --html replays/h-vs-ppo.html
python -m slinky.server --agent az-256 --port 8000   # Battlesnake API on 127.0.0.1
python -m pytest tests/test_env.py               # one test file
ruff check
```

At release, every test file passed on CPU with jax 0.11.2: 888 passed and 171
skipped, run one file at a time. The skips need tools that were not installed
there: the Go rules-engine oracle (Go ≥ 1.22; 158 tests), Node.js for the
dashboard and replay viewer scripts (10 tests), and PufferLib and PyTorch for
the three `tests/test_puffer_*.py` files (they run only in `.venv-puffer`).
`ruff check` passes on the whole tree.

## Known limitations

- **One seed.** Almost every trained result is a single seed, so the spread
  between runs of one algorithm is unknown. Three duel-specialist seeds of
  multi-mode PPO span 0.25–0.38 against the heuristic.
- **Campaign networks are not committed.** Only the multi-mode PPO networks
  are in `baselines/checkpoints/`. The default checkpoints of the
  `reinforce_modes`, `dqn_modes`, `rainbow_modes` and `az_modes` agents
  (`baselines/checkpoints/<kind>-seed0`) do not exist, so those agents need an
  explicit run directory.
- **Royale search sees the future zone.** The royale map state holds every
  shrink side of the game. A search that steps from a slinky state (`az-*`,
  `az_modes` with simulations, in training and in the benchmarks) sees the
  true future shrinks. The ladder server reads the zone from the request and
  does not know future sides, so its royale play has less information than the
  royale numbers above.
- **Constrictor** is the weak mode for every multi-mode agent, and AlphaZero M
  gets worse there with more than 64 simulations.
- **REINFORCE** collapses to a near-deterministic policy and stops improving,
  in the duel and in every multi-mode run.
- **jax versions.** The suite passes on jax 0.11.2. On jax 0.8.2 four
  `tests/test_mcts.py::test_tree_is_updated_in_place` cases fail: XLA copies
  the search tree there, which the in-place update in unbatched search was
  written to avoid. `pyproject.toml` allows `jax>=0.4.30`; no version older
  than 0.8.2 was tried.
- **Windows:** `jaxlib` is pinned below 0.11, which crashed in the
  networks' convolutions there.
- **ROCm GPUs.** On ROCm 7.2, some float32 GEMMs with 2^19 rows or more return
  wrong values. `slinky.unified` runs the convolution backward pass in pieces
  off the CPU to avoid the case that corrupted REINFORCE's gradients, and
  `reinforce_modes.py` skips non-finite or very large gradients, but other
  code paths are not protected. The ROCm 10.1 builds tested did not have the
  bug. With ROCm, set `JAX_PLATFORMS=rocm,cpu`: evaluations use
  `jax.debug.callback`, which fails with `rocm` alone.
- **Rules.** The engine's random-number stream is not reproduced, only its
  distributions, so a game cannot be replayed turn for turn against the
  engine. Hazard maps other than royale and healing pools are not
  implemented, and the server has no config for wrapped, solo or other maps
  (a safe-move picker plays those).
- **Server compilation.** Each new board config compiles in the background;
  until it is ready the server answers with a safe-move picker.
- **Not done:** population or league training.
