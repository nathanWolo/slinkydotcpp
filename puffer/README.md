# Battlesnake in PufferLib (`puffer/`)

slinky's game as a [PufferLib](https://github.com/PufferAI/PufferLib) 3.0 environment:
the rules, maps, rewards and observations ported to C, checked against slinky and the
official engine, and trained with PufferLib's PPO trainer (PuffeRL, in PyTorch). A trained
network exports as a `baselines/ppo.py` checkpoint, so slinky's `ppo:<run dir>` agent plays
it against every other agent in the repository with no new code.

| file | what it is |
|---|---|
| `battlesnake.h` | the game in C11: rules step, maps, autoreset, rewards, observations, log |
| `binding.c`, `setup.py` | the PufferLib binding (`env_binding.h`) and its build |
| `battlesnake.py` | `Battlesnake(pufferlib.PufferEnv)`, the Python environment |
| `obs_format.py` | the observation layout, shared by the environment and the policy |
| `policy.py` | ppo.py's network in PyTorch, and the export to ppo.py's checkpoint format |
| `train.py` | training with PuffeRL, evaluation with slinky's tools, checkpoints |
| `bench.py` | environment (and training) throughput |

On a 4-core cloud CPU (no GPU):

- **The environment** steps a duel with observations at 407k game-steps/s on one thread,
  3.7× slinky's JAX environment on all four cores (4.3M and 7.7× without observations,
  under uniformly random moves).
- **Training** with `train.py`'s defaults for 57 minutes (19.7M game steps, 3.75× ppo.py's
  budget), the network beats ppo.py's PPO 0.620 and scores 0.434 against the heuristic
  (ppo.py's PPO: 0.296). Head to head it is level with AlphaZero's raw network (`az-0`,
  0.501), the strongest network-only policy here, which scores 0.484 against the
  heuristic. AlphaZero with search (`az-64`) is still far ahead.
- **With ppo.py's own settings and budget**, PuffeRL trains 1.5–1.9× faster than ppo.py on
  the same cores (30 minutes on 4 cores; ppo.py took 62 on 3), but the network is weaker:
  0.198 against the heuristic, and 0.396 against ppo.py's network head to head (one seed
  each).
- **On a CPU the network is the bottleneck**, not the environment: 80–88% of training time
  goes to gradient steps.

## Setup

PufferLib 3.0.0 pins `numpy<2`, and jaxlib needs NumPy 2, so it lives in its own virtual
environment, `.venv-puffer`, built against NumPy 2 (the C extensions are compiled against
NumPy's C API, so they must be rebuilt after the upgrade). PufferLib 3.0.0 is only on PyPI
as a source package.

```bash
uv venv -p python3.12 .venv-puffer
uv pip install --python .venv-puffer/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-puffer/bin/python -e . --group dev setuptools wheel
uv pip install --python .venv-puffer/bin/python --no-build-isolation pufferlib==3.0.0
uv pip install --python .venv-puffer/bin/python numpy==2.3.5   # back to NumPy 2 for jaxlib
uv pip install --python .venv-puffer/bin/python --no-build-isolation --no-deps --reinstall \
    --no-cache pufferlib==3.0.0                                  # rebuilt against NumPy 2
.venv-puffer/bin/python puffer/setup.py build_ext --inplace     # puffer/binding.*.so
```

`--no-cache` is what forces the rebuild: without it uv reinstalls the wheel it built
against NumPy 1, whose extensions fail to import under NumPy 2. Each PufferLib build takes
about a minute on 4 cores. Tested with Python 3.12, uv 0.11, torch 2.14.1 (CPU), jax
0.11.2 and numpy 2.3.5. Importing the environment prints a notice that gym is unmaintained
(gym 0.23 comes with PufferLib); it is harmless.

For a GPU, install a CUDA (or ROCm) build of PyTorch instead of the CPU one. `import
pufferlib` creates a `resources` symlink in the current directory (ignored by git).

Tests: `.venv-puffer/bin/python -m pytest tests/test_puffer_*.py` (they are skipped in the
main `.venv`, which has no PufferLib). `BATTLESNAKE_DEBUG=1 .venv-puffer/bin/python
puffer/setup.py build_ext --inplace --force` builds with AddressSanitizer and UBSan; run the
tests with `LD_PRELOAD="$(gcc -print-file-name=libasan.so)
$(gcc -print-file-name=libstdc++.so)" ASAN_OPTIONS=detect_leaks=0`.

## The environment

```python
from puffer.battlesnake import Battlesnake

env = Battlesnake(num_envs=256, mode="duel")  # 256 games, 512 agent slots
obs, _ = env.reset(seed=0)  # uint8 [512, 5737]
obs, rewards, terminals, truncations, infos = env.step(actions)  # int32 [512] in 0..3
```

- **Games and slots.** `num_envs` C games of `num_snakes` snakes; agent `g * num_snakes +
  i` is snake `i` of game `g`. `mode` is one of the four leaderboard modes (`standard`,
  `duel`, `royale`, `constrictor`, as `slinky.modes`), or pass `ruleset`, `map` and
  `num_snakes`: every ruleset and map slinky has, boards of 1 to 32 cells a side, up to 16
  snakes. The game settings and `reward` (`win_loss` or `placement`) are slinky's.
- **Observations.** One flat `uint8` vector per agent: slinky's egocentric board
  (`[21, 21, 13]` on 11×11, in slinky's channel order) holding raw integers where slinky
  divides (body countdowns, health, length), then the 4-byte legal-move mask
  (`slinky.rules.action_mask`). `board * obs_format.channel_scale(w, h)` is slinky's float32
  observation, up to float32 rounding, while no length or body countdown exceeds 255 (the
  bytes saturate). At 5,737 bytes it is a quarter of the size of slinky's float32 one.
- **Per step**, `rewards[i]` and `terminals[i]` describe the transition just made, as
  PuffeRL expects. `terminals[i]` is 1 when snake `i` died, when its game ended, or when it
  was already dead. A finished game is replaced at once (autoreset).
- **Dead snakes.** A dead snake's slot waits for its game to end with reward 0, a zero
  board and an all-ones mask, terminal every step. PuffeRL 3.0 cannot drop rows, so these
  slots are stored and trained on (their value target is 0), and they count as agent
  steps. In 4-snake games they are a fifth to a third of the rows under random legal play
  (about a tenth to a sixth with 3 snakes, over half with 8); a duel ends with the first
  death, so it has none.
- **Truncation is terminal.** PuffeRL 3.0 ignores the truncation flag (its advantage
  kernel sees only terminals), so a game cut off by `max_turns` is reported as terminal.
- **Logging.** Every `report_interval` steps, `infos` holds averages over the games that
  finished since the last report (PufferLib's `vec_log`): game length, the share of draws
  and of games cut off by `max_turns`, and deaths per game by cause (wall, own body, other
  body, head-to-head, starvation, hazard), plus `n`, the number of games. It is an empty
  dict when no game finished.
- `env.render()` prints slinky's ASCII board of game 0; `puffer.battlesnake.env_creator(name)`
  is the PufferLib-style factory (`battlesnake`, or `battlesnake_<mode>` to fix the mode;
  a `puffer_` prefix is accepted).

**Rules fidelity.** The C rules step is a port of `slinky.rules.rules_step`,
and `tests/test_puffer_env.py` checks it the way slinky is checked against the engine:

- transitions: slinky rollouts, fuzzed engine states and scrambled states go through the C
  rules step and full step, and every field must equal slinky's, except spawned food,
  which must land where slinky's spawn mask allows;
- the C rules step must reproduce transitions generated by the official engine (these
  need the Go oracle in `tools/oracle/`, and are skipped without it);
- observations and masks must equal slinky's;
- new games, spawned food and royale's shrinking must follow slinky's probabilities;
- and PufferLib's slot semantics (rewards, terminals, autoreset, the log).

The rules tests also pass under AddressSanitizer and UBSan. Random numbers come from a
per-game PCG32 stream, so games are reproducible from the seed but are not slinky's games
for the same seed (slinky uses JAX's PRNG).

## Training

```bash
.venv-puffer/bin/python puffer/train.py                       # duel, CPU, defaults
.venv-puffer/bin/python puffer/train.py --mode standard --device cuda --num-envs 1024
.venv-puffer/bin/python puffer/train.py --resume runs/puffer-duel-<timestamp>
```

`puffer/train.py` drives PuffeRL with self-play and one shared network (every snake is
played by the same policy, as in ppo.py). The policy (`puffer/policy.py`) is ppo.py's
actor-critic: three 3×3 "SAME" convolutions (16, 32 and 32 channels, strides 2, 2 and 1),
a 256-unit dense layer, and policy and value heads. Illegal moves get the logit −1e9 inside
the network, so PuffeRL samples legal moves only and sees the masked policy's
log-probabilities and entropy, with no change to the trainer.

The defaults are close to ppo.py's but **not tuned**: Adam at 1e-3 with PuffeRL's cosine
decay, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.01, 128 games, 128-step segments, and
`update_epochs=2` with minibatches of 4,096. PuffeRL's update differs from ppo.py's: it draws
`update_epochs * batch / minibatch` minibatches of whole 128-step segments, and recomputes
the advantages before each one from the current values (V-trace style; at importance
ratio 1 this is GAE, except that the last step of each segment gets no advantage). The
module docstring of `train.py` explains each choice.

A run directory holds `params.npz` and `config.json` (the network in ppo.py's format),
`metrics.jsonl` (training statistics and evaluations), `checkpoint.pt` (for `--resume`) and
`snapshots/step-<N>/` (the network at each evaluation). Evaluation, at the end of the run
and every `--eval-every` steps, plays the snapshot against `random_legal` and the
heuristic with slinky's own tools. In a duel that is `play_match`, so the scores are on the
same scale as every other baseline's; with more snakes it is `play_table` (win share and
placement). Any network trained on the standard 11×11 board is then a slinky agent:

```bash
.venv/bin/python benchmarks/strength.py --a ppo:runs/puffer-duel-<timestamp> --b heuristic,ppo
```

## Throughput

`puffer/bench.py` steps the C games on one thread under a random policy; slinky's
`benchmarks/throughput.py` steps slinky's games as one jitted, vmapped program on all
cores. Both on the same 4-core cloud CPU (Xeon @ 2.1 GHz, no GPU). A *game step* is one
game advancing one turn; agent steps are game steps times the snakes.

| workload (duel) | slinky, 4 cores (best batch) | C, 1 thread (1,024 games) | C, 4 processes |
|---|--:|--:|--:|
| observations, `random_legal` | 111k game-steps/s | 407k (3.7×) | 1.56M (14×) |
| no observations, uniformly random moves | 559k | 4.3M (7.7×) | 16.3M (29×) |

| C, 1 thread, 1,024 games: agent-steps/s | duel | standard (4) | royale (4) | constrictor (4) |
|---|--:|--:|--:|--:|
| observations, `random_legal` | 815k | 1.06M | 931k | 689k |
| observations, uniformly random | 972k | 984k | 971k | 853k |
| no observations, uniformly random | 8.6M | 9.8M | 6.2M | 3.3M |

- **Workloads.** `random_legal` picks uniformly among the moves the mask allows, so a duel
  lasts about 70 turns, as in slinky's benchmark; uniformly random moves end a duel after
  2 or 3 turns (2.7 on average) and a 4-snake game after about 4, so new games are much of
  the work there. In 4-snake games under
  `random_legal`, slinky's egocentric observations run at 49k game-steps/s and the C ones
  at 265k (5.4×).
- **Why the gap.** The C observation is `uint8`, a quarter of the bytes of slinky's
  float32 one, and is written cell by cell from small per-cell summaries, while XLA
  builds and writes the full float tensor; on a CPU both are bound by memory. Without
  observations the rules step is cheap in both, and C has no batching overhead. slinky's
  figures are for a CPU. It is built for accelerators, where large batches should
  help it far more (not measured here); the C environment runs on the CPU only.
- **Four processes** are four copies of `puffer/battlesnake.py`'s benchmark, each pinned
  to a core, summed: what PuffeRL's `Multiprocessing` backend could reach without its
  communication overhead.
- **In training it barely matters** on a CPU. In the ppo.py-matched run below, 88% of the
  time went to gradient steps and the rest to collecting rollouts, the policy's forward
  pass included: a free environment would save at most 12% of the time.

Commands: `.venv-puffer/bin/python puffer/bench.py --num-envs 64,256,1024` (every mode,
both policies, with and without observations) and `.venv/bin/python
benchmarks/throughput.py --configs duel,standard-4p --variants sim,sim+ego-obs --policy
random_legal` (and `--policy random`). Their outputs, and the four-process runs, are in
`benchmarks/results/puffer_throughput/`.

## Results

Two duel runs, seed 0, then 1,024 fresh games against each opponent in the strength
benchmark (`benchmarks/strength.py`, results in `benchmarks/results/puffer.jsonl`). Both
networks are kept in `baselines/checkpoints/` and play as
`ppo:baselines/checkpoints/puffer-duel-seed0` and `...-matched-seed0`.

- **Run A, ppo.py's settings**: 64 games × 128 turns per update (16,384 agent samples),
  32 gradient steps of 2,048 per update (`--update-epochs 4 --minibatch-size 2048`), 5.24M
  game steps, everything else as the defaults.
- **Run B, `train.py`'s defaults**: 128 games (32,768 samples per update), 16 gradient steps
  of 4,096, 19.7M game steps (`--total-timesteps 39321600`). It was cut by a container
  restart at 29.8M agent steps and resumed from its checkpoint at 29.5M.

| network | game steps | training (evaluation excluded) | vs `random_legal` | vs heuristic | vs DQN | vs ppo.py's PPO | vs `az-0` |
|---|--:|---|--:|--:|--:|--:|--:|
| ppo.py's PPO (`ppo`) | 5.24M | 62 min, 3 cores | 0.998 | 0.296 | 0.868 | | 0.419 |
| run A (`puffer-duel-matched-seed0`) | 5.24M | 30 min, 4 cores | 0.999 | 0.198 | 0.797 | 0.396 | 0.471 |
| run B (`puffer-duel-seed0`) | 19.7M | 57 min, 4 cores | 0.996 | 0.434 | 0.857 | 0.620 | 0.501 |

The 95% intervals are at most ±0.031 (about ±0.02 against the DQN, under ±0.005 against
`random_legal`; ±0.04 for `az-0` against ppo.py's PPO, 512 games).
Run B wins 443, draws 3 and loses 578 of its games against the heuristic; run A 182 / 42 /
800. For scale, against the heuristic `az-0` scores 0.484 and the multi-mode PPO 0.388 in
the duel (`baselines/README.md`).

**During training** (256 games against the heuristic, greedy / sampled; ± at most 0.06):

| game steps | ppo.py | run A | run B |
|--:|---|---|---|
| 1.05M | 0.119 / 0.064 | 0.104 / 0.063 | |
| 2.10M | 0.121 / 0.098 | 0.104 / 0.064 | |
| 3.15M | 0.225 / 0.186 | 0.127 / 0.168 | |
| 3.93M | | | 0.113 / 0.066 |
| 4.19M | 0.232 / 0.242 | 0.129 / 0.143 | |
| 5.24M (end of A and ppo.py) | 0.338 / 0.268 | 0.232 / 0.199 | |
| 7.86M | | | 0.240 / 0.146 |
| 11.8M | | | 0.367 / 0.305 |
| 18.7M | | | 0.396 / 0.408 |
| 19.7M (end of B) | | | 0.398 / 0.475 |

Run B's evaluation due at 15.7M game steps was skipped: `--resume` restarted the
evaluation interval at the resume step (fixed since; schedules now count from step 0).

**Speed on the same machine** (game steps per second of training, evaluation excluded):

| trainer | settings | 3 cores | 4 cores |
|---|---|--:|--:|
| ppo.py (JAX) | its defaults | 1,409 (its run) | 1,518 (4 min) |
| PuffeRL | ppo.py's | 2,163 (150 s) | 2,875 (run A) |
| PuffeRL | `train.py`'s defaults | | 5,187 (150 s); 5,716 over run B |

The 3-core figures are pinned with `taskset`. Speeds vary by up to about 20% between
sessions here: run B's two sessions ran at 6,026 and 4,955 game steps per second. So the
ratios to ppo.py are 1.5–1.9× at ppo.py's settings and 3.4–3.8× at the defaults.
PuffeRL's update took 88% of run A's time (1,609 of 1,824 s) and about 80% at the
defaults, which do half the update epochs (2 instead of 4: half the gradient work per
sample). Both trainers spend most of their time in the same network, while slinky's
environment alone accounts for about 2% of ppo.py's time (5.24M game steps at 61k a
second for 64 games). So the gap most likely comes from PyTorch's CPU kernels against
XLA's for this network (and, at the defaults, from half the update epochs) rather than
from the faster environment.

**Why run A is weaker than ppo.py** at equal settings is not established: one seed each,
and no ablation. For scale, three seeds of the multi-mode duel specialist (a closely
related PPO at the same budget) span 0.25–0.38 against the heuristic, so the heuristic gap
alone is within seed noise. The head-to-head 0.396 is the stronger evidence: all three of
those seeds score 0.45–0.49 against ppo.py's network. PuffeRL's update differs from
ppo.py's in ways that could matter:

- it samples minibatches of segments independently, so a sample is trained on 4 times on
  average rather than exactly 4;
- before each minibatch it recomputes the advantages and value targets from the values
  and importance ratios left by earlier minibatches (V-trace, both clips at 1), so the
  targets drift within an update; ppo.py fixes them per update;
- the last step of each 128-step segment gets no advantage;
- its learning rate decays on a cosine instead of linearly.

(Games cut at 1,000 turns are terminal rather than bootstrapped, but no logged game in
either run reached the cap.)

Run B shows what the extra speed buys. In 57 minutes of training, about an hour with
evaluation, it trained on 3.75× the samples and ended clearly stronger than ppo.py's
network. Its settings differ from ppo.py's too, and ppo.py was not tried with them, so
speed and settings are confounded. Its greedy score against the heuristic levelled off
from about 11.8M game steps (0.367, 0.396, 0.398) as the learning rate decayed to 0, so
whether a longer run would help is untested.

## Assessment

**What PufferLib is.** A reinforcement-learning library built for speed. It has three
parts:

- environments written in C against a small header API ("Ocean": about 30 in 3.0 and about 80 in 5.0;
  its `snake` is a simple multi-snake game, not Battlesnake);
- vectorization (`Serial`, `Multiprocessing` over shared memory, and `Ray`);
- PuffeRL, a PPO trainer with extras. In 3.0 these are V-trace-style advantages recomputed
  per minibatch, prioritized segment sampling, the Muon optimizer by default, and
  hyperparameter sweeps. 5.0 keeps Muon and the sweeps, but takes minibatches in order and
  has V-trace off by default.

There are two lines. **3.0**, on PyPI, has the trainer in Python and PyTorch; this
directory targets it. **5.0**, on GitHub's main branch, rewrites the trainer in C and CUDA
as one binary with the environment compiled in. Its environment API is different (with
action masks, self-play pools and bot ladders), and it needs an NVIDIA GPU to train.

**Porting the game was the easy part.** The environment API is a struct of buffer pointers
and four functions (`c_step`, `c_reset`, `c_render`, `c_close`), plus a struct of
statistics that PufferLib averages. The game is about 1,000 lines of C and the binding
400. Most of the work went into showing that it is slinky's game: about 1,600 lines of
parity tests, a sanitizer build, and an adversarial review whose findings were each
checked, then fixed or documented.

**What it bought here:**

- an environment 3.7–29× faster than slinky's on this CPU (one thread or four processes,
  with or without observations);
- PPO training 1.5–1.9× faster at equal settings, and 3.4–3.8× at its defaults, about half
  of which comes from doing half the update epochs;
- a network level with AlphaZero's raw network, the best network-only policy here, in 57
  minutes of training on 4 CPU cores;
- interop: the networks play through slinky's own agents and benchmarks, with no new
  code.

**What it costs:**

- **Two copies of the rules.** A change to slinky's rules has to be ported to
  `battlesnake.h`. The parity tests catch drift, but only when someone runs them, and they
  need `.venv-puffer`.
- **A second environment to install.** PufferLib 3.0 pins `numpy<2`, which conflicts with
  jaxlib; hence the rebuild above. It also brings gym, pettingzoo, wandb and more.
- **Trainer gaps for this game.** PuffeRL 3.0 has no action masks (masked in the network
  instead), no truncation (cut games are terminal), and cannot drop dead snakes' rows (a
  fifth to a third of the rows in 4-snake games under random legal play).
- **Not plain PPO.** Its update differs from ppo.py's in ways that may cost sample
  efficiency here (run A, one seed), and its defaults were tuned on other games: Muon at learning
  rate 0.015, γ 0.995, prioritized sampling. We use Adam and ppo.py's values instead,
  untuned.
- **Churn.** 5.0 changed the environment API and the trainer, so this binding is for 3.0
  only. Porting to 5.0 would keep the game code but rewrite the binding, and training
  would then need an NVIDIA GPU.

**GPUs.** None was available here, so nothing on a GPU was measured.

- **NVIDIA:** `--device cuda` with a CUDA build of PyTorch.
- **AMD, PufferLib 3.0:** should work through a ROCm build of PyTorch, which exposes the
  GPU as `cuda`. PufferLib builds its CUDA advantage kernel only when `nvcc` is found.
  Without `nvcc` on the path, `pufferl.py` copies the buffers to the CPU before every
  minibatch and runs its C++ CPU kernel, which is slower but correct. Untested.
- **AMD, PufferLib 5.0:** no. Its trainer is CUDA C (cuBLAS, cuRAND, NCCL, NVML), with no
  HIP or ROCm build.
- **This environment** is C on the CPU in every case.

**Verdict.** It is worth keeping as the fast way to train PPO for this game on a CPU. It
matched the repository's best network-only result head to head (AlphaZero's raw network)
in 57 minutes of training. That is less a win of the C environment than of PyTorch's CPU kernels and of
cheaper settings (half the update epochs, which ppo.py was not tried with): on a CPU the
network costs more than either environment. The price is a second copy of the rules,
which the parity tests keep honest.

On a GPU the comparison would change in ways not measured here. slinky can keep the
environment, the network and search on the accelerator. This binding (PufferLib 3.0) keeps
the environment on the CPU and the network on the GPU; PufferLib 5.0 can also run
environments written in CUDA on the GPU.
