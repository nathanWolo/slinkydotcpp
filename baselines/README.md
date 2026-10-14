# Baselines

Single-file reinforcement-learning baselines for the 1v1 duel (11×11,
standard rules), trained by self-play on slinky, and one network for all four
leaderboard modes ([multi-mode PPO](#multi-mode-ppo-ppo_modespy)).

| baseline | file | training | vs `random_legal` | vs heuristic | checkpoint |
|---|---|---|---|---|---|
| Double DQN | `dqn.py` | 31 min, 1.28M env steps | 0.998 | 0.053 | `checkpoints/dqn-duel-seed0/` |
| PPO | `ppo.py` | 67 min, 5.24M env steps | 0.998 | 0.296 | `checkpoints/ppo-duel-seed0/` |
| Rainbow DQN | `rainbow.py` | 81 min, 1.28M env steps | 0.983 | 0.084 | `checkpoints/rainbow-duel-seed0/` |
| AlphaZero (`az-64`) | `az.py` | 9.7 h, 8.39M env steps | 0.999 | 0.853 | `checkpoints/az-duel-seed0/` |
| Multi-mode PPO (generalist, in the duel) | `ppo_modes.py` | 9.75 h, 21M env steps (5.24M per mode) | 1.000 | 0.388 | `checkpoints/ppo-modes-seed0/` |

- **Hardware:** the DQN and PPO trained on a 4-core cloud CPU (PPO pinned to
  3 cores), Rainbow on a thermally throttled laptop CPU, AlphaZero on the
  4-core cloud CPU, multi-mode PPO on 2 pinned cores of it; no GPU.
- **AlphaZero** is scored as `az-64`: the network with 64 simulations of
  search per move (7 ms per move with 32 games in flight on one core). The
  raw network alone (`az-0`) scores
  0.997 and 0.484. See [AlphaZero](#alphazero-azpy).
- **Head-to-head:** PPO beats the DQN 0.868 over 1,024 games (888 / 2 / 134).
  See [the three compared](#the-three-compared) for all the pairings.
- **Sources:** the DQN and PPO scores are from the strength benchmark
  ([`benchmarks/README.md`](../benchmarks/README.md); 1,024 games each), as
  are AlphaZero's (512 games each).
  Rainbow is not in that benchmark; its scores are from its own evaluation
  (5,000 games against `random_legal`, 1,000 against the heuristic). In its
  own 5,000-game evaluation below, the DQN scores 0.993 against
  `random_legal`.
- **Multi-mode PPO** is one network trained on standard, duel, royale and
  constrictor games at once, told the mode. Its duel scores are from the mode
  benchmark (`benchmarks/modes.py`; 256 games against `random_legal`, 1,024
  against the heuristic). It beats its single-mode specialists head to head
  in standard, the duel and royale, and is level with them in constrictor.
  See [multi-mode PPO](#multi-mode-ppo-ppo_modespy).

### The three compared

The three committed checkpoints in one harness (`slinky.evaluate.play_match`:
seats alternate, greedy play, the same seed 2026 for every match; score is
win + ½ draw with its 95% interval):

| match | games | W / D / L | score |
|---|--:|---|---|
| PPO vs DQN | 1,000 | 872 / 1 / 127 | 0.873 ± 0.021 |
| PPO vs Rainbow | 1,000 | 751 / 5 / 244 | 0.753 ± 0.027 |
| Rainbow vs DQN | 1,000 | 499 / 10 / 491 | 0.504 ± 0.031 |

| agent | vs `random_legal` (5,000 games) | vs heuristic (1,000 games) |
|---|---|---|
| DQN | 0.992 ± 0.002 (4951 / 16 / 33) | 0.047 ± 0.013 (45 / 4 / 951) |
| PPO | **0.996 ± 0.002** (4967 / 22 / 11) | **0.286 ± 0.025** (196 / 181 / 623) |
| Rainbow | 0.983 ± 0.003 (4896 / 33 / 71) | 0.084 ± 0.017 (78 / 12 / 910) |

- **PPO is the strongest of the three** on every count. It beats both DQNs
  head to head and scores 3–6 times as much as either against the heuristic.
  These agree with the strength benchmark (PPO vs DQN 0.868, PPO vs heuristic
  0.296).
- **Rainbow and the DQN are level** head to head, but Rainbow holds up
  better against PPO: 0.247 against PPO's 0.753, where the DQN gets 0.127.
  Rainbow also scores more against the heuristic (0.084 against 0.047).
- **The budgets differ.** PPO trained on 4.1 times as many env steps (5.24M
  against 1.28M). Its env steps are cheaper, so the wall time was similar.
  At about the same env steps, PPO's training evaluations (256 games, below)
  were already ahead: 0.615 against the DQN and 0.119 against the heuristic
  at 1.05M steps, where Rainbow and the DQN at 1.28M score 0.504 against each
  other and 0.084 and 0.047 against the heuristic. An equal-budget comparison
  would need longer DQN and Rainbow runs.
- **One seed each**, so the run-to-run spread of each algorithm is unknown.

```bash
uv pip install -e ".[rl]"            # adds optax
python baselines/dqn.py              # default run, ~30 min on a 4-core CPU
python baselines/ppo.py              # default run, ~70 min on 3 cores
python baselines/dqn.py --help       # every hyperparameter is a flag (same for ppo.py)
python baselines/dqn.py --eval-only runs/dqn-<timestamp> --eval-games 5000
python baselines/dqn.py --resume runs/dqn-<timestamp>   # continue an interrupted run
python baselines/rainbow.py          # Rainbow DQN; same flags, plus its own
python baselines/rainbow.py --n-step 1 --per-alpha 0    # ablate components
python baselines/az.py               # AlphaZero, 4-4.5 h on 4 cores (see below)
python baselines/ppo_modes.py --num-minibatches 16   # one network for all four modes, ~10 h on 2 cores
python baselines/dashboard.py        # live training dashboard (see below)
```

Each run writes to `runs/` (gitignored):

- `config.json`;
- `metrics.jsonl`, one record per log chunk (16k env steps for the DQN, 32k
  for PPO and AlphaZero) plus the evaluations;
- `params.npz`, the latest network;
- `runner.npz`, the full training state used by `--resume` (AlphaZero also
  saves its replay buffer, `replay.npz`).

## Watching training live

```bash
python baselines/dashboard.py        # then open http://127.0.0.1:8050
```

`dashboard.py` serves a page that lists every run under `runs/` and
`baselines/checkpoints/` (DQN, PPO, Rainbow and AlphaZero alike) and re-reads its
`metrics.jsonl` every 5 seconds, so a run in progress updates as it trains.

- **Run cards** show live / finished / stopped, progress and an ETA, and the
  latest greedy score against `random_legal`. Click a card to add the run to
  the charts or take it off (up to 8, each keeping its colour). Live runs and
  the committed checkpoints are shown at first.
- **Charts** (over env steps, one shared axis):
  - greedy score against each evaluation opponent, with its 95% interval:
    `random_legal` for every run, plus the heuristic and the DQN for PPO;
    AlphaZero runs plot their network-plus-search agent against the
    heuristic, the DQN, Rainbow, PPO and MCTS-64;
  - training loss (log scale; DQN's Huber loss, Rainbow's KL and
    AlphaZero's cross-entropy plus value error are not comparable), and
    PPO's or AlphaZero's value loss;
  - the value estimate: mean Q of the moves taken, or PPO's or AlphaZero's
    mean value-head output;
  - PPO's and AlphaZero's policy entropy;
  - self-play game length and draw rate, and throughput;
  - ε or the noisy-net σ.
- **Death causes**: the share of self-play eliminations by cause, per run.
- Hover (or focus a chart and use ←/→) for exact values; every chart also has
  a table view.

It needs only the standard library. It binds to `127.0.0.1` and answers only
requests addressed to localhost; `--host 0.0.0.0` serves it on the network,
and `--port` or extra directory arguments change where it listens and looks.

## Opponent: `random_legal`

The reference opponent is `slinky.evaluate.random_legal`, the "naive snake":

- Each turn it picks uniformly among the moves that don't *certainly* die
  next turn. It never steps into a wall or a body when it has another choice,
  and it applies the tail rule exactly.
- It ignores food, head-to-heads and being trapped.

Every score below comes from `slinky.evaluate.play_match`:

- seats alternate across games;
- win = 1, draw = ½, loss = 0;
- ± is a 95% confidence interval.

## DQN (`dqn.py`)

Double DQN with one network playing both snakes (parameter-sharing
self-play).

- **Acting.** Each snake picks from its own egocentric observation, a
  `21×21×13` board centred on its head. It plays epsilon-greedily, but only
  among its legal moves.
- **Learning.** The replay buffer stores compact game states and recomputes
  observations when it samples. Each game transition gives one training
  example per snake. Dying, winning (+1) and drawing end the episode for
  value purposes; truncation at `max_turns` does not, so it bootstraps.
- **Network.** Convolutions 32 → 64 → 64 with strides 2, 2, 1, then dense 256,
  then 4 Q-values (650k parameters).
- **Optimiser.** Adam at 5e-4, Huber loss, gradient clipping at 10, γ = 0.99,
  and a hard target-network update every 250 updates.
- **Schedule.** 32 games run in parallel, with 1 update of 128 game
  transitions per step. ε decays from 1 to 0.05 over the first 30% of
  training.

### Results

Default config, seed 0, 1.28M env steps (2.56M agent-transitions, 39.7k
gradient updates). It took **31 minutes** on a 4-core cloud CPU with no GPU,
at about 720 env-steps/s and 22 updates/s.

**Greedy DQN vs `random_legal` during training** (1,000 games each):

| env steps | wins | draws | losses | score | mean game length |
|--:|--:|--:|--:|---|--:|
| 256k | 982 | 7 | 11 | 0.986 ± 0.007 | 27 |
| 512k | 992 | 3 | 5 | 0.994 ± 0.005 | 27 |
| 768k | 996 | 3 | 1 | 0.998 ± 0.003 | 35 |
| 1.02M | 984 | 7 | 9 | 0.988 ± 0.006 | 36 |
| 1.28M | 993 | 4 | 3 | 0.995 ± 0.004 | 41 |

**Final checkpoint, re-evaluated with fresh seeds:**

| match | games | W / D / L | score |
|---|--:|---|---|
| **DQN (1.28M steps) vs `random_legal`** | 5,000 | 99.2% / 0.2% / 0.6% | **0.993 ± 0.002** |
| untrained network of the same architecture vs `random_legal` | 1,000 | 41% / 18% / 41% | 0.498 ± 0.028 |
| `random_legal` vs `random_legal` | 2,000 | 40% / 18% / 42% | 0.490 ± 0.020 |

The DQN row is 4,962 wins, 9 draws and 29 losses.

The second row uses initialisation seed 100. Initialisation seed 101 scored
0.423 ± 0.029.

**How it wins.** In a separate 4,000-game check (score 0.994), 91% of
`random_legal`'s deaths were head-to-head collisions. The DQN snake ends
games longer than its opponent (5.7 vs 4.5 segments on average), and most
games are short (median 26 turns). These statistics suggest it grows first,
then hunts: it stays close to the shorter snake, whose random moves then
walk into losing head-to-heads. This reading comes from aggregate statistics;
I did not inspect replays.

**Self-play dynamics.**

- Early on, almost all self-play deaths are head-to-heads (85% at 128k
  steps).
- Once exploration bottoms out, games between the two copies more than double
  in length, from about 60 to 130 turns. Head-to-heads fall to about 55–60%
  of deaths, and self- and body-collisions rise.
- The mean Q-value of taken actions peaks at +0.47 and settles at about
  +0.27. In a symmetric zero-sum game it would be near 0 with no bias, so the
  network overestimates its values. That did not stop it from beating the
  baseline.

### Caveats and next steps

- **One seed.** This is a single training run, not a seed sweep.
- **The opponent is weak.** `random_legal` is a sanity-check opponent, and
  this benchmark is now saturated. Against stronger reference agents
  ([`benchmarks/README.md`](../benchmarks/README.md); 1,024 or 512 games
  each):
  - the hand-written heuristic beats this DQN 0.947 (967 / 6 / 51);
  - on the MCTS scale, the DQN beats MCTS with 4 simulations (MCTS scores
    0.427) and loses to MCTS with 16 (0.947);
  - PPO self-play on the same setup (below) beats it 0.868.

  Still to try: past checkpoints and other algorithms (Elo or TrueSkill over
  a population).

The final network from this run is kept in
[`checkpoints/dqn-duel-seed0/`](checkpoints/dqn-duel-seed0/) (2.6 MB) as a
fixed opponent for benchmarks:
`python baselines/dqn.py --eval-only baselines/checkpoints/dqn-duel-seed0`.

## PPO (`ppo.py`)

PPO-clip with one actor-critic network playing both snakes (parameter-sharing
self-play), in the style of cleanRL and PureJaxRL. The file is self-contained:
it shares no code with `dqn.py`.

- **Acting.** Each snake samples from a softmax over its legal moves only:
  illegal moves get a logit of -1e9, so their probability is exactly 0 while
  the entropy and the gradients stay finite. The observation is the DQN's
  egocentric `21×21×13` board.
- **Rollouts.** 64 games run in parallel for 128 turns per iteration, and a
  finished game is replaced at once. Every turn gives one sample per living
  snake, so an iteration has 16,384 samples.
- **Episodes.** A snake's episode ends when it dies (-1), wins (+1) or draws
  (0), with no bootstrapping. A game cut off at `max_turns` bootstraps from the
  value of the state reached. GAE never crosses the end of an episode.
- **Network.** Convolutions 16 → 32 → 32 with strides 2, 2, 1, then dense 256,
  then a 4-logit policy head and a value head (312k parameters). That is half
  the DQN's channels: twice the env steps per second, and as strong per minute
  of training in pilots.
- **Optimiser.** 4 epochs of 8 minibatches (2,048 samples) per iteration;
  Adam at 1e-3 decayed linearly to 0, gradient clipping at 0.5. γ = 0.99,
  λ = 0.95, clip 0.2, value coefficient 0.5 (clipped value loss), entropy
  coefficient 0.01, advantages normalised per minibatch.
- **Speed.** The convolutions use a hand-written backward pass. XLA's own
  convolution gradient inside `lax.scan` is about 70 times slower on CPU.
  Training runs at about 1,400 env-steps/s on 3 pinned cores, and the
  updates take about 90% of the time.
- **Play mode.** The `ppo` agent in `slinky.agents` plays greedily (the masked
  argmax); `ppo-sample` samples. Greedy was ahead or within noise in every
  evaluation, and it beats sampled play 0.549 head-to-head (below).

### Results

Default config, seed 0, 5.24M env steps (10.5M agent samples, 20,480 gradient
steps). It took **67 minutes** on 3 pinned cores of a 4-core cloud CPU, five
evaluation rounds included.

**During training** (256 games per cell, greedy / sampled score; ± is at most
0.06 against the DQN and 0.05 against the heuristic):

| env steps | vs `random_legal` | vs DQN | vs heuristic | W / D / L vs heuristic (greedy) |
|--:|---|---|---|---|
| 1.05M | 0.986 / 0.992 | 0.615 / 0.510 | 0.119 / 0.064 | 17 / 27 / 212 |
| 2.10M | 1.000 / 0.994 | 0.686 / 0.541 | 0.121 / 0.098 | 23 / 16 / 217 |
| 3.15M | 0.996 / 0.982 | 0.777 / 0.791 | 0.225 / 0.186 | 34 / 47 / 175 |
| 4.19M | 0.986 / 0.992 | 0.836 / 0.828 | 0.232 / 0.242 | 33 / 53 / 170 |
| 5.24M | 1.000 / 0.996 | 0.836 / 0.848 | 0.338 / 0.268 | 60 / 53 / 143 |

**Final checkpoint in the strength benchmark** (fresh games;
[`benchmarks/README.md`](../benchmarks/README.md)):

| match | games | W / D / L | score |
|---|--:|---|---|
| PPO vs `random_legal` | 1,024 | 1020 / 3 / 1 | 0.998 [0.992, 0.999] |
| PPO vs DQN | 1,024 | 888 / 2 / 134 | 0.868 [0.846, 0.888] |
| PPO vs heuristic | 1,024 | 205 / 197 / 622 | 0.296 [0.269, 0.325] |
| PPO greedy vs PPO sampled | 512 | 280 / 2 / 230 | 0.549 [0.506, 0.591] |

- **Against the heuristic:** 0.296 over 1,024 games, against 0.338 in the
  final 256-game training evaluation; the two agree within noise.
- **On the MCTS scale**, PPO plays between 4 and 16 simulations. MCTS-4
  scores 0.127 against it and MCTS-16 0.642, while MCTS at 500 ms per move
  (24,000 simulations) scores 0.949.
- **How it loses:** against MCTS-2048 and MCTS-4096, PPO wins 6.4–6.6% of
  its games (the heuristic 0.4–2.7%). But it draws far less and loses more,
  so it scores 0.09 against them while the heuristic scores 0.15–0.16.

**Self-play dynamics.**

- Early on, 82% of self-play deaths are head-to-heads and games last about 55
  turns.
- By the end, games between the two copies last about 195 turns,
  head-to-heads are down to about 20% of deaths, and running into its own
  body is the most common death (about half).
- The policy's entropy falls from 0.90 to 0.24 nats.
- The value function stays weak: explained variance is only 0.1–0.2. A review
  found no bug behind it, and turning off value clipping or raising the value
  coefficient did not help in a short A/B run. The policy still gets the
  outcome through the GAE returns.

**Pilots.** The defaults come from 1M-step pilots on one core each (about 30
minutes):

- **A learning rate of 2.5e-4 collapses into opening head-on draws.** That
  pilot scored a misleading 0.36 against the heuristic with zero wins, while
  scoring 0.10 against the DQN and 0.83 against `random_legal`. Read the
  W/D/L, not just the score.
- **Smaller rollouts hurt.** 32 games × 64 turns at lr 2e-3 scored 0.07
  against the heuristic.
- **Within noise of each other:** lr 1e-3 and 2e-3, γ 0.99 and 0.995, a
  constant learning rate, and the DQN-sized network at equal wall time.
- **Greedy beats sampled.** In a 1,024-game paired evaluation, greedy scored
  0.196 against 0.160 vs the heuristic, and 0.567 against 0.504 vs the DQN.

### Caveats

- **One seed**, like the DQN.
- **Still improving.** Against the heuristic the score rose from 0.23 to 0.34
  over the last million steps, while the learning rate decayed to 0. A longer
  run would probably be stronger.
- **Resuming is exact only with the same number of CPU cores**: the
  floating-point reductions depend on the thread count.

The final network from this run is kept in
[`checkpoints/ppo-duel-seed0/`](checkpoints/ppo-duel-seed0/) (1.3 MB) as the
default `ppo` agent:
`python baselines/ppo.py --eval-only baselines/checkpoints/ppo-duel-seed0`.

## Rainbow DQN (`rainbow.py`)

All six extensions of Rainbow (Hessel et al., 2018) on top of the same
self-play setup as `dqn.py`: one network for both snakes, egocentric
observations, legal moves only, and compact game states in the replay
buffer. Every component except the distributional head can be switched off
from the command line, for ablations.

- **Distributional (C51).** Each move's return is a categorical distribution
  over 51 atoms. A snake's return in this game always lies in [−1, 1] (−1 for
  dying, +1 for winning, nothing in between), so the support is exactly
  [−1, 1] and no target is ever clipped. The loss is the KL divergence to the
  projected target, which has the same gradient as the cross-entropy.
- **Double Q-learning** (`--double`): the online network picks the bootstrap
  move (on expected values, legal moves only); the target network supplies
  its distribution.
- **Dueling heads** (`--dueling`): value and mean-centred advantage streams,
  per atom.
- **3-step returns** (`--n-step`). The last `n` one-step transitions of every
  game slot sit in a sliding window. Each step, the oldest becomes an `n`-step
  transition. The window stops at the end of a game: the rules ending it
  makes the return final, while a `max_turns` cut-off still bootstraps.
  Each snake gets its own discount, 0 once it is dead.
- **Prioritized replay** (`--per-alpha`, `--per-beta-*`): proportional, on the
  KL loss, α = 0.5, with importance weights annealed from β = 0.4 to 1.
  Sampling is a stratified draw from a cumulative sum over the buffer (about
  1.7 ms per batch here, cheaper than maintaining a sum tree in JAX).
  `--per-alpha 0` is uniform replay.
- **Noisy nets** (`--noisy`): factorized Gaussian noise (σ₀ = 0.5) on the
  dense layers instead of ε-greedy. Each snake draws its own noise when
  acting. Learning draws one sample per batch per network evaluation;
  per-sample noise there would cost about 40 ms per update. Evaluation plays
  the mean weights.

Other settings, mostly as `dqn.py`: the same convolutional trunk and dense
256 layer (1.37M parameters, since noisy layers hold a σ per weight), Adam
at 2.5e-4 with ε = 1.5e-4, gradient clipping at 10, γ = 0.99, a hard
target-network copy every 250 updates, a 100k buffer, learning from 10k, and
32 games with one update of 128 game transitions per step.

### Results

Default config, seed 0, the same budget as the DQN: 1.28M env steps, 39.7k
updates. It took **81 minutes** on a 12-thread laptop CPU that was thermally
throttled to about 2.2 GHz, at about 275 env-steps/s. Early in training,
on the same machine, Rainbow ran at about 79% of the DQN's speed (358 vs 454
env-steps/s).

**Greedy Rainbow during training** (1,000 games each against `random_legal`;
512 games each against the heuristic, played afterwards from saved
checkpoints):

| env steps | vs `random_legal` | vs heuristic |
|--:|---|---|
| 256k | 0.954 ± 0.012 | 0.026 ± 0.010 |
| 512k | 0.967 ± 0.011 | 0.045 ± 0.017 |
| 768k | 0.982 ± 0.008 | 0.051 ± 0.019 |
| 1.02M | 0.981 ± 0.008 | 0.045 ± 0.018 |
| 1.28M | 0.989 ± 0.006 | 0.087 ± 0.024 |

**Final checkpoints compared** (same seeds for both agents):

| match | games | Rainbow | DQN |
|---|--:|---|---|
| vs `random_legal` | 5,000 | 0.983 ± 0.003 (4896 / 33 / 71) | **0.992 ± 0.002** (4951 / 16 / 33) |
| vs heuristic | 1,000 | **0.084 ± 0.017** (78 / 12 / 910) | 0.047 ± 0.013 (45 / 4 / 951) |
| Rainbow vs DQN | 1,000 | 0.504 ± 0.031 (499 / 10 / 491) | |
| vs PPO | 1,000 | **0.247 ± 0.027** (244 / 5 / 751) | 0.127 ± 0.021 (127 / 1 / 872) |

At the same budget, the two are level head to head. Rainbow is slightly
weaker against the random snake but scores about 1.8× as much against the
heuristic, and its score there was still rising at the end of training. Both
gaps are outside the 95% intervals.

**Self-play dynamics.**

- **Better-calibrated values.** The mean Q of the moves taken peaks at
  +0.10 and settles at about +0.04, against the DQN's +0.47 and +0.27. In a
  symmetric zero-sum game it should be near 0.
- **Longer games.** Self-play games last about 240 turns by the end, against
  the DQN's 136. The share of head-to-head deaths falls from 72% at 256k
  steps to about a third. Self-collisions rise to about 40%.
- **New transitions are replayed a lot.** New transitions get the largest
  priority seen so far, as in the prioritized-replay paper, and that maximum
  grows to about 12 while typical priorities are below 0.1. Fresh data is
  therefore oversampled for its first replays. Using the buffer's current
  maximum instead is a variant worth trying.

### Caveats and next steps

- **One seed**, as for the DQN, and the default hyperparameters were not
  tuned. A 320k-step pilot with the same settings scored 0.931–0.945
  against `random_legal`.
- **The heuristic score was still rising** at 1.28M steps. A longer run is
  the obvious next experiment, followed by ablations of the six components
  (the flags above).

The final network is kept in
[`checkpoints/rainbow-duel-seed0/`](checkpoints/rainbow-duel-seed0/) (5.5 MB)
as the `rainbow` agent of `slinky.agents`:
`python baselines/rainbow.py --eval-only baselines/checkpoints/rainbow-duel-seed0`.

## AlphaZero (`az.py`)

AlphaZero (Silver et al., 2018) on top of `slinky.mcts`: one network gives
move priors and position values, the simultaneous-move search plays with
them, and the network learns from the search's own self-play games. It is
all JAX in one process: self-play, search, replay and learning run as one
jitted chunk per log line, so the same file runs on a GPU unchanged (raise
`--num-envs`). The file is self-contained: it shares no code with `ppo.py`.

- **Network.** PPO's torso (the same `21×21×13` egocentric observation,
  convolutions 16 → 32 → 32, dense 256), a 4-logit policy head with illegal
  moves masked and a `tanh` value head: 312k parameters, evaluated for each
  snake from its own view.
- **Search.** Decoupled PUCT in `slinky.mcts.search`: each snake picks its
  own move at a node by `q + c_puct · P · sqrt(max(N, 1)) / (1 + n)`, where
  `q` is the move's mean value mapped to [0, 1] (so `c_puct` 0.5 is
  AlphaZero's 1.0 on its [-1, 1] scale), `n` the move's visits and `N` the
  snake's visits at the node. The priors `P` and the leaf values come from
  the network (exact values at finished positions). Unvisited moves start
  from the node's own value minus 0.2 on the [-1, 1] scale (first-play
  urgency). The network enters as a closure over traced
  parameters, so new weights never recompile, and under `vmap` each
  simulation is one batched network call for every game in flight.
- **Self-play.** 64 games, 64 simulations per move, Dirichlet noise in the
  root priors (25%, α = 0.5). The first 16 turns sample each snake's move
  from its root visit counts, later turns play the most visited move.
- **Targets.** Each living snake's root visit shares (policy) and its game's
  outcome (value): win +1, loss -1, a mutual elimination -0.5 (the search's
  draw value, so the network and the tree agree), a cut-off game 0.
- **Replay.** Positions are stored as compact game states (under 1 KB) with
  their visit shares; a game's outcome goes into a side table when it ends,
  and positions of games still running are skipped when sampled. Each
  sampled position gets a random one of the 8 board symmetries (the rules,
  observations and legal masks commute with all 8 exactly; tested).
- **Learning.** Cross-entropy to the visit shares plus the value MSE, AdamW
  at 1e-3 decayed linearly to 1e-4, 512 positions (1,024 samples) per
  gradient step, about 4 uses of each stored position.

```bash
python baselines/az.py               # default run: 4.2M env steps, 4-4.5 h on 4 cores
python baselines/az.py --total-env-steps 8388608 --buffer-capacity 262144 \
    --eval-opponents heuristic,dqn,rainbow,ppo,mcts-64 --run-dir runs/az-duel-seed0   # the run below
python baselines/az.py --eval-only baselines/checkpoints/az-duel-seed0 --eval-simulations 256
```

The trained network plays as `az-<n>` in `slinky.agents`: `n` simulations
of PUCT search per move, or `az-0` for the raw network (the legal argmax of
its policy). Search settings can be changed at play time, for example
`az-64:c_puct=0.25` or `az-64:leaf=heuristic`.

### Results

Seed 0, 8.39M env steps (about 39,000 self-play games, 65,296 gradient
steps). It took **9.7 hours** on a 4-core cloud CPU: 8.0 hours of training
and 1.7 hours of evaluations. Training ran at about 350 env steps/s for
the first 2M steps and 250–290 after, while benchmarks shared the machine.

**Final checkpoint in the strength benchmark** (512 games per cell, 95%
Wilson intervals; [`benchmarks/README.md`](../benchmarks/README.md#alphazero)):

| agent | vs `random_legal` | vs DQN | vs Rainbow | vs PPO | vs heuristic | ms per move |
|---|---|---|---|---|---|--:|
| `az-0` (raw network) | 0.997 [0.987, 0.999] | 0.844 [0.810, 0.873] | 0.714 [0.673, 0.751] | 0.581 [0.538, 0.623] | 0.484 [0.441, 0.528] | 0.26 |
| `az-64` | 0.999 [0.991, 1.000] | 0.994 [0.983, 0.998] | 0.984 [0.969, 0.992] | 0.943 [0.920, 0.960] | 0.853 [0.819, 0.881] | 7.0 |
| `az-256` | 0.999 [0.991, 1.000] | 0.995 [0.984, 0.998] | 1.000 [0.993, 1.000] | 0.987 [0.973, 0.994] | 0.872 [0.840, 0.898] | 26.3 |

- **With search it beats every learned baseline almost every game.**
  `az-256` won all 512 games against Rainbow and lost one in 512 to PPO
  (500 / 11 / 1).
- **The raw network, with no search, is the strongest learned policy here.**
  It beats PPO 0.581 and scores 0.484 against the heuristic, where PPO scores
  0.296; that is between MCTS-16 and MCTS-32 (0.321 and 0.528).
- **Against the heuristic** `az-64` scores 0.853 (395 / 83 / 34), level with
  MCTS at 2048–4096 simulations (0.846, 0.840) and at 500 ms per move
  (24,000 simulations, 0.875 over 128 games). Plain MCTS's score against the
  heuristic has saturated there; head to head, MCTS at 500 ms still beats
  `az-64` (below). `az-256` scores 0.872 and loses 7 games of 512.

**During training** (the run's own evaluations: 128 games per cell, ± up to
0.09; raw network / with 64 simulations):

| env steps | vs heuristic | vs DQN | vs Rainbow | vs PPO | vs MCTS-64 |
|--:|---|---|---|---|---|
| 1.05M | 0.391 / 0.805 | 0.641 / 0.930 | 0.516 / 0.859 | 0.336 / 0.781 | 0.301 / 0.793 |
| 2.10M | 0.430 / 0.793 | 0.734 / 0.984 | 0.695 / 0.938 | 0.457 / 0.863 | 0.332 / 0.730 |
| 3.15M | 0.434 / 0.832 | 0.766 / 1.000 | 0.695 / 0.992 | 0.492 / 0.918 | 0.324 / 0.680 |
| 4.19M | 0.418 / 0.832 | 0.797 / 0.992 | 0.715 / 1.000 | 0.465 / 0.938 | 0.348 / 0.863 |
| 5.24M | 0.477 / 0.824 | 0.734 / 0.984 | 0.676 / 0.969 | 0.520 / 0.895 | 0.477 / 0.816 |
| 6.29M | 0.480 / 0.855 | 0.840 / 1.000 | 0.676 / 0.984 | 0.457 / 0.953 | 0.344 / 0.805 |
| 7.34M | 0.504 / 0.816 | 0.781 / 0.977 | 0.750 / 0.953 | 0.535 / 0.930 | 0.395 / 0.805 |
| 8.39M | 0.488 / 0.844 | 0.820 / 0.992 | 0.703 / 0.984 | 0.523 / 0.969 | 0.402 / 0.852 |

- After 1M steps (about 50 minutes) the network with search already beats
  PPO, which trained for 5.24M steps. The raw network passes the DQN and
  Rainbow by 2M steps and is about level with PPO from 3M.
- The search's scores against the heuristic and MCTS-64 level off early
  (0.68–0.86 from 1M steps on, within the ±0.09 of these evaluations), while
  the plain-MCTS ladder below shows gains until about 5M steps.

**What the network is worth in plain simulations.** `az-64` against plain
MCTS with more and more simulations, at each checkpoint (256 games per point
for the snapshots, 512 for the final network except 128 at 24,000
simulations): where a line crosses 0.5 is
how many plain simulations the network's 64 are worth.

![az-64 against plain MCTS by simulations, one line per checkpoint](../benchmarks/results/az_ladder.svg)

| checkpoint | vs MCTS-512 | vs MCTS-1024 | vs MCTS-2048 | vs MCTS-4096 | worth about |
|--:|---|---|---|---|---|
| 1.05M | 0.520 | 0.445 | 0.340 | 0.344 | 600 simulations |
| 2.10M | 0.557 | 0.523 | | | a little over 1,000 |
| 3.15M | 0.578 | 0.516 | | | a little over 1,000 |
| 4.19M | | 0.555 | 0.463 | | 1,500 |
| 5.24M | | 0.590 | 0.531 | 0.459 | 2,800 |
| 6.29M | | | 0.543 | 0.459 | 2,900 |
| 7.34M | | | 0.555 | 0.484 | 3,500 |
| 8.39M (final) | 0.640 | 0.588 | 0.489 | 0.501 | 2,000–4,000 |
| 8.39M, `c_puct` 0.25 at play time | | 0.664 | 0.612 | 0.592 | over 4,096 (0.574 against MCTS-24000) |

- **The crossings are rough.** A 256-game point is ±0.06, about one doubling
  of plain simulations on these slopes. Read the column as: the network's
  worth grew about fivefold over the first 5M steps and then flattened.
  The final network sits at 0.49–0.50 against both MCTS-2048 and
  MCTS-4096 (±0.043 each), so its crossing is no finer than 2,000–4,000.
- **At equal time.** `az-64` takes 7.0 ms per move on one core with 32 games
  in flight, what plain MCTS spends on about 1,190 simulations. At 1M steps
  the network was worth half its time in plain search; it drew level by
  2–3M steps, and from 5M on it is worth 2–3 times its time.
- **Against MCTS at 500 ms per move** (24,000 simulations, 27 times `az-64`'s
  time) `az-64` scores 0.387 [0.307, 0.473] over 128 games (36 / 27 / 65).
  `az-256`, at a seventh of that MCTS's time, scores 0.586 [0.499, 0.668]
  (57 / 36 / 35): level or slightly ahead.
- **On a GPU** the balance would likely shift further towards the network
  (not measured): almost all of `az-64`'s time is network evaluation (about
  0.1 ms per extra simulation, against 5–8 µs per extra plain simulation),
  which a GPU batches cheaply.

**Play-time settings** (final network; 512 games per cell, ± up to 0.045,
except 128 games, ± up to 0.085, against MCTS-24000):

| setting | vs MCTS-1024 | vs MCTS-2048 | vs MCTS-4096 | vs MCTS-24000 | vs heuristic | vs PPO |
|---|---|---|---|---|---|---|
| default (`c_puct` 0.5, network leaf values) | 0.588 | 0.489 | 0.501 | 0.387 | 0.853 | 0.943 |
| `c_puct=0.25` | 0.664 | 0.612 | 0.592 | 0.574 | 0.886 | 0.964 |
| `leaf=heuristic` (network priors, heuristic leaf values) | 0.574 | | | | | |

- **A smaller `c_puct` plays better** than the 0.5 the network trained with:
  higher in all six matchups, by 0.02 to 0.19. Against MCTS at 500 ms per
  move it scores 0.574 [0.488, 0.656] where the default scores 0.387
  [0.307, 0.473]: `az-64` with `c_puct` 0.25 is level with a search that
  takes 25 times as long. Whether self-play with 0.25 also trains a better
  network is untested.
- **The value head has caught up with the hand-written evaluation.** At 1M
  steps the heuristic leaf was worth +0.09 against MCTS-64 (pilots below);
  on the final network it is level within noise.

**Self-play dynamics.**

- Games between the two copies grew from about 180 turns (first million
  steps) to 240. Self-collisions are about half the deaths and head-to-heads
  a third (43% early on); draws stay at 1.4–3.2%: no collapse into opening
  head-on draws.
- The value head explains 11–17% of the outcome variance (averaged per
  million steps; 12.7% over the last million). That is measured on the
  network's own games, which grow longer and closer as it improves, so it
  understates progress. Measured against the heuristic leaf instead, the
  value head went from 0.09 behind it (the 1M-step pilot network, against
  MCTS-64) to level with it (the final network, against MCTS-1024).
- The policy loss sits about 0.04 nats above the entropy of its targets
  (0.73 against 0.69): the network nearly matches its search's visit
  shares, and the search improves on the network by the margins above.

**Pilots.** The configuration comes from 1M-step pilots on 2 cores each
(about an hour each), scored over 256 games with 64 simulations:

| pilot | vs heuristic | vs MCTS-64 | vs PPO |
|---|---|---|---|
| **from scratch** | **0.803** | **0.695** | **0.908** |
| expert iteration (0.5M steps learning from heuristic-leaf MCTS-256), then AlphaZero | 0.732 | 0.693 | 0.848 |
| the same, phasing the heuristic leaf out over the AlphaZero half | 0.715 | 0.660 | 0.812 |
| expert iteration only (0.5M steps) | 0.596 | 0.549 | 0.588 |

- **Learning from scratch was best.** Starting from a heuristic-MCTS teacher
  gave a fast start (0.305 raw against the heuristic after 24 minutes) but
  plateaued, and its value head carried over poorly.
- **`c_puct`** 0.25–0.5 at play time scored alike in the pilot (0.719 and
  0.695 against MCTS-64); 1.0 was clearly worse (0.609).
- **The value head was the bottleneck at 1M steps:** the heuristic leaf
  scored 0.783 against MCTS-64 where the network's own values scored 0.695.

### Caveats

- **One seed** and one network size (PPO's); a bigger network, more
  simulations in self-play or a longer run are untested.
- **CPU only.** Self-play is about 95% network evaluation on this CPU; the
  time comparisons above are for this machine.
- **The training evaluations are noisy** (128 games); the strength benchmark
  above has 512 games per cell.
- **The training snapshots are not committed**, only the final network; the
  ladder's snapshot rows are in `benchmarks/results/az_ladder.jsonl` with
  their parameter hashes.

The final network is kept in
[`checkpoints/az-duel-seed0/`](checkpoints/az-duel-seed0/) (1.3 MB) as the
default `az-<n>` agent.

## Multi-mode PPO (`ppo_modes.py`)

One network for all four Battlesnake leaderboard modes, told which mode it is
playing. It is `ppo.py` with one group of games per mode, a small side input
and a reward that ranks the finishers. Run on one mode (`--game-modes duel`),
the same file trains a *specialist*, so a generalist and its specialists can
differ only in the modes they see (the generalist reported here also takes
twice their gradient steps; see the minibatches below). The question: does one policy for every mode play
as well as one policy per mode, given the same experience of each mode?

| mode | rules | snakes |
|---|---|---|
| `standard` | standard rules and map | 4 |
| `duel` | standard rules and map | 2 |
| `royale` | royale rules and map: every 25 turns one more edge row or column, on a random side, becomes hazard; hazard cells cost 14 health a turn | 4 |
| `constrictor` | constrictor: snakes grow every turn and never shrink, food is inert | 4 |

All are on the 11×11 board (`slinky.modes`).

- **Inputs.** `ppo.py`'s egocentric `21×21×13` board, plus 7 numbers added
  into the dense layer: the mode one-hot, the turn, the phase of the royale
  shrink cycle and the share of opponents still alive. A real game request
  carries all of them. In constrictor the food channel is zeroed, because
  food is inert there; that follows the real ruleset, whatever mode the
  network is told.
- **Reward.** Placement: +1/(N−1) for each opponent a snake outlives and
  −1/(N−1) for each that outlives it, paid when it happens. It is zero-sum in
  every mode, so one value head serves all four, and in the duel it is the
  win/loss reward. In 4-snake games it pays second place more than fourth,
  where win/loss gives both −1.
- **Training.** 64 games per mode, 128 turns per iteration, and one network
  call per turn on every mode's snakes together. GAE and advantage
  normalisation are per mode, and every minibatch takes an equal share of each
  mode. Each mode's value loss is weighted by 1/variance of its returns,
  rescaled so that the weights average 1. The rest is `ppo.py`: its network
  (plus the 7 inputs, 314k parameters), learning rate, clipping and entropy
  bonus.
- **Budget.** 5.24M env steps per mode, a specialist's budget, so 21M in all:
  the generalist costs as much as its four specialists together.
- **Minibatches.** With `ppo.py`'s 8 minibatches the generalist's updates were
  too small. From 0.16M to 0.33M env steps per mode, its duel KL was 0.47× the
  duel specialist's and its clip fraction 0.49×. A rule fixed before training
  said to restart with 16 minibatches if that KL ratio was under one half, and
  the 16-minibatch run is the one reported here. (Against the 4-snake
  specialists, trained later, the 8-minibatch KL ratios were 0.59–0.74× and
  the clip-fraction ratios 0.58–0.78×.) With
  16, its KL over the first 1M env steps was 1.03–1.29× each specialist's.
  The specialists keep `ppo.py`'s 8, which leaves the generalist twice their
  gradient steps. Specialists retrained with 16 are the control below.

### Results

Seed 0 throughout, and three seeds of the duel specialist.

- **The generalist (G)** took **9.75 hours** on 2 pinned cores of the 4-core
  cloud CPU: 21M env steps, 59M agent samples and 40,960 gradient steps.
- **Each specialist (S)** took 1.3 hours (duel) or 2.7–2.8 hours (the 4-snake
  modes) on the other 2 cores, 9.5 hours together, with 20,480 gradient steps
  each.

All games below are fresh games from `benchmarks/modes.py`
([`benchmarks/README.md`](../benchmarks/README.md)), with the seats rotated.
*Share* is the share of the win, summed over a policy's seats: 1 for a sole
survivor, and if nobody survives, 1/k to each of the k snakes eliminated on
the last turn (a draw). No game reached the turn cap. Brackets are 95% Wilson
intervals.

**Head to head.** The primary result, fixed before the games were played, is
G's share when it holds 2 of the 4 seats against its specialist's 2 (G,S,G,S;
in the duel G against S), with both networks greedy. Parity is 0.5. W / T / L
counts the games where G's seats took all of the win, part of it (a draw) or
none. The sampled columns are the same line-ups with both networks sampling
their moves (`ppo_modes-sample`), added after the greedy results.

| mode | G vs S, greedy | W / T / L | G vs S, sampled | W / T / L |
|---|---|---|---|---|
| standard | **0.813** [0.788, 0.836] | 833 / 0 / 191 | **0.811** [0.786, 0.834] | 829 / 3 / 192 |
| duel | **0.624** [0.594, 0.653] | 637 / 4 / 383 | **0.678** [0.648, 0.706] | 688 / 12 / 324 |
| royale | **0.705** [0.676, 0.732] | 720 / 4 / 300 | **0.698** [0.669, 0.726] | 714 / 2 / 308 |
| constrictor | 0.406 (12 distinct games) | 327 / 177 / 520 | **0.494** [0.463, 0.524] | 455 / 102 / 467 |

- **The generalist beats its specialist in standard, the duel and royale**,
  greedy or sampled, 1,024 games each.
- **In constrictor the two are level under sampled play** (0.494), and the
  greedy result is fragile.
  - Constrictor food is inert and hidden from both networks, so with greedy
    play nothing random reaches the game. The 1,024 greedy games are draws
    from just 12 distinct games: the 2 start layouts of the standard map times
    6 seatings. G wins 4 of them, draws 2 and loses 6 (5/12 = 0.417).
  - The printed interval, [0.376, 0.436], only says which of the 12 came up.
    As evidence about strength the 12 games are the whole sample, and a
    bootstrap over them gives [0.17, 0.67].
  - Every other cell has a source of randomness: the heuristic breaks ties at
    random, `ppo` sees the food, and the other modes' food matters. `--table`
    now marks a line-up like this one *deterministic*.
- **Near-greedy play shows a small constrictor deficit.** With a random legal
  move 2%, 5% or 10% of the time, the generalist scores 0.417, 0.438 and 0.461
  head to head (2,048 games each). Only full sampling brings it to level.
  These come from the
  [investigation](../docs/research/multimode_constrictor.md)'s scripts.

**Against fixed opponents.** These are the share of one seat against three
copies of the opponent, or one copy in the duel. Parity is 0.25 at a 4-snake
table and 0.5 in the duel. H is the heuristic, P the `ppo` duel checkpoint
(playing the 4-snake modes zero-shot) and R `random_legal`.

| mode | G vs H | S vs H | G vs P | S vs P | G vs R | S vs R | P vs H |
|---|---|---|---|---|---|---|---|
| standard | 0.139 | 0.028 | 0.634 | 0.316 | 0.992 | 0.977 | 0.020 |
| duel | 0.388 | 0.350 | 0.644 | 0.492 | 1.000 | 0.996 | 0.296 |
| royale | 0.341 | 0.216 | 0.763 | 0.566 | 0.992 | 0.969 | 0.033 |
| constrictor | 0.225 | 0.237 | 0.398 | 0.343 | 0.742 | 0.706 | 0.098 |

Cells against R have 256 games, P against H 512 and the rest 1,024. The
duel's P vs H is from the strength benchmark (1,024 games, cut off at 500
turns).

**Paired G − S differences** in share per seat. These compare the same games
with G in place of S, and ± is the 95% interval of the per-game differences.
G,S,H,H puts both in one game against two heuristics.

| mode | G,S,H,H (within) | vs 3 H | vs 3 P |
|---|---|---|---|
| standard | +0.179 ± 0.030 | +0.111 ± 0.024 | +0.317 ± 0.042 |
| duel | | +0.038 ± 0.039 | +0.151 ± 0.041 |
| royale | +0.187 ± 0.044 | +0.125 ± 0.038 | +0.197 ± 0.039 |
| constrictor | +0.019 ± 0.037 | −0.012 ± 0.032 | +0.056 ± 0.039 |

- **Standard, royale: clear wins head to head, against the heuristic and
  against `ppo`.**
  - Against `random_legal` both networks score near 1 and are level
    (+0.016 ± 0.022 and +0.023 ± 0.024).
  - This is one seed of each. The gains against the heuristic (+0.111,
    +0.125) are about the size of the duel specialists' seed spread
    (0.251–0.376); the head-to-head margins are much larger.
  - The standard specialist's score against the heuristic stays at the floor.
    In its 256-game training evaluations it scores 0.016 at 1M env steps and
    0.025 at 5.24M, while the generalist rises from 0.020 to 0.111. It did
    learn something: it scores 0.316 against three copies of `ppo`, above
    parity.
  - In royale the generalist scores 0.341 against three heuristics, above
    parity; its specialist scores 0.216.
- **Duel.** The generalist beats all three duel specialists head to head:
  0.624, 0.714 and 0.698 against seeds 0, 1 and 2.
  - **Against the heuristic** its 0.388 is above all three (0.350, 0.376,
    0.251). Paired, it is level with seeds 0 and 1 (+0.038 ± 0.039,
    +0.011 ± 0.036) and ahead of seed 2 (+0.137 ± 0.038).
  - **Against `ppo`**, at 0.644, it is well above all three (0.492, 0.445,
    0.471). They play `ppo` level or slightly below it.
  - **It also plays a different game.** The specialists draw often against
    the heuristic (both snakes out on the same turn): 347, 521 and 112 of
    1,024 games, where the generalist draws 12. The generalist wins 391 of
    those games, specialist seed 0 185.
- **Constrictor: no overall gain.**
  - Level head to head under sampled play (0.494) and against three
    heuristics (−0.012 ± 0.032).
  - Ahead against `ppo` (+0.056 ± 0.039).
  - Behind against one heuristic and two `random_legal` (G,H,R,R against
    S,H,R,R: −0.046 ± 0.038).
  - Both are weak: about 0.7 against `random_legal`. The board fills and games
    are decided by turn 25–30.

**Controls: specialists with 16 minibatches.** The generalist takes twice its
specialists' gradient steps (40,960 against 20,480) for the same env steps per
mode. So each food-mode specialist was
trained again with 16 minibatches (S16), which gives it the generalist's
40,960 gradient steps. Each S16 cell has 1,024 games.

| mode | G vs S16 | S16 vs S | vs 3 H: S16 / S | vs 3 P: S16 / S | S16 − S, paired within S16,S | G − S16 vs 3 H, paired |
|---|---|---|---|---|---|---|
| standard | 0.830 | 0.481 | 0.042 / 0.028 | 0.260 / 0.316 | −0.019 ± 0.031 | +0.097 ± 0.024 |
| duel | 0.678 | 0.447 | 0.385 / 0.350 | 0.432 / 0.492 | −0.106 ± 0.049 | +0.002 ± 0.036 |
| royale | 0.717 | 0.456 | 0.166 / 0.216 | 0.538 / 0.566 | −0.044 ± 0.030 | +0.176 ± 0.037 |

- **More gradient steps do not make a specialist stronger head to head.**
  - S16 is level with S or behind it (the duel's −0.106 ± 0.049 excludes 0).
  - Against `ppo` it is level or behind too.
  - The generalist beats S16 by as much as it beats S, so its head-to-head
    lead is not its extra gradient steps.
- **Against the heuristic in the duel**, S16 is slightly stronger than S
  (+0.035 ± 0.031) and level with the generalist (+0.002 ± 0.036). The
  generalist's edge there is within the spread of the specialist seeds
  anyway (above).
- **The constrictor control** was stopped at 2.9M of its 5.24M env steps
  (see the caveats).

**Is the mode input used? Hardly.** Told a wrong mode, the generalist plays
about as well.

- **Strength.** G − G@m is measured over 256 paired games for each wrong mode
  against the heuristic, 12 pairs in all. The differences run from −0.023 to
  +0.064, averaging +0.018. Eleven of the twelve 95% intervals include 0. The
  twelfth, constrictor told duel, is +0.040 ± 0.039.
- **Moves.** Its greedy move changes on only 2–6% of decisions with at least
  two legal moves, in its own self-play games (from the investigation's
  scripts).
- **Why.** The board gives the mode away. The duel has two snakes, and
  constrictor's food channel is blank. Royale has hazards from turn 25, and
  before that it looks like 4-snake standard.

**Specialists out of their mode** play badly. This is the cross-mode matrix:
each network is told its own mode and plays one seat against three
heuristics, one in the duel. Off its own mode, each cell has 256 games.

| network, played in → | standard | duel | royale | constrictor |
|---|---|---|---|---|
| standard specialist | **0.028** | 0.100 | 0.098 | 0.108 |
| duel specialist | 0.020 | **0.350** | 0.031 | 0.066 |
| royale specialist | 0.025 | 0.045 | **0.216** | 0.109 |
| constrictor specialist | 0.000 | 0.010 | 0.004 | **0.237** |
| generalist | 0.139 | 0.388 | 0.341 | 0.225 |

- **No specialist carries over to another mode.**
  - Into standard there is nothing to lose, because every specialist is at
    the floor there (0.000–0.028).
  - The best transfers, about 0.1, are the standard and royale specialists in
    constrictor and the standard specialist in the duel and royale. Each is
    well under the mode's own specialist.
- **The generalist in every mode** is at least level with each mode's own
  specialist.

**Self-play dynamics** (the last 32k env steps of each mode):

- **Entropy.** The generalist's policy is sharper than the standard and
  royale specialists': 0.23 nats against 0.34 and 0.33. It is about as sharp
  as the duel specialist's (0.23 against 0.25). In constrictor it is less
  sharp: 0.15 against 0.12.
- **The 16 minibatches explain part of the gap.** The 16-minibatch specialists
  sit at 0.29 (standard, royale) and 0.24 (duel).
- **Game length.** Self-play games last 257, 208, 175 and 32 turns for the
  generalist (standard, duel, royale, constrictor), against 272, 151, 179 and
  32 for the specialists.

### Why no gain in constrictor

The generalist gains a lot in the three food modes and has no overall gain in
constrictor. No ablation was run that separates the causes, but an
investigation
([`docs/research/multimode_constrictor.md`](../docs/research/multimode_constrictor.md))
narrowed them down:

- **Specialist headroom** (consistent with the data, not tested).
  - Constrictor games last about 32 turns, against 150–270 in the other modes
    at the end of training. Per env step, its specialist completes 3–7 times
    as many games as the others over the run.
  - Its score against the heuristic is flat from 3M env steps on (0.230,
    0.230 and 0.234).
  - A flat curve does not show that it is near its ceiling, though. The
    standard specialist's curve is flat too, at the floor, and the generalist
    still beats it 0.81.
  - The royale specialist is still climbing at the end (0.146 at 3M env
    steps, 0.203 at 5.24M).
- **Shared skills.**
  - Royale is standard plus a visible hazard.
  - The food modes' value-loss gradients point the same way (de-noised cosine
    0.66–0.87). Constrictor's is much less aligned with theirs (0.04–0.46),
    and by the end of training it points against standard's.
  - That is mostly in the value head. On the shared layers there is no direct
    evidence that constrictor learns less.
- **Small interference costs.** These are real but too small to show in the
  results:
  - a moving-tail habit from the food modes, where constrictor's tails never
    move;
  - a value-loss weight of about 0.3 (8% of the total), from constrictor's
    larger return variance;
  - a dip in the middle of training that recovered.
- **Ruled out:**
  - mode confusion (above);
  - a broken constrictor critic.
- **Not tested: the extra gradient steps.** The food-mode controls show they
  do not explain the gains there, and they tend to weaken a specialist head
  to head. The constrictor control was stopped before the end.

### Caveats

- **One seed** of the generalist and of the 4-snake specialists. The duel has
  three specialist seeds, which spread from 0.251 to 0.376 against the
  heuristic, so a 4-snake seed spread of that size is possible.
- **The pre-registered constrictor head-to-head is degenerate** (12 distinct
  games); the sampled head-to-head is the one to read. The other modes'
  greedy head-to-heads are 1,024 distinct games.
- **The mode input is barely needed**, because the board shows the mode. A
  mode the board cannot reveal would need it.
- **Constrictor is weak for both** (about 0.7 against `random_legal`). A
  dedicated study (longer games on bigger boards, or a better reward) is open.
- **Two experiments were stopped unfinished:**
  - the 16-minibatch constrictor control, stopped at 2.9M of 5.24M env steps;
  - a fine-tuning test, in which the generalist and the constrictor specialist
    each get 1M more constrictor env steps and then play each other. The runs
    reached about 0.45M before a container restart stopped them (seen in
    their live logs, since overwritten); the last saved checkpoint is at
    0.26M.
  - Neither is reported.
- **The training snapshots are not committed**, only the final networks; the
  snapshot head-to-heads in the investigation are not reproducible from the
  repository alone.

### Checkpoints and commands

The final networks are kept in `checkpoints/` (1.3 MB each, with their
`config.json` and `metrics.jsonl`):

| directory | network |
|---|---|
| [`ppo-modes-seed0/`](checkpoints/ppo-modes-seed0/) | the generalist; the default `ppo_modes` agent |
| `ppo-modes-{standard,duel,royale,constrictor}-seed0/` | the specialists |
| `ppo-modes-duel-seed{1,2}/` | duel specialist seeds 1 and 2 |
| `ppo-modes-{standard,duel,royale}-m16-seed0/` | the 16-minibatch controls |

```bash
python baselines/ppo_modes.py --num-minibatches 16 --run-dir runs/ppo-modes-seed0   # the generalist
python baselines/ppo_modes.py --game-modes duel --run-dir runs/ppo-modes-duel-seed0 # a specialist
python baselines/ppo_modes.py --game-modes royale --num-minibatches 16 \
    --run-dir runs/ppo-modes-royale-m16-seed0                                      # a control
python benchmarks/modes.py --table      # the benchmark tables above, from benchmarks/results/modes.jsonl
```

The benchmark results were played from `runs/<directory>`, of which the
checkpoints are byte-identical copies. A cell's id includes the agent's
name, which is its path. So to replay or extend the benchmark without
replaying finished cells, first copy the checkpoints into `runs/`
(`mkdir -p runs && cp -r baselines/checkpoints/ppo-modes-* runs/`). Then
run:

```bash
python benchmarks/modes.py --protocol --generalist runs/ppo-modes-seed0 \
    --specialist standard=runs/ppo-modes-standard-seed0 --specialist duel=runs/ppo-modes-duel-seed0 \
    --specialist royale=runs/ppo-modes-royale-seed0 \
    --specialist constrictor=runs/ppo-modes-constrictor-seed0 --matrix
python benchmarks/modes.py --game-mode duel --seats G,S1 \
    --agent G=ppo_modes:runs/ppo-modes-seed0 --agent S1=ppo_modes:runs/ppo-modes-duel-seed1 --games 1024
```

The protocol then finds all its 68 cells finished. The extra cells (the
other duel seeds and the controls) are played one at a time, as in the last
command. The duel's P vs H is in the strength benchmark's results, and the
training numbers (curves, entropy, game lengths, value weights) are in each
checkpoint's `metrics.jsonl`. As an agent (`slinky.agents`) the generalist is `ppo_modes`, told the
mode of the game it plays; `ppo_modes:game_mode=<mode>` overrides what it is
told.
