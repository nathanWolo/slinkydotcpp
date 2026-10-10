# Baselines

Single-file reinforcement-learning baselines for the 1v1 duel (11×11,
standard rules), trained by self-play on slinky.

| baseline | file | training | vs `random_legal` | vs heuristic | checkpoint |
|---|---|---|---|---|---|
| Double DQN | `dqn.py` | 31 min, 1.28M env steps | 0.998 | 0.053 | `checkpoints/dqn-duel-seed0/` |
| PPO | `ppo.py` | 67 min, 5.24M env steps | 0.998 | 0.296 | `checkpoints/ppo-duel-seed0/` |
| Rainbow DQN | `rainbow.py` | 81 min, 1.28M env steps | 0.983 | 0.084 | `checkpoints/rainbow-duel-seed0/` |

- **Hardware:** the DQN and PPO trained on a 4-core cloud CPU (PPO pinned to
  3 cores), Rainbow on a thermally throttled laptop CPU; no GPU.
- **Head-to-head:** PPO beats the DQN 0.868 over 1,024 games (888 / 2 / 134).
  See [the three compared](#the-three-compared) for all the pairings.
- **Sources:** the DQN and PPO scores are from the strength benchmark
  ([`benchmarks/README.md`](../benchmarks/README.md); 1,024 games each).
  Rainbow is not in that benchmark; its scores are from its own evaluation
  (5,000 games against `random_legal`, 1,000 against the heuristic). In its
  own 5,000-game evaluation below, the DQN scores 0.993 against
  `random_legal`.

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
python baselines/dashboard.py        # live training dashboard (see below)
```

Each run writes to `runs/` (gitignored):

- `config.json`;
- `metrics.jsonl`, one record per log chunk (16k env steps for the DQN, 32k
  for PPO) plus the evaluations;
- `params.npz`, the latest network;
- `runner.npz`, the full training state used by `--resume`.

## Watching training live

```bash
python baselines/dashboard.py        # then open http://127.0.0.1:8050
```

`dashboard.py` serves a page that lists every run under `runs/` and
`baselines/checkpoints/` (DQN, PPO and Rainbow alike) and re-reads its
`metrics.jsonl` every 5 seconds, so a run in progress updates as it trains.

- **Run cards** show live / finished / stopped, progress and an ETA, and the
  latest greedy score against `random_legal`. Click a card to add the run to
  the charts or take it off (up to 8, each keeping its colour). Live runs and
  the committed checkpoints are shown at first.
- **Charts** (over env steps, one shared axis):
  - greedy score against each evaluation opponent, with its 95% interval:
    `random_legal` for every run, plus the heuristic and the DQN for PPO;
  - training loss (log scale; DQN's Huber loss and Rainbow's KL are not
    comparable), or PPO's value loss;
  - the value estimate: mean Q of the moves taken, or PPO's mean critic value;
  - PPO's policy entropy;
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
