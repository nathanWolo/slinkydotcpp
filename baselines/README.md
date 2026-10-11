# Baselines

Single-file reinforcement-learning baselines for the 1v1 duel (11×11,
standard rules), trained by self-play on slinky.

| baseline | file | training | vs `random_legal` | vs heuristic | checkpoint |
|---|---|---|---|---|---|
| Double DQN | `dqn.py` | 31 min, 1.28M env steps | 0.998 | 0.053 | `checkpoints/dqn-duel-seed0/` |
| PPO | `ppo.py` | 67 min, 5.24M env steps | 0.998 | 0.296 | `checkpoints/ppo-duel-seed0/` |
| Rainbow DQN | `rainbow.py` | 81 min, 1.28M env steps | 0.983 | 0.084 | `checkpoints/rainbow-duel-seed0/` |
| Rainbow DQN, trained on | `rainbow.py --extend-to` | 5.9 h, 4.61M env steps | 0.986 | 0.158 | `checkpoints/rainbow-long-seed0/` |
| REINFORCE | `reinforce.py` | 73 min, 5.25M env steps | 0.944 | 0.013 | `checkpoints/reinforce-duel-seed0/` |

- **Hardware:** the DQN and PPO trained on a 4-core cloud CPU (PPO pinned to
  3 cores), Rainbow and REINFORCE on a thermally throttled laptop CPU; no GPU.
- **Head-to-head:** PPO beats the DQN 0.868 over 1,024 games (888 / 2 / 134).
  See [the baselines compared](#the-baselines-compared) for all the pairings.
- **Sources:** the DQN and PPO scores are from the strength benchmark
  ([`benchmarks/README.md`](../benchmarks/README.md); 1,024 games each).
  Rainbow and REINFORCE are not in that benchmark; their scores are from the
  comparison below (5,000 games against `random_legal`, 1,000 against the
  heuristic; see [training on](#training-on) for the longer Rainbow run). In its
  own 5,000-game evaluation below, the DQN scores 0.993 against
  `random_legal`.

### The baselines compared

The committed checkpoints in one harness (`slinky.evaluate.play_match`:
seats alternate, greedy play, the same seed 2026 for every match; score is
win + ½ draw with its 95% interval):

| match | games | W / D / L | score |
|---|--:|---|---|
| PPO vs DQN | 1,000 | 872 / 1 / 127 | 0.873 ± 0.021 |
| PPO vs Rainbow | 1,000 | 751 / 5 / 244 | 0.753 ± 0.027 |
| Rainbow vs DQN | 1,000 | 499 / 10 / 491 | 0.504 ± 0.031 |
| PPO vs Rainbow 4.61M | 1,000 | 555 / 8 / 437 | 0.559 ± 0.031 |
| Rainbow 4.61M vs DQN | 1,000 | 689 / 7 / 304 | 0.693 ± 0.028 |
| Rainbow 4.61M vs Rainbow | 1,000 | 608 / 10 / 382 | 0.613 ± 0.030 |
| REINFORCE vs DQN | 1,000 | 327 / 4 / 669 | 0.329 ± 0.029 |
| REINFORCE vs PPO | 1,000 | 53 / 4 / 943 | 0.055 ± 0.014 |
| REINFORCE vs Rainbow | 1,000 | 282 / 4 / 714 | 0.284 ± 0.028 |
| REINFORCE vs Rainbow 4.61M | 1,000 | 149 / 2 / 849 | 0.150 ± 0.022 |

| agent | vs `random_legal` (5,000 games) | vs heuristic (1,000 games) |
|---|---|---|
| DQN | 0.992 ± 0.002 (4951 / 16 / 33) | 0.047 ± 0.013 (45 / 4 / 951) |
| PPO | **0.996 ± 0.002** (4967 / 22 / 11) | **0.286 ± 0.025** (196 / 181 / 623) |
| Rainbow | 0.983 ± 0.003 (4896 / 33 / 71) | 0.084 ± 0.017 (78 / 12 / 910) |
| Rainbow 4.61M | 0.986 ± 0.003 (4927 / 10 / 63) | 0.158 ± 0.023 (156 / 3 / 841) |
| REINFORCE | 0.944 ± 0.006 (4703 / 38 / 259) | 0.013 ± 0.007 (13 / 0 / 987) |

"Rainbow" is the committed 1.28M-step run; "Rainbow 4.61M" is the same run
trained on (see [training on](#training-on)). Its checkpoint was picked as
the best of 18 snapshots on these same seed-2026 games against PPO and the
heuristic, so those two of its scores are biased upward a little. On 2,000
fresh games (seed 7) it scores 0.419 ± 0.022 against PPO and 0.167 ± 0.016
against the heuristic.

- **PPO is the strongest** on every count. It beats both DQNs and REINFORCE
  head to head and scores 3–6 times as much as either DQN against the
  heuristic.
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
  would need longer DQN and Rainbow runs. Rainbow trained on to 4.61M env
  steps (close to PPO's 5.24M; see [training on](#training-on)) narrows the gap but does not close it: PPO
  still beats it (0.559 here, 0.581 on fresh games) and scores about
  1.8 times as much against the heuristic (0.286 against 0.158).
- **REINFORCE is the weakest** on every count, on the same network and env
  steps as PPO. It loses to the DQN 0.329 and to PPO 0.055, wins 13 of 1,000
  games against the heuristic, and loses 5.2% of its games to
  `random_legal`, where every other baseline loses under 1.5%. See [REINFORCE](#reinforce-reinforcepy)
  for why.
- **One seed each**, so the run-to-run spread of each algorithm is unknown.

```bash
uv pip install -e ".[rl]"            # adds optax
python baselines/dqn.py              # default run, ~30 min on a 4-core CPU
python baselines/ppo.py              # default run, ~70 min on 3 cores
python baselines/dqn.py --help       # every hyperparameter is a flag (same for ppo.py)
python baselines/dqn.py --eval-only runs/dqn-<timestamp> --eval-games 5000
python baselines/dqn.py --resume runs/dqn-<timestamp>   # continue an interrupted run
python baselines/rainbow.py          # Rainbow DQN; same flags, plus its own
python baselines/reinforce.py        # REINFORCE with a value baseline, ~75 min
python baselines/reinforce.py --baseline none   # plain REINFORCE
python baselines/rainbow.py --n-step 1 --per-alpha 0    # ablate components
python baselines/rainbow.py --resume runs/rainbow-<timestamp> --extend-to 5000000
                                     # train a finished run on, to 5M env steps
python baselines/dashboard.py        # live training dashboard (see below)
```

Each run writes to `runs/` (gitignored):

- `config.json`;
- `metrics.jsonl`, one record per log chunk (16k env steps for the DQN, 32k
  for PPO, 32 updates for REINFORCE, about 23k env steps) plus the
  evaluations;
- `params.npz`, the latest network;
- `runner.npz`, the full training state used by `--resume`.

## Watching training live

```bash
python baselines/dashboard.py        # then open http://127.0.0.1:8050
```

`dashboard.py` serves a page that lists every run under `runs/` and
`baselines/checkpoints/` (DQN, PPO, Rainbow and REINFORCE alike) and re-reads its
`metrics.jsonl` every 5 seconds, so a run in progress updates as it trains.

- **Run cards** show live / finished / stopped, progress and an ETA, and the
  latest greedy score against `random_legal`. Click a card to add the run to
  the charts or take it off (up to 8, each keeping its colour). Live runs and
  the committed checkpoints are shown at first.
- **Charts** (over env steps, one shared axis):
  - greedy score against each evaluation opponent, with its 95% interval:
    `random_legal` for every run, plus the heuristic and the DQN for PPO and
    REINFORCE;
  - training loss (log scale; DQN's Huber loss and Rainbow's KL are not
    comparable), or PPO's and REINFORCE's value loss;
  - the value estimate: mean Q of the moves taken, or the mean value-head
    estimate;
  - PPO's and REINFORCE's policy entropy;
  - self-play game length and draw rate, and throughput;
  - ε or the noisy-net σ.
- **Death causes**: the share of self-play eliminations by cause, per run.
- **Games tab**: recorded games in the replay viewer. It lists every
  `slinky-replay/1` file under `replays/` (gitignored; `--replays DIR` for
  others) with its matchup, score and game length; pick one to play it, or
  link to it with `#games=<file>`. Record games for it with, e.g.:

  ```bash
  python -m slinky.replay --a rainbow:baselines/checkpoints/rainbow-long-seed0 --b ppo \
      --games 8 --seed 2026 --max-turns 1000 --out replays/rainbow/rainbow-vs-ppo.json
  ```
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

## REINFORCE (`reinforce.py`)

Williams' (1992) REINFORCE with a learned value baseline: the textbook
policy gradient, on PPO's network and self-play setup (one network plays both
snakes, illegal moves masked out). It is the control for PPO: the same model
and env steps, without GAE, sample reuse or the clipped objective. The file is
self-contained.

- **Whole games.** Each update plays 8 fresh games to the end, in one jitted
  loop that stops once every game is over. Nothing is cut off mid-game or
  bootstrapped. A game that reached `max_turns` would end with return 0, like
  a draw; none did.
- **Returns.** Every move is weighted by its discounted Monte Carlo return
  G_t = r_t + γ G_{t+1}, minus the value head's estimate V(s_t). With
  `--baseline none` it is plain REINFORCE. The reward is the sparse win/loss
  reward, unshaped.
- **One gradient step per batch.** The loss is the mean over every move of
  every game (about 1,460 agent samples per batch), and each sample is used
  once. Adam at 3e-3, constant; gradient clipping at 0.5; γ = 0.95; value
  coefficient 0.5; entropy coefficient 0.001.
- **Memory.** The games are stored as compact game states (under 1 KB per
  turn). The update rebuilds the observations 50 turns at a time and skips
  the blocks after the longest game.
- **Network.** PPO's: convolutions 16 → 32 → 32 with strides 2, 2, 1, dense
  256, a policy head and a value head (312k parameters), with the same
  hand-written convolution backward pass.
- **Play mode.** Like PPO, the `reinforce` agent plays greedily and
  `reinforce-sample` samples. The two are within noise of each other (below).

### Results

Default config, seed 0, 5.25M env steps (10.5M agent samples, 7,200 gradient
steps). It took **73 minutes** on a thermally throttled 12-thread laptop CPU,
five evaluation rounds included: about 1,200 env steps/s overall, 1,310
while training.

**During training** (256 games per cell, greedy / sampled score; ± is at most
0.06):

| env steps | vs `random_legal` | vs DQN | vs heuristic | W / D / L vs heuristic (greedy) |
|--:|---|---|---|---|
| 1.06M | 0.994 / 0.992 | 0.379 / 0.402 | 0.057 / 0.084 | 7 / 15 / 234 |
| 2.10M | 0.955 / 0.961 | 0.246 / 0.293 | 0.014 / 0.029 | 3 / 1 / 252 |
| 3.17M | 0.965 / 0.959 | 0.381 / 0.404 | 0.031 / 0.027 | 3 / 10 / 243 |
| 4.20M | 0.887 / 0.871 | 0.064 / 0.062 | 0.008 / 0.008 | 2 / 0 / 254 |
| 5.25M | 0.959 / 0.943 | 0.316 / 0.275 | 0.012 / 0.039 | 3 / 0 / 253 |

**Final checkpoint** (the harness of [the baselines compared](#the-baselines-compared):
seed 2026, seats alternating; greedy / sampled):

| opponent | games | greedy W / D / L | greedy | sampled |
|---|--:|---|---|---|
| `random_legal` | 5,000 | 4703 / 38 / 259 | 0.944 ± 0.006 | 0.948 ± 0.006 |
| heuristic | 1,000 | 13 / 0 / 987 | 0.013 ± 0.007 | 0.022 ± 0.009 |
| DQN | 1,000 | 327 / 4 / 669 | 0.329 ± 0.029 | 0.286 ± 0.028 |
| PPO | 1,000 | 53 / 4 / 943 | 0.055 ± 0.014 | 0.045 ± 0.013 |
| Rainbow | 1,000 | 282 / 4 / 714 | 0.284 ± 0.028 | |
| Rainbow 4.61M | 1,000 | 149 / 2 / 849 | 0.150 ± 0.022 | |

- **It stopped improving after about 1M env steps.** No later evaluation
  round beat the first by more than noise; the scores swing from round to
  round instead of rising (0.38, 0.25, 0.38, 0.06 and 0.32 against the DQN).
- **The policy collapsed to near-determinism.** Its entropy fell from 0.92
  nats to 0.10 by 1M steps and averaged 0.07 over the last million (PPO ends
  at 0.24). Self-play between two copies of a near-deterministic policy keeps
  switching between strategies that beat the previous one, and the
  evaluations swing with it.
- **The value baseline barely helps.** Its explained variance peaked at
  about 0.3 near 0.5M steps and averaged 0.04 over the last million.

**Self-play dynamics.**

- For its first ~200k steps it played almost uniformly (entropy 0.92), with
  72% of deaths head-to-head and games about 70 turns long. So did every
  pilot but one (lr 3e-3 with no entropy bonus, below).
- Head-to-heads were still 69% of deaths at about 1M steps. Over the last
  million they fell to 46%, with running into its own body (29%) and the
  other snake's (18%) taking the rest, and games lasted about 117 turns. PPO
  ends at about 20% head-to-heads and 195-turn games.

**Why it is so far behind PPO.** The most obvious difference is how much
learning each sample buys: REINFORCE takes about a third as many gradient
steps (7,200 against 20,480), and each sample feeds one step where PPO's
feeds four. Short pilots (one seed each, 210–860k env steps) are only weakly
suggestive, and one of them cuts the other way:

- **PPO cut to one gradient step per batch** (16 envs × 64 turns, 1 epoch,
  1 minibatch, clipping off, lr 1e-3) lost default PPO's steady decline.
  Once it started learning, at about 260k steps, its entropy swung between
  0.43 and 0.83 up to 410k and its games between 48 and 98 turns. Default
  PPO's entropy fell steadily from 0.58 to about 0.47 over the same steps.
- **Fewer games per update** (4 instead of 8, at γ 0.99, lr 1e-3 and no
  entropy bonus) did not start learning any sooner per env step: still
  uniform at 244k steps after 864 updates, like its 8-game counterpart at
  213k after 384. Twice the gradient steps bought nothing there, though each
  step averaged half as many games.

The pilots also set the defaults:

- **Slow start.** At lr 1e-3, γ 0.9 and 0.95 started learning at about
  270–290k steps; at lr 3e-3 (γ 0.99) learning had started by 243k. The
  γ 0.99 pilots at lr 1e-3 stopped at about 230k, still uniform, too soon to
  compare γ. The defaults take lr 3e-3, and γ 0.95 to concentrate the credit
  on the last ~20 moves, like PPO's GAE λ = 0.95.
- **No entropy bonus collapses it.** At lr 3e-3 and entropy coefficient 0,
  entropy fell to 0.14 by 127k steps and 0.05 by 171k, with many starvation
  deaths, then swung for the rest of the 858k-step pilot. A coefficient of
  0.003 was still uniform when its pilot stopped at 250k steps (0.01 at the
  same lr and γ had started by 243k, so when learning starts is noisy). The
  default 0.001 keeps a small bonus.

### Caveats

- **One seed.**
- **Snapshot luck.** The scores swing between evaluation rounds, and only the
  last network was kept, so the checkpoint is one point of an oscillation. The
  first round (1.06M steps) was about as strong as any.
- **Not tuned to the end.** The pilots were short and single-seed. Larger
  batches, a decaying learning rate or a stronger entropy bonus might hold off
  the collapse; none was tried at full length.

The final network is kept in
[`checkpoints/reinforce-duel-seed0/`](checkpoints/reinforce-duel-seed0/) (1.3
MB) as the default `reinforce` agent:
`python baselines/reinforce.py --eval-only baselines/checkpoints/reinforce-duel-seed0`.

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

### Training on

The heuristic score was still rising at 1.28M steps, so the same run was
trained on with `--resume … --extend-to`. Extending keeps PER's β schedule at
its original length (β stays at 1); the noise has no schedule. Every 256k
env steps the network was snapshotted and played 1,000 games against PPO and
1,000 against the heuristic (seed 2026, the same games for every snapshot).
A plateau rule was to stop the run once the mean of the last four snapshots
beat the mean of the four before by at most 0.02 against both opponents.

The first extension, to PPO's 5.25M env steps, ended on its budget with the
score against PPO still rising. A second extension (to 8.06M) stopped at the
next snapshot, 5.38M, by the rule. The last-four window there spans
4.86M–5.38M.

| env steps | vs PPO | vs heuristic |
|--:|---|---|
| 1.28M | 0.247 | 0.084 |
| 1.54M | 0.326 | 0.095 |
| 1.79M | 0.307 | 0.099 |
| 2.05M | 0.300 | 0.100 |
| 2.30M | 0.304 | 0.128 |
| 2.56M | 0.275 | 0.120 |
| 2.82M | 0.323 | 0.114 |
| 3.07M | 0.346 | 0.150 |
| 3.33M | 0.355 | 0.150 |
| 3.58M | 0.333 | 0.151 |
| 3.84M | 0.351 | 0.131 |
| 4.10M | 0.350 | 0.150 |
| 4.35M | 0.418 | 0.157 |
| **4.61M** | **0.441** | 0.158 |
| 4.86M | 0.340 | 0.117 |
| 5.12M | 0.426 | **0.162** |
| 5.25M | 0.378 | 0.156 |
| 5.38M | 0.326 | 0.140 |

Each score has a 95% interval of about ±0.03 against PPO and ±0.02 against
the heuristic. The six snapshots from 4.35M on were played again on 2,000
fresh games (seed 7):

| snapshot | vs PPO | vs heuristic |
|--:|---|---|
| 4.35M | 0.414 ± 0.022 | 0.162 ± 0.016 |
| **4.61M** | **0.419 ± 0.022** | **0.167 ± 0.016** |
| 4.86M | 0.309 ± 0.020 | 0.131 ± 0.014 |
| 5.12M | 0.413 ± 0.022 | 0.154 ± 0.016 |
| 5.25M | 0.412 ± 0.022 | 0.142 ± 0.015 |
| 5.38M | 0.315 ± 0.020 | 0.123 ± 0.014 |

- **Against the heuristic** Rainbow levelled off at about 0.15 from 3M steps.
- **Against PPO** it climbed to about 0.41 by 4.35M steps. After that the
  network swings between snapshots: four score about 0.41 on fresh games and
  two (4.86M and 5.38M) about 0.31, far outside their intervals. Training is
  no longer improving the policy, only moving it, and the plateau rule fired
  on a weak snapshot.
- **Picking a snapshot therefore matters** as much as training longer. On the
  shared games the four good snapshots ranged from 0.378 to 0.441; on fresh
  games they are level against PPO (0.412–0.419), and 4.61M is best against
  the heuristic.

The 4.61M snapshot is kept as
[`checkpoints/rainbow-long-seed0/`](checkpoints/rainbow-long-seed0/) (5.5 MB;
its `metrics.jsonl` stops at 4.61M). The run to 4.61M took 5.7 hours of
training on the same laptop (5.9 hours with its built-in evaluations): about
265 env-steps/s for the first 1.28M and about 205 for the extension, which
shared the CPU with the snapshot evaluations. In
[the baselines compared](#the-baselines-compared):

- **It beats the 1.28M Rainbow** 0.613 ± 0.030 and **the DQN** 0.693 ± 0.028,
  and nearly doubles the heuristic score (0.158 against 0.084).
- **PPO is still stronger** at a similar budget (5.24M env steps): 0.559 ±
  0.031 here, 0.581 on fresh games.
- **Self-play changed little** after 1.28M steps. Games grow from about 240
  to 270 turns. Head-to-head deaths fall from about a third to 28% and
  self-collisions rise to 47%. The mean Q stays between +0.02 and +0.04, and
  the noise σ shrinks slowly (0.0084 to 0.0069, from 0.0104 at the start).
  The largest PER priority grows from 12 to about 22.

### Caveats and next steps

- **One seed**, as for the DQN, and the default hyperparameters were not
  tuned. A 320k-step pilot with the same settings scored 0.931–0.945
  against `random_legal`.
- **The policy oscillates late in training** with a constant learning rate
  and hard target-network copies. A decaying learning rate, softer or less
  frequent target updates, or averaging the weights of recent snapshots are
  the next things to try; then a larger network, more updates per step, and
  ablations of the six components (the flags above).

The 1.28M network is kept in
[`checkpoints/rainbow-duel-seed0/`](checkpoints/rainbow-duel-seed0/) (5.5 MB)
as the `rainbow` agent of `slinky.agents`:
`python baselines/rainbow.py --eval-only baselines/checkpoints/rainbow-duel-seed0`.
The longer run is the agent
`rainbow:baselines/checkpoints/rainbow-long-seed0`.
