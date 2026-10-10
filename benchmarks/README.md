# Benchmarks

| script | measures |
|---|---|
| `throughput.py` | environment steps per second (see the main README's Performance section) |
| `strength.py` | playing strength: every pair of two agent lists, as resumable JSONL results and markdown tables |
| `plot_strength.py` | the MCTS-strength-vs-simulations chart below, from `strength.py`'s results |
| `mcts_latency.py` | the latency of one MCTS move of one game, and the budget that fits a time limit |

## Strength of MCTS against the number of simulations

The score of simultaneous-move MCTS (`slinky.mcts`, default config) against
four fixed opponents, by simulations per move:

- **Setup:** the 11×11 duel on standard rules. Games are cut off at 500 turns
  and count as draws, which happened in at most 10 games per matchup.
- **Scoring:** seats alternate, and the score is (wins + draws/2) / games.
  Brackets are 95% Wilson intervals.
- **Games:** 512 per matchup, with three exceptions:
  - 256 for MCTS-2048 and MCTS-4096 against the DQN and the heuristic (they
    did not play `random_legal`);
  - 128 for MCTS-24000 against PPO and the heuristic;
  - 64 for MCTS-24000 against `random_legal` and the DQN.
- **MCTS-24000** is the 500 ms budget: the most simulations for which one
  game's move takes at most 500 ms at p90 on one core (see
  [below](#mcts-at-500-ms-per-move)).

![MCTS score against random_legal, the DQN, PPO and the heuristic, by simulations per move](results/strength.svg)

| simulations | vs `random_legal` | vs DQN | vs PPO | vs heuristic | W / D / L vs heuristic | ms per move |
|--:|---|---|---|---|---|--:|
| 1 | 0.504 [0.461, 0.547] | 0.004 [0.001, 0.014] | 0.007 [0.003, 0.019] | 0.009 [0.004, 0.021] | 1 / 7 / 504 | 0.06 |
| 4 | 0.860 [0.828, 0.888] | 0.427 [0.385, 0.470] | 0.127 [0.101, 0.159] | 0.106 [0.083, 0.136] | 9 / 91 / 412 | 0.08 |
| 16 | 0.990 [0.977, 0.996] | 0.947 [0.924, 0.964] | 0.642 [0.599, 0.682] | 0.321 [0.282, 0.363] | 108 / 113 / 291 | 0.10 |
| 32 | 0.992 [0.980, 0.997] | 0.943 [0.920, 0.960] | 0.785 [0.748, 0.819] | 0.528 [0.485, 0.571] | 172 / 197 / 143 | 0.20 |
| 64 | 0.994 [0.983, 0.998] | 0.980 [0.964, 0.989] | 0.800 [0.763, 0.832] | 0.603 [0.560, 0.644] | 200 / 217 / 95 | 0.34 |
| 128 | 0.995 [0.984, 0.998] | 0.954 [0.932, 0.969] | 0.817 [0.782, 0.848] | 0.625 [0.582, 0.666] | 200 / 240 / 72 | 0.79 |
| 256 | 0.993 [0.981, 0.997] | 0.984 [0.969, 0.992] | 0.812 [0.776, 0.844] | 0.715 [0.674, 0.752] | 260 / 212 / 40 | 0.97 |
| 512 | 0.991 [0.979, 0.996] | 0.967 [0.947, 0.979] | 0.841 [0.807, 0.870] | 0.718 [0.677, 0.755] | 253 / 229 / 30 | 2.1 |
| 1024 | 0.995 [0.984, 0.998] | 0.988 [0.975, 0.995] | 0.884 [0.853, 0.909] | 0.777 [0.739, 0.811] | 308 / 180 / 24 | 6.0 |
| 2048 | | 0.973 [0.945, 0.987] | 0.910 [0.882, 0.932] | 0.846 [0.796, 0.885] | 178 / 77 / 1 | 12 |
| 4096 | | 0.971 [0.942, 0.985] | 0.914 [0.887, 0.935] | 0.840 [0.790, 0.880] | 181 / 68 / 7 | 23 |
| **24000** (500 ms) | 0.992 [0.929, 0.999] | 1.000 [0.943, 1.000] | 0.949 [0.896, 0.976] | 0.875 [0.807, 0.922] | 99 / 26 / 3 | 186 |

The last column is the time for one move of one game against the heuristic:
MCTS's search, the heuristic's move and the environment step. It is a
throughput on one pinned core with 32 games in flight, measured on the first
turns. The search costs about 3.5–6 µs per simulation up to 4096 simulations,
and about 8 µs at 24,000. One game alone, unbatched, is slower per
simulation: about 18 µs (see the 500 ms section).

**Reference matchups** (same setup):

| match | games | W / D / L | score |
|---|--:|---|---|
| heuristic vs `random_legal` | 1,024 | 1001 / 22 / 1 | 0.988 [0.980, 0.993] |
| heuristic vs DQN | 1,024 | 967 / 6 / 51 | 0.947 [0.932, 0.959] |
| DQN vs `random_legal` | 1,024 | 1020 / 3 / 1 | 0.998 [0.992, 0.999] |
| heuristic vs heuristic | 512 | 15 / 476 / 21 | 0.494 [0.451, 0.537] |
| PPO vs `random_legal` | 1,024 | 1020 / 3 / 1 | 0.998 [0.992, 0.999] |
| PPO vs DQN | 1,024 | 888 / 2 / 134 | 0.868 [0.846, 0.888] |
| heuristic vs PPO | 1,024 | 622 / 197 / 205 | 0.704 [0.675, 0.731] |
| PPO (greedy) vs PPO (sampled) | 512 | 280 / 2 / 230 | 0.549 [0.506, 0.591] |

**Doubling ladder.** Each MCTS budget plays the budget below it. Elo
differences are computed from the score as 400·log10(s / (1 − s)).

| match | W / D / L | score | Elo | cumulative from MCTS-16 |
|---|---|---|--:|--:|
| MCTS-32 vs MCTS-16 | 325 / 55 / 132 | 0.688 [0.647, 0.727] | +138 | +138 |
| MCTS-64 vs MCTS-32 | 306 / 73 / 133 | 0.669 [0.627, 0.708] | +122 | +260 |
| MCTS-128 vs MCTS-64 | 253 / 122 / 137 | 0.613 [0.570, 0.654] | +80 | +340 |
| MCTS-256 vs MCTS-128 | 249 / 138 / 125 | 0.621 [0.578, 0.662] | +86 | +426 |
| MCTS-512 vs MCTS-256 | 241 / 147 / 124 | 0.614 [0.571, 0.655] | +81 | +507 |
| MCTS-1024 vs MCTS-512 | 225 / 148 / 139 | 0.584 [0.541, 0.626] | +59 | +566 |
| MCTS-2048 vs MCTS-1024 | 221 / 140 / 151 | 0.568 [0.525, 0.611] | +48 | +613 |
| MCTS-24000 vs MCTS-4096 (128 games) | 82 / 23 / 23 | 0.730 [0.648, 0.800] | +173 | |

The last row is not a doubling: 24,000 simulations is 2.55 doublings above
4096. MCTS-4096 vs MCTS-2048 was not played, so that row has no cumulative
figure.

### MCTS at 500 ms per move

Battlesnake's usual time control is 500 ms per move. A server answers one game
per request, so what matters is the latency of one unbatched search, not the
batched throughput above. `mcts_latency.py` measures it:

- **Program:** `jax.jit(slinky.mcts.search)` on one game, pinned to one core,
  with the other three cores busy. Dispatch and waiting for the result are
  included; compilation, JSON parsing and the network are not.
- **Positions:** 12 positions from games of MCTS-32 against the heuristic
  (turns 8, 50 and 150), 8 searches each.
- **Budget:** the largest multiple of 1,000 simulations whose p90 latency is
  within 500 ms.

That is **24,000 simulations**: median 439 ms, p90 489 ms, max 535 ms, about
18 µs per simulation. At the median latency, about 26,000 simulations fit.
For 400 ms, which leaves time for the network, about 18,000 fit at p90.

MCTS-24000 then played the benchmark like any other budget (batched, so the
games run faster than real time):

| opponent | games | W / D / L | score of MCTS-24000 |
|---|--:|---|---|
| `random_legal` | 64 | 63 / 1 / 0 | 0.992 [0.929, 0.999] |
| DQN | 64 | 64 / 0 / 0 | 1.000 [0.943, 1.000] |
| PPO | 128 | 119 / 5 / 4 | 0.949 [0.896, 0.976] |
| heuristic | 128 | 99 / 26 / 3 | 0.875 [0.807, 0.922] |
| MCTS-4096 | 128 | 82 / 23 / 23 | 0.730 [0.648, 0.800] |

### What the numbers say

- **Where the baselines sit on the MCTS scale.**
  - The heuristic plays like MCTS with about 32 simulations: MCTS-32 scores
    0.528 against it.
  - PPO sits between MCTS-4 and MCTS-16, nearer 16: MCTS-4 scores 0.127
    against it and MCTS-16 0.642. Against the heuristic, PPO scores 0.296,
    close to MCTS-16's 0.321.
  - The DQN sits between MCTS-4 and MCTS-16, nearer 4: MCTS-4 scores 0.427
    against it and MCTS-16 0.947.
  - Against the DQN, the heuristic scores 0.947 (like MCTS-16) and PPO 0.868.
  - The order is `random_legal` < DQN < PPO < heuristic ≈ MCTS-32.
  - So, as a target for RL agents: beating the heuristic is roughly beating
    MCTS-32, MCTS-2048 is about 475 Elo above MCTS-32 on the ladder above,
    and MCTS at 500 ms per move is the strongest reference.
- **PPO loses to MCTS differently from the heuristic.**
  - Against PPO, MCTS flattens at 0.79–0.84 from 32 to 512 simulations, then
    climbs to 0.91 at 2048–4096 and 0.95 at 500 ms.
  - PPO rarely draws. Against MCTS-32 to MCTS-4096 it draws 4–12% of its
    games, the heuristic 27–47%. At 2048 and 4096 alone it is 4–5% against
    27–30%.
  - PPO also wins more. From 256 simulations up, MCTS loses more games to PPO
    than to the heuristic: 6.4% and 6.6% at 2048 and 4096, against 0.4% and
    2.7%.
  - Even so, the heuristic scores more against strong MCTS (0.15–0.16 against
    MCTS-2048 and MCTS-4096, PPO 0.09), because it takes draws. It also beats
    PPO 0.704 head-to-head.
- **Against the heuristic, the curve keeps rising to 2048 simulations, then
  flattens.**
  - 0.85 at 2048, 0.84 at 4096.
  - The gap that remains is draws, not losses: MCTS-2048 and MCTS-4096 lost
    1 and 7 of 256 games but drew 77 and 68.
  - None of those draws are cut-off games; they are mutual eliminations.
  - We recorded 64 games each at 256 and 1024 simulations. Every draw in
    them (31 and 25 games) was a mutual head-on collision, with a median on
    turn 4: an equal-length clash in the opening.
  - The heuristic itself takes such trades (see contempt below). Avoiding them
    means reading its move in a simultaneous-move game, which DUCT does not
    model.
- **Against the DQN**, MCTS scores 0.94–0.99 from 16 to 4096 simulations. A
  few losses remain at every budget up to 4096 (6–7 of 256 at 2048 and 4096).
- **Against `random_legal`** the benchmark is saturated from 16 simulations.
  MCTS-1 is a sanity check: with one simulation the root has one visited move,
  picked uniformly among the legal ones, so it plays like `random_legal`
  (0.504).
- **500 ms per move.**
  - Against the heuristic, MCTS-24000 scores 0.875, inside the intervals of
    MCTS-2048 (0.846) and MCTS-4096 (0.840). Six times the simulations of
    MCTS-4096 does not move this score measurably: 26 of the 128 games were
    draws and 3 were losses.
  - Against PPO it scores 0.949 (5 draws, 4 losses). That is up from 0.914 at
    4096, but the intervals overlap.
  - Against MCTS-4096 it scores 0.730 [0.648, 0.800], about +173 Elo for
    2.55 doublings. That is about 68 Elo per doubling (the interval spans
    about 40–95), in line with the top of the ladder (+48 to +59).
  - So against another search the extra time still pays, even though the
    score against the heuristic has stopped moving.
  - It won all 64 games against the DQN, and 63 of 64 against `random_legal`
    (one draw).
- **Doubling the budget** is worth +138 Elo at 16→32, falling to +48 at
  1024→2048, which is the usual diminishing return. Draws grow from 11% of
  ladder games at 16→32 to 27–29% at the top.

### Reproducing

Results are in `results/strength.jsonl`. A matchup's games depend only on the
seed, the two canonical agent names and the game index, so these commands
replay exactly the same games.

- **First sweep** (MCTS against `random_legal`, the DQN and the heuristic, the
  ladder and the reference matchups): 44 minutes on a 4-core CPU with 4
  pinned workers, at commit `506a4c2`, all with one code fingerprint
  (`68a75d04928ededb`).
- **PPO and 500 ms runs:** about 4 hours of wall time on the same machine,
  with four pinned single-worker processes taking matchups from a queue.
  - The five MCTS-24000 matchups took 12 minutes to 2.9 hours each (6.9
    core-hours in all).
  - The PPO matchups against MCTS-4096 and below took about 2 core-hours.
  - They carry newer fingerprints, because `slinky/mcts.py` (the in-place
    unbatched search), `slinky/agents.py` (the PPO agents) and `strength.py`
    changed in between.
  - The batched search the benchmark runs is unchanged: re-running MCTS-16 and
    MCTS-256 against the heuristic with the new code gave the same games (same
    W/D/L and mean game length).

The first sweep:

```bash
python benchmarks/strength.py --a mcts-1,mcts-4,mcts-16,mcts-32,mcts-64,mcts-128,mcts-256,mcts-512,mcts-1024 \
    --b random_legal,heuristic,dqn --games 512 --workers 4
python benchmarks/strength.py --a mcts-2048,mcts-4096 --b heuristic,dqn --games 256 --workers 4
for k in 16 32 64 128 256 512 1024; do   # the ladder: one pair per call (--a x --b plays every pair)
    python benchmarks/strength.py --a mcts-$((2 * k)) --b mcts-$k --games 512
done
python benchmarks/strength.py --a heuristic,dqn --b random_legal --games 1024
python benchmarks/strength.py --a heuristic --b dqn --games 1024
python benchmarks/strength.py --a heuristic --b heuristic --games 512
```

PPO, and MCTS at 500 ms per move:

```bash
python benchmarks/mcts_latency.py --core 3 --sims 20000,24000 --stat p90 --repeats 8   # 24,000 at p90
python benchmarks/strength.py --a mcts-1,mcts-4,mcts-16,mcts-32,mcts-64,mcts-128,mcts-256,mcts-512,mcts-1024,mcts-2048,mcts-4096 \
    --b ppo --games 512 --workers 4
python benchmarks/strength.py --a ppo --b random_legal,dqn --games 1024
python benchmarks/strength.py --a heuristic --b ppo --games 1024
python benchmarks/strength.py --a ppo --b ppo-sample --games 512
python benchmarks/strength.py --a mcts-24000 --b heuristic,ppo,mcts-4096 --games 128 --mem-mb 2400 --workers 3
python benchmarks/strength.py --a mcts-24000 --b random_legal,dqn --games 64 --mem-mb 2400 --workers 2
python benchmarks/strength.py --table          # all tables, including W/D/L, game lengths and timings
python benchmarks/plot_strength.py             # results/strength.svg
```

`--mem-mb 2400` lets 32 games of MCTS-24000 run at once (its trees take about
25 MB per game); the default budget of 1,024 MB runs 20. The number of games in
flight changes the speed, not the games.

`strength.py --help` lists every option: `--dry-run` prints the plan and a
cost estimate, `--rerun` replays finished matchups, and `--target-ci` stops a
matchup early. The module docstring describes the result fields and the
resume rules. Agent names come from `slinky.agents`; for example,
`mcts-256-rm`, `mcts-128:exploration=1.0` or `dqn:<run dir>`.

## The opponents

- **`random_legal`** (`slinky.evaluate.random_legal`) moves uniformly at
  random among the moves that don't certainly die next turn.
- **DQN** is the self-play Double DQN in
  [`baselines/`](../baselines/README.md), checkpoint
  `baselines/checkpoints/dqn-duel-seed0` (31 minutes of CPU training).
- **PPO** is the self-play PPO in [`baselines/`](../baselines/README.md),
  checkpoint `baselines/checkpoints/ppo-duel-seed0` (67 minutes of CPU
  training), playing greedily (`ppo`; `ppo-sample` samples from its policy).
- **The heuristic** (`slinky.heuristic.heuristic`) is a hand-written snake;
  the module docstring has the details.
  - It uses time-aware flood fills that know when each body cell frees up,
    plus Voronoi territory, food inside that territory, length and hunger
    terms.
  - It chooses moves with a one-ply simultaneous-move search over the exact
    rules. Each of its moves is scored by the worst and the mean outcome over
    the opponent's replies.
  - Safety tiers come first: legal, then no losing head-to-head, then enough
    room to fit.
  - A move costs about 0.05 ms, including the environment step.

### Contempt

The heuristic values a mutual elimination (both snakes die on the same turn,
a draw) at −0.25 rather than 0. Raising that contempt makes it avoid
equal-length head-ons more, but not play better:

| contempt | heuristic vs itself: draws | turns | vs the default (0.25) | vs DQN | vs `random_legal` |
|--:|--:|--:|---|---|---|
| 0 | 96% | 10 | 0.488 (76% draws) | 0.949 | 0.978 |
| 0.1 | 94% | 13 | 0.494 (78% draws) | 0.949 | 0.986 |
| **0.25** (default) | 92% | 19 | (itself) | 0.958 | 0.992 |
| 0.4 | 67% | 61 | 0.484 (69% draws) | 0.957 | 0.995 |
| 0.6 | 34% | 125 | 0.412 (19% draws) | 0.957 | 0.996 |
| 0.9 | 4% | 188 | 0.383 (4% draws) | 0.954 | 0.997 |

These were measured before the review fixes: 500 mirror games, 1,000 against
the default and the DQN, 500 against `random_legal`.

- **Mirror games** between two copies end in an opening head-on more than 90%
  of the time at contempt 0–0.25: both go for the same cell at equal length.
  In 128 mirror games with the current code, all 119 draws were mutual head-on
  collisions, with a median on turn 4.
- **High contempt** makes those games decisive, but the snake then loses
  ground to the default, which takes the trade: 0.38 at contempt 0.9.
- **Contempt from 0 to 0.4** is equally strong against the default and the
  DQN.

0.25 is kept because the heuristic's job is to be a strong, fixed baseline to
measure RL agents against, not to minimise draws. A high mirror draw rate is
expected and does not show a weakness against other opponents. MCTS uses its
own, larger contempt inside the search (`draw_value=-0.5`; see the
`slinky.mcts` docstring); the benchmark score always counts a draw as ½.

## The MCTS

`slinky.mcts` is decoupled-UCT simultaneous-move MCTS:

- each player picks its own move with UCB1 over its own statistics, and the
  tree's children are joint moves;
- leaves are scored with the heuristic's evaluation instead of rollouts;
- the rules are exact, with no food spawns inside the tree;
- the final move is the most visited one.

The `slinky.mcts` module docstring gives every default with the measurement
behind it and the cost per simulation.
[`docs/research/simultaneous_move_mcts.md`](../docs/research/simultaneous_move_mcts.md)
surveys the literature it draws on (Tron, Goofspiel, earlier Battlesnake MCTS).
