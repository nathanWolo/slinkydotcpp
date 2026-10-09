# Baselines

Single-file reinforcement-learning baselines for the 1v1 duel (11×11,
standard rules), trained by self-play on slinky.

```bash
uv pip install -e ".[rl]"            # adds optax
python baselines/dqn.py              # default run, ~30 min on a 4-core CPU
python baselines/dqn.py --help       # every hyperparameter is a flag
python baselines/dqn.py --eval-only runs/dqn-<timestamp> --eval-games 5000
python baselines/dqn.py --resume runs/dqn-<timestamp>   # continue an interrupted run
```

Each run writes to `runs/` (gitignored):

- `config.json`;
- `metrics.jsonl`, one record per chunk of 16k env steps plus the evaluations;
- `params.npz`, the latest network;
- `runner.npz`, the full training state used by `--resume`.

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
  this benchmark is now saturated. More informative comparisons would be:
  - heuristic snakes (food-seeking, flood-fill space control);
  - past checkpoints and other algorithms, via `play_match` or Elo/TrueSkill
    over a population;
  - PPO self-play on the same setup.
