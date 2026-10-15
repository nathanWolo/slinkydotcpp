# slinky

A fast [Battlesnake](https://play.battlesnake.com) environment in JAX that
matches the official rules exactly, for benchmarking reinforcement learning
algorithms.

- **Exact rules.** Turn resolution is checked against the
  [official rules engine](https://github.com/BattlesnakeOfficial/rules) on
  tens of thousands of transitions. This includes every elimination cause,
  head-to-heads between 3 or more snakes, the tail rule and stacked hazards.
- **Pure functions.** `reset` and `step` are pure, so you can `jit` them,
  `vmap` them over thousands of games and `scan` them over turns, on CPU,
  GPU or TPU.
- **Multi-agent.** All snakes move at once; every agent gets its own
  observation, reward and action mask.
- **Extensible.** Rulesets (standard, solo, wrapped, constrictor) and maps
  (start positions, food, hazards) are separate pieces, as in the official
  engine.

## Quickstart

```bash
uv venv && uv pip install -e . --group dev   # or: pip install -e ".[rl]"
```

```python
import jax
from slinky.env import BattlesnakeEnv
from slinky.types import GameConfig
from slinky.policies import random_legal_policy
from slinky.render import render

env = BattlesnakeEnv(GameConfig())                # 11x11 1v1 ("duel"), standard rules
key = jax.random.key(0)
state, ts = env.reset(key)
print(ts.obs.shape)                               # (2, 21, 21, 13): egocentric, per agent

step = jax.jit(env.step)
while not state.done:
    key, k_act, k_step = jax.random.split(key, 3)
    actions = random_legal_policy(k_act, state, env)  # int32[2]
    state, ts = step(k_step, state, actions)
print(render(state))
```

To run many games in parallel, `vmap` everything:

```python
batch_reset = jax.jit(jax.vmap(env.reset))
batch_step = jax.jit(jax.vmap(env.step_autoreset))   # finished games restart automatically
states, ts = batch_reset(jax.random.split(key, 4096))
```

## The API

| | |
|---|---|
| `GameConfig` | Static settings: `width`, `height`, `num_snakes`, `ruleset` (`standard`, `solo`, `wrapped`, `constrictor`, `wrapped_constrictor`, `royale`), `map` (`standard`, `empty`, `royale`), `food_spawn_chance`, `minimum_food`, `hazard_damage_per_turn`, `shrink_every_n_turns` (royale), `max_turns`. Defaults are the official 1v1 setup. |
| `env.reset(key)` | Returns `(State, TimeStep)` at turn 0, with snakes placed and starting food. |
| `env.step(key, state, actions)` | `actions` is `int[N]`: `0=up, 1=down, 2=left, 3=right`. Any other value is an invalid move, and the snake repeats its last move, as in the engine. |
| `env.step_autoreset(...)` | Same as `step`, but a finished game is replaced by a new one. `ts.final_obs` keeps the observation of the state the game ended in, for bootstrapping values on truncation. |
| `TimeStep` | `obs[N, ...]`, `reward[N]`, `done`, `truncated`, `alive[N]`, `action_mask[N, 4]` (and `final_obs` from `step_autoreset`). |

- **Rewards** (default `win_loss_reward`):
  - −1 on the turn a snake is eliminated;
  - +1 to the last snake standing;
  - 0 for snakes that die on the final turn when nobody survives (a draw).
  - In solo games, dying is simply −1.
  - `slinky.env.placement_reward` is a zero-sum finishing-order alternative for games
    with more than two snakes. A snake gains 1/(N−1) for each snake that dies while
    it lives and loses 1/(N−1) for each snake still alive when it dies, so a
    4-snake game pays +1, +⅓, −⅓, −1 by place (ties averaged). In a duel it equals
    `win_loss_reward`.
  - Pass `reward_fn=` to use your own.
- **Observations** (`slinky.observations`):
  - **Egocentric** (the default): the board is centred on the agent's head
    and padded, giving `[2H-1, 2W-1, 13]`. Wrapped boards are rolled
    instead, giving `[H, W, 13]`.
  - **Allocentric:** `[H, W, 13]`.
  - The 13 channels are: board mask, food, hazards, own head, own body, own
    body countdown, opponent heads (split into longer-or-equal and shorter),
    opponent bodies, opponent body countdown, opponent health, own health and
    own length.
  - Pass `obs=` a function to use your own, or `None` to skip observations.
- **Royale:** `GameConfig(num_snakes=4, ruleset="royale", map="royale")` is the
  official 4-snake royale mode.
  - Every `shrink_every_n_turns` (default 25) turns the safe zone loses one
    edge row or column, on a random side. Everything outside it is hazard.
  - The sides of all shrinks are drawn at reset and kept in
    `state.map_state` (`RoyaleState`), so the hazards at any turn are a pure
    function of `(sides, turn)`: `slinky.rules.royale_hazards`.
  - The royale ruleset works with any map, as in the engine: `env.reset`
    always draws the sides. The standard ruleset on the royale map gives the
    same game.
  - `rules_step` on its own (search) needs a `RoyaleState` in `map_state`
    under the royale ruleset, and raises `ValueError` without one. Under the
    standard ruleset on the royale map, `rules_step` does not move the zone
    (the map does), so code that only calls `rules_step` keeps the root's
    hazards in that combination: `mcts.py` with its default `spawn_food=False`
    (so AlphaZero's search too) and the heuristic's duel search. With
    `spawn_food=True` the search steps through `env.step` and the zone moves.
  - The `az-*` agents can play royale (their search sees the true future
    zone, see below). Training AlphaZero with `augment=True` cannot: the board
    symmetries refuse states that carry a `map_state`.
  - The shrink sides follow the engine's distribution (iid uniform), not its
    random stream. (The engine's first side is slightly tied to the turn-0
    food roll; see `docs/battlesnake/ENGINE_RULES.md`.)
  - `map_state` holds the whole game's sides, so it shows the future zone.
    Observations and the heuristic never read it, but a search that simulates
    from a state would see the true future shrinks; redraw `sides[turn // N:]`
    at the search root if royale search is ever used.
- **Action mask:** masks out a move only if it certainly hits a wall or a
  body next turn, whatever the other snakes do.
  - It applies the tail rule exactly: a tail square is open unless that tail
    is stacked.
  - It ignores the bodies of snakes that are certain to starve first.
  - It does not consider hazards, your own starvation or head-to-heads.
  - **Rows are never all False.** Dead snakes, and snakes with no move that
    can survive, get all-True rows, so masked softmaxes never produce NaNs.

## Evaluating policies

`slinky.evaluate.play_match` plays two policies against each other over
thousands of games at once and reports wins, draws and losses from the
first policy's point of view, with a 95% confidence interval on the score
(win = 1, draw = ½). Seats are rotated across games, so neither policy gets
the better starting seat more often.

```python
from slinky.evaluate import greedy_from_q, play_match, random_legal

result = play_match(env, greedy_from_q(q_fn), random_legal(env), key, num_games=1000)
print(result.wins, result.draws, result.losses, result.score, "+/-", result.score_ci95)
```

A policy is a function `(key, state, timestep) -> int32[N]` for one game.
`random_legal(env)` is the reference opponent: it picks uniformly among the
moves that don't certainly die next turn.

`batch_size` is the number of game slots simulated at once: when a game
ends, its slot starts the next one, so no slot waits for the longest game
(game lengths are heavy-tailed; this made MCTS evaluations 3-3.5x faster).
Game `g` depends only on `(key, g)`, so the result is the same for every
`batch_size`. `run_match` also reports the loop iterations and slot
utilisation, plays a range of game ids (`first_game`) and takes a progress
callback.

For games with more than two snakes, `play_table` seats any number of
policies according to a line-up and scores each policy by its **share** of
the win (a sole survivor gets 1, snakes tied at the end split 1; a policy's
share is the sum over its seats, so parity is its seats / N) and by its mean
**placement** (parity 2.5 with four snakes), with 95% intervals. It also
counts each policy's wins, ties and losses by team share, and how its seats
were eliminated (`deaths`, by the engine's cause names):

```python
from slinky.evaluate import play_table, random_legal

env4 = BattlesnakeEnv(GameConfig(num_snakes=4))
# Two teams of two (policy_a is any policy for env4); game g rotates the
# line-up, so seats are balanced.
t = play_table(env4, (policy_a, random_legal(env4)), (0, 1, 0, 1), key, num_games=1024)
print(t.share[0], "+/-", t.share_ci95[0], t.placement[0], t.wins[0], t.ties[0], t.losses[0])
```

A game stops as soon as all living snakes belong to one policy (`early_stop`;
this changes no share, placement or count, only `mean_turns`, which then is
the turn at which games were decided), and `per_game=True` also returns each
game's share and placement per policy, so two tables played with the same
`key` can be compared game by game. As with `play_match`, `lineup`,
`num_games` and `max_turns` do not recompile, except that fewer games than
`batch_size` shrink the slots.

`benchmarks/strength.py` plays all pairs of two agent lists and keeps the
results in a resumable JSONL file (`--table` prints markdown tables);
`--workers 4` runs four matchups at once, each in a process pinned to its
own core, which is the fast path on a 4-core machine.

## Baselines

Reference opponents for the 1v1 duel, roughly from weakest to strongest, then
one policy for all four leaderboard modes. Full results and the MCTS sweep are
in [`benchmarks/README.md`](benchmarks/README.md); the multi-mode tables are in
[`baselines/README.md`](baselines/README.md#multi-mode-ppo-ppo_modespy).

- **`random_legal`** (`slinky.evaluate`): uniformly random among the moves
  that don't certainly die next turn.
- **REINFORCE** (`baselines/reinforce.py`, see
  [`baselines/README.md`](baselines/README.md#reinforce-reinforcepy)):
  Monte Carlo policy gradient with a value baseline, on PPO's network and env
  steps (73 minutes on a laptop CPU). It is the weakest learned baseline: it
  scores 0.944 against `random_legal`, 0.329 against the DQN, 0.055 against
  PPO and 0.013 against the heuristic. Its policy collapsed to
  near-determinism after about 1M env steps and stopped improving.
- **DQN** (`baselines/dqn.py`, see [`baselines/README.md`](baselines/README.md)):
  self-play Double DQN with one shared network playing both snakes. After 31
  minutes of training on a 4-core CPU, it scores 0.993 ± 0.002 against
  `random_legal` (99.2% wins over 5,000 games). It plays between MCTS with 4
  and with 16 simulations.
- **PPO** (`baselines/ppo.py`, see [`baselines/README.md`](baselines/README.md)):
  self-play PPO with one shared actor-critic network and masked illegal moves.
  After 67 minutes of training on 3 CPU cores, it beats the DQN 0.868 and
  `random_legal` 0.998, and scores 0.296 against the heuristic. It plays
  between MCTS with 4 and with 16 simulations, closer to 16.
- **Rainbow DQN** (`baselines/rainbow.py`, see
  [`baselines/README.md`](baselines/README.md#rainbow-dqn-rainbowpy)): all six
  Rainbow extensions on the same self-play setup, with a distributional head
  on the game's exact return range [−1, 1]. At the DQN's budget it ties the
  DQN head to head (0.504 ± 0.031). It scores 0.983 against `random_legal` and
  0.084 against the heuristic, against the DQN's 0.992 and 0.047. PPO, trained
  on 4 times the env steps, beats it 0.753 ± 0.027 (and the DQN 0.873 ±
  0.021 in the same harness); see
  [the baselines compared](baselines/README.md#the-baselines-compared). Trained on
  to 4.61M env steps, Rainbow beats its 1.28M self 0.613 and the DQN 0.693,
  and scores 0.158 against the heuristic, but PPO still beats it 0.559 ±
  0.031 ([training on](baselines/README.md#training-on)).
- **Heuristic** (`slinky.heuristic`): a hand-written snake.
  - It uses time-aware flood fills, Voronoi territory and food control.
  - It picks moves with a one-ply simultaneous-move search over the exact
    rules, after safety tiers.
  - It scores 0.988 against `random_legal` and 0.947 against the DQN, at about
    0.05 ms per move.
  - It plays like MCTS with about 32 simulations.
- **MCTS** (`slinky.mcts`): simultaneous-move MCTS (decoupled UCT, with the
  heuristic's evaluation at the leaves).
  - Against the heuristic it scores 0.53 at 32 simulations, 0.78 at 1024 and
    0.85 at 2048. It levels off there, held back by opening head-on draws.
  - At Battlesnake's 500 ms per move (24,000 simulations: one game's move on
    one core, at p90) it scores 0.875 against the heuristic and 0.949 against PPO.
  - It costs about 3.5–6 µs per simulation per CPU core in batches (up to
    4096 simulations), and about 18 µs for one game alone.
- **AlphaZero** (`baselines/az.py`, see
  [`baselines/README.md`](baselines/README.md#alphazero-azpy)): a network
  trained by self-play through `slinky.mcts`'s PUCT search, all in JAX. It
  trained for 9.7 hours on a 4-core CPU (8.39M env steps).
  - The raw network (`az-0`) is the strongest learned policy here. It beats
    PPO 0.581 and scores 0.484 against the heuristic, like MCTS with 32
    simulations.
  - With 64 simulations per move (`az-64`: 7 ms per move with 32 games in
    flight on one core, where MCTS-24000 takes 186 ms) it beats PPO 0.943
    and the heuristic 0.853. It is level with plain MCTS at 2048–4096
    simulations, so the network makes each simulation worth 30–60 plain ones.
    At equal time (about 1,190 plain simulations) it beats plain MCTS,
    scoring 0.588 against MCTS-1024.
  - With 256 simulations (`az-256`, 26 ms per move measured the same way) it
    is level with or slightly ahead of MCTS at 500 ms per move: 0.586
    [0.499, 0.668] over 128 games, at a seventh of the time. So is `az-64`
    with `c_puct` 0.25 at play time instead of the training's 0.5: 0.574
    [0.488, 0.656], at a 25th of the time.
- **Multi-mode PPO** (`baselines/ppo_modes.py`, see
  [`baselines/README.md`](baselines/README.md#multi-mode-ppo-ppo_modespy)):
  one PPO network for all four leaderboard modes (standard, duel, royale and
  constrictor, with 4, 2, 4 and 4 snakes), told which one it is playing. It
  trained for 9.75 hours on 2 CPU cores, with each mode's share being one
  specialist's budget (5.24M env steps per mode).
  - It beats the specialist trained on each mode alone head to head: 0.81 in
    standard, 0.62–0.71 in the duel (three specialist seeds) and 0.71 in
    royale.
  - In constrictor it has no overall gain. It is level with sampled play
    (0.494), and slightly behind close to greedy play (0.42–0.46 with
    occasional random moves).
  - In the duel it scores 0.388 against the heuristic and 0.644 against PPO.
  - Specialists retrained with its 16 minibatches (standard, duel, royale) are
    no stronger head to head, so its lead there is not its extra gradient
    steps.
- **PPO in PufferLib** (`puffer/`, see [`puffer/README.md`](puffer/README.md)):
  the game ported to C as a [PufferLib](https://github.com/PufferAI/PufferLib)
  3.0 environment (checked against slinky and the engine), and ppo.py's network
  trained with PufferLib's PyTorch trainer.
  - In 57 minutes of training on 4 CPU cores (19.7M env steps) it beats PPO
    0.620 and the DQN 0.857, and scores 0.434 against the heuristic. Head to
    head it is level with `az-0` (0.501), which scores 0.484 against the
    heuristic.
  - With ppo.py's settings and budget (5.24M env steps) it trains in 30
    minutes on 4 cores (ppo.py: 62 on 3), but ends weaker: 0.198 against the
    heuristic, 0.396 against PPO.

![MCTS score against random_legal, the DQN, PPO and the heuristic, by simulations per move](benchmarks/results/strength.svg)

## Watching games

`slinky.replay` records games between named agents and writes a
self-contained HTML viewer (no server; open the file in a browser):

```bash
python -m slinky.replay --a heuristic --b dqn --games 8 --html replays/heuristic-vs-dqn.html
python -m slinky.replay --a mcts-256 --b heuristic --games 4 --out replays/mcts.json   # JSON only
python -m slinky.replay --agents heuristic,random_legal,mcts-64,dqn --games 2 --html replays/four.html   # 4 snakes
python -m slinky.replay --game-mode royale --agents heuristic,random_legal,random_legal,random --games 2 --html replays/royale.html
```

- **Agents** come from the registry in `slinky.agents` (shared with
  `benchmarks/strength.py`): `random_legal`, `random`, `heuristic`, `dqn`
  (the checkpoint in `baselines/checkpoints/`) or `dqn:<run dir>`, `ppo`
  (greedy), `ppo-sample` or `ppo:<run dir>`,
  `ppo_modes[:<run dir>][:snapshot=<steps>][:game_mode=<mode>]` (the multi-mode
  PPO, told the mode of the game it plays; `snapshot` plays the run's
  `params-<steps>.npz`), `rainbow` or `rainbow:<run dir>`,
  `reinforce` (greedy), `reinforce-sample` or `reinforce:<run dir>`,
  `az-<simulations>` or `az-<simulations>:<run dir>` (the AlphaZero network
  with PUCT search; `az-0` is the raw network), and `mcts-<simulations>`
  with dash shorthands and/or `:field=value` overrides of `MCTSConfig`, e.g.
  `mcts-256-rm`, `mcts-64-rollout` or `mcts-128:exploration=0.5`. Names are
  shown in a canonical form that lists only non-default settings (an `az`
  name keeps every `:field=value` given, even one equal to its run's).
- **The viewer** shows the board (API coordinates, `(0, 0)` bottom-left),
  each snake's length and health, eliminations with the engine's cause, a
  health-by-turn strip that doubles as the scrubber, the match score and the
  game list. Keys: Space play/pause, ←/→ step, Home/End, `[`/`]` previous or
  next game. Another replay JSON can be opened from the page or dropped on it.
- **The format** (`slinky-replay/1`) is documented in `src/slinky/replay.py`:
  one frame per turn with head-first bodies, health, food and hazards, about
  250 bytes per turn in a duel. `replay.game_from_states` turns a sequence of
  states from your own loop into a game record.

## Playing on play.battlesnake.com

`slinky.server` serves any registry agent over the official Battlesnake API,
so it can play real games (for example on the Duels leaderboard: 11×11, two
snakes, 500 ms per move):

```bash
python -m slinky.server --agent ppo --port 8000   # binds 127.0.0.1; snake at /ppo
python -m slinky.server --agent az-256 --agent best=az-256:runs/az-x   # two snakes
```

- **One URL per snake.** Each `--agent` is served under its own path,
  `/<name>` (`/<name>/move` and so on), so one server can host several
  snakes; `GET /` lists them. The name is the agent's registry name, or
  `name` in `--agent name=agent` (needed when the agent names a run directory).
  Register `<server>/<name>` as the Battlesnake's URL.

- **Who plays.** The trained baselines play only the game they learned, the
  11×11 duel. Other standard games get the heuristic (`--fallback`), compiled
  for that board in the background; other rulesets, and a new board until it
  has compiled, get a simple safe-move picker.
- **Speed.** A duel move takes about 2 ms of compute on a laptop CPU, and
  about 4 ms round trip locally; 16 games' moves arriving together are all
  answered within 60 ms.
- **Reaching it.** Battlesnake's servers call the snake's URL, so a machine
  behind NAT needs a tunnel (e.g. `cloudflared tunnel --url
  http://127.0.0.1:8000`) or a forwarded port. Register that URL as a
  Battlesnake on play.battlesnake.com.

## How it works

Each snake is stored as a **countdown grid**. For snake `i`,
`state.body[i, y, x]` is the number of turns until cell `(x, y)` is vacated.
The head cell holds the snake's length, the tail holds 1, and a stacked tail
(the snake just ate) holds 2.

- **Moving:** subtract 1 from every cell, then write the length at the new
  head.
- **Eating:** add 1 to every cell. This reproduces the engine's duplicated
  tail segment exactly.
- **Collisions:** a lookup in the grid of each snake after it has moved.

So every step is a fixed number of dense array operations, with no
variable-length lists or per-segment loops. Turn resolution follows the
engine's stage order; see
[`docs/battlesnake/ENGINE_RULES.md`](docs/battlesnake/ENGINE_RULES.md).

## Rules fidelity and tests

```bash
.venv/bin/python -m pytest            # ~2-4 minutes; the oracle tests need Go >= 1.22
```

`tools/oracle` is a small Go program that links the official rules engine.
It is only used for testing; see its README. The tests use it in three ways:

- **Engine-generated games**
  (`tests/test_parity.py::test_transitions_match_engine`).
  - Thousands of games are played by the engine: 1v1, 4-, 8- and 12-player
    games, 7×7 to 19×19 boards, wrapped, constrictor, solo, the royale ruleset
    and map, and the hazard-pits, sinkholes and snail-mode maps.
  - Every transition must be reproduced exactly: bodies, health, elimination
    causes and turns, food and hazards.
- **Our own games checked by the engine**
  (`test_our_rollouts_match_engine`): states from our rollouts are sent to
  the engine, which must agree with our next state.
- **Spawning and setup:** engine food spawns must land on cells our spawn
  mask allows, at the engine's rate. Starting positions and starting food
  must follow the engine's rules, in both directions: engine states fit our
  rules, and our states are ones the engine could produce.

Two more test files check the rules directly:

- `tests/test_rules.py` has hand-written edge cases with explicit expected
  outcomes, each confirmed against the engine.
- `tests/test_fuzz_rules.py` fuzzes the rules against the engine with random
  states that games rarely reach: tightly coiled and fully stacked bodies,
  3- and 4-way head-ons with food on the square, stacked hazards with zero,
  negative or very large damage, 1-wide and non-square boards, up to 16
  snakes, and invalid moves.

An independent review ran more than 300k fuzzed and rollout transitions
through the engine and found no differences in the rules.

Food positions are random, so games can't be replayed turn for turn. The
random choices (start positions, food) follow the engine's *distributions*,
not Go's random-number stream.

## Performance

Run `python benchmarks/throughput.py` to measure steps per second (a step is
one game advancing one turn) with a random policy that avoids certain death.

Measured on a 4-core cloud CPU (no GPU) with jax 0.11.2, using a random
policy that avoids certain death and `step_autoreset`. A step is one game
advancing one turn, so multiply by the number of snakes for per-agent steps.

| config | batch | steps/s, no obs | steps/s, egocentric obs |
|---|--:|--:|--:|
| 11×11 duel | 1 | 32k | 18k |
| 11×11 duel | 1,024 | 381k | 99k |
| 11×11 duel | 8,192 | 627k | 69k |
| 11×11, 4 snakes | 8,192 | 420k | 29k |
| 19×19, 4 snakes | 8,192 | 212k | 10k |

Without observations the simulation is cheap: about 2–5 µs per game-step
on CPU. Building float32 observation tensors (a `[21, 21, 13]` image per
agent in a duel) then dominates, which is memory-bound on CPU. Accelerators
should scale much better with batch size.

The C port of the game in [`puffer/`](puffer/README.md#throughput) is faster on
a CPU. On one thread it steps a duel at 407k game-steps/s with `uint8`
observations under `random_legal`. slinky on four cores reaches 111k in a fresh
run of this benchmark (batch 1,024; this table's older run has 99k), so that is
3.7×. Without observations, under uniformly random moves (`--policy random`), it
reaches 4.3M against slinky's 559k (batch 8,192), 7.7×. The outputs are in
`benchmarks/results/puffer_throughput/`.

## Roadmap

1. **Core** (done): standard, duel, solo, wrapped, constrictor and royale
   rules; standard, empty and royale maps; observations; engine parity tests;
   benchmark.
2. **More modes:**
   - hazard maps other than royale, and healing pools (royale is done);
   - official API JSON (`/move` request) conversion, so a trained policy can
     play on the real Battlesnake servers.
3. **Baselines:**
   - self-play DQN and Rainbow DQN, with a live training dashboard (done; see
     `baselines/`);
   - heuristic snake and simultaneous-move MCTS, with a strength-vs-simulations
     benchmark (done; see `benchmarks/README.md`);
   - self-play PPO and REINFORCE (done; see `baselines/`);
   - population or league training;
   - AlphaZero-style search with a learned value and policy, on top of
     `slinky.mcts` (done; see `baselines/` and `benchmarks/README.md`);
   - one policy for all four leaderboard modes, told the mode (done; see
     `baselines/` and `docs/research/multimode_constrictor.md`).
   - the game in C as a PufferLib environment, trained with PufferLib's PPO
     (done; see `puffer/`).

## Layout

```
src/slinky/       types, rules (turn pipeline), maps, env, observations, render,
                  engine_json, policies, evaluate (matches between policies),
                  heuristic (hand-written snake), mcts (simultaneous-move MCTS),
                  agents (named-agent registry), replay (+ viewer.html), modes (the
                  four leaderboard modes and the multi-mode policy's side inputs)
baselines/        RL baselines (self-play DQN, PPO, Rainbow DQN, REINFORCE, AlphaZero and
                  multi-mode PPO), their checkpoints and a live training dashboard (dashboard.py)
puffer/           the game in C as a PufferLib 3.0 environment, ppo.py's network in PyTorch,
                  training with PufferLib's trainer (its own venv: puffer/README.md)
tests/            unit, edge-case and engine-parity tests
tools/oracle/     Go test oracle around the official rules engine (test-only)
benchmarks/       throughput, strength and mode benchmarks, results/ (their data)
docs/research/    literature notes (simultaneous-move MCTS, Battlesnake heuristics) and
                  the multi-mode constrictor investigation
docs/battlesnake/ official rules/API docs (MIT) + ENGINE_RULES.md
legacy/           the original C++ prototype
```
