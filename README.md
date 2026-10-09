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
uv venv && uv pip install -e . --group dev   # or: pip install -e .
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
| `GameConfig` | Static settings: `width`, `height`, `num_snakes`, `ruleset` (`standard`, `solo`, `wrapped`, `constrictor`, `wrapped_constrictor`), `map` (`standard`, `empty`), `food_spawn_chance`, `minimum_food`, `hazard_damage_per_turn`, `max_turns`. Defaults are the official 1v1 setup. |
| `env.reset(key)` | Returns `(State, TimeStep)` at turn 0, with snakes placed and starting food. |
| `env.step(key, state, actions)` | `actions` is `int[N]`: `0=up, 1=down, 2=left, 3=right`. Any other value is an invalid move, and the snake repeats its last move, as in the engine. |
| `env.step_autoreset(...)` | Same as `step`, but a finished game is replaced by a new one. |
| `TimeStep` | `obs[N, ...]`, `reward[N]`, `done`, `truncated`, `alive[N]`, `action_mask[N, 4]`. |

- **Rewards** (default `win_loss_reward`):
  - −1 on the turn a snake is eliminated;
  - +1 to the last snake standing;
  - 0 for snakes that die on the final turn when nobody survives (a draw).
  - In solo games, dying is simply −1.
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
- **Action mask:** marks moves that don't certainly hit a wall or a body next
  turn. It applies the tail rule exactly: a tail square is open unless that
  tail is stacked. It does not consider hazards, starvation or
  head-to-heads, since those depend on other snakes' choices.

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
    games, 7×7 to 19×19 boards, wrapped, constrictor, solo, the royale map,
    and the hazard-pits, sinkholes and snail-mode maps.
  - Every transition must be reproduced exactly: bodies, health, elimination
    causes and turns, food and hazards.
- **Our own games checked by the engine**
  (`test_our_rollouts_match_engine`): states from our rollouts are sent to
  the engine, which must agree with our next state.
- **Spawning and setup:** engine food spawns must land on cells our spawn
  mask allows, at the engine's rate. Starting positions and starting food
  must follow the engine's rules, in both directions: engine states fit our
  rules, and our states are ones the engine could produce.

`tests/test_rules.py` also has hand-written edge cases with explicit
expected outcomes, each confirmed against the engine.

Food positions are random, so games can't be replayed turn for turn. The
random choices (start positions, food) follow the engine's *distributions*,
not Go's random-number stream.

## Performance

Run `python benchmarks/throughput.py` to measure steps per second (a step is
one game advancing one turn) with a random policy that avoids certain death.

BENCHMARK_TABLE

## Roadmap

1. **Core** (done): standard, duel, solo, wrapped and constrictor rules;
   standard and empty maps; observations; engine parity tests; benchmark.
2. **More modes:**
   - royale, hazard maps and healing pools;
   - official API JSON (`/move` request) conversion, so a trained policy can
     play on the real Battlesnake servers.
3. **RL baselines:**
   - independent PPO with self-play;
   - population or league training;
   - search-based agents (MCTS/AlphaZero via `mctx`, adapted to
     simultaneous moves).

## Layout

```
src/slinky/       types, rules (turn pipeline), maps, env, observations, render,
                  engine_json, policies
tests/            unit, edge-case and engine-parity tests
tools/oracle/     Go test oracle around the official rules engine (test-only)
benchmarks/       throughput benchmark
docs/battlesnake/ official rules/API docs (MIT) + ENGINE_RULES.md
legacy/           the original C++ prototype
```
