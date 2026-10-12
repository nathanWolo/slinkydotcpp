"""Test helpers shared by several test files: hand-built boards, rollouts and engine comparison."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.engine_json import state_from_engine, state_to_engine
from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy
from slinky.rules import rules_step
from slinky.types import (
    ACTION_NAMES,
    GameConfig,
    RoyaleState,
    State,
    royale_num_sides,
)

try:
    from oracle import engine_royale_sides
except ImportError:
    from tests.oracle import engine_royale_sides

DUEL = GameConfig()
_MOVE_IDS = {name: i for i, name in enumerate(ACTION_NAMES)}


def board(snakes, config=DUEL, food=(), health=None, causes=None, turn=10) -> State:
    """A state from head-first body lists, as in the engine's JSON."""
    health = health or [90] * len(snakes)
    causes = causes or [("", 0)] * len(snakes)
    d = {
        "width": config.width,
        "height": config.height,
        "turn": turn,
        "snakes": [
            {
                "id": f"s{i}",
                "body": [list(c) for c in b],
                "health": health[i],
                "eliminated_cause": causes[i][0],
                "eliminated_on_turn": causes[i][1],
            }
            for i, b in enumerate(snakes)
        ],
        "food": [list(f) for f in food],
        "hazards": [],
    }
    return state_from_engine(d, config)


def both_seats(snakes, **kw):
    """The position with the hero as snake 0, and again as snake 1."""
    health = kw.pop("health", [90, 90])
    return [
        (board(snakes, health=health, **kw), 0),
        (board(snakes[::-1], health=health[::-1], **kw), 1),
    ]


@functools.cache
def env_for(config: GameConfig) -> BattlesnakeEnv:
    return BattlesnakeEnv(config, obs=None)


@functools.cache
def rollout_states(config: GameConfig, batch: int = 16, turns: int = 30, seed: int = 0) -> State:
    """A batch of mid-game states from random legal play (autoreset); cached, as it compiles."""
    env = env_for(config)
    k_reset, k_play = jax.random.split(jax.random.key(seed))
    states, _ = jax.vmap(env.reset)(jax.random.split(k_reset, batch))

    def step(states, key):
        k_pol, k_step = jax.random.split(key)
        acts = jax.vmap(lambda k, s: random_legal_policy(k, s, env))(
            jax.random.split(k_pol, batch), states
        )
        states, _ = jax.vmap(env.step_autoreset)(jax.random.split(k_step, batch), states, acts)
        return states, None

    states, _ = jax.jit(lambda s, k: jax.lax.scan(step, s, jax.random.split(k, turns)))(
        states, k_play
    )
    return states


# --- Comparison with the engine's output (test_parity.py, test_royale.py) ---------


def engine_sides(oracle, seed: int, config: GameConfig) -> RoyaleState:
    """The engine's royale shrink sides for a game seed, as a ``map_state``."""
    sides = engine_royale_sides(oracle, seed, royale_num_sides(config), config.width, config.height)
    return RoyaleState(jnp.asarray(sides, jnp.int8))


def stack_states(states):
    return jax.tree.map(lambda *xs: np.stack(xs), *states)


@functools.cache
def batched_step(config: GameConfig):
    return jax.jit(jax.vmap(lambda s, a: rules_step(s, a, config)))


def moves_to_ids(moves):
    return np.array([_MOVE_IDS.get(m, -1) for m in moves], np.int32)


def compare_states(ours, theirs, config: GameConfig, context) -> None:
    """Assert two stacked States agree on everything the engine defines."""
    o = jax.device_get(ours)
    t = jax.device_get(theirs)
    alive = t.alive
    checks = {
        "alive": (o.alive, t.alive),
        "elim_cause": (o.elim_cause, t.elim_cause),
        "elim_turn": (o.elim_turn, t.elim_turn),
        "health": (o.health, t.health),
        "head": (o.head, t.head),
        "food": (o.food, t.food),
        "hazard": (o.hazard, t.hazard),
        "length[alive]": (np.where(alive, o.length, 0), np.where(alive, t.length, 0)),
        "body[alive]": (o.body * alive[..., None, None], t.body * alive[..., None, None]),
        "last_move[alive]": (np.where(alive, o.last_move, 0), np.where(alive, t.last_move, 0)),
    }
    for name, (a, b) in checks.items():
        bad = np.nonzero(np.any((a != b).reshape(a.shape[0], -1), axis=1))[0]
        if len(bad):
            i = int(bad[0])
            ours_i = jax.tree.map(lambda x, i=i: x[i], ours)
            pytest.fail(
                f"{name} differs in {len(bad)} transition(s); first: {context(i)}\n"
                f"ours:   {state_to_engine(ours_i, config)}\n"
                f"engine: {context(i, engine=True)}"
            )
