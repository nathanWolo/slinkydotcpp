"""Distributions of the map's random choices (setup and food spawning)."""

from __future__ import annotations

import collections
import math

import jax
import jax.numpy as jnp
import numpy as np

from slinky.env import BattlesnakeEnv
from slinky.maps import spawn_food_standard, spawn_mask
from slinky.types import GameConfig


def _within(count: int, n: int, p: float, sigmas: float = 5.0) -> bool:
    return abs(count - n * p) < sigmas * math.sqrt(n * p * (1 - p))


def test_fixed_start_distribution():
    config = GameConfig()
    env = BattlesnakeEnv(config, obs=None)
    n = 20_000
    states, _ = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), n))
    heads = np.asarray(states.head)
    corners = {(1, 1), (1, 9), (9, 1), (9, 9)}
    is_corner = np.array([tuple(h) in corners for h in heads[:, 0]])
    assert _within(int(is_corner.sum()), n, 0.5)
    # Both snakes come from the same group, on distinct squares.
    assert all((tuple(a) in corners) == (tuple(b) in corners) for a, b in heads)
    assert not np.any(np.all(heads[:, 0] == heads[:, 1], axis=-1))
    # Each of the 8 squares is snake 0's start 1/8 of the time.
    counts = collections.Counter(map(tuple, heads[:, 0]))
    assert len(counts) == 8 and all(_within(c, n, 1 / 8) for c in counts.values())
    # Snake 0 on (1, 1): its food is (0, 2) or (2, 0), 50/50.
    food = np.asarray(states.food)
    at = np.all(heads[:, 0] == [1, 1], axis=-1)
    left = food[at, 2, 0]  # cell (0, 2)
    assert np.all(left ^ food[at, 0, 2]) and _within(int(left.sum()), int(at.sum()), 0.5)


def test_spawn_is_uniform_over_valid_cells():
    config = GameConfig(minimum_food=1)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    state = state._replace(food=jnp.zeros_like(state.food))  # force a top-up spawn
    valid = np.asarray(spawn_mask(state, config))
    n = 50_000
    keys = jax.random.split(jax.random.key(1), n)
    food = np.asarray(jax.jit(jax.vmap(lambda k: spawn_food_standard(k, state, config).food))(keys))
    assert np.all(food.sum(axis=(1, 2)) == 1)
    counts = food.sum(axis=0)
    assert not counts[~valid].any()
    k = int(valid.sum())
    assert all(_within(int(c), n, 1 / k) for c in counts[valid])


def test_multiple_spawns_are_distinct():
    config = GameConfig(minimum_food=5)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    state = state._replace(food=jnp.zeros_like(state.food))
    keys = jax.random.split(jax.random.key(2), 1000)
    food = np.asarray(jax.jit(jax.vmap(lambda k: spawn_food_standard(k, state, config).food))(keys))
    assert np.all(food.sum(axis=(1, 2)) == 5)
    assert not food[:, ~np.asarray(spawn_mask(state, config))].any()


def test_full_board_spawns_nothing():
    config = GameConfig(width=7, height=7, minimum_food=3)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    state = state._replace(food=jnp.ones_like(state.food).at[0, 0].set(False))
    full = state._replace(food=state.food | ~spawn_mask(state._replace(food=state.food), config))
    out = spawn_food_standard(jax.random.key(3), full, config)
    np.testing.assert_array_equal(out.food, full.food)
