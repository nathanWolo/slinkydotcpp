"""Tests for slinky.policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy, random_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig, State, empty_state

POLICIES = [random_policy, random_legal_policy]
CONFIGS = [
    GameConfig(),
    GameConfig(num_snakes=4),
    GameConfig(num_snakes=1),
    GameConfig(ruleset="wrapped"),
    GameConfig(ruleset="constrictor"),
]


def make_env(config: GameConfig) -> BattlesnakeEnv:
    return BattlesnakeEnv(config, obs=None)


def rollout_states(env: BattlesnakeEnv, batch: int = 32, turns: int = 60, seed: int = 0) -> State:
    """States visited by ``batch`` autoreset games under random legal play: leaves [T*B, ...]."""
    k_reset, k_play = jax.random.split(jax.random.key(seed))
    states, _ = jax.vmap(env.reset)(jax.random.split(k_reset, batch))

    def step(states, key):
        k_pol, k_step = jax.random.split(key)
        actions = jax.vmap(lambda k, s: random_legal_policy(k, s, env))(
            jax.random.split(k_pol, batch), states
        )
        new_states, _ = jax.vmap(env.step_autoreset)(
            jax.random.split(k_step, batch), states, actions
        )
        return new_states, states

    _, visited = jax.lax.scan(step, states, jax.random.split(k_play, turns))
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), visited)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("config", CONFIGS)
def test_shape_dtype_range(policy, config):
    env = make_env(config)
    state, _ = env.reset(jax.random.key(0))
    actions = policy(jax.random.key(1), state, env)
    assert actions.shape == (config.num_snakes,)
    assert actions.dtype == jnp.int32
    assert jnp.all((actions >= 0) & (actions < NUM_ACTIONS))


@pytest.mark.parametrize("policy", POLICIES)
def test_deterministic_given_key(policy):
    env = make_env(GameConfig(num_snakes=4))
    state, _ = env.reset(jax.random.key(0))
    a = policy(jax.random.key(7), state, env)
    b = policy(jax.random.key(7), state, env)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("policy", POLICIES)
def test_jit_vmap(policy):
    env = make_env(GameConfig(num_snakes=4))
    batch = 16
    keys = jax.random.split(jax.random.key(0), batch)
    states, _ = jax.vmap(env.reset)(keys)
    batched = jax.jit(jax.vmap(lambda k, s: policy(k, s, env)))
    actions = batched(keys, states)
    assert actions.shape == (batch, 4)
    assert actions.dtype == jnp.int32
    assert jnp.all((actions >= 0) & (actions < NUM_ACTIONS))
    # Each game gets its own key: they should not all pick the same moves.
    assert len({tuple(a) for a in np.asarray(actions)}) > 1
    # jit+vmap agrees with the unbatched policy.
    single = policy(keys[3], jax.tree.map(lambda x: x[3], states), env)
    np.testing.assert_array_equal(actions[3], single)


@pytest.mark.parametrize("policy", POLICIES)
def test_jit_scan(policy):
    env = make_env(GameConfig())

    @jax.jit
    def play(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = env.reset(k_reset)

        def body(state, key):
            k_pol, k_step = jax.random.split(key)
            state, ts = env.step_autoreset(k_step, state, policy(k_pol, state, env))
            return state, ts.done

        return jax.lax.scan(body, state, jax.random.split(k_play, 200))[1]

    assert play(jax.random.key(0)).any()  # games end and restart without errors


@pytest.mark.parametrize("policy", POLICIES)
def test_uses_all_actions(policy):
    env = make_env(GameConfig())
    state, _ = env.reset(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 256)
    actions = jax.vmap(lambda k: policy(k, state, env))(keys)  # [256, N]
    # At reset every move from the start position is legal, so both policies are uniform.
    for i in range(env.num_agents):
        counts = np.bincount(np.asarray(actions[:, i]), minlength=NUM_ACTIONS)
        assert (counts > 0).all(), counts


@pytest.mark.parametrize(
    "config",
    [
        GameConfig(),
        GameConfig(num_snakes=4),
        GameConfig(ruleset="wrapped"),
        GameConfig(ruleset="constrictor"),
    ],
)
def test_legal_policy_respects_mask(config):
    env = make_env(config)
    states = rollout_states(env)
    mask = np.asarray(jax.vmap(env.action_mask)(states))  # [S, N, 4]
    keys = jax.random.split(jax.random.key(3), mask.shape[0])
    actions = np.asarray(jax.vmap(lambda k, s: random_legal_policy(k, s, env))(keys, states))

    has_legal = mask.any(axis=-1)  # [S, N]
    chosen_legal = np.take_along_axis(mask, actions[..., None], axis=-1)[..., 0]
    assert chosen_legal[has_legal].all()
    # The check is not vacuous: the rollout contains partially masked rows.
    assert (has_legal & ~mask.all(axis=-1)).any()
    # Rows with nothing legal (dead or trapped snakes) still get a valid action.
    assert ((actions >= 0) & (actions < NUM_ACTIONS)).all()


def _boxed_in_state() -> State:
    """5x5 duel. Snake 0 sits in the corner (0, 0) with both exits blocked by snake 1.

    Snake 1 is at (1, 1) with its own body on its left and below, so only UP and
    RIGHT are legal for it.
    """
    config = GameConfig(width=5, height=5)
    state = empty_state(config)
    body = np.zeros((2, 5, 5), np.int16)
    body[0, 0, 0] = 3
    # Snake 1, length 4, head first: (1, 1), (1, 0), (0, 1), (0, 2). Countdowns 4, 3, 2, 1.
    for idx, (x, y) in enumerate([(1, 1), (1, 0), (0, 1), (0, 2)]):
        body[1, y, x] = 4 - idx
    return state._replace(
        body=jnp.asarray(body),
        head=jnp.array([[0, 0], [1, 1]], jnp.int32),
        length=jnp.array([3, 4], jnp.int32),
        health=jnp.full((2,), 100, jnp.int32),
        alive=jnp.ones((2,), bool),
    )


def test_legal_policy_restricts_to_legal_and_falls_back_when_trapped():
    env = BattlesnakeEnv(GameConfig(width=5, height=5), obs=None)
    state = _boxed_in_state()
    mask = np.asarray(env.action_mask(state))
    assert mask[0].all()  # trapped: the mask allows everything
    assert set(np.flatnonzero(mask[1])) == {UP, RIGHT}

    keys = jax.random.split(jax.random.key(0), 256)
    actions = np.asarray(jax.vmap(lambda k: random_legal_policy(k, state, env))(keys))
    assert set(actions[:, 1]) == {UP, RIGHT}  # only legal moves, and both of them
    assert set(actions[:, 0]) == {UP, DOWN, LEFT, RIGHT}  # trapped: uniform over all 4


def test_legal_policy_dead_snake_gets_valid_action():
    env = BattlesnakeEnv(GameConfig(width=5, height=5), obs=None)
    state = _boxed_in_state()._replace(alive=jnp.array([False, True]))
    actions = random_legal_policy(jax.random.key(0), state, env)
    assert actions.shape == (2,)
    assert ((actions >= 0) & (actions < NUM_ACTIONS)).all()
