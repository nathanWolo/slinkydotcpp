"""Environment API semantics: reset, step, rewards, termination, autoreset."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.observations import obs_shape
from slinky.types import LEFT, RIGHT, UP, GameConfig, Ruleset


def duel_state(config, snakes, food=(), turn=5):
    d = {
        "width": config.width,
        "height": config.height,
        "turn": turn,
        "snakes": [{"id": f"s{i}", "body": b, "health": 90} for i, b in enumerate(snakes)],
        "food": list(food),
        "hazards": [],
    }
    return state_from_engine(d, config)


def test_reset_shapes_and_dtypes():
    config = GameConfig()
    env = BattlesnakeEnv(config)
    state, ts = jax.jit(env.reset)(jax.random.key(0))
    assert ts.obs.shape == (2, *obs_shape(config))
    assert ts.reward.shape == (2,) and ts.reward.dtype == jnp.float32
    assert ts.action_mask.shape == (2, 4) and ts.action_mask.all()
    assert bool(ts.alive.all()) and not bool(ts.done)
    assert int(state.turn) == 0
    np.testing.assert_array_equal(state.length, [3, 3])
    np.testing.assert_array_equal(state.health, [100, 100])


def test_win_loss_rewards():
    config = GameConfig()
    env = BattlesnakeEnv(config, obs=None)
    state = duel_state(config, [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]])
    nxt, ts = env.step(jax.random.key(0), state, jnp.array([LEFT, UP]))
    assert bool(ts.done) and not bool(ts.truncated)
    np.testing.assert_array_equal(ts.reward, [-1.0, 1.0])
    assert int(nxt.turn) == 6


def test_draw_rewards():
    config = GameConfig()
    env = BattlesnakeEnv(config, obs=None)
    state = duel_state(config, [[[4, 5], [3, 5], [2, 5]], [[6, 5], [7, 5], [8, 5]]])
    _, ts = env.step(jax.random.key(0), state, jnp.array([RIGHT, LEFT]))
    assert bool(ts.done)
    np.testing.assert_array_equal(ts.reward, [0.0, 0.0])


def test_multiplayer_rewards():
    config = GameConfig(num_snakes=3)
    env = BattlesnakeEnv(config, obs=None)
    state = duel_state(
        config,
        [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]], [[5, 1], [5, 2], [5, 3]]],
    )
    state, ts = env.step(jax.random.key(0), state, jnp.array([LEFT, UP, RIGHT]))
    np.testing.assert_array_equal(ts.reward, [-1.0, 0.0, 0.0])
    assert not bool(ts.done)
    state, ts = env.step(jax.random.key(1), state, jnp.array([UP, UP, LEFT]))
    np.testing.assert_array_equal(ts.reward, [0.0, 1.0, -1.0])
    assert bool(ts.done)


def test_step_on_done_state_is_noop():
    config = GameConfig()
    env = BattlesnakeEnv(config, obs=None)
    state = duel_state(config, [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]])
    done_state, _ = env.step(jax.random.key(0), state, jnp.array([LEFT, UP]))
    again, ts = env.step(jax.random.key(1), done_state, jnp.array([UP, UP]))
    for a, b in zip(jax.tree.leaves(done_state), jax.tree.leaves(again), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ts.reward, [0.0, 0.0])
    assert bool(ts.done) and not bool(ts.truncated)


def test_truncation():
    config = GameConfig(max_turns=3)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    for t in range(3):
        actions = jnp.argmax(env.action_mask(state), axis=1)
        state, ts = env.step(jax.random.key(t), state, actions)
    assert bool(ts.done) and bool(ts.truncated)
    np.testing.assert_array_equal(ts.reward, [0.0, 0.0])
    assert bool(ts.alive.all())


def test_solo_game_ends_when_snake_dies():
    config = GameConfig(width=7, height=7, num_snakes=1)
    env = BattlesnakeEnv(config, obs=None)
    state = duel_state(config, [[[0, 3], [1, 3], [2, 3]]])
    assert not bool(state.done)
    _, ts = env.step(jax.random.key(0), state, jnp.array([LEFT]))
    assert bool(ts.done)
    np.testing.assert_array_equal(ts.reward, [-1.0])


def test_autoreset():
    config = GameConfig()
    env = BattlesnakeEnv(config)
    state = duel_state(config, [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]])
    nxt, ts = jax.jit(env.step_autoreset)(jax.random.key(0), state, jnp.array([LEFT, UP]))
    # The transition's outcome is reported, but the returned state is a new game.
    assert bool(ts.done)
    np.testing.assert_array_equal(ts.reward, [-1.0, 1.0])
    assert int(nxt.turn) == 0 and not bool(nxt.done)
    assert bool(ts.alive.all()) and bool(ts.action_mask.all())
    fresh_obs = env.observe(nxt)
    np.testing.assert_array_equal(ts.obs, fresh_obs)


def test_vmap_scan_rollout_is_zero_sum():
    config = GameConfig()
    env = BattlesnakeEnv(config, obs=None)

    def rollout(key):
        def body(state, k):
            k_a, k_s = jax.random.split(k)
            logits = jnp.where(env.action_mask(state), 0.0, -1e9)
            actions = jax.random.categorical(k_a, logits)
            state, ts = env.step_autoreset(k_s, state, actions)
            return state, ts.reward

        state, _ = env.reset(key)
        _, rewards = jax.lax.scan(body, state, jax.random.split(key, 300))
        return rewards

    rewards = jax.jit(jax.vmap(rollout))(jax.random.split(jax.random.key(3), 32))
    assert rewards.shape == (32, 300, 2)
    np.testing.assert_allclose(rewards.sum(-1), 0.0)  # 1v1 rewards are zero-sum
    assert np.abs(rewards).sum() > 0


def test_constrictor_reset_has_no_food():
    env = BattlesnakeEnv(GameConfig(ruleset=Ruleset.CONSTRICTOR), obs=None)
    state, _ = env.reset(jax.random.key(0))
    assert not bool(state.food.any())
