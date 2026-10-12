"""Tests for slinky.modes and env.placement_reward.

Royale is built separately: tests that need a royale environment are skipped while
``BattlesnakeEnv(mode_config("royale"))`` raises ``NotImplementedError`` (any other error
at construction fails them). Everything that only reads a config (mode_config, mode_of,
the royale features) runs regardless. Drop the guard once royale is merged.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import board, env_for, rollout_states

from slinky.env import BattlesnakeEnv, placement_reward, win_loss_reward
from slinky.modes import (
    DEFAULT_SNAKES,
    FEATURE_NAMES,
    MODE_NAMES,
    NUM_FEATURES,
    TURN_SCALE,
    hide_food,
    mode_config,
    mode_features,
    mode_of,
)
from slinky.observations import CHANNEL_NAMES, NUM_CHANNELS, obs_shape
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, RIGHT, UP, GameConfig, Ruleset, empty_state


def _royale_env_available() -> bool:
    try:
        BattlesnakeEnv(mode_config("royale"), obs=None)
    except NotImplementedError:
        return False
    return True


needs_royale = pytest.mark.skipif(
    not _royale_env_available(), reason="the royale env is not available yet"
)
MODES_WITH_ENV = ["duel", "standard", "constrictor", pytest.param("royale", marks=needs_royale)]


def _stub_state(config: GameConfig, turn: int = 0, alive=None):
    """A state with only ``turn`` and ``alive`` set: all mode_features reads."""
    state = empty_state(config)
    alive = jnp.ones(config.num_snakes, bool) if alive is None else jnp.asarray(alive)
    return state._replace(turn=jnp.int32(turn), alive=alive)


# --- Names and configs -----------------------------------------------------------------


def test_names_and_defaults():
    assert MODE_NAMES == ("standard", "duel", "royale", "constrictor")
    assert DEFAULT_SNAKES == {"standard": 4, "duel": 2, "royale": 4, "constrictor": 4}
    # Read-only: overrides must be built as a new dict, never written into the defaults.
    with pytest.raises(TypeError):
        DEFAULT_SNAKES["duel"] = 3
    assert DEFAULT_SNAKES["duel"] == 2
    assert NUM_FEATURES == len(FEATURE_NAMES) == 7
    # The first four features are the mode one-hot, in mode-id order.
    assert FEATURE_NAMES[:4] == tuple(f"is_{m}" for m in MODE_NAMES)
    assert FEATURE_NAMES[4:] == ("turn", "shrink_phase", "opponents_alive")


def test_mode_config_values():
    assert mode_config("standard") == GameConfig(num_snakes=4)
    assert mode_config("duel") == GameConfig(num_snakes=2) == GameConfig()
    assert mode_config("royale") == GameConfig(ruleset=Ruleset.ROYALE, map="royale", num_snakes=4)
    assert mode_config("constrictor") == GameConfig(ruleset=Ruleset.CONSTRICTOR, num_snakes=4)
    for mode in MODE_NAMES:
        c = mode_config(mode)
        assert (c.width, c.height) == (11, 11)
        assert c.num_snakes == DEFAULT_SNAKES[mode]
        assert c.max_turns is None


def test_mode_config_overrides():
    assert mode_config("standard", 3).num_snakes == 3
    assert mode_config("constrictor", 2).num_snakes == 2
    assert mode_config("royale", num_snakes=3).num_snakes == 3
    for mode in MODE_NAMES:
        assert mode_config(mode, max_turns=1000).max_turns == 1000
    assert mode_config("royale", 2, 500) == GameConfig(
        num_snakes=2, ruleset=Ruleset.ROYALE, map="royale", max_turns=500
    )


@pytest.mark.parametrize(
    ("mode", "n"),
    [("duel", 3), ("duel", 4), ("standard", 2), ("royale", 1), ("constrictor", 1), ("duel", 1)],
)
def test_mode_config_rejects_snake_counts_outside_the_mode(mode, n):
    with pytest.raises(ValueError, match="snakes"):
        mode_config(mode, n)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown game mode"):
        mode_config("arcade")
    with pytest.raises(ValueError, match="unknown game mode"):
        mode_features(_stub_state(GameConfig()), GameConfig(), "arcade")


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_mode_of_inverts_mode_config(mode):
    assert mode_of(mode_config(mode)) == mode
    assert mode_of(mode_config(mode, max_turns=1000)) == mode
    assert mode_of(dataclasses.replace(mode_config(mode), width=7, height=7)) == mode


def test_mode_of_by_snake_count():
    assert mode_of(GameConfig(num_snakes=2)) == "duel"
    assert mode_of(GameConfig(num_snakes=3)) == "standard"
    assert mode_of(GameConfig(num_snakes=8)) == "standard"
    # Royale and constrictor keep their mode at any count.
    assert mode_of(mode_config("constrictor", 2)) == "constrictor"
    assert mode_of(mode_config("royale", 2)) == "royale"


def test_mode_of_royale_ruleset_or_map():
    royale = mode_config("royale")
    # The engine treats the royale ruleset on the standard map like the royale map.
    assert mode_of(dataclasses.replace(royale, map="standard")) == "royale"
    # Hazards come from the map, so the standard ruleset on the royale map is royale play.
    assert mode_of(dataclasses.replace(royale, ruleset=Ruleset.STANDARD)) == "royale"
    assert mode_of(dataclasses.replace(royale, shrink_every_n_turns=5)) == "royale"


# (ruleset, map) pairs that play a mode with two or more snakes; the standard ruleset and map
# is "duel" for 2 snakes and "standard" for 3 or more. Every other pair is not a mode.
_PLAYED = {
    (Ruleset.STANDARD, "standard"): ("duel", "standard"),
    (Ruleset.STANDARD, "royale"): ("royale", "royale"),
    (Ruleset.ROYALE, "standard"): ("royale", "royale"),
    (Ruleset.ROYALE, "royale"): ("royale", "royale"),
    (Ruleset.CONSTRICTOR, "standard"): ("constrictor", "constrictor"),
}


@pytest.mark.parametrize("ruleset", list(Ruleset))
@pytest.mark.parametrize("map_name", ["standard", "royale", "empty"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mode_of_accepts_exactly_the_played_ruleset_and_map_pairs(ruleset, map_name, n):
    # One rule for every ruleset: only the standard and royale maps carry a mode, and the
    # royale ruleset gets no more latitude than constrictor (the empty map is a test map).
    config = GameConfig(num_snakes=n, ruleset=ruleset, map=map_name)
    played = _PLAYED.get((ruleset, map_name))
    if played is None or n < 2:
        with pytest.raises(ValueError, match="no game mode"):
            mode_of(config)
    else:
        assert mode_of(config) == played[0 if n == 2 else 1]


@pytest.mark.parametrize(
    "config",
    [
        GameConfig(num_snakes=1),
        GameConfig(ruleset=Ruleset.SOLO),
        GameConfig(ruleset=Ruleset.WRAPPED),
        GameConfig(ruleset=Ruleset.WRAPPED_CONSTRICTOR, num_snakes=4),
        GameConfig(map="empty"),
        GameConfig(ruleset=Ruleset.CONSTRICTOR, map="royale", num_snakes=4),
        GameConfig(ruleset=Ruleset.CONSTRICTOR, map="empty", num_snakes=4),
        GameConfig(ruleset=Ruleset.ROYALE, map="empty", num_snakes=4),
        GameConfig(ruleset=Ruleset.ROYALE, num_snakes=1),
    ],
)
def test_mode_of_rejects_other_games(config):
    with pytest.raises(ValueError, match="no game mode"):
        mode_of(config)


# --- Features --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_static_features_are_the_mode_one_hot(mode):
    config = mode_config(mode)
    feats = mode_features(_stub_state(config), config)
    assert feats.shape == (config.num_snakes, NUM_FEATURES) and feats.dtype == jnp.float32
    expected = np.zeros(len(MODE_NAMES), np.float32)
    expected[MODE_NAMES.index(mode)] = 1.0
    for row in np.asarray(feats):
        np.testing.assert_array_equal(row[:4], expected)
    # A fresh game: turn 0, nothing shrunk, everyone alive.
    np.testing.assert_array_equal(np.asarray(feats)[:, 4:], [[0.0, 0.0, 1.0]] * config.num_snakes)


def test_turn_feature_is_scaled_and_clamped():
    config = mode_config("standard")
    col = FEATURE_NAMES.index("turn")
    for turn, expected in [
        (0, 0.0),
        (1, 1 / TURN_SCALE),
        (250, 0.25),
        (TURN_SCALE, 1.0),
        (5000, 1.0),
    ]:
        feats = mode_features(_stub_state(config, turn), config)
        np.testing.assert_allclose(np.asarray(feats)[:, col], expected, rtol=1e-6)
    # Every mode carries the turn.
    for mode in MODE_NAMES:
        c = mode_config(mode)
        np.testing.assert_allclose(mode_features(_stub_state(c, 500), c)[:, col], 0.5)


def test_shrink_phase_is_the_royale_cycle_position():
    col = FEATURE_NAMES.index("shrink_phase")
    royale = mode_config("royale")
    for turn, expected in [(0, 0.0), (10, 0.4), (24, 0.96), (25, 0.0), (37, 0.48), (1000, 0.0)]:
        feats = mode_features(_stub_state(royale, turn), royale)
        np.testing.assert_allclose(np.asarray(feats)[:, col], expected, atol=1e-6)
    # It follows the config's shrink interval.
    fast = dataclasses.replace(royale, shrink_every_n_turns=5)
    for turn in range(13):
        feats = mode_features(_stub_state(fast, turn), fast)
        np.testing.assert_allclose(np.asarray(feats)[:, col], (turn % 5) / 5, atol=1e-6)
    # The other modes have no shrink cycle, whatever their turn.
    for mode in ("standard", "duel", "constrictor"):
        c = mode_config(mode)
        assert not np.asarray(mode_features(_stub_state(c, 17), c))[:, col].any()


def test_opponents_alive_follows_deaths():
    config = mode_config("standard")
    env = env_for(config)
    snakes = [
        [(0, 5), (1, 5), (2, 5)],  # runs into the left wall on the first move
        [(3, 8), (2, 8), (1, 8)],  # meets snake 2 head to head on the second
        [(7, 8), (8, 8), (9, 8)],
        [(5, 1), (5, 2), (5, 3)],
    ]
    col = FEATURE_NAMES.index("opponents_alive")
    state = board(snakes, config)
    np.testing.assert_array_equal(mode_features(state, config)[:, col], [1, 1, 1, 1])

    state, _ = env.step(jax.random.key(0), state, jnp.array([LEFT, RIGHT, LEFT, LEFT]))
    assert np.asarray(state.alive).tolist() == [False, True, True, True]
    # Survivors see 2 of their 3 opponents; the dead snake sees all 3 survivors.
    np.testing.assert_allclose(mode_features(state, config)[:, col], [1, 2 / 3, 2 / 3, 2 / 3])

    state, _ = env.step(jax.random.key(1), state, jnp.array([LEFT, RIGHT, LEFT, LEFT]))
    assert np.asarray(state.alive).tolist() == [False, False, False, True]
    np.testing.assert_allclose(mode_features(state, config)[:, col], [1 / 3, 1 / 3, 1 / 3, 0])


def test_opponents_alive_in_a_duel():
    config = mode_config("duel")
    col = FEATURE_NAMES.index("opponents_alive")
    alive = _stub_state(config, alive=[True, False])
    np.testing.assert_array_equal(mode_features(alive, config)[:, col], [0, 1])
    both = _stub_state(config, alive=[True, True])
    np.testing.assert_array_equal(mode_features(both, config)[:, col], [1, 1])
    # A single snake has no opponents to divide by.
    solo = GameConfig(num_snakes=1)
    feats = mode_features(_stub_state(solo), solo, "standard")
    assert np.isfinite(np.asarray(feats)).all() and feats[0, col] == 0


def test_told_mode_overrides_the_one_hot_only():
    config = mode_config("standard")
    state = _stub_state(config, turn=100)
    told = np.asarray(mode_features(state, config, "constrictor"))
    own = np.asarray(mode_features(state, config))
    np.testing.assert_array_equal(told[:, :4], [[0, 0, 0, 1]] * 4)
    np.testing.assert_array_equal(own[:, :4], [[1, 0, 0, 0]] * 4)
    np.testing.assert_array_equal(told[:, 4:], own[:, 4:])
    # Told "royale", the phase is computed from the config's shrink interval.
    royale = np.asarray(mode_features(state, config, "royale"))
    np.testing.assert_allclose(royale[:, 5], (100 % 25) / 25)


def test_features_under_jit_and_vmap():
    config = mode_config("standard")
    states = rollout_states(config, batch=8, turns=20)
    batched = jax.jit(jax.vmap(lambda s: mode_features(s, config)))(states)
    assert batched.shape == (8, 4, NUM_FEATURES)
    for g in range(8):
        one = jax.tree.map(lambda x, g=g: x[g], states)
        np.testing.assert_array_equal(batched[g], mode_features(one, config))
    # The turn is a traced value, not baked in.
    feats = jax.jit(lambda t: mode_features(_stub_state(config, 0)._replace(turn=t), config))(
        jnp.int32(500)
    )
    np.testing.assert_allclose(feats[:, FEATURE_NAMES.index("turn")], 0.5)


@needs_royale
def test_features_on_real_royale_games():
    config = dataclasses.replace(mode_config("royale"), shrink_every_n_turns=5)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    step = jax.jit(env.step_autoreset)
    for t in range(12):
        feats = np.asarray(mode_features(state, config))
        assert feats.shape == (4, NUM_FEATURES)
        np.testing.assert_array_equal(feats[:, :4], [[0, 0, 1, 0]] * 4)
        turn = int(state.turn)
        np.testing.assert_allclose(feats[:, 4], turn / TURN_SCALE, rtol=1e-6)
        np.testing.assert_allclose(feats[:, 5], (turn % 5) / 5, atol=1e-6)
        k_act, k_step = jax.random.split(jax.random.key(t))
        state, _ = step(k_step, state, random_legal_policy(k_act, state, env))


# --- Food hiding -----------------------------------------------------------------------

FOOD = CHANNEL_NAMES.index("food")


@pytest.mark.parametrize(
    "config",
    [
        mode_config("constrictor"),
        mode_config("constrictor", 2),
        GameConfig(ruleset=Ruleset.WRAPPED_CONSTRICTOR, num_snakes=4),
    ],
)
def test_hide_food_zeroes_only_the_food_channel_in_constrictor(config):
    rng = np.random.default_rng(0)
    obs = rng.uniform(0.1, 1.0, (config.num_snakes, *obs_shape(config))).astype(np.float32)
    hidden = np.asarray(hide_food(jnp.asarray(obs), config))
    assert not hidden[..., FOOD].any()
    keep = [c for c in range(NUM_CHANNELS) if c != FOOD]
    np.testing.assert_array_equal(hidden[..., keep], obs[..., keep])
    # Also for a single observation and under jit.
    np.testing.assert_array_equal(jax.jit(lambda o: hide_food(o, config))(obs[0]), hidden[0])


@pytest.mark.filterwarnings("error")  # a float literal scattered into uint8 or bool warns
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.uint8, jnp.int8, jnp.bool_])
def test_hide_food_keeps_the_observation_dtype(dtype):
    config = mode_config("constrictor")
    obs = jnp.ones((config.num_snakes, *obs_shape(config)), dtype)
    hidden = hide_food(obs, config)
    assert hidden.dtype == dtype and hidden.shape == obs.shape
    hidden = np.asarray(hidden.astype(jnp.float32))
    assert not hidden[..., FOOD].any()
    assert hidden[..., [c for c in range(NUM_CHANNELS) if c != FOOD]].all()


@pytest.mark.parametrize(
    "config",
    [
        mode_config("standard"),
        mode_config("duel"),
        mode_config("royale"),
        GameConfig(ruleset=Ruleset.WRAPPED),
        GameConfig(num_snakes=1),
    ],
)
def test_hide_food_leaves_every_other_ruleset_alone(config):
    obs = jnp.ones((config.num_snakes, *obs_shape(config)))
    assert hide_food(obs, config) is obs


def test_hide_food_hides_the_food_a_constrictor_game_respawns():
    config = mode_config("constrictor")
    env = BattlesnakeEnv(config)
    state, _ = env.reset(jax.random.key(0))
    assert not bool(state.food.any())  # reset removes it; the map spawns some every turn after
    step = jax.jit(env.step)
    for t in range(3):
        state, ts = step(jax.random.key(t), state, jnp.argmax(env.action_mask(state), axis=1))
    assert bool(state.food.any()) and np.asarray(ts.obs)[..., FOOD].any()
    hidden = hide_food(ts.obs, config)
    assert not np.asarray(hidden)[..., FOOD].any()


def test_hide_food_follows_the_config_not_the_told_mode():
    # A standard game told "constrictor" keeps its food: hiding reads only the config.
    config = mode_config("standard")
    env = BattlesnakeEnv(config)
    _, ts = env.reset(jax.random.key(0))
    assert np.asarray(ts.obs)[..., FOOD].any()
    assert mode_features(_stub_state(config), config, "constrictor")[0, 3] == 1.0
    np.testing.assert_array_equal(hide_food(ts.obs, config), ts.obs)


# --- Placement reward ------------------------------------------------------------------


def test_placement_reward_in_a_hand_built_four_snake_game():
    config = mode_config("standard")
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    snakes = [
        [(0, 5), (1, 5), (2, 5)],  # hits the wall on turn 1
        [(3, 8), (2, 8), (1, 8)],  # meets snake 2 head to head on turn 2: both die
        [(7, 8), (8, 8), (9, 8)],
        [(5, 1), (5, 2), (5, 3)],  # survives
    ]
    state = board(snakes, config)
    state, ts = env.step(jax.random.key(0), state, jnp.array([LEFT, RIGHT, LEFT, LEFT]))
    first = np.asarray(ts.reward)
    np.testing.assert_allclose(first, [-1, 1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert not bool(ts.done)
    state, ts = env.step(jax.random.key(1), state, jnp.array([LEFT, RIGHT, LEFT, LEFT]))
    second = np.asarray(ts.reward)
    np.testing.assert_allclose(second, [0, -1 / 3, -1 / 3, 2 / 3], atol=1e-6)
    assert bool(ts.done)
    # 1st +1, the tied 2nd/3rd average to 0, last -1.
    np.testing.assert_allclose(first + second, [-1, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(first.sum(), 0, atol=1e-6)
    np.testing.assert_allclose(second.sum(), 0, atol=1e-6)


def test_placement_reward_in_a_three_snake_game_and_after_it_ends():
    config = GameConfig(num_snakes=3)
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    snakes = [[(0, 5), (1, 5), (2, 5)], [(8, 8), (8, 7), (8, 6)], [(5, 1), (5, 2), (5, 3)]]
    state = board(snakes, config)
    state, ts = env.step(jax.random.key(0), state, jnp.array([LEFT, UP, RIGHT]))
    np.testing.assert_allclose(ts.reward, [-1, 0.5, 0.5], atol=1e-6)
    state, ts = env.step(jax.random.key(1), state, jnp.array([UP, UP, LEFT]))
    assert bool(ts.done)
    np.testing.assert_allclose(ts.reward, [0, 0.5, -0.5], atol=1e-6)
    # Stepping the finished game pays nothing.
    _, ts = env.step(jax.random.key(2), state, jnp.array([UP, UP, UP]))
    np.testing.assert_array_equal(ts.reward, 0)


def test_placement_reward_equals_win_loss_in_a_duel():
    config = mode_config("duel")
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    cases = {  # (bodies, actions, expected reward)
        "win": ([[(0, 5), (1, 5), (2, 5)], [(8, 8), (8, 7), (8, 6)]], [LEFT, UP], [-1, 1]),
        "draw": ([[(4, 5), (3, 5), (2, 5)], [(6, 5), (7, 5), (8, 5)]], [RIGHT, LEFT], [0, 0]),
        "loss": ([[(8, 8), (8, 7), (8, 6)], [(0, 5), (1, 5), (2, 5)]], [UP, LEFT], [1, -1]),
    }
    for name, (snakes, actions, expected) in cases.items():
        state = board(snakes, config)
        new, ts = env.step(jax.random.key(0), state, jnp.array(actions))
        assert bool(ts.done), name
        np.testing.assert_array_equal(ts.reward, expected, err_msg=name)
        np.testing.assert_array_equal(ts.reward, win_loss_reward(state, new, config), err_msg=name)


def test_placement_reward_truncation_is_a_tie():
    config = dataclasses.replace(mode_config("standard"), max_turns=3)
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    state, _ = env.reset(jax.random.key(0))
    total = jnp.zeros(4)
    for t in range(3):
        state, ts = env.step(jax.random.key(t), state, jnp.argmax(env.action_mask(state), axis=1))
        total += ts.reward
    assert bool(ts.done) and bool(ts.truncated) and bool(ts.alive.all())
    np.testing.assert_array_equal(total, 0)


def test_placement_reward_truncation_after_a_death_pays_the_survivors_as_tied():
    # Board turn 10, cut off at turn 12: one snake dies on the first step, three survive.
    config = dataclasses.replace(mode_config("standard"), max_turns=12)
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    snakes = [
        [(0, 5), (1, 5), (2, 5)],  # hits the wall on the first step
        [(3, 8), (2, 8), (1, 8)],
        [(7, 8), (8, 8), (9, 8)],
        [(5, 1), (5, 2), (5, 3)],
    ]
    state = board(snakes, config)
    state, first = env.step(jax.random.key(0), state, jnp.array([LEFT, RIGHT, LEFT, LEFT]))
    np.testing.assert_allclose(first.reward, [-1, 1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert not bool(first.done) and not bool(first.truncated)
    state, last = env.step(jax.random.key(1), state, jnp.array([LEFT, UP, UP, DOWN]))
    assert bool(last.done) and bool(last.truncated)
    assert np.asarray(state.alive).tolist() == [False, True, True, True]
    # Everything decided so far is already paid; the three survivors tie for first.
    np.testing.assert_array_equal(last.reward, 0)
    np.testing.assert_allclose(first.reward + last.reward, [-1, 1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_placement_reward_in_solo_games_is_plain_death_penalty():
    config = GameConfig(width=7, height=7, num_snakes=1)
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    state = board([[(0, 3), (1, 3), (2, 3)]], config)
    _, ts = env.step(jax.random.key(0), state, jnp.array([LEFT]))
    assert bool(ts.done)
    np.testing.assert_array_equal(ts.reward, [-1.0])
    _, ts = env.step(jax.random.key(0), state, jnp.array([DOWN]))
    np.testing.assert_array_equal(ts.reward, [0.0])


@functools.cache
def _random_games(config: GameConfig, games: int = 96, turns: int = 300):
    """Random legal play: final states, placement totals, win/loss totals, max |step sum|."""
    env = BattlesnakeEnv(config, obs=None, reward_fn=placement_reward)
    n = config.num_snakes

    def one(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = env.reset(k_reset)

        def body(carry, k):
            state, placement, win_loss, worst = carry
            k_act, k_step = jax.random.split(k)
            new, ts = env.step(k_step, state, random_legal_policy(k_act, state, env))
            wl = jnp.where(state.done, 0.0, win_loss_reward(state, new, config))
            worst = jnp.maximum(worst, jnp.abs(ts.reward.sum()))
            return (new, placement + ts.reward, win_loss + wl, worst), None

        zeros = jnp.zeros(n)
        init = (state, zeros, zeros, jnp.zeros(()))
        (state, placement, win_loss, worst), _ = jax.lax.scan(
            body, init, jax.random.split(k_play, turns)
        )
        return state, placement, win_loss, worst

    out = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(0), games))
    return jax.device_get(out)


def _pairwise_totals(state) -> np.ndarray:
    """(snakes outlasted - snakes that outlasted me) / (N - 1), from the elimination turns.

    A snake still alive has outlasted every eliminated one; ties are neither.
    """
    alive, elim = np.asarray(state.alive), np.asarray(state.elim_turn).astype(np.int64)
    elim = np.where(alive, np.iinfo(np.int64).max, elim)
    n = elim.shape[-1]
    outlasted = (elim[:, :, None] > elim[:, None, :]).sum(-1)  # [g, i]: j with elim_j < elim_i
    outlasted_by = (elim[:, :, None] < elim[:, None, :]).sum(-1)
    return (outlasted - outlasted_by) / (n - 1)


def _tied_place_totals(n: int) -> list[tuple[float, ...]]:
    """Every sorted (best first) total vector of an ``n``-snake game, one per way of tying.

    Place ``p`` (0-based) is worth ``1 - 2p / (n - 1)``; snakes tied over a run of
    consecutive places share the average of their values.
    """
    values = [1 - 2 * p / (n - 1) for p in range(n)]
    out: list[tuple[float, ...]] = []

    def split(start: int, acc: tuple[float, ...]) -> None:
        if start == n:
            out.append(acc)
            return
        for end in range(start + 1, n + 1):
            split(end, acc + (float(np.mean(values[start:end])),) * (end - start))

    split(0, ())
    return out


@pytest.mark.parametrize("mode", MODES_WITH_ENV)
def test_placement_reward_is_zero_sum_per_step_and_matches_the_pairwise_formula(mode):
    config = mode_config(mode)
    state, placement, _, worst = _random_games(config)
    assert np.max(worst) < 1e-5  # every step sums to zero
    assert (np.asarray(state.done).mean()) > 0.5  # most games finished, so deaths were paid
    np.testing.assert_allclose(placement, _pairwise_totals(state), atol=1e-5)
    np.testing.assert_allclose(placement.sum(-1), 0, atol=1e-5)
    assert np.abs(placement).max() <= 1 + 1e-5
    # A finished game pays out the places: sorted, the totals are 1st, 2nd, ... with tied
    # snakes sharing their places' average (e.g. +1, 0, 0, -1 for a tie for 2nd of 4).
    finished = np.asarray(state.done) & (np.asarray(state.alive).sum(-1) <= 1)
    assert finished.sum() > 10
    allowed = np.array(_tied_place_totals(config.num_snakes))
    ordered = -np.sort(-placement[finished], axis=-1)
    fits = np.isclose(ordered[:, None, :], allowed[None], atol=1e-5).all(-1).any(-1)
    assert fits.all(), ordered[~fits][:3]


def test_tied_place_totals_enumerates_the_ways_to_tie():
    def has(n, vector):
        return any(np.allclose(v, vector) for v in _tied_place_totals(n))

    assert len(_tied_place_totals(2)) == 2 and has(2, (1, -1)) and has(2, (0, 0))
    assert len(_tied_place_totals(4)) == 8  # compositions of 4
    assert has(4, (1, 1 / 3, -1 / 3, -1))  # no ties
    assert has(4, (1, 0, 0, -1))  # a tie for 2nd
    assert has(4, (0, 0, 0, 0))  # all die together
    assert has(4, (1, -1 / 3, -1 / 3, -1 / 3))  # three die together behind the winner
    assert not has(4, (0.5, 0.5, -0.5, -0.5))  # not a way to share four places
    assert not has(4, (1, 0, -1 / 3, -2 / 3))


def test_placement_reward_totals_equal_win_loss_in_a_duel():
    config = mode_config("duel")
    _, placement, win_loss, _ = _random_games(config)
    np.testing.assert_allclose(placement, win_loss, atol=1e-6)
    assert set(np.unique(np.round(placement, 6))) <= {-1.0, 0.0, 1.0}


def test_placement_reward_differs_from_win_loss_in_four_snake_games():
    # Win/loss pays every place but 1st the same -1; placement pays the order.
    config = mode_config("standard")
    state, placement, win_loss, _ = _random_games(config)
    assert not np.allclose(placement, win_loss)
    finished = np.asarray(state.done) & (np.asarray(state.alive).sum(-1) <= 1)
    assert finished.sum() > 10
    # Mean win/loss total is well below 0 (about -0.5): three losers to one winner.
    assert win_loss[finished].mean() < -0.3
