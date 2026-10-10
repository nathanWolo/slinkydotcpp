"""Tests for slinky.mcts: search invariants, shapes, tactics and a strength sanity check."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import mcts as M
from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.evaluate import play_match, random_legal
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig, State

DUEL = GameConfig()
DEFAULT = M.MCTSConfig(num_simulations=128)


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


@functools.cache
def env_for(config: GameConfig) -> BattlesnakeEnv:
    return BattlesnakeEnv(config, obs=None)


@functools.cache
def batched_search(game: GameConfig, config: M.MCTSConfig):
    """jit(vmap(search)) over keys, for one unbatched state."""
    env = env_for(game)
    return jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config), in_axes=(0, None)))


def run(state: State, config: M.MCTSConfig = DEFAULT, game: GameConfig = DUEL, n_keys: int = 4):
    """SearchOutput (numpy) with a leading axis of ``n_keys`` independent searches."""
    keys = jax.random.split(jax.random.key(0), n_keys)
    return jax.tree.map(np.asarray, batched_search(game, config)(keys, state))


def rollout_states(config: GameConfig, batch: int = 16, turns: int = 30, seed: int = 0) -> State:
    """A batch of mid-game states from random legal play (autoreset)."""
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


def both_seats(snakes, **kw):
    """The position with the hero as snake 0, and again as snake 1."""
    health = kw.pop("health", [90, 90])
    return [
        (board(snakes, health=health, **kw), 0),
        (board(snakes[::-1], health=health[::-1], **kw), 1),
    ]


# --- Invariants -----------------------------------------------------------------------


def test_config_validation_and_digits():
    with pytest.raises(ValueError):
        M.MCTSConfig(num_simulations=0)
    with pytest.raises(ValueError):
        M.MCTSConfig(selection="exp3")
    with pytest.raises(ValueError):
        M.MCTSConfig(final="argmax")
    hash(M.MCTSConfig())  # frozen and hashable (used as a cache key)
    d = M._digits(2)
    assert d.shape == (16, 2) and d[3 + 4 * 2].tolist() == [3, 2]


@pytest.mark.parametrize("selection", ["duct", "rm"])
def test_root_statistics_are_consistent(selection):
    config = M.MCTSConfig(num_simulations=64, selection=selection)
    states = rollout_states(DUEL, batch=16)
    env = env_for(DUEL)
    keys = jax.random.split(jax.random.key(3), 16)
    out = jax.tree.map(
        np.asarray, jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config)))(keys, states)
    )
    alive_root = ~np.asarray(states.done)
    assert alive_root.sum() >= 8
    # Every simulation passes through the root and takes one move per player.
    np.testing.assert_array_equal(out.visits[alive_root].sum(-1), 64)
    assert np.all(np.abs(out.q) <= 1) and np.all(np.abs(out.value) <= 1)
    np.testing.assert_allclose(out.policy.sum(-1), 1.0, rtol=1e-5)
    assert np.all(out.policy >= 0)
    assert np.all((out.nodes_used >= 1) & (out.nodes_used <= 65))
    assert np.all(out.depth <= 64)
    # Moves never leave env.action_mask, and visits stay on the selectable moves.
    mask = np.asarray(jax.vmap(env.action_mask)(states))
    assert np.take_along_axis(mask, out.action[..., None], -1).all()
    assert np.all(out.visits[~mask] == 0)
    # Root values are the visit-weighted mean of the move values.
    total = out.visits.sum(-1)
    mean = (out.visits * out.q).sum(-1) / np.maximum(total, 1)
    np.testing.assert_allclose(out.value[alive_root], mean[alive_root], atol=1e-5)


def test_deterministic_given_key():
    state = rollout_states(DUEL, batch=1)
    state = jax.tree.map(lambda x: x[0], state)
    a, b = run(state), run(state)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    # The keys differ across the batch, so the searches do too.
    assert not np.array_equal(a.visits[0], a.visits[1])


def test_finished_game_returns_valid_moves():
    a, b = [(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]
    over = board([a, b], causes=[("", 0), ("head-collision", 10)])
    assert bool(over.done)
    out = run(over, M.MCTSConfig(num_simulations=16))
    assert np.all((out.action >= 0) & (out.action < NUM_ACTIONS))
    assert np.all(out.visits == 0) and np.all(out.nodes_used == 1)
    np.testing.assert_array_equal(out.value, np.tile([1.0, -1.0], (4, 1)))
    # A drawn game, through the policy wrapper.
    draw = board([a, b], causes=[("head-collision", 10), ("head-collision", 10)])
    acts = M.mcts_policy(jax.random.key(0), draw, env_for(DUEL), M.MCTSConfig(num_simulations=8))
    assert acts.shape == (2,) and np.all(np.asarray(acts) >= 0)


@pytest.mark.parametrize("spawn_food", [True, False])
def test_truncation_inside_the_tree_is_a_draw(spawn_food):
    # One turn before max_turns every child ends the game: surviving is worth exactly 0.
    game = GameConfig(max_turns=11)
    hero = [(0, 5), (1, 5), (2, 5)]
    other = [(8, 8), (8, 7), (8, 6)]
    state = board([hero, other], game, turn=10)
    out = run(state, M.MCTSConfig(num_simulations=64, spawn_food=spawn_food), game)
    # The hero's LEFT runs into the wall (masked, never visited); its other moves survive.
    assert np.all(out.visits[:, 0, LEFT] == 0)
    visited = out.visits > 0
    np.testing.assert_array_equal(out.q[visited], 0.0)
    assert np.all(out.nodes_used <= 1 + 9)  # 3 x 3 joint moves, all terminal


@pytest.mark.parametrize("num_snakes", [1, 3, 4])
def test_shapes_with_other_numbers_of_snakes(num_snakes):
    game = GameConfig(num_snakes=num_snakes)
    states = rollout_states(game, batch=8, turns=20)
    env = env_for(game)
    config = M.MCTSConfig(num_simulations=32)
    keys = jax.random.split(jax.random.key(1), 8)
    out = jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config)))(keys, states)
    assert out.action.shape == (8, num_snakes) and out.action.dtype == jnp.int32
    assert out.visits.shape == (8, num_snakes, NUM_ACTIONS)
    assert out.q.shape == out.policy.shape == (8, num_snakes, NUM_ACTIONS)
    assert out.value.shape == (8, num_snakes)
    out = jax.tree.map(np.asarray, out)
    live = ~np.asarray(states.done)
    np.testing.assert_array_equal(out.visits[live].sum(-1), 32)
    # Dead snakes (in games still running) put every visit on their single move.
    dead = live[:, None] & ~np.asarray(states.alive)
    assert np.all(out.visits[dead][:, 1:] == 0)
    mask = np.asarray(jax.vmap(env.action_mask)(states))
    assert np.take_along_axis(mask, out.action[..., None], -1).all()


def test_too_many_snakes():
    game = GameConfig(num_snakes=5)
    state = env_for(game).init_state(jax.random.key(0))
    with pytest.raises(ValueError):
        M.search(jax.random.key(0), state, env_for(game), M.MCTSConfig(num_simulations=4))


def test_variants_run():
    """RM, UCB1-Tuned, sampled final moves, rollouts and the spawn-free model all run."""
    state = jax.tree.map(lambda x: x[0], rollout_states(DUEL, batch=1))
    for config in [
        M.MCTSConfig(num_simulations=32, ucb1_tuned=True, final="sample"),
        M.MCTSConfig(num_simulations=32, selection="rm", final="sample"),
        M.MCTSConfig(num_simulations=16, rollout_steps=10, leaf="none", spawn_food=False),
        M.MCTSConfig(num_simulations=8, rollout_steps=2, rollout_policy="heuristic"),
    ]:
        out = run(state, config, n_keys=2)
        np.testing.assert_array_equal(out.visits.sum(-1), config.num_simulations)
        assert np.all(np.abs(out.q) <= 1)


# --- Tactics (hero in both seats, several keys) ----------------------------------------


@pytest.mark.parametrize("selection", ["duct", "rm"])
def test_avoids_head_to_head_with_a_longer_snake(selection):
    config = M.MCTSConfig(num_simulations=128, selection=selection)
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    longer = [(5, 7), (5, 8), (5, 9), (6, 9), (7, 9), (8, 9)]
    for state, seat in both_seats([hero, longer]):
        assert UP not in set(run(state, config).action[:, seat])


@pytest.mark.parametrize("selection", ["duct", "rm"])
def test_takes_a_forced_kill_of_a_shorter_snake(selection):
    # The short snake's only legal move is (1, 2), which the hero's head can also reach.
    config = M.MCTSConfig(num_simulations=64, selection=selection)
    hero = [(1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    short = [(0, 2), (0, 1), (0, 0)]
    for state, seat in both_seats([hero, short]):
        out = run(state, config)
        assert set(out.action[:, seat]) == {DOWN}
        assert np.all(out.q[:, seat, DOWN] == 1.0)  # an exact win


def test_avoids_a_deep_pocket_the_mask_allows():
    # The hero (length 10) can go left into a 4-cell pocket walled by its own body
    # (with food in it, to tempt it): it is dead 5 moves later. Right is open.
    hero = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
    other = [(9, 9), (9, 8), (9, 7)]
    for state, seat in both_seats([hero, other], food=[(1, 0)], health=[20, 90]):
        assert bool(env_for(DUEL).action_mask(state)[seat, LEFT])
        assert set(run(state).action[:, seat]) == {RIGHT}


def test_tree_alone_finds_a_trap_three_moves_deep():
    # No heuristic (leaf="none": living snakes are worth 0), so only the search
    # can see it: left leads into a 2-cell dead end, and every line dies on move 3.
    hero = [(2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
    other = [(9, 9), (9, 8), (9, 7)]
    config = M.MCTSConfig(num_simulations=256, leaf="none")
    for state, seat in both_seats([hero, other]):
        assert bool(env_for(DUEL).action_mask(state)[seat, LEFT])
        out = run(state, config)
        assert set(out.action[:, seat]) == {RIGHT}
        assert np.all(out.q[:, seat, LEFT] < -0.5)


# --- Policy and strength ---------------------------------------------------------------


def test_policy_is_cached_and_beats_random_legal():
    env = BattlesnakeEnv(DUEL, obs=None)
    config = M.MCTSConfig(num_simulations=64)
    policy = M.mcts(env, config)
    assert M.mcts(env, config) is policy
    result = play_match(env, policy, random_legal(env), jax.random.key(7), num_games=64)
    assert result.score >= 0.9, result
