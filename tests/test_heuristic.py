"""Tests for slinky.heuristic: grid utilities, the evaluator and the heuristic policy."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import DUEL, board, both_seats, env_for, rollout_states

from slinky import heuristic as H
from slinky.env import BattlesnakeEnv
from slinky.evaluate import play_match, random_legal
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig, State

INF = H.INF
SOLO = GameConfig(num_snakes=1)
THREE = GameConfig(num_snakes=3)
FOUR = GameConfig(num_snakes=4)


@functools.cache
def jitted_policy(config: GameConfig, weights: H.Weights = H.DEFAULT_WEIGHTS):
    env = env_for(config)
    return jax.jit(jax.vmap(lambda k, s: H.heuristic_policy(k, s, env, weights), in_axes=(0, None)))


def moves(
    state: State, config: GameConfig = DUEL, n_keys: int = 8, weights: H.Weights = H.DEFAULT_WEIGHTS
) -> np.ndarray:
    """int[n_keys, N]: the policy's moves under several tie-breaking keys."""
    keys = jax.random.split(jax.random.key(0), n_keys)
    return np.asarray(jitted_policy(config, weights)(keys, state))


def every_seat(hero, others, config: GameConfig, food=(), hero_health: int = 90):
    """The position with the hero in each seat in turn (the others keep their order)."""
    positions = []
    for seat in range(config.num_snakes):
        rest = iter(others)
        snakes = [hero if i == seat else next(rest) for i in range(config.num_snakes)]
        health = [hero_health if i == seat else 90 for i in range(config.num_snakes)]
        positions.append((board(snakes, config, food=food, health=health), seat))
    return positions


# --- Grid utilities -----------------------------------------------------------------


def test_neighbours_plain_and_wrapped():
    m = np.zeros((3, 4), bool)
    m[0, 0] = True
    plain = np.asarray(H.neighbours(jnp.asarray(m)))
    assert plain.sum() == 2 and plain[1, 0] and plain[0, 1]
    wrapped = np.asarray(H.neighbours(jnp.asarray(m), wrapped=True))
    assert wrapped.sum() == 4 and wrapped[2, 0] and wrapped[0, 3]
    # Leading axes are batched.
    assert H.neighbours(jnp.zeros((5, 2, 3, 4), bool)).shape == (5, 2, 3, 4)


def test_distances_static_maze():
    # y = 0 is the first row. A wall splits the board except at the top.
    walls = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        bool,
    )
    src = np.zeros_like(walls)
    src[0, 0] = True
    dist = np.asarray(H.distances(jnp.asarray(~walls), jnp.asarray(src), 10))
    expected = np.array(
        [
            [0, INF, 6, 7],
            [1, INF, 5, INF],
            [2, 3, 4, INF],
        ]
    )
    np.testing.assert_array_equal(dist, expected)
    # max_dist cuts the search off.
    short = np.asarray(H.distances(jnp.asarray(~walls), jnp.asarray(src), 4))
    np.testing.assert_array_equal(short, np.where(expected <= 4, expected, INF))
    reach = np.asarray(H.flood_fill(jnp.asarray(~walls), jnp.asarray(src), 10))
    np.testing.assert_array_equal(reach, expected < INF)


def test_distances_wrapped_and_start():
    open_ = jnp.ones((1, 5), bool)
    src = jnp.zeros((1, 5), bool).at[0, 0].set(True)
    np.testing.assert_array_equal(H.distances(open_, src, 9)[0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(H.distances(open_, src, 9, wrapped=True)[0], [0, 1, 2, 2, 1])
    np.testing.assert_array_equal(H.distances(open_, src, 9, start=1)[0], [1, 2, 3, 4, 5])


def test_arrival_times_tail_rule_and_waiting():
    src = jnp.zeros((1, 4), bool).at[0, 0].set(True)
    # A tail (countdown 1) can be entered on the next move; a stacked tail (2) cannot.
    tail = jnp.array([[0, 1, 0, 0]])
    np.testing.assert_array_equal(H.arrival_times(tail, src, 9)[0], [0, 1, 2, 3])
    stacked = jnp.array([[0, 2, 0, 0]])
    assert int(H.arrival_times(stacked, src, 9)[0, 1]) == INF
    # A body cell with countdown k opens on move k. Without waiting, a cell that is
    # blocked when first reached is not retried; with waiting it is entered on move k.
    body = jnp.array([[0, 0, 3, 0]])
    np.testing.assert_array_equal(H.arrival_times(body, src, 9)[0], [0, 1, INF, INF])
    np.testing.assert_array_equal(H.arrival_times(body, src, 9, wait=True)[0], [0, 1, 3, 4])


def test_arrival_times_around_a_moving_body():
    # A length-3 snake lies across the middle row: head (0,1)=3, (1,1)=2, tail (2,1)=1.
    countdown = jnp.array([[0, 0, 0], [3, 2, 1], [0, 0, 0]])
    src = jnp.zeros((3, 3), bool).at[0, 0].set(True)
    arr = np.asarray(H.arrival_times(countdown, src, 9))
    expected = np.array([[0, 1, 2], [3, 2, 3], [4, 3, 4]])
    np.testing.assert_array_equal(arr, expected)


def test_voronoi_ties_follow_head_to_head_rule():
    arr = jnp.array([[[1, 2, 3, INF]], [[2, 2, 1, INF]], [[3, 2, 9, INF]]])
    claims = np.asarray(H.voronoi(arr, jnp.array([3, 3, 5])))
    # Cell 0: snake 0 first. Cell 1: three-way tie, snake 2 is strictly longest.
    # Cell 2: snake 1 first. Cell 3: nobody reaches it.
    np.testing.assert_array_equal(claims[:, 0], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
    tie = np.asarray(H.voronoi(arr[:2], jnp.array([4, 4])))
    assert not tie[:, 0, 1].any()  # equal arrival at equal length goes to nobody


@pytest.mark.parametrize(
    "config",
    [
        DUEL,
        GameConfig(num_snakes=4),
        GameConfig(ruleset="wrapped", width=7, height=9, num_snakes=3),
        GameConfig(ruleset="constrictor"),
        GameConfig(ruleset="wrapped", width=32, height=5, num_snakes=3),
        GameConfig(width=33, height=5),
    ],
    ids=["duel", "4p", "wrapped", "constrictor", "wrapped32", "width33"],
)
def test_packed_fill_matches_grid_reference(config):
    # fill_stats uses bit-packed rows up to width 32 (one uint32 per row) and the
    # boolean grids beyond.
    states = rollout_states(config, batch=32, turns=50 if config.width <= 11 else 25)
    steps = config.width + config.height
    for wait in (False, True):

        def both(s, wait=wait):
            args = (
                H.free_after(s, config),
                H._cells(s.head, config) & s.alive[:, None, None],
                s.length,
                s.food,
                H.tail_cells(s, config),
                steps,
            )
            fast = H.fill_stats(*args, wrapped=config.ruleset.wrapped, wait=wait)
            return fast, H._fill_stats_grid(*args, config.ruleset.wrapped, wait)

        packed, grid = jax.jit(jax.vmap(both))(states)
        for name, a, b in zip(packed._fields, packed, grid, strict=True):
            np.testing.assert_array_equal(a, b, err_msg=f"{name} (wait={wait})")


def test_packed_fill_is_the_same_under_any_batching():
    # The packed fill folds vmapped axes into its own batch (a custom_vmap). A single
    # game, a flat batch, a nested batch and a batch with shared arguments must all
    # agree with the grid reference, which is plain jax.
    config = GameConfig(ruleset="wrapped", width=9, height=7, num_snakes=3)
    states = rollout_states(config, batch=12, turns=40)

    def stats(fn):
        def run(countdown, head, alive, length, food, tail):
            heads = H._cells(head, config) & alive[:, None, None]
            return fn(countdown, heads, length, food, tail, 16, True, False)

        return run

    packed, grid = stats(H._fill_stats_packed), stats(H._fill_stats_grid)
    countdown = jax.vmap(lambda s: H.free_after(s, config))(states)
    tails = jax.vmap(lambda s: H.tail_cells(s, config))(states)
    args = (countdown, states.head, states.alive, states.length, states.food, tails)
    ref = jax.jit(jax.vmap(grid))(*args)
    flat = jax.jit(jax.vmap(packed))(*args)
    nested = jax.jit(jax.vmap(jax.vmap(packed)))(*(x.reshape(3, 4, *x.shape[1:]) for x in args))
    single = jax.jit(packed)(*(x[5] for x in args))
    # Every game's snakes on game 0's board: the countdown and the food are not batched.
    axes = (None, 0, 0, 0, None, 0)
    some = [x[0] if axis is None else x for x, axis in zip(args, axes, strict=True)]
    shared = jax.jit(jax.vmap(packed, in_axes=axes))(*some)
    shared_ref = jax.jit(jax.vmap(grid, in_axes=axes))(*some)
    for name, *xs in zip(ref._fields, ref, flat, nested, single, shared, shared_ref, strict=True):
        a, b, c, d, e, f = (np.asarray(x) for x in xs)
        np.testing.assert_array_equal(b, a, err_msg=f"{name} (flat)")
        np.testing.assert_array_equal(c.reshape(a.shape), a, err_msg=f"{name} (nested)")
        np.testing.assert_array_equal(d, a[5], err_msg=f"{name} (single)")
        np.testing.assert_array_equal(e, f, err_msg=f"{name} (shared)")


def test_constrictor_bodies_never_free():
    config = GameConfig(ruleset="constrictor")
    s = board([[(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]], config)
    tf = np.asarray(H.free_after(s, config))
    assert tf[3, 5] > 1000 and tf[0, 0] == 0


# --- Evaluator ----------------------------------------------------------------------


def test_evaluate_terminal_values():
    a, b = [(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]
    won = board([a, b], causes=[("", 0), ("head-collision", 10)])
    np.testing.assert_array_equal(H.evaluate(won, DUEL), [1.0, -1.0])
    draw = board([a, b], causes=[("head-collision", 10), ("head-collision", 10)])
    np.testing.assert_array_equal(H.evaluate(draw, DUEL), [0.0, 0.0])
    np.testing.assert_allclose(H.evaluate(draw, DUEL, draw_value=-0.4), [-0.4, -0.4])
    # Three snakes: one died earlier (-1), the last two died together (draw).
    three = GameConfig(num_snakes=3)
    c = [(8, 8), (8, 7), (8, 6)]
    causes = [("wall-collision", 4), ("head-collision", 9), ("head-collision", 9)]
    np.testing.assert_array_equal(
        H.evaluate(board([a, b, c], three, causes=causes), three), [-1, 0, 0]
    )
    # Not over yet: the dead snake gets -1, the living ones a heuristic value.
    v = np.asarray(H.evaluate(board([a, b, c], three, causes=[("", 0)] * 2 + [causes[0]]), three))
    assert v[2] == -1 and np.all(np.abs(v[:2]) < 1)
    # Solo: dying is -1, a comfortable snake is worth about 0.
    solo = GameConfig(num_snakes=1)
    assert float(H.evaluate(board([a], solo, causes=[("wall-collision", 3)]), solo)[0]) == -1
    assert -0.1 < float(H.evaluate(board([a], solo), solo)[0]) <= 0


def test_evaluate_matches_win_loss_reward_on_done_states():
    env = env_for(DUEL)
    s = board([[(4, 5), (3, 5), (2, 5)], [(6, 5), (7, 5), (8, 5)]])
    done, ts = env.step(jax.random.key(0), s, jnp.array([RIGHT, LEFT]))  # equal head-on
    assert bool(done.done)
    np.testing.assert_array_equal(H.evaluate(done, DUEL), ts.reward)
    s = board([[(0, 5), (1, 5), (2, 5)], [(8, 8), (8, 7), (8, 6)]])
    done, ts = env.step(jax.random.key(0), s, jnp.array([LEFT, UP]))  # snake 0 hits the wall
    np.testing.assert_array_equal(H.evaluate(done, DUEL), ts.reward)


def test_evaluate_bounded_and_antisymmetric():
    states = rollout_states(DUEL, batch=128, turns=60)
    v = np.asarray(jax.jit(jax.vmap(lambda s: H.evaluate(s, DUEL)))(states))
    assert v.shape == (128, 2) and v.dtype == np.float32
    assert np.all(np.abs(v) <= 1)
    both = np.asarray(states.alive).all(axis=1)
    assert both.sum() > 50
    np.testing.assert_array_equal(v[both, 0], -v[both, 1])
    assert np.all(np.abs(v[both]) <= np.float32(0.99))  # exact outcomes (+-1) always dominate
    # Swapping the seats swaps the values.
    swapped = jax.tree.map(lambda x: x[:, ::-1] if x.ndim >= 2 and x.shape[1] == 2 else x, states)
    swapped = swapped._replace(food=states.food, hazard=states.hazard)
    v_sw = np.asarray(jax.jit(jax.vmap(lambda s: H.evaluate(s, DUEL)))(swapped))
    np.testing.assert_allclose(v_sw[both], v[both][:, ::-1], atol=1e-6)


def test_evaluate_sign_in_a_clearly_winning_position():
    # Snake 0 is long and central; snake 1 is short and boxed into a corner by snake 0.
    big = [(3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
    small = [(1, 1), (1, 2), (1, 3)]
    v = np.asarray(H.evaluate(board([big, small], food=[(8, 8)]), DUEL))
    assert v[0] > 0.5 and v[1] == -v[0]
    terms = H.snake_terms(board([big, small], food=[(8, 8)]), DUEL)
    assert terms.territory[0] > terms.territory[1]


def test_evaluate_is_clipped_short_of_a_win():
    # Snake 0 is 21 longer, so tanh saturates; the clip keeps it below an exact win.
    path = [(x, 0) for x in range(11)] + [(x, 1) for x in range(10, -1, -1)] + [(0, 2), (1, 2)]
    v = np.asarray(H.evaluate(board([path[::-1], [(8, 8), (8, 7), (8, 6)]]), DUEL))
    np.testing.assert_array_equal(v, np.float32([0.99, -0.99]))


# --- Policy: hand-built positions -----------------------------------------------------


def test_avoids_a_dead_end_pocket_the_mask_allows():
    # The hero (length 10) can go left into a 4-cell pocket walled by its own body
    # (with food in it, to tempt it), or right into the open board.
    hero = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
    other = [(9, 9), (9, 8), (9, 7)]
    for state, seat in both_seats([hero, other], food=[(1, 0)], health=[20, 90]):
        assert bool(env_for(DUEL).action_mask(state)[seat, LEFT])
        assert set(moves(state)[:, seat]) == {RIGHT}


def test_avoids_head_to_head_with_a_longer_snake():
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    longer = [(5, 7), (5, 8), (5, 9), (6, 9), (7, 9), (8, 9)]
    for state, seat in both_seats([hero, longer]):
        assert UP not in set(moves(state)[:, seat])
    # Also when the opponent is the same length (a mutual elimination is worth -contempt).
    equal = [(5, 7), (5, 8), (5, 9), (6, 9)]
    for state, seat in both_seats([hero, equal]):
        assert UP not in set(moves(state)[:, seat])


def test_contempt_is_the_value_of_a_mutual_elimination():
    # The heads meet on (5, 6) after (UP, DOWN), at equal length.
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    equal = [(5, 7), (5, 8), (5, 9), (6, 9)]
    joint = jax.jit(H.joint_values, static_argnums=(1, 2))
    m = np.asarray(joint(board([hero, equal]), DUEL, H.Weights(contempt=0.6)))
    np.testing.assert_allclose(m[UP, DOWN], [-0.6, -0.6])
    # With a draw worth more than the alternatives, the snake goes for the head-on.
    seeks_draws = H.Weights(contempt=-0.5)
    for state, seat in both_seats([hero, equal]):
        assert set(moves(state, weights=seeks_draws)[:, seat]) == {UP}


def test_fits_tier_ignores_replies_that_end_the_game():
    # Equal lengths (16), and the opponent's only move is UP to (0, 2). Hero DOWN is a
    # certain mutual elimination. Hero UP enters a 3-cell pocket, but the opponent then
    # has no surviving move, so UP wins two moves later. A certain draw must not count
    # as "fitting" and outrank it on the tier.
    hero = [(0, 3), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (0, 7), (0, 8), (0, 9), (0, 10)]
    hero += [(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10)]
    other = [(0, 1), (0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0)]
    other += [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)]
    for state, seat in both_seats([hero, other]):
        assert set(moves(state)[:, seat]) == {UP}
    # Hero DOWN kills the short snake whatever it does, leaving the hero's head in a
    # pocket of its own body. The finished board is not checked for "fits".
    hero = [(0, 3), (1, 3), (1, 2), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    hero += [(7, 0), (8, 0), (9, 0), (10, 0), (10, 1), (10, 2)]
    short = [(0, 1), (0, 0), (1, 0)]
    for state, seat in both_seats([hero, short]):
        assert set(moves(state)[:, seat]) == {DOWN}


def test_takes_a_head_to_head_kill_against_a_shorter_snake():
    # The short snake's only legal move is (1, 2), which the hero's head can also reach.
    hero = [(1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    short = [(0, 2), (0, 1), (0, 0)]
    for state, seat in both_seats([hero, short]):
        mask = np.asarray(env_for(DUEL).action_mask(state))
        assert mask[1 - seat].tolist() == [False, False, False, True]  # only RIGHT
        assert set(moves(state)[:, seat]) == {DOWN}


@pytest.mark.parametrize("health", [1, 5])
def test_eats_adjacent_food_when_starving(health):
    hero = [(5, 5), (5, 4), (5, 3)]
    other = [(9, 9), (9, 8), (9, 7)]
    for state, seat in both_seats([hero, other], food=[(6, 5), (0, 10)], health=[health, 90]):
        assert set(moves(state)[:, seat]) == {RIGHT}


# --- Policy: other numbers of snakes (move_features, no lookahead) -------------------


def test_move_features_head_to_heads_and_kills():
    features = jax.jit(H.move_features, static_argnums=1)
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    equal = [(5, 7), (5, 8), (5, 9), (6, 9)]
    third = [(1, 1), (1, 0), (0, 0)]
    # Meeting an equal head: a draw in a duel, but a loss when another snake survives it.
    duel = features(board([hero, equal]), DUEL)
    assert duel.h2h_danger[0].tolist() == [True, False, False, False]
    assert not duel.h2h_loss.any()
    three = features(board([hero, equal, third], THREE), THREE)
    assert three.h2h_danger[0].tolist() == three.h2h_loss[0].tolist() == [True, False, False, False]
    # The short snake's only move is to (1, 2), which the hero reaches by moving DOWN.
    hero = [(1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    short = [(0, 2), (0, 1), (0, 0)]
    f = features(board([hero, short, [(9, 9), (9, 8), (9, 7)]], THREE, food=[(2, 3)]), THREE)
    np.testing.assert_array_equal(f.kill_chance[0], [0.0, 1.0, 0.0, 0.0])
    assert f.eat[0].tolist() == [False, False, False, True]
    assert f.legal[0].tolist() == [True, True, False, True]  # LEFT is the neck


@pytest.mark.parametrize("longer", [True, False], ids=["longer", "equal"])
def test_three_snakes_avoid_head_to_heads_that_lose(longer):
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    other = [(5, 7), (5, 8), (5, 9), (6, 9)] + ([(7, 9), (8, 9)] if longer else [])
    for state, seat in every_seat(hero, [other, [(1, 1), (1, 0), (0, 0)]], THREE):
        assert UP not in set(moves(state, THREE)[:, seat])


def test_three_snakes_take_a_forced_kill():
    hero = [(1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    short = [(0, 2), (0, 1), (0, 0)]
    for state, seat in every_seat(hero, [short, [(9, 9), (9, 8), (9, 7)]], THREE):
        assert set(moves(state, THREE)[:, seat]) == {DOWN}


def test_solo_snake_avoids_a_dead_end_pocket():
    hero = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
    state = board([hero], SOLO, food=[(1, 0)], health=[20])
    assert set(moves(state, SOLO)[:, 0]) == {RIGHT}


@pytest.mark.parametrize("health", [1, 5])
def test_four_snakes_eat_adjacent_food_when_starving(health):
    hero = [(5, 5), (5, 4), (5, 3)]
    others = [[(9, 9), (9, 8), (9, 7)], [(1, 9), (1, 8), (1, 7)], [(9, 1), (9, 2), (9, 3)]]
    for state, seat in every_seat(hero, others, FOUR, food=[(6, 5), (0, 10)], hero_health=health):
        assert set(moves(state, FOUR)[:, seat]) == {RIGHT}


# --- Policy: shapes, configs and strength ---------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        DUEL,
        GameConfig(num_snakes=4),
        GameConfig(num_snakes=1),
        GameConfig(ruleset="wrapped"),
        GameConfig(ruleset="constrictor"),
    ],
    ids=["duel", "4p", "solo", "wrapped", "constrictor"],
)
def test_policy_shapes_and_legality(config):
    # Every turn of 32 random games (autoreset), not just the last one.
    env = env_for(config)
    batch, turns = 32, 30
    k_reset, k_play = jax.random.split(jax.random.key(1))
    states, _ = jax.vmap(env.reset)(jax.random.split(k_reset, batch))

    def step(states, key):
        k_pol, k_step = jax.random.split(key)
        keys = jax.random.split(k_pol, batch)
        acts = jax.vmap(lambda k, s: random_legal_policy(k, s, env))(keys, states)
        states, _ = jax.vmap(env.step_autoreset)(jax.random.split(k_step, batch), states, acts)
        return states, states

    play = jax.jit(lambda s, k: jax.lax.scan(step, s, jax.random.split(k, turns))[1])
    states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), play(states, k_play))

    def policy(key, s):
        return H.heuristic_policy(key, s, env), H.legal_moves(s, config), env.action_mask(s)

    keys = jax.random.split(jax.random.key(2), batch * turns)
    acts, legal, mask = jax.jit(jax.vmap(policy))(keys, states)
    assert acts.shape == (batch * turns, config.num_snakes) and acts.dtype == jnp.int32
    acts, legal, mask = np.asarray(acts), np.asarray(legal), np.asarray(mask)
    assert np.all((acts >= 0) & (acts < NUM_ACTIONS))
    # Never a masked move, and a legal one (no certain starvation) whenever one exists.
    assert np.take_along_axis(mask, acts[..., None], axis=-1).all()
    chosen_legal = np.take_along_axis(legal, acts[..., None], axis=-1)[..., 0]
    assert chosen_legal[legal.any(-1)].all()
    values = jax.jit(jax.vmap(lambda s: H.evaluate(s, config)))(states)
    assert values.shape == (batch * turns, config.num_snakes)


def test_heuristic_policy_is_cached_and_beats_random_legal():
    env = BattlesnakeEnv(DUEL, obs=None)
    policy = H.heuristic(env)
    assert H.heuristic(env) is policy
    assert H.heuristic(env, H.Weights(contempt=0.4)) is not policy
    result = play_match(env, policy, random_legal(env), jax.random.key(7), num_games=100)
    assert result.score > 0.9, result
