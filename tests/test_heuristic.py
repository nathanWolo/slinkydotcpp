"""Tests for slinky.heuristic: grid utilities, the evaluator and the heuristic policy."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import heuristic as H
from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.evaluate import play_match, random_legal
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig, State

INF = H.INF
DUEL = GameConfig()


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


@functools.lru_cache(maxsize=None)
def env_for(config: GameConfig) -> BattlesnakeEnv:
    return BattlesnakeEnv(config, obs=None)


@functools.lru_cache(maxsize=None)
def jitted_policy(config: GameConfig):
    env = env_for(config)
    return jax.jit(jax.vmap(lambda k, s: H.heuristic_policy(k, s, env), in_axes=(0, None)))


def moves(state: State, config: GameConfig = DUEL, n_keys: int = 8) -> np.ndarray:
    """int[n_keys, N]: the policy's moves under several tie-breaking keys."""
    keys = jax.random.split(jax.random.key(0), n_keys)
    return np.asarray(jitted_policy(config)(keys, state))


def rollout_states(config: GameConfig, batch: int = 64, turns: int = 40, seed: int = 0) -> State:
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
    ],
    ids=["duel", "4p", "wrapped", "constrictor"],
)
def test_packed_fill_matches_grid_reference(config):
    states = rollout_states(config, batch=32, turns=50)
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
                config.ruleset.wrapped,
                wait,
            )
            return H._fill_stats_packed(*args), H._fill_stats_grid(*args)

        packed, grid = jax.jit(jax.vmap(both))(states)
        for name, a, b in zip(packed._fields, packed, grid, strict=True):
            np.testing.assert_array_equal(a, b, err_msg=f"{name} (wait={wait})")


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
    np.testing.assert_array_equal(H.evaluate(board([a, b, c], three, causes=causes), three), [-1, 0, 0])
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


# --- Policy: hand-built positions -----------------------------------------------------


def both_seats(snakes, **kw):
    """The position with the hero as snake 0, and again as snake 1 (for both seat indexings)."""
    health = kw.pop("health", [90, 90])
    return [
        (board(snakes, health=health, **kw), 0),
        (board(snakes[::-1], health=health[::-1], **kw), 1),
    ]


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
    states = rollout_states(config, batch=32, turns=30)
    env = env_for(config)
    keys = jax.random.split(jax.random.key(1), 32)
    acts = jax.jit(jax.vmap(lambda k, s: H.heuristic_policy(k, s, env)))(keys, states)
    assert acts.shape == (32, config.num_snakes) and acts.dtype == jnp.int32
    acts = np.asarray(acts)
    assert np.all((acts >= 0) & (acts < NUM_ACTIONS))
    mask = np.asarray(jax.vmap(env.action_mask)(states))
    chosen = np.take_along_axis(mask, acts[..., None], axis=-1)[..., 0]
    assert chosen.all()  # rows are never all-False, so a legal move always exists
    values = jax.jit(jax.vmap(lambda s: H.evaluate(s, config)))(states)
    assert values.shape == (32, config.num_snakes)


def test_heuristic_policy_is_cached_and_beats_random_legal():
    env = BattlesnakeEnv(DUEL, obs=None)
    policy = H.heuristic(env)
    assert H.heuristic(env) is policy
    result = play_match(env, policy, random_legal(env), jax.random.key(7), num_games=100)
    assert result.score > 0.9, result
