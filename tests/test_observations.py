"""Tests for slinky.observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.observations import (
    CHANNEL_NAMES,
    NUM_CHANNELS,
    allocentric_obs,
    egocentric_obs,
    make_obs_fn,
    obs_shape,
)
from slinky.types import GameConfig, Ruleset, State, empty_state

CH = {name: i for i, name in enumerate(CHANNEL_NAMES)}


def build_state(config: GameConfig, snakes, food=(), hazard=None) -> State:
    """Hand-build a State.

    ``snakes`` is a list (length N) of ``(points_head_first, health, alive)``;
    ``points`` may repeat a cell (stacked tail). ``hazard`` maps (x, y) -> count.
    """
    state = empty_state(config)
    n, h, w = config.num_snakes, config.height, config.width
    body = np.zeros((n, h, w), np.int16)
    head = np.zeros((n, 2), np.int32)
    length = np.zeros(n, np.int32)
    health = np.zeros(n, np.int32)
    alive = np.zeros(n, bool)
    for i, (points, hp, is_alive) in enumerate(snakes):
        length[i] = len(points)
        for idx, (x, y) in enumerate(points):
            body[i, y, x] = max(body[i, y, x], len(points) - idx)
        head[i] = points[0]
        health[i], alive[i] = hp, is_alive
    food_grid = np.zeros((h, w), bool)
    for x, y in food:
        food_grid[y, x] = True
    haz = np.zeros((h, w), np.int8)
    for (x, y), count in (hazard or {}).items():
        haz[y, x] = count
    return state._replace(
        body=jnp.asarray(body),
        head=jnp.asarray(head),
        length=jnp.asarray(length),
        health=jnp.asarray(health),
        alive=jnp.asarray(alive),
        food=jnp.asarray(food_grid),
        hazard=jnp.asarray(haz),
    )


def cells(obs, channel: str) -> set[tuple[int, int]]:
    """Set of (x, y) with a nonzero value in an allocentric channel."""
    ys, xs = np.nonzero(np.asarray(obs[..., CH[channel]]))
    return {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}


@pytest.fixture
def duel():
    """11x11, snake 0 (len 3, hp 80) vs snake 1 (len 4, hp 60), food and hazard."""
    config = GameConfig()
    snakes = [
        ([(5, 5), (4, 5), (3, 5)], 80, True),
        ([(8, 8), (8, 7), (8, 6), (8, 5)], 60, True),
    ]
    state = build_state(config, snakes, food=[(2, 2), (6, 5)], hazard={(0, 0): 2, (10, 10): 1})
    return config, state


def test_channel_layout():
    assert NUM_CHANNELS == 13 == len(CHANNEL_NAMES)
    assert CHANNEL_NAMES[0] == "on_board"
    assert CHANNEL_NAMES[-1] == "own_length"
    assert len(set(CHANNEL_NAMES)) == NUM_CHANNELS


def test_allocentric_channels_agent0(duel):
    config, state = duel
    obs = allocentric_obs(state, config, 0)
    assert obs.shape == (11, 11, NUM_CHANNELS)
    assert obs.dtype == jnp.float32
    assert np.all(np.isfinite(obs))
    cc = 121.0

    assert np.all(obs[..., CH["on_board"]] == 1)
    assert cells(obs, "food") == {(2, 2), (6, 5)}
    assert cells(obs, "hazard") == {(0, 0), (10, 10)}
    assert obs[0, 0, CH["hazard"]] == 2 and obs[10, 10, CH["hazard"]] == 1
    assert cells(obs, "own_head") == {(5, 5)}
    assert cells(obs, "own_body") == {(5, 5), (4, 5), (3, 5)}
    assert obs[5, 5, CH["own_countdown"]] == pytest.approx(3 / cc)  # head
    assert obs[5, 4, CH["own_countdown"]] == pytest.approx(2 / cc)
    assert obs[5, 3, CH["own_countdown"]] == pytest.approx(1 / cc)  # tail
    # Opponent (length 4 >= 3): its head is "ge".
    assert cells(obs, "opp_head_ge") == {(8, 8)}
    assert cells(obs, "opp_head_lt") == set()
    assert cells(obs, "opp_body") == {(8, 8), (8, 7), (8, 6), (8, 5)}
    assert obs[8, 8, CH["opp_countdown"]] == pytest.approx(4 / cc)
    assert obs[5, 8, CH["opp_countdown"]] == pytest.approx(1 / cc)
    assert cells(obs, "opp_head_health") == {(8, 8)}
    assert obs[8, 8, CH["opp_head_health"]] == pytest.approx(0.6)
    assert np.all(obs[..., CH["own_health"]] == pytest.approx(0.8))
    assert np.all(obs[..., CH["own_length"]] == pytest.approx(3 / cc))


def test_allocentric_swaps_with_agent(duel):
    config, state = duel
    obs = allocentric_obs(state, config, 1)
    assert cells(obs, "own_head") == {(8, 8)}
    assert cells(obs, "own_body") == {(8, 8), (8, 7), (8, 6), (8, 5)}
    assert obs[8, 8, CH["own_countdown"]] == pytest.approx(4 / 121)
    assert cells(obs, "opp_body") == {(5, 5), (4, 5), (3, 5)}
    # Opponent (length 3) is shorter than us (length 4).
    assert cells(obs, "opp_head_lt") == {(5, 5)}
    assert cells(obs, "opp_head_ge") == set()
    assert obs[5, 5, CH["opp_head_health"]] == pytest.approx(0.8)
    assert np.all(obs[..., CH["own_health"]] == pytest.approx(0.6))
    assert np.all(obs[..., CH["own_length"]] == pytest.approx(4 / 121))
    # Shared channels do not depend on the agent.
    other = allocentric_obs(state, config, 0)
    for name in ("on_board", "food", "hazard"):
        np.testing.assert_array_equal(obs[..., CH[name]], other[..., CH[name]])


def test_stacked_tail_countdown():
    config = GameConfig(num_snakes=1)
    state = build_state(config, [([(5, 5), (4, 5), (3, 5), (3, 5)], 100, True)])
    obs = allocentric_obs(state, config, 0)
    assert obs[5, 5, CH["own_countdown"]] == pytest.approx(4 / 121)
    assert obs[5, 3, CH["own_countdown"]] == pytest.approx(2 / 121)  # stacked tail
    assert cells(obs, "own_body") == {(5, 5), (4, 5), (3, 5)}


def test_head_ge_lt_equal_and_unequal_lengths():
    config = GameConfig(num_snakes=4)
    snakes = [
        ([(5, 5), (5, 4), (5, 3)], 100, True),  # agent, length 3
        ([(0, 0), (1, 0), (2, 0)], 90, True),  # equal length -> ge
        ([(10, 10), (9, 10), (8, 10), (7, 10), (6, 10)], 70, True),  # longer -> ge
        ([(0, 10), (1, 10)], 50, True),  # shorter -> lt
    ]
    state = build_state(config, snakes)
    obs = allocentric_obs(state, config, 0)
    assert cells(obs, "opp_head_ge") == {(0, 0), (10, 10)}
    assert cells(obs, "opp_head_lt") == {(0, 10)}
    assert obs[0, 0, CH["opp_head_health"]] == pytest.approx(0.9)
    assert obs[10, 10, CH["opp_head_health"]] == pytest.approx(0.7)
    assert obs[10, 0, CH["opp_head_health"]] == pytest.approx(0.5)
    assert len(cells(obs, "opp_body")) == 3 + 5 + 2
    # Agent 3 (shortest) sees everyone else as >= its own length.
    obs3 = allocentric_obs(state, config, 3)
    assert cells(obs3, "opp_head_ge") == {(5, 5), (0, 0), (10, 10)}
    assert cells(obs3, "opp_head_lt") == set()


def test_dead_opponent_is_ignored(duel):
    config, state = duel
    # Dead snake keeps a non-zero grid and an on-board head; it must still be ignored.
    state = state._replace(alive=jnp.array([True, False]))
    obs = allocentric_obs(state, config, 0)
    for name in (
        "opp_head_ge",
        "opp_head_lt",
        "opp_body",
        "opp_countdown",
        "opp_head_health",
    ):
        assert not np.any(np.asarray(obs[..., CH[name]])), name
    assert cells(obs, "own_head") == {(5, 5)}
    # Dead agent: own channels are zero, observation stays finite.
    obs1 = allocentric_obs(state, config, 1)
    assert np.all(np.isfinite(obs1))
    for name in ("own_head", "own_body", "own_countdown", "own_health", "own_length"):
        assert not np.any(np.asarray(obs1[..., CH[name]])), name
    assert cells(obs1, "opp_head_ge") | cells(obs1, "opp_head_lt") == {(5, 5)}


@pytest.mark.parametrize("head", [(5, 5), (0, 0), (10, 3), (4, 10)])
def test_egocentric_matches_shifted_allocentric(head):
    config = GameConfig()
    hx, hy = head
    snakes = [
        ([(hx, hy), (hx, hy - 1 if hy > 0 else hy + 1)], 77, True),
        ([(8, 8), (8, 7), (8, 6)], 60, True),
    ]
    state = build_state(config, snakes, food=[(2, 2)], hazard={(0, 0): 1})
    h, w = config.height, config.width
    allo = np.asarray(allocentric_obs(state, config, 0))
    ego = np.asarray(egocentric_obs(state, config, 0))
    assert ego.shape == (2 * h - 1, 2 * w - 1, NUM_CHANNELS)
    assert ego[h - 1, w - 1, CH["own_head"]] == 1
    assert ego[..., CH["own_head"]].sum() == 1
    for r in range(2 * h - 1):
        for c in range(2 * w - 1):
            y, x = hy + r - (h - 1), hx + c - (w - 1)
            if 0 <= y < h and 0 <= x < w:
                np.testing.assert_array_equal(ego[r, c], allo[y, x])
            else:
                assert not ego[r, c].any()  # padding is all zeros, incl. on_board
    assert ego[..., CH["on_board"]].sum() == h * w


def test_egocentric_food_next_to_head(duel):
    config, state = duel
    h, w = config.height, config.width
    ego = np.asarray(egocentric_obs(state, config, 0))
    assert ego[h - 1, w - 1, CH["own_head"]] == 1
    # Food at (6, 5) == head (5, 5) + (1, 0) lands one column to the right.
    assert ego[h - 1, w, CH["food"]] == 1
    assert ego[h - 1, w - 2, CH["food"]] == 0
    # Moving up one row increases the row index (y points up).
    assert ego[h - 1, w - 2, CH["own_body"]] == 1  # (4, 5): left of the head
    assert ego[h, w - 1, CH["on_board"]] == 1  # (5, 6)
    # Corner padding (head at 5,5 -> row 0 is y = -5, outside the board).
    assert ego[0, 0, CH["on_board"]] == 0
    assert ego[h - 1 - 5, w - 1, CH["on_board"]] == 1  # y = 0
    assert ego[h - 1 - 6, w - 1, CH["on_board"]] == 0  # y = -1


def test_egocentric_dead_agent_off_board_head():
    config = GameConfig()
    state = build_state(
        config,
        [([(5, 5), (4, 5), (3, 5)], 80, True), ([(2, 2), (2, 3)], 0, False)],
    )
    for bad_head in ([-1, -1], [11, 4], [3, 12]):
        s = state._replace(head=state.head.at[1].set(jnp.array(bad_head, jnp.int32)))
        ego = egocentric_obs(s, config, 1)
        assert ego.shape == obs_shape(config, "egocentric")
        assert np.all(np.isfinite(ego))


@pytest.mark.parametrize("head", [(5, 3), (0, 0), (6, 4), (2, 4)])
def test_wrapped_egocentric_rolls_head_to_center(head):
    config = GameConfig(width=7, height=5, ruleset=Ruleset.WRAPPED)
    h, w = config.height, config.width
    hx, hy = head
    snakes = [
        ([(hx, hy), ((hx - 1) % w, hy), ((hx - 2) % w, hy)], 90, True),
        ([(3, 1), (3, 0), (3, 4)], 50, True),  # body wraps over the top edge
    ]
    state = build_state(config, snakes, food=[((hx + 1) % w, hy)])
    ego = np.asarray(egocentric_obs(state, config, 0))
    allo = np.asarray(allocentric_obs(state, config, 0))
    assert ego.shape == (h, w, NUM_CHANNELS) == obs_shape(config, "egocentric")
    assert ego[h // 2, w // 2, CH["own_head"]] == 1
    assert ego[h // 2, w // 2 + 1, CH["food"]] == 1
    assert np.all(ego[..., CH["on_board"]] == 1)
    for r in range(h):
        for c in range(w):
            y, x = (hy + r - h // 2) % h, (hx + c - w // 2) % w
            np.testing.assert_array_equal(ego[r, c], allo[y, x])


def test_make_obs_fn_shapes_and_consistency(duel):
    config, state = duel
    for kind in ("egocentric", "allocentric"):
        fn = make_obs_fn(config, kind)
        out = fn(state)
        assert out.shape == (config.num_snakes, *obs_shape(config, kind))
        single = allocentric_obs if kind == "allocentric" else egocentric_obs
        for agent in range(config.num_snakes):
            np.testing.assert_allclose(out[agent], single(state, config, agent), rtol=1e-6)
    assert make_obs_fn(config).__call__(state).shape == (2, 21, 21, NUM_CHANNELS)
    assert obs_shape(config, "allocentric") == (11, 11, NUM_CHANNELS)
    wrapped = GameConfig(ruleset=Ruleset.WRAPPED_CONSTRICTOR)
    assert obs_shape(wrapped, "egocentric") == (11, 11, NUM_CHANNELS)
    assert make_obs_fn(wrapped)(empty_state(wrapped)).shape == (2, 11, 11, NUM_CHANNELS)
    with pytest.raises(ValueError):
        make_obs_fn(config, "bogus")
    with pytest.raises(ValueError):
        obs_shape(config, "bogus")


def test_jit_and_vmap(duel):
    config, state = duel
    for kind in ("egocentric", "allocentric"):
        fn = make_obs_fn(config, kind)
        np.testing.assert_allclose(jax.jit(fn)(state), fn(state), rtol=1e-6)
    # Traced agent index under jit.
    jitted = jax.jit(lambda s, a: egocentric_obs(s, config, a))
    np.testing.assert_allclose(
        jitted(state, jnp.int32(1)), egocentric_obs(state, config, 1), rtol=1e-6
    )
    # Batch of states: vmap over a stacked State.
    batch = jax.tree.map(lambda x: jnp.stack([x, x, x]), state)
    out = jax.jit(jax.vmap(make_obs_fn(config)))(batch)
    assert out.shape == (3, 2, 21, 21, NUM_CHANNELS)
    np.testing.assert_allclose(out[2], make_obs_fn(config)(state), rtol=1e-6)


def test_wrapped_jit(duel):
    _, state = duel
    config = GameConfig(ruleset=Ruleset.WRAPPED)
    out = jax.jit(make_obs_fn(config))(state)
    assert out.shape == (2, 11, 11, NUM_CHANNELS)
    assert np.all(np.isfinite(out))
