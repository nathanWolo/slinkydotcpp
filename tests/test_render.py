"""Tests for slinky.render."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from slinky.render import print_state, render
from slinky.types import Cause, GameConfig, State, empty_state


def build_state(config: GameConfig, snakes, food=(), hazard=()) -> State:
    """Hand-build a State; ``snakes`` is a list of ``(points_head_first, health, alive)``."""
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
    for x, y in hazard:
        haz[y, x] += 1
    return state._replace(
        body=jnp.asarray(body),
        head=jnp.asarray(head),
        length=jnp.asarray(length),
        health=jnp.asarray(health),
        alive=jnp.asarray(alive),
        food=jnp.asarray(food_grid),
        hazard=jnp.asarray(haz),
    )


def small_state() -> tuple[GameConfig, State]:
    config = GameConfig(width=5, height=4, num_snakes=3)
    snakes = [
        ([(2, 1), (1, 1), (0, 1)], 90, True),
        ([(3, 3), (3, 2), (3, 1)], 55, True),
        # Dead: grid is left non-zero on purpose; it must not be drawn.
        ([(4, 3), (4, 2)], 0, False),
    ]
    state = build_state(
        config,
        snakes,
        food=[(4, 0), (0, 3)],
        # (0, 1) is under snake A's body, (4, 3) is under the dead snake's grid.
        hazard=[(2, 2), (0, 1), (4, 3)],
    )
    state = state._replace(
        turn=jnp.int32(7),
        elim_cause=jnp.array([0, 0, int(Cause.HEAD_COLLISION)], jnp.int8),
    )
    return config, state


EXPECTED = """\
turn 7
3 * . . B ~
2 . . ~ b .
1 a a A b .
0 . . . . *
  0 1 2 3 4
A alive len=3 health=90
B alive len=3 health=55
C dead (HEAD_COLLISION) len=2 health=0"""


def test_render_small_board_exact():
    config, state = small_state()
    assert render(state, config) == EXPECTED
    assert render(state) == EXPECTED  # config is optional


def test_render_empty_cells_and_game_over():
    config = GameConfig(width=3, height=2, num_snakes=1)
    state = build_state(config, [([(1, 0), (0, 0)], 100, True)])
    out = render(state._replace(done=jnp.bool_(True)), config)
    assert out.splitlines() == [
        "turn 0  (game over)",
        "1 . . .",
        "0 a A .",
        "  0 1 2",
        "A alive len=2 health=100",
    ]


def test_render_two_digit_coordinates_aligned():
    config = GameConfig(width=12, height=11, num_snakes=1)
    state = build_state(config, [([(10, 10), (11, 10), (11, 9)], 100, True)], food=[(0, 0)])
    lines = render(state, config).splitlines()
    board = lines[1:12]
    axis = lines[12]
    assert board[0].startswith("10 ") and board[-1].startswith(" 0 ")
    col = lambda x: 3 + 3 * x + 1  # noqa: E731  (row label 2 + space, cells width 2 + sep)
    assert board[0][col(10)] == "A" and board[0][col(11)] == "a"
    assert board[1][col(11)] == "a"
    assert board[-1][col(0)] == "*"
    # x labels are right-aligned to the same columns as the cells.
    for x in range(12):
        assert axis[col(x)] == str(x)[-1]
    assert axis[col(10) - 1 : col(10) + 1] == "10"
    assert axis[col(11) - 1 : col(11) + 1] == "11"
    assert all(len(row) == len(axis) for row in board)


def test_render_hazard_only_on_empty_and_food_over_hazard():
    config = GameConfig(width=3, height=1, num_snakes=1)
    state = build_state(
        config, [([(0, 0)], 100, True)], food=[(1, 0)], hazard=[(0, 0), (1, 0), (2, 0)]
    )
    assert render(state, config).splitlines()[1] == "0 A * ~"


def test_print_state(capsys):
    config, state = small_state()
    print_state(state, config)
    assert capsys.readouterr().out == EXPECTED + "\n"
