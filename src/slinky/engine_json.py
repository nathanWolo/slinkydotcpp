"""Conversion between :class:`State` and the engine's board-state JSON.

The dict format (used by the Go oracle in ``tools/oracle``)::

    {"width": W, "height": H, "turn": T,
     "snakes": [{"id": "s0", "body": [[x, y], ...], "health": 90,
                 "eliminated_cause": "", "eliminated_on_turn": 0}, ...],
     "food": [[x, y], ...], "hazards": [[x, y], ...]}

Snake ``i`` in the list is agent ``i``. This module uses plain numpy and is
meant for testing, debugging and talking to external tools, not for use
inside jitted code.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from slinky.types import (
    ACTION_DELTAS,
    CAUSE_TO_ENGINE,
    ENGINE_TO_CAUSE,
    UP,
    Cause,
    GameConfig,
    State,
    empty_state,
)


def body_to_countdown(body: list[list[int]], height: int, width: int) -> np.ndarray:
    """int16[H, W] countdown grid for a head-first list of [x, y] segments."""
    grid = np.zeros((height, width), np.int16)
    n = len(body)
    for i, (x, y) in enumerate(body):
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = max(grid[y, x], n - i)
    return grid


def countdown_to_body(
    grid: np.ndarray, head: tuple[int, int], length: int, wrapped: bool = False
) -> list[list[int]]:
    """Inverse of :func:`body_to_countdown` for a well-formed (living) snake.

    Walks from the head to the neighbour whose countdown is one lower; once no
    such neighbour exists, the remaining segments are stacked on the last cell.
    """
    h, w = grid.shape
    x, y = int(head[0]), int(head[1])
    body = [[x, y]]
    value = int(grid[y, x])
    while len(body) < length:
        nxt = None
        for dx, dy in ACTION_DELTAS:
            nx, ny = x + dx, y + dy
            if wrapped:
                nx, ny = nx % w, ny % h
            if 0 <= nx < w and 0 <= ny < h and int(grid[ny, nx]) == value - 1 and value > 1:
                nxt = (nx, ny)
                break
        if nxt is not None:
            x, y = nxt
            value -= 1
        body.append([x, y])
    return body


def _direction(head: list[int], neck: list[int]) -> int:
    """The engine's ``getDefaultMove`` (neck -> head direction, wrap-aware)."""
    (hx, hy), (nx, ny) = head, neck
    if hx == nx + 1:
        return 3
    if hx == nx - 1:
        return 2
    if hy == ny + 1:
        return 0
    if hy == ny - 1:
        return 1
    if hx == 0 and nx > 0:
        return 3
    if nx == 0 and hx > 0:
        return 2
    if hy == 0 and ny > 0:
        return 0
    if ny == 0 and hy > 0:
        return 1
    return UP


def state_from_engine(d: dict[str, Any], config: GameConfig, map_state: Any = ()) -> State:
    """Build a :class:`State` from an engine board-state dict."""
    from slinky.rules import is_game_over

    h, w, n = config.height, config.width, config.num_snakes
    if (d["width"], d["height"]) != (w, h) or len(d["snakes"]) != n:
        raise ValueError("engine state does not match config")
    body = np.zeros((n, h, w), np.int16)
    head = np.zeros((n, 2), np.int32)
    length = np.zeros(n, np.int32)
    health = np.zeros(n, np.int32)
    alive = np.zeros(n, bool)
    last_move = np.full(n, UP, np.int8)
    cause = np.zeros(n, np.int8)
    elim_turn = np.zeros(n, np.int32)
    for i, s in enumerate(d["snakes"]):
        segs = s["body"]
        cause[i] = ENGINE_TO_CAUSE[s.get("eliminated_cause", "")]
        alive[i] = cause[i] == Cause.NONE
        elim_turn[i] = s.get("eliminated_on_turn", 0)
        health[i] = s["health"]
        length[i] = len(segs)
        if segs:
            head[i] = segs[0]
            if len(segs) >= 2:
                last_move[i] = _direction(segs[0], segs[1])
        if alive[i]:
            body[i] = body_to_countdown(segs, h, w)
    food = np.zeros((h, w), bool)
    for x, y in d["food"]:
        food[y, x] = True
    hazard = np.zeros((h, w), np.int8)
    for x, y in d.get("hazards", []):
        # Some maps (snail_mode) keep off-board hazards; they can never matter.
        if 0 <= x < w and 0 <= y < h:
            hazard[y, x] += 1
    base = empty_state(config, map_state)
    alive_j = jnp.asarray(alive)
    return base._replace(
        body=jnp.asarray(body),
        head=jnp.asarray(head),
        length=jnp.asarray(length),
        health=jnp.asarray(health),
        alive=alive_j,
        last_move=jnp.asarray(last_move),
        food=jnp.asarray(food),
        hazard=jnp.asarray(hazard),
        turn=jnp.asarray(d["turn"], jnp.int32),
        elim_cause=jnp.asarray(cause),
        elim_turn=jnp.asarray(elim_turn),
        done=is_game_over(alive_j, config),
    )


def state_to_engine(state: State, config: GameConfig) -> dict[str, Any]:
    """Engine board-state dict for a :class:`State`.

    Eliminated snakes' bodies are not tracked; they are emitted as three
    segments stacked on the last head position (possibly off-board), which the
    engine ignores for every rule.
    """
    s = jax.device_get(state)
    h, w = config.height, config.width
    snakes = []
    for i in range(config.num_snakes):
        hx, hy = int(s.head[i, 0]), int(s.head[i, 1])
        if s.alive[i]:
            body = countdown_to_body(
                s.body[i], (hx, hy), int(s.length[i]), wrapped=config.ruleset.wrapped
            )
        else:
            body = [[hx, hy]] * 3
        snakes.append(
            {
                "id": f"s{i}",
                "body": body,
                "health": int(s.health[i]),
                "eliminated_cause": CAUSE_TO_ENGINE[Cause(int(s.elim_cause[i]))],
                "eliminated_on_turn": int(s.elim_turn[i]),
            }
        )
    ys, xs = np.nonzero(s.food)
    hazards = []
    for y, x in zip(*np.nonzero(s.hazard), strict=True):
        hazards += [[int(x), int(y)]] * int(s.hazard[y, x])
    return {
        "width": w,
        "height": h,
        "turn": int(s.turn),
        "snakes": snakes,
        "food": [[int(x), int(y)] for x, y in zip(xs, ys, strict=True)],
        "hazards": hazards,
    }
