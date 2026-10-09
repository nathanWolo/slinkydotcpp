"""ASCII rendering of a game state for debugging and logs.

Plain numpy (no jit). The top printed row is ``y = H-1`` and the bottom is
``y = 0``, matching the Battlesnake API orientation. Living snake ``i`` is drawn
with an uppercase head (``'A' + i``) and lowercase body; ``*`` is food, ``~`` is
hazard (only shown on otherwise empty cells), ``.`` is empty. Dead snakes are
not drawn on the board and are listed below it with their elimination cause.
"""

from __future__ import annotations

import jax
import numpy as np

from slinky.types import Cause, GameConfig, State


def _cause_name(code: int) -> str:
    try:
        return Cause(code).name
    except ValueError:
        return str(code)


def render(state: State, config: GameConfig | None = None) -> str:
    """Render ``state`` as a multi-line ASCII string (at most 26 snakes).

    Board dimensions are read from the state's arrays; ``config`` is accepted for
    API symmetry and is not needed.
    """
    s = jax.device_get(state)
    body, head, alive = np.asarray(s.body), np.asarray(s.head), np.asarray(s.alive)
    n, h, w = body.shape

    grid = np.full((h, w), ".", dtype="<U1")
    grid[np.asarray(s.hazard) > 0] = "~"
    grid[np.asarray(s.food)] = "*"
    for i in range(n):  # bodies first so that heads always win
        if alive[i]:
            grid[body[i] > 0] = chr(ord("a") + i)
    for i in range(n):
        x, y = int(head[i, 0]), int(head[i, 1])
        if alive[i] and 0 <= x < w and 0 <= y < h:
            grid[y, x] = chr(ord("A") + i)

    # Column width fits the widest coordinate so 2-digit labels stay aligned.
    cw, yw = len(str(w - 1)), len(str(h - 1))
    lines = [f"turn {int(s.turn)}" + ("  (game over)" if bool(s.done) else "")]
    for y in range(h - 1, -1, -1):
        cells = " ".join(c.rjust(cw) for c in grid[y])
        lines.append(f"{y:>{yw}} {cells}")
    lines.append(" " * (yw + 1) + " ".join(str(x).rjust(cw) for x in range(w)))

    length, health = np.asarray(s.length), np.asarray(s.health)
    cause = np.asarray(s.elim_cause)
    for i in range(n):
        status = "alive" if alive[i] else f"dead ({_cause_name(int(cause[i]))})"
        lines.append(f"{chr(ord('A') + i)} {status} len={int(length[i])} health={int(health[i])}")
    return "\n".join(lines)


def print_state(state: State, config: GameConfig | None = None) -> None:
    """Print ``render(state, config)``."""
    print(render(state, config))
