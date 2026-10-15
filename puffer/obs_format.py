"""The observation layout of the PufferLib environment, shared by the env and the policy.

Each agent's observation is one flat ``uint8`` vector:

* **board** -- slinky's egocentric image (:mod:`slinky.observations`), ``rows x cols x 13``
  in the same order (row-major, channels last, row 0 the bottom of the window), but
  holding the raw integers that slinky divides by a scale: body countdowns, health and
  length. ``board * channel_scale(width, height)`` is slinky's float32 observation.
* **mask** -- 4 bytes, 1 where the move is legal (``slinky.rules.action_mask``).

A dead agent's board is all zeros and its mask all ones. Values above 255 saturate
(a body countdown or length past 255 needs a snake longer than 255 cells), so the
board equals slinky's exactly whenever no snake is that long.
"""

from __future__ import annotations

import numpy as np

# Same names and order as slinky.observations.CHANNEL_NAMES.
CHANNEL_NAMES = (
    "on_board",
    "food",
    "hazard",
    "own_head",
    "own_body",
    "own_countdown",
    "opp_head_ge",
    "opp_head_lt",
    "opp_body",
    "opp_countdown",
    "opp_head_health",
    "own_health",
    "own_length",
)
NUM_CHANNELS = len(CHANNEL_NAMES)
MASK_SIZE = 4
# Channels divided by the number of cells, and by 100 (health).
_PER_CELL = ("own_countdown", "opp_countdown", "own_length")
_PER_HEALTH = ("opp_head_health", "own_health")


def board_shape(width: int, height: int, wrapped: bool = False) -> tuple[int, int, int]:
    """``(rows, cols, NUM_CHANNELS)``: slinky's egocentric ``obs_shape``."""
    if wrapped:
        return (height, width, NUM_CHANNELS)
    return (2 * height - 1, 2 * width - 1, NUM_CHANNELS)


def board_size(width: int, height: int, wrapped: bool = False) -> int:
    rows, cols, channels = board_shape(width, height, wrapped)
    return rows * cols * channels


def obs_size(width: int, height: int, wrapped: bool = False) -> int:
    """Length of one agent's observation vector: the board, then the move mask."""
    return board_size(width, height, wrapped) + MASK_SIZE


def channel_scale(width: int, height: int) -> np.ndarray:
    """``float32[NUM_CHANNELS]`` multipliers that turn the raw board into slinky's floats.

    These are float32 reciprocals; slinky divides instead, so the product can differ
    from slinky's value by one float32 rounding step.
    """
    scale = np.ones(NUM_CHANNELS, np.float32)
    for name in _PER_CELL:
        scale[CHANNEL_NAMES.index(name)] = np.float32(1.0) / np.float32(width * height)
    for name in _PER_HEALTH:
        scale[CHANNEL_NAMES.index(name)] = np.float32(1.0) / np.float32(100.0)
    return scale
