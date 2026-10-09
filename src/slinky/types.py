"""Core data types: game configuration, state, and constants.

Coordinate conventions follow the official Battlesnake API:

* ``(x, y)`` with ``(0, 0)`` at the **bottom-left**; ``y`` grows upward.
* Grids are indexed ``grid[y, x]`` (so row 0 is the *bottom* row).
* Actions: ``UP=0`` (y+1), ``DOWN=1`` (y-1), ``LEFT=2`` (x-1), ``RIGHT=3`` (x+1).

Snake bodies use a *countdown grid* representation. For snake ``i``,
``state.body[i, y, x]`` is the number of tail-pops until cell ``(x, y)`` is
vacated, i.e. ``length - (smallest body index on that cell)``; 0 means the
snake does not occupy the cell. The head cell holds ``length`` and the tail
cell holds 1, or 2 if the tail segment is stacked because the snake just ate.
This makes moving (decrement + write head), eating (increment) and collision
checks dense, fixed-shape array ops. See ``docs/battlesnake/ENGINE_RULES.md``.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# --- Actions -----------------------------------------------------------------

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
NUM_ACTIONS = 4
ACTION_NAMES = ("up", "down", "left", "right")
# (dx, dy) per action, indexed by action id.
ACTION_DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))

MAX_HEALTH = 100
START_LENGTH = 3


class Cause(enum.IntEnum):
    """Elimination cause codes (mirror the engine's ``EliminatedCause`` strings)."""

    NONE = 0
    OUT_OF_HEALTH = 1  # "out-of-health"
    OUT_OF_BOUNDS = 2  # "wall-collision"
    SELF_COLLISION = 3  # "snake-self-collision"
    COLLISION = 4  # "snake-collision"
    HEAD_COLLISION = 5  # "head-collision"
    HAZARD = 6  # "hazard"


CAUSE_TO_ENGINE = {
    Cause.NONE: "",
    Cause.OUT_OF_HEALTH: "out-of-health",
    Cause.OUT_OF_BOUNDS: "wall-collision",
    Cause.SELF_COLLISION: "snake-self-collision",
    Cause.COLLISION: "snake-collision",
    Cause.HEAD_COLLISION: "head-collision",
    Cause.HAZARD: "hazard",
}
ENGINE_TO_CAUSE = {v: k for k, v in CAUSE_TO_ENGINE.items()}


class Ruleset(str, enum.Enum):
    """Official ruleset names (``game.ruleset.name`` in the API)."""

    STANDARD = "standard"
    SOLO = "solo"
    WRAPPED = "wrapped"
    CONSTRICTOR = "constrictor"
    WRAPPED_CONSTRICTOR = "wrapped_constrictor"
    ROYALE = "royale"

    @property
    def wrapped(self) -> bool:
        return self in (Ruleset.WRAPPED, Ruleset.WRAPPED_CONSTRICTOR)

    @property
    def constrictor(self) -> bool:
        return self in (Ruleset.CONSTRICTOR, Ruleset.WRAPPED_CONSTRICTOR)


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Static (hashable) game configuration.

    Everything here is a Python value fixed at trace time; changing it
    re-specializes jitted functions. Defaults match the official 1v1 "Duel"
    setup: an 11x11 standard game with two snakes.
    """

    width: int = 11
    height: int = 11
    num_snakes: int = 2
    ruleset: Ruleset = Ruleset.STANDARD
    map: str = "standard"
    food_spawn_chance: int = 15
    minimum_food: int = 1
    hazard_damage_per_turn: int = 14
    shrink_every_n_turns: int = 25
    # Optional episode truncation (the official engine has no turn limit).
    max_turns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ruleset", Ruleset(self.ruleset))
        if self.width < 1 or self.height < 1:
            raise ValueError("board dimensions must be positive")
        if self.num_snakes < 1:
            raise ValueError("need at least one snake")

    @property
    def solo(self) -> bool:
        """Game ends only when *no* snakes remain (engine: solo ruleset or 1 snake)."""
        return self.ruleset == Ruleset.SOLO or self.num_snakes == 1


class State(NamedTuple):
    """Full game state. A pytree; all leaves are fixed-shape arrays.

    ``N = num_snakes``, ``H = height``, ``W = width``.
    """

    body: jax.Array  # int16[N, H, W] countdown grid (see module docstring)
    head: jax.Array  # int32[N, 2] head (x, y); may be off-board for dead snakes
    length: jax.Array  # int32[N]
    health: jax.Array  # int32[N]
    alive: jax.Array  # bool[N]
    # int8[N] direction of the last applied move (UP for a fresh, stacked snake).
    # Equals the engine's neck->head "default move", used for invalid actions.
    last_move: jax.Array
    food: jax.Array  # bool[H, W]
    hazard: jax.Array  # int16[H, W] number of stacked hazards on each cell
    turn: jax.Array  # int32[] engine turn number (0 at reset)
    elim_cause: jax.Array  # int8[N] Cause code, 0 while alive
    elim_turn: jax.Array  # int32[N] engine "eliminatedOnTurn", 0 while alive
    done: jax.Array  # bool[] game over
    map_state: Any  # map-specific pytree (e.g. royale safe-zone bounds); () if unused


class TimeStep(NamedTuple):
    """Per-step output for all agents (leading axis ``N`` on per-agent fields)."""

    obs: Any  # observation pytree, leading axis N
    reward: jax.Array  # float32[N]
    done: jax.Array  # bool[] game over (terminated) or truncated
    truncated: jax.Array  # bool[] ended by max_turns rather than by the rules
    alive: jax.Array  # bool[N] which agents are still in the game
    action_mask: jax.Array  # bool[N, 4] moves not certainly fatal; never an all-False row
    # Observation of the state reached by the transition. Only set by
    # ``step_autoreset`` (where ``obs`` belongs to the next game when done).
    final_obs: Any = None


def empty_state(config: GameConfig, map_state: Any = ()) -> State:
    """An all-empty state with the right shapes and dtypes."""
    n, h, w = config.num_snakes, config.height, config.width
    return State(
        body=jnp.zeros((n, h, w), jnp.int16),
        head=jnp.zeros((n, 2), jnp.int32),
        length=jnp.zeros((n,), jnp.int32),
        health=jnp.zeros((n,), jnp.int32),
        alive=jnp.zeros((n,), bool),
        last_move=jnp.full((n,), UP, jnp.int8),
        food=jnp.zeros((h, w), bool),
        hazard=jnp.zeros((h, w), jnp.int16),
        turn=jnp.zeros((), jnp.int32),
        elim_cause=jnp.zeros((n,), jnp.int8),
        elim_turn=jnp.zeros((n,), jnp.int32),
        done=jnp.zeros((), bool),
        map_state=map_state,
    )
