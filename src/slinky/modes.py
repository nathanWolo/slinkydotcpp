"""The four leaderboard game modes, and what a single policy needs to tell them apart.

One network can play every mode if it is told which one it is in. This module is
the single place that defines the modes, so training, the agent registry and the
evaluator cannot drift apart:

=============== ============================================= ==============
mode            game                                          default snakes
=============== ============================================= ==============
``standard``    standard ruleset and map, 3 or more snakes    4
``duel``        standard ruleset and map, 2 snakes            2
``royale``      royale ruleset and map (shrinking hazards)    4
``constrictor`` constrictor ruleset (food inert, bodies stay) 4
=============== ============================================= ==============

All modes use the 11x11 board and the default food and hazard settings, and play on
the ``standard`` map (``royale`` on the ``royale`` map).

* :func:`mode_config` / :func:`mode_of` convert between a mode name and a
  :class:`~slinky.types.GameConfig`.
* :func:`mode_features` computes the per-snake side input for the network: a mode
  one-hot plus three numbers the board image does not carry (the turn, the phase of
  the royale shrink cycle, the fraction of opponents alive). Every feature is
  available from a real game request (ruleset, settings, turn, snakes), so a policy
  sees the same information in training and in play.
* :func:`hide_food` zeroes the food channel of an observation in constrictor games,
  where food is inert (the map keeps spawning it, but eating it changes nothing), so a
  shared network does not learn "food is good" there.
"""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from slinky.observations import CHANNEL_NAMES
from slinky.types import GameConfig, Ruleset, State

# Row order is the mode id; never reorder (it indexes the one-hot and stored runs).
MODE_NAMES = ("standard", "duel", "royale", "constrictor")
# Read-only, so an override such as the trainer's ``--mode-snakes`` is built as a new dict.
DEFAULT_SNAKES = MappingProxyType({"standard": 4, "duel": 2, "royale": 4, "constrictor": 4})
FEATURE_NAMES = (
    "is_standard",
    "is_duel",
    "is_royale",
    "is_constrictor",
    "turn",
    "shrink_phase",
    "opponents_alive",
)
NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 7
# The turn feature is min(turn, TURN_SCALE) / TURN_SCALE.
TURN_SCALE = 1000

_FOOD_CHANNEL = CHANNEL_NAMES.index("food")


def _reciprocal(x: int) -> np.float32:
    return np.float32(1.0 / x)


def _check_mode(mode: str) -> None:
    if mode not in MODE_NAMES:
        raise ValueError(f"unknown game mode {mode!r}; expected one of {MODE_NAMES}")


def mode_config(
    mode: str, num_snakes: int | None = None, max_turns: int | None = None
) -> GameConfig:
    """The 11x11 :class:`GameConfig` of a leaderboard mode.

    Args:
      mode: one of :data:`MODE_NAMES`.
      num_snakes: overrides :data:`DEFAULT_SNAKES`. The count must keep the config in
        its own mode: ``duel`` has exactly 2 snakes, ``standard`` 3 or more (2 is a
        duel), ``royale`` and ``constrictor`` 2 or more.
      max_turns: optional episode truncation (the official engine has none).
    """
    _check_mode(mode)
    n = DEFAULT_SNAKES[mode] if num_snakes is None else num_snakes
    if (mode == "duel" and n != 2) or (mode == "standard" and n < 3) or n < 2:
        raise ValueError(f"mode {mode!r} cannot have {n} snakes")
    if mode == "royale":
        return GameConfig(num_snakes=n, ruleset=Ruleset.ROYALE, map="royale", max_turns=max_turns)
    if mode == "constrictor":
        return GameConfig(num_snakes=n, ruleset=Ruleset.CONSTRICTOR, max_turns=max_turns)
    return GameConfig(num_snakes=n, max_turns=max_turns)


def mode_of(config: GameConfig) -> str:
    """The leaderboard mode a game configuration plays, from its ruleset, map and snake count.

    The inverse of :func:`mode_config`; board size, food and hazard settings and
    ``max_turns`` are not inspected. Only the ``standard`` and ``royale`` maps belong to a
    mode. The royale ruleset is royale on either (the engine treats it alike, and its
    hazards follow the ruleset), and so is the standard ruleset on the royale map (the
    hazards come from the map). Constrictor is a mode on the standard map only.

    Raises:
      ValueError: for anything outside the four modes (solo, wrapped rulesets, a single
        snake, any other map such as ``empty``, constrictor on the royale map).
    """
    ruleset, n = config.ruleset, config.num_snakes
    if n >= 2 and config.map in ("standard", "royale"):
        royale_map = config.map == "royale"
        if ruleset == Ruleset.ROYALE or (ruleset == Ruleset.STANDARD and royale_map):
            return "royale"
        if not royale_map:
            if ruleset == Ruleset.CONSTRICTOR:
                return "constrictor"
            if ruleset == Ruleset.STANDARD:
                return "duel" if n == 2 else "standard"
    raise ValueError(
        f"no game mode for ruleset={ruleset.value!r} map={config.map!r} num_snakes={n}; "
        f"the modes are {MODE_NAMES}"
    )


def mode_features(state: State, config: GameConfig, mode: str | None = None) -> jax.Array:
    """Side input for the policy network: ``f32[N, NUM_FEATURES]``, one row per snake.

    Columns follow :data:`FEATURE_NAMES`:

    * the mode one-hot (static);
    * ``turn``: ``min(turn, TURN_SCALE) / TURN_SCALE``, which the board image lacks;
    * ``shrink_phase``: ``(turn % N) / N`` with ``N = config.shrink_every_n_turns`` in
      royale, else 0. Royale looks like 4-snake standard until the first shrink;
    * ``opponents_alive``: ``(sum(alive) - alive_i) / (N_snakes - 1)``.

    Args:
      state: one unbatched game state (``vmap`` over games).
      config: the game being played.
      mode: the mode the network is told; defaults to ``mode_of(config)``. Pass another
        to measure how much the network relies on the input.
    """
    mode = mode_of(config) if mode is None else mode
    _check_mode(mode)
    n = config.num_snakes
    one_hot = jnp.zeros((len(MODE_NAMES),), jnp.float32).at[MODE_NAMES.index(mode)].set(1.0)
    # Multiply by a float32 reciprocal, which is what XLA does with a constant divisor under
    # jit, so jitted and eager features agree bit for bit.
    turn = jnp.minimum(state.turn, TURN_SCALE).astype(jnp.float32) * _reciprocal(TURN_SCALE)
    if mode == "royale":
        period = config.shrink_every_n_turns
        phase = (state.turn % period).astype(jnp.float32) * _reciprocal(period)
    else:
        phase = jnp.zeros((), jnp.float32)
    alive = state.alive.astype(jnp.float32)
    opponents = (jnp.sum(alive) - alive) * _reciprocal(max(n - 1, 1))
    shared = jnp.broadcast_to(jnp.stack([turn, phase]), (n, 2))
    return jnp.concatenate(
        [jnp.broadcast_to(one_hot, (n, len(MODE_NAMES))), shared, opponents[:, None]], axis=1
    )


def hide_food(obs: jax.Array, config: GameConfig) -> jax.Array:
    """Zero the food channel of observations (channels last) when the game is constrictor.

    Follows the real game, not the mode a network is told: it hides food exactly when
    ``config.ruleset`` is constrictor or wrapped constrictor, so telling a standard game
    "constrictor" does not blind the snake to food. Any other config returns ``obs``
    unchanged.
    """
    if not config.ruleset.constrictor:
        return obs
    obs = jnp.asarray(obs)
    # A zero of the observation's own dtype: a float literal is not a safe cast into uint8 or bool.
    return obs.at[..., _FOOD_CHANNEL].set(jnp.zeros((), obs.dtype))
