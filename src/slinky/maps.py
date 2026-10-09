"""Game maps: initial board setup and per-turn board updates.

A map decides where snakes start and where food and hazards appear; the
ruleset (:mod:`slinky.rules`) decides what happens when snakes move. This
mirrors the engine's ``maps.GameMap`` interface (``SetupBoard`` and
``PostUpdateBoard``). ``PreUpdateBoard`` is a no-op for every map implemented
so far, so it is omitted until a map needs it.

Random choices match the engine's *distributions*, not its exact random
stream (the engine uses Go's ``math/rand``).
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from slinky.rules import one_hot_cells, value_at
from slinky.types import MAX_HEALTH, START_LENGTH, GameConfig, State, empty_state


class GameMap(Protocol):
    """Interface implemented by every map."""

    def init_map_state(self, config: GameConfig) -> Any:
        """Map-specific pytree stored in ``State.map_state`` (``()`` if none)."""

    def setup(self, key: jax.Array, config: GameConfig) -> State:
        """Turn-0 state: snakes placed, initial food and hazards."""

    def post_update(self, key: jax.Array, state: State, config: GameConfig) -> State:
        """Runs after the rules pipeline each turn (before the turn increment)."""


# --- Snake placement -----------------------------------------------------------


def _fixed_start_points(config: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """The engine's 4 corner and 4 cardinal start points (square boards >= 7)."""
    mn, md, mx = 1, (config.width - 1) // 2, config.width - 2
    corners = np.array([[mn, mn], [mn, mx], [mx, mn], [mx, mx]], np.int32)
    cardinals = np.array([[mn, md], [md, mn], [md, mx], [mx, md]], np.int32)
    return corners, cardinals


def _place_fixed(key: jax.Array, config: GameConfig) -> jax.Array:
    k_corner, k_card, k_order = jax.random.split(key, 3)
    corners, cardinals = _fixed_start_points(config)
    corners = jax.random.permutation(k_corner, jnp.asarray(corners))
    cardinals = jax.random.permutation(k_card, jnp.asarray(cardinals))
    points = jnp.where(
        jax.random.bernoulli(k_order),
        jnp.concatenate([corners, cardinals]),
        jnp.concatenate([cardinals, corners]),
    )
    return points[: config.num_snakes]


def _quadrant_points(config: GameConfig) -> np.ndarray:
    """int32[4, 4, 2]: the engine's 4 candidate start points in each quadrant."""
    qh, qv = config.width // 2, config.height // 2
    ho, vo = qh // 3, qv // 3
    q0 = np.array([[ho, vo], [qh - ho, vo], [ho, qv - vo], [qh - ho, qv - vo]], np.int32)
    w1, h1 = config.width - 1, config.height - 1
    q1 = np.stack([w1 - q0[:, 0], q0[:, 1]], axis=1)
    q2 = np.stack([q0[:, 0], h1 - q0[:, 1]], axis=1)
    q3 = np.stack([w1 - q0[:, 0], h1 - q0[:, 1]], axis=1)
    return np.stack([q0, q1, q2, q3])


def _place_distributed(key: jax.Array, config: GameConfig) -> jax.Array:
    k_start, k_perm = jax.random.split(key)
    quads = jnp.asarray(_quadrant_points(config))
    perms = jax.vmap(lambda k: jax.random.permutation(k, 4))(jax.random.split(k_perm, 4))
    start = jax.random.randint(k_start, (), 0, 4)
    i = jnp.arange(config.num_snakes)
    quad = (start + i) % 4
    return quads[quad, perms[quad, i // 4]]


def _random_cells(key: jax.Array, valid: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Up to ``k`` distinct uniformly random cells from bool[H, W] ``valid``.

    Returns (xy int32[k, 2], ok bool[k]); ``ok`` is False where there were not
    enough valid cells.
    """
    h, w = valid.shape
    scores = jnp.where(valid.ravel(), jax.random.uniform(key, (h * w,)), -1.0)
    top, idx = jax.lax.top_k(scores, k)
    xy = jnp.stack([idx % w, idx // w], axis=1).astype(jnp.int32)
    return xy, top >= 0.0


def _place_random(key: jax.Array, config: GameConfig) -> jax.Array:
    """Random distinct cells with even x+y parity, excluding the centre."""
    ys, xs = np.mgrid[: config.height, : config.width]
    valid = (xs + ys) % 2 == 0
    valid[(config.height - 1) // 2, (config.width - 1) // 2] = False
    if valid.sum() < config.num_snakes:
        raise ValueError("not enough room to place snakes")
    xy, _ = _random_cells(key, jnp.asarray(valid), config.num_snakes)
    return xy


def place_snakes(key: jax.Array, config: GameConfig) -> jax.Array:
    """int32[N, 2] start points, following the engine's ``PlaceSnakesAutomatically``."""
    n, w = config.num_snakes, config.width
    if config.width == config.height:
        if n > 8 and w < 7:
            raise ValueError("too many snakes for this board size")
        if n <= 8 and w >= 7:
            return _place_fixed(key, config)
        if w >= 11:
            if n > 16:
                raise ValueError("too many snakes for distributed placement")
            return _place_distributed(key, config)
    return _place_random(key, config)


def snakes_at(state: State, points: jax.Array, config: GameConfig) -> State:
    """Place fresh length-3 snakes, all segments stacked on ``points[i]``."""
    cells = one_hot_cells(points, config)
    return state._replace(
        body=jnp.where(cells, START_LENGTH, 0).astype(state.body.dtype),
        head=points.astype(jnp.int32),
        length=jnp.full((config.num_snakes,), START_LENGTH, jnp.int32),
        health=jnp.full((config.num_snakes,), MAX_HEALTH, jnp.int32),
        alive=jnp.ones((config.num_snakes,), bool),
    )


# --- Food placement ------------------------------------------------------------


def spawn_mask(state: State, config: GameConfig, exclude_head_moves: bool = True) -> jax.Array:
    """bool[H, W] cells where the engine may place new food.

    Unoccupied by food or a living snake's body, and (by default) not
    orthogonally adjacent to a living snake's head. Hazards are allowed. As in
    the engine, adjacency does not wrap around the board edges.
    """
    occupied = state.food | jnp.any((state.body > 0) & state.alive[:, None, None], axis=0)
    if exclude_head_moves:
        offsets = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
        near = (state.head[:, None, :] + offsets[None]).reshape(-1, 2)
        near_alive = jnp.repeat(state.alive, 4)
        occupied |= jnp.any(one_hot_cells(near, config) & near_alive[:, None, None], axis=0)
    return ~occupied


def place_food_random(
    key: jax.Array, state: State, config: GameConfig, n: jax.Array, max_n: int
) -> State:
    """Add ``n`` (traced, ``<= max_n``) food at distinct random valid cells."""
    if max_n <= 0:
        return state
    xy, ok = _random_cells(key, spawn_mask(state, config), max_n)
    use = ok & (jnp.arange(max_n) < n)
    new = jnp.any(one_hot_cells(xy, config) & use[:, None, None], axis=0)
    return state._replace(food=state.food | new)


def _fixed_food_candidates(head: jax.Array, config: GameConfig) -> tuple[jax.Array, jax.Array]:
    """Diagonal neighbours of ``head`` that the engine allows for starting food.

    Excludes the centre, cells not on the far side of the head from the centre,
    and board corners. Returns (xy int32[4, 2], allowed bool[4]); the "not
    already food" check is applied by the caller.
    """
    cx, cy = (config.width - 1) // 2, (config.height - 1) // 2
    hx, hy = head[0], head[1]
    xy = head[None, :] + jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], jnp.int32)
    px, py = xy[:, 0], xy[:, 1]
    away = (
        ((px < hx) & (hx < cx))
        | ((cx < hx) & (hx < px))
        | ((py < hy) & (hy < cy))
        | ((cy < hy) & (hy < py))
    )
    corner = ((px == 0) | (px == config.width - 1)) & ((py == 0) | (py == config.height - 1))
    centre = (px == cx) & (py == cy)
    on_board = (px >= 0) & (px < config.width) & (py >= 0) & (py < config.height)
    return xy, away & ~corner & ~centre & on_board


def place_food_fixed(key: jax.Array, state: State, config: GameConfig) -> State:
    """The engine's ``PlaceFoodFixed``: one food diagonal to each head, one in the centre."""
    food = state.food
    small = config.width * config.height < 11 * 11
    if config.num_snakes <= 4 or not small:

        def place_one(food, inputs):
            head, k = inputs
            xy, allowed = _fixed_food_candidates(head, config)
            allowed &= ~value_at(food, xy, config)
            # Uniform choice among allowed candidates (the engine errors if none).
            pick = jax.random.categorical(k, jnp.where(allowed, 0.0, -jnp.inf))
            x, y = xy[pick, 0], xy[pick, 1]
            return food.at[y, x].set(food[y, x] | jnp.any(allowed)), None

        keys = jax.random.split(key, config.num_snakes)
        food, _ = jax.lax.scan(place_one, food, (state.head, keys))
    cx, cy = (config.width - 1) // 2, (config.height - 1) // 2
    return state._replace(food=food.at[cy, cx].set(True))


def place_initial_food(key: jax.Array, state: State, config: GameConfig) -> State:
    """The engine's ``PlaceFoodAutomatically``."""
    if config.width == config.height and config.width >= 7:
        return place_food_fixed(key, state, config)
    n = config.num_snakes
    return place_food_random(key, state, config, jnp.int32(n), n)


def spawn_food_standard(key: jax.Array, state: State, config: GameConfig) -> State:
    """Per-turn food spawning of the standard map.

    Tops food up to ``minimum_food``; otherwise spawns one food with the
    engine's probability ``(food_spawn_chance - 1) / 100`` (the engine tests
    ``100 - rand.Intn(100) < chance``, which is true for ``chance - 1`` of the
    100 outcomes).
    """
    k_chance, k_place = jax.random.split(key)
    count = jnp.sum(state.food, dtype=jnp.int32)
    roll = 100 - jax.random.randint(k_chance, (), 0, 100)
    chance = config.food_spawn_chance
    need = jnp.where(
        count < config.minimum_food,
        config.minimum_food - count,
        jnp.where((chance > 0) & (roll < chance), 1, 0),
    )
    return place_food_random(k_place, state, config, need, max(config.minimum_food, 1))


# --- Maps ----------------------------------------------------------------------


class StandardMap:
    """Standard snake placement and food spawning (map id ``standard``)."""

    def init_map_state(self, config: GameConfig) -> Any:
        return ()

    def setup(self, key: jax.Array, config: GameConfig) -> State:
        k_snakes, k_food = jax.random.split(key)
        state = empty_state(config, self.init_map_state(config))
        state = snakes_at(state, place_snakes(k_snakes, config), config)
        return place_initial_food(k_food, state, config)

    def post_update(self, key: jax.Array, state: State, config: GameConfig) -> State:
        return spawn_food_standard(key, state, config)


class EmptyMap:
    """Default snake placement with no food (map id ``empty``)."""

    def init_map_state(self, config: GameConfig) -> Any:
        return ()

    def setup(self, key: jax.Array, config: GameConfig) -> State:
        state = empty_state(config, self.init_map_state(config))
        return snakes_at(state, place_snakes(key, config), config)

    def post_update(self, key: jax.Array, state: State, config: GameConfig) -> State:
        return state


MAPS: dict[str, GameMap] = {
    "standard": StandardMap(),
    "empty": EmptyMap(),
}


def get_map(name: str) -> GameMap:
    try:
        return MAPS[name]
    except KeyError:
        raise ValueError(f"unknown map {name!r}; available: {sorted(MAPS)}") from None
