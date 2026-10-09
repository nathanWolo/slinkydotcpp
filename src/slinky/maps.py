"""Game maps: initial board setup and per-turn board updates.

A map decides where snakes start and where food and hazards appear; the
ruleset (:mod:`slinky.rules`) decides what happens when snakes move. This
mirrors the engine's ``maps.GameMap`` interface (``SetupBoard`` and
``PostUpdateBoard``). ``PreUpdateBoard`` is a no-op for every map implemented
so far, so it is omitted until a map needs it.

Random choices match the engine's *distributions*, not its exact random
stream (the engine uses Go's ``math/rand``). Each random operation draws one
block of raw bits and derives every choice from it: on CPU every PRNG call is
a small loop, so few, larger draws are much faster than many small ones.
"""

from __future__ import annotations

import functools
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

MAX_PLAYERS = 16  # the engine's standard and empty maps reject more snakes


def _fixed_start_points(config: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """The engine's 4 corner and 4 cardinal start points (square boards >= 7)."""
    mn, md, mx = 1, (config.width - 1) // 2, config.width - 2
    corners = np.array([[mn, mn], [mn, mx], [mx, mn], [mx, mx]], np.int32)
    cardinals = np.array([[mn, md], [md, mn], [md, mx], [mx, md]], np.int32)
    return corners, cardinals


def _scores(bits: jax.Array) -> jax.Array:
    """Positive int32 random scores from uint32 bits (0 is reserved for "invalid")."""
    return (bits >> 2).astype(jnp.int32) + 1


def _place_fixed(bits: jax.Array, config: GameConfig) -> jax.Array:
    """Shuffle corners and cardinals; corners first or cardinals first, 50/50."""
    corners, cardinals = _fixed_start_points(config)
    corners = jnp.asarray(corners)[jnp.argsort(bits[0:4])]
    cardinals = jnp.asarray(cardinals)[jnp.argsort(bits[4:8])]
    points = jnp.where(
        (bits[8] & 1) == 1,
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


def _place_distributed(bits: jax.Array, config: GameConfig) -> jax.Array:
    """Cycle through the quadrants from a random one, a random free point in each."""
    quads = jnp.asarray(_quadrant_points(config))
    perms = jnp.argsort(bits[:16].reshape(4, 4), axis=1)
    start = bits[16] % 4
    i = jnp.arange(config.num_snakes)
    quad = (start + i) % 4
    return quads[quad, perms[quad, i // 4]]


def _random_cells(bits: jax.Array, valid: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Up to ``k`` distinct uniformly random cells from bool[H, W] ``valid``.

    ``bits`` is uint32[H*W]. Returns (xy int32[k, 2], ok bool[k]); ``ok`` is
    False where there were not enough valid cells.
    """
    h, w = valid.shape
    scores = jnp.where(valid.ravel(), _scores(bits), 0)
    if k == 1:
        idx = jnp.argmax(scores)[None]
        top = scores[idx]
    else:
        top, idx = jax.lax.top_k(scores, k)
    xy = jnp.stack([idx % w, idx // w], axis=1).astype(jnp.int32)
    return xy, top > 0


def _random_start_cells(config: GameConfig) -> np.ndarray:
    """Cells the engine's random placement may use: even x+y parity, not the centre."""
    ys, xs = np.mgrid[: config.height, : config.width]
    valid = (xs + ys) % 2 == 0
    valid[(config.height - 1) // 2, (config.width - 1) // 2] = False
    if valid.sum() < config.num_snakes:
        raise ValueError("not enough room to place snakes")
    return valid


def placement_kind(config: GameConfig) -> str:
    """Which of the engine's ``PlaceSnakesAutomatically`` strategies applies."""
    n, w = config.num_snakes, config.width
    if n > MAX_PLAYERS:
        raise ValueError(f"the standard maps allow at most {MAX_PLAYERS} snakes")
    if config.width == config.height:
        if n > 8 and w < 7:
            raise ValueError("too many snakes for this board size")
        if n <= 8 and w >= 7:
            return "fixed"
        if w >= 11:
            return "distributed"
    return "random"


def _placement_bits(config: GameConfig) -> int:
    kind = placement_kind(config)
    return {"fixed": 9, "distributed": 17}.get(kind, config.width * config.height)


def place_snakes(bits: jax.Array, config: GameConfig) -> jax.Array:
    """int32[N, 2] start points from uint32 ``bits[_placement_bits(config)]``."""
    kind = placement_kind(config)
    if kind == "fixed":
        return _place_fixed(bits, config)
    if kind == "distributed":
        return _place_distributed(bits, config)
    xy, _ = _random_cells(bits, jnp.asarray(_random_start_cells(config)), config.num_snakes)
    return xy


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
        heads = jnp.any(one_hot_cells(state.head, config) & state.alive[:, None, None], axis=0)
        p = jnp.pad(heads, 1)  # dilate by one cell in the 4 directions, no wrapping
        occupied |= p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return ~occupied


def place_food_random(
    bits: jax.Array, state: State, config: GameConfig, n: jax.Array, max_n: int
) -> State:
    """Add ``n`` (traced, ``<= max_n``) food at distinct random valid cells.

    ``bits`` is uint32[H*W].
    """
    if max_n <= 0:
        return state
    xy, ok = _random_cells(bits, spawn_mask(state, config), max_n)
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


@functools.cache
def _fixed_candidates_disjoint(config: GameConfig) -> bool:
    """Whether no two fixed start points share an allowed starting-food cell.

    Then each snake's choice is independent of the others' (no "already food"
    exclusions), so the engine's sequential placement can be vectorized.
    Evaluated in numpy at trace time.
    """
    if placement_kind(config) != "fixed":
        return False
    cx, cy = (config.width - 1) // 2, (config.height - 1) // 2
    w, h = config.width, config.height

    def candidates(hx: int, hy: int) -> set[tuple[int, int]]:
        cells = set()
        for px, py in ((hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1), (hx + 1, hy + 1)):
            away = (px < hx < cx) or (cx < hx < px) or (py < hy < cy) or (cy < hy < py)
            corner = px in (0, w - 1) and py in (0, h - 1)
            on_board = 0 <= px < w and 0 <= py < h
            if away and not corner and (px, py) != (cx, cy) and on_board:
                cells.add((px, py))
        return cells

    corners, cardinals = _fixed_start_points(config)
    # With <= 4 snakes, all start on corners or all on cardinals.
    groups = (
        [corners, cardinals] if config.num_snakes <= 4 else [np.concatenate([corners, cardinals])]
    )
    for group in groups:
        seen: set[tuple[int, int]] = set()
        for hx, hy in group.tolist():
            cells = candidates(hx, hy)
            if cells & seen:
                return False
            seen |= cells
    return True


def place_food_fixed(bits: jax.Array, state: State, config: GameConfig) -> State:
    """The engine's ``PlaceFoodFixed``: one food diagonal to each head, one in the centre.

    ``bits`` is uint32[4 * N]: random scores for each snake's 4 candidates.
    """
    food = state.food
    small = config.width * config.height < 11 * 11
    scores = _scores(bits).reshape(config.num_snakes, 4)
    if config.num_snakes <= 4 or not small:
        xy, allowed = jax.vmap(lambda h: _fixed_food_candidates(h, config))(state.head)
        if _fixed_candidates_disjoint(config):
            # Uniform choice among each snake's allowed candidates.
            pick = jnp.argmax(jnp.where(allowed, scores, 0), axis=1)
            chosen = jnp.take_along_axis(xy, pick[:, None, None], axis=1)[:, 0]
            ok = jnp.any(allowed, axis=1)
            food |= jnp.any(one_hot_cells(chosen, config) & ok[:, None, None], axis=0)
        else:

            def place_one(food, inputs):
                xy, allowed, score = inputs
                allowed &= ~value_at(food, xy, config)
                # Uniform choice among allowed candidates (the engine errors if none).
                pick = jnp.argmax(jnp.where(allowed, score, 0))
                x, y = xy[pick, 0], xy[pick, 1]
                return food.at[y, x].set(food[y, x] | jnp.any(allowed)), None

            food, _ = jax.lax.scan(place_one, food, (xy, allowed, scores))
    cx, cy = (config.width - 1) // 2, (config.height - 1) // 2
    return state._replace(food=food.at[cy, cx].set(True))


def _initial_food_bits(config: GameConfig) -> int:
    if config.width == config.height and config.width >= 7:
        return 4 * config.num_snakes
    return config.width * config.height


def place_initial_food(bits: jax.Array, state: State, config: GameConfig) -> State:
    """The engine's ``PlaceFoodAutomatically``, from uint32 ``bits[_initial_food_bits]``."""
    if config.width == config.height and config.width >= 7:
        return place_food_fixed(bits, state, config)
    n = min(config.num_snakes, config.width * config.height)
    return place_food_random(bits, state, config, jnp.int32(n), n)


def spawn_food_standard(key: jax.Array, state: State, config: GameConfig) -> State:
    """Per-turn food spawning of the standard map.

    Tops food up to ``minimum_food``; otherwise spawns one food with the
    engine's probability ``(food_spawn_chance - 1) / 100`` (the engine tests
    ``100 - rand.Intn(100) < chance``, which is true for ``chance - 1`` of the
    100 outcomes).
    """
    bits = jax.random.bits(key, (1 + config.width * config.height,), jnp.uint32)
    count = jnp.sum(state.food, dtype=jnp.int32)
    roll = 100 - (bits[0] % 100).astype(jnp.int32)  # modulo bias ~1e-8, negligible
    chance = config.food_spawn_chance
    need = jnp.where(
        count < config.minimum_food,
        config.minimum_food - count,
        jnp.where((chance > 0) & (roll < chance), 1, 0),
    )
    max_n = min(max(config.minimum_food, 1), config.width * config.height)
    return place_food_random(bits[1:], state, config, need, max_n)


# --- Maps ----------------------------------------------------------------------


class StandardMap:
    """Standard snake placement and food spawning (map id ``standard``)."""

    def init_map_state(self, config: GameConfig) -> Any:
        return ()

    def setup(self, key: jax.Array, config: GameConfig) -> State:
        n_place = _placement_bits(config)
        bits = jax.random.bits(key, (n_place + _initial_food_bits(config),), jnp.uint32)
        state = empty_state(config, self.init_map_state(config))
        state = snakes_at(state, place_snakes(bits[:n_place], config), config)
        return place_initial_food(bits[n_place:], state, config)

    def post_update(self, key: jax.Array, state: State, config: GameConfig) -> State:
        return spawn_food_standard(key, state, config)


class EmptyMap:
    """Default snake placement with no food (map id ``empty``)."""

    def init_map_state(self, config: GameConfig) -> Any:
        return ()

    def setup(self, key: jax.Array, config: GameConfig) -> State:
        bits = jax.random.bits(key, (_placement_bits(config),), jnp.uint32)
        state = empty_state(config, self.init_map_state(config))
        return snakes_at(state, place_snakes(bits, config), config)

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
