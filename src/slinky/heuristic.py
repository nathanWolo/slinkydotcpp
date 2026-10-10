"""A hand-written heuristic snake: a benchmark opponent and an MCTS leaf evaluator.

The module has three layers. Each can be used on its own, and all of them are
pure, fixed-shape functions of one (unbatched) game that ``jit`` and ``vmap``.

1. **Grid utilities** (:func:`neighbours`, :func:`distances`,
   :func:`flood_fill`, :func:`free_after`, :func:`arrival_times`,
   :func:`voronoi`). These are breadth-first fills on ``[..., H, W]`` boolean
   grids, built from pad-and-shift 4-neighbour dilation (``jnp.roll`` on
   wrapped boards), as in ``maps.spawn_mask``. :func:`arrival_times` is the
   time-aware fill. It reads the countdown grid (:func:`free_after`): a body
   cell with countdown ``k`` can be entered on the ``d``-th move from now iff
   ``k <= d``, because that snake's tail has passed it by then (assuming it
   doesn't eat). Treating bodies as permanent walls instead plays much worse.

2. **A static evaluator**, :func:`evaluate`, which gives each snake a value in
   ``[-1, 1]``. Terminal states get the exact outcome, as in
   ``env.win_loss_reward``. Other states get ``tanh`` of a score difference
   (:class:`Weights`, :func:`snake_terms`). The score is built from Voronoi
   territory (cells a snake reaches strictly first, with ties going to the
   longer snake), food inside that territory, length, a starvation term and a
   trap term. In a duel the value is exactly antisymmetric. All snakes' fills
   run together as a race on bit-packed rows (one ``uint32`` per board row),
   which gives the same numbers as :func:`arrival_times` plus :func:`voronoi`
   about 4x faster. One evaluation costs about as much as 1.5 calls to
   ``env.step``, so it is cheap enough for every MCTS leaf.

3. **A policy**, :func:`heuristic_policy` (and :func:`heuristic`, the cached
   ``evaluate.Policy``). Each snake ranks its candidate moves
   lexicographically:

   * tier 1, legal: ``env.action_mask``, minus moves that certainly starve;
   * tier 2, no losing head-to-head: no opponent reply kills the snake while
     the opponent survives;
   * tier 3, fits: after the move, the time-aware reachable area holds the
     snake, or its own tail can be reached (trap avoidance);
   * then a strategic score.

   In a duel the tiers and the score come from a one-ply simultaneous-move
   search over the exact rules (:func:`duel_scores`). All 16 joint moves
   ``(a, b)`` are applied with ``rules.rules_step`` and evaluated, and move
   ``a`` scores ``min_b M[a, b] + mean_weight * mean_b M[a, b]`` over the
   opponent's legal replies ``b``. A mutual elimination is worth
   ``-contempt``, so the snake risks an equal-length head-on only when its
   other moves look worse than that.
   This one matrix covers both kinds of head-to-head exactly: danger from an
   equal-or-longer snake, and the chance to kill a shorter one. With other
   numbers of snakes (solo, or three or more), the tiers and a weighted sum
   come from per-move features instead (:func:`move_features`), with no
   simulation.

   Exact ties are broken uniformly at random with the policy's key.

The duel policy is the "P1" design of
``docs/research/battlesnake_heuristics.md``, with tiers added. Only the 11x11
duel on the ``standard`` ruleset was tuned. Wrapped boards and constrictor
(where bodies never shrink) are handled in the fills. Constrictor has no food
and keeps health full, so the hunger term there only charges small spaces
(see :func:`hunger`). It is kept on purpose: dropping it scored 0.436 +- 0.040
against the snake that keeps it (256 constrictor duels). Hazards only enter
through the exact rules step of the duel search.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from slinky import rules
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy
from slinky.types import ACTION_DELTAS, NUM_ACTIONS, GameConfig, State, TimeStep

INF = 1000  # distance / arrival time of cells that are not reached
_NEVER = 30_000  # countdown of cells that never free up (constrictor bodies)
_DELTAS = jnp.array(ACTION_DELTAS, jnp.int32)  # [4, 2] (dx, dy)
_ALL_BITS = 0xFFFFFFFF
_MAX_PACKED_WIDTH = 32  # boards up to 32 wide use bit-packed rows

# --- Grid utilities ---------------------------------------------------------------


def neighbours(mask: jax.Array, wrapped: bool = False) -> jax.Array:
    """bool[..., H, W]: cells orthogonally adjacent to a True cell of ``mask``.

    Operates on the last two axes (``[y, x]``). On a wrapped board the
    neighbourhood wraps around the edges; otherwise off-board cells are ignored.
    """
    if wrapped:
        return (
            jnp.roll(mask, 1, axis=-2)
            | jnp.roll(mask, -1, axis=-2)
            | jnp.roll(mask, 1, axis=-1)
            | jnp.roll(mask, -1, axis=-1)
        )
    p = jnp.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(1, 1), (1, 1)])
    return p[..., :-2, 1:-1] | p[..., 2:, 1:-1] | p[..., 1:-1, :-2] | p[..., 1:-1, 2:]


def distances(
    passable_at: jax.Array | Callable[[jax.Array], jax.Array],
    sources: jax.Array,
    max_dist: int,
    *,
    start: int = 0,
    wrapped: bool = False,
    wait: bool = False,
) -> jax.Array:
    """int32[..., H, W] breadth-first distance in moves from the nearest source cell.

    Args:
      passable_at: bool[..., H, W] cells that can be entered, or a function
        ``d -> bool[..., H, W]`` giving the cells that can be entered on move
        ``d`` (``d`` is a traced int32), for obstacles that change over time.
      sources: bool[..., H, W] start cells, at distance ``start``. A source is
        never tested for passability.
      max_dist: the largest distance computed (a Python int). Cells that are
        farther, or not reachable, get :data:`INF`.
      start: the distance of the sources (Python int). Use 1 for "after the
        move to this cell".
      wrapped: neighbours wrap around the board edges.
      wait: only matters when ``passable_at`` changes over time. If False (the
        default), only the newest frontier expands. A cell that is blocked when
        a neighbour first reaches it is not tried again from that neighbour,
        because snakes cannot stop. If True, every reached cell keeps
        expanding, as if the mover could wait in place.

    Leading axes broadcast, so ``[N, H, W]`` sources fill ``N`` maps at once.
    The loop has a fixed length of ``max_dist - start``.
    """
    sources = jnp.asarray(sources, bool)
    if callable(passable_at):
        open_at = passable_at
    else:
        static = jnp.asarray(passable_at, bool)

        def open_at(d: jax.Array) -> jax.Array:
            return static

    dist0 = jnp.where(sources, start, INF).astype(jnp.int32)

    def body(d, carry):
        dist, frontier = carry
        grow_from = (dist < INF) if wait else frontier
        new = neighbours(grow_from, wrapped) & (dist == INF) & open_at(d)
        return jnp.where(new, d, dist), new

    dist, _ = jax.lax.fori_loop(start + 1, max_dist + 1, body, (dist0, sources))
    return dist


def flood_fill(
    passable: jax.Array, seeds: jax.Array, steps: int, wrapped: bool = False
) -> jax.Array:
    """bool[..., H, W]: cells reachable from ``seeds`` in at most ``steps`` moves.

    Moves go through ``passable`` cells only. Seeds are always included.
    """
    return distances(passable, seeds, steps, wrapped=wrapped) < INF


def free_after(state: State, config: GameConfig) -> jax.Array:
    """int32[H, W]: moves until each cell is vacated by every living snake (0 = free now).

    This is the max over living snakes of the countdown grid. Under the
    constrictor rules snakes grow every turn and never vacate a cell, so every
    occupied cell gets a huge value.
    """
    tf = jnp.max(jnp.where(state.alive[:, None, None], state.body, 0), axis=0).astype(jnp.int32)
    if config.ruleset.constrictor:
        tf = jnp.where(tf > 0, _NEVER, 0)
    return tf


def arrival_times(
    countdown: jax.Array,
    sources: jax.Array,
    max_dist: int,
    *,
    start: int = 0,
    wrapped: bool = False,
    wait: bool = False,
) -> jax.Array:
    """int32[..., H, W] earliest move on which each cell can be entered (time-aware BFS).

    A cell can be entered on move ``d`` iff ``countdown <= d``. Use
    :func:`free_after` for ``countdown``: a body cell becomes free on the move
    its tail leaves it. This also gives the tail rule: countdown 1 can be
    entered on the next move, but a stacked tail (countdown 2) cannot. Other
    arguments are as in :func:`distances`. Two simplifications: snakes that
    eat keep their tail one move longer than assumed, and other snakes' future
    head positions are ignored.
    """
    countdown = jnp.asarray(countdown)
    return distances(
        lambda d: countdown <= d, sources, max_dist, start=start, wrapped=wrapped, wait=wait
    )


def voronoi(arrival: jax.Array, lengths: jax.Array) -> jax.Array:
    """bool[N, H, W]: cells each snake claims, following the head-to-head rule.

    ``arrival`` is int32[N, H, W] (INF where a snake cannot get to a cell; dead
    snakes should be INF everywhere), and ``lengths`` is int32[N]. A cell goes to
    the snake that arrives first. If several arrive together, it goes to the
    strictly longest of them, and to nobody if that is tied.
    """
    first = jnp.min(arrival, axis=0)
    at_first = (arrival == first) & (first < INF)
    lens = jnp.where(at_first, lengths[:, None, None], -1)
    longest = jnp.max(lens, axis=0)
    unique = jnp.sum(lens == longest, axis=0) == 1
    return at_first & (lens == longest) & unique


def _cells(xy: jax.Array, config: GameConfig) -> jax.Array:
    """bool[..., H, W] one-hot grids for points ``xy[..., 2]`` (all-False off the board)."""
    ys = jnp.arange(config.height)[:, None]
    xs = jnp.arange(config.width)[None, :]
    return (ys == xy[..., 1, None, None]) & (xs == xy[..., 0, None, None])


def _at(grid: jax.Array, xy: jax.Array, config: GameConfig) -> jax.Array:
    """``grid[..., y, x]`` for points ``xy[P, 2]``: returns ``[..., P]``, 0/False off the board."""
    x = jnp.clip(xy[:, 0], 0, config.width - 1)
    y = jnp.clip(xy[:, 1], 0, config.height - 1)
    vals = grid[..., y, x]
    return jnp.where(rules.in_bounds(xy, config), vals, jnp.zeros_like(vals))


# --- The race: all snakes' time-aware fills at once -----------------------------------


class FillStats(NamedTuple):
    """Per-snake results of one simultaneous time-aware fill (all ``[N]``)."""

    space: jax.Array  # int32: cells reachable within the fill (including the start cell)
    territory: jax.Array  # int32: cells claimed (see :func:`voronoi`)
    food_territory: jax.Array  # int32: food cells claimed
    dist_food: jax.Array  # int32: arrival time of the nearest reachable food, INF if none
    tail_reachable: jax.Array  # bool: the given tail cell is reachable


def fill_stats(
    countdown: jax.Array,
    heads: jax.Array,
    lengths: jax.Array,
    food: jax.Array,
    tails: jax.Array,
    steps: int,
    *,
    wrapped: bool = False,
    wait: bool = False,
) -> FillStats:
    """Space, Voronoi territory and food distance of ``N`` snakes from one time-aware race.

    Equivalent to running :func:`arrival_times` from each row of ``heads``
    (bool[N, H, W], all-False for dead snakes) and splitting the cells with
    :func:`voronoi`. ``tails`` is int32[N, 2], the (x, y) cell whose
    reachability is reported. Boards up to 32 wide use bit-packed rows, which
    is several times faster on CPU; wider boards use the boolean grids.
    """
    if heads.shape[-1] > _MAX_PACKED_WIDTH:
        return _fill_stats_grid(countdown, heads, lengths, food, tails, steps, wrapped, wait)
    return _fill_stats_packed(countdown, heads, lengths, food, tails, steps, wrapped, wait)


def _fill_stats_grid(countdown, heads, lengths, food, tails, steps, wrapped, wait) -> FillStats:
    """Reference implementation of :func:`fill_stats` with arrival-time grids."""
    arr = arrival_times(countdown, heads, steps, wrapped=wrapped, wait=wait)
    claims = voronoi(arr, lengths)
    reach = arr < INF
    h, w = food.shape
    tx, ty = jnp.clip(tails[:, 0], 0, w - 1), jnp.clip(tails[:, 1], 0, h - 1)
    return FillStats(
        space=jnp.sum(reach, axis=(1, 2), dtype=jnp.int32),
        territory=jnp.sum(claims, axis=(1, 2), dtype=jnp.int32),
        food_territory=jnp.sum(claims & food, axis=(1, 2), dtype=jnp.int32),
        dist_food=jnp.min(jnp.where(food, arr, INF), axis=(1, 2)),
        tail_reachable=reach[jnp.arange(reach.shape[0]), ty, tx],
    )


def _pack(grid: jax.Array) -> jax.Array:
    """uint32[..., H]: each row of a bool[..., H, W] grid as a bit mask (bit x is column x)."""
    bits = jnp.left_shift(jnp.uint32(1), jnp.arange(grid.shape[-1], dtype=jnp.uint32))
    return jnp.sum(jnp.where(grid, bits, jnp.uint32(0)), axis=-1, dtype=jnp.uint32)


def _packed_neighbours(rows: jax.Array, width: int, wrapped: bool) -> jax.Array:
    """:func:`neighbours` on bit-packed rows uint32[..., H, B] (board rows, then games)."""
    full = jnp.uint32((1 << width) - 1)
    if wrapped:
        left = (rows << 1) | (rows >> (width - 1))
        right = (rows >> 1) | (rows << (width - 1))
        return ((left | right) & full) | jnp.roll(rows, 1, axis=-2) | jnp.roll(rows, -1, axis=-2)
    zero = jnp.zeros_like(rows[..., :1, :])
    up = jnp.concatenate([zero, rows[..., :-1, :]], axis=-2)
    down = jnp.concatenate([rows[..., 1:, :], zero], axis=-2)
    return (((rows << 1) | (rows >> 1)) & full) | up | down


def _fill_stats_packed(countdown, heads, lengths, food, tails, steps, wrapped, wait) -> FillStats:
    """:func:`fill_stats` on bit-packed rows: one uint32 per board row and snake.

    One game is run as a batch of one (see :func:`_packed_fill`).
    """
    fill = _packed_fill(steps, wrapped, wait)
    out = fill(*(x[None] for x in (countdown, heads, lengths, food, tails)))
    return FillStats(*(x[0] for x in out))


@functools.cache
def _packed_fill(steps: int, wrapped: bool, wait: bool) -> Callable[..., tuple[jax.Array, ...]]:
    """The packed fill of a batch of games (a leading axis on every argument).

    It is a ``custom_vmap``. A plain ``vmap`` would put its axis in front, which
    leaves the dozen rows of one board as the fastest-varying axis. Instead, a
    ``vmap`` of this function (nested ones too) folds its axis into the batch,
    and the race runs with the games on the last axis, so XLA vectorizes across
    games. That makes a whole MCTS search 15-20% faster on CPU, with the same
    integer results.
    """

    @custom_vmap
    def fill(countdown, heads, lengths, food, tails):
        d = jnp.arange(1, steps + 1)
        open_rows = _pack(countdown[:, None] <= d[None, :, None, None])  # [B, steps, H]
        args = (open_rows, _pack(heads), lengths, _pack(food), tails)
        # Games last. The barrier keeps XLA from fusing these transposes into the loop.
        args = jax.lax.optimization_barrier(tuple(jnp.moveaxis(x, 0, -1) for x in args))
        out = _packed_race(*args, countdown.shape[-1], wrapped, wait)
        return tuple(jnp.moveaxis(x, -1, 0) for x in out)

    @fill.def_vmap
    def fold(axis_size, in_batched, *args):
        args = [
            x if batched else jnp.broadcast_to(x, (axis_size, *x.shape))
            for x, batched in zip(args, in_batched, strict=True)
        ]
        out = fill(*(x.reshape(-1, *x.shape[2:]) for x in args))
        return tuple(x.reshape(axis_size, -1, *x.shape[1:]) for x in out), (True,) * len(out)

    return fill


def _packed_race(open_rows, seeds, lengths, food_rows, tails, width, wrapped, wait):
    """The race of :func:`_fill_stats_packed`, for ``B`` games on the last axis.

    Each step, every snake's new cells are ``neighbours(frontier) & ~visited &
    open``, where ``open`` is ``countdown <= d``. A new cell that nobody reached
    earlier is claimed by the snake reaching it, unless another snake that is
    at least as long reaches it on the same step.

    Takes the open cells of each step ``open_rows`` uint32[steps, H, B], the
    packed heads ``seeds`` uint32[N, H, B], ``lengths`` int32[N, B],
    ``food_rows`` uint32[H, B] and ``tails`` int32[N, 2, B]. Returns the
    :class:`FillStats` fields, each ``[N, B]``.
    """
    steps, h, b = open_rows.shape
    n = seeds.shape[0]
    alive = jnp.any(seeds != 0, axis=1)

    def any_snake(rows: jax.Array) -> jax.Array:  # [N, H, B]: OR over snakes, broadcast
        merged = functools.reduce(jnp.bitwise_or, [rows[j] for j in range(n)])
        return jnp.broadcast_to(merged, rows.shape)

    # beats[i, j]: snake j takes a cell from snake i when both reach it on the same step.
    beats = ~jnp.eye(n, dtype=bool)[:, :, None] & alive[None] & (lengths[None] >= lengths[:, None])
    beat_masks = jnp.where(beats, jnp.uint32(_ALL_BITS), jnp.uint32(0))

    def claim(fresh: jax.Array) -> jax.Array:
        beaten = functools.reduce(
            jnp.bitwise_or, [fresh[j][None] & beat_masks[:, j, None] for j in range(n)]
        )
        return fresh & ~beaten

    food_rows = jnp.broadcast_to(food_rows, (n, h, b))
    # Pre-broadcasting the open cells to every snake keeps broadcasts out of the loop
    # body, which is noticeably faster on CPU.
    open_rows = jnp.broadcast_to(open_rows[:, None], (steps, n, h, b))
    d_all = jnp.arange(1, steps + 1)
    dist0 = jnp.where(jnp.any((seeds & food_rows) != 0, axis=1), 0, INF)

    def step(carry, x):
        d, open_d = x
        visited, frontier, claimed, reached, dist_food = carry
        grow_from = visited if wait else frontier
        new = _packed_neighbours(grow_from, width, wrapped) & ~visited & open_d
        claimed = claimed | claim(new & ~reached)
        hit = jnp.any((new & food_rows) != 0, axis=1)
        dist_food = jnp.where((dist_food == INF) & hit, d, dist_food)
        return (visited | new, new, claimed, reached | any_snake(new), dist_food), None

    init = (seeds, seeds, claim(seeds), any_snake(seeds), dist0)
    (visited, _, claimed, _, dist_food), _ = jax.lax.scan(step, init, (d_all, open_rows))

    def count(rows: jax.Array) -> jax.Array:
        return jnp.sum(jax.lax.population_count(rows), axis=1, dtype=jnp.int32)

    tx = jnp.clip(tails[:, 0], 0, width - 1).astype(jnp.uint32)
    ty = jnp.clip(tails[:, 1], 0, h - 1)
    tail_row = jnp.take_along_axis(visited, ty[:, None], axis=1)[:, 0]
    return (
        count(visited),
        count(claimed),
        count(claimed & food_rows),
        dist_food.astype(jnp.int32),
        ((tail_row >> tx) & 1) == 1,
    )


# --- Static evaluator ----------------------------------------------------------------


class Weights(NamedTuple):
    """Coefficients of the evaluator and the policy (Python numbers, hashable).

    Evaluator (per snake ``i``, see :func:`snake_terms`):

    ``score_i = territory * T_i + food_territory * F_i + length * L_i
    - hunger * hun_i - shortfall * short_i``. Then
    ``value_i = clip(tanh(sharpness * (score_i - max_{j != i} score_j)), -0.99, 0.99)``,
    where ``j`` ranges over living opponents.

    Policy:

    * ``contempt``: a mutual elimination (draw) is worth ``-contempt`` inside
      the duel lookahead. At 0 the snake trades heads freely, and a mirror
      match is nearly all draws. Higher values make it more passive.
    * ``mean_weight``: a move scores ``min + mean_weight * mean`` over the
      opponent's replies (0 is pure maximin).
    * ``greedy_*``: the weighted sum used when the game is not a duel (see
      :func:`greedy_scores`).
    * ``fill_steps``: length of every flood fill; ``None`` means ``H + W``.
    * ``fill_wait``: the fills' ``wait`` flag (see :func:`distances`). Letting
      fills wait played clearly worse in the duel.

    The evaluator coefficients start from the research note's fit
    (``docs/research/battlesnake_heuristics.md``: territory 0.40, food
    territory 0.35, length 0.15, hunger 0.50, shortfall 0.50, contempt 0.4,
    mean weight 0.25). A held-out tuning pass in the 11x11 duel then changed
    three of them: length 0.15 -> 0.25, contempt 0.4 -> 0.25, and mean weight
    0.25 -> 0.5. Each candidate played 1,024 games against the original
    weights, the feature-only snake and the DQN. Scores against the DQN stayed
    level, at about 0.95. The new weights won more often against the other two
    opponents. Lower contempt and a larger length weight both make the snake
    more willing to contest cells, and passive settings lost to aggressive
    opponents. The other coefficients, and both safety tiers, were flat within
    noise. The cost: two copies of the default snake trade heads early, so
    about 90% of mirror games are draws. With ``contempt=0.4`` mirror games
    last longer and about 70% are draws, at a small cost in strength.
    """

    territory: float = 0.40
    food_territory: float = 0.35
    length: float = 0.25
    hunger: float = 0.50
    shortfall: float = 0.50
    sharpness: float = 1.0
    contempt: float = 0.25
    mean_weight: float = 0.5
    greedy_territory: float = 1.5
    greedy_food_territory: float = 2.0
    greedy_hunger: float = 1.5
    greedy_eat: float = 2.0
    greedy_space: float = 0.25
    greedy_kill: float = 0.5
    fill_steps: int | None = None
    fill_wait: bool = False


DEFAULT_WEIGHTS = Weights()


def _fill_steps(config: GameConfig, weights: Weights) -> int:
    return config.height + config.width if weights.fill_steps is None else weights.fill_steps


def hunger(health: jax.Array, dist_food: jax.Array, space: jax.Array) -> jax.Array:
    """float32 starvation pressure: 0 when comfortable, 1 when about to starve.

    ``margin = health - dist_food``, the health left on reaching the nearest
    reachable food. If no food is reachable, ``margin = min(health, space) - 5``.
    The pressure is ``clip((30 - margin) / 30, 0, 1) ** 2``. With full health
    and no food (always the case under constrictor) it is a penalty on spaces
    smaller than 35 cells.
    """
    health = health.astype(jnp.float32)
    margin = jnp.where(
        dist_food < INF,
        health - dist_food.astype(jnp.float32),
        jnp.minimum(health, space.astype(jnp.float32)) - 5.0,
    )
    return jnp.clip((30.0 - margin) / 30.0, 0.0, 1.0) ** 2


class SnakeTerms(NamedTuple):
    """Per-snake evaluator terms (all ``[N]``; meaningless for dead snakes)."""

    space: jax.Array  # int32: cells reachable by the time-aware fill (including the head)
    tail_reachable: jax.Array  # bool: the fill reaches the snake's own tail cell
    territory: jax.Array  # float32: cells claimed / (H * W)
    food_territory: jax.Array  # float32: food claimed / max(1, food on the board)
    dist_food: jax.Array  # int32: moves to the nearest reachable food, INF if none
    hunger: jax.Array  # float32 in [0, 1], see :func:`hunger`
    shortfall: jax.Array  # float32 in [0, 1]: clip(1 - space / length, 0, 1)
    score: jax.Array  # float32: the weighted sum (see :class:`Weights`)


def tail_cells(state: State, config: GameConfig) -> jax.Array:
    """int32[N, 2] each snake's tail (x, y): its occupied cell with the smallest countdown."""
    body = state.body.reshape(config.num_snakes, -1)
    idx = jnp.argmin(jnp.where(body > 0, body, jnp.iinfo(body.dtype).max), axis=1)
    return jnp.stack([idx % config.width, idx // config.width], axis=1).astype(jnp.int32)


def snake_terms(state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS) -> SnakeTerms:
    """The evaluator's terms for every snake, from one time-aware race of all snakes."""
    h, w = config.height, config.width
    heads = _cells(state.head, config) & state.alive[:, None, None]
    stats = fill_stats(
        free_after(state, config),
        heads,
        state.length,
        state.food,
        tail_cells(state, config),
        _fill_steps(config, weights),
        wrapped=config.ruleset.wrapped,
        wait=weights.fill_wait,
    )
    territory = stats.territory / (h * w)
    food_territory = stats.food_territory / jnp.maximum(1, jnp.sum(state.food))
    hun = hunger(state.health, stats.dist_food, stats.space)
    length = state.length.astype(jnp.float32)
    short = jnp.clip(1.0 - stats.space / jnp.maximum(length, 1.0), 0.0, 1.0)
    score = (
        weights.territory * territory
        + weights.food_territory * food_territory
        + weights.length * length
        - weights.hunger * hun
        - weights.shortfall * short
    )
    return SnakeTerms(
        space=stats.space,
        tail_reachable=stats.tail_reachable,
        territory=territory.astype(jnp.float32),
        food_territory=food_territory.astype(jnp.float32),
        dist_food=stats.dist_food,
        hunger=hun,
        shortfall=short.astype(jnp.float32),
        score=score.astype(jnp.float32),
    )


def terminal_values(state: State, config: GameConfig, draw_value: float = 0.0) -> jax.Array:
    """float32[N] exact outcome values, as in ``env.win_loss_reward``.

    Dead snakes get -1. When the game is over, the survivor gets +1. If nobody
    survives, the snakes eliminated on the final turn get ``draw_value``. Living
    snakes in a game that is not over get 0 (use :func:`evaluate` for those).
    In solo games dying is simply -1.
    """
    alive = state.alive
    value = jnp.where(alive, 0.0, -1.0)
    if config.solo:
        return value.astype(jnp.float32)
    over = rules.is_game_over(alive, config)
    value = jnp.where(over & alive, 1.0, value)
    last = state.elim_turn == jnp.max(state.elim_turn)
    draw = over & ~jnp.any(alive) & last
    return jnp.where(draw, draw_value, value).astype(jnp.float32)


def _values(
    state: State, config: GameConfig, terms: SnakeTerms, weights: Weights, draw_value: float
) -> jax.Array:
    """:func:`evaluate` given precomputed :func:`snake_terms`."""
    alive = state.alive
    others = ~jnp.eye(config.num_snakes, dtype=bool) & alive[None, :]
    best_other = jnp.max(jnp.where(others, terms.score[None, :], -jnp.inf), axis=1)
    solo_z = -(weights.hunger * terms.hunger + weights.shortfall * terms.shortfall)
    z = jnp.where(jnp.any(others, axis=1), terms.score - best_other, solo_z)
    value = jnp.clip(jnp.tanh(weights.sharpness * z), -0.99, 0.99)
    value = jnp.where(alive, value, -1.0)
    over = rules.is_game_over(alive, config)
    return jnp.where(over, terminal_values(state, config, draw_value), value).astype(jnp.float32)


def evaluate(
    state: State,
    config: GameConfig,
    weights: Weights = DEFAULT_WEIGHTS,
    *,
    draw_value: float = 0.0,
) -> jax.Array:
    """float32[N] value estimate per snake in ``[-1, 1]`` (higher is better for that snake).

    * If the game is over (``rules.is_game_over``), the result is exact: dead
      -1, sole survivor +1, and ``draw_value`` for snakes that die together on
      the final turn. ``state.done`` is not consulted, so successors from
      ``rules.rules_step`` work too. A game truncated with snakes alive is
      evaluated like any other.
    * Otherwise dead snakes get -1 and each living snake gets
      ``clip(tanh(sharpness * z_i), -0.99, 0.99)``. Here ``z_i`` is its score
      (:func:`snake_terms`) minus the best living opponent's score. In a duel
      ``z_0 = -z_1``, so the values are exactly antisymmetric. A lone snake in
      a solo game uses only its hunger and shortfall terms, so its value is at
      most 0.

    The terms, from largest effect to smallest in the research ablations:

    * territory: Voronoi cells claimed;
    * food territory: food items inside the claimed region;
    * length: ``tanh`` saturates, so food matters less when far ahead;
    * hunger: health versus distance to food;
    * shortfall: reachable area smaller than the snake.
    """
    terms = snake_terms(state, config, weights)
    return _values(state, config, terms, weights, draw_value)


# --- Per-move features (any number of snakes) -------------------------------------------


class MoveFeatures(NamedTuple):
    """Features of each snake's four candidate moves (all ``[N, 4]``).

    They need no simulation, so they also make cheap MCTS priors.
    """

    legal: jax.Array  # bool: action mask, minus moves that certainly starve
    eat: jax.Array  # bool: food on the cell moved to
    space: jax.Array  # int32: time-aware cells reachable after the move
    fits: jax.Array  # bool: own tail reachable, or space >= length + eat
    h2h_danger: jax.Array  # bool: an equal-or-longer opponent can move to the same cell
    h2h_loss: jax.Array  # bool: ... and the meeting loses (equal length is a draw in a duel)
    kill_chance: jax.Array  # float32: sum over shorter opponents that can move there of 1/#moves
    territory: jax.Array  # float32: (claimed - claimed by opponents) / (H * W)
    food_territory: jax.Array  # float32: (food claimed - food opponents claim) / max(1, #food)
    dist_food: jax.Array  # int32: moves to the nearest reachable food (incl. this move), or INF
    hunger: jax.Array  # float32 in [0, 1], see :func:`hunger`


def next_cells(state: State, config: GameConfig) -> jax.Array:
    """int32[N, 4, 2] the cell each snake's head moves to for each action (wrapped if needed)."""
    nxt = state.head[:, None, :] + _DELTAS[None, :, :]
    if config.ruleset.wrapped:
        nxt = nxt % jnp.array([config.width, config.height], jnp.int32)
    return nxt


def legal_moves(state: State, config: GameConfig) -> jax.Array:
    """bool[N, 4] ``rules.action_mask`` minus moves that certainly starve (health 1, no food)."""
    eat = _at(state.food, next_cells(state, config).reshape(-1, 2), config)
    starves = (state.health <= 1)[:, None] & ~eat.reshape(config.num_snakes, NUM_ACTIONS)
    return rules.action_mask(state, config) & ~starves & state.alive[:, None]


def move_features(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> MoveFeatures:
    """Features of every snake's candidate moves (see :class:`MoveFeatures`).

    Each snake's field after move ``a`` is a time-aware fill from the cell moved
    to, starting at move 1. Opponents' fields start at their heads at move 0,
    so both run on one clock, which models the simultaneous move. Territory is
    the Voronoi split between the two. This costs ``5 N`` fills on boolean
    grids, so it is several times slower than :func:`evaluate`.
    """
    n, h, w = config.num_snakes, config.height, config.width
    k = _fill_steps(config, weights)
    wrapped, wait = config.ruleset.wrapped, weights.fill_wait
    alive, length = state.alive, state.length
    tf = free_after(state, config)
    nxt = next_cells(state, config)  # [N, 4, 2]
    flat = nxt.reshape(-1, 2)
    eat = _at(state.food, flat, config).reshape(n, NUM_ACTIONS)
    mask = rules.action_mask(state, config)

    # Opponents' fields from their heads (move 0), merged into one per snake.
    heads = _cells(state.head, config) & alive[:, None, None]
    base = arrival_times(tf, heads, k, wrapped=wrapped, wait=wait)
    others = ~jnp.eye(n, dtype=bool) & alive[None, :]  # [i, j]
    opp = jnp.where(others[:, :, None, None], base[None], INF)  # [i, j, H, W]
    opp_first = jnp.min(opp, axis=1)  # [N, H, W]
    tied = (opp == opp_first[:, None]) & (opp_first[:, None] < INF)
    opp_len = jnp.max(jnp.where(tied, length[None, :, None, None], -1), axis=1)  # [N, H, W]

    # My field after each move (move 1 onward); illegal moves get an empty field.
    seeds = _cells(nxt, config) & (mask & alive[:, None])[:, :, None, None]
    mine = arrival_times(tf, seeds, k + 1, start=1, wrapped=wrapped, wait=wait)  # [N, 4, H, W]
    reach = mine < INF
    me_len = length[:, None, None, None]
    b, lb = opp_first[:, None], opp_len[:, None]
    claim = (mine < b) | ((mine == b) & reach & (me_len > lb))
    lost = (b < mine) | ((mine == b) & (b < INF) & (lb > me_len))
    food = state.food
    n_food = jnp.maximum(1, jnp.sum(food))
    territory = (jnp.sum(claim, axis=(2, 3)) - jnp.sum(lost, axis=(2, 3))) / (h * w)
    food_territory = (
        jnp.sum(claim & food, axis=(2, 3)) - jnp.sum(lost & food, axis=(2, 3))
    ) / n_food
    space = jnp.sum(reach, axis=(2, 3), dtype=jnp.int32)
    dist_food = jnp.min(jnp.where(food, mine, INF), axis=(2, 3))

    tail = tail_cells(state, config)  # [N, 2]
    tail_ok = reach[jnp.arange(n), :, tail[:, 1], tail[:, 0]]  # [N, 4]
    fits = tail_ok | (space >= length[:, None] + eat)

    # Head-to-heads: which opponents can move to each of my cells next turn.
    can = _at(base == 1, flat, config).reshape(n, n, NUM_ACTIONS)  # [j, i, a]
    can = jnp.transpose(can, (1, 2, 0)) & others[:, None, :]  # [i, a, j]
    li, lj = length[:, None, None], length[None, None, :]
    draw_is_loss = jnp.sum(alive) > 2  # a mutual kill only draws when nobody else is left
    h2h_danger = jnp.any(can & (lj >= li), axis=2)
    h2h_loss = jnp.any(can & ((lj > li) | ((lj == li) & draw_is_loss)), axis=2)
    n_moves = jnp.maximum(1, jnp.sum(base == 1, axis=(1, 2)))  # [j]
    kill_chance = jnp.sum(jnp.where(can & (lj < li), 1.0 / n_moves[None, None, :], 0.0), axis=2)

    return MoveFeatures(
        legal=legal_moves(state, config),
        eat=eat,
        space=space,
        fits=fits,
        h2h_danger=h2h_danger,
        h2h_loss=h2h_loss,
        kill_chance=kill_chance.astype(jnp.float32),
        territory=territory.astype(jnp.float32),
        food_territory=food_territory.astype(jnp.float32),
        dist_food=dist_food,
        hunger=hunger(state.health[:, None], dist_food, space),
    )


def _tiers(legal: jax.Array, safe: jax.Array, fits: jax.Array) -> jax.Array:
    return 4 * legal.astype(jnp.int32) + 2 * safe.astype(jnp.int32) + fits.astype(jnp.int32)


def greedy_scores(
    features: MoveFeatures, state: State, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """float32[N, 4] strategic score from the move features alone (no lookahead).

    ``territory + food territory - hunger * distance to food + eating
    (more when hungry) + space / length (capped at 2) + kill chance``,
    weighted by the ``greedy_*`` fields of :class:`Weights`. This is the "P0"
    snake of the research note, used when the game is not a duel.
    """
    f, wt = features, weights
    dist = jnp.where(f.dist_food < INF, f.dist_food, 30).astype(jnp.float32)
    length = jnp.maximum(state.length[:, None], 1).astype(jnp.float32)
    return (
        wt.greedy_territory * f.territory
        + wt.greedy_food_territory * f.food_territory
        - wt.greedy_hunger * f.hunger * dist / 22.0
        + wt.greedy_eat * f.eat * (1.0 + f.hunger)
        + wt.greedy_space * jnp.minimum(f.space / length, 2.0)
        + wt.greedy_kill * f.kill_chance
    ).astype(jnp.float32)


def feature_scores(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> tuple[jax.Array, jax.Array]:
    """(tier int32[N, 4], score float32[N, 4]) from :func:`move_features` (any number of snakes).

    The tier is ``4 * legal + 2 * (no losing head-to-head) + fits``.
    """
    f = move_features(state, config, weights)
    return _tiers(f.legal, ~f.h2h_loss, f.fits), greedy_scores(f, state, weights)


# --- One-ply duel search ------------------------------------------------------------------


def _joint_outcomes(
    state: State, config: GameConfig, weights: Weights
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(value, alive, fits), each ``[4, 4, 2]`` = ``[action 0, action 1, snake]``."""
    if config.num_snakes != 2:
        raise ValueError("the joint-move search is for duels (num_snakes == 2)")
    a = jnp.arange(NUM_ACTIONS)
    acts = jnp.stack(jnp.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2)

    def outcome(joint: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        nxt = rules.rules_step(state, joint, config)
        terms = snake_terms(nxt, config, weights)
        value = _values(nxt, config, terms, weights, -weights.contempt)
        fits = terms.tail_reachable | (terms.space >= nxt.length)
        return value, nxt.alive, fits

    shape = (NUM_ACTIONS, NUM_ACTIONS, 2)
    return tuple(x.reshape(shape) for x in jax.vmap(outcome)(acts))


def joint_values(state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS) -> jax.Array:
    """float32[4, 4, 2] duel values after every joint move: ``[action 0, action 1, snake]``.

    Each joint move is applied with the exact ``rules.rules_step`` (no food
    spawn) and scored by :func:`evaluate`, with a draw worth ``-contempt``.
    """
    return _joint_outcomes(state, config, weights)[0]


def duel_scores(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> tuple[jax.Array, jax.Array]:
    """(tier int32[2, 4], score float32[2, 4]) from the one-ply simultaneous-move search.

    For snake ``i`` and its move ``a``, with ``b`` ranging over the opponent's
    replies allowed by ``rules.action_mask``:

    * score: ``min_b M[a, b] + mean_weight * mean_b M[a, b]``, where ``M`` is
      :func:`joint_values` from snake ``i``'s point of view;
    * tier: ``4 * legal + 2 * safe + fits``. ``safe`` means no reply ``b``
      leaves ``i`` dead and the opponent alive, e.g. a losing head-to-head or
      a lethal hazard. (Bodies, a stacked tail included, are already masked; a
      tail whose snake eats this turn is still vacated.) ``fits`` means that
      ``i`` survives at least one reply, and that after every reply both
      snakes survive, its reachable area holds it or reaches its tail. A
      certain mutual elimination therefore does not fit, and a reply that
      kills the opponent is not judged on the finished board.
    """
    value, alive, fits = _joint_outcomes(state, config, weights)
    mask = rules.action_mask(state, config)
    legal = legal_moves(state, config)
    tiers, scores = [], []
    for i in range(2):

        def own(x: jax.Array, snake: int, i: int = i) -> jax.Array:
            """[own move, opponent move] view of ``x[..., snake]`` for seat ``i``."""
            return x[:, :, snake] if i == 0 else x[:, :, snake].T

        m = own(value, i)
        replies = mask[1 - i][None, :]
        me_alive, op_alive = own(alive, i), own(alive, 1 - i)
        safe = ~jnp.any(replies & ~me_alive & op_alive, axis=1)
        both = me_alive & op_alive
        fit = jnp.all(~replies | ~both | own(fits, i), axis=1) & jnp.any(replies & me_alive, axis=1)
        worst = jnp.min(jnp.where(replies, m, jnp.inf), axis=1)
        mean = jnp.sum(jnp.where(replies, m, 0.0), axis=1) / jnp.sum(replies)
        tiers.append(_tiers(legal[i], safe, fit))
        scores.append(worst + weights.mean_weight * mean)
    return jnp.stack(tiers), jnp.stack(scores)


# --- Policy ------------------------------------------------------------------------------


def move_scores(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> tuple[jax.Array, jax.Array]:
    """(tier int32[N, 4], score float32[N, 4]) that :func:`heuristic_policy` ranks by.

    Duels use :func:`duel_scores`; other games use :func:`feature_scores`.
    """
    if config.num_snakes == 2:
        return duel_scores(state, config, weights)
    return feature_scores(state, config, weights)


def _lexicographic_choice(key: jax.Array, tier: jax.Array, score: jax.Array) -> jax.Array:
    """int32[N]: best tier first, then best score; exact ties broken uniformly at random."""
    top = tier == jnp.max(tier, axis=-1, keepdims=True)
    score = jnp.where(top, score, -jnp.inf)
    ties = top & (score >= jnp.max(score, axis=-1, keepdims=True) - 1e-6)
    return jax.random.categorical(key, jnp.where(ties, 0.0, -jnp.inf), axis=-1).astype(jnp.int32)


def heuristic_policy(
    key: jax.Array, state: State, env: BattlesnakeEnv, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """int32[N] the heuristic snake's move for every snake, each from its own point of view.

    Moves are ranked by safety tier, then by strategic score
    (:func:`move_scores`). Exact ties are broken at random with ``key``. Only
    ``env.config`` is used (observations are ignored), so close over ``env``
    rather than passing it through ``jit``/``vmap``.
    """
    tier, score = move_scores(state, env.config, weights)
    return _lexicographic_choice(key, tier, score)


def heuristic(env: BattlesnakeEnv, weights: Weights = DEFAULT_WEIGHTS) -> Policy:
    """The heuristic snake as an ``evaluate.Policy``: ``(key, state, timestep) -> int32[N]``.

    Cached, so repeated calls return the same object and hit ``play_match``'s
    jit cache.
    """
    return _heuristic(env, weights)


@functools.lru_cache(maxsize=16)
def _heuristic(env: BattlesnakeEnv, weights: Weights) -> Policy:
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return heuristic_policy(key, state, env, weights)

    return policy
