"""A hand-written heuristic snake: a benchmark opponent and an MCTS leaf evaluator.

The module has three layers. Each can be used on its own, and all of them are
pure, fixed-shape functions of one (unbatched) game that ``jit`` and ``vmap``.

1. **Grid utilities** (:func:`neighbours`, :func:`flood_fill`,
   :func:`distances`, :func:`arrival_times`, :func:`voronoi`). These are
   breadth-first fills on ``[..., H, W]`` boolean grids, built from
   pad-and-shift 4-neighbour dilation (``jnp.roll`` on wrapped boards), as in
   ``maps.spawn_mask``. :func:`arrival_times` is the time-aware fill. It reads
   the countdown grid (:func:`free_after`): a body cell with countdown ``k``
   can be entered on the ``d``-th move from now iff ``k <= d``, because that
   snake's tail has passed it by then (assuming it doesn't eat). Treating
   bodies as permanent walls instead plays much worse.

2. **A static evaluator**, :func:`evaluate`, which gives each snake a value in
   ``[-1, 1]``. Terminal states get the exact outcome, as in
   ``env.win_loss_reward``. Other states get ``tanh`` of a score difference
   (:class:`Weights`, :func:`snake_terms`). The score is built from Voronoi
   territory (cells a snake reaches strictly first, with ties going to the
   longer snake), food inside that territory, length, a starvation term and a
   trap term. In a duel the value is exactly antisymmetric. It costs two fills
   per state, so it is cheap enough to call at every MCTS leaf.

3. **A policy**, :func:`heuristic_policy` (and :func:`heuristic`, the cached
   ``evaluate.Policy``). Each snake ranks its candidate moves
   lexicographically:

   * tier 1, legal: ``env.action_mask``, minus moves that certainly starve;
   * tier 2, no losing head-to-head: no opponent that would survive the
     meeting can move to the same cell;
   * tier 3, fits: after the move, the time-aware reachable area holds the
     snake, or its own tail can be reached (trap avoidance);
   * then a strategic score. In a duel this is a one-ply simultaneous-move
     search over the exact rules. Every joint move ``(a, b)`` is applied with
     ``rules.rules_step`` and evaluated, and move ``a`` scores
     ``min_b M[a, b] + mean_weight * mean_b M[a, b]`` over the opponent's
     legal replies ``b``. A mutual elimination is worth ``-contempt``, so the
     snake trades heads only when it is otherwise losing. With other numbers
     of snakes (solo, or 3 or more), a weighted sum of the per-move features
     in :class:`MoveFeatures` is used instead.

   Exact ties are broken uniformly at random with the policy's key.

The policy follows the "P1" spec of ``docs/research/battlesnake_heuristics.md``
and was re-tuned on held-out games; see the docstring of :class:`Weights`.
Only the duel on the ``standard`` ruleset was tuned. Wrapped boards and
constrictor (where bodies never shrink) are handled in the fills. Hazards are
not modelled outside the exact one-ply rules step.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from slinky import rules
from slinky.env import BattlesnakeEnv
from slinky.types import ACTION_DELTAS, NUM_ACTIONS, GameConfig, State, TimeStep

INF = 1000  # distance / arrival time of cells that are not reached
_NEVER = 30_000  # countdown of cells that never free up (constrictor bodies)
_DELTAS = jnp.array(ACTION_DELTAS, jnp.int32)  # [4, 2] (dx, dy)

Policy = Callable[[jax.Array, State, TimeStep], jax.Array]

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
    its tail leaves it, which also gives the tail rule (countdown 1 can be
    entered on the next move, a stacked tail, countdown 2, cannot). Other
    arguments are as in :func:`distances`. This is pessimistic about snakes
    that eat (they keep their tail one move longer) and ignores where the
    other snakes' heads go.
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

    The defaults come from ``docs/research/battlesnake_heuristics.md`` (fitted
    on 46.5k sampled states, then hand-tuned), followed by a held-out tuning
    pass in the 11x11 duel.
    """

    territory: float = 0.40
    food_territory: float = 0.35
    length: float = 0.15
    hunger: float = 0.50
    shortfall: float = 0.50
    sharpness: float = 1.0
    contempt: float = 0.4
    mean_weight: float = 0.25
    greedy_territory: float = 1.5
    greedy_food_territory: float = 2.0
    greedy_hunger: float = 1.5
    greedy_eat: float = 2.0
    greedy_space: float = 0.25
    greedy_kill: float = 0.5
    fill_steps: int | None = None


DEFAULT_WEIGHTS = Weights()


def _fill_steps(config: GameConfig, weights: Weights) -> int:
    return config.height + config.width if weights.fill_steps is None else weights.fill_steps


def hunger(health: jax.Array, dist_food: jax.Array, space: jax.Array) -> jax.Array:
    """float32 starvation pressure: 0 when comfortable, 1 when about to starve.

    ``margin = health - dist_food``, the health left on reaching the nearest
    reachable food. If no food is reachable, ``margin = min(health, space) - 5``.
    The pressure is ``clip((30 - margin) / 30, 0, 1) ** 2``.
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
    territory: jax.Array  # float32: cells claimed / (H * W)
    food_territory: jax.Array  # float32: food claimed / max(1, food on the board)
    dist_food: jax.Array  # int32: moves to the nearest reachable food, INF if none
    hunger: jax.Array  # float32 in [0, 1], see :func:`hunger`
    shortfall: jax.Array  # float32 in [0, 1]: clip(1 - space / length, 0, 1)
    score: jax.Array  # float32: the weighted sum (see :class:`Weights`)


def snake_terms(state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS) -> SnakeTerms:
    """The evaluator's terms for every snake, from one time-aware fill per snake."""
    h, w = config.height, config.width
    tf = free_after(state, config)
    heads = _cells(state.head, config) & state.alive[:, None, None]
    arr = arrival_times(tf, heads, _fill_steps(config, weights), wrapped=config.ruleset.wrapped)
    claims = voronoi(arr, state.length)
    food = state.food
    reach = arr < INF
    space = jnp.sum(reach, axis=(1, 2))
    territory = jnp.sum(claims, axis=(1, 2)) / (h * w)
    food_territory = jnp.sum(claims & food, axis=(1, 2)) / jnp.maximum(1, jnp.sum(food))
    dist_food = jnp.min(jnp.where(food, arr, INF), axis=(1, 2))
    hun = hunger(state.health, dist_food, space)
    length = state.length.astype(jnp.float32)
    short = jnp.clip(1.0 - space / jnp.maximum(length, 1.0), 0.0, 1.0)
    score = (
        weights.territory * territory
        + weights.food_territory * food_territory
        + weights.length * length
        - weights.hunger * hun
        - weights.shortfall * short
    )
    return SnakeTerms(
        space=space,
        territory=territory.astype(jnp.float32),
        food_territory=food_territory.astype(jnp.float32),
        dist_food=dist_food,
        hunger=hun,
        shortfall=short,
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
    alive = state.alive
    n = config.num_snakes
    others = ~jnp.eye(n, dtype=bool) & alive[None, :]
    best_other = jnp.max(jnp.where(others, terms.score[None, :], -jnp.inf), axis=1)
    solo_z = -(weights.hunger * terms.hunger + weights.shortfall * terms.shortfall)
    z = jnp.where(jnp.any(others, axis=1), terms.score - best_other, solo_z)
    value = jnp.clip(jnp.tanh(weights.sharpness * z), -0.99, 0.99)
    value = jnp.where(alive, value, -1.0)
    over = rules.is_game_over(alive, config)
    return jnp.where(over, terminal_values(state, config, draw_value), value).astype(jnp.float32)


# --- Per-move features -----------------------------------------------------------------


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
    dist_food: jax.Array  # int32: moves to the nearest reachable food (this one counts), INF if none
    hunger: jax.Array  # float32 in [0, 1], see :func:`hunger`


def next_cells(state: State, config: GameConfig) -> jax.Array:
    """int32[N, 4, 2] the cell each snake's head moves to for each action (wrapped if needed)."""
    nxt = state.head[:, None, :] + _DELTAS[None, :, :]
    if config.ruleset.wrapped:
        nxt = nxt % jnp.array([config.width, config.height], jnp.int32)
    return nxt


def move_features(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> MoveFeatures:
    """Features of every snake's candidate moves (see :class:`MoveFeatures`).

    Each snake's field after move ``a`` is a time-aware fill from the cell moved
    to, starting at move 1. Opponents' fields start at their heads at move 0,
    so both run on one clock, which models the simultaneous move. Territory is
    the Voronoi split between the two.
    """
    n, h, w = config.num_snakes, config.height, config.width
    k = _fill_steps(config, weights)
    wrapped = config.ruleset.wrapped
    alive, length = state.alive, state.length
    tf = free_after(state, config)
    nxt = next_cells(state, config)  # [N, 4, 2]
    flat = nxt.reshape(-1, 2)
    eat = _at(state.food, flat, config).reshape(n, NUM_ACTIONS)
    mask = rules.action_mask(state, config)
    starves = (state.health <= 1)[:, None] & ~eat
    legal = mask & ~starves & alive[:, None]

    # Opponents' fields from their heads (move 0), merged into one per snake.
    base = arrival_times(tf, _cells(state.head, config) & alive[:, None, None], k, wrapped=wrapped)
    others = ~jnp.eye(n, dtype=bool) & alive[None, :]  # [i, j]
    opp = jnp.where(others[:, :, None, None], base[None], INF)  # [i, j, H, W]
    opp_first = jnp.min(opp, axis=1)  # [N, H, W]
    tied = (opp == opp_first[:, None]) & (opp_first[:, None] < INF)
    opp_len = jnp.max(jnp.where(tied, length[None, :, None, None], -1), axis=1)  # [N, H, W]

    # My field after each move (move 1 onward); illegal moves get an empty field.
    seeds = _cells(nxt, config) & (mask & alive[:, None])[:, :, None, None]
    mine = arrival_times(tf, seeds, k + 1, start=1, wrapped=wrapped)  # [N, 4, H, W]
    reach = mine < INF
    me_len = length[:, None, None, None]
    b, lb = opp_first[:, None], opp_len[:, None]
    claim = (mine < b) | ((mine == b) & reach & (me_len > lb))
    lost = (b < mine) | ((mine == b) & (b < INF) & (lb > me_len))
    food = state.food
    n_food = jnp.maximum(1, jnp.sum(food))
    territory = (jnp.sum(claim, axis=(2, 3)) - jnp.sum(lost, axis=(2, 3))) / (h * w)
    food_territory = (jnp.sum(claim & food, axis=(2, 3)) - jnp.sum(lost & food, axis=(2, 3))) / n_food
    space = jnp.sum(reach, axis=(2, 3))
    dist_food = jnp.min(jnp.where(food, mine, INF), axis=(2, 3))

    # Own tail: the occupied cell with the smallest countdown.
    body = state.body.reshape(n, -1)
    tail = jnp.argmin(jnp.where(body > 0, body, jnp.iinfo(body.dtype).max), axis=1)
    tail_ok = jnp.take_along_axis(reach.reshape(n, NUM_ACTIONS, -1), tail[:, None, None], axis=2)
    fits = tail_ok[..., 0] | (space >= length[:, None] + eat)

    # Head-to-heads: which opponents can move to each of my cells next turn.
    can = _at(base == 1, flat, config).reshape(n, n, NUM_ACTIONS)  # [j, i, a]
    can = jnp.transpose(can, (1, 2, 0)) & others[:, None, :]  # [i, a, j]
    li, lj = length[:, None, None], length[None, None, :]
    draw_is_loss = jnp.sum(alive) > 2  # a mutual kill only draws when nobody else is left
    h2h_danger = jnp.any(can & (lj >= li), axis=2)
    h2h_loss = jnp.any(can & ((lj > li) | ((lj == li) & draw_is_loss)), axis=2)
    n_moves = jnp.maximum(1, jnp.sum(base == 1, axis=(1, 2)))  # [j]
    kill_chance = jnp.sum(jnp.where(can & (lj < li), 1.0 / n_moves[None, None, :], 0.0), axis=2)

    hun = hunger(state.health[:, None], dist_food, space)
    return MoveFeatures(
        legal=legal,
        eat=eat,
        space=space,
        fits=fits,
        h2h_danger=h2h_danger,
        h2h_loss=h2h_loss,
        kill_chance=kill_chance.astype(jnp.float32),
        territory=territory.astype(jnp.float32),
        food_territory=food_territory.astype(jnp.float32),
        dist_food=dist_food,
        hunger=hun,
    )


def move_tiers(features: MoveFeatures) -> jax.Array:
    """int32[N, 4] lexicographic safety tier: 4 * legal + 2 * (no losing h2h) + fits."""
    f = features
    return 4 * f.legal.astype(jnp.int32) + 2 * (~f.h2h_loss).astype(jnp.int32) + f.fits


def greedy_scores(
    features: MoveFeatures, state: State, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """float32[N, 4] strategic score from the move features alone (no lookahead).

    ``territory + food territory - hunger * distance to food + eating
    (more when hungry) + space / length (capped at 2) + kill chance``,
    weighted by the ``greedy_*`` fields of :class:`Weights`. This is the "P0"
    snake of the research note. It is the policy's fallback when the game is
    not a duel.
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


def joint_values(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """float32[4, 4, 2] duel values after every joint move: ``[action 0, action 1, snake]``.

    Each joint move is applied with the exact ``rules.rules_step`` (no food
    spawn) and scored by :func:`evaluate`, with a draw worth ``-contempt``.
    """
    if config.num_snakes != 2:
        raise ValueError("joint_values is for duels (num_snakes == 2)")
    a = jnp.arange(NUM_ACTIONS)
    acts = jnp.stack(jnp.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2)

    def value(joint: jax.Array) -> jax.Array:
        nxt = rules.rules_step(state, joint, config)
        return evaluate(nxt, config, weights, draw_value=-weights.contempt)

    return jax.vmap(value)(acts).reshape(NUM_ACTIONS, NUM_ACTIONS, 2)


def lookahead_scores(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """float32[2, 4] one-ply simultaneous-move score of each snake's moves (duel).

    ``min_b M[a, b] + mean_weight * mean_b M[a, b]``, where ``M`` is
    :func:`joint_values` from the snake's own point of view and ``b`` ranges
    over the opponent's moves allowed by ``rules.action_mask``.
    """
    vals = joint_values(state, config, weights)
    mask = rules.action_mask(state, config)
    out = []
    for i in range(2):
        m = vals[:, :, 0] if i == 0 else vals[:, :, 1].T  # [own move, opponent move]
        replies = mask[1 - i][None, :]
        worst = jnp.min(jnp.where(replies, m, jnp.inf), axis=1)
        mean = jnp.sum(jnp.where(replies, m, 0.0), axis=1) / jnp.sum(replies)
        out.append(worst + weights.mean_weight * mean)
    return jnp.stack(out)


def _lexicographic_choice(key: jax.Array, tier: jax.Array, score: jax.Array) -> jax.Array:
    """int32[N]: best tier first, then best score; exact ties broken uniformly at random."""
    top = tier == jnp.max(tier, axis=-1, keepdims=True)
    score = jnp.where(top, score, -jnp.inf)
    ties = top & (score >= jnp.max(score, axis=-1, keepdims=True) - 1e-6)
    return jax.random.categorical(key, jnp.where(ties, 0.0, -jnp.inf), axis=-1).astype(jnp.int32)


def move_scores(
    state: State, config: GameConfig, weights: Weights = DEFAULT_WEIGHTS
) -> tuple[jax.Array, jax.Array]:
    """(tier int32[N, 4], score float32[N, 4]) that :func:`heuristic_policy` ranks by.

    The duel uses :func:`lookahead_scores`; other games use :func:`greedy_scores`.
    """
    features = move_features(state, config, weights)
    tier = move_tiers(features)
    if config.num_snakes == 2:
        score = lookahead_scores(state, config, weights)
    else:
        score = greedy_scores(features, state, weights)
    return tier, score


def heuristic_policy(
    key: jax.Array, state: State, env: BattlesnakeEnv, weights: Weights = DEFAULT_WEIGHTS
) -> jax.Array:
    """int32[N] the heuristic snake's move for every snake, each from its own point of view.

    Moves are ranked by safety tier (:func:`move_tiers`), then by strategic
    score (:func:`move_scores`), and exact ties are broken at random with
    ``key``. Only ``env.config`` is used (observations are ignored), so
    close over ``env`` rather than passing it through ``jit``/``vmap``.
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
