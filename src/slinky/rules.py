"""The deterministic rules pipeline: one engine ``ruleset.Execute`` call.

This mirrors the official engine's stage order (see
``docs/battlesnake/ENGINE_RULES.md``):

    game-over check -> move -> starvation -> hazard damage -> feed
    -> eliminate -> [constrictor: remove food, reset health, grow]
    -> [royale: regenerate the shrinking hazard zone]

Map behaviour (food spawning, hazard placement) and the turn increment are
*not* part of this pipeline; they live in :mod:`slinky.maps` and
:mod:`slinky.env`, exactly as in the engine. The royale ruleset is the one
ruleset that places hazards itself. Its zone comes from the shrink sides in
``state.map_state`` (a :class:`~slinky.types.RoyaleState`), which the
environment draws at reset whatever the map.

``rules_step`` alone does not update hazards for the *standard* ruleset on the
royale map; there the map does it (``RoyaleMap.post_update``). Code that only
calls ``rules_step`` (``mcts.py`` with its default ``spawn_food=False``, the
heuristic's duel search) therefore keeps the hazards of the root state in that
combination.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from slinky.types import (
    ACTION_DELTAS,
    MAX_HEALTH,
    NUM_ACTIONS,
    Cause,
    GameConfig,
    RoyaleState,
    Ruleset,
    State,
    royale_num_sides,
)

_DELTAS = jnp.array(ACTION_DELTAS, jnp.int32)  # [4, 2] (dx, dy)


def is_game_over(alive: jax.Array, config: GameConfig) -> jax.Array:
    """Engine game-over condition: <= 1 snake left (solo: none left)."""
    remaining = jnp.sum(alive)
    return remaining == 0 if config.solo else remaining <= 1


def in_bounds(xy: jax.Array, config: GameConfig) -> jax.Array:
    """bool[...] whether ``xy[..., (x, y)]`` lies on the board."""
    x, y = xy[..., 0], xy[..., 1]
    return (x >= 0) & (x < config.width) & (y >= 0) & (y < config.height)


def value_at(grid: jax.Array, xy: jax.Array, config: GameConfig) -> jax.Array:
    """Gather ``grid[..., y, x]`` at points ``xy[N, 2]``; 0/False when off-board.

    ``grid`` is ``[H, W]`` (returns ``[N]``) or ``[M, H, W]`` (returns ``[M, N]``).
    """
    x = jnp.clip(xy[:, 0], 0, config.width - 1)
    y = jnp.clip(xy[:, 1], 0, config.height - 1)
    vals = grid[..., y, x]
    return jnp.where(in_bounds(xy, config), vals, jnp.zeros_like(vals))


def one_hot_cells(xy: jax.Array, config: GameConfig) -> jax.Array:
    """bool[N, H, W] marking each point (all-False rows for off-board points)."""
    ys = jnp.arange(config.height)[None, :, None]
    xs = jnp.arange(config.width)[None, None, :]
    return (ys == xy[:, 1, None, None]) & (xs == xy[:, 0, None, None])


def next_heads(state: State, actions: jax.Array, config: GameConfig) -> tuple[jax.Array, jax.Array]:
    """Applied move and new head position for every snake (dead snakes don't move).

    Actions outside ``[0, 4)`` are invalid and fall back to the engine's default
    move (continue in the direction of the last move; ``up`` on turn 0).
    """
    actions = jnp.asarray(actions, jnp.int32)
    valid = (actions >= 0) & (actions < NUM_ACTIONS)
    move = jnp.where(valid, actions, state.last_move.astype(jnp.int32))
    head = state.head + _DELTAS[move]
    if config.ruleset.wrapped:
        head = head % jnp.array([config.width, config.height], jnp.int32)
    head = jnp.where(state.alive[:, None], head, state.head)
    return move, head


def royale_hazards(sides: jax.Array, turn: jax.Array, config: GameConfig) -> jax.Array:
    """int16[H, W] royale hazards (0 or 1) of the state shown at ``turn``.

    The zone has shrunk ``turn // shrink_every_n_turns`` times, so the first
    hazards are shown at turn ``shrink_every_n_turns`` and none before. Shrink
    ``k`` removes one edge row or column, ``sides[k]``: 0 left, 1 right,
    2 bottom (``y = 0``), 3 top. ``W + H - 1`` shrinks always cover the whole
    board, so ``sides`` holds that many and later shrinks change nothing.

    The engine regenerates the zone from scratch at ``turn + 1`` after each
    move (``PopulateHazardsRoyale``); callers pass ``state.turn + 1``.
    """
    used = jnp.arange(royale_num_sides(config)) < turn // config.shrink_every_n_turns
    c = jnp.sum((sides[:, None].astype(jnp.int32) == jnp.arange(4)) & used[:, None], axis=0)
    xs, ys = jnp.arange(config.width)[None, :], jnp.arange(config.height)[:, None]
    outside = (
        (xs < c[0]) | (xs > config.width - 1 - c[1]) | (ys < c[2]) | (ys > config.height - 1 - c[3])
    )
    return outside.astype(jnp.int16)


def rules_step(state: State, actions: jax.Array, config: GameConfig) -> State:
    """Apply one turn of the ruleset pipeline (no food spawning, no turn increment).

    Args:
      state: current state (the one the snakes were shown).
      actions: int[N] moves; dead snakes' entries are ignored.
      config: static game configuration.

    Returns the state after the engine's ``ruleset.Execute``. If the game was
    already over, the state is returned unchanged (as the engine does).

    The royale ruleset needs the shrink sides: ``state.map_state`` must be a
    :class:`~slinky.types.RoyaleState` (``ValueError`` at trace time otherwise).
    """
    if config.ruleset == Ruleset.ROYALE and not isinstance(state.map_state, RoyaleState):
        raise ValueError(
            "the royale ruleset needs state.map_state to be a RoyaleState (the shrink sides): "
            "BattlesnakeEnv draws them at reset, or pass map_state= to state_from_engine"
        )
    alive0 = state.alive
    move, head = next_heads(state, actions, config)
    on_board = in_bounds(head, config)
    head_cells = one_hot_cells(head, config) & alive0[:, None, None]

    # --- Move: pop the tail (decrement countdowns), then write the new head.
    body_dec = jnp.where(alive0[:, None, None], jnp.maximum(state.body - 1, 0), state.body)
    body = jnp.where(head_cells, state.length[:, None, None].astype(body_dec.dtype), body_dec)

    # --- Starvation.
    health = jnp.where(alive0, state.health - 1, state.health)

    # --- Hazard damage (skipped if the head's cell also has food; stacks per layer).
    food_at_head = value_at(state.food, head, config)
    layers = value_at(state.hazard, head, config).astype(jnp.int32)
    hit = alive0 & (layers > 0) & ~food_at_head
    # The engine clamps health to [0, 100] after each layer, so per-layer damage
    # beyond +-100 and more than 101 layers change nothing; saturating both keeps
    # the product far from int32 overflow.
    damage = max(-MAX_HEALTH, min(MAX_HEALTH, config.hazard_damage_per_turn))
    damaged = jnp.clip(health - jnp.minimum(layers, MAX_HEALTH + 1) * damage, 0, MAX_HEALTH)
    health = jnp.where(hit, damaged, health)
    hazard_elim = hit & (health <= 0)
    alive1 = alive0 & ~hazard_elim

    # --- Feed: every living snake whose head is on food eats it.
    eats = alive1 & food_at_head
    eaten_cells = jnp.any(head_cells & eats[:, None, None], axis=0)
    food = state.food & ~eaten_cells
    health = jnp.where(eats, MAX_HEALTH, health)
    length = state.length + eats.astype(jnp.int32)
    body = jnp.where(eats[:, None, None] & (body > 0), body + 1, body)

    # --- Eliminate, pass 1: out of health, then out of bounds.
    starved = alive1 & (health <= 0)
    walled = alive1 & ~starved & ~on_board
    alive2 = alive1 & ~starved & ~walled

    # --- Eliminate, pass 2: collisions among pass-1 survivors, applied together.
    # A head "hits" snake j's body if it lands on any segment of j with index >= 1,
    # i.e. a cell j still occupies after its tail pop (feeding doesn't change this).
    n = config.num_snakes
    hits = value_at(body_dec > 0, head, config).T  # [i, j]: head i on body of j
    others = ~jnp.eye(n, dtype=bool) & alive2[None, :]
    self_hit = alive2 & jnp.diagonal(hits)
    body_hit = alive2 & ~self_hit & jnp.any(hits & others, axis=1)
    same_cell = jnp.all(head[:, None, :] == head[None, :, :], axis=-1) & others
    loses_h2h = same_cell & (length[:, None] <= length[None, :])
    h2h = alive2 & ~self_hit & ~body_hit & jnp.any(loses_h2h, axis=1)
    alive3 = alive2 & ~self_hit & ~body_hit & ~h2h

    cause = jnp.select(
        [hazard_elim, starved, walled, self_hit, body_hit, h2h],
        [
            Cause.HAZARD,
            Cause.OUT_OF_HEALTH,
            Cause.OUT_OF_BOUNDS,
            Cause.SELF_COLLISION,
            Cause.COLLISION,
            Cause.HEAD_COLLISION,
        ],
        default=Cause.NONE,
    ).astype(jnp.int8)
    died = alive0 & ~alive3
    elim_cause = jnp.where(died, cause, state.elim_cause)
    elim_turn = jnp.where(died, state.turn + 1, state.elim_turn)

    # --- Constrictor: no food, full health, and grow unless the tail is already
    # stacked (tail cell countdown == 1 means a single, unstacked tail segment).
    if config.ruleset.constrictor:
        food = jnp.zeros_like(food)
        health = jnp.full_like(health, MAX_HEALTH)
        tail = jnp.min(jnp.where(body > 0, body, jnp.iinfo(body.dtype).max), axis=(1, 2))
        grow = alive3 & (tail == 1)
        length = length + grow.astype(jnp.int32)
        body = jnp.where(grow[:, None, None] & (body > 0), body + 1, body)

    # --- Royale: the zone for the next turn, from scratch (after elimination, as in
    # the engine). The freeze below skips it once the game is over, as the engine does.
    hazard = state.hazard
    if config.ruleset == Ruleset.ROYALE:
        hazard = royale_hazards(state.map_state.sides, state.turn + 1, config)

    body = jnp.where(alive3[:, None, None], body, 0)
    new_state = state._replace(
        body=body.astype(state.body.dtype),
        head=head,
        length=length,
        health=health,
        alive=alive3,
        last_move=jnp.where(alive0, move, state.last_move).astype(state.last_move.dtype),
        food=food,
        hazard=hazard,
        elim_cause=elim_cause,
        elim_turn=elim_turn,
    )
    # The engine's first stage ends the game before anything moves.
    over = is_game_over(alive0, config)
    return jax.tree.map(lambda old, new: jnp.where(over, old, new), state, new_state)


def _neighbours(head: jax.Array, config: GameConfig) -> jax.Array:
    """int32[N, 4, 2] cells reached by each action (wrapped if the ruleset wraps)."""
    nxt = head[:, None, :] + _DELTAS[None, :, :]
    if config.ruleset.wrapped:
        nxt = nxt % jnp.array([config.width, config.height], jnp.int32)
    return nxt


def certainly_starving(state: State, config: GameConfig) -> jax.Array:
    """bool[N] living snakes that will be eliminated before collisions next turn.

    A snake on 1 health dies of starvation (in the engine's first elimination
    pass, before collisions are checked) unless it eats, or unless a healing
    hazard (negative damage) restores it. Such a snake's body cannot block
    anyone.
    """
    nxt = _neighbours(state.head, config).reshape(-1, 2)
    rescue = value_at(state.food, nxt, config)
    if config.hazard_damage_per_turn < 0:
        rescue |= value_at(state.hazard, nxt, config) > 0
    rescue = jnp.any(rescue.reshape(config.num_snakes, NUM_ACTIONS), axis=1)
    return state.alive & (state.health <= 1) & ~rescue


def blocked_cells(state: State, config: GameConfig) -> jax.Array:
    """bool[H, W] cells that will hold a body segment next turn whatever anyone does.

    A cell is freed by a tail pop when its countdown is 1; anything above that
    (including a stacked tail, countdown 2) is still occupied after moving.
    Bodies of snakes that will certainly starve first don't count.
    """
    solid = state.alive & ~certainly_starving(state, config)
    return jnp.any((state.body > 1) & solid[:, None, None], axis=0)


def action_mask(state: State, config: GameConfig) -> jax.Array:
    """bool[N, 4] moves that don't certainly die to a wall or a body next turn.

    A move is masked out only if it leaves the board or enters a cell that
    will hold a body segment whatever the other snakes do (tail rule
    included). Hazards, starvation and head-to-heads are not considered.

    Rows are never all-False, so masked softmaxes stay finite: dead snakes,
    snakes certain to starve and snakes with no safe move get all-True rows
    (every action is equally good, or equally fatal, for them).
    """
    flat = _neighbours(state.head, config).reshape(-1, 2)
    ok = in_bounds(flat, config) & ~value_at(blocked_cells(state, config), flat, config)
    ok = ok.reshape(config.num_snakes, NUM_ACTIONS)
    trivial = ~state.alive | certainly_starving(state, config) | ~jnp.any(ok, axis=1)
    return ok | trivial[:, None]
