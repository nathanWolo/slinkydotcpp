"""The deterministic rules pipeline: one engine ``ruleset.Execute`` call.

This mirrors the official engine's stage order (see
``docs/battlesnake/ENGINE_RULES.md``):

    game-over check -> move -> starvation -> hazard damage -> feed
    -> eliminate -> [constrictor: remove food, reset health, grow]

Map behaviour (food spawning, hazard placement) and the turn increment are
*not* part of this pipeline; they live in :mod:`slinky.maps` and
:mod:`slinky.env`, exactly as in the engine.
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
    State,
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


def rules_step(state: State, actions: jax.Array, config: GameConfig) -> State:
    """Apply one turn of the ruleset pipeline (no food spawning, no turn increment).

    Args:
      state: current state (the one the snakes were shown).
      actions: int[N] moves; dead snakes' entries are ignored.
      config: static game configuration.

    Returns the state after the engine's ``ruleset.Execute``. If the game was
    already over, the state is returned unchanged (as the engine does).
    """
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
    damaged = jnp.clip(health - layers * config.hazard_damage_per_turn, 0, MAX_HEALTH)
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

    body = jnp.where(alive3[:, None, None], body, 0)
    new_state = state._replace(
        body=body.astype(state.body.dtype),
        head=head,
        length=length,
        health=health,
        alive=alive3,
        last_move=jnp.where(alive0, move, state.last_move).astype(state.last_move.dtype),
        food=food,
        elim_cause=elim_cause,
        elim_turn=elim_turn,
    )
    # The engine's first stage ends the game before anything moves.
    over = is_game_over(alive0, config)
    return jax.tree.map(lambda old, new: jnp.where(over, old, new), state, new_state)


def blocked_cells(state: State) -> jax.Array:
    """bool[H, W] cells that will hold a body segment next turn whatever anyone does.

    A cell is freed by a tail pop when its countdown is 1; anything above that
    (including a stacked tail, countdown 2) is still occupied after moving.
    """
    return jnp.any((state.body > 1) & state.alive[:, None, None], axis=0)


def action_mask(state: State, config: GameConfig) -> jax.Array:
    """bool[N, 4] moves that don't certainly die to a wall or a body next turn.

    Ignores hazards, starvation and head-to-heads (which depend on the other
    snakes' choices). Rows for dead snakes are all False; a living snake with
    no safe move also gets an all-False row.
    """
    head = state.head[:, None, :] + _DELTAS[None, :, :]  # [N, 4, 2]
    if config.ruleset.wrapped:
        head = head % jnp.array([config.width, config.height], jnp.int32)
    flat = head.reshape(-1, 2)
    ok = in_bounds(flat, config) & ~value_at(blocked_cells(state), flat, config)
    return ok.reshape(config.num_snakes, NUM_ACTIONS) & state.alive[:, None]
