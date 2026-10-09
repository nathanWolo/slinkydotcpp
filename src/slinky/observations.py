"""Observation encoders: dense multi-channel board images, per agent.

Every observation is a float32 ``[rows, cols, NUM_CHANNELS]`` image (channels
last). Two views are provided:

* **allocentric** -- the board as is, ``[H, W, C]``, indexed ``[y, x]`` (row 0 is
  the bottom row, same as ``State`` grids).
* **egocentric** -- the board re-centered on the agent's head. For non-wrapped
  rulesets the board is zero-padded (``on_board = 0``) so that the head sits at
  ``[H-1, W-1]`` of a ``[2H-1, 2W-1, C]`` image and output row ``r`` / column
  ``c`` is the cell at offset ``(c-(W-1), r-(H-1))`` from the head. For wrapped
  rulesets the board is a torus, so it is rolled (no padding) to ``[H, W, C]``
  with the head at ``[H//2, W//2]``.

Snakes with ``alive == False`` are ignored (masked by ``alive``, their grids
are never trusted). A dead agent's own channels are zero; its observation is
finite but otherwise unspecified. Everything is jit/vmap friendly: ``config`` is
static, ``agent`` may be a traced integer.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from slinky.types import GameConfig, State

CHANNEL_NAMES = (
    "on_board",  # 1 on real board cells, 0 on padding outside the board
    "food",
    "hazard",  # stacked hazard count
    "own_head",
    "own_body",  # occupancy incl. head
    "own_countdown",  # body value / (W*H)
    "opp_head_ge",  # heads of living opponents with length >= own length
    "opp_head_lt",  # heads of living opponents with length < own length
    "opp_body",  # occupancy by any living opponent, incl. heads
    "opp_countdown",  # max over living opponents of body value / (W*H)
    "opp_head_health",  # opponent health / 100 at that opponent's head cell
    "own_health",  # own health / 100, broadcast over on-board cells
    "own_length",  # own length / (W*H), broadcast over on-board cells
)
NUM_CHANNELS = len(CHANNEL_NAMES)
assert NUM_CHANNELS == 13

_KINDS = ("egocentric", "allocentric")


def allocentric_obs(state: State, config: GameConfig, agent: jax.Array | int) -> jax.Array:
    """Board observation ``f32[H, W, C]`` from ``agent``'s point of view."""
    h, w, n = config.height, config.width, config.num_snakes
    cells = float(h * w)
    agent = jnp.asarray(agent, jnp.int32)

    alive = state.alive
    me = jnp.arange(n) == agent
    opp = alive & ~me  # [N] living opponents
    own_alive = alive[agent]

    body = state.body.astype(jnp.float32)  # [N, H, W]
    ys = jnp.arange(h)[None, :, None]
    xs = jnp.arange(w)[None, None, :]
    # One-hot head maps; all-zero for off-board heads.
    head_map = (ys == state.head[:, 1, None, None]) & (xs == state.head[:, 0, None, None])
    opp_cell = opp[:, None, None]

    on_board = jnp.ones((h, w), jnp.float32)
    own_body = jnp.where(own_alive, body[agent], 0.0)
    own_head = head_map[agent] & own_alive
    opp_body = jnp.where(opp_cell, body, 0.0)

    longer = (state.length >= state.length[agent])[:, None, None]  # opp is at least as long
    opp_head = head_map & opp_cell
    opp_head_health = jnp.where(opp_head, state.health[:, None, None] / 100.0, 0.0).max(axis=0)

    own_health = jnp.where(own_alive, state.health[agent] / 100.0, 0.0)
    own_length = jnp.where(own_alive, state.length[agent] / cells, 0.0)

    return jnp.stack(
        [
            on_board,
            state.food.astype(jnp.float32),
            state.hazard.astype(jnp.float32),
            own_head.astype(jnp.float32),
            (own_body > 0).astype(jnp.float32),
            own_body / cells,
            (opp_head & longer).any(axis=0).astype(jnp.float32),
            (opp_head & ~longer).any(axis=0).astype(jnp.float32),
            (opp_body > 0).any(axis=0).astype(jnp.float32),
            opp_body.max(axis=0) / cells,
            opp_head_health,
            on_board * own_health,
            on_board * own_length,
        ],
        axis=-1,
    )


def egocentric_obs(state: State, config: GameConfig, agent: jax.Array | int) -> jax.Array:
    """Head-centered observation: ``f32[2H-1, 2W-1, C]``, or ``f32[H, W, C]`` if wrapped."""
    h, w = config.height, config.width
    agent = jnp.asarray(agent, jnp.int32)
    obs = allocentric_obs(state, config, agent)
    # Clamp so dead snakes with off-board heads still produce a valid window.
    hx = jnp.clip(state.head[agent, 0], 0, w - 1)
    hy = jnp.clip(state.head[agent, 1], 0, h - 1)
    if config.ruleset.wrapped:
        # Torus: roll the head to the center, no padding.
        return jnp.roll(jnp.roll(obs, h // 2 - hy, axis=0), w // 2 - hx, axis=1)
    # Pad to (3H-2, 3W-2); padded index of cell (y, x) is (y+H-1, x+W-1), so a
    # (2H-1, 2W-1) window starting at (hy, hx) has the head at [H-1, W-1].
    padded = jnp.pad(obs, ((h - 1, h - 1), (w - 1, w - 1), (0, 0)))
    return jax.lax.dynamic_slice(padded, (hy, hx, 0), (2 * h - 1, 2 * w - 1, NUM_CHANNELS))


def obs_shape(config: GameConfig, kind: str = "egocentric") -> tuple[int, int, int]:
    """Per-agent observation shape ``(rows, cols, NUM_CHANNELS)`` for ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {_KINDS}")
    h, w = config.height, config.width
    if kind == "egocentric" and not config.ruleset.wrapped:
        return (2 * h - 1, 2 * w - 1, NUM_CHANNELS)
    return (h, w, NUM_CHANNELS)


def make_obs_fn(config: GameConfig, kind: str = "egocentric") -> Callable[[State], jax.Array]:
    """Build ``obs_fn(state) -> f32[N, *obs_shape(config, kind)]`` over all agents."""
    if kind not in _KINDS:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {_KINDS}")
    single = egocentric_obs if kind == "egocentric" else allocentric_obs
    agents = jnp.arange(config.num_snakes)

    def obs_fn(state: State) -> jax.Array:
        return jax.vmap(lambda a: single(state, config, a))(agents)

    return obs_fn
