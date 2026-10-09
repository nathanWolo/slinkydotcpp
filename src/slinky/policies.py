"""Tiny baseline policies, handy for smoke tests, rollouts and benchmarks.

A policy is a pure function ``(key, state, env) -> int32[N]`` that picks one
action per snake for a single (unbatched) game. It composes with ``jax.jit``,
``jax.vmap`` over games and ``jax.lax.scan`` over turns::

    actions = random_legal_policy(key, state, env)           # int32[N]
    batched = jax.jit(jax.vmap(lambda k, s: random_legal_policy(k, s, env)))(keys, states)

``env`` is static configuration, not a JAX value, so close over it rather than
passing it through ``jit``/``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from slinky.env import BattlesnakeEnv
from slinky.types import NUM_ACTIONS, State


def random_policy(key: jax.Array, state: State, env: BattlesnakeEnv) -> jax.Array:
    """Uniformly random over all 4 actions, ignoring the state.

    Returns:
      int32[N] actions, one per snake (dead snakes' entries are ignored by the env).
    """
    return jax.random.randint(key, (env.num_agents,), 0, NUM_ACTIONS, dtype=jnp.int32)


def random_legal_policy(key: jax.Array, state: State, env: BattlesnakeEnv) -> jax.Array:
    """Uniformly random over each snake's legal moves (``env.action_mask(state)``).

    A snake whose mask row is all-False (it is dead, or every move is certain
    death) picks uniformly over all 4 actions instead, so the result is always
    a valid action id.

    Returns:
      int32[N] actions, one per snake.
    """
    mask = env.action_mask(state)  # bool[N, 4]
    mask = jnp.where(jnp.any(mask, axis=-1, keepdims=True), mask, True)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
