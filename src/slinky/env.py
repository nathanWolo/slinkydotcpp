"""Functional multi-agent environment API.

All snakes act simultaneously. Every method is a pure function of its
inputs, so it can be wrapped in ``jax.jit``, ``jax.vmap`` over a batch of games,
and ``jax.lax.scan`` over turns::

    env = BattlesnakeEnv(GameConfig())          # 11x11 1v1 duel
    state, ts = env.reset(key)
    state, ts = env.step(key, state, actions)   # actions: int[N]

Per-agent outputs (``ts.obs``, ``ts.reward``, ``ts.alive``, ``ts.action_mask``)
have a leading axis of size ``num_snakes``. Agent ``i`` is snake ``i``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from slinky import rules
from slinky.maps import get_map
from slinky.types import NUM_ACTIONS, GameConfig, Ruleset, State, TimeStep

RewardFn = Callable[[State, State, GameConfig], jax.Array]
ObsFn = Callable[[State], Any]


def win_loss_reward(prev: State, state: State, config: GameConfig) -> jax.Array:
    """Sparse zero-sum-style reward.

    * -1 on the turn a snake is eliminated,
    * +1 to the last snake standing when the game ends (not in solo games),
    * 0 to snakes eliminated on the final turn if *nobody* survives (a draw).
    """
    newly_dead = prev.alive & ~state.alive
    over = rules.is_game_over(state.alive, config) & ~prev.done
    draw = over & ~jnp.any(state.alive)
    reward = jnp.where(newly_dead & ~draw, -1.0, 0.0)
    if not config.solo:
        reward = jnp.where(over & state.alive, 1.0, reward)
    return reward.astype(jnp.float32)


class BattlesnakeEnv:
    """A Battlesnake game as a pure-functional JAX environment.

    Args:
      config: static game configuration (board size, ruleset, map, settings).
      obs: ``"egocentric"`` (default), ``"allocentric"``, a callable mapping a
        :class:`State` to a per-agent observation pytree, or ``None`` for no
        observations (e.g. when running search on raw states).
      reward_fn: ``(prev_state, state, config) -> float32[N]``.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        obs: str | ObsFn | None = "egocentric",
        reward_fn: RewardFn = win_loss_reward,
    ) -> None:
        self.config = config = config or GameConfig()
        if config.ruleset == Ruleset.ROYALE:
            raise NotImplementedError("the royale ruleset is not implemented yet")
        self.map = get_map(config.map)
        self.reward_fn = reward_fn
        if isinstance(obs, str):
            from slinky.observations import make_obs_fn

            self.obs_fn: ObsFn | None = make_obs_fn(config, obs)
        else:
            self.obs_fn = obs

    @property
    def num_agents(self) -> int:
        return self.config.num_snakes

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def reset(self, key: jax.Array) -> tuple[State, TimeStep]:
        """Start a new game (engine turn 0)."""
        state = self.map.setup(key, self.config)
        if self.config.ruleset.constrictor:
            # The engine runs the ruleset once at initialization; constrictor
            # removes all food then.
            state = state._replace(food=jnp.zeros_like(state.food))
        state = state._replace(done=rules.is_game_over(state.alive, self.config))
        zeros = jnp.zeros((self.num_agents,), jnp.float32)
        return state, self._timestep(state, zeros, jnp.zeros((), bool))

    def step(self, key: jax.Array, state: State, actions: jax.Array) -> tuple[State, TimeStep]:
        """Advance one turn. Stepping a finished game is a no-op with zero reward.

        Args:
          key: PRNG key for map randomness (food spawning).
          state: current state.
          actions: int[N]; ``UP=0, DOWN=1, LEFT=2, RIGHT=3``. Any other value is
            an invalid move and gets the engine's default (repeat the last move).
        """
        config = self.config
        new = rules.rules_step(state, actions, config)
        new = self.map.post_update(key, new, config)
        new = new._replace(turn=state.turn + 1)
        over = rules.is_game_over(new.alive, config)
        truncated = jnp.zeros((), bool)
        if config.max_turns is not None:
            truncated = ~over & (new.turn >= config.max_turns)
        new = new._replace(done=over | truncated)
        new = jax.tree.map(lambda old, cur: jnp.where(state.done, old, cur), state, new)
        reward = jnp.where(state.done, 0.0, self.reward_fn(state, new, config))
        return new, self._timestep(new, reward, truncated & ~state.done)

    def step_autoreset(
        self, key: jax.Array, state: State, actions: jax.Array
    ) -> tuple[State, TimeStep]:
        """Like :meth:`step`, but a finished game is replaced by a fresh one.

        ``reward``, ``done`` and ``truncated`` describe the transition that just
        happened; ``obs``, ``alive`` and ``action_mask`` (and the returned
        state) belong to the new game when ``done`` is True.
        """
        k_step, k_reset = jax.random.split(key)
        new, ts = self.step(k_step, state, actions)
        fresh, fresh_ts = self.reset(k_reset)

        def pick(a, b):
            return jax.tree.map(lambda x, y: jnp.where(ts.done, x, y), a, b)

        ts = ts._replace(
            obs=pick(fresh_ts.obs, ts.obs),
            alive=pick(fresh_ts.alive, ts.alive),
            action_mask=pick(fresh_ts.action_mask, ts.action_mask),
        )
        return pick(fresh, new), ts

    def observe(self, state: State) -> Any:
        return () if self.obs_fn is None else self.obs_fn(state)

    def action_mask(self, state: State) -> jax.Array:
        return rules.action_mask(state, self.config)

    def _timestep(self, state: State, reward: jax.Array, truncated: jax.Array) -> TimeStep:
        return TimeStep(
            obs=self.observe(state),
            reward=reward,
            done=state.done,
            truncated=truncated,
            alive=state.alive,
            action_mask=self.action_mask(state),
        )
