"""Head-to-head evaluation: play batches of games between two policies.

A *policy* here is a pure function ``(key, state, timestep) -> int32[N]`` for one
(unbatched) game. It returns actions for **all** snakes, but a match only uses
the entries of the seats that the policy controls, so a policy can be written
without knowing which seat it plays::

    env = BattlesnakeEnv(GameConfig())
    result = play_match(env, greedy_from_q(q_fn), random_legal(env), key, num_games=2048)
    print(result.score, "+/-", result.score_ci95)

:func:`play_match` runs ``batch_size`` games at a time under ``jit(vmap)``, with
a ``lax.while_loop`` that stops as soon as every game in the batch is over (or
``max_turns`` turns have been played). The jitted function is cached per
``(env, policy_a, policy_b, batch_size)``, so evaluating the same policy objects
repeatedly (e.g. during training) compiles once. Policies are traced, so any
parameters they close over are baked in as constants: a *new* closure (e.g. one
holding fresh network parameters) compiles again.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy
from slinky.types import State, TimeStep

# (key, state, timestep) -> int32[N]: actions for all snakes of one game.
Policy = Callable[[jax.Array, State, TimeStep], jax.Array]

# Per-game outcome codes from policy A's point of view.
_LOSS, _DRAW, _WIN = 0, 1, 2
_Z95 = 1.959963984540054


class MatchResult(NamedTuple):
    """Outcome counts of a match, from policy A's point of view.

    ``truncated`` games (cut off by ``max_turns`` while two or more snakes were
    alive) count as draws and are *also* counted in ``truncated``, so
    ``wins + draws + losses == num_games`` and ``truncated <= draws``.
    """

    wins: int
    draws: int
    losses: int
    truncated: int
    num_games: int
    mean_turns: float
    score: float  # (wins + 0.5 * draws) / num_games
    score_ci95: float  # half-width of a 95% normal-approx CI on ``score``


def random_legal(env: BattlesnakeEnv) -> Policy:
    """Uniformly random over each snake's legal moves (``policies.random_legal_policy``)."""
    return _random_legal(env)


@functools.lru_cache(maxsize=16)
def _random_legal(env: BattlesnakeEnv) -> Policy:
    # Cached so repeated calls return the same object and hit play_match's jit cache.
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return random_legal_policy(key, state, env)

    return policy


def greedy_from_q(q_fn: Callable[[Any], jax.Array]) -> Policy:
    """Greedy policy: ``argmax_a q_fn(obs)[a]`` over the legal moves in ``ts.action_mask``.

    Args:
      q_fn: ``obs[N, ...] -> q[N, 4]``. Illegal actions are masked with ``-inf``
        before the argmax; mask rows are never all-False, so a legal action
        always exists.
    """
    return _greedy_from_q(q_fn)


@functools.lru_cache(maxsize=16)
def _greedy_from_q(q_fn: Callable[[Any], jax.Array]) -> Policy:
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        q = jnp.where(ts.action_mask, q_fn(ts.obs), -jnp.inf)
        return jnp.argmax(q, axis=-1).astype(jnp.int32)

    return policy


@functools.lru_cache(maxsize=16)
def _compiled_batch(
    env: BattlesnakeEnv, policy_a: Policy, policy_b: Policy, batch_size: int
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Jitted ``(key, offset, num_games, max_turns) -> (outcome, truncated, turns)`` for one batch.

    Game ``offset + i`` puts policy A in seat ``(offset + i) % N``. Games with
    index ``>= num_games`` are padding: they start finished and are ignored.
    """
    n = env.num_agents

    def run(key, offset, num_games, max_turns):
        idx = offset + jnp.arange(batch_size, dtype=jnp.int32)
        seat = idx % n  # A's seat in each game
        mine = seat[:, None] == jnp.arange(n)  # [B, N] seats controlled by A

        # Independent keys per game; each game splits its own into a reset key
        # and a loop key that is folded with the turn number below.
        game_keys = jax.vmap(jax.random.split)(jax.random.split(key, batch_size))
        k_reset, k_loop = game_keys[:, 0], game_keys[:, 1]

        states, ts = jax.vmap(env.reset)(k_reset)
        states = states._replace(done=states.done | (idx >= num_games))

        def cond(carry):
            t, states, _ = carry
            return (t < max_turns) & ~jnp.all(states.done)

        def body(carry):
            t, states, ts = carry
            # Distinct keys for policy A, policy B and the env, per game and turn.
            ks = jax.vmap(lambda k: jax.random.split(jax.random.fold_in(k, t), 3))(k_loop)
            act_a = jax.vmap(policy_a)(ks[:, 0], states, ts)
            act_b = jax.vmap(policy_b)(ks[:, 1], states, ts)
            actions = jnp.where(mine, act_a, act_b).astype(jnp.int32)
            states, ts = jax.vmap(env.step)(ks[:, 2], states, actions)
            return t + 1, states, ts

        _, states, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), states, ts))

        remaining = jnp.sum(states.alive, axis=-1)  # [B]
        a_alive = jnp.take_along_axis(states.alive, seat[:, None], axis=1)[:, 0]
        outcome = jnp.where(remaining == 1, jnp.where(a_alive, _WIN, _LOSS), _DRAW)  # 0 alive: draw
        # Two or more alive means the game was cut off (max_turns), not decided.
        return outcome.astype(jnp.int32), remaining >= 2, states.turn

    return jax.jit(run)


def play_match(
    env: BattlesnakeEnv,
    policy_a: Policy,
    policy_b: Policy,
    key: jax.Array,
    num_games: int,
    max_turns: int = 1000,
    batch_size: int = 1024,
) -> MatchResult:
    """Play ``num_games`` games of ``policy_a`` against ``policy_b``.

    Game ``g`` puts A in seat ``g % N`` and B in every other seat, so seats are
    balanced. A game ends when the rules end it or after ``max_turns`` turns; in
    the latter case (or if ``env.config.max_turns`` cuts it off first) it is a
    draw and counted in ``truncated``. From A's seat, a win is being the only
    snake alive, a draw is nobody alive (or truncation), and a loss is being
    eliminated while another snake survives.

    ``batch_size`` games run at once (the last batch is padded). If
    ``num_games < batch_size`` the batch shrinks to ``num_games`` to avoid
    simulating padding.
    """
    if env.config.solo:
        raise ValueError("play_match needs a game with at least two snakes (not solo)")
    if num_games < 1 or max_turns < 1 or batch_size < 1:
        raise ValueError("num_games, max_turns and batch_size must be >= 1")
    batch_size = min(batch_size, num_games)
    run = _compiled_batch(env, policy_a, policy_b, batch_size)

    num_batches = -(-num_games // batch_size)
    batch_keys = jax.random.split(key, num_batches)
    results = [
        run(batch_keys[b], jnp.int32(b * batch_size), jnp.int32(num_games), jnp.int32(max_turns))
        for b in range(num_batches)  # dispatched asynchronously; synced once below
    ]
    outcome, truncated, turns = (
        np.concatenate(x)[:num_games] for x in zip(*jax.device_get(results), strict=True)
    )
    return _summarize(
        wins=int(np.sum(outcome == _WIN)),
        draws=int(np.sum(outcome == _DRAW)),
        losses=int(np.sum(outcome == _LOSS)),
        truncated=int(np.sum(truncated)),
        total_turns=int(np.sum(turns, dtype=np.int64)),
    )


def _summarize(wins: int, draws: int, losses: int, truncated: int, total_turns: int) -> MatchResult:
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n
    # Sample variance of the per-game outcomes (1 / 0.5 / 0).
    sum_sq = wins + 0.25 * draws
    var = max(sum_sq - n * score * score, 0.0) / (n - 1) if n > 1 else 0.0
    return MatchResult(
        wins=wins,
        draws=draws,
        losses=losses,
        truncated=truncated,
        num_games=n,
        mean_turns=total_turns / n,
        score=score,
        score_ci95=_Z95 * math.sqrt(var / n),
    )
