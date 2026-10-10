"""Head-to-head evaluation: play batches of games between two policies.

A *policy* here is a pure function ``(key, state, timestep) -> int32[N]`` for one
(unbatched) game. It returns actions for **all** snakes, but a match only uses
the entries of the seats that the policy controls, so a policy can be written
without knowing which seat it plays::

    env = BattlesnakeEnv(GameConfig())
    result = play_match(env, greedy_from_q(q_fn), random_legal(env), key, num_games=2048)
    print(result.score, "+/-", result.score_ci95)

:func:`play_match` simulates ``batch_size`` game *slots* at once under
``jit(vmap)``, in one ``lax.while_loop``: when a slot's game ends, the slot
starts the next game, and the loop stops when every game has been played. Game
lengths are heavy-tailed, so refilling slots saves most of the work that a
fixed batch spends on finished games waiting for the longest one. Game ``g``
depends only on ``(key, g)``, so the result is the same for every
``batch_size``. The jitted function is cached per ``(env, policy_a, policy_b,
slots)``, so evaluating the same policy objects repeatedly (e.g. during
training) compiles once. Policies are traced, so any parameters they close over
are baked in as constants: a *new* closure (e.g. one holding fresh network
parameters) compiles again.
"""

from __future__ import annotations

import functools
import itertools
import math
import sys
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

    ``truncated`` games (cut off by ``max_turns`` while A and at least one other
    snake were alive) count as draws and are *also* counted in ``truncated``, so
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


class MatchRun(NamedTuple):
    """What :func:`run_match` returns: the result and how the slots were used."""

    result: MatchResult
    slots: int  # games simulated at once: ``min(batch_size, num_games)``
    iterations: int  # turns of the slot loop; each advances every slot by one turn

    @property
    def utilization(self) -> float:
        """Fraction of slot-turns spent on games that count (1.0 = no slot ever idled)."""
        return self.result.mean_turns * self.result.num_games / (self.slots * self.iterations)


# Progress hooks of the run_match calls in flight, by call number (traced, so the
# compiled loop is shared by calls with and without a hook).
_REPORTS = 16
_CALLS = itertools.count()
_PROGRESS: dict[int, Callable[[int, int], None]] = {}


def _report(call: Any, finished: Any, iterations: Any) -> None:
    hook = _PROGRESS.get(int(call))
    if hook is not None:
        try:
            hook(int(finished), int(iterations))
        except Exception as e:  # a failing report must not abort the match
            print(f"progress callback failed: {type(e).__name__}: {e}", file=sys.stderr)


@functools.lru_cache(maxsize=16)
def _compiled_match(
    env: BattlesnakeEnv, policy_a: Policy, policy_b: Policy, slots: int
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Jitted ``(key, first_game, num_games, max_turns, call) -> (counts, turns, iterations)``.

    Plays games ``first_game ... first_game + num_games - 1`` in ``slots`` slots.
    ``counts[s]`` holds the wins, draws, losses and truncations (from A's point
    of view) of the games that slot ``s`` finished and ``turns[s]`` their total
    length; ``iterations`` is the number of loop turns. Every ``num_games / 16``
    finished games the loop calls ``_report(call, finished, iterations)`` on the
    host.
    """
    n = env.num_agents
    seats = jnp.arange(n)

    def start(key, game):
        # Everything random in a game derives from (key, game id) and its own turn number.
        k_reset, k_loop = jax.random.split(jax.random.fold_in(key, game))
        state, ts = env.reset(k_reset)
        return state, ts, k_loop

    def outcome(states, seat):
        remaining = jnp.sum(states.alive, axis=-1)  # [S]
        a_alive = jnp.take_along_axis(states.alive, seat[:, None], axis=1)[:, 0]
        a_elim = jnp.take_along_axis(states.elim_turn, seat[:, None], axis=1)[:, 0]
        # Two or more alive means the game was cut off (max_turns), not decided.
        cut_off = a_alive & (remaining >= 2)
        # A died on the final turn together with everyone else (as win_loss_reward).
        all_died = ~a_alive & (remaining == 0) & (a_elim == states.elim_turn.max(axis=-1))
        code = jnp.where(
            a_alive & (remaining == 1), _WIN, jnp.where(cut_off | all_died, _DRAW, _LOSS)
        )
        return code, cut_off

    def run(key, first_game, num_games, max_turns, call):
        report_every = jnp.maximum(num_games // _REPORTS, 1)
        game = jnp.arange(slots, dtype=jnp.int32)  # game held by each slot, from first_game
        states, ts, k_loop = jax.vmap(start, in_axes=(None, 0))(key, first_game + game)
        counts = jnp.zeros((slots, 4), jnp.int32)  # wins, draws, losses, truncated
        turns = jnp.zeros((slots,), jnp.int32)

        def cond(carry):
            return jnp.any(carry[0] < num_games)

        def body(carry):
            game, next_game, states, ts, k_loop, counts, turns, it = carry
            seat = (first_game + game) % n  # A's seat
            # Distinct keys for policy A, policy B and the env, per game and turn.
            ks = jax.vmap(lambda k, t: jax.random.split(jax.random.fold_in(k, t), 3))(
                k_loop, states.turn
            )
            act_a = jax.vmap(policy_a)(ks[:, 0], states, ts)
            act_b = jax.vmap(policy_b)(ks[:, 1], states, ts)
            actions = jnp.where(seat[:, None] == seats, act_a, act_b).astype(jnp.int32)
            states, ts = jax.vmap(env.step)(ks[:, 2], states, actions)

            ended = (game < num_games) & (states.done | (states.turn >= max_turns))
            code, cut_off = outcome(states, seat)
            won = jnp.stack([code == _WIN, code == _DRAW, code == _LOSS, cut_off], axis=-1)
            counts = counts + (ended[:, None] & won)
            turns = turns + jnp.where(ended, states.turn, 0)

            # Each slot whose game ended takes the next game id (ids >= num_games idle).
            game = jnp.where(ended, next_game + jnp.cumsum(ended) - 1, game).astype(jnp.int32)
            num_ended = jnp.sum(ended, dtype=jnp.int32)
            next_game = next_game + num_ended

            finished = next_game - slots  # games started minus the slots' current ones
            jax.lax.cond(
                finished // report_every != (finished - num_ended) // report_every,
                lambda: jax.debug.callback(_report, call, finished, it + 1),
                lambda: None,
            )

            def pick(fresh, old):
                return jnp.where(ended.reshape((-1,) + (1,) * (old.ndim - 1)), fresh, old)

            def refill(old):
                fresh = jax.vmap(start, in_axes=(None, 0))(key, first_game + game)
                return jax.tree.map(pick, fresh, old)

            states, ts, k_loop = jax.lax.cond(
                jnp.any(ended), refill, lambda old: old, (states, ts, k_loop)
            )
            return game, next_game, states, ts, k_loop, counts, turns, it + 1

        init = (game, jnp.int32(slots), states, ts, k_loop, counts, turns, jnp.int32(0))
        out = jax.lax.while_loop(cond, body, init)
        return out[5], out[6], out[7]

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
    draw and counted in ``truncated``. From A's seat, as in ``win_loss_reward``:
    a win is being the last snake standing; a draw is being eliminated on the
    final turn together with every other snake (or truncation while still
    alive); a loss is any other elimination. With more than two snakes, A can
    lose even if the game is later truncated or ends with nobody alive.

    ``batch_size`` is the number of game slots simulated at once (at most
    ``num_games``). When a slot's game ends, the slot starts the next game, so
    no slot waits for the longest game of a batch. Every random number of game
    ``g`` derives from ``(key, g)`` and the game's own turn number, so the
    result does not depend on ``batch_size`` (see :func:`run_match`).
    """
    return run_match(env, policy_a, policy_b, key, num_games, max_turns, batch_size).result


def run_match(
    env: BattlesnakeEnv,
    policy_a: Policy,
    policy_b: Policy,
    key: jax.Array,
    num_games: int,
    max_turns: int = 1000,
    batch_size: int = 1024,
    first_game: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> MatchRun:
    """:func:`play_match` that also reports how the slots were used.

    Plays games ``first_game, ..., first_game + num_games - 1``. Game ``g`` is
    the same game whatever ``batch_size``, ``num_games`` and ``first_game``
    are, so splitting a match into ranges of game ids gives the same games as
    playing it at once. The result is identical for every ``batch_size`` as
    long as the policies compute each game independently of the others in the
    batch (vmapped, as here) and XLA's arithmetic does not depend on the batch
    shape (true of the policies in this package on CPU; tested).

    ``progress``, if given, is called as ``progress(games_finished,
    iterations)`` from inside the loop (a host callback, on another thread)
    about 16 times over the match, for progress reports. Early games are the
    short ones, so the fraction of games finished runs ahead of the work done.

    The compiled loop is cached per ``(env, policy_a, policy_b, slots)``:
    ``num_games``, ``max_turns``, ``first_game`` and ``progress`` do not enter
    it, so changing them does not recompile, except that fewer games than
    ``batch_size`` shrink the slots to ``num_games``.
    """
    if env.config.solo:
        raise ValueError("play_match needs a game with at least two snakes (not solo)")
    if num_games < 1 or max_turns < 1 or batch_size < 1:
        raise ValueError("num_games, max_turns and batch_size must be >= 1")
    if first_game < 0 or first_game + num_games >= 2**31:
        raise ValueError("game ids must lie in [0, 2**31)")
    slots = min(batch_size, num_games)
    run = _compiled_match(env, policy_a, policy_b, slots)
    call = next(_CALLS)
    if progress is not None:
        _PROGRESS[call] = progress
    try:
        counts, turns, iterations = jax.device_get(
            run(key, jnp.int32(first_game), jnp.int32(num_games), jnp.int32(max_turns), call)
        )
    finally:
        _PROGRESS.pop(call, None)
    wins, draws, losses, truncated = (int(x) for x in counts.sum(axis=0, dtype=np.int64))
    result = _summarize(
        wins=wins,
        draws=draws,
        losses=losses,
        truncated=truncated,
        total_turns=int(turns.sum(dtype=np.int64)),
    )
    return MatchRun(result, slots, int(iterations))


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
