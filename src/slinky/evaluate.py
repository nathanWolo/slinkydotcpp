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

:func:`play_table` generalises this to ``k`` policies at one table, for games with
more than two snakes. A *line-up* says which policy holds each seat, and each
policy is scored by the share of the win its seats earn and by their mean
placement::

    env = BattlesnakeEnv(GameConfig(num_snakes=4))
    t = play_table(env, (a, b), lineup=(0, 1, 0, 1), key=key, num_games=1024)
    print(t.share[0], "+/-", t.share_ci95[0], t.placement[0])
"""

from __future__ import annotations

import functools
import itertools
import math
import sys
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy
from slinky.types import CAUSE_TO_ENGINE, Cause, State, TimeStep

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
            before = finished - num_ended
            jax.lax.cond(  # every report_every games, and once when the last game ends
                (finished // report_every != before // report_every)
                | ((finished >= num_games) & (before < num_games)),
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
    shape (true of the policies in this package on CPU; tested, except that the
    DQN's Q-values can differ in the last bit between a batch of one and larger
    batches, which has not been seen to change a move).

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


class TableResult(NamedTuple):
    """Outcome of a table of ``k`` policies; the tuple fields hold one entry per policy.

    A policy's *share* of a game is the sum over its seats of their share of the
    win: a sole survivor gets 1, and ``j`` snakes tied at the end get ``1/j``
    each (all eliminated on the final turn, or alive when the game is cut off).
    The shares of all policies sum to 1 in every game, so a policy with ``s``
    of the ``N`` seats has parity ``s / N``. Its *placement* in a game is the
    mean over its seats of ``1 + (snakes that outlasted the seat) + 0.5 *
    (other snakes eliminated on the same turn)``, so parity is ``(N + 1) / 2``
    whatever ``s``. For ``N = 2`` the share is :attr:`MatchResult.score`.

    A game is a *win* for a policy if its share is 1, a *loss* if it is 0 and a
    *tie* otherwise, so ``wins + ties + losses == num_games``. ``truncated``
    counts the games cut off by ``max_turns`` (or the env's own limit) with
    snakes of at least two policies alive. With ``early_stop`` a game stops as
    soon as it is decided between policies (see :func:`play_table`), so
    ``mean_turns`` is the mean turn at which games were *decided*; without it,
    the mean game length.

    The ``*_ci95`` fields are half-widths of normal-approximation 95% intervals
    (sample standard deviation of the per-game values); ``share_wilson`` is
    the Wilson score interval ``(low, high)`` of the share, which stays inside
    ``[0, 1]`` and is the better choice for shares near 0 or 1.

    ``deaths[p]`` maps the engine's cause names (``"out-of-health"``,
    ``"wall-collision"``, ``"snake-self-collision"``, ``"snake-collision"``,
    ``"head-collision"``, ``"hazard"``) to the number of policy ``p``'s seats
    that were eliminated by that cause, summed over the games. Only the
    eliminations while the game was contested (snakes of at least two policies
    alive when the turn began) count, as these are what ``early_stop`` leaves
    in, so the counts do not depend on it.

    ``game_share`` and ``game_placement`` are ``float32[num_games, k]``, row
    ``g - first_game`` for game ``g``, if ``per_game`` was requested (else
    ``None``). Game ``g`` starts from the same position and has the same food
    rolls whichever policies play (the env's random stream does not depend on
    ``k``), so results for the same ``key`` can be compared game by game.
    """

    num_games: int
    lineup: tuple[int, ...]  # policy index of each seat in game 0 (game g rotates it by g)
    seats: tuple[int, ...]  # seats held by each policy
    share: tuple[float, ...]
    share_ci95: tuple[float, ...]
    share_wilson: tuple[tuple[float, float], ...]
    placement: tuple[float, ...]
    placement_ci95: tuple[float, ...]
    wins: tuple[int, ...]
    ties: tuple[int, ...]
    losses: tuple[int, ...]
    deaths: tuple[dict[str, int], ...]
    truncated: int
    mean_turns: float
    game_share: np.ndarray | None = None
    game_placement: np.ndarray | None = None


class TableRun(NamedTuple):
    """What :func:`run_table` returns: the result and how the slots were used."""

    result: TableResult
    slots: int  # games simulated at once: ``min(batch_size, num_games)``
    iterations: int  # turns of the slot loop; each advances every slot by one turn

    @property
    def utilization(self) -> float:
        """Fraction of slot-turns spent on games that count (1.0 = no slot ever idled)."""
        return self.result.mean_turns * self.result.num_games / (self.slots * self.iterations)


# Games whose per-game values one compiled call can record (a ``per_game`` run of more
# games is played in chunks of this size, so the size of a run never changes the program).
_RECORDED_GAMES = 1 << 16

# Elimination causes in the order of ``TableResult.deaths``' histogram (all but NONE).
_DEATH_CAUSES = tuple(c for c in Cause if c != Cause.NONE)


@functools.lru_cache(maxsize=16)
def _compiled_table(
    env: BattlesnakeEnv, policies: tuple[Policy, ...], slots: int, early_stop: bool, capacity: int
) -> Callable[..., tuple[jax.Array, ...]]:
    """Jitted ``(key, lineup, first_game, num_games, max_turns, call) -> outcome arrays``.

    Plays games ``first_game ... first_game + num_games - 1`` in ``slots`` slots
    and returns ``(share_hist, place_hist, death_hist, truncated, turns,
    iterations, game_share, game_place)``. Per-game values are integers, so sums
    are exact and do not depend on the order in which slots finish:

    * a policy's share of a game in units of ``1 / scale``, with ``scale`` the
      least common multiple of ``1 .. N`` (shares are ``seats_in_group / group_size``);
    * twice its placement summed over its seats.

    ``share_hist[p, v]`` and ``place_hist[p, v]`` count the games in which policy
    ``p`` had value ``v``; ``death_hist[p, c]`` counts its seats eliminated by
    ``_DEATH_CAUSES[c]`` while the game was contested; ``turns[s]`` is the total
    length of the games slot ``s`` finished. ``game_share`` and ``game_place``
    (``[capacity, k]``, empty for ``capacity == 0``) hold the values of every game
    by game id.
    """
    n = env.num_agents
    k = len(policies)
    scale = math.lcm(*range(1, n + 1))
    seats = jnp.arange(n)
    policy_ids = jnp.arange(k)
    causes = jnp.array(_DEATH_CAUSES, jnp.int8)
    rows = jnp.broadcast_to(policy_ids, (slots, k))
    never = jnp.iinfo(jnp.int32).max  # "eliminated on turn infinity": still alive
    # Key 2 of a turn's split is the env's whatever k, as in play_match (whose policies
    # take keys 0 and 1), so tables of different sizes share the env's random stream.
    key_of = [i if i < 2 else i + 1 for i in range(k)]
    num_keys = max(k, 2) + 1

    def start(key, game):
        # Everything random in a game derives from (key, game id) and its own turn number.
        k_reset, k_loop = jax.random.split(jax.random.fold_in(key, game))
        state, ts = env.reset(k_reset)
        return state, ts, k_loop

    def outcome(states, mine):
        """Per policy: (share in 1/scale units, 2 * placement summed over seats, has living)."""
        alive, elim = states.alive, states.elim_turn  # [S, N]
        # The snakes sharing the win: the survivors, or if nobody survived the
        # ones eliminated on the final turn.
        last = jnp.max(jnp.where(alive, 0, elim), axis=-1)
        group = jnp.where(jnp.any(alive, axis=-1)[:, None], alive, elim == last[:, None])
        unit = scale // jnp.maximum(jnp.sum(group, axis=-1), 1)  # [S]
        share = jnp.sum(group[:, :, None] & mine, axis=1) * unit[:, None]
        # Twice the placement: 2 + 2 * (outlasted by) + (tied with), alive = eliminated at infinity.
        out = jnp.where(alive, never, elim)
        outlasted = jnp.sum(out[:, None, :] > out[:, :, None], axis=-1)
        tied = jnp.sum(out[:, None, :] == out[:, :, None], axis=-1) - 1
        place2 = jnp.sum((2 + 2 * outlasted + tied)[:, :, None] * mine, axis=1)
        return share, place2, jnp.any(alive[:, :, None] & mine, axis=1)

    def deaths(states, mine):
        """Per policy and cause: the seats eliminated by the turn just played, [S, k, causes]."""
        fell = ~states.alive & (states.elim_turn == states.turn[:, None])  # [S, N]
        by_cause = states.elim_cause[:, :, None] == causes  # [S, N, causes]
        return jnp.einsum(
            "snk,snc->skc", (fell[:, :, None] & mine).astype(jnp.int32), by_cause.astype(jnp.int32)
        )

    def run(key, lineup, first_game, num_games, max_turns, call):
        report_every = jnp.maximum(num_games // _REPORTS, 1)
        game = jnp.arange(slots, dtype=jnp.int32)  # game held by each slot, from first_game
        states, ts, k_loop = jax.vmap(start, in_axes=(None, 0))(key, first_game + game)
        acc = (
            jnp.zeros((k, scale + 1), jnp.int32),  # share_hist
            jnp.zeros((k, 2 * n * n + 1), jnp.int32),  # place_hist
            jnp.zeros((k, len(_DEATH_CAUSES)), jnp.int32),  # death_hist
            jnp.zeros((), jnp.int32),  # truncated
            jnp.zeros((slots,), jnp.int32),  # turns
            jnp.zeros((capacity, k), jnp.int32),  # game_share
            jnp.zeros((capacity, k), jnp.int32),  # game_place
        )
        contested = jnp.ones((slots,), bool)  # snakes of two policies alive at the turn's start

        def cond(carry):
            return jnp.any(carry[0] < num_games)

        def body(carry):
            game, next_game, states, ts, k_loop, contested, acc, it = carry
            share_hist, place_hist, death_hist, truncated, turns, game_share, game_place = acc
            # Seat s of game g holds lineup[(s - g) % N].
            owner = lineup[(seats[None, :] - (first_game + game)[:, None]) % n]  # [S, N]
            mine = owner[:, :, None] == policy_ids  # [S, N, k]: seat belongs to policy
            # Distinct keys for each policy and the env, per game and turn.
            ks = jax.vmap(lambda kk, t: jax.random.split(jax.random.fold_in(kk, t), num_keys))(
                k_loop, states.turn
            )
            acts = jnp.stack(
                [jax.vmap(p)(ks[:, key_of[i]], states, ts) for i, p in enumerate(policies)]
            )
            actions = jnp.take_along_axis(acts, owner[None], axis=0)[0].astype(jnp.int32)
            states, ts = jax.vmap(env.step)(ks[:, 2], states, actions)

            share, place2, present = outcome(states, mine)
            valid = game < num_games
            was_contested = contested
            contested = jnp.sum(present, axis=-1) >= 2  # snakes of at least two policies alive
            over = states.done | (states.turn >= max_turns)
            if early_stop:
                over = over | ~contested  # the rest is played between a policy's own snakes
            ended = valid & over
            count = jnp.broadcast_to(ended[:, None], (slots, k)).astype(jnp.int32)
            counted = (valid & was_contested)[:, None, None]
            acc = (
                share_hist.at[rows, share].add(count),
                place_hist.at[rows, place2].add(count),
                death_hist + jnp.sum(jnp.where(counted, deaths(states, mine), 0), axis=0),
                truncated + jnp.sum(ended & contested, dtype=jnp.int32),
                turns + jnp.where(ended, states.turn, 0),
                game_share,
                game_place,
            )
            if capacity:
                slot_game = jnp.where(ended, game, capacity)  # out of range: not recorded
                acc = acc[:5] + (
                    game_share.at[slot_game].set(share, mode="drop"),
                    game_place.at[slot_game].set(place2, mode="drop"),
                )
            contested = contested | ended  # a slot that takes a new game starts contested

            # Each slot whose game ended takes the next game id (ids >= num_games idle).
            game = jnp.where(ended, next_game + jnp.cumsum(ended) - 1, game).astype(jnp.int32)
            num_ended = jnp.sum(ended, dtype=jnp.int32)
            next_game = next_game + num_ended

            finished = next_game - slots  # games started minus the slots' current ones
            before = finished - num_ended
            jax.lax.cond(  # every report_every games, and once when the last game ends
                (finished // report_every != before // report_every)
                | ((finished >= num_games) & (before < num_games)),
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
            return game, next_game, states, ts, k_loop, contested, acc, it + 1

        init = (game, jnp.int32(slots), states, ts, k_loop, contested, acc, jnp.int32(0))
        out = jax.lax.while_loop(cond, body, init)
        share_hist, place_hist, death_hist, truncated, turns, game_share, game_place = out[6]
        return (
            share_hist, place_hist, death_hist, truncated, turns, out[7], game_share, game_place
        )  # fmt: skip

    return jax.jit(run)


def play_table(
    env: BattlesnakeEnv,
    policies: Sequence[Policy],
    lineup: Sequence[int],
    key: jax.Array,
    num_games: int,
    max_turns: int = 1000,
    batch_size: int = 1024,
    early_stop: bool = True,
    per_game: bool = False,
) -> TableResult:
    """Play ``num_games`` games of ``policies`` seated as ``lineup``.

    ``lineup[s]`` is the index into ``policies`` of the policy that holds seat
    ``s`` in game 0, e.g. ``(0, 1, 0, 1)`` for two teams of two. Game ``g``
    rotates it: seat ``s`` holds ``lineup[(s - g) % N]``, so every policy
    spends equal time in every seat of its line-up. For ``(0, 1)`` or ``(0, 1,
    1, 1)`` the policy 0 is in seat ``g % N``, as policy A is in
    :func:`play_match`, whose ``wins``, ``draws`` and ``losses`` are then the
    ``wins``, ``ties`` and ``losses`` of policy 0 (same ``key``, same games).
    Every policy must hold at least one seat. Each policy is called once per turn,
    for all seats, and each seat takes the entry of the policy that holds it, so
    list a policy once however many seats it has.

    With ``early_stop`` a game ends as soon as all living snakes belong to the
    same policy, and those survivors are scored as tied. That is exact for the
    quantities of :class:`TableResult` (the policy's seats earn the whole share
    and fill the top places whatever happens next), but ``mean_turns`` then
    measures when games were decided, and shares and placements of individual
    *seats* (not policies) are not observed. It skips the dead time in games
    that one policy has already won. It does not change ``truncated`` or
    ``deaths``.

    ``batch_size`` slots are simulated at once, refilled as in
    :func:`play_match`, and game ``g`` depends only on ``(key, g)``: the result
    does not depend on ``batch_size`` (to the extent that :func:`run_match`
    explains; the per-game values here are integers, so the tallies are exact).
    The jitted loop is cached per ``(env, policies, slots, early_stop,
    per_game)``; ``lineup``, ``num_games`` and ``max_turns`` are traced and do
    not recompile, except that fewer games than ``batch_size`` shrink ``slots``
    to ``num_games``. Each policy and the env draw their own random numbers, and
    the env's (food) do not depend on how many policies there are.
    """
    return run_table(
        env, policies, lineup, key, num_games, max_turns, batch_size, early_stop, per_game=per_game
    ).result


def run_table(
    env: BattlesnakeEnv,
    policies: Sequence[Policy],
    lineup: Sequence[int],
    key: jax.Array,
    num_games: int,
    max_turns: int = 1000,
    batch_size: int = 1024,
    early_stop: bool = True,
    first_game: int = 0,
    per_game: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> TableRun:
    """:func:`play_table` that also reports how the slots were used.

    Plays games ``first_game, ..., first_game + num_games - 1``, which are the
    same games whatever ``num_games`` and ``first_game`` (the rotation uses the
    absolute game id). ``progress`` is called as in :func:`run_match`. With
    ``per_game``, a run of more than 65,536 games is played in chunks of that many
    games, each with its own slot loop (and its own series of reports).
    """
    policies = tuple(policies)
    lineup = tuple(int(i) for i in lineup)
    if env.config.solo:
        raise ValueError("play_table needs a game with at least two snakes (not solo)")
    if num_games < 1 or max_turns < 1 or batch_size < 1:
        raise ValueError("num_games, max_turns and batch_size must be >= 1")
    if first_game < 0 or first_game + num_games >= 2**31:
        raise ValueError("game ids must lie in [0, 2**31)")
    if len(lineup) != env.num_agents:
        raise ValueError(f"lineup needs one policy per seat ({env.num_agents}), got {len(lineup)}")
    if not all(0 <= i < len(policies) for i in lineup):
        raise ValueError(f"lineup entries must index the {len(policies)} policies, got {lineup}")
    seats = tuple(lineup.count(i) for i in range(len(policies)))
    if 0 in seats:
        raise ValueError(f"every policy must hold a seat; none holds policy {seats.index(0)}")
    slots = min(batch_size, num_games)
    capacity = _RECORDED_GAMES if per_game else 0
    run = _compiled_table(env, policies, slots, early_stop, capacity)
    lineup_array = jnp.asarray(lineup, jnp.int32)

    parts, sizes, iterations = [], [], 0
    for done in range(0, num_games, capacity or num_games):
        games = min(capacity or num_games, num_games - done)
        call = next(_CALLS)
        if progress is not None:
            _PROGRESS[call] = _offset_progress(progress, done, iterations)
        try:
            parts.append(
                jax.device_get(
                    run(
                        key,
                        lineup_array,
                        jnp.int32(first_game + done),
                        jnp.int32(games),
                        jnp.int32(max_turns),
                        call,
                    )
                )
            )
        finally:
            _PROGRESS.pop(call, None)
        sizes.append(games)
        iterations += int(parts[-1][5])
    share_hist, place_hist, death_hist = (sum(p[i] for p in parts) for i in range(3))
    truncated = sum(int(p[3]) for p in parts)
    total_turns = sum(int(p[4].sum(dtype=np.int64)) for p in parts)
    scale = share_hist.shape[1] - 1
    shares = np.arange(scale + 1) / scale
    stats = []
    for p, held in enumerate(seats):
        places = np.arange(place_hist.shape[1]) / (2 * held)  # mean placement over the seats
        stats.append((_moments(share_hist[p], shares), _moments(place_hist[p], places)))
    wins = tuple(int(h[scale]) for h in share_hist)
    losses = tuple(int(h[0]) for h in share_hist)
    names = [CAUSE_TO_ENGINE[c] for c in _DEATH_CAUSES]
    result = TableResult(
        num_games=num_games,
        lineup=lineup,
        seats=seats,
        share=tuple(s[0][0] for s in stats),
        share_ci95=tuple(s[0][1] for s in stats),
        share_wilson=tuple(_wilson(s[0][0], num_games) for s in stats),
        placement=tuple(s[1][0] for s in stats),
        placement_ci95=tuple(s[1][1] for s in stats),
        wins=wins,
        ties=tuple(num_games - w - lo for w, lo in zip(wins, losses, strict=True)),
        losses=losses,
        deaths=tuple(dict(zip(names, (int(c) for c in h), strict=True)) for h in death_hist),
        truncated=truncated,
        mean_turns=total_turns / num_games,
    )
    if per_game:
        game_share = np.concatenate([p[6][:g] for p, g in zip(parts, sizes, strict=True)])
        game_place = np.concatenate([p[7][:g] for p, g in zip(parts, sizes, strict=True)])
        result = result._replace(
            game_share=(game_share / scale).astype(np.float32),
            game_placement=(game_place / (2.0 * np.array(seats))).astype(np.float32),
        )
    return TableRun(result, slots, iterations)


def _offset_progress(
    progress: Callable[[int, int], None], games: int, iterations: int
) -> Callable[[int, int], None]:
    """The hook for a chunk that follows ``games`` games played in ``iterations`` turns."""
    return lambda finished, turns: progress(games + finished, iterations + turns)


def _moments(counts: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Mean and 95% half-width of the mean, from counts of games per value."""
    counts = counts.astype(np.int64)
    n = int(counts.sum())
    mean = float(np.dot(counts, values) / n)
    var = float(np.dot(counts, (values - mean) ** 2) / (n - 1)) if n > 1 else 0.0
    return mean, _Z95 * math.sqrt(var / n)


def _wilson(p: float, n: int) -> tuple[float, float]:
    """Wilson score interval of a proportion ``p`` over ``n`` games."""
    z2 = _Z95 * _Z95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if p == 0 else centre - half, 1.0 if p == 1 else centre + half)
