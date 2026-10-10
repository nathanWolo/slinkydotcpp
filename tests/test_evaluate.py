"""Tests for slinky.evaluate."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import evaluate
from slinky.env import BattlesnakeEnv
from slinky.evaluate import MatchResult, greedy_from_q, play_match, random_legal
from slinky.types import LEFT, NUM_ACTIONS, RIGHT, GameConfig

# Policies here ignore observations, so skip computing them (much faster).
ENV = BattlesnakeEnv(GameConfig(), obs=None)


def always_left(key, state, ts):
    """Walks into the left wall within ~10 turns (spawns are at x in {1, 5, 9})."""
    return jnp.full((ENV.num_agents,), LEFT, jnp.int32)


def left_in_seat0_else_random(key, state, ts):
    """Suicidal in seat 0, random legal in the other seat."""
    rand = random_legal(ENV)(key, state, ts)
    return rand.at[0].set(LEFT)


def check_consistent(r: MatchResult, num_games: int):
    assert r.num_games == num_games
    assert r.wins + r.draws + r.losses == num_games
    assert 0 <= r.truncated <= r.draws
    assert r.score == pytest.approx((r.wins + 0.5 * r.draws) / num_games)
    assert r.score_ci95 >= 0


def test_random_vs_random_is_symmetric():
    p = random_legal(ENV)
    r = play_match(ENV, p, p, jax.random.key(0), num_games=2000, batch_size=500)
    check_consistent(r, 2000)
    assert abs(r.score - 0.5) < 0.05
    # Same score from B's side up to noise: wins and losses are about equal.
    assert abs(r.wins - r.losses) < 0.1 * r.num_games
    assert r.truncated == 0
    assert 5 < r.mean_turns < 300


def test_suicidal_policy_loses_to_random_legal():
    # max_turns is huge: the while_loop must exit as soon as all games are over.
    r = play_match(
        ENV, always_left, random_legal(ENV), jax.random.key(1), 400, max_turns=10**6, batch_size=200
    )
    check_consistent(r, 400)
    assert r.losses > 0.9 * r.num_games
    assert r.score < 0.05
    assert r.truncated == 0
    assert r.mean_turns <= 11


def test_random_legal_beats_suicidal_policy():
    r = play_match(ENV, random_legal(ENV), always_left, jax.random.key(2), 400, batch_size=200)
    assert r.wins > 0.9 * r.num_games
    assert r.score > 0.95


def test_seats_are_balanced():
    # A is suicidal only when it sits in seat 0, and so is B. With A in seat 0 in half
    # the games (and seat 1 in the other half), A loses one half and wins the other.
    n = 400
    r = play_match(
        ENV, left_in_seat0_else_random, left_in_seat0_else_random, jax.random.key(3), n,
        batch_size=100,
    )  # fmt: skip
    check_consistent(r, n)
    assert abs(r.wins - n / 2) < 0.1 * n
    assert abs(r.losses - n / 2) < 0.1 * n

    # With one slot, or a number of slots that is not a multiple of N, the seat still
    # alternates with the game index: refilled slots don't unbalance it.
    for slots in (1, 7):
        r1 = play_match(
            ENV, left_in_seat0_else_random, left_in_seat0_else_random, jax.random.key(3), 40,
            batch_size=slots,
        )  # fmt: skip
        assert abs(r1.wins - 20) <= 4 and abs(r1.losses - 20) <= 4

    # Only A suicidal-in-seat-0: A loses nearly all of its seat-0 games, ~even otherwise.
    p = random_legal(ENV)
    r2 = play_match(ENV, left_in_seat0_else_random, p, jax.random.key(4), n, batch_size=100)
    assert 0.2 < r2.score < 0.3  # ~0.5 * (0 + 0.5)


def test_max_turns_truncation_counts_as_draw():
    p = random_legal(ENV)
    r = play_match(ENV, p, p, jax.random.key(5), 64, max_turns=2, batch_size=32)
    check_consistent(r, 64)
    assert r.truncated == 64
    assert r.draws == 64 and r.wins == 0 and r.losses == 0
    assert r.mean_turns == 2
    assert r.score == 0.5

    # Truncation from the env's own max_turns is also reported.
    env = BattlesnakeEnv(GameConfig(max_turns=3), obs=None)
    p = random_legal(env)
    r = play_match(env, p, p, jax.random.key(6), 32, batch_size=32)
    assert r.truncated == 32 and r.draws == 32
    assert r.mean_turns == 3


def test_eliminated_policy_loses_with_more_snakes():
    # With 3 snakes the game goes on after A dies. A must score a loss even if the
    # survivors are then cut off by max_turns, or all die later on the same turn.
    env = BattlesnakeEnv(GameConfig(num_snakes=3), obs=None)

    def left(key, state, ts):
        return jnp.full((3,), LEFT, jnp.int32)

    p = random_legal(env)
    r = play_match(env, left, p, jax.random.key(12), 150, max_turns=15, batch_size=150)
    check_consistent(r, 150)
    assert r.truncated == 0  # A is dead by turn 15 in every game
    assert r.losses >= 0.9 * 150
    r = play_match(env, left, p, jax.random.key(13), 150, batch_size=150)
    check_consistent(r, 150)
    assert r.score < 0.1


def test_games_not_a_multiple_of_the_slots():
    p = random_legal(ENV)
    r = play_match(ENV, p, p, jax.random.key(7), num_games=70, max_turns=2, batch_size=32)
    assert r.num_games == 70 and r.truncated == 70 and r.draws == 70
    assert r.mean_turns == 2  # idle slots (and their extra games) would change this


def test_result_does_not_depend_on_batch_size():
    # Game g depends only on (key, g): the slots, and when a slot picks it up, don't matter.
    p = random_legal(ENV)
    runs = [
        evaluate.run_match(ENV, p, p, jax.random.key(20), 64, max_turns=300, batch_size=b)
        for b in (1, 5, 64)
    ]
    assert runs[0].result == runs[1].result == runs[2].result
    assert runs[0].result.truncated == 0
    # Fewer games than slots: the slots shrink to the number of games.
    small = evaluate.run_match(ENV, p, p, jax.random.key(20), 5, max_turns=300, batch_size=64)
    assert small.slots == 5
    assert small.result == evaluate.run_match(ENV, p, p, jax.random.key(20), 5, 300, 1).result
    # The heuristic (float scores) is just as invariant.
    from slinky.heuristic import heuristic

    h = heuristic(ENV)
    a = play_match(ENV, h, p, jax.random.key(21), 20, max_turns=300, batch_size=3)
    b = play_match(ENV, h, p, jax.random.key(21), 20, max_turns=300, batch_size=20)
    assert a == b


def test_refill_keeps_slots_busy():
    p = random_legal(ENV)
    key, games, slots = jax.random.key(22), 64, 8
    run = evaluate.run_match(ENV, p, p, key, games, max_turns=300, batch_size=slots)
    total_turns = round(run.result.mean_turns * games)
    assert run.slots == slots and run.iterations * slots >= total_turns
    # Without refill each block of 8 games would last as long as its longest game:
    # play the same games (by id) block by block to measure that.
    blocks = [
        evaluate.run_match(ENV, p, p, key, slots, 300, slots, first_game=g)
        for g in range(0, games, slots)
    ]
    fixed = sum(b.iterations for b in blocks)
    for field in ("wins", "draws", "losses", "truncated"):
        assert sum(getattr(b.result, field) for b in blocks) == getattr(run.result, field)
    assert sum(b.result.mean_turns * slots for b in blocks) == pytest.approx(total_turns)
    assert run.iterations < 0.75 * fixed
    assert run.utilization > 0.8
    # One slot is never idle: one loop turn per game turn.
    one = evaluate.run_match(ENV, p, p, key, 16, max_turns=300, batch_size=1)
    assert one.iterations == round(one.result.mean_turns * 16) and one.utilization == 1.0


def test_game_ranges_add_up():
    p = random_legal(ENV)
    key = jax.random.key(23)
    whole = evaluate.run_match(ENV, p, p, key, 30, 300, 4).result
    head = evaluate.run_match(ENV, p, p, key, 18, 300, 4).result
    tail = evaluate.run_match(ENV, p, p, key, 12, 300, 4, first_game=18).result
    assert (head.wins + tail.wins, head.draws + tail.draws, head.losses + tail.losses) == (
        whole.wins, whole.draws, whole.losses
    )  # fmt: skip
    assert head.mean_turns * 18 + tail.mean_turns * 12 == pytest.approx(whole.mean_turns * 30)
    with pytest.raises(ValueError):
        evaluate.run_match(ENV, p, p, key, 4, first_game=-1)


def test_progress_reports():
    p = random_legal(ENV)
    seen = []
    run = evaluate.run_match(
        ENV, p, p, jax.random.key(24), 48, 300, 8, progress=lambda f, i: seen.append((f, i))
    )
    finished = [f for f, _ in seen]
    assert finished == sorted(finished) and finished[-1] == 48 and len(seen) <= 16
    assert seen[-1][1] == run.iterations
    # A failing callback is reported but does not stop the match.
    assert evaluate.run_match(
        ENV, p, p, jax.random.key(24), 48, 300, 8, progress=lambda f, i: 1 / 0
    ) == run  # fmt: skip


def test_deterministic_given_key():
    p = random_legal(ENV)
    a = play_match(ENV, p, p, jax.random.key(8), 100, batch_size=50)
    b = play_match(ENV, p, p, jax.random.key(8), 100, batch_size=50)
    c = play_match(ENV, p, p, jax.random.key(9), 100, batch_size=50)
    assert a == b
    assert a != c


def test_jitted_function_is_cached():
    p = random_legal(ENV)
    assert random_legal(ENV) is p
    play_match(ENV, p, always_left, jax.random.key(10), 16, batch_size=16)
    before = evaluate._compiled_match.cache_info()
    # num_games, max_turns and first_game are traced: no recompilation.
    play_match(ENV, p, always_left, jax.random.key(11), 16, max_turns=50, batch_size=16)
    play_match(ENV, p, always_left, jax.random.key(11), 40, max_turns=7, batch_size=16)
    evaluate.run_match(ENV, p, always_left, jax.random.key(1), 20, 9, 16, first_game=3)
    after = evaluate._compiled_match.cache_info()
    assert after.misses == before.misses and after.hits == before.hits + 3


def test_solo_is_rejected():
    env = BattlesnakeEnv(GameConfig(num_snakes=1), obs=None)
    with pytest.raises(ValueError):
        play_match(env, random_legal(env), random_legal(env), jax.random.key(0), 4)


def test_summary_statistics():
    r = evaluate._summarize(wins=60, draws=20, losses=20, truncated=5, total_turns=3000)
    assert r.num_games == 100
    assert r.score == pytest.approx(0.7)
    assert r.mean_turns == pytest.approx(30.0)
    outcomes = np.array([1.0] * 60 + [0.5] * 20 + [0.0] * 20)
    expected = 1.96 * outcomes.std(ddof=1) / math.sqrt(100)
    assert r.score_ci95 == pytest.approx(expected, rel=1e-3)
    # A shutout has no variance.
    assert evaluate._summarize(10, 0, 0, 0, 100).score_ci95 == 0.0


def test_greedy_from_q_never_picks_illegal_action():
    env = BattlesnakeEnv(GameConfig())  # with observations
    state, ts = env.reset(jax.random.key(0))
    # Snake 0 and 1 stand on the left wall facing down it, so LEFT is a wall hit and UP is
    # the neck: only DOWN and RIGHT are legal.
    head = jnp.array([[0, 3], [0, 8]], jnp.int32)
    body = jnp.zeros_like(state.body)
    for i, (x, y) in enumerate(np.asarray(head)):
        body = body.at[i, y, x].set(3).at[i, y + 1, x].set(2).at[i, y + 2, x].set(1)
    state = state._replace(head=head, body=body)
    mask = env.action_mask(state)
    np.testing.assert_array_equal(np.asarray(mask), [[False, True, False, True]] * 2)

    def q_prefers_left(obs):
        assert obs.shape[0] == env.num_agents
        return jnp.tile(jnp.array([50.0, 0.5, 100.0, 1.0]), (obs.shape[0], 1))

    ts = ts._replace(action_mask=mask)
    actions = greedy_from_q(q_prefers_left)(jax.random.key(1), state, ts)
    assert actions.shape == (env.num_agents,) and actions.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(actions), RIGHT
    )  # best legal move is RIGHT (q=1 > 0.5)

    # Hand-built mask: the argmax is restricted to legal moves for every row.
    mask = jnp.array([[True, False, False, True], [False, True, True, False]])
    q = jnp.array([[0.1, 9.0, 9.0, 0.2], [9.0, 0.3, 0.4, 9.0]])
    out = greedy_from_q(lambda obs: q)(jax.random.key(0), state, ts._replace(action_mask=mask))
    np.testing.assert_array_equal(np.asarray(out), [3, 2])


def test_greedy_from_q_plays_a_match():
    env = BattlesnakeEnv(GameConfig())  # egocentric observations

    def q_fn(obs):  # obs: f32[N, 21, 21, 13]; arbitrary but deterministic scores
        return jnp.stack([jnp.sum(obs[:, 10, 10 + d, :], axis=-1) for d in range(4)], axis=-1)

    assert q_fn(env.reset(jax.random.key(0))[1].obs).shape == (env.num_agents, NUM_ACTIONS)
    r = play_match(env, greedy_from_q(q_fn), random_legal(env), jax.random.key(0), 8, 20, 8)
    check_consistent(r, 8)
