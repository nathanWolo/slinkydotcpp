"""Tests for slinky.evaluate."""

from __future__ import annotations

import math
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import evaluate
from slinky.env import BattlesnakeEnv
from slinky.evaluate import (
    MatchResult,
    TableResult,
    greedy_from_q,
    play_match,
    play_table,
    random_legal,
)
from slinky.types import CAUSE_TO_ENGINE, LEFT, NUM_ACTIONS, RIGHT, Cause, GameConfig

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


# --- play_table ----------------------------------------------------------------------

ENV4 = BattlesnakeEnv(GameConfig(num_snakes=4), obs=None)


def clone_of(policy, salt):
    """A new policy object that plays like ``policy`` with its own random stream."""

    def clone(key, state, ts):
        return policy(jax.random.fold_in(key, salt), state, ts)

    return clone


def suicidal_in_seat0(env):
    """A new policy: LEFT into the wall in seat 0 (and so dies), random legal elsewhere."""
    rand = random_legal(env)

    def policy(key, state, ts):
        return rand(key, state, ts).at[0].set(LEFT)

    return policy


RAND4 = random_legal(ENV4)
CLONES4 = (RAND4, clone_of(RAND4, 1), clone_of(RAND4, 2), clone_of(RAND4, 3))


def numbers(r: TableResult, skip=()):
    """The summary fields of a result (not the per-game arrays) as a dict."""
    return {f: getattr(r, f) for f in r._fields if not f.startswith("game_") and f not in skip}


def turn_keys(key, turn, k):
    """The keys of one turn as ``play_table`` hands them out: ``(policy keys, env key)``.

    The env's key is the third of the split whatever ``k``; policies 0 and 1 take the first
    two, as in ``play_match``, and policy ``i >= 2`` the key after the env's.
    """
    ks = jax.random.split(jax.random.fold_in(key, turn), max(k, 2) + 1)
    return [ks[i if i < 2 else i + 1] for i in range(k)], ks[2]


def reference_game(env, step, policies, lineup, key, game, max_turns):
    """Plays one game turn by turn and scores it straight from the definitions.

    ``step`` and ``policies`` are the jitted ``env.step`` and policies. Returns
    ``(share[k], placement[k], truncated, deaths[k])``, where ``deaths[p]`` counts the causes
    (by ``Cause`` code) of policy p's seats eliminated in turns that began with snakes of two
    or more policies alive.
    """
    n, k = env.num_agents, len(policies)
    k_reset, k_loop = jax.random.split(jax.random.fold_in(key, game))
    state, ts = env.reset(k_reset)
    owner = [lineup[(s - game) % n] for s in range(n)]
    deaths = [Counter() for _ in range(k)]
    while not state.done and state.turn < max_turns:
        policy_keys, env_key = turn_keys(k_loop, state.turn, k)
        acts = [np.asarray(p(policy_keys[i], state, ts)) for i, p in enumerate(policies)]
        actions = jnp.array([acts[owner[s]][s] for s in range(n)], jnp.int32)
        contested = len({owner[s] for s in range(n) if state.alive[s]}) >= 2
        before = np.asarray(state.alive)
        state, ts = step(env_key, state, actions)
        if contested:
            for s in np.flatnonzero(before & ~np.asarray(state.alive)):
                deaths[owner[s]][Cause(int(state.elim_cause[s]))] += 1
    alive, elim = np.asarray(state.alive), np.asarray(state.elim_turn)
    sharing = alive if alive.any() else elim == elim.max()  # the survivors, else the last to die
    seat_share = sharing / sharing.sum()
    gone = np.where(alive, np.inf, elim)  # elimination turn, infinity while alive
    seat_place = [
        1 + (gone > gone[s]).sum() + 0.5 * ((gone == gone[s]).sum() - 1) for s in range(n)
    ]
    share = [sum(seat_share[s] for s in range(n) if owner[s] == p) for p in range(k)]
    place = [np.mean([seat_place[s] for s in range(n) if owner[s] == p]) for p in range(k)]
    truncated = len({owner[s] for s in range(n) if alive[s]}) >= 2
    return share, place, truncated, deaths


def test_table_matches_play_match_with_two_players():
    # Policy A of play_match is policy 0 of the line-up (0, 1): same games, same counts.
    p, q = random_legal(ENV), suicidal_in_seat0(ENV)
    key = jax.random.key(30)
    m = play_match(ENV, p, q, key, 200, batch_size=50)
    assert m.wins > 0 and m.draws > 0 and m.losses > 0
    for early_stop in (False, True):
        t = play_table(ENV, (p, q), (0, 1), key, 200, batch_size=50, early_stop=early_stop)
        assert (t.wins[0], t.ties[0], t.losses[0]) == (m.wins, m.draws, m.losses)
        assert (t.wins[1], t.ties[1], t.losses[1]) == (m.losses, m.draws, m.wins)
        assert t.truncated == m.truncated and t.num_games == 200
        assert t.share[0] == pytest.approx(m.score) and t.share[1] == pytest.approx(1 - m.score)
        assert t.share_ci95[0] == pytest.approx(m.score_ci95)
        assert t.seats == (1, 1) and t.lineup == (0, 1)
        assert t.game_share is None and t.game_placement is None
        # Games of two snakes end when one is out, so there is nothing to stop early.
        assert t.mean_turns == m.mean_turns


def test_table_matches_play_match_one_against_three():
    # Line-up (0, 1, 1, 1) is play_match's seating. The game is decided when A is out or
    # alone, so early stopping shortens it but changes no count.
    p, q = CLONES4[0], CLONES4[1]
    key = jax.random.key(31)
    m = play_match(ENV4, p, q, key, 128, max_turns=150, batch_size=32)
    assert m.draws > 0 and m.truncated > 0  # ties and cut-off games are in the comparison
    full = play_table(ENV4, (p, q), (0, 1, 1, 1), key, 128, 150, 32, early_stop=False)
    stopped = play_table(ENV4, (p, q), (0, 1, 1, 1), key, 128, 150, 32, early_stop=True)
    for t in (full, stopped):
        assert (t.wins[0], t.ties[0], t.losses[0]) == (m.wins, m.draws, m.losses)
        assert t.truncated == m.truncated and t.seats == (1, 3)
    assert full.mean_turns == m.mean_turns
    assert stopped.mean_turns < 0.9 * m.mean_turns
    assert numbers(full, skip=("mean_turns",)) == numbers(stopped, skip=("mean_turns",))


def engine_counts(counters):
    """Sums Counters of ``Cause`` codes into ``{engine cause name: count}`` (as ``deaths[p]``)."""
    return {CAUSE_TO_ENGINE[c]: sum(x[c] for x in counters) for c in Cause if c != Cause.NONE}


def test_table_scores_games_by_the_definitions():
    # Compare with games played one by one and scored from the final state by plain numpy,
    # with and without early stopping, so truncation (60 turns) and ties are covered.
    policies = CLONES4[:3]
    key, max_turns, games = jax.random.key(32), 60, 12
    step, jitted = jax.jit(ENV4.step), [jax.jit(p) for p in policies]
    for lineup in ((0, 1, 2, 2), (2, 0, 1, 2)):
        want = [
            reference_game(ENV4, step, jitted, lineup, key, g, max_turns) for g in range(games)
        ]  # fmt: skip
        assert any(w[2] for w in want) and not all(w[2] for w in want)
        deaths = tuple(engine_counts([w[3][p] for w in want]) for p in range(3))
        total = Counter()
        for d in deaths:
            total.update(d)
        assert sum(total.values()) > 10 and sum(count > 0 for count in total.values()) >= 2
        for early_stop in (False, True):
            t = play_table(
                ENV4, policies, lineup, key, games, max_turns, 4, early_stop, per_game=True
            )
            np.testing.assert_allclose(t.game_share, [w[0] for w in want], atol=1e-6)
            np.testing.assert_allclose(t.game_placement, [w[1] for w in want], atol=1e-6)
            assert t.truncated == sum(w[2] for w in want)
            assert t.deaths == deaths


def test_table_shares_sum_to_one():
    key = jax.random.key(33)
    for policies, lineup in ((CLONES4[:2], (0, 1, 0, 1)), (CLONES4[:3], (0, 1, 2, 2))):
        t = play_table(ENV4, policies, lineup, key, 100, 300, 32, per_game=True)
        assert t.game_share.shape == t.game_placement.shape == (100, len(policies))
        assert t.game_share.dtype == t.game_placement.dtype == np.float32
        np.testing.assert_allclose(t.game_share.sum(axis=1), 1.0, atol=1e-6)
        assert sum(t.share) == pytest.approx(1.0)
        # A place is shared out among the seats: the seats' places always sum to 1+2+3+4.
        seats = np.array(t.seats)
        np.testing.assert_allclose((t.game_placement * seats).sum(axis=1), 10.0, atol=1e-5)
        # The aggregates are those of the per-game values.
        for p in range(len(policies)):
            share, place = t.game_share[:, p], t.game_placement[:, p]
            assert t.share[p] == pytest.approx(share.mean(), abs=1e-6)
            assert t.placement[p] == pytest.approx(place.mean(), abs=1e-6)
            assert t.share_ci95[p] == pytest.approx(1.96 * share.std(ddof=1) / 10, rel=1e-4)
            assert t.placement_ci95[p] == pytest.approx(1.96 * place.std(ddof=1) / 10, rel=1e-4)
            assert (t.wins[p], t.losses[p]) == (np.sum(share == 1), np.sum(share == 0))
            assert t.wins[p] + t.ties[p] + t.losses[p] == 100
            assert sum(t.deaths[p].values()) <= t.seats[p] * 100  # a seat is eliminated once
            low, high = t.share_wilson[p]
            assert 0 <= low <= t.share[p] <= high <= 1


def test_table_early_stop_changes_nothing_but_the_length():
    key = jax.random.key(34)
    policies = CLONES4[:3]
    for lineup in ((0, 1, 2, 2), (0, 0, 1, 2), (0, 1, 2, 0)):
        full, stopped = (
            play_table(ENV4, policies, lineup, key, 64, 50, 16, es, per_game=True)
            for es in (False, True)
        )
        np.testing.assert_array_equal(full.game_share, stopped.game_share)
        np.testing.assert_array_equal(full.game_placement, stopped.game_placement)
        assert numbers(full, skip=("mean_turns",)) == numbers(stopped, skip=("mean_turns",))
        assert full.truncated > 0  # some games are cut off at 50 turns in both runs
        assert stopped.mean_turns < full.mean_turns


def test_table_does_not_depend_on_batch_size():
    key = jax.random.key(35)
    policies = CLONES4[:2]
    runs = [
        play_table(ENV4, policies, (0, 1, 0, 1), key, 40, 120, b, per_game=True)
        for b in (1, 7, 40, 64)  # 64 slots for 40 games: the slots shrink to 40
    ]
    for t in runs[1:]:
        assert numbers(t) == numbers(runs[0])
        np.testing.assert_array_equal(t.game_share, runs[0].game_share)
        np.testing.assert_array_equal(t.game_placement, runs[0].game_placement)
    # Without per-game arrays the numbers are the same, and the game ranges add up.
    plain = play_table(ENV4, policies, (0, 1, 0, 1), key, 40, 120, 8)
    assert numbers(plain) == numbers(runs[0])
    head = evaluate.run_table(ENV4, policies, (0, 1, 0, 1), key, 15, 120, 8, per_game=True)
    tail = evaluate.run_table(
        ENV4, policies, (0, 1, 0, 1), key, 25, 120, 8, first_game=15, per_game=True
    )  # fmt: skip
    both = np.concatenate([head.result.game_share, tail.result.game_share])
    np.testing.assert_array_equal(both, runs[0].game_share)


def test_table_rotation_balances_seats():
    # Every policy is suicidal when it sits in seat 0, so whoever holds seat 0 finishes last
    # (barring a random snake dying even sooner). Seat 0 of game g holds lineup[-g % 4]; over
    # a multiple of 4 games each policy sits there equally often, so all end up at parity.
    policies = tuple(suicidal_in_seat0(ENV4) for _ in range(4))
    key, lineup, games = jax.random.key(36), (0, 1, 2, 3), 256
    t = play_table(ENV4, policies, lineup, key, games, 300, 64, per_game=True)
    game = np.arange(games)
    holder = np.array([lineup[-g % 4] for g in game])
    in_seat0 = np.zeros((games, 4), bool)
    in_seat0[game, holder] = True
    assert np.mean(t.game_placement[in_seat0] >= 3.5) > 0.85
    assert np.mean(t.game_placement[~in_seat0] >= 3.5) < 0.15
    for p in range(4):
        assert abs(t.placement[p] - 2.5) < 0.4  # 3.9 or 2.0 if one policy kept seat 0
        assert abs(t.share[p] - 0.25) < 0.1
    # Teams of two: either team holds seat 0 half of the time.
    team = play_table(ENV4, policies[:2], (0, 1, 0, 1), key, games, 300, 64)
    assert abs(team.placement[0] - 2.5) < 0.4 and abs(team.share[0] - 0.5) < 0.1


@pytest.mark.parametrize("lineup", [(0, 1, 1, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 2, 2)])
def test_table_identical_policies_are_at_parity(lineup):
    # Copies of one policy: share seats/N and placement 2.5 (the mean of 1, 2, 3, 4).
    policies = CLONES4[: max(lineup) + 1]
    t = play_table(ENV4, policies, lineup, jax.random.key(37), 512, 300, 128)
    for p, seats in enumerate(t.seats):
        assert abs(t.share[p] - seats / 4) < 2.5 * t.share_ci95[p], (p, t.share[p])
        assert abs(t.placement[p] - 2.5) < 2.5 * t.placement_ci95[p], (p, t.placement[p])
    assert sum(t.share) == pytest.approx(1.0)
    # At most one policy wins a game outright, and in every game some policy has a share.
    assert sum(t.wins) <= 512 <= sum(t.wins) + sum(t.ties)
    assert all(0 < w < 512 for w in t.wins)


def test_table_truncation():
    key = jax.random.key(38)
    for early_stop in (False, True):
        t = play_table(ENV4, CLONES4[:2], (0, 1, 0, 1), key, 64, 2, 32, early_stop)
        # Nobody is out after two turns: all four tie, so each team gets half and 2.5.
        assert t.truncated == 64 and t.mean_turns == 2
        assert t.ties == (64, 64) and t.wins == t.losses == (0, 0)
        assert t.share == (0.5, 0.5) and t.placement == (2.5, 2.5)
        assert t.share_ci95 == (0.0, 0.0)
    # Games cut off by the env's own limit count as truncated too.
    env = BattlesnakeEnv(GameConfig(num_snakes=4, max_turns=3), obs=None)
    t = play_table(env, (random_legal(env),) * 2, (0, 1, 1, 0), key, 16, 1000, 16)
    assert t.truncated == 16 and t.mean_turns == 3


def test_table_jit_cache_ignores_lineup_and_sizes():
    policies = CLONES4[:2]
    play_table(ENV4, policies, (0, 1, 1, 1), jax.random.key(39), 16, 20, 16)
    before = evaluate._compiled_table.cache_info()
    play_table(ENV4, policies, (0, 1, 0, 1), jax.random.key(40), 16, 30, 16)
    play_table(ENV4, policies, (1, 1, 0, 1), jax.random.key(40), 16, 9, 16)
    evaluate.run_table(ENV4, policies, (0, 1, 1, 1), jax.random.key(1), 16, 9, 16, first_game=5)
    after = evaluate._compiled_table.cache_info()
    assert after.misses == before.misses and after.hits == before.hits + 3

    # The same holds with per-game arrays, whatever the number of games: 9, 16 and 100 games
    # in 8 slots are one program (the slots only shrink when there are fewer games than slots).
    fresh = (clone_of(RAND4, 11), clone_of(RAND4, 12))  # not compiled by any other test
    before = evaluate._compiled_table.cache_info()
    runs = [
        evaluate.run_table(ENV4, fresh, (0, 1, 1, 1), jax.random.key(39), g, 20, 8, per_game=True)
        for g in (9, 16, 100)
    ]
    assert [r.slots for r in runs] == [8, 8, 8]
    assert [r.result.game_share.shape for r in runs] == [(9, 2), (16, 2), (100, 2)]
    after = evaluate._compiled_table.cache_info()
    assert (after.misses, after.hits) == (before.misses + 1, before.hits + 2)
    play_table(ENV4, fresh, (0, 1, 0, 1), jax.random.key(41), 40, 20, 8, per_game=True)
    assert evaluate._compiled_table.cache_info().hits == after.hits + 1
    # The program depends on the slots: fewer games than slots shrink them.
    play_table(ENV4, fresh, (0, 1, 0, 1), jax.random.key(41), 5, 20, 8, per_game=True)
    assert evaluate._compiled_table.cache_info().misses == after.misses + 1


def test_table_records_more_games_than_one_buffer_holds(monkeypatch):
    # More games than the per-game buffer holds are played in chunks, with the same result.
    policies, lineup, key = CLONES4[:3], (0, 1, 2, 2), jax.random.key(42)
    whole = evaluate.run_table(ENV4, policies, lineup, key, 21, 60, 8, per_game=True)
    monkeypatch.setattr(evaluate, "_RECORDED_GAMES", 8)
    seen = []
    chunked = evaluate.run_table(
        ENV4, policies, lineup, key, 21, 60, 8, per_game=True,
        progress=lambda games, turns: seen.append((games, turns)),
    )  # fmt: skip
    assert chunked.slots == whole.slots == 8
    assert numbers(chunked.result) == numbers(whole.result)
    np.testing.assert_array_equal(chunked.result.game_share, whole.result.game_share)
    np.testing.assert_array_equal(chunked.result.game_placement, whole.result.game_placement)
    assert chunked.result.game_share.shape == (21, 3)
    # The reports continue across chunks: games and turns only grow, up to the totals.
    assert seen == sorted(seen) and seen[-1] == (21, chunked.iterations)


def test_table_env_stream_does_not_depend_on_the_number_of_policies():
    # Policies that ignore their key and are clones of each other: a line-up of three policies
    # (0, 1, 2, 1) must play the very games of the line-up (0, 1, 0, 1) of two, which needs the
    # env to draw the same food in both.
    def first_legal(key, state, ts):
        return jnp.argmax(ts.action_mask, axis=-1).astype(jnp.int32)

    def twin(key, state, ts):
        return first_legal(key, state, ts)

    def other(key, state, ts):  # the last legal move
        return (NUM_ACTIONS - 1 - jnp.argmax(ts.action_mask[..., ::-1], axis=-1)).astype(jnp.int32)

    def other_twin(key, state, ts):
        return other(key, state, ts)

    key = jax.random.key(43)
    two = play_table(ENV4, (first_legal, other), (0, 1, 0, 1), key, 48, 150, 16, per_game=True)
    three = play_table(
        ENV4, (first_legal, other, other_twin), (0, 1, 0, 2), key, 48, 150, 16, per_game=True
    )
    four = play_table(
        ENV4, (first_legal, other, twin, other_twin), (0, 1, 2, 3), key, 48, 150, 16, per_game=True
    )
    # Policy 0 and its twin are one team, as are policy 1 and its twin.
    np.testing.assert_allclose(three.game_share[:, 0], two.game_share[:, 0], atol=1e-6)
    np.testing.assert_allclose(
        three.game_share[:, 1] + three.game_share[:, 2], two.game_share[:, 1], atol=1e-6
    )
    np.testing.assert_allclose(
        four.game_share[:, 0] + four.game_share[:, 2], two.game_share[:, 0], atol=1e-6
    )
    # The games differ from each other (they are not all drawn, say).
    assert 0 < np.mean(two.game_share[:, 0] == 0.5) < 1


def test_table_progress_reports():
    seen = []
    run = evaluate.run_table(
        ENV4, CLONES4[:2], (0, 1, 1, 1), jax.random.key(41), 48, 100, 8,
        progress=lambda f, i: seen.append((f, i)),
    )  # fmt: skip
    finished = [f for f, _ in seen]
    assert finished == sorted(finished) and finished[-1] == 48 and len(seen) <= 16
    assert seen[-1][1] == run.iterations
    assert 0.5 < run.utilization <= 1.0


def test_table_rejects_bad_arguments():
    key, p = jax.random.key(0), CLONES4[:2]
    with pytest.raises(ValueError, match="lineup needs one policy per seat"):
        play_table(ENV4, p, (0, 1, 0), key, 4)
    with pytest.raises(ValueError, match="must index"):
        play_table(ENV4, p, (0, 1, 2, 0), key, 4)
    with pytest.raises(ValueError, match="must hold a seat"):
        play_table(ENV4, p, (0, 0, 0, 0), key, 4)
    with pytest.raises(ValueError, match=">= 1"):
        play_table(ENV4, p, (0, 1, 0, 1), key, 0)
    with pytest.raises(ValueError, match="game ids"):
        evaluate.run_table(ENV4, p, (0, 1, 0, 1), key, 4, first_game=-1)
    solo = BattlesnakeEnv(GameConfig(num_snakes=1), obs=None)
    with pytest.raises(ValueError, match="not solo"):
        play_table(solo, (random_legal(solo),), (0,), key, 4)


def test_table_wilson_interval():
    low, high = evaluate._wilson(0.5, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert evaluate._wilson(0.0, 100)[0] == 0.0 and evaluate._wilson(1.0, 100)[1] == 1.0
    assert 0.0 < evaluate._wilson(0.0, 100)[1] < 0.04  # not a zero-width interval
