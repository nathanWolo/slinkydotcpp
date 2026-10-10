"""Tests for the self-play Rainbow DQN baseline (``baselines/rainbow.py``)."""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.observations import obs_shape
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    """Import ``baselines/<filename>`` from its path (``baselines/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "baselines" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


rb = _load("baselines_rainbow", "rainbow.py")
GAMMA = 0.9
# A tiny network (and buffer) so tests compile and run fast.
TINY = rb.RainbowConfig(
    conv_channels=(4,), conv_strides=(2,), hidden=16, num_atoms=11, gamma=GAMMA,
    num_envs=4, buffer_capacity=64, learning_starts=8,
)  # fmt: skip
NO_FOOD = {"food_spawn_chance": 0, "minimum_food": 0}  # deterministic steps
net = jax.jit(rb.network, static_argnames=("cfg", "per_sample"))


def make_state(config: GameConfig, snakes: list[dict], turn: int = 5):
    d = {
        "width": config.width,
        "height": config.height,
        "turn": turn,
        "snakes": [{"id": f"s{i}", "health": 90, **s} for i, s in enumerate(snakes)],
        "food": [],
        "hazards": [],
    }
    return state_from_engine(d, config)


def stack(items):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


@functools.cache
def jitted_step(env: BattlesnakeEnv):
    return jax.jit(env.step)


def play(env: BattlesnakeEnv, state, moves) -> list:
    """The one-step transitions (``rb.Step``) of playing ``moves`` (a list of joint moves)."""
    steps = []
    for k, actions in enumerate(moves):
        actions = jnp.asarray(actions, jnp.int32)
        nxt, ts = jitted_step(env)(jax.random.key(k), state, actions)
        steps.append(rb.Step(state, actions, ts.reward, nxt, ts.truncated))
        state = nxt
    return steps


def make_window(slots: list[list]):
    """Stack per-slot step lists (each of length n) into a window with leaves ``[n, E, ...]``."""
    n = len(slots[0])
    return stack([stack([slot[k] for slot in slots]) for k in range(n)])


def reference_nstep(steps: list, gamma: float):
    """Brute force: ``(returns, discount, bootstrap state)`` of the oldest step's n-step target."""
    returns = np.zeros(np.shape(steps[0].rewards), np.float64)
    for k, s in enumerate(steps):
        returns += gamma**k * np.asarray(s.rewards, np.float64)
        if bool(s.next_state.done):
            break  # the slot holds a new game after this step
    last = steps[k]
    rules_ended = bool(last.next_state.done) and not bool(last.truncated)
    alive = np.asarray(last.next_state.alive)
    discount = np.where(alive & (not rules_ended), gamma ** (k + 1), 0.0)
    return returns, discount, last.next_state


def assert_trees_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def check_window(window, slots, gamma):
    tr = rb.nstep_transition(window, gamma)
    for e, slot in enumerate(slots):
        returns, discount, next_state = reference_nstep(slot, gamma)
        np.testing.assert_allclose(tr.returns[e], returns, rtol=1e-6, atol=1e-7, err_msg=str(e))
        np.testing.assert_allclose(tr.discount[e], discount, rtol=1e-6, err_msg=str(e))
        assert_trees_equal(jax.tree.map(lambda x, e=e: x[e], tr.next_state), next_state)
        assert_trees_equal(jax.tree.map(lambda x, e=e: x[e], tr.state), slot[0].state)
        np.testing.assert_array_equal(tr.actions[e], slot[0].actions)
    return tr


# --- (1) n-step transitions --------------------------------------------------------------

DUEL = BattlesnakeEnv(GameConfig(**NO_FOOD), obs=None)
SHORT = BattlesnakeEnv(GameConfig(max_turns=7, **NO_FOOD), obs=None)  # turn 5 -> 7 truncates
UP_SNAKE = {"body": [[5, 5], [5, 4], [5, 3]]}
LOW_SNAKE = {"body": [[2, 2], [2, 1], [2, 0]]}  # moves RIGHT along y = 2


def fresh_game_with_a_death():
    """A new game (turn 0) whose first step kills snake 1 (rewards +1, -1): must be ignored."""
    state = make_state(DUEL.config, [UP_SNAKE, {"body": [[10, 5], [9, 5], [8, 5]]}], turn=0)
    return play(DUEL, state, [[UP, RIGHT]])


@functools.cache
def duel_slots():
    """``(slots, expected returns, expected discounts)`` for a 3-step window; g = GAMMA."""
    g = GAMMA
    st = make_state
    cfg = DUEL.config
    slots, ret, disc = [], [], []
    # Ordinary mid-game: nothing happens, bootstrap gamma**3.
    slots.append(play(DUEL, st(cfg, [UP_SNAKE, LOW_SNAKE]), [[UP, RIGHT]] * 3))
    ret.append([0, 0]), disc.append([g**3, g**3])
    # Snake 0 runs into the left wall at step 1: loses -g, snake 1 wins +g; a fresh
    # game follows in the slot (with its own rewards, which must be excluded).
    s = st(cfg, [{"body": [[1, 5], [2, 5], [3, 5]]}, {"body": [[8, 8], [8, 7], [8, 6]]}])
    slots.append(play(DUEL, s, [[LEFT, LEFT]] * 2) + fresh_game_with_a_death())
    ret.append([-g, g]), disc.append([0, 0])
    # Snake 1 dies on the window's last step: rewards discounted by g**2.
    s = st(cfg, [UP_SNAKE, {"body": [[8, 3], [7, 3], [6, 3]]}])
    slots.append(play(DUEL, s, [[UP, RIGHT]] * 3))
    ret.append([g**2, -(g**2)]), disc.append([0, 0])
    # Equal-length head-on at step 2: a draw, 0 for both, no bootstrap.
    s = st(cfg, [{"body": [[2, 5], [1, 5], [0, 5]]}, {"body": [[8, 5], [9, 5], [10, 5]]}])
    slots.append(play(DUEL, s, [[RIGHT, LEFT]] * 3))
    ret.append([0, 0]), disc.append([0, 0])
    # max_turns truncation at step 1: bootstrap from the truncated state with g**2.
    s = st(cfg, [UP_SNAKE, LOW_SNAKE])
    slots.append(play(SHORT, s, [[UP, RIGHT]] * 2) + fresh_game_with_a_death())
    ret.append([0, 0]), disc.append([g**2, g**2])
    # Game over at step 0 (snake 1 hits the right wall); two fresh-game steps follow.
    s = st(cfg, [UP_SNAKE, {"body": [[10, 8], [9, 8], [8, 8]]}])
    fresh = make_state(cfg, [UP_SNAKE, LOW_SNAKE], turn=0)
    slots.append(play(DUEL, s, [[UP, RIGHT]]) + play(DUEL, fresh, [[UP, RIGHT]] * 2))
    ret.append([1, -1]), disc.append([0, 0])
    return slots, np.array(ret, np.float64), np.array(disc, np.float64)


def test_duel_step_rewards_are_as_assumed():
    slots, _, _ = duel_slots()
    rewards = np.array([[np.asarray(s.rewards) for s in slot] for slot in slots])
    np.testing.assert_array_equal(rewards[1, :2], [[0, 0], [-1, 1]])
    np.testing.assert_array_equal(rewards[1, 2], [1, -1])  # the fresh game's step
    np.testing.assert_array_equal(rewards[2, 2], [1, -1])
    np.testing.assert_array_equal(rewards[3], 0)
    assert bool(slots[3][2].next_state.done) and not slots[3][2].next_state.alive.any()
    assert bool(slots[4][1].truncated) and bool(slots[4][1].next_state.done)
    assert int(slots[4][1].next_state.turn) == 7


def test_nstep_transition_matches_brute_force():
    slots, returns, discount = duel_slots()
    tr = check_window(make_window(slots), slots, GAMMA)
    # Hand-computed values too, not just the brute-force reference.
    np.testing.assert_allclose(tr.returns, returns, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(tr.discount, discount, rtol=1e-6)
    assert tr.discount.dtype == jnp.float32
    # Bootstrap states: the step that ended the game, or the newest step.
    np.testing.assert_array_equal(tr.next_state.turn, [8, 7, 8, 8, 7, 6])


def test_nstep_one_equals_one_step_td():
    slots, _, _ = duel_slots()
    first = [slot[:1] for slot in slots]
    tr = check_window(make_window(first), first, GAMMA)
    for e, slot in enumerate(slots):
        s = slot[0]
        np.testing.assert_array_equal(tr.returns[e], s.rewards)
        over = bool(s.next_state.done) and not bool(s.truncated)
        expected = np.where(np.asarray(s.next_state.alive) & (not over), GAMMA, 0.0)
        np.testing.assert_allclose(tr.discount[e], expected, rtol=1e-6)
        assert_trees_equal(jax.tree.map(lambda x, e=e: x[e], tr.next_state), s.next_state)


def test_nstep_three_snakes_one_dies_while_two_continue():
    env = BattlesnakeEnv(GameConfig(num_snakes=3, **NO_FOOD), obs=None)
    snakes = [UP_SNAKE, LOW_SNAKE, {"body": [[1, 8], [2, 8], [3, 8]]}]
    slot = play(env, make_state(env.config, snakes), [[UP, RIGHT, LEFT]] * 3)
    np.testing.assert_array_equal(slot[1].rewards, [0, 0, -1])  # snake 2 left the board
    assert not bool(slot[1].next_state.done)  # two snakes are still playing
    np.testing.assert_array_equal(slot[2].rewards, [0, 0, 0])
    tr = check_window(make_window([slot]), [slot], GAMMA)
    np.testing.assert_allclose(tr.returns[0], [0, 0, -GAMMA], rtol=1e-6)
    np.testing.assert_allclose(tr.discount[0], [GAMMA**3, GAMMA**3, 0], rtol=1e-6)


def test_window_push_slides_oldest_first():
    w = rb.Step(*(jnp.arange(3 * 2).reshape(3, 2) + 10 * i for i in range(5)))
    s = rb.Step(*(jnp.array([-1, -2]) - 10 * i for i in range(5)))
    out = rb.window_push(w, s)
    for leaf, old, new in zip(out, w, s, strict=True):
        np.testing.assert_array_equal(leaf[:2], old[1:])
        np.testing.assert_array_equal(leaf[2], new)


# --- (2) C51 projection ----------------------------------------------------------------


def reference_projection(returns, discount, probs, atoms):
    """The paper's projection (Bellemare et al. 2017, Alg. 1), with l == u kept."""
    atoms = np.asarray(atoms, np.float64)
    v_min, v_max, num = atoms[0], atoms[-1], len(atoms)
    dz = (v_max - v_min) / (num - 1)
    out = np.zeros(np.shape(probs), np.float64)
    for idx in np.ndindex(np.shape(returns)):
        for j in range(num):
            tz = np.clip(returns[idx] + discount[idx] * atoms[j], v_min, v_max)
            b = (tz - v_min) / dz
            lo, hi = int(np.floor(b)), int(np.ceil(b))
            p = probs[idx][j]
            if lo == hi:
                out[idx][lo] += p
            else:
                out[idx][lo] += p * (hi - b)
                out[idx][hi] += p * (b - lo)
    return out


@pytest.mark.parametrize(("v_min", "v_max", "num"), [(-1.0, 1.0, 51), (-3.0, 5.0, 9)])
def test_projection_matches_floor_ceil_reference(v_min, v_max, num):
    rng = np.random.default_rng(0)
    atoms = np.linspace(v_min, v_max, num)
    span = v_max - v_min
    returns = rng.uniform(v_min - 0.5 * span, v_max + 0.5 * span, size=(6, 5))
    returns[0, :] = [v_min, v_max, 0.0, v_min, v_max]  # support edges
    discount = rng.choice([0.0, 0.5, 0.9, 0.99, 1.0], size=(6, 5))
    discount[0, :] = [0.0, 0.0, 1.0, 1.0, 0.9]
    probs = rng.dirichlet(np.full(num, 0.3), size=(6, 5))
    out = rb.project(
        jnp.asarray(returns, jnp.float32),
        jnp.asarray(discount, jnp.float32),
        jnp.asarray(probs, jnp.float32),
        jnp.linspace(v_min, v_max, num),
    )
    ref = reference_projection(returns, discount, probs, atoms)
    np.testing.assert_allclose(out, ref, atol=2e-5)
    np.testing.assert_allclose(out.sum(-1), 1.0, atol=1e-5)  # mass is conserved
    assert float(out.min()) >= 0.0


def test_projection_special_cases():
    atoms = jnp.linspace(-1.0, 1.0, 5)  # -1, -0.5, 0, 0.5, 1
    p = jnp.array([0.1, 0.2, 0.3, 0.25, 0.15])

    def proj(r, d):
        return np.asarray(rb.project(jnp.float32(r), jnp.float32(d), p, atoms))

    np.testing.assert_allclose(proj(0.0, 1.0), p, atol=1e-6)  # identity
    np.testing.assert_allclose(proj(1.0, 0.0), [0, 0, 0, 0, 1], atol=1e-6)  # point mass
    np.testing.assert_allclose(proj(-1.0, 0.0), [1, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(proj(0.25, 0.0), [0, 0, 0.5, 0.5, 0], atol=1e-6)
    np.testing.assert_allclose(proj(5.0, 0.5), [0, 0, 0, 0, 1], atol=1e-6)  # all clipped
    # Shift up one atom: the top atom collects what falls off the support.
    np.testing.assert_allclose(proj(0.5, 1.0), [0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)
    # Shrink by half: atom j lands on (z_j / 2), halfway between atoms for odd j.
    expected = reference_projection(np.array(0.0), np.array(0.5), np.asarray(p), np.asarray(atoms))
    np.testing.assert_allclose(proj(0.0, 0.5), expected, atol=1e-6)


# --- (3) Targets and the loss --------------------------------------------------------------

SMALL = dataclasses.replace(TINY, num_atoms=5)  # atoms -1, -0.5, 0, 0.5, 1


def test_target_distribution_bootstraps_from_the_legal_argmax_of_the_selector():
    atoms = np.linspace(-1.0, 1.0, 5)
    rng = np.random.default_rng(1)
    # Selector: expected returns ordered 1 > 3 > 0 > 2 for both agents (point masses).
    point = np.full((NUM_ACTIONS, 5), -30.0)
    point[[0, 1, 2, 3], [2, 4, 0, 3]] = 0.0  # Q = 0, 1, -1, 0.5
    select = jnp.asarray(np.broadcast_to(point, (2, NUM_ACTIONS, 5)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(2, NUM_ACTIONS, 5)), jnp.float32)
    mask = jnp.array([[True, False, True, True], [True, True, True, True]])  # 1 illegal for 0
    tr = rb.Transition(None, None, jnp.array([0.2, -0.3]), None, jnp.array([0.81, 0.0]))
    out = rb.target_distribution(tr, select, target, mask, SMALL)
    probs = np.asarray(jax.nn.softmax(target, axis=-1))
    chosen = probs[[0, 1], [RIGHT, DOWN]]  # agent 0 may not take DOWN, agent 1 may
    ref = reference_projection(np.array([0.2, -0.3]), np.array([0.81, 0.0]), chosen, atoms)
    np.testing.assert_allclose(out, ref, atol=1e-5)
    np.testing.assert_allclose(out[1], [0, 0.6, 0.4, 0, 0], atol=1e-5)  # discount 0: return only


@functools.cache
def one_step_batch(size: int, seed: int = 0) -> rb.Transition:
    """One-step duel transitions of random legal play, mid-game states included."""
    env = BattlesnakeEnv(SMALL.game, obs=None)
    states = duel_rollout()
    pick = np.random.default_rng(seed).choice(states.turn.shape[0], size, replace=False)
    states = jax.tree.map(lambda x: x[pick], states)
    keys = jax.random.split(jax.random.key(seed), size)

    @jax.jit
    def act_and_step(keys, states):
        actions = jax.vmap(lambda k, s: random_legal_policy(k, s, env))(keys, states)
        return (actions, *jax.vmap(env.step)(keys, states, actions))

    actions, nxt, ts = act_and_step(keys, states)
    ended = nxt.done & ~ts.truncated
    discount = jnp.where(nxt.alive & ~ended[:, None], GAMMA, 0.0)
    return rb.Transition(states, actions, ts.reward, nxt, discount.astype(jnp.float32))


@pytest.mark.parametrize("double", [True, False])
def test_loss_matches_manual_computation(double):
    """Loss, priorities, mean Q and gradients against a hand-written C51 loss (no noise)."""
    cfg = dataclasses.replace(SMALL, noisy=False, double=double)
    env = BattlesnakeEnv(cfg.game)
    tr = one_step_batch(12)
    tr = tr._replace(returns=tr.returns + jnp.linspace(-0.3, 0.3, 24).reshape(12, 2))
    params = rb.init_network(jax.random.key(0), cfg)
    target_params = rb.init_network(jax.random.key(1), cfg)
    weights = jnp.linspace(0.2, 1.0, 12)
    loss, out = jax.jit(rb.make_loss_fn(env, cfg))(
        params, target_params, tr, weights, jax.random.key(2)
    )

    atoms = np.linspace(-1.0, 1.0, 5)
    next_obs = jax.vmap(env.observe)(tr.next_state)
    mask = np.asarray(jax.vmap(env.action_mask)(tr.next_state))
    online_next = np.asarray(net(params, next_obs, cfg), np.float64)
    target_next = np.asarray(net(target_params, next_obs, cfg), np.float64)

    def softmax(x):
        z = np.exp(x - x.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    q_online = softmax(online_next) @ atoms
    q_target = softmax(target_next) @ atoms
    q_select = q_online if double else q_target
    a_star = np.where(mask, q_select, -np.inf).argmax(-1)  # [B, N]
    a_other = np.where(mask, q_target if double else q_online, -np.inf).argmax(-1)
    disagree = (a_star != a_other) & (np.asarray(tr.discount) > 0)
    assert disagree.any()  # the two networks pick different bootstrap moves somewhere
    p_next = np.take_along_axis(softmax(target_next), a_star[..., None, None], -2)[..., 0, :]
    m = reference_projection(np.asarray(tr.returns), np.asarray(tr.discount), p_next, atoms)

    obs = jax.vmap(env.observe)(tr.state)
    valid = np.asarray(tr.state.alive & ~tr.state.done[:, None], np.float64)

    def manual_ce(p):
        logits = rb.network(p, obs, cfg)
        taken = jnp.take_along_axis(logits, tr.actions[..., None, None], axis=-2)[..., 0, :]
        ce = -jnp.sum(m * jax.nn.log_softmax(taken), axis=-1)  # [B, N]
        return jnp.sum(weights[:, None] * valid * ce) / valid.sum()

    taken = np.take_along_axis(
        np.asarray(net(params, obs, cfg), np.float64),
        np.asarray(tr.actions)[..., None, None],
        -2,
    )[..., 0, :]
    log_p = taken - np.log(np.exp(taken).sum(-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.sum(np.where(m > 0, m * (np.log(m) - log_p), 0.0), axis=-1)
    expected_loss = np.sum(np.asarray(weights)[:, None] * valid * kl) / valid.sum()
    np.testing.assert_allclose(float(loss), expected_loss, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out.priorities, (valid * kl).sum(-1) / valid.sum(-1), rtol=1e-4)
    mean_q = np.sum(valid * (np.exp(log_p) @ atoms)) / valid.sum()
    np.testing.assert_allclose(float(out.mean_q), mean_q, rtol=1e-4, atol=1e-6)
    # The KL's gradient is the cross-entropy's.
    for a, b in zip(
        jax.tree.leaves(out.grads),
        jax.tree.leaves(jax.jit(jax.grad(manual_ce))(params)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6)


@functools.cache
def three_snake_batch():
    config = GameConfig(num_snakes=3)
    env = BattlesnakeEnv(config)  # egocentric obs, same shape as the duel's
    snakes = [
        {"body": [[1, 1], [1, 2], [1, 3]]},
        {"body": [[8, 8], [8, 7], [8, 6]]},
        {"body": [[5, 0], [5, 0], [5, 0]], "eliminated_cause": "wall-collision"},
    ]
    state = make_state(config, snakes)
    actions = jnp.array([RIGHT, UP, UP], jnp.int32)
    sim = BattlesnakeEnv(config, obs=None)
    nxt, ts = jitted_step(sim)(jax.random.key(0), state, actions)
    discount = jnp.where(nxt.alive, GAMMA, 0.0)
    one = rb.Transition(state, actions, ts.reward, nxt, discount)
    tr = stack([one] * 4)
    # Different returns per item for the living agents, so the items' losses differ.
    tr = tr._replace(returns=tr.returns.at[:, :2].add(jnp.array([[-0.5], [0.0], [0.3], [0.6]])))
    return env, tr


@functools.cache
def three_snake_loss_fn():
    return jax.jit(rb.make_loss_fn(three_snake_batch()[0], TINY))


def test_agent_dead_at_t_is_excluded_from_the_loss():
    _, tr = three_snake_batch()
    np.testing.assert_array_equal(rb.valid_agents(tr.state)[0], [True, True, False])
    params = rb.init_network(jax.random.key(0), TINY)
    loss_fn = three_snake_loss_fn()
    ones, key = jnp.ones(4), jax.random.key(1)
    loss, out = loss_fn(params, params, tr, ones, key)
    assert np.isfinite(float(loss)) and np.isfinite(out.priorities).all()
    # The dead agent's return does not matter; a living agent's does.
    dead = tr._replace(returns=tr.returns.at[:, 2].set(0.9))
    live = tr._replace(returns=tr.returns.at[:, 1].set(0.9))
    loss_dead, out_dead = loss_fn(params, params, dead, ones, key)
    loss_live, out_live = loss_fn(params, params, live, ones, key)
    assert float(loss_dead) == pytest.approx(float(loss))
    np.testing.assert_allclose(out_dead.priorities, out.priorities, rtol=1e-6)
    assert abs(float(loss_live) - float(loss)) > 1e-3
    # Uniform weights: the loss is the mean of the per-item priorities (2 valid agents each).
    np.testing.assert_allclose(float(loss), float(out.priorities.mean()), rtol=1e-5)


def test_importance_weights_scale_the_loss():
    _, tr = three_snake_batch()
    params = rb.init_network(jax.random.key(0), TINY)
    loss_fn = three_snake_loss_fn()
    key = jax.random.key(1)
    loss1, out1 = loss_fn(params, params, tr, jnp.ones(4), key)
    loss2, out2 = loss_fn(params, params, tr, jnp.full(4, 0.5), key)
    assert float(loss2) == pytest.approx(0.5 * float(loss1), rel=1e-5)
    np.testing.assert_allclose(out2.priorities, out1.priorities, rtol=1e-6)  # unweighted
    for a, b in zip(jax.tree.leaves(out2.grads), jax.tree.leaves(out1.grads), strict=True):
        np.testing.assert_allclose(a, 0.5 * b, rtol=1e-4, atol=1e-8)
    w = jnp.array([1.0, 0.0, 0.25, 0.0])
    loss_w, _ = loss_fn(params, params, tr, w, key)
    np.testing.assert_allclose(float(loss_w), float(jnp.mean(w * out1.priorities)), rtol=1e-5)


# --- (4) Prioritized replay buffer ---------------------------------------------------------


def test_buffer_add_wraps_and_assigns_max_priority():
    example = {"x": jnp.zeros((), jnp.int32), "y": jnp.zeros((2,), jnp.float32)}
    buf = rb.buffer_init(example, capacity=6)
    assert float(buf.max_priority) == 1.0 and not buf.priority.any()
    add = jax.jit(rb.buffer_add)

    def items(start):
        x = jnp.arange(start, start + 2, dtype=jnp.int32)
        return {"x": x, "y": jnp.stack([x, -x], axis=-1).astype(jnp.float32)}

    buf = add(add(buf, items(1)), items(3))
    assert (int(buf.ptr), int(buf.size)) == (4, 4)
    np.testing.assert_array_equal(buf.priority, [1, 1, 1, 1, 0, 0])
    prio, max_prio = rb.update_priorities(
        buf.priority, buf.max_priority, jnp.array([0, 2]), jnp.array([0.5, 3.0])
    )
    np.testing.assert_allclose(prio, [0.5, 1, 3, 1, 0, 0])
    assert float(max_prio) == 3.0
    buf = buf._replace(priority=prio, max_priority=max_prio)
    buf = add(buf, items(5))  # full; new items get the max priority so far
    assert (int(buf.ptr), int(buf.size)) == (0, 6)
    np.testing.assert_allclose(buf.priority, [0.5, 1, 3, 1, 3, 3])
    buf = add(buf, items(7))  # overwrites the two oldest
    assert (int(buf.ptr), int(buf.size)) == (2, 6)
    np.testing.assert_array_equal(buf.data["x"], [7, 8, 3, 4, 5, 6])
    np.testing.assert_array_equal(buf.data["y"][:, 1], -buf.data["x"])
    np.testing.assert_allclose(buf.priority, [3, 3, 3, 1, 3, 3])
    # Lower priorities never lower the maximum.
    _, m = rb.update_priorities(buf.priority, buf.max_priority, jnp.array([1]), jnp.array([0.1]))
    assert float(m) == 3.0
    with pytest.raises(ValueError, match="multiple"):
        rb.buffer_add(buf, {"x": jnp.zeros(4, jnp.int32), "y": jnp.zeros((4, 2))})


def sample_many(priority, size, alpha, beta, batch=16, draws=3000):
    keys = jax.random.split(jax.random.key(0), draws)
    fn = jax.jit(jax.vmap(lambda k: rb.buffer_sample(priority, size, k, batch, alpha, beta)))
    idx, w = fn(keys)
    return np.asarray(idx), np.asarray(w)


@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_buffer_sample_frequencies_follow_priorities(alpha):
    # Slots 6.. are stale (e.g. an uncommitted write) and must never be drawn.
    priority = jnp.array([0.1, 1.0, 2.0, 4.0, 0.5, 3.0, 50.0, 50.0], jnp.float32)
    size = jnp.int32(6)
    idx, _ = sample_many(priority, size, alpha, 0.4)
    assert idx.dtype == np.int32 and idx.min() >= 0 and idx.max() < 6
    p = np.asarray(priority[:6], np.float64) ** alpha
    p /= p.sum()
    counts = np.bincount(idx.ravel(), minlength=6)
    n = idx.size
    chi2 = np.sum((counts - n * p) ** 2 / (n * p))
    assert chi2 < 30.0, (counts / n, p)  # 5 dof; stratification makes it much smaller
    if alpha == 0.0:
        np.testing.assert_allclose(counts / n, 1 / 6, atol=0.01)  # uniform


def test_buffer_sample_importance_weights():
    priority = jnp.array([0.1, 1.0, 2.0, 4.0, 0.5, 9.0, 0.0, 0.0], jnp.float32)
    size, alpha, beta = 6, 0.7, 0.5
    idx, w = sample_many(priority, jnp.int32(size), alpha, beta, batch=8, draws=50)
    p = np.asarray(priority[:size], np.float64) ** alpha
    prob = p / p.sum()
    raw = (size * prob[idx]) ** -beta
    np.testing.assert_allclose(w, raw / raw.max(axis=-1, keepdims=True), rtol=1e-5)
    assert np.allclose(w.max(axis=-1), 1.0)
    # beta = 0: no correction.
    _, w0 = sample_many(priority, jnp.int32(size), alpha, 0.0, batch=8, draws=10)
    np.testing.assert_allclose(w0, 1.0)


def test_buffer_sample_full_buffer_and_single_item():
    priority = jnp.array([1.0, 2.0, 3.0, 4.0])
    idx, _ = sample_many(priority, jnp.int32(4), 1.0, 1.0, batch=4, draws=500)
    np.testing.assert_allclose(np.bincount(idx.ravel()) / idx.size, [0.1, 0.2, 0.3, 0.4], atol=0.02)
    idx, w = sample_many(priority, jnp.int32(1), 1.0, 1.0, batch=4, draws=20)
    assert (idx == 0).all() and np.allclose(w, 1.0)


# --- (5) Network ---------------------------------------------------------------------------


def f_noise(key, shape):
    """Factorized NoisyNet noise f(e) = sign(e) sqrt(|e|), e ~ N(0, 1)."""
    e = np.asarray(jax.random.normal(key, shape), np.float64)
    return np.sign(e) * np.sqrt(np.abs(e))


def noisy_layer(key, fan_in=5, fan_out=3):
    k = jax.random.split(key, 4)
    return {
        "w": jax.random.normal(k[0], (fan_in, fan_out)),
        "b": jax.random.normal(k[1], (fan_out,)),
        "w_sigma": jax.random.uniform(k[2], (fan_in, fan_out), minval=0.1, maxval=1.0),
        "b_sigma": jax.random.uniform(k[3], (fan_out,), minval=0.1, maxval=1.0),
    }


def test_noisy_dense_matches_explicit_noisy_weights():
    p = noisy_layer(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (4, 5))
    w, b, ws, bs = (np.asarray(p[k], np.float64) for k in ("w", "b", "w_sigma", "b_sigma"))
    xn = np.asarray(x, np.float64)
    mean = xn @ w + b
    np.testing.assert_allclose(rb.dense(p, x, None, False), mean, rtol=1e-5, atol=1e-5)
    plain = {"w": p["w"], "b": p["b"]}
    np.testing.assert_allclose(rb.dense(plain, x, jax.random.key(2), True), mean, rtol=1e-5)

    key = jax.random.key(3)
    k_in, k_out = jax.random.split(key)
    # Shared noise: one noisy weight matrix for the batch.
    e_in, e_out = f_noise(k_in, (5,)), f_noise(k_out, (3,))
    expected = xn @ (w + ws * np.outer(e_in, e_out)) + b + bs * e_out
    np.testing.assert_allclose(rb.dense(p, x, key, False), expected, rtol=1e-4, atol=1e-5)
    # Per-sample noise: a noisy weight matrix per row.
    e_in, e_out = f_noise(k_in, (4, 5)), f_noise(k_out, (4, 3))
    expected = np.stack(
        [xn[r] @ (w + ws * np.outer(e_in[r], e_out[r])) + b + bs * e_out[r] for r in range(4)]
    )
    np.testing.assert_allclose(rb.dense(p, x, key, True), expected, rtol=1e-4, atol=1e-5)


def test_per_sample_noise_differs_per_row_and_shared_noise_does_not():
    p = noisy_layer(jax.random.key(0))
    x = jnp.tile(jax.random.normal(jax.random.key(1), (1, 5)), (6, 1))  # identical rows
    shared = np.asarray(rb.dense(p, x, jax.random.key(2), False))
    np.testing.assert_array_equal(shared, np.broadcast_to(shared[0], shared.shape))
    per = np.asarray(rb.dense(p, x, jax.random.key(2), True))
    assert len({tuple(np.round(r, 5)) for r in per}) == 6

    obs = jnp.tile(jax.random.uniform(jax.random.key(3), (1, *obs_dims(TINY))), (5, 1, 1, 1))
    params = rb.init_network(jax.random.key(4), TINY)
    shared = np.asarray(net(params, obs, TINY, jax.random.key(5)))
    np.testing.assert_allclose(shared, np.broadcast_to(shared[0], shared.shape), atol=1e-6)
    per = np.asarray(net(params, obs, TINY, jax.random.key(5), per_sample=True))
    assert all(not np.allclose(per[i], per[j]) for i in range(5) for j in range(i))


def obs_dims(cfg):
    return obs_shape(cfg.game)


@pytest.mark.parametrize("dueling", [True, False])
@pytest.mark.parametrize("noisy", [True, False])
def test_network_shapes_noise_and_q_values(dueling, noisy):
    cfg = dataclasses.replace(TINY, dueling=dueling, noisy=noisy, num_atoms=7)
    params = rb.init_network(jax.random.key(0), cfg)
    assert ("w_sigma" in params["hidden"]) == noisy
    assert set(params) == {"conv0", "hidden"} | ({"value", "advantage"} if dueling else {"logits"})
    obs = jax.random.uniform(jax.random.key(1), (2, 3, *obs_dims(cfg)))
    mean = net(params, obs, cfg)
    assert mean.shape == (2, 3, NUM_ACTIONS, 7)
    noisy_out = net(params, obs, cfg, jax.random.key(2))
    if noisy:
        assert not np.allclose(noisy_out, mean)
        again = net(params, obs, cfg, jax.random.key(2))
        np.testing.assert_array_equal(noisy_out, again)  # deterministic given the key
    else:
        np.testing.assert_array_equal(noisy_out, mean)  # no noisy layers: key ignored

    logits = np.asarray(mean, np.float64)
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    q = probs @ np.linspace(cfg.v_min, cfg.v_max, 7)
    np.testing.assert_allclose(rb.q_values(mean, cfg), q, rtol=1e-5, atol=1e-6)
    assert rb.q_values(mean, cfg).shape == (2, 3, NUM_ACTIONS)


def test_dueling_head_centres_the_advantage():
    cfg = dataclasses.replace(TINY, num_atoms=7, noisy=False)
    params = rb.init_network(jax.random.key(0), cfg)
    adv_bias = jax.random.normal(jax.random.key(1), (NUM_ACTIONS * 7,))
    params["advantage"] = {"w": jnp.zeros_like(params["advantage"]["w"]), "b": adv_bias}
    obs = jax.random.uniform(jax.random.key(2), (5, *obs_dims(cfg)))
    logits = np.asarray(net(params, obs, cfg))
    a = np.asarray(adv_bias).reshape(NUM_ACTIONS, 7)
    # A constant advantage stream: logits = value + a - mean_a(a) for every input.
    np.testing.assert_allclose(
        logits - logits.mean(axis=1, keepdims=True),
        np.broadcast_to(a - a.mean(0), logits.shape),
        atol=1e-5,
    )


def test_noisy_initialization():
    cfg = dataclasses.replace(TINY, noisy_sigma0=0.4)
    params = rb.init_network(jax.random.key(0), cfg)
    for name in ("hidden", "value", "advantage"):
        p = params[name]
        fan_in = p["w"].shape[0]
        bound = 1 / np.sqrt(fan_in)
        assert float(jnp.abs(p["w"]).max()) <= bound and float(jnp.abs(p["b"]).max()) <= bound
        np.testing.assert_allclose(p["w_sigma"], 0.4 / np.sqrt(fan_in), rtol=1e-6)
        np.testing.assert_allclose(p["b_sigma"], 0.4 / np.sqrt(fan_in), rtol=1e-6)


@pytest.mark.parametrize("stride", [1, 2])
def test_conv3x3_gradient_matches_xla_convolution(stride):
    x = jax.random.normal(jax.random.key(0), (2, 11, 11, 3))
    w = jax.random.normal(jax.random.key(1), (3, 3, 3, 5))

    def reference(x, w):
        dn = ("NHWC", "HWIO", "NHWC")
        return jax.lax.conv_general_dilated(x, w, (stride, stride), "SAME", dimension_numbers=dn)

    def loss(conv):
        return lambda x, w: jnp.sum(jnp.sin(conv(x, w)))

    ours = jax.grad(loss(lambda x, w: rb.conv3x3(x, w, stride)), argnums=(0, 1))(x, w)
    ref = jax.grad(loss(reference), argnums=(0, 1))(x, w)
    for a, b in zip(ours, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


# --- (6) Action selection and schedules -----------------------------------------------------


def rollout_states(env: BattlesnakeEnv, batch: int = 16, turns: int = 40):
    """States visited by ``batch`` autoreset games of random legal play: leaves [T*B, ...]."""
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), batch))

    def step(states, key):
        k_act, k_step = jax.random.split(key)
        act = jax.vmap(lambda k, s: random_legal_policy(k, s, env))
        actions = act(jax.random.split(k_act, batch), states)
        new, _ = jax.vmap(env.step_autoreset)(jax.random.split(k_step, batch), states, actions)
        return new, states

    _, visited = jax.lax.scan(step, states, jax.random.split(jax.random.key(1), turns))
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), visited)


@functools.cache
def duel_rollout():
    return rollout_states(BattlesnakeEnv(SMALL.game, obs=None), batch=8, turns=25)


def test_action_selection_never_picks_a_masked_move():
    env = BattlesnakeEnv(GameConfig())
    states = duel_rollout()
    mask = jax.vmap(env.action_mask)(states)  # [S, 2, 4]
    assert (~mask).any(axis=-1).mean() > 0.5  # most rows rule something out
    # Adversarial Q-values: every illegal move beats every legal one.
    noise = jax.random.normal(jax.random.key(2), mask.shape)
    q = jnp.where(mask, noise, noise + 100.0)
    for eps in (0.0, 0.3, 1.0):
        actions = rb.epsilon_greedy(jax.random.key(3), q, mask, jnp.float32(eps))
        assert actions.dtype == jnp.int32
        assert bool(jnp.take_along_axis(mask, actions[..., None], axis=-1).all()), eps
    greedy = rb.epsilon_greedy(jax.random.key(4), q, mask, jnp.float32(0.0))
    np.testing.assert_array_equal(greedy, jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1))
    # The noisy path: per-snake noisy Q-values, epsilon 0.
    params = rb.init_network(jax.random.key(5), TINY)
    obs = jax.jit(jax.vmap(env.observe))(states)
    logits = net(params, obs, TINY, jax.random.key(6), per_sample=True)
    eps = rb.epsilon(TINY, jnp.int32(0))
    actions = rb.epsilon_greedy(jax.random.key(7), rb.q_values(logits, TINY), mask, eps)
    assert bool(jnp.take_along_axis(mask, actions[..., None], axis=-1).all())


def test_epsilon_and_beta_schedules():
    cfg = rb.RainbowConfig(total_env_steps=1000, per_beta_start=0.4, per_beta_end=1.0)
    for t in (0, 500, 5000):
        assert float(rb.epsilon(cfg, jnp.int32(t))) == 0.0  # noisy nets explore instead
    beta = [float(rb.per_beta(cfg, jnp.int32(t))) for t in (0, 250, 500, 1000, 3000)]
    np.testing.assert_allclose(beta, [0.4, 0.55, 0.7, 1.0, 1.0], rtol=1e-6)
    cfg = dataclasses.replace(cfg, noisy=False, eps_decay_fraction=0.5, eps_end=0.1)
    eps = [float(rb.epsilon(cfg, jnp.int32(t))) for t in (0, 250, 500, 1000)]
    np.testing.assert_allclose(eps, [1.0, 0.55, 0.1, 0.1], rtol=1e-6)


# --- (7) The collection loop ------------------------------------------------------------


def test_train_loop_stores_the_nstep_transitions_of_its_steps():
    """Step the jitted loop one iteration at a time; check the buffer against the steps taken.

    Noisy-net acting; the epsilon path's legality is checked in section (6).
    """
    n, e, iters = 3, 8, 30
    cfg = dataclasses.replace(
        TINY, n_step=n, num_envs=e, max_turns=15, buffer_capacity=e * iters,
        learning_starts=e * iters, total_env_steps=e * iters,
    )  # fmt: skip
    env, sim = rb.make_envs(cfg)
    runner = rb.init_runner(cfg, sim, jax.random.key(0))
    chunk = jax.jit(rb.make_train_chunk(cfg, env, sim, 1))
    steps = []
    for i in range(iters):
        runner, m = chunk(runner)
        steps.append(jax.tree.map(lambda x: x[-1], runner.window))  # the newest step
        stored = max(0, i + 2 - n) * e
        assert (int(runner.buffer.size), int(runner.buffer.ptr)) == (stored, stored % (e * iters))
        assert int(m["updates"]) == 0
    steps = jax.device_get(steps)

    # Every acting snake took a legal move.
    for s in steps:
        mask = jax.vmap(sim.action_mask)(s.state)
        legal = np.take_along_axis(np.asarray(mask), s.actions[..., None], -1)[..., 0]
        assert legal[np.asarray(rb.valid_agents(s.state))].all()

    data = jax.device_get(runner.buffer.data)
    seen = {"death": 0, "truncated": 0, "bootstrap": 0}
    for j in range(iters - n + 1):
        for slot in range(e):
            window = [jax.tree.map(lambda x, s=slot: x[s], steps[j + k]) for k in range(n)]
            returns, discount, next_state = reference_nstep(window, cfg.gamma)
            item = jax.tree.map(lambda x, i=j * e + slot: x[i], data)
            np.testing.assert_allclose(item.returns, returns, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(item.discount, discount, rtol=1e-6)
            assert_trees_equal(item.next_state, next_state)
            assert_trees_equal(item.state, window[0].state)
            np.testing.assert_array_equal(item.actions, window[0].actions)
            assert not item.state.done
            seen["death"] += bool((returns != 0).any())
            seen["truncated"] += any(bool(s.truncated) for s in window)
            seen["bootstrap"] += bool((discount > 0).any())
    assert all(seen.values()), seen  # the cases above were exercised


# --- (8) End to end, configs and checkpoints -------------------------------------------------


def tiny_argv(run_dir: str) -> list[str]:
    return [
        "--num-envs", "4", "--total-env-steps", "800", "--log-every", "400",
        "--learning-starts", "200", "--batch-size", "8", "--n-step", "3",
        "--buffer-capacity", "402",  # rounded down to 400: wraps around
        "--conv-channels", "4,8", "--conv-strides", "2,2", "--hidden", "16", "--num-atoms", "11",
        "--target-update-period", "10", "--tau", "0.5", "--max-turns", "100",
        "--eval-every", "800", "--eval-games", "8", "--run-dir", run_dir,
    ]  # fmt: skip


def test_tiny_training_run_checkpoints_and_eval_only(tmp_path, capsys):
    run_dir = str(tmp_path / "run")
    cfg, _ = rb.parse_args(tiny_argv(run_dir))
    assert cfg.conv_channels == (4, 8) and cfg.noisy and cfg.double and cfg.dueling
    assert cfg.lr == rb.RainbowConfig.lr and cfg.adam_eps == 1.5e-4
    out_dir, params = rb.train(cfg)
    assert out_dir == run_dir

    records = [json.loads(line) for line in Path(run_dir, "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    assert [r["env_steps"] for r in train] == [400, 800]
    assert train[-1]["buffer_size"] == 400
    assert sum(r["agent_transitions"] for r in train) == 2 * 800  # both snakes, every step
    # Storing starts at iteration n = 3, learning once 200 transitions are stored.
    first_learning_iter = 200 // 4 + 3 - 1  # 1-based
    assert train[-1]["total_updates"] == 800 // 4 - first_learning_iter + 1
    assert train[0]["eps"] == 0.0 and train[-1]["beta"] == pytest.approx(1.0)
    assert train[-1]["noise_sigma"] > 0 and train[-1]["max_priority"] >= 1.0
    losses = [r["loss"] for r in train if r["loss"] is not None]
    assert losses and all(np.isfinite(losses)) and min(losses) >= 0.0
    assert all(r["games"] > 0 for r in train)
    assert len(evals) == 1 and evals[0]["num_games"] == 8 and evals[0]["env_steps"] == 800
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))

    loaded = rb.load_params(run_dir)
    assert jax.tree.structure(loaded) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert rb.load_config(run_dir) == cfg
    assert json.loads(Path(run_dir, "config.json").read_text())["algorithm"] == "rainbow"

    capsys.readouterr()
    rb.main(["--eval-only", run_dir, "--eval-games", "8"])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["num_games"] == 8
    assert result["wins"] + result["draws"] + result["losses"] == 8


def test_resume_continues_exactly(tmp_path, monkeypatch):
    def argv(run_dir):
        return [
            "--num-envs", "4", "--total-env-steps", "600", "--log-every", "200",
            "--checkpoint-every", "200", "--learning-starts", "60", "--batch-size", "8",
            "--buffer-capacity", "300", "--conv-channels", "4", "--conv-strides", "2",
            "--hidden", "16", "--num-atoms", "11", "--target-update-period", "20",
            "--eval-every", "600", "--eval-games", "0", "--run-dir", run_dir,
        ]  # fmt: skip

    full_dir, cut_dir = str(tmp_path / "full"), str(tmp_path / "cut")
    _, full = rb.train(rb.parse_args(argv(full_dir))[0])

    # Interrupt the second run during its third chunk, after two full checkpoints.
    calls = []
    chunk_record = rb.chunk_record

    def interrupted(*args):
        calls.append(None)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return chunk_record(*args)

    monkeypatch.setattr(rb, "chunk_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rb.train(rb.parse_args(argv(cut_dir))[0])
    monkeypatch.setattr(rb, "chunk_record", chunk_record)
    rb.main(["--resume", cut_dir])

    resumed = rb.load_params(cut_dir)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True):
        np.testing.assert_array_equal(a, b)

    def records(run_dir):
        lines = Path(run_dir, "metrics.jsonl").read_text().splitlines()
        drop = ("seconds", "elapsed", "updates_per_s", "env_steps_per_s")
        return [{k: v for k, v in json.loads(x).items() if k not in drop} for x in lines]

    assert records(cut_dir) == records(full_dir)  # no duplicated or missing records


def test_load_config_rejects_a_dqn_run(tmp_path):
    dqn = _load("baselines_dqn", "dqn.py")
    dqn.save_config(str(tmp_path), dqn.DQNConfig())
    with pytest.raises(ValueError, match="not a Rainbow run"):
        rb.load_config(str(tmp_path))
    cfg = dataclasses.replace(TINY, conv_channels=(4, 8), conv_strides=(2, 1), run_dir="x")
    rb.save_config(str(tmp_path), cfg)
    assert rb.load_config(str(tmp_path)) == cfg


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"buffer_capacity": 10_000, "num_envs": 32}, "learning_starts"),  # rounds to 9,984
        ({"conv_channels": (4, 8), "conv_strides": (2,)}, "same length"),
        ({"n_step": 0}, "n_step"),
        ({"num_atoms": 1}, "num_atoms"),
        ({"v_min": 1.0, "v_max": 1.0}, "v_min"),
        ({"v_max": 0.5}, "must cover the returns"),  # would clip every win
        ({"v_min": -0.9}, "must cover the returns"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": 1.5}, "gamma"),
        ({"tau": 0.0}, "tau"),
        ({"per_alpha": -0.1}, "per_alpha"),
        ({"per_beta_start": 1.5}, "per_beta"),
        ({"per_eps": 0.0}, "per_eps"),
        ({"num_envs": 64, "buffer_capacity": 32, "learning_starts": 0}, "buffer_capacity"),
    ],
)
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rb.RainbowConfig(**kwargs)


def test_train_refuses_a_run_that_would_never_learn(tmp_path):
    # 800 env steps store 800 - (3 - 1) * 4 = 792 transitions: learning_starts 800 is unreachable.
    argv = tiny_argv(str(tmp_path / "run"))
    argv[argv.index("--learning-starts") + 1] = "800"
    argv[argv.index("--buffer-capacity") + 1] = "800"
    with pytest.raises(ValueError, match="never learn"):
        rb.train(rb.parse_args(argv)[0])
    argv[argv.index("--learning-starts") + 1] = "792"
    rb.parse_args(argv)  # the boundary case is valid (and stores exactly enough)


def test_importance_weights_use_the_sampled_probability_mass():
    """With ~1e5 slots, float32 rounding of the cumulative sum changes how often tiny
    priorities are drawn; their weights must follow the draw, not the exact priority."""
    size = 99_968
    rng = np.random.default_rng(0)
    priority = jnp.asarray(rng.uniform(0, 1, size) ** 2 + 1e-6, jnp.float32)
    alpha, beta = 0.5, 1.0
    idx, w = jax.jit(rb.buffer_sample, static_argnums=(3,))(
        priority, jnp.int32(size), jax.random.key(0), 4096, alpha, beta
    )
    # The same float32 cumulative sum (jnp's summation order differs from numpy's).
    cdf = np.asarray(jax.jit(lambda p: jnp.cumsum(p**alpha))(priority))
    idx = np.asarray(idx)
    mass = cdf[idx] - np.where(idx > 0, cdf[np.maximum(idx - 1, 0)], 0.0)
    raw = (size * mass / cdf[-1]) ** -beta
    np.testing.assert_allclose(w, raw / raw.max(), rtol=1e-4)


def test_config_accepts_a_buffer_that_fits_learning_starts():
    rb.RainbowConfig(buffer_capacity=10_016, num_envs=32)
    cfg, _ = rb.parse_args(["--noisy", "false", "--per-alpha", "0", "--v-min", "-2"])
    assert not cfg.noisy and cfg.per_alpha == 0.0 and cfg.v_min == -2.0
