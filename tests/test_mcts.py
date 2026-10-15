"""Tests for slinky.mcts: search invariants, shapes, tactics and a strength sanity check."""

from __future__ import annotations

import dataclasses
import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import DUEL, board, both_seats, env_for, rollout_states

from slinky import mcts as M
from slinky.env import BattlesnakeEnv
from slinky.evaluate import play_match, random_legal
from slinky.observations import make_obs_fn
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig, State

DEFAULT = M.MCTSConfig(num_simulations=128)


@functools.cache
def batched_search(game: GameConfig, config: M.MCTSConfig, evaluator: M.Evaluator | None = None):
    """jit(vmap(search)) over keys, for one unbatched state."""
    env = env_for(game)
    return jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config, evaluator), in_axes=(0, None)))


def run(
    state: State,
    config: M.MCTSConfig = DEFAULT,
    game: GameConfig = DUEL,
    n_keys: int = 4,
    evaluator: M.Evaluator | None = None,
):
    """SearchOutput (numpy) with a leading axis of ``n_keys`` independent searches."""
    keys = jax.random.split(jax.random.key(0), n_keys)
    return jax.tree.map(np.asarray, batched_search(game, config, evaluator)(keys, state))


@functools.cache
def constant_evaluator(value: tuple[float, ...] = (0.3, -0.2), logits: tuple = ((0.0,) * 4,) * 2):
    """An evaluator that ignores the state (cached, so ``batched_search`` compiles it once)."""

    def evaluator(state: State) -> tuple[jax.Array, jax.Array]:
        return jnp.asarray(value, jnp.float32), jnp.asarray(logits, jnp.float32)

    return evaluator


def tiny_params(key: jax.Array) -> dict[str, jax.Array]:
    k_conv, k_value, k_policy = jax.random.split(key, 3)
    return {
        "conv": jax.random.normal(k_conv, (3, 3, 13, 8)) * 0.3,
        "value": jax.random.normal(k_value, (8,)),
        "policy": jax.random.normal(k_policy, (8, NUM_ACTIONS)),
    }


def tiny_network(params: dict[str, jax.Array]) -> M.Evaluator:
    """A small network on egocentric observations, closing over (possibly traced) params."""
    obs_fn = make_obs_fn(DUEL)

    def evaluator(state: State) -> tuple[jax.Array, jax.Array]:
        x = jax.lax.conv_general_dilated(
            obs_fn(state),
            params["conv"],
            (2, 2),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        x = jnp.mean(jax.nn.relu(x), axis=(1, 2))  # [N, 8]
        return jnp.tanh(x @ params["value"]), x @ params["policy"]

    return evaluator


@functools.cache
def mid_game_state() -> State:
    """One live duel state after 30 turns of random legal play."""
    state = jax.tree.map(lambda x: x[0], rollout_states(DUEL, batch=1))
    assert not bool(state.done)
    return state


# --- Invariants -----------------------------------------------------------------------


def test_config_validation_and_digits():
    for bad in [
        dict(num_simulations=0),
        dict(max_depth=0),
        dict(rollout_steps=-1),
        dict(selection="exp3"),
        dict(leaf="dqn"),
        dict(rollout_policy="greedy"),
        dict(final="argmax"),
        dict(exploration=-0.1),
        dict(exploration=float("nan")),
        dict(tie_noise=float("inf")),
        dict(rm_gamma=1.5),
        dict(rm_gamma=-0.1),
        dict(draw_value=-2.0),
        dict(prior="heuristic"),
        dict(selection="puct", c_puct=-1.0),
        dict(selection="puct", c_puct=float("nan")),
        dict(selection="puct", fpu_reduction=float("nan")),
        dict(selection="puct", root_noise_frac=1.5),
        dict(selection="puct", root_dirichlet_alpha=0.0),
        dict(prior="network"),  # priors and root noise need PUCT
        dict(selection="rm", root_noise_frac=0.25),
        dict(selection="puct", ucb1_tuned=True),  # DUCT's variance bound
        dict(leaf="network", rollout_steps=10),
    ]:
        with pytest.raises(ValueError):
            M.MCTSConfig(**bad)
    hash(M.MCTSConfig())  # frozen and hashable (used as a cache key)
    d = M._digits(2)
    assert d.shape == (16, 2) and d[3 + 4 * 2].tolist() == [3, 2]


def test_only_ucb1_tuned_duct_keeps_squared_values():
    env, state = env_for(DUEL), mid_game_state()
    for kwargs, kept in [
        (dict(ucb1_tuned=True), True),
        (dict(selection="rm", ucb1_tuned=True), False),  # accepted, but RM never reads them
        (dict(), False),
    ]:
        tree = M._init_tree(state, env, M.MCTSConfig(num_simulations=4, **kwargs), 5)
        assert (tree.value_sq is not None) == kept, kwargs


@pytest.mark.parametrize("selection", ["duct", "rm", "puct"])
def test_root_statistics_are_consistent(selection):
    config = M.MCTSConfig(num_simulations=64, selection=selection)
    states = rollout_states(DUEL, batch=16)
    env = env_for(DUEL)
    keys = jax.random.split(jax.random.key(3), 16)
    out = jax.tree.map(
        np.asarray, jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config)))(keys, states)
    )
    alive_root = ~np.asarray(states.done)
    assert alive_root.sum() >= 8
    # Every simulation passes through the root and takes one move per player.
    np.testing.assert_array_equal(out.visits[alive_root].sum(-1), 64)
    assert np.all(np.abs(out.q) <= 1) and np.all(np.abs(out.value) <= 1)
    np.testing.assert_allclose(out.policy.sum(-1), 1.0, rtol=1e-5)
    assert np.all(out.policy >= 0)
    assert np.all((out.nodes_used >= 1) & (out.nodes_used <= 65))
    # Moves never leave env.action_mask, and visits stay on the selectable moves.
    mask = np.asarray(jax.vmap(env.action_mask)(states))
    assert np.take_along_axis(mask, out.action[..., None], -1).all()
    assert np.all(out.visits[~mask] == 0)
    # Root values are the visit-weighted mean of the move values.
    total = out.visits.sum(-1)
    mean = (out.visits * out.q).sum(-1) / np.maximum(total, 1)
    np.testing.assert_allclose(out.value[alive_root], mean[alive_root], atol=1e-5)
    # final="max": a move with the largest policy weight, ties broken by mean value.
    policy, q = out.policy[alive_root], out.q[alive_root]
    action = out.action[alive_root][..., None]
    top = policy >= policy.max(-1, keepdims=True) - 1e-6
    assert np.take_along_axis(top, action, -1).all()
    np.testing.assert_array_equal(
        np.take_along_axis(q, action, -1)[..., 0], np.where(top, q, -np.inf).max(-1)
    )
    if selection != "rm":  # the policy is the root visit distribution
        np.testing.assert_allclose(policy, out.visits[alive_root] / 64, rtol=1e-6)
    # Without network priors or root noise, the root priors are uniform over the moves.
    legal = np.asarray(jax.vmap(lambda s: M._legal(s, env.action_mask(s), DUEL))(states))
    np.testing.assert_allclose(out.root_prior, legal / legal.sum(-1, keepdims=True), rtol=1e-6)


def test_deterministic_given_key():
    state = mid_game_state()
    a, b = run(state), run(state)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    # The keys differ across the batch, so the searches do too.
    assert not np.array_equal(a.visits[0], a.visits[1])


# Root visits, moves and values of two searches of ``mid_game_state()`` (keys split from
# ``key(0)``), recorded with the search as it was before PUCT, priors and network leaves
# were added (commit 45ec415). The configs that existed then must keep their exact results.
# (A JAX or XLA upgrade that changes float rounding could change them too: re-record only
# after checking that the search itself did not change.)
RECORDED = {
    "default": (
        M.MCTSConfig(num_simulations=32),
        [[[0, 0, 12, 20], [0, 10, 14, 8]], [[0, 0, 13, 19], [0, 10, 14, 8]]],
        [[3, 2], [3, 2]],
        [[-0.259293, 0.259293], [-0.279073, 0.247823]],
    ),
    "rm": (
        M.MCTSConfig(num_simulations=32, selection="rm"),
        [[[0, 0, 7, 25], [0, 2, 27, 3]], [[0, 0, 5, 27], [0, 6, 23, 3]]],
        [[3, 2], [3, 2]],
        [[-0.296204, 0.296204], [-0.280412, 0.249162]],
    ),
    "tuned_sample": (
        M.MCTSConfig(num_simulations=32, ucb1_tuned=True, final="sample"),
        [[[0, 0, 13, 19], [0, 10, 12, 10]], [[0, 0, 13, 19], [0, 10, 13, 9]]],
        [[3, 3], [3, 2]],
        [[-0.246915, 0.215665], [-0.271789, 0.240539]],
    ),
}


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_existing_configs_reproduce_recorded_searches(name):
    config, visits, action, value = RECORDED[name]
    out = run(mid_game_state(), config, n_keys=2)
    np.testing.assert_array_equal(out.visits, visits)
    np.testing.assert_array_equal(out.action, action)
    np.testing.assert_allclose(out.value, value, atol=2e-6)
    # PUCT's settings do nothing outside PUCT.
    inert = dataclasses.replace(config, c_puct=3.0, fpu_reduction=-0.5, root_dirichlet_alpha=0.1)
    for x, y in zip(run(mid_game_state(), inert, n_keys=2), out, strict=True):
        np.testing.assert_array_equal(x, y)


def test_finished_game_returns_valid_moves():
    a, b = [(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]
    over = board([a, b], causes=[("", 0), ("head-collision", 10)])
    assert bool(over.done)
    out = run(over, M.MCTSConfig(num_simulations=16))
    assert np.all((out.action >= 0) & (out.action < NUM_ACTIONS))
    assert np.all(out.visits == 0) and np.all(out.nodes_used == 1)
    np.testing.assert_array_equal(out.value, np.tile([1.0, -1.0], (4, 1)))
    # A drawn game, through the policy wrapper.
    draw = board([a, b], causes=[("head-collision", 10), ("head-collision", 10)])
    acts = M.mcts_policy(jax.random.key(0), draw, env_for(DUEL), M.MCTSConfig(num_simulations=8))
    assert acts.shape == (2,) and np.all(np.asarray(acts) >= 0)


@pytest.mark.parametrize("draw_value", [None, 0.0])
def test_draw_value_scores_mutual_eliminations(draw_value):
    # None checks the default (-0.5, "contempt").
    if draw_value is None:
        config, expected = M.MCTSConfig(num_simulations=16), -0.5
    else:
        config, expected = M.MCTSConfig(num_simulations=16, draw_value=draw_value), draw_value
    # The hero's only move is UP and the opponent's only move is DOWN, onto the same
    # cell at equal length: every simulation ends in a mutual elimination.
    hero = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    other = [(0, 2), (1, 2), (1, 3), (0, 3), (0, 4)]
    for state, seat in both_seats([hero, other]):
        assert np.asarray(env_for(DUEL).action_mask(state)).sum(-1).tolist() == [1, 1]
        out = run(state, config)
        np.testing.assert_array_equal(out.value, expected)
        np.testing.assert_array_equal(out.q[:, seat, UP], expected)
    # A finished, drawn root is worth the same.
    a, b = [(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]
    draw = board([a, b], causes=[("head-collision", 10), ("head-collision", 10)])
    np.testing.assert_array_equal(run(draw, config).value, expected)


@pytest.mark.parametrize("spawn_food", [True, False])
def test_truncation_inside_the_tree_is_a_draw(spawn_food):
    # One turn before max_turns every child ends the game: surviving is worth exactly 0.
    game = GameConfig(max_turns=11)
    hero = [(0, 5), (1, 5), (2, 5)]
    other = [(8, 8), (8, 7), (8, 6)]
    state = board([hero, other], game, turn=10)
    out = run(state, M.MCTSConfig(num_simulations=64, spawn_food=spawn_food), game)
    # The hero's LEFT (wall) and RIGHT (neck) are masked and never visited.
    assert np.all(out.visits[:, 0, LEFT] == 0) and np.all(out.visits[:, 0, RIGHT] == 0)
    visited = out.visits > 0
    np.testing.assert_array_equal(out.q[visited], 0.0)
    np.testing.assert_array_equal(out.value, 0.0)
    np.testing.assert_array_equal(out.nodes_used, 1 + 2 * 3)  # every joint move, all terminal


@pytest.mark.parametrize(("selection", "max_depth"), [("duct", 1), ("rm", 2)])
def test_max_depth_caps_the_descent(selection, max_depth):
    # A selection that reaches max_depth backs up the stored value there and expands nothing.
    state = mid_game_state()
    config = M.MCTSConfig(num_simulations=64, selection=selection, max_depth=max_depth)
    out = run(state, config)
    np.testing.assert_array_equal(out.depth, max_depth)
    np.testing.assert_array_equal(out.visits.sum(-1), 64)
    assert np.all(np.abs(out.q) <= 1)
    if max_depth == 1:  # only the root's joint actions are ever expanded
        legal = M._legal(state, env_for(DUEL).action_mask(state), DUEL)
        assert np.all(out.nodes_used <= 1 + np.prod(np.asarray(legal).sum(-1)))


def test_unvisited_moves_are_tried_in_uniformly_random_order():
    # Regression: a priority of 1e6 + U(0, 1) rounds to a few float32 values, so ties
    # among unvisited moves went to the lowest index (UP first 28% of the time).
    config = M.MCTSConfig()
    state = board([[(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]])
    tree = M._init_tree(state, env_for(DUEL), config, 2)
    tree = tree._replace(legal=jnp.ones_like(tree.legal))
    u = jax.random.uniform(jax.random.key(0), (100_000, 2, NUM_ACTIONS))

    @jax.jit
    def first_moves(tree):
        moves = jax.vmap(lambda u: M._duct_moves(tree, 0, u, config))(u)
        return jnp.stack([jnp.bincount(moves[:, p], length=NUM_ACTIONS) for p in range(2)])

    np.testing.assert_allclose(first_moves(tree) / len(u), 0.25, atol=0.01)
    # With UP and DOWN visited (and looking good), LEFT and RIGHT still come first, evenly.
    tree = tree._replace(
        visits=tree.visits.at[0, :, :2].set(3), value_sum=tree.value_sum.at[0, :, :2].set(3.0)
    )
    np.testing.assert_allclose(first_moves(tree) / len(u), [[0, 0, 0.5, 0.5]] * 2, atol=0.01)


def test_each_depth_of_a_descent_draws_its_own_noise():
    # Regression: every depth reused one draw shifted by a constant, so the move sampled
    # at a node depended on the one sampled at its parent (RM's samples below the root
    # were then confined to a window of its distribution).
    config = M.MCTSConfig(selection="rm")
    state = board([[(5, 5), (5, 4), (5, 3)], [(1, 1), (1, 2), (1, 3)]])
    tree = M._init_tree(state, env_for(DUEL), config, 3)
    # Every joint action of the root leads to node 1, whose joint actions are unexpanded.
    # All moves are legal and nothing is visited, so RM picks a uniformly random joint
    # action at each of the two levels.
    tree = tree._replace(legal=jnp.ones_like(tree.legal), children=tree.children.at[0].set(1))
    digits = jnp.asarray(M._digits(2))
    bits = jax.random.bits(jax.random.key(0), (4096, 2, NUM_ACTIONS), jnp.uint32)
    path = np.asarray(jax.vmap(lambda b: M._descend(tree, b, config, digits, 2).path_joint)(bits))
    pairs = np.zeros((16, 16), int)
    np.add.at(pairs, (path[:, 0], path[:, 1]), 1)
    assert pairs.min() > 0  # every (parent, child) pair occurs; 16 are expected per cell


@pytest.mark.parametrize("num_snakes", [1, 3, 4])
def test_shapes_with_other_numbers_of_snakes(num_snakes):
    game = GameConfig(num_snakes=num_snakes)
    states = rollout_states(game, batch=8, turns=20)
    env = env_for(game)
    config = M.MCTSConfig(num_simulations=32)
    keys = jax.random.split(jax.random.key(1), 8)
    out = jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config)))(keys, states)
    assert out.action.shape == (8, num_snakes) and out.action.dtype == jnp.int32
    assert out.visits.shape == (8, num_snakes, NUM_ACTIONS)
    assert out.q.shape == out.policy.shape == (8, num_snakes, NUM_ACTIONS)
    assert out.value.shape == (8, num_snakes)
    out = jax.tree.map(np.asarray, out)
    live = ~np.asarray(states.done)
    np.testing.assert_array_equal(out.visits[live].sum(-1), 32)
    # Dead snakes (in games still running) put every visit on their single move.
    dead = live[:, None] & ~np.asarray(states.alive)
    assert np.all(out.visits[dead][:, 1:] == 0)
    mask = np.asarray(jax.vmap(env.action_mask)(states))
    assert np.take_along_axis(mask, out.action[..., None], -1).all()


def test_too_many_snakes():
    game = GameConfig(num_snakes=5)
    state = env_for(game).init_state(jax.random.key(0))
    config = M.MCTSConfig(num_simulations=4)
    with pytest.raises(ValueError):
        M.search(jax.random.key(0), state, env_for(game), config)
    with pytest.raises(ValueError):  # when the policy is built, not when it is traced
        M.mcts(env_for(game), config)


def test_variants_run():
    """RM, UCB1-Tuned, sampled final moves, rollouts and sampled food spawns all run."""
    state = mid_game_state()
    mask = np.asarray(env_for(DUEL).action_mask(state))
    for config in [
        M.MCTSConfig(num_simulations=32, ucb1_tuned=True, final="sample"),
        M.MCTSConfig(num_simulations=32, selection="rm", final="sample"),
        M.MCTSConfig(num_simulations=16, rollout_steps=10, leaf="none", spawn_food=True),
        M.MCTSConfig(num_simulations=8, rollout_steps=2, rollout_policy="heuristic"),
    ]:
        out = run(state, config)
        np.testing.assert_array_equal(out.visits.sum(-1), config.num_simulations)
        assert np.all(np.abs(out.q) <= 1)
        # Sampled or not, every move is one the mask allows.
        assert mask[np.arange(2), out.action].all()


def tree_copies(fn, *args, num_nodes: int) -> list[str]:
    """``copy`` instructions of tree-sized arrays (a ``num_nodes`` axis) in ``fn``'s loops.

    Those are the HLO computations other than the entry, which runs once per call.
    """
    hlo = jax.jit(fn).lower(*args).compile().as_text()
    copies, entry = [], False
    for line in hlo.splitlines():
        if line.endswith("{") and not line.startswith(" "):  # a computation's header
            entry = line.startswith("ENTRY")
        elif not entry and re.search(rf"= \w+\[[\d,]*\b{num_nodes}\b[\d,]*\]\S* copy\(", line):
            copies.append(line.strip())
    return copies


@pytest.mark.xfail(
    tuple(int(x) for x in jax.__version__.split(".")[:2]) < (0, 11),
    reason="XLA in jax < 0.11 (0.8.2 checked) copies the tree; the search is correct but slower",
    strict=False,
)
@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize("batch", [None, 1, 2])
def test_tree_is_updated_in_place(batch, network):
    # XLA compiled one unbatched game (and a vmap of one) with a copy of every
    # Tree.state array on every simulation, about 1 MB per copy at 1024 simulations,
    # so the search was quadratic in num_simulations. 41 nodes: a size no other axis has.
    # ``network``: PUCT with network priors, a network leaf and root noise, with the
    # network's parameters traced (an argument of the jitted function).
    config = M.MCTSConfig(num_simulations=40)
    if network:
        config = dataclasses.replace(
            config, selection="puct", prior="network", leaf="network", root_noise_frac=0.25
        )
    env, key, state = env_for(DUEL), jax.random.key(0), mid_game_state()
    params = tiny_params(jax.random.key(1))

    def search(k, s, p):
        return M.search(k, s, env, config, tiny_network(p) if network else None)

    if batch is not None:
        search = jax.vmap(search, in_axes=(0, 0, None))
        key = jax.random.split(key, batch)
        state = jax.tree.map(lambda x: jnp.stack([x] * batch), state)
    assert tree_copies(search, key, state, params, num_nodes=41) == []


def test_spawn_free_transition_matches_env_step():
    # On a map without food the deterministic model must equal env.step exactly: the
    # turn count, max_turns truncation, freezing finished games and the action mask.
    # Three snakes, so some die while the game goes on.
    game = GameConfig(num_snakes=3, ruleset="wrapped_constrictor", map="empty", max_turns=45)
    env = env_for(game)
    model, sampled = M.MCTSConfig(spawn_food=False), M.MCTSConfig(spawn_food=True)

    def play(key):
        def step(carry, k):
            state, mask = carry
            k_act, k_step = jax.random.split(k)
            acts = jax.random.categorical(k_act, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
            acts = acts.astype(jnp.int32)
            a = M._transition(k_step, state, acts, env, model)
            b = M._transition(k_step, state, acts, env, sampled)
            return b, (a, b)

        state, ts = env.reset(key)
        return jax.lax.scan(step, (state, ts.action_mask), jax.random.split(key, 60))[1]

    a, b = jax.jit(jax.vmap(play))(jax.random.split(jax.random.key(2), 16))
    assert bool(jnp.all(b[0].done[:, -1]))  # every game ended, some by truncation
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


# --- PUCT, priors, network leaves and root noise -----------------------------------------

# Far apart (8 moves between the heads), so no game can end within 3 moves of the root.
APART = ([(1, 5), (1, 4), (1, 3)], [(9, 5), (9, 4), (9, 3)])
# Both snakes' only move is onto the same cell at equal length: a mutual elimination.
COLLIDE = (
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 2), (1, 2), (1, 3), (0, 3), (0, 4)],
)


def test_node_priors_are_distributions_over_the_legal_moves():
    legal = jnp.array([[True, False, True, True], [True, False, False, False]])
    logits = jnp.array([[1.0, 50.0, 0.0, -1.0], [0.0, 0.0, 9.0, 0.0]])
    puct = M.MCTSConfig(selection="puct")
    uniform = np.asarray(M._node_prior(logits, legal, puct))
    np.testing.assert_allclose(uniform, [[1 / 3, 0, 1 / 3, 1 / 3], [1, 0, 0, 0]], rtol=1e-6)
    network = dataclasses.replace(puct, prior="network")
    prior = np.asarray(M._node_prior(logits, legal, network))
    e = np.exp([1.0, 0.0, -1.0])
    np.testing.assert_allclose(prior, [[*(e / e.sum())[:1], 0, *(e / e.sum())[1:]], [1, 0, 0, 0]])
    # Infinite logits on every legal move: uniform over them (or over the +inf ones), not NaN.
    for row, expected in [
        ([-np.inf, 0.0, -np.inf, -np.inf], [1 / 3, 0, 1 / 3, 1 / 3]),
        ([np.inf, 0.0, np.inf, 1.0], [0.5, 0, 0.5, 0]),
    ]:
        logits = jnp.array([row, [-np.inf] * 4])
        prior = np.asarray(M._node_prior(logits, legal, network))
        np.testing.assert_allclose(prior, [expected, [1, 0, 0, 0]], rtol=1e-6)
    # ... also through the search, at every node (a forced row included).
    evaluator = constant_evaluator((0.0, 0.0), ((-np.inf,) * 4,) * 2)
    config = M.MCTSConfig(num_simulations=16, selection="puct", prior="network")
    for state, _ in both_seats(list(COLLIDE)) + [(mid_game_state(), None)]:
        out = run(state, config, evaluator=evaluator)
        assert np.isfinite(out.root_prior).all() and np.isfinite(out.q).all()
        np.testing.assert_allclose(out.root_prior.sum(-1), 1.0, rtol=1e-6)


def test_puct_with_uniform_priors_is_deterministic_and_legal():
    config = M.MCTSConfig(num_simulations=48, selection="puct")
    state = mid_game_state()
    a, b = run(state, config), run(state, config)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.visits[0], a.visits[1])  # different keys, different searches
    legal = np.asarray(M._legal(state, env_for(DUEL).action_mask(state), DUEL))
    np.testing.assert_array_equal(a.visits.sum(-1), 48)
    assert np.all(a.visits[:, ~legal] == 0)
    assert legal[np.arange(2), a.action].all()


@pytest.mark.parametrize("target", [LEFT, RIGHT])
def test_a_concentrated_prior_gets_the_most_visits(target):
    # Both snakes can go UP, LEFT or RIGHT (DOWN is the neck). A prior of about 0.95 on
    # one move makes it the most visited at a small budget, whichever it is.
    logits = np.zeros((2, NUM_ACTIONS))
    logits[:, target] = 4.0
    logits[:, DOWN] = 9.0  # masked: the prior of an illegal move is 0 whatever its logit
    evaluator = constant_evaluator((0.0, 0.0), tuple(map(tuple, logits)))
    config = M.MCTSConfig(num_simulations=16, selection="puct", prior="network")
    for state, _ in both_seats(list(APART)):
        out = run(state, config, evaluator=evaluator)
        assert np.all(out.visits.argmax(-1) == target)
        assert np.all(out.root_prior[..., DOWN] == 0)
        np.testing.assert_allclose(out.root_prior[..., target], np.exp(4) / (np.exp(4) + 2))


@pytest.mark.parametrize("selection", ["duct", "puct"])
def test_network_leaf_uses_the_evaluator_value(selection):
    config = M.MCTSConfig(num_simulations=32, selection=selection, leaf="network", max_depth=3)
    evaluator = constant_evaluator((0.3, -0.2))
    out = run(board(list(APART)), config, evaluator=evaluator)
    expected = np.broadcast_to(np.array([0.3, -0.2])[:, None], out.q.shape)
    np.testing.assert_allclose(out.value, expected[..., 0], atol=1e-6)
    visited = out.visits > 0
    np.testing.assert_allclose(out.q[visited], expected[visited], atol=1e-6)
    # Exact outcomes still override the evaluator: every line here is a mutual elimination.
    for state, _ in both_seats(list(COLLIDE)):
        np.testing.assert_array_equal(run(state, config, evaluator=evaluator).value, -0.5)


def test_evaluator_is_batched_and_params_are_traced():
    env, config = env_for(DUEL), M.MCTSConfig(num_simulations=16, selection="puct")
    config = dataclasses.replace(config, prior="network", leaf="network")
    shapes = []

    def evaluator(params):
        def fn(state):
            shapes.append(state.length.shape)  # what the evaluator sees, at trace time
            return tiny_network(params)(state)

        return fn

    @jax.jit
    def batched(keys, states, params):
        return jax.vmap(lambda k, s: M.search(k, s, env, config, evaluator(params)))(keys, states)

    states = rollout_states(DUEL, batch=4)
    keys = jax.random.split(jax.random.key(5), 4)
    a = batched(keys, states, tiny_params(jax.random.key(1)))
    traced = len(shapes)
    # One unbatched state per call (the root and the simulation loop), batched by vmap.
    assert traced >= 2 and set(shapes) == {(2,)}
    b = batched(keys, states, tiny_params(jax.random.key(2)))
    assert len(shapes) == traced  # new parameters, no new trace
    assert not np.allclose(a.root_prior, b.root_prior)
    # Each game of the batch got its own evaluation: the same root priors as alone.
    single = jax.jit(lambda k, s, p: M.search(k, s, env, config, evaluator(p)))
    for g in range(4):
        out = single(
            keys[g], jax.tree.map(lambda x, g=g: x[g], states), tiny_params(jax.random.key(1))
        )
        np.testing.assert_allclose(out.root_prior, a.root_prior[g], atol=1e-5)


def test_root_noise_keeps_the_priors_on_the_legal_moves():
    state = mid_game_state()
    legal = np.asarray(M._legal(state, env_for(DUEL).action_mask(state), DUEL))
    uniform = legal / legal.sum(-1, keepdims=True)
    config = M.MCTSConfig(num_simulations=16, selection="puct", root_noise_frac=0.5)
    out = run(state, config)
    prior = out.root_prior
    np.testing.assert_allclose(prior.sum(-1), 1.0, rtol=1e-6)
    assert np.all(prior[:, ~legal] == 0) and np.all(prior >= 0.5 * uniform - 1e-6)
    multi = legal.sum(-1) > 1
    assert multi.any()
    assert not np.allclose(prior[:, multi], uniform[multi], atol=1e-3)  # it changed
    assert not np.allclose(prior[0], prior[1], atol=1e-3)  # and differs across keys
    # Off, the priors are uniform; a player with one legal move keeps its 1 either way.
    off = run(state, dataclasses.replace(config, root_noise_frac=0.0))
    np.testing.assert_allclose(off.root_prior, np.broadcast_to(uniform, prior.shape), rtol=1e-6)
    for forced, _ in both_seats(list(COLLIDE)):
        mask = np.asarray(env_for(DUEL).action_mask(forced))
        np.testing.assert_array_equal(
            run(forced, config).root_prior, np.broadcast_to(mask, prior.shape)
        )


@pytest.mark.parametrize("alpha", [0.003, 0.01, 0.5, 30.0])
def test_root_noise_is_a_distribution_for_any_alpha(alpha):
    # Below alpha 0.1 every Gamma draw of a row can underflow float32; normalizing the
    # draws themselves left 5% of the rows at alpha 0.01 summing to less than 0.999.
    legal = jnp.array([[True, True, True, False], [True, False, False, False]])
    prior = legal / jnp.sum(legal, -1, keepdims=True)
    config = M.MCTSConfig(selection="puct", root_noise_frac=1.0, root_dirichlet_alpha=alpha)
    keys = jax.random.split(jax.random.key(0), 4096)

    def noisy(k):
        return M._noisy_root_prior(M.root_noise_draws(k, 2, config), prior, legal, config)

    noise = np.asarray(jax.vmap(noisy)(keys))
    assert np.isfinite(noise).all() and np.all(noise[:, ~np.asarray(legal)] == 0)
    np.testing.assert_allclose(noise.sum(-1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(noise[:, 1], np.broadcast_to([1, 0, 0, 0], (4096, 4)))
    np.testing.assert_allclose(noise[:, 0, :3].mean(0), 1 / 3, atol=0.03)  # E[Dirichlet]


def test_network_configs_need_an_evaluator():
    env, state, key = env_for(DUEL), mid_game_state(), jax.random.key(0)
    for config in [
        M.MCTSConfig(num_simulations=4, leaf="network"),
        M.MCTSConfig(num_simulations=4, selection="puct", prior="network"),
    ]:
        assert config.uses_evaluator
        with pytest.raises(ValueError, match="needs an evaluator"):
            M.search(key, state, env, config)
        with pytest.raises(ValueError, match="needs an evaluator"):  # no cached Policy
            M.mcts(env, config)
    with pytest.raises(ValueError, match="does not use it"):
        M.search(key, state, env, M.MCTSConfig(num_simulations=4), constant_evaluator())
    with pytest.raises(ValueError, match="must return"):

        def flat(state):
            return jnp.zeros(2), jnp.zeros(NUM_ACTIONS)

        M.search(key, state, env, M.MCTSConfig(num_simulations=4, leaf="network"), flat)


# --- Tactics (hero in both seats, several keys) ----------------------------------------


@pytest.mark.parametrize("selection", ["duct", "rm", "puct"])
def test_avoids_head_to_head_with_a_longer_snake(selection):
    config = M.MCTSConfig(num_simulations=128, selection=selection)
    hero = [(5, 5), (5, 4), (5, 3), (5, 2)]
    longer = [(5, 7), (5, 8), (5, 9), (6, 9), (7, 9), (8, 9)]
    for state, seat in both_seats([hero, longer]):
        assert UP not in set(run(state, config).action[:, seat])


@pytest.mark.parametrize("selection", ["duct", "rm", "puct"])
def test_takes_a_forced_kill_of_a_shorter_snake(selection):
    # The short snake's only legal move is (1, 2), which the hero's head can also reach.
    config = M.MCTSConfig(num_simulations=64, selection=selection)
    hero = [(1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    short = [(0, 2), (0, 1), (0, 0)]
    for state, seat in both_seats([hero, short]):
        out = run(state, config)
        assert set(out.action[:, seat]) == {DOWN}
        assert np.all(out.q[:, seat, DOWN] == 1.0)  # an exact win


def test_avoids_a_pocket_the_mask_allows():
    # The hero (length 10) can go left into a 4-cell pocket walled by its own body
    # (with food in it, to tempt it): it is dead 5 moves later. Right is open. The
    # heuristic leaf already sees this; the next test needs the tree.
    hero = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
    other = [(9, 9), (9, 8), (9, 7)]
    for state, seat in both_seats([hero, other], food=[(1, 0)], health=[20, 90]):
        assert bool(env_for(DUEL).action_mask(state)[seat, LEFT])
        assert set(run(state).action[:, seat]) == {RIGHT}


def test_tree_alone_finds_a_trap_three_moves_deep():
    # No heuristic (leaf="none": living snakes are worth 0), so only the search
    # can see it: left leads into a 2-cell dead end, and every line dies on move 3.
    hero = [(2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
    other = [(9, 9), (9, 8), (9, 7)]
    config = M.MCTSConfig(num_simulations=256, leaf="none")
    for state, seat in both_seats([hero, other]):
        assert bool(env_for(DUEL).action_mask(state)[seat, LEFT])
        out = run(state, config)
        assert set(out.action[:, seat]) == {RIGHT}
        # Deaths were found below LEFT (the leaves are worth 0 without them).
        assert np.all(out.q[:, seat, LEFT] < 0)
        assert np.all(out.visits[:, seat, LEFT] < out.visits[:, seat, RIGHT])


# --- Policy and strength ---------------------------------------------------------------


def test_policy_is_cached_and_beats_random_legal():
    env = BattlesnakeEnv(DUEL, obs=None)
    config = M.MCTSConfig(num_simulations=64)
    policy = M.mcts(env, config)
    assert M.mcts(env, config) is policy
    result = play_match(env, policy, random_legal(env), jax.random.key(7), num_games=64)
    assert result.score >= 0.9, result
