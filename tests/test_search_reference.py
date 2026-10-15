"""Bitwise equivalence of the search with the frozen 1952d9f code (``reference_mcts.py``).

The GPU form of the descent (masked steps, ``_masked_descent_on_cpu``) and the CPU form must
both give exactly the outputs, trees and descents of the original ``while_loop``, and AZ
self-play must give the same positions and metrics as the original trainer.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference_az_modes as RA
import reference_mcts as R
from helpers import env_for, rollout_states
from test_mcts import mid_game_state, tree_copies

from slinky import mcts as M
from slinky import modes, unified
from slinky.types import NUM_ACTIONS, GameConfig

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]
MODES = {name: modes.mode_config(name, None, 1000) for name in modes.MODE_NAMES}
SEEDS = (0, 1, 2)
BATCH = 6
TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)


def fake_eval(state):
    """Elementwise per-snake network stand-in (exactly row-independent)."""
    n = state.alive.shape[0]
    h = state.head.astype(jnp.float32)
    value = jnp.tanh(0.1 * h[:, 0] - 0.07 * h[:, 1] + 0.01 * state.length.astype(jnp.float32))
    logits = jnp.stack(
        [h[:, 0] * 0.1, h[:, 1] * -0.1, state.health.astype(jnp.float32) * 0.01, jnp.zeros(n)], -1
    )
    return value, logits


# name -> (MCTSConfig kwargs, evaluator)
CONFIGS = {
    "duct": (dict(num_simulations=64), None),
    "rm": (dict(num_simulations=40, selection="rm"), None),
    "tuned": (dict(num_simulations=40, ucb1_tuned=True), None),
    "az": (
        dict(num_simulations=64, selection="puct", prior="network", leaf="network",
             root_noise_frac=0.25, value="placement"),
        fake_eval,
    ),
    "deep": (dict(num_simulations=160, selection="puct", c_puct=0.05, tie_noise=0.0), None),
    "shallow": (dict(num_simulations=64, max_depth=5), None),
}  # fmt: skip
FULL_MODES = ("standard", "duel")
CASES = [
    (c, m) for m in MODES for c in CONFIGS if m in FULL_MODES or c in ("az", "deep")
]  # fmt: skip


def _cfg(module, name):
    kwargs, evaluator = CONFIGS[name]
    return module.MCTSConfig(**kwargs), evaluator


def _states(mode, seed):
    return rollout_states(MODES[mode], batch=BATCH, turns=25, seed=seed)


def _keys(seed):
    return jax.random.split(jax.random.key(7 + seed), BATCH)


def _arr(x):
    return np.asarray(jax.random.key_data(x) if jnp.issubdtype(x.dtype, jax.dtypes.prng_key) else x)


def _equal(a, b):
    """Same fields (None included) and bitwise equal arrays; the types may differ (M against R)."""
    la = jax.tree.leaves(a, is_leaf=lambda x: x is None)
    lb = jax.tree.leaves(b, is_leaf=lambda x: x is None)
    return len(la) == len(lb) and all(
        (x is None and y is None)
        or (x is not None and y is not None and np.array_equal(_arr(x), _arr(y)))
        for x, y in zip(la, lb, strict=True)
    )


@pytest.fixture(params=[False, True], ids=["cpu_loop", "masked"])
def masked(request, monkeypatch):
    monkeypatch.setattr(M, "_masked_descent_on_cpu", request.param)
    return request.param


@functools.cache
def _reference_outputs(name, mode, seed):
    config, evaluator = _cfg(R, name)
    env = env_for(MODES[mode])
    states, keys = _states(mode, seed), _keys(seed)

    def one(k, s):
        return R.search(k, s, env, config, evaluator)

    batched = jax.jit(jax.vmap(one))(keys, states)
    single = [
        jax.jit(one)(keys[i], jax.tree.map(lambda x, i=i: x[i], states)) for i in range(2)
    ]  # fmt: skip
    return batched, single


@pytest.mark.parametrize("name,mode", CASES)
def test_search_matches_reference(masked, name, mode):
    config, evaluator = _cfg(M, name)
    env = env_for(MODES[mode])

    def one(k, s):
        return M.search(k, s, env, config, evaluator)

    batched_fn, single_fn = jax.jit(jax.vmap(one)), jax.jit(one)  # fresh: the flag is traced in
    for seed in SEEDS:
        states, keys = _states(mode, seed), _keys(seed)
        ref_batched, ref_single = _reference_outputs(name, mode, seed)
        assert _equal(batched_fn(keys, states), ref_batched)
        for i in range(2):
            out = single_fn(keys[i], jax.tree.map(lambda x, i=i: x[i], states))
            assert _equal(out, ref_single[i])


def reference_tree(key, state, env, config, evaluator):
    """The 1952d9f ``search`` up to ``_output``, on the reference's internals."""
    game = env.config
    n = game.num_snakes
    env = env if env.obs_fn is None else R._search_env(game)
    num_sims = config.num_simulations
    max_depth = min(config.max_depth, num_sims)
    digits = jnp.asarray(R._digits(n))
    k_noise, k_spawn, k_roll, k_final = jax.random.split(key, 4)
    noise = jax.random.bits(k_noise, (num_sims, n, NUM_ACTIONS), jnp.uint32)

    def simulate(sim, carry):
        tree, deepest, expanded = carry
        path = R._descend(tree, noise[sim], config, digits, max_depth)
        leaf, slot = path.node, sim + 1
        joint = path.path_joint[jnp.maximum(path.depth - 1, 0)]
        parent = R._unbatched_barrier(jax.tree.map(lambda x: x[leaf], tree.state))
        child, mask = R._transition(
            jax.random.fold_in(k_spawn, slot), parent, digits[joint], env, config
        )
        new_value, new_logits = R._leaf_value(
            jax.random.fold_in(k_roll, slot), child, mask, env, config, evaluator
        )
        value = jnp.where(path.expand, new_value, path.stored)
        tree = tree._replace(
            state=jax.tree.map(lambda a, x: a.at[slot].set(x), tree.state, child),
            children=tree.children.at[jnp.where(path.expand, leaf, num_sims + 1), joint].set(
                slot, mode="drop"
            ),
            legal=tree.legal.at[slot].set(R._legal(child, mask, game)),
            terminal=tree.terminal.at[slot].set(child.done),
            value=tree.value.at[slot].set(new_value),
        )
        if tree.prior is not None:
            prior = R._node_prior(new_logits, R._legal(child, mask, game), config)
            tree = tree._replace(prior=tree.prior.at[slot].set(prior.astype(jnp.float32)))
        tree = R._backup(tree, path, value, config, digits)
        deepest = jnp.maximum(deepest, jnp.where(path.expand, path.depth, 0))
        return tree, deepest, expanded + path.expand.astype(jnp.int32)

    zero = jnp.zeros((), jnp.int32)
    k_root = jax.random.fold_in(key, R._ROOT_NOISE_STREAM) if config.root_noise_frac > 0 else None
    tree = R._init_tree(state, env, config, num_sims + 1, evaluator, k_root)
    tree, deepest, expanded = jax.lax.fori_loop(0, num_sims, simulate, (tree, zero, zero))
    return tree, deepest, expanded, k_final


def test_reference_tree_helper_is_the_reference_search():
    config, evaluator = _cfg(R, "az")
    env, mode = env_for(MODES["duel"]), "duel"
    states, keys = _states(mode, 0), _keys(0)

    def check(k, s):
        tree, deepest, expanded, k_final = reference_tree(k, s, env, config, evaluator)
        return R._output(k_final, tree, config, deepest, expanded)

    got = jax.jit(jax.vmap(check))(keys, states)
    assert _equal(got, _reference_outputs("az", mode, 0)[0])


@pytest.mark.parametrize("name,mode", [c for c in CASES if c[1] in FULL_MODES][::2])
def test_tree_matches_reference(masked, name, mode):
    config, evaluator = _cfg(M, name)
    ref_config, _ = _cfg(R, name)
    env = env_for(MODES[mode])
    new = jax.jit(jax.vmap(lambda k, s: M._run(k, s, env, config, evaluator)))
    ref = jax.jit(jax.vmap(lambda k, s: reference_tree(k, s, env, ref_config, evaluator)))
    for seed in SEEDS[:2]:
        states, keys = _states(mode, seed), _keys(seed)
        assert _equal(new(keys, states), ref(keys, states))  # structure covers None leaves


@pytest.mark.parametrize("name", ["duct", "rm", "az", "deep"])
def test_descent_paths_match_reference(monkeypatch, name):
    config, evaluator = _cfg(M, name)
    ref_config, _ = _cfg(R, name)
    mode = "standard"
    game = MODES[mode]
    env = env_for(game)
    n = game.num_snakes
    states, keys = _states(mode, 0), _keys(0)
    trees = jax.jit(jax.vmap(lambda k, s: M._run(k, s, env, config, evaluator)[0]))(keys, states)
    sims = config.num_simulations
    max_depth = min(config.max_depth, sims)
    digits = jnp.asarray(M._digits(n))
    bits = jax.vmap(
        lambda k: jax.random.bits(jax.random.split(k, 4)[0], (sims, n, NUM_ACTIONS), jnp.uint32)
    )(keys)  # [E, sims, N, 4]
    monkeypatch.setattr(M, "_masked_descent_on_cpu", True)

    def descents(module, cfg):
        def one(t, b):
            return module._descend(t, b, cfg, digits, max_depth)

        batched = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, 0)))
        single = jax.jit(one)
        tree0 = jax.tree.map(lambda x: x[0], trees)
        return batched(trees, bits), [single(tree0, bits[0, i]) for i in range(0, sims, 7)]

    assert _equal(descents(M, config), descents(R, ref_config))


@pytest.mark.parametrize("batch", [None, 1, 2])
@pytest.mark.parametrize("network", [False, True])
def test_masked_tree_is_updated_in_place(monkeypatch, batch, network):
    monkeypatch.setattr(M, "_masked_descent_on_cpu", True)
    config = M.MCTSConfig(num_simulations=40)
    if network:
        config = dataclasses.replace(
            config, selection="puct", prior="network", leaf="network", root_noise_frac=0.25
        )
    env, key, state = env_for(MODES["duel"]), jax.random.key(0), mid_game_state()

    def search(k, s):
        return M.search(k, s, env, config, fake_eval if network else None)

    if batch is not None:
        search = jax.vmap(search)
        key = jax.random.split(key, batch)
        state = jax.tree.map(lambda x: jnp.stack([x] * batch), state)
    assert tree_copies(search, key, state, num_nodes=41) == []


def _load_az_modes():
    name = "baselines_az_modes"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "baselines" / "az_modes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_az_selfplay_matches_reference(masked):
    # The joint search calls the real network once over every mode's rows; on the CPU at
    # these shapes (and at 256 games of S, M and L) that rounds as the per-mode calls did.
    azm = _load_az_modes()
    cfg = azm.Config(**TINY, num_envs=4, num_steps=3, num_simulations=16, max_turns=40)
    groups = unified.make_groups(cfg)
    runner = azm.init_runner(cfg, groups, jax.random.key(0))
    args = (runner.params, runner.env_states, runner.uids, jax.random.key(9))
    new = jax.jit(azm.make_selfplay(cfg, groups))(*args)
    ref = jax.jit(RA.make_selfplay(cfg, groups))(*args)
    assert _equal(new, ref)


@pytest.mark.parametrize("mode", list(MODES))
def test_root_draws_match_the_search_draws(masked, mode):
    config, evaluator = _cfg(M, "az")
    game = MODES[mode]
    env = env_for(game)
    n = game.num_snakes
    states, keys = _states(mode, 0), _keys(0)

    own = jax.jit(jax.vmap(lambda k, s: M.search(k, s, env, config, evaluator)))
    pre = jax.jit(
        jax.vmap(
            lambda k, s: M.search(
                k, s, env, config, evaluator, root_draws=M.root_noise_draws(k, n, config)
            )
        )
    )
    assert _equal(own(keys, states), pre(keys, states))
    ref_config, _ = _cfg(R, "az")
    ref = jax.jit(jax.vmap(lambda k, s: R.search(k, s, env, ref_config, evaluator)))
    assert _equal(pre(keys, states), ref(keys, states))
    single = jax.tree.map(lambda x: x[0], states)
    draws = M.root_noise_draws(keys[0], n, config)
    assert _equal(
        jax.jit(lambda k, s, d: M.search(k, s, env, config, evaluator, root_draws=d))(
            keys[0], single, draws
        ),
        jax.tree.map(lambda x: x[0], ref(keys, states)),
    )


def test_root_draws_are_the_same_under_any_batching():
    config = M.MCTSConfig(selection="puct", root_noise_frac=0.25, root_dirichlet_alpha=0.5)
    keys = jax.random.split(jax.random.key(3), 21).reshape(3, 7)
    nested = jax.jit(jax.vmap(jax.vmap(lambda k: M.root_noise_draws(k, 4, config))))(keys)
    flat = jax.jit(jax.vmap(lambda k: M.root_noise_draws(k, 4, config)))(keys.reshape(-1))
    draw = jax.jit(lambda k: M.root_noise_draws(k, 4, config))
    each = jnp.stack([draw(k) for k in keys.reshape(-1)])
    assert nested.shape == (3, 7, 4, NUM_ACTIONS)
    assert np.array_equal(np.asarray(nested).reshape(21, 4, 4), np.asarray(flat))
    assert np.array_equal(np.asarray(flat), np.asarray(each))


def test_root_draws_are_checked():
    game = MODES["duel"]
    env = env_for(game)
    state = jax.tree.map(lambda x: x[0], _states("duel", 0))
    key = jax.random.key(0)
    noisy, evaluator = _cfg(M, "az")
    with pytest.raises(ValueError, match="root_draws"):
        M.search(key, state, env, noisy, evaluator, root_draws=jnp.zeros((3, 4)))
    quiet = dataclasses.replace(noisy, root_noise_frac=0.0)
    with pytest.raises(ValueError, match="root_draws"):
        M.search(key, state, env, quiet, evaluator, root_draws=jnp.zeros((2, 4)))


# --- search_groups: every mode in one lockstep search -------------------------------------

GROUP_GAMES = {**MODES, "three": GameConfig(num_snakes=3)}


def groups_eval(states):
    """``fake_eval`` of every game of every group (row-independent, like the contract asks)."""
    return tuple(jax.vmap(fake_eval)(s) for s in states)


@functools.cache
def _group_inputs(e):
    states = tuple(
        jax.tree.map(lambda x: x[:e], rollout_states(g, batch=BATCH, turns=25, seed=3))
        for g in GROUP_GAMES.values()
    )
    keys = tuple(jax.random.split(jax.random.key(11 + i), e) for i in range(len(GROUP_GAMES)))
    return keys, states


@functools.cache
def _reference_groups(name, e):
    config, evaluator = _cfg(R, name)
    keys, states = _group_inputs(e)
    outs = []
    for g, k, s in zip(GROUP_GAMES.values(), keys, states, strict=True):
        env = env_for(g)

        def one(k, s, env=env):
            return R.search(k, s, env, config, evaluator)

        outs.append(jax.jit(jax.vmap(one))(k, s))
    return tuple(outs)


@pytest.mark.parametrize("e", [5, 1])
@pytest.mark.parametrize("name", ["duct", "rm", "az", "deep"])
def test_search_groups_matches_per_group(masked, name, e):
    # Every mode and a 3-snake game in one search, against each group's own 1952d9f search.
    config, evaluator = _cfg(M, name)
    envs = tuple(env_for(g) for g in GROUP_GAMES.values())
    keys, states = _group_inputs(e)
    joint = jax.jit(
        lambda ks, ss: M.search_groups(ks, ss, envs, config, groups_eval if evaluator else None)
    )(keys, states)
    ref = _reference_groups(name, e)
    if name == "deep":  # the lockstep walk's loop runs after its straight-line steps
        assert max(int(jnp.max(r.depth)) for r in ref) > M._DESCENT_STEPS
    assert len(joint) == len(ref)
    for g, (a, b) in enumerate(zip(joint, ref, strict=True)):
        assert _equal(a, b), list(GROUP_GAMES)[g]


def test_search_groups_takes_root_draws():
    config, _ = _cfg(M, "az")
    envs = tuple(env_for(g) for g in GROUP_GAMES.values())
    keys, states = _group_inputs(5)
    draws = tuple(
        jax.vmap(lambda k, n=g.num_snakes: M.root_noise_draws(k, n, config))(k)
        for g, k in zip(GROUP_GAMES.values(), keys, strict=True)
    )
    run = jax.jit(lambda ks, ss, ds: M.search_groups(ks, ss, envs, config, groups_eval, ds))
    assert _equal(run(keys, states, draws), _reference_groups("az", 5))


def test_search_groups_checks_its_arguments():
    envs = (env_for(MODES["duel"]), env_for(MODES["standard"]))
    keys, states = _group_inputs(2)
    pick = [list(GROUP_GAMES).index(m) for m in ("duel", "standard")]
    keys, states = tuple(keys[i] for i in pick), tuple(states[i] for i in pick)
    noisy, _ = _cfg(M, "az")
    with pytest.raises(ValueError, match="evaluate"):
        M.search_groups(keys, states, envs, noisy)
    with pytest.raises(ValueError, match="evaluate"):
        M.search_groups(keys, states, envs, M.MCTSConfig(num_simulations=4), groups_eval)
    with pytest.raises(ValueError, match="one batch"):
        M.search_groups(keys[:1], states, envs, M.MCTSConfig(num_simulations=4))

    def wrong_shape(ss):
        return tuple((v[:, :1], lg) for v, lg in groups_eval(ss))

    with pytest.raises(ValueError, match="evaluate must return"):
        M.search_groups(keys, states, envs, noisy, wrong_shape)
    with pytest.raises(ValueError, match="root_draws"):
        M.search_groups(keys, states, envs, noisy, groups_eval, (jnp.zeros((2, 2, 4)),) * 2)


@pytest.mark.parametrize("on_cpu", [False, True], ids=["cpu_loop", "masked"])
@pytest.mark.parametrize("e", [1, 2])
def test_search_groups_tree_is_updated_in_place(monkeypatch, on_cpu, e):
    monkeypatch.setattr(M, "_masked_descent_on_cpu", on_cpu)
    config = M.MCTSConfig(
        num_simulations=40, selection="puct", prior="network", leaf="network",
        root_noise_frac=0.25,
    )  # fmt: skip
    games = (MODES["duel"], MODES["standard"])
    envs = tuple(env_for(g) for g in games)
    states = tuple(
        jax.tree.map(lambda x: x[:e], rollout_states(g, batch=BATCH, turns=25, seed=3))
        for g in games
    )
    keys = tuple(jax.random.split(jax.random.key(1 + i), e) for i in range(len(games)))

    def run(ks, ss):
        return M.search_groups(ks, ss, envs, config, groups_eval)

    assert tree_copies(run, keys, states, num_nodes=41) == []


def rowwise_network(params, boards, feats, cfg):
    """A stand-in for ``az_modes.network`` that is exactly row-independent (slices only)."""
    lead = boards.shape[:-3]
    b = boards.reshape(-1, *boards.shape[-3:])
    f = feats.reshape(-1, feats.shape[-1])
    logits = b[:, 10, 9:13, 1] * 0.7 - b[:, 9, 10:14, 2] * 0.3 + f[:, :4] * params["policy"]["b"]
    value = jnp.tanh(b[:, 11, 10, 3] - b[:, 10, 11, 4] + 0.1 * f[:, 0])
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


def test_az_selfplay_joint_with_rowwise_net(masked, monkeypatch):
    # With a network whose rows are independent, the joint search's self-play is the
    # 1952d9f per-mode self-play bit for bit; only the real network's batch rounding differs.
    azm = _load_az_modes()
    monkeypatch.setattr(azm, "network", rowwise_network)
    monkeypatch.setattr(RA, "network", rowwise_network)
    cfg = azm.Config(**TINY, num_envs=4, num_steps=4, num_simulations=16, max_turns=40)
    groups = unified.make_groups(cfg)
    runner = azm.init_runner(cfg, groups, jax.random.key(0))
    bias = jnp.array([1.0, -2.0, 3.0, 0.0])
    params = dict(runner.params, policy={**runner.params["policy"], "b": bias})
    args = (params, runner.env_states, runner.uids, jax.random.key(9))
    new = jax.jit(azm.make_selfplay(cfg, groups))(*args)
    ref = jax.jit(RA.make_selfplay(cfg, groups))(*args)
    assert _equal(new, ref)


@pytest.mark.parametrize("size", ["tiny", "M"])
def test_groups_evaluator_matches_per_group_evaluator(size):
    # The joint search's one source of difference: one network call over every mode's
    # rows instead of one per mode (the same rows; a GEMM's rounding may depend on the batch).
    # With these perturbed parameters (outputs up to about 2) M differed by up to 1.6e-6.
    azm = _load_az_modes()
    if size == "tiny":
        shape = TINY
    else:
        shape = dict(zip(("conv_channels", "hidden"), unified.SIZES[size], strict=True))
    cfg = azm.Config(**shape, num_envs=8)
    groups = unified.make_groups(cfg)
    params = azm.init_network(jax.random.key(2), cfg)
    params = jax.tree.map(lambda x: x + 0.1 * jax.random.normal(jax.random.key(3), x.shape), params)
    states = tuple(rollout_states(g.game, batch=cfg.num_envs, turns=25, seed=4) for g in groups)
    joint = jax.jit(lambda p, ss: azm.make_groups_evaluator(p, cfg, groups)(ss))(params, states)
    worst = 0.0
    for g, s, (value, logits) in zip(groups, states, joint, strict=True):
        own = jax.vmap(lambda st, p, g=g: azm.make_evaluator(p, cfg, g)(st), (0, None))
        own_v, own_l = jax.jit(own)(s, params)
        np.testing.assert_allclose(value, own_v, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(logits, own_l, atol=1e-5, rtol=1e-5)
        for a, b in ((value, own_v), (logits, own_l)):
            worst = max(worst, float(jnp.max(jnp.abs(a - b))))
    print(f"{size}: max abs difference of the joint network call {worst:.3g}")
