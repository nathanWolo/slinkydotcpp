"""Tests for the multi-mode PPO trainer (``baselines/ppo_modes.py``)."""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import modes
from slinky.env import BattlesnakeEnv, placement_reward, win_loss_reward
from slinky.evaluate import play_table
from slinky.observations import CHANNEL_NAMES
from slinky.types import NUM_ACTIONS, RoyaleState, TimeStep

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, file_name: str):
    """Import ``baselines/<file_name>`` from its path (``baselines/`` is not a package)."""
    if module_name in sys.modules:  # e.g. already loaded by slinky.agents or another test
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "baselines" / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


ppo = _load("baselines_ppo", "ppo.py")
pm = _load("baselines_ppo_modes", "ppo_modes.py")
# A tiny network so tests compile and run fast.
TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)
FOOD = CHANNEL_NAMES.index("food")
HAZARD = CHANNEL_NAMES.index("hazard")
TURN = modes.FEATURE_NAMES.index("turn")
PHASE = modes.FEATURE_NAMES.index("shrink_phase")


def _equal_trees(a, b) -> None:
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _with_random_feat(params, key):
    """``params`` with a non-zero ``feat/w`` (and policy head), so the features matter."""
    k_feat, k_policy = jax.random.split(key)
    w, policy = params["feat"]["w"], params["policy"]["w"]
    return {
        **params,
        "feat": {"w": jax.random.normal(k_feat, w.shape) * 0.5},
        "policy": {**params["policy"], "w": jax.random.normal(k_policy, policy.shape) * 0.3},
    }


# --- (1) Config ---------------------------------------------------------------------


def test_config_checks():
    with pytest.raises(ValueError, match="unknown game_modes"):
        pm.ModesConfig(game_modes=("duel", "solo"))
    with pytest.raises(ValueError, match="duplicates"):
        pm.ModesConfig(game_modes=("duel", "duel"))
    with pytest.raises(ValueError, match="at least one"):
        pm.ModesConfig(game_modes=())
    with pytest.raises(ValueError, match="<mode>=<snakes>"):
        pm.ModesConfig(mode_snakes=("constrictor",))
    with pytest.raises(ValueError, match="not in game_modes"):
        pm.ModesConfig(game_modes=("duel",), mode_snakes=("constrictor=2",))
    with pytest.raises(ValueError, match="cannot have 3 snakes"):
        pm.ModesConfig(mode_snakes=("duel=3",))
    with pytest.raises(ValueError, match="twice"):
        pm.ModesConfig(mode_snakes=("royale=2", "royale=3"))
    with pytest.raises(ValueError, match="unknown reward"):
        pm.ModesConfig(reward="score")
    with pytest.raises(ValueError, match="unknown features"):
        pm.ModesConfig(features="mode")
    with pytest.raises(ValueError, match="one weight per game mode"):
        pm.ModesConfig(mode_weights=(1.0, 2.0))
    with pytest.raises(ValueError, match="positive sum"):
        pm.ModesConfig(game_modes=("duel",), mode_weights=(0.0,))
    with pytest.raises(ValueError, match="eval_modes"):
        pm.ModesConfig(eval_modes=("argmax",))
    with pytest.raises(ValueError, match="snapshot_at"):
        pm.ModesConfig(snapshot_at=(0,))
    # 3 * 5 * 2 = 30 duel samples are not a multiple of 4 (the 4-snake modes' 60 are).
    with pytest.raises(ValueError, match="duel: num_envs"):
        pm.ModesConfig(num_envs=3, num_steps=5, num_minibatches=4)
    pm.ModesConfig(game_modes=("standard",), num_envs=3, num_steps=5, num_minibatches=4)

    cfg = pm.ModesConfig(mode_snakes=("constrictor=2",), mode_weights=(1, 2, 1, 0))
    assert cfg.snakes == {"standard": 4, "duel": 2, "royale": 4, "constrictor": 2}
    assert cfg.games["constrictor"] == modes.mode_config("constrictor", 2, 1000)
    assert cfg.games["royale"] == modes.mode_config("royale", 4, 1000)
    assert cfg.game == cfg.games["standard"] and cfg.game.max_turns == 1000
    assert cfg.loss_weights == (0.25, 0.5, 0.25, 0.0)
    assert pm.ModesConfig(game_modes=("duel", "royale")).loss_weights == (0.5, 0.5)
    s = pm.schedule(pm.ModesConfig(num_envs=4, num_steps=8, log_every=100, total_env_steps=200))
    assert (s.chunk_iters, s.chunk_steps, s.num_chunks, s.total_steps) == (3, 96, 3, 288)
    assert s.total_updates == 9 * 4 * 8  # per-mode iterations x epochs x minibatches


def test_parser_takes_element_types_from_annotations():
    cfg, args = pm.parse_args(
        [
            "--game-modes", "duel,royale", "--mode-snakes", "royale=3", "--mode-weights", "2,1",
            "--value-norm", "false", "--features", "no_mode", "--init-from", "some/run",
            "--conv-channels", "4,8", "--conv-strides", "2,2", "--lr", "3e-4",
            "--total-env-steps", "1_000", "--snapshot-at", "8192,16384",
        ]
    )  # fmt: skip
    assert cfg.game_modes == ("duel", "royale") and cfg.mode_snakes == ("royale=3",)
    assert cfg.mode_weights == (2.0, 1.0) and isinstance(cfg.mode_weights[0], float)
    assert cfg.value_norm is False and cfg.features == "no_mode" and cfg.init_from == "some/run"
    assert cfg.conv_channels == (4, 8) and cfg.lr == 3e-4 and cfg.total_env_steps == 1000
    assert cfg.snapshot_at == (8192, 16384)
    assert cfg.snakes == {"duel": 2, "royale": 3}
    assert args.resume is None and args.eval_only is None
    default, _ = pm.parse_args([])
    assert default == pm.ModesConfig() and default.game_modes == modes.MODE_NAMES
    assert default.mode_weights == () and default.run_dir is None
    assert default.snapshot_at == (5_242_880 // 4,) and default.value_norm is True


# --- (2) Single mode = ppo.py ------------------------------------------------------------


@pytest.mark.parametrize(
    "reward, value_norm", [("win_loss", False), ("placement", True)], ids=["ppo", "defaults"]
)
def test_single_duel_group_matches_ppo(reward, value_norm):
    """game_modes=duel without features: ppo.py, bit for bit.

    With win/loss and no value normalisation, and with the defaults (the
    placement reward is win/loss in a duel, and value_norm's factor is exactly
    1 for a single mode). The runner, a rollout (short games, so truncations
    and terminal steps occur) and one full iteration (GAE, 2 epochs of 4
    minibatches) are compared.
    """
    kw = dict(num_envs=8, num_steps=16, max_turns=12, num_minibatches=4, update_epochs=2, **TINY)
    pcfg = ppo.PPOConfig(**kw)
    mcfg = pm.ModesConfig(
        game_modes=("duel",), features="none", reward=reward, value_norm=value_norm, **kw
    )
    env, sim = ppo.make_envs(pcfg)
    groups = pm.make_groups(mcfg)
    assert groups[0].game == pcfg.game
    assert groups[0].sim.reward_fn is pm.REWARDS[reward] and sim.reward_fn is win_loss_reward
    key = jax.random.key(3)
    p_runner = ppo.init_runner(pcfg, sim, key)
    m_runner = pm.init_runner(mcfg, groups, key)
    _equal_trees(m_runner.params, p_runner.params)  # no feat layer: ppo.py's parameters
    _equal_trees(m_runner.env_states, (p_runner.env_states,))

    k = jax.random.key(5)
    _, p_traj, p_metrics = jax.jit(ppo.make_rollout(pcfg, env, sim))(
        p_runner.params, p_runner.env_states, k
    )
    _, (m_traj,), m_metrics = jax.jit(pm.make_rollout(mcfg, groups))(
        m_runner.params, m_runner.env_states, k
    )
    for field in ppo.Rollout._fields:
        np.testing.assert_array_equal(getattr(m_traj, field), getattr(p_traj, field), field)
    for name, value in p_metrics.items():
        np.testing.assert_array_equal(m_metrics[name][0], value, name)
    assert int(p_metrics["truncated"]) > 0 and int(p_metrics["games"]) > int(p_metrics["truncated"])

    p_after, p_m = jax.jit(ppo.make_train_chunk(pcfg, env, sim, 1))(p_runner)
    m_after, m_m = jax.jit(pm.make_train_chunk(mcfg, groups, 1))(m_runner)
    # The keys are the same all the way, and so is every operation: identical parameters.
    _equal_trees(m_after.params, p_after.params)
    _equal_trees(m_after.env_states, (p_after.env_states,))
    assert int(m_after.env_steps) == int(p_after.env_steps)
    assert int(m_after.updates) == int(p_after.updates)
    for name in ("loss", "grad_norm"):
        np.testing.assert_allclose(m_m[name], p_m[name], rtol=1e-6)
    for name in (*ppo.LOSS_METRICS[1:], "explained_var", "value_mean"):
        np.testing.assert_allclose(m_m[name][0], p_m[name], rtol=1e-6, atol=1e-9)
    assert float(m_m["value_weight"][0]) == 1.0


def test_features_start_as_the_ppo_network():
    """feat/w starts at zero: the network is ppo.py's whatever the features."""
    pcfg = ppo.PPOConfig(**TINY)
    params = ppo.init_network(jax.random.key(0), pcfg)
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.random((6, 2, 21, 21, 13)), jnp.float32)
    feats = jnp.asarray(rng.random((6, 2, modes.NUM_FEATURES)), jnp.float32)
    p_logits, p_values = ppo.network(params, obs, pcfg)
    for features in ("all", "no_mode"):
        cfg = pm.ModesConfig(features=features, **TINY)
        m_params = pm.init_network(jax.random.key(0), cfg)
        assert m_params["feat"]["w"].shape == (modes.NUM_FEATURES, 16)
        assert not m_params["feat"]["w"].any()
        _equal_trees({k: v for k, v in m_params.items() if k != "feat"}, params)
        logits, values = pm.network(m_params, obs, feats, cfg)
        np.testing.assert_array_equal(logits, p_logits)
        np.testing.assert_array_equal(values, p_values)
    none = pm.ModesConfig(features="none", **TINY)
    _equal_trees(pm.init_network(jax.random.key(0), none), params)


def test_policy_features_and_food_hiding_follow_the_config():
    """no_mode keeps only the turn and opponents; food hides by the real ruleset."""
    game = modes.mode_config("royale")
    env = BattlesnakeEnv(game)
    state, ts = env.reset(jax.random.key(0))
    state = state._replace(turn=jnp.int32(37))
    full = pm.policy_features(state, game, pm.ModesConfig())
    np.testing.assert_array_equal(full, modes.mode_features(state, game))
    no_mode = pm.policy_features(state, game, pm.ModesConfig(features="no_mode"))
    keep = [TURN, modes.FEATURE_NAMES.index("opponents_alive")]
    np.testing.assert_array_equal(no_mode[:, keep], full[:, keep])
    assert full[:, PHASE].any() and not np.delete(np.asarray(no_mode), keep, axis=1).any()
    assert not pm.policy_features(state, game, pm.ModesConfig(features="none")).any()
    told = pm.policy_features(state, game, pm.ModesConfig(), "duel")
    np.testing.assert_array_equal(told, modes.mode_features(state, game, "duel"))

    cfg = pm.ModesConfig()
    for mode in modes.MODE_NAMES:
        g = modes.mode_config(mode)
        e = BattlesnakeEnv(g)
        s, t = e.reset(jax.random.key(1))
        s, t = e.step(jax.random.key(2), s, jnp.zeros(g.num_snakes, jnp.int32))
        assert t.obs[..., FOOD].any()  # the map spawned food (minimum_food=1), even constrictor
        for told_mode in modes.MODE_NAMES:  # whatever the network is told
            boards, _ = pm.policy_inputs(t.obs, s, g, cfg, told_mode)
            assert boards[..., FOOD].any() == (mode != "constrictor")
            np.testing.assert_array_equal(np.delete(boards, FOOD, -1), np.delete(t.obs, FOOD, -1))
        shown = pm.ModesConfig(hide_constrictor_food=False)
        np.testing.assert_array_equal(pm.policy_inputs(t.obs, s, g, shown)[0], t.obs)


# --- (3) Groups, rollouts and the update -------------------------------------------------


@pytest.fixture(scope="module")
def four_modes():
    """A 6-turn rollout of 3 games per mode with features on (a random, non-zero feat/w)."""
    cfg = pm.ModesConfig(num_envs=3, num_steps=6, num_minibatches=2, **TINY)
    groups = pm.make_groups(cfg)
    runner = pm.init_runner(cfg, groups, jax.random.key(0))
    params = _with_random_feat(runner.params, jax.random.key(9))
    states, trajs, metrics = jax.jit(pm.make_rollout(cfg, groups))(
        params, runner.env_states, jax.random.key(1)
    )
    return cfg, groups, params, states, trajs, metrics


def test_group_shapes_offsets_and_inputs(four_modes):
    cfg, groups, params, states, trajs, metrics = four_modes
    assert [g.name for g in groups] == list(modes.MODE_NAMES)
    assert [g.mode_id for g in groups] == [0, 1, 2, 3]
    assert [g.num_snakes for g in groups] == [4, 2, 4, 4]
    assert all(g.sim.reward_fn is placement_reward for g in groups)  # the default reward
    assert pm.group_offsets(groups, 3) == ((0, 12), (12, 6), (18, 12), (30, 12))
    T, E = 6, 3
    for i, (g, traj, s) in enumerate(zip(groups, trajs, states, strict=True)):
        n = g.num_snakes
        assert traj.obs.shape == (T, E, n, 21, 21, 13)
        assert traj.feats.shape == (T, E, n, modes.NUM_FEATURES)
        assert traj.mask.shape == (T, E, n, NUM_ACTIONS)
        for name in ("actions", "log_probs", "values", "rewards", "valid", "terminal"):
            assert getattr(traj, name).shape == (T, E, n), name
        assert traj.done.shape == (T, E)
        assert s.body.shape == (E, n, 11, 11)
        one_hot = np.asarray(traj.feats[..., : len(modes.MODE_NAMES)])
        assert (one_hot == np.eye(len(modes.MODE_NAMES))[g.mode_id]).all()
        # Each group's moves come from its own rows of the shared network call: recompute
        # the first turn's outputs per group and compare.
        logits, values = pm.network(params, traj.obs[0], traj.feats[0], cfg)
        np.testing.assert_allclose(traj.values[0], values, rtol=1e-5, atol=1e-6)
        logp = jnp.take_along_axis(
            pm.masked_log_probs(logits, traj.mask[0]), traj.actions[0][..., None], -1
        )[..., 0]
        np.testing.assert_allclose(traj.log_probs[0], logp, rtol=1e-5, atol=1e-5)
        assert bool(jnp.take_along_axis(traj.mask, traj.actions[..., None], -1).all())
        # Constrictor boards never show food, the others do; the raw observation has it.
        assert traj.obs[..., FOOD].any() == (g.name != "constrictor")
        assert jax.vmap(g.env.observe)(s)[..., FOOD].any()
        assert int(metrics["agent_samples"][i]) == int(traj.valid.sum())
    assert metrics["deaths"].shape == (4, len(pm.Cause))


def test_placement_reward_reaches_the_rollouts():
    """4-snake rewards are zero-sum thirds with placement, not with win/loss; the duel's agree."""
    trajs = {}
    for reward in ("placement", "win_loss"):
        cfg = pm.ModesConfig(
            game_modes=("standard", "duel"), reward=reward, num_envs=4, num_steps=40, **TINY
        )
        groups = pm.make_groups(cfg)
        assert all(g.sim.reward_fn is pm.REWARDS[reward] for g in groups)
        runner = pm.init_runner(cfg, groups, jax.random.key(0))
        _, trajs[reward], _ = jax.jit(pm.make_rollout(cfg, groups))(
            runner.params, runner.env_states, jax.random.key(1)
        )
    (std, duel), (std_wl, duel_wl) = trajs["placement"], trajs["win_loss"]
    # The reward never changes play: the same games, only the payments differ.
    np.testing.assert_array_equal(std.actions, std_wl.actions)
    rewards = np.asarray(std.rewards)
    assert (rewards != 0).any()
    np.testing.assert_allclose(rewards.sum(axis=-1), 0.0, atol=1e-6)  # zero-sum every turn
    np.testing.assert_allclose(rewards * 3, np.rint(rewards * 3), atol=1e-5)  # multiples of 1/3
    assert np.isclose(np.abs(rewards), 1 / 3).any()  # outlasting one of three opponents
    assert (np.abs(np.asarray(std_wl.rewards).sum(axis=-1)) > 0.5).any()  # win/loss is not
    np.testing.assert_array_equal(duel.rewards, duel_wl.rewards)  # in a duel they are one
    assert (np.asarray(duel.rewards) != 0).any()


def test_stratified_minibatches():
    groups = pm.make_groups(pm.ModesConfig(game_modes=("duel", "royale")))
    key = jax.random.key(4)
    idx = pm.minibatch_indices(key, groups, (48, 96), 4)
    assert [i.shape for i in idx] == [(4, 12), (4, 24)]
    for i, size in zip(idx, (48, 96), strict=True):
        assert sorted(np.asarray(i).ravel().tolist()) == list(range(size))
    # One group: ppo.py's permutation, from the epoch key itself.
    (single,) = pm.minibatch_indices(key, groups[:1], (48,), 4)
    np.testing.assert_array_equal(single, jax.random.permutation(key, 48).reshape(4, -1))
    np.testing.assert_array_equal(
        idx[0], jax.random.permutation(jax.random.fold_in(key, 1), 48).reshape(4, -1)
    )


def _batch(rng, b, n_valid):
    obs = jnp.asarray(rng.random((b, 21, 21, 13)), jnp.float32)
    feats = jnp.asarray(rng.random((b, modes.NUM_FEATURES)), jnp.float32)
    mask = jnp.asarray(rng.random((b, NUM_ACTIONS)) < 0.7).at[:, 0].set(True)
    return pm.Batch(
        obs=obs,
        feats=feats,
        mask=mask,
        actions=jnp.zeros(b, jnp.int32),
        log_probs=jnp.asarray(rng.normal(-1.0, 0.1, b), jnp.float32),
        values=jnp.asarray(rng.normal(0, 0.3, b), jnp.float32),
        advantages=jnp.asarray(rng.normal(size=b), jnp.float32),
        returns=jnp.asarray(rng.uniform(-1, 1, b), jnp.float32),
        valid=jnp.arange(b) < n_valid,
    )


@pytest.mark.parametrize("value_norm", [False, True])
def test_loss_is_the_weighted_mean_of_per_group_losses(value_norm):
    """One forward pass on the concatenation; each group normalised and weighted on its own."""
    cfg = pm.ModesConfig(
        game_modes=("duel", "royale"), mode_weights=(3.0, 1.0), value_norm=value_norm, **TINY
    )
    params = _with_random_feat(pm.init_network(jax.random.key(0), cfg), jax.random.key(1))
    rng = np.random.default_rng(0)
    mbs = (_batch(rng, 8, 6), _batch(rng, 16, 11))
    return_vars = jnp.array([0.002, 0.3], jnp.float32)  # below and above MIN_RETURN_VAR
    weights = pm.value_loss_weights(return_vars, cfg)
    if value_norm:  # 1/Var (at least 1/MIN_RETURN_VAR), with weighted mean 1
        inv = np.array([1 / pm.MIN_RETURN_VAR, 1 / 0.3])
        np.testing.assert_allclose(weights, inv / (0.75 * inv[0] + 0.25 * inv[1]), rtol=1e-6)
        assert 0.75 * float(weights[0]) + 0.25 * float(weights[1]) == pytest.approx(1.0)
    else:
        np.testing.assert_array_equal(weights, [1.0, 1.0])
    loss, parts = jax.jit(lambda p, m, w: pm.modes_loss(p, m, w, cfg))(params, mbs, weights)
    assert parts["pg_loss"].shape == (2,)

    expected = []
    for g, mb in enumerate(mbs):
        logits, values = pm.network(params, mb.obs, mb.feats, cfg)
        loss_g, parts_g = pm.group_loss(logits, values, mb, weights[g], cfg)
        for name, value in parts_g.items():
            np.testing.assert_allclose(parts[name][g], value, rtol=1e-5, atol=1e-7)
        total = (
            parts_g["pg_loss"]
            + cfg.vf_coef * parts_g["v_loss"] * float(weights[g])
            - cfg.ent_coef * parts_g["entropy"]
        )
        np.testing.assert_allclose(loss_g, total, rtol=1e-5)
        expected.append(loss_g)
    np.testing.assert_allclose(loss, 0.75 * expected[0] + 0.25 * expected[1], rtol=1e-5)
    # Advantages are normalised per group: shifting and scaling one group's leaves both alone.
    shifted = (mbs[0]._replace(advantages=mbs[0].advantages * 5.0 + 2.0), mbs[1])
    loss2, parts2 = pm.modes_loss(params, shifted, weights, cfg)
    np.testing.assert_allclose(loss2, loss, rtol=1e-4)
    np.testing.assert_allclose(parts2["pg_loss"], parts["pg_loss"], rtol=1e-4, atol=1e-6)


def test_value_weights_come_from_the_iterations_returns():
    """return_var is the valid-weighted variance of each group's GAE returns; value_weight
    is 1/return_var normalised to a mean of 1 under mode_weights."""
    cfg = pm.ModesConfig(
        game_modes=("duel", "royale"), mode_weights=(3.0, 1.0), num_envs=3, num_steps=12,
        num_minibatches=2, update_epochs=1, **TINY,
    )  # fmt: skip
    groups = pm.make_groups(cfg)
    runner = pm.init_runner(cfg, groups, jax.random.key(0))
    # A larger value head, so the returns vary by more than MIN_RETURN_VAR.
    params = {**runner.params, "value": {**runner.params["value"]}}
    params["value"]["w"] = params["value"]["w"] * 8.0
    runner = runner._replace(params=params)
    _, m = jax.jit(pm.make_train_chunk(cfg, groups, 1))(runner)

    # The iteration's rollout, recomputed from the runner's key.
    _, k_roll, _ = jax.random.split(runner.key, 3)
    states, trajs, _ = jax.jit(pm.make_rollout(cfg, groups))(params, runner.env_states, k_roll)
    last_values = jax.jit(pm.make_values(cfg, groups))(params, states)
    expected = []
    for traj, last in zip(trajs, last_values, strict=True):
        adv, ret = pm.compute_gae(
            traj.rewards, traj.values, traj.valid, traj.terminal, traj.done,
            traj.boot_values, last, cfg.gamma, cfg.gae_lambda,
        )  # fmt: skip
        valid = np.asarray(traj.valid)
        expected.append(np.asarray(ret)[valid].var())
        assert not np.isclose(np.asarray(adv)[valid].var(), expected[-1], rtol=1e-2)
    np.testing.assert_allclose(m["return_var"], expected, rtol=1e-4)
    assert min(expected) > pm.MIN_RETURN_VAR and not np.isclose(*expected, rtol=0.05)
    inv = 1.0 / np.asarray(expected)
    np.testing.assert_allclose(m["value_weight"], inv / (0.75 * inv[0] + 0.25 * inv[1]), rtol=1e-4)


def test_royale_group_carries_map_state_through_resets_and_save(tmp_path):
    """Royale's shrink sides survive the scan, are redrawn at autoreset and saved exactly."""
    cfg = pm.ModesConfig(
        game_modes=("duel", "royale"),
        mode_snakes=("royale=2",),
        num_envs=8,
        num_steps=40,
        max_turns=30,
        **TINY,
    )
    groups = pm.make_groups(cfg)
    runner = pm.init_runner(cfg, groups, jax.random.key(0))
    sides0 = runner.env_states[1].map_state.sides
    assert isinstance(runner.env_states[1].map_state, RoyaleState) and sides0.shape == (8, 21)
    assert runner.env_states[0].map_state == ()
    states, trajs, _ = jax.jit(pm.make_rollout(cfg, groups))(
        runner.params, runner.env_states, jax.random.key(1)
    )
    royale = trajs[1]
    assert isinstance(states[1].map_state, RoyaleState)
    assert states[1].map_state.sides.dtype == sides0.dtype
    assert np.asarray(royale.done).any(axis=0).all()  # every game ended (max_turns=30 < 40)
    # A reset draws new sides (the same 21 sides again has probability 4**-21).
    assert (np.asarray(states[1].map_state.sides) != np.asarray(sides0)).any(axis=1).all()
    # Hazards appear on the board from turn 25 on and disappear in the next game.
    valid = np.asarray(royale.valid)
    turns = np.rint(np.asarray(royale.feats[..., TURN]) * modes.TURN_SCALE).astype(int)
    hazard = np.asarray(royale.obs[..., HAZARD]).sum(axis=(-1, -2))
    assert (valid & (turns >= 25)).any() and (valid & (turns < 25)).any()
    assert (hazard[valid & (turns >= 25)] >= 11).all()
    assert (hazard[valid & (turns < 25)] == 0).all()
    np.testing.assert_allclose(royale.feats[..., PHASE], turns % 25 / 25, atol=1e-6)
    assert not np.asarray(trajs[0].feats[..., PHASE]).any()

    saved = runner._replace(env_states=states)
    pm.save_runner(str(tmp_path), saved, 123)
    loaded, agent_samples = pm.load_runner(str(tmp_path), runner)
    assert agent_samples == 123 and isinstance(loaded.env_states[1].map_state, RoyaleState)
    _equal_trees(loaded.env_states, states)
    np.testing.assert_array_equal(jax.random.key_data(loaded.key), jax.random.key_data(saved.key))


def test_truncation_bootstraps_from_the_reached_state_with_its_features():
    """boot_values is V of the state the truncated game reached, features included.

    With ``hidden/w = 0`` the network sees only the features, which are known for
    the reached state: its mode, ``turn = max_turns``, the royale phase and the
    snakes still alive (the non-terminal ones of a truncated step).
    """
    cfg = pm.ModesConfig(
        game_modes=("duel", "royale"), num_envs=6, num_steps=30, max_turns=12, **TINY
    )
    groups = pm.make_groups(cfg)
    runner = pm.init_runner(cfg, groups, jax.random.key(2))
    params = _with_random_feat(runner.params, jax.random.key(3))
    k_b, k_v = jax.random.split(jax.random.key(4))
    params["hidden"] = {
        "w": jnp.zeros_like(params["hidden"]["w"]),
        "b": jax.random.normal(k_b, (16,)),
    }
    params["value"] = {**params["value"], "w": jax.random.normal(k_v, (16, 1))}
    _, trajs, metrics = jax.jit(pm.make_rollout(cfg, groups))(
        params, runner.env_states, jax.random.key(5)
    )
    assert (np.asarray(metrics["truncated"]) > 0).all()
    for g, traj in zip(groups, trajs, strict=True):
        terminal, done = np.asarray(traj.terminal), np.asarray(traj.done)
        truncated = done & ~terminal.all(axis=-1)  # survivors: the rules did not end it
        boot = np.asarray(traj.boot_values)
        assert truncated.any() and not boot[~truncated].any()
        template = g.sim.init_state(jax.random.key(0))
        for t, e in zip(*np.nonzero(truncated), strict=True):
            reached = template._replace(turn=jnp.int32(12), alive=jnp.asarray(~terminal[t, e]))
            feats = modes.mode_features(reached, g.game)
            boards = jnp.zeros((g.num_snakes, 21, 21, 13))
            expected = pm.network(params, boards, feats, cfg)[1]
            np.testing.assert_allclose(boot[t, e], expected, rtol=1e-5, atol=1e-6)


# --- (4) Checkpoints, --init-from and the registry policy ---------------------------------


def test_config_params_and_runner_round_trip(tmp_path):
    run_dir = str(tmp_path)
    cfg = pm.ModesConfig(
        game_modes=("royale", "duel"), mode_snakes=("royale=3",), mode_weights=(2.0, 1.0),
        init_from="x", num_envs=2, num_steps=4, run_dir=run_dir, **TINY,
    )  # fmt: skip
    pm.save_config(run_dir, cfg)
    assert json.loads((tmp_path / "config.json").read_text())["algorithm"] == "ppo_modes"
    assert pm.load_config(run_dir) == cfg
    groups = pm.make_groups(cfg)
    runner = pm.init_runner(cfg, groups, jax.random.key(0))
    params = _with_random_feat(runner.params, jax.random.key(1))
    pm.save_checkpoint(run_dir, params)
    pm.save_checkpoint(run_dir, params, pm.snapshot_name(4096))
    _equal_trees(pm.load_params(run_dir), params)
    _equal_trees(pm.load_params(run_dir, cfg, "params-4096.npz"), params)
    pm.save_runner(run_dir, runner, 7)
    loaded, samples = pm.load_runner(run_dir, runner)
    assert samples == 7
    _equal_trees(loaded._replace(key=None), runner._replace(key=None))
    other = pm.init_runner(dataclasses.replace(cfg, num_envs=4), groups, jax.random.key(0))
    with pytest.raises(ValueError, match="does not match"):
        pm.load_runner(run_dir, other)
    # A ppo.py run is not a ppo_modes run.
    ppo.save_config(str(tmp_path), ppo.PPOConfig(**TINY))
    with pytest.raises(ValueError, match="not a ppo_modes run"):
        pm.load_config(run_dir)


def _ppo_run(run_dir: Path, **kw) -> dict:
    """A ppo.py checkpoint (random parameters) in ``run_dir``; returns its parameters."""
    pcfg = ppo.PPOConfig(**{**TINY, **kw})
    params = ppo.init_network(jax.random.key(7), pcfg)
    run_dir.mkdir()
    ppo.save_config(str(run_dir), pcfg)
    ppo.save_checkpoint(str(run_dir), params)
    return params


def test_init_from_a_ppo_run(tmp_path):
    source = _ppo_run(tmp_path / "ppo")
    cfg = pm.ModesConfig(**TINY)
    params = pm.init_network(jax.random.key(0), cfg)
    out, copied = pm.init_from_run(params, cfg, str(tmp_path / "ppo"))
    assert sorted(copied) == sorted(f"{k}/{n}" for k in source for n in source[k])
    _equal_trees({k: v for k, v in out.items() if k != "feat"}, source)
    assert not out["feat"]["w"].any()  # no features in the source: still zero
    obs = jax.random.uniform(jax.random.key(1), (5, 21, 21, 13))
    feats = jax.random.uniform(jax.random.key(2), (5, modes.NUM_FEATURES))
    np.testing.assert_array_equal(
        pm.network(out, obs, feats, cfg)[0], ppo.network(source, obs, ppo.PPOConfig(**TINY))[0]
    )
    # From a ppo_modes run the feature layer comes along.
    modes_dir = tmp_path / "modes"
    modes_dir.mkdir()
    trained = _with_random_feat(out, jax.random.key(3))
    pm.save_config(str(modes_dir), cfg)
    pm.save_checkpoint(str(modes_dir), trained)
    again, copied = pm.init_from_run(params, cfg, str(modes_dir))
    assert "feat/w" in copied
    _equal_trees(again, trained)
    # Other network sizes and other algorithms are refused.
    _ppo_run(tmp_path / "wide", hidden=32)
    with pytest.raises(ValueError, match="hidden 32 != 16"):
        pm.init_from_run(params, cfg, str(tmp_path / "wide"))
    (tmp_path / "az").mkdir()
    (tmp_path / "az" / "config.json").write_text(json.dumps({"algorithm": "az", "hidden": 16}))
    with pytest.raises(ValueError, match="not a ppo.py or ppo_modes run"):
        pm.init_from_run(params, cfg, str(tmp_path / "az"))


def test_make_policy_logits_equal_the_training_network(four_modes):
    """The registry's path (its env's observations, make_policy) gives the trainer's logits."""
    from slinky.agents import make_env

    cfg, groups, params, states, _, _ = four_modes
    for g, s in zip(groups, states, strict=True):
        boards, feats, mask = pm.group_inputs(cfg, g, s)
        train_logits = pm.network(params, boards, feats, cfg)[0]
        play_env = make_env(g.game, True)  # what slinky.agents gives an agent with needs_obs
        ts = TimeStep(
            obs=jax.vmap(play_env.observe)(s),
            reward=jnp.zeros((3, g.num_snakes)),
            done=s.done,
            truncated=jnp.zeros((3,), bool),
            alive=s.alive,
            action_mask=jax.vmap(play_env.action_mask)(s),
        )
        logits_fn = pm.make_logits_fn(params, cfg, g.game)
        play_logits = jax.jit(jax.vmap(logits_fn))(s, ts.obs)
        np.testing.assert_allclose(play_logits, train_logits, rtol=1e-6, atol=1e-6)
        greedy = jax.vmap(pm.make_policy(params, cfg, g.game, greedy=True))
        keys = jax.random.split(jax.random.key(0), 3)
        expected = jnp.argmax(jnp.where(mask, train_logits, -jnp.inf), -1)
        np.testing.assert_array_equal(greedy(keys, s, ts), expected)
        sampled = jax.vmap(pm.make_policy(params, cfg, g.game, greedy=False))(keys, s, ts)
        assert bool(jnp.take_along_axis(ts.action_mask, sampled[..., None], -1).all())
        # Told another mode, the features (and so the logits) change, the boards do not.
        other = "duel" if g.name != "duel" else "standard"
        told = jax.vmap(pm.make_logits_fn(params, cfg, g.game, other))(s, ts.obs)
        assert not np.allclose(told, play_logits)
        told_inputs = functools.partial(pm.policy_inputs, config=g.game, cfg=cfg, game_mode=other)
        told_boards, _ = jax.vmap(told_inputs)(ts.obs, s)
        np.testing.assert_array_equal(told_boards, boards)


def test_run_evals_records_match_play_table():
    """Each eval record is the network's seat of a direct play_table call in its own mode.

    The network is told the mode it plays (a non-zero feat/w, so that matters), sits
    in one seat against the opponent in all others, with the documented key.
    """
    from slinky.agents import make_agent

    cfg = pm.ModesConfig(game_modes=("royale", "duel"), max_turns=200, **TINY)
    groups = pm.make_groups(cfg)
    params = _with_random_feat(pm.init_network(jax.random.key(0), cfg), jax.random.key(1))
    key = jax.random.key(11)
    written = []
    records = pm.run_evals(
        params, cfg, groups, key, 8, ("random_legal",), ("greedy",), 77, written.append
    )
    assert written == records
    assert [(r["game_mode"], r["num_snakes"]) for r in records] == [("royale", 4), ("duel", 2)]
    for g, rec in zip(groups, records, strict=True):
        policies = (
            pm.make_policy(params, cfg, g.game, g.name, greedy=True),
            make_agent("random_legal", g.game).policy,
        )
        lineup = (0,) + (1,) * (g.num_snakes - 1)
        k = jax.random.fold_in(key, g.mode_id)
        r = play_table(g.env, policies, lineup, k, 8, max_turns=cfg.max_turns)
        assert rec["type"] == "eval" and rec["env_steps"] == 77
        assert (rec["opponent"], rec["mode"], rec["num_games"]) == ("random_legal", "greedy", 8)
        assert rec["score"] == r.share[0] and rec["score_ci95"] == r.share_ci95[0]
        assert rec["score_wilson"] == list(r.share_wilson[0])
        assert rec["placement"] == r.placement[0]
        assert rec["placement_ci95"] == r.placement_ci95[0]
        assert (rec["wins"], rec["draws"], rec["losses"]) == (r.wins[0], r.ties[0], r.losses[0])
        assert rec["mean_turns"] == r.mean_turns and rec["truncated"] == r.truncated
        assert rec["deaths"] == r.deaths[0]
    # The record reads the network's entries (policy 0), whatever the values.
    fake = r._replace(share=(0.125, 0.875), placement=(1.25, 1.75), wins=(1, 7), losses=(7, 1))
    rec = pm.eval_record(fake, groups[1], "random_legal", "greedy", 5, 0.0)
    assert (rec["score"], rec["placement"], rec["wins"], rec["losses"]) == (0.125, 1.25, 1, 7)


def test_chunk_record_weights_losses_and_takes_the_headline_mode():
    """Top-level losses are the mode_weights mean; game statistics are the first mode's."""
    cfg = pm.ModesConfig(game_modes=("royale", "duel"), mode_weights=(3.0, 1.0), num_envs=2)
    n_upd = 4
    per_update = {  # per-mode means per gradient step
        "mode_loss": [0.5, -0.3], "pg_loss": [0.1, -0.2], "v_loss": [0.04, 0.2],
        "entropy": [1.2, 0.6], "approx_kl": [0.01, 0.03], "clip_frac": [0.1, 0.3],
    }  # fmt: skip
    m = {k: np.array(v) * n_upd for k, v in per_update.items()}
    m.update(
        updates=n_upd, env_steps=48, total_updates=96, lr=1e-4, loss=0.3 * n_upd,
        grad_norm=0.25 * n_upd, games=np.array([3, 5]), game_turns=np.array([90, 50]),
        agent_samples=np.array([70, 30]), draws=np.array([1, 2]), truncated=np.array([0, 1]),
        deaths=np.arange(2 * len(pm.Cause)).reshape(2, -1), explained_var=np.array([0.2, 0.4]),
        value_mean=np.array([0.05, -0.1]), return_var=np.array([0.1, 0.3]),
        value_weight=np.array([1.2, 0.4]),
    )  # fmt: skip
    r = pm.chunk_record(m, 2.0, 24, cfg)
    for k, (royale, duel) in per_update.items():
        assert r["game_modes"]["royale"][k if k != "mode_loss" else "loss"] == pytest.approx(royale)
        if k != "mode_loss":
            assert r[k] == pytest.approx(0.75 * royale + 0.25 * duel), k
    assert r["loss"] == pytest.approx(0.3) and r["grad_norm"] == pytest.approx(0.25)
    for k in pm.HEADLINE_FIELDS:
        assert r[k] == r["game_modes"]["royale"][k], k
    assert (r["games"], r["mean_game_len"], r["draws"]) == (3, 30.0, 1)
    assert r["agent_samples"] == 100 and r["env_steps_all"] == 96
    assert r["game_modes"]["duel"]["value_weight"] == 0.4
    assert r["game_modes"]["duel"]["valid_frac"] == pytest.approx(30 / (24 * 2))


# --- (5) End to end --------------------------------------------------------------------


def _argv(run_dir, *extra):
    return [
        "--num-envs", "4", "--num-steps", "8", "--total-env-steps", "64", "--log-every", "32",
        "--update-epochs", "2", "--conv-channels", "4", "--conv-strides", "2", "--hidden", "16",
        "--eval-every", "64", "--eval-games", "8", "--eval-opponents", "random_legal",
        "--snapshot-at", "32", "--run-dir", run_dir, *extra,
    ]  # fmt: skip


MODE_FIELDS = {
    "num_snakes", "games", "mean_game_len", "draws", "truncated", "deaths", "agent_samples",
    "valid_frac", "loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac",
    "explained_var", "value_mean", "return_var", "value_weight",
}  # fmt: skip


def test_tiny_four_mode_run(tmp_path, capsys):
    """All four modes, warm-started from a ppo.py run: records, evals, snapshots, --eval-only."""
    source = _ppo_run(tmp_path / "ppo")
    run_dir = str(tmp_path / "run")
    cfg, _ = pm.parse_args(_argv(run_dir, "--init-from", str(tmp_path / "ppo")))
    out_dir, params = pm.train(cfg)
    assert out_dir == run_dir
    told = [x for x in capsys.readouterr().out.splitlines() if x.startswith("initialised")]
    assert len(told) == 1 and told[0].endswith(f"from {tmp_path / 'ppo'}")
    assert "conv0/w" in told[0] and "value/b" in told[0] and "feat" not in told[0]
    assert not np.array_equal(params["hidden"]["w"], source["hidden"]["w"])  # trained on

    records = [json.loads(x) for x in Path(run_dir, "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    assert [r["env_steps"] for r in train] == [32, 64]
    assert [r["env_steps_all"] for r in train] == [128, 256]
    assert train[-1]["total_updates"] == 2 * 2 * 8  # iterations x epochs x minibatches
    assert train[0]["lr"] > train[-1]["lr"] == pytest.approx(0.0, abs=1e-12)
    for r in train:
        assert list(r["game_modes"]) == list(modes.MODE_NAMES)
        for name, m in r["game_modes"].items():
            assert set(m) == MODE_FIELDS, name
            assert m["num_snakes"] == modes.DEFAULT_SNAKES[name]
            assert 0 < m["valid_frac"] <= 1 and m["agent_samples"] > 0
            for key in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac"):
                assert np.isfinite(m[key]), (name, key)
        head = r["game_modes"]["standard"]
        for key in pm.HEADLINE_FIELDS:
            assert r[key] == head[key], key
        assert r["agent_samples"] == sum(m["agent_samples"] for m in r["game_modes"].values())
        weighted = np.mean([m["pg_loss"] for m in r["game_modes"].values()])
        assert r["pg_loss"] == pytest.approx(weighted, rel=1e-6, abs=1e-12)
        assert r["loss"] == pytest.approx(np.mean([m["loss"] for m in r["game_modes"].values()]))
    assert sum(r["game_modes"]["constrictor"]["games"] for r in train) > 0
    assert [(e["game_mode"], e["num_snakes"]) for e in evals] == [
        (m, modes.DEFAULT_SNAKES[m]) for m in modes.MODE_NAMES
    ]
    for e in evals:
        assert e["env_steps"] == 64 and e["opponent"] == "random_legal" and e["mode"] == "greedy"
        assert e["num_games"] == 8 and e["wins"] + e["draws"] + e["losses"] == 8
        assert 0 <= e["score"] <= 1 and 1 <= e["placement"] <= e["num_snakes"]
        assert set(e["deaths"]) >= {"hazard", "head-collision"}

    files = sorted(os.listdir(run_dir))
    assert files == [
        "config.json", "metrics.jsonl", "params-32.npz", "params-64.npz", "params.npz",
        "runner.npz",
    ]  # fmt: skip
    _equal_trees(pm.load_params(run_dir), params)
    _equal_trees(pm.load_params(run_dir, name="params-64.npz"), params)
    first = pm.load_params(run_dir, name="params-32.npz")  # --snapshot-at 32: after chunk 1
    assert not np.array_equal(first["hidden"]["w"], params["hidden"]["w"])
    assert not np.array_equal(first["hidden"]["w"], source["hidden"]["w"])
    assert pm.load_config(run_dir) == cfg

    capsys.readouterr()
    pm.main(["--eval-only", run_dir, "--eval-games", "6", "--eval-opponents", "random_legal"])
    lines = capsys.readouterr().out.strip().splitlines()
    results = [json.loads(x) for x in lines if x.startswith("{")]
    assert [r["game_mode"] for r in results] == list(modes.MODE_NAMES)
    assert all(r["wins"] + r["draws"] + r["losses"] == 6 for r in results)


def test_resume_continues_exactly(tmp_path, monkeypatch):
    """Two modes (one royale), interrupted after two of four chunks, then --resume."""

    def argv(run_dir):
        return [
            "--game-modes", "duel,royale", "--num-envs", "2", "--num-steps", "8",
            "--total-env-steps", "64", "--log-every", "16", "--checkpoint-every", "16",
            "--eval-every", "32", "--eval-games", "0", "--update-epochs", "1",
            "--num-minibatches", "2", "--conv-channels", "4", "--conv-strides", "2",
            "--hidden", "16", "--run-dir", run_dir,
        ]  # fmt: skip

    full_dir, cut_dir = str(tmp_path / "full"), str(tmp_path / "cut")
    _, full = pm.train(pm.parse_args(argv(full_dir))[0])

    calls = []
    chunk_record = pm.chunk_record

    def interrupted(*args):
        calls.append(None)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return chunk_record(*args)

    monkeypatch.setattr(pm, "chunk_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pm.train(pm.parse_args(argv(cut_dir))[0])
    monkeypatch.setattr(pm, "chunk_record", chunk_record)
    pm.main(["--resume", cut_dir])

    _equal_trees(pm.load_params(cut_dir), full)
    assert sorted(os.listdir(cut_dir)) == sorted(os.listdir(full_dir))
    _equal_trees(
        pm.load_params(cut_dir, name="params-32.npz"),
        pm.load_params(full_dir, name="params-32.npz"),
    )

    def records(run_dir):
        lines = Path(run_dir, "metrics.jsonl").read_text().splitlines()
        drop = ("seconds", "elapsed", "env_steps_per_s")
        return [{k: v for k, v in json.loads(x).items() if k not in drop} for x in lines]

    assert records(cut_dir) == records(full_dir)  # no duplicated or missing records


def test_bad_sources_fail_before_touching_the_run_dir(tmp_path):
    """An unknown eval opponent or a bad --init-from leaves an old run's files alone."""
    old = tmp_path / "old"
    old.mkdir()
    files = {"config.json": "{}", "metrics.jsonl": "{}\n", "runner.npz": "r", "params.npz": "p"}
    for name, text in files.items():
        (old / name).write_text(text)
    cfg = pm.ModesConfig(
        eval_opponents=("alphasnake",), run_dir=str(old), num_envs=2, num_steps=8, **TINY
    )
    with pytest.raises(ValueError, match="unknown agent"):
        pm.train(cfg)
    _ppo_run(tmp_path / "wide", hidden=32)
    bad = dataclasses.replace(
        cfg, eval_opponents=("random_legal",), init_from=str(tmp_path / "wide")
    )
    with pytest.raises(ValueError, match="hidden"):
        pm.train(bad)
    assert {p.name: p.read_text() for p in old.iterdir()} == files
