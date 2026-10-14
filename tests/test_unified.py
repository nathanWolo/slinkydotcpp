"""Tests for the multi-mode trainers on :mod:`slinky.unified` and their agents.

``baselines/{reinforce,dqn,rainbow,az}_modes.py``: tiny runs of every trainer (snapshots,
checkpoints, an exact resume, the STOP file and the cooldown), the shared torso and inputs,
placement values in the search, ``ppo_modes``' cooldown, and the registry names.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import agents, heuristic, mcts, modes, unified
from slinky.env import BattlesnakeEnv, placement_reward
from slinky.types import State

optax = pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, file_name: str):
    """Import ``baselines/<file_name>`` from its path (``baselines/`` is not a package)."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "baselines" / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rm = _load("baselines_reinforce_modes", "reinforce_modes.py")
rf = _load("baselines_reinforce", "reinforce.py")
dq = _load("baselines_dqn_modes", "dqn_modes.py")
rb = _load("baselines_rainbow_modes", "rainbow_modes.py")
azm = _load("baselines_az_modes", "az_modes.py")
pm = _load("baselines_ppo_modes", "ppo_modes.py")

TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16, max_turns=40, eval_games=0)
# Per trainer: a config small enough to run a few chunks in seconds.
TINY_RUNS = {
    "reinforce_modes": (
        rm,
        dict(mode_envs=("standard=2", "duel=2", "royale=2", "constrictor=2"), block_turns=20,
             updates_per_log=1),
    ),
    "dqn_modes": (
        dq,
        dict(num_envs=4, buffer_capacity=64, learning_starts=16, batch_size=8, log_every=32,
             eps_decay_steps=64),
    ),
    "rainbow_modes": (
        rb,
        dict(num_envs=4, buffer_capacity=64, learning_starts=16, batch_size=8, log_every=32),
    ),
    "az_modes": (
        azm,
        dict(num_envs=2, num_steps=8, num_simulations=4, buffer_capacity=64, learning_starts=16,
             batch_size=8, log_every=16),
    ),
}  # fmt: skip


def _tiny(kind: str, tmp_path: Path, **kw):
    module, settings = TINY_RUNS[kind]
    run_dir = str(tmp_path / kind)
    cfg = module.Config(
        **TINY, **settings, snapshot_every=32, checkpoint_every=32, run_dir=run_dir, **kw
    )
    return module, cfg


def _records(run_dir: str) -> list[dict]:
    lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def _strip(record: dict) -> dict:
    """A train record without its timings."""
    drop = ("seconds", "elapsed", "env_steps_per_s")
    return {k: v for k, v in record.items() if k not in drop}


# --- Shared pieces -----------------------------------------------------------------------


def test_settings_checks_and_sizes():
    with pytest.raises(ValueError, match="compute_dtype"):
        rm.Config(compute_dtype="float16")
    with pytest.raises(ValueError, match="cooldown_from"):
        dq.Config(cooldown_from=10**9)
    with pytest.raises(ValueError, match="mode_envs"):
        rm.Config(mode_envs=("duel",))
    assert rm.Config(mode_envs=("duel=3",)).envs == (8, 3, 8, 8)
    for size, (channels, hidden) in unified.SIZES.items():
        cfg = dq.Config(conv_channels=channels, hidden=hidden)
        n = unified.count_params(unified.init_torso(jax.random.key(0), cfg))
        assert n > 0, size
    s, m, big = (
        unified.count_params(azm.init_network(jax.random.key(0), azm.Config(
            conv_channels=c, hidden=h)))
        for c, h in unified.SIZES.values()
    )  # fmt: skip
    assert s < m < big
    assert m == 314_021  # ppo_modes.py's default network with its feat/w


def test_lr_factor_is_constant_then_cools_down():
    cfg = dq.Config(total_env_steps=1000)
    assert float(unified.lr_factor(jnp.int32(900), cfg)) == 1.0
    cool = dataclasses.replace(cfg, cooldown_from=600)
    got = [float(unified.lr_factor(jnp.int32(x), cool)) for x in (0, 600, 800, 1000, 1200)]
    np.testing.assert_allclose(got, [1.0, 1.0, 0.5, 0.0, 0.0])


def test_bfloat16_torso_close_to_float32():
    cfg = rm.Config(**TINY)
    bf16 = dataclasses.replace(cfg, compute_dtype="bfloat16")
    keys = iter(jax.random.split(jax.random.key(1), 16))
    params = jax.tree.map(
        lambda x: x + 0.05 * jax.random.normal(next(keys), x.shape),
        rm.init_network(jax.random.key(0), cfg),
    )
    group = unified.make_groups(cfg)[0]
    states = jax.vmap(group.sim.init_state)(jax.random.split(jax.random.key(2), 4))
    boards, feats, _ = unified.group_inputs(cfg, group, states)
    jaxpr = str(jax.make_jaxpr(lambda p: rm.network(p, boards, feats, cfg))(params))
    assert "convert_element_type" not in jaxpr  # float32 traces no casts at all
    logits, value = rm.network(params, boards, feats, bf16)
    ref_logits, ref_value = rm.network(params, boards, feats, cfg)
    assert logits.dtype == value.dtype == jnp.float32
    np.testing.assert_allclose(logits, ref_logits, atol=0.02 * float(jnp.abs(ref_logits).max()))
    np.testing.assert_allclose(value, ref_value, atol=0.02 * float(jnp.abs(ref_value).max()))


def test_join_and_split_are_inverse():
    cfg = dq.Config(**TINY)
    groups = unified.make_groups(cfg)
    envs = (3, 2, 1, 2)
    xs = [jnp.arange(e * g.num_snakes * 5).reshape(e, g.num_snakes, 5) + 100 * i
          for i, (g, e) in enumerate(zip(groups, envs, strict=True))]  # fmt: skip
    joined = unified.join(xs)
    assert joined.shape == (sum(e * g.num_snakes for g, e in zip(groups, envs, strict=True)), 5)
    for x, y in zip(xs, unified.split(joined, groups, envs), strict=True):
        np.testing.assert_array_equal(x, y)


# --- REINFORCE's gradient ---------------------------------------------------------------------


def test_reinforce_modes_duel_gradient_is_reinforce_py_s():
    """Playing only duels and told nothing, reinforce_modes' loss is reinforce.py's: the same
    gradient and sums on the same games, with a first-update norm far below the guard (the
    GPU campaign's first chunk logged a mean raw norm of 1.5e6)."""
    net = dict(conv_channels=(4, 8), conv_strides=(2, 2), hidden=16, max_turns=60, block_turns=20)
    cfg = rm.Config(
        **net, game_modes=("duel",), mode_envs=("duel=4",), features="none", reward="win_loss",
        eval_games=0,
    )  # fmt: skip
    ref_cfg = rf.ReinforceConfig(**net, num_envs=4)
    params = rf.init_network(jax.random.key(0), ref_cfg)
    assert jax.tree.structure(params) == jax.tree.structure(rm.init_network(jax.random.key(0), cfg))
    env, sim = rf.make_envs(ref_cfg)
    games = jax.jit(rf.make_play(ref_cfg, env, sim))(params, jax.random.key(1))
    want, want_sums = jax.jit(lambda p, g: rf.batch_gradient(p, g, ref_cfg, env))(params, games)
    groups = unified.make_groups(cfg)
    same = rm.Games(*(getattr(games, f) for f in rm.Games._fields))
    got, got_sums = jax.jit(lambda p, g, t: rm.batch_gradient(p, (g,), t, cfg, groups))(
        params, same, games.turns
    )
    norm = float(optax.tree.norm(want))
    assert 0.0 < norm < rm.MAX_RAW_GRAD_NORM
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6 * norm)
    for k in rm.SUMS:
        np.testing.assert_allclose(got_sums[k][0], want_sums[k], rtol=1e-4, atol=1e-5)
    assert float(got_sums["samples"][0]) == float(want_sums["samples"])


@pytest.mark.parametrize("fault", [1e12, float("nan")])
def test_reinforce_modes_skips_a_faulty_gradient(fault, monkeypatch):
    """A raw gradient that is not finite or absurdly large changes neither the parameters nor
    Adam's state, and the record says so; a healthy one is taken."""
    cfg = rm.Config(**TINY, **TINY_RUNS["reinforce_modes"][1])
    groups = unified.make_groups(cfg)
    runner = rm.init_runner(cfg, groups, jax.random.key(0))
    healthy, m = jax.jit(rm.make_train_chunk(cfg, groups))(runner)
    assert int(m["skipped"]) == 0 and float(m["grad_norm_max"]) < rm.MAX_RAW_GRAD_NORM
    assert not np.array_equal(healthy.params["value"]["b"], runner.params["value"]["b"])

    real = rm.batch_gradient

    def faulty(*args):
        grads, sums = real(*args)
        return jax.tree.map(lambda g: g * fault, grads), sums

    monkeypatch.setattr(rm, "batch_gradient", faulty)
    after, m = jax.jit(rm.make_train_chunk(cfg, groups))(runner)
    jax.tree.map(
        np.testing.assert_array_equal,
        (after.params, after.opt_state), (runner.params, runner.opt_state),
    )  # fmt: skip
    assert int(after.env_steps) > 0  # the games were played all the same
    record = rm.chunk_record(jax.device_get(m), 1.0, 1, cfg)
    assert record["skipped_updates"] == cfg.updates_per_log and record["grad_norm"] == 0.0
    assert (record["grad_norm_max"] is None) == bool(np.isnan(fault))
    assert all(p["value_std"] >= 0.0 for p in record["game_modes"].values())


# --- Placement values -----------------------------------------------------------------------


def _state(alive, elim_turn, n=4) -> tuple[State, object]:
    game = modes.mode_config("standard" if n > 2 else "duel")
    s = BattlesnakeEnv(game, obs=None).init_state(jax.random.key(0))
    return s._replace(
        alive=jnp.array(alive), elim_turn=jnp.array(elim_turn, s.elim_turn.dtype),
        done=jnp.array(sum(alive) <= 1),
    ), game  # fmt: skip


def test_placement_values():
    # Places 1 to 4: +1, +1/3, -1/3, -1.
    s, game = _state([True, False, False, False], [0, 30, 20, 10])
    np.testing.assert_allclose(mcts.placement_values(s, game), [1, 1 / 3, -1 / 3, -1], atol=1e-6)
    # Two out on the same turn are tied; a running game's dead snakes are already exact.
    s, game = _state([True, True, False, False], [0, 0, 20, 20])
    np.testing.assert_allclose(mcts.placement_values(s, game)[2:], [-2 / 3, -2 / 3], atol=1e-6)
    # The last two out together: each outlasted two, and the contempt scales their tie.
    s, game = _state([False, False, False, False], [30, 30, 20, 10])
    v = mcts.placement_values(s, game, draw_value=-0.5)
    np.testing.assert_allclose(v, [2 / 3 - 0.5 / 3, 2 / 3 - 0.5 / 3, -1 / 3, -1], atol=1e-6)
    np.testing.assert_allclose(sum(mcts.placement_values(s, game)), 0.0, atol=1e-6)
    # In a duel: heuristic.terminal_values.
    for alive, elim in (([True, False], [0, 9]), ([False, False], [9, 9]), ([True, True], [0, 0])):
        s, game = _state(alive, elim, n=2)
        for draw in (0.0, -0.5):
            np.testing.assert_allclose(
                mcts.placement_values(s, game, draw), heuristic.terminal_values(s, game, draw)
            )


def test_placement_values_sum_the_placement_reward():
    game = modes.mode_config("standard", None, 60)
    env = BattlesnakeEnv(game, obs=None, reward_fn=placement_reward)
    for seed in range(4):
        key = jax.random.key(seed)
        s = env.init_state(key)
        total = jnp.zeros(4)
        for t in range(60):
            k = jax.random.fold_in(key, t)
            mask = env.action_mask(s)
            a = jax.random.categorical(k, jnp.where(mask, 0.0, -jnp.inf)).astype(jnp.int32)
            s, ts = env.step(k, s, a)
            total = total + ts.reward
            if bool(s.done):
                break
        np.testing.assert_allclose(total, mcts.placement_values(s, game), atol=1e-5)


def test_placement_search_runs_in_four_snake_games():
    cfg = azm.Config(**TINY, num_envs=2, num_steps=4, num_simulations=8)
    params = azm.init_network(jax.random.key(0), cfg)
    group = unified.make_groups(cfg)[0]
    state = group.sim.init_state(jax.random.key(1))
    out = mcts.search(
        jax.random.key(2), state, group.sim, azm.agent_search(cfg, 8),
        azm.make_evaluator(params, cfg, group),
    )  # fmt: skip
    assert out.action.shape == (4,) and bool(jnp.all(jnp.abs(out.value) <= 1.0))
    # value="win_loss" is left out of every older agent's identity.
    assert "value" not in agents._mcts_fields(mcts.MCTSConfig())
    assert agents._mcts_fields(azm.agent_search(cfg, 8))["value"] == "placement"


# --- Tiny runs -------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(TINY_RUNS))
def test_tiny_run_snapshots_resume_stop_and_cooldown(kind, tmp_path):
    module, cfg = _tiny(kind, tmp_path, total_env_steps=96)
    run_dir, params = unified.train(module, cfg)
    full = _records(run_dir)
    assert full[-1]["env_steps"] >= 96
    names = {p.name for p in Path(run_dir).iterdir()}
    assert {"config.json", "metrics.jsonl", "params.npz", "runner.npz"} <= names
    snaps = sorted(int(n[7:-4]) for n in names if n.startswith("params-"))
    assert len(snaps) >= 2 and snaps[-1] == full[-1]["env_steps"]
    for name, mode in zip(cfg.game_modes, full[-1]["game_modes"], strict=True):
        assert name == mode
    # The snapshot loads into the registry's agent, which plays every mode.
    for mode in modes.MODE_NAMES:
        game = modes.mode_config(mode, None, 40)
        run = os.path.relpath(run_dir)  # names split on ":", so no drive letter
        agent = agents.make_agent(f"{kind}:{run}:snapshot={snaps[0]}", game)
        assert agent.needs_obs

    # Resume after the first chunk: the same records, bit for bit.
    module2, cfg2 = _tiny(kind, tmp_path / "again", total_env_steps=full[0]["env_steps"])
    run2, _ = unified.train(module2, cfg2)
    cont = dataclasses.replace(cfg2, total_env_steps=96)
    unified.train(module2, cont, resume=True)
    again = _records(run2)
    assert [_strip(r) for r in again] == [_strip(r) for r in full]

    # STOP ends a resumed run after one chunk; --cooldown then decays the rate to 0.
    longer = dataclasses.replace(cont, total_env_steps=10**6)
    (Path(run2) / "STOP").write_text("{}")
    unified.train(module2, longer, resume=True)
    assert (Path(run2) / "STOPPED").exists() and not (Path(run2) / "STOP").exists()
    steps = unified.runner_env_steps(run2)
    # (REINFORCE updates once per batch of whole games, about 70 steps here: a longer one.)
    length = 256 if kind == "reinforce_modes" else 64
    cooled = unified.cooldown_config(longer, steps, length)
    unified.train(module2, cooled, resume=True)
    lrs = [r["lr"] for r in _records(run2) if r["env_steps"] > steps]
    assert _records(run2)[-1]["env_steps"] >= steps + length
    assert lrs == sorted(lrs, reverse=True) and lrs[-1] < 0.5 * cfg.lr


def test_registry_names_of_the_new_kinds():
    run = "runs/m"  # names split on ":", so no drive letter
    for kind in ("reinforce_modes", "dqn_modes", "rainbow_modes", "az_modes"):
        spec = agents.parse_agent(f"{kind}-sample:{run}:game_mode=duel:snapshot=64")
        assert spec.kind == kind and spec.greedy is False and spec.snapshot == 64
        assert agents.parse_agent(spec.name) == spec
        with pytest.raises(ValueError, match="unknown"):
            agents.parse_agent(f"{kind}:{run}:weights=1")
    spec = agents.parse_agent(f"az_modes:{run}:simulations=16")
    assert dict(spec.overrides) == {"simulations": 16} and spec.name.endswith(":simulations=16")
    with pytest.raises(ValueError, match="most visited"):
        agents.parse_agent(f"az_modes-sample:{run}:simulations=16")
    with pytest.raises(ValueError, match="unknown"):
        agents.parse_agent(f"dqn_modes:{run}:simulations=16")


def test_ppo_modes_cooldown_and_stop(tmp_path):
    base = dict(
        conv_channels=(4,), conv_strides=(2,), hidden=16, num_envs=2, num_steps=8,
        num_minibatches=2, update_epochs=1, max_turns=40, eval_games=0, anneal_lr=False,
        log_every=16, checkpoint_every=16,
    )  # fmt: skip
    with pytest.raises(ValueError, match="anneal_lr"):
        pm.ModesConfig(**{**base, "anneal_lr": True}, cooldown_from=0)
    cfg = pm.ModesConfig(**base, total_env_steps=64, cooldown_from=32)
    lr = pm.learning_rate(cfg)
    per_iter = cfg.update_epochs * cfg.num_minibatches
    assert float(lr(32 // 16 * per_iter)) == pytest.approx(cfg.lr)
    assert float(lr(pm.schedule(cfg).total_updates)) == pytest.approx(0.0, abs=1e-9)
    # The STOP file: a checkpoint, a snapshot, STOPPED, and the run ends.
    run = tmp_path / "pm"
    run.mkdir()
    first = pm.ModesConfig(**base, total_env_steps=16, run_dir=str(run))
    pm.train(first)
    (run / "STOP").write_text("{}")
    pm.train(dataclasses.replace(first, total_env_steps=10**4), resume=True)
    assert (run / "STOPPED").exists() and (run / "params-32.npz").exists()
    # The new field is left out of the agent's identity at its old value.
    assert "cooldown_from" not in agents._run_config_fields(first)
