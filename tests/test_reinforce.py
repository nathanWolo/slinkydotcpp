"""Tests for the self-play REINFORCE baseline (``baselines/reinforce.py``)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load_reinforce():
    """Import ``baselines/reinforce.py`` from its path (``baselines/`` is not a package)."""
    if "baselines_reinforce" in sys.modules:  # e.g. already loaded by slinky.agents
        return sys.modules["baselines_reinforce"]
    spec = importlib.util.spec_from_file_location(
        "baselines_reinforce", ROOT / "baselines" / "reinforce.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


rf = _load_reinforce()
# A tiny network so tests compile and run fast.
TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)
# Short games, some cut off by max_turns; blocks that don't divide max_turns.
SHORT = dict(max_turns=30, num_envs=6, block_turns=7, **TINY)


def test_discounted_returns_hand_computed():
    rewards = jnp.array([[0.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])  # [T=4, 2 snakes]
    g = rf.discounted_returns(rewards, 0.5)
    np.testing.assert_allclose(g[:, 0], [0.25, 0.5, 1.0, 0.0])
    np.testing.assert_allclose(g[:, 1], [-0.5, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(rf.discounted_returns(rewards, 1.0)[0], [1.0, -1.0])


@pytest.fixture(scope="module")
def short_games():
    cfg = rf.ReinforceConfig(**SHORT)
    env, sim = rf.make_envs(cfg)
    params = rf.init_network(jax.random.key(0), cfg)
    games = jax.jit(rf.make_play(cfg, env, sim))(params, jax.random.key(3))
    return cfg, env, params, jax.device_get(games)


def test_play_records_whole_games(short_games):
    cfg, _, _, g = short_games
    t_buf, e, n = g.valid.shape
    assert (t_buf, e, n) == (cfg.buffer_turns, cfg.num_envs, 2) and t_buf == 35
    length = np.asarray(g.final.turn)  # turns each game lasted
    assert int(g.turns) == length.max() and bool(np.all(g.final.done))
    assert bool(np.any(g.truncated)) and bool(np.any(~g.truncated))  # both kinds of ending
    assert np.all(length[g.truncated] == cfg.max_turns)
    for game in range(e):
        played = np.arange(t_buf) < length[game]
        # A snake moves while alive in a running game, and only then.
        np.testing.assert_array_equal(g.valid[:, game].any(-1), played)
        alive = np.asarray(g.states.alive[:, game]) & ~np.asarray(g.states.done[:, game])[:, None]
        np.testing.assert_array_equal(g.valid[:, game], alive & played[:, None])
        np.testing.assert_array_equal(g.states.turn[: length[game], game], np.arange(length[game]))
        assert not np.any(g.rewards[~played, game])
        # Win +1, loss -1, draw or cut off 0, paid on the snake's last move.
        outcome = g.rewards[:, game].sum(0)
        if g.truncated[game]:
            np.testing.assert_array_equal(outcome, 0)
        else:
            survivor = bool(g.final.alive[game].any())  # else a draw: both out at once
            expected = np.where(g.final.alive[game], 1, -1 if survivor else 0)
            np.testing.assert_array_equal(outcome, expected)
        returns = np.asarray(rf.discounted_returns(jnp.asarray(g.rewards), cfg.gamma))
        for snake in range(n):
            last = np.flatnonzero(g.valid[:, game, snake]).max()
            assert returns[0, game, snake] == pytest.approx(cfg.gamma**last * outcome[snake])


def full_batch_gradient(params, games, cfg, env):
    """The batch loss over every move at once (no blocks), differentiated directly."""
    obs = jax.vmap(jax.vmap(env.observe))(games.states)
    mask = jax.vmap(jax.vmap(env.action_mask))(games.states)
    returns = rf.discounted_returns(games.rewards, cfg.gamma)
    count = jnp.maximum(jnp.sum(games.valid), 1).astype(jnp.float32)
    args = (obs, mask, games.actions, returns, games.valid, count, cfg)
    return jax.grad(rf.block_loss, has_aux=True)(params, *args)


@pytest.mark.parametrize("baseline", rf.BASELINES)
def test_blocked_gradient_equals_the_full_batch_gradient(short_games, baseline):
    cfg, env, params, games = short_games
    cfg = dataclasses.replace(cfg, baseline=baseline)
    games = jax.tree.map(jnp.asarray, games)
    grads, sums = jax.jit(rf.batch_gradient, static_argnums=(2, 3))(params, games, cfg, env)
    expected, expected_sums = full_batch_gradient(params, games, cfg, env)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-7)
    for k, v in expected_sums.items():
        np.testing.assert_allclose(sums[k], v, rtol=2e-4, atol=1e-6)
    assert int(sums["samples"]) == int(games.valid.sum())
    # Plain REINFORCE never trains the value head; the baseline does.
    value_grad = float(jnp.abs(grads["value"]["w"]).sum())
    assert (value_grad == 0.0) == (baseline == "none")


def test_policy_gradient_raises_the_log_prob_of_rewarded_moves():
    """One sample, return +1, no entropy bonus: the step makes its move more likely."""
    cfg = rf.ReinforceConfig(baseline="none", ent_coef=0.0, **TINY)
    env, _ = rf.make_envs(cfg)
    params = rf.init_network(jax.random.key(0), cfg)
    _, ts = env.reset(jax.random.key(0))
    obs, mask = ts.obs[:1], ts.action_mask[:1]
    action = jnp.argmax(mask[0]).astype(jnp.int32)[None]
    args = (obs, mask, action, jnp.ones(1), jnp.ones(1, bool), jnp.float32(1.0), cfg)
    grads, _ = jax.grad(rf.block_loss, has_aux=True)(params, *args)
    stepped = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    def log_prob(p):
        return rf.masked_log_probs(rf.network(p, obs, cfg)[0], mask)[0, action[0]]

    assert float(log_prob(stepped)) > float(log_prob(params))


def test_policies_play_legal_moves():
    cfg = rf.ReinforceConfig(**TINY)
    env, _ = rf.make_envs(cfg)
    params = rf.init_network(jax.random.key(0), cfg)
    state, ts = env.reset(jax.random.key(0))
    greedy = rf.make_policy(params, cfg, greedy=True)
    sample = rf.make_policy(params, cfg, greedy=False)
    for policy in (greedy, sample):
        a = jax.jit(policy)(jax.random.key(1), state, ts)
        assert a.shape == (2,) and a.dtype == jnp.int32
        assert bool(jnp.take_along_axis(ts.action_mask, a[:, None], -1).all())
    logits = rf.network(params, ts.obs, cfg)[0]
    expected = jnp.argmax(jnp.where(ts.action_mask, logits, -jnp.inf), -1)
    np.testing.assert_array_equal(greedy(jax.random.key(2), state, ts), expected)


def test_config_checks():
    with pytest.raises(ValueError, match="unknown baseline"):
        rf.ReinforceConfig(baseline="mean")
    with pytest.raises(ValueError, match="eval_modes"):
        rf.ReinforceConfig(eval_modes=("argmax",))
    with pytest.raises(ValueError, match="block_turns must be >= 1"):
        rf.ReinforceConfig(block_turns=0)
    cfg = rf.ReinforceConfig(max_turns=1000, block_turns=64)
    assert (cfg.num_blocks, cfg.buffer_turns) == (16, 1024)


def test_load_config_refuses_other_runs(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"gae_lambda": 0.95}))  # a PPO run
    with pytest.raises(ValueError, match="not a REINFORCE run"):
        rf.load_config(str(tmp_path))


# --- End to end ---------------------------------------------------------------------


def _argv(run_dir, *extra):
    return [
        "--num-envs", "4", "--max-turns", "40", "--block-turns", "16",
        "--total-env-steps", "300", "--updates-per-log", "2",
        "--conv-channels", "4,8", "--conv-strides", "2,2", "--hidden", "16",
        "--eval-every", "100000", "--eval-games", "4", "--eval-opponents", "random_legal",
        "--run-dir", run_dir, *extra,
    ]  # fmt: skip


def _records(run_dir):
    lines = Path(run_dir, "metrics.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_tiny_training_run_checkpoints_and_eval_only(tmp_path, capsys):
    run_dir = str(tmp_path / "run")
    cfg, _ = rf.parse_args(_argv(run_dir, "--baseline", "none"))
    assert cfg.conv_channels == (4, 8) and cfg.baseline == "none"
    assert cfg.lr == rf.ReinforceConfig.lr
    out_dir, params = rf.train(cfg)
    assert out_dir == run_dir
    assert json.loads(Path(run_dir, "config.json").read_text())["algorithm"] == "reinforce"

    records = _records(run_dir)
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    steps = [r["env_steps"] for r in train]
    # Whole games: the run stops at the first chunk that reaches the budget.
    assert steps == sorted(steps) and steps[-1] >= 300 > (steps[-2] if len(steps) > 1 else 0)
    assert [r["total_updates"] for r in train] == [2 * (i + 1) for i in range(len(train))]
    assert all(r["games"] == 2 * 4 for r in train)
    assert all(r["v_loss"] is None and r["value_mean"] is None for r in train)  # no baseline
    for key in ("pg_loss", "entropy", "grad_norm", "return_mean", "busy"):
        assert all(np.isfinite(r[key]) for r in train), key
    assert all(0 < r["busy"] <= 1 and r["max_game_len"] <= 40 for r in train)
    assert sum(sum(r["deaths"].values()) for r in train) > 0  # some games are cut off
    assert [(e["env_steps"], e["mode"]) for e in evals] == [(steps[-1], "greedy")] + [
        (steps[-1], "sample")
    ]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))

    loaded = rf.load_params(run_dir)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert rf.load_config(run_dir) == cfg

    capsys.readouterr()
    rf.main(["--eval-only", run_dir, "--eval-games", "6", "--eval-opponents", "random_legal"])
    lines = capsys.readouterr().out.strip().splitlines()
    results = [json.loads(x) for x in lines if x.startswith("{")]
    assert [r["mode"] for r in results] == ["greedy", "sample"]
    assert all(r["wins"] + r["draws"] + r["losses"] == 6 for r in results)


def test_resume_continues_exactly(tmp_path, monkeypatch):
    def argv(run_dir):
        return _argv(
            run_dir, "--total-env-steps", "900", "--checkpoint-every", "150",
            "--eval-modes", "sample",
        )  # fmt: skip

    full_dir, cut_dir = str(tmp_path / "full"), str(tmp_path / "cut")
    _, full = rf.train(rf.parse_args(argv(full_dir))[0])
    assert len([r for r in _records(full_dir) if r["type"] == "train"]) >= 4

    # Interrupt the second run during its fourth chunk, after some full checkpoints.
    calls = []
    chunk_record = rf.chunk_record

    def interrupted(*args):
        calls.append(None)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return chunk_record(*args)

    monkeypatch.setattr(rf, "chunk_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rf.train(rf.parse_args(argv(cut_dir))[0])
    monkeypatch.setattr(rf, "chunk_record", chunk_record)
    rf.main(["--resume", cut_dir])

    resumed = rf.load_params(cut_dir)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True):
        np.testing.assert_array_equal(a, b)

    def records(run_dir):
        drop = ("seconds", "elapsed", "env_steps_per_s")
        return [{k: v for k, v in r.items() if k not in drop} for r in _records(run_dir)]

    assert records(cut_dir) == records(full_dir)  # no duplicated or missing records
