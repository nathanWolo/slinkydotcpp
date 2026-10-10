"""Tests for the self-play PPO baseline (``baselines/ppo.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.types import NUM_ACTIONS

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load_ppo():
    """Import ``baselines/ppo.py`` from its path (``baselines/`` is not a package)."""
    if "baselines_ppo" in sys.modules:  # e.g. already loaded by slinky.agents
        return sys.modules["baselines_ppo"]
    spec = importlib.util.spec_from_file_location("baselines_ppo", ROOT / "baselines" / "ppo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


ppo = _load_ppo()
# A tiny network so tests compile and run fast.
TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)


# --- (1) GAE ---------------------------------------------------------------------------


def reference_gae(rewards, values, valid, terminal, done, boot, last, gamma, lam):
    """A per-agent loop over explicit trajectories (numpy), for checking compute_gae."""
    t_len, n_env, n_agent = rewards.shape
    adv = np.zeros_like(rewards)
    for e in range(n_env):
        for i in range(n_agent):
            following = 0.0  # A_{t+1} of the same trajectory, 0 across a boundary
            for t in reversed(range(t_len)):
                if not valid[t, e, i]:
                    following = 0.0
                    continue
                if terminal[t, e, i]:
                    next_value, same = 0.0, False
                elif done[t, e]:  # truncated: the slot holds the next game at t+1
                    next_value, same = boot[t, e, i], False
                else:
                    next_value = last[e, i] if t == t_len - 1 else values[t + 1, e, i]
                    same = True
                delta = rewards[t, e, i] + gamma * next_value - values[t, e, i]
                adv[t, e, i] = delta + gamma * lam * (following if same else 0.0)
                following = adv[t, e, i]
    return adv


def test_gae_hand_computed_boundaries():
    """Terminal, truncation, autoreset and dead-snake boundaries, 3 snakes, 2 envs, 4 turns."""
    gamma, lam = 0.9, 0.5
    T, E, N = 4, 2, 3
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, (T, E, N)).astype(np.float32)
    last = rng.uniform(-1, 1, (E, N)).astype(np.float32)
    rewards = np.zeros((T, E, N), np.float32)
    valid = np.ones((T, E, N), bool)
    terminal = np.zeros((T, E, N), bool)
    done = np.zeros((T, E), bool)
    boot = np.zeros((T, E, N), np.float32)
    # Env 0: snake 2 dies at t=0 (-1) while the game goes on; its slot is dead at t=1.
    # The game is truncated at t=1 -> 2 (snakes 0 and 1 bootstrap from boot), and a new
    # game (all three alive) runs at t=2, 3 and continues past the rollout.
    rewards[0, 0, 2], terminal[0, 0, 2] = -1.0, True
    valid[1, 0, 2], terminal[1, 0, 2] = False, True
    done[1, 0] = True
    boot[1, 0] = [0.7, -0.3, 0.0]
    # Env 1: snakes 1 and 2 die at t=0 (both -1), snake 0 wins (+1): the game ends. In the
    # next game everyone dies on the same turn at t=2: a draw (0, terminal). A third game
    # starts at t=3.
    rewards[0, 1] = [1.0, -1.0, -1.0]
    terminal[0, 1] = terminal[2, 1] = True
    done[0, 1] = done[2, 1] = True
    # A dead snake's leftovers must never leak in: poison values/rewards where not valid.
    values[1, 0, 2], rewards[1, 0, 2] = 100.0, 100.0

    args = (rewards, values, valid, terminal, done, boot, last, gamma, lam)
    adv, ret = ppo.compute_gae(*map(jnp.asarray, args[:7]), gamma, lam)
    expected = reference_gae(*args)
    np.testing.assert_allclose(adv, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ret, np.where(valid, expected + values, 0.0), rtol=1e-5, atol=1e-6)

    v = values
    # Terminal steps: A = r - V, nothing from later steps (the next game's values).
    np.testing.assert_allclose(adv[0, 1], [1.0 - v[0, 1, 0], -1 - v[0, 1, 1], -1 - v[0, 1, 2]])
    np.testing.assert_allclose(adv[2, 1], -v[2, 1], rtol=1e-6)
    assert adv[0, 0, 2] == pytest.approx(-1.0 - v[0, 0, 2])
    assert adv[1, 0, 2] == 0.0 and ret[1, 0, 2] == 0.0  # dead: no sample
    # Truncation: bootstrap from boot, not from values[2] (the next game).
    a_trunc = 0.0 + gamma * 0.7 - v[1, 0, 0]
    assert adv[1, 0, 0] == pytest.approx(a_trunc, rel=1e-5)
    d0 = gamma * v[1, 0, 0] - v[0, 0, 0]
    assert adv[0, 0, 0] == pytest.approx(d0 + gamma * lam * a_trunc, rel=1e-5)
    # The last step bootstraps from last_values; the step before chains into it.
    a3 = gamma * last[0, 1] - v[3, 0, 1]
    assert adv[3, 0, 1] == pytest.approx(a3, rel=1e-5)
    assert adv[2, 0, 1] == pytest.approx(gamma * v[3, 0, 1] - v[2, 0, 1] + gamma * lam * a3)
    # Env 1's game starting at t=1 ends in the draw at t=2: A_1 chains into A_2 only.
    a1 = gamma * v[2, 1, 0] - v[1, 1, 0] + gamma * lam * (-v[2, 1, 0])
    assert adv[1, 1, 0] == pytest.approx(a1, rel=1e-5)


def test_gae_on_real_rollouts_gives_monte_carlo_returns():
    """With gamma = lambda = 1, the return of every sample is its own episode's outcome.

    That outcome is the snake's final reward (+1, -1 or 0) if its trajectory ends
    within the rollout, the value of the reached state if the game is cut off by
    max_turns, and V(s_T) otherwise, whatever the values in between. Short games
    (max_turns=12) make every kind of boundary common.
    """
    cfg = ppo.PPOConfig(num_envs=16, num_steps=40, max_turns=12, **TINY)
    env, sim = ppo.make_envs(cfg)
    runner = ppo.init_runner(cfg, sim, jax.random.key(0))
    states, traj, metrics = jax.jit(ppo.make_rollout(cfg, env, sim))(
        runner.params, runner.env_states, jax.random.key(1)
    )
    last = ppo.network(runner.params, jax.vmap(env.observe)(states), cfg)[1]
    _, ret = ppo.compute_gae(
        traj.rewards, traj.values, traj.valid, traj.terminal, traj.done, traj.boot_values,
        last, 1.0, 1.0,
    )  # fmt: skip
    tr = jax.device_get(traj)
    ret, last = np.asarray(ret), np.asarray(last)

    assert tr.valid.all()  # in a duel both snakes are alive in every running game
    assert tr.done.sum() == int(metrics["games"]) and int(metrics["truncated"]) > 0
    assert int(metrics["games"]) > int(metrics["truncated"])  # some games end by the rules
    assert np.take_along_axis(tr.mask, tr.actions[..., None], -1).all()  # legal moves only
    # Rewards only on terminal steps; bootstrap values only on truncated ones.
    assert not (tr.rewards != 0)[~tr.terminal].any()
    assert not (tr.boot_values != 0)[~tr.done].any()

    T, E, N = tr.rewards.shape
    expected = np.zeros((T, E, N))
    for e in range(E):
        for i in range(N):
            outcome = last[e, i]  # trajectory still running at the end of the rollout
            for t in reversed(range(T)):
                if tr.terminal[t, e, i]:
                    outcome = tr.rewards[t, e, i]
                elif tr.done[t, e]:
                    outcome = tr.boot_values[t, e, i]
                expected[t, e, i] = outcome
    np.testing.assert_allclose(ret, expected, rtol=1e-5, atol=1e-5)
    # Truncation bootstraps from the value of the state reached, not of the next game.
    truncated = tr.done & ~tr.terminal.any(-1)
    assert truncated.any() and (tr.boot_values[truncated] != 0).all()


# --- (2) The masked policy -------------------------------------------------------------


def test_masked_log_probs_entropy_and_sampling():
    rng = np.random.default_rng(0)
    mask = rng.random((500, NUM_ACTIONS)) < 0.6
    mask[np.arange(500), rng.integers(0, NUM_ACTIONS, 500)] = True  # never all-False
    mask[0] = [False, False, True, False]  # one legal move
    logits = jnp.asarray(rng.normal(0, 3, mask.shape), jnp.float32)
    mask = jnp.asarray(mask)

    logp = ppo.masked_log_probs(logits, mask)
    p = np.exp(np.asarray(logp, np.float64))
    assert (p[~np.asarray(mask)] == 0).all()
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-5)
    # Same as the softmax over the legal moves alone.
    legal = np.where(mask, np.exp(np.asarray(logits, np.float64)), 0.0)
    np.testing.assert_allclose(p, legal / legal.sum(-1, keepdims=True), rtol=1e-4, atol=1e-7)
    ent = np.asarray(ppo.masked_entropy(logp, mask))
    ref = -np.sum(np.where(mask, p * np.log(np.where(p > 0, p, 1.0)), 0.0), -1)
    np.testing.assert_allclose(ent, ref, rtol=1e-4, atol=1e-6)
    assert ent[0] == pytest.approx(0.0, abs=1e-6) and float(logp[0, 2]) == pytest.approx(0.0)

    # Gradients stay finite even though illegal moves have probability 0.
    def objective(logits):
        lp = ppo.masked_log_probs(logits, mask)
        return jnp.sum(ppo.masked_entropy(lp, mask)) + jnp.sum(jnp.where(mask, lp, 0.0))

    g = jax.grad(objective)(logits)
    assert bool(jnp.isfinite(g).all())
    assert bool((g[~mask] == 0).all())  # illegal logits get no gradient

    # Sampling: only legal moves, with the policy's frequencies.
    keys = jax.random.split(jax.random.key(0), 4000)
    many = jax.vmap(lambda k: ppo.sample_actions(k, logits[:4], mask[:4]))(keys)
    assert many.dtype == jnp.int32
    assert bool(jnp.take_along_axis(mask[:4][None].repeat(4000, 0), many[..., None], -1).all())
    freq = np.stack([(np.asarray(many) == a).mean(0) for a in range(NUM_ACTIONS)], -1)
    np.testing.assert_allclose(freq, p[:4], atol=0.03)


def test_loss_ignores_invalid_samples_and_starts_unclipped():
    cfg = ppo.PPOConfig(**TINY)
    env, _ = ppo.make_envs(cfg)
    params = ppo.init_network(jax.random.key(0), cfg)
    rng = np.random.default_rng(1)
    b = 64
    obs = jnp.asarray(rng.random((b, *env.observe(env.init_state(jax.random.key(0))).shape[1:])))
    mask = jnp.ones((b, NUM_ACTIONS), bool)
    logits, values = ppo.network(params, obs, cfg)
    actions = jnp.asarray(rng.integers(0, NUM_ACTIONS, b), jnp.int32)
    logp = jnp.take_along_axis(ppo.masked_log_probs(logits, mask), actions[:, None], -1)[:, 0]
    valid = jnp.arange(b) % 4 != 0
    mb = ppo.Batch(
        obs=obs,
        mask=mask,
        actions=actions,
        log_probs=logp,
        values=values,
        advantages=jnp.asarray(rng.normal(size=b), jnp.float32),
        returns=jnp.asarray(rng.uniform(-1, 1, b), jnp.float32),
        valid=valid,
    )
    loss, parts = ppo.ppo_loss(params, mb, cfg)
    # The policy that collected the data: ratio 1, nothing clipped, KL 0.
    assert float(parts["approx_kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(parts["clip_frac"]) == 0.0
    assert float(parts["pg_loss"]) == pytest.approx(0.0, abs=1e-5)  # normalised advantages
    poisoned = mb._replace(
        advantages=jnp.where(valid, mb.advantages, 1e6),
        returns=jnp.where(valid, mb.returns, -1e6),
        log_probs=jnp.where(valid, mb.log_probs, 50.0),
    )
    loss2, parts2 = ppo.ppo_loss(params, poisoned, cfg)
    assert float(loss2) == pytest.approx(float(loss), rel=1e-5)
    grads = jax.grad(lambda p: ppo.ppo_loss(p, poisoned, cfg)[0])(params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(grads))


def test_policies_play_legal_moves():
    cfg = ppo.PPOConfig(**TINY)
    env, _ = ppo.make_envs(cfg)
    params = ppo.init_network(jax.random.key(0), cfg)
    state, ts = env.reset(jax.random.key(0))
    greedy = ppo.make_policy(params, cfg, greedy=True)
    sample = ppo.make_policy(params, cfg, greedy=False)
    for policy in (greedy, sample):
        a = jax.jit(policy)(jax.random.key(1), state, ts)
        assert a.shape == (2,) and a.dtype == jnp.int32
        assert bool(jnp.take_along_axis(ts.action_mask, a[:, None], -1).all())
    logits = ppo.network(params, ts.obs, cfg)[0]
    expected = jnp.argmax(jnp.where(ts.action_mask, logits, -jnp.inf), -1)
    np.testing.assert_array_equal(greedy(jax.random.key(2), state, ts), expected)


def test_config_checks():
    with pytest.raises(ValueError, match="num_minibatches"):
        ppo.PPOConfig(num_envs=3, num_steps=5, num_minibatches=4)  # 30 agent samples
    with pytest.raises(ValueError, match="eval_modes"):
        ppo.PPOConfig(eval_modes=("argmax",))
    cfg = ppo.PPOConfig(num_envs=4, num_steps=8, log_every=100, total_env_steps=200)
    s = ppo.schedule(cfg)
    assert (s.chunk_iters, s.chunk_steps, s.num_chunks, s.total_steps) == (3, 96, 3, 288)
    assert s.total_updates == 9 * cfg.update_epochs * cfg.num_minibatches
    assert cfg.minibatch_size == 4 * 8 * 2 // cfg.num_minibatches


def test_bad_eval_opponent_fails_before_training(tmp_path):
    cfg = ppo.PPOConfig(eval_opponents=("alphasnake",), run_dir=str(tmp_path), **TINY)
    with pytest.raises(ValueError, match="unknown agent"):
        ppo.train(cfg)
    assert not (tmp_path / "metrics.jsonl").exists()


# --- (3) End to end ---------------------------------------------------------------


def _argv(run_dir, *extra):
    return [
        "--num-envs", "4", "--num-steps", "16", "--total-env-steps", "256",
        "--log-every", "128", "--num-minibatches", "4", "--update-epochs", "2",
        "--conv-channels", "4,8", "--conv-strides", "2,2", "--hidden", "16",
        "--eval-every", "128", "--eval-games", "4", "--eval-opponents", "random_legal",
        "--run-dir", run_dir, *extra,
    ]  # fmt: skip


def test_tiny_training_run_checkpoints_and_eval_only(tmp_path, capsys):
    run_dir = str(tmp_path / "run")
    cfg, _ = ppo.parse_args(_argv(run_dir, "--clip-vloss", "false"))
    assert cfg.conv_channels == (4, 8) and not cfg.clip_vloss and cfg.lr == ppo.PPOConfig.lr
    assert cfg.eval_opponents == ("random_legal",) and cfg.eval_modes == ("greedy", "sample")
    out_dir, params = ppo.train(cfg)
    assert out_dir == run_dir

    records = [json.loads(line) for line in Path(run_dir, "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    assert [r["env_steps"] for r in train] == [128, 256]
    assert sum(r["agent_samples"] for r in train) == 2 * 256  # both snakes, every step
    assert train[-1]["total_updates"] == 4 * 2 * 4  # iterations x epochs x minibatches
    assert train[0]["lr"] > train[-1]["lr"] == pytest.approx(0.0, abs=1e-12)  # annealed
    for key in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"):
        assert all(np.isfinite(r[key]) for r in train), key
    assert sum(r["games"] for r in train) > 0 and sum(train[-1]["deaths"].values()) > 0
    assert [(e["env_steps"], e["mode"]) for e in evals] == [
        (128, "greedy"), (128, "sample"), (256, "greedy"), (256, "sample"),
    ]  # fmt: skip
    assert all(e["num_games"] == 4 and e["opponent"] == "random_legal" for e in evals)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))

    loaded = ppo.load_params(run_dir)
    assert jax.tree.structure(loaded) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert ppo.load_config(run_dir) == cfg

    capsys.readouterr()
    ppo.main(["--eval-only", run_dir, "--eval-games", "6", "--eval-opponents", "random_legal"])
    lines = capsys.readouterr().out.strip().splitlines()
    results = [json.loads(x) for x in lines if x.startswith("{")]
    assert [r["mode"] for r in results] == ["greedy", "sample"]
    assert all(r["wins"] + r["draws"] + r["losses"] == 6 for r in results)


def test_resume_continues_exactly(tmp_path, monkeypatch):
    def argv(run_dir):
        return _argv(
            run_dir, "--total-env-steps", "512", "--checkpoint-every", "128",
            "--eval-every", "512", "--eval-modes", "sample",
        )  # fmt: skip

    full_dir, cut_dir = str(tmp_path / "full"), str(tmp_path / "cut")
    _, full = ppo.train(ppo.parse_args(argv(full_dir))[0])

    # Interrupt the second run during its third chunk, after two full checkpoints.
    calls = []
    chunk_record = ppo.chunk_record

    def interrupted(*args):
        calls.append(None)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return chunk_record(*args)

    monkeypatch.setattr(ppo, "chunk_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ppo.train(ppo.parse_args(argv(cut_dir))[0])
    monkeypatch.setattr(ppo, "chunk_record", chunk_record)
    ppo.main(["--resume", cut_dir])

    resumed = ppo.load_params(cut_dir)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True):
        np.testing.assert_array_equal(a, b)

    def records(run_dir):
        lines = Path(run_dir, "metrics.jsonl").read_text().splitlines()
        drop = ("seconds", "elapsed", "env_steps_per_s")
        return [{k: v for k, v in json.loads(x).items() if k not in drop} for x in lines]

    assert records(cut_dir) == records(full_dir)  # no duplicated or missing records
