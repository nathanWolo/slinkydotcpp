"""Tests for the AlphaZero baseline (``baselines/az.py``): replay, symmetries, loss, runs."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import DUEL, board, env_for, rollout_states

from slinky import agents as agents_mod
from slinky import heuristic, rules
from slinky.evaluate import play_match
from slinky.observations import make_obs_fn
from slinky.types import NUM_ACTIONS, GameConfig

pytest.importorskip("optax")

az = agents_mod.load_az_module()  # the module the registry uses
BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
# A tiny network so tests compile and run fast.
TINY = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)


# --- (1) Board symmetries -----------------------------------------------------------------


def test_action_permutations():
    """Flip y swaps UP/DOWN, flip x swaps LEFT/RIGHT, the transpose maps UP<->RIGHT."""
    expected = {1: [1, 0, 2, 3], 2: [0, 1, 3, 2], 4: [3, 2, 1, 0], 0: [0, 1, 2, 3]}
    for sym, perm in expected.items():
        assert az.action_permutation(sym).tolist() == perm
    tab = az.symmetry_tables(11, 11)
    assert tab.perm.shape == (8, 4) and len({tuple(p) for p in tab.perm}) == 8
    for s in range(8):
        assert sorted(tab.source[s]) == list(range(121))  # a permutation of the cells
        np.testing.assert_array_equal(tab.perm[s][tab.inverse[s]], np.arange(4))
    assert az.symmetry_tables(11, 7).perm.shape == (4, 4)  # no transpose on a non-square board


def _positions() -> object:
    """Mid-game states from random play, with hazards sprinkled in to exercise that channel."""
    parts = [rollout_states(DUEL, batch=32, turns=t, seed=t) for t in (8, 25, 40)]
    states = jax.tree.map(lambda *x: jnp.concatenate(x), *parts)
    hazard = jax.random.randint(jax.random.key(0), states.hazard.shape, 0, 3) * (
        jax.random.uniform(jax.random.key(1), states.hazard.shape) < 0.1
    )
    return states._replace(hazard=hazard.astype(states.hazard.dtype))


def test_symmetries_are_exact_on_real_positions():
    """Observation, legal mask, rules and heuristic all commute with every symmetry."""
    states = _positions()
    n = states.done.shape[0]
    assert n > 16
    env = env_for(DUEL)
    obs_fn = jax.jit(jax.vmap(make_obs_fn(DUEL)))
    mask_fn = jax.jit(jax.vmap(env.action_mask))
    step = jax.jit(jax.vmap(lambda s, a: rules.rules_step(s, a, DUEL)))
    value_fn = jax.jit(jax.vmap(lambda s: heuristic.evaluate(s, DUEL)))
    transform = jax.jit(jax.vmap(lambda s, g: az.transform_state(s, g, DUEL), in_axes=(0, None)))
    acts = jax.random.randint(jax.random.key(2), (n, 2), 0, NUM_ACTIONS)
    obs, mask, nxt, values = obs_fn(states), mask_fn(states), step(states, acts), value_fn(states)
    for sym in range(8):
        t_states = transform(states, sym)
        np.testing.assert_array_equal(obs_fn(t_states), az.transform_obs(obs, sym), err_msg=sym)
        np.testing.assert_array_equal(mask_fn(t_states), az.transform_moves(mask, sym, DUEL))
        perm = jnp.asarray(az.action_permutation(sym))
        expected = transform(nxt, sym)
        stepped = jax.tree.leaves(step(t_states, perm[acts]))
        for got, want in zip(stepped, jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_allclose(value_fn(t_states), values, atol=1e-5)
    # A traced symmetry per position (as sampling uses it) gives the same states.
    syms = jnp.arange(n) % 8
    traced = jax.jit(jax.vmap(lambda s, g: az.transform_state(s, g, DUEL)))(states, syms)
    for i in (0, 5, 11):
        one = az.transform_state(jax.tree.map(lambda x, i=i: x[i], states), i % 8, DUEL)
        for got, want in zip(jax.tree.leaves(traced), jax.tree.leaves(one), strict=True):
            np.testing.assert_array_equal(got[i], want)


# --- (2) Replay buffer and outcome table --------------------------------------------------


def _toy_replay(capacity: int):
    example = jax.tree.map(lambda x: x[0], rollout_states(DUEL, batch=16, turns=30))
    return az.replay_init(example, capacity), example


def test_outcome_table_masks_unfinished_games_and_never_aliases():
    """A random schedule of games: a position counts iff its game ended, with that outcome.

    ``E`` env slots each play games of random length (1-9 turns); every
    iteration of ``T`` turns writes ``T * E`` positions and the outcomes of the
    games that ended. A reference model tracks every game's true outcome.
    """
    rng = np.random.default_rng(0)
    E, T, C = 3, 2, 12  # the buffer holds 4 turns per slot: games often outlive it
    replay, example = _toy_replay(C)
    add = jax.jit(az.replay_add)
    record = jax.jit(az.replay_record_outcomes)
    lookup = jax.jit(az.replay_lookup)
    states = jax.tree.map(lambda x: jnp.stack([x] * (T * E)), example)
    policy = jnp.zeros((T * E, 2, NUM_ACTIONS))
    uids = np.arange(E)
    left = rng.integers(1, 10, E)  # turns left in each slot's game
    outcomes: dict[int, np.ndarray] = {}  # uid -> outcome of the games that ended
    written: list[int] = []  # uid of every position, in write order
    for _ in range(60):
        pos_uid, done, out = [], [], []
        for _t in range(T):
            for e in range(E):
                pos_uid.append(uids[e])
                left[e] -= 1
                done.append(left[e] == 0)
                out.append(rng.choice([-1.0, -0.5, 0.0, 1.0], 2))
            for e in range(E):
                if left[e] == 0:
                    outcomes[int(uids[e])] = out[-E + e]
                    uids[e] += E
                    left[e] = rng.integers(1, 10)
        replay = add(replay, states, policy, jnp.asarray(pos_uid, jnp.int32))
        replay = record(
            replay, jnp.asarray(pos_uid, jnp.int32), jnp.asarray(done), jnp.asarray(np.array(out))
        )
        written += pos_uid
        finished, z = map(np.asarray, lookup(replay, jnp.arange(C)))
        stored = np.asarray(replay.uid)
        assert int(replay.size) == min(len(written), C)
        for slot in range(int(replay.size)):
            u = int(stored[slot])
            assert finished[slot] == (u in outcomes), (slot, u)
            if u in outcomes:
                np.testing.assert_array_equal(z[slot], outcomes[u])
    assert 0 < sum(finished) < C  # both kinds were present at the end


def test_terminal_values_give_the_outcome_targets():
    """Win/loss, a mutual elimination (draw_value) and truncation (0) as self-play sees them."""
    env = env_for(GameConfig(max_turns=11))
    a = [(5, 5), (5, 4), (5, 3)]
    b = [(5, 7), (5, 8), (5, 9)]
    up, down, left = 0, 1, 2  # b's neck is above its head: b never moves up

    def outcome(snakes, acts, turn=5):
        state = board(snakes, turn=turn)
        new, ts = env.step(jax.random.key(0), state, jnp.asarray(acts))
        return bool(ts.done), bool(ts.truncated), heuristic.terminal_values(new, env.config, -0.5)

    done, trunc, z = outcome([a, b], [up, down])  # head to head at equal length
    assert done and not trunc and z.tolist() == [-0.5, -0.5]
    done, trunc, z = outcome([[(0, 5), (1, 5), (2, 5)], b], [left, left])  # into the wall
    assert done and not trunc and z.tolist() == [-1.0, 1.0]
    done, trunc, z = outcome([a, b], [left, left], turn=10)  # cut off at max_turns
    assert done and trunc and z.tolist() == [0.0, 0.0]
    done, _, z = outcome([a, b], [left, left])  # still running: no outcome yet
    assert not done


def test_sample_batch_weights_and_augmentation():
    cfg = az.AZConfig(num_envs=2, num_steps=2, buffer_capacity=8, learning_starts=4, batch_size=64,
                      **TINY)  # fmt: skip
    env = az.make_env(cfg)
    replay, example = _toy_replay(8)
    states = jax.tree.map(lambda x: x[:4], rollout_states(DUEL, batch=16, turns=30))
    target = jax.nn.softmax(jax.random.normal(jax.random.key(0), (4, 2, 4)))
    target = jnp.where(jax.vmap(env.action_mask)(states), target, 0.0)
    target = target / target.sum(-1, keepdims=True)
    uid = jnp.array([0, 1, 0, 1], jnp.int32)
    replay = az.replay_add(replay, states, target, uid)
    replay = az.replay_record_outcomes(
        replay, uid, jnp.array([False, True, False, False]), jnp.array([[0, 0], [1, -1.0]] * 2)
    )
    batch = jax.jit(lambda r, k: az.sample_batch(r, k, cfg, env))(replay, jax.random.key(1))
    uids = np.asarray(replay.uid)[: int(replay.size)]
    assert batch.obs.shape == (64, 2, 21, 21, 13)
    np.testing.assert_array_equal(batch.finished, batch.w_value.any(-1))
    assert 0 < batch.finished.mean() < 1 and set(uids) == {0, 1}
    np.testing.assert_array_equal(np.asarray(batch.z)[np.asarray(batch.finished)], [[1, -1]] * int(
        batch.finished.sum()))  # fmt: skip
    # Targets stay on the (transformed) legal moves and still sum to 1.
    assert bool(jnp.all(jnp.where(batch.legal, True, batch.target == 0)))
    np.testing.assert_allclose(batch.target.sum(-1), 1.0, rtol=1e-6)


# --- (3) Loss ---------------------------------------------------------------------------


def test_loss_on_a_hand_made_batch():
    """A network that ignores its input: logits = policy bias, value = tanh(value bias)."""
    cfg = az.AZConfig(value_coef=0.7, **TINY)
    params = jax.tree.map(jnp.zeros_like, az.init_network(jax.random.key(0), cfg))
    bias = np.array([1.0, -0.5, 0.3, 2.0], np.float32)
    params["policy"]["b"] = jnp.asarray(bias)
    params["value"]["b"] = jnp.asarray([0.4])
    legal = np.array([[[1, 1, 1, 1], [1, 0, 1, 0]], [[0, 1, 1, 1], [1, 1, 1, 1]]], bool)
    target = np.array([[[0.25, 0.25, 0.5, 0.0], [0.6, 0.0, 0.4, 0.0]],
                       [[0.0, 1.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]]], np.float32)  # fmt: skip
    z = np.array([[1.0, -1.0], [-0.5, -0.5]], np.float32)
    w_value = np.array([[True, True], [True, False]])
    w_policy = np.array([[True, False], [True, False]])
    batch = az.Batch(
        obs=jnp.zeros((2, 2, 21, 21, 13)), legal=jnp.asarray(legal), target=jnp.asarray(target),
        z=jnp.asarray(z), w_policy=jnp.asarray(w_policy), w_value=jnp.asarray(w_value),
        finished=jnp.array([True, False]),
    )  # fmt: skip
    loss, parts = az.az_loss(params, batch, cfg)

    def log_softmax(row_mask):  # 0 on illegal moves, where the targets are 0
        return np.where(row_mask, bias - np.log(np.sum(np.exp(bias[row_mask]))), 0.0)

    ce = [-np.sum(np.where(legal[b, i], target[b, i] * log_softmax(legal[b, i]), 0.0))
          for b, i in ((0, 0), (1, 0))]  # fmt: skip
    v = np.tanh(0.4)
    value_loss = np.mean([(v - 1.0) ** 2, (v + 1.0) ** 2, (v + 0.5) ** 2])
    np.testing.assert_allclose(parts["policy_loss"], np.mean(ce), rtol=1e-5)
    np.testing.assert_allclose(parts["value_loss"], value_loss, rtol=1e-5)
    np.testing.assert_allclose(loss, np.mean(ce) + 0.7 * value_loss, rtol=1e-5)
    h = [-np.sum(t[t > 0] * np.log(t[t > 0])) for t in (target[0, 0], target[1, 0])]
    np.testing.assert_allclose(parts["target_entropy"], np.mean(h), rtol=1e-5)
    np.testing.assert_allclose(parts["value_abs_mean"], v, rtol=1e-5)
    np.testing.assert_allclose(parts["value_target_var"], np.var([1.0, -1.0, -0.5]), rtol=1e-5)
    np.testing.assert_allclose(parts["masked_frac"], 0.5)
    # No samples at all: zero loss and zero gradients, not NaN.
    empty = batch._replace(w_policy=jnp.zeros_like(batch.w_policy), w_value=batch.w_policy & False)
    grads = jax.grad(lambda p: az.az_loss(p, empty, cfg)[0])(params)
    assert all(float(jnp.abs(g).max()) == 0.0 for g in jax.tree.leaves(grads))


# --- (4) Config, schedule and policies --------------------------------------------------


def test_config_checks_and_schedule():
    with pytest.raises(ValueError, match="buffer_capacity must hold"):
        az.AZConfig(num_envs=8, num_steps=8, buffer_capacity=32)
    with pytest.raises(ValueError, match="learning_starts"):
        az.AZConfig(num_envs=8, num_steps=8, buffer_capacity=100, learning_starts=65)
    with pytest.raises(ValueError, match="eval_modes"):
        az.AZConfig(eval_modes=("greedy",))
    with pytest.raises(ValueError, match="c_puct"):
        az.AZConfig(c_puct=-1.0)
    with pytest.raises(ValueError, match="root_noise_frac"):
        az.AZConfig(root_noise_frac=1.5)
    cfg = az.AZConfig(num_envs=4, num_steps=8, buffer_capacity=100, learning_starts=10,
                      batch_size=16, replay_ratio=2.0, log_every=70, total_env_steps=200,
                      teacher_env_steps=65)  # fmt: skip
    assert cfg.capacity == 96 and cfg.updates_per_iteration == 4  # 2 * 32 / 16
    s = az.schedule(cfg)
    assert (s.chunk_iters, s.chunk_steps, s.num_chunks, s.total_steps) == (2, 64, 4, 256)
    assert s.teacher_chunks == 2 and s.teacher_steps == 128
    assert [s.phase(c) for c in range(4)] == ["teacher", "teacher", "network", "network"]
    assert s.total_updates == 4 * 2 * 4
    # The learning-rate decay counts only the iterations that learn: with 70 positions to
    # wait for, the first two (32 and 64 positions) take no gradient steps.
    late = dataclasses.replace(cfg, learning_starts=70)
    assert az.schedule(late).total_updates == (4 * 2 - 2) * 4
    assert az.schedule(dataclasses.replace(cfg, learning_starts=0)).total_updates == 4 * 2 * 4
    assert az.agent_search(cfg, 8, final="sample").final == "sample"
    with pytest.raises(ValueError, match="can't be set"):
        az.agent_search(cfg, 8, selection="duct")
    # 10 env steps per iteration; the network phase starts after 5 iterations (50 steps).
    blend = az.AZConfig(num_envs=2, num_steps=5, buffer_capacity=100, learning_starts=10,
                        blend_env_steps=100, blend_start=0.8)  # fmt: skip
    assert az.blend_at(az.AZConfig(), jnp.int32(0), 0) is None
    np.testing.assert_allclose(
        [az.blend_at(blend, jnp.int32(i), 50) for i in (5, 10, 15, 40)], [0.8, 0.4, 0.0, 0.0]
    )
    # The registry keeps its own copy of the settable search fields.
    assert agents_mod._az_search_fields() == az.AGENT_SEARCH_FIELDS


def test_policies_play_legal_moves():
    cfg = az.AZConfig(**TINY)
    params = az.init_network(jax.random.key(0), cfg)
    env = env_for(DUEL)
    states = jax.tree.map(lambda x: x[:8], rollout_states(DUEL, batch=16, turns=30))
    mask = jax.vmap(env.action_mask)(states)
    for sims in (0, 3):
        policy = az.make_policy(params, cfg, DUEL, sims)
        acts = jax.jit(jax.vmap(policy, in_axes=(0, 0, None)))(
            jax.random.split(jax.random.key(1), 8), states, None
        )
        assert acts.shape == (8, 2) and bool(jnp.take_along_axis(mask, acts[..., None], -1).all())
    logits = jax.vmap(make_obs_fn(DUEL))(states)
    logits = az.network(params, logits, cfg)[0]
    raw = jax.vmap(az.make_policy(params, cfg, DUEL, 0), in_axes=(None, 0, None))
    np.testing.assert_array_equal(
        raw(jax.random.key(0), states, None), jnp.argmax(jnp.where(mask, logits, -jnp.inf), -1)
    )


def test_init_from_a_ppo_run(tmp_path):
    ppo = agents_mod.load_ppo_module()
    ppo_cfg = ppo.PPOConfig(**TINY)
    ppo.save_config(str(tmp_path), ppo_cfg)
    ppo_params = ppo.init_network(jax.random.key(5), ppo_cfg)
    ppo.save_checkpoint(str(tmp_path), ppo_params)
    cfg = az.AZConfig(**TINY)
    fresh = az.init_network(jax.random.key(0), cfg)
    params = az.init_from_run(fresh, cfg, str(tmp_path))
    for layer in ("conv0", "hidden", "policy"):
        np.testing.assert_array_equal(params[layer]["w"], ppo_params[layer]["w"])
    np.testing.assert_array_equal(params["value"]["w"], fresh["value"]["w"])  # re-initialised
    with pytest.raises(ValueError, match="hidden"):
        az.init_from_run(fresh, dataclasses.replace(cfg, hidden=8), str(tmp_path))


# --- (5) End to end ---------------------------------------------------------------------


def _argv(run_dir, *extra):
    return [
        "--num-envs", "4", "--num-steps", "4", "--log-every", "16", "--total-env-steps", "32",
        "--num-simulations", "4", "--teacher-simulations", "4", "--teacher-env-steps", "16",
        "--conv-channels", "4", "--conv-strides", "2", "--hidden", "16", "--max-turns", "6",
        "--buffer-capacity", "64", "--learning-starts", "16", "--batch-size", "8",
        "--eval-every", "32", "--eval-games", "2", "--eval-opponents", "random_legal",
        "--eval-simulations", "2", "--run-dir", run_dir, *extra,
    ]  # fmt: skip


def _records(run_dir):
    return [json.loads(x) for x in Path(run_dir, "metrics.jsonl").read_text().splitlines()]


def test_two_phase_training_run_checkpoints_eval_only_and_registry(tmp_path, capsys):
    """Teacher then network phase, checkpoints, --eval-only, and the az agents built from it."""
    run_dir = str(tmp_path / "run")
    cfg, _ = az.parse_args(_argv(run_dir, "--augment", "false"))
    assert cfg.conv_channels == (4,) and not cfg.augment and cfg.eval_opponents == ("random_legal",)
    out_dir, params = az.train(cfg)
    assert out_dir == run_dir

    records = _records(run_dir)
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    assert [(r["phase"], r["env_steps"]) for r in train] == [("teacher", 16), ("network", 32)]
    # Learning starts with 16 positions; 4 uses per position = 8 steps of 8 per iteration.
    assert [r["updates"] for r in train] == [8, 8]
    assert train[-1]["buffer_size"] == 32 and train[-1]["total_updates"] == 16
    # max_turns 6: every game ends within 6 turns, so outcomes arrive and samples count.
    assert sum(r["games"] for r in train) >= 4 and train[-1]["masked_frac"] < 1
    for key in (*az.LOSS_METRICS, "grad_norm"):
        assert all(np.isfinite(r[key]) for r in train), key
    assert all(r["explained_var"] is None or r["explained_var"] <= 1 for r in train)
    assert [(e["mode"], e["agent"]) for e in evals] == [("search", "az-2"), ("raw", "az-0")]
    assert all(e["num_games"] == 2 and e["env_steps"] == 32 for e in evals)
    for name in ("params.npz", "config.json", "runner.npz", "replay.npz"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    loaded = az.load_params(run_dir)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert az.load_config(run_dir) == cfg

    capsys.readouterr()
    az.main(["--eval-only", run_dir, "--eval-games", "2", "--eval-opponents", "random_legal",
             "--eval-simulations", "3"])  # fmt: skip
    lines = capsys.readouterr().out.strip().splitlines()
    results = [json.loads(x) for x in lines if x.startswith("{")]
    assert [r["agent"] for r in results] == ["az-3", "az-0"]
    assert all(r["wins"] + r["draws"] + r["losses"] == 2 for r in results)

    # The registry plays the checkpoint, raw and with search.
    game = GameConfig()
    env = agents_mod.make_env(game, False)
    for name in (f"az-0:{run_dir}", f"az-4:{run_dir}"):
        agent = agents_mod.make_agent(name, game)
        assert not agent.needs_obs and agent.name == name
        other = agents_mod.make_agent("random_legal", game).policy
        r = play_match(env, agent.policy, other, jax.random.key(0), 2, max_turns=10)
        assert r.wins + r.draws + r.losses == 2
    info = agents_mod.agent_config(agents_mod.parse_agent(f"az-4:{run_dir}"))
    assert info["simulations"] == 4 and info["search"]["leaf"] == "network"
    assert info["network"]["max_turns"] == 6 and info["search"]["root_noise_frac"] == 0.0

    template = jax.jit(az.init_runner, static_argnums=(0, 1))(
        cfg, az.make_env(cfg), jax.random.key(0)
    )
    runner, samples, restored = az.load_runner(run_dir, template)
    assert restored and int(runner.replay.size) == 32 and int(runner.iterations) == 2
    assert samples == train[-1]["agent_samples_total"]
    # A crash between the two files of a later checkpoint leaves its buffer beside this
    # runner: the buffer would repeat the games the runner replays, so it is not used.
    runner_path = os.path.join(run_dir, "runner.npz")
    shutil.copy(runner_path, tmp_path / "runner.npz")
    az.save_runner(run_dir, runner._replace(iterations=runner.iterations + 1), samples, True)
    shutil.copy(tmp_path / "runner.npz", runner_path)
    runner, _, restored = az.load_runner(run_dir, template)
    assert not restored and int(runner.replay.size) == 0 and int(runner.iterations) == 2
    # Without replay.npz a resumed run starts with an empty buffer too.
    os.remove(os.path.join(run_dir, "replay.npz"))
    runner, _, restored = az.load_runner(run_dir, template)
    assert not restored and int(runner.replay.size) == 0 and int(runner.iterations) == 2


def test_resume_continues_exactly(tmp_path, monkeypatch):
    def argv(run_dir):
        return _argv(
            run_dir, "--total-env-steps", "48", "--teacher-env-steps", "0", "--checkpoint-every",
            "16", "--eval-every", "48", "--eval-modes", "raw", "--learning-starts", "32",
        )  # fmt: skip

    full_dir, cut_dir = str(tmp_path / "full"), str(tmp_path / "cut")
    cfg = az.parse_args(argv(full_dir))[0]
    _, full = az.train(cfg)
    # The first of the three iterations waits for positions, and the learning rate still
    # decays to its floor over the gradient steps actually taken.
    last = [r for r in _records(full_dir) if r["type"] == "train"][-1]
    assert last["total_updates"] == az.schedule(cfg).total_updates == 2 * 8
    np.testing.assert_allclose(last["lr"], cfg.lr * cfg.final_lr_frac, rtol=1e-5)

    # Interrupt the second run during its third chunk, after two full checkpoints.
    calls = []
    chunk_record = az.chunk_record

    def interrupted(*args):
        calls.append(None)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return chunk_record(*args)

    monkeypatch.setattr(az, "chunk_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        az.train(az.parse_args(argv(cut_dir))[0])
    monkeypatch.setattr(az, "chunk_record", chunk_record)
    az.main(["--resume", cut_dir])

    resumed = az.load_params(cut_dir)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True):
        np.testing.assert_array_equal(a, b)

    def records(run_dir):
        drop = ("seconds", "elapsed", "env_steps_per_s", "samples_per_s")
        return [{k: v for k, v in r.items() if k not in drop} for r in _records(run_dir)]

    assert records(cut_dir) == records(full_dir)  # no duplicated or missing records


# --- (6) The strength benchmark knows the az agents --------------------------------------


def test_strength_benchmark_costs_and_sorts_az_agents(tmp_path):
    sys.path.insert(0, str(BENCHMARKS))
    import strength as S

    cfg = az.AZConfig(**TINY)
    az.save_config(str(tmp_path), cfg)
    az.save_checkpoint(str(tmp_path), az.init_network(jax.random.key(0), cfg))
    raw, searched = S.load_agent(f"az-0:{tmp_path}"), S.load_agent(f"az-16:{tmp_path}")
    mcts = S.load_agent("mcts-16")
    heur = S.load_agent("heuristic")
    assert searched.sims == 16 and raw.sims == 0
    costs = [S.Matchup(a, heur, 64).rough_seconds() for a in (mcts, raw, searched)]
    assert costs[0] < costs[2] and costs[1] < costs[2]
    assert S.tree_bytes_per_game(raw, DUEL) == 0
    assert S.tree_bytes_per_game(searched, DUEL) == S.tree_bytes_per_game(mcts, DUEL) + 17 * 32
    names = ["mcts-16", "az-64", "ppo", "az-0", "heuristic", "az-128", "dqn"]
    assert sorted(names, key=S.agent_sort_key) == [
        "heuristic", "az-0", "dqn", "ppo", "az-64", "az-128", "mcts-16"
    ]  # fmt: skip
    assert "baselines_az" in [name for _, name, _ in agents_mod.BASELINES.values()]
