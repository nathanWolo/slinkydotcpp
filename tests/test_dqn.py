"""Tests for the self-play DQN baseline (``baselines/dqn.py``)."""

from __future__ import annotations

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
from slinky.policies import random_legal_policy
from slinky.types import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GameConfig

pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]


def _load_dqn():
    """Import ``baselines/dqn.py`` from its path (``baselines/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("baselines_dqn", ROOT / "baselines" / "dqn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


dqn = _load_dqn()
GAMMA = 0.9
# A tiny network so tests compile and run fast.
TINY = dqn.DQNConfig(conv_channels=(4,), conv_strides=(2,), hidden=16, gamma=GAMMA)


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


def make_transition(env: BattlesnakeEnv, state, actions):
    """The transition ``env.step`` produces from ``state`` (the reached state, no reset)."""
    actions = jnp.asarray(actions, jnp.int32)
    nxt, ts = env.step(jax.random.key(0), state, actions)
    return dqn.Transition(state, actions, ts.reward, nxt, ts.truncated)


def stack(items):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


# --- (1) Per-agent targets ------------------------------------------------------------


def test_td_targets_on_hand_built_transitions():
    duel = BattlesnakeEnv(GameConfig(), obs=None)
    short = BattlesnakeEnv(GameConfig(max_turns=6), obs=None)  # turn 5 -> 6 truncates
    far = {"body": [[8, 8], [8, 7], [8, 6]]}
    up_snake = {"body": [[5, 5], [5, 4], [5, 3]]}
    cases = {
        # Snake 0 hits the left wall: it loses (-1), snake 1 wins (+1); both terminal.
        "win_loss": (duel, [{"body": [[0, 5], [1, 5], [2, 5]]}, far], [LEFT, UP]),
        # Equal-length head-on: both die, nobody survives, a draw (0, terminal).
        "draw": (
            duel,
            [{"body": [[4, 5], [3, 5], [2, 5]]}, {"body": [[6, 5], [7, 5], [8, 5]]}],
            [RIGHT, LEFT],
        ),
        # Both survive but max_turns cuts the game off: bootstrap.
        "truncated": (short, [up_snake, far], [UP, UP]),
        # An ordinary mid-game step: bootstrap.
        "ongoing": (duel, [up_snake, far], [UP, UP]),
    }
    tr = stack([make_transition(env, make_state(env.config, s), a) for env, s, a in cases.values()])
    np.testing.assert_array_equal(tr.rewards, [[-1, 1], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(tr.truncated, [False, False, True, False])

    next_mask = jax.vmap(duel.action_mask)(tr.next_state)
    # After moving UP from (5, 5), DOWN is the neck: illegal. Make the online
    # network prefer it, so the target must use the best *legal* move (LEFT).
    assert not next_mask[2, 0, DOWN] and not next_mask[3, 0, DOWN]
    q_online = jnp.broadcast_to(jnp.array([1.0, 10.0, 2.0, 0.0]), (4, 2, NUM_ACTIONS))
    q_target = jnp.broadcast_to(jnp.array([0.1, 0.2, 0.3, 0.4]), (4, 2, NUM_ACTIONS))
    target, valid = dqn.td_targets(tr, q_online, q_target, next_mask, GAMMA)

    valid_f, terminal = dqn.agent_flags(tr.state, tr.next_state, tr.truncated)
    np.testing.assert_array_equal(valid, valid_f)
    assert bool(valid.all())
    np.testing.assert_array_equal(terminal, [[1, 1], [1, 1], [0, 0], [0, 0]])
    np.testing.assert_allclose(target[0], [-1.0, 1.0])  # loser and winner: no bootstrap
    np.testing.assert_allclose(target[1], [0.0, 0.0])  # draw
    # Snake 0 bootstraps from LEFT (0.3). Snake 1 (head (8, 9), neck below) also
    # can't go DOWN, so its legal argmax is LEFT as well.
    np.testing.assert_allclose(target[2], [GAMMA * 0.3, GAMMA * 0.3], rtol=1e-6)
    np.testing.assert_allclose(target[3], [GAMMA * 0.3, GAMMA * 0.3], rtol=1e-6)


def test_agent_dead_at_t_is_excluded_from_the_loss():
    config = GameConfig(num_snakes=3)
    env = BattlesnakeEnv(config)  # egocentric obs, same shape as the duel's
    snakes = [
        {"body": [[1, 1], [1, 2], [1, 3]]},
        {"body": [[8, 8], [8, 7], [8, 6]]},
        {"body": [[5, 0], [5, 0], [5, 0]], "eliminated_cause": "wall-collision"},
    ]
    tr = stack([make_transition(env, make_state(config, snakes), [RIGHT, UP, UP])] * 4)
    valid, terminal = dqn.agent_flags(tr.state, tr.next_state, tr.truncated)
    np.testing.assert_array_equal(valid[0], [True, True, False])
    np.testing.assert_array_equal(terminal[0], [False, False, True])

    params = dqn.init_network(jax.random.key(0), TINY)
    loss_fn = jax.jit(dqn.make_loss_fn(env, TINY))
    loss, (grads, _) = loss_fn(params, params, tr)
    assert np.isfinite(float(loss))
    # The dead agent's reward does not matter; a living agent's does.
    loss_dead, _ = loss_fn(params, params, tr._replace(rewards=tr.rewards.at[:, 2].set(1e3)))
    loss_live, _ = loss_fn(params, params, tr._replace(rewards=tr.rewards.at[:, 1].set(1e3)))
    assert float(loss_dead) == pytest.approx(float(loss))
    assert float(loss_live) > float(loss) + 100


# --- (2) Replay buffer -------------------------------------------------------------


def test_replay_buffer_wraps_and_samples_only_filled_slots():
    example = {"x": jnp.zeros((), jnp.int32), "y": jnp.zeros((2,), jnp.float32)}
    buf = dqn.buffer_init(example, capacity=6)
    add = jax.jit(dqn.buffer_add)

    def items(start):
        x = jnp.arange(start, start + 2, dtype=jnp.int32)
        return {"x": x, "y": jnp.stack([x, -x], axis=-1).astype(jnp.float32)}

    def sample(buf, seed):
        s = jax.device_get(dqn.buffer_sample(buf, jax.random.key(seed), 2000))
        np.testing.assert_array_equal(s["y"], np.stack([s["x"], -s["x"]], -1))  # rows intact
        return set(s["x"].tolist())

    buf = add(add(buf, items(1)), items(3))  # 1, 2, 3, 4 and two empty slots
    assert (int(buf.ptr), int(buf.size)) == (4, 4)
    assert sample(buf, 0) == {1, 2, 3, 4}  # never the zero-filled slots

    buf = add(buf, items(5))  # full
    assert (int(buf.ptr), int(buf.size)) == (0, 6)
    buf = add(buf, items(7))  # overwrites the two oldest
    assert (int(buf.ptr), int(buf.size)) == (2, 6)
    np.testing.assert_array_equal(buf.data["x"], [7, 8, 3, 4, 5, 6])
    assert sample(buf, 1) == {3, 4, 5, 6, 7, 8}

    with pytest.raises(ValueError, match="multiple"):
        dqn.buffer_add(buf, {"x": jnp.zeros(4, jnp.int32), "y": jnp.zeros((4, 2))})


# --- (3) End to end ---------------------------------------------------------------


def test_tiny_training_run_checkpoints_and_eval_only(tmp_path, capsys):
    run_dir = str(tmp_path / "run")
    argv = [
        "--num-envs", "4", "--total-env-steps", "800", "--log-every", "400",
        "--learning-starts", "200", "--batch-size", "8",
        "--buffer-capacity", "402",  # rounded down to 400: wraps around once
        "--conv-channels", "4,8", "--conv-strides", "2,2", "--hidden", "16", "--dueling", "true",
        "--target-update-period", "10", "--tau", "0.5",
        "--eval-every", "400", "--eval-games", "8", "--run-dir", run_dir,
    ]  # fmt: skip
    cfg, _ = dqn.parse_args(argv)
    assert cfg.conv_channels == (4, 8) and cfg.dueling and cfg.lr == dqn.DQNConfig.lr
    out_dir, params = dqn.train(cfg)
    assert out_dir == run_dir

    records = [json.loads(line) for line in Path(run_dir, "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    assert [r["env_steps"] for r in train] == [400, 800]
    assert train[-1]["buffer_size"] == 400
    assert sum(r["agent_transitions"] for r in train) == 2 * 800  # both snakes, every step
    assert train[-1]["total_updates"] == (800 - 200) // 4 + 1
    losses = [r["loss"] for r in train if r["loss"] is not None]
    assert losses and all(np.isfinite(losses))
    assert all(r["games"] > 0 for r in train)
    assert len(evals) == 2 and all(e["num_games"] == 8 for e in evals)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))

    loaded = dqn.load_params(run_dir)
    assert jax.tree.structure(loaded) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert dqn.load_config(run_dir) == cfg

    capsys.readouterr()
    dqn.main(["--eval-only", run_dir, "--eval-games", "8"])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["num_games"] == 8
    assert result["wins"] + result["draws"] + result["losses"] == 8


# --- (4) Action selection ---------------------------------------------------------


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


def test_epsilon_greedy_never_picks_a_masked_action():
    env = BattlesnakeEnv(GameConfig(), obs=None)
    mask = jax.vmap(env.action_mask)(rollout_states(env))  # [S, 2, 4]
    assert (~mask).any(axis=-1).mean() > 0.5  # most rows rule something out
    # Adversarial Q-values: every illegal move beats every legal one.
    noise = jax.random.normal(jax.random.key(2), mask.shape)
    q = jnp.where(mask, noise, noise + 100.0)
    for eps in (0.0, 0.3, 1.0):
        actions = dqn.epsilon_greedy(jax.random.key(3), q, mask, jnp.float32(eps))
        assert actions.dtype == jnp.int32
        assert bool(jnp.take_along_axis(mask, actions[..., None], axis=-1).all()), eps
    greedy = dqn.epsilon_greedy(jax.random.key(4), q, mask, jnp.float32(0.0))
    np.testing.assert_array_equal(greedy, jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1))
    explore = dqn.epsilon_greedy(jax.random.key(5), q, mask, jnp.float32(1.0))
    assert (explore != greedy).mean() > 0.2  # eps = 1 is random among legal moves


def test_epsilon_schedule():
    cfg = dqn.DQNConfig(total_env_steps=1000, eps_decay_fraction=0.5, eps_end=0.1)
    eps = [float(dqn.epsilon(cfg, jnp.int32(t))) for t in (0, 250, 500, 1000)]
    np.testing.assert_allclose(eps, [1.0, 0.55, 0.1, 0.1], rtol=1e-6)


@pytest.mark.parametrize("stride", [1, 2])
def test_conv3x3_gradient_matches_xla_convolution(stride):
    x = jax.random.normal(jax.random.key(0), (3, 21, 21, 5))
    w = jax.random.normal(jax.random.key(1), (3, 3, 5, 7))

    def reference(x, w):
        dn = ("NHWC", "HWIO", "NHWC")
        return jax.lax.conv_general_dilated(x, w, (stride, stride), "SAME", dimension_numbers=dn)

    def loss(conv):
        return lambda x, w: jnp.sum(jnp.sin(conv(x, w)))

    ours = jax.grad(loss(lambda x, w: dqn.conv3x3(x, w, stride)), argnums=(0, 1))(x, w)
    ref = jax.grad(loss(reference), argnums=(0, 1))(x, w)
    for a, b in zip(ours, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
