"""Tests for ``puffer/policy.py``: the PyTorch policy is ``baselines/ppo.py``'s network.

The oracle is ppo.py itself (``network``, ``load_params``, ``masked_log_probs``): the
policy's masked logits and value, on observations built from slinky's own states, must
equal ppo.py's on the checkpoint the policy exports, and slinky's ``ppo:<run dir>`` agent
must play that checkpoint. None of this needs the C environment: the uint8 observation is
rebuilt from slinky's float one (``round(obs / scale)``, the contract of
``puffer/obs_format.py``), and the "environment" the policy needs is a namespace with the
board shape, observation size and channel scale.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rollout_states

from slinky.agents import make_agent
from slinky.env import BattlesnakeEnv
from slinky.types import GameConfig, TimeStep

torch = pytest.importorskip("torch")
pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `puffer` is a directory of the repository, not installed
    sys.path.append(str(ROOT))

from puffer import obs_format  # noqa: E402
from puffer import policy as P  # noqa: E402


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

DUEL = GameConfig()
STANDARD4 = GameConfig(num_snakes=4)
# Even sides: stride 2 gives "SAME" padding of 0 before and 1 after, which torch's
# padding=1 does not do.
WRAPPED = GameConfig(width=10, height=8, num_snakes=3, ruleset="wrapped")

DEFAULT_NET = dict(conv_channels=(16, 32, 32), conv_strides=(2, 2, 1), hidden=256)
SMALL_NET = dict(conv_channels=(8,), conv_strides=(2,), hidden=32)
STRIDES_1_2 = dict(conv_channels=(8, 8), conv_strides=(1, 2), hidden=32)
CASES = {
    "duel": (DUEL, DEFAULT_NET),
    "standard4": (STANDARD4, DEFAULT_NET),
    "wrapped-even-board": (WRAPPED, DEFAULT_NET),
    "duel-small-net": (DUEL, SMALL_NET),
    "duel-strides-1-2": (DUEL, STRIDES_1_2),
    "wrapped-small-net": (WRAPPED, SMALL_NET),
}


# --- Helpers ----------------------------------------------------------------------------


def env_like(game: GameConfig) -> types.SimpleNamespace:
    """What the policy needs of the environment, from ``puffer/obs_format.py``."""
    w, h, wrapped = game.width, game.height, game.ruleset.wrapped
    return types.SimpleNamespace(
        board_shape=obs_format.board_shape(w, h, wrapped),
        obs_size=obs_format.obs_size(w, h, wrapped),
        channel_scale=obs_format.channel_scale(w, h),
    )


def ppo_cfg(game: GameConfig, net: dict) -> types.SimpleNamespace:
    """What ppo.py's ``init_network``/``network``/``load_params`` read of a ``PPOConfig``.

    (``PPOConfig`` itself always describes the 11x11 duel board.)
    """
    return types.SimpleNamespace(game=game, **net)


def make_policy(game: GameConfig, net: dict, seed: int = 0) -> P.BattlesnakePolicy:
    torch.manual_seed(seed)
    return P.BattlesnakePolicy(env_like(game), **net)


def randomize(policy: P.BattlesnakePolicy, seed: int = 1) -> P.BattlesnakePolicy:
    """Give every parameter an O(1) effect (the initial policy head and biases are ~0)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            else:
                gain = 1.0 if name.startswith(("policy_head", "value_head")) else 2.0
                p.copy_(torch.randn(p.shape, generator=gen) * (gain / p[0].numel()) ** 0.5)
    return policy


class AgentData:
    """Every living agent of a batch of slinky states, as the policy would see it."""

    def __init__(self, game: GameConfig, batch: int = 8, turns: int = 40):
        env = BattlesnakeEnv(game, obs="egocentric")
        self.game = game
        self.states = rollout_states(game, batch=batch, turns=turns, every_turn=True)
        all_obs = jax.jit(jax.vmap(env.observe))(self.states)  # [S, N, rows, cols, C]
        all_mask = jax.jit(jax.vmap(env.action_mask))(self.states)  # [S, N, 4]
        num_states = len(self.states.turn)
        # The inputs slinky's own agents are given.
        self.timesteps = TimeStep(
            obs=all_obs,
            reward=jnp.zeros((num_states, game.num_snakes), jnp.float32),
            done=self.states.done,
            truncated=jnp.zeros(num_states, bool),
            alive=self.states.alive,
            action_mask=all_mask,
        )
        obs, mask = np.asarray(all_obs), np.asarray(all_mask)
        self.alive = np.asarray(self.states.alive) & ~np.asarray(self.states.done)[:, None]
        assert self.alive.any()
        self.float_obs, self.mask = obs[self.alive], mask[self.alive]
        scale = obs_format.channel_scale(game.width, game.height)
        board = np.rint(self.float_obs / scale)
        assert board.min() >= 0 and board.max() <= 255
        # The contract of obs_format: raw integers times the scale are slinky's floats.
        np.testing.assert_allclose(board * scale, self.float_obs, rtol=1e-6, atol=0)
        flat = np.concatenate(
            [board.reshape(len(board), -1), self.mask.astype(np.float64)], axis=1
        ).astype(np.uint8)
        self.obs = torch.from_numpy(flat)  # uint8 [M, obs_size]
        assert flat.shape[1] == env_like(game).obs_size


@functools.cache
def agent_data(game: GameConfig) -> AgentData:
    return AgentData(game)


def reference(run_dir, game: GameConfig, net: dict, float_obs: np.ndarray):
    """ppo.py's ``(logits, value)`` for the checkpoint in ``run_dir``: the oracle."""
    cfg = ppo_cfg(game, net)
    params = ppo.load_params(str(run_dir), cfg)  # also checks every shape against init_network
    logits, value = ppo.network(params, jnp.asarray(float_obs), cfg)
    return np.asarray(logits), np.asarray(value)


def masked_argmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(argmax, gap)``: the best move and its lead over the runner-up (both legal moves)."""
    order = np.sort(logits, axis=-1)
    return np.argmax(logits, axis=-1), order[:, -1] - order[:, -2]


# --- (1) The network equals ppo.py's ----------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_masked_logits_and_value_match_ppo_network(case, tmp_path):
    game, net = CASES[case]
    data = agent_data(game)
    policy = randomize(make_policy(game, net))
    P.export_ppo_checkpoint(policy, tmp_path)
    ref_logits, ref_value = reference(tmp_path, game, net, data.float_obs)
    ref_masked = np.where(data.mask, ref_logits, ppo.MASKED_LOGIT)

    with torch.no_grad():
        logits, value = policy(data.obs)
    assert logits.dtype == torch.float32 and logits.shape == (len(data.obs), 4)
    assert value.dtype == torch.float32 and value.shape == (len(data.obs), 1)
    np.testing.assert_allclose(logits.numpy(), ref_masked, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(value.numpy()[:, 0], ref_value, rtol=1e-4, atol=1e-4)
    # The comparison is only meaningful if the network is not degenerate.
    legal = ref_logits[data.mask]
    assert legal.std() > 0.05 and ref_value.std() > 0.01
    assert (logits.numpy()[~data.mask] == ppo.MASKED_LOGIT).all()
    assert not data.mask.all(), "want states with illegal moves"
    # forward is forward_eval (PufferLib calls both).
    with torch.no_grad():
        again, value_again = policy.forward_eval(data.obs, {"unused": 1})
    assert torch.equal(again, logits) and torch.equal(value_again, value)


def test_masked_logit_and_defaults_are_ppos():
    assert P.MASKED_LOGIT == ppo.MASKED_LOGIT
    default = ppo.PPOConfig()
    assert default.conv_channels == P.DEFAULT_CONV_CHANNELS
    assert default.conv_strides == P.DEFAULT_CONV_STRIDES
    assert default.hidden == P.DEFAULT_HIDDEN
    assert default.max_turns == P.DEFAULT_MAX_TURNS


@pytest.mark.parametrize(
    ("rows", "cols", "strides"),
    [
        (6, 7, (3, 2, 1)),
        (1, 5, (2, 1)),
        (8, 8, (2, 2, 2)),
        (9, 10, (1,)),
        (4, 12, (4, 3)),
        (5, 5, ()),
    ],
)
def test_same_padding_on_arbitrary_board_shapes(rows, cols, strides):
    """Strides and sizes the games do not use: random boards against ppo.py's "SAME" convs."""
    shape = (rows, cols, obs_format.NUM_CHANNELS)
    size = rows * cols * obs_format.NUM_CHANNELS
    scale = np.linspace(0.5, 2.0, obs_format.NUM_CHANNELS, dtype=np.float32)
    env = types.SimpleNamespace(board_shape=shape, obs_size=size + 4, channel_scale=scale)
    torch.manual_seed(3)
    policy = randomize(P.BattlesnakePolicy(env, (4,) * len(strides), strides, hidden=8))
    rng = np.random.default_rng(0)
    board = rng.integers(0, 4, (5, size))
    mask = rng.integers(0, 2, (5, 4))
    mask[:, 0] = 1  # a legal move exists
    obs = torch.from_numpy(np.concatenate([board, mask], axis=1).astype(np.uint8))

    params: dict = {}
    for key, array in P.ppo_params(policy).items():
        layer, name = key.split("/")
        params.setdefault(layer, {})[name] = jnp.asarray(array)
    float_obs = board.reshape(5, *shape).astype(np.float32) * scale
    logits, value = ppo.network(
        params, jnp.asarray(float_obs), types.SimpleNamespace(conv_strides=strides)
    )
    with torch.no_grad():
        got_logits, got_value = policy(obs)
    np.testing.assert_allclose(
        got_logits.numpy(),
        np.where(mask == 1, np.asarray(logits), ppo.MASKED_LOGIT),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(got_value.numpy()[:, 0], np.asarray(value), rtol=1e-4, atol=1e-4)


def test_same_padding_matches_xla():
    for size in range(1, 25):
        for stride in range(1, 6):
            xla = jax.lax.padtype_to_pads((size,), (3,), (stride,), "SAME")
            assert [P.same_padding(size, stride)] == xla, (size, stride)


# --- (2) Initialisation and parameter count ---------------------------------------------


def test_initialisation_follows_ppo_init_network():
    torch.manual_seed(0)
    mine = P.ppo_params(make_policy(DUEL, DEFAULT_NET))
    ref = ppo.init_network(jax.random.key(0), ppo.PPOConfig())
    assert set(mine) == {f"{layer}/{name}" for layer in ref for name in ("w", "b")}
    gains = {"policy": 1e-4, "value": 1.0}
    for layer, ref_layer in ref.items():
        w, b = mine[f"{layer}/w"], mine[f"{layer}/b"]
        assert w.shape == ref_layer["w"].shape and b.shape == ref_layer["b"].shape
        assert (b == 0).all()
        std = (gains.get(layer, 2.0) / np.prod(w.shape[:-1])) ** 0.5  # sqrt(gain / fan_in)
        assert w.std() == pytest.approx(std, rel=0.25), layer
        assert w.std() == pytest.approx(float(ref_layer["w"].std()), rel=0.25), layer
        assert abs(w.mean()) < 5 * std / np.sqrt(w.size)  # centred


@pytest.mark.parametrize("case", CASES)
def test_parameter_count_equals_ppo(case):
    game, net = CASES[case]
    n_torch = sum(p.numel() for p in make_policy(game, net).parameters())
    ref = ppo.init_network(jax.random.key(0), ppo_cfg(game, net))
    assert n_torch == sum(x.size for x in jax.tree.leaves(ref))
    if case == "duel":  # the default run's network, built from a real PPOConfig
        assert n_torch == sum(
            x.size for x in jax.tree.leaves(ppo.init_network(jax.random.key(0), ppo.PPOConfig()))
        )


# --- (3) Masking with PufferLib's own sampling ------------------------------------------


@pytest.mark.parametrize("game", [DUEL, STANDARD4], ids=["duel", "standard4"])
def test_sample_logits_never_picks_an_illegal_move_and_scores_the_masked_policy(game, tmp_path):
    pufferlib_pytorch = pytest.importorskip("pufferlib.pytorch")
    data = agent_data(game)
    policy = randomize(make_policy(game, DEFAULT_NET))
    with torch.no_grad():
        policy.policy_head.bias.copy_(torch.tensor([20.0, 0.0, 0.0, 0.0]))  # UP, unless masked
        P.export_ppo_checkpoint(policy, tmp_path)
        logits, _ = policy(data.obs)
    ref_logits, _ = reference(tmp_path, game, DEFAULT_NET, data.float_obs)
    assert not data.mask[:, 0].all(), "want states where UP is illegal"

    rows = np.arange(len(data.obs))
    torch.manual_seed(0)
    for _ in range(3):
        action, logprob, entropy = pufferlib_pytorch.sample_logits(logits)
        assert data.mask[rows, action.numpy()].all()
        ref_logprob = ppo.masked_log_probs(jnp.asarray(ref_logits), jnp.asarray(data.mask))
        np.testing.assert_allclose(
            logprob.numpy(), np.asarray(ref_logprob)[rows, action.numpy()], rtol=1e-4, atol=1e-4
        )
        ref_entropy = ppo.masked_entropy(ref_logprob, jnp.asarray(data.mask))
        np.testing.assert_allclose(entropy.numpy(), np.asarray(ref_entropy), rtol=1e-4, atol=1e-4)
    assert (torch.softmax(logits, -1).numpy()[~data.mask] == 0).all()
    # A trained step through the masked logits: finite gradients everywhere.
    _, logprob, entropy = pufferlib_pytorch.sample_logits(policy(data.obs)[0], action=action)
    (logprob.mean() + entropy.mean() + policy(data.obs)[1].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters())


def test_dead_agents_and_autocast(tmp_path):
    """A dead agent's observation (board 0, mask all ones) is a plain input; bf16 autocast works."""
    game = DUEL
    policy = randomize(make_policy(game, DEFAULT_NET))
    size = env_like(game).obs_size
    dead = torch.zeros(3, size, dtype=torch.uint8)
    dead[:, -4:] = 1
    with torch.no_grad():
        logits, value = policy(dead)
        assert torch.isfinite(logits).all() and (logits > P.MASKED_LOGIT / 2).all()
        data = agent_data(game)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            half_logits, half_value = policy(data.obs)
        full_logits, full_value = policy(data.obs)
    assert half_logits.dtype == torch.float32 and half_value.dtype == torch.float32
    assert ((half_logits == P.MASKED_LOGIT) == (full_logits == P.MASKED_LOGIT)).all()
    legal = torch.from_numpy(data.mask)
    torch.testing.assert_close(half_logits[legal], full_logits[legal], rtol=0.1, atol=0.1)
    torch.testing.assert_close(half_value, full_value, rtol=0.1, atol=0.1)


# --- (4) Export and load ----------------------------------------------------------------


@pytest.mark.parametrize("case", ["duel-small-net", "wrapped-even-board"])
def test_export_load_round_trip(case, tmp_path):
    game, net = CASES[case]
    data = agent_data(game)
    policy = randomize(make_policy(game, net))
    run_dir = tmp_path / "nested" / "run"  # created
    path = P.export_ppo_checkpoint(
        policy, run_dir, max_turns=77, trainer="pufferlib", total_timesteps=123_456
    )
    assert path == str(run_dir / "params.npz")
    assert sorted(os.listdir(run_dir)) == ["config.json", "params.npz"]  # no temp files left

    with np.load(path) as f:
        template = ppo.init_network(jax.random.key(0), ppo_cfg(game, net))
        shapes = {
            f"{layer}/{name}": leaf.shape
            for layer, leaves in template.items()
            for name, leaf in leaves.items()
        }
        assert set(f.files) == set(shapes)
        for key, shape in shapes.items():
            assert f[key].shape == shape and f[key].dtype == np.float32, key

    loaded = P.load_ppo_checkpoint(run_dir, env_like(game))
    assert isinstance(loaded, P.BattlesnakePolicy)
    assert (
        loaded.conv_channels == net["conv_channels"] and loaded.conv_strides == net["conv_strides"]
    )
    assert loaded.hidden_size == net["hidden"]
    for (name, a), (_, b) in zip(
        policy.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    with torch.no_grad():
        for x, y in zip(policy(data.obs), loaded(data.obs), strict=True):
            assert torch.equal(x, y)

    # config.json: what ppo.py's load_config reads, plus the PufferLib keys it ignores.
    record = json.loads((run_dir / "config.json").read_text())
    assert record["trainer"] == "pufferlib" and record["total_timesteps"] == 123_456
    cfg = ppo.load_config(str(run_dir))
    assert cfg.conv_channels == net["conv_channels"] and cfg.conv_strides == net["conv_strides"]
    assert (cfg.hidden, cfg.max_turns) == (net["hidden"], 77)

    # Exporting again replaces the files; load_ppo_checkpoint reads whatever ppo.py wrote too.
    other = randomize(make_policy(game, net), seed=5)
    P.export_ppo_checkpoint(other, run_dir)
    assert json.loads((run_dir / "config.json").read_text())["max_turns"] == P.DEFAULT_MAX_TURNS
    reloaded = P.load_ppo_checkpoint(run_dir, env_like(game))
    assert not torch.equal(reloaded.dense.weight, policy.dense.weight)
    assert torch.equal(reloaded.dense.weight, other.dense.weight)


def test_loading_a_ppo_py_checkpoint(tmp_path):
    """A run directory written by baselines/ppo.py itself loads, and its network is played."""
    cfg = ppo.PPOConfig(**SMALL_NET)
    params = ppo.init_network(jax.random.key(4), cfg)
    params = jax.tree.map(
        lambda x: x + 0.05 * jax.random.normal(jax.random.key(5), x.shape), params
    )
    ppo.save_config(str(tmp_path), cfg)
    ppo.save_checkpoint(str(tmp_path), params)
    data = agent_data(DUEL)
    policy = P.load_ppo_checkpoint(tmp_path, env_like(DUEL))
    logits, value = ppo.network(params, jnp.asarray(data.float_obs), cfg)
    with torch.no_grad():
        got_logits, got_value = policy(data.obs)
    np.testing.assert_allclose(
        got_logits.numpy(),
        np.where(data.mask, np.asarray(logits), ppo.MASKED_LOGIT),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(got_value.numpy()[:, 0], np.asarray(value), rtol=1e-4, atol=1e-4)


def test_state_dict_round_trip_and_scale_buffer():
    """PufferLib saves and restores ``state_dict()``; the scale belongs to the env, not to it."""
    policy = randomize(make_policy(DUEL, SMALL_NET))
    state = policy.state_dict()
    assert "channel_scale" not in state
    np.testing.assert_array_equal(
        policy.channel_scale.numpy(), obs_format.channel_scale(DUEL.width, DUEL.height)
    )
    fresh = make_policy(DUEL, SMALL_NET, seed=9)
    fresh.load_state_dict(state)
    data = agent_data(DUEL)
    with torch.no_grad():
        assert torch.equal(policy(data.obs)[0], fresh(data.obs)[0])


# --- (5) The exported checkpoint plays in slinky -----------------------------------------


@pytest.mark.parametrize(
    ("game", "net"),
    [(DUEL, DEFAULT_NET), (DUEL, SMALL_NET), (DUEL, STRIDES_1_2), (STANDARD4, DEFAULT_NET)],
    ids=["duel", "duel-small-net", "duel-strides-1-2", "standard4"],
)
def test_exported_checkpoint_plays_the_torch_policys_moves(game, net, tmp_path):
    data = agent_data(game)
    policy = randomize(make_policy(game, net, seed=2), seed=3)
    P.export_ppo_checkpoint(policy, tmp_path, max_turns=1000)
    agent = make_agent(f"ppo:{tmp_path}", game)  # the default ppo mode is greedy
    assert agent.needs_obs and "greedy" in agent.description

    states = data.states
    keys = jax.random.split(jax.random.key(0), len(states.turn))
    slinky_moves = np.asarray(jax.vmap(agent.policy)(keys, states, data.timesteps))[data.alive]

    with torch.no_grad():
        logits, _ = policy(data.obs)
    best, gap = masked_argmax(logits.numpy())
    assert data.mask[np.arange(len(best)), slinky_moves].all()
    decided = gap > 1e-4  # a float32 near-tie may break either way
    assert decided.mean() > 0.99
    np.testing.assert_array_equal(slinky_moves[decided], best[decided])
    assert len(set(best.tolist())) > 1, "want a policy that plays different moves"


# --- (6) Argument checks ----------------------------------------------------------------


def test_construction_and_loading_errors(tmp_path):
    env = env_like(DUEL)
    bad_channels = types.SimpleNamespace(**{**vars(env), "board_shape": (21, 21, 12)})
    with pytest.raises(ValueError, match="13 channels"):
        P.BattlesnakePolicy(bad_channels)
    with pytest.raises(ValueError, match="obs_size"):
        P.BattlesnakePolicy(types.SimpleNamespace(**{**vars(env), "obs_size": env.obs_size + 1}))
    with pytest.raises(ValueError, match="channel_scale"):
        P.BattlesnakePolicy(types.SimpleNamespace(**{**vars(env), "channel_scale": np.ones(12)}))
    with pytest.raises(ValueError, match="same length"):
        P.BattlesnakePolicy(env, conv_channels=(4, 4), conv_strides=(1,))
    with pytest.raises(ValueError, match="conv_strides"):
        P.BattlesnakePolicy(env, conv_channels=(4,), conv_strides=(0,))
    with pytest.raises(ValueError, match="hidden"):
        P.BattlesnakePolicy(env, hidden=0)
    # Lists (as in a config file) are fine.
    assert P.BattlesnakePolicy(env, [4], [2], 8).conv_strides == (2,)

    policy = make_policy(DUEL, SMALL_NET)
    with pytest.raises(ValueError, match="size"):
        policy(torch.zeros(2, env.obs_size - 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="contradicts"):
        P.export_ppo_checkpoint(policy, tmp_path, hidden=64)
    with pytest.raises(ValueError, match="contradicts"):
        P.export_ppo_checkpoint(policy, tmp_path, conv_strides=[1])
    with pytest.raises(ValueError, match="max_turns"):
        P.export_ppo_checkpoint(policy, tmp_path, max_turns=0)
    assert not os.listdir(tmp_path)  # nothing half written
    P.export_ppo_checkpoint(
        policy, tmp_path, conv_strides=(2,), hidden=np.int64(32), mask=np.ones(2)
    )

    # A checkpoint of another board or network is refused, naming the parameter.
    with pytest.raises(ValueError, match=r"hidden/w: checkpoint shape"):
        P.load_ppo_checkpoint(tmp_path, env_like(WRAPPED))
    with np.load(tmp_path / "params.npz") as f:
        arrays = {k: f[k] for k in f.files}
    del arrays["value/b"]
    with pytest.raises(ValueError, match=r"missing parameters \['value/b'\]"):
        P.set_ppo_params(policy, arrays)
    arrays["value/b"] = np.zeros(1, np.float32)
    arrays["value/extra"] = np.zeros(1, np.float32)
    with pytest.raises(ValueError, match="unexpected parameters"):
        P.set_ppo_params(policy, arrays)
