"""The numpy PPO port (deploy/lambda) against the JAX agent, and its Lambda handler."""

from __future__ import annotations

import base64
import importlib
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from slinky.agents import load_ppo_module, make_agent, make_env, parse_agent
from slinky.engine_json import state_to_engine
from slinky.types import GameConfig

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_DIR = ROOT / "deploy" / "lambda"
CHECKPOINT = ROOT / "baselines" / "checkpoints" / "ppo-duel-seed0"
sys.path.insert(0, str(LAMBDA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snake_numpy as sn  # noqa: E402
from test_server import api_request  # noqa: E402

DUEL = GameConfig()


def positions(n: int, seed: int) -> list:
    """``n`` duel states with both snakes alive, from mixed PPO / heuristic / random play.

    Some have a snake on 1 health, to exercise the starvation rule of the mask.
    """
    env = make_env(DUEL, True)
    agents = [make_agent(a, DUEL) for a in ("ppo", "heuristic", "random_legal")]
    policies = [jax.jit(a.policy) for a in agents]
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    out = []
    while len(out) < n:
        key, k = jax.random.split(key)
        state, ts = env.reset(k)
        mine, theirs = rng.choice(3, 2)
        for _ in range(int(rng.integers(0, 150))):
            key, k1, k2, k3 = jax.random.split(key, 4)
            a = policies[mine](k1, state, ts)
            b = policies[theirs](k2, state, ts)
            state, ts = step(k3, state, a.at[1].set(b[1]))
            if state.done:
                break
        if not bool(state.alive.all()):
            continue
        if rng.random() < 0.1:
            state = state._replace(health=state.health.at[int(rng.integers(2))].set(1))
        out.append(state)
    return out


@pytest.fixture(scope="module")
def jax_ppo():
    ppo = load_ppo_module()
    spec = parse_agent("ppo")
    cfg = ppo.load_config(str(spec.checkpoint))
    params = ppo.load_params(str(spec.checkpoint), cfg)
    env = make_env(DUEL, True)
    return (
        jax.jit(env.observe),
        jax.jit(env.action_mask),
        jax.jit(lambda o: ppo.network(params, o, cfg)[0]),
    )


def test_matches_the_jax_agent(jax_ppo) -> None:
    observe, mask_fn, logits_fn = jax_ppo
    net = sn.PPO(str(CHECKPOINT / "params.npz"))
    agent = make_agent("ppo", DUEL)
    env = make_env(DUEL, True)
    states = positions(300, seed=0)
    near_ties = 0
    for i, state in enumerate(states):
        obs, mask, logits = observe(state), mask_fn(state), logits_fn(observe(state))
        ts = env._timestep(state, np.zeros(2, np.float32), np.asarray(False))
        want = np.asarray(agent.policy(jax.random.key(0), state, ts))
        d = state_to_engine(state, DUEL)
        for seat in (0, 1):
            req = api_request(d, seat, game_id=f"g{i}")
            board = sn.Board(req)
            assert sn.is_duel(req)
            np.testing.assert_array_equal(sn.observation(board), np.asarray(obs[seat]))
            np.testing.assert_array_equal(sn.legal_moves(board), np.asarray(mask[seat]))
            ours = net.logits(sn.observation(board))
            np.testing.assert_allclose(ours, np.asarray(logits[seat]), rtol=1e-4, atol=1e-4)
            legal = np.sort(np.where(np.asarray(mask[seat]), np.asarray(logits[seat]), -np.inf))
            if legal[-1] - legal[-2] < 1e-3:
                near_ties += 1  # float32 rounding may legitimately pick the other move
                continue
            assert net.move(board) == int(want[seat])
    assert near_ties < 10  # 600 positions; a handful at most


@pytest.fixture(scope="module")
def handler(monkeypatch_module):
    monkeypatch_module.setenv("PPO_PARAMS", str(CHECKPOINT / "params.npz"))
    sys.modules.pop("lambda_function", None)
    return importlib.import_module("lambda_function").lambda_handler


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def event(method: str, path: str, body: dict | bytes | None = None, b64: bool = False) -> dict:
    raw = (
        body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else b""
    )
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "body": base64.b64encode(raw).decode() if b64 else raw.decode(),
        "isBase64Encoded": b64,
    }


def test_handler_endpoints(handler) -> None:
    out = handler(event("GET", "/ppo"), None)
    assert out["statusCode"] == 200 and json.loads(out["body"])["apiversion"] == "1"
    state = positions(1, seed=5)[0]
    req = api_request(state_to_engine(state, DUEL), 0, game_id="h")
    for b64 in (False, True):
        out = handler(event("POST", "/ppo/move", req, b64), None)
        assert out["statusCode"] == 200
        move = json.loads(out["body"])["move"]
        assert move == sn.MOVE_NAMES[sn.PPO(str(CHECKPOINT / "params.npz")).move(sn.Board(req))]
    assert handler(event("POST", "/ppo/start", req), None)["statusCode"] == 200
    assert handler(event("POST", "/ppo/end", req), None)["statusCode"] == 200
    assert handler(event("POST", "/dqn/move", req), None)["statusCode"] == 404
    assert handler(event("GET", "/ppo/move"), None)["statusCode"] == 404
    assert handler(event("POST", "/ppo/move", b"{nope"), None)["statusCode"] == 400
    assert handler(event("POST", "/ppo/move", b"x" * 70000), None)["statusCode"] == 400


def test_handler_other_games_and_junk(handler) -> None:
    four = GameConfig(num_snakes=4)
    env = make_env(four, False)
    s4 = env.reset(jax.random.key(3))[0]
    req = api_request(state_to_engine(s4, four), 0, game_id="four")
    assert not sn.is_duel(req)
    out = handler(event("POST", "/ppo/move", req), None)
    assert json.loads(out["body"])["move"] in sn.MOVE_NAMES
    # Parses, has a game id, but no usable board: still a move.
    out = handler(event("POST", "/ppo/move", {"game": {"id": "x"}, "board": 3}), None)
    assert out["statusCode"] == 200 and json.loads(out["body"])["move"] == "up"
