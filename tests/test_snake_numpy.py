"""The numpy port of the trained baselines (deploy/lambda) against the JAX agents, and its
Lambda handler."""

from __future__ import annotations

import base64
import importlib
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from slinky.agents import load_baseline_module, make_agent, make_env, parse_agent
from slinky.engine_json import state_to_engine
from slinky.types import GameConfig

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_DIR = ROOT / "deploy" / "lambda"
CHECKPOINTS = ROOT / "baselines" / "checkpoints"
sys.path.insert(0, str(LAMBDA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambda_function as lf  # noqa: E402  (loads no network until asked)
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


# Lambda snake -> the slinky.agents name of the same network.
AGENT_NAMES = {
    "ppo": "ppo",
    "reinforce": "reinforce",
    "dqn": "dqn",
    "rainbow": f"rainbow:{CHECKPOINTS / 'rainbow-long-seed0'}",
    "az-0": "az-0",
}


def numpy_net(name: str) -> sn.Net:
    d = CHECKPOINTS / lf.SNAKES[name][0]
    return sn.Net(str(d / "params.npz"), json.loads((d / "config.json").read_text()))


def jax_scores(name: str):
    """The JAX function of observations ``[N, ...]`` whose legal argmax the agent plays."""
    spec = parse_agent(AGENT_NAMES[name])
    mod = load_baseline_module(spec.kind)
    cfg = mod.load_config(str(spec.checkpoint))
    params = mod.load_params(str(spec.checkpoint), cfg)
    if spec.kind == "dqn":
        return jax.jit(lambda o: mod.q_network(params, o, cfg))
    if spec.kind == "rainbow":
        return jax.jit(lambda o: mod.q_values(mod.network(params, o, cfg), cfg))
    return jax.jit(lambda o: mod.network(params, o, cfg)[0])


@pytest.fixture(scope="module")
def states():
    return positions(150, seed=0)


def test_snakes_are_the_agents() -> None:
    assert set(lf.SNAKES) == set(AGENT_NAMES)


def test_encoding_matches_jax(states) -> None:
    env = make_env(DUEL, True)
    observe, mask_fn = jax.jit(env.observe), jax.jit(env.action_mask)
    for i, state in enumerate(states):
        obs, mask = observe(state), mask_fn(state)
        d = state_to_engine(state, DUEL)
        for seat in (0, 1):
            req = api_request(d, seat, game_id=f"g{i}")
            board = sn.Board(req)
            assert sn.is_duel(req)
            np.testing.assert_array_equal(sn.observation(board), np.asarray(obs[seat]))
            np.testing.assert_array_equal(sn.legal_moves(board), np.asarray(mask[seat]))


@pytest.mark.parametrize("name", list(AGENT_NAMES))
def test_moves_match_the_jax_agent(name, states) -> None:
    net, scores_fn = numpy_net(name), jax_scores(name)
    agent = make_agent(AGENT_NAMES[name], DUEL)
    env = make_env(DUEL, True)
    policy = jax.jit(agent.policy)
    near_ties = 0
    for i, state in enumerate(states):
        ts = env._timestep(state, np.zeros(2, np.float32), np.asarray(False))
        want = np.asarray(policy(jax.random.key(0), state, ts))
        scores, mask = np.asarray(scores_fn(ts.obs)), np.asarray(ts.action_mask)
        d = state_to_engine(state, DUEL)
        for seat in (0, 1):
            board = sn.Board(api_request(d, seat, game_id=f"g{i}"))
            ours = net.scores(sn.observation(board))
            np.testing.assert_allclose(ours, scores[seat], rtol=1e-4, atol=1e-5)
            legal = np.where(mask[seat], scores[seat], -np.inf)
            tol = 1e-4 * max(1.0, abs(legal.max()))
            move = net.move(board)
            if np.sort(legal)[-2] > legal.max() - tol:
                # A near-tie: float32 rounding may legitimately pick the other move,
                # but it must be one of the tied best.
                near_ties += 1
                assert legal[move] > legal.max() - tol, (name, i, seat)
                continue
            assert move == int(want[seat]), (name, i, seat)
    assert near_ties < 30  # of 300 positions (Rainbow's returns saturate near +-1)


@pytest.fixture(scope="module")
def handler(monkeypatch_module):
    monkeypatch_module.setenv("CHECKPOINTS", str(CHECKPOINTS))
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
    state = positions(1, seed=5)[0]
    req = api_request(state_to_engine(state, DUEL), 0, game_id="h")
    colors = set()
    for name in AGENT_NAMES:
        out = handler(event("GET", f"/{name}"), None)
        body = json.loads(out["body"])
        assert out["statusCode"] == 200 and body["apiversion"] == "1" and name in body["version"]
        colors.add(body["color"])
        assert handler(event("POST", f"/{name}/start", req), None)["statusCode"] == 200
        for b64 in (False, True):
            out = handler(event("POST", f"/{name}/move", req, b64), None)
            assert out["statusCode"] == 200
            want = sn.MOVE_NAMES[numpy_net(name).move(sn.Board(req))]
            assert json.loads(out["body"])["move"] == want
        assert handler(event("POST", f"/{name}/end", req), None)["statusCode"] == 200
    assert len(colors) == len(AGENT_NAMES)  # each snake looks different on the board
    assert handler(event("POST", "/nope/move", req), None)["statusCode"] == 404
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
