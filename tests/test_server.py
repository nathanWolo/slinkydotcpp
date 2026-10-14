"""The Battlesnake API server: request conversion, agent choice and HTTP endpoints."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import jax
import numpy as np
import pytest

from slinky.agents import make_agent, make_env
from slinky.engine_json import state_to_engine
from slinky.server import (
    MOVE_NAMES,
    Mover,
    Server,
    make_handler,
    parse_snake,
    request_to_engine,
    safe_move,
)
from slinky.types import GameConfig

DUEL = GameConfig()


def api_request(d: dict, you: int, game_id: str = "g", ruleset: str = "standard") -> dict:
    """A Battlesnake API move request for engine dict ``d`` from snake ``you``'s side."""
    snakes = [
        {
            "id": s["id"],
            "name": s["id"],
            "health": s["health"],
            "body": [{"x": x, "y": y} for x, y in s["body"]],
            "head": {"x": s["body"][0][0], "y": s["body"][0][1]},
            "length": len(s["body"]),
        }
        for s in d["snakes"]
        if not s.get("eliminated_cause")
    ]
    return {
        "game": {"id": game_id, "ruleset": {"name": ruleset}, "map": "standard", "timeout": 500},
        "turn": d["turn"],
        "board": {
            "width": d["width"],
            "height": d["height"],
            "food": [{"x": x, "y": y} for x, y in d["food"]],
            "hazards": [{"x": x, "y": y} for x, y in d["hazards"]],
            "snakes": snakes,
        },
        "you": next(s for s in snakes if s["id"] == d["snakes"][you]["id"]),
    }


def midgame_states(config: GameConfig, n: int, seed: int = 0) -> list:
    """``n`` states with every snake alive, from heuristic self-play."""
    env = make_env(config, True)
    agent = make_agent("heuristic", config)
    policy, step = jax.jit(agent.policy), jax.jit(env.step)
    key = jax.random.key(seed)
    states = []
    while len(states) < n:
        key, k = jax.random.split(key)
        state, ts = env.reset(k)
        for _ in range(int(jax.random.randint(k, (), 5, 60))):
            key, k1, k2 = jax.random.split(key, 3)
            state, ts = step(k2, state, policy(k1, state, ts))
            if state.done:
                break
        if bool(state.alive.all()):
            states.append(state)
    return states


def test_request_puts_you_first_and_pads_the_dead() -> None:
    req = {
        "turn": 3,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 1, "y": 2}],
            "snakes": [
                {"id": "x", "health": 90, "body": [{"x": 0, "y": 0}]},
                {"id": "me", "health": 80, "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}]},
            ],
        },
        "you": {"id": "me"},
    }
    d = request_to_engine(req, num_snakes=4)
    assert [s["id"] for s in d["snakes"][:2]] == ["me", "x"]
    assert d["snakes"][0]["body"] == [[5, 5], [5, 4]]
    assert [s["eliminated_cause"] for s in d["snakes"][2:]] == ["snake-collision"] * 2
    assert d["food"] == [[1, 2]] and d["hazards"] == [] and d["turn"] == 3


@pytest.mark.parametrize("you", [0, 1])
def test_duel_moves_match_the_agent(you: int) -> None:
    """The server's move for either seat is the PPO agent's move in that seat.

    The keys differ, so this relies on the default PPO agent being greedy.
    """
    agent = make_agent("ppo", DUEL)
    env = make_env(DUEL, True)
    mover = Mover("ppo")
    for i, state in enumerate(midgame_states(DUEL, 6)):
        ts = env._timestep(state, np.zeros(2, np.float32), np.asarray(False))
        want = int(agent.policy(jax.random.key(0), state, ts)[you])
        req = api_request(state_to_engine(state, DUEL), you, game_id=f"g{i}")
        assert mover.move(req) == (want, "ppo")


def test_other_games_use_the_fallbacks() -> None:
    config = GameConfig(num_snakes=4)
    state = midgame_states(config, 1)[0]
    d = state_to_engine(state, config)
    mover = Mover("ppo")
    req = api_request(d, 0, game_id="four")
    assert mover.start(req) == "safe_move until compiled"
    assert mover.wait_idle(timeout=300)
    assert mover.start(req) == "heuristic"
    # A snake dies: the dead seat is padded back in and the same policy plays.
    del req["board"]["snakes"][3]
    action, name = mover.move(req)
    assert name == "heuristic" and 0 <= action < 4
    assert set(mover._fns) == {DUEL, config}
    mover.end(req)
    assert mover._games == {}
    assert mover.move(api_request(d, 0, ruleset="royale"))[1] == "safe_move"


def test_compiling_a_new_board_does_not_block_duels(monkeypatch: pytest.MonkeyPatch) -> None:
    mover = Mover("ppo")
    release = threading.Event()
    build = Mover._build

    def slow_build(self: Mover, config: GameConfig):
        assert release.wait(timeout=60)
        return build(self, config)

    monkeypatch.setattr(Mover, "_build", slow_build)
    seven = GameConfig(width=7, height=7)
    small = api_request(state_to_engine(midgame_states(seven, 1)[0], seven), 0, game_id="7x7")
    assert mover.start(small) == "safe_move until compiled"
    assert mover.move(small)[1] == "safe_move"  # while it compiles
    duel = api_request(state_to_engine(midgame_states(DUEL, 1)[0], DUEL), 0, game_id="duel")
    assert mover.move(duel)[1] == "ppo"  # not stuck behind the compilation
    release.set()
    assert mover.wait_idle(timeout=300)
    assert mover.move(small)[1] == "heuristic"


def test_oversized_lists_are_refused() -> None:
    req = api_request(state_to_engine(midgame_states(DUEL, 1)[0], DUEL), 0)
    req["board"]["food"] = [{"x": 0, "y": 0}] * 122
    with pytest.raises(ValueError, match="more food"):
        Mover("ppo", precompile=()).move(req)


def test_safe_move_avoids_walls_and_bodies() -> None:
    # Head in the corner (0, 0) facing up its own neck at (0, 1): only right is safe.
    me = {
        "id": "me",
        "health": 50,
        "length": 3,
        "head": {"x": 0, "y": 0},
        "body": [{"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2}],
    }
    req = {
        "game": {"ruleset": {"name": "royale"}},
        "board": {"width": 11, "height": 11, "snakes": [me]},
        "you": me,
    }
    assert MOVE_NAMES[safe_move(req)] == "right"


@pytest.fixture(scope="module")
def server():
    info = {
        "apiversion": "1",
        "author": "test",
        "color": "#123456",
        "head": "default",
        "tail": "default",
    }
    srv = Server(("127.0.0.1", 0), make_handler({"ppo": (Mover("ppo"), info)}))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_endpoints(server: str) -> None:
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        assert json.loads(r.read()) == {"snakes": ["/ppo"]}
    for path in ("/ppo", "/ppo/"):
        with urllib.request.urlopen(server + path, timeout=10) as r:
            assert json.loads(r.read())["apiversion"] == "1"
    snake = server + "/ppo"
    state = midgame_states(DUEL, 1, seed=3)[0]
    req = json.dumps(api_request(state_to_engine(state, DUEL), 0, game_id="http")).encode()
    assert _post(snake + "/start", req) == (200, {})
    code, body = _post(snake + "/move", req)
    assert code == 200 and body["move"] in MOVE_NAMES
    assert _post(snake + "/end", req) == (200, {})
    assert _post(snake + "/move", b"{not json")[0] == 400
    assert _post(snake + "/nope", req)[0] == 404
    assert _post(snake + "/move", b"[1, 2]")[0] == 400
    # Junk that parses: still a move, never an error or a hang.
    assert _post(snake + "/move", b'{"board": 3}') == (200, {"move": "up"})
    huge = json.loads(req)
    huge["board"]["width"] = huge["board"]["height"] = 99  # past MAX_SIDE: no compile
    code, body = _post(snake + "/move", json.dumps(huge).encode())
    assert code == 200 and body["move"] in MOVE_NAMES


def test_routes_are_per_snake(server: str) -> None:
    state = midgame_states(DUEL, 1, seed=3)[0]
    req = json.dumps(api_request(state_to_engine(state, DUEL), 0, game_id="route")).encode()
    assert _post(server + "/move", req)[0] == 404  # no snake at the root
    assert _post(server + "/dqn/move", req)[0] == 404  # not a snake on this server
    assert _post(server + "/ppo/move/extra", req)[0] == 404
    for path in ("/dqn", "/ppo/move"):  # GET only serves a snake's info
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + path, timeout=10)
        assert e.value.code == 404


def test_parse_snake() -> None:
    assert parse_snake("az-256") == ("az-256", "az-256")
    assert parse_snake("best=az-256:runs/loop/gen-02") == ("best", "az-256:runs/loop/gen-02")
    for bad in ("az-64:runs/x", "a b=ppo", "=ppo"):
        with pytest.raises(ValueError):
            parse_snake(bad)


def test_concurrent_moves(server: str) -> None:
    """Many games' moves at once (more than the default listen backlog of 5)."""
    from concurrent.futures import ThreadPoolExecutor

    reqs = [
        json.dumps(api_request(state_to_engine(st, DUEL), i % 2, game_id=f"c{i}")).encode()
        for i, st in enumerate(midgame_states(DUEL, 8, seed=9))
    ] * 4
    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(lambda r: _post(server + "/ppo/move", r), reqs))
    assert all(code == 200 and body["move"] in MOVE_NAMES for code, body in results)
