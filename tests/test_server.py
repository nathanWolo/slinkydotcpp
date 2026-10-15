"""The Battlesnake API server: request conversion, agent choice and HTTP endpoints."""

from __future__ import annotations

import dataclasses
import json
import threading
import urllib.error
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import modes
from slinky.agents import make_agent, make_env, parse_agent
from slinky.engine_json import state_to_engine
from slinky.rules import royale_hazards, rules_step
from slinky.server import (
    MODE_CONFIGS,
    MOVE_NAMES,
    UNKNOWN_SIDE,
    Mover,
    Server,
    game_config,
    make_handler,
    parse_snake,
    request_state,
    request_to_engine,
    royale_sides,
    safe_move,
)
from slinky.types import GameConfig, Ruleset, State

DUEL = GameConfig()
# A small multi-mode checkpoint trained on all four modes (no search: fast to compile).
MODES_AGENT = "ppo_modes:baselines/checkpoints/ppo-modes-seed0"


def api_request(
    d: dict,
    you: int,
    game_id: str = "g",
    ruleset: str = "standard",
    game_map: str = "standard",
    settings: dict | None = None,
) -> dict:
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
        "game": {
            "id": game_id,
            "ruleset": {"name": ruleset, "version": "v1.2.3", "settings": settings or {}},
            "map": game_map,
            "source": "custom",
            "timeout": 500,
        },
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
    # Rulesets the agents don't model get the safe-move picker, and nothing compiles.
    assert mover.move(api_request(d, 0, ruleset="wrapped"))[1] == "safe_move"
    assert set(mover._fns) == {DUEL, config} and not mover._pending


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


# --- Every game mode -------------------------------------------------------------------


def official_settings(config: GameConfig) -> dict:
    """``game.ruleset.settings`` as the API sends it (every setting, always)."""
    return {
        "foodSpawnChance": config.food_spawn_chance,
        "minimumFood": config.minimum_food,
        "hazardDamagePerTurn": config.hazard_damage_per_turn,
        "royale": {"shrinkEveryNTurns": config.shrink_every_n_turns},
        "squad": {
            "allowBodyCollisions": False,
            "sharedElimination": False,
            "sharedHealth": False,
            "sharedLength": False,
        },
    }


def mode_request(
    state: State,
    config: GameConfig,
    you: int = 0,
    game_id: str = "g",
    ruleset: str | None = None,
    game_map: str | None = None,
) -> dict:
    """The API move request of a slinky ``state`` in a game with ``config``."""
    return api_request(
        state_to_engine(state, config),
        you,
        game_id=game_id,
        ruleset=ruleset or config.ruleset.value,
        game_map=game_map or config.map,
        settings=official_settings(config),
    )


# (label, config): the default config of each mode, and royale with other settings.
GAMES = [
    *MODE_CONFIGS.items(),
    (
        "royale-fast",
        dataclasses.replace(
            MODE_CONFIGS["royale"],
            shrink_every_n_turns=5,
            hazard_damage_per_turn=30,
            food_spawn_chance=25,
            minimum_food=2,
        ),
    ),
]
WANTED_TURNS = (0, 1, 4, 5, 6, 13, 24, 25, 26, 33, 49, 50, 51, 64)


def played_states(config: GameConfig, seed: int = 0, games: int = 12) -> dict[int, State]:
    """Undecided states of ``random_legal`` self-play at :data:`WANTED_TURNS`, by turn."""
    env = make_env(config, False)
    agent = make_agent("random_legal", config)
    policy, step = jax.jit(agent.policy), jax.jit(env.step)
    key = jax.random.key(seed)
    found: dict[int, State] = {}
    for _ in range(games):
        key, k = jax.random.split(key)
        state, ts = env.reset(k)
        while not bool(state.done) and int(state.turn) <= max(WANTED_TURNS):
            if int(state.turn) in WANTED_TURNS:
                found.setdefault(int(state.turn), state)
            key, k1, k2 = jax.random.split(key, 3)
            state, ts = step(k2, state, policy(k1, state, ts))
        if len(found) == len(WANTED_TURNS):
            break
    return found


@pytest.fixture(scope="module")
def mode_games() -> dict[str, tuple[GameConfig, dict[int, State]]]:
    return {label: (config, played_states(config)) for label, config in GAMES}


def seat_order(state: State, you: int) -> list[int]:
    """The seats of ``state`` in the server's order: ``you``, the other living, the dead."""
    alive = np.asarray(state.alive)
    n = len(alive)
    others = [i for i in range(n) if i != you and alive[i]]
    return [you, *others, *(i for i in range(n) if not alive[i])]


SEAT_FIELDS = ("body", "head", "length", "health", "alive", "last_move", "elim_cause", "elim_turn")


def reorder(state: State, order: list[int]) -> State:
    idx = jnp.asarray(order)
    return state._replace(**{f: getattr(state, f)[idx] for f in SEAT_FIELDS})


def assert_same_state(got: State, want: State, config: GameConfig) -> None:
    """``got`` (from a request) is ``want``, but for what the API can't say.

    The API lists no dead snakes: their seats only agree on being dead (and empty). Of the
    royale shrink sides only those already shown are known (see ``royale_sides``).
    """
    got, want = jax.device_get(got), jax.device_get(want)
    alive = np.asarray(want.alive)
    for field in State._fields:
        if field == "map_state":
            continue
        a, b = np.asarray(getattr(got, field)), np.asarray(getattr(want, field))
        assert a.dtype == b.dtype and a.shape == b.shape, field
        if field in SEAT_FIELDS and field not in ("body", "alive"):
            a, b = a[alive], b[alive]
        np.testing.assert_array_equal(a, b, err_msg=field)
    np.testing.assert_array_equal(np.asarray(got.elim_cause) != 0, ~alive)
    if config.ruleset != Ruleset.ROYALE:
        assert got.map_state == () and want.map_state == ()
        return
    sides, true_sides = np.asarray(got.map_state.sides), np.asarray(want.map_state.sides)
    assert sides.dtype == true_sides.dtype and sides.shape == true_sides.shape
    period, turn = config.shrink_every_n_turns, int(want.turn)
    shrinks = turn // period
    assert (sides[shrinks:] == UNKNOWN_SIDE).all()
    if not np.asarray(want.hazard).all():  # the order of the shown shrinks is not known
        np.testing.assert_array_equal(np.sort(sides[:shrinks]), np.sort(true_sides[:shrinks]))
    # The same zone until the next shrink, which nobody can know.
    for t in range(shrinks * period, (shrinks + 1) * period):
        np.testing.assert_array_equal(
            royale_hazards(sides, t, config), royale_hazards(true_sides, t, config), err_msg=str(t)
        )


@pytest.mark.parametrize("label", [label for label, _ in GAMES])
def test_requests_rebuild_the_game(label: str, mode_games) -> None:
    """Config and state from the API request of every mode, as a search would simulate them."""
    config, states = mode_games[label]
    assert len(states) >= 6 and {0, 1, 4, 5, 6} <= set(states), sorted(states)
    if config.ruleset == Ruleset.ROYALE:  # some zone has shrunk
        assert any(np.asarray(s.hazard).any() for s in states.values())
    key = jax.random.key(1)
    step = jax.jit(rules_step, static_argnums=2)
    for turn, state in states.items():
        alive = np.flatnonzero(np.asarray(state.alive))
        for you in {int(alive[0]), int(alive[-1])}:
            req = mode_request(state, config, you=you, game_id=label)
            # The game's start (all snakes) says the snake count; the request lists the living.
            assert game_config(req, config.num_snakes) == config
            assert (game_config(req) == config) == (len(alive) == config.num_snakes)
            got, want = request_state(req, config), reorder(state, seat_order(state, you))
            assert_same_state(got, want, config)
            # One rules step from both (a search's model of the next turn) agrees too, unless
            # it shows a new shrink.
            if (turn + 1) % config.shrink_every_n_turns or config.ruleset != Ruleset.ROYALE:
                key, k = jax.random.split(key)
                actions = jax.random.randint(k, (config.num_snakes,), 0, 4)
                assert_same_state(step(got, actions, config), step(want, actions, config), config)
            if config.ruleset == Ruleset.ROYALE:  # the other two ways to play royale
                for ruleset, game_map in (("royale", "standard"), ("standard", "royale")):
                    other = mode_request(state, config, you, label, ruleset, game_map)
                    assert game_config(other, config.num_snakes) == config
                    assert_same_state(request_state(other, config), want, config)


def test_settings() -> None:
    req = example_request_for(MODE_CONFIGS["royale"])
    req["game"]["ruleset"]["settings"] = {
        "foodSpawnChance": "20",  # the engine CLI sends strings
        "minimumFood": True,  # junk keeps the default
        "hazardDamagePerTurn": 10**12,
        "shrinkEveryNTurns": 7,  # the CLI's flat name
    }
    want = dataclasses.replace(MODE_CONFIGS["royale"], food_spawn_chance=20, shrink_every_n_turns=7)
    assert game_config(req) == want
    req["game"]["ruleset"]["settings"]["royale"] = {"shrinkEveryNTurns": 9}  # the API's wins
    assert game_config(req).shrink_every_n_turns == 9
    req["game"]["ruleset"]["settings"]["royale"] = {"shrinkEveryNTurns": 0}  # an engine error
    assert game_config(req) is None
    # Only royale reads the shrink period: other modes keep their one config.
    for mode in ("standard", "duel", "constrictor"):
        req = example_request_for(MODE_CONFIGS[mode])
        req["game"]["ruleset"]["settings"] = {"royale": {"shrinkEveryNTurns": 5}}
        assert game_config(req) == MODE_CONFIGS[mode]
        del req["game"]["ruleset"]["settings"]
        assert game_config(req) == MODE_CONFIGS[mode]


def example_request_for(config: GameConfig) -> dict:
    state = make_env(config, False).reset(jax.random.key(0))[0]
    return mode_request(state, config)


def test_games_without_a_model() -> None:
    req = example_request_for(MODE_CONFIGS["standard"])
    for ruleset, game_map in (
        ("wrapped", "standard"),
        ("wrapped_constrictor", "standard"),
        ("solo", "standard"),
        ("constrictor", "royale"),
        ("standard", "empty"),
        ("royale", "arcade_maze"),
        ("squad", "standard"),
    ):
        req["game"]["ruleset"]["name"], req["game"]["map"] = ruleset, game_map
        assert game_config(req) is None, (ruleset, game_map)
    # Standard games on other boards keep their config (the fallback agent plays them).
    seven = GameConfig(width=7, height=7, num_snakes=3)
    assert game_config(example_request_for(seven)) == seven


def test_royale_sides_when_the_zone_is_gone() -> None:
    config = MODE_CONFIGS["royale"]
    everything = [[x, y] for x in range(11) for y in range(11)]
    for turn in (11 * 25, 21 * 25, 40 * 25):
        sides = royale_sides(everything, turn, config).sides
        assert np.asarray(royale_hazards(sides, turn, config)).all()
        assert np.asarray(royale_hazards(sides, turn + 1, config)).all()


@pytest.fixture(scope="module")
def modes_mover() -> Mover:
    return Mover(MODES_AGENT)  # precompiles all four modes


def test_multi_mode_agent_plays_every_mode(modes_mover: Mover, mode_games) -> None:
    """In each mode the server answers with the multi-mode agent's move, told that mode."""
    name = parse_agent(MODES_AGENT).name
    assert set(modes_mover._fns) == set(MODE_CONFIGS.values())
    assert all(entry is not None and entry[0] == name for entry in modes_mover._fns.values())
    for label, (config, states) in mode_games.items():
        if label not in MODE_CONFIGS:
            continue
        policy = jax.jit(make_agent(MODES_AGENT, config).policy)  # greedy: keys change nothing
        env = make_env(config, True)
        checked = 0
        for turn, state in states.items():
            if not bool(state.alive.all()):
                continue
            req = mode_request(state, config, game_id=f"modes-{label}-{turn}")
            assert modes_mover.start(req) == name
            ts = env._timestep(state, np.zeros(config.num_snakes, np.float32), np.asarray(False))
            want = int(policy(jax.random.key(0), state, ts)[0])
            assert modes_mover.move(req) == (want, name), (label, turn)
            checked += 1
        assert checked >= 2, label
    # A dead snake's seat is padded back in: the same compiled policy plays on.
    for label in ("standard", "royale", "constrictor"):
        config, states = mode_games[label]
        dying = [s for s in states.values() if not bool(s.alive.all())]
        if dying:
            modes_mover.start(mode_request(states[0], config, game_id=f"padded-{label}"))
            you = int(np.flatnonzero(np.asarray(dying[-1].alive))[0])
            req = mode_request(dying[-1], config, you=you, game_id=f"padded-{label}")
            assert modes_mover.move(req)[1] == name
    assert set(modes_mover._fns) == set(MODE_CONFIGS.values()) and not modes_mover._pending


def test_multi_mode_agent_falls_back_outside_the_modes(modes_mover: Mover) -> None:
    # A wrapped game: no model, the safe-move picker, nothing compiled.
    req = example_request_for(DUEL)
    req["game"]["ruleset"]["name"] = "wrapped"
    assert modes_mover.start(req) == "safe_move"
    assert modes_mover.move(req)[1] == "safe_move"
    # Another board: the fallback agent once compiled.
    seven = example_request_for(GameConfig(width=7, height=7))
    assert modes_mover.start(seven) == "safe_move until compiled"
    assert modes_mover.wait_idle(timeout=300)
    assert modes_mover.move(seven)[1] == "heuristic"


def test_other_agents_in_the_modes() -> None:
    """A duel-only baseline and a single-mode specialist leave the other modes to the fallback."""
    royale = example_request_for(MODE_CONFIGS["royale"])
    duel = example_request_for(DUEL)
    ppo = Mover("ppo", precompile=())
    assert not ppo.multi_mode and ppo.plays(DUEL)
    assert not any(ppo.plays(c) for m, c in MODE_CONFIGS.items() if m != "duel")
    specialist = Mover("ppo_modes:baselines/checkpoints/ppo-modes-duel-seed0", precompile=())
    assert specialist.multi_mode
    for mover, name in ((ppo, "ppo"), (specialist, specialist.agent)):
        mover.start(royale)
        mover.start(duel)
        assert mover.wait_idle(timeout=300)
        assert mover.move(royale)[1] == "heuristic"
        assert mover.move(duel)[1] == name
