"""A Battlesnake API server for agents in the registry (:mod:`slinky.agents`).

It speaks the official Battlesnake HTTP API (version 1). Each agent is a
separate snake under its own path, so one server can host several and a
registered snake on play.battlesnake.com points at ``<server>/<name>``::

    python -m slinky.server --agent az-256 --agent ppo --port 8000
    python -m slinky.server --agent best=az-256:runs/loop/gen-02   # name=agent
    curl -s http://127.0.0.1:8000/az-256    # snake info

* ``GET /<name>`` returns the snake's info (API version, author, colour, head,
  tail); ``GET /`` lists the snakes.
* ``POST /<name>/move`` converts the request's board to a
  :class:`~slinky.types.State` with the requesting snake in seat 0 and answers
  with that seat's move.
* ``POST /<name>/start`` starts compiling the policy for the game's board if
  it is new (in the background; the duel is compiled at startup);
  ``POST /<name>/end`` logs the result.

A snake's name is the agent's registry name, or the ``name`` of
``--agent name=agent`` (letters, digits, ``.``, ``_`` and ``-``), which is
needed for agents whose registry name has a run directory in it.

The agent plays the games it can: the trained baselines read 11x11 egocentric
observations, so they only play 11x11 standard games with two snakes (the
Duels leaderboard), even where they could run. Any other standard game gets
the ``--fallback`` agent (the heuristic by default), compiled for that board;
until it is compiled, and in other rulesets (royale, wrapped, constrictor,
...), a plain safe-move picker plays.

It binds to ``127.0.0.1`` by default: put a tunnel or reverse proxy in front of
it to reach it from the internet. Requests run in parallel threads.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from slinky.agents import BASELINES, make_agent, make_env, parse_agent
from slinky.engine_json import state_from_engine
from slinky.types import ACTION_DELTAS, GameConfig

DUEL = GameConfig()  # the game the trained baselines learned: 11x11 standard, 2 snakes

MOVE_NAMES = ("up", "down", "left", "right")  # index = action, as in slinky.types
MAX_BODY = 1 << 16  # bytes; a duel request is under 2 KB
# Limits on what requests can make the server compile and remember: a new
# board shape costs about a second of compilation and some memory.
MAX_SIDE, MAX_SNAKES, MAX_CONFIGS, MAX_GAMES = 25, 8, 8, 1000

log = logging.getLogger("slinky.server")


def request_to_engine(req: dict[str, Any], num_snakes: int | None = None) -> dict[str, Any]:
    """The engine board-state dict for a Battlesnake API request, ``you`` in seat 0.

    The API lists only the living snakes. With ``num_snakes`` (the game's
    snake count at the start) the dead ones are padded back in as eliminated
    seats, so the board keeps one shape (and one compiled policy) all game.
    """
    board, you = req["board"], req["you"]
    snakes = sorted(board["snakes"], key=lambda s: s["id"] != you["id"])  # stable
    if not snakes or snakes[0]["id"] != you["id"]:
        raise ValueError("'you' is not on the board")
    dead = [
        {"id": f"gone-{i}", "body": [], "health": 0, "eliminated_cause": "snake-collision"}
        for i in range(max(0, (num_snakes or 0) - len(snakes)))
    ]
    return {
        "width": board["width"],
        "height": board["height"],
        "turn": req["turn"],
        "snakes": [
            {"id": s["id"], "body": [[p["x"], p["y"]] for p in s["body"]], "health": s["health"]}
            for s in snakes
        ]
        + dead,
        "food": [[p["x"], p["y"]] for p in board["food"]],
        "hazards": [[p["x"], p["y"]] for p in board.get("hazards", [])],
    }


def game_config(req: dict[str, Any], num_snakes: int | None = None) -> GameConfig | None:
    """The :class:`GameConfig` of a standard-ruleset request, or None for other rulesets."""
    game = req.get("game", {})
    if game.get("ruleset", {}).get("name", "standard") != "standard":
        return None
    if game.get("map", "standard") != "standard":
        return None
    board = req["board"]
    w, h, n = board["width"], board["height"], max(num_snakes or 0, len(board["snakes"]))
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE and 1 <= n <= MAX_SNAKES):
        return None
    return GameConfig(width=w, height=h, num_snakes=n)


def safe_move(req: dict[str, Any]) -> int:
    """A move that doesn't leave the board or enter a body segment that stays put.

    Only for rulesets the agents don't model; prefers moves away from bigger
    snakes' possible next heads.
    """
    board, you = req["board"], req["you"]
    w, h = board["width"], board["height"]
    blocked = {(p["x"], p["y"]) for s in board["snakes"] for p in s["body"][:-1]}
    danger = {
        (s["head"]["x"] + dx, s["head"]["y"] + dy)
        for s in board["snakes"]
        if s["id"] != you["id"] and s["length"] >= you["length"]
        for dx, dy in ACTION_DELTAS
    }
    hx, hy = you["head"]["x"], you["head"]["y"]
    wrapped = "wrapped" in req.get("game", {}).get("ruleset", {}).get("name", "")
    best, best_score = 0, -1
    for a, (dx, dy) in enumerate(ACTION_DELTAS):
        p = (hx + dx, hy + dy)
        if wrapped:
            p = (p[0] % w, p[1] % h)
        elif not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        score = 0 if p in blocked else 1 + (p not in danger)
        if score > best_score:
            best, best_score = a, score
    return best


def check_sizes(req: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the request lists more than its board can hold."""
    board = req["board"]
    cells = board["width"] * board["height"]
    if (
        len(board["food"]) > cells
        or len(board.get("hazards", [])) > 4 * cells  # hazards can stack
        or any(len(s["body"]) > cells + 3 for s in board["snakes"])
    ):
        raise ValueError("more food, hazards or body than the board holds")


class Mover:
    """Chooses moves: the main agent where it can play, the fallback elsewhere.

    A board shape is compiled once, in a background thread (one at a time), so a
    new game never stalls the moves of games already being played. Until its
    shape is ready a game gets :func:`safe_move`. The shapes in ``precompile``
    (the duel by default) are compiled before the constructor returns.
    """

    def __init__(
        self,
        agent: str,
        fallback: str = "heuristic",
        seed: int = 0,
        precompile: tuple[GameConfig, ...] = (DUEL,),
    ) -> None:
        self.agent = parse_agent(agent).name
        self.fallback = parse_agent(fallback).name
        # Compiled shapes; None marks a shape whose compilation failed.
        self._fns: dict[GameConfig, tuple[str, Any] | None] = {}
        self._pending: set[GameConfig] = set()
        self._games: dict[str, int] = {}  # game id -> snakes at the start
        self._key = jax.random.key(seed)
        self._moves = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._compile_lock = threading.Lock()
        for config in precompile:
            self._pending.add(config)
            self._compile(config)

    def _build(self, config: GameConfig) -> tuple[str, Any]:
        agent = None
        # The trained baselines only play the game they were trained on.
        if parse_agent(self.agent).kind not in BASELINES or config == DUEL:
            with contextlib.suppress(ValueError):  # a game it can't play
                name, agent = self.agent, make_agent(self.agent, config)
        if agent is None:
            name, agent = self.fallback, make_agent(self.fallback, config)
        env = make_env(config, agent.needs_obs)

        @jax.jit
        def act(key: jax.Array, state: Any) -> jax.Array:
            n = config.num_snakes
            ts = env._timestep(state, jnp.zeros(n, jnp.float32), jnp.asarray(False))
            return agent.policy(key, state, ts)[0]

        state = env.reset(jax.random.key(0))[0]
        act(jax.random.key(0), state).block_until_ready()  # compile now
        return name, act

    def _compile(self, config: GameConfig) -> None:
        entry = None
        try:
            with self._compile_lock:
                t = time.perf_counter()
                entry = self._build(config)
                log.info(
                    "compiled %s for %dx%d, %d snakes in %.1f s",
                    entry[0],
                    config.width,
                    config.height,
                    config.num_snakes,
                    time.perf_counter() - t,
                )
        except Exception:
            log.exception("compiling for %s failed: those games get safe_move", config)
        finally:
            with self._lock:
                self._fns[config] = entry
                self._pending.discard(config)
                self._idle.notify_all()

    def _lookup(self, config: GameConfig) -> tuple[str, Any] | None:
        """The compiled policy for ``config``, starting its compilation if new (holds the lock)."""
        if config in self._fns:
            return self._fns[config]
        if config not in self._pending and len(self._fns) + len(self._pending) < MAX_CONFIGS:
            self._pending.add(config)
            threading.Thread(target=self._compile, args=(config,), daemon=True).start()
        return None

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no compilation is pending; False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: not self._pending, timeout)

    def start(self, req: dict[str, Any]) -> str:
        """Remember the game's snake count and start compiling its board; who will play."""
        game, n = req["game"].get("id"), len(req["board"]["snakes"])
        config = game_config(req)
        with self._lock:
            if game is not None:
                if len(self._games) >= MAX_GAMES:  # games whose /end never came
                    del self._games[next(iter(self._games))]
                self._games[game] = n
            if config is None:
                return "safe_move"
            entry = self._lookup(config)
        return entry[0] if entry else "safe_move until compiled"

    def end(self, req: dict[str, Any]) -> None:
        with self._lock:
            self._games.pop(req.get("game", {}).get("id"), None)

    def move(self, req: dict[str, Any]) -> tuple[int, str]:
        """``(action, agent name)`` for the requesting snake."""
        check_sizes(req)
        with self._lock:
            n = self._games.get(req.get("game", {}).get("id"))
            config = game_config(req, n)
            entry = None if config is None else self._lookup(config)
            self._moves += 1
            move_index = self._moves
        if entry is None:
            return safe_move(req), "safe_move"
        name, act = entry
        state = state_from_engine(request_to_engine(req, config.num_snakes), config)
        key = jax.random.fold_in(self._key, move_index)
        return int(act(key, state)), name


SNAKE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def parse_snake(text: str) -> tuple[str, str]:
    """``(name, agent)`` of an ``--agent`` value: ``agent`` or ``name=agent``."""
    name, sep, agent = text.partition("=")
    if not sep:
        name = agent = text
    if not SNAKE_NAME.fullmatch(name):
        raise ValueError(
            f"snake name {name!r} can't be a URL path: write --agent name={agent}, "
            "with letters, digits, '.', '_' and '-' in the name"
        )
    return name, agent


def make_handler(snakes: dict[str, tuple[Mover, dict[str, str]]]) -> type[BaseHTTPRequestHandler]:
    """A handler serving each ``name -> (mover, info)`` snake under ``/<name>``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "slinky"
        timeout = 5  # seconds a client may take to send its request

        def log_message(self, fmt: str, *args: Any) -> None:  # requests are logged below
            pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self) -> tuple[str, str] | None:
            """``(snake name, action)`` of ``/<name>[/<action>]``, or None."""
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts or parts[0] not in snakes or len(parts) > 2:
                return None
            return parts[0], parts[1] if len(parts) == 2 else ""

        def do_GET(self) -> None:
            if self.path.split("?")[0] in ("", "/"):
                self._send(200, {"snakes": [f"/{name}" for name in snakes]})
                return
            route = self._route()
            if route is None or route[1] != "":
                self._send(404, {"error": "not found"})
                return
            self._send(200, snakes[route[0]][1])

        def do_POST(self) -> None:
            route = self._route()
            if route is None or route[1] not in ("start", "move", "end"):
                self._send(404, {"error": "not found"})
                return
            snake, path = route
            mover = snakes[snake][0]
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_BODY:
                    raise ValueError(f"bad Content-Length {size}")
                req = json.loads(self.rfile.read(size))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
                return
            if not isinstance(req, dict) or not isinstance(req.get("game", {}), dict):
                self._send(400, {"error": "not a Battlesnake request"})
                return
            game = req.get("game", {}).get("id", "?")
            if path == "move":
                t = time.perf_counter()
                try:
                    action, name = mover.move(req)
                except Exception:  # never time out on our side: answer something legal-ish
                    log.exception(
                        "%s game %s turn %s: falling back to safe_move",
                        snake,
                        game,
                        req.get("turn"),
                    )
                    try:
                        action, name = safe_move(req), "safe_move"
                    except Exception:  # not even a board: any answer will do
                        action, name = 0, "none"
                ms = (time.perf_counter() - t) * 1e3
                self._send(200, {"move": MOVE_NAMES[action]})
                log.info(
                    "%s game %s turn %s: %s (%s, %.1f ms)",
                    snake,
                    game,
                    req.get("turn"),
                    MOVE_NAMES[action],
                    name,
                    ms,
                )
            elif path == "start":
                self._send(200, {})  # answer first; compiling may take a few seconds
                try:
                    name = mover.start(req)
                    b = req["board"]
                    log.info(
                        "%s game %s start: %sx%s, %d snakes, %s, playing %s",
                        snake,
                        game,
                        b["width"],
                        b["height"],
                        len(b["snakes"]),
                        req["game"].get("ruleset", {}).get("name"),
                        name,
                    )
                except Exception:
                    log.exception("%s game %s: bad /start request", snake, game)
            else:
                self._send(200, {})
                try:
                    mover.end(req)
                    left = [s["id"] for s in req["board"]["snakes"]]
                    log.info(
                        "%s game %s end at turn %s: %s",
                        snake,
                        game,
                        req.get("turn"),
                        "won" if left == [req["you"]["id"]] else "lost or drew",
                    )
                except Exception:
                    log.exception("%s game %s: bad /end request", snake, game)

    return Handler


class Server(ThreadingHTTPServer):
    # The default backlog of 5 drops connections when several games' moves
    # arrive together, and each drop costs a 1 s SYN retransmit.
    request_queue_size = 128
    daemon_threads = True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--agent",
        action="append",
        help="a snake to serve at /<name>: an agent's registry name, or name=agent "
        "(repeat for several snakes; default: ppo)",
    )
    p.add_argument(
        "--fallback",
        default="heuristic",
        help="agent for standard games the main agent can't play (default: heuristic)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--author", default="slinky")
    p.add_argument("--color", default="#3b82f6")
    p.add_argument("--head", default="default")
    p.add_argument("--tail", default="default")
    args = p.parse_args(argv)
    try:
        specs = dict(parse_snake(text) for text in args.agent or ["ppo"])
    except ValueError as e:
        p.error(str(e))
    if len(specs) < len(args.agent or ["ppo"]):
        p.error("two snakes have the same name")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    duel = {
        "game": {"ruleset": {"name": "standard"}, "map": "standard"},
        "turn": 0,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 5, "y": 5}],
            "hazards": [],
            "snakes": [
                {"id": i, "body": [{"x": x, "y": 1}] * 3, "health": 100}
                for i, x in (("a", 1), ("b", 9))
            ],
        },
        "you": {"id": "a"},
    }
    snakes = {}
    for snake, agent in specs.items():
        mover = Mover(agent, args.fallback)
        times = []
        for _ in range(20):
            t = time.perf_counter()
            action, name = mover.move(duel)
            times.append((time.perf_counter() - t) * 1e3)
        log.info("/%s: %s plays 11x11 duels: %.1f ms per move (median of 20)",
                 snake, name, np.median(times))  # fmt: skip
        info = {
            "apiversion": "1",
            "author": args.author,
            "color": args.color,
            "head": args.head,
            "tail": args.tail,
            "version": f"slinky {agent}",
        }
        snakes[snake] = (mover, info)
    server = Server((args.host, args.port), make_handler(snakes))
    for snake, agent in specs.items():
        log.info("serving %s on http://%s:%d/%s", agent, args.host, args.port, snake)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
