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
  it is new (in the background; see "Compiling" below);
  ``POST /<name>/end`` logs the result.

A snake's name is the agent's registry name, or the ``name`` of
``--agent name=agent`` (letters, digits, ``.``, ``_`` and ``-``), which is
needed for agents whose registry name has a run directory in it.

Games. :func:`game_config` reads the exact :class:`~slinky.types.GameConfig`
the rules need from a request: the ruleset (``game.ruleset.name``), the map
(``game.map``), the board and the settings ``foodSpawnChance``,
``minimumFood``, ``hazardDamagePerTurn`` and ``royale.shrinkEveryNTurns``
(``game.ruleset.settings``; a missing or malformed one keeps its
``GameConfig`` default). It models the four leaderboard modes
(:mod:`slinky.modes`) and the standard game on any board:

* ``standard`` ruleset on the ``standard`` map (``duel`` with two snakes);
* royale: the ``royale`` ruleset on the ``standard`` or ``royale`` map, or the
  ``standard`` ruleset on the ``royale`` map. The engine plays all three
  alike, so they share one config, the royale ruleset on the royale map, whose
  rules step regenerates the shrinking zone inside a search too. The shrink
  sides (``State.map_state``) are read off the zone in the request
  (:func:`royale_sides`); the sides of shrinks still to come are unknown (the
  engine draws them at random), so the search assumes the zone stays put until
  the next shrink is shown;
* ``constrictor`` ruleset on the ``standard`` map, or on the ``empty`` map (the
  official constrictor ladder's). Constrictor removes all food every turn, and
  the empty map only lacks the standard map's food spawns, so a search that
  never spawns food (every agent here by default; ``spawn_food`` is the
  exception) plays both identically: the empty-map game is read as the
  standard-map constrictor config the agents are trained on. A search that
  samples spawns would add food the empty map never has, so it gets no config.

Any other ruleset or map (wrapped, solo, constrictor on the royale map, the
standard ruleset on the ``empty`` map, ...) has no config, and a plain
safe-move picker plays.

Settings that cannot change a game are not part of its config, so that the
ladder's values (its ``hazardDamagePerTurn`` is not the default) reach the
precompiled configs: see :func:`game_config`.

Agents. A multi-mode agent (:data:`slinky.agents.MODES_KINDS`) plays every
11x11 game that has a mode (:func:`slinky.modes.mode_of`), told that mode (or
its ``:game_mode=``), if its run was trained on it. The other trained
baselines read 11x11 egocentric observations, so they only play 11x11 standard
games with two snakes (the Duels leaderboard), even where they could run.
Games the agent does not play get the ``--fallback`` agent (the heuristic by
default), compiled for that game.

Compiling. Each game config is compiled once, in a background thread, and
until it is ready its games get the safe-move picker. The duel is compiled at
startup, and for a multi-mode agent also the default configs of the three
other modes (startup then takes a few times longer; the log says how long).

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

from slinky import modes
from slinky.agents import BASELINES, MODES_KINDS, make_agent, make_env, parse_agent
from slinky.engine_json import state_from_engine, state_to_engine
from slinky.types import (
    ACTION_DELTAS,
    GameConfig,
    RoyaleState,
    Ruleset,
    State,
    royale_num_sides,
)

DUEL = GameConfig()  # the game the trained baselines learned: 11x11 standard, 2 snakes
# The default config of each leaderboard mode; DUEL is MODE_CONFIGS["duel"].
MODE_CONFIGS = {mode: modes.mode_config(mode) for mode in modes.MODE_NAMES}

MOVE_NAMES = ("up", "down", "left", "right")  # index = action, as in slinky.types
MAX_BODY = 1 << 16  # bytes; a duel request is under 2 KB
# Limits on what requests can make the server compile and remember: a new
# board shape costs about a second of compilation and some memory (a search
# agent's take longer). A multi-mode agent precompiles four.
MAX_SIDE, MAX_SNAKES, MAX_CONFIGS, MAX_GAMES = 25, 8, 16, 1000
# Settings outside this range keep their default (they would overflow int32 arithmetic).
MAX_SETTING = 1 << 20
# A shrink side that moves no edge: :func:`royale_sides` fills the shrinks not yet shown with it.
UNKNOWN_SIDE = 4

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


def _setting(settings: Any, key: str, default: int) -> int:
    """The integer ``settings[key]`` (the engine CLI sends strings), or ``default``."""
    value = settings.get(key) if isinstance(settings, dict) else None
    if isinstance(value, bool):
        return default
    try:
        value = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if -MAX_SETTING <= value <= MAX_SETTING else default


def game_config(
    req: dict[str, Any], num_snakes: int | None = None, food_spawns: bool = False
) -> GameConfig | None:
    """The :class:`GameConfig` of a request's game, or None for a game the rules don't model.

    Standard, royale and constrictor games (see the module docstring); the three royale
    variants share the config of the royale ruleset on the royale map. ``num_snakes`` is
    the game's snake count at the start, if known (the request lists the living only).
    The settings come from ``game.ruleset.settings``, but only those that can change
    play are kept; the others stay at the :class:`GameConfig` defaults, so the games
    of one mode share one compiled config whatever the ladder sends:

    * ``foodSpawnChance`` and ``minimumFood`` are read only by the map's food spawning
      (``maps.spawn_food_standard``), which a search runs only with ``spawn_food``
      (``food_spawns``); the rules step, the state rebuild and the observations never
      read them (the request lists the food that exists). Kept only if ``food_spawns``.
    * ``hazardDamagePerTurn`` is read by ``rules.rules_step`` where a snake stands in
      a hazard. Outside royale, the standard and empty maps place none, so with none
      on the board it cannot matter; royale (hazards of its own) and a board that
      lists hazards keep it.
    * the shrink period is read only in royale (zone, shrink phase feature).

    Constrictor on the ``empty`` map is the constrictor mode's config (standard map)
    unless ``food_spawns``, see the module docstring.
    """
    game = req.get("game", {})
    ruleset = game.get("ruleset", {}) or {}
    name = ruleset.get("name", "standard")
    map_name = game.get("map", "standard")
    if name not in ("standard", "royale", "constrictor") or map_name not in (
        "standard",
        "royale",
        "empty",
    ):
        return None
    if map_name == "empty" and (name != "constrictor" or food_spawns):
        return None
    royale = name == "royale" or map_name == "royale"
    if royale and name == "constrictor":
        return None  # no mode, and the engine's royale map would put hazards in
    board = req["board"]
    w, h, n = board["width"], board["height"], max(num_snakes or 0, len(board["snakes"]))
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE and 1 <= n <= MAX_SNAKES):
        return None
    settings = ruleset.get("settings") or {}
    d = GameConfig()
    fields: dict[str, Any] = {"width": w, "height": h, "num_snakes": n}
    if food_spawns:
        fields["food_spawn_chance"] = _setting(settings, "foodSpawnChance", d.food_spawn_chance)
        fields["minimum_food"] = _setting(settings, "minimumFood", d.minimum_food)
    if royale or board.get("hazards"):
        fields["hazard_damage_per_turn"] = _setting(
            settings, "hazardDamagePerTurn", d.hazard_damage_per_turn
        )
    if royale:
        royale_settings = settings.get("royale") if isinstance(settings, dict) else None
        # The API nests it under "royale"; the engine CLI's flat name is read too.
        shrink = _setting(settings, "shrinkEveryNTurns", d.shrink_every_n_turns)
        shrink = _setting(royale_settings, "shrinkEveryNTurns", shrink)
        fields.update(ruleset=Ruleset.ROYALE, map="royale", shrink_every_n_turns=shrink)
    elif name == "constrictor":
        fields.update(ruleset=Ruleset.CONSTRICTOR)  # map "standard", also for "empty"
    try:
        return GameConfig(**fields)
    except ValueError:  # e.g. shrinkEveryNTurns 0, an engine error
        return None


def game_mode(config: GameConfig) -> str | None:
    """The leaderboard mode of ``config`` (:func:`slinky.modes.mode_of`), or None."""
    try:
        return modes.mode_of(config)
    except ValueError:
        return None


def royale_sides(hazards: list[list[int]], turn: int, config: GameConfig) -> RoyaleState:
    """Royale shrink sides (``State.map_state``) that give the zone shown at ``turn``.

    The zone at ``turn`` has had ``turn // config.shrink_every_n_turns`` shrinks, and
    its hazards (all the cells outside the safe rectangle) say how many came from each
    side; their order changes nothing. The sides of the shrinks still to come are
    unknown and get :data:`UNKNOWN_SIDE`, which moves no edge: a search assumes the
    zone stays put after the shrinks it has seen.
    """
    h, w = config.height, config.width
    hazard = np.zeros((h, w), bool)
    for x, y in hazards:
        if 0 <= x < w and 0 <= y < h:
            hazard[y, x] = True
    num = royale_num_sides(config)
    shrinks = min(max(turn, 0) // config.shrink_every_n_turns, num)
    if hazard.all():  # no safe cell left: any shrinks that cover the board
        counts = [w, 0, 0, 0] if shrinks >= w else [0, 0, h, 0]
    else:
        ys, xs = np.nonzero(~hazard)
        counts = [xs.min(), w - 1 - xs.max(), ys.min(), h - 1 - ys.max()]
    seen = np.repeat(np.arange(4, dtype=np.int8), counts)[:shrinks]
    sides = np.full(num, UNKNOWN_SIDE, np.int8)
    sides[: len(seen)] = seen
    return RoyaleState(jnp.asarray(sides))


def request_state(req: dict[str, Any], config: GameConfig) -> State:
    """The :class:`State` of a request in a game with ``config``, ``you`` in seat 0.

    The game's dead snakes are padded in as eliminated seats up to
    ``config.num_snakes``; a royale state gets its shrink sides (:func:`royale_sides`).
    """
    d = request_to_engine(req, config.num_snakes)
    map_state: Any = ()
    if config.ruleset == Ruleset.ROYALE:
        map_state = royale_sides(d["hazards"], d["turn"], config)
    return state_from_engine(d, config, map_state)


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


def example_request(config: GameConfig, seed: int = 0) -> dict[str, Any]:
    """A move request for the start of a game with ``config`` (from seat 0), as the API sends."""
    state = make_env(config, False).reset(jax.random.key(seed))[0]
    d = state_to_engine(state, config)
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
    ]
    settings = {
        "foodSpawnChance": config.food_spawn_chance,
        "minimumFood": config.minimum_food,
        "hazardDamagePerTurn": config.hazard_damage_per_turn,
        "royale": {"shrinkEveryNTurns": config.shrink_every_n_turns},
    }
    return {
        "game": {
            "id": f"example-{game_mode(config) or 'game'}",
            "ruleset": {"name": config.ruleset.value, "settings": settings},
            "map": config.map,
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
        "you": snakes[0],
    }


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

    A game config is compiled once, in a background thread (one at a time), so a
    new game never stalls the moves of games already being played. Until its
    config is ready a game gets :func:`safe_move`. The configs in ``precompile``
    are compiled before the constructor returns; by default the duel, and for a
    multi-mode agent the default configs of all four modes (:data:`MODE_CONFIGS`).
    """

    def __init__(
        self,
        agent: str,
        fallback: str = "heuristic",
        seed: int = 0,
        precompile: tuple[GameConfig, ...] | None = None,
    ) -> None:
        spec = parse_agent(agent)
        self.agent = spec.name
        self.multi_mode = spec.kind in MODES_KINDS
        self.fallback = parse_agent(fallback).name
        # Whether the agent's search samples food spawns (then the food settings matter).
        self.food_spawns = bool(getattr(spec.mcts, "spawn_food", False))
        if precompile is None:
            precompile = tuple(MODE_CONFIGS.values()) if self.multi_mode else (DUEL,)
            precompile = (DUEL, *(c for c in precompile if c != DUEL))
        # Compiled shapes; None marks a shape whose compilation failed.
        self._fns: dict[GameConfig, tuple[str, Any] | None] = {}
        self._pending: set[GameConfig] = set()
        self._games: dict[str, int] = {}  # game id -> snakes at the start
        self._key = jax.random.key(seed)
        self._moves = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._compile_lock = threading.Lock()
        t = time.perf_counter()
        for config in precompile:
            self._pending.add(config)
            self._compile(config)
        if precompile:
            log.info(
                "%s: precompiled %d game configs (%s) in %.1f s",
                self.agent,
                len(precompile),
                ", ".join(f"{game_mode(c) or 'no mode'} {c.num_snakes}p" for c in precompile),
                time.perf_counter() - t,
            )

    def plays(self, config: GameConfig) -> bool:
        """Whether the main agent is meant for games with ``config`` (it may still refuse)."""
        mode = game_mode(config)
        if self.multi_mode:  # every mode, told it (11x11 and its run's modes: make_agent)
            return mode is not None
        if parse_agent(self.agent).kind in BASELINES:  # the game they were trained on
            return mode == "duel" and (config.width, config.height) == (11, 11)
        return True

    def _build(self, config: GameConfig) -> tuple[str, Any]:
        agent = None
        if self.plays(config):
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
                    "compiled %s for %dx%d, %d snakes, %s (%s) in %.1f s",
                    entry[0],
                    config.width,
                    config.height,
                    config.num_snakes,
                    config.ruleset.value,
                    game_mode(config) or "no mode",
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
        config = game_config(req, food_spawns=self.food_spawns)
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
            config = game_config(req, n, self.food_spawns)
            entry = None if config is None else self._lookup(config)
            self._moves += 1
            move_index = self._moves
        if entry is None:
            return safe_move(req), "safe_move"
        name, act = entry
        state = request_state(req, config)
        key = jax.random.fold_in(self._key, move_index)
        return int(act(key, state)), name


API_DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}  # y points up


class MoveLedger:
    """Whether the engine applied each answer, and a JSON line per answer.

    The engine never says that an answer came too late: it just moves the snake on
    as if it had repeated its last move. Each game's next request (``/move`` or
    ``/end``) shows where the snake's head went, so a head that moved another way
    than the answer means the answer missed the game's timeout, from compute or the
    network. A late answer that happens to be the default move goes unnoticed, so
    the count is a lower bound. A turn that skips one means a request that never
    reached us.

    Each answer appends one line to ``path`` (when given): time, snake, game, turn,
    ruleset, map, snake count, source (e.g. ``ladder``), the game's timeout in ms,
    the agent, the move, our compute ms and ``prev``, the verdict on this game's
    previous answer (``applied``, ``missed``, ``gap`` or None), plus ``wall`` (ms from
    the request reaching the handler to the answer sent) and ``colo`` (the Cloudflare
    data centre of the request's ``Cf-Ray``) when known. A timeout with a short
    ``wall`` was lost outside the server. ``/end`` adds a line with ``"end": true``
    and the verdict on the last answer (None when our snake died before the end: the
    engine stops asking a dead snake). Detected misses are also logged as warnings.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._last: dict[tuple[str, str], tuple[int, str, int, int]] = {}  # turn, move, x, y

    def verdict(self, snake: str, req: dict[str, Any]) -> str | None:
        """The fate of this game's previous answer, from the head in this request."""
        try:
            game, turn = req["game"]["id"], int(req["turn"])
            head, board = req["you"]["head"], req["board"]
            x, y, w, h = int(head["x"]), int(head["y"]), int(board["width"]), int(board["height"])
        except (KeyError, TypeError, ValueError):
            return None
        with self._lock:
            prev = self._last.get((snake, game))
        if prev is None:
            return None
        prev_turn, move, px, py = prev
        if turn != prev_turn + 1:
            return "gap"
        # Shortest signed step, so wrapped boards count an edge crossing as one step.
        dx = (x - px + w // 2) % w - w // 2 if w else x - px
        dy = (y - py + h // 2) % h - h // 2 if h else y - py
        return "applied" if (dx, dy) == API_DELTAS[move] else "missed"

    def record(
        self, snake: str, req: dict[str, Any], move: str | None, agent: str, ms: float,
        end: bool = False, wall_ms: float | None = None, colo: str | None = None,
    ) -> str | None:  # fmt: skip
        """Judge the previous answer, remember this one (``move``; None at ``/end``)."""
        verdict = self.verdict(snake, req)
        if end and verdict == "gap":
            verdict = None  # our snake died earlier; nothing was asked since
        game = req.get("game", {}) if isinstance(req.get("game"), dict) else {}
        key = (snake, str(game.get("id")))
        try:
            head = req["you"]["head"]
            last = (int(req["turn"]), move, int(head["x"]), int(head["y"]))
        except (KeyError, TypeError, ValueError):
            last = None
        with self._lock:
            if end or last is None or move is None:
                self._last.pop(key, None)
            else:
                if key not in self._last and len(self._last) >= MAX_GAMES:
                    del self._last[next(iter(self._last))]  # games whose /end never came
                self._last[key] = last
        if verdict in ("missed", "gap"):
            log.warning(
                "%s game %s turn %s: our previous answer was not applied (%s; timeout %s ms)",
                snake, key[1], req.get("turn"), verdict, game.get("timeout"),
            )  # fmt: skip
        if self.path:
            ruleset = game.get("ruleset") if isinstance(game.get("ruleset"), dict) else {}
            board = req.get("board") if isinstance(req.get("board"), dict) else {}
            line = {
                "t": round(time.time(), 3), "snake": snake, "game": key[1], "turn": req.get("turn"),
                "ruleset": ruleset.get("name"), "map": game.get("map"),
                "snakes": len(board.get("snakes") or []), "source": game.get("source"),
                "timeout": game.get("timeout"), "agent": agent, "move": move,
                "ms": round(ms, 1), "prev": verdict,
            }  # fmt: skip
            if wall_ms is not None:
                line["wall"] = round(wall_ms, 1)
            if colo:
                line["colo"] = colo
            if end:
                line["end"] = True
            try:
                with self._lock, open(self.path, "a") as f:
                    f.write(json.dumps(line) + "\n")
            except OSError:
                log.exception("cannot write the move log %s", self.path)
        return verdict


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


def make_handler(
    snakes: dict[str, tuple[Mover, dict[str, str]]], ledger: MoveLedger | None = None
) -> type[BaseHTTPRequestHandler]:
    """A handler serving each ``name -> (mover, info)`` snake under ``/<name>``; ``ledger``
    checks that the engine applied each answer (see :class:`MoveLedger`)."""
    ledger = ledger or MoveLedger()

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
            t_in = time.perf_counter()
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
                wall = (time.perf_counter() - t_in) * 1e3
                ray = self.headers.get("Cf-Ray", "")  # "<id>-<colo>" behind Cloudflare
                try:
                    ledger.record(
                        snake, req, MOVE_NAMES[action], name, ms, wall_ms=wall,
                        colo=ray.rpartition("-")[2] or None,
                    )  # fmt: skip
                except Exception:  # bookkeeping must never break play
                    log.exception("%s game %s: move ledger", snake, game)
            elif path == "start":
                self._send(200, {})  # answer first; compiling may take a few seconds
                try:
                    name = mover.start(req)
                    b = req["board"]
                    log.info(
                        "%s game %s start: %sx%s, %d snakes, %s on map %s, playing %s",
                        snake,
                        game,
                        b["width"],
                        b["height"],
                        len(b["snakes"]),
                        req["game"].get("ruleset", {}).get("name"),
                        req["game"].get("map"),
                        name,
                    )
                except Exception:
                    log.exception("%s game %s: bad /start request", snake, game)
            else:
                self._send(200, {})
                try:
                    ledger.record(snake, req, None, "", 0.0, end=True)
                except Exception:
                    log.exception("%s game %s: move ledger", snake, game)
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
        help="agent for the games the main agent doesn't play (default: heuristic)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--author", default="slinky")
    p.add_argument("--color", default="#3b82f6")
    p.add_argument("--head", default="default")
    p.add_argument("--tail", default="default")
    p.add_argument(
        "--move-log",
        help="append a JSON line per answer here, with whether the engine applied the "
        "game's previous answer (a lower bound on timeouts; see MoveLedger)",
    )
    args = p.parse_args(argv)
    try:
        specs = dict(parse_snake(text) for text in args.agent or ["ppo"])
    except ValueError as e:
        p.error(str(e))
    if len(specs) < len(args.agent or ["ppo"]):
        p.error("two snakes have the same name")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    snakes = {}
    for snake, agent in specs.items():
        t = time.perf_counter()
        mover = Mover(agent, args.fallback)
        log.info("/%s: ready to play in %.1f s", snake, time.perf_counter() - t)
        for mode in modes.MODE_NAMES if mover.multi_mode else ("duel",):
            req, count, times = example_request(MODE_CONFIGS[mode]), 20, []
            for _ in range(count if mode == "duel" else 5):
                t = time.perf_counter()
                action, name = mover.move(req)
                times.append((time.perf_counter() - t) * 1e3)
            log.info("/%s: %s plays 11x11 %s: %.1f ms per move (median of %d)",
                     snake, name, mode, np.median(times), len(times))  # fmt: skip
        info = {
            "apiversion": "1",
            "author": args.author,
            "color": args.color,
            "head": args.head,
            "tail": args.tail,
            "version": f"slinky {agent}",
        }
        snakes[snake] = (mover, info)
    server = Server((args.host, args.port), make_handler(snakes, MoveLedger(args.move_log)))
    for snake, agent in specs.items():
        log.info("serving %s on http://%s:%d/%s", agent, args.host, args.port, snake)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
