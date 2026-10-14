"""AWS Lambda handler: Battlesnake API snakes behind one Function URL.

Each snake is a path prefix, so one function serves them all::

    https://<id>.lambda-url.<region>.on.aws/ppo        <- register this URL
        GET  /ppo          snake info
        POST /ppo/start    (loads the network if needed; logged)
        POST /ppo/move     {"move": "up" | "down" | "left" | "right"}
        POST /ppo/end      (logged)

The snakes (:data:`SNAKES`) are the trained baselines of ``slinky.agents`` that
need no search, under their registry names. A snake's network loads the first
time the execution environment serves it (usually its ``/start``) and stays
loaded. A move request always gets a move: if anything goes wrong it falls back
to :func:`snake_numpy.safe_move`, then to "up".

Settings (environment variables): ``SNAKE_AUTHOR``, ``SNAKE_HEAD``,
``SNAKE_TAIL`` and ``SNAKE_COLOR_<NAME>`` (e.g. ``SNAKE_COLOR_PPO``) for the
info responses; ``CHECKPOINTS`` for the checkpoint directories (default:
``checkpoints/`` next to this file, as ``build.sh`` packages them).
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import snake_numpy as sn

HERE = Path(__file__).resolve().parent
MAX_BODY = 1 << 16  # bytes; a duel request is under 2 KB
CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", str(HERE / "checkpoints")))
# URL name -> (checkpoint directory under CHECKPOINTS, default colour): each
# baseline's fully trained checkpoint. The names are slinky.agents', except that
# "rainbow" is the longer run (slinky.agents' rainbow:baselines/checkpoints/
# rainbow-long-seed0); az-0 is the raw AlphaZero network, without its search.
SNAKES = {
    "ppo": ("ppo-duel-seed0", "#3b82f6"),
    "reinforce": ("reinforce-duel-seed0", "#14b8a6"),
    "dqn": ("dqn-duel-seed0", "#f59e0b"),
    "rainbow": ("rainbow-long-seed0", "#a855f7"),  # trained on to 4.61M env steps
    "az-0": ("az-duel-seed0", "#ef4444"),
}
_nets: dict[str, sn.Net] = {}


def net(name: str) -> sn.Net:
    """The snake's network, loaded on first use."""
    if name not in _nets:
        t = time.perf_counter()
        d = CHECKPOINTS / SNAKES[name][0]
        _nets[name] = sn.Net(str(d / "params.npz"), json.loads((d / "config.json").read_text()))
        log(snake=name, event="load", ms=round((time.perf_counter() - t) * 1e3, 1))
    return _nets[name]


def info(name: str) -> dict[str, str]:
    return {
        "apiversion": "1",
        "author": os.environ.get("SNAKE_AUTHOR", "slinky"),
        "color": os.environ.get(f"SNAKE_COLOR_{name.upper().replace('-', '_')}", SNAKES[name][1]),
        "head": os.environ.get("SNAKE_HEAD", "default"),
        "tail": os.environ.get("SNAKE_TAIL", "default"),
        "version": f"slinky {name} (numpy)",
    }


def response(code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def choose(name: str, req: dict[str, Any]) -> tuple[int, str]:
    """``(move, who chose it)`` for the request."""
    try:
        if sn.is_duel(req):
            return net(name).move(sn.Board(req)), name
        return sn.safe_move(req), "safe_move"
    except Exception as e:  # never fail a move: answer something
        print(json.dumps({"error": repr(e)[:300]}))
        try:
            return sn.safe_move(req), "safe_move"
        except Exception:
            return 0, "none"


def log(**fields: Any) -> None:
    print(json.dumps(fields, separators=(",", ":")))  # one line per event in CloudWatch


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    parts = [p for p in event.get("rawPath", "/").split("/") if p]
    if not parts or parts[0] not in SNAKES or len(parts) > 2:
        return response(404, {"error": "not found"})
    name, action = parts[0], parts[1] if len(parts) == 2 else ""
    if method == "GET" and action == "":
        return response(200, info(name))
    if method != "POST" or action not in ("start", "move", "end"):
        return response(404, {"error": "not found"})

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8", "replace")
    if len(raw) > MAX_BODY:
        return response(400, {"error": "request too large"})
    try:
        req = json.loads(raw)
        game = req["game"]["id"]
        turn = req.get("turn")
    except (ValueError, KeyError, TypeError):
        return response(400, {"error": "not a Battlesnake request"})

    if action == "move":
        t = time.perf_counter()
        move, who = choose(name, req)
        ms = round((time.perf_counter() - t) * 1e3, 2)
        log(snake=name, game=game, turn=turn, move=sn.MOVE_NAMES[move], by=who, ms=ms)
        return response(200, {"move": sn.MOVE_NAMES[move]})
    if action == "start":
        try:
            net(name)
        except Exception as e:  # the moves will report it too
            log(snake=name, event="load failed", error=repr(e)[:300])
        board = req.get("board", {})
        log(
            snake=name,
            game=game,
            event="start",
            ruleset=req["game"].get("ruleset", {}).get("name"),
            size=f"{board.get('width')}x{board.get('height')}",
            snakes=len(board.get("snakes", [])),
            source=req["game"].get("source"),
        )
    else:
        left = [s.get("id") for s in req.get("board", {}).get("snakes", [])]
        won = left == [req.get("you", {}).get("id")]
        log(snake=name, game=game, event="end", turn=turn, result="won" if won else "lost or drew")
    return response(200, {})
