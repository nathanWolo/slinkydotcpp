"""AWS Lambda handler: Battlesnake API snakes behind one Function URL.

Each snake is a path prefix, so one function serves them all::

    https://<id>.lambda-url.<region>.on.aws/ppo        <- register this URL
        GET  /ppo          snake info
        POST /ppo/start    (nothing to do; logged)
        POST /ppo/move     {"move": "up" | "down" | "left" | "right"}
        POST /ppo/end      (logged)

The networks load once per execution environment, at cold start. A move
request always gets a move: if anything goes wrong it falls back to
:func:`snake_numpy.safe_move`, then to "up".

Settings (environment variables): ``SNAKE_AUTHOR``, ``SNAKE_COLOR``,
``SNAKE_HEAD``, ``SNAKE_TAIL`` for the info response; ``PPO_PARAMS`` for the
network (default: ``ppo.npz`` next to this file, as ``build.sh`` packages it).
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
SNAKES = {"ppo": sn.PPO(os.environ.get("PPO_PARAMS", str(HERE / "ppo.npz")))}


def info(name: str) -> dict[str, str]:
    return {
        "apiversion": "1",
        "author": os.environ.get("SNAKE_AUTHOR", "slinky"),
        "color": os.environ.get("SNAKE_COLOR", "#3b82f6"),
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
            return SNAKES[name].move(sn.Board(req)), name
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
