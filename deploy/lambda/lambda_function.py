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

The native snakes (:data:`NATIVE`) are the multi-mode AlphaZero agent with its
search (all four ladder modes), played by the Rust engine ``bin/slinky-engine``
(``native/crates/lambda``). The handler starts it once per execution
environment, on first use, and talks to it in JSON lines (:class:`Engine`); the
engine loads the weights once and keeps the ``/start`` snake counts. Every
search stops at the request's deadline, ``game.timeout`` (500 ms when missing)
less ``NATIVE_MARGIN_MS``, counted from the handler's start, and plays its best
move so far. If the engine does not answer by then (plus a little grace), dies
or answers nonsense, the move is :func:`snake_numpy.safe_move` and the engine is
restarted on the next request.

Settings (environment variables): ``SNAKE_AUTHOR``, ``SNAKE_HEAD``,
``SNAKE_TAIL`` and ``SNAKE_COLOR_<NAME>`` (e.g. ``SNAKE_COLOR_PPO``) for the
info responses; ``CHECKPOINTS`` for the checkpoint directories (default:
``checkpoints/`` next to this file, as ``build.sh`` packages them). Native
snakes: ``SNAKE_SIMULATIONS_<NAME>`` (simulations per move, default from the
name: 256 for ``az-256``), ``SNAKE_BUDGET_MS_<NAME>`` (optional: also stop that
many ms after the request's arrival), ``NATIVE_MARGIN_MS`` (default 150: the
time kept for the network and Lambda's overhead), ``WEIGHTS_DIR`` (default
``weights/``) and ``ENGINE_BIN`` (default ``bin/slinky-engine``).
"""

from __future__ import annotations

import base64
import json
import os
import select
import subprocess
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
# Native snakes: URL name -> (weights directory under WEIGHTS_DIR, default colour,
# default simulations per move). az-256 is the az_modes M network at 256
# simulations, the ladder agent (native/crates/lambda: same moves as Python's).
NATIVE = {
    "az-256": ("az-modes", "#ef4444", 256),
}
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", str(HERE / "weights")))
ENGINE_BIN = Path(os.environ.get("ENGINE_BIN", str(HERE / "bin" / "slinky-engine")))
MARGIN_MS = int(os.environ.get("NATIVE_MARGIN_MS", "150"))
GRACE_MS = 30  # how long past the engine's own deadline the handler waits for it
DEFAULT_TIMEOUT_MS = 500


def net(name: str) -> sn.Net:
    """The snake's network, loaded on first use."""
    if name not in _nets:
        t = time.perf_counter()
        d = CHECKPOINTS / SNAKES[name][0]
        _nets[name] = sn.Net(str(d / "params.npz"), json.loads((d / "config.json").read_text()))
        log(snake=name, event="load", ms=round((time.perf_counter() - t) * 1e3, 1))
    return _nets[name]


def env_key(name: str) -> str:
    """The ``<NAME>`` of a snake's settings: ``az-256`` -> ``AZ_256``."""
    return name.upper().replace("-", "_").replace(".", "_")


def info(name: str) -> dict[str, str]:
    native = name in NATIVE
    return {
        "apiversion": "1",
        "author": os.environ.get("SNAKE_AUTHOR", "slinky"),
        "color": os.environ.get(
            f"SNAKE_COLOR_{env_key(name)}", (NATIVE[name] if native else SNAKES[name])[1]
        ),
        "head": os.environ.get("SNAKE_HEAD", "default"),
        "tail": os.environ.get("SNAKE_TAIL", "default"),
        "version": f"slinky {name} ({'native' if native else 'numpy'})",
    }


def game_timeout_ms(req: Any) -> int:
    """``game.timeout`` as the engine reads it: 500 when missing or not a positive number."""
    try:
        t = req["game"].get("timeout")
    except (AttributeError, KeyError, TypeError):
        return DEFAULT_TIMEOUT_MS
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not t >= 1:
        return DEFAULT_TIMEOUT_MS
    return int(min(t, 60_000))


class Engine:
    """The native engine process (``bin/slinky-engine``): one request line in, one answer
    line out, matched by id. Started on first use; stopped (and so restarted on the next
    request) when it is late, dies or answers nonsense."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self.buf = b""
        self.next_id = 0

    def env(self) -> dict[str, str]:
        specs = []
        for name, (sub, _, sims) in NATIVE.items():
            sims = os.environ.get(f"SNAKE_SIMULATIONS_{env_key(name)}", sims)
            spec = f"{name}={sub}:simulations={sims}"
            budget = os.environ.get(f"SNAKE_BUDGET_MS_{env_key(name)}")
            if budget:
                spec += f":budget_ms={budget}"
            specs.append(spec)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "WEIGHTS_DIR": str(WEIGHTS_DIR),
            "SNAKES": ",".join(specs),
            "DEADLINE_MARGIN_MS": str(MARGIN_MS),
        }
        for k in ("NATIVE_THREADS", "TIE_SEED", "MEMORY_LIMIT_MB", "DEBUG_PANIC"):
            if os.environ.get(k):
                env["THREADS" if k == "NATIVE_THREADS" else k] = os.environ[k]
        return env

    def start(self) -> subprocess.Popen[bytes]:
        t = time.perf_counter()
        proc = subprocess.Popen(
            [str(ENGINE_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env()
        )  # its log lines go to our stderr: CloudWatch
        assert proc.stdin is not None and proc.stdout is not None
        os.set_blocking(proc.stdin.fileno(), False)
        self.proc, self.buf = proc, b""
        log(event="engine start", pid=proc.pid, ms=round((time.perf_counter() - t) * 1e3, 1))
        return proc

    def stop(self, why: str) -> None:
        proc, self.proc, self.buf = self.proc, None, b""
        if proc is None:
            return
        log(event="engine stop", pid=proc.pid, why=why)
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:  # already gone, or a zombie for the OS to reap
            pass

    def _write(self, fd: int, data: bytes, deadline: float) -> bool:
        while data:
            try:
                data = data[os.write(fd, data) :]
            except BlockingIOError:
                left = deadline - time.perf_counter()
                if left <= 0 or not select.select([], [fd], [], left)[1]:
                    return False
            except OSError:  # the engine is gone (BrokenPipeError)
                return False
        return True

    def _readline(self, fd: int, deadline: float) -> bytes | None:
        while b"\n" not in self.buf:
            left = deadline - time.perf_counter()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                return None
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return None  # it exited
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        return line

    def ask(self, name: str, kind: str, raw: str, t0: float, wait: float | None) -> dict | None:
        """Sends a request; returns the engine's answer (``{"code", "reply"}``), ``{}`` when
        not waiting (``wait`` None), or None after stopping the engine (late, gone, bad)."""
        proc = self.proc if self.proc is not None and self.proc.poll() is None else None
        if proc is None:
            if self.proc is not None:
                self.stop(f"exited with {self.proc.returncode}")
            proc = self.start()
        assert proc.stdin is not None and proc.stdout is not None
        self.next_id += 1
        rid = self.next_id
        msg = {
            "id": rid,
            "snake": name,
            "kind": kind,
            "body": raw,
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 3),
        }
        line = json.dumps(msg, ensure_ascii=False).encode("utf-8", "replace") + b"\n"
        deadline = time.perf_counter() + (wait if wait is not None else 1.0)
        if not self._write(proc.stdin.fileno(), line, deadline):
            self.stop("cannot write")
            return None
        if wait is None:
            return {}
        while True:
            out = self._readline(proc.stdout.fileno(), deadline)
            if out is None:
                self.stop("no answer in time" if proc.poll() is None else "exited")
                return None
            try:
                ans = json.loads(out)
            except ValueError:
                self.stop(f"bad output {out[:100]!r}")
                return None
            if ans.get("id") == rid:
                return ans
            if not isinstance(ans.get("id"), int) or ans["id"] > rid:
                self.stop(f"unexpected answer {out[:100]!r}")
                return None
            # else: the answer to an earlier request not waited for (/end): skip it


_engine = Engine()


def native_move(name: str, raw: str, req: Any, t0: float) -> tuple[int, str]:
    """``(move, who chose it)`` from the engine, or the safe move."""
    wait = (game_timeout_ms(req) - MARGIN_MS + GRACE_MS) / 1e3 - (time.perf_counter() - t0)
    try:
        ans = _engine.ask(name, "move", raw, t0, max(wait, 0.02))
        mv = ans and ans.get("code") == 200 and ans.get("reply", {}).get("move")
        if mv in sn.MOVE_NAMES:
            return sn.MOVE_NAMES.index(mv), "native"
        why = "no answer" if ans is None else f"answer {json.dumps(ans)[:200]}"
    except Exception as e:  # never fail a move
        _engine.stop(f"error {e!r}"[:300])
        why = repr(e)[:300]
    print(json.dumps({"error": f"engine: {why}"}))
    try:
        return sn.safe_move(req), "safe_move"
    except Exception:
        return 0, "none"


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
    t0 = time.perf_counter()
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    parts = [p for p in event.get("rawPath", "/").split("/") if p]
    if not parts or (parts[0] not in SNAKES and parts[0] not in NATIVE) or len(parts) > 2:
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

    native = name in NATIVE
    if action == "move":
        t = time.perf_counter()
        move, who = native_move(name, raw, req, t0) if native else choose(name, req)
        ms = round((time.perf_counter() - t) * 1e3, 2)
        log(snake=name, game=game, turn=turn, move=sn.MOVE_NAMES[move], by=who, ms=ms)
        return response(200, {"move": sn.MOVE_NAMES[move]})
    if action == "start":
        try:
            if native:  # starts the engine (and loads its weights) before the first move
                ok = _engine.ask(name, "start", raw, t0, 1.0)
                if ok is None:
                    raise RuntimeError("the engine did not answer /start")
            else:
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
        if native:
            try:
                _engine.ask(name, "end", raw, t0, None)  # not waited for
            except Exception as e:
                _engine.stop(f"error {e!r}"[:300])
        left = [s.get("id") for s in req.get("board", {}).get("snakes", [])]
        won = left == [req.get("you", {}).get("id")]
        log(snake=name, game=game, event="end", turn=turn, result="won" if won else "lost or drew")
    return response(200, {})
