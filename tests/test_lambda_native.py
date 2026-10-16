"""The Lambda handler's native snake (``/az-256``): the Rust engine process behind
``deploy/lambda/lambda_function.py``, run locally with a host (x86) build of the engine.

Needs ``slinky-engine`` (``$SLINKY_ENGINE_BIN``, default the unwinding build in
``/dev/shm/naz-target-deployprep-unwind/release/``; build it with
``CARGO_PROFILE_RELEASE_PANIC=unwind cargo build --release -p slinky-lambda --bin slinky-engine``)
and the az_modes M run dir (``$NAZ_RUN``); skips without them. No JAX: only the handler.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_DIR = ROOT / "deploy" / "lambda"
ENGINE = Path(
    os.environ.get(
        "SLINKY_ENGINE_BIN", "/dev/shm/naz-target-deployprep-unwind/release/slinky-engine"
    )
)
RUN = Path(os.environ.get("NAZ_RUN", "/home/nathan/slinky-serve/az-modes-M-s0"))
EXAMPLES = json.loads((ROOT / "native/crates/server/data/examples.json").read_text())["requests"]
MODES = ("standard", "duel", "royale", "constrictor")
MOVES = ("up", "down", "left", "right")

pytestmark = pytest.mark.skipif(
    not (ENGINE.exists() and (RUN / "params.npz").exists()),
    reason=f"needs {ENGINE} and {RUN}",
)
sys.path.insert(0, str(LAMBDA_DIR))


@pytest.fixture
def lf(tmp_path, monkeypatch):
    """A fresh ``lambda_function`` module wired to the local engine and weights."""
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "az-modes").symlink_to(RUN)
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("ENGINE_BIN", str(ENGINE))
    monkeypatch.setenv("CHECKPOINTS", str(ROOT / "baselines" / "checkpoints"))
    for k in [k for k in os.environ if k.startswith(("SNAKE_", "NATIVE_"))]:
        monkeypatch.delenv(k)
    import lambda_function

    mod = importlib.reload(lambda_function)
    yield mod
    mod._engine.stop("test over")


def event(method: str, path: str, body=None, b64: bool = False) -> dict:
    raw = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    ev = {"rawPath": path, "requestContext": {"http": {"method": method}}}
    if raw is not None:
        ev["body"] = base64.b64encode(raw).decode() if b64 else raw.decode("utf-8", "replace")
        ev["isBase64Encoded"] = b64
    return ev


def call(lf, method: str, path: str, body=None, b64: bool = False) -> tuple[int, dict]:
    out = lf.lambda_handler(event(method, path, body, b64), None)
    return out["statusCode"], json.loads(out["body"])


def move_logs(capsys) -> list[dict]:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    return [d for d in map(json.loads, lines) if "move" in d]


class Reference:
    """The engine binary on its own (same weights and settings): the moves the handler
    should relay."""

    def __init__(self, lf) -> None:
        self.p = subprocess.Popen(
            [str(ENGINE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=lf._engine.env(),
        )
        self.i = 0

    def ask(self, kind: str, body: dict) -> dict:
        self.i += 1
        line = {"id": self.i, "snake": "az-256", "kind": kind, "body": json.dumps(body)}
        self.p.stdin.write(json.dumps(line).encode() + b"\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["reply"]

    def close(self) -> None:
        self.p.kill()
        self.p.wait()


def test_info(lf) -> None:
    code, info = call(lf, "GET", "/az-256")
    assert code == 200
    assert info == {
        "apiversion": "1",
        "author": "slinky",
        "color": "#ef4444",
        "head": "default",
        "tail": "default",
        "version": "slinky az-256 (native)",
    }
    os.environ["SNAKE_COLOR_AZ_256"] = "#123456"
    try:
        assert call(lf, "GET", "/az-256/")[1]["color"] == "#123456"
    finally:
        del os.environ["SNAKE_COLOR_AZ_256"]
    assert call(lf, "GET", "/az-256/move")[0] == 404
    assert call(lf, "POST", "/az-256")[0] == 404
    assert call(lf, "POST", "/az-256/move/x", EXAMPLES["duel"])[0] == 404


def test_games_in_every_mode_match_the_engine(lf, capsys) -> None:
    ref = Reference(lf)
    try:
        for mode in MODES:
            req = EXAMPLES[mode]
            assert call(lf, "POST", "/az-256/start", req) == (200, {})
            ref.ask("start", req)
            for turn, b64 in ((0, False), (1, True), (7, False)):
                r = dict(req, turn=turn)
                code, out = call(lf, "POST", "/az-256/move", r, b64)
                assert code == 200 and out["move"] in MOVES
                assert out == ref.ask("move", r), (mode, turn)
            assert call(lf, "POST", "/az-256/end", req) == (200, {})
    finally:
        ref.close()
    logs = move_logs(capsys)
    assert len(logs) == 12 and all(d["by"] == "native" for d in logs), logs
    assert lf._engine.proc is not None and lf._engine.proc.poll() is None


def test_malformed_requests(lf, capsys) -> None:
    assert call(lf, "POST", "/az-256/move", b"{nope")[0] == 400
    assert call(lf, "POST", "/az-256/move", b"")[0] == 400
    assert call(lf, "POST", "/az-256/move", {"turn": 3})[0] == 400
    assert call(lf, "POST", "/az-256/move", b"x" * 70000)[0] == 400
    # Requests the engine cannot play get the safe move (as the numpy snakes' fallback).
    weird = []
    r = dict(EXAMPLES["standard"])
    r["board"] = dict(r["board"], width=19, height=19)
    weird.append(r)
    weird.append({"game": {"id": "x"}, "board": 3})
    weird.append({"game": {"id": 5}, "turn": 1})
    one = dict(EXAMPLES["duel"])
    one["board"] = dict(one["board"], snakes=one["board"]["snakes"][:1])
    weird.append(one)
    for w in weird:
        code, out = call(lf, "POST", "/az-256/move", w)
        assert code == 200 and out["move"] in MOVES, w
    # The engine is still the same process and still plays.
    pid = lf._engine.proc.pid
    assert call(lf, "POST", "/az-256/move", EXAMPLES["royale"])[0] == 200
    assert lf._engine.proc.pid == pid
    assert move_logs(capsys)[-1]["by"] == "native"


@pytest.mark.parametrize("timeout", [200, 500])
def test_game_timeout_bounds_the_answer(lf, capsys, timeout) -> None:
    os.environ["SNAKE_SIMULATIONS_AZ_256"] = "1048576"  # far more than fits
    try:
        req = dict(EXAMPLES["standard"])
        req["game"] = dict(req["game"], timeout=timeout)
        call(lf, "POST", "/az-256/start", req)
        for _ in range(3):
            t = time.perf_counter()
            code, out = call(lf, "POST", "/az-256/move", req)
            ms = (time.perf_counter() - t) * 1e3
            assert code == 200 and out["move"] in MOVES
            due = timeout - lf.MARGIN_MS
            assert due - 5 < ms < due + 25, f"timeout {timeout}: answered after {ms:.0f} ms"
        assert all(d["by"] == "native" for d in move_logs(capsys))
    finally:
        del os.environ["SNAKE_SIMULATIONS_AZ_256"]


def test_engine_crash_and_hang_recover(lf, capsys) -> None:
    req = EXAMPLES["duel"]
    call(lf, "POST", "/az-256/start", req)
    assert call(lf, "POST", "/az-256/move", req)[0] == 200
    first = lf._engine.proc.pid
    # Killed between moves: the next move restarts it and is still the engine's.
    os.kill(first, signal.SIGKILL)
    time.sleep(0.05)
    code, out = call(lf, "POST", "/az-256/move", dict(req, turn=1))
    assert code == 200 and out["move"] in MOVES
    second = lf._engine.proc.pid
    assert second != first
    assert move_logs(capsys)[-1]["by"] == "native"
    # Frozen mid-game: the move is the safe move in time, and the engine is replaced.
    os.kill(second, signal.SIGSTOP)
    t = time.perf_counter()
    code, out = call(lf, "POST", "/az-256/move", dict(req, turn=2))
    ms = (time.perf_counter() - t) * 1e3
    assert code == 200 and out["move"] in MOVES
    assert ms < 500 - lf.MARGIN_MS + lf.GRACE_MS + 20, ms
    assert move_logs(capsys)[-1]["by"] == "safe_move"
    assert lf._engine.proc is None
    code, out = call(lf, "POST", "/az-256/move", dict(req, turn=3))
    assert code == 200 and lf._engine.proc.pid not in (first, second)
    assert move_logs(capsys)[-1]["by"] == "native"
    # Killed while a /end is in flight, then a new game: fine.
    call(lf, "POST", "/az-256/end", req)
    os.kill(lf._engine.proc.pid, signal.SIGKILL)
    assert call(lf, "POST", "/az-256/start", req) == (200, {})
    assert call(lf, "POST", "/az-256/move", req)[0] == 200
    assert move_logs(capsys)[-1]["by"] == "native"


def test_broken_engine_gives_the_safe_move(lf, capsys, monkeypatch, tmp_path) -> None:
    junk = tmp_path / "junk.sh"
    junk.write_text("#!/bin/sh\nwhile read l; do echo 'not json'; done\n")
    junk.chmod(0o755)
    for bad in ("/nonexistent/slinky-engine", "/bin/false", str(junk)):
        monkeypatch.setattr(lf, "ENGINE_BIN", Path(bad))
        lf._engine.stop("next engine")
        assert call(lf, "POST", "/az-256/start", EXAMPLES["duel"]) == (200, {})
        code, out = call(lf, "POST", "/az-256/move", EXAMPLES["duel"])
        assert code == 200 and out["move"] in MOVES, bad
        assert move_logs(capsys)[-1]["by"] == "safe_move", bad
        assert call(lf, "POST", "/az-256/end", EXAMPLES["duel"]) == (200, {})


def test_numpy_snakes_are_unchanged(lf) -> None:
    """Every numpy snake answers as the handler before the native snakes (git HEAD's
    ``lambda_function.py`` if it has no NATIVE, else the first commit without it)."""
    src = None
    for rev in ("HEAD", "c3b75ae"):
        text = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{rev}:deploy/lambda/lambda_function.py"],
            capture_output=True,
            text=True,
        ).stdout
        if text and "NATIVE" not in text:
            src = text
            break
    if src is None:
        pytest.skip("no earlier lambda_function.py in git")
    old = types.ModuleType("lambda_function_old")
    old.__file__ = str(LAMBDA_DIR / "lambda_function.py")
    exec(compile(src, "lambda_function_old.py", "exec"), old.__dict__)
    reqs = [EXAMPLES["duel"], dict(EXAMPLES["duel"], turn=9), EXAMPLES["standard"]]
    for name in lf.SNAKES:
        for method, path, body in [("GET", f"/{name}", None)] + [
            ("POST", f"/{name}/{a}", r) for r in reqs for a in ("start", "move", "end")
        ]:
            assert lf.lambda_handler(event(method, path, body), None) == old.lambda_handler(
                event(method, path, body), None
            ), (name, path)
