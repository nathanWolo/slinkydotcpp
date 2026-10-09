"""Python bridge to the Go rules-engine oracle in ``tools/oracle``.

The oracle links the official Battlesnake rules engine (AGPL-3.0); it is a
test-only tool and is never imported by, or shipped with, the ``slinky``
package. See ``tools/oracle/README.md`` for the wire format.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_DIR = REPO_ROOT / "tools" / "oracle"
ORACLE_BIN = ORACLE_DIR / "bin" / "oracle"
CACHE_DIR = REPO_ROOT / ".oracle-cache"

# The engine's CLI defaults (cli/commands/play.go), keyed as the engine expects.
DEFAULT_SETTINGS: dict[str, str] = {
    "foodSpawnChance": "15",
    "minimumFood": "1",
    "damagePerTurn": "14",
    "shrinkEveryNTurns": "25",
}

# Defaults of `oracle games`, as Python keyword names (dashes -> underscores).
DEFAULT_GAME_FLAGS: dict[str, Any] = {
    "ruleset": "standard",
    "map": "standard",
    "width": 11,
    "height": 11,
    "snakes": 2,
    "games": 10,
    "seed": 1,
    "max_turns": 500,
    "food_spawn_chance": 15,
    "minimum_food": 1,
    "hazard_damage": 14,
    "shrink_every_n_turns": 25,
    "invalid_move_prob": 0.0,
    "p_random": 0.03,
    "p_aggressive": 0.3,
    "p_food": 0.3,
    "p_food_averse": 0.15,
    "p_careful": 0.75,
}


class OracleError(RuntimeError):
    """The oracle process misbehaved (crashed, bad output, ...)."""


class OracleUnavailable(OracleError):
    """The oracle cannot be built here (no Go toolchain, build failure, ...)."""


# --------------------------------------------------------------------------- build


def find_go() -> str | None:
    """Locate the ``go`` binary: $ORACLE_GO, then PATH, then /usr/local/go."""
    env = os.environ.get("ORACLE_GO")
    if env:
        return env
    found = shutil.which("go")
    if found:
        return found
    fallback = Path("/usr/local/go/bin/go")
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None


def _source_files() -> list[Path]:
    files = [p for p in ORACLE_DIR.rglob("*.go") if "bin" not in p.relative_to(ORACLE_DIR).parts]
    files += [p for p in (ORACLE_DIR / "go.mod", ORACLE_DIR / "go.sum") if p.exists()]
    return sorted(files)


def _source_digest() -> str:
    """Hash of the oracle sources; part of the games cache key."""
    h = hashlib.sha256()
    for p in _source_files():
        h.update(str(p.relative_to(ORACLE_DIR)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _needs_build(binary: Path) -> bool:
    if not binary.exists():
        return True
    bin_mtime = binary.stat().st_mtime
    return any(p.stat().st_mtime > bin_mtime for p in _source_files())


def build_oracle(force: bool = False) -> Path:
    """Build ``tools/oracle/bin/oracle`` if missing or stale; return its path.

    Raises OracleUnavailable if a build is needed but Go is missing or the
    build fails (the first build downloads the engine module).
    """
    if not force and not _needs_build(ORACLE_BIN):
        return ORACLE_BIN
    go = find_go()
    if go is None:
        raise OracleUnavailable("Go toolchain not found (install Go or set ORACLE_GO)")
    ORACLE_BIN.parent.mkdir(parents=True, exist_ok=True)
    # Build to a temporary name and rename, so concurrent builders never
    # observe a half-written binary.
    tmp = ORACLE_BIN.with_name(f".oracle.{os.getpid()}.tmp")
    try:
        try:
            proc = subprocess.run(
                [go, "build", "-o", str(tmp), "."],
                cwd=ORACLE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as e:
            raise OracleUnavailable(f"could not run {go!r}: {e}") from e
        if proc.returncode != 0:
            raise OracleUnavailable(f"go build failed:\n{proc.stdout}{proc.stderr}".rstrip())
        os.replace(tmp, ORACLE_BIN)
    finally:
        if tmp.exists():
            tmp.unlink()
    return ORACLE_BIN


# --------------------------------------------------------------------------- step


def make_step_request(
    ruleset: str,
    state: dict,
    moves: Sequence[str] | None,
    settings: dict | None = None,
    solo: bool = False,
    seed: int = 1,
) -> dict:
    """Build one ``oracle step`` request.

    ``moves[i]`` is the move of snake ``i`` (send "" for eliminated snakes);
    ``moves=None`` calls ``Execute(state, nil)`` like the CLI's turn-0 init.
    ``settings`` default to the CLI defaults; values are stringified.
    """
    if settings is None:
        settings = DEFAULT_SETTINGS
    return {
        "ruleset": ruleset,
        "solo": bool(solo),
        "settings": {str(k): str(v) for k, v in settings.items()},
        "seed": int(seed),
        "state": state,
        "moves": None if moves is None else [str(m) for m in moves],
    }


class OracleStepper:
    """A persistent ``oracle step`` subprocess.

    Use as a context manager::

        with OracleStepper() as oracle:
            resp = oracle.step("standard", state, ["up", "left"])
            resp["game_over"], resp["state"], resp["error"]
    """

    def __init__(self, binary: Path | str | None = None):
        self.binary = Path(binary) if binary is not None else build_oracle()
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- lifecycle
    def start(self) -> OracleStepper:
        if self._proc is None:
            self._proc = subprocess.Popen(
                [str(self.binary), "step"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        finally:
            if proc.stdout:
                proc.stdout.close()

    def __enter__(self) -> OracleStepper:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- protocol
    def _pipes(self):
        proc = self.start()._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        if proc.poll() is not None:
            raise OracleError(f"oracle step process exited with code {proc.returncode}")
        return proc.stdin, proc.stdout

    def _read_response(self, stdout) -> dict:
        line = stdout.readline()
        if not line:
            code = self._proc.poll() if self._proc else None
            raise OracleError(f"oracle step process closed its output (exit code {code})")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise OracleError(f"oracle step returned invalid JSON: {line!r}") from e

    def request(self, req: dict) -> dict:
        """Send one raw request dict, return the response dict."""
        with self._lock:
            stdin, stdout = self._pipes()
            stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            stdin.flush()
            return self._read_response(stdout)

    def step(
        self,
        ruleset: str,
        state: dict,
        moves: Sequence[str] | None,
        settings: dict | None = None,
        solo: bool = False,
        seed: int = 1,
    ) -> dict:
        """Run ``ruleset.Execute(state, moves)``.

        Returns ``{"game_over": bool, "state": dict | None, "error": str}``;
        ``state`` is None and ``error`` non-empty if the engine errored.
        """
        return self.request(make_step_request(ruleset, state, moves, settings, solo, seed))

    def step_many(self, requests: Iterable[dict]) -> list[dict]:
        """Pipeline many requests (dicts as from :func:`make_step_request`).

        Requests are written from a background thread while responses are
        read, so arbitrarily many can be sent without filling the pipes.
        """
        payload = [json.dumps(r, separators=(",", ":")) + "\n" for r in requests]
        with self._lock:
            stdin, stdout = self._pipes()
            write_error: list[BaseException] = []

            def writer() -> None:
                try:
                    for line in payload:
                        stdin.write(line)
                    stdin.flush()
                except BaseException as e:  # surfaced below
                    write_error.append(e)

            t = threading.Thread(target=writer, daemon=True)
            t.start()
            try:
                responses = [self._read_response(stdout) for _ in payload]
            finally:
                t.join()
            if write_error:
                raise OracleError(f"writing to oracle failed: {write_error[0]!r}")
            return responses


# --------------------------------------------------------------------------- games


def _games_args(flags: dict[str, Any]) -> list[str]:
    args = []
    for k, v in flags.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    return args


def generate_games(
    *, binary: Path | str | None = None, use_cache: bool = True, **flags: Any
) -> list[dict]:
    """Run ``oracle games`` and return its parsed JSON Lines records.

    Keyword flags use underscores (``max_turns=200``, ``invalid_move_prob=0.05``);
    see ``DEFAULT_GAME_FLAGS``. Output is cached under ``.oracle-cache/``,
    keyed by the full flag set and a hash of the oracle sources.
    """
    unknown = set(flags) - set(DEFAULT_GAME_FLAGS)
    if unknown:
        raise TypeError(f"unknown oracle games flags: {sorted(unknown)}")
    full = {**DEFAULT_GAME_FLAGS, **flags}

    cache_file = None
    if use_cache:
        key_src = json.dumps({"flags": full, "oracle": _source_digest()}, sort_keys=True)
        key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
        cache_file = CACHE_DIR / f"games-{key}.jsonl"
        if cache_file.exists():
            return _parse_jsonl(cache_file.read_text(encoding="utf-8"))

    exe = Path(binary) if binary is not None else build_oracle()
    proc = subprocess.run(
        [str(exe), "games", *_games_args(full)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise OracleError(f"oracle games failed ({proc.returncode}): {proc.stderr.strip()}")
    records = _parse_jsonl(proc.stdout)

    if cache_file is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, prefix=".games-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        os.replace(tmp, cache_file)
    return records


def _parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def split_games(records: Iterable[dict]) -> list[dict]:
    """Group ``generate_games`` records per game.

    Returns ``[{"game": g, "config": ..., "initial": STATE, "transitions": [...]}, ...]``.
    """
    games: dict[int, dict] = {}
    for r in records:
        if r["kind"] == "initial":
            games[r["game"]] = {
                "game": r["game"],
                "config": r["config"],
                "initial": r["state"],
                "transitions": [],
            }
        elif r["kind"] == "transition":
            games[r["game"]]["transitions"].append(r)
        else:
            raise OracleError(f"unknown record kind {r['kind']!r}")
    return [games[g] for g in sorted(games)]
