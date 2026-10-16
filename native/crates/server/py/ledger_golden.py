"""Golden vectors for the native server's MoveLedger port (native/crates/server/src/ledger.rs).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/crates/server/py/ledger_golden.py [--out DIR] [--seed S]

Drives Python's ``server.MoveLedger.record`` the way the HTTP handler does (``/move``: a move,
``end=False``, ``wall_ms`` and ``colo``; ``/end``: no move, ``end=True``, no wall) over a
deterministic fuzz: games played on plain and wrapped boards of several sizes (edge and corner
crossings), answers applied, missed and skipped (gaps), ``/end`` after our death (turn jumps)
and at the next turn, games interleaved across two snakes, more than ``MAX_GAMES`` games (the
oldest is forgotten), and junk (turn, coordinates and sizes as strings, floats, bools, None,
NaN, infinity, huge ints; game ids of every JSON type; ``board.snakes`` of every type).

Output: ``<out>/server/ledger.json`` (default out /dev/shm/naz-golden; RAM, never committed):

    {"calls": [{"snake": str, "request": str (json.dumps text, NaN/Infinity allowed),
                "move": str | null, "agent": str, "ms": float, "end": bool,
                "wall": float | null, "colo": str | null,
                "verdict": str | null,          # record's return value
                "raised": str | null,           # exception class name when record raised
                "line": str | null}]}           # the move-log line written, with "t" removed

``line`` is the text after ``"t": <t>, `` (the line starts ``{`` + that text), or null when no
line was written.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from pathlib import Path

from slinky import server

MOVES = ["up", "down", "left", "right"]
DELTA = server.API_DELTAS


def request(gid, turn, x, y, w, h, *, snakes=2, ruleset="standard", extra=None):
    req = {
        "game": {
            "id": gid,
            "timeout": 500,
            "source": "ladder",
            "map": "standard",
            "ruleset": {"name": ruleset},
        },
        "turn": turn,
        "board": {"width": w, "height": h, "snakes": [{"id": f"s{i}"} for i in range(snakes)]},
        "you": {"id": "s0", "head": {"x": x, "y": y}},
    }
    if extra:
        extra(req)
    return req


class Recorder:
    def __init__(self, path: str) -> None:
        self.path = path
        self.ledger = server.MoveLedger(path)
        self.calls: list[dict] = []
        self.seen = 0

    def call(self, snake, req, move, *, end=False, ms=1.0, wall=None, colo=None, agent="az"):
        text = json.dumps(req)
        # The handler hands record the parsed body.
        parsed = json.loads(text)
        verdict = raised = None
        try:
            if end:
                verdict = self.ledger.record(snake, parsed, None, "", 0.0, end=True)
            else:
                verdict = self.ledger.record(
                    snake, parsed, move, agent, ms, wall_ms=wall, colo=colo
                )
        except Exception as e:  # noqa: BLE001 - recorded as the expected outcome
            raised = type(e).__name__
        with open(self.path) as f:
            lines = f.read().splitlines()
        line = None
        if len(lines) > self.seen:
            assert len(lines) == self.seen + 1
            line = lines[-1]
            m = re.match(r'\{"t": [^,]+, (.*)$', line)
            assert m, line
            line = m.group(1)
            self.seen = len(lines)
        self.calls.append(
            {
                "snake": snake,
                "request": text,
                "move": None if end else move,
                "agent": "" if end else agent,
                "ms": 0.0 if end else ms,
                "end": end,
                "wall": None if end else wall,
                "colo": None if end else colo,
                "verdict": verdict,
                "raised": raised,
                "line": line,
            }
        )
        return verdict


def play_game(rec: Recorder, rng: random.Random, snake: str, gid, w: int, h: int, wrapped: bool):
    """A game of random length; heads follow the answers (or not), with gaps and an /end."""
    x, y = rng.randrange(max(w, 1)), rng.randrange(max(h, 1))
    turn = 0
    for _ in range(rng.randrange(2, 30)):
        move = rng.choice(MOVES)
        ms = round(rng.uniform(0, 60), rng.choice([1, 2, 3, 6]))
        wall = rng.choice([None, ms + rng.uniform(0, 5), 2.25, 0.05, 0.15])
        colo = rng.choice([None, "", "SEA", "ORD", "8d0f2a7b1c3e4f56"])
        rec.call(snake, request(gid, turn, x, y, w, h), move, ms=ms, wall=wall, colo=colo)
        dx, dy = DELTA[move]
        r = rng.random()
        if r < 0.1:  # missed: the engine moved us another way
            dx, dy = DELTA[rng.choice([m for m in MOVES if m != move])]
        nx, ny = x + dx, y + dy
        if wrapped:
            nx, ny = nx % w, ny % h
        x, y = nx, ny
        turn += 1
        if rng.random() < 0.08:  # gap: a request never reached us
            turn += 1
    # /end: at the next turn (the last answer judged), or later (we died earlier).
    if rng.random() < 0.5:
        turn += rng.randrange(1, 6)
    rec.call(snake, request(gid, turn, x, y, w, h), None, end=True)


def junk(rec: Recorder) -> None:
    """Values Python's int(), str() and len() read in odd ways."""
    base = lambda **kw: request(kw.pop("gid", "j"), 0, 3, 3, 11, 11, **kw)  # noqa: E731

    def setp(path, value):
        def f(req):
            d = req
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value

        return f

    values = [
        "5", " 7 ", "1_0", "+3", "-2", "x", "", 2.9, -2.9, 5.0, True, False, None,
        float("nan"), float("inf"), float("-inf"), 10**40, -(10**40), 1e40, [], {}, [1], {"a": 1},
    ]  # fmt: skip
    for field in [("turn",), ("you", "head", "x"), ("you", "head", "y"),
                  ("board", "width"), ("board", "height")]:  # fmt: skip
        for v in values:
            gid = f"junk-{'.'.join(field)}-{values.index(v)}"
            # a first answer, then the next turn with the junk value
            rec.call("s", request(gid, 0, 3, 3, 11, 11), "up")
            req = request(gid, 1, 3, 4, 11, 11, extra=setp(field, v))
            rec.call("s", req, "left")
            rec.call("s", request(gid, 2, 2, 4, 11, 11), "up")
            rec.call("s", request(gid, 3, 2, 5, 11, 11), None, end=True)
    # Game ids of every type: str, int, float, bool, None, missing, list, dict, nested.
    for gid in ["g", 7, 7.0, 7.5, True, None, [1], {"a": 1}, "", "é😀", 10**30, float("nan")]:
        rec.call("s", request(gid, 0, 1, 1, 11, 11), "right")
        rec.call("s", request(gid, 1, 2, 1, 11, 11), "up")
        rec.call("s", request(gid, 2, 2, 2, 11, 11), None, end=True)
    no_id = request("x", 0, 1, 1, 11, 11)
    del no_id["game"]["id"]
    rec.call("s", no_id, "up")
    no_game = request("x", 0, 1, 1, 11, 11)
    del no_game["game"]
    rec.call("s", no_game, "up")
    # board.snakes of every type, a missing board, ruleset and game fields of odd types.
    for snakes in [None, 0, 3, 2.5, "abc", "", {"a": 1, "b": 2}, {}, [], [1, 2, 3], True]:
        rec.call("s", base(gid="sn", extra=setp(("board", "snakes"), snakes)), "up")
    for path, v in [(("game", "ruleset"), "standard"), (("game", "ruleset"), None),
                    (("game", "ruleset", "name"), 3), (("game", "map"), [1, {"x": None}]),
                    (("game", "timeout"), 1.5e300), (("game", "source"), "é\n\t\"\\"),
                    (("turn",), 2**70), (("turn",), 1e16), (("turn",), 1e-7), (("board",), 3),
                    (("board",), None), (("you",), "x"), (("you", "head"), [3, 3])]:  # fmt: skip
        rec.call("s", base(gid="odd", extra=setp(path, v)), "down")
    # Huge but exact coordinates and board sizes: Python's arithmetic is exact.
    big = 10**40
    rec.call("s", request("bigw", 0, 3, 3, big, big), "up")
    rec.call("s", request("bigw", 1, 3, 4, big, big), "right")
    rec.call("s", request("bigx", 0, big, big, 11, 11), "up")
    rec.call("s", request("bigx", 1, big, big + 1, 11, 11), "right")
    rec.call("s", request("bigt", 2**80, 3, 3, 11, 11), "up")
    rec.call("s", request("bigt", 2**80 + 1, 3, 4, 11, 11), "up")
    # Zero and negative board sizes (floor arithmetic, w == 0 skips the modulo).
    for w in [0, -1, -4, 1, 2]:
        gid = f"w{w}"
        rec.call("s", request(gid, 0, 1, 1, w, w), "up")
        rec.call("s", request(gid, 1, 1, 2, w, w), "right")
        rec.call("s", request(gid, 2, 2, 2, w, w), "up")
    # A float turn and coordinates, and a numeric string turn.
    rec.call("s", request("fl", 4.7, 2.9, 3.1, 11.0, 11), "up")
    rec.call("s", request("fl", "5", 2, 4, "11", 11), "left")
    rec.call("s", request("fl", 6.0, 1.5, 4, 11, 11.9), None, end=True)
    # ms and wall rounding.
    for ms in [0.05, 0.15, 0.25, 2.25, 2.35, 1e-5, 12345.65, 999999.95, 0.0, 33.0]:
        rec.call("s", request("ms", 0, 3, 3, 11, 11), "up", ms=ms, wall=ms + 0.05)


def eviction(rec: Recorder) -> None:
    """More games than MAX_GAMES: the oldest is forgotten, an updated game keeps its place."""
    n = server.MAX_GAMES
    for i in range(n):
        rec.call("e", request(f"e{i}", 0, 5, 5, 11, 11), "up")
    rec.call("e", request("e0", 1, 5, 6, 11, 11), "up")  # e0 updated: keeps its place (oldest)
    rec.call("e", request("new", 0, 5, 5, 11, 11), "up")  # evicts e0
    rec.call("e", request("e0", 2, 5, 7, 11, 11), "up")  # e0 forgotten: null; evicts e1
    rec.call("e", request("e1", 1, 5, 6, 11, 11), "up")  # null; evicts e2
    rec.call("e", request("e3", 1, 5, 6, 11, 11), "up")  # applied
    rec.call("e", request("e5", 1, 5, 6, 11, 11), None, end=True)  # removes e5
    rec.call("e", request("new2", 0, 5, 5, 11, 11), "up")  # room: nothing evicted
    rec.call("e", request("e4", 1, 5, 6, 11, 11), "up")  # applied
    # A missing turn pops the key (last is None) ...
    req = request("e6", 1, 5, 6, 11, 11)
    del req["turn"]
    rec.call("e", req, "up")
    rec.call("e", request("e6", 2, 5, 7, 11, 11), "up")  # null


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/dev/shm/naz-golden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=400)
    args = p.parse_args()
    rng = random.Random(args.seed)
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        rec = Recorder(path)
        sizes = [(11, 11, False), (11, 11, True), (7, 7, True), (19, 7, True), (7, 19, False),
                 (1, 5, True), (2, 2, True), (25, 25, True)]  # fmt: skip
        live = []
        for g in range(args.games):
            w, h, wrapped = rng.choice(sizes)
            snake = rng.choice(["s", "t"])
            live.append((snake, f"game-{g}", w, h, wrapped))
        for snake, gid, w, h, wrapped in live:
            play_game(rec, rng, snake, gid, w, h, wrapped)
        junk(rec)
        eviction(rec)
    finally:
        os.unlink(path)
    out = Path(args.out) / "server"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ledger.json").write_text(json.dumps({"calls": rec.calls}))
    verdicts = {}
    for c in rec.calls:
        verdicts[c["raised"] or c["verdict"]] = verdicts.get(c["raised"] or c["verdict"], 0) + 1
    print(f"{len(rec.calls)} calls -> {out / 'ledger.json'}: {verdicts}")


if __name__ == "__main__":
    main()
