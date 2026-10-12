"""Play games against a deployed snake URL and check its moves and latency.

The remote snake plays seat 0 through the real Battlesnake API (``/start``,
``/move``, ``/end``); a local agent plays seat 1. Every remote move is compared
with the same snake's local numpy network (``snake_numpy.py``) on the same
request::

    python deploy/lambda/check_live.py https://<id>.lambda-url.us-east-2.on.aws/ppo \
        --games 8 --parallel 4 --opponent heuristic
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snake_numpy as sn  # noqa: E402
from lambda_function import SNAKES  # noqa: E402  (loads no network)

from slinky.agents import make_agent, make_env  # noqa: E402
from slinky.engine_json import state_to_engine  # noqa: E402
from slinky.types import GameConfig  # noqa: E402

DUEL = GameConfig()


def to_request(d: dict, game_id: str, turn: int) -> dict:
    every = [
        {
            "id": s["id"],
            "name": s["id"],
            "health": s["health"],
            "body": [{"x": x, "y": y} for x, y in s["body"]],
            "head": {"x": s["body"][0][0], "y": s["body"][0][1]},
            "length": len(s["body"]),
            "latency": "0",
            "shout": "",
        }
        for s in d["snakes"]
    ]
    # The board lists the living snakes; "you" is seat 0 even once it is out (at /end).
    snakes = [
        s for s, raw in zip(every, d["snakes"], strict=True) if not raw.get("eliminated_cause")
    ]
    return {
        "game": {
            "id": game_id,
            "ruleset": {"name": "standard", "version": "v1.2.3"},
            "map": "standard",
            "timeout": 500,
            "source": "custom",
        },
        "turn": turn,
        "board": {
            "width": d["width"],
            "height": d["height"],
            "food": [{"x": x, "y": y} for x, y in d["food"]],
            "hazards": [{"x": x, "y": y} for x, y in d["hazards"]],
            "snakes": snakes,
        },
        "you": every[0],
    }


def post(url: str, body: dict) -> tuple[float, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read() or b"{}")
    return (time.perf_counter() - t) * 1e3, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("url", help="a snake's base URL (ending in its name, e.g. /ppo)")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--opponent", default="heuristic")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    base = args.url.rstrip("/")

    t = time.perf_counter()
    with urllib.request.urlopen(base, timeout=10) as r:
        print(f"GET {base}: {json.loads(r.read())} ({(time.perf_counter() - t) * 1e3:.0f} ms)")

    name = base.rsplit("/", 1)[-1]
    if name not in SNAKES:
        p.error(f"the URL must end in a snake name: {', '.join(SNAKES)}")
    folder = ROOT / "baselines" / "checkpoints" / SNAKES[name][0]
    net = sn.Net(str(folder / "params.npz"), json.loads((folder / "config.json").read_text()))
    env = make_env(DUEL, False)
    opp = make_agent(args.opponent, DUEL)
    reset, step, opp_policy = jax.jit(env.reset), jax.jit(env.step), jax.jit(opp.policy)
    lock = threading.Lock()
    lat, mismatches, results = [], [], []

    def play(g: int) -> None:
        key = jax.random.key(args.seed * 1000 + g)
        key, k = jax.random.split(key)
        with lock:  # jitted calls are thread-safe, but compile once
            state, ts = reset(k)
        gid = f"check-{args.seed}-{g}"
        post(base + "/start", to_request(state_to_engine(state, DUEL), gid, 0))
        turn = 0
        while not bool(state.done):
            req = to_request(state_to_engine(state, DUEL), gid, turn)
            ms, out = post(base + "/move", req)
            remote = sn.MOVE_NAMES.index(out["move"])
            local = net.move(sn.Board(req))
            key, k1, k2 = jax.random.split(key, 3)
            theirs = opp_policy(k1, state, ts)
            state, ts = step(k2, state, theirs.at[0].set(remote))
            turn += 1
            with lock:
                lat.append(ms)
                if remote != local:
                    mismatches.append((gid, turn, out["move"], sn.MOVE_NAMES[local]))
        end = to_request(state_to_engine(state, DUEL), gid, turn)
        post(base + "/end", end)
        alive = np.asarray(state.alive)
        with lock:
            results.append(
                "W" if alive[0] and not alive[1] else "L" if alive[1] and not alive[0] else "D"
            )

    t = time.perf_counter()
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(play, range(args.games)))
    a = np.sort(np.array(lat))
    print(
        f"{args.games} games vs {args.opponent} ({args.parallel} at a time) in "
        f"{time.perf_counter() - t:.0f} s: W/D/L {results.count('W')}/{results.count('D')}/"
        f"{results.count('L')}"
    )
    print(
        f"{len(a)} moves: round trip median {np.median(a):.0f} ms, "
        f"p95 {a[int(0.95 * len(a))]:.0f} ms, "
        f"max {a[-1]:.0f} ms; over 400 ms: {(a > 400).sum()}"
    )
    print(f"moves differing from the local numpy {name}: {len(mismatches)}", mismatches[:5])


if __name__ == "__main__":
    main()
