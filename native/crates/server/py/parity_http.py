"""HTTP parity: the Python ladder server and the native server, same requests, same answers.

Starts the Python server (``slinky.server.main``, in a thread of this process, the deployed
agent ``az_modes:<run>:simulations=256``) and the native server (``--native-bin``, a child
process) with the same snake, then sends both the same sequence of requests, one at a time and
in the same order (so both move counters, and so both tie-noise keys, stay in step):

* real-shaped games in every ladder mode and spelling (standard 4p, duel, royale on the royale
  map with the royale or the standard ruleset and on the standard map, constrictor on the
  standard and the empty map), played by the heuristic for the other snakes and by the
  servers' own answers for ours (so the move log sees applied answers), with some turns
  skipped (a ``gap``) and some answers overridden (a ``missed``), each game opened with
  ``/start`` and closed with ``/end``, with the ladder's settings and extra fields;
* junk between games: bodies that are not JSON or not objects, unknown rulesets, boards past
  the size limit, too much food, unhashable or missing game ids, float sizes, a missing turn,
  info and 404 routes.

It compares every response (status and body bytes) and then the two move logs line by line,
field by field (all but ``t``, ``ms`` and ``wall``). Run (CPU, memory capped; the native
server runs under the same cap):

    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/crates/server/py/parity_http.py \\
        --native-bin /dev/shm/naz-target-server/release/slinky-native-server \\
        [--run /home/nathan/slinky-serve/az-modes-M-s0] [--games 3] [--out DIR]

Writes ``<out>/py.jsonl`` and ``<out>/rs.jsonl`` (the move logs), ``<out>/report.json`` and
prints a summary. Exit status 0 when everything matched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import jax
import numpy as np

from slinky import server
from slinky.agents import make_agent, make_env
from slinky.engine_json import state_to_engine

AGENT = "az_modes:{run}:simulations=256"
LOOK = ["--author", "nathanWolo", "--color", "#ef4444", "--head", "cosmic-horror",
        "--tail", "cosmic-horror"]  # fmt: skip
LADDER = {
    "foodSpawnChance": 15,
    "minimumFood": 1,
    "hazardDamagePerTurn": 14,
    "hazardMap": "",
    "hazardMapAuthor": "",
    "royale": {"shrinkEveryNTurns": 25},
    "squad": {"allowBodyCollisions": False, "sharedElimination": False,
              "sharedHealth": False, "sharedLength": False},
}  # fmt: skip
# (label, mode config, ruleset name, map name) of the ways the ladder spells each mode.
GAMES = [
    ("standard", "standard", "standard", "standard"),
    ("duel", "duel", "standard", "standard"),
    ("royale", "royale", "royale", "royale"),
    ("royale-std-ruleset", "royale", "standard", "royale"),
    ("royale-std-map", "royale", "royale", "standard"),
    ("constrictor", "constrictor", "constrictor", "standard"),
    ("constrictor-empty", "constrictor", "constrictor", "empty"),
]
MOVES = server.MOVE_NAMES


def http(port: int, method: str, path: str, body: bytes | None = None, ray: str | None = None):
    """``(status, body bytes)`` of one request."""
    headers = {"Content-Type": "application/json"}
    if ray:
        headers["Cf-Ray"] = ray
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", body, headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_up(port: int, timeout: float) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            if http(port, "GET", "/")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"server on port {port} did not start in {timeout} s")


def api_snake(s: dict) -> dict:
    return {
        "id": s["id"], "name": f"snake {s['id']}", "health": s["health"],
        "body": [{"x": x, "y": y} for x, y in s["body"]],
        "head": {"x": s["body"][0][0], "y": s["body"][0][1]},
        "length": len(s["body"]), "latency": "42", "shout": "",
        "squad": "", "customizations": {"color": "#888888", "head": "default", "tail": "default"},
    }  # fmt: skip


def request(d: dict, you: str, game_id: str, ruleset: str, game_map: str) -> dict:
    """The API request of engine dict ``d`` from snake ``you``'s side (the living snakes)."""
    snakes = [api_snake(s) for s in d["snakes"] if not s.get("eliminated_cause")]
    me = next((s for s in snakes if s["id"] == you), None)
    if me is None:  # our snake is dead (only for /end): the engine still sends the last you
        me = api_snake(next(s for s in d["snakes"] if s["id"] == you))
    return {
        "game": {"id": game_id, "ruleset": {"name": ruleset, "version": "v1.2.3",
                 "settings": LADDER}, "map": game_map, "timeout": 500, "source": "ladder"},
        "turn": d["turn"],
        "board": {"height": d["height"], "width": d["width"],
                  "food": [{"x": x, "y": y} for x, y in d["food"]],
                  "hazards": [{"x": x, "y": y} for x, y in d["hazards"]], "snakes": snakes},
        "you": me,
    }  # fmt: skip


class Duo:
    """Sends each request to both servers and compares the answers."""

    def __init__(self, py_port: int, rs_port: int) -> None:
        self.ports = (py_port, rs_port)
        self.count = 0
        self.mismatches: list[dict] = []
        self.known: dict[str, int] = {}  # expected differences, by kind
        self.kinds: dict[str, int] = {}

    def send(self, kind: str, method: str, path: str, body: bytes | None = None, ray=None):
        py = http(self.ports[0], method, path, body, ray)
        rs = http(self.ports[1], method, path, body, ray)
        self.count += 1
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        same = py[0] == rs[0] and (py[1] == rs[1] or (py[0] == 400 and rs[0] == 400))
        if py[0] == rs[0] == 400 and py[1] != rs[1]:
            self.known["400 error text"] = self.known.get("400 error text", 0) + 1
        # The info's version names the engine: "slinky <agent>" vs "slinky-native <agent>".
        if not same and py[0] == rs[0] == 200 and py[1].replace(
            b'"version": "slinky ', b'"version": "slinky-native '
        ) == rs[1]:
            self.known["info version"] = self.known.get("info version", 0) + 1
            same = True
        if not same:
            self.mismatches.append({"kind": kind, "path": path, "python": [py[0], py[1].decode(
                "utf-8", "replace")], "native": [rs[0], rs[1].decode("utf-8", "replace")],
                "body": (body or b"")[:300].decode("utf-8", "replace")})  # fmt: skip
        return py, rs


def play_game(duo: Duo, label: str, mode: str, ruleset: str, game_map: str, seed: int,
              max_turns: int, rng: random.Random) -> dict:  # fmt: skip
    """One game: /start, a /move per turn while our snake lives, /end."""
    config = server.MODE_CONFIGS[mode]
    if game_map == "empty":
        import dataclasses

        config = dataclasses.replace(config, map="empty")  # no food spawns, as the ladder's
    heur = make_agent("heuristic", config)
    env = make_env(config, heur.needs_obs)
    policy, step = jax.jit(heur.policy), jax.jit(env.step)
    key = jax.random.key(seed)
    key, k = jax.random.split(key)
    state, ts = env.reset(k)
    seat = rng.randrange(config.num_snakes)
    you = f"s{seat}"
    gid = f"parity-{label}-{seed}"
    path = "/az-256"
    stats = {"moves": 0, "skipped": 0, "overridden": 0}
    d = state_to_engine(state, config)
    duo.send("start", "POST", path + "/start", json.dumps(request(d, you, gid, ruleset, game_map)).encode())
    for _ in range(max_turns):
        if bool(state.done) or not bool(state.alive[seat]):
            break
        key, k1, k2 = jax.random.split(key, 3)
        actions = np.array(policy(k1, state, ts))
        d = state_to_engine(state, config)
        if rng.random() < 0.05 and int(state.turn) > 0:
            stats["skipped"] += 1  # this turn's request "never reaches us": a gap next turn
        else:
            body = json.dumps(request(d, you, gid, ruleset, game_map)).encode()
            ray = f"{rng.getrandbits(64):016x}-{rng.choice(['SEA', 'YVR', 'PDX'])}"
            (code, answer), _ = duo.send("move", "POST", path + "/move", body, ray)
            stats["moves"] += 1
            if code == 200:
                actions[seat] = MOVES.index(json.loads(answer)["move"])
            if rng.random() < 0.05:
                actions[seat] = (actions[seat] + 1) % 4  # the engine "missed" our answer
                stats["overridden"] += 1
        state, ts = step(k2, state, jax.numpy.asarray(actions))
    d = state_to_engine(state, config)
    duo.send("end", "POST", path + "/end", json.dumps(request(d, you, gid, ruleset, game_map)).encode())
    stats["turns"] = int(state.turn)
    return stats


def junk(duo: Duo, rng: random.Random, base: dict) -> None:
    """Requests that are not games: errors, fallbacks, routes."""
    p = "/az-256"
    for body in (b"{not json", b"[1, 2]", b'{"game": 3}', b'{"game": []}', b"null"):
        duo.send("junk-400", "POST", p + "/move", body)
    variants = []
    r = json.loads(json.dumps(base))
    r["game"]["ruleset"]["name"] = "wrapped"
    variants.append(r)
    r = json.loads(json.dumps(base))
    r["board"]["width"] = r["board"]["height"] = 99
    variants.append(r)
    r = json.loads(json.dumps(base))
    r["board"]["food"] = [{"x": 0, "y": 0}] * 122
    variants.append(r)
    r = json.loads(json.dumps(base))
    r["game"]["id"] = ["unhashable"]
    variants.append(r)
    r = json.loads(json.dumps(base))
    del r["game"]["id"]
    variants.append(r)
    r = json.loads(json.dumps(base))
    r["board"]["width"] = 11.0
    variants.append(r)
    r = json.loads(json.dumps(base))
    del r["turn"]
    variants.append(r)
    r = json.loads(json.dumps(base))
    r["game"]["id"] = 12345
    variants.append(r)
    variants.append({"board": 3})
    variants.append({})
    rng.shuffle(variants)
    for r in variants:
        duo.send("junk-move", "POST", p + "/move", json.dumps(r).encode())
    duo.send("junk-start", "POST", p + "/start", b'{"board": 3}')
    duo.send("junk-end", "POST", p + "/end", b'{"game": {"id": [1]}}')
    for method, path in (("GET", "/"), ("GET", p), ("GET", p + "/"), ("GET", p + "/move"),
                         ("GET", "/nope"), ("POST", "/nope/move"), ("POST", p + "/x")):  # fmt: skip
        duo.send("route", method, path, b"{}" if method == "POST" else None)


def compare_logs(py_path: str, rs_path: str) -> dict:
    py = [json.loads(line) for line in open(py_path)]
    rs = [json.loads(line) for line in open(rs_path)]
    out = {"python_lines": len(py), "native_lines": len(rs), "line_mismatches": [],
           "field_mismatch_counts": {}, "known": {}}  # fmt: skip
    for i, (a, b) in enumerate(zip(py, rs)):
        if list(a) != list(b):
            out["line_mismatches"].append({"line": i, "keys": [list(a), list(b)]})
            continue
        for k in a:
            if k in ("t", "ms", "wall"):
                continue
            if k == "agent" and (a[k], b[k]) == ("none", "safe_move"):
                # Python names a move "none" where its own safe_move raised; native does not
                # tell those apart (the move is "up" in both).
                out["known"]["agent none"] = out["known"].get("agent none", 0) + 1
                continue
            if a[k] != b[k]:
                out["field_mismatch_counts"][k] = out["field_mismatch_counts"].get(k, 0) + 1
                if len(out["line_mismatches"]) < 30:
                    out["line_mismatches"].append({"line": i, "field": k, "python": a[k],
                                                   "native": b[k]})  # fmt: skip
    verdicts: dict[str, int] = {}
    for a in py:
        verdicts[str(a["prev"])] = verdicts.get(str(a["prev"]), 0) + 1
    out["python_verdicts"] = verdicts
    out["ms_python_median"] = float(np.median([a["ms"] for a in py if not a.get("end")]))
    out["ms_native_median"] = float(np.median([b["ms"] for b in rs if not b.get("end")]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--native-bin", required=True)
    ap.add_argument("--run", default="/home/nathan/slinky-serve/az-modes-M-s0")
    ap.add_argument("--py-port", type=int, default=18100)
    ap.add_argument("--rs-port", type=int, default=18200)
    ap.add_argument("--games", type=int, default=3, help="games per mode spelling")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/dev/shm/naz-server-parity")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    py_log, rs_log = os.path.join(args.out, "py.jsonl"), os.path.join(args.out, "rs.jsonl")
    for f in (py_log, rs_log):
        if os.path.exists(f):
            os.remove(f)
    agent = AGENT.format(run=args.run)
    native = subprocess.Popen(
        [args.native_bin, "--agent", f"az-256={agent}", "--port", str(args.rs_port), *LOOK,
         "--move-log", rs_log, "--threads", "2"],
        stdout=open(os.path.join(args.out, "native.log"), "w"), stderr=subprocess.STDOUT,
    )  # fmt: skip
    py_argv = ["--agent", f"az-256={agent}", "--port", str(args.py_port), *LOOK,
               "--move-log", py_log]  # fmt: skip
    threading.Thread(target=server.main, args=(py_argv,), daemon=True).start()
    try:
        t = time.time()
        wait_up(args.rs_port, 60)
        wait_up(args.py_port, 1500)
        print(f"both servers up after {time.time() - t:.0f} s", flush=True)
        duo = Duo(args.py_port, args.rs_port)
        info = {}
        for path in ("/", "/az-256"):
            py, rs = http(args.py_port, "GET", path), http(args.rs_port, "GET", path)
            info[path] = {"python": py[1].decode(), "native": rs[1].decode()}
        rng = random.Random(args.seed)
        games, seed = [], 1000 * args.seed
        base = None
        for g in range(args.games):
            order = list(GAMES)
            rng.shuffle(order)
            for label, mode, ruleset, game_map in order:
                seed += 1
                stats = play_game(duo, label, mode, ruleset, game_map, seed, args.max_turns, rng)
                games.append({"label": label, "seed": seed, **stats})
                if base is None:
                    d = state_to_engine(make_env(server.MODE_CONFIGS["duel"], False).reset(
                        jax.random.key(seed))[0], server.MODE_CONFIGS["duel"])
                    base = request(d, "s0", "junk-game", "standard", "standard")
                junk(duo, rng, base)
            print(f"round {g + 1}/{args.games}: {duo.count} requests, "
                  f"{len(duo.mismatches)} mismatches", flush=True)  # fmt: skip
    finally:
        native.send_signal(signal.SIGTERM)
        native.wait(timeout=30)
    time.sleep(0.5)
    logs = compare_logs(py_log, rs_log)
    report = {"requests": duo.count, "kinds": duo.kinds, "response_mismatches": duo.mismatches,
              "known_response_differences": duo.known,
              "info": info, "games": games, "logs": logs}  # fmt: skip
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    moves = duo.kinds.get("move", 0)
    print(json.dumps({k: v for k, v in report.items() if k not in ("games",)}, indent=1)[:6000])
    print(f"requests {duo.count} (game moves {moves}), response mismatches "
          f"{len(duo.mismatches)}; log lines {logs['python_lines']} / {logs['native_lines']}, "
          f"field mismatches {logs['field_mismatch_counts']}")  # fmt: skip
    moves_logged = sum(1 for line in open(py_log) if '"end": true' not in line)
    print(f"move lines {moves_logged}; known differences {duo.known} {logs['known']}")
    ok = not duo.mismatches and not logs["line_mismatches"] and (
        logs["python_lines"] == logs["native_lines"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
