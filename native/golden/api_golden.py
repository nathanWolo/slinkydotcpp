"""Golden vectors for the Rust api crate (native/crates/api).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/api_golden.py [--scale F] [--out DIR] [--seed S]

Builds Battlesnake API move requests and records what the Python server makes of each one, so
the Rust request -> game-state conversion can be compared exactly. Realistic requests come from
rollouts of slinky games in all four modes (``rules_golden.rollouts``/``perturb``: heuristic and
forager play with random moves, low health, extra food, stacked hazards, royale turns at shrink
boundaries); the snakes are listed in shuffled order, dead snakes are absent, ``you`` is a random
living snake, the ruleset/map spelling and the settings vary the way the ladder and the engine
CLI send them. Hand-written edge cases (stacked tails at turn 0, duplicate food, odd settings,
unsupported rulesets, royale zones of every shape, ...) are added to the random ones.

Output: ``<out>/api/cases.jsonl`` (default out /dev/shm/naz-golden; RAM, never committed), one
JSON object per line:

    {"name": str,
     "starting": int | null,         # snakes at /start (the caller's cache); null if unknown
     "request": {...},                # the API request body, exactly as the Rust crate parses it
     "expect": {
        "ok": bool,                   # Python made a config, a mode and a state for >= 2 and <= 4
                                      # seats on a board of at most 11x11 (what the Rust crate
                                      # supports); false for everything the Python server answers
                                      # with safe_move, or the Rust crate refuses
        "why": str,                   # failure reason when not ok (informational)
        "safe": int | null,           # server.safe_move(request): action id 0..3; null if it raised
        "mode": str,                  # when ok: 'standard' | 'duel' | 'royale' | 'constrictor'
        "config": {"width","height","num_snakes","ruleset","hazard_damage","shrink_every"},
        "state": {                    # when ok: server.request_state(request, config), seat 0 = you
           "body": [[int * (H*W)] * N],   # countdown grids, index y * W + x
           "head": [[x, y] * N], "length": [N], "health": [N], "alive": [N bools],
           "last_move": [N], "elim_cause": [N], "elim_turn": [N],
           "food": [H*W of 0/1], "hazard": [H*W], "turn": int, "done": bool,
           "sides": [W+H-1] | null}      # royale shrink sides (null outside royale)
     }}

The script also records, for each case, only what Python computes without a JAX search: it
never runs the network or the search.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_golden as rg  # noqa: E402  (also puts jax on the CPU through JAX_PLATFORMS)

from slinky import modes, server  # noqa: E402
from slinky.engine_json import countdown_to_body  # noqa: E402
from slinky.types import GameConfig, Ruleset  # noqa: E402

MAX_DUMP = 4


def engine_dict(d, i, game):
    """Engine board dict of numpy state ``i`` of ``d`` (like engine_json.state_to_engine)."""
    h, w, n = game.height, game.width, game.num_snakes
    snakes = []
    for s in range(n):
        alive = bool(d["alive"][i, s])
        head = (int(d["head"][i, s, 0]), int(d["head"][i, s, 1]))
        body = (
            countdown_to_body(d["body"][i, s], head, int(d["length"][i, s]))
            if alive
            else [list(head)] * 3
        )
        snakes.append({"id": f"s{s}", "body": body, "health": int(d["health"][i, s]), "alive": alive})
    ys, xs = np.nonzero(d["food"][i])
    hazards = []
    for y, x in zip(*np.nonzero(d["hazard"][i])):
        hazards += [[int(x), int(y)]] * int(d["hazard"][i, y, x])
    return {
        "width": w, "height": h, "turn": int(d["turn"][i]), "snakes": snakes,
        "food": [[int(x), int(y)] for x, y in zip(xs, ys)], "hazards": hazards,
    }


def api_snake(s):
    return {
        "id": s["id"], "name": s["id"], "health": s["health"],
        "body": [{"x": x, "y": y} for x, y in s["body"]],
        "head": {"x": s["body"][0][0], "y": s["body"][0][1]},
        "length": len(s["body"]), "latency": "12", "shout": "", "squad": "",
        "customizations": {"color": "#888888", "head": "default", "tail": "default"},
    }


def official_settings(config, rng=None):
    """``game.ruleset.settings`` as the API sends it (every setting); strings sometimes."""
    s = {
        "foodSpawnChance": config.food_spawn_chance,
        "minimumFood": config.minimum_food,
        "hazardDamagePerTurn": config.hazard_damage_per_turn,
        "royale": {"shrinkEveryNTurns": config.shrink_every_n_turns},
        "squad": {"allowBodyCollisions": False, "sharedElimination": False,
                  "sharedHealth": False, "sharedLength": False},
    }
    if rng is not None and rng.random() < 0.2:  # the engine CLI sends strings
        s["hazardDamagePerTurn"] = str(s["hazardDamagePerTurn"])
        s["royale"]["shrinkEveryNTurns"] = str(s["royale"]["shrinkEveryNTurns"])
    return s


def api_request(d, you, game_id="g", ruleset="standard", game_map="standard", settings=None,
                shuffle=None):
    """API move request for engine dict ``d`` from snake index ``you`` (dead snakes absent)."""
    snakes = [api_snake(s) for s in d["snakes"] if s["alive"]]
    me = next(s for s in snakes if s["id"] == d["snakes"][you]["id"])
    if shuffle is not None:
        shuffle.shuffle(snakes)
    return {
        "game": {"id": game_id, "ruleset": {"name": ruleset, "version": "v1.2.3",
                                            "settings": settings if settings is not None else {}},
                 "map": game_map, "source": "custom", "timeout": 500},
        "turn": d["turn"],
        "board": {"width": d["width"], "height": d["height"],
                  "food": [{"x": x, "y": y} for x, y in d["food"]],
                  "hazards": [{"x": x, "y": y} for x, y in d["hazards"]],
                  "snakes": snakes},
        "you": me,
    }


def spelling(game, rng):
    """(ruleset name, map) a game of ``game``'s config may be announced with."""
    if game.ruleset == Ruleset.ROYALE:
        return [("royale", "royale"), ("royale", "standard"), ("standard", "royale")][int(rng.integers(3))]
    if game.ruleset == Ruleset.CONSTRICTOR:
        return ("constrictor", "standard" if rng.random() < 0.5 else "empty")
    return ("standard", "standard")


def expectation(req, starting):
    """What the Python server makes of ``req``: see the module docstring."""
    out = {"ok": False, "why": "", "safe": None}
    try:
        out["safe"] = int(server.safe_move(req))
    except Exception as e:  # noqa: BLE001
        out["safe_error"] = repr(e)
    try:
        server.check_sizes(req)
        config = server.game_config(req, starting)
        if config is None:
            out["why"] = "no config"
            return out
        mode = server.game_mode(config)
        if mode is None:
            out["why"] = "no mode"
            return out
        state = server.request_state(req, config)
    except Exception as e:  # noqa: BLE001
        out["why"] = f"python raised {e!r}"[:200]
        return out
    if config.num_snakes > 4:
        out["why"] = "more than 4 seats"
        return out
    if config.width > 11 or config.height > 11:
        out["why"] = "board larger than 11x11"
        return out
    royale = config.ruleset == Ruleset.ROYALE
    out.update(
        ok=True, mode=mode,
        config={"width": config.width, "height": config.height, "num_snakes": config.num_snakes,
                "ruleset": config.ruleset.value, "hazard_damage": config.hazard_damage_per_turn,
                "shrink_every": config.shrink_every_n_turns},
        state={
            "body": np.asarray(state.body).reshape(config.num_snakes, -1).tolist(),
            "head": np.asarray(state.head).tolist(),
            "length": np.asarray(state.length).tolist(),
            "health": np.asarray(state.health).tolist(),
            "alive": np.asarray(state.alive).tolist(),
            "last_move": np.asarray(state.last_move).tolist(),
            "elim_cause": np.asarray(state.elim_cause).tolist(),
            "elim_turn": np.asarray(state.elim_turn).tolist(),
            "food": np.asarray(state.food).astype(int).reshape(-1).tolist(),
            "hazard": np.asarray(state.hazard).reshape(-1).tolist(),
            "turn": int(state.turn), "done": bool(state.done),
            "sides": np.asarray(state.map_state.sides).tolist() if royale else None,
        },
    )
    return out


class Writer:
    def __init__(self, path):
        self.f = open(path, "w")
        self.n = self.ok = 0

    def add(self, name, req, starting):
        exp = expectation(req, starting)
        self.f.write(json.dumps({"name": name, "starting": starting, "request": req,
                                 "expect": exp}) + "\n")
        self.n += 1
        self.ok += exp["ok"]


# --- Random games ---------------------------------------------------------------------------


def configs(scale):
    m = modes.mode_config
    fast = dataclasses.replace(m("royale"), shrink_every_n_turns=3, hazard_damage_per_turn=30)
    return [
        ("duel", m("duel"), int(500 * scale), {}),
        ("standard", m("standard"), int(500 * scale), {}),
        ("standard3", m("standard", 3), int(150 * scale), {}),
        ("royale", m("royale"), int(700 * scale), {"royale_turns": True}),
        ("royale_fast", fast, int(200 * scale), {"royale_turns": True}),
        ("royale2", m("royale", 2), int(150 * scale), {"royale_turns": True}),
        ("constrictor", m("constrictor"), int(500 * scale), {"no_low_health": True}),
        ("constrictor2", m("constrictor", 2), int(100 * scale), {"no_low_health": True}),
        ("hazard_std", m("standard"), int(200 * scale), {"hazards": 4}),
        ("small", GameConfig(width=7, height=7, num_snakes=4), int(100 * scale), {}),
        ("rect", GameConfig(width=9, height=11, num_snakes=3), int(100 * scale), {}),
    ]


def random_cases(w, scale, seed):
    for k, (label, game, count, opts) in enumerate(configs(scale)):
        t0 = time.time()
        rng = np.random.default_rng(seed * 100 + k)
        turns = 150 if game.ruleset == Ruleset.CONSTRICTOR else 300
        traj = rg.to_numpy(rg.rollouts(game, 12, turns, seed * 100 + k), game)
        total = len(traj["turn"])
        # Turn-0 states are over-represented (stacked tails) next to the whole range of turns.
        first = np.nonzero(traj["turn"] == 0)[0]
        for c in range(count):
            i = int(rng.choice(first)) if c % 10 == 0 else int(rng.integers(total))
            src = rg.take(traj, np.array([i]))
            rg.perturb(src, game, opts, rng)
            alive = np.nonzero(src["alive"][0])[0]
            if len(alive) == 0:
                continue
            you = int(rng.choice(alive))
            d = engine_dict(src, 0, game)
            name_ruleset, game_map = spelling(game, rng)
            if rng.random() < 0.15:
                settings = {}
            elif game.ruleset != Ruleset.ROYALE and rng.random() < 0.4:
                settings = {**official_settings(game, rng), "hazardDamagePerTurn": 100}  # the ladder
            else:
                settings = official_settings(game, rng)
            req = api_request(d, you, game_id=f"{label}-{c}", ruleset=name_ruleset,
                              game_map=game_map, settings=settings, shuffle=rng)
            r = rng.random()
            starting = game.num_snakes if r < 0.7 else (None if r < 0.9 else len(alive))
            w.add(f"{label}-{c}", req, starting)
        print(f"{label}: {count} random cases, {time.time() - t0:.1f}s", flush=True)


# --- Hand-written edge cases ----------------------------------------------------------------


def base_request(game, seed, turn_index=0, **kw):
    traj = rg.to_numpy(rg.rollouts(game, 2, 120, seed), game)
    full = np.nonzero(traj["alive"].all(axis=1))[0]  # every snake still alive
    idx = full[min(turn_index, len(full) - 1)]
    d = engine_dict(traj, int(idx), game)
    ruleset, game_map = {"royale": ("royale", "royale"), "constrictor": ("constrictor", "standard")}.get(
        game.ruleset.value, ("standard", "standard"))
    return d, api_request(d, 0, ruleset=ruleset, game_map=game_map,
                          settings=official_settings(game), **kw)


def edge_cases(w, seed):
    m = modes.mode_config
    mid = 40
    games = {name: m(name) for name in ("duel", "standard", "royale", "constrictor")}
    for name, game in games.items():
        n = game.num_snakes
        d0, r0 = base_request(game, seed)
        _, rmid = base_request(game, seed + 1, turn_index=mid)
        w.add(f"{name}-turn0-stacked", r0, n)
        w.add(f"{name}-turn0-nostart", r0, None)
        w.add(f"{name}-mid", rmid, n)
        # Fewer snakes alive than at the start (dead absent), with and without the cache.
        if n > 2:
            for keep in range(1, n):
                r = json.loads(json.dumps(rmid))
                r["board"]["snakes"] = r["board"]["snakes"][:keep]
                if r["you"]["id"] not in [s["id"] for s in r["board"]["snakes"]]:
                    r["you"] = r["board"]["snakes"][0]
                w.add(f"{name}-alive{keep}-start{n}", r, n)
                w.add(f"{name}-alive{keep}-nostart", r, None)
                w.add(f"{name}-alive{keep}-start5", r, 5)
        # you in every seat, board listed in every rotation
        for k in range(len(rmid["board"]["snakes"])):
            r = json.loads(json.dumps(rmid))
            sn = r["board"]["snakes"]
            r["board"]["snakes"] = sn[k:] + sn[:k]
            r["you"] = sn[(k + 1) % len(sn)]
            w.add(f"{name}-rot{k}", r, n)
        # duplicate / off-board food, a food on a snake
        r = json.loads(json.dumps(rmid))
        if r["board"]["food"]:
            r["board"]["food"].append(dict(r["board"]["food"][0]))
            w.add(f"{name}-dup-food", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["food"].append({"x": 11, "y": 3})
        w.add(f"{name}-food-off-right", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["food"].append({"x": 3, "y": 11})
        w.add(f"{name}-food-off-top", r, n)
        r = json.loads(json.dumps(rmid))
        h = r["board"]["snakes"][0]["head"]
        r["board"]["food"].append(dict(h))
        w.add(f"{name}-food-under-head", r, n)
        # hazards: off-board dropped, stacked, with a damage setting (string / int / junk)
        for dmg in (40, "40", 100, "x", None, True, -15):
            r = json.loads(json.dumps(rmid))
            r["board"]["hazards"] = [{"x": 2, "y": 2}, {"x": 2, "y": 2}, {"x": -1, "y": 4},
                                     {"x": 11, "y": 11}, {"x": 5, "y": 0}]
            r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = dmg
            w.add(f"{name}-hazards-dmg-{dmg}", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["hazards"] = [{"x": -1, "y": 4}]
        r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = 33
        w.add(f"{name}-only-offboard-hazard", r, n)
        r = json.loads(json.dumps(rmid))
        del r["board"]["hazards"]
        w.add(f"{name}-no-hazards-key", r, n)
        r = json.loads(json.dumps(rmid))
        del r["game"]["ruleset"]["settings"]
        w.add(f"{name}-no-settings", r, n)
        r = json.loads(json.dumps(rmid))
        r["game"]["ruleset"]["settings"] = [1, 2]
        w.add(f"{name}-settings-list", r, n)
        r = json.loads(json.dumps(rmid))
        del r["game"]
        w.add(f"{name}-no-game", r, n)
        # malformed
        for key in ("turn", "board", "you"):
            r = json.loads(json.dumps(rmid))
            del r[key]
            w.add(f"{name}-missing-{key}", r, n)
        r = json.loads(json.dumps(rmid))
        r["you"]["id"] = "nobody"
        w.add(f"{name}-you-not-on-board", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["snakes"] = []
        w.add(f"{name}-no-snakes", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["snakes"].append(dict(r["board"]["snakes"][-1]))
        w.add(f"{name}-dup-snake-id", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["snakes"][0]["body"] = [{"x": 12, "y": 3}, {"x": 11, "y": 3}, {"x": -1, "y": 2}]
        w.add(f"{name}-body-off-board", r, n)
        r = json.loads(json.dumps(rmid))
        r["board"]["snakes"][0]["body"] = []
        w.add(f"{name}-empty-body", r, n)
        for width, height in ((12, 11), (11, 12), (25, 25), (7, 7), (1, 1), (0, 5), (3, 11)):
            r = json.loads(json.dumps(rmid))
            r["board"]["width"], r["board"]["height"] = width, height
            w.add(f"{name}-board-{width}x{height}", r, n)
        r = json.loads(json.dumps(rmid))
        r["turn"] = 10**10
        w.add(f"{name}-turn-huge", r, n)
        r = json.loads(json.dumps(rmid))
        r["turn"] = -7
        w.add(f"{name}-turn-negative", r, n)
        # unsupported rulesets and maps
        for ruleset in ("wrapped", "solo", "squad", "wrapped_constrictor", "royale_x", "", None):
            r = json.loads(json.dumps(rmid))
            r["game"]["ruleset"]["name"] = ruleset
            w.add(f"{name}-ruleset-{ruleset}", r, n)
        for gm in ("arcade_maze", "hz_islands", "snail_mode", "", None, "royale", "empty", "standard"):
            for ruleset in ("standard", "royale", "constrictor"):
                r = json.loads(json.dumps(rmid))
                r["game"]["map"] = gm
                r["game"]["ruleset"]["name"] = ruleset
                w.add(f"{name}-map-{gm}-{ruleset}", r, n)
        r = json.loads(json.dumps(rmid))
        del r["game"]["map"]
        w.add(f"{name}-no-map", r, n)
        del r["game"]["ruleset"]["name"]
        w.add(f"{name}-no-ruleset-name", r, n)
        # snake counts: the cache bigger than the board, too big, tiny
        for st in (0, 1, 2, 3, 4, 5, 8, 9):
            w.add(f"{name}-starting-{st}", rmid, st)

    # Royale settings in every spelling.
    _, rr = base_request(games["royale"], seed + 2, turn_index=60)
    variants = ["7", 7, 7.0, 7.9, "  12 ", "1_0", "1__0", "_1", "0x10", "3.5", "2e1", "+8", "-8", "007",
                True, False, None, [7], {"a": 1}, 0, -3, "0", 2**20, 2**20 + 1, -(2**20), 2**40,
                "99999999999", 1, 3, 25, 50, ""]
    for i, v in enumerate(variants):
        for where in ("nested", "flat", "both"):
            r = json.loads(json.dumps(rr))
            st = r["game"]["ruleset"]["settings"]
            if where in ("nested", "both"):
                st["royale"] = {"shrinkEveryNTurns": v}
            else:
                del st["royale"]
            if where in ("flat", "both"):
                st["shrinkEveryNTurns"] = v if where == "flat" else 9
            w.add(f"royale-shrink-{i}-{where}", r, 4)
    for v in variants:
        r = json.loads(json.dumps(rr))
        r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = v
        w.add(f"royale-damage-{v!r}", r, 4)
    r = json.loads(json.dumps(rr))
    r["game"]["ruleset"]["settings"]["royale"] = "oops"
    w.add("royale-royale-string", r, 4)
    r = json.loads(json.dumps(rr))
    r["game"]["ruleset"]["settings"] = None
    w.add("royale-settings-null", r, 4)
    r = json.loads(json.dumps(rr))
    r["game"]["ruleset"] = None
    w.add("royale-ruleset-null", r, 4)

    # Royale zones of every shape: rectangle from known sides at assorted turns, the whole
    # board, none, scattered cells, columns only, stacked hazards.
    game = games["royale"]
    d0, r0 = base_request(game, seed + 3)
    rng = np.random.default_rng(seed + 7)
    S = 25
    for t in (0, 1, 24, 25, 26, 49, 50, 99, 100, 249, 250, 274, 275, 276, 300, 399, 400, 1000):
        for shape in ("none", "full", "rect", "rect2", "cols", "rows", "scatter", "stacked", "one"):
            r = json.loads(json.dumps(r0))
            r["turn"] = t
            hz = np.zeros((11, 11), int)
            if shape == "full":
                hz[:] = 1
            elif shape == "rect":
                shrinks = min(t // S, 21)
                x0, x1, y0, y1 = 0, 10, 0, 10
                for k in range(shrinks):
                    side = int(rng.integers(4))
                    if x0 <= x1 and y0 <= y1:
                        if side == 0: x0 += 1
                        elif side == 1: x1 -= 1
                        elif side == 2: y0 += 1
                        else: y1 -= 1
                hz[:] = 1
                if x0 <= x1 and y0 <= y1:
                    hz[y0:y1 + 1, x0:x1 + 1] = 0
            elif shape == "rect2":
                hz[:] = 1
                hz[2:9, 3:7] = 0
            elif shape == "cols":
                hz[:, : t // S % 12] = 1
            elif shape == "rows":
                hz[: t // S % 12, :] = 1
            elif shape == "scatter":
                hz = (rng.random((11, 11)) < 0.3).astype(int)
            elif shape == "stacked":
                hz = rng.integers(0, 4, (11, 11))
            elif shape == "one":
                hz[5, 5] = 1
            haz = []
            for y, x in zip(*np.nonzero(hz)):
                haz += [{"x": int(x), "y": int(y)}] * int(hz[y, x])
            r["board"]["hazards"] = haz
            w.add(f"royale-zone-{shape}-t{t}", r, 4)
    # All shrink periods against the full board.
    for S2 in (1, 2, 3, 5, 11, 12, 25):
        for t in (0, 5, 10, 11, 12, 21, 22, 30, 100):
            r = json.loads(json.dumps(r0))
            r["turn"] = t
            r["game"]["ruleset"]["settings"]["royale"]["shrinkEveryNTurns"] = S2
            r["board"]["hazards"] = [{"x": x, "y": y} for x in range(11) for y in range(11)]
            w.add(f"royale-full-S{S2}-t{t}", r, 4)
            r["board"]["hazards"] = [{"x": x, "y": y} for x in range(11) for y in range(11)
                                     if not (3 <= x <= 5 and 2 <= y <= 8)]
            w.add(f"royale-rect-S{S2}-t{t}", r, 4)

    # Standard ruleset on the royale map and royale ruleset on the standard map.
    for ruleset, gm in (("standard", "royale"), ("royale", "standard"), ("royale", "royale")):
        r = json.loads(json.dumps(r0))
        r["game"]["ruleset"]["name"], r["game"]["map"] = ruleset, gm
        w.add(f"royale-spelling-{ruleset}-{gm}", r, 4)
        w.add(f"royale-spelling-{ruleset}-{gm}-2p", r, 2)

    # Constrictor on the empty map, with food, and on the royale map.
    d0, rc = base_request(games["constrictor"], seed + 4, turn_index=30)
    for gm in ("standard", "empty", "royale"):
        r = json.loads(json.dumps(rc))
        r["game"]["map"] = gm
        r["board"]["food"] = [{"x": 1, "y": 1}, {"x": 9, "y": 9}]
        w.add(f"constrictor-map-{gm}-food", r, 4)
    # A game with a 2-snake duel that lost nobody, and a 3-snake standard game.
    d3, r3 = base_request(modes.mode_config("standard", 3), seed + 5, turn_index=20)
    w.add("standard-3p", r3, 3)
    w.add("standard-3p-nostart", r3, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="'random' or 'edge' (default both)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "api"), exist_ok=True)
    w = Writer(os.path.join(args.out, "api", "cases.jsonl"))
    t0 = time.time()
    if args.only in ("", "edge"):
        edge_cases(w, args.seed)
        print(f"edge cases: {w.n} so far ({w.ok} ok), {time.time() - t0:.1f}s", flush=True)
    if args.only in ("", "random"):
        random_cases(w, args.scale, args.seed)
    w.f.close()
    print(f"api golden: {w.n} cases ({w.ok} with a state) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
