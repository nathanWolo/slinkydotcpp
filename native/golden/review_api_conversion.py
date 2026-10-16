"""Adversarial golden vectors for the Rust api crate (native/crates/api), written by the review
of the request -> game conversion. Complements ``api_golden.py`` (realistic and edge requests)
with malformed and hostile request bodies, and records what the *whole* Python server answers,
not only what ``request_state`` builds.

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_api_conversion.py [--out DIR]

Output: ``<out>/api/review_cases.jsonl`` (default out /dev/shm/naz-golden; RAM, never
committed), one JSON object per line:

    {"name": str,                      # unique case name
     "group": str,                     # the review question the case belongs to (the Rust test
                                       # file has one test per group)
     "text": str,                      # the raw HTTP body, parsed by Python with json.loads
                                       # exactly as server.py's handler does (NaN, Infinity and
                                       # duplicate keys included)
     "starting": int | null,           # snakes at /start (the Mover's cache); null if unknown
     "outcome": str,                   # what the Python server does with the body:
         # "agent"    -> the multi-mode agent searches (config, mode and 11x11 accepted)
         # "fallback" -> the fallback heuristic plays (no mode, e.g. one snake, or a board other
         #               than 11x11: agents.check_game refuses, Mover._build falls back)
         # "safe"     -> server.safe_move answers (no config, more than 4 seats, or an exception
         #               in check_sizes / game_config / request_state)
         # "http400"  -> the HTTP layer answers 400 (not a JSON object, or a non-dict "game")
     "why": str,                       # the reason (informational)
     "answer": int | null,             # the action the server sends for "safe": safe_move, or 0
                                       # ("up") when safe_move itself raised; null otherwise
     "expect": {...}                   # api_golden.expectation(request, starting): "ok" is true
                                       # when Python built a config, mode and state with 2..4
                                       # seats on a board of at most 11x11; then "config",
                                       # "mode" and "state" as documented in api_golden.py
    }

The script never runs the network or the search (``request_state`` only).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_golden  # noqa: E402  (puts jax on the CPU through rules_golden)

from slinky import server  # noqa: E402

RUN_MODES = ("standard", "duel", "royale", "constrictor")  # az-modes-M-s0 game_modes


def outcome(req, starting):
    """(outcome, why, answer) of the Python server for parsed body ``req`` (module docstring)."""

    def safe(why):
        try:
            return "safe", why, int(server.safe_move(req))
        except Exception as e:  # noqa: BLE001  the handler answers "up"
            return "safe", f"{why}; safe_move raised {e!r}"[:200], 0

    # server.make_handler.do_POST
    if not isinstance(req, dict) or not isinstance(req.get("game", {}), dict):
        return "http400", "not a Battlesnake request", None
    # Mover.move
    try:
        server.check_sizes(req)
        config = server.game_config(req, starting, False)
    except Exception as e:  # noqa: BLE001
        return safe(f"raised {e!r}"[:200])
    if config is None:
        return safe("no config")
    mode = server.game_mode(config)
    # Mover._build: a multi-mode agent plays only a game with a mode; agents.check_game refuses
    # any board but 11x11 (ValueError -> the fallback heuristic). More than 4 seats fail the
    # search's compile (mcts._check_game), outside the ValueError suppression -> safe_move.
    if config.num_snakes > 4 and mode is not None:
        return safe("more than 4 seats: the search refuses to compile")
    if mode is None:
        kind, why = "fallback", "no mode"
    elif (config.width, config.height) != (11, 11):
        kind, why = "fallback", "not 11x11"
    elif mode not in RUN_MODES:
        kind, why = "fallback", "mode not trained"
    else:
        kind, why = "agent", mode
    try:
        server.request_state(req, config)
    except Exception as e:  # noqa: BLE001
        return safe(f"request_state raised {e!r}"[:200])
    return kind, why, None


def parse(text, starting):
    """(parsed body or None, outcome triple) as server.py's POST handler sees ``text``."""
    if len(text.encode()) > server.MAX_BODY:  # the handler's Content-Length check
        return None, ("http400", "body over server.MAX_BODY", None)
    try:
        req = json.loads(text.encode())  # bytes, as the handler reads them
    except ValueError:
        return None, ("http400", "not JSON", None)
    return req, outcome(req, starting)


class Writer:
    def __init__(self, path):
        self.f = open(path, "w")
        self.n = 0
        self.names = set()

    def add(self, group, name, body, starting=4):
        """``body``: a dict (dumped with json.dumps) or the raw text."""
        name = f"{group}/{name}"
        assert name not in self.names, name
        self.names.add(name)
        text = body if isinstance(body, str) else json.dumps(body)
        req, (kind, why, answer) = parse(text, starting)
        if isinstance(req, dict):
            try:
                exp = api_golden.expectation(req, starting)
            except Exception as e:  # noqa: BLE001
                exp = {"ok": False, "why": f"expectation raised {e!r}"[:200], "safe": None}
        else:
            exp = {"ok": False, "why": "not an object", "safe": None}
        if kind == "http400":
            exp["ok"] = False
        self.f.write(json.dumps({"name": name, "group": group, "text": text, "starting": starting,
                                 "outcome": kind, "why": why, "answer": answer,
                                 "expect": exp}) + "\n")
        self.n += 1


# --- Building blocks ---------------------------------------------------------------------------


def seg(x, y):
    return {"x": x, "y": y}


def snake(sid, body, health=90):
    return {"id": sid, "name": sid, "health": health, "body": [seg(x, y) for x, y in body],
            "head": seg(*body[0]) if body else seg(0, 0), "length": len(body), "latency": "10",
            "shout": "", "customizations": {"color": "#123456", "head": "default", "tail": "default"}}


# A mid-game four-player position: a just-ate snake (stacked tail), a long one, two short ones.
BODIES = {
    "a": [(5, 5), (5, 4), (5, 3), (5, 3)],
    "b": [(1, 9), (2, 9), (3, 9), (4, 9), (4, 8), (4, 7)],
    "c": [(9, 1), (9, 2), (9, 3)],
    "d": [(8, 8), (8, 7), (7, 7)],
}


def request(ids=("a", "b", "c", "d"), you="a", ruleset="standard", game_map="standard",
            turn=37, food=((0, 0), (10, 10), (6, 2)), hazards=(), settings=None, w=11, h=11):
    snakes = [snake(i, BODIES[i]) for i in ids]
    me = next((s for s in snakes if s["id"] == you), snake(you, BODIES.get(you, [(0, 0)])))
    if settings is None:
        settings = {"foodSpawnChance": 15, "minimumFood": 1, "hazardDamagePerTurn": 14,
                    "royale": {"shrinkEveryNTurns": 25}}
    return {
        "game": {"id": "rev", "ruleset": {"name": ruleset, "version": "v1", "settings": settings},
                 "map": game_map, "timeout": 500, "source": "league"},
        "turn": turn,
        "board": {"width": w, "height": h, "food": [seg(*f) for f in food],
                  "hazards": [seg(*z) for z in hazards], "snakes": snakes},
        "you": me,
    }


def copy(r):
    return json.loads(json.dumps(r))


def hazards_except(x0, x1, y0, y1, w=11, h=11):
    return [(x, y) for y in range(h) for x in range(w) if not (x0 <= x <= x1 and y0 <= y <= y1)]


# --- Cases -----------------------------------------------------------------------------------


def game_object(w):
    g = "game_object"
    for label, value in (("list", []), ("zero", 0), ("empty_string", ""), ("false", False),
                         ("null", None), ("string", "x"), ("number", 3), ("list1", [1]),
                         ("empty", {})):
        r = request()
        r["game"] = value
        w.add(g, label, r)
    for label, value in (("list", []), ("zero", 0), ("false", False), ("string", "standard"),
                         ("number", 5), ("null", None), ("empty", {})):
        r = request()
        r["game"]["ruleset"] = value
        w.add(g, f"ruleset-{label}", r)
    for label, value in (("null", None), ("number", 5), ("list", ["standard"]), ("caps", "Standard"),
                         ("space", "standard "), ("bool", True)):
        r = request()
        r["game"]["ruleset"]["name"] = value
        w.add(g, f"name-{label}", r)
        r = request()
        r["game"]["map"] = value
        w.add(g, f"map-{label}", r)
    r = request()
    r["game"]["map"] = "empty"
    r["game"]["ruleset"]["name"] = "constrictor"
    w.add(g, "constrictor-empty", r)


def you_object(w):
    g = "you_object"
    for drop in ("health", "body", "head", "length", "name"):
        r = request()
        del r["you"][drop]
        w.add(g, f"you-without-{drop}", r)
    r = request()
    r["you"] = {"id": "a"}
    w.add(g, "you-only-id", r)
    for label, key, value in (("length-string", "length", "4"), ("length-float", "length", 4.5),
                              ("head-null", "head", None), ("head-partial", "head", {"x": 1}),
                              ("name-number", "name", 5), ("health-string", "health", "90"),
                              ("body-huge-coord", "body", [seg(2**40, 0)]),
                              ("health-huge", "health", 2**40)):
        r = request()
        r["you"][key] = value
        w.add(g, label, r)
    r = request()
    r["you"]["body"] = [seg(1, 1)] * 400
    w.add(g, "you-body-400", r)
    r = request()
    for s in r["board"]["snakes"]:
        s["id"] = {"a": 1, "b": 2, "c": 3, "d": 4}[s["id"]]
    r["you"]["id"] = 1
    w.add(g, "numeric-ids", r)


def board_snakes(w):
    g = "board_snakes"
    for label, key, value in (("length-string", "length", "4"), ("head-partial", "head", {"x": 1}),
                              ("head-null", "head", None), ("name-number", "name", 7),
                              ("name-null", "name", None), ("customizations-null", "customizations", None),
                              ("extra-field", "zzz", {"deep": [1, 2, {"x": "y"}]})):
        r = request()
        r["board"]["snakes"][1][key] = value
        w.add(g, f"other-{label}", r)
    for label, value in (("float", 89.0), ("float-frac", 89.7), ("negative", -5), ("zero", 0),
                         ("over", 250), ("i32max", 2**31 - 1), ("i32over", 2**31),
                         ("string", "90"), ("null", None), ("bool", True)):
        r = request()
        r["board"]["snakes"][1]["health"] = value
        w.add(g, f"health-{label}", r)
    r = request()
    del r["board"]["snakes"][2]["health"]
    w.add(g, "health-missing", r)
    # Too many snakes listed, with and without /start.
    for k in (5, 8, 9):
        r = request()
        for i in range(4, k):
            r["board"]["snakes"].append(snake(f"x{i}", [(i % 11, 0)]))
        w.add(g, f"listed-{k}", r, None)
        w.add(g, f"listed-{k}-start4", r, 4)
    # Two snakes carry our id; our id only in the board, not in you (another body).
    r = request()
    r["board"]["snakes"].append(dict(r["board"]["snakes"][0], body=[seg(0, 5)]))
    r["board"]["snakes"] = r["board"]["snakes"][:2] + r["board"]["snakes"][4:]
    w.add(g, "two-yous", r)
    r = request()
    r["you"]["body"] = [seg(0, 1)]
    r["you"]["head"] = seg(0, 1)
    w.add(g, "you-body-differs", r)
    r = request(ids=("b", "c", "d"), you="a")
    w.add(g, "you-absent", r)
    r = request(ids=(), you="a")
    w.add(g, "no-snakes", r, None)
    r = request()
    r["board"]["snakes"] = None
    w.add(g, "snakes-null", r)


def bodies(w):
    g = "bodies"
    cases = {
        "stacked-tail": [(5, 5), (5, 4), (5, 4)],
        "stacked-3-tail": [(5, 5), (5, 4), (5, 4), (5, 4)],
        "all-stacked": [(5, 5)] * 3,
        "head-on-tail": [(5, 5), (5, 4), (4, 4), (4, 5), (5, 5)],
        "self-crossing": [(1, 1), (1, 2), (1, 1), (1, 2)],
        "teleport": [(0, 0), (10, 10), (5, 5)],
        "single": [(3, 3)],
        "empty": [],
        "head-offboard": [(-1, 5), (0, 5), (1, 5)],
        "tail-offboard": [(0, 5), (-1, 5), (-2, 5)],
        "middle-offboard": [(10, 5), (11, 5), (10, 4)],
        "head-top": [(5, 11), (5, 10)],
        "long": [(x, y) for y in range(11) for x in (range(11) if y % 2 == 0 else range(10, -1, -1))]
        + [(0, 10)] * 3,
    }
    for label, body in cases.items():
        for who in (0, 1):
            r = request()
            r["board"]["snakes"][who]["body"] = [seg(x, y) for x, y in body]
            if who == 0:
                r["you"]["body"] = r["board"]["snakes"][0]["body"]
            w.add(g, f"{label}-seat{who}", r)
    for label, coord in (("x-i32max", (2**31 - 1, 3)), ("x-i32over", (2**31, 3)),
                         ("x-2pow40", (2**40, 3)), ("y-neg-2pow40", (3, -(2**40))),
                         ("x-i64over", (2**64, 3))):
        r = request()
        r["board"]["snakes"][1]["body"].append(seg(*coord))  # a non-head segment
        w.add(g, f"tail-{label}", r)
        r = request()
        r["board"]["snakes"][1]["body"][0] = seg(*coord)  # the head
        w.add(g, f"head-{label}", r)
    for label, value in (("float", 1.0), ("float-frac", 1.5), ("string", "1"), ("null", None)):
        r = request()
        r["board"]["snakes"][1]["body"][2]["x"] = value
        w.add(g, f"coord-{label}", r)
    r = request()
    r["board"]["snakes"][1]["body"][2] = [1, 2]
    w.add(g, "coord-as-list", r)
    r = request()
    r["board"]["snakes"][1]["body"][2]["z"] = 7
    w.add(g, "coord-extra-key", r)


def food(w):
    g = "food"
    for label, cells in (("dup", [(0, 0), (0, 0)]), ("neg-x", [(-1, 0)]), ("neg-y", [(3, -1)]),
                         ("neg-wrap-dup", [(-1, 0), (10, 0)]), ("neg-far", [(-12, 0)]),
                         ("off-right", [(11, 0)]), ("off-top", [(0, 11)]), ("huge", [(2**40, 0)]),
                         ("under-head", [(5, 5)]), ("under-body", [(5, 4)]), ("none", []),
                         ("all", [(x, y) for x in range(11) for y in range(11)])):
        r = request(food=cells)
        w.add(g, label, r)
    r = request(food=[(x, y) for x in range(11) for y in range(11)] + [(0, 0)])
    w.add(g, "more-than-cells", r)
    for label, value in (("null", None), ("float", 1.0), ("string", "1")):
        r = request()
        r["board"]["food"][0]["x"] = value
        w.add(g, f"coord-{label}", r)
    r = request()
    r["board"]["food"] = None
    w.add(g, "food-null", r)
    r = request()
    del r["board"]["food"]
    w.add(g, "food-missing", r)


def hazards(w):
    g = "hazards"
    for ruleset, gm in (("standard", "standard"), ("royale", "royale"), ("constrictor", "standard")):
        for label, cells in (("huge-x", [(2**40, 1)]), ("i32over", [(2**31, 1)]),
                             ("neg", [(-1, 1), (1, -1)]), ("stack-4", [(2, 2)] * 4),
                             ("stack-484", [(2, 2)] * 484), ("over-4-per-cell", [(2, 2)] * 485)):
            r = request(ruleset=ruleset, game_map=gm, hazards=cells, turn=60)
            w.add(g, f"{ruleset}-{label}", r)
        r = request(ruleset=ruleset, game_map=gm)
        r["board"]["hazards"] = None
        w.add(g, f"{ruleset}-null", r)
        r = request(ruleset=ruleset, game_map=gm, hazards=[(1, 1)])
        r["board"]["hazards"][0]["x"] = 1.0
        w.add(g, f"{ruleset}-float", r)


def turns(w):
    g = "turns"
    for ruleset, gm in (("standard", "standard"), ("royale", "royale")):
        for label, value in (("float", 30.0), ("float-frac", 30.6), ("string", "30"), ("bool", True),
                             ("null", None), ("neg", -1), ("i32max", 2**31 - 1),
                             ("i32over", 2**31), ("huge", 2**70)):
            r = request(ruleset=ruleset, game_map=gm, hazards=hazards_except(1, 10, 0, 10))
            r["turn"] = value
            w.add(g, f"{ruleset}-{label}", r)


def raw_text(w):
    """Bodies only text can express: NaN/Infinity, duplicate keys, deep nesting, junk."""
    g = "raw_text"
    base = json.dumps(request())
    w.add(g, "dup-turn", base[:-1] + ', "turn": 40}')
    w.add(g, "dup-game", base[:-1] + ', "game": {"ruleset": {"name": "wrapped"}}}')
    w.add(g, "dup-coord-key", base.replace('{"x": 6, "y": 2}', '{"x": 6, "y": 2, "x": 7}'))
    w.add(g, "dup-width", base.replace('"width": 11', '"width": 9, "width": 11'))
    for label, lit in (("nan", "NaN"), ("inf", "Infinity"), ("neginf", "-Infinity")):
        for rs in ("standard", "royale"):
            r = request(ruleset=rs, game_map=rs)
            text = json.dumps(r).replace('"hazardDamagePerTurn": 14', f'"hazardDamagePerTurn": {lit}')
            w.add(g, f"damage-{label}-{rs}", text)
        r = request(ruleset="royale", game_map="royale")
        text = json.dumps(r).replace('"shrinkEveryNTurns": 25', f'"shrinkEveryNTurns": {lit}')
        w.add(g, f"shrink-{label}", text)
        text = json.dumps(request()).replace('"timeout": 500', f'"timeout": {lit}')
        w.add(g, f"timeout-{label}", text)
    deep = "[" * 200 + "]" * 200
    w.add(g, "deep-unused-field", base[:-1] + f', "extra": {deep}}}')
    deep = "[" * 500 + "]" * 500
    w.add(g, "deeper-unused-field", base[:-1] + f', "extra": {deep}}}')
    w.add(g, "deep-in-game", base.replace('"timeout": 500', f'"timeout": {deep}'))
    w.add(g, "deep-in-snake", base.replace('"latency": "10"', f'"latency": {deep}', 1))
    w.add(g, "over-64kb", base[:-1] + ', "extra": "' + "x" * 70000 + '"}')
    w.add(g, "trailing-junk", base + " x")
    w.add(g, "trailing-space", base + " \n")
    w.add(g, "bom", "﻿" + base)
    w.add(g, "array", "[" + base + "]")
    w.add(g, "empty", "")
    w.add(g, "null", "null")
    w.add(g, "big-int-unused", base[:-1] + ', "extra": 1' + "0" * 400 + "}")
    w.add(g, "big-int-setting",
          json.dumps(request()).replace('"minimumFood": 1', '"minimumFood": 1' + "0" * 400))


def seats(w):
    """A four-player game down to fewer snakes, with and without the /start count."""
    g = "seats"
    for rs, gm in (("standard", "standard"), ("royale", "royale"), ("constrictor", "standard"),
                   ("royale", "standard"), ("standard", "royale"), ("constrictor", "empty")):
        for ids in (("a", "b", "c", "d"), ("a", "b", "c"), ("a", "b"), ("a",), ("c", "a")):
            for starting in (None, 0, 1, 2, 3, 4, 5, 8, 9):
                r = request(ids=ids, ruleset=rs, game_map=gm,
                            hazards=hazards_except(1, 10, 0, 10) if "royale" in (rs, gm) else ())
                w.add(g, f"{rs}-{gm}-{''.join(ids)}-start{starting}", r, starting)


def royale_zones(w):
    """Royale hazards that are not the outside of a rectangle."""
    g = "royale_zones"
    shapes = {
        "L": [(x, y) for x in range(11) for y in range(11) if x < 3 or y < 3]
        + [(x, y) for x in range(8, 11) for y in range(8, 11)],
        "center-only": [(x, y) for x in range(4, 7) for y in range(4, 7)],
        "checker": [(x, y) for x in range(11) for y in range(11) if (x + y) % 2],
        "ring-gap": [(x, y) for (x, y) in hazards_except(2, 8, 2, 8) if (x, y) != (0, 5)],
        "one-safe": [(x, y) for x in range(11) for y in range(11) if (x, y) != (7, 3)],
        "two-safe-corners": [(x, y) for x in range(11) for y in range(11)
                             if (x, y) not in ((0, 0), (10, 10))],
        "left-cols": [(x, y) for x in range(3) for y in range(11)],
        "top-rows-stacked": [(x, y) for x in range(11) for y in range(9, 11)] * 2,
        "full": [(x, y) for x in range(11) for y in range(11)],
        "full-plus-offboard": [(x, y) for x in range(-1, 12) for y in range(-1, 12)],
        "offboard-only": [(-1, y) for y in range(11)] + [(11, y) for y in range(11)],
        "none": [],
        "more-shrinks-than-turns": hazards_except(4, 6, 4, 6),
    }
    for label, cells in shapes.items():
        for turn in (-30, 0, 24, 25, 26, 74, 75, 150, 274, 275, 525, 10**6):
            r = request(ruleset="royale", game_map="royale", hazards=cells, turn=turn)
            w.add(g, f"{label}-t{turn}", r)
    # Smaller boards (not 11x11): the conversion still runs, the agent is not told to play.
    for wd, ht in ((7, 7), (5, 9), (11, 3), (1, 11)):
        cells = hazards_except(1, wd - 1, 0, ht - 2, wd, ht)
        r = request(ids=("a", "c"), ruleset="royale", game_map="royale", hazards=cells, turn=60,
                    food=(), w=wd, h=ht)
        for s in r["board"]["snakes"] + [r["you"]]:
            s["body"] = [seg(min(p["x"], wd - 1), min(p["y"], ht - 1)) for p in s["body"]]
        w.add(g, f"board-{wd}x{ht}", r)


def settings(w):
    g = "settings"
    values = ["\x1c12", "\x0c12", " 12", "　" + "12", "١٢", "１２",
              " +1_2 ", "12\n", "\t-3\t", "1_2_3", "0_0", "-0", "+-1", "--1", "1 2", "", " ",
              -2**20, -(2**20) - 1, 2**20, 2**20 + 1, 1048576.9, 1048577.0, -1048576.9, 1e20, -1e20,
              1e-300, -0.0, 0.5, -0.5, 1.9, "1e3", "1.0", [], {}, [12], {"v": 12}, True, False, None,
              2**63, 2**63 - 1, -(2**63), 2**64, 10**30, "9" * 30, "0" * 50 + "7"]
    for i, v in enumerate(values):
        r = request(ruleset="royale", game_map="royale", hazards=hazards_except(2, 10, 0, 9), turn=90)
        r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = v
        w.add(g, f"royale-damage-{i}", r)
        r = request(ruleset="royale", game_map="royale", hazards=hazards_except(2, 10, 0, 9), turn=90)
        r["game"]["ruleset"]["settings"]["royale"]["shrinkEveryNTurns"] = v
        w.add(g, f"royale-shrink-nested-{i}", r)
        r = request(ruleset="royale", game_map="royale", hazards=hazards_except(2, 10, 0, 9), turn=90)
        del r["game"]["ruleset"]["settings"]["royale"]
        r["game"]["ruleset"]["settings"]["shrinkEveryNTurns"] = v
        w.add(g, f"royale-shrink-flat-{i}", r)
        r = request(hazards=[(3, 3), (3, 3)])
        r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = v
        w.add(g, f"standard-hazards-damage-{i}", r)
    for label, st in (("null", None), ("list", [1]), ("string", "x"), ("zero", 0), ("true", True),
                      ("royale-null", {"royale": None, "shrinkEveryNTurns": 5}),
                      ("royale-list", {"royale": [1], "shrinkEveryNTurns": 5}),
                      ("royale-string", {"royale": "x", "shrinkEveryNTurns": 5}),
                      ("royale-bad-flat-good", {"royale": {"shrinkEveryNTurns": "x"},
                                                "shrinkEveryNTurns": 5}),
                      ("royale-zero-flat-good", {"royale": {"shrinkEveryNTurns": 0},
                                                 "shrinkEveryNTurns": 5}),
                      ("flat-zero-nested-good", {"royale": {"shrinkEveryNTurns": 7},
                                                 "shrinkEveryNTurns": 0})):
        r = request(ruleset="royale", game_map="royale", hazards=hazards_except(2, 10, 0, 9), turn=90)
        r["game"]["ruleset"]["settings"] = st
        w.add(g, f"royale-settings-{label}", r)
    # The ladder's 100 damage outside royale, with and without listed hazards.
    for hz in ((), ((4, 4),), ((-1, -1),)):
        r = request(hazards=hz)
        r["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = 100
        w.add(g, f"standard-damage-100-hazards{len(hz)}-{hz[:1]}", r)


def boards(w):
    g = "boards"
    for wd, ht in ((11, 11), (12, 11), (11, 12), (19, 19), (25, 25), (26, 25), (1000, 1000),
                   (0, 11), (-11, -11), (-1, 5), (2**31, 1), (2**62, 4), (2**63, 1), (7, 7), (1, 1),
                   (2, 1)):
        r = request(ids=("a", "c"), w=wd, h=ht, food=())
        for s in r["board"]["snakes"] + [r["you"]]:
            s["body"] = [seg(0, 0)]
            s["head"] = seg(0, 0)
        w.add(g, f"{wd}x{ht}", r, 2)
    for label, value in (("float", 11.0), ("string", "11"), ("null", None), ("bool", True)):
        r = request()
        r["board"]["width"] = value
        w.add(g, f"width-{label}", r)


def safe_moves(w):
    """Requests only ``safe_move`` answers (a wrapped game, or a broken one), with hostile fields."""
    g = "safe_moves"
    big, small = 2**63 - 1, -(2**63)
    for rs in ("wrapped", "solo"):
        for label, edit in (
            ("you-head-i64max", lambda r: r["you"].update(head=seg(big, 5))),
            ("you-head-i64min", lambda r: r["you"].update(head=seg(small, small))),
            ("other-head-i64max", lambda r: r["board"]["snakes"][1].update(head=seg(big, big))),
            ("other-head-i64min", lambda r: r["board"]["snakes"][1].update(head=seg(small, 0))),
            ("you-no-length", lambda r: r["you"].pop("length")),
            ("you-no-head", lambda r: r["you"].pop("head")),
            ("other-no-head", lambda r: r["board"]["snakes"][1].pop("head")),
            ("other-no-length", lambda r: r["board"]["snakes"][1].pop("length")),
            ("other-no-id", lambda r: r["board"]["snakes"][1].pop("id")),
            ("you-length-float", lambda r: r["you"].update(length=3.5)),
            ("other-length-string", lambda r: r["board"]["snakes"][1].update(length="9")),
            ("body-floats", lambda r: [p.update(x=float(p["x"])) for s in r["board"]["snakes"]
                                       for p in s["body"]]),
            ("head-not-body0", lambda r: r["you"].update(head=seg(0, 0))),
            ("width-zero", lambda r: r["board"].update(width=0)),
            ("width-huge", lambda r: r["board"].update(width=2**64)),
            ("negative-head", lambda r: r["you"].update(head=seg(-1, -1))),
            ("plain", lambda r: None),
        ):
            r = request(ruleset=rs)
            r["board"]["snakes"][1]["body"] = [seg(5, 6), seg(6, 6), seg(7, 6)]
            r["board"]["snakes"][1]["head"] = seg(5, 6)
            r["board"]["snakes"][1]["length"] = 6
            edit(r)
            w.add(g, f"{rs}-{label}", r)
    r = request()
    r["game"]["ruleset"]["name"] = 5
    w.add(g, "name-number", r)
    r = request()
    r["game"]["ruleset"]["name"] = ["wrapped"]
    w.add(g, "name-list", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "api"), exist_ok=True)
    w = Writer(os.path.join(args.out, "api", "review_cases.jsonl"))
    for part in (game_object, you_object, board_snakes, bodies, food, hazards, turns, raw_text,
                 seats, royale_zones, settings, boards, safe_moves):
        before = w.n
        part(w)
        print(f"{part.__name__}: {w.n - before} cases", flush=True)
    w.f.close()
    print(f"review api golden: {w.n} cases")


if __name__ == "__main__":
    main()
