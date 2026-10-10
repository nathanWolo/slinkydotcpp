"""Tests for replay recording (``slinky.replay``) and the viewer page (agents: test_agents.py)."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from importlib import resources
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import replay as replay_mod
from slinky.agents import Agent, make_agent, make_env
from slinky.engine_json import countdown_to_body, state_from_engine
from slinky.policies import random_legal_policy
from slinky.replay import (
    FORMAT,
    game_from_states,
    merge_replays,
    record_games,
    render_html,
    to_json,
)
from slinky.types import ACTION_DELTAS, CAUSE_TO_ENGINE, Cause, GameConfig, Ruleset

CONFIG = GameConfig()
CAUSES = set(CAUSE_TO_ENGINE.values()) - {""}


@pytest.fixture(scope="module")
def replay():
    """Heuristic vs random_legal, seats rotated; short enough that some games are cut off."""
    return record_games(CONFIG, ["heuristic", "random_legal"], jax.random.key(0), 6, max_turns=80)


def _on_board(p, config=CONFIG):
    return 0 <= p[0] < config.width and 0 <= p[1] < config.height


def _connected(body, wrapped=False, config=CONFIG):
    """Consecutive segments are neighbours or stacked."""
    for a, b in zip(body, body[1:], strict=False):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if wrapped:
            dx, dy = min(dx, config.width - dx), min(dy, config.height - dy)
        if dx + dy > 1:
            return False
    return True


def test_frames_match_env_states():
    env = make_env(CONFIG, False)
    step = jax.jit(env.step)
    key = jax.random.key(1)
    state, _ = env.reset(key)
    states = [state]
    for _ in range(120):
        if state.done:
            break
        key, k_act, k_step = jax.random.split(key, 3)
        state, _ = step(k_step, state, random_legal_policy(k_act, state, env))
        states.append(state)
    game = game_from_states(states, CONFIG, seats=["a", "b"])
    assert game["seats"] == ["a", "b"]
    assert len(game["frames"]) == len(states) == game["result"]["turns"] + 1
    for frame, s in zip(game["frames"], jax.device_get(states), strict=True):
        assert frame["turn"] == int(s.turn)
        ys, xs = np.nonzero(s.food)
        assert sorted(map(tuple, frame["food"])) == sorted(
            zip(xs.tolist(), ys.tolist(), strict=True)
        )
        assert frame["hazards"] == []
        for i, snake in enumerate(frame["snakes"]):
            assert snake["health"] == int(s.health[i])
            assert snake["alive"] == bool(s.alive[i])
            assert snake["length"] == int(s.length[i])
            if s.alive[i]:
                body = snake["body"]
                assert body[0] == [int(s.head[i, 0]), int(s.head[i, 1])]
                assert len(body) == int(s.length[i])
                assert body == countdown_to_body(s.body[i], s.head[i], int(s.length[i]))
                assert _connected(body)
            else:
                assert snake["cause"] == CAUSE_TO_ENGINE[Cause(int(s.elim_cause[i]))]
    final = jax.device_get(states[-1])
    for e in game["result"]["elims"]:
        assert e["turn"] == int(final.elim_turn[e["seat"]])
        assert e["cause"] == CAUSE_TO_ENGINE[Cause(int(final.elim_cause[e["seat"]]))]


def test_game_from_states_requires_frames_from_turn_zero():
    """Frame t is turn t: the viewer reads the frame index as the turn (counter, deaths)."""
    env = make_env(CONFIG, False)
    step = jax.jit(env.step)
    key = jax.random.key(4)
    state, _ = env.reset(key)
    states = [state]
    for _ in range(6):
        key, k_act, k_step = jax.random.split(key, 3)
        state, _ = step(k_step, state, random_legal_policy(k_act, state, env))
        states.append(state)
    assert not states[-1].done
    assert [f["turn"] for f in game_from_states(states, CONFIG)["frames"]] == list(range(7))
    with pytest.raises(ValueError, match="from turn 0"):
        game_from_states(states[3:], CONFIG)  # picked up mid-game: turns 3, 4, 5, 6
    with pytest.raises(ValueError, match="got turns 0, 1, 3"):
        game_from_states([*states[:2], *states[3:]], CONFIG)  # a skipped turn
    # States after the first finished one are ignored, whatever their turn.
    done = states[-1]._replace(done=jnp.asarray(True))
    game = game_from_states([*states[:-1], done, states[2]], CONFIG)
    assert len(game["frames"]) == 7


def test_replay_structure_and_results(replay):
    assert replay["format"] == FORMAT
    assert replay["board"] == {"width": 11, "height": 11}
    assert replay["ruleset"] == "standard" and replay["map"] == "standard"
    assert replay["agents"] == ["heuristic", "random_legal"]
    assert replay["max_turns"] == 80
    assert [g["id"] for g in replay["games"]] == list(range(6))
    for g in replay["games"]:
        frames, r = g["frames"], g["result"]
        assert len(frames) == r["turns"] + 1
        assert [f["turn"] for f in frames] == list(range(len(frames)))
        assert r["turns"] <= 80
        last = frames[-1]["snakes"]
        alive = [s["alive"] for s in last]
        if r["truncated"]:
            assert r["turns"] == 80 and sum(alive) >= 2 and r["winner"] is None and not r["draw"]
        elif r["draw"]:
            assert sum(alive) == 0 and r["winner"] is None
        else:
            assert alive == [i == r["winner"] for i in range(2)]
        # Eliminations: one per dead seat, matching each frame's flags.
        assert sorted(e["seat"] for e in r["elims"]) == [i for i, a in enumerate(alive) if not a]
        for e in r["elims"]:
            assert e["cause"] in CAUSES
            death = frames[e["turn"]]["snakes"][e["seat"]]
            before = frames[e["turn"] - 1]["snakes"][e["seat"]]
            assert before["alive"] and not death["alive"] and death["cause"] == e["cause"]
            # Drawable where it died: the body after the fatal move (grown if it ate).
            moved = [death["body"][0], *before["body"][:-1]]
            moved += [moved[-1]] * (death["length"] - len(moved))
            assert death["body"] == moved
            assert _on_board(death["body"][0]) == (e["cause"] != "wall-collision")
            for later in frames[e["turn"] + 1 :]:
                gone = later["snakes"][e["seat"]]
                assert gone["body"] == [] and not gone["alive"] and gone["cause"] == e["cause"]
        for f in frames:
            for s in f["snakes"]:
                if s["alive"]:
                    assert len(s["body"]) == s["length"] and _connected(s["body"])
                    assert 0 < s["health"] <= 100
    assert any(g["result"]["winner"] is not None for g in replay["games"])


def _direction(a, b):
    return ACTION_DELTAS.index((b[0] - a[0], b[1] - a[1]))


def test_frames_follow_the_rules(replay):
    """Stepping the env from frame t with the moves the frames show gives frame t + 1's snakes."""
    env = make_env(CONFIG, False)
    step = jax.jit(env.step)
    for g in replay["games"][:3]:
        frames = g["frames"]
        for f0, f1 in zip(frames, frames[1:], strict=False):
            d = {
                "width": 11,
                "height": 11,
                "turn": f0["turn"],
                "snakes": [
                    {
                        "id": f"s{i}",
                        "body": s["body"],
                        "health": s["health"],
                        "eliminated_cause": s.get("cause", ""),
                        "eliminated_on_turn": 0,
                    }
                    for i, s in enumerate(f0["snakes"])
                ],
                "food": f0["food"],
                "hazards": [],
            }
            state = state_from_engine(d, CONFIG)
            actions = [
                _direction(s0["body"][0], s1["body"][0]) if s0["alive"] else 0
                for s0, s1 in zip(f0["snakes"], f1["snakes"], strict=True)
            ]
            nxt, _ = step(jax.random.key(0), state, jnp.asarray(actions, jnp.int32))
            nxt = jax.device_get(nxt)
            for i, s1 in enumerate(f1["snakes"]):
                if not f0["snakes"][i]["alive"]:
                    continue
                assert bool(nxt.alive[i]) == s1["alive"]
                assert int(nxt.health[i]) == s1["health"]
                assert int(nxt.length[i]) == s1["length"]
                assert [int(nxt.head[i, 0]), int(nxt.head[i, 1])] == s1["body"][0]
                if s1["alive"]:
                    assert s1["body"] == countdown_to_body(
                        nxt.body[i], nxt.head[i], int(nxt.length[i])
                    )
                else:
                    assert s1["cause"] == CAUSE_TO_ENGINE[Cause(int(nxt.elim_cause[i]))]
            # Food only disappears when eaten (spawning is random, so not compared).
            heads = {tuple(s["body"][0]) for s in f1["snakes"] if s["body"]}
            assert {tuple(p) for p in f0["food"]} - heads <= {tuple(p) for p in f1["food"]}


def test_seats_rotate_like_play_match(replay):
    assert [g["seats"] for g in replay["games"]] == [
        ["heuristic", "random_legal"],
        ["random_legal", "heuristic"],
    ] * 3
    four = GameConfig(num_snakes=4)
    r = record_games(four, ["random_legal", "random"], jax.random.key(2), 5, max_turns=3)
    for g in r["games"]:
        assert g["seats"] == ["random_legal" if s == g["id"] % 4 else "random" for s in range(4)]
    fixed = record_games(
        four, ["random_legal", "random", "random", "random_legal"], jax.random.key(2), 2, 3
    )
    # Repeated names get roster letters, which can't be mistaken for the 0-based seats.
    assert fixed["agents"] == ["random_legal A", "random B", "random C", "random_legal D"]
    assert all(g["seats"] == fixed["agents"] for g in fixed["games"])
    shifted = record_games(
        four, ["random_legal", "random", "random", "random_legal"], jax.random.key(2), 2, 3,
        rotate=True,
    )  # fmt: skip
    names = shifted["agents"]
    assert shifted["games"][1]["seats"] == [names[3], names[0], names[1], names[2]]
    with pytest.raises(ValueError, match="one agent per seat"):
        record_games(four, ["random"] * 3, jax.random.key(0), 1, 3)


def test_truncation_and_determinism():
    a = record_games(CONFIG, ["random_legal", "random_legal"], jax.random.key(5), 2, 4)
    assert a["agents"] == ["random_legal A", "random_legal B"]
    for g in a["games"]:
        r = g["result"]
        if r["truncated"]:
            assert r["turns"] == 4 and len(g["frames"]) == 5 and r["winner"] is None
    assert any(g["result"]["truncated"] for g in a["games"])
    b = record_games(CONFIG, ["random_legal", "random_legal"], jax.random.key(5), 2, 4)
    assert to_json(a) == to_json(b)


def test_json_round_trip_and_merge(replay):
    assert json.loads(to_json(replay)) == replay
    assert "\n" not in to_json(replay)
    other = record_games(CONFIG, ["random_legal", "random"], jax.random.key(3), 2, max_turns=5)
    merged = merge_replays(replay, other)
    assert merged["agents"] == ["heuristic", "random_legal", "random"]
    assert [g["id"] for g in merged["games"]] == list(range(8))
    assert merged["games"][6]["frames"] == other["games"][0]["frames"]
    with pytest.raises(ValueError, match="same board"):
        merge_replays(replay, {**other, "ruleset": "wrapped"})


def _embedded(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="replay-data">(.*?)</script>', html, re.S)
    assert m is not None
    return json.loads(m.group(1))


def test_render_html_fragment_and_document(replay):
    fragment = render_html(replay, standalone=False)
    assert fragment.lstrip().startswith("<title>")
    lowered = fragment.lower()
    for tag in ("!doctype", "html", "head", "body"):
        assert re.search(rf"<{tag}[\s>]", lowered) is None, tag
    assert _embedded(fragment) == replay
    doc = render_html(replay)
    assert doc.startswith("<!doctype html>")
    assert '<meta name="viewport"' in doc and "viewport-fit=cover" in doc
    assert fragment in doc and doc.rstrip().endswith("</html>")
    # The viewer contract: tokens for both themes, no external scripts.
    assert "@media (prefers-color-scheme: dark)" in fragment
    assert ':root[data-theme="dark"]' in fragment
    assert "<script src" not in fragment


def test_render_html_escapes_script_breakouts():
    evil = "</script><script>alert(1)</script><!--"
    base = make_agent("random_legal", CONFIG)
    agent = Agent(evil, base.policy, False, "")
    r = record_games(CONFIG, [agent, "random"], jax.random.key(0), 1, max_turns=2)
    html = render_html(r, standalone=False)
    data_start = html.index('id="replay-data">')
    data_end = html.index("</script>", data_start)
    assert evil not in html and "<!--" not in html[data_start:data_end]
    assert _embedded(html)["agents"][0] == evil
    assert html.count("</script>") == 2  # the data block and the viewer script


# --- The viewer's drawing logic (its pure helpers, run under node) ------------------------


def _viewer_js(fn: str, data: Any) -> Any:
    """``fn(input)`` evaluated after viewer.html's ``@pure-begin``/``@pure-end`` block."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    page = resources.files("slinky").joinpath("viewer.html").read_text(encoding="utf-8")
    m = re.search(r"// @pure-begin\n(.*?)// @pure-end", page, re.S)
    assert m is not None
    code = (
        m.group(1)
        + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        + f"\nprocess.stdout.write(JSON.stringify(({fn})(input)));"
    )
    out = subprocess.run(
        [node, "-e", code], input=json.dumps(data), capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


_EATERS = "(games) => games.map((fs) => fs.slice(1).map((f, t) => eaters(fs[t], f)))"


def _head_on_food(frames):
    """Per turn t >= 1: the live snakes whose head is on a food cell of frame t - 1."""
    return [
        [i for i, s in enumerate(f["snakes"]) if s["alive"] and s["body"][0] in prev["food"]]
        for prev, f in zip(frames, frames[1:], strict=False)
    ]


def test_viewer_eating_is_head_on_food_not_growth(replay):
    """'X ate on this turn' must not fire for constrictor snakes, which grow every turn."""
    shown = _viewer_js(_EATERS, [g["frames"] for g in replay["games"]])
    meals = 0
    for g, ate in zip(replay["games"], shown, strict=True):
        frames = g["frames"]
        assert ate == _head_on_food(frames)
        # Standard rules: a live snake grows exactly when it eats.
        grew = [
            [
                i
                for i, s in enumerate(f["snakes"])
                if s["alive"] and s["length"] > prev["snakes"][i]["length"]
            ]
            for prev, f in zip(frames, frames[1:], strict=False)
        ]
        assert ate == grew
        meals += sum(map(len, ate))
    assert meals > 0

    config = GameConfig(ruleset=Ruleset.CONSTRICTOR)
    constrictor = record_games(config, ["random_legal", "random_legal"], jax.random.key(0), 2, 30)
    frames = [g["frames"] for g in constrictor["games"]]
    shown = _viewer_js(_EATERS, frames)
    grown = 0
    for fs, ate in zip(frames, shown, strict=True):
        assert ate == _head_on_food(fs)
        grown += sum(
            s["alive"] and s["length"] > prev["snakes"][i]["length"]
            for prev, f in zip(fs, fs[1:], strict=False)
            for i, s in enumerate(f["snakes"])
        )
    assert grown > sum(map(len, sum(shown, [])))  # growth without eating happened


def _arc(x, y, r, a0, a1, anticlockwise, n=48):
    """Points along a canvas ``arc()`` (angles grow clockwise on screen, y down)."""
    sweep = -((a0 - a1) % (2 * math.pi)) if anticlockwise else (a1 - a0) % (2 * math.pi)
    return [
        (x + r * math.cos(a0 + sweep * k / n), y + r * math.sin(a0 + sweep * k / n))
        for k in range(n + 1)
    ]


def _inside(pt, poly):
    x, y = pt
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def test_viewer_tail_tip_is_rounded_outward():
    """The taper ends in a round tip covering the tail cell, not a notch cut into it."""
    w0, w1, length = 11.2, 5.8, 40.0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    paths = _viewer_js(
        f"() => {json.dumps(dirs)}.map(([dx, dy]) => taperPath(100, 100, dx, dy, {length}, "
        f"{w0}, {w1}))",
        None,
    )
    for (dx, dy), ops in zip(dirs, paths, strict=True):
        poly = []
        for op, *args in ops:
            poly += _arc(*args) if op == "arc" else [tuple(args)]
        bx, by = 100 + dx * length, 100 + dy * length
        for t in (-0.6, 0.0, 0.6):  # the tail cell's centre and either side of it
            assert _inside((bx + dx * w1 * t + dy * 0.01, by + dy * w1 * t + dx * 0.01), poly)
        assert not _inside((bx + dx * w1 * 1.2, by + dy * w1 * 1.2), poly)
        assert _inside((100 + dx * length / 2 + dy * 0.01, 100 + dy * length / 2 + dx * 0.01), poly)


def _geometry(css_width, size=11):
    """The board geometry ``layoutBoard()`` computes for a canvas this wide."""
    gl = round(min(24, max(16, css_width * 0.04)))
    pad = math.ceil(0.37 * (css_width - gl) / (size + 0.37))
    return {"gl": gl, "pad": pad, "cell": (css_width - gl - pad) / size, "W": size, "H": size}


_LABELS = """({deaths, geoms}) => {
  const board = {width: 11, height: 11};
  const ds = deaths.map(
    (d) => ({...d, anchor: d.body[0], pos: deathPoint(d.body, d.cause, board)}),
  );
  return {
    pos: ds.map((d) => d.pos),
    labels: geoms.map((g) => {
      const fs = Math.max(10, Math.min(12, g.cell * 0.32));
      return layoutDeathLabels(ds, 9, g, fs, (t) => t.length * fs * 0.6);
    }),
  };
}"""


def _overlap(a, b):
    x = a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
    return x and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]


def _box(x, y, half):
    return {"x": x - half, "y": y - half, "w": 2 * half, "h": 2 * half}


def test_viewer_death_labels_leave_crosses_and_heads_visible():
    """A head-on draw (two deaths on one cell) gets one label, and no label covers a cross."""
    deaths = [
        # Head-on on (1, 5): seat 0 came from the left, seat 3 from above.
        {"seat": 0, "turn": 9, "cause": "head-collision", "body": [[1, 5], [0, 5], [0, 4]]},
        {"seat": 3, "turn": 9, "cause": "head-collision", "body": [[1, 5], [1, 6], [1, 7]]},
        # Two deaths in neighbouring cells, and one on the top row.
        {"seat": 1, "turn": 9, "cause": "snake-collision", "body": [[6, 5], [6, 4]]},
        {"seat": 2, "turn": 9, "cause": "out-of-health", "body": [[7, 5], [7, 4]]},
        {"seat": 4, "turn": 9, "cause": "head-collision", "body": [[4, 10], [4, 9]]},
        {"seat": 5, "turn": 9, "cause": "head-collision", "body": [[4, 10], [3, 10]]},
    ]
    geoms = [_geometry(368), _geometry(560)]  # phone and desktop canvas widths
    out = _viewer_js(_LABELS, {"deaths": deaths, "geoms": geoms})
    # Head-collision crosses sit on the edge each loser came in through, off the shared cell.
    assert out["pos"] == [[0.5, 5], [1, 5.5], [6, 5], [7, 5], [4, 9.5], [3.5, 10]]
    for g, labels in zip(geoms, out["labels"], strict=True):
        cell = g["cell"]
        centres = {  # canvas pixels of each cross and each head
            (x, y): (g["gl"] + (x + 0.5) * cell, g["pad"] + (10 - y + 0.5) * cell)
            for x, y in [*map(tuple, out["pos"]), *(tuple(d["body"][0]) for d in deaths)]
        }
        texts = [lb["text"] for lb in labels]
        assert texts == [
            "head-collision (seats 0, 3)",
            "snake-collision",
            "out-of-health",
            "head-collision (seats 4, 5)",
        ]
        # A cross's arms plus its halo; a head's radius.
        crosses = [_box(*centres[tuple(p)], 0.24 * cell * 1.475) for p in out["pos"]]
        heads = [_box(*centres[tuple(d["body"][0])], 0.56 * 0.62 * cell) for d in deaths]
        for i, lb in enumerate(labels):
            assert g["gl"] <= lb["x"] and lb["x"] + lb["w"] <= g["gl"] + 11 * cell
            assert g["pad"] <= lb["y"] and lb["y"] + lb["h"] <= g["pad"] + 11 * cell
            assert not any(_overlap(lb, box) for box in crosses + heads), (lb, g)
            assert not any(_overlap(lb, other) for other in labels[:i])


def test_cli_writes_json_and_html(tmp_path, capsys):
    out, html = tmp_path / "r" / "replay.json", tmp_path / "r" / "replay.html"
    replay_mod.main(
        ["--a", "random_legal", "--b", "random", "--games", "3", "--max-turns", "20",
         "--seed", "1", "--out", str(out), "--html", str(html)]
    )  # fmt: skip
    printed = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("game ") for line in printed) == 3
    assert any(line.startswith("random_legal ") and " - " in line for line in printed)
    data = json.loads(out.read_text())
    assert data["format"] == FORMAT and len(data["games"]) == 3
    assert html.read_text().startswith("<!doctype html>")
    frag = tmp_path / "frag.html"
    replay_mod.main(
        ["--agents", "random_legal,random,random", "--games", "1", "--max-turns", "5",
         "--html", str(frag), "--fragment"]
    )  # fmt: skip
    text = frag.read_text()
    assert text.startswith("<title>") and len(_embedded(text)["games"][0]["seats"]) == 3
    with pytest.raises(SystemExit):
        replay_mod.main(["--a", "random"])
