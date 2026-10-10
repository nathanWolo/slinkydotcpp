"""Tests for replay recording (``slinky.replay``), the viewer page and the agent registry."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import agents as agents_mod
from slinky import replay as replay_mod
from slinky.agents import Agent, make_agent, make_env, parse_mcts_name
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
from slinky.types import ACTION_DELTAS, CAUSE_TO_ENGINE, Cause, GameConfig

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
    assert fixed["agents"] == ["random_legal (1)", "random (2)", "random (3)", "random_legal (4)"]
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
    assert a["agents"] == ["random_legal (1)", "random_legal (2)"]
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


def test_agents_registry():
    for name in ("random_legal", "random", "heuristic"):
        agent = make_agent(name, CONFIG)
        assert agent.name == name and not agent.needs_obs and agent.description
        assert make_agent(name, CONFIG) is agent  # cached: play_match's jit cache hits
    assert (
        make_agent("heuristic", CONFIG).policy
        is not make_agent("heuristic", GameConfig(num_snakes=4)).policy
    )
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("alphasnake", CONFIG)
    with pytest.raises(FileNotFoundError, match="no DQN checkpoint"):
        make_agent("dqn:/nonexistent/run", CONFIG)


def test_dqn_agent_plays_from_observations():
    pytest.importorskip("optax")
    if not agents_mod.DEFAULT_DQN_CHECKPOINT.is_dir():
        pytest.skip("no DQN checkpoint in this checkout")
    agent = make_agent("dqn", CONFIG)
    assert agent.needs_obs
    assert make_agent("dqn:baselines/checkpoints/dqn-duel-seed0", CONFIG).needs_obs
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent("dqn", GameConfig(width=7, height=7))
    r = record_games(CONFIG, ["dqn", "random_legal"], jax.random.key(0), 2, max_turns=6)
    assert [g["seats"][0] for g in r["games"]] == ["dqn", "random_legal"]


@pytest.mark.skipif(importlib.util.find_spec("slinky.mcts") is None, reason="no slinky.mcts")
def test_mcts_agent_names():
    try:
        mcts = importlib.import_module("slinky.mcts")
    except Exception as e:  # pragma: no cover - module under development
        pytest.skip(f"slinky.mcts does not import: {e}")
    n, overrides = parse_mcts_name("mcts-16:exploration=1.0:spawn_food=true:selection=rm")
    assert n == 16 and overrides == {"exploration": 1.0, "spawn_food": True, "selection": "rm"}
    with pytest.raises(ValueError, match="unknown mcts option 'bogus'"):
        parse_mcts_name("mcts-16:bogus=1")
    with pytest.raises(ValueError, match="expected a float"):
        parse_mcts_name("mcts-16:exploration=lots")
    with pytest.raises(ValueError, match="field=value"):
        parse_mcts_name("mcts-16:exploration")
    agent = make_agent("mcts-2:max_depth=2", CONFIG)
    assert not agent.needs_obs and "2 simulations" in agent.description
    assert agent is make_agent("mcts-2:max_depth=2", CONFIG)
    assert {"exploration", "selection"} <= set(mcts.MCTSConfig.__dataclass_fields__)
    r = record_games(CONFIG, [agent, "random_legal"], jax.random.key(0), 1, max_turns=2)
    assert len(r["games"][0]["frames"]) >= 2


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
