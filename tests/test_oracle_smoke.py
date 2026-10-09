"""Smoke tests for the Go rules-engine oracle bridge (tests/oracle.py)."""

from __future__ import annotations

import os

import pytest

try:
    from oracle import generate_games, make_step_request, split_games
except ImportError:
    from tests.oracle import generate_games, make_step_request, split_games

pytestmark = pytest.mark.oracle

CAUSES = {
    "",
    "out-of-health",
    "wall-collision",
    "snake-self-collision",
    "snake-collision",
    "head-collision",
    "hazard",
}


def mk_snake(sid, body, health=100):
    return {
        "id": sid,
        "body": [list(p) for p in body],
        "health": health,
        "eliminated_cause": "",
        "eliminated_on_turn": 0,
        "eliminated_by": "",
    }


def mk_state(snakes, food=(), hazards=(), width=11, height=11, turn=5):
    return {
        "width": width,
        "height": height,
        "turn": turn,
        "snakes": snakes,
        "food": [list(p) for p in food],
        "hazards": [list(p) for p in hazards],
    }


def by_id(state):
    return {s["id"]: s for s in state["snakes"]}


# --------------------------------------------------------------------------- step


def test_oracle_builds(oracle_bin):
    assert oracle_bin.is_file()
    assert os.access(oracle_bin, os.X_OK)


def test_step_wall_collision(oracle):
    st = mk_state(
        [
            mk_snake("a", [(0, 5), (1, 5), (2, 5)]),
            mk_snake("b", [(5, 5), (5, 4), (5, 3)]),
        ]
    )
    resp = oracle.step("standard", st, ["left", "up"])
    assert resp["error"] == ""
    assert resp["game_over"] is False  # the game-over check runs before moving
    out = resp["state"]
    assert out["turn"] == 5  # Execute does not advance the turn
    a, b = by_id(out)["a"], by_id(out)["b"]
    assert a["eliminated_cause"] == "wall-collision"
    assert a["eliminated_on_turn"] == 6
    assert a["body"] == [[-1, 5], [0, 5], [1, 5]]  # engine keeps the off-board head
    assert b["eliminated_cause"] == ""
    assert b["body"] == [[5, 6], [5, 5], [5, 4]]
    assert b["health"] == 99


def test_step_equal_head_to_head_both_die(oracle):
    st = mk_state(
        [
            mk_snake("a", [(4, 5), (3, 5), (2, 5)]),
            mk_snake("b", [(6, 5), (7, 5), (8, 5)]),
        ]
    )
    out = oracle.step("standard", st, ["right", "left"])["state"]
    a, b = by_id(out)["a"], by_id(out)["b"]
    assert (a["eliminated_cause"], a["eliminated_by"]) == ("head-collision", "b")
    assert (b["eliminated_cause"], b["eliminated_by"]) == ("head-collision", "a")


def test_step_eat_food_duplicates_tail(oracle):
    st = mk_state(
        [
            mk_snake("a", [(4, 5), (3, 5), (2, 5)], health=50),
            mk_snake("b", [(9, 9), (9, 8), (9, 7)]),
        ],
        food=[(5, 5), (0, 0)],
    )
    out = oracle.step("standard", st, ["right", "left"])["state"]
    a = by_id(out)["a"]
    assert a["health"] == 100
    assert a["body"] == [[5, 5], [4, 5], [3, 5], [3, 5]]
    assert out["food"] == [[0, 0]]


@pytest.mark.parametrize("bad", ["", "bogus", "UP"])
def test_step_invalid_move_continues_straight(oracle, bad):
    st = mk_state(
        [mk_snake("a", [(4, 5), (3, 5), (2, 5)]), mk_snake("b", [(9, 9), (9, 8), (9, 7)])]
    )
    out = oracle.step("standard", st, [bad, "left"])["state"]
    assert by_id(out)["a"]["body"][0] == [5, 5]


def test_step_game_over_when_one_snake_left(oracle):
    dead = mk_snake("b", [(9, 9), (9, 8), (9, 7)])
    dead.update(eliminated_cause="wall-collision", eliminated_on_turn=5)
    st = mk_state([mk_snake("a", [(4, 5), (3, 5), (2, 5)]), dead])
    resp = oracle.step("standard", st, ["up", ""])
    assert resp["game_over"] is True
    assert resp["state"] == st  # pipeline stops at the game-over stage


def test_step_init_constrictor_removes_food(oracle):
    st = mk_state(
        [mk_snake("a", [(1, 1)] * 3), mk_snake("b", [(9, 9)] * 3)], food=[(0, 2), (5, 5)], turn=0
    )
    out = oracle.step("constrictor", st, None)["state"]
    assert out["food"] == []
    assert [len(s["body"]) for s in out["snakes"]] == [3, 3]


def test_step_reports_errors(oracle):
    resp = oracle.step("no-such-ruleset", mk_state([]), [])
    assert resp["state"] is None and "unknown ruleset" in resp["error"]
    # a living snake without a move is an engine error
    st = mk_state(
        [mk_snake("a", [(4, 5), (3, 5), (2, 5)]), mk_snake("b", [(9, 9), (9, 8), (9, 7)])]
    )
    resp = oracle.step("standard", st, ["up"])
    assert resp["state"] is None and resp["error"] == "move not provided for snake"
    # the process survives errors
    assert oracle.step("standard", st, ["up", "up"])["error"] == ""


def test_step_many_matches_step(oracle):
    st = mk_state(
        [mk_snake("a", [(4, 5), (3, 5), (2, 5)]), mk_snake("b", [(6, 5), (7, 5), (8, 5)])]
    )
    moves = [
        [m1, m2] for m1 in ("up", "down", "left", "right") for m2 in ("up", "down", "left", "right")
    ]
    reqs = [make_step_request("wrapped", st, m) for m in moves] * 50
    batch = oracle.step_many(reqs)
    assert len(batch) == len(reqs)
    for m, resp in zip(moves, batch[: len(moves)], strict=True):
        assert resp == oracle.step("wrapped", st, m)


# --------------------------------------------------------------------------- games


GAME_CONFIGS = [
    dict(ruleset="standard", map="standard", snakes=2),
    dict(ruleset="wrapped_constrictor", map="standard", snakes=2),
    dict(ruleset="royale", map="royale", snakes=4, invalid_move_prob=0.05),
    dict(ruleset="solo", map="standard", snakes=1, width=7, height=7),
]


@pytest.fixture(scope="module", params=GAME_CONFIGS, ids=lambda c: c["ruleset"])
def games(request, oracle_bin):
    records = generate_games(binary=oracle_bin, games=3, max_turns=150, seed=11, **request.param)
    return records, split_games(records)


def test_games_well_formed_and_chained(games):
    records, per_game = games
    assert [r["kind"] for r in records].count("initial") == 3
    for g in per_game:
        cfg = g["config"]
        n = cfg["snakes"]
        assert cfg["game_seed"] == cfg["seed"] + g["game"]
        assert cfg["solo"] == (n == 1)
        init = g["initial"]
        assert init["turn"] == 0
        assert [s["id"] for s in init["snakes"]] == [f"s{i}" for i in range(n)]
        assert (init["width"], init["height"]) == (cfg["width"], cfg["height"])

        trans = g["transitions"]
        assert 0 < len(trans) <= cfg["max_turns"]
        prev = init
        for t, tr in enumerate(trans):
            # PreUpdateBoard is a no-op for the standard and royale maps
            assert tr["pre"] == prev
            assert tr["pre"]["turn"] == t
            assert tr["post_rules"]["turn"] == t
            assert tr["post_map"]["turn"] == t + 1
            assert len(tr["moves"]) == n
            for s, m in zip(tr["pre"]["snakes"], tr["moves"], strict=True):
                if s["eliminated_cause"]:
                    assert m == ""
            for st in (tr["post_rules"], tr["post_map"]):
                assert [s["id"] for s in st["snakes"]] == [f"s{i}" for i in range(n)]
                for s in st["snakes"]:
                    assert s["eliminated_cause"] in CAUSES
                    assert all(len(p) == 2 for p in s["body"])
            # snakes never change between post_rules and post_map
            assert tr["post_rules"]["snakes"] == tr["post_map"]["snakes"]
            prev = tr["post_map"]
        # only the final transition may report game over
        assert not any(tr["game_over"] for tr in trans[:-1])
        assert trans[-1]["game_over"] or len(trans) == cfg["max_turns"]
        if trans[-1]["game_over"]:
            last = trans[-1]
            assert last["post_rules"] == last["pre"]  # nothing happens on the game-over step


def test_games_replay_through_step(games, oracle):
    """Each recorded ruleset transition is reproduced by `oracle step`."""
    _, per_game = games
    reqs, expected = [], []
    for g in per_game:
        cfg = g["config"]
        for tr in g["transitions"]:
            reqs.append(
                make_step_request(
                    cfg["ruleset"],
                    tr["pre"],
                    tr["moves"],
                    cfg["settings"],
                    cfg["solo"],
                    cfg["game_seed"],
                )
            )
            expected.append((tr["game_over"], tr["post_rules"]))
    got = oracle.step_many(reqs)
    for (game_over, post), resp in zip(expected, got, strict=True):
        assert resp["error"] == ""
        assert resp["game_over"] == game_over
        assert resp["state"] == post


def test_generate_games_is_cached_and_deterministic(oracle_bin):
    a = generate_games(binary=oracle_bin, games=2, max_turns=40, seed=3)
    b = generate_games(binary=oracle_bin, games=2, max_turns=40, seed=3)
    c = generate_games(binary=oracle_bin, games=2, max_turns=40, seed=3, use_cache=False)
    assert a == b == c
