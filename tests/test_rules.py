"""Hand-written edge-case scenarios for the rules pipeline.

Each scenario states the expected outcome explicitly, and (when Go is
available) is also checked against the official engine via the oracle.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from slinky.engine_json import state_from_engine, state_to_engine
from slinky.rules import action_mask, rules_step
from slinky.types import ACTION_NAMES, DOWN, LEFT, RIGHT, UP, Cause, GameConfig, Ruleset


@dataclasses.dataclass
class Scenario:
    name: str
    snakes: list[dict]  # {"body": [[x, y], ...], "health": int}
    moves: list[int]
    expect_alive: list[bool]
    expect_cause: list[Cause] | None = None
    expect_length: list[int] | None = None
    expect_health: list[int] | None = None
    food: list[list[int]] = dataclasses.field(default_factory=list)
    hazards: list[list[int]] = dataclasses.field(default_factory=list)
    ruleset: Ruleset = Ruleset.STANDARD
    size: int = 11
    turn: int = 5
    hazard_damage: int = 14


def snake(body, health=90):
    return {"body": body, "health": health}


SCENARIOS = [
    Scenario(
        "wall",
        [snake([[0, 5], [1, 5], [2, 5]]), snake([[8, 8], [8, 7], [8, 6]])],
        [LEFT, UP],
        [False, True],
        [Cause.OUT_OF_BOUNDS, Cause.NONE],
    ),
    Scenario(
        "neck_reversal",
        [snake([[5, 5], [5, 4], [5, 3]]), snake([[8, 8], [8, 7], [8, 6]])],
        [DOWN, UP],
        [False, True],
        [Cause.SELF_COLLISION, Cause.NONE],
    ),
    Scenario(
        "chase_own_tail_is_safe",
        [snake([[2, 2], [2, 1], [1, 1], [1, 2]]), snake([[8, 8], [8, 7], [8, 6]])],
        [LEFT, UP],
        [True, True],
        expect_length=[4, 3],
    ),
    Scenario(
        "own_stacked_tail_is_deadly",
        [snake([[2, 2], [2, 1], [1, 1], [1, 2], [1, 2]]), snake([[8, 8], [8, 7], [8, 6]])],
        [LEFT, UP],
        [False, True],
        [Cause.SELF_COLLISION, Cause.NONE],
    ),
    Scenario(
        "opponent_tail_is_safe",
        [snake([[3, 5], [2, 5], [1, 5]]), snake([[4, 7], [4, 6], [4, 5]])],
        [RIGHT, UP],
        [True, True],
    ),
    Scenario(
        "opponent_stacked_tail_is_deadly",
        [snake([[3, 5], [2, 5], [1, 5]]), snake([[4, 7], [4, 6], [4, 5], [4, 5]])],
        [RIGHT, UP],
        [False, True],
        [Cause.COLLISION, Cause.NONE],
    ),
    Scenario(
        "opponent_tail_safe_even_if_it_eats_now",
        [snake([[3, 5], [2, 5], [1, 5]]), snake([[4, 7], [4, 6], [4, 5]])],
        [RIGHT, UP],
        [True, True],
        expect_length=[3, 4],
        food=[[4, 8]],
    ),
    Scenario(
        "head_to_head_equal",
        [snake([[4, 5], [3, 5], [2, 5]]), snake([[6, 5], [7, 5], [8, 5]])],
        [RIGHT, LEFT],
        [False, False],
        [Cause.HEAD_COLLISION, Cause.HEAD_COLLISION],
    ),
    Scenario(
        "head_to_head_longer_wins",
        [snake([[4, 5], [3, 5], [2, 5], [1, 5]]), snake([[6, 5], [7, 5], [8, 5]])],
        [RIGHT, LEFT],
        [True, False],
        [Cause.NONE, Cause.HEAD_COLLISION],
    ),
    Scenario(
        "head_to_head_on_food_both_grow",
        [snake([[4, 5], [3, 5], [2, 5], [1, 5]]), snake([[6, 5], [7, 5], [8, 5]])],
        [RIGHT, LEFT],
        [True, False],
        [Cause.NONE, Cause.HEAD_COLLISION],
        expect_length=[5, 4],
        food=[[5, 5]],
    ),
    Scenario(
        "three_way_head_on",
        [
            snake([[4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]),
            snake([[6, 5], [7, 5], [8, 5], [9, 5], [10, 5]]),
            snake([[5, 4], [5, 3], [5, 2], [5, 1]]),
        ],
        [RIGHT, LEFT, UP],
        [False, False, False],
        [Cause.HEAD_COLLISION] * 3,
    ),
    Scenario(
        "eat_at_one_health_survives",
        [snake([[4, 5], [3, 5], [2, 5]], 1), snake([[8, 8], [8, 7], [8, 6]], 1)],
        [RIGHT, UP],
        [True, False],
        [Cause.NONE, Cause.OUT_OF_HEALTH],
        expect_health=[100, 0],
        food=[[5, 5]],
    ),
    Scenario(
        "starved_snake_body_does_not_block",
        [snake([[4, 5], [4, 4], [4, 3]], 1), snake([[3, 4], [2, 4], [1, 4]])],
        [UP, RIGHT],
        [False, True],
        [Cause.OUT_OF_HEALTH, Cause.NONE],
    ),
    Scenario(
        "hazard_damage",
        [snake([[4, 5], [3, 5], [2, 5]], 50), snake([[8, 8], [8, 7], [8, 6]])],
        [RIGHT, UP],
        [True, True],
        expect_health=[35, 89],
        hazards=[[5, 5]],
    ),
    Scenario(
        "stacked_hazard_kills",
        [snake([[4, 5], [3, 5], [2, 5]], 25), snake([[8, 8], [8, 7], [8, 6]])],
        [RIGHT, UP],
        [False, True],
        [Cause.HAZARD, Cause.NONE],
        hazards=[[5, 5], [5, 5]],
    ),
    Scenario(
        "food_on_hazard_cancels_damage",
        [snake([[4, 5], [3, 5], [2, 5]], 10), snake([[8, 8], [8, 7], [8, 6]])],
        [RIGHT, UP],
        [True, True],
        expect_health=[100, 89],
        food=[[5, 5]],
        hazards=[[5, 5]],
    ),
    Scenario(
        "invalid_move_continues_straight",
        [snake([[4, 5], [3, 5], [2, 5]]), snake([[8, 8], [8, 7], [8, 6]])],
        [-1, -1],
        [True, True],
    ),
    Scenario(
        "wrapped_edge",
        [snake([[10, 5], [9, 5], [8, 5]]), snake([[0, 7], [0, 6], [0, 5]])],
        [RIGHT, UP],
        [True, True],
        ruleset=Ruleset.WRAPPED,
    ),
    Scenario(
        "wrapped_body_collision_across_edge",
        [snake([[10, 6], [9, 6], [8, 6]]), snake([[0, 7], [0, 6], [0, 5]])],
        [RIGHT, UP],
        [False, True],
        [Cause.COLLISION, Cause.NONE],
        ruleset=Ruleset.WRAPPED,
    ),
    Scenario(
        "constrictor_grows",
        [snake([[4, 5], [3, 5], [2, 5]], 50), snake([[8, 8], [8, 7], [8, 6]], 50)],
        [RIGHT, UP],
        [True, True],
        expect_length=[4, 4],
        expect_health=[100, 100],
        ruleset=Ruleset.CONSTRICTOR,
    ),
    Scenario(
        "constrictor_stacked_tail_no_extra_growth",
        [snake([[4, 5]] * 3, 50), snake([[8, 8], [8, 7], [8, 6]], 50)],
        [RIGHT, UP],
        [True, True],
        expect_length=[3, 4],
        ruleset=Ruleset.CONSTRICTOR,
    ),
]


def run_scenario(sc: Scenario):
    config = GameConfig(
        width=sc.size,
        height=sc.size,
        num_snakes=len(sc.snakes),
        ruleset=sc.ruleset,
        hazard_damage_per_turn=sc.hazard_damage,
    )
    d = {
        "width": sc.size,
        "height": sc.size,
        "turn": sc.turn,
        "snakes": [
            {
                "id": f"s{i}",
                "body": s["body"],
                "health": s["health"],
                "eliminated_cause": "",
                "eliminated_on_turn": 0,
            }
            for i, s in enumerate(sc.snakes)
        ],
        "food": sc.food,
        "hazards": sc.hazards,
    }
    state = state_from_engine(d, config)
    out = jax.jit(rules_step, static_argnums=2)(state, np.array(sc.moves), config)
    return config, d, state, out


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_scenario(sc: Scenario):
    config, _, _, out = run_scenario(sc)
    assert list(np.asarray(out.alive)) == sc.expect_alive
    if sc.expect_cause is not None:
        assert [Cause(int(c)) for c in out.elim_cause] == sc.expect_cause
        died = ~np.asarray(out.alive)
        assert np.all(np.asarray(out.elim_turn)[died] == sc.turn + 1)
    if sc.expect_length is not None:
        assert list(np.asarray(out.length)) == sc.expect_length
    if sc.expect_health is not None:
        assert list(np.asarray(out.health)) == sc.expect_health
    # Dead snakes vacate the board.
    assert not np.asarray(out.body)[~np.asarray(out.alive)].any()


@pytest.mark.oracle
@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_scenario_matches_engine(sc: Scenario, oracle):
    config, d, _, out = run_scenario(sc)
    moves = [ACTION_NAMES[m] if 0 <= m < 4 else "" for m in sc.moves]
    settings = {"damagePerTurn": str(sc.hazard_damage)}
    resp = oracle.step(sc.ruleset.value, d, moves, settings=settings)
    assert not resp.get("error"), resp
    expected = resp["state"]
    ours = state_to_engine(out, config)
    for i, (e, o) in enumerate(zip(expected["snakes"], ours["snakes"], strict=True)):
        assert e["eliminated_cause"] == o["eliminated_cause"], f"snake {i}"
        if not e["eliminated_cause"]:
            assert e["body"] == o["body"], f"snake {i}"
            assert e["health"] == o["health"], f"snake {i}"
    assert sorted(map(tuple, expected["food"])) == sorted(map(tuple, ours["food"]))


def test_turn_zero_invalid_move_goes_up():
    config = GameConfig(num_snakes=2)
    d = {
        "width": 11,
        "height": 11,
        "turn": 0,
        "snakes": [
            {"id": "s0", "body": [[1, 1]] * 3, "health": 100},
            {"id": "s1", "body": [[9, 9]] * 3, "health": 100},
        ],
        "food": [],
        "hazards": [],
    }
    out = rules_step(state_from_engine(d, config), np.array([-1, 7]), config)
    assert np.asarray(out.head).tolist() == [[1, 2], [9, 10]]


def test_game_over_is_a_noop():
    config = GameConfig(num_snakes=2)
    d = {
        "width": 11,
        "height": 11,
        "turn": 9,
        "snakes": [
            {"id": "s0", "body": [[4, 5], [3, 5], [2, 5]], "health": 50},
            {
                "id": "s1",
                "body": [[8, 8], [8, 7], [8, 6]],
                "health": 50,
                "eliminated_cause": "wall-collision",
                "eliminated_on_turn": 3,
            },
        ],
        "food": [],
        "hazards": [],
    }
    state = state_from_engine(d, config)
    out = rules_step(state, np.array([UP, UP]), config)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(out), strict=True):
        np.testing.assert_array_equal(a, b)


def test_action_mask():
    config = GameConfig(num_snakes=2)
    d = {
        "width": 11,
        "height": 11,
        "turn": 5,
        "snakes": [
            # In the bottom-left corner, heading down: left and down are walls,
            # up is the neck.
            {"id": "s0", "body": [[0, 0], [0, 1], [0, 2]], "health": 50},
            # Stacked tail at (5, 3) blocks; its own neck blocks left.
            {"id": "s1", "body": [[5, 4], [4, 4], [4, 3], [5, 3], [5, 3]], "health": 50},
        ],
        "food": [],
        "hazards": [],
    }
    mask = np.asarray(action_mask(state_from_engine(d, config), config))
    assert mask[0].tolist() == [False, False, False, True]
    assert mask[1].tolist() == [True, False, False, True]


def test_countdown_roundtrip():
    config = GameConfig(num_snakes=2)
    d = {
        "width": 11,
        "height": 11,
        "turn": 5,
        "snakes": [
            {
                "id": "s0",
                "body": [[2, 2], [2, 1], [1, 1], [1, 2], [1, 3], [1, 3]],
                "health": 50,
                "eliminated_cause": "",
                "eliminated_on_turn": 0,
            },
            {
                "id": "s1",
                "body": [[7, 7]] * 3,
                "health": 100,
                "eliminated_cause": "",
                "eliminated_on_turn": 0,
            },
        ],
        "food": [[0, 0]],
        "hazards": [[3, 3], [3, 3]],
    }
    back = state_to_engine(state_from_engine(d, config), config)
    assert back["snakes"] == d["snakes"]
    assert back["food"] == d["food"] and back["hazards"] == d["hazards"]
