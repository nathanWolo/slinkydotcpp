"""Royale: the shrinking hazard zone, as a function, in the environment, and against the engine.

The zone shown at turn ``t`` is a pure function of the game's fixed sequence of
random shrink sides and ``t // shrink_every_n_turns`` (``rules.royale_hazards``).
The engine's own sequence comes from Go's ``math/rand``, which we do not
reproduce, so the engine tests read its sides (``oracle.engine_royale_sides``)
and inject them; our own sides are tested for their distribution instead.
"""

from __future__ import annotations

import dataclasses
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import batched_step, compare_states, engine_sides, moves_to_ids, stack_states

from slinky import mcts
from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.maps import MAPS, RoyaleMap, StandardMap, WithRoyaleSides, get_map
from slinky.policies import random_legal_policy
from slinky.rules import is_game_over, royale_hazards, rules_step
from slinky.types import GameConfig, RoyaleState, Ruleset, royale_num_sides

try:
    from oracle import engine_royale_sides, engine_royale_zones, generate_games, split_games
except ImportError:
    from tests.oracle import engine_royale_sides, engine_royale_zones, generate_games, split_games

ROYALE = GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE, map="royale", shrink_every_n_turns=5)
LEFT_COLUMN, RIGHT_COLUMN, BOTTOM_ROW, TOP_ROW = range(4)


def _zone(sides, turn: int, config: GameConfig) -> np.ndarray:
    return np.asarray(royale_hazards(jnp.asarray(sides, jnp.int8), jnp.int32(turn), config))


def _reference_zone(sides, turn: int, config: GameConfig) -> np.ndarray:
    """The zone by plain Python: one inward step of the safe rectangle per shrink."""
    lo_x, hi_x, lo_y, hi_y = 0, config.width - 1, 0, config.height - 1
    for side in sides[: turn // config.shrink_every_n_turns]:
        lo_x, hi_x, lo_y, hi_y = (
            lo_x + (side == LEFT_COLUMN),
            hi_x - (side == RIGHT_COLUMN),
            lo_y + (side == BOTTOM_ROW),
            hi_y - (side == TOP_ROW),
        )
    return np.array(
        [
            [x < lo_x or x > hi_x or y < lo_y or y > hi_y for x in range(config.width)]
            for y in range(config.height)
        ],
        np.int16,
    )


# --- The hazard function ---------------------------------------------------------


def test_no_hazards_before_the_first_shrink():
    sides = np.random.default_rng(0).integers(0, 4, royale_num_sides(ROYALE))
    n = ROYALE.shrink_every_n_turns
    assert not any(_zone(sides, t, ROYALE).any() for t in range(n))
    assert _zone(sides, n, ROYALE).any()


@pytest.mark.parametrize("side", range(4))
def test_each_side_shrinks_its_own_edge(side):
    config = GameConfig(width=7, height=5, shrink_every_n_turns=3)  # not square
    edge = np.zeros((5, 7), np.int16)  # grid[y, x]
    edge[:, 0 if side == LEFT_COLUMN else 6] = side in (LEFT_COLUMN, RIGHT_COLUMN)
    if side in (BOTTOM_ROW, TOP_ROW):
        edge[0 if side == BOTTOM_ROW else 4, :] = 1
    sides = np.full(royale_num_sides(config), side)
    assert np.array_equal(_zone(sides, 3, config), edge)
    assert _zone(sides, 3, config).sum() == (5 if side < 2 else 7)
    assert np.array_equal(_zone(sides, 5, config), edge)  # still one shrink


@pytest.mark.parametrize("width, height, n", [(11, 11, 25), (7, 5, 3), (4, 9, 1), (1, 1, 1)])
def test_zone_matches_python_reference(width, height, n):
    config = GameConfig(width=width, height=height, shrink_every_n_turns=n)
    k = royale_num_sides(config)
    rng = np.random.default_rng(width * 100 + height)
    # Shrinks beyond the k that cover the board (extra sides, long turns) change nothing.
    sides = rng.integers(0, 4, (400, k + 8))
    turns = rng.integers(0, (k + 5) * n, 400)
    turns[:8] = [0, 1, n - 1, n, n + 1, k * n - 1, k * n, (k + 5) * n - 1]
    zone = jax.jit(jax.vmap(lambda s, t: royale_hazards(s, t, config)))(
        jnp.asarray(sides[:, :k], jnp.int8), jnp.asarray(turns, jnp.int32)
    )
    want = np.stack([_reference_zone(s, int(t), config) for s, t in zip(sides, turns, strict=True)])
    assert np.array_equal(np.asarray(zone), want)


def test_zone_is_zero_one_monotone_and_collapses():
    config = GameConfig(shrink_every_n_turns=2)
    k = royale_num_sides(config)
    rng = np.random.default_rng(1)
    for _ in range(20):
        sides = rng.integers(0, 4, k)
        zones = [_zone(sides, t, config) for t in range(2 * (k + 3))]
        assert all(set(np.unique(z)) <= {0, 1} for z in zones)
        assert all(np.all(a <= b) for a, b in zip(zones, zones[1:], strict=False))
        # Too few shrinks to cover the board leave a safe cell; k shrinks always cover it.
        assert all(not z.all() for z in zones[: 2 * config.width - 1])
        assert all(z.all() for z in zones[2 * k :])


def test_zone_changes_only_when_a_shrink_is_due():
    config = GameConfig(width=9, height=6, shrink_every_n_turns=4)
    sides = np.random.default_rng(2).integers(0, 4, royale_num_sides(config))
    for t in range(80):
        same = np.array_equal(_zone(sides, t, config), _zone(sides, t + 1, config))
        assert same or (t + 1) % 4 == 0, t


def test_shrink_interval_must_be_positive():
    for n in (0, -1):
        with pytest.raises(ValueError, match="shrink_every_n_turns"):
            GameConfig(shrink_every_n_turns=n)


# --- The ruleset stage -----------------------------------------------------------


def _skirmish(turn: int, sides, config: GameConfig, hazards=()):
    """A two-snake position at ``turn`` with the given shrink sides."""
    d = {
        "width": config.width,
        "height": config.height,
        "turn": turn,
        "food": [],
        "hazards": [list(h) for h in hazards],
        "snakes": [
            {"id": "s0", "body": [[1, 5], [2, 5], [3, 5]], "health": 100},
            {"id": "s1", "body": [[8, 8], [8, 7], [8, 6]], "health": 100},
        ],
    }
    return state_from_engine(d, config, RoyaleState(jnp.asarray(sides, jnp.int8)))


def test_first_damage_lands_on_the_move_after_the_zone_appears():
    config = GameConfig(ruleset=Ruleset.ROYALE, shrink_every_n_turns=5)
    sides = np.zeros(royale_num_sides(config), np.int8)  # always the left column
    # Turn 4 is shown without hazards, so stepping onto the column that is about to
    # become hazard costs only the usual 1 health, and the column appears at turn 5.
    state = _skirmish(4, sides, config)
    assert not state.hazard.any()
    after = rules_step(state, np.array([2, 0]), config)  # s0 moves left to (0, 5)
    assert int(after.health[0]) == 99
    assert np.array_equal(np.asarray(after.hazard), _zone(sides, 5, config))
    assert np.asarray(after.hazard)[:, 0].all() and np.asarray(after.hazard).sum() == 11
    # Stepping within the column at turn 5 costs 14 more.
    after = rules_step(after._replace(turn=jnp.int32(5)), np.array([0, 0]), config)
    assert int(after.health[0]) == 99 - 1 - 14


def test_zone_is_not_updated_once_the_game_is_over():
    config = GameConfig(ruleset=Ruleset.ROYALE, shrink_every_n_turns=5)
    sides = np.array([2, 3, 0, 1] * 5 + [0], np.int8)
    state = _skirmish(24, sides, config)
    over = state._replace(alive=jnp.array([True, False]), done=jnp.array(True))
    assert not np.array_equal(_zone(sides, 25, config), _zone(sides, 24, config))
    out = rules_step(over, np.array([0, 0]), config)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, out, over))
    live = rules_step(state, np.array([0, 0]), config)  # the engine runs the stage here
    assert np.array_equal(np.asarray(live.hazard), _zone(sides, 25, config))


# --- The maps and the environment ------------------------------------------------

ROYALE_COMBINATIONS = [
    pytest.param(Ruleset.ROYALE, "royale", id="royale_ruleset_royale_map"),
    pytest.param(Ruleset.STANDARD, "royale", id="standard_ruleset_royale_map"),
    pytest.param(Ruleset.ROYALE, "standard", id="royale_ruleset_standard_map"),
    pytest.param(Ruleset.ROYALE, "empty", id="royale_ruleset_empty_map"),
]


def _play(env: BattlesnakeEnv, key, steps: int):
    """Random legal play with autoreset; returns the state after every step."""

    def body(state, k):
        k_pol, k_step = jax.random.split(k)
        state, ts = env.step_autoreset(k_step, state, random_legal_policy(k_pol, state, env))
        return state, (state, ts.done)

    state, _ = env.reset(key)
    _, out = jax.lax.scan(body, state, jax.random.split(key, steps))
    return out


@pytest.mark.parametrize("ruleset, map_id", ROYALE_COMBINATIONS)
def test_environment_follows_the_hazard_function(ruleset, map_id):
    config = GameConfig(num_snakes=4, ruleset=ruleset, map=map_id, shrink_every_n_turns=3)
    env = BattlesnakeEnv(config, obs=None)
    states, done = jax.jit(jax.vmap(lambda k: _play(env, k, 150)))(
        jax.random.split(jax.random.key(0), 16)
    )
    sides = np.asarray(states.map_state.sides)  # [games, steps, K]
    k = royale_num_sides(config)
    assert sides.shape[-1] == k and sides.min() >= 0 and sides.max() <= 3
    want = jax.vmap(jax.vmap(lambda s, t: royale_hazards(s, t, config)))(
        states.map_state.sides, states.turn
    )
    assert np.array_equal(np.asarray(states.hazard), np.asarray(want))
    assert np.asarray(states.hazard).any()  # games reach the first shrink
    # The sides are drawn once per game: they change exactly when a new game starts.
    new_game = np.asarray(states.turn)[:, 1:] == 0
    changed = np.any(sides[:, 1:] != sides[:, :-1], axis=-1)
    assert np.array_equal(changed, new_game)
    assert np.asarray(done).any() and new_game.any()


def test_stepping_a_finished_royale_game_changes_nothing():
    env = BattlesnakeEnv(ROYALE, obs=None)

    def play(key):
        def body(state, k):
            k_pol, k_step = jax.random.split(k)
            return env.step(k_step, state, random_legal_policy(k_pol, state, env))[0], None

        state, _ = env.reset(key)
        state, _ = jax.lax.scan(body, state, jax.random.split(key, 150))
        return state, env.step(key, state, jnp.zeros(4, jnp.int32))[0]

    finished, again = jax.jit(jax.vmap(play))(jax.random.split(jax.random.key(3), 8))
    assert np.asarray(finished.done).all() and np.asarray(finished.hazard).any()
    assert jax.tree.all(jax.tree.map(jnp.array_equal, finished, again))


def test_standard_ruleset_on_the_royale_map_is_the_same_game():
    # Only the hazards right after the ruleset differ, and the map update repairs them.
    standard = GameConfig(num_snakes=4, map="royale", shrink_every_n_turns=4)
    royale = dataclasses.replace(standard, ruleset=Ruleset.ROYALE)
    envs = [BattlesnakeEnv(c, obs=None) for c in (standard, royale)]
    steps = [jax.jit(env.step) for env in envs]
    key = jax.random.key(5)
    pair = [env.reset(key)[0] for env in envs]
    assert jax.tree.all(jax.tree.map(jnp.array_equal, *pair))
    for t in range(60):
        k_act, k_step = jax.random.split(jax.random.fold_in(key, t))
        actions = jax.random.randint(k_act, (4,), -1, 4)
        pair = [step(k_step, s, actions)[0] for step, s in zip(steps, pair, strict=True)]
        assert jax.tree.all(jax.tree.map(jnp.array_equal, *pair)), t
    assert np.asarray(pair[0].hazard).any()


def test_standard_ruleset_zone_moves_in_the_map_not_in_the_rules_step():
    # Code that only calls rules_step (a spawn-free search) keeps the root's hazards in this
    # combination; stepping through the environment (spawn_food=True) moves the zone.
    config = GameConfig(num_snakes=4, map="royale", shrink_every_n_turns=2)
    env = BattlesnakeEnv(config, obs=None)
    state = env.reset(jax.random.key(0))[0]._replace(turn=jnp.int32(1))  # the zone shows at 2
    actions = jnp.zeros(4, jnp.int32)
    assert not rules_step(state, actions, config).hazard.any()
    zone = royale_hazards(state.map_state.sides, 2, config)
    assert zone.any()
    for spawn_food in (False, True):
        after, _ = mcts._transition(
            jax.random.key(1), state, actions, env, mcts.MCTSConfig(spawn_food=spawn_food)
        )
        assert np.array_equal(after.hazard, zone if spawn_food else state.hazard)


def test_royale_ruleset_works_with_every_map():
    for map_id in MAPS:
        config = GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE, map=map_id)
        state, _ = BattlesnakeEnv(config, obs=None).reset(jax.random.key(0))
        assert isinstance(state.map_state, RoyaleState)
        assert state.map_state.sides.shape == (royale_num_sides(config),)
        assert not state.hazard.any()


def test_royale_ruleset_keeps_the_other_maps_setup():
    # Only the sides are added: placement and food are the wrapped map's own.
    config = GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE, map="standard")
    wrapped = get_map("standard", Ruleset.ROYALE)
    assert wrapped is not MAPS["standard"] and get_map("royale", Ruleset.ROYALE) is MAPS["royale"]
    assert get_map("standard", Ruleset.STANDARD) is MAPS["standard"]
    for seed in range(4):
        key = jax.random.key(seed)
        ours = wrapped.setup(key, config)
        base = StandardMap().setup(jax.random.split(key)[0], config)
        assert jax.tree.all(jax.tree.map(jnp.array_equal, ours._replace(map_state=()), base))
        assert ours.map_state.sides.shape == (royale_num_sides(config),)


def test_royale_sides_do_not_replace_another_map_state():
    class Stateful(StandardMap):
        def init_map_state(self, config):
            return jnp.zeros(3, jnp.int32)

    config = GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE)
    for base in (Stateful(), RoyaleMap()):  # a map with its own state, and wrapping twice
        with pytest.raises(ValueError, match="map_state"):
            WithRoyaleSides(base).setup(jax.random.key(0), config)


def test_royale_map_keeps_the_standard_setup_apart_from_sides():
    config = GameConfig(num_snakes=4, map="royale")
    state = RoyaleMap().setup(jax.random.key(0), config)
    assert not state.hazard.any() and int(state.turn) == 0
    assert state.food.sum() == 5  # one per snake plus the centre
    assert jnp.all(state.length == 3) and jnp.all(state.health == 100)


def test_royale_rules_step_alone_needs_the_sides():
    # rules_step on its own (as in search) cannot invent the sides.
    state, _ = BattlesnakeEnv(ROYALE, obs=None).reset(jax.random.key(0))
    with pytest.raises(ValueError, match="RoyaleState"):
        rules_step(state._replace(map_state=()), jnp.zeros(4, jnp.int32), ROYALE)


# --- Distribution of our sides ---------------------------------------------------


def _p_value(stat: float, df: int) -> float:
    return float(jax.scipy.special.gammaincc(df / 2, stat / 2))


def _independence_p(table: np.ndarray) -> float:
    """p-value of Pearson's independence test on a table of counts."""
    table = table.astype(np.float64)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    return _p_value(float(((table - expected) ** 2 / expected).sum()), (len(table) - 1) * 3)


@pytest.mark.parametrize("map_id", ["royale", "standard", "empty"])
def test_sides_are_uniform_independent_and_unrelated_to_the_start_layout(map_id):
    # The royale map draws the sides with its placement; other maps get them from `get_map`.
    config = GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE, map=map_id)
    game_map = get_map(map_id, Ruleset.ROYALE)
    n, k = 100_000, royale_num_sides(config)

    def draw(key):
        state = game_map.setup(key, config)
        corner = jnp.all((state.head[0] == 1) | (state.head[0] == 9))  # else cardinal starts
        return state.map_state.sides, corner

    sides, corner = jax.jit(jax.vmap(draw))(jax.random.split(jax.random.key(0), n))
    sides, corner = np.asarray(sides, np.int64), np.asarray(corner, np.int64)
    assert 0.48 < corner.mean() < 0.52

    def table(a, b):  # counts[a value, b value], b in 0..3
        return np.bincount(a * 4 + b, minlength=4 * (a.max() + 1)).reshape(-1, 4)

    counts = [np.bincount(sides[:, i], minlength=4) for i in range(k)]
    uniform = [_p_value(float(((c - n / 4) ** 2 / (n / 4)).sum()), 3) for c in counts]
    neighbours = [_independence_p(table(sides[:, i], sides[:, i + 1])) for i in range(k - 1)]
    layout = [_independence_p(table(corner, sides[:, i])) for i in range(k)]
    # About 60 tests at once; the keys are fixed, so this is deterministic, and loose.
    for name, ps in (("uniform", uniform), ("neighbours", neighbours), ("layout", layout)):
        assert min(ps) > 1e-4, (name, np.round(ps, 5))


# --- Against the engine ----------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "width, height, n", [(11, 11, 1), (11, 11, 3), (11, 11, 25), (9, 7, 4), (5, 13, 2)]
)
def test_zone_matches_engine_through_collapse(oracle, width, height, n):
    config = GameConfig(width=width, height=height, ruleset=Ruleset.ROYALE, shrink_every_n_turns=n)
    k = royale_num_sides(config)
    turns = range((k + 3) * n)  # well past the whole-board collapse
    for seed in random.Random(width * 31 + height).sample(range(1, 2**62), 8):
        sides = engine_royale_sides(oracle, seed, k, width, height)
        zones = engine_royale_zones(oracle, seed, turns, n, width, height)
        for t, hazards in zip(turns, zones, strict=True):
            engine = np.zeros((height, width), np.int16)
            for x, y in hazards:
                engine[y, x] += 1
            assert np.array_equal(_zone(sides, t + 1, config), engine), (seed, t)


# (ruleset, map, shrink_every_n_turns) and extra `oracle games` flags. N = 1 reaches the
# whole-board collapse; the last case is a small non-square board with three snakes.
CORPORA = [
    (ruleset, map_id, n, {})
    for ruleset, map_id in (("royale", "royale"), ("standard", "royale"), ("royale", "standard"))
    for n in (25, 5, 1)
] + [("royale", "royale", 4, dict(width=9, height=7, snakes=3))]


@pytest.fixture(
    scope="module",
    params=CORPORA,
    ids=[f"{r}_ruleset_{m}_map_every{n}" + ("_9x7" if x else "") for r, m, n, x in CORPORA],
)
def royale_corpus(request, oracle_bin, oracle):
    ruleset, map_id, n, extra = request.param
    flags = (
        dict(ruleset=ruleset, map=map_id, snakes=4, games=24, seed=2000, shrink_every_n_turns=n)
        | extra
    )
    config = GameConfig(
        width=flags.get("width", 11),
        height=flags.get("height", 11),
        num_snakes=flags["snakes"],
        ruleset=ruleset,
        map=map_id,
        shrink_every_n_turns=n,
    )
    games = split_games(generate_games(binary=oracle_bin, **flags))
    sides = {g["game"]: engine_sides(oracle, g["config"]["game_seed"], config) for g in games}
    return config, games, sides


@pytest.mark.oracle
def test_ruleset_step_matches_engine(royale_corpus):
    config, games, sides = royale_corpus
    trans = [t for g in games for t in g["transitions"]]
    assert trans
    pre = stack_states([state_from_engine(t["pre"], config, sides[t["game"]]) for t in trans])
    post = stack_states(
        [state_from_engine(t["post_rules"], config, sides[t["game"]]) for t in trans]
    )
    actions = np.stack([moves_to_ids(t["moves"]) for t in trans])
    ours = batched_step(config)(pre, actions)

    def context(i, engine=False):
        t = trans[i]
        return (
            t["post_rules"] if engine else f"game {t['game']} turn {t['pre']['turn']}\n{t['pre']}"
        )

    # Hazards included: the royale ruleset regenerates them here, other rulesets leave them.
    compare_states(ours, post, config, context)
    over = jax.vmap(lambda a: is_game_over(a, config))(pre.alive)
    assert np.array_equal(np.asarray(over), [t["game_over"] for t in trans])


@pytest.mark.oracle
def test_map_update_matches_engine(royale_corpus):
    config, games, sides = royale_corpus
    if config.map != "royale":
        pytest.skip("only the royale map places hazards in its map update")
    trans = [t for g in games for t in g["transitions"]]
    post_rules = stack_states(
        [state_from_engine(t["post_rules"], config, sides[t["game"]]) for t in trans]
    )
    post_map = stack_states([_hazard_grid(t["post_map"], config) for t in trans])
    keys = jax.random.split(jax.random.key(0), len(trans))
    out = jax.jit(jax.vmap(lambda k, s: get_map("royale").post_update(k, s, config)))(
        keys, post_rules
    )
    # Also on the game-over call, where the ruleset skips its stage but the map does not.
    assert np.array_equal(np.asarray(out.hazard), post_map)


def _hazard_grid(state: dict, config: GameConfig) -> np.ndarray:
    grid = np.zeros((config.height, config.width), np.int16)
    for x, y in state["hazards"]:
        grid[y, x] += 1
    return grid


@pytest.mark.oracle
def test_every_state_shows_the_hazard_function(royale_corpus):
    config, games, sides = royale_corpus
    shown, zone_turns, owners, labels = [], [], [], []
    for g in games:
        records = [("initial", g["initial"], g["initial"]["turn"])]
        for t in g["transitions"]:
            records.append((f"pre {t['pre']['turn']}", t["pre"], t["pre"]["turn"]))
            # The call that finds the game over still gets a map update, but the royale
            # ruleset's own stage is skipped. A map without hazards leaves the zone as it
            # was, so the final state (turn T + 1) shows the zone of turn T and misses a
            # shrink that falls due then; the royale map regenerates it for T + 1.
            stale = t["game_over"] and config.map != "royale"
            zone_turn = t["pre"]["turn"] if stale else t["post_map"]["turn"]
            records.append((f"post_map {t['post_map']['turn']}", t["post_map"], zone_turn))
        for label, d, zone_turn in records:
            shown.append(_hazard_grid(d, config))
            zone_turns.append(zone_turn)
            owners.append(np.asarray(sides[g["game"]].sides))
            labels.append((g["game"], label))
    want = jax.vmap(lambda s, t: royale_hazards(s, t, config))(
        jnp.asarray(np.stack(owners)), jnp.asarray(zone_turns, jnp.int32)
    )
    bad = np.nonzero(np.any(np.asarray(want) != np.stack(shown), axis=(1, 2)))[0]
    assert not len(bad), [labels[i] for i in bad[:5]]
    assert np.stack(shown).any(), "no game reached the first shrink"
