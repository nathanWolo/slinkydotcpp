"""Exact parity with the official rules engine, at scale.

* Engine-generated games: every transition ``pre -> post_rules`` must be
  reproduced exactly by :func:`slinky.rules.rules_step`, and food spawned by
  the map must land on cells our spawn mask allows.
* Our own rollouts: states produced by :class:`BattlesnakeEnv` are sent to the
  engine, which must agree with our next state.

All tests here need Go (they are skipped otherwise).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky.engine_json import state_from_engine, state_to_engine
from slinky.env import BattlesnakeEnv
from slinky.maps import spawn_food_standard, spawn_mask
from slinky.rules import is_game_over, rules_step
from slinky.types import ACTION_NAMES, GameConfig, Ruleset

try:
    from oracle import generate_games, make_step_request, split_games
except ImportError:
    from tests.oracle import generate_games, make_step_request, split_games

pytestmark = pytest.mark.oracle

MOVE_IDS = {name: i for i, name in enumerate(ACTION_NAMES)}

# (id, oracle `games` flags). Hazard maps exercise hazard damage and stacking;
# the rules pipeline is map-independent so any map's transitions are valid tests.
GAME_CONFIGS = [
    ("duel", dict(games=150)),
    ("duel_invalid_moves", dict(games=60, invalid_move_prob=0.05)),
    ("standard_7x7_4p", dict(width=7, height=7, snakes=4, games=80)),
    ("standard_19x19_4p", dict(width=19, height=19, snakes=4, games=20)),
    ("standard_11x11_8p", dict(snakes=8, games=30)),
    ("standard_11x11_12p", dict(snakes=12, games=20)),
    ("wrapped", dict(ruleset="wrapped", games=60)),
    ("constrictor", dict(ruleset="constrictor", games=60)),
    ("wrapped_constrictor", dict(ruleset="wrapped_constrictor", games=40)),
    ("solo_7x7", dict(ruleset="solo", width=7, height=7, snakes=1, games=40)),
    ("royale_map_4p", dict(map="royale", snakes=4, games=40, shrink_every_n_turns=10)),
    ("hazard_pits", dict(map="hz_hazard_pits", snakes=4, games=30)),
    ("sinkholes", dict(map="sinkholes", snakes=4, games=30)),
    ("snail_mode", dict(map="snail_mode", snakes=4, games=30)),
]


def _game_config(flags: dict) -> GameConfig:
    return GameConfig(
        width=flags.get("width", 11),
        height=flags.get("height", 11),
        num_snakes=flags.get("snakes", 2),
        ruleset=flags.get("ruleset", "standard"),
        food_spawn_chance=flags.get("food_spawn_chance", 15),
        minimum_food=flags.get("minimum_food", 1),
        hazard_damage_per_turn=flags.get("hazard_damage", 14),
    )


def _stack(states):
    return jax.tree.map(lambda *xs: np.stack(xs), *states)


@functools.cache
def _batched_step(config: GameConfig):
    return jax.jit(jax.vmap(lambda s, a: rules_step(s, a, config)))


def _moves_to_ids(moves):
    return np.array([MOVE_IDS.get(m, -1) for m in moves], np.int32)


def _compare(ours, theirs, config: GameConfig, context) -> None:
    """Assert two stacked States agree on everything the engine defines."""
    o = jax.device_get(ours)
    t = jax.device_get(theirs)
    alive = t.alive
    checks = {
        "alive": (o.alive, t.alive),
        "elim_cause": (o.elim_cause, t.elim_cause),
        "elim_turn": (o.elim_turn, t.elim_turn),
        "health": (o.health, t.health),
        "head": (o.head, t.head),
        "food": (o.food, t.food),
        "hazard": (o.hazard, t.hazard),
        "length[alive]": (np.where(alive, o.length, 0), np.where(alive, t.length, 0)),
        "body[alive]": (o.body * alive[..., None, None], t.body * alive[..., None, None]),
        "last_move[alive]": (np.where(alive, o.last_move, 0), np.where(alive, t.last_move, 0)),
    }
    for name, (a, b) in checks.items():
        bad = np.nonzero(np.any((a != b).reshape(a.shape[0], -1), axis=1))[0]
        if len(bad):
            i = int(bad[0])
            ours_i = jax.tree.map(lambda x, i=i: x[i], ours)
            pytest.fail(
                f"{name} differs in {len(bad)} transition(s); first: {context(i)}\n"
                f"ours:   {state_to_engine(ours_i, config)}\n"
                f"engine: {context(i, engine=True)}"
            )


@pytest.fixture(scope="module", params=GAME_CONFIGS, ids=[c[0] for c in GAME_CONFIGS])
def corpus(request, oracle_bin):
    name, flags = request.param
    records = generate_games(binary=oracle_bin, seed=1000, **flags)
    return name, flags, _game_config(flags), split_games(records)


def test_transitions_match_engine(corpus):
    name, flags, config, games = corpus
    trans = [t for g in games for t in g["transitions"]]
    assert trans, "no transitions generated"
    pre = _stack([state_from_engine(t["pre"], config) for t in trans])
    post = _stack([state_from_engine(t["post_rules"], config) for t in trans])
    actions = np.stack([_moves_to_ids(t["moves"]) for t in trans])
    ours = _batched_step(config)(pre, actions)

    def context(i, engine=False):
        t = trans[i]
        if engine:
            return t["post_rules"]
        return f"game {t['game']} turn {t['pre']['turn']} moves {t['moves']}\npre: {t['pre']}"

    _compare(ours, post, config, context)
    # The engine's game-over flag is its *pre*-move check.
    over = jax.vmap(lambda a: is_game_over(a, config))(pre.alive)
    assert np.array_equal(np.asarray(over), [t["game_over"] for t in trans])


def test_food_spawns_are_legal(corpus):
    name, flags, config, games = corpus
    if flags.get("map", "standard") != "standard" or config.ruleset.constrictor:
        pytest.skip("only the standard map's spawning is implemented")
    trans = [t for g in games for t in g["transitions"]]
    post = _stack([state_from_engine(t["post_rules"], config) for t in trans])
    masks = np.asarray(jax.vmap(lambda s: spawn_mask(s, config))(post))
    # The engine draws turn T's spawn roll from a generator seeded with
    # game_seed + T, so games with overlapping seed ranges share rolls. Collect
    # one outcome per distinct seed (and check that shared seeds agree).
    roll_by_seed: dict[int, int] = {}
    index = {id(t): i for i, t in enumerate(trans)}
    for g in games:
        for t in g["transitions"]:
            i = index[id(t)]
            before = {tuple(p) for p in t["post_rules"]["food"]}
            after = {tuple(p) for p in t["post_map"]["food"]}
            assert before <= after, "map removed food"
            new = after - before
            for x, y in new:
                assert masks[i, y, x], f"illegal spawn at {(x, y)} in game {t['game']}"
            room = int(masks[i].sum())
            if len(before) < config.minimum_food:
                assert len(new) == min(config.minimum_food - len(before), room)
            elif room > 0:
                assert len(new) <= 1
                seed = g["config"]["game_seed"] + t["pre"]["turn"]
                roll = roll_by_seed.setdefault(seed, len(new))
                assert roll == len(new), "same seed, different roll"
    # Engine spawn probability is (chance - 1) / 100 (see maps.spawn_food_standard).
    trials = len(roll_by_seed)
    if trials > 200:
        p = (config.food_spawn_chance - 1) / 100
        spawned = sum(roll_by_seed.values())
        z = (spawned - trials * p) / math.sqrt(trials * p * (1 - p))
        assert abs(z) < 4.5, (spawned, trials, p)


def test_our_spawn_probability():
    """Our per-turn chance spawn happens with probability (chance - 1) / 100."""
    config = GameConfig(food_spawn_chance=15)
    env = BattlesnakeEnv(config, obs=None)
    state, _ = env.reset(jax.random.key(0))
    state = state._replace(food=jnp.zeros_like(state.food).at[0, 0].set(True))

    def spawned(key):
        return jnp.sum(spawn_food_standard(key, state, config).food) - 1

    n = 200_000
    hits = int(jnp.sum(jax.jit(jax.vmap(spawned))(jax.random.split(jax.random.key(1), n))))
    p = 0.14
    assert abs(hits - n * p) < 5 * math.sqrt(n * p * (1 - p)), hits / n


def _initial_ok(d: dict, config: GameConfig) -> str | None:
    """Check an initial standard-map state against the engine's setup rules."""
    w = config.width
    heads = [tuple(s["body"][0]) for s in d["snakes"]]
    if any(len({tuple(p) for p in s["body"]}) != 1 or len(s["body"]) != 3 for s in d["snakes"]):
        return "snakes must start as 3 stacked segments"
    if len(set(heads)) != len(heads):
        return "two snakes share a start square"
    if w == config.height and w >= 7 and config.num_snakes <= 8:
        mn, md, mx = 1, (w - 1) // 2, w - 2
        corners = {(mn, mn), (mn, mx), (mx, mn), (mx, mx)}
        cardinals = {(mn, md), (md, mn), (md, mx), (mx, md)}
        k = config.num_snakes
        if not (set(heads[: min(k, 4)]) <= corners or set(heads[: min(k, 4)]) <= cardinals):
            return f"first snakes must all be corners or all cardinals: {heads}"
    food = {tuple(p) for p in d["food"]}
    if config.ruleset.constrictor:
        return None if not food else "constrictor starts without food"
    centre = ((w - 1) // 2, (config.height - 1) // 2)
    if w == config.height and w >= 7:
        if centre not in food:
            return "missing centre food"
        if config.num_snakes <= 4 or w * w >= 121:
            if len(food) != config.num_snakes + 1:
                return f"expected one food per snake plus centre, got {sorted(food)}"
            for hx, hy in heads:
                diag = {(hx + dx, hy + dy) for dx in (-1, 1) for dy in (-1, 1)}
                if not diag & food:
                    return f"no starting food diagonal to head {(hx, hy)}"
    return None


def test_initial_states(corpus):
    name, flags, config, games = corpus
    if flags.get("map", "standard") != "standard":
        pytest.skip("only the standard map's setup is implemented")
    for g in games:
        err = _initial_ok(g["initial"], config)
        assert err is None, f"engine initial state violates our model: {err}"
    env = BattlesnakeEnv(config, obs=None)
    reset = jax.jit(jax.vmap(env.reset))
    states, _ = reset(jax.random.split(jax.random.key(0), 64))
    for i in range(64):
        d = state_to_engine(jax.tree.map(lambda x, i=i: x[i], states), config)
        err = _initial_ok(d, config)
        assert err is None, f"our initial state is not one the engine could produce: {err}"


# --- Our rollouts, checked by the engine ------------------------------------------

ROLLOUT_CONFIGS = [
    GameConfig(),
    GameConfig(width=7, height=7, num_snakes=4),
    GameConfig(width=19, height=19, num_snakes=6),
    GameConfig(ruleset=Ruleset.WRAPPED),
    GameConfig(ruleset=Ruleset.CONSTRICTOR),
    GameConfig(width=7, height=7, num_snakes=1),
    GameConfig(minimum_food=3, food_spawn_chance=40),
]


def _rollout(env: BattlesnakeEnv, key, steps: int, eps: float):
    """Random play that mostly avoids certain death; returns (states, actions)."""

    def body(state, k):
        k_pol, k_eps, k_step = jax.random.split(k, 3)
        mask = env.action_mask(state)
        logits = jnp.where(mask, 0.0, -1e9)
        greedy = jax.random.categorical(k_pol, logits)
        rand = jax.random.randint(k_eps, greedy.shape, -1, 4)  # -1: invalid move
        actions = jnp.where(jax.random.uniform(k_eps, greedy.shape) < eps, rand, greedy)
        nxt, _ = env.step_autoreset(k_step, state, actions)
        return nxt, (state, actions)

    state, _ = env.reset(key)
    _, (states, actions) = jax.lax.scan(body, state, jax.random.split(key, steps))
    return states, actions


@pytest.mark.parametrize(
    "config",
    ROLLOUT_CONFIGS,
    ids=lambda c: f"{c.ruleset.value}_{c.width}x{c.height}_{c.num_snakes}p_food{c.minimum_food}",
)
def test_our_rollouts_match_engine(config, oracle):
    env = BattlesnakeEnv(config, obs=None)
    states, actions = jax.jit(jax.vmap(lambda k: _rollout(env, k, 150, 0.08)))(
        jax.random.split(jax.random.key(7), 16)
    )
    flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (states, actions))
    states, actions = jax.device_get(flat)
    keep = ~np.asarray(states.done)
    idx = np.nonzero(keep)[0]
    pre = jax.tree.map(lambda x: x[idx], states)
    acts = actions[idx]
    pre_dicts = [
        state_to_engine(jax.tree.map(lambda x, i=i: x[i], pre), config) for i in range(len(idx))
    ]
    requests = [
        make_step_request(
            config.ruleset.value,
            d,
            [ACTION_NAMES[a] if 0 <= a < 4 else "" for a in acts[i]],
            settings={"damagePerTurn": str(config.hazard_damage_per_turn)},
            solo=config.solo,
        )
        for i, d in enumerate(pre_dicts)
    ]
    responses = oracle.step_many(requests)
    errors = [r["error"] for r in responses if r.get("error")]
    assert not errors, errors[:3]
    theirs = _stack([state_from_engine(r["state"], config) for r in responses])
    # Round-tripping through JSON must be lossless for our own states.
    roundtrip = _stack([state_from_engine(d, config) for d in pre_dicts])
    _compare(roundtrip, pre, config, lambda i, engine=False: f"roundtrip {i}")
    ours = _batched_step(config)(pre, acts)
    _compare(
        ours,
        theirs,
        config,
        lambda i, engine=False: (
            responses[i]["state"] if engine else f"pre {pre_dicts[i]} moves {acts[i]}"
        ),
    )
