"""The PufferLib environment (``puffer/``): its C port of the game, checked against slinky.

slinky is checked against the official engine (``test_parity.py``, ``test_fuzz_rules.py``), so
the C environment is checked against slinky, through the binding's test hooks:

* **transitions**: slinky rollouts, fuzzed engine states (``test_fuzz_rules.random_case``),
  scrambled states (arbitrary grids, overlapping snakes) and states from C play go through
  the C rules step and full step; the result equals slinky's on every field, except that
  spawned food only has to land where slinky's spawn mask allows, as often as slinky would;
* **observations**: the uint8 board times ``channel_scale`` is slinky's egocentric
  observation and the mask bytes are ``action_mask``;
* **new games**: legal placements and starting food, and the distribution of every random
  choice (fixed seeds and generous thresholds, so nothing is flaky);
* **agent slots**: PufferLib's reward / terminal / autoreset semantics through
  ``Battlesnake.step``, and the log;
* **engine** (marked ``oracle``): engine-generated transitions, reproduced by the C rules
  step.

Needs the built binding (``.venv-puffer/bin/python puffer/setup.py build_ext --inplace``);
run with ``.venv-puffer/bin/python -m pytest tests/test_puffer_env.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import subprocess
import sys
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import board, compare_states, moves_to_ids, rollout_states
from test_fuzz_rules import random_case

from slinky.engine_json import _direction, body_to_countdown, state_from_engine
from slinky.env import BattlesnakeEnv, placement_reward
from slinky.maps import (
    _fixed_food_candidates,
    _fixed_start_points,
    _quadrant_points,
    _random_start_cells,
    placement_kind,
    spawn_mask,
)
from slinky.modes import MODE_NAMES, mode_config
from slinky.observations import make_obs_fn
from slinky.render import render
from slinky.rules import action_mask, royale_hazards, rules_step
from slinky.types import (
    ENGINE_TO_CAUSE,
    LEFT,
    RIGHT,
    UP,
    Cause,
    GameConfig,
    RoyaleState,
    Ruleset,
    State,
    royale_num_sides,
)

pytest.importorskip("pufferlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `puffer` is a directory of the repository, not installed
    sys.path.append(str(ROOT))
binding = pytest.importorskip(
    "puffer.binding", reason="build it: .venv-puffer/bin/python puffer/setup.py build_ext --inplace"
)
from puffer import battlesnake, obs_format  # noqa: E402

STATE_FIELDS = (
    "body",
    "head",
    "length",
    "health",
    "alive",
    "last_move",
    "food",
    "hazard",
    "turn",
    "elim_cause",
    "elim_turn",
    "done",
)
# z-scores (of counts, chi-square statistics or binomial counts) beyond this fail, about a
# 1e-6 chance per check. The seeds are fixed, so a pass is deterministic; the margin only
# keeps a correct implementation far from the edge.
Z_MAX = 5.0
FAST_COMPILE = {"xla_backend_optimization_level": 0}


# --- The C game and slinky states --------------------------------------------------------


def uses_sides(config: GameConfig) -> bool:
    return config.ruleset == Ruleset.ROYALE or config.map == "royale"


def c_kwargs(config: GameConfig, reward: str = "win_loss", map: str | None = None) -> dict:
    return dict(
        width=config.width,
        height=config.height,
        num_snakes=config.num_snakes,
        ruleset=config.ruleset.value,
        map=map or config.map,
        food_spawn_chance=config.food_spawn_chance,
        minimum_food=config.minimum_food,
        hazard_damage_per_turn=config.hazard_damage_per_turn,
        shrink_every_n_turns=config.shrink_every_n_turns,
        max_turns=config.max_turns or 0,
        reward=reward,
    )


class CGame:
    """One C game with buffers of its own, driven through the binding's test hooks."""

    def __init__(self, config: GameConfig, reward: str = "win_loss", seed: int = 0, map=None):
        n, w, h = config.num_snakes, config.width, config.height
        self.config = config
        self.board_size = obs_format.board_size(w, h, config.ruleset.wrapped)
        self.obs = np.zeros((n, obs_format.obs_size(w, h, config.ruleset.wrapped)), np.uint8)
        self.actions = np.zeros(n, np.int32)
        self.rewards = np.zeros(n, np.float32)
        self.terminals = np.zeros(n, bool)
        self.truncations = np.zeros(n, bool)
        self.h = binding.env_init(
            self.obs,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            seed,
            **c_kwargs(config, reward, map),
        )

    def __del__(self):
        if hasattr(self, "h"):
            binding.env_close(self.h)

    def get(self) -> dict:
        return binding.env_get(self.h)

    def put(self, fields: dict) -> None:
        binding.env_put(self.h, **fields)


def put_fields(state: State, config: GameConfig) -> dict:
    """``env_put`` keywords for one unbatched slinky state."""
    fields = {f: np.asarray(getattr(state, f)) for f in STATE_FIELDS}
    if isinstance(state.map_state, RoyaleState):
        fields["sides"] = np.asarray(state.map_state.sides, np.int8)
    else:
        fields["sides"] = np.zeros(royale_num_sides(config), np.int8)
    return fields


def to_state(fields: list[dict], config: GameConfig) -> State:
    """A batched slinky state from ``env_get`` / :func:`put_fields` dicts."""
    arrays = {f: np.stack([d[f] for d in fields]) for f in STATE_FIELDS}
    sides = RoyaleState(np.stack([d["sides"] for d in fields])) if uses_sides(config) else ()
    return State(**arrays, map_state=sides)


def game_over(alive: np.ndarray, config: GameConfig) -> np.ndarray:
    """slinky.rules.is_game_over over a batch, in numpy."""
    remaining = np.asarray(alive).sum(axis=-1)
    return remaining == 0 if config.solo else remaining <= 1


def unbatch(state: State, i: int) -> State:
    return jax.tree.map(lambda x: x[i], state)


def describe(fields: dict, actions, config: GameConfig) -> str:
    return f"\n{render(unbatch(to_state([fields], config), 0))}\nactions {list(actions)}"


# --- State sources -------------------------------------------------------------------------


def random_sides(rng, config: GameConfig) -> np.ndarray:
    return rng.integers(0, 4, royale_num_sides(config)).astype(np.int8)


def engine_fields(d: dict, config: GameConfig, sides: np.ndarray | None) -> dict:
    """``state_from_engine`` in plain numpy, from the same helpers (much faster than its jnp
    version; :func:`fuzz_fields` checks the two agree)."""
    h, w, n = config.height, config.width, config.num_snakes
    body = np.zeros((n, h, w), np.int16)
    head = np.zeros((n, 2), np.int32)
    last_move = np.full(n, UP, np.int8)
    cause = np.array([ENGINE_TO_CAUSE[s.get("eliminated_cause", "")] for s in d["snakes"]], np.int8)
    for i, s in enumerate(d["snakes"]):
        segs = s["body"]
        if segs:
            head[i] = segs[0]
            if len(segs) >= 2:
                last_move[i] = _direction(segs[0], segs[1])
        if cause[i] == Cause.NONE:
            body[i] = body_to_countdown(segs, h, w)
    food = np.zeros((h, w), bool)
    for x, y in d["food"]:
        if food[y, x]:
            raise ValueError(f"duplicate food at {(x, y)} is not representable")
        food[y, x] = True
    hazard = np.zeros((h, w), np.int16)
    for x, y in d.get("hazards", []):
        if 0 <= x < w and 0 <= y < h:
            hazard[y, x] += 1
    alive = cause == Cause.NONE
    return dict(
        body=body,
        head=head,
        length=np.array([len(s["body"]) for s in d["snakes"]], np.int32),
        health=np.array([s["health"] for s in d["snakes"]], np.int32),
        alive=alive,
        last_move=last_move,
        food=food,
        hazard=hazard,
        turn=np.asarray(d["turn"], np.int32),
        elim_cause=cause,
        elim_turn=np.array([s.get("eliminated_on_turn", 0) for s in d["snakes"]], np.int32),
        done=np.asarray(game_over(alive, config)),
        sides=sides if sides is not None else np.zeros(royale_num_sides(config), np.int8),
    )


def fuzz_fields(config: GameConfig, count: int, rng) -> tuple[list[dict], list]:
    """``test_fuzz_rules.random_case`` states: coiled and stacked bodies, dead snakes with
    off-board heads, food and stacked hazards next to heads, boundary health, invalid moves."""
    fields, actions = [], []
    for k in range(count):
        d, moves = random_case(
            rng, config.width, config.height, config.num_snakes, config.ruleset.wrapped
        )
        sides = random_sides(rng, config) if uses_sides(config) else None
        fields.append(engine_fields(d, config, sides))
        actions.append(moves_to_ids(moves))
        if k < 5:
            ms = RoyaleState(jnp.asarray(sides)) if sides is not None else ()
            reference = put_fields(jax.device_get(state_from_engine(d, config, ms)), config)
            for f, v in reference.items():
                assert np.array_equal(fields[-1][f], v) and fields[-1][f].dtype == v.dtype, f
    return fields, actions


def scrambled_fields(config: GameConfig, count: int, rng) -> tuple[list[dict], list]:
    """Arbitrary grids: living snakes overlap, countdowns need not form bodies, heads may be
    off the board. The rules and observations are defined on any state, and slinky's
    definitions must hold literally there too."""
    n, h, w = config.num_snakes, config.height, config.width
    fields, actions = [], []
    for _ in range(count):
        alive = rng.random(n) < 0.8
        body = np.where(rng.random((n, h, w)) < 0.2, rng.integers(1, 7, (n, h, w)), 0)
        head = np.stack([rng.integers(0, w, n), rng.integers(0, h, n)], axis=1)
        length = rng.integers(1, 9, n)
        for i in range(n):
            if rng.random() < 0.8:
                body[i, head[i, 1], head[i, 0]] = length[i]
            if rng.random() < 0.05:
                head[i, 0] = -1 if rng.random() < 0.5 else w
        fields.append(
            dict(
                body=body.astype(np.int16),
                head=head.astype(np.int32),
                length=length.astype(np.int32),
                health=rng.integers(0, 102, n).astype(np.int32),
                alive=alive,
                last_move=rng.integers(0, 4, n).astype(np.int8),
                food=rng.random((h, w)) < 0.15,
                hazard=np.where(rng.random((h, w)) < 0.2, rng.integers(1, 4, (h, w)), 0).astype(
                    np.int16
                ),
                turn=np.int32(rng.integers(0, 200)),
                elim_cause=np.where(alive, 0, rng.integers(1, 7, n)).astype(np.int8),
                elim_turn=np.where(alive, 0, rng.integers(1, 100, n)).astype(np.int32),
                done=np.bool_(rng.random() < 0.05),
                sides=random_sides(rng, config)
                if uses_sides(config)
                else np.zeros(royale_num_sides(config), np.int8),
            )
        )
        actions.append(rng.integers(-1, 5, n))
    return fields, actions


def play_actions(game: CGame, rng, eps: float = 0.1) -> np.ndarray:
    """Mostly uniformly random legal moves (by the C mask), sometimes anything, invalid included."""
    binding.observe(game.h)
    mask = game.obs[:, game.board_size :] > 0
    acts = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    noisy = rng.random(len(acts)) < eps
    acts[noisy] = rng.integers(-1, 5, int(noisy.sum()))
    return acts


def c_play_fields(config: GameConfig, steps: int, rng) -> tuple[list[dict], list]:
    """Every state of C self-play (finished games included, then a new game)."""
    game = CGame(config, seed=int(rng.integers(1 << 30)))
    fields, actions = [], []
    for _ in range(steps):
        d = game.get()
        fields.append(d)
        actions.append(play_actions(game, rng))
        if d["done"]:
            binding.new_game(game.h)
        else:
            binding.game_step(game.h, actions[-1])
    return fields, actions


def slinky_rollout_fields(config: GameConfig, rng) -> tuple[list[dict], list]:
    states = jax.device_get(rollout_states(config, batch=8, turns=30, every_turn=True))
    count = states.alive.shape[0]
    fields = [put_fields(unbatch(states, i), config) for i in range(count)]
    return fields, list(rng.integers(-1, 5, (count, config.num_snakes)))


# --- Per-config transition bundles ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Case:
    id: str
    config: GameConfig
    fuzz: int = 120
    scrambled: int = 80
    play: int = 300
    slinky_rollout: bool = False


CASES = [
    Case("duel", GameConfig(), slinky_rollout=True),
    Case("standard_4p_max60", GameConfig(num_snakes=4, max_turns=60)),
    Case("9x7_5p_dmg100", GameConfig(width=9, height=7, num_snakes=5, hazard_damage_per_turn=100)),
    Case("1x8_2p", GameConfig(width=1, height=8, num_snakes=2)),
    Case(
        "19x19_16p",
        GameConfig(
            width=19,
            height=19,
            num_snakes=16,
            hazard_damage_per_turn=30,
            minimum_food=3,
            food_spawn_chance=40,
        ),
        fuzz=25,
        scrambled=10,
        play=40,
    ),
    Case("one_snake_max150", GameConfig(num_snakes=1, max_turns=150)),
    Case("solo_7x7_3p", GameConfig(width=7, height=7, num_snakes=3, ruleset="solo")),
    # Odd width, even height: both roll offsets of the wrapped view.
    Case("wrapped_11x8_3p", GameConfig(width=11, height=8, num_snakes=3, ruleset="wrapped")),
    Case("constrictor_7x7_4p", GameConfig(width=7, height=7, num_snakes=4, ruleset="constrictor")),
    Case(
        "wrapped_constrictor_6x6",
        GameConfig(
            width=6,
            height=6,
            num_snakes=4,
            ruleset="wrapped_constrictor",
            hazard_damage_per_turn=20,
        ),
    ),
    Case(
        "royale_4p",
        GameConfig(num_snakes=4, ruleset="royale", map="royale", shrink_every_n_turns=5),
    ),
    Case(  # the royale ruleset on the standard map, with healing hazards
        "royale_rules_7x9_heal",
        GameConfig(
            width=7,
            height=9,
            num_snakes=3,
            ruleset="royale",
            shrink_every_n_turns=3,
            hazard_damage_per_turn=-7,
        ),
    ),
    Case(  # the standard ruleset on the royale map: the map places the zone
        "royale_map_standard_rules",
        GameConfig(num_snakes=4, map="royale", shrink_every_n_turns=4, hazard_damage_per_turn=100),
    ),
    Case(  # the empty map (no food), under the royale ruleset, with harmless hazards
        "royale_rules_empty_5x5_dmg0",
        GameConfig(
            width=5,
            height=5,
            num_snakes=3,
            ruleset="royale",
            map="empty",
            shrink_every_n_turns=2,
            hazard_damage_per_turn=0,
        ),
    ),
]
CASE_IDS = [c.id for c in CASES]


@functools.cache
def slinky_reference(config: GameConfig):
    """Everything slinky says about a batch of (state, actions), in one compiled function."""
    env = BattlesnakeEnv(config, obs=None)
    obs_fn = make_obs_fn(config)

    def one(key, state, actions):
        after_rules = rules_step(state, actions, config)
        stepped, ts = env.step(key, state, actions)
        return dict(
            rules=after_rules,
            step=stepped,
            win_loss=ts.reward,
            placement=jnp.where(state.done, 0.0, placement_reward(state, stepped, config)),
            truncated=ts.truncated,
            spawn=spawn_mask(after_rules, config),
            obs=obs_fn(state),
            mask=action_mask(state, config),
        )

    # The batches are small: LLVM's optimisations would only cost compile time.
    return jax.jit(jax.vmap(one), compiler_options=FAST_COMPILE)


@functools.cache
def bundle(case: Case) -> dict:
    """States of every source, slinky's answers, and the C environment's, for one config."""
    config = case.config
    rng = np.random.default_rng(zlib.crc32(case.id.encode()))
    sources = [
        fuzz_fields(config, case.fuzz, rng),
        scrambled_fields(config, case.scrambled, rng),
        c_play_fields(config, case.play, rng),
    ]
    if case.slinky_rollout:
        sources.append(slinky_rollout_fields(config, rng))
    pre = [f for fields, _ in sources for f in fields]
    actions = np.array([a for _, acts in sources for a in acts], np.int32)
    keys = jax.random.split(jax.random.key(0), len(pre))
    ref = jax.device_get(slinky_reference(config)(keys, to_state(pre, config), actions))

    game, placement = CGame(config), CGame(config, reward="placement")
    ours = dict(rules=[], step=[], win_loss=[], placement=[], obs=[])
    for fields, acts in zip(pre, actions, strict=True):
        game.put(fields)
        binding.rules_step(game.h, acts)
        ours["rules"].append(game.get())
        game.put(fields)
        ours["win_loss"].append(binding.game_step(game.h, acts))
        ours["step"].append(game.get())
        placement.put(fields)
        ours["placement"].append(binding.game_step(placement.h, acts))
        game.put(fields)
        binding.observe(game.h)
        ours["obs"].append(game.obs.copy())
    return dict(pre=pre, actions=actions, ref=ref, ours=ours, board_size=game.board_size)


def assert_same(ours: list[dict], theirs: State, names, b: dict, config: GameConfig, what: str):
    for name in names:
        a = np.stack([d[name] for d in ours])
        t = np.asarray(getattr(theirs, name))
        bad = np.nonzero((a != t).reshape(len(a), -1).any(axis=1))[0]
        if len(bad):
            i = int(bad[0])
            pytest.fail(
                f"{what}: {name} differs in {len(bad)} of {len(a)} states; first (#{i}):"
                f"{describe(b['pre'][i], b['actions'][i], config)}\n"
                f"C: {ours[i][name].tolist()}\nslinky: {t[i].tolist()}"
            )


# --- 1. Rules parity -----------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_rules_step_matches_slinky(case):
    b = bundle(case)
    assert_same(b["ours"]["rules"], b["ref"]["rules"], STATE_FIELDS, b, case.config, "rules_step")
    # The rules never touch the shrink sides.
    sides = np.stack([d["sides"] for d in b["ours"]["rules"]])
    assert np.array_equal(sides, np.stack([d["sides"] for d in b["pre"]]))


def test_rules_cases_cover_the_interesting_events():
    """The transition corpus actually exercises every elimination cause and the freeze."""
    causes, frozen, ate = set(), 0, 0
    for case in CASES:
        b = bundle(case)
        pre, post = to_state(b["pre"], case.config), b["ref"]["rules"]
        died = pre.alive & ~post.alive
        causes |= set(np.asarray(post.elim_cause)[died].tolist())
        frozen += int(np.sum(game_over(pre.alive, case.config)))
        ate += int(np.sum(post.alive & (post.length > pre.length)))
    assert causes == {int(c) for c in Cause if c != Cause.NONE}
    assert frozen > 50 and ate > 100


# --- 2. Full step parity -------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_game_step_matches_slinky(case):
    config = case.config
    b = bundle(case)
    ref, ours = b["ref"], b["ours"]
    exact = [f for f in STATE_FIELDS if f != "food"]
    assert_same(ours["step"], ref["step"], exact, b, config, "game_step")
    np.testing.assert_array_equal(np.stack(ours["win_loss"]), ref["win_loss"])
    # XLA turns slinky's division by N - 1 into a multiplication by its reciprocal.
    np.testing.assert_allclose(np.stack(ours["placement"]), ref["placement"], rtol=1e-6, atol=0)
    pre = to_state(b["pre"], config)
    alive = np.stack([d["alive"] for d in ours["step"]])
    done = np.stack([d["done"] for d in ours["step"]])
    np.testing.assert_array_equal(done & ~game_over(alive, config) & ~pre.done, ref["truncated"])
    if uses_sides(config):
        # The zone of the turn now shown, from the game's sides; the royale ruleset skips it
        # on the game-over call (no royale map to redo it).
        zones = np.asarray(
            jax.jit(jax.vmap(lambda s, t: royale_hazards(s, t, config)))(
                np.stack([d["sides"] for d in ours["step"]]),
                np.stack([d["turn"] for d in ours["step"]]),
            )
        )
        regenerated = ~pre.done & ((config.map == "royale") | ~game_over(pre.alive, config))
        hazard = np.stack([d["hazard"] for d in ours["step"]])
        np.testing.assert_array_equal(hazard[regenerated], zones[regenerated])
        assert regenerated.sum() > 100

    # Food: what the rules left, plus new food only on spawn-mask cells, as much as
    # spawn_food_standard adds (minimum_food top-up, else at most one).
    rules_food, spawn = np.asarray(ref["rules"].food), np.asarray(ref["spawn"])
    max_n = min(max(config.minimum_food, 1), config.width * config.height)
    for i, d in enumerate(ours["step"]):
        if pre.done[i]:
            assert np.array_equal(d["food"], b["pre"][i]["food"])
            continue
        context = describe(b["pre"][i], b["actions"][i], config)
        assert np.all(d["food"] >= rules_food[i]), f"food removed by the map{context}"
        new = d["food"] & ~rules_food[i]
        assert not np.any(new & ~spawn[i]), f"food spawned off the spawn mask{context}"
        count, room, spawned = int(rules_food[i].sum()), int(spawn[i].sum()), int(new.sum())
        if config.map == "empty":
            assert spawned == 0
        elif count < config.minimum_food:
            assert spawned == min(config.minimum_food - count, room, max_n), context
        else:
            assert spawned <= min(1, room), context


def test_finished_game_steps_are_no_ops():
    config = GameConfig(num_snakes=3)
    game = CGame(config)
    fields = put_fields(board([[[0, 0], [0, 1], [0, 2]]] * 3, config, turn=40), config)
    fields["alive"] = np.array([True, False, False])
    fields["done"] = np.bool_(True)
    game.put(fields)
    rewards = binding.game_step(game.h, [UP, UP, UP])
    assert rewards.dtype == np.float32 and not rewards.any()
    after = game.get()
    for f in fields:
        np.testing.assert_array_equal(after[f], fields[f])


def test_rewards_and_truncation_by_hand():
    """Hand-built endings: win, draw, solo death, placement ties, truncation."""

    def step(config, snakes, actions, reward="win_loss", **kw):
        game = CGame(config, reward=reward)
        game.put(put_fields(board(snakes, config, **kw), config))
        return binding.game_step(game.h, actions).tolist(), game.get()

    duel = GameConfig()
    # Snake 0 walks into the wall; snake 1 wins.
    r, d = step(duel, [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]], [LEFT, UP])
    assert r == [-1.0, 1.0] and bool(d["done"]) and int(d["turn"]) == 11
    assert d["elim_cause"].tolist() == [Cause.OUT_OF_BOUNDS, 0] and d["elim_turn"][0] == 11
    # Head-to-head with equal lengths: both die, a draw.
    r, d = step(duel, [[[4, 5], [3, 5], [2, 5]], [[6, 5], [7, 5], [8, 5]]], [RIGHT, LEFT])
    assert r == [0.0, 0.0] and not d["alive"].any() and bool(d["done"])
    # Solo: dying is -1, and only the last death ends the game.
    solo = GameConfig(num_snakes=2, ruleset="solo")
    r, d = step(solo, [[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]], [LEFT, UP])
    assert r == [-1.0, 0.0] and not bool(d["done"])
    # Placement: two of four die on one turn and tie; the survivors outlast both.
    four = GameConfig(num_snakes=4)
    snakes = [
        [[0, 5], [1, 5], [2, 5]],
        [[10, 2], [9, 2], [8, 2]],
        [[5, 9], [5, 8], [5, 7]],
        [[3, 2], [3, 1], [3, 0]],
    ]
    r, _ = step(four, snakes, [LEFT, RIGHT, UP, UP], reward="placement")
    np.testing.assert_allclose(r, [-2 / 3, -2 / 3, 2 / 3, 2 / 3], rtol=1e-6)
    # Truncation by max_turns: survivors get 0 under win_loss.
    capped = GameConfig(max_turns=11)
    r, d = step(capped, [[[4, 4], [4, 3], [4, 2]], [[8, 8], [8, 7], [8, 6]]], [UP, UP])
    assert r == [0.0, 0.0] and bool(d["done"]) and d["alive"].all()


# --- 3. Observations and masks -------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_observations_match_slinky(case):
    config = case.config
    b = bundle(case)
    rows, cols, channels = obs_format.board_shape(
        config.width, config.height, config.ruleset.wrapped
    )
    obs = np.stack(b["ours"]["obs"])  # [B, N, obs_size]
    raw = obs[..., : b["board_size"]].reshape(*obs.shape[:2], rows, cols, channels)
    mask = obs[..., b["board_size"] :]
    alive = np.stack([d["alive"] for d in b["pre"]])
    scale = obs_format.channel_scale(config.width, config.height)
    ref_obs = np.asarray(b["ref"]["obs"])
    assert ref_obs.shape[2:] == (rows, cols, channels)
    np.testing.assert_allclose(
        raw[alive].astype(np.float32) * scale, ref_obs[alive], rtol=1e-6, atol=0
    )
    # The inverse map the policy tests use: round(slinky / scale) is the C board.
    np.testing.assert_array_equal(np.rint(ref_obs[alive] / scale), raw[alive])
    np.testing.assert_array_equal(mask.astype(bool), np.asarray(b["ref"]["mask"]))
    assert not raw[~alive].any() and (mask[~alive] == 1).all()


def test_observation_channels_by_hand():
    """A readable spot check of the raw layout: rows from the bottom, channels last."""
    config = GameConfig()
    game = CGame(config)
    state = board(
        [[[5, 5], [5, 4], [5, 3], [5, 3]], [[7, 5], [8, 5], [9, 5]]],
        config,
        food=[(6, 5)],
        health=[42, 77],
    )
    game.put(put_fields(state, config))
    binding.observe(game.h)
    rows, cols, ch = obs_format.board_shape(11, 11)
    me = game.obs[0, : game.board_size].reshape(rows, cols, ch)
    c = obs_format.CHANNEL_NAMES.index
    centre = (10, 10)  # own head at [H-1, W-1]
    assert me[centre][c("own_head")] == 1 and me[centre][c("own_countdown")] == 4
    assert me[centre][c("own_health")] == 42 and me[centre][c("own_length")] == 4
    assert me[10, 11][c("food")] == 1  # (6, 5): one column right
    assert me[10, 12][c("opp_head_lt")] == 1 and me[10, 12][c("opp_head_health")] == 77
    assert me[9, 10][c("own_countdown")] == 3 and me[8, 10][c("own_countdown")] == 2
    assert me[0, 0][c("on_board")] == 0 and me[5, 5][c("on_board")] == 1
    assert game.obs[0, game.board_size :].tolist() == [1, 0, 1, 1]  # down is its own body


# --- 4. New games --------------------------------------------------------------------------


def new_games(config: GameConfig, count: int, seed: int) -> list[dict]:
    game = CGame(config)
    binding.seed(game.h, seed)
    out = []
    for _ in range(count):
        binding.new_game(game.h)
        out.append(game.get())
    return out


def chi2_z(counts) -> float:
    """z-score of the chi-square statistic of ``counts`` against uniform (Wilson-Hilferty).

    One-sided: only large statistics (a misfit) are large z.
    """
    counts = np.asarray(counts, float)
    k = len(counts) - 1
    expected = counts.sum() / len(counts)
    x = float(((counts - expected) ** 2).sum() / expected)
    return ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))


def binomial_z(hits: int, n: int, p: float) -> float:
    return (hits - n * p) / math.sqrt(n * p * (1 - p))


SETUP_CONFIGS = [
    GameConfig(),
    GameConfig(num_snakes=4),
    GameConfig(num_snakes=6),
    GameConfig(width=7, height=7, num_snakes=4),
    GameConfig(width=7, height=7, num_snakes=6),
    GameConfig(width=9, height=9, num_snakes=9),
    GameConfig(num_snakes=12),
    GameConfig(width=19, height=19, num_snakes=16),
    GameConfig(width=7, height=9, num_snakes=3),
    GameConfig(width=1, height=8, num_snakes=2),
    GameConfig(width=32, height=3, num_snakes=16),
    GameConfig(num_snakes=4, ruleset="constrictor"),
    GameConfig(width=6, height=6, num_snakes=4, ruleset="wrapped_constrictor"),
    GameConfig(num_snakes=4, map="empty"),
    GameConfig(num_snakes=4, ruleset="royale", map="royale"),
    GameConfig(width=7, height=9, num_snakes=3, ruleset="royale"),
    GameConfig(num_snakes=1),
]


def _setup_id(c: GameConfig) -> str:
    return f"{c.ruleset.value}_{c.map}_{c.width}x{c.height}_{c.num_snakes}p"


@functools.cache
def _candidates(config: GameConfig):
    return jax.jit(jax.vmap(jax.vmap(lambda h: _fixed_food_candidates(h, config))))


def food_candidates(games: list[dict], config: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """slinky's ``_fixed_food_candidates`` for every head of every game: (xy, allowed)."""
    heads = np.stack([d["head"] for d in games])
    return jax.device_get(_candidates(config)(heads))


def fixed_food_components(xy: np.ndarray, allowed: np.ndarray):
    """The exact law of ``place_food_fixed``'s per-snake food for one game's candidates.

    Snakes whose allowed cells overlap form components, processed in snake order with the
    "already food" exclusion: ``[(snakes, cells, {food subset: probability}), ...]``.
    """
    n = len(xy)
    options = [[tuple(map(int, xy[i, k])) for k in range(4) if allowed[i, k]] for i in range(n)]
    comps: list[list[int]] = []
    for i in range(n):
        joined = [c for c in comps if any(set(options[i]) & set(options[j]) for j in c)]
        merged = sorted([i, *[j for c in joined for j in c]])
        comps = [c for c in comps if c not in joined] + [merged]
    result = []
    for comp in comps:
        law: dict[frozenset, float] = {}

        def walk(k, taken, p, comp=comp, law=law):
            if k == len(comp):
                law[taken] = law.get(taken, 0.0) + p
                return
            opts = [c for c in options[comp[k]] if c not in taken]
            if not opts:
                walk(k + 1, taken, p)
            for c in opts:
                walk(k + 1, taken | {c}, p / len(opts))

        walk(0, frozenset(), 1.0)
        result.append((comp, set().union(*(set(options[i]) for i in comp)), law))
    return result


def assert_marginals(observed: np.ndarray, expected: np.ndarray):
    """Independent games, cell indicators with known probabilities: per-cell z-scores."""
    o, p = observed.sum(axis=0), expected.sum(axis=0)
    var = (expected * (1 - expected)).sum(axis=0)
    assert np.all(observed[expected == 0] == 0), "food where it can never be"
    assert np.all(observed[expected == 1] == 1), "no food where it must be"
    z = np.where(var > 0, (o - p) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    assert np.abs(z).max() < Z_MAX, (np.abs(z).max(), np.unravel_index(np.abs(z).argmax(), z.shape))


@pytest.mark.parametrize("config", SETUP_CONFIGS, ids=_setup_id)
def test_new_games_are_legal(config):
    count = 300
    games = new_games(config, count, seed=1)
    n, w, h = config.num_snakes, config.width, config.height
    kind = placement_kind(config)
    corners, cardinals = (set(map(tuple, p.tolist())) for p in _fixed_start_points(config))
    centre = ((w - 1) // 2, (h - 1) // 2)
    random_food = not (w == h and w >= 7 or config.ruleset.constrictor or config.map == "empty")
    if w == h and w >= 7:
        xy, allowed = food_candidates(games, config)
    if random_food:  # place_food_random uses the spawn mask of the board without food
        states = to_state(games, config)
        masks = np.asarray(
            jax.jit(jax.vmap(lambda s: spawn_mask(s, config)))(
                states._replace(food=np.zeros_like(states.food))
            )
        )
    for g, d in enumerate(games):
        heads = [tuple(p) for p in d["head"].tolist()]
        assert len(set(heads)) == n
        for i, (x, y) in enumerate(heads):
            expect = np.zeros((h, w), np.int16)
            expect[y, x] = 3
            assert np.array_equal(d["body"][i], expect)
        assert d["length"].tolist() == [3] * n and d["health"].tolist() == [100] * n
        assert d["alive"].all() and not d["last_move"].any() and int(d["turn"]) == 0
        assert not d["elim_cause"].any() and not d["elim_turn"].any() and not d["hazard"].any()
        assert not bool(d["done"])
        if uses_sides(config):
            assert d["sides"].min() >= 0 and d["sides"].max() <= 3
        else:
            assert not d["sides"].any()
        if kind == "fixed":
            first = set(heads[: min(n, 4)])
            assert first <= corners or first <= cardinals
            assert set(heads) <= corners | cardinals
        elif kind == "distributed":
            quads = [set(map(tuple, q.tolist())) for q in _quadrant_points(config)]
            q0 = next(q for q in range(4) if heads[0] in quads[q])
            assert all(heads[i] in quads[(q0 + i) % 4] for i in range(n))
        else:
            valid = _random_start_cells(config)
            assert all(valid[y, x] for x, y in heads)
        food = {(int(x), int(y)) for y, x in zip(*np.nonzero(d["food"]), strict=True)}
        if config.ruleset.constrictor or config.map == "empty":
            assert not food
        elif w == h and w >= 7:
            assert centre in food
            if n <= 4 or w * h >= 121:
                rest = set(food) - {centre}
                for _, cells, law in fixed_food_components(xy[g], allowed[g]):
                    assert law.get(frozenset(rest & cells), 0.0) > 0, (heads, food)
                    rest -= cells
                assert not rest
            else:
                assert food == {centre}
        else:
            room = int(masks[g].sum())
            assert len(food) == min(n, w * h, room)
            assert all(masks[g][y, x] for x, y in food)


def test_fixed_placement_distribution():
    """Corners or cardinals first 50/50, each group shuffled uniformly."""
    config = GameConfig()
    games = new_games(config, 25000, seed=2)  # enough to see a 52/48 coin
    points = [tuple(p) for p in np.concatenate(_fixed_start_points(config)).tolist()]
    first = [points.index(tuple(d["head"][0].tolist())) for d in games]
    pairs = [(points.index(tuple(d["head"][0])), points.index(tuple(d["head"][1]))) for d in games]
    assert chi2_z(np.bincount(first, minlength=8)) < Z_MAX
    combos = sorted({(a, b) for a in range(8) for b in range(8) if a != b and (a < 4) == (b < 4)})
    assert set(pairs) <= set(combos)
    assert chi2_z([pairs.count(c) for c in combos]) < Z_MAX
    corners_first = sum(i < 4 for i in first)
    assert abs(binomial_z(corners_first, len(games), 0.5)) < Z_MAX


def test_distributed_placement_distribution():
    config = GameConfig(num_snakes=12)
    games = new_games(config, 6000, seed=3)
    quads = [[tuple(p) for p in q.tolist()] for q in _quadrant_points(config)]
    flat = [p for q in quads for p in q]
    first = [flat.index(tuple(d["head"][0].tolist())) for d in games]
    assert chi2_z(np.bincount(first, minlength=16)) < Z_MAX
    # Snakes 0 and 4 share a quadrant: an ordered pair of its points, uniformly.
    pair = [(flat.index(tuple(d["head"][0])), flat.index(tuple(d["head"][4]))) for d in games]
    combos = sorted({(a, b) for a in range(16) for b in range(16) if a // 4 == b // 4 and a != b})
    assert set(pair) <= set(combos)
    assert chi2_z([pair.count(c) for c in combos]) < Z_MAX


def test_random_placement_distribution():
    config = GameConfig(width=7, height=9, num_snakes=3)
    games = new_games(config, 6000, seed=4)
    valid = _random_start_cells(config)
    cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(valid), strict=True)]
    for i in range(3):
        idx = [cells.index(tuple(d["head"][i].tolist())) for d in games]
        assert chi2_z(np.bincount(idx, minlength=len(cells))) < Z_MAX


@pytest.mark.parametrize(
    "config",
    [GameConfig(), GameConfig(num_snakes=12), GameConfig(width=19, height=19, num_snakes=16)],
    ids=_setup_id,
)
def test_fixed_food_distribution(config):
    """Uniform among each snake's allowed cells, sequentially excluding food already placed
    (11x11 with 12 snakes has neighbours that share a candidate cell)."""
    games = new_games(config, 3000, seed=5)
    h, w = config.height, config.width
    observed = np.stack([d["food"] for d in games]).astype(float)
    expected = np.zeros_like(observed)
    shared = 0
    xy, allowed = food_candidates(games, config)
    for g in range(len(games)):
        expected[g, (h - 1) // 2, (w - 1) // 2] = 1.0
        for comp, _, law in fixed_food_components(xy[g], allowed[g]):
            shared += len(comp) > 1
            for subset, p in law.items():
                for x, y in subset:
                    expected[g, y, x] += p
    if config.num_snakes == 12:
        assert shared > 0, "no overlapping candidates: the exclusion is not exercised"
    assert_marginals(observed, expected)


def test_random_initial_food_distribution():
    config = GameConfig(width=7, height=9, num_snakes=3)
    games = new_games(config, 4000, seed=6)
    states = to_state(games, config)
    masks = np.asarray(
        jax.jit(jax.vmap(lambda s: spawn_mask(s, config)))(
            states._replace(food=np.zeros_like(states.food))
        )
    )
    room = masks.reshape(len(games), -1).sum(axis=1)
    expected = masks * (np.minimum(3, room) / np.maximum(room, 1))[:, None, None]
    assert_marginals(np.stack([d["food"] for d in games]).astype(float), expected)


def test_royale_sides_distribution():
    config = GameConfig(num_snakes=4, ruleset="royale", map="royale")
    games = new_games(config, 2000, seed=7)
    sides = np.stack([d["sides"] for d in games]).astype(int)
    assert sides.shape == (2000, royale_num_sides(config))
    assert chi2_z(np.bincount(sides.ravel(), minlength=4)) < Z_MAX
    pairs = (4 * sides[:, :-1] + sides[:, 1:]).ravel()
    assert chi2_z(np.bincount(pairs, minlength=16)) < Z_MAX
    assert chi2_z(np.bincount(sides[:, 0], minlength=4)) < Z_MAX


# Food spawning, from one fixed state stepped many times: the rules part is deterministic.
SPAWN_SNAKES = [[[3, 3], [3, 2], [3, 1]], [[7, 7], [7, 6], [7, 5]]]


@functools.cache
def _rules_and_spawn_mask(config: GameConfig):
    def f(state, actions):
        after = rules_step(state, actions, config)
        return after.food, spawn_mask(after, config)

    return jax.jit(f, compiler_options=FAST_COMPILE)


def spawn_samples(config: GameConfig, food, count: int, seed: int):
    state = board(SPAWN_SNAKES, config, food=food)
    fields = put_fields(state, config)
    # Neither depends on the food settings, so every spawn test shares one compilation.
    rules_config = dataclasses.replace(config, food_spawn_chance=15, minimum_food=1)
    rules_food, mask = jax.device_get(
        _rules_and_spawn_mask(rules_config)(state, jnp.array([UP, UP]))
    )
    game = CGame(config)
    binding.seed(game.h, seed)
    new = []
    for _ in range(count):
        game.put(fields)
        binding.game_step(game.h, [UP, UP])
        new.append(game.get()["food"] & ~rules_food)
    return np.stack(new), mask


def test_food_spawn_cells_are_uniform_on_the_spawn_mask():
    config = GameConfig(food_spawn_chance=101)  # the roll always passes
    new, mask = spawn_samples(config, food=[(0, 0)], count=20000, seed=8)
    assert (new.reshape(len(new), -1).sum(axis=1) == 1).all()
    assert not (new & ~mask).any()
    assert chi2_z(new.sum(axis=0)[mask]) < Z_MAX


@pytest.mark.parametrize("chance,p", [(15, 0.14), (50, 0.49), (1, 0.0), (0, 0.0), (100, 0.99)])
def test_food_spawn_probability(chance, p):
    """The engine's 100 - rand.Intn(100) < chance: probability (chance - 1) / 100."""
    config = GameConfig(food_spawn_chance=chance)
    new, _ = spawn_samples(config, food=[(0, 0)], count=20000, seed=9)
    hits = int(new.any(axis=(1, 2)).sum())
    if p == 0.0:
        assert hits == 0
    else:
        assert abs(binomial_z(hits, len(new), p)) < Z_MAX, hits / len(new)


def test_minimum_food_top_up_is_uniform():
    config = GameConfig(minimum_food=3, food_spawn_chance=0)
    new, mask = spawn_samples(config, food=[], count=10000, seed=10)
    assert (new.reshape(len(new), -1).sum(axis=1) == 3).all()
    assert not (new & ~mask).any()
    room = int(mask.sum())
    assert_marginals(new.astype(float), np.broadcast_to(mask * (3 / room), new.shape))


def test_seed_reproducibility():
    config = GameConfig(num_snakes=4, ruleset="royale", map="royale")
    a, b = new_games(config, 5, seed=11), new_games(config, 5, seed=11)
    c = new_games(config, 5, seed=12)
    for x, y in zip(a, b, strict=True):
        for f in x:
            np.testing.assert_array_equal(x[f], y[f])
    assert any(not np.array_equal(x["head"], z["head"]) for x, z in zip(a, c, strict=True))


# --- 5. Agent slots through Battlesnake.step -----------------------------------------------


def put_env(env, state: State, config: GameConfig, game: int = 0) -> None:
    binding.env_put(env.handles[game], **put_fields(jax.device_get(state), config))


@functools.cache
def _obs_fn(config: GameConfig):
    return jax.jit(make_obs_fn(config))


def slinky_obs(fields: dict, config: GameConfig) -> np.ndarray:
    return np.asarray(_obs_fn(config)(unbatch(to_state([fields], config), 0)))


def raw_board(env, agent: int) -> np.ndarray:
    return env.observations[agent, : env.obs_size - 4].reshape(env.board_shape)


def test_death_and_win_reset_the_game():
    env = battlesnake.Battlesnake(num_envs=2, mode="duel", report_interval=1)
    env.reset(0)
    config = GameConfig(**env.game_config)
    put_env(
        env, board([[[0, 5], [1, 5], [2, 5]], [[8, 8], [8, 7], [8, 6]]], config, turn=30), config
    )
    other = binding.env_get(env.handles[1])
    _, rewards, terminals, _, infos = env.step(np.array([LEFT, UP, UP, UP]))
    assert rewards[:2].tolist() == [-1.0, 1.0] and terminals[:2].tolist() == [True, True]
    assert infos and infos[0]["n"] == 1 and infos[0]["deaths_wall"] == 1
    assert infos[0]["episode_length"] == 31 and infos[0]["draws"] == 0
    # Autoreset: game 0 is a new game and its observations are the new game's.
    fresh = binding.env_get(env.handles[0])
    assert int(fresh["turn"]) == 0 and fresh["alive"].all()
    scale = env.channel_scale
    for i in range(2):
        np.testing.assert_allclose(
            raw_board(env, i) * scale, slinky_obs(fresh, config)[i], rtol=1e-6, atol=0
        )
    snapshot = env.observations.copy()
    binding.observe(env.handles[0])
    np.testing.assert_array_equal(env.observations, snapshot)
    # The other game just moved on.
    assert int(binding.env_get(env.handles[1])["turn"]) == int(other["turn"]) + 1
    assert not terminals[2:].any() and not rewards[2:].any()


def test_waiting_slots_of_dead_snakes():
    env = battlesnake.Battlesnake(num_envs=1, mode="standard", report_interval=1)
    env.reset(0)
    config = GameConfig(**env.game_config)
    snakes = [
        [[0, 5], [1, 5], [2, 5]],
        [[8, 8], [8, 7], [8, 6]],
        [[5, 2], [5, 1], [5, 0]],
        [[2, 9], [2, 8], [2, 7]],
    ]
    put_env(env, board(snakes, config), config)
    _, rewards, terminals, _, infos = env.step(np.array([LEFT, UP, UP, UP]))
    assert rewards.tolist() == [-1, 0, 0, 0] and terminals.tolist() == [1, 0, 0, 0]
    assert infos == [{}]  # nothing finished yet
    assert not raw_board(env, 0).any() and env.observations[0, -4:].tolist() == [1, 1, 1, 1]
    for _ in range(3):  # the dead slot keeps one-step episodes with reward 0
        _, rewards, terminals, _, _ = env.step(np.array([UP, LEFT, UP, RIGHT]))
        assert rewards.tolist() == [0, 0, 0, 0] and terminals.tolist() == [1, 0, 0, 0]
        assert not raw_board(env, 0).any()
    state = binding.env_get(env.handles[0])
    np.testing.assert_allclose(
        raw_board(env, 1) * env.channel_scale, slinky_obs(state, config)[1], rtol=1e-6, atol=0
    )


def test_draw_and_placement_rewards_through_step():
    env = battlesnake.Battlesnake(num_envs=1, num_snakes=4, reward="placement", report_interval=1)
    env.reset(0)
    config = GameConfig(**env.game_config)
    snakes = [
        [[4, 5], [3, 5], [2, 5]],
        [[6, 5], [7, 5], [8, 5]],
        [[5, 9], [5, 8], [5, 7]],
        [[5, 1], [5, 0], [6, 0]],
    ]
    state = board(snakes, config)
    put_env(env, state._replace(alive=jnp.array([True, True, False, False])), config)
    _, rewards, terminals, _, infos = env.step(np.array([RIGHT, LEFT, UP, UP]))
    # Snakes 2 and 3 were already dead (waiting slots); 0 and 1 meet head-on: a draw.
    assert rewards.tolist() == [0, 0, 0, 0] and terminals.all()
    log = infos[0]
    assert log["draws"] == 1 and log["deaths_head"] == 2 and log["n"] == 1


def test_truncation_is_terminal():
    env = battlesnake.Battlesnake(num_envs=1, max_turns=20, report_interval=1)
    env.reset(0)
    config = GameConfig(**env.game_config)
    put_env(
        env, board([[[4, 4], [4, 3], [4, 2]], [[8, 8], [8, 7], [8, 6]]], config, turn=19), config
    )
    _, rewards, terminals, truncations, infos = env.step(np.array([UP, UP]))
    assert terminals.all() and not rewards.any() and not truncations.any()
    assert infos[0]["truncated"] == 1 and infos[0]["episode_length"] == 20
    assert int(binding.env_get(env.handles[0])["turn"]) == 0


def test_solo_game_through_step():
    env = battlesnake.Battlesnake(num_envs=1, width=7, height=7, num_snakes=1, report_interval=1)
    env.reset(0)
    config = GameConfig(**env.game_config)
    put_env(env, board([[[0, 3], [1, 3], [2, 3]]], config), config)
    _, rewards, terminals, _, infos = env.step(np.array([LEFT]))
    assert rewards.tolist() == [-1.0] and terminals.tolist() == [True]
    assert infos[0]["deaths_wall"] == 1 and infos[0]["draws"] == 1


def test_random_play_logs():
    env = battlesnake.Battlesnake(num_envs=64, mode="standard", report_interval=50)
    env.reset(1)
    rng = np.random.default_rng(0)
    logs, deaths, finished = [], 0, 0
    for _ in range(200):
        _, rewards, terminals, _, infos = env.step(rng.integers(0, 4, env.num_agents))
        assert set(np.unique(rewards)) <= {-1.0, 0.0, 1.0}
        logs += [i for i in infos if i]
        deaths += int((rewards == -1).sum())
        finished += int(terminals.reshape(64, 4).all(axis=1).sum())
    keys = {"n", "episode_length", "draws", "truncated"} | {
        f"deaths_{c}" for c in ("wall", "self", "body", "head", "starve", "hazard")
    }
    assert logs and all(set(log) == keys for log in logs)
    for log in logs:
        # Each game ends with at most one survivor: deaths = snakes - 1 + draws.
        total = sum(log[k] for k in keys if k.startswith("deaths_"))
        assert total == pytest.approx(4 - 1 + log["draws"], abs=1e-4)
        assert log["truncated"] == 0 and log["episode_length"] > 1
    assert finished > 100 and deaths > 300


def test_reset_seeds_and_compute_obs():
    a = battlesnake.Battlesnake(num_envs=8, mode="royale")
    b = battlesnake.Battlesnake(num_envs=8, mode="royale")
    obs_a = a.reset(5)[0].copy()
    assert np.array_equal(obs_a, b.reset(5)[0])
    assert not np.array_equal(obs_a, b.reset(6)[0])
    raw = battlesnake.Battlesnake(num_envs=8, mode="royale", compute_obs=False)
    first = raw.reset(5)[0].copy()
    assert np.array_equal(first, obs_a)  # reset always observes
    raw.step(np.zeros(raw.num_agents, np.int32))
    assert np.array_equal(raw.observations, first)
    for env in (a, b, raw):
        env.close()


def test_env_attributes():
    env = battlesnake.Battlesnake(num_envs=3, width=10, height=8, num_snakes=3, ruleset="wrapped")
    assert env.num_agents == 9 and env.wrapped
    assert env.board_shape == (8, 10, 13) == obs_format.board_shape(10, 8, True)
    assert env.single_observation_space.shape == (env.obs_size,)
    assert env.observations.shape == (9, env.obs_size) and env.observations.dtype == np.uint8
    assert env.single_action_space.n == 4
    np.testing.assert_array_equal(env.channel_scale, obs_format.channel_scale(10, 8))
    assert GameConfig(**env.game_config) == GameConfig(
        width=10, height=8, num_snakes=3, ruleset="wrapped"
    )


VECTOR_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import numpy as np
import pufferlib.vector
from puffer import battlesnake, obs_format

kwargs = dict(num_workers=2, batch_size=2) if {backend!r} == "Multiprocessing" else {{}}
vecenv = pufferlib.vector.make(
    battlesnake.env_creator("battlesnake_duel"),
    env_kwargs=dict(num_envs=4),
    backend=getattr(pufferlib.vector, {backend!r}),
    num_envs=2,
    seed=0,
    **kwargs,
)
vecenv.async_reset(0)
obs = vecenv.recv()[0]
assert obs.shape == (16, obs_format.obs_size(11, 11)), obs.shape
rng = np.random.default_rng(0)
ended = 0
for _ in range(200):
    vecenv.send(rng.integers(0, 4, 16).astype(np.int32))
    obs, rewards, terminals = vecenv.recv()[:3]
    ended += int(terminals.sum())
assert ended > 0 and obs[:, -4:].any()
vecenv.close()
assert "jax" not in sys.modules
print("ok")
"""


@pytest.mark.parametrize("backend", ["Serial", "Multiprocessing"])
def test_pufferlib_vector_backends(backend, tmp_path):
    """``pufferlib.vector.make`` with the env creator, in a fresh interpreter: forking this
    process (JAX threads) is unsafe, and the env module must not need JAX."""
    script = VECTOR_SCRIPT.format(root=str(ROOT), backend=backend)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,  # pufferlib links `resources` into the working directory
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-3000:]


def test_render_matches_slinky(capfd):
    config = GameConfig(width=12, height=10, num_snakes=3)
    state = board(
        [[[0, 0], [0, 1], [1, 1]], [[5, 5], [5, 6], [5, 7]], [[9, 9], [9, 8], [9, 7]]],
        config,
        food=[(3, 3), (11, 9)],
        causes=[("", 0), ("", 0), ("wall-collision", 7)],
    )
    state = state._replace(hazard=state.hazard.at[2, 4].set(2).at[3, 3].set(1))
    game = CGame(config)
    game.put(put_fields(jax.device_get(state), config))
    capfd.readouterr()
    binding.env_render(game.h)
    assert capfd.readouterr().out == render(state) + "\n"


# --- Validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=6, height=6, num_snakes=9),  # too many snakes for the board
        dict(width=1, height=1, num_snakes=1),  # no room: the only cell is the centre
        dict(num_snakes=17),
        dict(num_snakes=0),
        dict(width=33),
        dict(shrink_every_n_turns=0),
        dict(ruleset="chess"),
        dict(map="arcade_maze"),
        dict(reward="score"),
        dict(max_turns=-1),
    ],
    ids=str,
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        battlesnake.Battlesnake(num_envs=1, **kwargs)
    # The binding rejects them too, without the wrapper.
    full = dict(c_kwargs(GameConfig()), **kwargs)
    n = max(full["num_snakes"], 1)
    bufs = [np.zeros((n, 8), np.uint8), np.zeros(n, np.int32), np.zeros(n, np.float32)]
    bufs += [np.zeros(n, bool), np.zeros(n, bool)]
    with pytest.raises(ValueError):
        binding.env_init(*bufs, 0, **full)


def test_env_put_validates():
    config = GameConfig()
    game = CGame(config)
    good = game.get()
    game.put(good)
    for f, bad, err in [
        ("body", good["body"].astype(np.int32), TypeError),
        ("body", good["body"][:1], ValueError),
        ("alive", good["alive"].astype(np.uint8), TypeError),
        ("sides", np.zeros(3, np.int8), ValueError),
        ("turn", np.float32(3), TypeError),
        ("last_move", np.array([0, 7], np.int8), ValueError),
        ("head", good["head"].tolist(), TypeError),
    ]:
        with pytest.raises(err):
            game.put(dict(good, **{f: bad}))
    with pytest.raises(TypeError):
        game.put({k: v for k, v in good.items() if k != "food"})
    with pytest.raises(TypeError):
        game.put(dict(good, extra=np.zeros(1)))
    after = game.get()
    for f in good:  # a rejected put changes nothing
        np.testing.assert_array_equal(after[f], good[f])
    game.put(dict(good, turn=7, done=True))  # Python scalars are fine
    assert int(game.get()["turn"]) == 7 and bool(game.get()["done"])


# --- 6. Engine oracle ----------------------------------------------------------------------


def _parity():
    try:
        import test_parity
    except ImportError:
        from tests import test_parity
    return test_parity


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name,flags", _parity().GAME_CONFIGS, ids=[c[0] for c in _parity().GAME_CONFIGS]
)
def test_engine_transitions(name, flags, oracle_bin, oracle):
    """Engine-generated ``pre -> post_rules`` transitions, reproduced by the C rules step."""
    tp = _parity()
    config = tp._game_config(flags)
    games = tp.split_games(tp.generate_games(binary=oracle_bin, seed=1000, **flags))
    sides = tp._map_states(oracle, games, config)
    # The rules pipeline does not depend on the map (hazard maps only place hazards).
    game = CGame(config, map=config.map if config.map in battlesnake.MAPS else "standard")
    trans = [t for g in games for t in g["transitions"]]
    assert trans
    ours, post = [], []
    for k, t in enumerate(trans):
        game_sides = sides[t["game"]]
        raw_sides = np.asarray(game_sides.sides, np.int8) if game_sides else None
        pre = engine_fields(t["pre"], config, raw_sides)
        post.append(engine_fields(t["post_rules"], config, raw_sides))
        if k % 500 == 0:  # the fast conversion is state_from_engine's
            for d, s in ((pre, t["pre"]), (post[-1], t["post_rules"])):
                reference = put_fields(
                    jax.device_get(state_from_engine(s, config, game_sides)), config
                )
                assert all(np.array_equal(d[f], v) for f, v in reference.items())
        game.put(pre)
        binding.rules_step(game.h, moves_to_ids(t["moves"]))
        ours.append(game.get())

    def context(i, engine=False):
        t = trans[i]
        if engine:
            return t["post_rules"]
        return f"game {t['game']} turn {t['pre']['turn']} moves {t['moves']}\npre: {t['pre']}"

    compare_states(to_state(ours, config), to_state(post, config), config, context)


# --- 7. Modes ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_mode_table_matches_slinky(mode):
    env = battlesnake.Battlesnake(num_envs=1, mode=mode)
    assert GameConfig(**env.game_config) == mode_config(mode)
    for n in range(1, 17):
        try:
            expected = mode_config(mode, num_snakes=n)
        except ValueError:
            with pytest.raises(ValueError):
                battlesnake.resolve_game(mode, n, None, None)
            continue
        ruleset, map_name, snakes = battlesnake.resolve_game(mode, n, None, None)
        assert (ruleset, map_name, snakes) == (expected.ruleset.value, expected.map, n)
    assert set(battlesnake.MODES) == set(MODE_NAMES)
    with pytest.raises(ValueError):
        battlesnake.Battlesnake(num_envs=1, mode=mode, ruleset="standard")
    with pytest.raises(ValueError):
        battlesnake.Battlesnake(num_envs=1, mode=mode, map="standard")


def test_env_creator():
    assert battlesnake.env_creator()().num_snakes == 2
    env = battlesnake.env_creator("puffer_battlesnake_royale")(num_envs=2)
    assert env.game_config["ruleset"] == "royale" and env.num_agents == 8
    with pytest.raises(ValueError):
        battlesnake.env_creator("battlesnake_chess")
