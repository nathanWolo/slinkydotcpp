"""Random-state fuzzing of the rules pipeline against the official engine.

Unlike the engine-generated corpora, states here are drawn directly: tightly
coiled self-avoiding bodies, stacked tails (incl. fully stacked mid-game),
dead snakes (incl. off-board heads), food next to heads and on head-on
squares, stacked hazards with zero/negative/huge damage, boundary health,
odd/even/non-square/1-wide boards, up to 16 snakes, and invalid moves.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
import pytest

from slinky.engine_json import state_from_engine
from slinky.rules import is_game_over, rules_step
from slinky.types import ACTION_DELTAS, ACTION_NAMES, GameConfig

try:
    from oracle import make_step_request
except ImportError:
    from tests.oracle import make_step_request

pytestmark = pytest.mark.oracle

CAUSES = [
    "wall-collision",
    "snake-collision",
    "snake-self-collision",
    "head-collision",
    "out-of-health",
    "hazard",
]


def _nbrs(p, w, h, wrapped):
    out = []
    for dx, dy in ACTION_DELTAS:
        x, y = p[0] + dx, p[1] + dy
        if wrapped:
            x, y = x % w, y % h
        if 0 <= x < w and 0 <= y < h:
            out.append((x, y))
    return out


def random_case(rng, w, h, n, wrapped):
    """A random engine state (living snakes never overlap) and a move list."""
    cells = [(x, y) for x in range(w) for y in range(h)]
    target = cells[rng.integers(len(cells))]  # head-on magnet
    ring = _nbrs(target, w, h, wrapped)
    taken, snakes, heads = set(), [], []
    for i in range(n):
        dead = rng.random() < 0.15
        length = 3 if rng.random() < 0.1 else int(rng.integers(3, max(4, min(w * h, 40))))
        cells_wanted = (
            1 if length == 3 and rng.random() < 0.3 else length - int(rng.choice([0, 0, 1, 2]))
        )
        free = [c for c in cells if dead or c not in taken]
        if not free:
            dead, free = True, cells
        start = (
            ring[i]
            if i < len(ring) and ring[i] in free and rng.random() < 0.5
            else free[rng.integers(len(free))]
        )
        walk, seen = [start], {start}
        while len(walk) < cells_wanted:
            opts = [
                q
                for q in _nbrs(walk[-1], w, h, wrapped)
                if q not in seen and (dead or q not in taken)
            ]
            if not opts:
                break
            if rng.random() < 0.5:  # coil: prefer cells with many occupied neighbours
                score = [
                    sum(r in seen or r in taken for r in _nbrs(q, w, h, wrapped)) for q in opts
                ]
                opts = [q for q, s in zip(opts, score, strict=True) if s == max(score)]
            walk.append(opts[rng.integers(len(opts))])
            seen.add(walk[-1])
        body = walk + [walk[-1]] * (length - len(walk))
        if dead and not wrapped and rng.random() < 0.3:  # head off the board
            body = [(-1, body[0][1])] + body[:-1]
        if not dead:
            taken.update(body)
            heads.append(body[0])
        snakes.append(
            {
                "id": f"s{i}",
                "body": [list(p) for p in body],
                "health": int(rng.choice([1, 1, 2, 14, 15, 16, 99, 100, rng.integers(1, 101)])),
                "eliminated_cause": CAUSES[rng.integers(len(CAUSES))] if dead else "",
                "eliminated_on_turn": 3 if dead else 0,
            }
        )

    def near_head():
        if heads and rng.random() < 0.7:
            nb = _nbrs(heads[rng.integers(len(heads))], w, h, wrapped)
            if nb:
                return nb[rng.integers(len(nb))]
        return cells[rng.integers(len(cells))]

    food = {near_head() for _ in range(rng.integers(0, 2 + 2 * n))} | (
        {target} if rng.random() < 0.3 else set()
    )
    hazards = [list(near_head()) for _ in range(rng.integers(0, 3 + 2 * n))]
    hazards += hazards[: rng.integers(0, len(hazards) + 1)]  # stacked layers
    moves = []
    for s in snakes:
        if s["eliminated_cause"]:
            moves.append("")
        elif rng.random() < 0.08:
            moves.append(str(rng.choice(["", "bogus", "UP"])))
        else:
            hx, hy = s["body"][0]
            to_target = [
                a
                for a, (dx, dy) in enumerate(ACTION_DELTAS)
                if ((hx + dx) % w, (hy + dy) % h) == target
            ]
            a = to_target[0] if to_target and rng.random() < 0.7 else int(rng.integers(4))
            moves.append(ACTION_NAMES[a])
    state = {
        "width": w,
        "height": h,
        "turn": int(rng.integers(0, 300)),
        "snakes": snakes,
        "food": [list(c) for c in sorted(food) if c not in taken],
        "hazards": hazards,
    }
    return state, moves


FUZZ_CONFIGS = [  # (width, height, snakes, ruleset, hazard damage)
    (11, 11, 2, "standard", 14),
    (7, 9, 4, "standard", -7),
    (9, 7, 5, "standard", 100),
    (5, 5, 3, "standard", 0),
    (1, 8, 2, "standard", 14),
    (19, 19, 16, "standard", 30),
    (11, 11, 1, "standard", 14),
    (7, 7, 3, "solo", 14),
    (11, 11, 4, "wrapped", 14),
    (2, 5, 2, "wrapped", 14),
    (7, 7, 4, "constrictor", 14),
    (6, 6, 4, "wrapped_constrictor", 20),
]


@functools.cache
def _step(config):
    return jax.jit(jax.vmap(lambda s, a: rules_step(s, a, config)))


@pytest.mark.parametrize(
    "w,h,n,ruleset,damage", FUZZ_CONFIGS, ids=lambda v: str(v) if not isinstance(v, int) else None
)
def test_random_states_match_engine(w, h, n, ruleset, damage, oracle):
    config = GameConfig(
        width=w, height=h, num_snakes=n, ruleset=ruleset, hazard_damage_per_turn=damage
    )
    rng = np.random.default_rng([w, h, n, damage + 1000])
    cases = [random_case(rng, w, h, n, config.ruleset.wrapped) for _ in range(500)]
    resps = oracle.step_many(
        [
            make_step_request(
                ruleset, d, mv, settings={"damagePerTurn": str(damage)}, solo=config.solo
            )
            for d, mv in cases
        ]
    )
    assert not [r["error"] for r in resps if r["error"]]
    stack = lambda xs: jax.tree.map(lambda *a: np.stack(a), *xs)  # noqa: E731
    pre = stack([state_from_engine(d, config) for d, _ in cases])
    acts = np.array(
        [[ACTION_NAMES.index(m) if m in ACTION_NAMES else -1 for m in mv] for _, mv in cases]
    )
    ours = jax.device_get(_step(config)(pre, acts))
    theirs = stack([state_from_engine(r["state"], config) for r in resps])
    over = np.asarray(jax.vmap(lambda a: is_game_over(a, config))(pre.alive))
    assert over.tolist() == [r["game_over"] for r in resps]
    alive = theirs.alive
    wh = np.array([w, h])
    delta = np.array(ACTION_DELTAS)

    def default_cell(s):  # where an invalid move would go (wrap-equivalent moves agree)
        p = s.head + delta[s.last_move.astype(int)]
        return np.where(alive[..., None], p % wh if config.ruleset.wrapped else p, 0)

    for name, a, b in [
        ("alive", ours.alive, theirs.alive),
        ("elim_cause", ours.elim_cause, theirs.elim_cause),
        ("elim_turn", ours.elim_turn, theirs.elim_turn),
        ("health", ours.health, theirs.health),
        ("head", ours.head, theirs.head),
        ("food", ours.food, theirs.food),
        ("length", ours.length * alive, theirs.length * alive),
        ("body", ours.body * alive[..., None, None], theirs.body * alive[..., None, None]),
        ("default move", default_cell(ours), default_cell(theirs)),
    ]:
        bad = np.nonzero((a != b).reshape(len(cases), -1).any(1))[0]
        assert not len(bad), f"{name} differs in {len(bad)} cases; first: {cases[bad[0]]}"
