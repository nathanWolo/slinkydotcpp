"""Adversarial golden vectors for the Rust rules crate (native/crates/rules/tests/review_rules.rs).

Written by the rules review: hand-built and fuzzed positions aimed at the corners most likely to
break parity with slinky's jitted rules, rather than at typical play. Groups (one directory each):

* ``headon_*``: 2-, 3- and 4-way head-ons on one cell, length combinations in {2, 3, 4} (all of
  them up to 3-way; for 4-way the all-equal ones and a random 24),
  with and without food on the cell, with low health and stacked tails (all 4^N joint actions).
* ``stacked_*``: a snake that ate last turn (stacked tail) next to another head and to its own
  head; chains of two steps (every joint action, then every joint action again) so the second
  step moves into tails that stacked in the first.
* ``neck``: snakes of length 1, 2 and 3 (stacked and not) turning back into their necks.
* ``hazard_*``: health set around the exact-kill boundary (``health - 1 - layers * damage`` in
  -2..2) for damage 1, 14, 50, 100, 101 (clamped), 0 and -15 (healing), 1 to 7 stacked layers,
  with and without food on the hazard cell, and two snakes entering the same hazard cell.
* ``crowd_*``: random crowded small boards (3x3 to 6x6) with every ruleset and 2 to 4 seats,
  random low health, food, stacked hazards, dead seats with random elimination turns, invalid
  action ids, malformed bodies (overlaps, off-board heads), extreme turn/health/length values;
  every joint action, plus random walks of up to 25 steps (search-like chains).
* ``royale_*``: royale with shrink every 1, 2, 3 and 25 turns, turns at shrink boundaries
  (S*k - 1, S*k, S*k + 1), weird side values (negative, 4, > 4), zones that disagree with the
  sides, turns near i32::MAX (turn + 1 wraps), and walks until the zone closes.
* ``constrictor_*``: growth and starvation in constrictor, food at the root, health 1 at the root.
* ``lasttwo``: 4p positions where two seats died earlier and the last two (or three, or four)
  die on the same turn, for the draw values.

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_rules.py [--out DIR] [--only NAME,...]

Output under ``<out>/review_rules/`` (default out /dev/shm/naz-golden; RAM, never committed).
All little-endian. Per group directory:

* ``meta.txt``: ``key value`` lines: width, height, num_snakes, ruleset, hazard_damage,
  shrink_every, num_states, num_transitions, draws (comma-separated draw values, as Python
  floats, used for the placement values in derived.bin).
* ``states.bin``: one record per distinct state (sources and children together): i32 turn,
  i32 done; per seat s < N: i32 head_x, head_y, length, health, alive, last_move, elim_cause,
  elim_turn; i8 sides[W+H-1] (zeros outside royale); u8 food[H*W]; i16 hazard[H*W];
  i16 body[N][H*W]. Grids are ``y * W + x`` with y = 0 the bottom row.
* ``derived.bin``: per state, from the jitted Python: u8 action_mask bits[N] (bit a = move a,
  ``rules.action_mask``), u8 legal bits[N] (``mcts._legal``), u8 certainly_starving bits
  (bit s), u8 is_game_over(alive), u8 blocked_cells[16] (u128 bitboard), then for each draw
  value in ``draws``: f32 placement_values[N] (``mcts.placement_values``).
* ``trans.bin``: per transition (``mcts._transition`` without food spawning: ``rules_step``,
  ``turn + 1``, ``done``): u32 parent index, u32 child index, u8 actions[4] (seats >= N are 0;
  some ids are invalid, 4..255, and repeat last_move).
* ``labels.txt``: ``index label`` lines naming the hand-built source states.

Plus ``<out>/review_rules/placement.bin``: records of i32 n, f64 draw, u8 alive[4],
i32 elim_turn[4], f32 values[4] (seats >= n zero), from ``jax.jit(mcts.placement_values)``
with the draw static, and ``<out>/review_rules/royale_hazards.bin``: records of i32 width,
i32 height, i32 shrink_every, i32 turn, i8 sides[W+H-1], u8 hazard[H*W]
(``rules.royale_hazards``).
"""
import argparse, dataclasses, os, struct, time
import numpy as np
import jax, jax.numpy as jnp

from slinky import mcts, rules
from slinky.env import BattlesnakeEnv
from slinky.types import GameConfig, RoyaleState, Ruleset, State

DELTAS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
DRAWS = [-0.5, -0.3, -0.7, -0.1, -0.45, 0.0, 0.3, -1.0, -0.123, -0.9, -0.2, -0.6]
FIELDS = ("body", "head", "length", "health", "alive", "last_move", "food", "hazard", "turn",
          "elim_cause", "elim_turn", "done")


# --- Building states --------------------------------------------------------------------------


def blank(game):
    n, h, w = game.num_snakes, game.height, game.width
    return dict(body=np.zeros((n, h, w), np.int16), head=np.zeros((n, 2), np.int32),
                length=np.zeros(n, np.int32), health=np.zeros(n, np.int32),
                alive=np.zeros(n, bool), last_move=np.zeros(n, np.int8),
                food=np.zeros((h, w), bool), hazard=np.zeros((h, w), np.int16),
                turn=np.int32(0), elim_cause=np.zeros(n, np.int8),
                elim_turn=np.zeros(n, np.int32), done=np.bool_(False),
                sides=np.zeros(w + h - 1, np.int8))


def direction(segs):
    if len(segs) < 2:
        return 0
    (hx, hy), (nx, ny) = segs[0], segs[1]
    if hx == nx + 1:
        return 3
    if hx == nx - 1:
        return 2
    if hy == ny + 1:
        return 0
    if hy == ny - 1:
        return 1
    return 0


def put(d, game, s, segs, health=100):
    """Seat s alive with head-first segments (engine_json body_to_countdown)."""
    L = len(segs)
    g = np.zeros((game.height, game.width), np.int16)
    for k, (x, y) in enumerate(segs):
        if 0 <= x < game.width and 0 <= y < game.height:
            g[y, x] = max(g[y, x], L - k)
    d["body"][s] = g
    d["head"][s] = segs[0] if segs else (0, 0)
    d["length"][s] = L
    d["health"][s] = health
    d["alive"][s] = True
    d["elim_cause"][s] = 0
    d["elim_turn"][s] = 0
    d["last_move"][s] = direction(segs)


def kill(d, s, elim_turn, cause=4, head=(0, 0), length=3):
    d["body"][s] = 0
    d["alive"][s] = False
    d["elim_turn"][s] = elim_turn
    d["elim_cause"][s] = cause
    d["head"][s] = head
    d["length"][s] = length
    d["health"][s] = 0


def finish(d):
    d["done"] = np.bool_(d["alive"].sum() <= 1)
    return d


def royale_zone(game, sides, turn):
    return np.asarray(rules.royale_hazards(jnp.asarray(sides, jnp.int8), jnp.int32(turn), game))


def all_actions(n):
    j = np.arange(4 ** n)
    return ((j[:, None] // 4 ** np.arange(n)[None, :]) % 4).astype(np.int32)


# --- Python functions ---------------------------------------------------------------------------


def build_fns(game):
    senv = BattlesnakeEnv(game, obs=None)
    mcfg = mcts.MCTSConfig()
    assert not mcfg.spawn_food
    royale = game.ruleset == Ruleset.ROYALE

    def mk(d):
        ms = RoyaleState(d["sides"]) if royale else ()
        return State(**{f: d[f] for f in FIELDS}, map_state=ms)

    def unmk(s):
        out = {f: getattr(s, f) for f in FIELDS}
        out["sides"] = s.map_state.sides if royale else jnp.zeros(
            game.width + game.height - 1, jnp.int8)
        return out

    def trans(d, a):
        child, _ = mcts._transition(None, mk(d), a, senv, mcfg)
        return unmk(child)

    def derived(d):
        s = mk(d)
        mask = rules.action_mask(s, game)
        out = dict(mask=mask, legal=mcts._legal(s, mask, game),
                   starving=rules.certainly_starving(s, game),
                   over=rules.is_game_over(s.alive, game),
                   blocked=rules.blocked_cells(s, game))
        for k, dv in enumerate(DRAWS):
            out[f"pv{k}"] = mcts.placement_values(s, game, dv)
        return out

    return jax.jit(jax.vmap(trans)), jax.jit(jax.vmap(derived))


def stack(ds):
    return {k: np.stack([np.asarray(d[k]) for d in ds]) for k in ds[0]}


def unstack(b, i):
    return {k: np.asarray(v[i]).copy() for k, v in b.items()}


# --- Writer -------------------------------------------------------------------------------------


class Group:
    """Collects distinct states, their derived values and transitions for one game config."""

    def __init__(self, name, game, root):
        self.name, self.game = name, game
        self.dir = os.path.join(root, name)
        self.trans_fn, self.derived_fn = build_fns(game)
        self.index = {}
        self.states = []  # bytes
        self.dicts = []
        self.trans = []
        self.labels = []

    def key(self, d):
        g = self.game
        n = g.num_snakes
        ints = [int(d["turn"]), int(bool(d["done"]))]
        for s in range(n):
            ints += [int(d["head"][s, 0]), int(d["head"][s, 1]), int(d["length"][s]),
                     int(d["health"][s]), int(bool(d["alive"][s])), int(d["last_move"][s]),
                     int(d["elim_cause"][s]), int(d["elim_turn"][s])]
        return b"".join([
            struct.pack(f"<{len(ints)}i", *ints),
            np.asarray(d["sides"], np.int8).tobytes(),
            np.asarray(d["food"], np.uint8).tobytes(),
            np.asarray(d["hazard"], "<i2").tobytes(),
            np.asarray(d["body"], "<i2").tobytes(),
        ])

    def add(self, d, label=None):
        k = self.key(d)
        i = self.index.get(k)
        if i is None:
            i = len(self.states)
            self.index[k] = i
            self.states.append(k)
            self.dicts.append(d)
        if label is not None:
            self.labels.append((i, label))
        return i

    def expand(self, srcs, actions_per_src):
        """Children of each source under its joint actions; returns child dicts per source."""
        n = self.game.num_snakes
        flat_d, flat_a, owners = [], [], []
        for si, (d, acts) in enumerate(zip(srcs, actions_per_src)):
            for a in acts:
                flat_d.append(d)
                flat_a.append(np.asarray(a, np.int32))
                owners.append(si)
        out = [[] for _ in srcs]
        B = 512
        for b0 in range(0, len(flat_d), B):
            bd = stack(flat_d[b0:b0 + B])
            ba = np.stack(flat_a[b0:b0 + B])
            res = jax.tree.map(np.asarray, self.trans_fn(bd, ba))
            for k in range(len(ba)):
                child = unstack(res, k)
                p = self.add(flat_d[b0 + k])
                c = self.add(child)
                acts = np.zeros(4, np.int64)
                acts[:n] = ba[k]
                self.trans.append((p, c, bytes(int(v) & 0xFF for v in acts)))
                out[owners[b0 + k]].append(child)
        return out

    def write(self):
        g = self.game
        n = g.num_snakes
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "states.bin"), "wb") as f:
            f.write(b"".join(self.states))
        with open(os.path.join(self.dir, "derived.bin"), "wb") as f:
            B = 512
            for b0 in range(0, len(self.dicts), B):
                dv = jax.tree.map(np.asarray, self.derived_fn(stack(self.dicts[b0:b0 + B])))
                for i in range(len(self.dicts[b0:b0 + B])):
                    blocked = dv["blocked"][i].reshape(-1)
                    bb = sum(1 << int(c) for c in np.nonzero(blocked)[0])
                    parts = [bytes(bits(dv["mask"][i, s]) for s in range(n)),
                             bytes(bits(dv["legal"][i, s]) for s in range(n)),
                             bytes([bits(dv["starving"][i]), int(dv["over"][i])]),
                             bb.to_bytes(16, "little")]
                    for k in range(len(DRAWS)):
                        parts.append(np.asarray(dv[f"pv{k}"][i], "<f4").tobytes())
                    f.write(b"".join(parts))
        with open(os.path.join(self.dir, "trans.bin"), "wb") as f:
            for p, c, a in self.trans:
                f.write(struct.pack("<II", p, c) + a)
        with open(os.path.join(self.dir, "labels.txt"), "w") as f:
            for i, lab in self.labels:
                f.write(f"{i} {lab}\n")
        with open(os.path.join(self.dir, "meta.txt"), "w") as f:
            f.write(f"width {g.width}\nheight {g.height}\nnum_snakes {n}\n"
                    f"ruleset {g.ruleset.value}\nhazard_damage {g.hazard_damage_per_turn}\n"
                    f"shrink_every {g.shrink_every_n_turns}\nnum_states {len(self.states)}\n"
                    f"num_transitions {len(self.trans)}\n"
                    f"draws {','.join(repr(x) for x in DRAWS)}\n")
        print(f"{self.name}: {len(self.states)} states, {len(self.trans)} transitions", flush=True)


def bits(row):
    return sum(1 << a for a, b in enumerate(np.asarray(row).tolist()) if b)


# --- Scenarios ----------------------------------------------------------------------------------


def arm(cx, cy, dx, dy, L, stacked=0):
    """Segments of a straight snake whose head is next to (cx, cy) on side (dx, dy), body
    extending away, with ``stacked`` extra copies of the tail."""
    segs = [(cx + dx * (k + 1), cy + dy * (k + 1)) for k in range(L)]
    return segs + [segs[-1]] * stacked


def headon_states(game, rng):
    """k snakes around a centre cell, all length combinations, food or not."""
    n = game.num_snakes
    out = []
    cx, cy = game.width // 2, game.height // 2
    sides = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    import itertools
    for k in range(2, n + 1):
        combos = list(itertools.product([2, 3, 4], repeat=k))
        if k == 4:  # keep the 4-way files small: every all-equal combo plus a sample
            pick = rng.choice(len(combos), 24, replace=False)
            combos = sorted(set(combos[i] for i in pick) | {(2,) * 4, (3,) * 4, (4,) * 4})
        for lens in combos:
            for food in (False, True):
                d = blank(game)
                for s in range(n):
                    if s < k:
                        dx, dy = sides[s]
                        stacked = int(rng.integers(0, 2))
                        L = lens[s]
                        segs = arm(cx, cy, dx, dy, L - stacked, stacked) if L - stacked >= 1 else arm(cx, cy, dx, dy, L)
                        put(d, game, s, segs, health=int(rng.choice([100, 100, 1, 2, 50])))
                    else:
                        if rng.random() < 0.5:
                            kill(d, s, int(rng.integers(0, 4)), head=(cx, cy))
                        else:
                            put(d, game, s, [(0, 0), (0, 0)], health=100)
                d["food"][cy, cx] = food
                d["turn"] = np.int32(rng.integers(0, 5))
                if game.ruleset == Ruleset.ROYALE:
                    d["sides"][:] = 4
                out.append((finish(d), f"headon k={k} lens={lens} food={food}"))
    return out


def stacked_states(game, rng):
    """Snake A next to food: after eating its tail is stacked; B's head next to A's tail."""
    out = []
    n = game.num_snakes
    # A: head (5,5), body down to (5,2); food at (5,6). B: head at (6,2), next to A's tail.
    for a_len in (2, 3, 4):
        for food in (True, False):
            for b_len in (2, 3, 5):
                for hb in (100, 1):
                    d = blank(game)
                    a = [(5, 5 - k) for k in range(a_len)]
                    put(d, game, 0, a)
                    tail = a[-1]
                    b = [(tail[0] + 1 + k, tail[1]) for k in range(b_len)]
                    put(d, game, 1, b, health=hb)
                    for s in range(2, n):
                        put(d, game, s, [(1 + 2 * s, 10), (1 + 2 * s, 9)])
                    if food:
                        d["food"][6, 5] = True
                    d["turn"] = np.int32(7)
                    if game.ruleset == Ruleset.ROYALE:
                        d["sides"][:] = 4
                    out.append((finish(d), f"stacked a_len={a_len} food={food} b_len={b_len} hb={hb}"))
    # Own stacked tail next to own head: a 2x2 loop that just ate.
    for st in (0, 1, 2):
        d = blank(game)
        segs = [(5, 5), (5, 4), (4, 4), (4, 5)] + [(4, 5)] * st
        put(d, game, 0, segs)
        put(d, game, 1, [(8, 8), (8, 7), (8, 6)])
        for s in range(2, n):
            put(d, game, s, [(1 + 2 * s, 1), (1 + 2 * s, 0)])
        if game.ruleset == Ruleset.ROYALE:
            d["sides"][:] = 4
        out.append((finish(d), f"own-loop stacked={st}"))
    return out


def neck_states(game, rng):
    out = []
    n = game.num_snakes
    for segs in ([(5, 5)], [(5, 5), (5, 4)], [(5, 5), (5, 5)], [(5, 5), (5, 4), (5, 4)],
                 [(5, 5), (5, 4), (5, 3)], [(5, 5), (5, 5), (5, 5)], [(5, 5), (5, 4), (5, 4), (5, 4)],
                 [(5, 5), (4, 5)], [(0, 0), (1, 0)], [(10, 10), (10, 9)]):
        for hl in (100, 1):
            d = blank(game)
            put(d, game, 0, list(segs), health=hl)
            put(d, game, 1, [(5, 6), (6, 6)] if (5, 6) not in segs else [(8, 8), (8, 7)])
            for s in range(2, n):
                put(d, game, s, [(1 + 2 * s, 1), (1 + 2 * s, 0)])
            if rng.random() < 0.3:
                d["food"][5, 6] = True  # (6, 5)
            out.append((finish(d), f"neck segs={segs} health={hl}"))
    return out


def hazard_states(game, rng):
    """Health around the exact-kill boundary on a hazard cell next to the head."""
    out = []
    n = game.num_snakes
    dmg = max(-100, min(100, game.hazard_damage_per_turn))
    for layers in (1, 2, 3, 7, 150):
        base = 1 + min(layers, 101) * dmg
        hs = sorted({h for h in range(base - 2, base + 3)} | {1, 2, 0, 100, 99, 101, -1})
        for h in hs:
            for food in (False, True):
                for twin in (False, True):
                    d = blank(game)
                    put(d, game, 0, [(5, 5), (5, 4), (5, 3)], health=h)
                    d["hazard"][5, 6] = layers  # cell (6, 5), right of seat 0's head
                    if food:
                        d["food"][5, 6] = True
                    if twin:  # seat 1 enters the same cell from the right
                        put(d, game, 1, [(7, 5), (8, 5), (9, 5)], health=max(h, 1))
                    else:
                        put(d, game, 1, [(8, 8), (8, 7), (8, 6)], health=h)
                        d["hazard"][9, 8] = layers  # above seat 1's head
                    for s in range(2, n):
                        put(d, game, s, [(1 + 2 * s, 1), (1 + 2 * s, 0)], health=h)
                        d["hazard"][2, 1 + 2 * s] = layers
                    out.append((finish(d), f"hazard layers={layers} health={h} food={food} twin={twin}"))
    return out


def random_snake(rng, game, occupied, L, allow_overlap=False):
    w, h = game.width, game.height
    free = np.argwhere(~occupied) if not allow_overlap else np.argwhere(np.ones_like(occupied))
    if len(free) == 0:
        return None
    y, x = free[rng.integers(len(free))]
    segs = [(int(x), int(y))]
    occupied[y, x] = True
    while len(segs) < L:
        px, py = segs[-1]
        opts = [(px + a, py + b) for a, b in DELTAS]
        opts = [(a, b) for a, b in opts if 0 <= a < w and 0 <= b < h and (allow_overlap or not occupied[b, a])]
        if not opts:
            break
        a, b = opts[rng.integers(len(opts))]
        segs.append((a, b))
        occupied[b, a] = True
    return segs


def crowd_state(game, rng, weird=False):
    w, h, n = game.width, game.height, game.num_snakes
    d = blank(game)
    occupied = np.zeros((h, w), bool)
    turn = int(rng.choice([0, 1, 2, 3, 24, 25, 26, 49, 50, int(rng.integers(0, 200))]))
    if weird and rng.random() < 0.15:
        turn = int(rng.choice([2**31 - 1, 2**31 - 2, -1, -26]))
    d["turn"] = np.int32(turn)
    n_dead = int(rng.integers(0, n - 1)) if rng.random() < 0.4 else 0
    for s in range(n):
        if s >= n - n_dead:
            kill(d, s, int(rng.integers(0, max(1, min(turn, 50)) + 1)) if turn >= 0 else 0,
                 cause=int(rng.integers(1, 7)),
                 head=(int(rng.integers(-1, w + 1)), int(rng.integers(-1, h + 1))),
                 length=int(rng.integers(0, 6)))
            continue
        L = int(rng.choice([1, 2, 2, 3, 3, 4, 5, 7]))
        segs = random_snake(rng, game, occupied, L, allow_overlap=weird and rng.random() < 0.1)
        if segs is None:
            segs = [(0, 0)]
        segs += [segs[-1]] * int(rng.choice([0, 0, 0, 1, 2]))
        health = int(rng.choice([1, 1, 2, 3, 15, 16, 50, 100]))
        if weird and rng.random() < 0.1:
            health = int(rng.choice([0, -5, 2**31 - 1, -2**31, 250]))
        put(d, game, s, segs, health=health)
        if weird and rng.random() < 0.05:
            d["head"][s] = (-1, int(rng.integers(0, h)))  # alive with an off-board head
        if weird and rng.random() < 0.05:
            d["length"][s] = int(rng.choice([0, 300, 32767, 40000]))
        d["last_move"][s] = int(rng.integers(0, 4))
    for _ in range(int(rng.integers(0, 4))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if not occupied[y, x] or (weird and rng.random() < 0.3):
            d["food"][y, x] = True
    # Food next to heads, sometimes.
    for s in range(n):
        if d["alive"][s] and rng.random() < 0.3:
            dx, dy = DELTAS[rng.integers(4)]
            x, y = d["head"][s, 0] + dx, d["head"][s, 1] + dy
            if 0 <= x < w and 0 <= y < h:
                d["food"][y, x] = True
    if game.ruleset == Ruleset.ROYALE:
        S = game.shrink_every_n_turns
        sides = rng.integers(0, 5, w + h - 1).astype(np.int8)
        if weird and rng.random() < 0.3:
            sides = rng.choice(np.array([-128, -1, 0, 1, 2, 3, 4, 5, 127], np.int8), w + h - 1)
        if rng.random() < 0.6:
            t = int(S * rng.integers(0, w + h + 1) + rng.integers(-1, 2))
            if not (weird and turn > 2**30):
                turn = max(t, 0)
            d["turn"] = np.int32(turn)
        d["sides"] = sides
        d["hazard"] = royale_zone(game, sides, turn).astype(np.int16)
        if rng.random() < 0.15:  # a request zone that disagrees with the sides
            d["hazard"] = rng.integers(0, 2, (h, w)).astype(np.int16)
    elif game.hazard_damage_per_turn != 14 or rng.random() < 0.3:
        k = int(rng.integers(0, w * h))
        for _ in range(k):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            d["hazard"][y, x] += int(rng.choice([1, 1, 1, 2, 3, 200]))
    return finish(d)


def lasttwo_states(game, rng):
    """4p: seats 2 and 3 died earlier (same or different turns, or padded at 0); seats 0 and 1
    meet head-on (equal or unequal lengths, food or not)."""
    out = []
    for e2, e3 in ((3, 5), (5, 5), (0, 0), (0, 7), (10, 10), (11, 3)):
        for la, lb in ((3, 3), (3, 4), (4, 3)):
            for food in (False, True):
                d = blank(game)
                put(d, game, 0, arm(5, 5, -1, 0, la))
                put(d, game, 1, arm(5, 5, 1, 0, lb))
                kill(d, 2, e2)
                kill(d, 3, e3)
                d["food"][5, 5] = food
                d["turn"] = np.int32(10)
                if game.ruleset == Ruleset.ROYALE:
                    d["sides"][:] = 4
                out.append((finish(d), f"lasttwo elim=({e2},{e3}) lens=({la},{lb}) food={food}"))
    # Three or four die together on one cell, with one or none dead before.
    for dead_before in (0, 1):
        for lens in ((3, 3, 3, 3), (3, 3, 3, 4), (4, 4, 3, 3), (2, 3, 4, 5)):
            for food in (False, True):
                d = blank(game)
                for s, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):
                    put(d, game, s, arm(5, 5, dx, dy, lens[s]))
                if dead_before:
                    kill(d, 3, 4)
                d["food"][5, 5] = food
                d["turn"] = np.int32(10)
                if game.ruleset == Ruleset.ROYALE:
                    d["sides"][:] = 4
                out.append((finish(d), f"pileup dead_before={dead_before} lens={lens} food={food}"))
    return out


def constrictor_states(game, rng):
    out = []
    n = game.num_snakes
    for segs in ([(5, 5)], [(5, 5), (5, 4)], [(5, 5), (5, 5), (5, 5)], [(5, 5), (5, 4), (5, 4)],
                 [(5, 5), (5, 4), (5, 3)], [(5, 5), (5, 4), (5, 3), (5, 3)]):
        for health in (100, 1, 2, 0):
            for food in (False, True):
                d = blank(game)
                put(d, game, 0, list(segs), health=health)
                put(d, game, 1, [(6, 6), (7, 6), (7, 6)], health=100)
                for s in range(2, n):
                    put(d, game, s, [(1 + 2 * s, 1), (1 + 2 * s, 0)], health=health)
                if food:
                    d["food"][6, 5] = True
                    d["food"][5, 4] = True
                out.append((finish(d), f"constrictor segs={segs} health={health} food={food}"))
    return out


# --- Groups ---------------------------------------------------------------------------------------


def walk(group, starts, rng, steps):
    """Random search-like chains: one joint action per step, biased towards legal moves."""
    game = group.game
    n = game.num_snakes
    cur = list(starts)
    for _ in range(steps):
        if not cur:
            break
        acts = []
        for d in cur:
            a = rng.integers(0, 4, n)
            if rng.random() < 0.05:
                a[rng.integers(n)] = int(rng.choice([4, 5, 7, 200, 255]))
            acts.append([a])
        children = group.expand(cur, acts)
        cur = [c[0] for c in children if not bool(c[0]["done"])]


def run_group(name, game, root, rng, builders, full=True, chain=False, walks=0, walk_steps=25):
    t0 = time.time()
    g = Group(name, game, root)
    n = game.num_snakes
    srcs = []
    for b in builders:
        srcs += b(game, rng)
    for d, lab in srcs:
        g.add(d, lab)
    ds = [d for d, _ in srcs]
    if full:
        kids = g.expand(ds, [all_actions(n)] * len(ds))
        if chain:  # every joint action again from every distinct living child
            seen, nxt = set(), []
            for ks in kids:
                for c in ks:
                    k = g.key(c)
                    if k not in seen and not bool(c["done"]):
                        seen.add(k)
                        nxt.append(c)
            g.expand(nxt, [all_actions(n)] * len(nxt))
    if walks:
        walk(g, [d for d in ds for _ in range(walks)], rng, walk_steps)
    g.write()
    print(f"  {name}: {time.time() - t0:.1f}s", flush=True)


def placement_fuzz(root, rng):
    """Random alive / elim_turn patterns for N = 2, 3, 4 and every draw value."""
    recs = []
    for n in (2, 3, 4):
        game = GameConfig(num_snakes=n)
        for draw in DRAWS:
            fn = jax.jit(jax.vmap(lambda a, e: mcts.placement_values(
                empty_state(game)._replace(alive=a, elim_turn=e), game, draw)))
            m = 600
            alive = rng.random((m, n)) < rng.choice([0.0, 0.2, 0.5], (m, 1))
            elim = rng.integers(0, 4, (m, n)).astype(np.int32)
            big = rng.random((m, n)) < 0.05
            elim[big] = rng.choice(np.array([2**31 - 1, -1, 2**31 - 2], np.int32), big.sum())
            vals = np.asarray(fn(jnp.asarray(alive), jnp.asarray(elim)))
            for i in range(m):
                a4 = np.zeros(4, np.uint8); a4[:n] = alive[i]
                e4 = np.zeros(4, np.int32); e4[:n] = elim[i]
                v4 = np.zeros(4, np.float32); v4[:n] = vals[i]
                recs.append(struct.pack("<id", n, draw) + a4.tobytes() + e4.astype("<i4").tobytes()
                            + v4.astype("<f4").tobytes())
    with open(os.path.join(root, "placement.bin"), "wb") as f:
        f.write(b"".join(recs))
    print(f"placement: {len(recs)} records", flush=True)


def empty_state(game):
    from slinky.types import empty_state as es
    return es(game)


def hazards_fuzz(root, rng):
    recs = []
    for (w, h) in ((11, 11), (5, 5), (3, 7), (1, 1), (11, 1)):
        for S in (1, 2, 3, 25):
            game = GameConfig(width=w, height=h, num_snakes=2, ruleset="royale", shrink_every_n_turns=S)
            fn = jax.jit(lambda s, t: rules.royale_hazards(s, t, game))
            num = w + h - 1
            for _ in range(60):
                sides = rng.choice(np.array([-128, -1, 0, 1, 2, 3, 4, 5, 127], np.int8), num)
                if rng.random() < 0.5:
                    sides = rng.integers(0, 4, num).astype(np.int8)
                t = int(rng.choice([0, S - 1, S, S + 1, S * num - 1, S * num, S * num + 5,
                                    2**31 - 1, -2**31, -1, -S, -S - 1, int(rng.integers(0, S * num + 3))]))
                hz = np.asarray(fn(jnp.asarray(sides), jnp.int32(t)))
                recs.append(struct.pack("<iiii", w, h, S, t) + sides.astype(np.int8).tobytes()
                            + hz.astype(np.uint8).tobytes())
    with open(os.path.join(root, "royale_hazards.bin"), "wb") as f:
        f.write(b"".join(recs))
    print(f"royale_hazards: {len(recs)} records", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    root = os.path.join(args.out, "review_rules")
    os.makedirs(root, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    rng = np.random.default_rng(20261015)
    G = GameConfig
    rep = dataclasses.replace

    def crowd_builder(count, weird):
        return lambda game, rng: [(crowd_state(game, rng, weird), f"crowd weird={weird}") for _ in range(count)]

    jobs = {
        "headon_std4": (G(num_snakes=4), [headon_states], dict(full=True)),
        "headon_std3": (G(num_snakes=3), [headon_states], dict(full=True)),
        "headon_duel": (G(num_snakes=2), [headon_states, neck_states], dict(full=True, chain=True)),
        "headon_con4": (G(num_snakes=4, ruleset="constrictor"), [headon_states], dict(full=True)),
        "headon_roy3": (G(num_snakes=3, ruleset="royale"), [headon_states], dict(full=True)),
        "stacked_duel": (G(num_snakes=2), [stacked_states, neck_states], dict(full=True, chain=True)),
        "stacked_std3": (G(num_snakes=3), [stacked_states], dict(full=True, walks=4)),
        "stacked_con2": (G(num_snakes=2, ruleset="constrictor"), [stacked_states, constrictor_states], dict(full=True, chain=True)),
        "neck_std4": (G(num_snakes=4), [neck_states], dict(full=True)),
        "hazard_d14": (G(num_snakes=2), [hazard_states], dict(full=True)),
        "hazard_d1": (rep(G(num_snakes=2), hazard_damage_per_turn=1), [hazard_states], dict(full=True)),
        "hazard_d50": (rep(G(num_snakes=3), hazard_damage_per_turn=50), [hazard_states], dict(full=True)),
        "hazard_d100": (rep(G(num_snakes=2), hazard_damage_per_turn=100), [hazard_states], dict(full=True)),
        "hazard_d101": (rep(G(num_snakes=2), hazard_damage_per_turn=101), [hazard_states], dict(full=True)),
        "hazard_d0": (rep(G(num_snakes=2), hazard_damage_per_turn=0), [hazard_states], dict(full=True)),
        "hazard_heal": (rep(G(num_snakes=2), hazard_damage_per_turn=-15), [hazard_states], dict(full=True)),
        "hazard_royale3": (G(num_snakes=3, ruleset="royale"), [hazard_states], dict(full=True)),
        "lasttwo_std4": (G(num_snakes=4), [lasttwo_states], dict(full=True)),
        "lasttwo_roy4": (G(num_snakes=4, ruleset="royale"), [lasttwo_states], dict(full=True)),
        "lasttwo_con4": (G(num_snakes=4, ruleset="constrictor"), [lasttwo_states], dict(full=True)),
        "constrictor_4": (G(num_snakes=4, ruleset="constrictor"), [constrictor_states], dict(full=True, walks=2, walk_steps=30)),
        "crowd_std4_5": (G(width=5, height=5, num_snakes=4), [crowd_builder(120, False), crowd_builder(40, True)], dict(full=True, walks=2)),
        "crowd_std3_4": (G(width=4, height=4, num_snakes=3), [crowd_builder(200, False), crowd_builder(60, True)], dict(full=True, walks=2)),
        "crowd_duel_3": (G(width=3, height=3, num_snakes=2), [crowd_builder(300, False), crowd_builder(100, True)], dict(full=True, chain=True, walks=2)),
        "crowd_duel_6": (G(width=6, height=4, num_snakes=2), [crowd_builder(300, False), crowd_builder(100, True)], dict(full=True, chain=True, walks=3)),
        "crowd_haz4": (rep(G(width=5, height=5, num_snakes=4), hazard_damage_per_turn=30), [crowd_builder(100, False), crowd_builder(40, True)], dict(full=True, walks=2)),
        "crowd_heal3": (rep(G(width=5, height=4, num_snakes=3), hazard_damage_per_turn=-7), [crowd_builder(150, False), crowd_builder(50, True)], dict(full=True, walks=2)),
        "crowd_con4": (G(width=5, height=5, num_snakes=4, ruleset="constrictor"), [crowd_builder(100, False), crowd_builder(40, True)], dict(full=True, walks=2)),
        "crowd_con2": (G(width=4, height=4, num_snakes=2, ruleset="constrictor"), [crowd_builder(300, False), crowd_builder(100, True)], dict(full=True, chain=True, walks=3)),
        "royale_s1": (G(width=5, height=5, num_snakes=4, ruleset="royale", shrink_every_n_turns=1), [crowd_builder(100, False), crowd_builder(40, True)], dict(full=True, walks=3)),
        "royale_s2": (G(width=6, height=5, num_snakes=3, ruleset="royale", shrink_every_n_turns=2), [crowd_builder(150, False), crowd_builder(50, True)], dict(full=True, walks=3)),
        "royale_s3": (G(width=4, height=4, num_snakes=2, ruleset="royale", shrink_every_n_turns=3), [crowd_builder(300, False), crowd_builder(100, True)], dict(full=True, chain=True, walks=3)),
        "royale_s25": (G(width=11, height=11, num_snakes=4, ruleset="royale"), [crowd_builder(60, False), crowd_builder(20, True)], dict(full=True, walks=2)),
        "royale_heal": (rep(G(width=5, height=5, num_snakes=2, ruleset="royale", shrink_every_n_turns=2), hazard_damage_per_turn=-20), [crowd_builder(200, False), crowd_builder(60, True)], dict(full=True, chain=True, walks=2)),
    }
    for name, (game, builders, kw) in jobs.items():
        if only and name not in only:
            continue
        run_group(name, game, root, rng, builders, **kw)
    if not only or "placement" in only:
        placement_fuzz(root, rng)
    if not only or "royale_hazards" in only:
        hazards_fuzz(root, rng)


if __name__ == "__main__":
    main()
