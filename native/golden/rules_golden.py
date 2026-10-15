"""Golden vectors for the Rust rules crate (native/crates/rules).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/rules_golden.py [--scale F] [--only NAME,...] [--out DIR]

Generates transitions with slinky's own jitted rules: source states come from rollouts of
``BattlesnakeEnv`` (heuristic agent with an epsilon of random legal moves, food spawning on),
some perturbed (low health, extra food beside heads, random stacked hazards, royale turns at
shrink boundaries with server-reconstructed shrink sides). Every source state is expanded with
``mcts._transition`` (``rules.rules_step`` without food spawning, ``turn + 1``, ``done``) for
sampled joint actions (all 16 in duels, all 4^N for 4p states with heads close together).

Output: one directory per game config under ``<out>/rules/<name>/`` (default out
/dev/shm/naz-golden; RAM, never committed). Main configs (``--scale 1``: >= 20,000 transitions
each): ``duel``, ``standard`` (4p), ``royale`` (4p), ``constrictor`` (4p). Extra configs (about
3,000 each): ``standard3``, ``royale2``, ``constrictor2``, ``royale_fast`` (shrink every 3 turns,
whole board hazardous late), ``hazard_std`` (4p standard with random stacked hazards, damage 14),
``hazard100`` (duel, damage 100), ``heal_duel`` (duel, damage -15, healing hazards), ``small``
(7x7 standard 4p), ``rect`` (9 wide x 11 high, 3 snakes). Each directory holds (all little-endian):

* ``meta.txt``: ``key value`` lines: width, height, num_snakes, ruleset (standard, constrictor,
  royale), hazard_damage, shrink_every, num_states, num_transitions.
* ``states.bin``: ``num_states`` fixed-size records, sources and children together, each:
  i32 turn, i32 done, then per seat s < N: i32 head_x, head_y, length, health, alive,
  last_move, elim_cause, elim_turn; then i8 sides[W+H-1] (zeros outside royale); u8 food[H*W];
  u8 hazard[H*W]; u8 body[N][H*W] (countdowns; the script asserts they fit in u8). Grids are
  ``y * W + x`` with y = 0 the bottom row.
* ``derived.bin``: per state, the jitted Python of: u8 action_mask bits[N] (bit a = move a,
  ``rules.action_mask``), u8 legal bits[N] (``mcts._legal``), u8 certainly_starving bits (bit s),
  u8 is_game_over(alive), u8 blocked_cells[16] (u128 bitboard, ``rules.blocked_cells``),
  f32 placement_values[N] with draw -0.5, f32 placement_values[N] with draw -0.3.
* ``trans.bin``: per transition: u32 parent index, u32 child index, u8 actions[4] (seats >= N
  are 0; a few actions are invalid ids 4..7, which repeat last_move).

Plus ``<out>/rules/royale_hazards.bin``: records of i32 width, i32 height, i32 shrink_every,
i32 turn, i8 sides[W+H-1], u8 hazard[H*W] (``rules.royale_hazards``), with W+H-1 and H*W taken
from the record's own width and height.
"""
import argparse, dataclasses, os, struct, sys, time
import numpy as np
import jax, jax.numpy as jnp

from slinky import mcts, modes, rules, server
from slinky.agents import make_agent
from slinky.env import BattlesnakeEnv
from slinky.types import GameConfig, RoyaleState, Ruleset

MAIN = 20_000
DELTAS = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)])
EXTRA = 3_000


def configs():
    m = modes.mode_config
    royale_fast = dataclasses.replace(m("royale"), shrink_every_n_turns=3)
    return [
        ("duel", m("duel"), MAIN, {}),
        ("standard", m("standard"), MAIN, {}),
        ("royale", m("royale"), MAIN, {"royale_turns": True}),
        ("constrictor", m("constrictor"), MAIN, {"no_low_health": True}),
        ("standard3", m("standard", 3), EXTRA, {}),
        ("royale2", m("royale", 2), EXTRA, {"royale_turns": True}),
        ("constrictor2", m("constrictor", 2), EXTRA, {"no_low_health": True}),
        ("royale_fast", royale_fast, EXTRA, {"royale_turns": True}),
        ("hazard_std", m("standard"), EXTRA, {"hazards": 8}),
        ("hazard100", dataclasses.replace(m("duel"), hazard_damage_per_turn=100), EXTRA, {"hazards": 2}),
        ("heal_duel", dataclasses.replace(m("duel"), hazard_damage_per_turn=-15), EXTRA, {"hazards": 3}),
        ("small", GameConfig(width=7, height=7, num_snakes=4), EXTRA, {}),
        ("rect", GameConfig(width=9, height=11, num_snakes=3), EXTRA, {}),
    ]


# --- Rollouts -------------------------------------------------------------------------------


def forager(key, state, mask, game):
    """A cautious food-seeking policy: legal moves, away from cells next to the heads of
    snakes at least as long, towards the nearest food, with random tie-breaks. Its games last
    long and grow long snakes (the heuristic agent's duels end within a few dozen turns)."""
    d = jnp.asarray(DELTAS, jnp.int32)
    nxt = state.head[:, None, :] + d[None]  # [N, 4, 2]
    ys, xs = jnp.mgrid[: game.height, : game.width]
    dist = jnp.abs(nxt[..., 0, None, None] - xs) + jnp.abs(nxt[..., 1, None, None] - ys)
    fd = jnp.min(jnp.where(state.food, dist, 99), axis=(-1, -2))  # [N, 4]
    other_head = state.head[None, None, :, :]  # [1, 1, N, 2]
    near = jnp.abs(nxt[:, :, None, :] - other_head).sum(-1) == 1  # [N, 4, N]
    big = (state.length[None, :] >= state.length[:, None]) & state.alive[None, :]
    big = big & ~jnp.eye(game.num_snakes, dtype=bool)
    danger = jnp.any(near & big[:, None, :], axis=-1)
    hungry = (state.health < 60)[:, None]
    # Room: free cells around the target (2 steps), a crude guard against dead ends.
    solid = jnp.any((state.body > 1) & state.alive[:, None, None], axis=0)
    pad = jnp.pad(~solid, 1)
    free = sum(pad[1 + dy : 1 + dy + game.height, 1 + dx : 1 + dx + game.width]
               for dx, dy in DELTAS.tolist()).astype(jnp.float32)
    room = sum(jnp.pad(free, 1)[1 + dy : 1 + dy + game.height, 1 + dx : 1 + dx + game.width]
               for dx, dy in DELTAS.tolist())
    tx = jnp.clip(nxt[..., 0], 0, game.width - 1)
    ty = jnp.clip(nxt[..., 1], 0, game.height - 1)
    score = -8.0 * danger - jnp.where(hungry, 1.0, 0.2) * fd + 0.5 * free[ty, tx] + 0.15 * room[ty, tx]
    score = score + jax.random.gumbel(key, score.shape)
    return jnp.argmax(jnp.where(mask, score, -jnp.inf), axis=-1)


def rollouts(game, n_games, turns, seed):
    """Stacked states [n_games, turns]: half heuristic games, half forager games, each with a
    per-game epsilon of uniformly random legal moves."""
    env = BattlesnakeEnv(game, obs="egocentric")
    pol = make_agent("heuristic", game).policy

    def body(carry, key):
        state, ts, eps, use_forager = carry
        k1, k2, k3, k4 = jax.random.split(key, 4)
        a = jnp.where(use_forager, forager(k1, state, ts.action_mask, game), pol(k1, state, ts))
        rnd = jax.random.categorical(k2, jnp.where(ts.action_mask, 0.0, -jnp.inf))
        a = jnp.where(jax.random.uniform(k3, a.shape) < eps, rnd, a)
        ns, nts = env.step(k4, state, a)
        return (ns, nts, eps, use_forager), ns

    def one(key, eps, use_forager):
        s, ts = env.reset(key)
        _, traj = jax.lax.scan(body, (s, ts, eps, use_forager), jax.random.split(key, turns - 1))
        # Keep the turn-0 state (three segments stacked on one cell) too.
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), s, traj)

    keys = jax.random.split(jax.random.key(seed), n_games)
    eps = jnp.asarray(np.resize([0.0, 0.03, 0.1, 0.0, 0.25, 0.6], n_games), jnp.float32)
    use_forager = jnp.asarray(np.arange(n_games) % 2 == 1)
    traj = jax.jit(jax.vmap(one))(keys, eps, use_forager)
    return jax.tree.map(lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), traj)


FIELDS = ("body", "head", "length", "health", "alive", "last_move", "food", "hazard", "turn",
          "elim_cause", "elim_turn", "done")


def to_numpy(traj, game):
    d = {f: np.array(getattr(traj, f)) for f in FIELDS}
    if game.ruleset == Ruleset.ROYALE:
        d["sides"] = np.array(traj.map_state.sides)
    else:
        d["sides"] = np.zeros((len(d["turn"]), game.width + game.height - 1), np.int8)
    return d


def take(d, idx):
    return {k: v[idx].copy() for k, v in d.items()}


# --- Perturbations --------------------------------------------------------------------------

def perturb(d, game, opts, rng):
    """In-place edge-case perturbations of a batch of numpy states."""
    m = len(d["turn"])
    w, h, n = game.width, game.height, game.num_snakes
    for i in range(m):
        alive = d["alive"][i]
        r = rng.random()
        if not opts.get("no_low_health") and r < 0.2:
            for s in range(n):
                if alive[s] and rng.random() < 0.5:
                    d["health"][i, s] = rng.integers(1, 4)
        if rng.random() < 0.25:
            occupied = (d["body"][i] > 0).any(0)
            for s in range(n):
                if alive[s] and rng.random() < 0.6:
                    dx, dy = DELTAS[rng.integers(4)]
                    x, y = d["head"][i, s, 0] + dx, d["head"][i, s, 1] + dy
                    if 0 <= x < w and 0 <= y < h and not occupied[y, x]:
                        d["food"][i, y, x] = True
        if opts.get("hazards") and rng.random() < 0.8:
            k = rng.integers(1, w * h // 2)
            ys, xs = rng.integers(0, h, k), rng.integers(0, w, k)
            for y, x in zip(ys, xs):
                d["hazard"][i, y, x] += rng.integers(1, opts["hazards"] + 1)
            # Hazards on heads' neighbours, sometimes with food.
            for s in range(n):
                if alive[s] and rng.random() < 0.5:
                    dx, dy = DELTAS[rng.integers(4)]
                    x, y = d["head"][i, s, 0] + dx, d["head"][i, s, 1] + dy
                    if 0 <= x < w and 0 <= y < h:
                        d["hazard"][i, y, x] += 1
        if opts.get("royale_turns") and rng.random() < 0.35:
            # A turn at a shrink boundary, with the zone shown at that turn, as the server
            # rebuilds the sides from the hazards (unknown future shrinks).
            S = game.shrink_every_n_turns
            t = int(S * rng.integers(0, (w + h + 2)) + rng.integers(-1, 2))
            t = max(t, 0)
            sides = d["sides"][i]
            hz = np.asarray(rules.royale_hazards(jnp.asarray(sides), jnp.int32(t), game))
            d["turn"][i] = t
            d["hazard"][i] = hz
            if rng.random() < 0.6:
                ys, xs = np.nonzero(hz)
                rs = server.royale_sides([[int(x), int(y)] for x, y in zip(xs, ys)], t, game)
                d["sides"][i] = np.asarray(rs.sides)


# --- Expansion and derived values -----------------------------------------------------------


def build_fns(game):
    senv = BattlesnakeEnv(game, obs=None)
    mcfg = mcts.MCTSConfig()
    assert not mcfg.spawn_food

    def mk(d):
        from slinky.types import State
        ms = RoyaleState(d["sides"]) if game.ruleset == Ruleset.ROYALE else ()
        return State(**{f: d[f] for f in FIELDS}, map_state=ms)

    def unmk(s):
        out = {f: getattr(s, f) for f in FIELDS}
        out["sides"] = s.map_state.sides if game.ruleset == Ruleset.ROYALE else jnp.zeros(
            game.width + game.height - 1, jnp.int8)
        return out

    def trans(d, a):
        child, _ = mcts._transition(None, mk(d), a, senv, mcfg)
        return unmk(child)

    def derived(d):
        s = mk(d)
        mask = rules.action_mask(s, game)
        return dict(
            mask=mask,
            legal=mcts._legal(s, mask, game),
            starving=rules.certainly_starving(s, game),
            over=rules.is_game_over(s.alive, game),
            blocked=rules.blocked_cells(s, game),
            pv5=mcts.placement_values(s, game, -0.5),
            pv3=mcts.placement_values(s, game, -0.3),
        )

    return jax.jit(jax.vmap(trans)), jax.jit(jax.vmap(derived))


def sample_actions(d, i, game, rng, k, full=False):
    """Joint actions for state i: all 4^N for duels, hand-built crowds (``full``) and some states
    with heads close together; else k random ones, with a few invalid ids (4 to 7)."""
    n = game.num_snakes
    alive, head = d["alive"][i], d["head"][i]
    if n == 2 or full or (close_heads(alive, head) and rng.random() < 0.3):
        j = np.arange(4 ** n)
        return ((j[:, None] // 4 ** np.arange(n)[None, :]) % 4).astype(np.int32)
    out = rng.integers(0, 4, (k, n))
    bad = rng.random((k, n)) < 0.02
    out[bad] = rng.integers(4, 8, bad.sum())
    return out.astype(np.int32)


def huddle(d, i, game, opts, rng):
    """Overwrite state i with a hand-built crowd: 2 to 4 snakes whose heads all touch one free
    cell (random lengths, often equal, stacked tails, low health, food or hazard on that cell),
    so the enumerated joint actions give 3- and 4-way head-to-heads."""
    w, h, n = game.width, game.height, game.num_snakes
    cx, cy = int(rng.integers(1, w - 1)), int(rng.integers(1, h - 1))
    for f in ("body", "food", "hazard"):
        d[f][i] = 0
    occupied = np.zeros((h, w), bool)
    occupied[cy, cx] = True
    dirs = rng.permutation(4)
    n_dead = int(rng.integers(0, n - 1)) if n >= 3 and rng.random() < 0.3 else 0
    turn = int(rng.integers(1, 300))
    base_len = int(rng.integers(1, 9))
    for s in range(n):
        if s >= n - n_dead:
            d["alive"][i, s] = False
            d["body"][i, s] = 0
            d["head"][i, s] = rng.integers(-1, max(w, h) + 1, 2)
            d["length"][i, s] = rng.integers(0, 10)
            d["health"][i, s] = rng.integers(0, 101)
            d["elim_cause"][i, s] = rng.integers(1, 7)
            d["elim_turn"][i, s] = rng.integers(0, turn + 1)
            d["last_move"][i, s] = rng.integers(0, 4)
            continue
        dx, dy = DELTAS[dirs[s]]
        x, y = cx + dx, cy + dy
        segs = []
        if 0 <= x < w and 0 <= y < h and not occupied[y, x]:
            segs.append((x, y)); occupied[y, x] = True
        target = base_len if rng.random() < 0.5 else int(rng.integers(1, 9))
        while segs and len(segs) < target:
            px, py = segs[-1]
            opts_ = [(px + a, py + b) for a, b in DELTAS.tolist()]
            opts_ = [(a, b) for a, b in opts_ if 0 <= a < w and 0 <= b < h and not occupied[b, a]]
            if not opts_:
                break
            a, b = opts_[rng.integers(len(opts_))]
            segs.append((a, b)); occupied[b, a] = True
        if not segs:  # no room: an off-board-free fallback, a stacked snake in a free cell
            free = np.argwhere(~occupied)
            yy, xx = free[rng.integers(len(free))]
            segs = [(int(xx), int(yy))]; occupied[yy, xx] = True
        segs += [segs[-1]] * int(rng.choice([0, 0, 1, 2]))
        if len(segs) < 2:
            segs += [segs[-1]]
        L = len(segs)
        grid = np.zeros((h, w), np.int16)
        for k, (a, b) in enumerate(segs):
            grid[b, a] = max(grid[b, a], L - k)
        d["body"][i, s] = grid
        d["head"][i, s] = segs[0]
        d["length"][i, s] = L
        d["health"][i, s] = rng.choice([1, 2, int(rng.integers(1, 101)), 100])
        if opts.get("no_low_health"):
            d["health"][i, s] = 100
        d["alive"][i, s] = True
        d["elim_cause"][i, s] = 0
        d["elim_turn"][i, s] = 0
        d["last_move"][i, s] = rng.integers(0, 4)
    if rng.random() < 0.4:
        d["food"][i, cy, cx] = True
    for _ in range(int(rng.integers(0, 4))):
        a, b = int(rng.integers(0, w)), int(rng.integers(0, h))
        if not occupied[b, a]:
            d["food"][i, b, a] = True
    d["turn"][i] = turn
    if game.ruleset == Ruleset.ROYALE:
        d["sides"][i] = rng.integers(0, 5, w + h - 1)
        d["hazard"][i] = np.asarray(rules.royale_hazards(jnp.asarray(d["sides"][i]), jnp.int32(turn), game))
    elif opts.get("hazards") and rng.random() < 0.5:
        d["hazard"][i, cy, cx] = rng.integers(1, opts["hazards"] + 1)
    d["done"][i] = d["alive"][i].sum() <= 1


def close_heads(alive, head):
    idx = np.nonzero(alive)[0]
    for a in idx:
        for b in idx:
            if a < b and np.abs(head[a] - head[b]).sum() <= 2:
                return True
    return False


# --- Writing --------------------------------------------------------------------------------


def state_bytes(d, i, game):
    n = game.num_snakes
    ints = [int(d["turn"][i]), int(d["done"][i])]
    for s in range(n):
        ints += [int(d["head"][i, s, 0]), int(d["head"][i, s, 1]), int(d["length"][i, s]),
                 int(d["health"][i, s]), int(d["alive"][i, s]), int(d["last_move"][i, s]),
                 int(d["elim_cause"][i, s]), int(d["elim_turn"][i, s])]
    body, hazard = d["body"][i], d["hazard"][i]
    assert body.min() >= 0 and body.max() < 256 and hazard.min() >= 0 and hazard.max() < 256
    return b"".join([
        struct.pack(f"<{len(ints)}i", *ints),
        np.asarray(d["sides"][i], np.int8).tobytes(),
        np.asarray(d["food"][i], np.uint8).tobytes(),
        np.asarray(hazard, np.uint8).tobytes(),
        np.asarray(body, np.uint8).tobytes(),
    ])


def bits(row):
    return sum(1 << a for a, b in enumerate(row) if b)


def derived_bytes(dv, i, game):
    n = game.num_snakes
    blocked = dv["blocked"][i].reshape(-1)
    bb = sum(1 << c for c in np.nonzero(blocked)[0].tolist())
    return b"".join([
        bytes(bits(dv["mask"][i, s]) for s in range(n)),
        bytes(bits(dv["legal"][i, s]) for s in range(n)),
        bytes([bits(dv["starving"][i]), int(dv["over"][i])]),
        bb.to_bytes(16, "little"),
        np.asarray(dv["pv5"][i], "<f4").tobytes(),
        np.asarray(dv["pv3"][i], "<f4").tobytes(),
    ])


def generate(name, game, target, opts, out, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = game.num_snakes
    turns = 300 if game.ruleset != Ruleset.CONSTRICTOR else 150
    n_games = 24 if target >= MAIN else 8
    traj = to_numpy(rollouts(game, n_games, turns, seed), game)
    trans_fn, derived_fn = build_fns(game)
    total = len(traj["turn"])
    live = np.nonzero(~traj["done"])[0]
    dead = np.nonzero(traj["done"])[0]
    os.makedirs(out, exist_ok=True)
    fs, fd, ft = (open(os.path.join(out, f), "wb") for f in ("states.bin", "derived.bin", "trans.bin"))
    n_states = n_trans = 0
    k = 32
    max_len = 0
    while n_trans < target:
        m = 128
        pick = rng.choice(live, m)
        if len(dead):
            pick[: max(1, m // 50)] = rng.choice(dead, max(1, m // 50))
        src = take(traj, pick)
        perturb(src, game, opts, rng)
        n_huddle = 4 if n >= 3 else 8
        for i in range(m - n_huddle, m):
            huddle(src, i, game, opts, rng)
        acts, parent = [], []
        for i in range(m):
            a = sample_actions(src, i, game, rng, k, full=i >= m - n_huddle)
            acts.append(a)
            parent += [i] * len(a)
        acts = np.concatenate(acts)
        parent = np.asarray(parent)
        # Pad to a multiple of 2048 rows so the jitted functions compile only a few shapes.
        nt = len(acts)
        padded = -(-nt // 2048) * 2048
        prow = np.concatenate([parent, np.full(padded - nt, parent[-1])])
        pacts = np.concatenate([acts, np.repeat(acts[-1:], padded - nt, 0)])
        rep = {kk: jnp.asarray(v[prow]) for kk, v in src.items()}
        child = {kk: np.asarray(v)[:nt] for kk, v in trans_fn(rep, jnp.asarray(pacts)).items()}
        dsrc = {kk: np.asarray(v) for kk, v in derived_fn({kk: jnp.asarray(v) for kk, v in src.items()}).items()}
        cpad = {kk: jnp.asarray(np.concatenate([v, np.repeat(v[-1:], padded - nt, 0)])) for kk, v in child.items()}
        dch = {kk: np.asarray(v)[:nt] for kk, v in derived_fn(cpad).items()}
        base = n_states
        for i in range(m):
            fs.write(state_bytes(src, i, game)); fd.write(derived_bytes(dsrc, i, game))
        for r in range(len(acts)):
            fs.write(state_bytes(child, r, game)); fd.write(derived_bytes(dch, r, game))
            a4 = list(acts[r]) + [0] * (4 - n)
            ft.write(struct.pack("<II4B", base + int(parent[r]), base + m + r, *a4))
        n_states += m + len(acts)
        n_trans += len(acts)
        max_len = max(max_len, int(child["length"].max()))
    for f in (fs, fd, ft):
        f.close()
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"width {game.width}\nheight {game.height}\nnum_snakes {n}\n"
                f"ruleset {game.ruleset.value}\nhazard_damage {game.hazard_damage_per_turn}\n"
                f"shrink_every {game.shrink_every_n_turns}\nnum_states {n_states}\n"
                f"num_transitions {n_trans}\n")
    print(f"{name}: {n_trans} transitions, {n_states} states from {total} rollout states, "
          f"max length {max_len}, {time.time() - t0:.1f}s", flush=True)


def royale_table(path, rng, count=3000):
    recs = []
    for (w, h, S) in ((11, 11, 25), (11, 11, 3), (7, 9, 5), (11, 5, 1)):
        game = GameConfig(width=w, height=h, num_snakes=4, ruleset=Ruleset.ROYALE, map="royale",
                          shrink_every_n_turns=S)
        nside = w + h - 1
        sides = rng.integers(0, 4, (count // 4, nside)).astype(np.int8)
        unk = rng.random(sides.shape) < 0.15
        sides[unk] = rng.choice(np.array([4, 5, -1, 100], np.int8), unk.sum())
        turns = rng.integers(-2 * S, S * (nside + 3), count // 4).astype(np.int32)
        f = jax.jit(jax.vmap(lambda s, t: rules.royale_hazards(s, t, game)))
        hz = np.asarray(f(jnp.asarray(sides), jnp.asarray(turns)))
        for i in range(len(turns)):
            recs.append(struct.pack("<4i", w, h, S, int(turns[i])) + sides[i].tobytes()
                        + hz[i].astype(np.uint8).tobytes())
    with open(path, "wb") as f:
        f.write(b"".join(recs))
    print(f"royale_hazards: {len(recs)} cases", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    root = os.path.join(args.out, "rules")
    os.makedirs(root, exist_ok=True)
    if not only or "royale_hazards" in only:
        royale_table(os.path.join(root, "royale_hazards.bin"), np.random.default_rng(args.seed))
    for k, (name, game, target, opts) in enumerate(configs()):
        if only and name not in only:
            continue
        generate(name, game, int(target * args.scale), opts, os.path.join(root, name),
                 args.seed * 1000 + k)


if __name__ == "__main__":
    main()
