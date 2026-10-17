"""Golden data for the Rust env crate (native/crates/env).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/env_golden.py [--only trans,dist] \\
        [--configs NAME,...] [--scale F] [--out DIR] [--seed S]

Two parts, both from slinky's own jitted ``BattlesnakeEnv`` (env.py, maps.py, modes.py):

``trans``: exact transitions. Source states come from random-legal rollouts of each config,
some perturbed (low health, food next to heads, a few finished games). Each is stepped once by
``env.step`` with a fresh key and a sampled joint action (random legal, uniformly random, or an
invalid id 4..7 that repeats the last move). Alongside, the script records what the map's
randomness decided: the spawn roll ``100 - bits[0] % 100`` that ``spawn_food_standard`` derives
from that key, the spawn mask of the post-rules state (``maps.spawn_mask``) and the cells that
were spawned (``next.food & ~rules_step(state).food``). The Rust step, scripted with the same roll
and cells, must reproduce everything else exactly.

    <out>/env/trans/<name>/meta.txt: ``key value`` lines: width, height, num_snakes, ruleset,
        map, food_spawn_chance, minimum_food, max_turns (-1 for none), hazard_damage,
        shrink_every, num_transitions.
    <out>/env/trans/<name>/trans.bin: records (little-endian):
        state (pre), u8 actions[4], i32 roll, u8 spawn_mask[16] (u128, bit y*W+x),
        u8 spawned[16] (u128), u8 truncated, f32 win_loss[N], f32 placement[N], state (next).
      A state is the rules golden format: i32 turn, i32 done, per seat i32 head_x, head_y,
      length, health, alive, last_move, elim_cause, elim_turn; i8 sides[W+H-1] (zeros outside
      royale); u8 food[H*W]; u8 hazard[H*W]; u8 body[N][H*W].

``dist``: distributions. G whole games per config from ``env.reset`` with random-legal play
(``policies.random_legal_policy``) until every game is over, aggregated into
``<out>/env/dist/<name>.json``:

    {"name", "config": {...}, "games": G,
     "start_joint": {"x,y;x,y;...": count},   # seat-ordered start heads
     "start_seat_cell": [[count per cell] per seat],
     "start_food_count": [count per food count], "start_food_cell": [count per cell],
     "sides_pos": [[count of side s at shrink k] per k], "sides_pair": [16 counts of (k0, k1)],
     "first_hazard_turn": {turn: games},         # royale: first turn showing a hazard
     "turn_active": [games not over at turn t], "food_sum": [...], "food_sq": [...],
     "hazard_sum": [...],                         # per turn t, over games not over at turn t
     "spawn_low": [transitions, food spawned, sum of squares],  # post-rules food < minimum_food
     "spawn_high": [transitions, transitions that spawned],
     "length": [games ending at turn t], "truncated": count,
     "alive_end": [games ending with k alive], "winner": [wins per seat],
     "causes": [eliminations per cause code 0..6]}

Default out /dev/shm/naz-golden (RAM; never committed).
"""
import argparse, dataclasses, json, os, struct, time
import numpy as np
import jax, jax.numpy as jnp

from slinky import maps, modes, rules
from slinky.env import BattlesnakeEnv, placement_reward, win_loss_reward
from slinky.types import GameConfig, Ruleset

MAIN, EXTRA = 20_000, 5_000


def configs():
    m = modes.mode_config
    c4 = GameConfig(num_snakes=4, ruleset=Ruleset.CONSTRICTOR, map="empty")
    return [
        # name, config, transitions, games for the distributions (0: none)
        ("duel", m("duel"), MAIN, 32768),
        ("standard", m("standard"), MAIN, 16384),
        ("royale", m("royale"), MAIN, 16384),
        ("constrictor", m("constrictor"), MAIN, 16384),
        ("constrictor_empty", c4, MAIN, 16384),
        ("standard3", m("standard", 3), EXTRA, 8192),
        ("royale2", m("royale", 2), EXTRA, 8192),
        ("royale_fast", dataclasses.replace(m("royale"), shrink_every_n_turns=3), EXTRA, 8192),
        ("royale_stdmap", GameConfig(num_snakes=4, ruleset=Ruleset.ROYALE, map="standard"), EXTRA, 0),
        ("food_rich", dataclasses.replace(m("duel"), minimum_food=3, food_spawn_chance=60), EXTRA, 8192),
        ("packed", GameConfig(width=7, height=7, num_snakes=4, minimum_food=40, food_spawn_chance=0), EXTRA, 0),
        ("max_turns", dataclasses.replace(m("standard"), max_turns=40), EXTRA, 8192),
        ("small", GameConfig(width=7, height=7, num_snakes=4), EXTRA, 8192),
        ("rect", GameConfig(width=9, height=11, num_snakes=3), EXTRA, 8192),
    ]


FIELDS = ("body", "head", "length", "health", "alive", "last_move", "food", "hazard", "turn",
          "elim_cause", "elim_turn", "done")


def sides_of(state, game):
    if game.ruleset == Ruleset.ROYALE:
        return state.map_state.sides
    lead = state.turn.shape
    return jnp.zeros(lead + (game.width + game.height - 1,), jnp.int8)


def to_numpy(s, game):
    d = {f: np.array(getattr(s, f)) for f in FIELDS}
    d["sides"] = np.array(sides_of(s, game))
    return d


def mk(d, game):
    from slinky.types import RoyaleState, State
    ms = RoyaleState(d["sides"]) if game.ruleset == Ruleset.ROYALE else ()
    return State(**{f: d[f] for f in FIELDS}, map_state=ms)


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


def bitboard(grid):
    """u128 little-endian bytes of a bool[H, W] grid (bit y*W+x)."""
    v = 0
    for c in np.nonzero(np.asarray(grid).ravel())[0]:
        v |= 1 << int(c)
    return v.to_bytes(16, "little")


def legal_actions(key, state, game):
    mask = rules.action_mask(state, game)
    return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf), axis=-1).astype(jnp.int32)


# --- Exact transitions ----------------------------------------------------------------------


def rollouts(game, n_games, turns, seed):
    env = BattlesnakeEnv(game, obs=None)

    def body(s, key):
        k1, k2 = jax.random.split(key)
        ns, _ = env.step(k2, s, legal_actions(k1, s, game))
        return ns, ns

    def one(key):
        s, _ = env.reset(key)
        _, traj = jax.lax.scan(body, s, jax.random.split(key, turns - 1))
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), s, traj)

    keys = jax.random.split(jax.random.key(seed), n_games)
    traj = jax.jit(jax.vmap(one))(keys)
    return jax.tree.map(lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), traj)


def build_trans(game):
    env = BattlesnakeEnv(game, obs=None)
    cells = game.width * game.height

    def trans(d, key, a):
        s = mk(d, game)
        post = rules.rules_step(s, a, game)
        mask = maps.spawn_mask(post, game)
        bits = jax.random.bits(key, (1 + cells,), jnp.uint32)
        roll = 100 - (bits[0] % 100).astype(jnp.int32)
        new, ts = env.step(key, s, a)
        spawned = new.food & ~post.food & ~s.done
        wl = jnp.where(s.done, 0.0, win_loss_reward(s, new, game))
        pl = jnp.where(s.done, 0.0, placement_reward(s, new, game))
        out = {f: getattr(new, f) for f in FIELDS}
        out["sides"] = sides_of(new, game)
        return out, roll, mask, spawned, ts.truncated, wl, pl, ts.reward

    return jax.jit(jax.vmap(trans))


def perturb(d, game, rng):
    m = len(d["turn"])
    w, h, n = game.width, game.height, game.num_snakes
    deltas = ((0, 1), (0, -1), (-1, 0), (1, 0))
    for i in range(m):
        alive = d["alive"][i]
        if game.ruleset != Ruleset.CONSTRICTOR and rng.random() < 0.15:
            for s in range(n):
                if alive[s] and rng.random() < 0.5:
                    d["health"][i, s] = rng.integers(1, 4)
        if rng.random() < 0.25:
            occupied = (d["body"][i] > 0).any(0)
            for s in range(n):
                if alive[s] and rng.random() < 0.6:
                    dx, dy = deltas[rng.integers(4)]
                    x, y = d["head"][i, s, 0] + dx, d["head"][i, s, 1] + dy
                    if 0 <= x < w and 0 <= y < h and not occupied[y, x]:
                        d["food"][i, y, x] = True


def gen_trans(name, game, target, out, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = game.num_snakes
    turns = 40 if game.max_turns else 250
    traj = to_numpy(rollouts(game, 64, turns, seed), game)
    live = np.nonzero(~traj["done"])[0]
    dead = np.nonzero(traj["done"])[0]
    fn = build_trans(game)
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "trans.bin"), "wb")
    count, spawns, multi = 0, 0, 0
    batch = 1024
    while count < target:
        pick = rng.choice(live, batch)
        if len(dead):
            k = batch // 50
            pick[:k] = rng.choice(dead, k)
        src = {k: v[pick].copy() for k, v in traj.items()}
        perturb(src, game, rng)
        keys = jax.random.split(jax.random.key(seed * 7919 + count), batch)
        legal = np.asarray(jax.jit(jax.vmap(lambda k, d: legal_actions(k, mk(d, game), game)))(
            keys, {k: jnp.asarray(v) for k, v in src.items()}))
        r = rng.random(batch)
        acts = legal.copy()
        rnd = rng.integers(0, 4, (batch, n))
        acts[r < 0.15] = rnd[r < 0.15]
        bad = r > 0.95
        acts[bad] = np.where(rng.random((bad.sum(), n)) < 0.5, rng.integers(4, 8, (bad.sum(), n)),
                             acts[bad])
        skeys = jax.random.split(jax.random.key(seed * 104729 + count + 1), batch)
        res = fn({k: jnp.asarray(v) for k, v in src.items()}, skeys, jnp.asarray(acts, jnp.int32))
        nxt, roll, mask, spawned, trunc, wl, pl, reward = jax.tree.map(np.asarray, res)
        assert np.array_equal(reward, wl), "env reward is win_loss_reward"
        for i in range(batch):
            a4 = [int(v) for v in acts[i]] + [0] * (4 - n)
            ns = spawned[i].sum()
            spawns += ns > 0
            multi += ns > 1
            f.write(state_bytes(src, i, game) + struct.pack("<4Bi", *a4, int(roll[i]))
                    + bitboard(mask[i]) + bitboard(spawned[i]) + struct.pack("<B", int(trunc[i]))
                    + np.asarray(wl[i], "<f4").tobytes() + np.asarray(pl[i], "<f4").tobytes()
                    + state_bytes(nxt, i, game))
        count += batch
    f.close()
    with open(os.path.join(out, "meta.txt"), "w") as m:
        m.write(f"width {game.width}\nheight {game.height}\nnum_snakes {n}\n"
                f"ruleset {game.ruleset.value}\nmap {game.map}\n"
                f"food_spawn_chance {game.food_spawn_chance}\nminimum_food {game.minimum_food}\n"
                f"max_turns {game.max_turns if game.max_turns is not None else -1}\n"
                f"hazard_damage {game.hazard_damage_per_turn}\n"
                f"shrink_every {game.shrink_every_n_turns}\nnum_transitions {count}\n")
    print(f"trans {name}: {count} transitions ({spawns} with spawns, {multi} with several), "
          f"{time.time() - t0:.1f}s", flush=True)


# --- Distributions --------------------------------------------------------------------------


def gen_dist(name, game, n_games, out, seed, chunk=64, cap=5000):
    t0 = time.time()
    env = BattlesnakeEnv(game, obs=None)
    n, cells = game.num_snakes, game.width * game.height
    royale = game.ruleset == Ruleset.ROYALE
    s, _ = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(seed), n_games))
    d0 = to_numpy(s, game)
    res = {"name": name, "games": n_games, "config": {
        "width": game.width, "height": game.height, "num_snakes": n,
        "ruleset": game.ruleset.value, "map": game.map,
        "food_spawn_chance": game.food_spawn_chance, "minimum_food": game.minimum_food,
        "max_turns": game.max_turns if game.max_turns is not None else -1,
        "shrink_every": game.shrink_every_n_turns}}
    joint = {}
    seat_cell = np.zeros((n, cells), np.int64)
    for g in range(n_games):
        key = ";".join(f"{int(x)},{int(y)}" for x, y in d0["head"][g])
        joint[key] = joint.get(key, 0) + 1
        for i in range(n):
            x, y = d0["head"][g, i]
            seat_cell[i, y * game.width + x] += 1
    res["start_joint"] = joint
    res["start_seat_cell"] = seat_cell.tolist()
    food0 = d0["food"].reshape(n_games, cells)
    res["start_food_count"] = np.bincount(food0.sum(1), minlength=cells + 1).tolist()
    res["start_food_cell"] = food0.sum(0).tolist()
    if royale:
        sides = d0["sides"].astype(np.int64)
        res["sides_pos"] = [np.bincount(sides[:, k], minlength=4).tolist()
                            for k in range(sides.shape[1])]
        res["sides_pair"] = np.bincount(sides[:, 0] * 4 + sides[:, 1], minlength=16).tolist()

    def body(s, key):
        k1, k2 = jax.random.split(key)
        a = legal_actions(k1, s, game)
        post = rules.rules_step(s, a, game)
        ns, _ = env.step(k2, s, a)
        rec = dict(active=~s.done, turn=s.turn, food=jnp.sum(s.food),
                   hazard=jnp.sum(s.hazard > 0),
                   low=jnp.sum(post.food) < game.minimum_food,
                   spawned=jnp.sum(ns.food & ~post.food))
        return ns, rec

    @jax.jit
    def run(s, keys):
        return jax.vmap(lambda s, k: jax.lax.scan(body, s, k))(s, keys)

    T = cap + 2
    act = np.zeros(T, np.int64)
    fsum = np.zeros(T, np.int64)
    fsq = np.zeros(T, np.int64)
    hsum = np.zeros(T, np.int64)
    first_hz = {}
    seen_hz = np.zeros(n_games, bool)
    spawn_low = [0, 0, 0]
    spawn_high = [0, 0]
    key = jax.random.key(seed + 1)
    turn = 0
    while True:
        key, sub = jax.random.split(key)
        keys = jax.vmap(lambda k: jax.random.split(k, chunk))(jax.random.split(sub, n_games))
        s, rec = run(s, keys)
        rec = jax.tree.map(np.asarray, rec)
        a = rec["active"]
        tt = rec["turn"]
        np.add.at(act, tt[a], 1)
        np.add.at(fsum, tt[a], rec["food"][a])
        np.add.at(fsq, tt[a], rec["food"][a].astype(np.int64) ** 2)
        np.add.at(hsum, tt[a], rec["hazard"][a])
        low = a & rec["low"]
        high = a & ~rec["low"]
        spawn_low[0] += int(low.sum()); spawn_low[1] += int(rec["spawned"][low].sum())
        spawn_low[2] += int((rec["spawned"][low].astype(np.int64) ** 2).sum())
        spawn_high[0] += int(high.sum()); spawn_high[1] += int((rec["spawned"][high] > 0).sum())
        if royale:
            for g in range(n_games):
                if seen_hz[g]:
                    continue
                hit = np.nonzero(a[g] & (rec["hazard"][g] > 0))[0]
                if len(hit):
                    t = int(tt[g, hit[0]])
                    first_hz[t] = first_hz.get(t, 0) + 1
                    seen_hz[g] = True
        turn += chunk
        if bool(np.all(np.asarray(s.done))) or turn >= cap:
            break
    assert bool(np.all(np.asarray(s.done))), f"{name}: games still running at turn {turn}"
    dn = to_numpy(s, game)
    last = int(np.max(np.nonzero(act)[0])) + 1
    res["turn_active"] = act[:last].tolist()
    res["food_sum"] = fsum[:last].tolist()
    res["food_sq"] = fsq[:last].tolist()
    res["hazard_sum"] = hsum[:last].tolist()
    res["first_hazard_turn"] = {str(k): v for k, v in sorted(first_hz.items())}
    res["spawn_low"] = spawn_low
    res["spawn_high"] = spawn_high
    res["length"] = np.bincount(dn["turn"], minlength=last + 1).tolist()
    trunc = (dn["alive"].sum(1) > 1)
    res["truncated"] = int(trunc.sum())
    res["alive_end"] = np.bincount(dn["alive"].sum(1), minlength=n + 1).tolist()
    res["winner"] = (dn["alive"] & (dn["alive"].sum(1, keepdims=True) == 1)).sum(0).tolist()
    res["causes"] = np.bincount(dn["elim_cause"].ravel().astype(np.int64), minlength=7).tolist()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(res, f)
    print(f"dist {name}: {n_games} games, mean length {np.mean(dn['turn']):.1f}, "
          f"max {int(dn['turn'].max())}, {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    ap.add_argument("--only", default="trans,dist")
    ap.add_argument("--configs", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    parts = set(args.only.split(","))
    names = set(filter(None, args.configs.split(",")))
    for k, (name, game, target, games) in enumerate(configs()):
        if names and name not in names:
            continue
        if "trans" in parts:
            gen_trans(name, game, max(1, int(target * args.scale)),
                      os.path.join(args.out, "env", "trans", name), args.seed * 1000 + k)
        if "dist" in parts and games:
            gen_dist(name, game, max(64, int(games * args.scale)),
                     os.path.join(args.out, "env", "dist"), args.seed * 1000 + 500 + k)


if __name__ == "__main__":
    main()
