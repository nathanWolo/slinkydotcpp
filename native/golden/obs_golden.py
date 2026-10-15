"""Golden vectors for the Rust observation crate (native/crates/obs).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/obs_golden.py [--scale F] [--only NAME,...] [--out DIR]

Source states come from rollouts of ``BattlesnakeEnv`` (food spawning on): half the games play
slinky's heuristic agent and half play uniformly random legal moves, each game with its own
epsilon of random moves. About a third of the sampled states are then perturbed to reach edge
cases the network inputs must handle: odd turns (0, 999 to 1001, far past 1000, negative, around
shrink boundaries), health outside 1..100, stacked hazards, extra food, length ties, dead seats
with off-board heads and leftover (ignored) body grids, a living snake with an off-board head and
two living opponents whose heads share a cell. Each state is fed to the jitted Python:

* ``boards, feats = unified.policy_inputs(env.observe(state), state, game, cfg, mode_of(game))``
  with the deployed run's config (az-modes-M-s0: ``hide_constrictor_food=True``,
  ``features="all"``), i.e. exactly what ``az_modes.make_evaluator`` feeds the network;
* ``env.observe(state)`` alone (the raw egocentric windows, food not hidden);
* ``modes.mode_features(state, game, m)`` for each of the four told modes ``m``.

Output: one directory per config under ``<out>/obs/<name>/`` (default out /dev/shm/naz-golden;
RAM, never committed). Main configs (``--scale 1``: 2,500 states each): ``duel``, ``standard``
(4p), ``royale`` (4p), ``constrictor`` (4p). Extra configs (500 each): ``standard3``, ``royale2``,
``constrictor2``, ``royale7`` (4p royale, shrink every 7 turns), ``small`` (7x7 standard 4p),
``rect`` (9 wide x 11 high, 3 snakes standard). Each directory holds (all little-endian):

* ``meta.txt``: ``key value`` lines: width, height, num_snakes, ruleset (standard, constrictor,
  royale), shrink_every, hide_food (1 when the boards have the food plane zeroed), rows, cols
  (the window, 2H-1 and 2W-1), num_states.
* ``states.bin``: ``num_states`` records: i32 turn, i32 done, then per seat s < N: i32 head_x,
  head_y, length, health, alive, elim_turn; then u8 food[H*W]; i16 hazard[H*W];
  i16 body[N][H*W]. Grids are ``y * W + x`` with y = 0 the bottom row.
* ``boards.bin``: per state, per seat: the ``policy_inputs`` board ``f32[rows][cols][13]`` in
  sparse form: u32 count, u16 index[count] (flat index ``(r * cols + c) * 13 + ch``), u32
  bits[count] (the f32 bit patterns). Every index not listed holds bit pattern 0 (+0.0).
* ``raw_diff.bin``: per state, per seat, same sparse form: the entries where ``env.observe``
  (food not hidden) differs from the board, with the raw value's bits. Empty outside
  constrictor (the script asserts it).
* ``feats.bin``: per state: f32 ``policy_inputs`` features [N][7], then f32
  ``mode_features`` [4 told modes in MODE_NAMES order][N][7].
"""
import argparse, dataclasses, os, struct, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/home/nathan/slinky-release/baselines")
import az_modes  # noqa: E402
from slinky import modes, unified  # noqa: E402
from slinky.agents import make_agent  # noqa: E402
from slinky.env import BattlesnakeEnv  # noqa: E402
from slinky.types import GameConfig, RoyaleState, Ruleset, State  # noqa: E402

RUN = "/home/nathan/slinky-serve/az-modes-M-s0"
MAIN, EXTRA = 2_500, 500
FIELDS = ("body", "head", "length", "health", "alive", "last_move", "food", "hazard", "turn",
          "elim_cause", "elim_turn", "done")


def configs():
    m = modes.mode_config
    return [
        ("duel", m("duel"), MAIN),
        ("standard", m("standard"), MAIN),
        ("royale", m("royale"), MAIN),
        ("constrictor", m("constrictor"), MAIN),
        ("standard3", m("standard", 3), EXTRA),
        ("royale2", m("royale", 2), EXTRA),
        ("constrictor2", m("constrictor", 2), EXTRA),
        ("royale7", dataclasses.replace(m("royale"), shrink_every_n_turns=7), EXTRA),
        ("small", GameConfig(width=7, height=7, num_snakes=4), EXTRA),
        ("rect", GameConfig(width=9, height=11, num_snakes=3), EXTRA),
    ]


def rollouts(game, n_games, turns, seed):
    """Stacked numpy states [n_games * turns]: even games heuristic, odd games random legal,
    each with a per-game epsilon of random legal moves."""
    env = BattlesnakeEnv(game, obs=None)
    pol = make_agent("heuristic", game).policy

    def body(carry, key):
        state, ts, eps, rnd_game = carry
        k1, k2, k3, k4 = jax.random.split(key, 4)
        rnd = jax.random.categorical(k2, jnp.where(ts.action_mask, 0.0, -jnp.inf))
        a = jnp.where(rnd_game, rnd, pol(k1, state, ts))
        a = jnp.where(jax.random.uniform(k3, a.shape) < eps, rnd, a)
        ns, nts = env.step(k4, state, a)
        return (ns, nts, eps, rnd_game), ns

    def one(key, eps, rnd_game):
        s, ts = env.reset(key)
        _, traj = jax.lax.scan(body, (s, ts, eps, rnd_game), jax.random.split(key, turns - 1))
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), s, traj)

    keys = jax.random.split(jax.random.key(seed), n_games)
    eps = jnp.asarray(np.resize([0.0, 0.05, 0.2, 0.5], n_games), jnp.float32)
    rnd_game = jnp.asarray(np.arange(n_games) % 2 == 1)
    traj = jax.jit(jax.vmap(one))(keys, eps, rnd_game)
    d = {f: np.array(getattr(traj, f)).reshape((-1,) + getattr(traj, f).shape[2:]) for f in FIELDS}
    if game.ruleset == Ruleset.ROYALE:
        sides = np.array(traj.map_state.sides)
        d["sides"] = sides.reshape((-1,) + sides.shape[2:])
    return d


def perturb(d, i, game, rng):
    """Edge cases for state i (in place). The observation is a pure function of the fields, so
    states need not be reachable."""
    w, h, n = game.width, game.height, game.num_snakes
    S = game.shrink_every_n_turns
    r = rng.random()
    if r < 0.25:
        d["turn"][i] = rng.choice([0, 1, 998, 999, 1000, 1001, 1500, 123456, -1, -7,
                                   S * int(rng.integers(0, 30)) + int(rng.integers(-1, 2)),
                                   int(rng.integers(0, 3000))])
    alive = d["alive"][i]
    for s in range(n):
        if alive[s] and rng.random() < 0.1:
            d["health"][i, s] = rng.choice([0, 1, 2, 99, 100, 101, 250, -5, int(rng.integers(1, 101))])
        if alive[s] and rng.random() < 0.1:
            d["length"][i, s] += int(rng.integers(-2, 3))
    if rng.random() < 0.3:
        k = int(rng.integers(1, w * h))
        ys, xs = rng.integers(0, h, k), rng.integers(0, w, k)
        for y, x in zip(ys, xs):
            d["hazard"][i, y, x] += int(rng.integers(1, 4))
    if rng.random() < 0.3:
        for _ in range(int(rng.integers(1, 6))):
            d["food"][i, rng.integers(0, h), rng.integers(0, w)] = True
    live = np.nonzero(alive)[0]
    if len(live) >= 2 and rng.random() < 0.15:
        # Length ties.
        a, b = rng.choice(live, 2, replace=False)
        d["length"][i, a] = d["length"][i, b]
    if len(live) >= 3 and rng.random() < 0.3:
        # Two living snakes' heads on one cell (seen from a third), different health.
        a, b = rng.choice(live, 2, replace=False)
        d["head"][i, a] = d["head"][i, b]
        d["health"][i, a] = int(rng.integers(1, 101))
    if len(live) >= 2 and rng.random() < 0.2:
        # A dead seat: off-board (or on-board) head, body grid left behind, arbitrary fields.
        s = rng.choice(live)
        d["alive"][i, s] = False
        d["head"][i, s] = rng.integers(-3, max(w, h) + 3, 2)
        d["elim_cause"][i, s] = int(rng.integers(1, 7))
        d["elim_turn"][i, s] = int(rng.integers(0, 300))
        if rng.random() < 0.5:
            d["body"][i, s] = 0
    if len(live) >= 1 and rng.random() < 0.05:
        # A living snake whose head is off the board (just walled, before elimination).
        s = rng.choice(live)
        d["head"][i, s] = rng.choice([[-1, int(rng.integers(0, h))], [w, int(rng.integers(0, h))],
                                      [int(rng.integers(0, w)), -1], [int(rng.integers(0, w)), h]])
    d["done"][i] = d["alive"][i].sum() <= 1


def build_fn(game, cfg):
    env = BattlesnakeEnv(game, obs="egocentric")
    mode = modes.mode_of(game)

    def mk(d):
        ms = RoyaleState(d["sides"]) if game.ruleset == Ruleset.ROYALE else ()
        return State(**{f: d[f] for f in FIELDS}, map_state=ms)

    def fn(d):
        s = mk(d)
        raw = env.observe(s)
        boards, feats = unified.policy_inputs(raw, s, game, cfg, mode)
        mf = jnp.stack([modes.mode_features(s, game, m) for m in modes.MODE_NAMES])
        return raw, boards, feats, mf

    return jax.jit(jax.vmap(fn))


def sparse(a):
    flat = np.ascontiguousarray(a, np.float32).reshape(-1).view(np.uint32)
    idx = np.nonzero(flat)[0]
    assert flat.size < 65536
    return b"".join([struct.pack("<I", len(idx)), idx.astype("<u2").tobytes(),
                     flat[idx].astype("<u4").tobytes()])


def sparse_diff(raw, board):
    rb = np.ascontiguousarray(raw, np.float32).reshape(-1).view(np.uint32)
    bb = np.ascontiguousarray(board, np.float32).reshape(-1).view(np.uint32)
    idx = np.nonzero(rb != bb)[0]
    return b"".join([struct.pack("<I", len(idx)), idx.astype("<u2").tobytes(),
                     rb[idx].astype("<u4").tobytes()])


def state_bytes(d, i, game):
    n = game.num_snakes
    ints = [int(d["turn"][i]), int(d["done"][i])]
    for s in range(n):
        ints += [int(d["head"][i, s, 0]), int(d["head"][i, s, 1]), int(d["length"][i, s]),
                 int(d["health"][i, s]), int(d["alive"][i, s]), int(d["elim_turn"][i, s])]
    return b"".join([
        struct.pack(f"<{len(ints)}i", *ints),
        np.asarray(d["food"][i], np.uint8).tobytes(),
        np.asarray(d["hazard"][i], "<i2").tobytes(),
        np.asarray(d["body"][i], "<i2").tobytes(),
    ])


def generate(name, game, target, cfg, out, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n_games = 128 if target >= MAIN else 32
    turns = 160
    traj = rollouts(game, n_games, turns, seed)
    total = len(traj["turn"])
    live = np.nonzero(~traj["done"])[0]
    dead = np.nonzero(traj["done"])[0]
    pick = rng.choice(live, target, replace=len(live) < target)
    if len(dead):
        k = max(1, target // 100)
        pick[:k] = rng.choice(dead, k)
    d = {k: v[pick].copy() for k, v in traj.items()}
    for i in range(target):
        if rng.random() < 0.35:
            perturb(d, i, game, rng)
    fn = build_fn(game, cfg)
    hide = bool(cfg.hide_constrictor_food and game.ruleset.constrictor)
    os.makedirs(out, exist_ok=True)
    files = {k: open(os.path.join(out, k + ".bin"), "wb") for k in ("states", "boards", "raw_diff", "feats")}
    n = game.num_snakes
    rows, cols = 2 * game.height - 1, 2 * game.width - 1
    B = 128
    for a in range(0, target, B):
        sl = {k: v[a:a + B] for k, v in d.items()}
        raw, boards, feats, mf = (np.asarray(x) for x in fn(sl))
        assert boards.shape[1:] == (n, rows, cols, 13), boards.shape
        if hide:
            assert not boards[..., 1].any()
            assert np.array_equal(np.delete(raw, 1, -1).view(np.uint32), np.delete(boards, 1, -1).view(np.uint32))
        else:
            assert np.array_equal(raw.view(np.uint32), boards.view(np.uint32))
        for j in range(len(boards)):
            files["states"].write(state_bytes(sl, j, game))
            for s in range(n):
                files["boards"].write(sparse(boards[j, s]))
                files["raw_diff"].write(sparse_diff(raw[j, s], boards[j, s]))
            files["feats"].write(np.asarray(feats[j], "<f4").tobytes() + np.asarray(mf[j], "<f4").tobytes())
    for f in files.values():
        f.close()
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"width {game.width}\nheight {game.height}\nnum_snakes {n}\n"
                f"ruleset {game.ruleset.value}\nshrink_every {game.shrink_every_n_turns}\n"
                f"hide_food {int(hide)}\nrows {rows}\ncols {cols}\nnum_states {target}\n")
    print(f"{name}: {target} states sampled from {len(live)} live rollout states ({total} in all), "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--only", default="")
    p.add_argument("--out", default="/dev/shm/naz-golden")
    args = p.parse_args()
    cfg = az_modes.load_config(RUN)
    assert cfg.hide_constrictor_food and cfg.features == "all"
    only = set(filter(None, args.only.split(",")))
    for k, (name, game, target) in enumerate(configs()):
        if only and name not in only:
            continue
        generate(name, game, max(16, int(target * args.scale)), cfg,
                 os.path.join(args.out, "obs", name), 1000 + k)


if __name__ == "__main__":
    main()
