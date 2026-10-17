"""Adversarial golden vectors for the network and observation crates (review stage).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_network_and_observations.py [--out DIR]

Read by native/crates/net/tests/review_network_and_observations.rs. Everything goes to
``<out>/review_netobs/`` (default out /dev/shm/naz-golden; RAM, never committed).

Unlike obs_golden.py and net_golden.py (random rollouts, the deployed (2, 2, 1) strides), the
states here are hand-built to hit the places most likely to break: heads in all four corners and
on every edge, long self-avoiding snakes coiled around their own head, stacked tails, turn 0 and
late turns (999, 1000, 1001, far past 1000), royale turns around the shrinks (24, 25, 26, 49, 50,
...) with the zone the server would show, shrink periods 1, 7, 25 and 1000, dead seats (padded
from the API, or freshly dead with an off-board head and a leftover body grid), the agent's own
seat dead, constrictor games with food (hidden), coinciding heads, length ties, health 0, 1, 100,
101 and negative, stacked hazards, 3-snake standard and 2-snake royale/constrictor, and a few
small and rectangular boards (observation only). The networks include the deployed M net, fresh
S and L nets and nets with odd architectures (strides 1, 3, 4 and five stride-2 layers that end
at 1x1; channel counts 5, 7, 12, 24 and 64; hidden widths 16, 20, 37 and 100; features "all",
"no_mode" and "none"), all with random non-zero heads and biases so every part matters.

Output files:

* ``nets/<name>/config.json`` + ``params.npz``: loadable run dirs (``M`` is symlinked to the
  deployed /home/nathan/slinky-serve/az-modes-M-s0).
* ``nets/<name>/cases.npz`` (plain ``numpy.savez``): float32 ``obs [n,21,21,13]``, ``feats
  [n,7]`` (raw ``modes.mode_features``, before any no_mode mask), ``logits [n,4]`` and ``value
  [n]`` from the jitted ``az_modes.network`` (fed ``policy_features`` of that net's config, i.e.
  the no_mode mask applied in Python). The boards are synthetic: single pixels in each corner of
  the window, the board in each corner of the window, dense random values of both signs, large
  hazard counts.
* ``e2e.jsonl``: one JSON object per state:
  ``name`` (a label), ``width``, ``height``, ``num_snakes``, ``ruleset`` (standard, royale,
  constrictor), ``shrink_every``, ``hide_food`` (0/1, what the boards show), ``turn``, ``done``,
  ``head [N][2]``, ``length [N]``, ``health [N]``, ``alive [N]`` (0/1), ``elim_turn [N]``,
  ``food`` (list of cells ``y*W+x``), ``hazard [H*W]``, ``body [N][H*W]`` (countdown grids,
  index ``y*W+x``, y = 0 the bottom row); ``boards``: per seat the ``unified.policy_inputs``
  board as ``[[index, f32 bits], ...]`` over the non-zero bit patterns (index ``(r*cols+c)*13+ch``);
  ``feats``: the ``policy_inputs`` feature bits ``[N*7]`` (deployed config: features "all");
  ``eval``: for 11x11 boards, per net name ``{"value": [N], "logits": [N][4]}`` from
  ``az_modes.make_evaluator(params, cfg, group, mode_of(game))(state)``, i.e. the whole Python
  evaluator (observation, food hiding, features with that net's mask, forward pass).
"""
import argparse, dataclasses, json, os, shutil, sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/home/nathan/slinky-release/baselines")
import az_modes  # noqa: E402
from slinky import modes, unified, rules  # noqa: E402
from slinky.env import BattlesnakeEnv  # noqa: E402
from slinky.types import GameConfig, RoyaleState, Ruleset, State  # noqa: E402

M_RUN = "/home/nathan/slinky-serve/az-modes-M-s0"
DIRS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# name -> (conv_channels, conv_strides, hidden, features)
NETS = {
    "S": ((8, 16, 16), (2, 2, 1), 128, "all"),
    "L": ((32, 64, 64), (2, 2, 1), 512, "all"),
    "odd_a": ((5, 12, 7), (1, 3, 2), 37, "all"),
    "odd_b": ((24, 64), (3, 1), 100, "no_mode"),
    "deep": ((8, 8, 8, 8, 8), (2, 2, 2, 2, 2), 16, "none"),
    "s4": ((16,), (4,), 64, "all"),
    "s1": ((6, 6), (1, 1), 20, "no_mode"),
}


# --- Nets -------------------------------------------------------------------------------------


def make_params(name, cfg):
    params = az_modes.init_network(jax.random.key(sum(map(ord, name))), cfg)
    rng = np.random.default_rng(sum(map(ord, name)) + 1)
    out = {}
    for layer, d in params.items():
        out[layer] = {}
        for k, v in d.items():
            v = np.asarray(v, np.float32)
            if layer in ("policy", "value", "feat") and k == "w":
                v = rng.normal(0, 0.3 if layer != "feat" else 1.0, v.shape).astype(np.float32)
            elif k == "b":
                v = rng.normal(0, 0.2, v.shape).astype(np.float32)
            out[layer][k] = jnp.asarray(v)
    return out


def nets():
    mcfg = az_modes.load_config(M_RUN)
    out = {"M": (mcfg, az_modes.load_params(M_RUN, mcfg))}
    for name, (ch, st, hid, feat) in NETS.items():
        cfg = dataclasses.replace(mcfg, conv_channels=ch, conv_strides=st, hidden=hid, features=feat)
        out[name] = (cfg, make_params(name, cfg))
    return out


def synthetic_boards(rng):
    """Boards aimed at the padding and stride arithmetic."""
    bs = []
    for (r, c) in [(0, 0), (0, 20), (20, 0), (20, 20), (10, 10), (1, 19), (19, 1), (0, 10)]:
        b = np.zeros((21, 21, 13), np.float32)
        b[r, c, :] = rng.random(13).astype(np.float32) + 0.5
        bs.append(b)
    for (r0, c0) in [(0, 0), (0, 10), (10, 0), (10, 10), (5, 3)]:
        b = np.zeros((21, 21, 13), np.float32)
        blk = (rng.random((11, 11, 13)) < 0.3).astype(np.float32)
        blk[..., 0] = 1.0
        blk[..., 2] = rng.integers(0, 6, (11, 11))
        b[r0:r0 + 11, c0:c0 + 11] = blk
        bs.append(b)
    for _ in range(6):
        bs.append(rng.normal(0, 1, (21, 21, 13)).astype(np.float32))
    b = np.ones((21, 21, 13), np.float32)
    b[..., 2] = 9.0
    bs.append(b)
    bs.append(np.zeros((21, 21, 13), np.float32))
    feats = rng.random((len(bs), 7)).astype(np.float32)
    feats[:, :4] = np.eye(4, dtype=np.float32)[rng.integers(0, 4, len(bs))]
    return np.stack(bs), feats


# --- States -----------------------------------------------------------------------------------


def walk(rng, head, length, w, h, taken, coil=True):
    """A self-avoiding body from ``head`` (head first), preferring to turn back towards the head
    so that it coils around it. May be shorter than ``length`` when stuck."""
    segs = [tuple(head)]
    own = {tuple(head)}
    while len(segs) < length:
        x, y = segs[-1]
        opts = [(x + dx, y + dy) for dx, dy in DIRS
                if 0 <= x + dx < w and 0 <= y + dy < h and (x + dx, y + dy) not in own
                and (x + dx, y + dy) not in taken]
        if not opts:
            break
        if coil:
            hx, hy = head
            opts.sort(key=lambda p: abs(p[0] - hx) + abs(p[1] - hy) + rng.random() * 1.5)
            p = opts[0]
        else:
            p = opts[rng.integers(len(opts))]
        segs.append(p)
        own.add(p)
    return segs


class Builder:
    def __init__(self, game, turn=0):
        self.g = game
        n, h, w = game.num_snakes, game.height, game.width
        self.body = np.zeros((n, h, w), np.int16)
        self.head = np.zeros((n, 2), np.int32)
        self.length = np.zeros(n, np.int32)
        self.health = np.zeros(n, np.int32)
        self.alive = np.zeros(n, bool)
        self.elim_cause = np.zeros(n, np.int8)
        self.elim_turn = np.zeros(n, np.int32)
        self.food = np.zeros((h, w), bool)
        self.hazard = np.zeros((h, w), np.int16)
        self.turn = turn
        self.sides = np.full(w + h - 1, 4, np.int8)
        self.taken = set()

    def snake(self, seat, segs, health=100):
        n = len(segs)
        for i, (x, y) in enumerate(segs):
            if 0 <= x < self.g.width and 0 <= y < self.g.height:
                self.body[seat, y, x] = max(self.body[seat, y, x], n - i)
            self.taken.add((x, y))
        self.head[seat] = segs[0]
        self.length[seat] = n
        self.health[seat] = health
        self.alive[seat] = True

    def dead(self, seat, head=(0, 0), elim_turn=0, keep_body=False, length=0):
        """A padded API seat (head (0,0), empty body) or a freshly dead one."""
        if not keep_body:
            self.body[seat] = 0
        self.head[seat] = head
        self.alive[seat] = False
        self.elim_cause[seat] = 6
        self.elim_turn[seat] = elim_turn
        self.length[seat] = length
        self.health[seat] = 0

    def state(self):
        royale = self.g.ruleset == Ruleset.ROYALE
        return State(
            body=jnp.asarray(self.body), head=jnp.asarray(self.head),
            length=jnp.asarray(self.length), health=jnp.asarray(self.health),
            alive=jnp.asarray(self.alive), last_move=jnp.zeros(self.g.num_snakes, jnp.int8),
            food=jnp.asarray(self.food), hazard=jnp.asarray(self.hazard),
            turn=jnp.int32(self.turn), elim_cause=jnp.asarray(self.elim_cause),
            elim_turn=jnp.asarray(self.elim_turn), done=jnp.asarray(self.alive.sum() <= 1),
            map_state=RoyaleState(jnp.asarray(self.sides)) if royale else (),
        )

    def record(self):
        g = self.g
        return {
            "width": g.width, "height": g.height, "num_snakes": g.num_snakes,
            "ruleset": g.ruleset.value, "shrink_every": g.shrink_every_n_turns,
            "turn": int(self.turn), "done": int(self.alive.sum() <= 1),
            "head": self.head.tolist(), "length": self.length.tolist(),
            "health": self.health.tolist(), "alive": self.alive.astype(int).tolist(),
            "elim_turn": self.elim_turn.tolist(),
            "food": [int(c) for c in np.nonzero(self.food.reshape(-1))[0]],
            "hazard": self.hazard.reshape(-1).astype(int).tolist(),
            "body": self.body.reshape(g.num_snakes, -1).astype(int).tolist(),
        }


def random_snakes(b, rng, seats, lo=3, hi=12, coil=False, heads=None):
    w, h = b.g.width, b.g.height
    for k, s in enumerate(seats):
        for _ in range(50):
            hd = heads[k] if heads else (int(rng.integers(0, w)), int(rng.integers(0, h)))
            if hd not in b.taken:
                break
        segs = walk(rng, hd, int(rng.integers(lo, hi + 1)), w, h, b.taken, coil)
        if rng.random() < 0.3:
            segs.append(segs[-1])  # stacked tail: just ate
        b.snake(s, segs, int(rng.integers(1, 101)))


def add_food(b, rng, k):
    for _ in range(k):
        b.food[rng.integers(0, b.g.height), rng.integers(0, b.g.width)] = True


def cases(rng):
    """(label, Builder) pairs."""
    m = modes.mode_config
    out = []
    duel = m("duel")
    corners = [(0, 0), (10, 0), (0, 10), (10, 10)]
    # Turn-0 duels with heads in corners: stacked 3-segment snakes.
    for i, c in enumerate(corners):
        b = Builder(duel, 0)
        b.snake(0, [c] * 3)
        b.snake(1, [corners[3 - i]] * 3)
        b.food[5, 5] = True
        b.food[c[1], 10 - c[0]] = True
        out.append((f"duel_t0_corner{i}", b))
    # Long coiled snakes with the head in a corner or on an edge, late turns.
    edge_heads = corners + [(0, 5), (10, 5), (5, 0), (5, 10), (1, 1), (9, 9)]
    for i, hd in enumerate(edge_heads):
        b = Builder(duel, [1, 57, 999, 1000, 1001, 4000, 250, 0, 998, 123456][i])
        segs = walk(rng, hd, int(rng.integers(25, 70)), 11, 11, set(), coil=True)
        b.snake(0, segs + [segs[-1]] * int(i % 2), int(rng.integers(1, 101)))
        random_snakes(b, rng, [1], 5, 30, coil=True)
        add_food(b, rng, 3)
        out.append((f"duel_coil{i}", b))
    # The opponent coiled near our head; length ties; health extremes.
    for i, hp in enumerate([0, 1, 100, 101, -3, 50]):
        b = Builder(duel, 30 + i)
        random_snakes(b, rng, [0], 4, 8, coil=False)
        segs = walk(rng, (int(b.head[0][0] + 1) % 11, int(b.head[0][1])), 40, 11, 11, b.taken, True)
        b.snake(1, segs, hp)
        if i % 2 == 0:
            b.length[1] = b.length[0]
        out.append((f"duel_oppcoil{i}", b))
    # 4p standard: deaths of every kind, own seat dead, ties, coinciding heads, stacked hazards.
    std = m("standard")
    for i in range(14):
        b = Builder(std, [0, 3, 100, 999, 1000, 1001, 2000, 7, 64, 500, 12, 1, 77, 300][i])
        random_snakes(b, rng, [0, 1, 2, 3], 3, 25, coil=bool(i % 2))
        if i % 3 == 0:
            b.dead(3)  # padded seat from the API
        if i % 4 == 1:
            b.dead(1, head=(-1, 4), elim_turn=b.turn, keep_body=True, length=int(b.length[1]))
        if i % 5 == 2:
            b.dead(2, head=(11, 11), elim_turn=3)
            b.dead(3)
        if i == 6 or i == 13:
            b.dead(0, head=(4, -1), elim_turn=b.turn, keep_body=True, length=int(b.length[0]))
        if i % 4 == 3:
            b.length[1] = b.length[0]
            b.length[2] = b.length[0] + 1
        if i in (8, 9):
            b.head[2] = b.head[1]  # two living heads on one cell
            b.health[2] = 17
        if i % 3 == 1:
            b.hazard[:] = rng.integers(0, 6, (11, 11))
        add_food(b, rng, int(rng.integers(0, 6)))
        out.append((f"std4_{i}", b))
    # 3-snake standard.
    for i in range(5):
        b = Builder(m("standard", 3), 10 * i)
        random_snakes(b, rng, [0, 1, 2], 3, 30, coil=True)
        if i == 4:
            b.dead(1)
        add_food(b, rng, 2)
        out.append((f"std3_{i}", b))
    # Royale: the zone the server shows at turns around the shrinks, several periods, 2 and 4 seats.
    for period, n in [(25, 4), (7, 4), (1, 4), (1000, 4), (25, 2)]:
        game = dataclasses.replace(m("royale", n), shrink_every_n_turns=period)
        for t in [0, 1, period - 1, period, period + 1, 2 * period - 1, 2 * period, 5 * period + 3, 999, 1000, 1003]:
            b = Builder(game, t)
            sides = rng.integers(0, 4, 21).astype(np.int8)
            b.sides[:] = sides
            b.hazard[:] = np.asarray(rules.royale_hazards(jnp.asarray(sides), jnp.int32(t), game))
            random_snakes(b, rng, list(range(n)), 3, 15, coil=bool(t % 2))
            if n == 4 and t % 3 == 0:
                b.dead(int(rng.integers(1, 4)))
            out.append((f"royale{n}_p{period}_t{t}", b))
    # Constrictor with food on the board (hidden), food under heads.
    for n in (4, 2):
        for i in range(5):
            b = Builder(m("constrictor", n), [0, 1, 50, 999, 1000][i])
            random_snakes(b, rng, list(range(n)), 3, 40, coil=True)
            add_food(b, rng, 4)
            b.food[b.head[0][1], b.head[0][0]] = True
            if n == 4 and i % 2 == 1:
                b.dead(2)
            out.append((f"constrictor{n}_{i}", b))
    # Small and rectangular boards (observation only; the nets read 11x11).
    for (w, h, n) in [(7, 7, 4), (9, 11, 3), (11, 5, 2), (1, 3, 2), (3, 1, 2), (2, 11, 2)]:
        game = GameConfig(width=w, height=h, num_snakes=n)
        b = Builder(game, 5)
        random_snakes(b, rng, list(range(n)), 1, 6, coil=True)
        add_food(b, rng, 1)
        out.append((f"board{w}x{h}", b))
    return out


def sparse_bits(a):
    flat = np.ascontiguousarray(a, np.float32).reshape(-1).view(np.uint32)
    idx = np.nonzero(flat)[0]
    return [[int(i), int(flat[i])] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    a = ap.parse_args()
    root = os.path.join(a.out, "review_netobs")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "nets"))
    rng = np.random.default_rng(2026)
    all_nets = nets()
    mcfg = all_nets["M"][0]

    # Run dirs and synthetic network cases.
    sb, sf = synthetic_boards(rng)
    for name, (cfg, params) in all_nets.items():
        d = os.path.join(root, "nets", name)
        os.makedirs(d)
        if name == "M":
            for f in ("config.json", "params.npz"):
                os.symlink(os.path.join(M_RUN, f), os.path.join(d, f))
        else:
            unified.save_config(d, "az_modes", cfg)
            unified.save_params(d, params)
        f = sf * unified._NO_MODE_KEEP if cfg.features == "no_mode" else sf
        if cfg.features == "none":
            f = np.zeros_like(sf)
        lg, v = jax.jit(lambda b, f, p=params, c=cfg: az_modes.network(p, b, f, c))(sb, f)
        np.savez(os.path.join(d, "cases.npz"), obs=sb, feats=sf, logits=np.asarray(lg, np.float32),
                 value=np.asarray(v, np.float32))
        print("net", name, unified.count_params(params), "params", flush=True)

    # End-to-end states.
    lines = []
    by_game = {}
    for label, b in cases(rng):
        by_game.setdefault(b.g, []).append((label, b))
    for game, items in by_game.items():
        mode = modes.mode_of(game)
        env = BattlesnakeEnv(game, obs="egocentric")
        inputs = jax.jit(lambda s, g=game, e=env, md=mode: unified.policy_inputs(e.observe(s), s, g, mcfg, md))
        group = unified.Group(mode, modes.MODE_NAMES.index(mode), game, env, None)
        evals = {}
        if (game.width, game.height) == (11, 11):
            evals = {name: jax.jit(az_modes.make_evaluator(p, c, group, mode)) for name, (c, p) in all_nets.items()}
        hide = int(bool(mcfg.hide_constrictor_food and game.ruleset.constrictor))
        for label, b in items:
            s = b.state()
            boards, feats = inputs(s)
            rec = {"name": label, "hide_food": hide, **b.record()}
            rec["boards"] = [sparse_bits(np.asarray(boards[i])) for i in range(game.num_snakes)]
            rec["feats"] = [int(x) for x in np.asarray(feats, np.float32).reshape(-1).view(np.uint32)]
            rec["eval"] = {}
            for name, ev in evals.items():
                v, lg = ev(s)
                rec["eval"][name] = {"value": np.asarray(v, np.float64).tolist(),
                                     "logits": np.asarray(lg, np.float64).tolist()}
            lines.append(json.dumps(rec))
        print(f"{mode} {game.width}x{game.height} n={game.num_snakes} S={game.shrink_every_n_turns}: "
              f"{len(items)} states", flush=True)
    with open(os.path.join(root, "e2e.jsonl"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("e2e states", len(lines))


if __name__ == "__main__":
    main()
