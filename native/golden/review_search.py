"""Adversarial golden vectors for the Rust search crate (native/crates/search/tests/review_search.rs).

Where ``search_golden.py`` samples typical rollout positions, this script aims at the cases most
likely to break parity: hand-built positions (forced moves of dead and certainly-starving snakes,
our own snake included; no safe move; one safe move; head-to-heads; finished roots; 4-snake
placement backups with dead seats eliminated on different turns; royale stacked hazards and food
on hazards; constrictor roots with food), exact ties settled by the noise bits or by the final
move's tie-breaks (``tie_noise=0``, constant evaluators), the depth cutoff (small ``max_depth``
and a sharp evaluator whose principal line runs to depth 32), uniformly random play (odd
positions with many dead seats), and API requests shaped like the ladder log's games (standard
on the royale map, constrictor on the empty map, 2 to 4 snakes, known or unknown ``/start``
count, a lone survivor).

Run (CPU, memory capped, through the gate), one part at a time:
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_search.py --only PART [--out DIR]

PART is one of ``crafted``, ``random``, ``ladder``, ``variants`` (comma separated for several). Output goes to
``<out>/search/`` (default out /dev/shm/naz-golden; RAM, never committed).

Files (JSON lines; f32 values as their u32 bit patterns; ``game``, ``state`` and ``out`` are
the formats of ``search_golden.py``):

* ``review_<part>.jsonl`` for parts ``crafted``, ``random`` and ``variants`` (the crafted
  positions with other ``c_puct``, ``fpu_reduction`` and ``draw_value``), one search per line:
  ``name`` (case label), ``kind`` (``mech``: ``prior="uniform", leaf="none"``, reproduce with
  the Rust ``UniformEvaluator``; ``replay``: a custom JAX evaluator, ``sharp`` or ``const``;
  ``agent``: the deployed network as ``make_policy`` plays it), ``evaluator`` (``uniform``,
  ``sharp``, ``const``, ``net``), ``config`` (name), ``game``, ``mode`` (the told mode, agent
  only), ``search`` (``sims``; ``c_puct``, ``fpu``, ``draw``, ``tie_noise`` as f32 bits;
  ``max_depth``), ``seed`` (noise of ``jax.random.key(seed)``), ``state``, ``out``, and for
  ``replay``/``agent`` lines ``node_offset`` (bytes) and ``num_nodes``: the tree's per-node
  ``value[N]`` then ``prior[N][4]`` (f32 little-endian, creation order, root first) in
  ``review_<part>_nodes.bin``.
* ``review_ladder.jsonl``: per line ``name``, ``request`` (the API JSON object),
  ``starting`` (the ``/start`` snake count passed, or null), ``seed``, and either ``skip``
  (why the Python server would not search it: no config, or fewer than 2 seats) or ``mode``,
  ``num_snakes`` and ``out`` of the agent's search on ``server.request_state(req,
  server.game_config(req, n))`` (``search_golden.part_e2e``'s path).
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nathan/slinky-release/baselines")
import search_golden as sg  # noqa: E402
import az_modes  # noqa: E402
from slinky import engine_json, mcts, modes, rules, server  # noqa: E402
from slinky.env import BattlesnakeEnv  # noqa: E402
from slinky.types import Ruleset  # noqa: E402

F32 = np.float32


def as_dict(state, game):
    """numpy dict of a ``State`` (the keys ``search_golden.state_json`` reads)."""
    s = jax.device_get(state)
    d = {f: np.asarray(getattr(s, f)) for f in sg.FIELDS}
    if game.ruleset == Ruleset.ROYALE:
        d["sides"] = np.asarray(s.map_state.sides)
    else:
        d["sides"] = np.zeros(game.width + game.height - 1, np.int8)
    return d


def snake(body, health=100, dead_turn=None, sid=None):
    s = {"id": sid or "x", "body": [list(p) for p in body], "health": health}
    if dead_turn is not None:
        s.update(eliminated_cause="snake-collision", eliminated_on_turn=dead_turn, body=[])
    return s


def build(game, turn, snakes, food=(), hazards=(), sides=None):
    """A State from an engine-style dict (``engine_json.state_from_engine``)."""
    d = {"width": game.width, "height": game.height, "turn": turn,
         "snakes": [dict(s, id=f"s{i}") for i, s in enumerate(snakes)],
         "food": [list(p) for p in food], "hazards": [list(p) for p in hazards]}
    ms = ()
    if game.ruleset == Ruleset.ROYALE:
        if sides is None:
            ms = server.royale_sides(d["hazards"], turn, game)
        else:
            from slinky.types import RoyaleState
            ms = RoyaleState(jnp.asarray(np.resize(np.asarray(sides, np.int8), game.width + game.height - 1)))
    return engine_json.state_from_engine(d, game, ms)


def crafted_positions():
    """(name, config name, State) hand-built edge cases."""
    g = sg.configs()
    duel, std, roy, con, std3 = g["duel"], g["standard"], g["royale"], g["constrictor"], g["standard3"]
    me = [(5, 5), (5, 4), (5, 3)]
    opp = [(2, 8), (3, 8), (4, 8)]
    out = []
    add = lambda name, cfg, st: out.append((name, cfg, st))  # noqa: E731
    add("duel_me_starving", "duel", build(duel, 30, [snake(me, 1), snake(opp, 80)], food=[(9, 1)]))
    add("duel_me_starving_food_adjacent", "duel",
        build(duel, 30, [snake(me, 1), snake(opp, 80)], food=[(6, 5), (9, 1)]))
    add("duel_both_starving", "duel", build(duel, 30, [snake(me, 1), snake(opp, 1)], food=[(9, 1)]))
    add("duel_opp_starving", "duel", build(duel, 30, [snake(me, 50), snake(opp, 1)]))
    add("duel_me_health2", "duel", build(duel, 30, [snake(me, 2), snake(opp, 60)], food=[(9, 9)]))
    add("duel_h2h_equal", "duel",
        build(duel, 12, [snake([(5, 5), (5, 4), (5, 3)], 90), snake([(5, 7), (5, 8), (5, 9)], 90)]))
    add("duel_h2h_me_longer", "duel",
        build(duel, 12, [snake([(5, 5), (5, 4), (5, 3), (5, 2)], 90), snake([(5, 7), (5, 8), (5, 9)], 90)]))
    add("duel_h2h_opp_longer", "duel",
        build(duel, 12, [snake([(5, 5), (5, 4), (5, 3)], 90), snake([(5, 7), (5, 8), (5, 9), (5, 10)], 90)]))
    add("duel_no_safe_move", "duel",
        build(duel, 40, [snake([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)], 70), snake(opp, 70)]))
    add("duel_one_safe_move", "duel",
        build(duel, 40, [snake([(0, 5), (0, 4), (0, 3)], 70),
                         snake([(1, 8), (1, 7), (1, 6), (1, 5), (1, 4)], 70)]))
    add("duel_turn0_stacked", "duel",
        build(duel, 0, [snake([(1, 1)] * 3), snake([(9, 9)] * 3)], food=[(0, 2), (10, 8), (5, 5)]))
    add("duel_tail_chase", "duel",
        build(duel, 50, [snake([(3, 3), (3, 4), (4, 4), (4, 3)], 90),
                         snake([(8, 8), (8, 7), (7, 7), (7, 8)], 90)]))
    add("duel_corner_trap_opp", "duel",
        build(duel, 60, [snake([(2, 1), (3, 1), (4, 1), (5, 1)], 90),
                         snake([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)], 90)]))
    # 4 seats with dead ones.
    s4a = snake([(2, 2), (2, 3), (2, 4)], 70)
    s4b = snake([(8, 8), (8, 7), (8, 6)], 70)
    s4c = snake([(8, 2), (7, 2), (6, 2)], 70)
    add("std_two_dead_padded", "standard", build(std, 80, [s4a, s4b, snake([], 0, 0), snake([], 0, 0)]))
    add("std_dead_varied_turns", "standard", build(std, 80, [s4a, snake([], 0, 9), s4b, snake([], 0, 41)]))
    add("std_me_starving_4p", "standard", build(std, 80, [dict(s4a, health=1), s4b, s4c, snake([], 0, 20)]))
    add("std_opp_starving_4p", "standard", build(std, 80, [s4a, dict(s4b, health=1), s4c, snake([], 0, 20)]))
    add("std_all_starving", "standard", build(std, 80, [dict(s, health=1) for s in
                                                        (s4a, s4b, s4c, snake([(5, 9), (4, 9), (3, 9)]))]))
    add("std_last_two_h2h", "standard",
        build(std, 33, [snake([(5, 5), (5, 4), (5, 3)], 40), snake([], 0, 7), snake([(5, 7), (5, 8), (5, 9)], 40),
                        snake([], 0, 7)]))
    add("std_three_way_h2h", "standard",
        build(std, 33, [snake([(5, 4), (5, 3), (5, 2)], 40), snake([(4, 5), (3, 5), (2, 5)], 40),
                        snake([(6, 5), (7, 5), (8, 5)], 40), snake([(5, 6), (5, 7), (5, 8)], 40)]))
    add("std_done_root_me_alone", "standard",
        build(std, 90, [s4a, snake([], 0, 50), snake([], 0, 0), snake([], 0, 0)]))
    add("std_done_root_me_alone_starving", "standard",
        build(std, 90, [dict(s4a, health=1), snake([], 0, 50), snake([], 0, 0), snake([], 0, 0)]))
    add("std3_one_dead", "standard3", build(std3, 25, [s4a, snake([], 0, 4), s4b]))
    add("std3_turn0", "standard3", build(std3, 0, [snake([(1, 1)] * 3), snake([(9, 1)] * 3), snake([(5, 9)] * 3)],
                                         food=[(0, 2), (10, 2), (4, 10), (5, 5)]))
    # Royale: stacked hazards (counts 2-3) next to heads, hazard under food, low health in hazard.
    hz = [(x, y) for x in range(11) for y in range(11) if x < 2 or y < 1]
    hz_stacked = hz + [(4, 5), (4, 5), (6, 5), (6, 5), (6, 5), (5, 6)]
    add("royale_stacked_hazards", "royale",
        build(roy, 52, [snake(me, 30), snake([(8, 8), (8, 7), (8, 6)], 15), snake([(2, 2), (3, 2), (4, 2)], 14),
                        snake([], 0, 30)], food=[(5, 6)], hazards=hz_stacked))
    add("royale_low_health_in_zone", "royale",
        build(roy, 75, [snake([(1, 5), (2, 5), (3, 5)], 13), snake([(0, 3), (0, 4), (0, 5)], 15),
                        snake([(8, 8), (8, 7), (8, 6)], 60), snake([(6, 0), (7, 0), (8, 0)], 9)],
              food=[(0, 2), (1, 6)], hazards=hz))
    for t in (24, 25, 26, 49, 50):
        S = roy.shrink_every_n_turns
        sides = [0, 2, 1, 3] * 6
        zone = np.asarray(rules.royale_hazards(jnp.asarray(np.asarray(sides[:21], np.int8)), jnp.int32(t), roy))
        ys, xs = np.nonzero(zone)
        add(f"royale_turn{t}", "royale",
            build(roy, t, [snake([(5, 5), (5, 4), (5, 3)], 20), snake([(1, 8), (2, 8), (3, 8)], 20),
                           snake([(9, 2), (9, 3), (9, 4)], 5), snake([(9, 9), (8, 9), (7, 9)], 2)],
                  hazards=list(zip(xs.tolist(), ys.tolist())), sides=sides[: max(0, t // S)] + [4] * 21))
    add("royale2_duel_zone", "royale2",
        build(g["royale2"], 101, [snake([(5, 5), (5, 4), (5, 3), (4, 3)], 25), snake([(6, 7), (6, 8), (7, 8)], 20)],
              hazards=[(x, y) for x in range(11) for y in range(11) if x < 3 or x > 8 or y < 2 or y > 9]))
    add("constrictor_root_food", "constrictor",
        build(con, 15, [snake([(5, 5), (5, 4), (5, 3), (5, 3)], 100), snake([(2, 8), (3, 8), (4, 8)], 100),
                        snake([(8, 2), (8, 3), (8, 4)], 100), snake([], 0, 6)], food=[(6, 5), (2, 7)]))
    add("constrictor_cramped", "constrictor",
        build(con, 70, [snake([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 1)], 100),
                        snake([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)], 100),
                        snake([(10, 10), (9, 10), (8, 10)], 100), snake([(10, 0), (10, 1), (10, 2)], 100)]))
    return out


# --- Evaluators -------------------------------------------------------------------------------


def sharp_evaluator(game):
    """A mask-aware evaluator with a strong prior on one safe move and high, slowly varying
    values: the tree follows one line deep (down to the depth cutoff)."""
    n = game.num_snakes

    def ev(state):
        mask = rules.action_mask(state, game)
        h = state.head
        pref = (h[:, 0] * 3 + h[:, 1] * 5 + state.turn) % 4
        rank = (4 - ((jnp.arange(4)[None, :] - pref[:, None]) % 4)).astype(jnp.float32)
        logits = jnp.where(mask, rank * 3.0, -4.0)
        value = 0.55 + 0.004 * state.health.astype(jnp.float32) - 0.01 * (jnp.arange(n) == 1)
        return value.astype(jnp.float32), logits.astype(jnp.float32)

    return ev


def const_evaluator(game, v=0.5):
    """Equal logits and one exact value everywhere: visits split evenly, so the final move's
    tie-breaks (equal visits, equal q, lowest index) decide."""
    n = game.num_snakes

    def ev(state):
        return jnp.full((n,), v, jnp.float32), jnp.zeros((n, 4), jnp.float32)

    return ev


# --- Runners ---------------------------------------------------------------------------------


class Writer:
    def __init__(self, out_dir, part):
        self.f = open(os.path.join(out_dir, f"review_{part}.jsonl"), "w")
        self.nodes = open(os.path.join(out_dir, f"review_{part}_nodes.bin"), "wb")
        self.offset = 0
        self.count = 0

    def write(self, rec, used=None, value=None, prior=None, n=None):
        if used is not None:
            idx = np.nonzero(np.asarray(used))[0]
            rows = np.concatenate([np.asarray(value)[idx].reshape(len(idx), n),
                                   np.asarray(prior)[idx].reshape(len(idx), n * 4)], axis=1)
            data = np.asarray(rows, "<f4").tobytes()
            self.nodes.write(data)
            rec["node_offset"] = self.offset
            rec["num_nodes"] = int(len(idx))
            assert len(idx) == rec["out"]["nodes_used"], (len(idx), rec["out"]["nodes_used"])
            self.offset += len(data)
        self.f.write(json.dumps(rec) + "\n")
        self.count += 1

    def close(self):
        self.f.close()
        self.nodes.close()


_cache = {}


def runner(kind, cfgname, game, sims, max_depth=32, tie_noise=0.01, c_puct=0.5, fpu=0.2, draw=-0.5,
           agent=None):
    """A jitted ``run(key, state) -> (out, used, value, prior)`` and its search settings."""
    k = (kind, cfgname, sims, max_depth, tie_noise, c_puct, fpu, draw)
    if k in _cache:
        return _cache[k]
    mode = None
    if kind == "uniform":
        mcfg = mcts.MCTSConfig(num_simulations=sims, selection="puct", prior="uniform", leaf="none",
                               value="placement", c_puct=c_puct, fpu_reduction=fpu, draw_value=draw,
                               tie_noise=tie_noise, max_depth=max_depth)
        run = sg.searcher(game, mcfg, None)
    elif kind == "net":
        cfg, params = agent
        run, _, mode = sg.agent_fns(cfg, params, game)
        mcfg = az_modes.agent_search(cfg, sims)
        assert sims == 256 and max_depth == 32 and tie_noise == 0.01
        c_puct, fpu, draw = cfg.c_puct, cfg.fpu_reduction, cfg.draw_value
    else:
        ev = sharp_evaluator(game) if kind == "sharp" else const_evaluator(game)
        mcfg = mcts.MCTSConfig(num_simulations=sims, selection="puct", prior="network", leaf="network",
                               value="placement", c_puct=c_puct, fpu_reduction=fpu, draw_value=draw,
                               tie_noise=tie_noise, max_depth=max_depth)
        run = sg.searcher(game, mcfg, ev)
    settings = {"sims": sims, "c_puct": sg.f32bits(c_puct), "fpu": sg.f32bits(fpu), "draw": sg.f32bits(draw),
                "tie_noise": sg.f32bits(tie_noise), "max_depth": max_depth}
    _cache[k] = (run, settings, mode)
    return _cache[k]


def search_case(w, name, cfgname, state, kind, seed, **kw):
    game = sg.configs()[cfgname]
    run, settings, mode = runner(kind, cfgname, game, **kw)
    o, used, value, prior = jax.device_get(run(jax.random.key(seed), state))
    rec = {"name": name, "kind": {"uniform": "mech", "net": "agent"}.get(kind, "replay"), "evaluator": kind,
           "config": cfgname, "game": sg.game_json(game), "mode": mode, "search": settings, "seed": seed,
           "state": sg.state_json(as_dict(state, game), game), "out": sg.out_json(o)}
    if kind == "uniform":
        w.write(rec)
    else:
        w.write(rec, used, value, prior, game.num_snakes)
    return o


def part_crafted(out_dir):
    agent = sg.load_agent()
    w = Writer(out_dir, "crafted")
    rng = np.random.default_rng(77)
    seeds = lambda k: [int(x) for x in rng.integers(0, 2**32, k)]  # noqa: E731
    for name, cfgname, state in crafted_positions():
        t0 = time.time()
        for s in seeds(3):
            search_case(w, name, cfgname, state, "uniform", s, sims=256)
        for s in seeds(2):
            search_case(w, name, cfgname, state, "uniform", s, sims=256, tie_noise=0.0)
        for md in (1, 2, 3, 6):
            search_case(w, name, cfgname, state, "uniform", seeds(1)[0], sims=256, max_depth=md)
        search_case(w, name, cfgname, state, "uniform", seeds(1)[0], sims=40, max_depth=32)
        for s in seeds(2):
            search_case(w, name, cfgname, state, "sharp", s, sims=256)
        search_case(w, name, cfgname, state, "sharp", seeds(1)[0], sims=256, max_depth=5)
        search_case(w, name, cfgname, state, "sharp", seeds(1)[0], sims=256, tie_noise=0.0)
        search_case(w, name, cfgname, state, "const", seeds(1)[0], sims=256)
        search_case(w, name, cfgname, state, "const", seeds(1)[0], sims=256, tie_noise=0.0)
        for s in seeds(4):
            search_case(w, name, cfgname, state, "net", s, sims=256, agent=agent)
        print(f"crafted {name}: {time.time() - t0:.1f}s", flush=True)
    w.close()
    print("crafted:", w.count, "searches", flush=True)


VARIANTS = [  # (c_puct, fpu_reduction, draw_value): constants a retrained run's config.json may hold
    (1.1, 0.33, -0.45), (0.3, 0.0, -1 / 3), (0.75, 0.5, -0.3), (0.5, 0.2, 0.25), (0.5, 0.2, -0.7)]


def part_variants(out_dir):
    """Crafted positions searched with other search constants (``c_puct``, ``fpu_reduction``,
    ``draw_value``), with the uniform and the sharp evaluators."""
    w = Writer(out_dir, "variants")
    rng = np.random.default_rng(91)
    for name, cfgname, state in crafted_positions():
        for c, fpu, draw in VARIANTS:
            kw = dict(sims=256, c_puct=c, fpu=fpu, draw=draw)
            search_case(w, name, cfgname, state, "uniform", int(rng.integers(0, 2**32)), **kw)
            search_case(w, name, cfgname, state, "sharp", int(rng.integers(0, 2**32)), **kw)
    w.close()
    print("variants:", w.count, "searches", flush=True)


def random_play(game, n_games, turns, seed):
    """States of uniformly random legal play (``action_mask``), done states included."""
    env = BattlesnakeEnv(game, obs=None)

    def body(carry, key):
        state, mask = carry
        k1, k2 = jax.random.split(key)
        a = jax.random.categorical(k1, jnp.where(mask, 0.0, -jnp.inf))
        ns, nts = env.step(k2, state, a)
        return (ns, nts.action_mask), ns

    def one(key):
        s, ts = env.reset(key)
        _, traj = jax.lax.scan(body, (s, ts.action_mask), jax.random.split(key, turns))
        return traj

    traj = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(seed), n_games))
    return jax.tree.map(lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), traj)


def part_random(out_dir):
    from slinky.types import RoyaleState, State
    agent = sg.load_agent()
    w = Writer(out_dir, "random")
    rng = np.random.default_rng(31)
    plan = [("duel", 60), ("standard", 60), ("royale", 50), ("constrictor", 40), ("standard3", 30),
            ("royale2", 20), ("constrictor2", 20)]
    for i, (cfgname, count) in enumerate(plan):
        t0 = time.time()
        game = sg.configs()[cfgname]
        traj = random_play(game, 16, 60 if game.ruleset != Ruleset.ROYALE else 120, 500 + i)
        n_states = len(traj.turn)
        alive_n = np.asarray(traj.alive).sum(-1)
        done = np.asarray(traj.done)
        # Prefer positions with dead seats and few alive (odd placements), plus some finished.
        weight = np.where(done, 0.3, 1.0) * (1.0 + (alive_n < game.num_snakes))
        weight[np.asarray(traj.turn) == 0] = 0.0
        pick = rng.choice(n_states, count, replace=False, p=weight / weight.sum())
        for j in pick:
            st = jax.tree.map(lambda x: x[j], traj)
            d = as_dict(st, game)
            ms = RoyaleState(jnp.asarray(d["sides"])) if game.ruleset == Ruleset.ROYALE else ()
            state = State(**{f: jnp.asarray(d[f]) for f in sg.FIELDS}, map_state=ms)
            nm = f"random_{cfgname}_{int(j)}"
            search_case(w, nm, cfgname, state, "uniform", int(rng.integers(0, 2**32)), sims=256)
            search_case(w, nm, cfgname, state, "sharp", int(rng.integers(0, 2**32)), sims=256)
            search_case(w, nm, cfgname, state, "net", int(rng.integers(0, 2**32)), sims=256, agent=agent)
        print(f"random {cfgname}: {count} positions, {time.time() - t0:.1f}s", flush=True)
    w.close()
    print("random:", w.count, "searches", flush=True)


# --- Ladder-shaped API requests -----------------------------------------------------------------


def ladder_request(d, game, you, ruleset, game_map, damage):
    req = sg.api_request(d, game, you, game_id=f"g{int(d['turn'])}-{you}")
    req["game"]["ruleset"]["name"] = ruleset
    req["game"]["map"] = game_map
    req["game"]["ruleset"]["settings"]["hazardDamagePerTurn"] = damage
    req["game"]["source"] = "arena"
    return req


def part_ladder(out_dir):
    cfg, params = sg.load_agent()
    games = sg.configs()
    # (label, rollout config, API ruleset, API map, hazard damage, count) as in moves.jsonl.
    plan = [("standard_standard_2", "duel", "standard", "standard", 100, 70),
            ("standard_royale_4", "royale", "standard", "royale", 14, 40),
            ("standard_royale_2", "royale2", "standard", "royale", 14, 30),
            ("standard_standard_4", "standard", "standard", "standard", 100, 30),
            ("standard_standard_3", "standard3", "standard", "standard", 100, 25),
            ("constrictor_empty_4", "constrictor", "constrictor", "empty", 100, 25),
            ("constrictor_empty_2", "constrictor2", "constrictor", "empty", 100, 15)]
    rng = np.random.default_rng(57)
    fns = {}
    path = os.path.join(out_dir, "review_ladder.jsonl")
    count = 0
    with open(path, "w") as f:
        for i, (label, cfgname, rs, mp, dmg, n_cases) in enumerate(plan):
            t0 = time.time()
            game = games[cfgname]
            pos = sg.positions(label, game, n_cases, 7000 + i, False, 0.3, 1.0)
            for d in pos:
                alive = np.nonzero(d["alive"])[0]
                if len(alive) == 0:
                    continue
                you = int(rng.choice(alive))
                req = ladder_request(d, game, you, rs, mp, dmg)
                for starting in (game.num_snakes, None):
                    n = len(req["board"]["snakes"]) if starting is None else starting
                    seed = int(rng.integers(0, 2**32))
                    rec = {"name": label, "request": req, "starting": starting, "seed": seed}
                    config = server.game_config(req, n)
                    if config is None or config.num_snakes < 2:
                        rec["skip"] = "no config" if config is None else "fewer than 2 seats"
                    else:
                        state = server.request_state(req, config)
                        if config not in fns:
                            fns[config] = sg.agent_fns(cfg, params, config)
                        run, _, mode = fns[config]
                        o, *_ = run(jax.random.key(seed), state)
                        rec.update(mode=mode, num_snakes=config.num_snakes, out=sg.out_json(o))
                    f.write(json.dumps(rec) + "\n")
                    count += 1
            print(f"ladder {label}: {time.time() - t0:.1f}s", flush=True)
    print("ladder:", count, "requests ->", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="crafted,random,ladder,variants")
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, "search")
    os.makedirs(out_dir, exist_ok=True)
    for part in a.only.split(","):
        t0 = time.time()
        {"crafted": part_crafted, "random": part_random, "ladder": part_ladder,
         "variants": part_variants}[part](out_dir)
        print(f"{part} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
