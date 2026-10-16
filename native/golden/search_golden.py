"""Golden vectors for the Rust search crate (native/crates/search).

Run (CPU, memory capped, through the gate), one part at a time:
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/search_golden.py --only PART [--scale F] [--out DIR]

PART is one of ``noise``, ``mech``, ``agent``, ``e2e`` (comma separated for several).
``--scale`` multiplies the case counts (default 1). Output goes to ``<out>/search/`` (default
out /dev/shm/naz-golden; RAM, never committed).

Positions come from rollouts of ``BattlesnakeEnv`` (``rules_golden.rollouts``: the heuristic
agent and a food-seeking policy with some random moves), some perturbed by
``rules_golden.perturb`` (low health, food beside heads, royale turns at shrink boundaries).
Royale positions get the shrink sides the server rebuilds from the hazards
(``server.royale_sides``: unknown future shrinks), except for some mechanics cases. Configs are
the server's (``modes.mode_config``, ``max_turns=None``): ``duel``, ``standard`` (4p),
``royale`` (4p), ``constrictor`` (4p), ``standard3``, ``royale2``, ``constrictor2``.

Every search uses the key ``jax.random.key(seed)`` (the case's ``seed``), whose noise bits
the Rust ``JaxNoise::from_seed(seed)`` reproduces (checked by part ``noise``).

Files (all JSON lines unless said otherwise; f32 values are given as their u32 bit patterns):

* ``noise.jsonl``: per line ``seed``, ``key`` (``jax.random.key_data(key(seed))``, 2 u32),
  ``split`` (``key_data(split(key, 4))``, 4 x 2 u32), ``fold`` (``[data, k0, k1]`` for
  ``fold_in(key, data)``), ``n`` and ``bits`` (the flat ``mcts._search_keys(key, cfg, n).noise``
  for ``sims`` simulations, u32[sims * n * 4]), ``sims``.
* ``mech.jsonl``: the mechanics test of SPEC.md 7.5: ``mcts.search`` with ``prior="uniform",
  leaf="none", value="placement"``. Per line: ``config`` (name), ``game`` (width, height,
  num_snakes, ruleset, hazard_damage, shrink_every), ``search`` (sims, c_puct, fpu, draw as
  f32 bits), ``seed``, ``state`` (below) and ``out`` (below).
* ``agent.jsonl`` + ``agent_nodes.bin``: the real agent (``az_modes.agent_search(cfg, 256)``
  with ``make_evaluator``, as ``make_policy`` plays) on the deployed run
  /home/nathan/slinky-serve/az-modes-M-s0. Lines as in ``mech.jsonl`` plus ``root_logits``
  (f32 bits [N][4]) and ``root_values`` (f32 bits [N]): the evaluator's raw output at the
  root; ``node_offset`` and ``num_nodes``: the case's records in ``agent_nodes.bin``, one per
  tree node in creation order (root first), each ``f32 value[N]`` then ``f32 prior[N][4]``
  (little-endian; ``tree.value`` and ``tree.prior`` of the node), ``node_offset`` counting
  records of that case's N (bytes = sum of earlier cases' ``num_nodes * N * 5 * 4``; the
  offset is given in bytes).
* ``e2e.jsonl``: API requests built from agent positions (``engine_json.state_to_engine``
  and the API request layout of tests/test_server.py, from a random living seat's side).
  Per line: ``request`` (the JSON object), ``starting`` (the snake count passed as the
  ``/start`` count, or null for the living snakes), ``seed``, ``mode``, ``num_snakes`` and
  ``out`` of the agent's search on ``server.request_state(req, server.game_config(req, n))``.

``state`` fields: turn, done (0/1), head [[x, y]] per seat, length, health, alive (0/1),
last_move, elim_cause, elim_turn (lists per seat), sides (W+H-1 ints), food (list of cell
indices ``y * W + x``), hazard (H*W counts), body (per seat H*W countdowns).

``out`` fields (``mcts.SearchOutput``): visits [N][4], q [N][4] bits, policy [N][4] bits,
value [N] bits, root_prior [N][4] bits, action [N], nodes_used, depth.
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nathan/slinky-release/baselines")
import rules_golden as rg  # noqa: E402
import az_modes  # noqa: E402
from slinky import engine_json, mcts, modes, server  # noqa: E402
from slinky.env import BattlesnakeEnv  # noqa: E402
from slinky.types import RoyaleState, Ruleset, State  # noqa: E402

RUN = "/home/nathan/slinky-serve/az-modes-M-s0"
FIELDS = rg.FIELDS


def configs():
    m = modes.mode_config
    return {
        "duel": m("duel"), "standard": m("standard"), "royale": m("royale"),
        "constrictor": m("constrictor"), "standard3": m("standard", 3),
        "royale2": m("royale", 2), "constrictor2": m("constrictor", 2),
    }


def f32bits(x):
    return np.asarray(x, np.float32).view(np.uint32).tolist()


def mk(d, game):
    ms = RoyaleState(jnp.asarray(d["sides"])) if game.ruleset == Ruleset.ROYALE else ()
    return State(**{f: jnp.asarray(d[f]) for f in FIELDS}, map_state=ms)


def state_json(d, game):
    n, w = game.num_snakes, game.width
    ys, xs = np.nonzero(d["food"])
    return {
        "turn": int(d["turn"]), "done": int(d["done"]),
        "head": np.asarray(d["head"]).astype(int).tolist(),
        "length": np.asarray(d["length"]).astype(int).tolist(),
        "health": np.asarray(d["health"]).astype(int).tolist(),
        "alive": np.asarray(d["alive"]).astype(int).tolist(),
        "last_move": np.asarray(d["last_move"]).astype(int).tolist(),
        "elim_cause": np.asarray(d["elim_cause"]).astype(int).tolist(),
        "elim_turn": np.asarray(d["elim_turn"]).astype(int).tolist(),
        "sides": np.asarray(d["sides"]).astype(int).tolist(),
        "food": sorted(int(y) * w + int(x) for y, x in zip(ys, xs)),
        "hazard": np.asarray(d["hazard"]).reshape(-1).astype(int).tolist(),
        "body": [np.asarray(d["body"][s]).reshape(-1).astype(int).tolist() for s in range(n)],
    }


def game_json(game):
    return {"width": game.width, "height": game.height, "num_snakes": game.num_snakes,
            "ruleset": game.ruleset.value, "hazard_damage": game.hazard_damage_per_turn,
            "shrink_every": game.shrink_every_n_turns}


def out_json(out):
    o = jax.device_get(out)
    return {"visits": np.asarray(o.visits).tolist(), "q": f32bits(o.q), "policy": f32bits(o.policy),
            "value": f32bits(o.value), "root_prior": f32bits(o.root_prior),
            "action": np.asarray(o.action).tolist(), "nodes_used": int(o.nodes_used),
            "depth": int(o.depth)}


def server_sides(d, game):
    hz = np.asarray(d["hazard"])
    ys, xs = np.nonzero(hz)
    rs = server.royale_sides([[int(x), int(y)] for x, y in zip(xs, ys)], int(d["turn"]), game)
    return np.asarray(rs.sides)


def positions(name, game, count, seed, live_only, perturb_frac, server_frac):
    """``count`` numpy state dicts from rollouts of ``game``."""
    rng = np.random.default_rng(seed)
    turns = 300 if game.ruleset != Ruleset.CONSTRICTOR else 150
    n_games = max(8, min(32, count // 20))
    traj = rg.to_numpy(rg.rollouts(game, n_games, turns, seed), game)
    live = np.nonzero(~traj["done"])[0]
    done = np.nonzero(traj["done"])[0]
    pick = rng.choice(live, count, replace=count > len(live))
    if not live_only and len(done):
        k = max(1, count // 50)
        pick[:k] = rng.choice(done, k)
    d = rg.take(traj, pick)
    sel = rng.random(count) < perturb_frac
    if sel.any():
        sub = rg.take(d, np.nonzero(sel)[0])
        opts = {"royale_turns": game.ruleset == Ruleset.ROYALE,
                "no_low_health": game.ruleset == Ruleset.CONSTRICTOR}
        rg.perturb(sub, game, opts, rng)
        for k, v in sub.items():
            d[k][np.nonzero(sel)[0]] = v
    out = []
    for i in range(count):
        s = {k: v[i] for k, v in d.items()}
        if game.ruleset == Ruleset.ROYALE and rng.random() < server_frac:
            s["sides"] = server_sides(s, game)
        # Keep ``done`` consistent with the perturbed state (perturb never kills).
        out.append(s)
    print(f"{name}: {count} positions from {len(traj['turn'])} rollout states", flush=True)
    return out


def searcher(game, mcfg, evaluator):
    senv = BattlesnakeEnv(game, obs=None)

    def run(key, s):
        tree, deepest, expanded, kf = mcts._run(key, s, senv, mcfg, evaluator)
        out = mcts._output(kf, tree, mcfg, deepest, expanded)
        m = tree.terminal.shape[0]
        ch = tree.children.reshape(-1)
        used = jnp.zeros(m, bool).at[jnp.where(ch >= 0, ch, m)].set(True, mode="drop")
        used = used.at[0].set(True)
        return out, used, tree.value, tree.prior

    return jax.jit(run)


# --- Parts ------------------------------------------------------------------------------------


def part_noise(out_dir):
    path = os.path.join(out_dir, "noise.jsonl")
    with open(path, "w") as f:
        for seed in (0, 1, 2, 7, 123, 4096, 2**31 - 1, 2**31 + 5, 2**32 - 1):
            key = jax.random.key(seed)
            kd = np.asarray(jax.random.key_data(key)).tolist()
            sp = np.asarray(jax.random.key_data(jax.random.split(key, 4))).tolist()
            folds = []
            for data in (0, 1, 99, 2**32 - 1):
                folds.append([data] + np.asarray(jax.random.key_data(jax.random.fold_in(key, data))).tolist())
            for n in (2, 3, 4):
                sims = 64
                bits = mcts._search_keys(key, mcts.MCTSConfig(num_simulations=sims), n).noise
                f.write(json.dumps({"seed": seed, "key": kd, "split": sp, "fold": folds, "n": n,
                                    "sims": sims, "bits": np.asarray(bits).reshape(-1).tolist()}) + "\n")
    print("noise:", path, flush=True)


def part_mech(out_dir, scale):
    games = configs()
    plan = [  # (config, count, sims, c_puct, fpu, draw)
        ("duel", 600, 256, 0.5, 0.2, -0.5), ("standard", 500, 256, 0.5, 0.2, -0.5),
        ("royale", 400, 256, 0.5, 0.2, -0.5), ("constrictor", 400, 256, 0.5, 0.2, -0.5),
        ("standard3", 200, 256, 0.5, 0.2, -0.5), ("royale2", 200, 256, 0.5, 0.2, -0.5),
        ("constrictor2", 200, 256, 0.5, 0.2, -0.5),
        ("duel", 120, 20, 0.5, 0.2, -0.5), ("standard", 120, 20, 0.5, 0.2, -0.5),
        ("duel", 120, 128, 1.25, 0.0, -0.3), ("standard", 120, 96, 0.25, 0.35, 0.0),
    ]
    rng = np.random.default_rng(5)
    path = os.path.join(out_dir, "mech.jsonl")
    total = 0
    with open(path, "w") as f:
        for i, (name, count, sims, c, fpu, draw) in enumerate(plan):
            t0 = time.time()
            game = games[name]
            count = max(1, int(count * scale))
            mcfg = mcts.MCTSConfig(num_simulations=sims, selection="puct", prior="uniform",
                                   leaf="none", value="placement", c_puct=c, fpu_reduction=fpu,
                                   draw_value=draw)
            run = searcher(game, mcfg, None)
            pos = positions(name, game, count, 1000 + i, False, 0.3, 0.5)
            for d in pos:
                seed = int(rng.integers(0, 2**32))
                o, *_ = run(jax.random.key(seed), mk(d, game))
                f.write(json.dumps({
                    "config": name, "game": game_json(game),
                    "search": {"sims": sims, "c_puct": f32bits(c), "fpu": f32bits(fpu), "draw": f32bits(draw)},
                    "seed": seed, "state": state_json(d, game), "out": out_json(o)}) + "\n")
            total += count
            print(f"mech {name} sims {sims}: {count} cases, {time.time() - t0:.1f}s", flush=True)
    print("mech:", total, "cases ->", path, flush=True)


def load_agent():
    cfg = az_modes.load_config(RUN)
    params = az_modes.load_params(RUN, cfg)
    return cfg, params


def agent_fns(cfg, params, game):
    from slinky import unified
    mode = unified.told_mode(cfg, game, None)
    group = unified.Group(mode or "none", -1, game, BattlesnakeEnv(game, obs="egocentric"), None)
    ev = az_modes.make_evaluator(params, cfg, group, mode)
    mcfg = az_modes.agent_search(cfg, 256)
    return searcher(game, mcfg, ev), jax.jit(ev), mode


def part_agent(out_dir, scale):
    cfg, params = load_agent()
    games = configs()
    plan = [("duel", 1000), ("standard", 400), ("royale", 300), ("constrictor", 300),
            ("standard3", 100), ("royale2", 100), ("constrictor2", 100)]
    rng = np.random.default_rng(9)
    path = os.path.join(out_dir, "agent.jsonl")
    nodes_f = open(os.path.join(out_dir, "agent_nodes.bin"), "wb")
    offset = 0
    total = 0
    with open(path, "w") as f:
        for i, (name, count) in enumerate(plan):
            t0 = time.time()
            game = games[name]
            n = game.num_snakes
            count = max(1, int(count * scale))
            run, ev, mode = agent_fns(cfg, params, game)
            pos = positions(name, game, count, 2000 + i, True, 0.15, 1.0)
            policy = jax.jit(lambda k, s, p=az_modes.make_policy(params, cfg, game, mode, True, simulations=256): p(k, s, None))
            for j, d in enumerate(pos):
                seed = int(rng.integers(0, 2**32))
                key = jax.random.key(seed)
                s = mk(d, game)
                o, used, value, prior = jax.device_get(run(key, s))
                if j < 3:  # the search is make_policy's
                    assert np.array_equal(np.asarray(policy(key, s)), np.asarray(o.action)), name
                rv, rl = jax.device_get(ev(s))
                idx = np.nonzero(np.asarray(used))[0]
                rec = np.concatenate([np.asarray(value)[idx].reshape(len(idx), n),
                                      np.asarray(prior)[idx].reshape(len(idx), n * 4)], axis=1)
                nodes_f.write(np.asarray(rec, "<f4").tobytes())
                f.write(json.dumps({
                    "config": name, "game": game_json(game), "mode": mode,
                    "search": {"sims": 256, "c_puct": f32bits(cfg.c_puct), "fpu": f32bits(cfg.fpu_reduction),
                               "draw": f32bits(cfg.draw_value)},
                    "seed": seed, "state": state_json(d, game), "out": out_json(o),
                    "root_logits": f32bits(rl), "root_values": f32bits(rv),
                    "node_offset": offset, "num_nodes": int(len(idx))}) + "\n")
                assert len(idx) == int(o.nodes_used)
                offset += rec.size * 4
            total += count
            print(f"agent {name}: {count} cases, {time.time() - t0:.1f}s", flush=True)
    nodes_f.close()
    print("agent:", total, "cases ->", path, flush=True)


def api_request(d, game, you, game_id="g"):
    """The API move request of state ``d`` from seat ``you``'s side (tests/test_server.py)."""
    eng = engine_json.state_to_engine(mk(d, game), game)
    snakes = [{"id": s["id"], "name": s["id"], "health": s["health"],
               "body": [{"x": x, "y": y} for x, y in s["body"]],
               "head": {"x": s["body"][0][0], "y": s["body"][0][1]}, "length": len(s["body"])}
              for s in eng["snakes"] if not s.get("eliminated_cause")]
    me = next(s for s in snakes if s["id"] == f"s{you}")
    settings = {"foodSpawnChance": game.food_spawn_chance, "minimumFood": game.minimum_food,
                "hazardDamagePerTurn": game.hazard_damage_per_turn,
                "royale": {"shrinkEveryNTurns": game.shrink_every_n_turns}}
    return {
        "game": {"id": game_id, "ruleset": {"name": game.ruleset.value, "version": "v1.2.3",
                                           "settings": settings},
                 "map": game.map, "source": "custom", "timeout": 500},
        "turn": int(d["turn"]),
        "board": {"height": game.height, "width": game.width,
                  "food": [{"x": x, "y": y} for x, y in eng["food"]],
                  "hazards": [{"x": x, "y": y} for x, y in eng["hazards"]], "snakes": snakes},
        "you": me,
    }


def part_e2e(out_dir, scale):
    cfg, params = load_agent()
    games = configs()
    plan = [("duel", 80), ("standard", 50), ("royale", 40), ("constrictor", 40)]
    rng = np.random.default_rng(13)
    fns = {}
    path = os.path.join(out_dir, "e2e.jsonl")
    total = 0
    with open(path, "w") as f:
        for i, (name, count) in enumerate(plan):
            t0 = time.time()
            game = games[name]
            count = max(1, int(count * scale))
            pos = positions(name, game, count, 3000 + i, True, 0.0, 1.0)
            for d in pos:
                alive = np.nonzero(d["alive"])[0]
                you = int(rng.choice(alive))
                req = api_request(d, game, you)
                # Mostly the /start count; sometimes unknown (the living snakes).
                starting = None if rng.random() < 0.25 else game.num_snakes
                n = len(req["board"]["snakes"]) if starting is None else starting
                config = server.game_config(req, n)
                if config is None or config.num_snakes < 2:
                    continue
                state = server.request_state(req, config)
                if config not in fns:
                    fns[config] = agent_fns(cfg, params, config)
                run, _, mode = fns[config]
                seed = int(rng.integers(0, 2**32))
                o, *_ = run(jax.random.key(seed), state)
                f.write(json.dumps({"request": req, "starting": starting, "seed": seed, "mode": mode,
                                    "num_snakes": config.num_snakes, "out": out_json(o)}) + "\n")
                total += 1
            print(f"e2e {name}: {time.time() - t0:.1f}s", flush=True)
    print("e2e:", total, "requests ->", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="noise,mech,agent,e2e")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, "search")
    os.makedirs(out_dir, exist_ok=True)
    for part in a.only.split(","):
        t0 = time.time()
        if part == "noise":
            part_noise(out_dir)
        elif part == "mech":
            part_mech(out_dir, a.scale)
        elif part == "agent":
            part_agent(out_dir, a.scale)
        elif part == "e2e":
            part_e2e(out_dir, a.scale)
        else:
            raise SystemExit(f"unknown part {part}")
        print(f"{part} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
