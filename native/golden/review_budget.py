"""Golden vectors for the review of the long (1024+ simulation) searches of the Rust search crate.

The faithful mode is checked at 256 simulations by ``search_golden.py``. The time-budget work
made larger fixed simulation counts (1024, 4096) the reproducible mode for strength tests and
claims that "nothing in the tree is sized for 256 simulations". This script writes Python's
``mcts.search`` at those counts so the Rust tests (``crates/search/tests/review_budget.rs``) can
check that a fixed-simulation Rust search is still exactly Python's search with
``simulations=N``.

Run (CPU, memory capped, through the gate), one part at a time:
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_budget.py --only PART [--scale F] [--out DIR]

PART is ``mech`` or ``agent`` (comma separated for both). Output goes to ``<out>/search/``
(default /dev/shm/naz-golden): ``review_long_mech.jsonl`` and ``review_long_agent.jsonl`` +
``review_long_agent_nodes.bin``, in exactly the formats of ``mech.jsonl`` and ``agent.jsonl`` +
``agent_nodes.bin`` (see ``search_golden.py``), with ``search.sims`` set to the long count.
"""
import argparse, json, os, sys, time
import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_golden as sg  # noqa: E402
import az_modes  # noqa: E402  (search_golden put baselines on the path)
from slinky import mcts  # noqa: E402


def part_mech(out_dir, scale):
    games = sg.configs()
    plan = [  # (config, count, sims)
        ("duel", 24, 1024), ("standard", 16, 1024), ("royale", 12, 1024),
        ("constrictor", 12, 1024), ("standard3", 8, 1024), ("royale2", 8, 1024),
        ("constrictor2", 8, 1024),
        ("duel", 6, 4096), ("standard", 4, 4096), ("royale", 4, 4096), ("constrictor", 4, 4096),
        ("duel", 6, 777), ("standard", 4, 1500),
    ]
    rng = np.random.default_rng(77)
    path = os.path.join(out_dir, "review_long_mech.jsonl")
    total = 0
    with open(path, "w") as f:
        for i, (name, count, sims) in enumerate(plan):
            t0 = time.time()
            game = games[name]
            count = max(1, int(count * scale))
            c, fpu, draw = 0.5, 0.2, -0.5
            mcfg = mcts.MCTSConfig(num_simulations=sims, selection="puct", prior="uniform",
                                   leaf="none", value="placement", c_puct=c, fpu_reduction=fpu,
                                   draw_value=draw)
            run = sg.searcher(game, mcfg, None)
            pos = sg.positions(name, game, count, 7000 + i, True, 0.3, 0.5)
            for d in pos:
                seed = int(rng.integers(0, 2**32))
                o, *_ = run(jax.random.key(seed), sg.mk(d, game))
                f.write(json.dumps({
                    "config": name, "game": sg.game_json(game),
                    "search": {"sims": sims, "c_puct": sg.f32bits(c), "fpu": sg.f32bits(fpu),
                               "draw": sg.f32bits(draw)},
                    "seed": seed, "state": sg.state_json(d, game), "out": sg.out_json(o)}) + "\n")
            total += count
            print(f"long mech {name} sims {sims}: {count} cases, {time.time() - t0:.1f}s", flush=True)
    print("long mech:", total, "cases ->", path, flush=True)


def part_agent(out_dir, scale):
    from slinky import unified
    from slinky.env import BattlesnakeEnv
    cfg, params = sg.load_agent()
    games = sg.configs()
    plan = [  # (config, count, sims)
        ("duel", 16, 1024), ("standard", 10, 1024), ("royale", 8, 1024), ("constrictor", 8, 1024),
        ("standard3", 4, 1024), ("royale2", 4, 1024), ("constrictor2", 4, 1024),
        ("duel", 4, 4096), ("standard", 3, 4096), ("royale", 3, 4096), ("constrictor", 3, 4096),
    ]
    rng = np.random.default_rng(78)
    path = os.path.join(out_dir, "review_long_agent.jsonl")
    nodes_f = open(os.path.join(out_dir, "review_long_agent_nodes.bin"), "wb")
    offset = 0
    total = 0
    with open(path, "w") as f:
        for i, (name, count, sims) in enumerate(plan):
            t0 = time.time()
            game = games[name]
            n = game.num_snakes
            count = max(1, int(count * scale))
            mode = unified.told_mode(cfg, game, None)
            group = unified.Group(mode or "none", -1, game, BattlesnakeEnv(game, obs="egocentric"), None)
            ev = az_modes.make_evaluator(params, cfg, group, mode)
            mcfg = az_modes.agent_search(cfg, sims)
            run = sg.searcher(game, mcfg, ev)
            jev = jax.jit(ev)
            pos = sg.positions(name, game, count, 8000 + i, True, 0.15, 1.0)
            policy = jax.jit(lambda k, s, p=az_modes.make_policy(params, cfg, game, mode, True, simulations=sims): p(k, s, None))
            for j, d in enumerate(pos):
                seed = int(rng.integers(0, 2**32))
                key = jax.random.key(seed)
                s = sg.mk(d, game)
                o, used, value, prior = jax.device_get(run(key, s))
                if j < 1:  # the search is make_policy's at this simulation count
                    assert np.array_equal(np.asarray(policy(key, s)), np.asarray(o.action)), name
                rv, rl = jax.device_get(jev(s))
                idx = np.nonzero(np.asarray(used))[0]
                rec = np.concatenate([np.asarray(value)[idx].reshape(len(idx), n),
                                      np.asarray(prior)[idx].reshape(len(idx), n * 4)], axis=1)
                nodes_f.write(np.asarray(rec, "<f4").tobytes())
                f.write(json.dumps({
                    "config": name, "game": sg.game_json(game), "mode": mode,
                    "search": {"sims": sims, "c_puct": sg.f32bits(cfg.c_puct),
                               "fpu": sg.f32bits(cfg.fpu_reduction), "draw": sg.f32bits(cfg.draw_value)},
                    "seed": seed, "state": sg.state_json(d, game), "out": sg.out_json(o),
                    "root_logits": sg.f32bits(rl), "root_values": sg.f32bits(rv),
                    "node_offset": offset, "num_nodes": int(len(idx))}) + "\n")
                assert len(idx) == int(o.nodes_used)
                offset += rec.size * 4
            total += count
            print(f"long agent {name} sims {sims}: {count} cases, {time.time() - t0:.1f}s", flush=True)
    nodes_f.close()
    print("long agent:", total, "cases ->", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mech,agent")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, "search")
    os.makedirs(out_dir, exist_ok=True)
    for part in a.only.split(","):
        t0 = time.time()
        if part == "mech":
            part_mech(out_dir, a.scale)
        elif part == "agent":
            part_agent(out_dir, a.scale)
        else:
            raise SystemExit(f"unknown part {part}")
        print(f"{part} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
