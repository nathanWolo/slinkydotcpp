"""Latency benchmark: how long does one MCTS move of ONE game take, and what budget fits?

The Battlesnake server setting: a request carries one game, the move must be back
within a time limit (usually 500 ms, about 400 ms of it usable after network latency),
and nothing is batched. The measured program is ``jax.jit(slinky.mcts.search)`` with
the default :class:`~slinky.mcts.MCTSConfig` (``num_simulations=n``) on ONE unbatched
state; the clock covers dispatch, the search and waiting for the result, and excludes
compilation (untimed warmup calls). Its distribution is reported per ``n``: median,
p90 and max over ``--repeats`` searches (fresh keys) of every position.

``--batch B`` (``B >= 2``) measures ``jax.jit(jax.vmap(search))`` on ``B`` copies of the
position instead, i.e. the latency of a ``B``-game call. Both programs update the
tree in place, so the cost per simulation grows only slowly with ``n`` (cache misses,
deeper descents) and serving one game needs no batch. (Before ``mcts._unbatched_barrier``,
XLA copied the tree's state arrays on every simulation of the unbatched search and of
``B = 1``, which made it quadratic in ``n``.)

Positions come from real games, so the cost is not that of a lucky opening: a batch of
``--games`` games of MCTS (``--player``, a cheap ``mcts-<n>``) against the heuristic is
played, and for each game phase (``--phases name:turn,...``) up to ``--states-per-phase``
games that are still running at that turn each contribute their state.

After the sweep over ``--sims``, each ``--budget-ms`` is calibrated: the largest ``n``
that is a multiple of ``--round`` and whose median latency (``--stat p90`` for the
tail) is within the budget, found by interpolating the sweep and then measuring (and,
if needed, lowering ``n``) until it fits. The report also gives, per ``n``, the memory
of the tree and the depth of the deepest node (the search caps depth at
``max_depth=32``; ``deepest < max_depth`` means the cap never bound).

Run it pinned to ONE core and with the machine otherwise quiet, since other busy cores
disturb timings (``--core`` pins this process; ``taskset -c 3 python ...`` also works)::

    taskset -c 3 python benchmarks/mcts_latency.py
    python benchmarks/mcts_latency.py --core 3 --sims 4096,16384 --budget-ms 500
    python benchmarks/mcts_latency.py --core 3 --sims 256,1024 --budget-ms 50 --quick

Progress goes to stderr; stdout carries only the markdown report. The numbers describe
the search plus dispatch, not JSON parsing or the network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import resource
import sys
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from slinky import agents, mcts
from slinky.env import BattlesnakeEnv
from slinky.types import GameConfig, State

DEFAULT_SIMS = (1024, 4096, 16384, 65536)
DEFAULT_BUDGETS_MS = (500.0, 400.0)  # the time control; and what is left after the network
DEFAULT_PHASES = "early:8,mid:50,late:150"  # name:turn of the positions (turn = engine turn)
MAX_PROBES = 6  # calibration measures at most this many extra values of n per budget
GAME_TURNS = 200  # turns the position-collecting games are played for


@dataclasses.dataclass
class Row:
    """Latency statistics of ``n`` simulations."""

    n: int
    ms: np.ndarray  # float64[P, R] latency of repeat r on position p
    phase: list[str]  # phase of each position
    depth: np.ndarray  # int[P, R] deepest node of each search
    nodes: np.ndarray  # int[P, R] nodes expanded (root included)
    tree_bytes: int  # tree arrays + pre-drawn noise of one search
    compile_s: float
    rss_mb: float  # process peak resident set after this measurement

    def stat(self, name: str) -> float:
        flat = self.ms.ravel()
        return {"median": np.median(flat), "p90": np.percentile(flat, 90), "max": flat.max()}[name]


# --- Positions ------------------------------------------------------------------------


def collect_positions(
    game: GameConfig, phases: dict[str, int], args: argparse.Namespace
) -> tuple[list[str], list[State]]:
    """``(phase names, states)`` from MCTS-vs-heuristic games, at the phases' turns."""
    env = BattlesnakeEnv(game, obs=None)
    mine = agents.make_agent(args.player, game).policy
    other = agents.make_agent("heuristic", game).policy
    turns = max(phases.values()) + 1

    def play(key):
        k_reset, k_play = jax.random.split(key)
        state, ts = env.reset(k_reset)

        def body(carry, k):
            state, ts = carry
            ka, kb, ks = jax.random.split(k, 3)
            seat0 = jnp.arange(game.num_snakes) == 0  # the MCTS plays seat 0
            actions = jnp.where(seat0, mine(ka, state, ts), other(kb, state, ts))
            new_state, new_ts = env.step(ks, state, actions)
            return (new_state, new_ts), state  # the state the move was made in

        _, states = jax.lax.scan(body, (state, ts), jax.random.split(k_play, turns))
        return states

    keys = jax.random.split(jax.random.key(args.seed), args.games)
    states = jax.tree.map(np.asarray, jax.jit(jax.vmap(play))(keys))  # leaves [G, T, ...]
    names, picked = [], []
    for name, turn in phases.items():
        alive = np.flatnonzero(~states.done[:, turn])[: args.states_per_phase]
        log(f"[positions] {name} (turn {turn}): {len(alive)} of {args.games} games still running")
        for g in alive:
            names.append(name)
            picked.append(jax.tree.map(lambda x, g=g, turn=turn: jnp.asarray(x[g, turn]), states))
    if not picked:
        raise SystemExit("no running game at any phase; raise --games or lower the phase turns")
    return names, picked


# --- Measurement ----------------------------------------------------------------------


def tree_bytes(env: BattlesnakeEnv, state: State, config: mcts.MCTSConfig) -> int:
    """Bytes of one search's tree (``num_simulations + 1`` nodes) plus its pre-drawn noise."""
    n = config.num_simulations
    tree = jax.eval_shape(lambda s: mcts._init_tree(s, env, config, n + 1), state)
    noise = n * env.config.num_snakes * 4 * 4  # uint32[n, N, 4]
    return sum(math.prod(x.shape) * x.dtype.itemsize for x in jax.tree.leaves(tree)) + noise


def make_program(env: BattlesnakeEnv, config: mcts.MCTSConfig, batch: int) -> Callable:
    """``program(key, state) -> SearchOutput``: one search, or ``batch`` copies under vmap."""

    def search(key, state):
        return mcts.search(key, state, env, config)

    if batch == 1:
        return jax.jit(search)
    return jax.jit(jax.vmap(search))


def measure(
    n: int, env: BattlesnakeEnv, names: list[str], states: list[State], args: argparse.Namespace
) -> Row:
    """Latency of ``n``-simulation searches: ``repeats`` per position, positions interleaved."""
    config = mcts.MCTSConfig(num_simulations=n)
    program = make_program(env, config, args.batch)
    base = jax.random.key(args.seed + 1)
    if args.batch == 1:
        inputs = states
        keys = [
            [jax.random.fold_in(base, 1000 * p + r) for r in range(args.repeats)]
            for p in range(len(states))
        ]
    else:  # the position repeated ``batch`` times, each copy searched with its own key
        inputs = [jax.tree.map(lambda x: jnp.stack([x] * args.batch), s) for s in states]
        keys = [
            [
                jax.random.split(jax.random.fold_in(base, 1000 * p + r), args.batch)
                for r in range(args.repeats)
            ]
            for p in range(len(states))
        ]
    jax.block_until_ready((inputs, keys))  # inputs are ready before the clock starts
    t0 = time.perf_counter()
    jax.block_until_ready(program(keys[0][0], inputs[0]))  # compile + first run
    compile_s = time.perf_counter() - t0
    jax.block_until_ready(program(keys[0][-1], inputs[0]))  # first run's page faults etc.
    ms = np.zeros((len(states), args.repeats))
    depth = np.zeros(ms.shape, int)
    nodes = np.zeros(ms.shape, int)
    for r in range(args.repeats):
        for p in range(len(states)):
            t0 = time.perf_counter()
            out = jax.block_until_ready(program(keys[p][r], inputs[p]))
            ms[p, r] = 1e3 * (time.perf_counter() - t0)
            depth[p, r], nodes[p, r] = int(np.max(out.depth)), int(np.max(out.nodes_used))
    row = Row(
        n=n,
        ms=ms,
        phase=names,
        depth=depth,
        nodes=nodes,
        tree_bytes=args.batch * tree_bytes(env, states[0], config),
        compile_s=compile_s,
        rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KiB on Linux
    )
    log(
        f"[n={n:>6}] median {row.stat('median'):8.1f} ms  p90 {row.stat('p90'):8.1f}  "
        f"max {row.stat('max'):8.1f}  deepest {depth.max()}  (compile {compile_s:.1f}s)"
    )
    jax.clear_caches()  # keep the compiled programs of other n from piling up
    return row


def estimate_n(points: list[tuple[int, float]], budget: float) -> float:
    """The ``n`` at which the line through the two nearest (n, latency) points meets ``budget``."""
    points = sorted(points)
    for lo, hi in zip(points, points[1:], strict=False):
        if lo[1] <= budget < hi[1]:
            break
    else:  # outside the measured range: extrapolate from the two nearest points
        lo, hi = (points[-2], points[-1]) if budget >= points[-1][1] else (points[0], points[1])
    (n0, t0), (n1, t1) = lo, hi
    return n0 + (budget - t0) * (n1 - n0) / (t1 - t0)


def calibrate(
    budget: float, rows: dict[int, Row], run: Callable[[int], Row], args: argparse.Namespace
) -> Row | None:
    """The row of the largest multiple of ``--round`` measured to fit ``budget`` ms (or None).

    Latency grows with ``n``, so a measured ``n`` over the budget rules out every larger
    multiple, and a fit is only trusted where it was measured. With ``lo`` and ``hi`` the
    bounds on ``n / round``, each probe is the interpolated ``n`` (from all measurements,
    rounded down and kept strictly between the bounds) and moves one of them. It stops
    when they are adjacent, so the answer errs low by less than one ``--round`` (and by
    noise), or after ``MAX_PROBES`` probes.
    """
    unit = args.round
    if len(rows) < 2:
        log("calibration needs at least two --sims values")
        return None

    def bounds() -> tuple[int, int | None]:
        lo = max((n // unit for n, r in rows.items() if n % unit == 0 and fits(r)), default=0)
        over = [-(-n // unit) for n, r in rows.items() if not fits(r)]  # first multiple >= n
        return lo, min(over, default=None)

    def fits(r: Row) -> bool:
        return r.stat(args.stat) <= budget

    for _ in range(MAX_PROBES):
        lo, hi = bounds()
        if hi is not None and hi - lo <= 1:
            break
        guess = int(estimate_n([(n, r.stat(args.stat)) for n, r in rows.items()], budget) // unit)
        k = max(guess, lo + 1) if hi is None else min(max(guess, lo + 1), hi - 1)
        rows[k * unit] = run(k * unit)
    lo, _ = bounds()
    return rows.get(lo * unit)


# --- Reporting --------------------------------------------------------------------------


def table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-flavoured table with right-aligned columns."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    cells = [[c.rjust(w) for c, w in zip(r, widths, strict=True)] for r in (headers, *rows)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join(
        [lines[0], "| " + " | ".join("-" * (w - 1) + ":" for w in widths) + " |"] + lines[1:]
    )


def latency_table(rows: list[Row], max_depth: int) -> str:
    phases = list(dict.fromkeys(rows[0].phase))
    headers = ["simulations", "median ms", "p90 ms", "max ms", "us/sim"]
    headers += [f"median {p}" for p in phases]
    headers += ["tree MB", "RSS MB", "deepest", "median deepest"]
    body = []
    for r in sorted(rows, key=lambda r: r.n):
        by_phase = [np.median(r.ms[[p == ph for p in r.phase]]) for ph in phases]
        body.append(
            [f"{r.n:,}", f"{r.stat('median'):.1f}", f"{r.stat('p90'):.1f}", f"{r.stat('max'):.1f}"]
            + [f"{1e3 * r.stat('median') / r.n:.2f}"]
            + [f"{x:.1f}" for x in by_phase]
            + [f"{r.tree_bytes / 2**20:.1f}", f"{r.rss_mb:.0f}"]
            + [f"{r.depth.max()}" + (" (CAP)" if r.depth.max() >= max_depth else "")]
            + [f"{np.median(r.depth):.0f}"]
        )
    return table(headers, body)


def fit_table(fits: dict[float, Row | None], args: argparse.Namespace) -> str:
    headers = ["budget ms", "simulations", "median ms", "p90 ms", "max ms", "deepest", "tree MB"]
    body = []
    for budget, r in fits.items():
        if r is None:
            body.append([f"{budget:g}", "none found"] + ["-"] * 5)
            continue
        body.append(
            [f"{budget:g}", f"{r.n:,}", f"{r.stat('median'):.1f}", f"{r.stat('p90'):.1f}"]
            + [f"{r.stat('max'):.1f}", f"{r.depth.max()}", f"{r.tree_bytes / 2**20:.1f}"]
        )
    return table(headers, body)


def device_info() -> list[str]:
    lines = [
        f"- jax {jax.__version__}, numpy {np.__version__}, python {platform.python_version()}",
        f"- backend: {jax.default_backend()}, host: {platform.machine()}, "
        f"{os.cpu_count()} logical CPUs, affinity {sorted(os.sched_getaffinity(0))}",
    ]
    if hasattr(os, "getloadavg"):
        lines.append(f"- load average at exit: {', '.join(f'{x:.1f}' for x in os.getloadavg())}")
    return lines


# --- CLI --------------------------------------------------------------------------------


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--sims",
        type=lambda s: sorted({int(x) for x in _csv(s)}),
        default=list(DEFAULT_SIMS),
        metavar="N,N",
        help=f"simulation counts to sweep (default: {','.join(map(str, DEFAULT_SIMS))})",
    )
    p.add_argument(
        "--budget-ms",
        type=lambda s: [float(x) for x in _csv(s)],
        default=list(DEFAULT_BUDGETS_MS),
        metavar="MS,MS",
        help="time budgets to calibrate n for (default: 500,400); '' skips calibration",
    )
    p.add_argument(
        "--round", type=int, default=1000, help="calibrated n is a multiple (default 1000)"
    )
    p.add_argument(
        "--stat",
        choices=("median", "p90", "max"),
        default="median",
        help="latency statistic that must be within the budget (default: median)",
    )
    p.add_argument("--repeats", type=int, default=5, help="searches per position and n (default 5)")
    p.add_argument(
        "--phases", default=DEFAULT_PHASES, help=f"name:turn,... (default {DEFAULT_PHASES})"
    )
    p.add_argument(
        "--states-per-phase", type=int, default=4, help="positions per phase (default 4)"
    )
    p.add_argument("--games", type=int, default=64, help="games played to find positions")
    p.add_argument("--player", default="mcts-32", help="agent for seat 0 in those games")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="1: jit(search) of one game (default); B >= 2: jit(vmap(search)) of B copies",
    )
    p.add_argument("--core", type=int, default=None, help="pin this process to one CPU core")
    p.add_argument(
        "--quick", action="store_true", help="smoke test: 1 repeat, 2 positions per phase"
    )
    p.add_argument("--json", metavar="PATH", help="also write all measurements as JSON")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.quick:
        args.repeats, args.states_per_phase = 1, 2
    args.budget_ms = args.budget_ms or []
    if min(args.round, args.repeats, args.states_per_phase, args.batch) < 1:
        p.error("--round, --repeats, --states-per-phase and --batch must be >= 1")
    try:
        args.phases = {k: int(v) for k, v in (kv.split(":") for kv in _csv(args.phases))}
    except ValueError:
        p.error("--phases must look like name:turn,name:turn")
    if args.phases and max(args.phases.values()) >= GAME_TURNS:
        p.error(f"phase turns must be below {GAME_TURNS}")
    return args


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.core is not None:  # before the first jax computation, so XLA's threads inherit it
        os.sched_setaffinity(0, {args.core})
    if len(os.sched_getaffinity(0)) > 1:
        log("warning: running on several cores; pin with --core or taskset for stable timings")
    game = GameConfig()
    env = BattlesnakeEnv(game, obs=None)
    names, states = collect_positions(game, args.phases, args)

    rows: dict[int, Row] = {}

    def run(n: int) -> Row:
        return measure(n, env, names, states, args)

    for n in args.sims:
        rows[n] = run(n)
    fits = {b: calibrate(b, rows, run, args) for b in args.budget_ms}

    program = (
        "jit(search), one game" if args.batch == 1 else f"jit(vmap(search)), {args.batch} copies"
    )
    print(f"# slinky MCTS latency ({program}; default MCTSConfig)\n")
    print("\n".join(device_info()))
    print(
        f"- positions: {len(states)} from {args.player} vs heuristic games "
        f"({', '.join(f'{k}: turn {v}' for k, v in args.phases.items())}), "
        f"{args.repeats} searches each, {len(states) * args.repeats} per row"
    )
    print(f"- max_depth {mcts.MCTSConfig().max_depth}; compile time excluded\n")
    print("## Latency by simulations\n")
    print(latency_table(list(rows.values()), mcts.MCTSConfig().max_depth))
    print("\n`us/sim` = median latency / simulations; `tree MB` = tree arrays + pre-drawn noise;")
    print("`RSS MB` = peak resident set of this process so far; `deepest` = deepest expanded node")
    print("over all searches (the cap is max_depth; `(CAP)` = reached).")
    if fits:
        print(
            f"\n## Largest n (multiple of {args.round}) within the budget ({args.stat} latency)\n"
        )
        print(fit_table(fits, args))

    if args.json:
        payload = {
            "meta": {"jax": jax.__version__, "affinity": sorted(os.sched_getaffinity(0))},
            "args": {k: v for k, v in vars(args).items()},
            "rows": [
                {
                    "n": r.n,
                    "ms": r.ms.tolist(),
                    "phase": r.phase,
                    "depth": r.depth.tolist(),
                    "nodes": r.nodes.tolist(),
                    "tree_bytes": r.tree_bytes,
                    "compile_s": r.compile_s,
                    "rss_mb": r.rss_mb,
                }
                for r in rows.values()
            ],
            "fits": {str(b): (r.n if r else None) for b, r in fits.items()},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
