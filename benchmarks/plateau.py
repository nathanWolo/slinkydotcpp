"""Plateau watcher: stop a multi-mode training run once its strength stops rising.

Watches a run directory of any multi-mode trainer (``ppo_modes``, ``reinforce_modes``,
``dqn_modes``, ``rainbow_modes``, ``az_modes``) for its snapshots ``params-<steps>.npz``
and plays every new one with :mod:`benchmarks.modes` cells, on the CPU (``JAX_PLATFORMS``
is set to ``cpu``), so it runs beside the trainer on the GPU:

* **Strength.** In every game mode the snapshot against the heuristic: ``X,H,H,H``
  (``X,H`` in the duel), ``--games`` games. Every cell of a mode plays the same games
  (:func:`modes.cell_key`), so two snapshots' cells pair game by game.
* **Gain.** At the ``k``-th snapshot, ``D`` = the mean over the modes of the paired gain
  in share per seat of snapshot ``k`` over snapshot ``k - gap``, with its 95% interval
  (the modes' intervals combined as independent).
* **Head to head.** Snapshot ``k`` against snapshot ``k - gap`` with both sampling their
  moves: ``A,B,A,B`` (``A,B`` in the duel), ``--h2h-games`` games per mode; ``H2H`` is
  A's share (parity 0.5 in every mode), the mean over the modes.

Snapshot ``k`` is *flat* when ``D < --threshold``, the interval of ``D`` covers 0 and
``H2H`` lies in ``--h2h-band``. Once ``--confirm`` consecutive snapshots are flat and the
run has ``--min-steps`` env steps per mode, the watcher writes ``STOP`` in the run
directory (every multi-mode trainer saves a snapshot and its ``runner.npz`` after the chunk
and exits; ``--resume --cooldown <steps>`` then anneals the rate to 0) and exits; ``--pid``
also sends the trainer SIGINT. Constrictor cells use the sampled agent (``<kind>-sample``):
greedy networks play a game fixed by its start there.

Results: the cells go to ``<run>/plateau-cells.jsonl`` (the :mod:`benchmarks.modes`
format), one decision line per snapshot to ``<run>/plateau.jsonl``. Rerunning skips the
finished cells, so a watcher can be restarted. Usage::

    python benchmarks/plateau.py --run runs/dqn-modes-M --kind dqn_modes
    python benchmarks/plateau.py --run runs/ppo-modes-M --kind ppo_modes
    python benchmarks/plateau.py --run runs/dqn-modes-M --kind dqn_modes --once   # no waiting
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import modes as bench  # noqa: E402  (benchmarks/modes.py)
import numpy as np  # noqa: E402

from slinky import modes  # noqa: E402

SNAPSHOT = re.compile(r"params-([0-9]+)\.npz")
KINDS = ("ppo_modes", "reinforce_modes", "dqn_modes", "rainbow_modes", "az_modes")


def snapshots(run: Path) -> list[int]:
    """The env steps of every snapshot in ``run``, in order."""
    found = (SNAPSHOT.fullmatch(p.name) for p in run.iterdir())
    return sorted(int(m[1]) for m in found if m)


def run_path(text: str) -> Path:
    """The run directory as an agent name can hold it: names split on ``:``, so on Windows a
    path on a drive is made relative to the working directory (UNC paths have no colon)."""
    run = Path(text).resolve()
    return Path(os.path.relpath(run)) if ":" in str(run) else run


def agent_name(kind: str, run: Path, steps: int, sample: bool) -> str:
    head = f"{kind}-sample" if sample else kind
    return f"{head}:{run}:snapshot={steps}"


def strength_cell(kind: str, run: Path, steps: int, mode: str, games: int) -> bench.Cell:
    n = modes.DEFAULT_SNAKES[mode]
    agents = {"X": agent_name(kind, run, steps, mode == "constrictor"), "H": "heuristic"}
    return bench.make_cell(mode, ["X"] + ["H"] * (n - 1), agents, games)


def h2h_cell(kind: str, run: Path, a: int, b: int, mode: str, games: int) -> bench.Cell:
    n = modes.DEFAULT_SNAKES[mode]
    agents = {"A": agent_name(kind, run, a, True), "B": agent_name(kind, run, b, True)}
    return bench.make_cell(mode, ["A", "B"] * (n // 2), agents, games)


def _init_worker(cores: Any) -> None:
    """A worker process: pin it to its share of the cores (Linux), before jax starts."""
    mine = cores.get()
    if mine and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, mine)


def _run_cell(cell: bench.Cell, s: bench.Settings, tag: str) -> dict[str, Any]:
    bench.log(f"{tag}: {cell.games} games, {bench.agents_text(cell)}")
    record = bench.run_cell(cell, s, tag)
    bench.log(f"{tag} done: {bench.describe(record)}")
    return record


def make_pool(jobs: int) -> concurrent.futures.ProcessPoolExecutor | None:
    """``jobs`` worker processes, each pinned to its share of this process's cores."""
    if jobs <= 1:
        return None
    ctx = multiprocessing.get_context("spawn")
    cores = ctx.Queue()
    have = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    for i in range(jobs):
        cores.put(set(have[i::jobs]) if len(have) >= jobs else None)
    return concurrent.futures.ProcessPoolExecutor(
        jobs, mp_context=ctx, initializer=_init_worker, initargs=(cores,)
    )


def play_all(
    cells: dict[Any, tuple[bench.Cell, str]],
    s: bench.Settings,
    out: Path,
    pool: concurrent.futures.ProcessPoolExecutor | None,
) -> dict[Any, dict[str, Any]]:
    """Each cell's result line from ``out``, the missing ones played (in ``pool``, in
    parallel) and appended first. ``cells``: key -> (cell, log tag)."""
    done = {}
    have = {r["cell_id"]: r for r in bench.read_records(out)[0]}
    todo = {}
    for key, (cell, tag) in cells.items():
        r = have.get(bench.cell_id(cell, s))
        if r is not None and r["games"] >= cell.games:
            done[key] = r
        else:
            todo[key] = (cell, tag)
    if pool is None:
        for key, (cell, tag) in todo.items():
            done[key] = _run_cell(cell, s, tag)
            bench.strength.append_record(out, done[key])
        return done
    futures = {pool.submit(_run_cell, cell, s, tag): key for key, (cell, tag) in todo.items()}
    for future in concurrent.futures.as_completed(futures):
        done[futures[future]] = record = future.result()
        bench.strength.append_record(out, record)
    return done


def gain(new: dict[str, Any], old: dict[str, Any]) -> tuple[float, float]:
    """Paired gain in share per seat of X in ``new`` over ``old`` (same games): (mean, half)."""
    a = bench.decode_per_game(new)["X"]["share"]
    b = bench.decode_per_game(old)["X"]["share"]
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    return float(d.mean()), bench.Z95 * float(d.std(ddof=1)) / math.sqrt(n)


def decide(
    records: dict[tuple[int, str], dict[str, Any]],
    h2h: dict[tuple[int, str], dict[str, Any]],
    steps: list[int],
    k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The decision line of snapshot index ``k`` (against ``k - gap``)."""
    a, b = steps[k], steps[k - args.gap]
    per_mode, gains, halves, shares = {}, [], [], []
    for mode in args.modes:
        g, h = gain(records[(a, mode)], records[(b, mode)])
        r = h2h[(a, mode)]
        share = r["results"]["A"]["share"]
        per_mode[mode] = {
            "share_vs_heuristic": records[(a, mode)]["results"]["X"]["share"],
            "gain": g,
            "gain_ci95": h,
            "h2h_share": share,
            "h2h_ci95": r["results"]["A"]["share_ci95"],
        }
        gains.append(g)
        halves.append(h)
        shares.append(share)
    d = float(np.mean(gains))
    half = math.sqrt(sum(x * x for x in halves)) / len(halves)
    h2h_share = float(np.mean(shares))
    lo, hi = args.h2h_band
    flat = d < args.threshold and d - half <= 0.0 <= d + half and lo <= h2h_share <= hi
    return {
        "type": "plateau_check",
        "env_steps": a,
        "against": b,
        "gain": d,
        "gain_ci95": half,
        "h2h_share": h2h_share,
        "flat": flat,
        "share_vs_heuristic": float(
            np.mean([records[(a, m)]["results"]["X"]["share"] for m in args.modes])
        ),
        "game_modes": per_mode,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def stop_run(run: Path, steps: int, args: argparse.Namespace) -> None:
    (run / "STOP").write_text(json.dumps({"plateau_at": steps, "by": "benchmarks/plateau.py"}))
    if args.pid:
        os.kill(args.pid, signal.SIGINT)
    bench.log(f"plateau at {steps:,} env steps per mode: wrote {run / 'STOP'}")


def watch(args: argparse.Namespace) -> int:
    run = run_path(args.run)
    out = run / "plateau-cells.jsonl"
    decisions = run / "plateau.jsonl"
    s = bench.Settings(max_turns=bench.MAX_TURNS, seed=args.seed, slots=args.slots)
    records: dict[tuple[int, str], dict[str, Any]] = {}
    h2h: dict[tuple[int, str], dict[str, Any]] = {}
    done: dict[int, dict[str, Any]] = {}
    if decisions.is_file():
        for line in decisions.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["env_steps"]] = rec
    pool = make_pool(args.jobs)
    while True:
        steps = snapshots(run)
        if args.every:
            steps = [x for x in steps if x % args.every == 0] or steps
        for k, x in enumerate(steps):
            # This snapshot's cells: against the heuristic, and against snapshot k - gap.
            cells = {
                ("H", mode): (
                    strength_cell(args.kind, run, x, mode, args.games),
                    f"{x:,} {mode} vs H",
                )
                for mode in args.modes
                if (x, mode) not in records
            }
            if k >= args.gap:
                b = steps[k - args.gap]
                for mode in args.modes:
                    if (x, mode) not in h2h:
                        cell = h2h_cell(args.kind, run, x, b, mode, args.h2h_games)
                        cells[("A", mode)] = (cell, f"{x:,} {mode} h2h")
            for (what, mode), r in play_all(cells, s, out, pool).items():
                (records if what == "H" else h2h)[(x, mode)] = r
            if k < args.gap:
                continue
            if x not in done:
                rec = decide(records, h2h, steps, k, args)
                done[x] = rec
                with open(decisions, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                bench.log(
                    f"{x:,}: gain {rec['gain']:+.3f} +/- {rec['gain_ci95']:.3f} over "
                    f"{rec['against']:,}, h2h {rec['h2h_share']:.3f}, vs heuristic "
                    f"{rec['share_vs_heuristic']:.3f} -> {'flat' if rec['flat'] else 'rising'}"
                )
            recent = [done[y]["flat"] for y in steps[: k + 1] if y in done][-args.confirm :]
            if len(recent) == args.confirm and all(recent) and x >= args.min_steps:
                stop_run(run, x, args)
                return 0
        if args.once or (run / "STOPPED").exists():
            return 0
        time.sleep(args.poll)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="the run directory")
    p.add_argument("--kind", required=True, choices=KINDS, help="the agent kind of the run")
    p.add_argument("--modes", default=",".join(modes.MODE_NAMES), type=lambda t: t.split(","))
    p.add_argument("--games", type=int, default=1024, help="games per strength cell")
    p.add_argument("--h2h-games", type=int, default=512, help="games per head-to-head cell")
    p.add_argument("--gap", type=int, default=2, help="compare snapshot k with k - gap")
    p.add_argument("--every", type=int, default=0, help="only snapshots at multiples of this")
    p.add_argument("--threshold", type=float, default=0.02, help="largest gain that is flat")
    p.add_argument("--h2h-band", type=float, nargs=2, default=(0.45, 0.55))
    p.add_argument("--confirm", type=int, default=2, help="consecutive flat snapshots to stop")
    p.add_argument("--min-steps", type=int, default=5_242_880, help="env steps per mode")
    p.add_argument("--pid", type=int, help="also send SIGINT to this trainer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slots", type=int, default=bench.SLOTS)
    p.add_argument(
        "--jobs", type=int, default=1,
        help="play a snapshot's cells in this many processes, each pinned to its share of the "
        "cores (Linux)",
    )  # fmt: skip
    p.add_argument("--poll", type=float, default=60.0, help="seconds between directory scans")
    p.add_argument("--once", action="store_true", help="check the snapshots there and exit")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(watch(parse_args()))
