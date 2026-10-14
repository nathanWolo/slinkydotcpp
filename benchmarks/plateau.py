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
* **Trend.** The share against the heuristic (mean over the modes) of every snapshot played
  so far, from the first one: at snapshot ``k`` the last ``--window`` ``W`` of them (``x`` =
  env steps in millions) get an ordinary least squares line ``y = a + b x``. Its residual sd
  is ``s = sqrt(SSR / (W - 2))``, the slope's standard error ``se_b = s / sqrt(Sxx)``; over
  the window's span ``h = x_last - x_first`` the line predicts a gain ``G = b h`` with the
  95% half-width ``C = t(0.975, W - 2) se_b h`` (Student t: scipy if installed, else a table).

The stop decision is ``--rule``:

* ``trend`` (the default): snapshot ``k`` is *flat* when ``G < --threshold`` and the upward
  trend is not significant (``G - C <= 0``; a significant decline is flat). Before ``W``
  snapshots have a share, it is not flat. A churning strength (DQN moves by +/-0.017 from one
  snapshot to the next with no trend) makes the pairwise tests significant one way or the
  other almost every time, so this rule looks at the whole window instead of at two snapshots.
* ``pair`` (the first rule, kept to reproduce old runs): flat when ``D < --threshold``, the
  interval of ``D`` covers 0 and ``H2H`` lies in ``--h2h-band``.

Once ``--confirm`` consecutive snapshots are flat and the run has ``--min-steps`` env steps per
mode, the watcher writes ``STOP`` in the run directory (every multi-mode trainer saves a
snapshot and its ``runner.npz`` after the chunk and exits; ``--resume --cooldown <steps>``
then anneals the rate to 0) and exits; ``--pid`` also sends the trainer SIGINT. Constrictor
cells use the sampled agent (``<kind>-sample``): greedy networks play a game fixed by its
start there.

Results: the cells go to ``<run>/plateau-cells.jsonl`` (the :mod:`benchmarks.modes`
format), one decision line per snapshot (from the ``--gap``-th on) to ``<run>/plateau.jsonl``:
``flat`` is the rule's decision, and ``gain``, ``gain_ci95``, ``h2h_share`` and
``share_vs_heuristic`` are always there (the pairwise numbers also under ``trend``, for
information, with their own verdict in ``pair_flat``). The ``trend`` lines add ``rule``,
``window``, ``trend_gain`` (G), ``trend_gain_ci95`` (C), ``trend_slope_per_M`` (b, share per
million env steps) and ``trend_points`` (W), the last four ``null`` until there are ``W``
points; the lines of the first rule have none of them. Rerunning skips the finished cells and
decisions, and the trend reads the shares of the earlier snapshots from the cells (or, failing
those, from the stored decision lines), so a watcher can be restarted. Only the verdicts of the
active rule count towards ``--confirm``: restarted with ``--rule trend`` on a file of the first
rule's lines, the watcher needs ``--confirm`` flat trend checks of its own. Usage::

    python benchmarks/plateau.py --run runs/dqn-modes-M --kind dqn_modes
    python benchmarks/plateau.py --run runs/ppo-modes-M --kind ppo_modes
    python benchmarks/plateau.py --run runs/dqn-modes-M --kind dqn_modes --once   # no waiting
    python benchmarks/plateau.py --run runs/dqn-modes-M --kind dqn_modes --rule pair
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
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
RULES = ("trend", "pair")
# t(0.975, dof) for dof = 1..30, for when scipy is not installed.
T975 = (
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
)  # fmt: skip


def snapshots(run: Path) -> list[int]:
    """The env steps of every snapshot in ``run``, in order."""
    found = (SNAPSHOT.fullmatch(p.name) for p in run.iterdir())
    return sorted({int(m[1]) for m in found if m})  # a set: params-01.npz and params-1.npz


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


def t_table(dof: int) -> float:
    """``t(0.975, dof)`` from :data:`T975`, past it from the Cornish-Fisher expansion."""
    if dof < 1:
        raise ValueError(f"dof must be at least 1, got {dof}")
    if dof <= len(T975):
        return T975[dof - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * dof) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * dof**2)


@functools.cache
def t_crit(dof: int) -> float:
    """The two-sided 95% Student t quantile ``t(0.975, dof)``: scipy's if installed."""
    try:
        from scipy import stats
    except ImportError:
        return t_table(dof)
    return float(stats.t.ppf(0.975, dof))


def trend_check(points: list[tuple[int, float]], window: int, threshold: float) -> dict[str, Any]:
    """The trend rule on ``points`` = (env steps, share against the heuristic), in order.

    Fits ``y = a + b x`` (``x`` = env steps in millions) to the last ``window`` points. The
    window's span is ``h = x_last - x_first``, the predicted gain ``trend_gain = b h`` and its
    95% half-width ``trend_gain_ci95 = t(0.975, window - 2) se_b h``. ``flat`` is
    ``trend_gain < threshold`` and ``trend_gain - trend_gain_ci95 <= 0``. With fewer than
    ``window`` points it is not flat and the trend fields are ``None``. A point whose share is
    not a finite number counts as missing.
    """
    if window < 3:
        raise ValueError(f"the trend window needs at least 3 points, got {window}")
    out: dict[str, Any] = {
        "rule": "trend",
        "window": window,
        "flat": False,
        "trend_gain": None,
        "trend_gain_ci95": None,
        "trend_slope_per_M": None,
        "trend_points": None,
    }
    points = [p for p in points if math.isfinite(p[1])]
    if len(points) < window:
        return out
    last = points[-window:]
    x = np.array([p[0] for p in last], np.float64) / 1e6
    y = np.array([p[1] for p in last], np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float(xc @ xc)
    if not sxx > 0.0:  # the snapshots of the window are all the same: no line to fit
        return out
    slope = float(xc @ yc) / sxx
    resid = yc - slope * xc
    se_slope = math.sqrt(float(resid @ resid) / (window - 2)) / math.sqrt(sxx)
    span = float(x[-1] - x[0])
    trend_gain = slope * span
    half = t_crit(window - 2) * se_slope * span
    out.update(
        flat=bool(trend_gain < threshold and trend_gain - half <= 0.0),
        trend_gain=trend_gain,
        trend_gain_ci95=half,
        trend_slope_per_M=slope,
        trend_points=window,
    )
    return out


def finite(value: Any) -> float | None:
    """``value`` as a float, or None if it is missing or not a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def cell_share(cells: list[dict[str, Any]]) -> float | None:
    """The mean over the modes' cell lines of X's share, None if one has no usable share."""
    try:
        return finite(np.mean([c["results"]["X"]["share"] for c in cells]))
    except (KeyError, TypeError):
        return None


def share_points(
    records: dict[tuple[int, str], dict[str, Any]],
    done: dict[int, dict[str, Any]],
    steps: list[int],
    k: int,
    mode_names: list[str],
) -> list[tuple[int, float]]:
    """(env steps, share against the heuristic, the mean over the modes) of snapshots
    ``steps[: k + 1]``: from the cells played or read back in this process, else from the
    ``share_vs_heuristic`` of the stored decision line (a restarted watcher). Each snapshot
    counts once; one with no usable share (missing, or not a finite number) is left out."""
    points = []
    for y in dict.fromkeys(steps[: k + 1]):
        share = None
        if all((y, m) in records for m in mode_names):
            share = cell_share([records[(y, m)] for m in mode_names])
        if share is None:
            share = finite(done.get(y, {}).get("share_vs_heuristic"))
        if share is not None:
            points.append((y, share))
    return points


def verdict(rec: dict[str, Any], rule: str) -> bool:
    """Whether the stored decision line ``rec`` is flat by ``rule``. A line of the first rule
    has no ``rule`` field. A line of the other rule is never flat by the trend rule (the
    watcher restarted with a new ``--rule`` must see ``--confirm`` flat checks of its own); by
    the first rule it is flat when its ``pair_flat`` says so."""
    if rec.get("rule", "pair") == rule:
        return bool(rec["flat"])
    return rule == "pair" and bool(rec.get("pair_flat", False))


def plateaued(flats: list[bool], steps: int, confirm: int, min_steps: int) -> bool:
    """Whether the last ``confirm`` verdicts (oldest first) are all flat at ``steps``."""
    recent = flats[-confirm:]
    return len(recent) == confirm and all(recent) and steps >= min_steps


def decide(
    records: dict[tuple[int, str], dict[str, Any]],
    h2h: dict[tuple[int, str], dict[str, Any]],
    steps: list[int],
    k: int,
    args: argparse.Namespace,
    done: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The decision line of snapshot index ``k`` (against ``k - gap``).

    ``done``: the decision lines stored so far, the trend's source for the snapshots whose
    cells are not in ``records``."""
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
    pair_flat = d < args.threshold and d - half <= 0.0 <= d + half and lo <= h2h_share <= hi
    rec = {
        "type": "plateau_check",
        "env_steps": a,
        "against": b,
        "gain": d,
        "gain_ci95": half,
        "h2h_share": h2h_share,
        "flat": pair_flat,
        "share_vs_heuristic": float(
            np.mean([records[(a, m)]["results"]["X"]["share"] for m in args.modes])
        ),
        "game_modes": per_mode,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.rule == "trend":
        points = share_points(records, done or {}, steps, k, args.modes)
        rec["pair_flat"] = pair_flat
        rec.update(trend_check(points, args.window, args.threshold))  # sets "flat"
    else:
        rec["rule"] = "pair"
    return rec


def log_line(rec: dict[str, Any]) -> str:
    """The ``plateau.log`` line of a decision."""
    label = "pair gain" if rec.get("rule") == "trend" else "gain"
    pair = (
        f"{label} {rec['gain']:+.3f} +/- {rec['gain_ci95']:.3f} over {rec['against']:,}, "
        f"h2h {rec['h2h_share']:.3f}, vs heuristic {rec['share_vs_heuristic']:.3f}"
    )
    verdict = "flat" if rec["flat"] else "rising"
    if rec.get("rule") != "trend":
        return f"{rec['env_steps']:,}: {pair} -> {verdict}"
    if rec["trend_gain"] is None:
        trend = f"trend waiting for {rec['window']} points"
        verdict = "not flat"
    else:
        trend = (
            f"trend gain {rec['trend_gain']:+.3f} +/- {rec['trend_gain_ci95']:.3f} "
            f"(slope {rec['trend_slope_per_M']:+.4f}/M, {rec['trend_points']} points)"
        )
    return f"{rec['env_steps']:,}: {trend}; {pair} -> {verdict}"


def read_decisions(path: Path) -> dict[int, dict[str, Any]]:
    """The decision lines stored in ``path`` by env steps (none if it is not there yet)."""
    done: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["env_steps"]] = rec
    return done


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
    done = read_decisions(decisions)
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
                rec = decide(records, h2h, steps, k, args, done)
                done[x] = rec
                with open(decisions, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                bench.log(log_line(rec))
            flats = [verdict(done[y], args.rule) for y in steps[: k + 1] if y in done]
            if plateaued(flats, x, args.confirm, args.min_steps):
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
    p.add_argument("--rule", choices=RULES, default="trend", help="the flat/rising verdict")
    p.add_argument("--window", type=int, default=6, help="trend rule: snapshots in the fit")
    p.add_argument(
        "--threshold", type=float, default=0.02,
        help="largest gain that is flat (the trend's over the window, or the pair's)",
    )  # fmt: skip
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
    args = p.parse_args(argv)
    if args.window < 3:
        p.error("--window must be at least 3 (the fit has window - 2 degrees of freedom)")
    return args


if __name__ == "__main__":
    raise SystemExit(watch(parse_args()))
