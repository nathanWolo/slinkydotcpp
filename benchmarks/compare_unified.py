"""Final comparison of the unified-agents campaign: every finished multi-mode run, side by side.

The agents are the run directories under ``--runs`` named ``<algo>-modes-<S|M|L>-s<seed>``
with a ``DONE`` file (``rocdxg-work/campaign/train.sh``: trained to a plateau or the cap,
then cooled down); each plays with its final ``params.npz``. Cells are
:mod:`benchmarks.modes` cells (the same games in every cell of a mode), played on the CPU:

* **Panel** (``--panel``): in every game mode, each agent ``X`` against the heuristic
  (``X,H,H,H``; duel ``X,H``) and against the committed generalist G0
  (``baselines/checkpoints/ppo-modes-seed0``; ``X,G,G,G``) at ``--games``, and against
  random agents (``X,R,R,R``) at ``--random-games``. In constrictor ``X`` samples its moves
  (greedy networks play a game fixed by its start there; G0 too).
* **Round robin** (``--round-robin``): every pair, both sampling, ``A,B,A,B`` (duel ``A,B``)
  at ``--rr-games`` per mode, fitted to Bradley-Terry ratings per mode and over all modes
  (Elo scale, the mean agent at 0).
* ``--az-sims N``: also ``az_modes`` agents with an N-simulation search, panel cells only,
  at ``--az-games`` (set ``JAX_PLATFORMS`` to use the GPU, with ``--jobs 1``).

``--table`` prints both from ``--out``. Usage::

    python benchmarks/compare_unified.py --runs /root/runs --out final.jsonl --panel --jobs 6
    python benchmarks/compare_unified.py --runs /root/runs --out final.jsonl --round-robin --jobs 6
    python benchmarks/compare_unified.py --runs /root/runs --out final.jsonl --table
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plateau  # noqa: E402  (benchmarks/plateau.py: sets JAX_PLATFORMS=cpu by default)

bench = plateau.bench
from slinky import modes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "baselines" / "checkpoints" / "ppo-modes-seed0"
RUN = re.compile(r"(ppo|reinforce|dqn|rainbow|az)-modes-([SML])-s([0-9]+)")
ALGORITHMS = ("ppo", "reinforce", "dqn", "rainbow", "az")
ELO = 400 / math.log(10)


def find_agents(runs: Path, az_sims: int = 0) -> dict[str, tuple[str, Path, int]]:
    """``{symbol: (kind, run dir, simulations)}`` of the finished runs, in algorithm and size
    order; ``az_sims`` adds each ``az_modes`` run with that search as ``<symbol>@<n>``."""
    found = []
    for path in runs.iterdir():
        m = RUN.fullmatch(path.name)
        if m and (path / "DONE").is_file():
            key = (ALGORITHMS.index(m[1]), "SML".index(m[2]), int(m[3]))
            found.append((key, f"{m[1]}-{m[2]}-s{m[3]}", f"{m[1]}_modes", path))
    agents = {}
    for _, symbol, kind, path in sorted(found):
        agents[symbol] = (kind, plateau.run_path(str(path)), 0)
        if az_sims and kind == "az_modes":
            agents[f"{symbol}@{az_sims}"] = (kind, plateau.run_path(str(path)), az_sims)
    return agents


def agent_name(kind: str, run: Path, sims: int, sample: bool) -> str:
    head = f"{kind}-sample" if sample else kind
    return f"{head}:{run}" + (f":simulations={sims}" if sims else "")


def panel_cells(agents: dict, args: argparse.Namespace) -> dict[Any, tuple[bench.Cell, str]]:
    g0 = plateau.run_path(str(G0))
    cells = {}
    for symbol, (kind, run, sims) in agents.items():
        for mode in args.modes:
            n = modes.DEFAULT_SNAKES[mode]
            constrictor = mode == "constrictor"
            names = {
                symbol: agent_name(kind, run, sims, constrictor and not sims),
                "H": "heuristic",
                "R": "random_legal",
                "G": agent_name("ppo_modes", g0, 0, constrictor),
            }
            for other, games in (("H", args.games), ("G", args.games), ("R", args.random_games)):
                if sims:
                    if other != "H":
                        continue
                    games = args.az_games
                cell = bench.make_cell(mode, [symbol] + [other] * (n - 1), names, games)
                cells[("panel", symbol, other, mode)] = (cell, f"{mode} {symbol} vs {other}")
    return cells


def round_robin_cells(agents: dict, args: argparse.Namespace) -> dict[Any, tuple[bench.Cell, str]]:
    plain = {s: a for s, a in agents.items() if not a[2]}
    cells = {}
    for a, b in itertools.combinations(plain, 2):
        for mode in args.modes:
            n = modes.DEFAULT_SNAKES[mode]
            names = {x: agent_name(*plain[x][:2], 0, True) for x in (a, b)}
            cell = bench.make_cell(mode, [a, b] * (n // 2), names, args.rr_games)
            cells[("rr", a, b, mode)] = (cell, f"{mode} {a} v {b}")
    return cells


def bradley_terry(scores: dict[tuple[str, str], float], iters: int = 2000) -> dict[str, float]:
    """Ratings (Elo scale, mean 0) from ``scores[(i, j)]``: i's total score against j."""
    players = sorted({p for pair in scores for p in pair})
    strength = dict.fromkeys(players, 1.0)
    wins = {p: sum(s for (i, _), s in scores.items() if i == p) for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            denom = 0.0
            for q in players:
                n = scores.get((p, q), 0.0) + scores.get((q, p), 0.0)
                if q != p and n:
                    denom += n / (strength[p] + strength[q])
            new[p] = wins[p] / denom if denom and wins[p] > 0 else 1e-6
        mean = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        strength = {p: v / mean for p, v in new.items()}
    return {p: ELO * math.log(v) for p, v in strength.items()}


def table(out: Path, args: argparse.Namespace) -> int:
    records = bench.best_records(bench.read_records(out)[0], None, None)
    if not records:
        bench.log(f"no results in {out}")
        return 1
    panel: dict[tuple[str, str, str], float] = {}  # (symbol, opponent, mode) -> share per seat
    rr: dict[str, dict[tuple[str, str], float]] = {m: {} for m in modes.MODE_NAMES}
    for r in records:
        seats, mode = r["seats"], r["game_mode"]
        policies = list(dict.fromkeys(seats))
        if len(policies) == 2 and seats[1:] == [policies[1]] * (len(seats) - 1):
            x, other = policies
            if other in ("H", "G", "R"):
                panel[(x, other, mode)] = r["results"][x]["share"] / r["results"][x]["seats"]
                continue
        if len(policies) == 2 and seats == policies * (len(seats) // 2):
            a, b = policies
            for p, q in ((a, b), (b, a)):
                rr[mode][(p, q)] = r["games"] * r["results"][p]["share"]
    symbols = list(dict.fromkeys(s for s, _, _ in panel))
    mode_cols = list(modes.MODE_NAMES)
    print("Share of the win per seat (parity: 1/N; duel 0.5); constrictor X samples.\n")
    for other, label in (("H", "heuristic"), ("G", "G0"), ("R", "random")):
        rows = []
        for s in symbols:
            vals = [panel.get((s, other, m)) for m in mode_cols]
            if all(v is None for v in vals):
                continue
            known = [v for v in vals if v is not None]
            mean = sum(known) / len(known) if len(known) == len(vals) else None
            rows.append([s] + [f"{v:.3f}" if v is not None else "-" for v in vals + [mean]])
        if rows:
            print(f"### Against {label}\n")
            print(bench.markdown_table(["Agent"] + mode_cols + ["Mean"], rows) + "\n")
    if any(rr.values()):
        ratings = {m: bradley_terry(s) for m, s in rr.items() if s}
        overall = bradley_terry({k: sum(rr[m].get(k, 0.0) for m in rr) for m in rr for k in rr[m]})
        order = sorted(overall, key=overall.get, reverse=True)
        cols = [m for m in mode_cols if m in ratings]
        rows = [
            [p, f"{overall[p]:+.0f}"] + [f"{ratings[m].get(p, float('nan')):+.0f}" for m in cols]
            for p in order
        ]
        print("### Round robin: Bradley-Terry ratings (Elo scale, mean 0; both sampling)\n")
        print(bench.markdown_table(["Agent", "All modes"] + cols, rows) + "\n")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, required=True, help="the directory of the run dirs")
    p.add_argument("--out", type=Path, required=True, help="results file (modes.py lines)")
    p.add_argument("--panel", action="store_true")
    p.add_argument("--round-robin", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--modes", default=",".join(modes.MODE_NAMES), type=lambda t: t.split(","))
    p.add_argument("--games", type=int, default=1024, help="panel games against H and G0")
    p.add_argument("--random-games", type=int, default=256, help="panel games against random")
    p.add_argument("--rr-games", type=int, default=512, help="round-robin games per pair and mode")
    p.add_argument("--az-sims", type=int, default=0, help="also az_modes with this search")
    p.add_argument("--az-games", type=int, default=256, help="panel games of the searching AZ")
    p.add_argument("--only", help="comma-separated agent symbols (default: all finished runs)")
    p.add_argument("--jobs", type=int, default=1, help="cells played at once (processes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slots", type=int, default=bench.SLOTS)
    p.add_argument("--dry-run", action="store_true", help="list the cells to play and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.table:
        return table(args.out, args)
    agents = find_agents(args.runs, args.az_sims)
    if args.only:
        keep = set(args.only.split(","))
        agents = {s: a for s, a in agents.items() if s in keep}
    bench.log(f"{len(agents)} agents: {', '.join(agents)}")
    cells = {}
    if args.panel:
        cells |= panel_cells(agents, args)
    if args.round_robin:
        cells |= round_robin_cells(agents, args)
    s = bench.Settings(max_turns=bench.MAX_TURNS, seed=args.seed, slots=args.slots)
    if args.dry_run:
        done = (
            {r["cell_id"] for r in bench.read_records(args.out)[0]} if args.out.exists() else set()
        )
        todo = [tag for cell, tag in cells.values() if bench.cell_id(cell, s) not in done]
        print(f"{len(cells)} cells, {len(todo)} to play:\n" + "\n".join(todo))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plateau.play_all(cells, s, args.out, plateau.make_pool(args.jobs))
    return table(args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
