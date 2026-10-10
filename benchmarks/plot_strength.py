"""Plot MCTS strength against the number of simulations from ``strength.py`` results.

Writes a self-contained SVG (no plotting library needed): one line per fixed
opponent, the score of ``mcts-<n>`` (default config) against it, with the 95%
Wilson interval of each point as a whisker::

    python benchmarks/plot_strength.py      # results/strength.jsonl -> results/strength.svg
    python benchmarks/plot_strength.py --out other.svg --opponents heuristic,dqn
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_STYLE = {  # opponent -> (label, colour)
    "random_legal": ("vs random_legal", "#8a8f98"),
    "dqn": ("vs DQN (self-play)", "#2563c9"),
    "heuristic": ("vs heuristic", "#d4580f"),
}
W, H = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 64, 150, 24, 58


def load(path: Path, opponents: list[str], seed: int) -> dict[str, list[tuple[int, dict]]]:
    """``opponent -> [(simulations, record)]`` for plain ``mcts-<n>`` rows (newest line wins)."""
    series: dict[str, dict[int, dict]] = {b: {} for b in opponents}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.fullmatch(r"mcts-(\d+)", rec.get("a", ""))
        if m and rec.get("b") in series and rec.get("seed") == seed:
            series[rec["b"]][int(m.group(1))] = rec
    return {b: sorted(pts.items()) for b, pts in series.items() if pts}


def render(series: dict[str, list[tuple[int, dict]]]) -> str:
    sims = sorted({n for pts in series.values() for n, _ in pts})
    lo, hi = math.log2(sims[0]), math.log2(sims[-1])
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def x(n: float) -> float:
        return LEFT + (math.log2(n) - lo) / (hi - lo) * pw

    def y(s: float) -> float:
        return TOP + (1 - s) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for s in (0, 0.25, 0.5, 0.75, 1.0):  # horizontal grid
        dash = ' stroke-dasharray="4 3"' if s == 0.5 else ""
        colour = "#9aa0a8" if s == 0.5 else "#e3e6ea"
        out.append(
            f'<line x1="{LEFT}" x2="{LEFT + pw}" y1="{y(s):.1f}" y2="{y(s):.1f}" '
            f'stroke="{colour}"{dash}/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{y(s) + 4:.1f}" text-anchor="end" fill="#555b63">{s:g}</text>'
        )
    out.append(
        f'<text x="{LEFT + pw - 4}" y="{y(0.5) - 6:.1f}" text-anchor="end" fill="#6b7179">'
        "even (0.5)</text>"
    )
    for n in sims:  # x ticks at every budget played
        out.append(
            f'<line x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{TOP + ph}" y2="{TOP + ph + 5}" '
            'stroke="#555b63"/>'
        )
        out.append(
            f'<text x="{x(n):.1f}" y="{TOP + ph + 19}" text-anchor="middle" '
            f'fill="#555b63">{n}</text>'
        )
    out.append(
        f'<line x1="{LEFT}" x2="{LEFT + pw}" y1="{TOP + ph}" y2="{TOP + ph}" stroke="#555b63"/>'
    )
    out.append(
        f'<text x="{LEFT + pw / 2:.1f}" y="{H - 14}" text-anchor="middle" fill="#30353b">'
        "MCTS simulations per move (log scale)</text>"
    )
    out.append(
        f'<text transform="translate(18 {TOP + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
        'fill="#30353b">score of MCTS (win 1, draw ½, loss 0)</text>'
    )

    label_y: list[float] = []
    for b, pts in series.items():
        label, colour = SERIES_STYLE.get(b, (f"vs {b}", "#444444"))
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(n):.1f},{y(r['score']):.1f}"
            for i, (n, r) in enumerate(pts)
        )
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for n, r in pts:
            lo_s, hi_s = r.get("ci95_low", r["score"]), r.get("ci95_high", r["score"])
            out.append(
                f'<line x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{y(hi_s):.1f}" y2="{y(lo_s):.1f}" '
                f'stroke="{colour}" stroke-width="1.2" opacity="0.7"/>'
            )
            out.append(
                f'<circle cx="{x(n):.1f}" cy="{y(r["score"]):.1f}" r="3.2" fill="{colour}">'
                f"<title>mcts-{n} {label}: {r['score']:.3f} [{lo_s:.3f}, {hi_s:.3f}], "
                f"{r['wins']}/{r['draws']}/{r['losses']} over {r['games']} games</title></circle>"
            )
        # Direct label in the right margin at the height of the line's last point, nudged
        # apart if two are close.
        ly = y(pts[-1][1]["score"])
        while any(abs(ly - other) < 14 for other in label_y):
            ly += 14
        label_y.append(ly)
        out.append(
            f'<text x="{LEFT + pw + 12}" y="{ly + 4:.1f}" fill="{colour}" '
            f'font-weight="bold">{label}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=HERE / "results" / "strength.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "results" / "strength.svg")
    p.add_argument("--opponents", default="random_legal,dqn,heuristic")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    series = load(args.results, args.opponents.split(","), args.seed)
    if not series:
        raise SystemExit(f"no mcts-<n> results against {args.opponents} in {args.results}")
    args.out.write_text(render(series))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
