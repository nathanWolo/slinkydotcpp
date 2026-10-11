"""Plot what AlphaZero's network is worth in plain MCTS simulations, over training.

Reads ``strength.py`` results of ``az-64`` (the network with 64 simulations of
PUCT search) against plain ``mcts-<n>`` and writes a self-contained SVG: one
line per checkpoint, the score of ``az-64`` against MCTS with ``n`` simulations,
with 95% Wilson whiskers. Where a line crosses 0.5 is the number of plain
simulations that 64 network-guided ones are worth at that checkpoint. A dashed
line marks the plain budget that costs as much time per move as ``az-64`` on the
benchmark machine (from the two agents' timing probes against the heuristic)::

    python benchmarks/plot_az_ladder.py   # results/*.jsonl -> results/az_ladder.svg

Rows ``az-64:<dir>/<env steps>`` are training snapshots (``results/az_ladder.jsonl``;
the snapshots themselves are not committed) and plain ``az-64`` is the committed
checkpoint, ``baselines/checkpoints/az-duel-seed0``. A ``:c_puct=<x>`` play-time
setting gets its own dashed line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 800, 430
LEFT, RIGHT, TOP, BOTTOM = 64, 210, 24, 58
AGENT = re.compile(r"az-64(?::(?P<path>[^:]*?/(?P<steps>\d+)))?(?::c_puct=(?P<c_puct>[0-9.]+))?")
Key = tuple[int, str | None]  # (env steps, play-time c_puct or None)
FINAL_STEPS = 8_388_608  # the committed checkpoint's training length


def load(paths: list[Path], seed: int) -> tuple[dict[Key, dict[int, dict]], dict[str, dict]]:
    """``((env steps, c_puct) -> {mcts sims -> record}, {agent name -> record vs heuristic})``.

    The newest line wins, as in ``strength.py``'s tables.
    """
    series: dict[Key, dict[int, dict]] = {}
    vs_heuristic: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("seed") != seed:
                continue
            a, b = rec.get("a", ""), rec.get("b", "")
            if b == "heuristic":
                vs_heuristic[a] = rec
            agent, opponent = AGENT.fullmatch(a), re.fullmatch(r"mcts-(\d+)", b)
            if agent and opponent:
                steps = int(agent["steps"]) if agent["steps"] else FINAL_STEPS
                key = (steps, agent["c_puct"])
                series.setdefault(key, {})[int(opponent.group(1))] = rec
    return series, vs_heuristic


def equal_time_sims(vs_heuristic: dict[str, dict]) -> float | None:
    """The plain-MCTS budget whose move takes as long as ``az-64``'s (log-log interpolation).

    Both times are ``strength.py``'s probe against the heuristic: one move of one
    game with 32 games in flight, including the heuristic's move and the env step.
    """
    az = vs_heuristic.get("az-64", {}).get("seconds_per_move_probe")
    plain = sorted(
        (int(m.group(1)), r["seconds_per_move_probe"])
        for a, r in vs_heuristic.items()
        if (m := re.fullmatch(r"mcts-(\d+)", a)) and r.get("seconds_per_move_probe")
    )
    if not az or len(plain) < 2:
        return None
    for (n0, t0), (n1, t1) in zip(plain, plain[1:], strict=False):
        if t0 <= az <= t1:
            f = (math.log(az) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0)))
    return None


def steps_label(key: Key) -> str:
    steps, c_puct = key
    text = f"{steps / 1e6:.1f}M env steps" + (", final" if steps == FINAL_STEPS else "")
    return text + f", c_puct {c_puct}" if c_puct else text


def colour(i: int, count: int) -> str:
    """Light to dark blue along the training run."""
    t = i / max(count - 1, 1)
    light, dark = (158, 202, 225), (8, 48, 107)
    r, g, b = (round(c0 + t * (c1 - c0)) for c0, c1 in zip(light, dark, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def render(series: dict[Key, dict[int, dict]], equal_time: float | None) -> str:
    sims = sorted({n for pts in series.values() for n in pts})
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
    for s in (0, 0.25, 0.5, 0.75, 1.0):
        dash = ' stroke-dasharray="4 3"' if s == 0.5 else ""
        grid = "#9aa0a8" if s == 0.5 else "#e3e6ea"
        out.append(
            f'<line x1="{LEFT}" x2="{LEFT + pw}" y1="{y(s):.1f}" y2="{y(s):.1f}" '
            f'stroke="{grid}"{dash}/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{y(s) + 4:.1f}" text-anchor="end" fill="#555b63">{s:g}</text>'
        )
    out.append(f'<text x="{LEFT + 4}" y="{y(0.5) + 16:.1f}" fill="#6b7179">even (0.5)</text>')
    for n in sims:
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
    if equal_time and sims[0] <= equal_time <= sims[-1]:
        out.append(
            f'<line x1="{x(equal_time):.1f}" x2="{x(equal_time):.1f}" y1="{TOP}" '
            f'y2="{TOP + ph}" stroke="#d4580f" stroke-dasharray="3 3"/>'
        )
        out.append(
            f'<text x="{x(equal_time) + 5:.1f}" y="{TOP + 12}" fill="#d4580f">'
            f"same time per move as az-64 (~{equal_time:,.0f})</text>"
        )
    out.append(
        f'<text x="{LEFT + pw / 2:.1f}" y="{H - 14}" text-anchor="middle" fill="#30353b">'
        "plain MCTS simulations per move (log scale)</text>"
    )
    out.append(
        f'<text transform="translate(18 {TOP + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
        'fill="#30353b">score of az-64 (win 1, draw ½, loss 0)</text>'
    )

    out.append(f'<text x="{LEFT + pw + 12}" y="{TOP + 12}" fill="#30353b">checkpoint</text>')
    ordered = sorted(series, key=lambda k: (k[0], k[1] or ""))
    checkpoints = sorted({steps for steps, _ in series})
    for i, key in enumerate(ordered):
        pts = sorted(series[key].items())
        steps, variant = key
        c = colour(checkpoints.index(steps), len(checkpoints))
        final = steps == FINAL_STEPS
        width = 2.6 if final else 1.6
        dash = ' stroke-dasharray="6 4"' if variant else ""
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{x(n):.1f},{y(r['score']):.1f}"
            for j, (n, r) in enumerate(pts)
        )
        out.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="{width}"{dash}/>')
        for n, r in pts:
            lo_s, hi_s = r.get("ci95_low", r["score"]), r.get("ci95_high", r["score"])
            out.append(
                f'<line x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{y(hi_s):.1f}" y2="{y(lo_s):.1f}" '
                f'stroke="{c}" stroke-width="1.2" opacity="0.6"/>'
            )
            out.append(
                f'<circle cx="{x(n):.1f}" cy="{y(r["score"]):.1f}" r="3" fill="{c}">'
                f"<title>{steps_label(key)}: az-64 vs mcts-{n}: {r['score']:.3f} "
                f"[{lo_s:.3f}, {hi_s:.3f}], {r['wins']}/{r['draws']}/{r['losses']} over "
                f"{r['games']} games</title></circle>"
            )
        # Legend in the right margin, in training order.
        ly = TOP + 30 + 18 * i
        weight = ' font-weight="bold"' if final else ""
        out.append(
            f'<line x1="{LEFT + pw + 12}" x2="{LEFT + pw + 30}" y1="{ly:.1f}" y2="{ly:.1f}" '
            f'stroke="{c}" stroke-width="{width}"{dash}/>'
        )
        label = steps_label(key)
        lines = label.split(", c_puct ")  # a play-time variant gets a second line
        out.append(
            f'<text x="{LEFT + pw + 36}" y="{ly + 4:.1f}" fill="{c}"{weight}>{lines[0]}</text>'
        )
        if len(lines) > 1:
            out.append(
                f'<text x="{LEFT + pw + 36}" y="{ly + 18:.1f}" fill="{c}"{weight}>'
                f"c_puct {lines[1]} at play time</text>"
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[HERE / "results" / "strength.jsonl", HERE / "results" / "az_ladder.jsonl"],
    )
    p.add_argument("--out", type=Path, default=HERE / "results" / "az_ladder.svg")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    series, vs_heuristic = load([r for r in args.results if r.exists()], args.seed)
    if not series:
        raise SystemExit(f"no az-64 vs mcts-<n> results in {[str(r) for r in args.results]}")
    args.out.write_text(render(series, equal_time_sims(vs_heuristic)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
