"""Tabulate snapshot cells: A's share with Wilson 95% CI (from the result line) and paired comparisons."""
import json, sys, math
import numpy as np

D = "docs/research/multimode_constrictor_analysis/snapshots"
SNAPS = [1048576, 2097152, 3145728, 4194304, 5242880]

def unbase36(text, width):
    return np.array([int(text[i:i+width], 36) for i in range(0, len(text), width)], np.int64)

def pergame(rec, sym):
    pg = rec["per_game"]
    return unbase36(pg["share"][sym], pg["width"]) / pg["share_scale"]

cells = {}
for line in open(f"{D}/cells.jsonl"):
    r = json.loads(line)
    cid = r["cell_id"]
    if cid not in cells or r["games"] > cells[cid]["games"]:
        cells[cid] = r

def find(mode, lineup, a, b):
    """a, b: (run_dir, snapshot or None) ; b None for heuristic"""
    def nm(x):
        if x == "heuristic": return "heuristic"
        d, s = x
        return f"ppo_modes:{d}" + (f":snapshot={s}" if s else "")
    for r in cells.values():
        if r["game_mode"] == mode and r["lineup"] == lineup and r["seat_names"][0] == nm(a):
            sn = r["seat_names"]
            if b is None or sn[1] == nm(b):
                return r
    return None

def fmt(r, sym="A"):
    x = r["results"][sym]
    lo, hi = x["share_wilson"]
    return f"{x['share']:.3f} [{lo:.3f}, {hi:.3f}]", x

def pair_diff(r1, r2, sym="A"):
    """paired per-seat share difference r1 - r2 (same games), normal 95% CI; both cells hold sym in same seats"""
    s1, s2 = r1["results"][sym]["seats"], r2["results"][sym]["seats"]
    assert s1 == s2
    n = min(len(pergame(r1, sym)), len(pergame(r2, sym)))
    d = (pergame(r1, sym)[:n] - pergame(r2, sym)[:n]) / s1
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    return m, 1.96 * se, n

out = []
def P(s=""):
    print(s); out.append(s)

G = "runs/ppo-modes-seed0"
for mode in ["constrictor", "royale"]:
    S = f"runs/ppo-modes-{mode}-seed0"
    ng = 1024 if mode == "constrictor" else 512
    P(f"## {mode}  (games per head-to-head cell as run; A's team share, parity 0.5; Wilson 95%)")
    P()
    P("| k (env steps/mode) | (a) G@k vs S_final | W/T/L | (c) G@k vs S@k | W/T/L | (b) S@k vs S_final | W/T/L |")
    P("|---|---|---|---|---|---|---|")
    for k in SNAPS:
        row = [f"{k/1e6:.2f}M"]
        for key in ("a", "c", "b"):
            if key == "a": r = find(mode, "A,B,A,B", (G, k), (S, None))
            elif key == "c": r = find(mode, "A,B,A,B", (G, k), (S, k))
            else: r = find(mode, "A,B,A,B", (S, k), (S, None)) if k < SNAPS[-1] else None
            if r is None:
                row += ["n/a", ""]
            else:
                s, x = fmt(r)
                row += [f"{s} (n={r['games']})", f"{x['wins']}/{x['ties']}/{x['losses']}"]
        P("| " + " | ".join(row) + " |")
    P()
    # paired differences between snapshots in (a)
    P(f"Paired change in G's per-seat share vs S_final between snapshots (cell (a), same games, normal 95% CI on per-game differences; per-seat share = share/2, so multiply by 2 for team share):")
    P()
    P("| from -> to | delta team share (A) | 95% CI | n games |")
    P("|---|---|---|---|")
    for k1, k2 in [(SNAPS[i], SNAPS[j]) for i in range(5) for j in range(i+1, 5)]:
        r1, r2 = find(mode, "A,B,A,B", (G, k1), (S, None)), find(mode, "A,B,A,B", (G, k2), (S, None))
        if r1 and r2:
            m, hw, n = pair_diff(r2, r1)
            P(f"| {k1/1e6:.2f}M -> {k2/1e6:.2f}M | {2*m:+.3f} | +-{2*hw:.3f} [{2*(m-hw):+.3f}, {2*(m+hw):+.3f}] | {n} |")
    P()
    if mode == "constrictor":
        P("### (d) constrictor vs 3 heuristics (A,H,H,H, 512 games, parity 0.25; Wilson 95%)")
        P()
        P("| k | G@k | W/T/L | S@k | W/T/L | G - S (paired, share) |")
        P("|---|---|---|---|---|---|")
        for k in SNAPS:
            rg = find(mode, "A,H,H,H", (G, k), "heuristic")
            rs = find(mode, "A,H,H,H", (S, k), "heuristic")
            row = [f"{k/1e6:.2f}M"]
            for r in (rg, rs):
                if r is None: row += ["n/a", ""]
                else:
                    s, x = fmt(r); row += [s, f"{x['wins']}/{x['ties']}/{x['losses']}"]
            if rg and rs:
                m, hw, n = pair_diff(rg, rs)
                row.append(f"{m*1:+.3f} +-{hw:.3f} (n={n})")
            else: row.append("")
            P("| " + " | ".join(row) + " |")
        P()
        P("Paired change in G@k vs 3 heuristics between snapshots (share, parity 0.25):")
        P()
        P("| from -> to | G delta | S delta |")
        P("|---|---|---|")
        for k1, k2 in [(SNAPS[i], SNAPS[i+1]) for i in range(4)] + [(SNAPS[0], SNAPS[2]), (SNAPS[2], SNAPS[4]), (SNAPS[0], SNAPS[4]), (SNAPS[1], SNAPS[4])]:
            cols = []
            for run in (G, S):
                r1, r2 = find(mode, "A,H,H,H", (run, k1), "heuristic"), find(mode, "A,H,H,H", (run, k2), "heuristic")
                if r1 and r2:
                    m, hw, n = pair_diff(r2, r1); cols.append(f"{m:+.3f} +-{hw:.3f}")
                else: cols.append("n/a")
            P(f"| {k1/1e6:.2f}M -> {k2/1e6:.2f}M | {cols[0]} | {cols[1]} |")
        P()
open(f"{D}/tables.md", "w").write("\n".join(out))
