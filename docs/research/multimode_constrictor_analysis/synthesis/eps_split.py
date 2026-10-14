"""Split the behaviour investigation's eps-greedy constrictor G,S,G,S games by the move at which the first
effective random move (applied move != the owning policy's greedy move) happened during the contest.
Question: is G's near-greedy deficit (0.417 at eps=0.02) carried by games that still follow the 12 deterministic
greedy opening lines, or does it persist after the opening has been perturbed?"""
import sys
BEH = "docs/research/multimode_constrictor_analysis/behaviour"
sys.path.insert(0, BEH)
import numpy as np
from common import load, contest_outcome, Boot, fmt
for f in ["constrictor2048_eps0.02", "constrictor2048_eps0.05", "constrictor2048_eps0.1"]:
    d = contest_outcome(load(f"{BEH}/{f}.npz", dedupe=False))
    Gm, T, N = d["G"], d["T"], d["N"]
    pol = d["pol"]  # [G, N]
    own = np.where(pol[:, None, :] == 0, d["ag"], d["a_s"])
    t = np.arange(T)[None, :, None]
    live = d["alive"] & ~d["done"][:, :, None] & (t < d["K_end"][:, None, None])
    dev = live & (d["act"] != own)
    anydev = dev.any(-1)  # [G, T]
    first = np.where(anydev.any(1), np.argmax(anydev, axis=1), 10**6)
    # who made the first effective random move (G seat, S seat, or both on the same move)
    fd = np.zeros(Gm, int)  # 0 none, 1 G only, 2 S only, 3 both
    for g in range(Gm):
        if first[g] < 10**6:
            seats = np.where(dev[g, first[g]])[0]
            ps = set(pol[g, seats].tolist())
            fd[g] = 3 if ps == {0, 1} else (1 if ps == {0} else 2)
    sh = d["share"][:, 0]
    b = Boot(Gm)
    print(f"\n== {f}: {Gm} games, overall G share {fmt(b.ratio(sh))}")
    buckets = [("no effective random move in the contest", first >= 10**6),
               ("first random move at move 0-2", first <= 2),
               ("first random move at move 3-9", (first >= 3) & (first <= 9)),
               ("first random move at move 10-19", (first >= 10) & (first <= 19)),
               ("first random move at move >=20", (first >= 20) & (first < 10**6)),
               ("first random move at move <=9", first <= 9)]
    for name, m in buckets:
        if m.sum() == 0: continue
        ci = b.ratio(sh * m, m.astype(float))
        # deviator-balanced: mean of the G-deviated and S-deviated subsets (removes the deviator's own handicap)
        mg, ms = m & (fd == 1), m & (fd == 2)
        extra = ""
        if mg.sum() and ms.sum():
            bal = 0.5 * (sh[mg].mean() + sh[ms].mean())
            boots = 0.5 * ((b.W @ (sh * mg)) / (b.W @ mg) + (b.W @ (sh * ms)) / (b.W @ ms))
            extra = (f" | G deviated first: {sh[mg].mean():.3f} (n={mg.sum()}), S deviated first: {sh[ms].mean():.3f} (n={ms.sum()}),"
                     f" balanced {bal:.3f} [{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]")
        print(f"  {name:42s} n={m.sum():5d} ({m.mean():.1%})  G share {fmt(ci)}{extra}")
