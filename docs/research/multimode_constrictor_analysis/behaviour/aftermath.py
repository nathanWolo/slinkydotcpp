"""What does a G-S head-on leave behind? Territory of the four roles just before the first head-on
(first head-on = exactly one G and one S snake entering the same cell, all four alive)."""
import sys, json
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
from geom import voronoi, region_sizes
d = contest_outcome(load(sys.argv[1], dedupe=False))
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]; boot = Boot(Gm)
og = d["owner_grid"].astype(int); head = d["head"].astype(int)
rows = []
for g in range(Gm):
    for t in range(min(T, int(d["K_end"][g]))):
        if d["done"][g, t]: break
        al = d["alive"][g, t]; nh = d["newhead"][g, t]; cells = {}
        for i in range(N):
            if al[i]: cells.setdefault(tuple(nh[i]), []).append(i)
        ev = [m for m in cells.values() if len(m) >= 2]
        if ev:
            m = ev[0]
            if len(m) == 2 and pol[g, m[0]] != pol[g, m[1]] and al.all():
                gi = m[0] if pol[g, m[0]] == 0 else m[1]; si = m[1] if gi == m[0] else m[0]
                oth = [i for i in range(N) if i not in m]
                gc = [i for i in oth if pol[g, i] == 0][0]; sd = [i for i in oth if pol[g, i] == 1][0]
                rows.append((g, t, gi, si, gc, sd))
            break
print(f"games whose first head-on is a G-S collision among four living snakes: {len(rows)} of {Gm}")
R = np.array(rows)
gs, ts = R[:, 0], R[:, 1]
free = og[gs, ts] == -1
terr = voronoi(free, head[gs, ts], d["alive"][gs, ts])
reg = region_sizes(free, head[gs, ts], d["alive"][gs, ts])
roles = {"G in the collision": 2, "S in the collision": 3, "other G (survivor)": 4, "other S (survivor)": 5}
res = {}
print("Voronoi cells of each role at the state before the collision (mean, game-level bootstrap):")
num = {k: np.zeros(Gm) for k in roles}; den = np.zeros(Gm)
np.add.at(den, gs, 1)
for k, c in roles.items():
    np.add.at(num[k], gs, terr[np.arange(len(gs)), R[:, c]])
    res[k] = boot.ratio(num[k], den)
    print(f"   {k:22s} {fmt(res[k], 2)}")
dd = boot.diff(num["other G (survivor)"], den, num["other S (survivor)"], den)
print("   other G minus other S:", fmt(dd, 2), "*" if sig(dd) else "")
res["survivors_diff"] = dd
dd2 = boot.diff(num["G in the collision"], den, num["S in the collision"], den)
print("   G minus S in the collision:", fmt(dd2, 2), "*" if sig(dd2) else "")
# the duel afterwards
sh = d["share"][gs, 0]
nsh = np.zeros(Gm); np.add.at(nsh, gs, sh)
res["G_share_after"] = boot.ratio(nsh, den)
print("G share of these games:", fmt(res["G_share_after"]))
# is the survivor with more territory the winner?
adv = terr[np.arange(len(gs)), R[:, 4]] - terr[np.arange(len(gs)), R[:, 5]]
print("share of games won by the survivor that had more Voronoi cells at the collision (decisive games only): ", end="")
dec = (sh == 0) | (sh == 1)
win_more = ((adv > 0) & (sh == 1)) | ((adv < 0) & (sh == 0))
print(f"{win_more[dec & (adv != 0)].mean():.3f} (n={int((dec & (adv != 0)).sum())})")
# mean territory-advantage of survivors by who is the survivor, and share of games where S_d has >= 3 more
for lab, m in (("S survivor has >=3 more cells", adv <= -3), ("G survivor has >=3 more cells", adv >= 3), ("within 2 cells", np.abs(adv) <= 2)):
    n_ = np.zeros(Gm); d_ = np.zeros(Gm); np.add.at(n_, gs[m], sh[m]); np.add.at(d_, gs[m], 1)
    ci = boot.ratio(n_, d_)
    fr = boot.ratio(d_, den)
    res[lab] = dict(frac=fr, G_share=ci, n=int(m.sum()))
    print(f"   {lab:32s} {fmt(fr, pct=True):>22s} of these games, G share {fmt(ci)} (n={int(m.sum())})")
json.dump(res, open(sys.argv[2], "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
