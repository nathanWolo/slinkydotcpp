"""Royale contrast: how long are G's and S's snakes, and how do head-on collisions end?"""
import json, sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
d = contest_outcome(load(sys.argv[1], dedupe=False))
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]; boot = Boot(Gm)
cv = d["alive"] & ~d["done"][:, :, None] & (np.arange(T)[None, :] < d["K_end"][:, None])[:, :, None]
res = {}
print(f"{Gm} games; mean length of living snakes (contested states) at move t")
for t in (0, 25, 50, 75, 100, 150, 200):
    if t >= T: continue
    s_ = cv[:, t, :]
    ng = (d["length"][:, t, :] * s_ * (pol == 0)).sum(1); dg = (s_ * (pol == 0)).sum(1).astype(float)
    ns = (d["length"][:, t, :] * s_ * (pol == 1)).sum(1); ds = (s_ * (pol == 1)).sum(1).astype(float)
    if dg.sum() < 5 or ds.sum() < 5: continue
    a, b, df = boot.ratio(ng, dg), boot.ratio(ns, ds), boot.diff(ng, dg, ns, ds)
    res[f"len_t{t}"] = dict(G=a, S=b, diff=df)
    print(f"  t={t:3d}: G {fmt(a, 2)}  S {fmt(b, 2)}  G-S {fmt(df, 2)} {'*' if sig(df) else ''}")
# pooled over contested states
ng = (d["length"] * cv * (pol[:, None, :] == 0)).sum((1, 2)); dg = (cv * (pol[:, None, :] == 0)).sum((1, 2)).astype(float)
ns = (d["length"] * cv * (pol[:, None, :] == 1)).sum((1, 2)); ds = (cv * (pol[:, None, :] == 1)).sum((1, 2)).astype(float)
res["len_pooled"] = dict(G=boot.ratio(ng, dg), S=boot.ratio(ns, ds), diff=boot.diff(ng, dg, ns, ds))
print("  pooled over all contested snake-moves: G", fmt(res["len_pooled"]["G"], 2), "S", fmt(res["len_pooled"]["S"], 2), "G-S", fmt(res["len_pooled"]["diff"], 2))
# fraction of GS head-on collisions where the G snake is longer
rows = []
for g in range(Gm):
    for t in range(min(T, int(d["K_end"][g]))):
        if d["done"][g, t]: break
        al = d["alive"][g, t]; nh = d["newhead"][g, t]
        for i in range(N):
            for j in range(i + 1, N):
                if al[i] and al[j] and (nh[i] == nh[j]).all() and pol[g, i] != pol[g, j] and 0 <= nh[i, 0] < 11 and 0 <= nh[i, 1] < 11:
                    gi, si = (i, j) if pol[g, i] == 0 else (j, i)
                    rows.append((g, int(d["length"][g, t, gi]), int(d["length"][g, t, si])))
rows = np.array(rows)
dl = rows[:, 1] - rows[:, 2]
print(f"  G-vs-S head-on events: {len(rows)}; G longer {int((dl>0).sum())}, equal {int((dl==0).sum())}, S longer {int((dl<0).sum())}; mean length difference (G - S) at the collision {dl.mean():.2f}")
num = np.zeros(Gm); den = np.zeros(Gm)
np.add.at(num, rows[:, 0], (dl > 0)); np.add.at(den, rows[:, 0], 1)
res["headon_G_longer"] = boot.ratio(num, den)
print("  share of G-S head-ons where G is longer:", fmt(res["headon_G_longer"], pct=True))
# the same, if G and S had equal length it would be 0.5 minus ties: report how often the longer snake survives
surv_long = 0
print("  (the longer snake survives by rule; equal length kills both)")
json.dump(res, open(sys.argv[2], "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
