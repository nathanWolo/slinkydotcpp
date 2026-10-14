"""First-head-on accounting and aggregated fate classes with game-level bootstrap (non-deterministic data)."""
import json, sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
path, tabpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = contest_outcome(load(path, dedupe=False)); tab = np.load(tabpath, allow_pickle=True)
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]; boot = Boot(Gm)
share = d["share"][:, 0]; res = {}
cause, cdead, cls = tab["cause"], tab["contested_death"], tab["cls"]
trapped = cdead & np.isin(cause, [1, 2, 3, 4, 6])
fates = (("no safe move left (wall/self/any body)", trapped), ("head-on with an opponent", cdead & (cls == "head-on with opponent")), ("head-on with a teammate", cdead & (cls == "head-on with teammate")), ("alive when the contest ended", ~cdead))
print("-- fate of a seat during the contest (fraction of seats)")
res["fate"] = {}
for nm, ind in fates:
    ng, ns = by_pol(ind.astype(float), pol, 0), by_pol(ind.astype(float), pol, 1)
    a, b, df = boot.ratio(ng, np.full(Gm, 2.0)), boot.ratio(ns, np.full(Gm, 2.0)), boot.diff(ng, np.full(Gm, 2.0), ns, np.full(Gm, 2.0))
    res["fate"][nm] = dict(G=a, S=b, diff=df)
    print(f"   {nm:40s} G {fmt(a, pct=True):>22s}  S {fmt(b, pct=True):>22s}  G-S {fmt(df, 3):>22s} {'*' if sig(df) else ' '}")
kind = np.full(Gm, "none", object); move = np.full(Gm, -1)
for g in range(Gm):
    for t_ in range(min(T, int(d["K_end"][g]))):
        if d["done"][g, t_]: break
        al = d["alive"][g, t_]; nh_ = d["newhead"][g, t_]; cells = {}
        for i in range(N):
            if al[i]: cells.setdefault(tuple(nh_[i]), []).append(i)
        ev = ["".join(sorted(POLICY[pol[g, i]] for i in m)) for c, m in cells.items() if len(m) >= 2]
        if ev:
            kind[g] = ev[0]; move[g] = t_; break
print("-- outcome by the type of the first head-on of the contest (G share; contribution to G's deficit in share-units per game)")
res["first_headon"] = {}
tot = (share - 0.5).mean()
for k in sorted(set(kind)):
    sel = kind == k
    ci = boot.ratio(share * sel, sel.astype(float))
    contrib = boot.ratio((share - 0.5) * sel)
    frac = boot.ratio(sel.astype(float))
    res["first_headon"][k] = dict(frac=frac, share=ci, contrib=contrib, W=int(((share == 1) & sel).sum()), D=int(((share > 0) & (share < 1) & sel).sum()), L=int(((share == 0) & sel).sum()))
    print(f"   first head-on {k:5s}: {fmt(frac, pct=True):>22s} of games, G share {fmt(ci)}, W/D/L {res['first_headon'][k]['W']}/{res['first_headon'][k]['D']}/{res['first_headon'][k]['L']}, contribution {fmt(contrib, 4)}")
print(f"   total G deficit vs parity per game: {tot:+.4f}")
json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
