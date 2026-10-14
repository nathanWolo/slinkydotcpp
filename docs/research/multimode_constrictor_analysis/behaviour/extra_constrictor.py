"""Constrictor extras on the 96 distinct games: aggregated death classes, outcome by start type and by
first head-on, and the exact accounting of G's deficit."""
import json, sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
path, tabpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = contest_outcome(load(path, dedupe=True))
tab = np.load(tabpath, allow_pickle=True)
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]; boot = Boot(Gm, cluster=d["cluster"])
res = {}
cause, dead, cdead, cls = tab["cause"], tab["dead"], tab["contested_death"], tab["cls"]
share = d["share"][:, 0]
print(f"{Gm} distinct games, {d['cluster'].max()+1} placements, multiplicity min/median/max in the 512-game sample: {d['mult'].min()}/{int(np.median(d['mult']))}/{d['mult'].max()}")
print(f"exact G share (unweighted over all {Gm} starts x seat parities): {share.mean():.4f}; wins {int((share==1).sum())} draws {int((share==0.5).sum())} losses {int((share==0).sum())}")
print(f"512-game sample (multiplicity-weighted): {(share*d['mult']).sum()/d['mult'].sum():.4f}  <- what the benchmark's first 512 games estimate")
# sampling noise of the benchmark estimator: games are draws (with replacement) of the 96 configs; sd of a mean of n draws
pop_sd = share.std()
print(f"per-game sd of G share over the 96 configs {pop_sd:.3f}; so an n-game benchmark estimate has sampling sd {pop_sd/np.sqrt(512):.3f} (n=512) and {pop_sd/np.sqrt(1024):.3f} (n=1024) about the exact value; its reported +-1.96*se is about the same size -- but note the draws are NOT independent games of a continuing process, they are repeats of a fixed set")
# 1. aggregated death classes (contested deaths), per seat
trapped = cdead & np.isin(cause, [1, 2, 3, 4, 6])
ho_opp = cdead & (cls == "head-on with opponent")
ho_mate = cdead & (cls == "head-on with teammate")
surv = ~cdead
print("-- fate of a seat during the contest (fraction of seats), start-level 95% intervals")
res["fate"] = {}
for nm, ind in (("no safe move left (wall/self/body: forced)", trapped), ("head-on with an opponent (both die)", ho_opp), ("head-on with a teammate (both die)", ho_mate), ("alive when the contest ended", surv)):
    ng, ns = by_pol(ind.astype(float), pol, 0), by_pol(ind.astype(float), pol, 1)
    a, b, df = boot.ratio(ng, np.full(Gm, 2.0)), boot.ratio(ns, np.full(Gm, 2.0)), boot.diff(ng, np.full(Gm, 2.0), ns, np.full(Gm, 2.0))
    res["fate"][nm] = dict(G=a, S=b, diff=df, nG=int(ng.sum()), nS=int(ns.sum()))
    print(f"   {nm:45s} G {fmt(a, pct=True):>22s}  S {fmt(b, pct=True):>22s}  G-S {fmt(df, 3):>22s} {'*' if sig(df) else ' '} (seats {int(ng.sum())} of {2*Gm}, {int(ns.sum())} of {2*Gm})")
# 2. start type
pos0 = d["head"][:, 0]  # [G,N,2]
is_corner = np.array([(abs(p[0, 0] - 5) > 2 and abs(p[0, 1] - 5) > 2) or (abs(p[0, 0] - 5) > 2 and abs(p[0, 1] - 5) > 2) for p in pos0])
# corner starts: both coordinates in {1, 9}
is_corner = np.array([all(x in (1, 9) and y in (1, 9) for x, y in p) for p in pos0])
par = d["game_id"] % 2
print("-- G share by start type and seat parity (exact over the games in each cell)")
res["start_type"] = {}
for nm, sel in (("corner starts", is_corner), ("cardinal starts", ~is_corner), ("G in seats 0,2 (even game id)", par == 0), ("G in seats 1,3 (odd game id)", par == 1)):
    ci = boot.ratio(share * sel, sel.astype(float))
    w_ = ((share == 1) & sel).sum(); dr = ((share == .5) & sel).sum(); l_ = ((share == 0) & sel).sum()
    res["start_type"][nm] = dict(n=int(sel.sum()), share=ci, wins=int(w_), draws=int(dr), losses=int(l_))
    print(f"   {nm:32s} n={int(sel.sum()):3d}  G share {share[sel].mean():.3f} [{ci[1]:.3f},{ci[2]:.3f}]  W/D/L {w_}/{dr}/{l_}")
# per placement: average of the two parities
cl = d["cluster"]; C = cl.max() + 1
pshare = np.array([share[cl == c].mean() for c in range(C)])
print(f"   per placement (mean of both parities): G ahead in {int((pshare > .5).sum())}, level in {int((pshare == .5).sum())}, behind in {int((pshare < .5).sum())} of {C}")
res["placements"] = dict(ahead=int((pshare > .5).sum()), level=int((pshare == .5).sum()), behind=int((pshare < .5).sum()), n=int(C))
# sign-test style: among decided placements
a_, b_ = int((pshare > .5).sum()), int((pshare < .5).sum())
from math import comb
n_ = a_ + b_
p_two = min(1.0, 2 * sum(comb(n_, k) for k in range(0, min(a_, b_) + 1)) / 2 ** n_)
print(f"   sign test over placements (ties dropped): {a_} vs {b_}, two-sided p = {p_two:.3f}")
res["sign_test_p"] = p_two
# 3. outcome by first head-on
print("-- outcome by the first head-on of the contest")
first = {}
for g in range(Gm):
    kind = "no head-on"
    for t_ in range(min(T, int(d["K_end"][g]))):
        if d["done"][g, t_]:
            break
        al = d["alive"][g, t_]; nh_ = d["newhead"][g, t_]
        cells = {}
        for i in range(N):
            if al[i]:
                cells.setdefault(tuple(nh_[i]), []).append(i)
        ev = [("".join(sorted(POLICY[pol[g, i]] for i in m))) for c, m in cells.items() if len(m) >= 2]
        if ev:
            kind = "head-on " + ev[0]; first[g] = (kind, t_); break
    first.setdefault(g, (kind, -1))
kinds = sorted({v[0] for v in first.values()})
res["by_first_headon"] = {}
tot_games = Gm
for k in kinds:
    sel = np.array([first[g][0] == k for g in range(Gm)])
    ci = boot.ratio(share * sel, sel.astype(float))
    w_ = ((share == 1) & sel).sum(); dr = ((share == .5) & sel).sum(); l_ = ((share == 0) & sel).sum()
    mt = np.mean([first[g][1] for g in range(Gm) if sel[g]]) if k != "no head-on" else float("nan")
    res["by_first_headon"][k] = dict(n=int(sel.sum()), G_share_mean=float(share[sel].mean()), wins=int(w_), draws=int(dr), losses=int(l_))
    print(f"   {k:14s} n={int(sel.sum()):3d} ({100*sel.mean():.0f}% of games; mean move {mt:.1f})  G share {share[sel].mean():.3f}  W/D/L {w_}/{dr}/{l_}")
# contribution to the deficit relative to parity: sum over groups of n*(share-0.5)
print("-- accounting of G's deficit vs parity (exact): sum over games of (G share - 0.5) =", f"{(share - 0.5).sum():.1f}", f"(= {(share-0.5).mean():.4f} per game)")
for k in kinds:
    sel = np.array([first[g][0] == k for g in range(Gm)])
    print(f"   {k:14s} contributes {(share[sel]-0.5).sum():+6.1f} game-units ({(share[sel]-0.5).sum()/Gm:+.4f} per game)")
json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
