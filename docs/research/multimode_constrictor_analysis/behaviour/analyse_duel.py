"""Constrictor: the 1 G vs 1 S duels (first state with exactly one snake of each policy alive).

For each such game: are the two snakes still connected through free cells, how much room does each
have (free cells reachable from its head), who wins. Separated duels are pure pocket races.
"""
import json, sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from scipy import ndimage
from common import *
from geom import DX, H, W, voronoi
path, out_path = sys.argv[1], sys.argv[2]
tab = np.load(sys.argv[3], allow_pickle=True)
d = contest_outcome(load(path))
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]; boot = Boot(Gm, cluster=d.get("cluster"))
og = d["owner_grid"].astype(int); head = d["head"].astype(int); dm = d["move_dead"]
cls = tab["cls"]; cause = tab["cause"]
seq = d["alive_seq"]
cnt = np.stack([(seq & (pol[:, None, :] == p)).sum(-1) for p in (0, 1)], -1)
rows = []
for g in range(Gm):
    w = np.where((cnt[g, :, 0] == 1) & (cnt[g, :, 1] == 1))[0]
    if not len(w) or w[0] >= int(d["K_end"][g]) or w[0] >= T or d["done"][g, w[0]]:
        continue
    k0 = int(w[0])
    iG = [i for i in range(N) if d["alive"][g, k0, i] and pol[g, i] == 0][0]
    iS = [i for i in range(N) if d["alive"][g, k0, i] and pol[g, i] == 1][0]
    free = og[g, k0] == -1
    lab, n = ndimage.label(free)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    def comps(i):
        x, y = head[g, k0, i]
        cs = set()
        for k in range(4):
            px, py = x + DX[k, 0], y + DX[k, 1]
            if 0 <= px < W and 0 <= py < H and lab[py, px] > 0:
                cs.add(int(lab[py, px]))
        return cs
    cg, cs_ = comps(iG), comps(iS)
    connected = bool(cg & cs_)
    rg, rs = sum(sizes[c] for c in cg), sum(sizes[c] for c in cs_)
    tv = voronoi(free[None], head[g, k0][None], d["alive"][g, k0][None])[0]
    # route: how did the other two seats die (cause classes), in order
    others = [i for i in range(N) if i not in (iG, iS)]
    route = sorted(f"{POLICY[pol[g,i]]}:{cls[g,i]}" for i in others)
    rows.append(dict(g=g, k0=k0, connected=connected, rG=int(rg), rS=int(rs), tG=int(tv[iG]), tS=int(tv[iS]), share=float(d["share"][g, 0]), route=route, iG=iG, iS=iS))
R = rows
print(f"games with a 1G-vs-1S state while undecided: {len(R)} of {Gm}")
sh = np.array([r["share"] for r in R]); gidx = np.array([r["g"] for r in R])
def gci(mask_):
    num = np.zeros(Gm); den = np.zeros(Gm)
    np.add.at(num, gidx[mask_], sh[mask_]); np.add.at(den, gidx[mask_], 1)
    return boot.ratio(num, den)
con = np.array([r["connected"] for r in R]); rG = np.array([r["rG"] for r in R]); rS = np.array([r["rS"] for r in R])
tG = np.array([r["tG"] for r in R]); tS = np.array([r["tS"] for r in R])
res = {"n": len(R)}
print(f"  G share: all {fmt(gci(np.ones(len(R), bool)))}; connected {fmt(gci(con))} (n={int(con.sum())}); separated {fmt(gci(~con))} (n={int((~con).sum())})")
res["share_all"], res["share_conn"], res["share_sep"] = gci(np.ones(len(R), bool)), gci(con), gci(~con)
print(f"  mean k0 {np.mean([r['k0'] for r in R]):.1f}")
print("  --- separated duels: pure pocket races. room = free cells reachable from the head.")
sep = ~con
dr = rG - rS
print(f"  mean room: G {rG[sep].mean():.2f}  S {rS[sep].mean():.2f}  (G-S {dr[sep].mean():.2f})")
for lo, hi in [(-999, -10), (-9, -4), (-3, -1), (0, 0), (1, 3), (4, 9), (10, 999)]:
    s_ = sep & (dr >= lo) & (dr <= hi)
    if s_.sum():
        ci = gci(s_)
        res[f"sep_room_{lo}..{hi}"] = {"n": int(s_.sum()), "G_share": ci}
        print(f"     G room - S room in [{lo:4d},{hi:4d}] n={int(s_.sum()):3d}  G share {fmt(ci)}")
# symmetric view: share of the bigger-room side, split by policy
print("  the side with the bigger room wins how often, by policy:")
for lo, hi in [(1, 3), (4, 9), (10, 999)]:
    sg = sep & (dr >= lo) & (dr <= hi)  # G bigger
    ss_ = sep & (-dr >= lo) & (-dr <= hi)  # S bigger
    cg_, cs2_ = gci(sg), None
    num = np.zeros(Gm); den = np.zeros(Gm); np.add.at(num, gidx[ss_], 1 - sh[ss_]); np.add.at(den, gidx[ss_], 1)
    cs2_ = boot.ratio(num, den)
    # difference S-big minus G-big (paired bootstrap not possible across disjoint subsets of games; use independent resample difference)
    numg = np.zeros(Gm); deng = np.zeros(Gm); np.add.at(numg, gidx[sg], sh[sg]); np.add.at(deng, gidx[sg], 1)
    df = boot.diff(num, den, numg, deng)
    res[f"bigger_room_{lo}..{hi}"] = {"nG_big": int(sg.sum()), "nS_big": int(ss_.sum()), "G_big_win": cg_, "S_big_win": cs2_, "S_minus_G": df}
    print(f"     room advantage {lo}..{hi}: G bigger n={int(sg.sum())} wins {fmt(cg_)} | S bigger n={int(ss_.sum())} wins {fmt(cs2_)} | S-G {fmt(df)} {'*' if sig(df) else ''}")
# loser efficiency in separated duels
print("  loser filling in separated duels (moves survived after k0 / room at k0):")
for p, nm in enumerate(POLICY):
    sel = [r for r in R if (not r["connected"]) and r["share"] == (0.0 if p == 0 else 1.0)]
    mv = np.array([ (dm[r["g"], r["iG" if p == 0 else "iS"]] + 1 - r["k0"]) for r in sel], float)
    rm = np.array([ (r["rG"] if p == 0 else r["rS"]) for r in sel], float)
    gg = np.array([r["g"] for r in sel])
    n_ = np.zeros(Gm); d_ = np.zeros(Gm); r_ = np.zeros(Gm)
    np.add.at(n_, gg, mv); np.add.at(d_, gg, 1); np.add.at(r_, gg, rm)
    res[f"loser_fill_{nm}"] = {"n": len(sel), "moves": boot.ratio(n_, d_), "room": boot.ratio(r_, d_), "moves_per_room": boot.ratio(n_, r_)}
    print(f"     {nm} as loser: n={len(sel)} moves {fmt(boot.ratio(n_, d_), 2)}, room {fmt(boot.ratio(r_, d_), 2)}, moves/room {fmt(boot.ratio(n_, r_), 3)}")
# when the loser's room was within 3 cells of the winner's or larger: i.e., it lost a race it was not smaller in
print("  separated duels lost by the side that had >= the room of the winner (it wasted room):")
for p, nm in enumerate(POLICY):
    if p == 0:
        s_ = sep & (sh == 0) & (rG >= rS)
        tot = sep & (rG >= rS)
    else:
        s_ = sep & (sh == 1) & (rS >= rG)
        tot = sep & (rS >= rG)
    num = np.zeros(Gm); den = np.zeros(Gm); np.add.at(num, gidx[s_], 1); np.add.at(den, gidx[tot], 1)
    ci = boot.ratio(num, den)
    res[f"lost_with_geq_room_{nm}"] = {"n_lost": int(s_.sum()), "n_geq": int(tot.sum()), "frac": ci}
    print(f"     {nm}: lost {int(s_.sum())} of {int(tot.sum())} duels in which it had >= the other's room: {fmt(ci, pct=True)}")
df_ = boot.diff(*(lambda a, b: (a[0], a[1], b[0], b[1]))(*[ (np.bincount(gidx[(sep & (sh == (0 if p==0 else 1)) & ((rG >= rS) if p==0 else (rS >= rG)))], minlength=Gm).astype(float), np.bincount(gidx[sep & ((rG >= rS) if p==0 else (rS >= rG))], minlength=Gm).astype(float)) for p in (0, 1)]))
res["lost_with_geq_room_diff_G_minus_S"] = df_
print(f"     G - S: {fmt(df_, 3)} {'*' if sig(df_) else ''}")
print("  --- connected duels (the fight): territory (Voronoi) and outcome")
cs_ = con
print(f"  n={int(cs_.sum())}; draws {int(((sh == 0.5) & cs_).sum())}; G wins {int(((sh == 1) & cs_).sum())}; S wins {int(((sh == 0) & cs_).sum())}")
res["conn_counts"] = {"draw": int(((sh == 0.5) & cs_).sum()), "G": int(((sh == 1) & cs_).sum()), "S": int(((sh == 0) & cs_).sum())}
print(f"  mean Voronoi cells G {tG[cs_].mean():.2f} S {tS[cs_].mean():.2f}; mean room G {rG[cs_].mean():.1f} S {rS[cs_].mean():.1f}")
da = tG - tS
for lo, hi in [(-999, -10), (-9, -4), (-3, 3), (4, 9), (10, 999)]:
    s_ = cs_ & (da >= lo) & (da <= hi)
    if s_.sum():
        n_g = ((sh == 1) & s_).sum(); n_d = ((sh == .5) & s_).sum(); n_s = ((sh == 0) & s_).sum()
        res[f"conn_terr_{lo}..{hi}"] = {"n": int(s_.sum()), "G_share": gci(s_), "G": int(n_g), "draw": int(n_d), "S": int(n_s)}
        print(f"     Voronoi G-S in [{lo:4d},{hi:4d}] n={int(s_.sum()):3d}  G wins {n_g}, draws {n_d}, S wins {n_s}; G share {fmt(gci(s_))}")
# route into the duel
from collections import Counter
routes = Counter(" + ".join(r["route"]) for r in R)
print("  route into the duel (the two other seats' death classes):")
for k, v in routes.most_common():
    s_ = np.array([" + ".join(r["route"]) == k for r in R])
    print(f"     {v:3d}  {k:70s} connected {int((con & s_).sum())}/{int(s_.sum())}  G share {fmt(gci(s_))}")
res["routes"] = {k: v for k, v in routes.items()}
json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

if len(sys.argv) > 4:
    from collections import Counter
    c = Counter((r["connected"], r["rG"], r["rS"], r["share"]) for r in R)
    for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:30]:
        print(v, k)
    print("k0 distribution", Counter(r["k0"] for r in R).most_common(12))
