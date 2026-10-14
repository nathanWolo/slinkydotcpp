"""Constrictor: when do the two teams get separated, who has more room then, and who fills better?

After the teams are separated (no free cell reachable from a G snake and from an S snake) the game
is a pure pocket race: nobody can interact any more, and tails never move, so the side with more
usable room outlasts the other. This splits the result into (a) the fight for room before the
separation and (b) the filling afterwards. Intervals: cluster bootstrap over games (4000).
"""
import json
import sys

sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from scipy import ndimage
from common import *
from geom import DX, H, W

path, out_path = sys.argv[1], sys.argv[2]
d = contest_outcome(load(path))
Gm, T, N = d["G"], d["T"], d["N"]
pol = d["pol"]
boot = Boot(Gm, cluster=d.get("cluster"))
res = {}
og = d["owner_grid"].astype(int)
head = d["head"].astype(int)
dm = d["move_dead"]
K_end = d["K_end"]

k_sep = np.full(Gm, -1)
room = np.zeros((Gm, 2))  # at separation
n_alive_sep = np.zeros((Gm, 2), int)
conn_hist = np.zeros((Gm, T), int)  # 1 if teams connected at t (for contested states)
for g in range(Gm):
    for t in range(int(K_end[g])):
        if d["done"][g, t]:
            break
        free = og[g, t] == -1
        lab, n = ndimage.label(free)  # 4-connectivity
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        comps = {}
        for i in range(N):
            if not d["alive"][g, t, i]:
                continue
            x, y = head[g, t, i]
            cs = set()
            for k in range(4):
                px, py = x + DX[k, 0], y + DX[k, 1]
                if 0 <= px < W and 0 <= py < H and lab[py, px] > 0:
                    cs.add(int(lab[py, px]))
            comps[i] = cs
        gs_ = [i for i in comps if pol[g, i] == 0]
        ss_ = [i for i in comps if pol[g, i] == 1]
        connected = any(comps[i] & comps[j] for i in gs_ for j in ss_)
        conn_hist[g, t] = int(connected)
        if not connected and k_sep[g] < 0:
            k_sep[g] = t
            for p, members in ((0, gs_), (1, ss_)):
                u = set().union(*[comps[i] for i in members]) if members else set()
                room[g, p] = sum(sizes[c] for c in u)
                n_alive_sep[g, p] = len(members)
            break
sep = k_sep >= 0
res["separated_before_contest_end"] = boot.ratio(sep.astype(float))
res["k_sep_mean"] = boot.ratio(np.where(sep, k_sep, 0).astype(float), sep.astype(float))
print(f"games where the teams were separated while both were alive: {int(sep.sum())} of {Gm}  ({fmt(res['separated_before_contest_end'], pct=True)}); mean separation move {fmt(res['k_sep_mean'], 2)}")
print("  composition at separation (G alive, S alive):", {k: int(v) for k, v in zip(*np.unique([f"{a}G+{b}S" for a, b in n_alive_sep[sep]], return_counts=True))})
share = d["share"][:, 0]
print(f"  G share: games separated {fmt(boot.ratio(share * sep, sep.astype(float)))}, not separated {fmt(boot.ratio(share * ~sep, (~sep).astype(float)))}")
res["share_sep"] = boot.ratio(share * sep, sep.astype(float)); res["share_nosep"] = boot.ratio(share * ~sep, (~sep).astype(float))
print(f"  draws: separated {int((d['draw'] & sep).sum())}, not separated {int((d['draw'] & ~sep).sum())}")
diff = room[:, 0] - room[:, 1]
print(f"  at separation: mean room G {room[sep,0].mean():.1f}  S {room[sep,1].mean():.1f}  (G-S) {fmt(boot.ratio(diff * sep, sep.astype(float)), 2)}")
res["room_diff"] = boot.ratio(diff * sep, sep.astype(float))
res["room_G"] = boot.ratio(room[:, 0] * sep, sep.astype(float)); res["room_S"] = boot.ratio(room[:, 1] * sep, sep.astype(float))
print(f"  room G {fmt(res['room_G'], 2)}   room S {fmt(res['room_S'], 2)}")
# same, per composition
for comp in sorted({(int(a), int(b)) for a, b in n_alive_sep[sep]}):
    s_ = sep & (n_alive_sep[:, 0] == comp[0]) & (n_alive_sep[:, 1] == comp[1])
    print(f"   {comp[0]}G+{comp[1]}S: n={int(s_.sum()):3d}  G share {fmt(boot.ratio(share * s_, s_.astype(float)))}  room G {room[s_,0].mean():.1f} S {room[s_,1].mean():.1f}")

# calibration: G share against room difference at separation
print("  G share by room difference at separation (G room - S room):")
res["calib"] = {}
edges = [(-999, -10), (-9, -4), (-3, 3), (4, 9), (10, 999)]
for lo, hi in edges:
    s_ = sep & (diff >= lo) & (diff <= hi)
    if s_.sum() == 0:
        continue
    ci = boot.ratio(share * s_, s_.astype(float))
    res["calib"][f"{lo}..{hi}"] = {"n": int(s_.sum()), "G_share": ci}
    print(f"     [{lo:5d}, {hi:4d}] n={int(s_.sum()):3d}  G share {fmt(ci)}")
# symmetric check: P(bigger room wins) when G is bigger vs when S is bigger
for lab, sel_ in (("G has >=4 more room", sep & (diff >= 4)), ("S has >=4 more room", sep & (diff <= -4))):
    wins_big = np.where(diff >= 4, share, 1 - share) if "G has" in lab else 1 - share
    ci = boot.ratio(wins_big * sel_, sel_.astype(float))
    res["calib"][lab] = {"n": int(sel_.sum()), "big_room_side_share": ci}
    print(f"     {lab}: n={int(sel_.sum())}, share of the bigger-room side {fmt(ci)}")
# who has more room (|diff|>=4): frequency
a = (sep & (diff >= 4)).astype(float); b = (sep & (diff <= -4)).astype(float)
res["more_room"] = {"G_ahead": boot.ratio(a), "S_ahead": boot.ratio(b), "diff": boot.diff(a, np.ones(Gm), b, np.ones(Gm))}
print(f"  per game: G has >=4 more room {fmt(res['more_room']['G_ahead'], pct=True)}, S has >=4 more room {fmt(res['more_room']['S_ahead'], pct=True)}, diff {fmt(res['more_room']['diff'], 3)}")
# a non-parametric view: P(G wins) among separated games with |diff|<=3 (a fair race)
# filling efficiency of the side that ran out first: moves survived after separation / room at separation
print("  filling efficiency after separation: for the side that lost the pocket race, (moves survived after separation) / (room at separation)")
eff = {0: [], 1: []}
for g in np.where(sep)[0]:
    for p in (0, 1):
        if share[g] in ((0.0, 1.0)[p::1] if False else ()) :
            pass
    for p in (0, 1):
        mine_share = d["share"][g, p]
        if mine_share == 0:  # p lost
            members = [i for i in range(N) if pol[g, i] == p]
            last = max(int(dm[g, i]) for i in members) + 1  # moves played by the last of its snakes
            moves_after = last - k_sep[g]
            if room[g, p] > 0:
                eff[p].append((g, moves_after, room[g, p]))
res["fill"] = {}
for p, nm in enumerate(POLICY):
    arr = np.array(eff[p])
    num = np.zeros(Gm); den = np.zeros(Gm); roomsum = np.zeros(Gm)
    np.add.at(num, arr[:, 0].astype(int), arr[:, 1]); np.add.at(den, arr[:, 0].astype(int), 1); np.add.at(roomsum, arr[:, 0].astype(int), arr[:, 2])
    res["fill"][nm] = {"n": len(arr), "moves_after": boot.ratio(num, den), "room": boot.ratio(roomsum, den), "eff": boot.ratio(num, roomsum)}
    print(f"   {nm} lost {len(arr)} separated races: moves survived after separation {fmt(res['fill'][nm]['moves_after'], 2)}, room at separation {fmt(res['fill'][nm]['room'], 2)}, moves/room {fmt(res['fill'][nm]['eff'], 3)}")
# restrict to races with equal-ish room? compare loser efficiency
json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
np.savez_compressed(out_path.replace(".json", "_arrays.npz"), k_sep=k_sep, room=room, n_alive_sep=n_alive_sep, conn_hist=conn_hist)

# ------- when does the first death happen and what is it
