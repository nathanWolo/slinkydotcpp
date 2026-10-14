import sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
from geom import *
# unit test voronoi: empty 11x11, two sources at (0,5) and (10,5): each should get 55-ish minus the shared column
free = np.ones((1, 11, 11), bool)
src = np.array([[[0, 5], [10, 5]]])
free[0, 5, 0] = False; free[0, 5, 10] = False
print(voronoi(free, src, np.array([[True, True]])))
# delay: if source 0 moves first (delay 1) into (1,5)
src2 = np.array([[[1, 5], [10, 5]]]); free2 = np.ones((1,11,11), bool); free2[0,5,10]=False
print(voronoi(free2, src2, np.array([[True, True]]), np.array([[1, 0]])))
d = contest_outcome(load(sys.argv[1]))
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]
seq = d["alive_seq"]
cnt = np.stack([(seq & (pol[:, None, :] == p)).sum(-1) for p in (0, 1)], -1)
shown = 0
for g in range(Gm):
    w = np.where((cnt[g, :, 0] == 1) & (cnt[g, :, 1] == 1))[0]
    if len(w) and shown < 4:
        k0 = w[0]
        og = d["owner_grid"][g, k0].astype(int)
        print(f"game {g} duel start at state {k0}; G share {d['share'][g,0]}; seats alive {np.where(d['alive'][g,k0])[0]}  pol {pol[g]}")
        for yy in range(10, -1, -1):
            row = ""
            for xx in range(11):
                o = og[yy, xx]
                hd = [i for i in range(N) if d["alive"][g, k0, i] and tuple(d["head"][g, k0, i]) == (xx, yy)]
                if hd: row += POLICY[pol[g, hd[0]]]
                elif o >= 0: row += POLICY[pol[g, o]].lower()
                else: row += "."
            print(row)
        free_ = og[None] == -1
        t_ = voronoi(free_, d["head"][g, k0][None].astype(int), d["alive"][g, k0][None])
        print("terr", t_)
        shown += 1
