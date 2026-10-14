import sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
d = contest_outcome(load(sys.argv[1]))
g = int(sys.argv[2]); t0 = int(sys.argv[3]); t1 = int(sys.argv[4])
pol = d["pol"]; N = 4
print("pol", pol[g], "share", d["share"][g], "elim", d["final_elim_turn"][g], "cause", d["final_elim_cause"][g])
for t in range(t0, t1):
    og = d["owner_grid"][g, t].astype(int)
    print(f"--- state {t}: alive {d['alive'][g,t].astype(int)} act {d['act'][g,t]} (ag {d['ag'][g,t]}, as {d['a_s'][g,t]}) mask {d['mask'][g,t].astype(int).tolist()} done {d['done'][g,t]}")
    for yy in range(10, -1, -1):
        row = ""
        for xx in range(11):
            o = og[yy, xx]
            hd = [i for i in range(N) if d["alive"][g, t, i] and tuple(d["head"][g, t, i]) == (xx, yy)]
            if hd: row += POLICY[pol[g, hd[0]]]
            elif o >= 0: row += POLICY[pol[g, o]].lower()
            else: row += "."
        print(row)
