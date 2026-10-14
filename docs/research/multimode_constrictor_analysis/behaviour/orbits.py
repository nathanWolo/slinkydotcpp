"""Are the 96 (placement, parity) constrictor games really only 12 distinct games?
Canonical key: (set of start cells, set of start cells held by G). Check that every key's games have the same
per-position action sequences (up to relabelling snakes of the same team) and the same outcome."""
import sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from collections import defaultdict
from common import *
d = contest_outcome(load(sys.argv[1], dedupe=True))
Gm, T, N = d["G"], d["T"], d["N"]; pol = d["pol"]
groups = defaultdict(list)
for g in range(Gm):
    cells = [tuple(d["head"][g, 0, i]) for i in range(N)]
    gset = tuple(sorted(c for i, c in enumerate(cells) if pol[g, i] == 0))
    groups[(tuple(sorted(cells)), gset)].append(g)
print("distinct (start set, G start cells) groups:", len(groups), " group sizes:", sorted({len(v) for v in groups.values()}))
bad = 0
for k, gl in groups.items():
    sigs = set()
    for g in gl:
        # action sequence per start cell, truncated at the contest end; dead snakes' actions are irrelevant
        traj = {}
        for i in range(N):
            cell = tuple(d["head"][g, 0, i])
            seq = []
            for t in range(T):
                if d["alive"][g, t, i] and not d["done"][g, t]:
                    seq.append(int(d["act"][g, t, i]))
            traj[cell] = tuple(seq)
        sigs.add((tuple(sorted(traj.items())), tuple(d["share"][g])))
    if len(sigs) != 1:
        bad += 1
print("groups whose games differ:", bad)
kinds = {}
for (cells, gset), gl in sorted(groups.items()):
    corner = all(x in (1, 9) and y in (1, 9) for x, y in cells)
    sh = d["share"][gl[0], 0]
    kinds[(corner, gset)] = sh
    print(("corners  " if corner else "cardinal ") + "G on " + str(list(gset)) + f" -> G share {sh}")
vals = np.array(list(kinds.values()))
print("12 distinct games: G wins", int((vals == 1).sum()), "draws", int((vals == .5).sum()), "losses", int((vals == 0).sum()), "mean share", vals.mean())
