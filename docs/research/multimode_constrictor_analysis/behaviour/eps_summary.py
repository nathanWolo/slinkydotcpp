import sys, hashlib
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
D = "docs/research/multimode_constrictor_analysis/behaviour"
print("policy noise      games  distinct  G share [95% CI]         G wins / draws / G losses")
for name, f in (("greedy (benchmark)", "constrictor512"), ("eps=0.02", "constrictor2048_eps0.02"), ("eps=0.05", "constrictor2048_eps0.05"), ("eps=0.10", "constrictor2048_eps0.1"), ("sampled (T=1)", "constrictor2048_sampled")):
    raw = np.load(f"{D}/{f}.npz")
    tr = {hashlib.md5(raw["act"][g].tobytes() + raw["alive"][g].tobytes()).hexdigest() for g in range(raw["act"].shape[0])}
    d = contest_outcome(load(f"{D}/{f}.npz", dedupe=False))
    b = Boot(d["G"]); sh = d["share"][:, 0]
    ci = b.ratio(sh)
    w, dr, l = (sh == 1).sum(), (sh == .5).sum() + ((sh > 0) & (sh < 1) & (sh != .5)).sum(), (sh == 0).sum()
    print(f"{name:18s} {d['G']:5d}  {len(tr):6d}   {fmt(ci)}     {w} / {dr} / {l}")
