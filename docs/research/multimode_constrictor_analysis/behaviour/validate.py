import sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
mode, path = sys.argv[1], sys.argv[2]
d = contest_outcome(load(path))
bs, bp = bench_pergame(mode, d["G"])
print("games", d["G"])
print("share G mine/bench max abs diff", np.abs(d["share"][:, 0] - bs["G"]).max(), " S:", np.abs(d["share"][:, 1] - bs["S"]).max())
pg = np.stack([d["place"][np.arange(d["G"])[:,None]*0 + np.arange(d["G"])[:,None], np.arange(4)[None,:]][:, :]],0)
# placement by policy: mean over its seats
pmine = np.stack([(d["place"] * (d["pol"] == p)).sum(1) / 2 for p in (0, 1)], 1)
print("place G diff", np.abs(pmine[:, 0] - bp["G"]).max(), "S", np.abs(pmine[:, 1] - bp["S"]).max())
print("G wins/ties/losses", (d["share"][:,0]==1).sum(), ((d["share"][:,0]>0)&(d["share"][:,0]<1)).sum(), (d["share"][:,0]==0).sum(), "mean G share", d["share"][:,0].mean())
print("bench   wins/ties/losses", (bs["G"]==1).sum(), ((bs["G"]>0)&(bs["G"]<1)).sum(), (bs["G"]==0).sum())
print("draw games", d["draw"].sum())
