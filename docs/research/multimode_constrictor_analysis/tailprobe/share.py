"""Share of the win of G's seats in collected games (bootstrap over games). Usage: python share.py games.npz [dedup]"""
import sys
import numpy as np
d = np.load(sys.argv[1])
fa, et, sg = d["final_alive"], d["final_elim_turn"], d["seat_g"]
G = fa.shape[0]
keep = np.arange(G)
if len(sys.argv) > 2:
    sig = [d["body"][g].tobytes() + sg[g].tobytes() for g in range(G)]
    _, keep = np.unique(sig, return_index=True)
share = np.zeros(fa.shape)
for g in range(G):
    if fa[g].any():
        share[g] = fa[g] / fa[g].sum()
    else:
        w = et[g] == et[g].max(); share[g] = w / w.sum()
x = (share * sg).sum(1)[keep]
rng = np.random.default_rng(3)
bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
print(f"{sys.argv[1]} games={len(x)} G share (2 seats of 4, parity 0.5) = {x.mean():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  G wins={int((x==1).sum())} S wins={int((x==0).sum())} split={int(((x>0)&(x<1)).sum())}")
