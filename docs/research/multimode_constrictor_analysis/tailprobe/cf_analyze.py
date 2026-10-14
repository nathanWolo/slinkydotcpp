"""Behaviour of G and S under counterfactual countdown inputs (cf_offline.py output).

Usage: python cf_analyze.py decisions.npz cf.npz [OUT.json]
Bootstrap over games (B=2000), paired: cf - real on the same decisions.
"""

import json
import sys

import numpy as np

from analyze import Boot, fmt, pick

T = 8
d = dict(np.load(sys.argv[1]))
c = dict(np.load(sys.argv[2]))
bt = Boot(d["game"])
L, ac, ast = d["legal"], d["a_con"], d["a_std"]
amax = np.where(L, ac, -1).max(1)
amin = np.where(L, ac, 10**6).min(1)
trapmv = L & (ac < T)
safe = (L & (ac >= 2 * T)).any(1)
elig = trapmv.any(1) & safe
masked_mv = trapmv & (ast >= 2 * T)
em = masked_mv.any(1) & safe
plain_elig = (trapmv & (ast < 2 * T)).any(1) & safe & ~masked_mv.any(1)
cbest = L & (ac == amax[:, None])
spread = (amax - amin) >= 5
smax = np.where(L, ast, -1).max(1)
sbest = L & (ast == smax[:, None])
div5 = (np.where(sbest, ac, -1).max(1) <= amax - 5) & (smax > 0)
one = np.ones(len(L), bool)

metrics = {
    "flip": (None, one),
    "trapT8": (trapmv, elig),
    "maskedtrapT8": (masked_mv, em),
    "plaintrapT8": (trapmv, plain_elig),
    "conbest_spread5": (cbest, spread),
    "div5_stdbest": (sbest, div5),
}
out = {}
variants = [k[4:] for k in c if k.startswith("c_g_") and k != "c_g_real"]
print(f"{sys.argv[1]}: {len(L)} decisions, {bt.ng} games")
for p in ("g", "s"):
    real = c[f"c_{p}_real"]
    for m, (mv, den) in metrics.items():
        line = f"{p.upper()} {m:16s} n={int(den.sum()):6d}"
        if mv is not None:
            e = bt.ratio(pick(mv, real) & den, den)
            line += f" real={e[0]:.3f} [{e[1]:.3f},{e[2]:.3f}]"
            out[f"{p}_{m}_real"] = fmt(*e[:3])
        for v in variants:
            cf = c[f"c_{p}_{v}"]
            if mv is None:
                e = bt.ratio((cf != real) & den, den)
                line += f" {v}={e[0]:.3f} [{e[1]:.3f},{e[2]:.3f}]"
                out[f"{p}_{m}_{v}"] = fmt(*e[:3])
            else:
                e = bt.paired(pick(mv, cf) & den, pick(mv, real) & den, den)
                line += f" {v}:{e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]"
                out[f"{p}_{m}_{v}-real"] = fmt(*e)
        print(line)
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
