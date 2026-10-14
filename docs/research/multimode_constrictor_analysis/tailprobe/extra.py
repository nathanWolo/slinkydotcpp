"""Pooled divergent-decision rates (3 sampled constrictor datasets) and the masked-vs-plain interaction."""
import numpy as np
from analyze import Boot, load, pick
d = load(["con_gsgs_s_dec.npz", "con_ssss_s_dec.npz", "con_gggg_s_dec.npz"])
bt = Boot(d["game"])
L, ac, ast = d["legal"], d["a_con"], d["a_std"]
amax = np.where(L, ac, -1).max(1)
smax = np.where(L, ast, -1).max(1)
sbest = L & (ast == smax[:, None])
print("pooled sampled constrictor:", len(L), "decisions", bt.ng, "games")
for k in (1, 3, 5, 10):
    div = (np.where(sbest, ac, -1).max(1) <= amax - k) & (smax > 0)
    g = pick(sbest, d["c_lg"]) & div; s = pick(sbest, d["c_ls"]) & div
    eg, eg_lo, eg_hi, _ = bt.ratio(g, div); es, es_lo, es_hi, _ = bt.ratio(s, div)
    df = bt.paired(g, s, div)
    print(f" k={k:2d} n={div.sum():4d} G picks std-best {eg:.3f} [{eg_lo:.3f},{eg_hi:.3f}]  S {es:.3f} [{es_lo:.3f},{es_hi:.3f}]  G-S {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}]")
for name, files in (("gsgs", ["con_gsgs_s_dec.npz"]), ("pooled", ["con_gsgs_s_dec.npz", "con_ssss_s_dec.npz", "con_gggg_s_dec.npz"])):
    d = load(files); bt = Boot(d["game"])
    L, ac, ast = d["legal"], d["a_con"], d["a_std"]
    T = 8
    trapmv = L & (ac < T); safe = (L & (ac >= 2 * T)).any(1)
    mm = trapmv & (ast >= 2 * T); em = mm.any(1) & safe
    eu = (trapmv & (ast < 2 * T)).any(1) & safe & ~mm.any(1)
    res = {}
    for p, key in (("G", "c_lg"), ("S", "c_ls")):
        _, _, _, bm = bt.ratio(pick(mm, d[key]) & em, em)
        _, _, _, bu = bt.ratio(pick(trapmv, d[key]) & eu, eu)
        res[p] = (bm, bu)
    est_m = (pick(mm, d["c_lg"]) & em).sum() / em.sum() - (pick(mm, d["c_ls"]) & em).sum() / em.sum()
    est_u = (pick(trapmv, d["c_lg"]) & eu).sum() / eu.sum() - (pick(trapmv, d["c_ls"]) & eu).sum() / eu.sum()
    inter = (res["G"][0] - res["S"][0]) - (res["G"][1] - res["S"][1])
    lo, hi = np.percentile(inter, [2.5, 97.5])
    # ratio form: G/S odds of trap in masked vs plain
    print(f"{name}: T8 masked G-S {est_m:+.3f}, plain G-S {est_u:+.3f}, interaction {est_m-est_u:+.3f} [{lo:+.3f},{hi:+.3f}]; "
          f"masked decisions/game {em.sum()/bt.ng:.2f}, plain {eu.sum()/bt.ng:.2f}")
