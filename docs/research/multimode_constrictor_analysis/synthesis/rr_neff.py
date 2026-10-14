"""Re-run the snapshots investigation's transitivity Monte Carlo with per-pair noise for 12 distinct games
(greedy-vs-greedy constrictor has only 12 distinct games per pair; see behaviour/orbits.py) instead of 512."""
import json, math, numpy as np
D = "docs/research/multimode_constrictor_analysis/snapshots"
SNAPS = [1048576, 2097152, 3145728, 4194304, 5242880]
G, S = "runs/ppo-modes-seed0", "runs/ppo-modes-constrictor-seed0"
NG = 512
def unbase36(t, w): return np.array([int(t[i:i+w], 36) for i in range(0, len(t), w)], np.int64)
def norm(name):
    if name == f"ppo_modes:{S}": return "S5"
    for i, k in enumerate(SNAPS, 1):
        if name == f"ppo_modes:{G}:snapshot={k}": return f"G{i}"
        if name == f"ppo_modes:{S}:snapshot={k}": return f"S{i}"
recs = {}
for line in open(f"{D}/cells.jsonl"):
    r = json.loads(line)
    if r["game_mode"] != "constrictor" or r["lineup"] != "A,B,A,B": continue
    a, b = norm(r["seat_names"][0]), norm(r["seat_names"][1])
    if a is None or b is None or a == b: continue
    if (a, b) not in recs or r["games"] > recs[(a, b)]["games"]: recs[(a, b)] = r
labels = [f"G{i}" for i in range(1, 6)] + [f"S{i}" for i in range(1, 6)]
idx = {l: i for i, l in enumerate(labels)}; n = 10
pg = {}
for (a, b), r in recs.items():
    p = r["per_game"]; sh = unbase36(p["share"]["A"], p["width"]) / p["share_scale"]
    pg[(a, b)] = sh[:NG]; pg[(b, a)] = 1 - sh[:NG]
M = np.full((n, n), np.nan)
for (a, b), v in pg.items(): M[idx[a], idx[b]] = v.mean()
D_ = M - 0.5; np.fill_diagonal(D_, 0)
r_ = D_.mean(axis=1); fit = r_[:, None] - r_[None, :]
off = ~np.eye(n, dtype=bool)
obs = np.sum((D_ - fit)[off] ** 2)
# distinct per-game share values per cell (a crude check of the 12-game structure: share takes values in {0,1/2,1})
vals = {k: sorted(set(np.round(v, 3))) for k, v in pg.items() if k[0] < k[1]}
print("observed residual SS", round(obs, 4))
rng = np.random.default_rng(1)
iu = np.triu_indices(n, 1)
for neff in [512, 96, 24, 12]:
    noise = np.zeros((n, n))
    for (a, b), v in pg.items(): noise[idx[a], idx[b]] = v.var(ddof=1) / neff
    sims = []
    for _ in range(4000):
        e = np.zeros((n, n)); e[iu] = rng.normal(size=len(iu[0])) * np.sqrt(noise[iu]); e = e - e.T
        Ds = fit + e; rs = Ds.mean(axis=1); f2 = rs[:, None] - rs[None, :]
        sims.append(np.sum((Ds - f2)[off] ** 2))
    sims = np.array(sims)
    print(f"n_eff={neff:4d}: transitive-null residual mean {sims.mean():.3f}, 95% range [{np.percentile(sims,2.5):.3f}, {np.percentile(sims,97.5):.3f}], P(null >= observed) = {(sims >= obs).mean():.3f}")
# Exact 12-game resampling null: draw 12 per-game outcomes per pair from {0,.5,1} with the pair's fitted mean and the observed draw rate
