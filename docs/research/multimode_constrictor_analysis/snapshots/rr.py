"""Constrictor round robin over {G@k, S@k}: matrix, field score with cluster bootstrap, intransitivity."""
import json, math
import numpy as np

D = "docs/research/multimode_constrictor_analysis/snapshots"
SNAPS = [1048576, 2097152, 3145728, 4194304, 5242880]
G, S = "runs/ppo-modes-seed0", "runs/ppo-modes-constrictor-seed0"
NG = 512  # games used per pair (first 512 game ids; common random numbers across cells)

def unbase36(text, width):
    return np.array([int(text[i:i+width], 36) for i in range(0, len(text), width)], np.int64)

def norm(name):
    # canonical policy label from a seat name; S at 5242880 may be written with or without snapshot
    if name == f"ppo_modes:{S}": return "S5"
    for i, k in enumerate(SNAPS, 1):
        if name == f"ppo_modes:{G}:snapshot={k}": return f"G{i}"
        if name == f"ppo_modes:{S}:snapshot={k}": return f"S{i}"
    return None

recs = {}
for line in open(f"{D}/cells.jsonl"):
    r = json.loads(line)
    if r["game_mode"] != "constrictor" or r["lineup"] != "A,B,A,B": continue
    a, b = norm(r["seat_names"][0]), norm(r["seat_names"][1])
    if a is None or b is None or a == b: continue
    key = (a, b)
    if key not in recs or r["games"] > recs[key]["games"]:
        recs[key] = r
labels = [f"G{i}" for i in range(1, 6)] + [f"S{i}" for i in range(1, 6)]
idx = {l: i for i, l in enumerate(labels)}
n = len(labels)
# per-game team share of row policy vs column policy (first NG games); both orientations
pg = {}
for (a, b), r in recs.items():
    p = r["per_game"]
    sh = unbase36(p["share"]["A"], p["width"]) / p["share_scale"]
    assert len(sh) >= NG, (a, b, len(sh))
    pg[(a, b)] = sh[:NG]
    pg[(b, a)] = 1.0 - sh[:NG]   # two-symbol table: shares of a game sum to 1
missing = [(a, b) for a in labels for b in labels if a != b and (a, b) not in pg]
print("missing pairs:", missing)

M = np.full((n, n), np.nan)
for (a, b), v in pg.items():
    M[idx[a], idx[b]] = v.mean()
def pm(x): return f"{x:.3f}"
print("\nRound-robin matrix: row policy's team share vs column policy (parity 0.5), first %d games per pair" % NG)
print("| row \\ col | " + " | ".join(labels) + " | field mean |")
print("|---|" + "---|" * (n + 1))
for a in labels:
    row = [("-" if a == b else pm(M[idx[a], idx[b]])) for b in labels]
    print(f"| {a} | " + " | ".join(row) + f" | {np.nanmean(M[idx[a]]):.3f} |")

# field score: mean over all 9 opponents; cluster bootstrap over game ids (same resample for every cell)
rng = np.random.default_rng(0)
B = 4000
ids = rng.integers(0, NG, size=(B, NG))
def field_scores(sample_idx, opp_filter=None):
    out = np.zeros(n)
    for a in labels:
        vals = [pg[(a, b)][sample_idx].mean() for b in labels if b != a and (opp_filter is None or opp_filter(a, b))]
        out[idx[a]] = np.mean(vals)
    return out
base = field_scores(slice(None))
boot = np.array([field_scores(ids[i]) for i in range(B)])
print(f"\nField score = mean over the 9 other policies of the team share (A,B,A,B tables); 95 pct cluster bootstrap over game ids ({B} resamples)")
print("| policy | field score | 95% CI |")
print("|---|---|---|")
for a in labels:
    lo, hi = np.percentile(boot[:, idx[a]], [2.5, 97.5])
    print(f"| {a} | {base[idx[a]]:.3f} | [{lo:.3f}, {hi:.3f}] |")
print("\nMatched G@k minus S@k field score (paired bootstrap)")
print("| k | G@k - S@k | 95% CI |")
print("|---|---|---|")
for i in range(1, 6):
    d = boot[:, idx[f"G{i}"]] - boot[:, idx[f"S{i}"]]
    print(f"| {SNAPS[i-1]/1e6:.2f}M | {base[idx[f'G{i}']]-base[idx[f'S{i}']]:+.3f} | [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] |")

# same, but field = the other generation's policies vs only within-run excluded (G vs S opponents only: cross-run field)
def cross_field(sample_idx):
    out = np.zeros(n)
    for a in labels:
        vals = [pg[(a, b)][sample_idx].mean() for b in labels if b[0] != a[0]]
        out[idx[a]] = np.mean(vals)
    return out
cbase = cross_field(slice(None)); cboot = np.array([cross_field(ids[i]) for i in range(B)])
print("\nCross-run field score (G@k vs the five S snapshots; S@k vs the five G snapshots)")
print("| policy | cross-run score | 95% CI |")
print("|---|---|---|")
for a in labels:
    print(f"| {a} | {cbase[idx[a]]:.3f} | [{np.percentile(cboot[:, idx[a]],2.5):.3f}, {np.percentile(cboot[:, idx[a]],97.5):.3f}] |")

# intransitivity: fit d_ij = M_ij - 0.5 ~ r_i - r_j (least squares; for a complete matrix r_i = row mean of d), residual vs noise
D_ = M - 0.5
np.fill_diagonal(D_, 0)
r = np.nanmean(D_, axis=1) * n / (n)  # LS solution up to scale: r_i = mean_j d_ij  (complete skew matrix)
fit = r[:, None] - r[None, :]
off = ~np.eye(n, dtype=bool)
num = np.sum((D_ - fit)[off] ** 2); den = np.sum(D_[off] ** 2)
# sampling noise of each pair's mean share: var of per-game team share / NG
noise = np.zeros((n, n))
for (a, b), v in pg.items():
    noise[idx[a], idx[b]] = v.var(ddof=1) / NG
# For a skew matrix fit, expected residual SS from pure sampling noise (transitive truth): each unordered pair counted twice
# Monte Carlo: simulate truth = fit, add Gaussian noise with the same variances, refit, compute residual SS
sim = []
for _ in range(2000):
    e = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    z = rng.normal(size=len(iu[0])) * np.sqrt(noise[iu])
    e[iu] = z; e = e - e.T
    Ds = fit + e
    rs = Ds.sum(axis=1) / (n - 0)  # row means (diag 0)
    rs = Ds.mean(axis=1)
    f2 = rs[:, None] - rs[None, :]
    sim.append(np.sum((Ds - f2)[off] ** 2))
print(f"\nTransitive (Borda/Elo-like) fit d_ij = r_i - r_j: residual SS fraction = {num/den:.3f} of total SS of (share-0.5).")
print(f"Residual SS = {num:.4f}; under a transitive truth with this sampling noise it would be {np.mean(sim):.4f} (95% range {np.percentile(sim,2.5):.4f}-{np.percentile(sim,97.5):.4f}) -> {'cycles beyond noise' if num > np.percentile(sim,97.5) else 'consistent with noise'}.")

# explicit significantly-cyclic triples: each edge's share CI (pair normal) excludes 0.5 in the cyclic direction
def beats(a, b):
    v = pg[(a, b)]; m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
    return m - 1.96 * se > 0.5
cyc = []
for i in range(n):
    for j in range(n):
        for k in range(n):
            if len({i, j, k}) == 3 and i < j and i < k:
                a, b, c = labels[i], labels[j], labels[k]
                if beats(a, b) and beats(b, c) and beats(c, a):
                    cyc.append((a, b, c))
print(f"\nStrict 3-cycles (a beats b, b beats c, c beats a, each with the pair's 95% interval excluding 0.5): {len(cyc)}")
for a, b, c in cyc:
    print(f"  {a} > {b} > {c} > {a}: " + ", ".join(f"{x}>{y} {pg[(x,y)].mean():.3f}" for x, y in [(a,b),(b,c),(c,a)]))
