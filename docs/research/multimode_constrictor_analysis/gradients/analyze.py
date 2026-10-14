"""Cosine matrices, noise ceilings, normalised alignment and norms from grads.py outputs, with a
cluster bootstrap over env blocks (each block = 4 envs x 128 turns of one mode in one rollout,
resampled with replacement within its (rollout, mode); 16 blocks per cell).

Usage: python analyze.py out.json label=file.npz [label=file.npz ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

import os
B_REPS = int(os.environ.get("B_REPS", 2000))
PARTS = ("pg", "v", "total", "ent")
LAYERS = ("torso", "all", "conv", "hidden", "feat_shared", "feat_onehot", "policy", "value")
FOOD = ("standard", "duel", "royale")


def load(path):
    z = np.load(path, allow_pickle=False)
    grams = {k[5:]: z[k] for k in z.files if k.startswith("gram_")}
    grams["torso"] = grams["conv"] + grams["hidden"]
    grams["all"] = sum(grams[k] for k in ("conv", "hidden", "feat_onehot", "feat_shared", "policy", "value"))
    return dict(
        grams=grams,
        counts=z["counts"],
        vw=z["value_weights"],
        vf=float(z["vf_coef"]),
        ef=float(z["ent_coef"]),
        modes=[str(m) for m in z["modes"]],
        return_vars=z["return_vars"],
        explained_var=z["explained_var"],
        diag={k[5:]: z[k] for k in z.files if k.startswith("diag_")},
    )


def weight_matrix(d, block_w):
    """W [R*G*nb*3, R*G*len(PARTS)]: columns are the (rollout, mode, part) gradients as they
    enter the loss (pg term, vf*vw*v term, total, -ef*entropy term)."""
    R, G, nb = d["counts"].shape
    W = np.zeros((R * G * nb * 3, R * G * len(PARTS)))
    for r in range(R):
        for g in range(G):
            for b in range(nb):
                w = block_w[r, g, b]
                base = ((r * G + g) * nb + b) * 3
                c = (r * G + g) * len(PARTS)
                coef = {"pg": (1.0, 0, 0), "v": (0, d["vf"] * d["vw"][r, g], 0),
                        "ent": (0, 0, -d["ef"])}
                coef["total"] = tuple(sum(x) for x in zip(coef["pg"], coef["v"], coef["ent"]))
                for pi, p in enumerate(PARTS):
                    for k in range(3):
                        W[base + k, c + pi] += w * coef[p][k]
    return W


def stats_from_D(D, R, G, modes):
    """D: [R*G*P, R*G*P] inner products. Returns per part dict of metrics."""
    nP = len(PARTS)
    out = {}
    ci = modes.index("constrictor")
    ri = modes.index("royale")
    food = [modes.index(m) for m in FOOD]

    def idx(r, g, p):
        return (r * G + g) * nP + p

    for pi, p in enumerate(PARTS):
        dot = lambda r1, a, r2, b: D[idx(r1, a, pi), idx(r2, b, pi)]  # noqa: E731
        norm = np.array([[np.sqrt(max(dot(r, g, r, g), 0.0)) for g in range(G)] for r in range(R)])
        if not np.all(norm > 0):
            continue

        def cos(r1, a, r2, b):
            return dot(r1, a, r2, b) / (norm[r1, a] * norm[r2, b])

        ceil = np.array([cos(0, g, 1, g) for g in range(G)])
        cross = np.full((G, G), np.nan)
        within = np.full((G, G), np.nan)
        for a in range(G):
            for b in range(G):
                if a != b:
                    cross[a, b] = 0.5 * (cos(0, a, 1, b) + cos(1, a, 0, b))
                    within[a, b] = 0.5 * (cos(0, a, 0, b) + cos(1, a, 1, b))
        with np.errstate(invalid="ignore", divide="ignore"):
            normed = cross / np.sqrt(np.outer(ceil, ceil))
            normed[np.outer(ceil, ceil) <= 0] = np.nan
        # Each mode vs the sum of the other three (equal loss weights): what the shared update
        # does for it, from the other modes' side.
        vs_others, vs_others_norm, others_ceil = np.zeros(G), np.zeros(G), np.zeros(G)
        for a in range(G):
            oth = [b for b in range(G) if b != a]
            def odot(r1, r2):  # others(r1) . others(r2)
                return sum(dot(r1, b, r2, c) for b in oth for c in oth)
            def adot(r1, r2):  # g_a(r1) . others(r2)
                return sum(dot(r1, a, r2, b) for b in oth)
            on = [np.sqrt(odot(r, r)) for r in range(R)]
            c12 = adot(0, 1) / (norm[0, a] * on[1])
            c21 = adot(1, 0) / (norm[1, a] * on[0])
            vs_others[a] = 0.5 * (c12 + c21)
            others_ceil[a] = odot(0, 1) / (on[0] * on[1])
            vs_others_norm[a] = (vs_others[a] / np.sqrt(ceil[a] * others_ceil[a])
                                 if ceil[a] > 0 and others_ceil[a] > 0 else np.nan)
        def pairs_mean(M, a, bs):
            return float(np.mean([M[a, b] for b in bs]))
        key = {
            "C_vs_food_raw": pairs_mean(cross, ci, food),
            "R_vs_SD_raw": pairs_mean(cross, ri, [modes.index("standard"), modes.index("duel")]),
            "C_vs_food_norm": pairs_mean(normed, ci, food),
            "R_vs_SD_norm": pairs_mean(normed, ri, [modes.index("standard"), modes.index("duel")]),
            "food_pairs_raw": float(np.mean([cross[a, b] for i, a in enumerate(food) for b in food[i + 1:]])),
            "food_pairs_norm": float(np.mean([normed[a, b] for i, a in enumerate(food) for b in food[i + 1:]])),
        }
        key["diff_raw"] = key["C_vs_food_raw"] - key["R_vs_SD_raw"]
        key["diff_norm"] = key["C_vs_food_norm"] - key["R_vs_SD_norm"]
        key["diff_foodpairs_norm"] = key["C_vs_food_norm"] - key["food_pairs_norm"]
        out[p] = dict(
            norm=norm.mean(0), norm_r=norm, ceil=ceil, cross=cross, within=within, normed=normed,
            vs_others=vs_others, vs_others_norm=vs_others_norm, others_ceil=others_ceil, key=key,
        )
    return out


def flatten_stats(s):
    """Scalar dict of everything in a stats_from_D output, for bootstrapping."""
    flat = {}
    for p, m in s.items():
        for name in ("norm", "ceil", "vs_others", "vs_others_norm", "others_ceil"):
            for i, v in enumerate(m[name]):
                flat[(p, name, i)] = v
        for name in ("cross", "within", "normed"):
            G = m[name].shape[0]
            for a in range(G):
                for b in range(G):
                    flat[(p, name, a, b)] = m[name][a, b]
        for k, v in m["key"].items():
            flat[(p, "key", k)] = v
    return flat


def analyse(path, rng):
    d = load(path)
    R, G, nb = d["counts"].shape
    assert R == 2
    w0 = d["counts"] / d["counts"].sum(2, keepdims=True)
    W = weight_matrix(d, w0)
    res = {}
    boots = {}
    # Bootstrap weights: multinomial resample of blocks within each (r, g).
    reps = []
    for _ in range(B_REPS):
        mult = np.stack([[rng.multinomial(nb, np.full(nb, 1 / nb)) for g in range(G)] for r in range(R)])
        wc = mult * d["counts"]
        reps.append(wc / wc.sum(2, keepdims=True))
    for L in LAYERS:
        Gm = d["grams"][L]
        D = W.T @ Gm @ W
        s = stats_from_D(D, R, G, d["modes"])
        flat0 = flatten_stats(s)
        samples = {k: [] for k in flat0}
        for wb in reps:
            Wb = weight_matrix_fast(d, wb, W)
            sb = flatten_stats(stats_from_D(Wb.T @ Gm @ Wb, R, G, d["modes"]))
            for k in samples:
                samples[k].append(sb.get(k, np.nan))
        ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
              for k, v in samples.items()}
        res[L] = s
        boots[L] = ci
    return d, res, boots


_PATTERN_CACHE = {}


def weight_matrix_fast(d, block_w, W0):
    """weight_matrix with block weights block_w, reusing W0's sparsity: W0 was built with w0, so
    each row's entries scale by block_w / w0 of its block."""
    R, G, nb = d["counts"].shape
    w0 = d["counts"] / d["counts"].sum(2, keepdims=True)
    scale = (block_w / w0).reshape(-1)  # per block, (r, g, b) order
    rows = np.repeat(scale, 3)
    return W0 * rows[:, None]


def fmt_matrix(M, modes, lo=None, hi=None):
    abbrev = [m[:4] for m in modes]
    lines = ["        " + " ".join(f"{a:>17}" for a in abbrev)]
    for a in range(len(modes)):
        cells = []
        for b in range(len(modes)):
            v = M[a, b]
            if np.isnan(v):
                cells.append(f"{'-':>17}")
            elif lo is not None:
                cells.append(f"{v:6.3f} [{lo[a, b]:5.2f},{hi[a, b]:5.2f}]")
            else:
                cells.append(f"{v:17.3f}")
        lines.append(f"{abbrev[a]:>8}" + " ".join(cells))
    return "\n".join(lines)


def main():
    out_path = sys.argv[1]
    rng = np.random.default_rng(0)
    report = {}
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        d, res, boots = analyse(path, rng)
        modes = d["modes"]
        G = len(modes)
        rep = {"modes": modes, "value_weights": d["vw"].tolist(), "return_vars": d["return_vars"].tolist(),
               "explained_var": d["explained_var"].tolist(),
               "approx_kl_max": float(np.max(np.abs(d["diag"]["approx_kl"]))),
               "clip_frac_max": float(np.max(d["diag"]["clip_frac"])), "layers": {}}
        for L, s in res.items():
            ci = boots[L]
            rep["layers"][L] = {}
            for p, m in s.items():
                entry = {}
                for name in ("norm", "ceil", "vs_others", "vs_others_norm", "others_ceil"):
                    entry[name] = {modes[i]: [float(m[name][i]), *ci[(p, name, i)]] for i in range(G)}
                for name in ("cross", "within", "normed"):
                    entry[name] = {
                        f"{modes[a]}|{modes[b]}": [float(m[name][a, b]), *ci[(p, name, a, b)]]
                        for a in range(G) for b in range(G) if a < b
                    }
                entry["key"] = {k: [float(v), *ci[(p, "key", k)]] for k, v in m["key"].items()}
                rep["layers"][L][p] = entry
        report[label] = rep
        # Print a summary.
        print(f"\n=== {label} ({path}) ===")
        print(f"value_weights r0 {np.round(d['vw'][0], 3)} r1 {np.round(d['vw'][1], 3)}; "
              f"return_var r0 {np.round(d['return_vars'][0], 3)}; max|approx_kl| {rep['approx_kl_max']:.2e}")
        for L in ("torso", "all", "conv", "hidden", "policy", "value", "feat_shared"):
            for p in ("pg", "v", "total"):
                if p not in res[L]:
                    continue
                m = res[L][p]
                ci = boots[L]
                lo = np.array([[ci.get((p, "cross", a, b), (np.nan, np.nan))[0] for b in range(G)] for a in range(G)])
                hi = np.array([[ci.get((p, "cross", a, b), (np.nan, np.nan))[1] for b in range(G)] for a in range(G)])
                print(f"\n[{L} / {p}] norms {np.round(m['norm'], 4)}  ceil {np.round(m['ceil'], 3)}")
                print("cross cos (rollout1 x rollout2, symmetrised) [95% boot CI]:")
                print(fmt_matrix(m["cross"], modes, lo, hi))
                lo = np.array([[ci.get((p, "normed", a, b), (np.nan, np.nan))[0] for b in range(G)] for a in range(G)])
                hi = np.array([[ci.get((p, "normed", a, b), (np.nan, np.nan))[1] for b in range(G)] for a in range(G)])
                print("normalised alignment cross/sqrt(ceil_a ceil_b):")
                print(fmt_matrix(m["normed"], modes, lo, hi))
                print("vs sum of others (raw, norm):",
                      {modes[i]: (round(m['vs_others'][i], 3), round(m['vs_others_norm'][i], 3)) for i in range(G)})
                print("key:", {k: f"{v:.3f} [{ci[(p, 'key', k)][0]:.3f},{ci[(p, 'key', k)][1]:.3f}]"
                               for k, v in m["key"].items()})
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
