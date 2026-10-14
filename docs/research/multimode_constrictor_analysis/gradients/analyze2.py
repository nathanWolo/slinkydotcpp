"""Gradient alignment between modes from grads.py outputs, pooled over files of the same params.

Units: blocks = 4 envs x 128 turns of one mode in one rollout (16 per rollout and mode); blocks are
independent games. Per layer group L and loss part p (pg term, value term vf*vw*v_loss, entropy term
-ef*H, total = their sum, all as they enter group_loss):

* split-half (the requested protocol): g_a^k = mode a's full gradient in rollout k.
  ceil_a = mean_{k != k'} cos(g_a^k, g_a^k');  cross_ab = mean_{k != k'} cos(g_a^k, g_b^k');
  normed_ab = cross_ab / sqrt(ceil_a ceil_b).
* pooled U-statistic (disattenuated cosine between the expected gradients mu_a, mu_b):
  m_a = count-weighted mean of all of mode a's blocks over all rollouts,
  U_aa = (|m_a|^2 - sum_i w_i^2 |x_i|^2) / (1 - sum_i w_i^2)   (unbiased for |mu_a|^2),
  U_ab = m_a . m_b (a != b; different games, unbiased for mu_a . mu_b),
  rho_ab = U_ab / sqrt(U_aa U_bb).   rho_aO: a against the sum of the other three modes.
* 95% intervals: percentile bootstrap, blocks resampled with replacement within each
  (rollout, mode).

Usage: python analyze2.py out.json label=f1.npz[+f2.npz ...] ...
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

B_REPS = int(os.environ.get("B_REPS", 1000))
PARTS = ("pg", "v", "ent", "total")
LAYERS = ("all", "torso", "conv", "hidden", "feat_shared", "feat_onehot", "policy", "value")
BASE = ("conv", "hidden", "feat_onehot", "feat_shared", "policy", "value")


def adam_metric(runner_path, P):
    """sqrt(P) [P]: Adam's preconditioner 1/(sqrt(nu)+eps) from a runner.npz (nu = leaves 27-39)."""
    z = np.load(runner_path)
    nu = np.concatenate([z[f"leaf{i}"].ravel() for i in range(27, 40)]).astype(np.float64)
    count = int(z["leaf13"])
    nu_hat = nu / (1 - 0.999**count)
    assert nu.size == P, (nu.size, P)
    return np.sqrt(1.0 / (np.sqrt(nu_hat) + 1e-5))


def load_dataset(files, metric=None):
    zs = [np.load(f) for f in files]
    modes = [str(m) for m in zs[0]["modes"]]
    vf, ef = float(zs[0]["vf_coef"]), float(zs[0]["ent_coef"])
    counts = np.concatenate([z["counts"] for z in zs])  # [K, G, nb]
    vw = np.concatenate([z["value_weights"] for z in zs])  # [K, G]
    if len(files) == 1 and metric is None:
        grams = {k: zs[0]["gram_" + k] for k in BASE}
    else:
        X = np.concatenate([np.load(f.removesuffix(".npz") + "-blocks.npy", mmap_mode="r") for f in files])
        # slices: rebuild from a file's Gram sizes is impossible, so recompute layer index ranges
        sl = layer_index(files[0])
        flat = X.reshape(-1, X.shape[-1])
        sP = adam_metric(metric, flat.shape[1]) if metric else None
        grams = {}
        for k in BASE:
            idx = sl[k]
            acc = np.zeros((flat.shape[0], flat.shape[0]))
            for c in range(0, len(idx), 16384):
                sub = np.asarray(flat[:, idx[c : c + 16384]], np.float64)
                if sP is not None:
                    sub = sub * sP[idx[c : c + 16384]][None]
                acc += sub @ sub.T
            grams[k] = acc
        del X
    grams["torso"] = grams["conv"] + grams["hidden"]
    grams["all"] = sum(grams[k] for k in BASE)
    return dict(modes=modes, vf=vf, ef=ef, counts=counts, vw=vw, grams=grams, files=files,
                return_vars=np.concatenate([z["return_vars"] for z in zs]),
                explained_var=np.concatenate([z["explained_var"] for z in zs]),
                approx_kl=max(float(np.abs(z["diag_approx_kl"]).max()) for z in zs))


def layer_index(npz_path):
    """Layer-group column indices, from the trainer's network template (same as grads.py)."""
    sys.path.insert(0, "/home/user/slinkydotcpp/baselines")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jax

    import ppo_modes as pm
    from grads import layer_slices

    cfg = pm.load_config("/home/user/slinkydotcpp/runs/ppo-modes-seed0")
    params = pm.init_network(jax.random.key(0), cfg)
    return layer_slices(params)[0]


def coef_matrix(d):
    """C [K, G, 3, 4]: raw part (pg, v_loss, entropy) -> loss part (pg, v term, ent term, total)."""
    K, G, _ = d["counts"].shape
    C = np.zeros((K, G, 3, 4))
    C[:, :, 0, 0] = 1.0
    C[:, :, 1, 1] = d["vf"] * d["vw"]
    C[:, :, 2, 2] = -d["ef"]
    C[:, :, :, 3] = C[:, :, :, 0] + C[:, :, :, 1] + C[:, :, :, 2]
    return C


def block_sqnorms(gram, d, C):
    """q [K, G, nb, 4]: |x_i^p|^2 of every block for every loss part."""
    K, G, nb = d["counts"].shape
    Gm = gram.reshape(K, G, nb, 3, K, G, nb, 3)
    q = np.zeros((K, G, nb, 4))
    for k in range(K):
        for g in range(G):
            for b in range(nb):
                M = Gm[k, g, b, :, k, g, b, :]
                q[k, g, b] = np.einsum("qp,qr,rp->p", C[k, g], M, C[k, g])
    return q


def stats(gram, q, w, tot_kg, d, C):
    """All statistics for within-(k,g) block weights w [K, G, nb] (summing to 1 over nb)."""
    K, G, nb = w.shape
    nP = 4
    # A: [K*G*nb*3, K*G*4]; column (k, g, p) = rollout-mean of loss part p.
    A = np.zeros((K, G, nb, 3, K, G, nP))
    for k in range(K):
        for g in range(G):
            A[k, g, :, :, k, g, :] = w[k, g][:, None, None] * C[k, g][None]
    A = A.reshape(K * G * nb * 3, K * G * nP)
    D = (A.T @ gram @ A).reshape(K, G, nP, K, G, nP)
    out = {}
    for p in range(nP):
        Dp = D[:, :, p, :, :, p]  # [K, G, K, G]
        nrm = np.sqrt(np.maximum(np.einsum("kgkg->kg", Dp), 0))  # [K, G]
        with np.errstate(invalid="ignore", divide="ignore"):
            cosm = Dp / (nrm[:, :, None, None] * nrm[None, None, :, :])
        off = ~np.eye(K, dtype=bool)
        ceil = np.array([cosm[:, a, :, a][off].mean() for a in range(G)])
        cross = np.array([[cosm[:, a, :, b][off].mean() for b in range(G)] for a in range(G)])
        # Pooled U-statistics.
        pi = tot_kg / tot_kg.sum(0, keepdims=True)  # [K, G] rollout share of mode a's samples
        Mdot = np.einsum("ka,kalb,lb->ab", pi, Dp, pi)  # m_a . m_b
        sw2 = np.einsum("ka,kab->a", pi**2, w**2)  # sum_i (pooled w_i)^2
        sq = np.einsum("ka,kab,kab->a", pi**2, w**2, q[..., p])  # sum_i w_i^2 |x_i|^2
        U = Mdot.copy()
        U[np.diag_indices(G)] = (np.diag(Mdot) - sq) / (1 - sw2)
        with np.errstate(invalid="ignore", divide="ignore"):
            dU = np.sqrt(np.where(np.diag(U) > 0, np.diag(U), np.nan))
            rho = U / np.outer(dU, dU)
            normed = cross / np.sqrt(np.where(np.outer(ceil, ceil) > 0, np.outer(ceil, ceil), np.nan))
            rho_oth = np.zeros(G)
            for a in range(G):
                o = [b for b in range(G) if b != a]
                uao = U[a, o].sum()
                uoo = U[np.ix_(o, o)].sum()
                rho_oth[a] = uao / np.sqrt(U[a, a] * uoo) if U[a, a] > 0 and uoo > 0 else np.nan
        out[PARTS[p]] = dict(
            norm=nrm.mean(0), signal_norm=dU, reliability=np.diag(U) / np.diag(Mdot), ceil=ceil,
            cross=cross, normed=normed, rho=rho, rho_others=rho_oth, U=U,
        )
    return out


def key_stats(s, modes):
    i = {m: modes.index(m) for m in modes}
    S, Du, R, Cn = i["standard"], i["duel"], i["royale"], i["constrictor"]
    out = {}
    for name in ("cross", "normed", "rho"):
        M = s[name]
        cf = np.mean([M[Cn, S], M[Cn, Du], M[Cn, R]])
        rsd = np.mean([M[R, S], M[R, Du]])
        fp = np.mean([M[S, Du], M[S, R], M[Du, R]])
        out[f"{name}:C_vs_food"] = cf
        out[f"{name}:R_vs_SD"] = rsd
        out[f"{name}:food_pairs"] = fp
        out[f"{name}:diff_C-R"] = cf - rsd
        out[f"{name}:diff_C-foodpairs"] = cf - fp
    out["rho_others:C"] = s["rho_others"][Cn]
    out["rho_others:R"] = s["rho_others"][R]
    out["rho_others:diff_C-R"] = s["rho_others"][Cn] - s["rho_others"][R]
    out["signal_norm_ratio:C/mean_food"] = s["signal_norm"][Cn] / np.mean(s["signal_norm"][[S, Du, R]])
    out["norm_ratio:C/mean_food"] = s["norm"][Cn] / np.mean(s["norm"][[S, Du, R]])
    return out


def flatten(s, modes):
    flat = {}
    G = len(modes)
    for name in ("norm", "signal_norm", "reliability", "ceil", "rho_others"):
        for a in range(G):
            flat[(name, modes[a])] = s[name][a]
    for name in ("cross", "normed", "rho"):
        for a in range(G):
            for b in range(a + 1, G):
                flat[(name, f"{modes[a]}|{modes[b]}")] = s[name][a, b]
    for k, v in key_stats(s, modes).items():
        flat[("key", k)] = v
    return flat


def analyse(d, rng, layers=LAYERS):
    K, G, nb = d["counts"].shape
    C = coef_matrix(d)
    w0 = d["counts"] / d["counts"].sum(2, keepdims=True)
    reps = []
    for _ in range(B_REPS):
        n = rng.multinomial(nb, np.full(nb, 1 / nb), size=(K, G)).astype(float)
        wc = n * d["counts"]
        reps.append((wc / wc.sum(2, keepdims=True), wc.sum(2)))
    res = {}
    pr = {}
    for L in ("all", "torso"):
        gram = d["grams"][L]
        Gm = gram.reshape(K, G, nb, 3, K, G, nb, 3)
        pr[L] = {}
        for pi_, pname in enumerate(("pg", "v")):
            vals = np.zeros((K, G))
            for k in range(K):
                for g in range(G):
                    M = Gm[k, g, :, pi_, k, g, :, pi_]  # [nb, nb]
                    w = w0[k, g]
                    J = np.eye(nb) - np.outer(np.ones(nb), w)  # x_i - weighted mean
                    Mc = J @ M @ J.T
                    lam = np.clip(np.linalg.eigvalsh(Mc * np.sqrt(np.outer(w, w)) * nb), 0, None)
                    vals[k, g] = lam.sum() ** 2 / (lam**2).sum()
            pr[L][pname] = {m: float(vals[:, g].mean()) for g, m in enumerate(d["modes"])}
    res["_participation_ratio"] = pr
    for L in layers:
        gram = d["grams"][L]
        q = block_sqnorms(gram, d, C)
        s0 = stats(gram, q, w0, d["counts"].sum(2), d, C)
        res[L] = {}
        boot = {p: [] for p in PARTS}
        for wb, tb in reps:
            sb = stats(gram, q, wb, tb, d, C)
            for p in PARTS:
                boot[p].append(flatten(sb[p], d["modes"]))
        for p in PARTS:
            f0 = flatten(s0[p], d["modes"])
            if max(f0[("norm", m)] for m in d["modes"]) == 0:
                continue  # e.g. the pg term in the value head
            entry = {}
            for key, v in f0.items():
                arr = np.array([b[key] for b in boot[p]], float)
                ok = np.isfinite(arr)
                lo, hi = (np.percentile(arr[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan))
                se = float(np.std(arr[ok])) if ok.sum() > 10 else np.nan
                entry["/".join(key)] = [float(v), float(lo), float(hi), float(1 - ok.mean()), se]
            res[L][p] = entry
    return res


def main():
    out_path = sys.argv[1]
    rng = np.random.default_rng(12345)
    report = {}
    for arg in sys.argv[2:]:
        label, files = arg.split("=", 1)
        files, _, metric = files.partition("@adam:")
        files = files.split("+")
        d = load_dataset(files, metric or None)
        res = analyse(d, rng)
        report[label] = dict(
            files=files, modes=d["modes"], rollouts=int(d["counts"].shape[0]),
            value_weights=d["vw"].tolist(), return_vars=d["return_vars"].tolist(),
            explained_var=d["explained_var"].tolist(), max_abs_approx_kl=d["approx_kl"],
            valid_samples=d["counts"].sum(2).tolist(), layers=res,
        )
        print(f"done {label}", flush=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
