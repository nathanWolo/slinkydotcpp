"""Metrics for the tail probe, with bootstrap CIs over games.

Usage: python analyze.py OUT.json NAME=file1.npz[,file2.npz] ...
(decision files from decisions.py; several files of one dataset are concatenated with
game ids offset).

Every metric is a ratio sum(num)/sum(den) over decisions; the CI resamples games with
replacement (B=2000 multinomial weights), percentiles 2.5/97.5. G and S are both scored
on the same decisions, so G-S is a paired difference (same resample).
"""

import json
import sys

import numpy as np

B = 2000
RNG = np.random.default_rng(12345)


def load(files, dedupe=False):
    parts, off = [], 0
    seen = set()
    for f in files:
        d = dict(np.load(f))
        if dedupe:
            keep_games = []
            for g in np.unique(d["game"]):
                s = int(d["game_sig"][d["game"] == g][0])
                if s not in seen:
                    seen.add(s)
                    keep_games.append(g)
            sel = np.isin(d["game"], keep_games)
            d = {k: v[sel] for k, v in d.items()}
        d["game"] = d["game"] + off
        off = d["game"].max() + 1
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class Boot:
    def __init__(self, game):
        _, self.gid = np.unique(game, return_inverse=True)
        self.ng = self.gid.max() + 1
        self.W = RNG.multinomial(self.ng, np.full(self.ng, 1.0 / self.ng), size=B).astype(np.float64)

    def agg(self, x):
        return np.bincount(self.gid, weights=x.astype(np.float64), minlength=self.ng)

    def ratio(self, num, den):
        n, d = self.agg(num), self.agg(den)
        est = n.sum() / d.sum() if d.sum() > 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (self.W @ n) / (self.W @ d)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if d.sum() > 0 else (np.nan, np.nan)
        return est, lo, hi, bs

    def paired(self, num_a, num_b, den):
        """(a - b) where a=sum(num_a)/sum(den), b=sum(num_b)/sum(den)."""
        ea, _, _, ba = self.ratio(num_a, den)
        eb, _, _, bb = self.ratio(num_b, den)
        diff = ba - bb
        lo, hi = np.nanpercentile(diff, [2.5, 97.5]) if den.sum() > 0 else (np.nan, np.nan)
        return ea - eb, lo, hi


def fmt(est, lo, hi):
    return {"est": round(float(est), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def pick(arr, choice):
    return np.take_along_axis(arr, choice[:, None], axis=1)[:, 0]


def analyze(d):
    bt = Boot(d["game"])
    L = d["legal"]
    ac, ast, asw = d["a_con"], d["a_std"], d["a_stdw"]
    K = len(L)
    out = {"decisions": int(K), "games": int(bt.ng)}
    amax = np.where(L, ac, -1).max(1)
    amin = np.where(L, ac, 10**6).min(1)
    choices = {"G": d["c_lg"], "S": d["c_ls"]}
    one = np.ones(K, bool)

    def both(num_fn, den, label):
        res = {"n": int(den.sum())}
        nums = {p: num_fn(c) & den for p, c in choices.items()}
        for p in nums:
            res[p] = fmt(*bt.ratio(nums[p], den)[:3])
        res["G-S"] = fmt(*bt.paired(nums["G"], nums["S"], den))
        out[label] = res

    # 1. divergent decisions
    for name, s in (("std", ast), ("stdw", asw)):
        smax = np.where(L, s, -1).max(1)
        sbest = L & (s == smax[:, None])
        cbest = L & (ac == amax[:, None])
        for k in (1, 3, 5, 10):
            div = (np.where(sbest, ac, -1).max(1) <= amax - k) & (smax > 0)
            both(lambda c: pick(sbest, c), div, f"div_{name}_k{k}_pick_stdbest")
            both(lambda c: pick(cbest, c), div, f"div_{name}_k{k}_pick_conbest")

    # 2. trap rates
    for T in (5, 8, 12):
        trapmv = L & (ac < T)
        safe = L & (ac >= 2 * T)
        elig = trapmv.any(1) & safe.any(1)
        both(lambda c: pick(trapmv, c), elig, f"trap_T{T}_all")
        for name, s in (("std", ast), ("stdw", asw)):
            masked_mv = trapmv & (s >= 2 * T)
            unmasked_mv = trapmv & (s < 2 * T)
            em = masked_mv.any(1) & safe.any(1)
            eu = unmasked_mv.any(1) & safe.any(1) & ~masked_mv.any(1)
            # masked traps: choosing a trap move that tail-moving semantics calls roomy
            both(lambda c: pick(masked_mv, c), em, f"trap_T{T}_{name}_masked")
            both(lambda c: pick(trapmv, c), eu, f"trap_T{T}_{name}_unmasked_only")

    # 3. con-optimality when areas differ
    for k in (1, 5, 20):
        inf_ = (amax - amin) >= k
        cbest = L & (ac == amax[:, None])
        both(lambda c: pick(cbest, c), inf_, f"conbest_when_spread_ge{k}")
    smax = np.where(L, ast, -1).max(1)
    smin = np.where(L, ast, 10**6).min(1)
    for k in (1, 5, 20):
        inf_ = (smax - smin) >= k
        sbest = L & (ast == smax[:, None])
        both(lambda c: pick(sbest, c), inf_, f"stdbest_when_std_spread_ge{k}")
        inf2 = inf_ & (amax == amin)
        both(lambda c: pick(sbest, c), inf2, f"stdbest_when_std_spread_ge{k}_con_flat")

    # 4. tail adjacency
    ta = d["tail_adj"]
    mixed = ta.any(1) & ~(ta | ~L).all(1)
    both(lambda c: pick(ta, c), mixed, "tail_adjacent_when_mixed")
    mixed_flat = mixed & (amax == amin)
    both(lambda c: pick(ta, c), mixed_flat, "tail_adjacent_when_mixed_con_flat")

    # 5. counterfactual inputs
    for cf in ("far", "near"):
        for sub_name, sub in (("all", one), ("con_spread", amax != amin)):
            res = {"n": int(sub.sum())}
            flips = {}
            for p, key in (("G", "lg"), ("S", "ls")):
                f = (d[f"c_{key}_{cf}"] != d[f"c_{key}"]) & sub
                flips[p] = f
                res[p + "_flip"] = fmt(*bt.ratio(f, sub)[:3])
                # among flips: did the counterfactual choice gain area_con?
                gain = pick(ac, d[f"c_{key}_{cf}"]) - pick(ac, d[f"c_{key}"])
                res[p + "_mean_dcon_per_decision"] = fmt(*bt.ratio(np.where(sub, gain, 0), sub)[:3])
                res[p + "_frac_flips_up"] = fmt(*bt.ratio((gain > 0) & f, f)[:3])
                res[p + "_frac_flips_down"] = fmt(*bt.ratio((gain < 0) & f, f)[:3])
            res["G-S_flip"] = fmt(*bt.paired(flips["G"], flips["S"], sub))
            out[f"cf_{cf}_{sub_name}"] = res
        # behaviour under the counterfactual input vs the real one (paired, same policy)
        T = 8
        trapmv = L & (ac < T)
        safe = L & (ac >= 2 * T)
        elig = trapmv.any(1) & safe.any(1)
        masked_mv = trapmv & (ast >= 2 * T)
        em = masked_mv.any(1) & safe.any(1)
        cbest = L & (ac == amax[:, None])
        spread = (amax - amin) >= 5
        res = {}
        for p, key in (("G", "lg"), ("S", "ls")):
            real, cfc = d[f"c_{key}"], d[f"c_{key}_{cf}"]
            for mname, mv, den in (("trapT8", trapmv, elig), ("maskedtrapT8", masked_mv, em),
                                   ("conbest_spread5", cbest, spread)):
                res[f"{p}_{mname}_real"] = fmt(*bt.ratio(pick(mv, real) & den, den)[:3])
                res[f"{p}_{mname}_cf"] = fmt(*bt.ratio(pick(mv, cfc) & den, den)[:3])
                res[f"{p}_{mname}_cf-real"] = fmt(*bt.paired(pick(mv, cfc) & den, pick(mv, real) & den, den))
                res[f"n_{mname}"] = int(den.sum())
        out[f"cf_{cf}_behaviour"] = res

    # 6. validation: actually played moves; died within a_con+1 moves?
    played = d["played"]
    a_pl = pick(ac, played)
    s_pl = pick(ast, played)
    died = ~d["final_alive"]
    moves_left = np.where(died, d["elim_turn"] - d["game_turn"], 10**6)
    val = {}
    for T in (5, 8, 12):
        trap = a_pl < T
        val[f"played_acon_lt{T}"] = {
            "n": int(trap.sum()),
            "died_within_acon+1": float(np.mean(moves_left[trap] <= a_pl[trap] + 1)) if trap.any() else None,
            "died_within_acon+1_when_astd_ge2T": float(np.mean((moves_left <= a_pl + 1)[trap & (s_pl >= 2 * T)])) if (trap & (s_pl >= 2 * T)).any() else None,
            "n_astd_ge2T": int((trap & (s_pl >= 2 * T)).sum()),
        }
    big = a_pl >= 30
    val["played_acon_ge30_died_within_10"] = float(np.mean(moves_left[big] <= 10))
    out["validation"] = val
    return out


def main():
    outpath = sys.argv[1]
    results = {}
    for spec in sys.argv[2:]:
        name, files = spec.split("=")
        dedupe = name.endswith("_dedup")
        d = load(files.split(","), dedupe=dedupe)
        results[name] = analyze(d)
        print(name, results[name]["decisions"], "decisions", results[name]["games"], "games", flush=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
