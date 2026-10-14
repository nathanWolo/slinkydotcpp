"""Print compact tables from analyze2.py JSON outputs. Usage: summarize.py a.json [b.json ...]
[--layers all,torso] [--parts pg,v,total]"""

import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
layers = opts.get("layers", "all,torso").split(",")
parts = opts.get("parts", "pg,v,total").split(",")
modes = ["standard", "duel", "royale", "constrictor"]
ab = {"standard": "S", "duel": "D", "royale": "R", "constrictor": "C"}


def f(e, k, nd=3):
    v = e.get(k)
    if v is None:
        return "-"
    x, lo, hi, nan, se = v
    # Split-half statistics (noisy norms, cosines) are biased by resampling duplicates, so they
    # get estimate +- 1.96 bootstrap SE; U-statistics get the percentile interval.
    if k.split("/")[0] in ("norm", "ceil", "cross", "normed") or k.startswith("key/cross") or \
            k.startswith("key/normed") or k.startswith("key/norm_ratio"):
        lo, hi = x - 1.96 * se, x + 1.96 * se
    s = f"{x:.{nd}f} [{lo:.{nd - 1}f},{hi:.{nd - 1}f}]"
    if nan > 0.05:
        s += f" (nan {nan:.0%})"
    return s


report = {}
for a in args:
    report.update(json.load(open(a)))
for label, r in report.items():
    print(f"\n##### {label}: {r['rollouts']} rollouts, valid samples/rollout "
          f"{[int(x) for x in r['valid_samples'][0]]}, max|approx_kl| {r['max_abs_approx_kl']:.1e}")
    print("value weights per rollout:", [[round(x, 3) for x in v] for v in r["value_weights"]])
    if "_participation_ratio" in r["layers"]:
        print("noise participation ratio (16 blocks, max 15):", json.dumps(r["layers"]["_participation_ratio"]))
    for L in layers:
        for p in parts:
            e = r["layers"][L].get(p)
            if e is None:
                continue
            print(f"\n[{label} | {L} | {p}]")
            print("  mode          norm(1 rollout)        signal norm            split-half ceil      "
                  "reliab(pooled)        rho vs other 3")
            for m in modes:
                print(f"  {m:12s}  {f(e, 'norm/' + m, 4):22s} {f(e, 'signal_norm/' + m, 4):22s} "
                      f"{f(e, 'ceil/' + m):20s} {f(e, 'reliability/' + m):20s} {f(e, 'rho_others/' + m)}")
            print("  pair    split-half cos (r_k x r_k')   normed=cos/sqrt(ceil ceil)   rho (pooled U-stat)")
            for i, m1 in enumerate(modes):
                for m2 in modes[i + 1:]:
                    k = f"{m1}|{m2}"
                    print(f"  {ab[m1]}-{ab[m2]}     {f(e, 'cross/' + k):28s} {f(e, 'normed/' + k):28s} "
                          f"{f(e, 'rho/' + k)}")
            for name in ("cross", "rho"):
                print(f"  key {name:5s}: C-vs-food {f(e, f'key/{name}:C_vs_food')} | R-vs-SD "
                      f"{f(e, f'key/{name}:R_vs_SD')} | food pairs {f(e, f'key/{name}:food_pairs')} | "
                      f"diff C-R {f(e, f'key/{name}:diff_C-R')} | diff C-foodpairs "
                      f"{f(e, f'key/{name}:diff_C-foodpairs')}")
            print(f"  rho vs others: C {f(e, 'key/rho_others:C')}, R {f(e, 'key/rho_others:R')}, diff "
                  f"{f(e, 'key/rho_others:diff_C-R')}; |g| C/food {f(e, 'key/norm_ratio:C/mean_food')}, "
                  f"signal C/food {f(e, 'key/signal_norm_ratio:C/mean_food')}")
