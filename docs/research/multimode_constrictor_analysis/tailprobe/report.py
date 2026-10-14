"""Print the tail-probe results JSON as compact tables. Usage: python report.py results.json [keys...]"""

import json
import sys


def ci(v):
    return f"{v['est']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]"


def main():
    r = json.load(open(sys.argv[1]))
    filt = sys.argv[2:]
    for ds, res in r.items():
        print(f"\n=== {ds}: {res['decisions']} decisions, {res['games']} games")
        for k, v in res.items():
            if filt and not any(f in k for f in filt):
                continue
            if isinstance(v, dict) and "G" in v and isinstance(v["G"], dict):
                print(f"{k:42s} n={v['n']:6d}  G={ci(v['G'])}  S={ci(v['S'])}  G-S={ci(v['G-S'])}")
            elif isinstance(v, dict) and k.startswith("cf_") and k.endswith("behaviour"):
                print(k)
                for p in ("G", "S"):
                    for m in ("trapT8", "maskedtrapT8", "conbest_spread5"):
                        print(f"   {p} {m:16s} n={v['n_' + m]:5d} real={ci(v[f'{p}_{m}_real'])} "
                              f"cf={ci(v[f'{p}_{m}_cf'])} cf-real={ci(v[f'{p}_{m}_cf-real'])}")
            elif isinstance(v, dict) and k.startswith("cf_"):
                print(f"{k}: n={v['n']}")
                for p in ("G", "S"):
                    print(f"   {p} flip={ci(v[p + '_flip'])} mean_dcon={ci(v[p + '_mean_dcon_per_decision'])} "
                          f"up={ci(v[p + '_frac_flips_up'])} down={ci(v[p + '_frac_flips_down'])}")
                print(f"   G-S flip={ci(v['G-S_flip'])}")
            elif k == "validation":
                print("validation", json.dumps(v))


if __name__ == "__main__":
    main()
