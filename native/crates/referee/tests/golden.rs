//! The ported statistics against the eval-speed branch's strength.py
//! (`native/golden/referee_golden.py --only seq` writes `$NAZ_GOLDEN/referee/seq.json`).
//! Skips when the file is missing.

use std::path::PathBuf;

use serde_json::Value;
use slinky_referee::stats;

fn golden() -> Option<Value> {
    let dir = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let path = PathBuf::from(dir).join("referee/seq.json");
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(serde_json::from_str(&text).expect("seq.json is JSON")),
        Err(_) => {
            println!(
                "SKIP: no {} (run native/golden/referee_golden.py)",
                path.display()
            );
            None
        }
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn sequential_test_matches_python() {
    let Some(g) = golden() else { return };
    let cases = g["seq"].as_array().unwrap();
    assert!(cases.len() >= 100);
    let (mut max_dlog, mut exact) = (0.0f64, 0);
    for (i, c) in cases.iter().enumerate() {
        let rounds: Vec<(u64, u64, u64)> = c["rounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let r = r.as_array().unwrap();
                (
                    r[0].as_u64().unwrap(),
                    r[1].as_u64().unwrap(),
                    r[2].as_u64().unwrap(),
                )
            })
            .collect();
        let st = stats::seq_test(
            &rounds,
            f(&c["p0"]),
            f(&c["margin"]),
            f(&c["alpha"]),
            f(&c["beta"]),
        );
        let decision = st.decision.map(stats::Decision::name);
        assert_eq!(decision, c["decision"].as_str(), "case {i}: decision");
        assert_eq!(
            st.stop_round.map(|r| r as u64),
            c["stop_round"].as_u64(),
            "case {i}: stop round"
        );
        assert_eq!(st.games, c["games"].as_u64().unwrap(), "case {i}: games");
        for (side, key) in [(0, "a"), (1, "b")] {
            assert_eq!(
                st.ended[side].map(|e| e.name()),
                c["ended"][key].as_str(),
                "case {i}"
            );
        }
        for (k, name) in stats::SEQ_TESTS.iter().enumerate() {
            let want = f(&c["log_e"][*name]);
            max_dlog = max_dlog.max((st.log_e[k] - want).abs());
            exact += usize::from(st.log_e[k] == want);
            assert!(
                close(st.log_e[k], want, 1e-12),
                "case {i} {name}: {} vs {want}",
                st.log_e[k]
            );
        }
    }
    println!(
        "{} sequential cases (decisions, stops and test ends all equal), log e-values bit-exact \
         {exact}/{}, max |dlog e| {max_dlog:.1e}",
        cases.len(),
        4 * cases.len()
    );
}

#[test]
fn intervals_bets_and_elo_match_python() {
    let Some(g) = golden() else { return };
    let mut max_cs = 0.0f64;
    for c in g["intervals"].as_array().unwrap() {
        let w = &c["wdl"];
        let (wi, d, l) = (
            w[0].as_u64().unwrap(),
            w[1].as_u64().unwrap(),
            w[2].as_u64().unwrap(),
        );
        let (lo, hi) = stats::score_cs95(wi, d, l);
        // Bisection on a lgamma sum: the bounds agree to rounding of the 50th halving.
        for (x, want) in [(lo, f(&c["cs"][0])), (hi, f(&c["cs"][1]))] {
            max_cs = max_cs.max((x - want).abs());
            assert!((x - want).abs() < 1e-9, "cs {c}: {x} vs {want}");
        }
        let (el, eh) = stats::wilson_eff_interval(wi, d, l);
        assert!(
            close(el, f(&c["eff"][0]), 1e-12) && close(eh, f(&c["eff"][1]), 1e-12),
            "eff {c}"
        );
        let (s, wl, wh) = stats::score_interval(wi, d, l);
        assert!(close(s, f(&c["wilson"][0]), 1e-15), "wilson {c}");
        assert!(
            close(wl, f(&c["wilson"][1]), 1e-12) && close(wh, f(&c["wilson"][2]), 1e-12),
            "wilson {c}"
        );
        let (n, score, ci) = stats::outcome_stats(wi, d, l);
        assert_eq!(n, c["wald"][0].as_u64().unwrap());
        assert!(
            close(score, f(&c["wald"][1]), 1e-15) && close(ci, f(&c["wald"][2]), 1e-12),
            "wald {c}"
        );
    }
    let (mut max_dlam, mut exact) = (0.0f64, 0);
    let bets = g["kelly"].as_array().unwrap();
    for k in bets {
        let lam = stats::kelly_bet(f(&k[0]), f(&k[1]), f(&k[2]));
        assert!(close(lam, f(&k[3]), 1e-12), "kelly {k}: {lam}");
        max_dlam = max_dlam.max((lam - f(&k[3])).abs());
        exact += usize::from(lam == f(&k[3]));
    }
    println!(
        "kelly bets: {exact}/{} bit-exact, max |dlambda| {max_dlam:.1e}",
        bets.len()
    );
    for e in g["elo"].as_array().unwrap() {
        assert!(
            close(stats::score_to_elo(f(&e[0])), f(&e[1]), 1e-12),
            "elo {e}"
        );
    }
    println!("max |dCS| {max_cs:.1e}");
}
