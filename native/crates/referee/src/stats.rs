//! Match statistics: slinky's score intervals and the eval-speed branch's anytime-valid
//! sequential test, ported from `benchmarks/strength.py` (branch `eval-speed`, commit
//! `d4c3328`: `outcome_stats`, `score_interval`, `wilson_eff_interval`, `elo_to_score`,
//! `score_to_elo`, `score_margin`, `kelly_bet`, `log_wealth`, `seq_test`, `_log_mix`,
//! `score_cs`) and `slinky.evaluate` (`_summarize`, `_moments`, `_wilson`).
//!
//! These are pure functions of win/draw/loss counts (per round for the sequential test), so a
//! rerun with the same seed and `--round-games` reproduces the same stop. The functions use
//! the same float operations in the same order as the Python; `tests/` compares them with
//! the Python on random inputs (`native/golden/referee_golden.py --only seq`).

/// The 95% normal quantile (strength.py `Z95`).
pub const Z95: f64 = 1.959963984540054;
/// `--target-ci` never stops before this many games (strength.py `MIN_GAMES_TO_STOP`).
pub const MIN_GAMES_TO_STOP: u64 = 64;
/// Version of the sequential rule (strength.py `SEQ_VERSION`).
pub const SEQ_VERSION: u32 = 1;
/// Draw rate assumed before any game: the shrinkage target of the bet's plug-in.
pub const DRAW_PRIOR: f64 = 0.15;
/// The weight of [`DRAW_PRIOR`] in games.
pub const DRAW_PRIOR_GAMES: f64 = 32.0;
/// Concentration of the confidence sequence's beta mixture.
pub const CS_CONC: f64 = 64.0;

/// `(games, score, ci95)` as `slinky.evaluate` computes them: the score counts a draw as 1/2
/// and `ci95` is the normal-approximation half-width from the sample variance. NaN for no games.
pub fn outcome_stats(wins: u64, draws: u64, losses: u64) -> (u64, f64, f64) {
    let n = wins + draws + losses;
    if n == 0 {
        return (0, f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let score = (wins as f64 + 0.5 * draws as f64) / nf;
    let var = if n > 1 {
        (wins as f64 + 0.25 * draws as f64 - nf * score * score).max(0.0) / (nf - 1.0)
    } else {
        0.0
    };
    (n, score, Z95 * (var / nf).sqrt())
}

/// `(score, low, high)`: the Wilson score interval of the score as a proportion of `n` games
/// (strength.py `score_interval`). NaNs for no games.
pub fn score_interval(wins: u64, draws: u64, losses: u64) -> (f64, f64, f64) {
    let n = (wins + draws + losses) as f64;
    if n == 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let p = (wins as f64 + 0.5 * draws as f64) / n;
    let (low, high) = wilson(p, n);
    (p, low, high)
}

/// The Wilson score interval of a proportion `p` over `n` trials (`evaluate._wilson`).
pub fn wilson(p: f64, n: f64) -> (f64, f64) {
    let z2 = Z95 * Z95;
    let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
    (
        if p == 0.0 { 0.0 } else { centre - half },
        if p == 1.0 { 1.0 } else { centre + half },
    )
}

/// Draw-aware Wilson interval (strength.py `wilson_eff_interval`): `n` is replaced by the
/// effective `n p (1 - p) / sample variance`, never below `n`. For a fixed number of games.
/// `(0, 1)` for no games.
pub fn wilson_eff_interval(wins: u64, draws: u64, losses: u64) -> (f64, f64) {
    let n = (wins + draws + losses) as f64;
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let (w, d) = (wins as f64, draws as f64);
    let p = (w + d / 2.0) / n;
    let v = (w + d / 4.0 - n * p * p).max(0.0) / (n - 1.0).max(1.0);
    let ne = if v > 0.0 {
        (n * p * (1.0 - p) / v).max(n)
    } else {
        n
    };
    wilson(p, ne)
}

/// The expected score of an Elo difference.
pub fn elo_to_score(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

/// The Elo difference of a score (clamped to `[1e-9, 1 - 1e-9]`).
pub fn score_to_elo(s: f64) -> f64 {
    let s = s.clamp(1e-9, 1.0 - 1e-9);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// The duel margin in score units for an Elo difference (35 Elo is 0.0502).
pub fn score_margin(elo: f64) -> f64 {
    elo_to_score(elo) - 0.5
}

/// The Kelly fraction: the `lambda` maximising `E log(1 + lambda (X - m_null))` when `X` takes
/// 1, 1/2, 0 with mean `m_alt` and draw rate `draw` (bisection on the derivative, 60 steps).
pub fn kelly_bet(m_null: f64, m_alt: f64, draw: f64) -> f64 {
    let draw = draw.min(0.999 * 2.0 * m_alt.min(1.0 - m_alt));
    let q = [
        (m_alt - draw / 2.0, 1.0 - m_null),
        (draw, 0.5 - m_null),
        (1.0 - m_alt - draw / 2.0, -m_null),
    ];
    let (mut lo, mut hi) = (-0.999 / (1.0 - m_null), 0.999 / m_null);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        let g = py_sum(q.iter().map(|&(qx, x)| qx * x / (1.0 + mid * x)));
        if g > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Python 3.12's `sum` of floats: Neumaier's compensated summation from 0.
fn py_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut c) = (0.0f64, 0.0f64);
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

/// `log` of the product over a round's games of `1 + lam (x - m_null)`.
pub fn log_wealth(wins: u64, draws: u64, losses: u64, lam: f64, m_null: f64) -> f64 {
    let mut out = 0.0;
    for (count, x) in [(wins, 1.0), (draws, 0.5), (losses, 0.0)] {
        if count > 0 {
            out += count as f64 * (1.0 + lam * (x - m_null)).ln();
        }
    }
    out
}

/// How a one-sided pair of tests ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideEnd {
    /// The claim test crossed (this side is stronger).
    Claim,
    /// The futility test crossed (this side is not stronger by the margin).
    Futile,
}

impl SideEnd {
    /// `"claim"` or `"futile"` (strength.py's `seq_tests` values).
    pub fn name(self) -> &'static str {
        match self {
            SideEnd::Claim => "claim",
            SideEnd::Futile => "futile",
        }
    }
}

/// The decision of the sequential test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// A is stronger than B.
    AStronger,
    /// B is stronger than A.
    BStronger,
    /// Neither is stronger by the margin.
    Within,
}

impl Decision {
    /// `"A>B"`, `"B>A"` or `"within"` (strength.py's `seq_decision`).
    pub fn name(self) -> &'static str {
        match self {
            Decision::AStronger => "A>B",
            Decision::BStronger => "B>A",
            Decision::Within => "within",
        }
    }
}

/// Names of the four e-processes, in strength.py's order.
pub const SEQ_TESTS: [&str; 4] = ["e_a", "f_a", "e_b", "f_b"];

/// The sequential test's state after the last round given (logs in nats).
#[derive(Debug, Clone, PartialEq)]
pub struct SeqState {
    /// Games seen.
    pub games: u64,
    /// Log e-values of `e_a`, `f_a`, `e_b`, `f_b` ([`SEQ_TESTS`]).
    pub log_e: [f64; 4],
    /// How each side's tests ended: `[a, b]`.
    pub ended: [Option<SideEnd>; 2],
    /// The decision, once there is one.
    pub decision: Option<Decision>,
    /// The 1-based round of the decision.
    pub stop_round: Option<usize>,
}

/// The parameters of `--sequential`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeqSpec {
    /// A's score when the agents are equally strong.
    pub p0: f64,
    /// The smallest score difference that matters, in score units.
    pub margin: f64,
    /// False-claim rate per side.
    pub alpha: f64,
    /// Futility error rate per side.
    pub beta: f64,
    /// Two players at `p0 = 1/2`: margins and results are also given in Elo.
    pub duel: bool,
}

/// Two one-sided betting tests around `p0`, each frozen at its first boundary (strength.py
/// `seq_test`):
///
/// * `e_a` against "score <= p0" (bets on `p0 + margin`): crosses `1/alpha` -> `A>B`;
/// * `e_b` against "score >= p0" (bets on `p0 - margin`): crosses `1/alpha` -> `B>A`;
/// * `f_a` against "score >= p0 + margin" (bets on `p0`): crosses `1/beta` -> A not stronger
///   by the margin;
/// * `f_b` against "score <= p0 - margin" (bets on `p0`): crosses `1/beta` -> B not stronger
///   by the margin.
///
/// Stops at the first claim, or "within" once both futility tests crossed. Each bet uses the
/// Kelly fraction for a trinomial whose draw rate is estimated from earlier rounds only.
pub fn seq_test(
    rounds: &[(u64, u64, u64)],
    p0: f64,
    margin: f64,
    alpha: f64,
    beta: f64,
) -> SeqState {
    let (hi, lo) = (p0 + margin, p0 - margin);
    let claim_thr = (1.0 / alpha).ln();
    let futile_thr = (1.0 / beta).ln();
    // (null mean, alternative mean, threshold), in SEQ_TESTS order.
    let tests = [
        (p0, hi, claim_thr),
        (hi, p0, futile_thr),
        (p0, lo, claim_thr),
        (lo, p0, futile_thr),
    ];
    let mut st = SeqState {
        games: 0,
        log_e: [0.0; 4],
        ended: [None, None],
        decision: None,
        stop_round: None,
    };
    let (mut d_total, mut n_total) = (0u64, 0u64);
    for (k, &(w, d, loss)) in rounds.iter().enumerate() {
        let draw_hat =
            (d_total as f64 + DRAW_PRIOR * DRAW_PRIOR_GAMES) / (n_total as f64 + DRAW_PRIOR_GAMES);
        for (i, &(m_null, m_alt, _)) in tests.iter().enumerate() {
            let lam = kelly_bet(m_null, m_alt, draw_hat);
            st.log_e[i] += log_wealth(w, d, loss, lam, m_null);
        }
        d_total += d;
        n_total += w + d + loss;
        st.games = n_total;
        for side in 0..2 {
            if st.ended[side].is_none() {
                let (e, f) = (2 * side, 2 * side + 1);
                if st.log_e[e] >= tests[e].2 {
                    st.ended[side] = Some(SideEnd::Claim);
                } else if st.log_e[f] >= tests[f].2 {
                    st.ended[side] = Some(SideEnd::Futile);
                }
            }
        }
        if st.ended[0] == Some(SideEnd::Claim) {
            st.decision = Some(Decision::AStronger);
        } else if st.ended[1] == Some(SideEnd::Claim) {
            st.decision = Some(Decision::BStronger);
        } else if st.ended == [Some(SideEnd::Futile), Some(SideEnd::Futile)] {
            st.decision = Some(Decision::Within);
        }
        if st.decision.is_some() {
            st.stop_round = Some(k + 1);
            break;
        }
    }
    st
}

/// `ln Gamma(x)` for `x > 0`: CPython's `math.lgamma` (Lanczos approximation, `g = 6.0247`,
/// 13 terms), so the confidence sequence matches the Python to rounding. The constants are
/// CPython's as written (`Modules/mathmodule.c`).
#[allow(clippy::excessive_precision)]
pub fn lgamma(x: f64) -> f64 {
    const G: f64 = 6.024_680_040_776_729_583_740_234_375;
    const NUM: [f64; 13] = [
        23_531_376_880.410_759,
        42_919_803_642.649_1,
        35_711_959_237.355_67,
        17_921_034_426.037_21,
        6_039_542_586.352_028,
        1_439_720_407.311_721_6,
        248_874_557.862_054_16,
        31_426_415.585_400_194,
        2_876_370.628_935_372_4,
        186_056.265_395_223_5,
        8_071.672_002_365_816,
        210.824_277_751_579_35,
        2.506_628_274_631_000_3,
    ];
    const DEN: [f64; 13] = [
        0.0,
        39_916_800.0,
        120_543_840.0,
        150_917_976.0,
        105_258_076.0,
        45_995_730.0,
        13_339_535.0,
        2_637_558.0,
        357_423.0,
        32_670.0,
        1_925.0,
        66.0,
        1.0,
    ];
    if !x.is_finite() {
        return x.abs();
    }
    if x == x.floor() && x <= 2.0 {
        return if x <= 0.0 { f64::INFINITY } else { 0.0 };
    }
    let absx = x.abs();
    if absx < 1e-20 {
        return -absx.ln();
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    if absx < 5.0 {
        for i in (0..13).rev() {
            num = num * absx + NUM[i];
            den = den * absx + DEN[i];
        }
    } else {
        for i in 0..13 {
            num = num / absx + NUM[i];
            den = den / absx + DEN[i];
        }
    }
    let mut r = (num / den).ln() - G;
    r += (absx - 0.5) * ((absx + G - 0.5).ln() - 1.0);
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        r = pi.ln() - (pi * absx).sin().abs().ln() - absx.ln() - r;
    }
    r
}

/// The log of the beta-mixture martingale at `p` (strength.py `_log_mix`).
fn log_mix(s: f64, n: f64, p: f64, conc: f64) -> f64 {
    let (a, b) = (conc * p, conc * (1.0 - p));
    let mut out = lgamma(a + s) + lgamma(b + n - s) - lgamma(a + b + n);
    out -= lgamma(a) + lgamma(b) - lgamma(a + b);
    if s > 0.0 {
        out -= s * p.ln();
    }
    if n - s > 0.0 {
        out -= (n - s) * (1.0 - p).ln();
    }
    out
}

/// The always-valid confidence sequence `(low, high)` of the score (strength.py `score_cs`):
/// `{p : M_n(p) < 1/alpha}` for a beta mixture of concentration `conc`, valid simultaneously
/// at all rotation-group boundaries, so at any stopping time.
pub fn score_cs(wins: u64, draws: u64, losses: u64, alpha: f64, conc: f64) -> (f64, f64) {
    let n = (wins + draws + losses) as f64;
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let s = wins as f64 + draws as f64 / 2.0;
    let phat = s / n;
    let thr = (1.0 / alpha).ln();
    let bound = |far: f64, edge: f64| -> f64 {
        if log_mix(s, n, far, conc) < thr {
            return edge;
        }
        let (mut out, mut inside) = (far, phat);
        for _ in 0..50 {
            let mid = (out + inside) / 2.0;
            if log_mix(s, n, mid, conc) < thr {
                inside = mid;
            } else {
                out = mid;
            }
        }
        out
    };
    (bound(1e-9, 0.0), bound(1.0 - 1e-9, 1.0))
}

/// [`score_cs`] at strength.py's defaults (alpha 0.05, [`CS_CONC`]).
pub fn score_cs95(wins: u64, draws: u64, losses: u64) -> (f64, f64) {
    score_cs(wins, draws, losses, 0.05, CS_CONC)
}

/// Mean and 95% half-width of the mean (sample variance; `evaluate._moments`). NaNs when empty.
pub fn mean_ci(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = if values.len() > 1 {
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, Z95 * (var / n).sqrt())
}

/// The `q`-quantile (0 to 1) of sorted values with linear interpolation (numpy's default).
/// NaN when empty.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let frac = pos - i as f64;
    match sorted.get(i + 1) {
        Some(&next) => sorted[i] + frac * (next - sorted[i]),
        None => sorted[i],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wald_and_wilson_match_hand_values() {
        // 3 wins, 2 draws, 5 losses: score 0.4, sample variance (3 + 0.5 - 1.6) / 9.
        let (n, score, ci) = outcome_stats(3, 2, 5);
        assert_eq!(n, 10);
        assert!((score - 0.4).abs() < 1e-15);
        let var: f64 = (3.0 + 0.5 - 10.0 * 0.16) / 9.0;
        assert!((ci - Z95 * (var / 10.0).sqrt()).abs() < 1e-15);
        let (p, lo, hi) = score_interval(3, 2, 5);
        assert!((p - 0.4).abs() < 1e-15);
        // Wilson at p = 0.4, n = 10 (z = 1.96): about [0.168, 0.687].
        assert!(
            (lo - 0.16818).abs() < 1e-4 && (hi - 0.68736).abs() < 1e-4,
            "{lo} {hi}"
        );
        assert_eq!(score_interval(0, 0, 7).1, 0.0);
        assert_eq!(score_interval(7, 0, 0).2, 1.0);
        assert!(outcome_stats(0, 0, 0).1.is_nan());
    }

    #[test]
    fn draws_narrow_the_effective_interval() {
        let (_, wl, wh) = score_interval(30, 40, 30);
        let (el, eh) = wilson_eff_interval(30, 40, 30);
        assert!(eh - el < wh - wl);
        // No draws: the same as Wilson.
        let (_, wl, wh) = score_interval(30, 0, 70);
        let (el, eh) = wilson_eff_interval(30, 0, 70);
        assert!((wl - el).abs() < 1e-12 && (wh - eh).abs() < 1e-12);
    }

    #[test]
    fn elo_round_trips() {
        for e in [-300.0, -35.0, 0.0, 19.0, 200.0] {
            assert!((score_to_elo(elo_to_score(e)) - e).abs() < 1e-9);
        }
        assert!((score_margin(35.0) - 0.0502).abs() < 1e-4);
        assert!(score_to_elo(1.0).is_finite() && score_to_elo(0.0).is_finite());
    }

    #[test]
    fn kelly_bet_is_the_log_optimal_fraction() {
        let lam = kelly_bet(0.5, 0.55, 0.2);
        assert!(lam > 0.0);
        let growth = |l: f64| {
            let (w, d) = (0.55 - 0.1, 0.2);
            let lo = 1.0 - w - d;
            w * (1.0 + l * 0.5).ln() + d * 1.0f64.ln() + lo * (1.0 - l * 0.5).ln()
        };
        assert!(growth(lam) >= growth(lam * 0.9) && growth(lam) >= growth(lam * 1.1));
        assert!(kelly_bet(0.5, 0.45, 0.2) < 0.0);
    }

    #[test]
    fn sequential_test_decides_obvious_cases() {
        let all_wins = vec![(64, 0, 0); 16];
        let st = seq_test(&all_wins, 0.5, score_margin(35.0), 0.05, 0.05);
        assert_eq!(st.decision, Some(Decision::AStronger));
        assert_eq!(st.stop_round, Some(1));
        let all_losses = vec![(0, 0, 64); 16];
        let st = seq_test(&all_losses, 0.5, score_margin(35.0), 0.05, 0.05);
        assert_eq!(st.decision, Some(Decision::BStronger));
        // Exactly even rounds: eventually "within".
        let even = vec![(24, 16, 24); 64];
        let st = seq_test(&even, 0.5, score_margin(35.0), 0.05, 0.05);
        assert_eq!(st.decision, Some(Decision::Within));
        assert_eq!(st.ended, [Some(SideEnd::Futile), Some(SideEnd::Futile)]);
        // Nothing yet.
        let st = seq_test(&[(30, 4, 30)], 0.5, score_margin(35.0), 0.05, 0.05);
        assert_eq!(st.decision, None);
        assert_eq!(st.games, 64);
    }

    #[test]
    fn lgamma_matches_known_values() {
        assert_eq!(lgamma(1.0), 0.0);
        assert_eq!(lgamma(2.0), 0.0);
        assert!((lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
        // ln(10!) = ln Gamma(11).
        let ln_fact10 = (1..=10).map(|k| (k as f64).ln()).sum::<f64>();
        assert!((lgamma(11.0) - ln_fact10).abs() < 1e-12);
        assert!(
            (lgamma(1000.5) - 5_908.674_175_848_678).abs() < 1e-9,
            "{}",
            lgamma(1000.5)
        );
    }

    #[test]
    fn confidence_sequence_contains_the_score_and_is_wider_than_wilson() {
        let (lo, hi) = score_cs95(300, 100, 300);
        let (_, wl, wh) = score_interval(300, 100, 300);
        assert!(lo < 0.5 && hi > 0.5);
        assert!(hi - lo > wh - wl);
        assert_eq!(score_cs95(0, 0, 0), (0.0, 1.0));
        let (lo, hi) = score_cs95(64, 0, 0);
        assert!(lo > 0.5 && hi == 1.0);
    }

    #[test]
    fn quantiles_interpolate() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 1.0), 4.0);
        assert!((quantile(&v, 0.5) - 2.5).abs() < 1e-15);
        assert!(quantile(&[], 0.5).is_nan());
        assert_eq!(quantile(&[7.0], 0.9), 7.0);
        let (m, h) = mean_ci(&[1.0, 1.0, 1.0]);
        assert_eq!((m, h), (1.0, 0.0));
    }
}
