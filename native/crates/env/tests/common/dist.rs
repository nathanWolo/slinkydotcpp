//! Distribution statistics of whole random-legal games, in the layout `env_golden.py --only dist`
//! writes, and their side-by-side comparison with tolerances. Shared by `tests/distributions.rs`
//! and `examples/env_stats.rs`.

#![allow(dead_code)]

use serde_json::Value;
use slinky_env::{EnvConfig, MapKind, Rng, random_legal};
use slinky_rules::{GameConfig, Ruleset};
use std::collections::BTreeMap;

/// The aggregates of one config's games (see the docstring of `env_golden.py`).
#[derive(Clone, Debug, Default)]
pub struct Dist {
    pub games: u64,
    pub start_joint: BTreeMap<String, u64>,
    pub start_seat_cell: Vec<Vec<u64>>,
    pub start_food_count: Vec<u64>,
    pub start_food_cell: Vec<u64>,
    pub sides_pos: Vec<Vec<u64>>,
    pub sides_pair: Vec<u64>,
    pub first_hazard_turn: BTreeMap<i64, u64>,
    pub turn_active: Vec<u64>,
    pub food_sum: Vec<u64>,
    pub food_sq: Vec<u64>,
    pub hazard_sum: Vec<u64>,
    /// Sum of squared hazard counts per turn (Rust only; the null-hypothesis spread).
    pub hazard_sq: Vec<u64>,
    pub spawn_low: [u64; 3],
    pub spawn_high: [u64; 2],
    pub length: Vec<u64>,
    pub truncated: u64,
    pub alive_end: Vec<u64>,
    pub winner: Vec<u64>,
    pub causes: Vec<u64>,
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect())
        .unwrap_or_default()
}

/// The environment a golden file's `config` describes.
pub fn env_of(v: &Value) -> EnvConfig {
    let c = &v["config"];
    let num = |k: &str| c[k].as_i64().unwrap();
    let max_turns = num("max_turns");
    let rules = GameConfig::new(
        num("width") as usize,
        num("height") as usize,
        num("num_snakes") as usize,
        Ruleset::from_name(c["ruleset"].as_str().unwrap()).unwrap(),
    )
    .unwrap()
    .with_shrink_every(num("shrink_every") as i32)
    .unwrap()
    .with_max_turns((max_turns >= 0).then_some(max_turns as i32));
    EnvConfig::new(
        rules,
        MapKind::from_name(c["map"].as_str().unwrap()).unwrap(),
    )
    .unwrap()
    .with_food(num("food_spawn_chance") as i32, num("minimum_food") as i32)
}

impl Dist {
    /// A golden JSON file's aggregates.
    pub fn from_json(v: &Value) -> Dist {
        let fixed = |k: &str, len: usize| {
            let mut out = u64s(&v[k]);
            out.resize(len, 0);
            out
        };
        let low = fixed("spawn_low", 3);
        let high = fixed("spawn_high", 2);
        Dist {
            games: v["games"].as_u64().unwrap(),
            start_joint: v["start_joint"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, c)| (k.clone(), c.as_u64().unwrap()))
                .collect(),
            start_seat_cell: v["start_seat_cell"]
                .as_array()
                .unwrap()
                .iter()
                .map(u64s)
                .collect(),
            start_food_count: u64s(&v["start_food_count"]),
            start_food_cell: u64s(&v["start_food_cell"]),
            sides_pos: v["sides_pos"]
                .as_array()
                .map(|a| a.iter().map(u64s).collect())
                .unwrap_or_default(),
            sides_pair: u64s(&v["sides_pair"]),
            first_hazard_turn: v["first_hazard_turn"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, c)| (k.parse().unwrap(), c.as_u64().unwrap()))
                        .collect()
                })
                .unwrap_or_default(),
            turn_active: u64s(&v["turn_active"]),
            food_sum: u64s(&v["food_sum"]),
            food_sq: u64s(&v["food_sq"]),
            hazard_sum: u64s(&v["hazard_sum"]),
            hazard_sq: u64s(&v["hazard_sq"]),
            spawn_low: [low[0], low[1], low[2]],
            spawn_high: [high[0], high[1]],
            length: u64s(&v["length"]),
            truncated: v["truncated"].as_u64().unwrap(),
            alive_end: u64s(&v["alive_end"]),
            winner: u64s(&v["winner"]),
            causes: u64s(&v["causes"]),
        }
    }

    /// Play `games` random-legal games of `env` (game `g` uses `Rng::with_stream(seed, g)`) and
    /// aggregate them exactly as the Python script does.
    pub fn simulate(env: &EnvConfig, games: u64, seed: u64) -> Dist {
        let cfg = *env.rules();
        let (n, cells) = (cfg.num_snakes(), cfg.cells());
        let mut d = Dist {
            games,
            start_seat_cell: vec![vec![0; cells]; n],
            start_food_count: vec![0; cells + 1],
            start_food_cell: vec![0; cells],
            causes: vec![0; 7],
            alive_end: vec![0; n + 1],
            winner: vec![0; n],
            ..Dist::default()
        };
        if env.has_sides() {
            d.sides_pos = vec![vec![0; 4]; cfg.num_sides()];
            d.sides_pair = vec![0; 16];
        }
        let grow = |v: &mut Vec<u64>, t: usize| {
            if v.len() <= t {
                v.resize(t + 1, 0);
            }
        };
        for g in 0..games {
            let mut rng = Rng::with_stream(seed, g);
            let mut s = env.reset(&mut rng);
            let key: Vec<String> = (0..n)
                .map(|i| format!("{},{}", s.head[i][0], s.head[i][1]))
                .collect();
            *d.start_joint.entry(key.join(";")).or_default() += 1;
            for i in 0..n {
                if let Some(c) = s.head_cell(&cfg, i) {
                    d.start_seat_cell[i][c] += 1;
                }
            }
            d.start_food_count[s.food.count_ones() as usize] += 1;
            for c in 0..cells {
                d.start_food_cell[c] += u64::from(s.has_food(c));
            }
            if env.has_sides() {
                for k in 0..cfg.num_sides() {
                    d.sides_pos[k][s.sides[k] as usize] += 1;
                }
                d.sides_pair[(s.sides[0] * 4 + s.sides[1]) as usize] += 1;
            }
            let mut seen_hazard = false;
            while !s.done {
                let t = s.turn as usize;
                for v in [
                    &mut d.turn_active,
                    &mut d.food_sum,
                    &mut d.food_sq,
                    &mut d.hazard_sum,
                    &mut d.hazard_sq,
                ] {
                    grow(v, t);
                }
                let food = u64::from(s.food.count_ones());
                let hazard = s.hazard[..cells].iter().filter(|&&h| h > 0).count() as u64;
                d.turn_active[t] += 1;
                d.food_sum[t] += food;
                d.food_sq[t] += food * food;
                d.hazard_sum[t] += hazard;
                d.hazard_sq[t] += hazard * hazard;
                if hazard > 0 && !seen_hazard {
                    seen_hazard = true;
                    *d.first_hazard_turn.entry(t as i64).or_default() += 1;
                }
                let actions = random_legal(&mut rng, &cfg, &s);
                let info = env.step(&mut rng, &mut s, &actions);
                let spawned = u64::from(info.spawned);
                let post = i64::from(s.food.count_ones()) - spawned as i64;
                if post < i64::from(env.minimum_food()) {
                    d.spawn_low[0] += 1;
                    d.spawn_low[1] += spawned;
                    d.spawn_low[2] += spawned * spawned;
                } else {
                    d.spawn_high[0] += 1;
                    d.spawn_high[1] += u64::from(spawned > 0);
                }
            }
            let t = s.turn as usize;
            grow(&mut d.length, t);
            d.length[t] += 1;
            let alive = s.num_alive(&cfg);
            d.truncated += u64::from(alive > 1);
            d.alive_end[alive] += 1;
            for i in 0..n {
                if alive == 1 && s.alive[i] {
                    d.winner[i] += 1;
                }
                d.causes[s.elim_cause[i].clamp(0, 6) as usize] += 1;
            }
        }
        d
    }

    fn mean_length(&self) -> f64 {
        let (s, c) = self
            .length
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(s, c), (t, &k)| {
                (s + t as f64 * k as f64, c + k as f64)
            });
        s / c.max(1.0)
    }

    fn length_moments(&self) -> (f64, f64) {
        let m = self.mean_length();
        let (v, c) = self
            .length
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(v, c), (t, &k)| {
                (v + (t as f64 - m).powi(2) * k as f64, c + k as f64)
            });
        (m, v / (c - 1.0).max(1.0))
    }

    fn quantile(&self, q: f64) -> usize {
        let total: u64 = self.length.iter().sum();
        let target = (q * total as f64).ceil().max(1.0) as u64;
        let mut acc = 0;
        for (t, &k) in self.length.iter().enumerate() {
            acc += k;
            if acc >= target {
                return t;
            }
        }
        self.length.len()
    }
}

/// One compared statistic.
#[derive(Clone, Debug)]
pub struct Row {
    pub stat: String,
    pub python: String,
    pub rust: String,
    /// |z| (or the KS distance) and its tolerance.
    pub score: f64,
    pub tol: f64,
}

impl Row {
    pub fn ok(&self) -> bool {
        self.score <= self.tol
    }
}

/// The |z| tolerance of every proportion and mean comparison. With a few thousand comparisons
/// over all configs, a false alarm at 4.5 sigma has a probability of about 1%.
pub const Z_TOL: f64 = 4.5;
/// Two-sample Kolmogorov-Smirnov coefficient for alpha = 1e-4: `sqrt(-ln(alpha / 2) / 2)`.
pub const KS_C: f64 = 2.226;

fn z_prop(c1: u64, n1: u64, c2: u64, n2: u64) -> f64 {
    if n1 == 0 || n2 == 0 {
        return 0.0;
    }
    let (p1, p2) = (c1 as f64 / n1 as f64, c2 as f64 / n2 as f64);
    let p = (c1 + c2) as f64 / (n1 + n2) as f64;
    let se = (p * (1.0 - p) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    if se == 0.0 {
        if p1 == p2 { 0.0 } else { f64::INFINITY }
    } else {
        ((p1 - p2) / se).abs()
    }
}

/// |z| of two sample means from (count, sum, sum of squares).
fn z_mean(n1: u64, s1: f64, q1: f64, n2: u64, s2: f64, q2: f64) -> (f64, f64, f64) {
    let (n1f, n2f) = (n1 as f64, n2 as f64);
    let (m1, m2) = (s1 / n1f, s2 / n2f);
    let v1 = ((q1 - n1f * m1 * m1) / (n1f - 1.0).max(1.0)).max(0.0);
    let v2 = ((q2 - n2f * m2 * m2) / (n2f - 1.0).max(1.0)).max(0.0);
    let se = (v1 / n1f + v2 / n2f).sqrt();
    let z = if se == 0.0 {
        if (m1 - m2).abs() < 1e-12 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((m1 - m2) / se).abs()
    };
    (m1, m2, z)
}

fn prop_row(stat: String, c1: u64, n1: u64, c2: u64, n2: u64) -> Row {
    Row {
        stat,
        python: format!("{:.4}", c1 as f64 / n1.max(1) as f64),
        rust: format!("{:.4}", c2 as f64 / n2.max(1) as f64),
        score: z_prop(c1, n1, c2, n2),
        tol: Z_TOL,
    }
}

fn hist_rows(rows: &mut Vec<Row>, name: &str, a: &[u64], na: u64, b: &[u64], nb: u64) {
    for k in 0..a.len().max(b.len()) {
        let (x, y) = (
            a.get(k).copied().unwrap_or(0),
            b.get(k).copied().unwrap_or(0),
        );
        if x + y > 0 {
            rows.push(prop_row(format!("{name}[{k}]"), x, na, y, nb));
        }
    }
}

/// Every comparison of slinky's aggregates `py` with the Rust ones `rs`.
pub fn compare(py: &Dist, rs: &Dist, cfg: &GameConfig) -> Vec<Row> {
    let (g1, g2) = (py.games, rs.games);
    let mut rows = Vec::new();
    // Start positions: the joint seat-ordered heads when the support is small, else per seat.
    let keys: std::collections::BTreeSet<&String> =
        py.start_joint.keys().chain(rs.start_joint.keys()).collect();
    if keys.len() <= 200 {
        for k in keys {
            let (x, y) = (
                py.start_joint.get(k).copied().unwrap_or(0),
                rs.start_joint.get(k).copied().unwrap_or(0),
            );
            rows.push(prop_row(format!("start heads {k}"), x, g1, y, g2));
        }
    }
    rows.push(Row {
        stat: "distinct start head tuples".into(),
        python: py.start_joint.len().to_string(),
        rust: rs.start_joint.len().to_string(),
        score: 0.0,
        tol: 0.0,
    });
    for (i, (a, b)) in py
        .start_seat_cell
        .iter()
        .zip(&rs.start_seat_cell)
        .enumerate()
    {
        for c in 0..a.len().min(b.len()) {
            if a[c] + b[c] > 0 {
                let (x, y) = cfg.xy(c);
                rows.push(prop_row(
                    format!("seat {i} starts at ({x},{y})"),
                    a[c],
                    g1,
                    b[c],
                    g2,
                ));
            }
        }
    }
    hist_rows(
        &mut rows,
        "start food count",
        &py.start_food_count,
        g1,
        &rs.start_food_count,
        g2,
    );
    for c in 0..py.start_food_cell.len().min(rs.start_food_cell.len()) {
        let (a, b) = (py.start_food_cell[c], rs.start_food_cell[c]);
        if a + b > 0 {
            let (x, y) = cfg.xy(c);
            rows.push(prop_row(format!("start food at ({x},{y})"), a, g1, b, g2));
        }
    }
    // Royale sides.
    for (k, (a, b)) in py.sides_pos.iter().zip(&rs.sides_pos).enumerate() {
        hist_rows(&mut rows, &format!("shrink {k} side"), a, g1, b, g2);
    }
    hist_rows(
        &mut rows,
        "sides (k0, k1) pair",
        &py.sides_pair,
        g1,
        &rs.sides_pair,
        g2,
    );
    if !py.first_hazard_turn.is_empty() || !rs.first_hazard_turn.is_empty() {
        let fmt = |m: &BTreeMap<i64, u64>| format!("{:?}", m.keys().collect::<Vec<_>>());
        let same = py.first_hazard_turn.keys().eq(rs.first_hazard_turn.keys());
        rows.push(Row {
            stat: "first turn showing a hazard".into(),
            python: fmt(&py.first_hazard_turn),
            rust: fmt(&rs.first_hazard_turn),
            score: if same { 0.0 } else { 1.0 },
            tol: 0.0,
        });
    }
    // Per-turn food and hazard means where both sides have at least 200 games running.
    let turns = py.turn_active.len().min(rs.turn_active.len());
    for t in (0..turns).filter(|t| t % 5 == 0 || *t < 5) {
        let (n1, n2) = (py.turn_active[t], rs.turn_active[t]);
        if n1 < 200 || n2 < 200 {
            continue;
        }
        let (m1, m2, z) = z_mean(
            n1,
            py.food_sum[t] as f64,
            py.food_sq[t] as f64,
            n2,
            rs.food_sum[t] as f64,
            rs.food_sq[t] as f64,
        );
        rows.push(Row {
            stat: format!("mean food at turn {t}"),
            python: format!("{m1:.3}"),
            rust: format!("{m2:.3}"),
            score: z,
            tol: Z_TOL,
        });
        rows.push(prop_row(
            format!("games running at turn {t}"),
            n1,
            g1,
            n2,
            g2,
        ));
        if py.hazard_sum[t] + rs.hazard_sum[t] > 0 {
            let (h1, h2) = (
                py.hazard_sum[t] as f64 / n1 as f64,
                rs.hazard_sum[t] as f64 / n2 as f64,
            );
            // The spread under the null hypothesis (same distribution): the variance of the
            // sample that has the squares (the Rust one).
            let (sq, n0, sum) = if rs.hazard_sq.len() > t {
                (rs.hazard_sq[t] as f64, n2 as f64, rs.hazard_sum[t] as f64)
            } else {
                (
                    py.hazard_sq.get(t).copied().unwrap_or(0) as f64,
                    n1 as f64,
                    py.hazard_sum[t] as f64,
                )
            };
            let var = ((sq - sum * sum / n0) / (n0 - 1.0).max(1.0)).max(0.0);
            let se = (var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
            rows.push(Row {
                stat: format!("mean hazard cells at turn {t}"),
                python: format!("{h1:.3}"),
                rust: format!("{h2:.3}"),
                score: if se > 0.0 {
                    (h1 - h2).abs() / se
                } else if (h1 - h2).abs() < 1e-12 {
                    0.0
                } else {
                    f64::INFINITY
                },
                tol: Z_TOL,
            });
        }
    }
    // Spawns.
    if py.spawn_low[0] + rs.spawn_low[0] > 0 {
        let (m1, m2, z) = z_mean(
            py.spawn_low[0],
            py.spawn_low[1] as f64,
            py.spawn_low[2] as f64,
            rs.spawn_low[0],
            rs.spawn_low[1] as f64,
            rs.spawn_low[2] as f64,
        );
        rows.push(Row {
            stat: "food spawned per step below minimum_food".into(),
            python: format!("{m1:.4} ({} steps)", py.spawn_low[0]),
            rust: format!("{m2:.4} ({} steps)", rs.spawn_low[0]),
            score: z,
            tol: Z_TOL,
        });
    }
    rows.push(prop_row(
        "spawn rate at or above minimum_food".into(),
        py.spawn_high[1],
        py.spawn_high[0],
        rs.spawn_high[1],
        rs.spawn_high[0],
    ));
    // Game length: KS distance, mean, quantiles.
    let (cdf1, cdf2) = (cdf(&py.length), cdf(&rs.length));
    let d = (0..cdf1.len().max(cdf2.len()))
        .map(|t| (at(&cdf1, t) - at(&cdf2, t)).abs())
        .fold(0.0, f64::max);
    let (n1, n2) = (g1 as f64, g2 as f64);
    rows.push(Row {
        stat: "game length KS distance".into(),
        python: format!("{} games", g1),
        rust: format!("{} games", g2),
        score: d,
        tol: KS_C * ((n1 + n2) / (n1 * n2)).sqrt(),
    });
    let ((m1, v1), (m2, v2)) = (py.length_moments(), rs.length_moments());
    rows.push(Row {
        stat: "mean game length (sd)".into(),
        python: format!("{m1:.2} ({:.1})", v1.sqrt()),
        rust: format!("{m2:.2} ({:.1})", v2.sqrt()),
        score: (m1 - m2).abs() / (v1 / n1 + v2 / n2).sqrt().max(1e-12),
        tol: Z_TOL,
    });
    let qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    rows.push(Row {
        stat: format!("length quantiles {qs:?}"),
        python: format!("{:?}", qs.map(|q| py.quantile(q))),
        rust: format!("{:?}", qs.map(|q| rs.quantile(q))),
        score: 0.0,
        tol: 0.0,
    });
    rows.push(prop_row(
        "truncated".into(),
        py.truncated,
        g1,
        rs.truncated,
        g2,
    ));
    hist_rows(
        &mut rows,
        "snakes alive at the end",
        &py.alive_end,
        g1,
        &rs.alive_end,
        g2,
    );
    hist_rows(&mut rows, "wins by seat", &py.winner, g1, &rs.winner, g2);
    let n = cfg.num_snakes() as u64;
    hist_rows(
        &mut rows,
        "elimination cause",
        &py.causes,
        g1 * n,
        &rs.causes,
        g2 * n,
    );
    rows
}

fn cdf(hist: &[u64]) -> Vec<f64> {
    let total: u64 = hist.iter().sum();
    let mut acc = 0;
    hist.iter()
        .map(|&k| {
            acc += k;
            acc as f64 / total.max(1) as f64
        })
        .collect()
}

fn at(cdf: &[f64], t: usize) -> f64 {
    cdf.get(t).copied().unwrap_or(1.0)
}

/// A table of the rows: the summary ones always, the rest only when failing (or `all`).
pub fn table(name: &str, rows: &[Row], all: bool) -> String {
    let mut out = format!("== {name}\n");
    out += &format!(
        "{:<44} {:>22} {:>22} {:>8} {:>7}  ok\n",
        "statistic", "slinky (Python)", "slinky-env (Rust)", "|z|/D", "tol"
    );
    let summary = |s: &str| {
        s.starts_with("mean")
            || s.starts_with("spawn")
            || s.starts_with("food spawned")
            || s.starts_with("game length")
            || s.starts_with("length")
            || s.starts_with("first")
            || s.starts_with("distinct")
            || s.starts_with("truncated")
            || s.starts_with("snakes alive")
            || s.starts_with("elimination")
            || s.starts_with("wins")
            || s.starts_with("start food count")
    };
    let mut hidden = 0;
    for r in rows {
        let periodic = r.stat.contains(" at turn ")
            && !r.stat.ends_with(" 0")
            && !r.stat.ends_with(" 25")
            && !r.stat.ends_with(" 50")
            && !r.stat.ends_with(" 100");
        if all || !r.ok() || (summary(&r.stat) && !periodic) {
            out += &format!(
                "{:<44} {:>22} {:>22} {:>8.3} {:>7.3}  {}\n",
                r.stat,
                r.python,
                r.rust,
                r.score,
                r.tol,
                if r.ok() { "yes" } else { "NO" }
            );
        } else {
            hidden += 1;
        }
    }
    let worst = rows
        .iter()
        .filter(|r| r.tol > 0.0)
        .map(|r| r.score / r.tol)
        .fold(0.0, f64::max);
    let failed = rows.iter().filter(|r| !r.ok()).count();
    out += &format!(
        "({hidden} more rows within tolerance; {} rows, {failed} out of tolerance, worst score/tol {worst:.2})\n",
        rows.len()
    );
    out
}
