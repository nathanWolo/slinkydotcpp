//! Adversarial review of slinky-env against slinky's Python env (env.py, maps.py, modes.py).
//!
//! Part 1 needs no golden data: exact or statistical checks of what the Python code implies
//! (start positions and starting food in 4p and on other boards, spawns on full and nearly
//! full boards, the spawn probability for each chance, `minimum_food`, constrictor food, royale
//! shrink timing and side frequencies, game end, `max_turns`, rewards, determinism per seed, and
//! requests). Part 2 replays `$NAZ_GOLDEN/env_review` (written by `native/golden/review_env.py`):
//! exact transitions and whole-game distributions on configs the main golden set leaves out; it
//! skips when the files are missing.
//!
//! The tolerances are wide enough for a false alarm rate far below 1e-4 per assertion with the
//! fixed seeds used here.

#![allow(clippy::needless_range_loop)] // parallel per-seat arrays read more clearly by index

mod common;

use common::dist::{Dist, compare, env_of, table};
use serde_json::json;
use slinky_api::Request;
use slinky_env::{
    EnvConfig, EnvRandom, Game, MapKind, Rng, StepInfo, from_request, nth_set_bit, random_legal,
    random_moves, to_request_json,
};
use slinky_rules::{GameConfig, GameState, MAX_HEALTH, MAX_SNAKES, Mode, Ruleset, royale_hazards};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------
// Helpers.

/// Chi-square goodness of fit of `counts` against `probs` (summing to 1): the Wilson-Hilferty
/// z-score of the statistic (about N(0, 1) under the null hypothesis).
fn chi2_z(counts: &[u64], probs: &[f64]) -> f64 {
    assert_eq!(counts.len(), probs.len());
    let n: u64 = counts.iter().sum();
    let mut x = 0.0;
    let mut k = 0usize;
    for (&c, &p) in counts.iter().zip(probs) {
        if p == 0.0 {
            assert_eq!(c, 0, "an outcome of probability 0 happened");
            continue;
        }
        let e = p * n as f64;
        x += (c as f64 - e).powi(2) / e;
        k += 1;
    }
    let df = (k - 1).max(1) as f64;
    let t = 2.0 / (9.0 * df);
    ((x / df).cbrt() - (1.0 - t)) / t.sqrt()
}

fn assert_uniform(counts: &[u64], what: &str) {
    let p = vec![1.0 / counts.len() as f64; counts.len()];
    let z = chi2_z(counts, &p);
    assert!(z < 5.0, "{what}: not uniform (chi2 z = {z:.2}): {counts:?}");
}

/// |z| of a binomial count against probability `p`.
fn binom_z(count: u64, n: u64, p: f64) -> f64 {
    let sd = (n as f64 * p * (1.0 - p)).sqrt();
    if sd == 0.0 {
        return if count as f64 == n as f64 * p {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (count as f64 - n as f64 * p).abs() / sd
}

fn cells_of(mask: u128) -> Vec<usize> {
    (0..128).filter(|&c| mask >> c & 1 == 1).collect()
}

fn stacked(cfg: &GameConfig, s: &mut GameState, seat: usize, x: i32, y: i32) {
    s.set_snake(cfg, seat, &[(x, y); 3], MAX_HEALTH, None)
        .unwrap();
}

/// Play one random-legal game, calling `f(prev, info, next)` after every step.
fn play<F: FnMut(&GameState, &StepInfo, &GameState)>(env: &EnvConfig, seed: u64, mut f: F) {
    let mut g = Game::new(*env, seed);
    let mut guard = 0;
    while !g.state.done {
        let prev = g.state;
        let a = random_legal(&mut g.rng, env.rules(), &g.state);
        let info = g.step(&a);
        f(&prev, &info, &g.state);
        guard += 1;
        assert!(guard < 20_000, "game did not end");
    }
}

/// Answers every `below` with a fixed value (clamped below `n`), for exhaustive enumeration.
struct Fixed(u32);

impl EnvRandom for Fixed {
    fn next_u64(&mut self) -> u64 {
        unreachable!("Fixed overrides below")
    }
    fn below(&mut self, n: u32) -> u32 {
        self.0.min(n.saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------------------------
// Start positions (maps._place_fixed, _random_start_cells, place_snakes).

/// 4p on 11x11 (every 4-seat ladder mode): the seats take all 4 corners or all 4 cardinals, in
/// a uniformly random order, corners and cardinals 50/50: 48 equally likely joint starts.
#[test]
fn start_positions_4p_are_48_uniform_tuples() {
    for mode in [Mode::Standard, Mode::Royale, Mode::Constrictor] {
        let env = EnvConfig::ladder(mode).unwrap();
        let cfg = *env.rules();
        let corners = [(1, 1), (1, 9), (9, 1), (9, 9)];
        let cardinals = [(1, 5), (5, 1), (5, 9), (9, 5)];
        let mut joint: BTreeMap<Vec<(i32, i32)>, u64> = BTreeMap::new();
        let mut rng = Rng::new(0xC0FFEE + mode as u64);
        let games = 48 * 1500;
        for _ in 0..games {
            let s = env.reset(&mut rng);
            let heads: Vec<(i32, i32)> = (0..4).map(|i| (s.head[i][0], s.head[i][1])).collect();
            let mut sorted = heads.clone();
            sorted.sort();
            assert!(
                sorted == corners || sorted == cardinals,
                "{mode:?}: heads {heads:?} are not all corners or all cardinals"
            );
            for (i, &(x, y)) in heads.iter().enumerate() {
                assert_eq!(s.body_at(&cfg, i, x, y), 3);
                assert_eq!((s.length[i], s.health[i], s.alive[i]), (3, 100, true));
                assert_eq!(
                    s.last_move[i], 0,
                    "turn-0 last move is UP (types.empty_state)"
                );
            }
            assert!(!s.done && s.turn == 0);
            *joint.entry(heads).or_default() += 1;
        }
        assert_eq!(
            joint.len(),
            48,
            "{mode:?}: {} distinct joint starts",
            joint.len()
        );
        let counts: Vec<u64> = joint.values().copied().collect();
        assert_uniform(&counts, &format!("{mode:?} joint start"));
    }
}

/// A duel takes 2 of the same group: 2 x 12 ordered pairs, uniform.
#[test]
fn start_positions_duel_are_24_uniform_pairs() {
    let env = EnvConfig::ladder(Mode::Duel).unwrap();
    let mut joint: BTreeMap<[[i32; 2]; 2], u64> = BTreeMap::new();
    let mut rng = Rng::new(17);
    for _ in 0..24 * 2000 {
        let s = env.reset(&mut rng);
        *joint.entry([s.head[0], s.head[1]]).or_default() += 1;
    }
    assert_eq!(joint.len(), 24);
    assert_uniform(
        &joint.values().copied().collect::<Vec<_>>(),
        "duel joint start",
    );
}

/// Random placement (non-square boards and squares below 7): distinct cells of even `x + y`,
/// not the centre, each seat uniform over them, and on a 3x3 board with 4 seats a uniform
/// permutation of the only 4 such cells.
#[test]
fn random_placement_is_uniform_on_even_cells() {
    for (w, h, n) in [
        (9, 11, 3),
        (11, 7, 4),
        (6, 6, 4),
        (5, 5, 2),
        (1, 11, 2),
        (3, 3, 4),
    ] {
        let cfg = GameConfig::new(w, h, n, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(cfg, MapKind::Standard).unwrap();
        let valid = slinky_env::random_start_cells(&cfg);
        let cells = cells_of(valid);
        let centre = ((w as i32 - 1) / 2, (h as i32 - 1) / 2);
        for &c in &cells {
            let (x, y) = cfg.xy(c);
            assert_eq!((x + y) % 2, 0);
            assert_ne!((x, y), centre);
        }
        let mut per_seat = vec![vec![0u64; cells.len()]; n];
        let mut rng = Rng::new(w as u64 * 100 + h as u64);
        let games = 3000 * cells.len() as u64;
        for _ in 0..games {
            let s = env.reset(&mut rng);
            let mut seen = 0u128;
            for (i, seat) in per_seat.iter_mut().enumerate() {
                let c = s.head_cell(&cfg, i).expect("head on board");
                assert!(
                    valid >> c & 1 == 1,
                    "{w}x{h}: start {:?} not allowed",
                    cfg.xy(c)
                );
                assert!(seen >> c & 1 == 0, "{w}x{h}: two snakes start on one cell");
                seen |= 1 << c;
                seat[cells.iter().position(|&v| v == c).unwrap()] += 1;
            }
        }
        for (i, seat) in per_seat.iter().enumerate() {
            assert_uniform(seat, &format!("{w}x{h} seat {i} start cell"));
        }
    }
}

/// Fixed placement on every square board from 7 to 11: the start points of
/// `maps._fixed_start_points` (1, (w - 1) // 2, w - 2).
#[test]
fn fixed_placement_on_every_square_size() {
    for w in 7..=11usize {
        let cfg = GameConfig::new(w, w, 4, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(cfg, MapKind::Standard).unwrap();
        let (mn, md, mx) = (1, (w as i32 - 1) / 2, w as i32 - 2);
        let mut corners = vec![(mn, mn), (mn, mx), (mx, mn), (mx, mx)];
        let mut cardinals = vec![(mn, md), (md, mn), (md, mx), (mx, md)];
        corners.sort();
        cardinals.sort();
        let mut rng = Rng::new(w as u64);
        let (mut nc, games) = (0u64, 20_000u64);
        for _ in 0..games {
            let s = env.reset(&mut rng);
            let mut h: Vec<(i32, i32)> = (0..4).map(|i| (s.head[i][0], s.head[i][1])).collect();
            h.sort();
            if h == corners {
                nc += 1;
            } else {
                assert_eq!(h, cardinals, "{w}x{w}");
            }
        }
        assert!(
            binom_z(nc, games, 0.5) < 5.0,
            "{w}x{w}: corners {nc}/{games}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Starting food (maps.place_food_fixed, place_initial_food, env.init_state).

/// The diagonal food candidates of `_fixed_food_candidates`, written out independently.
fn py_candidates(w: i32, h: i32, (hx, hy): (i32, i32)) -> Vec<(i32, i32)> {
    let (cx, cy) = ((w - 1) / 2, (h - 1) / 2);
    let mut out = Vec::new();
    for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        let (px, py) = (hx + dx, hy + dy);
        let away = (px < hx && hx < cx)
            || (cx < hx && hx < px)
            || (py < hy && hy < cy)
            || (cy < hy && hy < py);
        let corner = (px == 0 || px == w - 1) && (py == 0 || py == h - 1);
        let on = (0..w).contains(&px) && (0..h).contains(&py);
        if away && !corner && (px, py) != (cx, cy) && on {
            out.push((px, py));
        }
    }
    out
}

/// On square boards of side 7 or more: the centre always has food, and each snake gets exactly
/// one food on one of its candidates, uniformly; nothing else. Under constrictor (either map)
/// and on the empty map there is no food at all.
#[test]
fn starting_food_fixed_is_one_uniform_candidate_per_snake_plus_centre() {
    for (w, n) in [(11usize, 4usize), (11, 2), (7, 4), (8, 4), (9, 3), (10, 4)] {
        let cfg = GameConfig::new(w, w, n, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(cfg, MapKind::Standard).unwrap();
        let wi = w as i32;
        let centre = cfg.cell((wi - 1) / 2, (wi - 1) / 2).unwrap();
        let mut per_head: HashMap<(i32, i32), HashMap<(i32, i32), u64>> = HashMap::new();
        let mut rng = Rng::new(31 * w as u64 + n as u64);
        for _ in 0..40_000 {
            let s = env.reset(&mut rng);
            assert!(s.has_food(centre), "{w}x{w}: no centre food");
            let mut expect = 1u128 << centre;
            for i in 0..n {
                let head = (s.head[i][0], s.head[i][1]);
                let cands = py_candidates(wi, wi, head);
                let here: Vec<(i32, i32)> = cands
                    .iter()
                    .copied()
                    .filter(|&(x, y)| s.food_at(&cfg, x, y))
                    .collect();
                assert_eq!(
                    here.len(),
                    1,
                    "{w}x{w} head {head:?}: food on {here:?} of {cands:?}"
                );
                expect |= 1 << cfg.cell(here[0].0, here[0].1).unwrap();
                *per_head
                    .entry(head)
                    .or_default()
                    .entry(here[0])
                    .or_default() += 1;
            }
            assert_eq!(
                s.food, expect,
                "{w}x{w}: food beyond the candidates and the centre"
            );
            assert_eq!(s.food.count_ones() as usize, n + 1);
        }
        for (head, m) in per_head {
            let k = py_candidates(wi, wi, head).len();
            assert_eq!(m.len(), k, "{w}x{w} head {head:?}: {m:?}");
            assert_uniform(
                &m.values().copied().collect::<Vec<_>>(),
                "starting food candidate",
            );
        }
    }
    let mut rng = Rng::new(5);
    let mut c2 = EnvConfig::for_mode(Mode::Constrictor, Some(2)).unwrap();
    for env in [
        EnvConfig::ladder(Mode::Constrictor).unwrap(),
        EnvConfig::for_mode(Mode::Constrictor, None).unwrap(),
        c2,
    ] {
        for _ in 0..500 {
            assert_eq!(env.reset(&mut rng).food, 0);
        }
    }
    // The empty map under the standard ruleset (not a ladder mode, but a valid env).
    c2 = EnvConfig::new(
        GameConfig::new(11, 11, 4, Ruleset::Standard).unwrap(),
        MapKind::Empty,
    )
    .unwrap();
    assert_eq!(c2.reset(&mut rng).food, 0);
}

/// Elsewhere: `min(N, cells)` food on distinct uniformly random spawn-allowed cells (no food, no
/// body, not next to a head), fewer only when the board has no more room.
#[test]
fn starting_food_random_boards() {
    for (w, h, n) in [
        (9, 11, 3),
        (11, 7, 4),
        (5, 5, 2),
        (6, 6, 4),
        (3, 3, 4),
        (1, 11, 2),
    ] {
        let cfg = GameConfig::new(w, h, n, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(cfg, MapKind::Standard).unwrap();
        let mut rng = Rng::new(7 + w as u64 * h as u64);
        // Per cell: food count, and its expectation and variance given each game's heads
        // (`want` of the `k` allowed cells, uniformly: probability want / k each).
        let mut hits = vec![0f64; cfg.cells()];
        let mut mean = vec![0f64; cfg.cells()];
        let mut var = vec![0f64; cfg.cells()];
        for _ in 0..20_000 {
            let s = env.reset(&mut rng);
            let mut bare = s;
            bare.food = 0;
            let allowed = env.spawn_mask(&bare);
            assert_eq!(
                s.food & !allowed,
                0,
                "{w}x{h}: starting food on a forbidden cell"
            );
            let want = n.min(allowed.count_ones() as usize);
            assert_eq!(s.food.count_ones() as usize, want, "{w}x{h}");
            let k = allowed.count_ones() as f64;
            for c in cells_of(allowed) {
                let p = want as f64 / k;
                mean[c] += p;
                var[c] += p * (1.0 - p);
            }
            for c in cells_of(s.food) {
                hits[c] += 1.0;
            }
        }
        for c in 0..cfg.cells() {
            let z = if var[c] > 0.0 {
                (hits[c] - mean[c]).abs() / var[c].sqrt()
            } else {
                (hits[c] - mean[c]).abs()
            };
            assert!(
                z < 5.0,
                "{w}x{h} cell {:?}: food {} vs expected {:.1}",
                cfg.xy(c),
                hits[c],
                mean[c]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Food spawns (maps.spawn_food_standard).

/// The engine spawns when `100 - rand.Intn(100) < chance`: exactly `chance - 1` of the 100 roll
/// outcomes, clamped to 0..=100, and only when the minimum is met.
#[test]
fn spawn_chance_counts_exactly_chance_minus_one_rolls() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    let mut s = GameState::new(&cfg);
    stacked(&cfg, &mut s, 0, 1, 1);
    stacked(&cfg, &mut s, 1, 9, 9);
    s.food = 1 << 60;
    for chance in [-5, 0, 1, 2, 15, 50, 99, 100, 101, 1000] {
        let env = EnvConfig::new(cfg, MapKind::Standard)
            .unwrap()
            .with_food(chance, 1);
        let mut spawned = 0;
        for v in 0..100 {
            let mut t = s;
            let k = env.spawn_food(&mut Fixed(v), &mut t);
            assert!(k <= 1);
            spawned += k;
        }
        assert_eq!(
            spawned as i32,
            (chance - 1).clamp(0, 100),
            "chance {chance}"
        );
    }
}

/// Below `minimum_food` the board is topped up to it (no roll), never past the free cells, and
/// the new food lands only on spawn-allowed cells.
#[test]
fn minimum_food_top_up_and_full_boards() {
    let cfg = GameConfig::new(5, 5, 2, Ruleset::Standard).unwrap();
    let mut s = GameState::new(&cfg);
    stacked(&cfg, &mut s, 0, 0, 0);
    stacked(&cfg, &mut s, 1, 4, 4);
    s.refresh_done(&cfg);
    let free = EnvConfig::new(cfg, MapKind::Standard)
        .unwrap()
        .spawn_mask(&s);
    assert_eq!(free.count_ones(), 19);
    let mut rng = Rng::new(1);
    for (min, have, chance) in [(3, 0, 0), (3, 2, 0), (19, 5, 0), (40, 0, 100), (40, 18, 0)] {
        let env = EnvConfig::new(cfg, MapKind::Standard)
            .unwrap()
            .with_food(chance, min);
        for _ in 0..200 {
            let mut t = s;
            for c in cells_of(free).into_iter().take(have) {
                t.food |= 1 << c;
            }
            let before = t.food;
            let k = env.spawn_food(&mut rng, &mut t);
            let want = (min - have as i32).clamp(0, 19 - have as i32);
            assert_eq!(k as i32, want, "min {min} have {have}");
            assert_eq!(t.food & before, before, "food was removed");
            assert_eq!((t.food & !before) & !free, 0, "food off the spawn mask");
            assert_eq!(t.food.count_ones(), before.count_ones() + k);
        }
    }
    // A board with no free cell: nothing spawns, whatever the settings, and nothing panics.
    let env = EnvConfig::new(cfg, MapKind::Standard)
        .unwrap()
        .with_food(101, 50);
    let mut t = s;
    t.food = free;
    assert_eq!(env.spawn_mask(&t), 0);
    for _ in 0..100 {
        let mut u = t;
        assert_eq!(env.spawn_food(&mut rng, &mut u), 0);
        assert_eq!(u.food, free);
    }
    // One free cell: the chance-100 spawn (99%) and the minimum both put food exactly there.
    let last = cells_of(free)[7];
    t.food = free & !(1 << last);
    let mut hits = 0;
    for _ in 0..10_000 {
        let mut u = t;
        let k = EnvConfig::new(cfg, MapKind::Standard)
            .unwrap()
            .with_food(100, 0)
            .spawn_food(&mut rng, &mut u);
        if k == 1 {
            assert_eq!(u.food, free);
            hits += 1;
        }
    }
    assert!(binom_z(hits, 10_000, 0.99) < 5.0, "{hits}");
}

/// Spawned cells are uniform over the spawn mask (the argmax of random scores in slinky).
#[test]
fn spawned_cell_is_uniform_over_the_mask() {
    let cfg = GameConfig::new(7, 7, 2, Ruleset::Standard).unwrap();
    let env = EnvConfig::new(cfg, MapKind::Standard)
        .unwrap()
        .with_food(0, 1);
    let mut s = GameState::new(&cfg);
    s.set_snake(&cfg, 0, &[(3, 3), (3, 2), (3, 1)], 90, None)
        .unwrap();
    stacked(&cfg, &mut s, 1, 6, 6);
    s.refresh_done(&cfg);
    let mask = env.spawn_mask(&s);
    let cells = cells_of(mask);
    let mut counts = vec![0u64; cells.len()];
    let mut rng = Rng::new(3);
    for _ in 0..2000 * cells.len() {
        let mut t = s;
        assert_eq!(env.spawn_food(&mut rng, &mut t), 1);
        let c = nth_set_bit(t.food, 0);
        counts[cells.iter().position(|&v| v == c).unwrap()] += 1;
    }
    assert_uniform(&counts, "spawned cell");
}

/// Whole games: after every step the board holds at least `minimum_food` food unless no
/// spawn-allowed cell is left, and at most `max(minimum_food, 1)` food appeared.
#[test]
fn minimum_food_holds_after_every_step() {
    let configs = [
        EnvConfig::ladder(Mode::Duel).unwrap().with_food(15, 4),
        EnvConfig::new(
            GameConfig::new(5, 5, 2, Ruleset::Standard).unwrap(),
            MapKind::Standard,
        )
        .unwrap()
        .with_food(0, 30),
        EnvConfig::new(
            GameConfig::new(7, 7, 4, Ruleset::Royale).unwrap(),
            MapKind::Royale,
        )
        .unwrap()
        .with_food(50, 25),
        EnvConfig::ladder(Mode::Standard).unwrap(),
    ];
    for (k, env) in configs.iter().enumerate() {
        for seed in 0..200 {
            play(env, seed * 10 + k as u64, |prev, info, next| {
                let min = env.minimum_food().max(0) as u32;
                if !prev.done && next.food.count_ones() < min {
                    assert_eq!(env.spawn_mask(next), 0, "config {k}: below the minimum");
                }
                assert!(info.spawned <= env.minimum_food().max(1) as u32);
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Constrictor food.

/// The ladder's constrictor (empty map) never has food. slinky's training constrictor (standard
/// map) spawns inert food after the rules removed it, so a state holds at most this turn's spawn.
#[test]
fn constrictor_food() {
    let ladder = EnvConfig::ladder(Mode::Constrictor).unwrap();
    let training = EnvConfig::for_mode(Mode::Constrictor, None).unwrap();
    let mut spawned_some = false;
    for seed in 0..300 {
        play(&ladder, seed, |_, info, next| {
            assert_eq!(next.food, 0);
            assert_eq!(info.spawned, 0);
        });
        play(&training, seed, |prev, info, next| {
            assert_eq!(
                next.food.count_ones(),
                info.spawned,
                "old food survived the rules"
            );
            assert!(info.spawned <= 1);
            spawned_some |= info.spawned > 0;
            for i in 0..4 {
                if next.alive[i] {
                    assert_eq!(next.health[i], MAX_HEALTH);
                    assert!(next.length[i] >= prev.length[i]);
                }
            }
        });
    }
    assert!(spawned_some);
}

// ---------------------------------------------------------------------------------------------
// Royale (RoyaleMap, WithRoyaleSides, rules.royale_hazards).

/// Every state of a royale game shows the zone of its own turn, from the sides drawn at reset
/// (hazards first at turn `shrink_every_n_turns`, never before), on the royale map and under
/// the royale ruleset on the standard and empty maps; the sides never change.
#[test]
fn royale_zone_matches_the_turn_in_every_state() {
    let base = GameConfig::new(11, 11, 4, Ruleset::Royale).unwrap();
    let small = GameConfig::new(7, 7, 3, Ruleset::Royale).unwrap();
    let configs = [
        EnvConfig::ladder(Mode::Royale).unwrap(),
        EnvConfig::new(base.with_shrink_every(3).unwrap(), MapKind::Royale).unwrap(),
        EnvConfig::new(base.with_shrink_every(7).unwrap(), MapKind::Standard).unwrap(),
        EnvConfig::new(base.with_shrink_every(5).unwrap(), MapKind::Empty).unwrap(),
        EnvConfig::new(small.with_shrink_every(1).unwrap(), MapKind::Royale).unwrap(),
        EnvConfig::for_mode(Mode::Royale, Some(2))
            .unwrap()
            .with_max_turns(Some(60)),
    ];
    for (k, env) in configs.iter().enumerate() {
        let cfg = *env.rules();
        let every = cfg.shrink_every_n_turns();
        for seed in 0..150 {
            let g = Game::new(*env, seed);
            assert_eq!(g.state.hazard, [0; slinky_rules::MAX_CELLS]);
            let sides = g.state.sides;
            assert!(sides[..cfg.num_sides()].iter().all(|v| (0..4).contains(v)));
            assert!(sides[cfg.num_sides()..].iter().all(|&v| v == 0));
            play(env, seed, |_, _, next| {
                assert_eq!(next.sides, sides, "config {k}: sides changed");
                assert_eq!(
                    next.hazard,
                    royale_hazards(&cfg, &sides, next.turn),
                    "config {k} seed {seed}: zone at turn {} is not the turn's zone",
                    next.turn
                );
                let any = next.hazard[..cfg.cells()].iter().any(|&h| h > 0);
                assert_eq!(
                    any,
                    next.turn >= every,
                    "config {k}: hazards at turn {}",
                    next.turn
                );
            });
        }
    }
}

/// Shrink sides are iid uniform over 0..4 at every position (`maps._random_sides`).
#[test]
fn royale_sides_are_iid_uniform() {
    let env = EnvConfig::ladder(Mode::Royale).unwrap();
    let ns = env.rules().num_sides();
    let mut pos = vec![vec![0u64; 4]; ns];
    let mut pairs = vec![0u64; 16];
    let mut rng = Rng::new(99);
    for _ in 0..40_000 {
        let s = env.reset(&mut rng);
        for k in 0..ns {
            pos[k][s.sides[k] as usize] += 1;
        }
        for k in 0..ns - 1 {
            pairs[(s.sides[k] * 4 + s.sides[k + 1]) as usize] += 1;
        }
    }
    for (k, p) in pos.iter().enumerate() {
        assert_uniform(p, &format!("side {k}"));
    }
    assert_uniform(&pairs, "adjacent side pairs");
    // Non-royale games keep the blank sides.
    let duel = EnvConfig::ladder(Mode::Duel).unwrap();
    assert_eq!(duel.reset(&mut rng).sides, [0; slinky_rules::MAX_SIDES]);
}

/// A state whose `done` flag was not refreshed although at most one snake is alive (e.g. built
/// by hand): the rules step is skipped, and on the royale map the zone is still regenerated for
/// the next turn (`RoyaleMap.post_update`), food still spawns, and the game ends with +1 to the
/// survivor. Under the royale ruleset on the standard map the zone stays (`WithRoyaleSides`).
#[test]
fn skipped_rules_step_in_royale() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Royale)
        .unwrap()
        .with_shrink_every(2)
        .unwrap();
    let mut s = GameState::new(&cfg);
    stacked(&cfg, &mut s, 0, 5, 5);
    s.turn = 5;
    s.sides[..21].copy_from_slice(&[
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
    ]);
    s.done = false;
    for map in [MapKind::Royale, MapKind::Standard] {
        let env = EnvConfig::new(cfg, map).unwrap().with_food(0, 2);
        let mut t = s;
        let info = env.step(&mut Rng::new(0), &mut t, &[1, 1, 0, 0]);
        assert!(info.done && !info.truncated);
        assert_eq!(t.turn, 6);
        assert_eq!(t.head[0], [5, 5], "a finished game's snakes do not move");
        assert_eq!(t.food.count_ones(), 2);
        assert_eq!(info.win_loss[..2], [1.0, 0.0]);
        let zone = if map == MapKind::Royale {
            royale_hazards(&cfg, &s.sides, 6)
        } else {
            s.hazard
        };
        assert_eq!(t.hazard, zone, "{map:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Game end, max_turns and rewards (env.step, win_loss_reward, placement_reward).

#[test]
fn game_end_truncation_and_rewards() {
    let configs = [
        EnvConfig::ladder(Mode::Duel).unwrap(),
        EnvConfig::ladder(Mode::Standard).unwrap(),
        EnvConfig::ladder(Mode::Royale).unwrap(),
        EnvConfig::ladder(Mode::Constrictor).unwrap(),
        EnvConfig::for_mode(Mode::Standard, Some(3))
            .unwrap()
            .with_max_turns(Some(40)),
        EnvConfig::ladder(Mode::Duel)
            .unwrap()
            .with_max_turns(Some(1)),
        EnvConfig::ladder(Mode::Royale)
            .unwrap()
            .with_max_turns(Some(0)),
    ];
    for (k, env) in configs.iter().enumerate() {
        let cfg = *env.rules();
        let n = cfg.num_snakes();
        let recip = 1.0 / (n - 1) as f64;
        for seed in 0..300 {
            let mut total = [0.0f64; MAX_SNAKES];
            let mut total_wl = [0.0f64; MAX_SNAKES];
            let mut truncs = 0;
            let mut last = StepInfo::default();
            let mut end = GameState::new(&cfg);
            play(env, seed, |prev, info, next| {
                assert_eq!(next.turn, prev.turn + 1);
                let alive = next.num_alive(&cfg);
                let by_turns = cfg.max_turns().is_some_and(|m| next.turn >= m);
                assert_eq!(info.done, next.done);
                assert_eq!(next.done, alive <= 1 || by_turns, "config {k}");
                assert_eq!(info.truncated, alive > 1 && by_turns, "config {k}");
                truncs += u32::from(info.truncated);
                let sum: f32 = info.placement[..n].iter().sum();
                assert!(
                    sum.abs() < 1e-6,
                    "placement not zero-sum: {:?}",
                    info.placement
                );
                for i in 0..MAX_SNAKES {
                    if i >= n {
                        assert_eq!((info.win_loss[i], info.placement[i]), (0.0, 0.0));
                        continue;
                    }
                    let died = prev.alive[i] && !next.alive[i];
                    if died {
                        assert_eq!(next.elim_turn[i], next.turn);
                        assert_ne!(next.elim_cause[i], 0);
                    }
                    if !next.alive[i] && !died {
                        assert_eq!(info.placement[i], 0.0, "a dead seat was paid");
                    }
                    total[i] += f64::from(info.placement[i]);
                    total_wl[i] += f64::from(info.win_loss[i]);
                }
                last = *info;
                end = *next;
            });
            assert!(truncs <= 1);
            assert!(last.done);
            // Finished games are frozen and pay nothing.
            let mut frozen = end;
            let info = env.step(&mut Rng::new(1), &mut frozen, &[0, 1, 2, 3]);
            assert_eq!(frozen, end);
            assert!(info.done && !info.truncated && info.spawned == 0);
            assert_eq!((info.win_loss, info.placement), ([0.0; 4], [0.0; 4]));
            // Placement totals: (outlasted - outlasted_by) / (N - 1) for every seat.
            for i in 0..n {
                let out = (0..n).filter(|&j| j != i && outlasted(&end, i, j)).count() as f64;
                let by = (0..n).filter(|&j| j != i && outlasted(&end, j, i)).count() as f64;
                assert!(
                    (total[i] - (out - by) * recip).abs() < 1e-5,
                    "config {k} seed {seed} seat {i}: placement total {} != {}",
                    total[i],
                    (out - by) * recip
                );
            }
            let alive = end.num_alive(&cfg);
            for i in 0..n {
                let want = if last.truncated {
                    if end.alive[i] { 0.0 } else { -1.0 }
                } else if alive == 1 && end.alive[i] {
                    1.0
                } else if alive == 0 && end.elim_turn[i] == end.turn {
                    0.0 // a draw: everyone eliminated on the last turn
                } else {
                    -1.0
                };
                assert_eq!(
                    total_wl[i], want,
                    "config {k} seed {seed} seat {i} win/loss"
                );
            }
        }
    }
}

/// Whether seat `i` outlasted seat `j` in a finished game.
fn outlasted(end: &GameState, i: usize, j: usize) -> bool {
    match (end.alive[i], end.alive[j]) {
        (true, false) => true,
        (false, false) => end.elim_turn[i] > end.elim_turn[j],
        _ => false,
    }
}

/// `max_turns = 1`: one step, then the game is over (truncated unless the rules ended it).
#[test]
fn max_turns_one() {
    let env = EnvConfig::ladder(Mode::Duel)
        .unwrap()
        .with_max_turns(Some(1));
    let mut rng = Rng::new(4);
    let mut s = env.reset(&mut rng);
    assert!(
        !s.done,
        "max_turns does not end a game at reset (env.init_state)"
    );
    let a = random_legal(&mut rng, env.rules(), &s);
    let info = env.step(&mut rng, &mut s, &a);
    assert!(info.done && info.truncated && s.done && s.turn == 1);
}

// ---------------------------------------------------------------------------------------------
// Determinism per seed.

fn trajectory(env: &EnvConfig, seed: u64) -> Vec<GameState> {
    let mut out = Vec::new();
    let mut g = Game::new(*env, seed);
    out.push(g.state);
    while !g.state.done {
        let a = random_legal(&mut g.rng, env.rules(), &g.state);
        g.step(&a);
        out.push(g.state);
    }
    out
}

#[test]
fn games_are_fixed_by_their_seed() {
    let mut envs: Vec<EnvConfig> = Mode::ALL
        .iter()
        .map(|&m| EnvConfig::ladder(m).unwrap())
        .collect();
    envs.push(EnvConfig::for_mode(Mode::Constrictor, None).unwrap());
    envs.push(
        EnvConfig::new(
            GameConfig::new(9, 11, 3, Ruleset::Standard).unwrap(),
            MapKind::Standard,
        )
        .unwrap(),
    );
    for env in &envs {
        let mut distinct = std::collections::HashSet::new();
        for seed in [0u64, 1, 2, 3, u64::MAX, 0x9E37_79B9_7F4A_7C15] {
            let a = trajectory(env, seed);
            let b = trajectory(env, seed);
            assert_eq!(a, b, "seed {seed}: two runs differ");
            distinct.insert(format!("{:?}", a.last().unwrap()));
        }
        assert!(
            distinct.len() >= 5,
            "{:?}: seeds give the same games",
            env.mode()
        );
        // step_autoreset over many games: the same stream gives the same sequence.
        let run = |seed| {
            let mut rng = Rng::with_stream(seed, 3);
            let mut s = env.reset(&mut rng);
            let mut finals = Vec::new();
            for _ in 0..3000 {
                let a = random_moves(&mut rng, env.rules());
                if let (_, Some(f)) = env.step_autoreset(&mut rng, &mut s, &a) {
                    finals.push(f);
                }
            }
            (finals, s)
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }
}

/// Copies of a generator replay the same choices; parallel streams are not shifted copies of
/// each other (the first outputs of 64 streams are all distinct, and no stream's first 8 outputs
/// appear in another's first 64).
#[test]
fn streams_are_independent() {
    let mut firsts = std::collections::HashSet::new();
    let mut all: HashMap<u64, u64> = HashMap::new();
    for stream in 0..64u64 {
        let mut r = Rng::with_stream(42, stream);
        let mut c = r;
        assert_eq!(r.next_u64(), c.next_u64());
        let mut r = Rng::with_stream(42, stream);
        for k in 0..64 {
            let v = r.next_u64();
            if k == 0 {
                assert!(firsts.insert(v));
            }
            if let Some(&other) = all.get(&v) {
                assert_eq!(
                    other, stream,
                    "streams {other} and {stream} share an output"
                );
            }
            all.insert(v, stream);
        }
    }
    // Seeds one SplitMix64 increment apart (Rng::new feeds seed + k * gamma to SplitMix64).
    let gamma = 0x9E37_79B9_7F4A_7C15u64;
    let mut a = Rng::new(1);
    let mut b = Rng::new(1u64.wrapping_add(gamma));
    let xs: Vec<u64> = (0..64).map(|_| a.next_u64()).collect();
    let ys: Vec<u64> = (0..64).map(|_| b.next_u64()).collect();
    assert!(xs.iter().all(|x| !ys.contains(x)));
}

// ---------------------------------------------------------------------------------------------
// Requests.

fn snake_json(id: &str, body: &[(i32, i32)]) -> serde_json::Value {
    let pts: Vec<_> = body.iter().map(|&(x, y)| json!({"x": x, "y": y})).collect();
    json!({"id": id, "name": id, "health": 90, "body": pts, "head": pts[0], "length": body.len()})
}

/// A request on a board where the env could not *reset* (random placement needs one even cell
/// per seat) is still a valid position to continue from: `from_request` should accept it as the
/// API crate does, since nothing is placed. (Fixed: it builds the env with
/// `EnvConfig::for_position`, which skips the reset-time placement check.)
#[test]
fn from_request_accepts_small_boards_it_could_not_reset() {
    // 3x2 board, 3 snakes: even-parity cells other than the centre (1, 0) are (0, 0), (2, 0)
    // and (1, 1): 3 cells, enough. 2x2 with 3 snakes: (1, 1) only.
    let a = [(0, 0), (0, 1)];
    let b = [(1, 0)];
    let c = [(1, 1)];
    let body = json!({
        "game": {"id": "g", "ruleset": {"name": "standard"}, "map": "standard"},
        "turn": 3,
        "board": {"width": 2, "height": 2, "food": [], "hazards": [],
                  "snakes": [snake_json("a", &a), snake_json("b", &b), snake_json("c", &c)]},
        "you": snake_json("a", &a),
    });
    let req = Request::from_value(&body).unwrap();
    assert!(slinky_api::game_from_request(&req, None).is_ok());
    let r = from_request(&req, None, &mut Rng::new(0));
    assert!(r.is_ok(), "{r:?}");
}

/// `to_request_json` then `from_request` keeps the environment for every env the crate builds.
/// (Fixed: the map comes from the request's map name, and the `empty` map is read under every
/// ruleset through `slinky_api::game_from_request_any_map`.)
#[test]
fn request_round_trip_keeps_the_env() {
    let base = GameConfig::new(11, 11, 4, Ruleset::Royale).unwrap();
    for map in [MapKind::Standard, MapKind::Empty] {
        let env = EnvConfig::new(base, map).unwrap();
        let s = env.reset(&mut Rng::new(1));
        let body = to_request_json(&env, &s, 0, "g").unwrap();
        let req = Request::from_value(&body).unwrap();
        let back = from_request(&req, Some(4), &mut Rng::new(2));
        assert_eq!(back.map(|(e, _)| e), Ok(env), "{map:?}");
    }
}

/// `from_request` keeps the shown zone and draws only the shrinks not shown yet: at every turn
/// of royale games, the request's position shows the true zone, and the next turn's zone equals
/// the true one except on a shrink turn, where it differs by the one new (random) shrink.
#[test]
fn from_request_keeps_the_shown_zone() {
    let env = EnvConfig::ladder(Mode::Royale).unwrap().with_food(15, 1);
    let fast = EnvConfig::new(env.rules().with_shrink_every(2).unwrap(), MapKind::Royale).unwrap();
    let mut checked = 0;
    for e in [env, fast] {
        let cfg = *e.rules();
        let every = cfg.shrink_every_n_turns();
        for seed in 0..60 {
            let mut g = Game::new(e, seed);
            while !g.state.done {
                let you = (0..4).find(|&i| g.state.alive[i]).unwrap();
                let body = to_request_json(&e, &g.state, you, "g").unwrap();
                let req = Request::from_value(&body).unwrap();
                let (e2, t) = from_request(&req, Some(4), &mut Rng::new(seed)).unwrap();
                assert_eq!(e2, e);
                assert_eq!(t.hazard, g.state.hazard);
                assert_eq!(royale_hazards(&cfg, &t.sides, t.turn), g.state.hazard);
                let next = t.turn + 1;
                let mine = royale_hazards(&cfg, &t.sides, next);
                let truth = royale_hazards(&cfg, &g.state.sides, next);
                let cells = |z: &[i16; slinky_rules::MAX_CELLS]| -> i32 {
                    z.iter().map(|&v| i32::from(v)).sum()
                };
                if next % every != 0 {
                    assert_eq!(mine, truth, "turn {}", t.turn);
                } else {
                    // One new shrink each: the same number of new hazard cells up to the side
                    // (a row or a column of the current safe box).
                    let now = cells(&g.state.hazard);
                    assert!(cells(&mine) >= now && cells(&truth) >= now);
                }
                checked += 1;
                let a = random_legal(&mut g.rng, e.rules(), &g.state);
                g.step(&a);
            }
        }
    }
    assert!(checked > 1000);
}

// ---------------------------------------------------------------------------------------------
// Part 2: review golden data (native/golden/review_env.py).

fn review_root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("env_review")
}

#[test]
fn review_distributions_match_slinky() {
    let dir = review_root().join("dist");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("SKIP: no env review distributions at {}", dir.display());
        return;
    };
    let games: u64 = std::env::var("NAZ_ENV_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let mut failed = Vec::new();
    for (k, path) in files.iter().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let env = env_of(&v);
        let rows = compare(
            &Dist::from_json(&v),
            &Dist::simulate(&env, games, 5000 + k as u64),
            env.rules(),
        );
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        eprintln!("{}", table(&name, &rows, false));
        if rows.iter().any(|r| !r.ok()) {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "out of tolerance: {failed:?}");
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u128(&mut self) -> u128 {
        u128::from_le_bytes(self.bytes(16).try_into().unwrap())
    }
    fn state(&mut self, cfg: &GameConfig) -> GameState {
        let (n, cells) = (cfg.num_snakes(), cfg.cells());
        let mut s = GameState::new(cfg);
        s.turn = self.i32();
        s.done = self.i32() != 0;
        for i in 0..n {
            s.head[i] = [self.i32(), self.i32()];
            s.length[i] = self.i32();
            s.health[i] = self.i32();
            s.alive[i] = self.i32() != 0;
            s.last_move[i] = self.i32() as u8;
            s.elim_cause[i] = self.i32() as i8;
            s.elim_turn[i] = self.i32();
        }
        for k in 0..cfg.num_sides() {
            s.sides[k] = self.bytes(1)[0] as i8;
        }
        for c in 0..cells {
            if self.bytes(1)[0] != 0 {
                s.food |= 1u128 << c;
            }
        }
        for c in 0..cells {
            s.hazard[c] = i16::from(self.bytes(1)[0]);
        }
        for i in 0..n {
            for c in 0..cells {
                s.body[i][c] = i16::from(self.bytes(1)[0]);
            }
        }
        s
    }
}

/// Hands out slinky's roll and its spawned cells (lowest first) and counts the picks.
struct Scripted {
    roll: i32,
    cells: u128,
    extra: bool,
    outside: bool,
}

impl EnvRandom for Scripted {
    fn next_u64(&mut self) -> u64 {
        panic!("the step drew randomness outside the roll and the cell picks");
    }
    fn food_roll(&mut self) -> i32 {
        self.roll
    }
    fn pick_cell(&mut self, mask: u128) -> Option<usize> {
        if self.cells == 0 {
            self.extra |= mask != 0;
            return None;
        }
        let c = nth_set_bit(self.cells, 0);
        self.cells &= !(1u128 << c);
        self.outside |= mask >> c & 1 == 0;
        Some(c)
    }
}

fn meta_env(dir: &Path) -> EnvConfig {
    let meta: HashMap<String, String> = std::fs::read_to_string(dir.join("meta.txt"))
        .unwrap()
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect();
    let num = |k: &str| -> i32 { meta[k].parse().unwrap() };
    let max_turns = num("max_turns");
    let rules = GameConfig::new(
        num("width") as usize,
        num("height") as usize,
        num("num_snakes") as usize,
        Ruleset::from_name(&meta["ruleset"]).unwrap(),
    )
    .unwrap()
    .with_hazard_damage(num("hazard_damage"))
    .with_shrink_every(num("shrink_every"))
    .unwrap()
    .with_max_turns((max_turns >= 0).then_some(max_turns));
    EnvConfig::new(rules, MapKind::from_name(&meta["map"]).unwrap())
        .unwrap()
        .with_food(num("food_spawn_chance"), num("minimum_food"))
}

#[test]
fn review_transitions_match_slinky() {
    let root = review_root().join("trans");
    let Ok(entries) = std::fs::read_dir(&root) else {
        eprintln!("SKIP: no env review transitions at {}", root.display());
        return;
    };
    let mut dirs: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
    dirs.sort();
    let mut problems = Vec::new();
    for dir in dirs.iter().filter(|d| d.join("trans.bin").exists()) {
        let env = meta_env(dir);
        let cfg = *env.rules();
        let n = cfg.num_snakes();
        let buf = std::fs::read(dir.join("trans.bin")).unwrap();
        let mut r = Reader { buf: &buf, pos: 0 };
        let (mut count, mut bad) = (0, 0);
        while r.pos < buf.len() {
            let pre = r.state(&cfg);
            let mut actions = [0u8; MAX_SNAKES];
            actions.copy_from_slice(r.bytes(4));
            let roll = r.i32();
            let mask = r.u128();
            let spawned = r.u128();
            let truncated = r.bytes(1)[0] != 0;
            let wl: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            let pl: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            let next = r.state(&cfg);
            let mut rng = Scripted {
                roll,
                cells: spawned,
                extra: false,
                outside: false,
            };
            let mut s = pre;
            let info = env.step(&mut rng, &mut s, &actions);
            let mut why = Vec::new();
            if s != next {
                why.push(format!("next state differs: rust {s:?} python {next:?}"));
            }
            if rng.cells != 0 || rng.extra || rng.outside {
                why.push(format!(
                    "spawn count or cells differ (left {:#x}, extra {}, outside {})",
                    rng.cells, rng.extra, rng.outside
                ));
            }
            if !pre.done && env.spawns_food() {
                let mut post = pre;
                post.rules_step(&cfg, &actions);
                if env.spawn_mask(&post) != mask {
                    why.push("spawn mask differs".into());
                }
            }
            if info.truncated != truncated || info.done != next.done {
                why.push("truncated/done differ".into());
            }
            for i in 0..n {
                if info.win_loss[i].to_bits() != wl[i].to_bits()
                    || info.placement[i].to_bits() != pl[i].to_bits()
                {
                    why.push(format!(
                        "rewards {:?}/{:?} != {wl:?}/{pl:?}",
                        &info.win_loss[..n],
                        &info.placement[..n]
                    ));
                    break;
                }
            }
            if !why.is_empty() {
                bad += 1;
                if problems.len() < 10 {
                    problems.push(format!("{} #{count}: {}", dir.display(), why.join("; ")));
                }
            }
            count += 1;
        }
        eprintln!(
            "env review {}: {count} transitions, {bad} mismatches",
            dir.file_name().unwrap().to_string_lossy()
        );
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

// ---------------------------------------------------------------------------------------------
// No panics on network input: random (often malformed) requests through from_request, then
// steps and to_request_json. Run it in a debug build too, where integer overflow panics.

fn fuzz_request(rng: &mut Rng) -> serde_json::Value {
    let mut pick = |n: u32| rng.below(n) as i64;
    let w = 1 + pick(11);
    let h = 1 + pick(11);
    let coord = |p: &mut dyn FnMut(u32) -> i64, lim: i64| -> i64 {
        match p(10) {
            0 => -1 - p(3),
            1 => lim + p(3),
            _ => p(lim.max(1) as u32),
        }
    };
    let n = 1 + pick(4);
    let mut snakes = Vec::new();
    for i in 0..n {
        let len = pick(8);
        let mut body = Vec::new();
        let (mut x, mut y) = (coord(&mut pick, w), coord(&mut pick, h));
        for _ in 0..len {
            body.push(json!({"x": x, "y": y}));
            match pick(5) {
                0 => x += 1,
                1 => x -= 1,
                2 => y += 1,
                3 => y -= 1,
                _ => {}
            }
        }
        let health = match pick(6) {
            0 => i64::from(i32::MAX),
            1 => i64::from(i32::MIN),
            2 => -pick(5),
            _ => pick(130),
        };
        snakes.push(json!({"id": format!("s{i}"), "health": health, "body": body}));
    }
    let you = snakes[pick(snakes.len() as u32) as usize].clone();
    let mut food = Vec::new();
    for _ in 0..pick((w * h) as u32 + 1) {
        food.push(json!({"x": coord(&mut pick, w), "y": coord(&mut pick, h)}));
    }
    let mut hazards = Vec::new();
    for _ in 0..pick((w * h) as u32 + 1) {
        hazards.push(json!({"x": coord(&mut pick, w), "y": coord(&mut pick, h)}));
    }
    let names = ["standard", "royale", "constrictor"];
    let maps = ["standard", "royale", "empty"];
    let turn = match pick(5) {
        0 => i64::from(i32::MAX),
        1 => -pick(30),
        _ => pick(400),
    };
    json!({
        "game": {"id": "fuzz", "map": maps[pick(3) as usize],
                 "ruleset": {"name": names[pick(3) as usize], "settings": {
                     "foodSpawnChance": pick(250) - 50,
                     "minimumFood": match pick(4) { 0 => i64::from(i32::MAX), 1 => -pick(9), _ => pick(150) },
                     "hazardDamagePerTurn": match pick(4) { 0 => i64::from(i32::MIN), 1 => i64::from(i32::MAX), _ => pick(120) - 20 },
                     "royale": {"shrinkEveryNTurns": match pick(4) { 0 => i64::from(i32::MAX), _ => 1 + pick(30) }}}}},
        "turn": turn,
        "board": {"width": w, "height": h, "food": food, "hazards": hazards, "snakes": snakes},
        "you": you,
    })
}

#[test]
fn malformed_requests_never_panic() {
    let mut rng = Rng::new(2024);
    let mut accepted = 0;
    for _ in 0..30_000 {
        let body = fuzz_request(&mut rng);
        let Ok(req) = Request::from_value(&body) else {
            continue;
        };
        let starting = match rng.below(3) {
            0 => None,
            k => Some(k as usize + 1),
        };
        let Ok((env, mut s)) = from_request(&req, starting, &mut rng) else {
            continue;
        };
        accepted += 1;
        for _ in 0..30 {
            if s.done {
                break;
            }
            let a = if rng.below(2) == 0 {
                random_legal(&mut rng, env.rules(), &s)
            } else {
                [rng.below(9) as u8, rng.below(9) as u8, 200, 255]
            };
            env.step(&mut rng, &mut s, &a);
            if let Some(you) = (0..env.rules().num_snakes()).find(|&i| s.alive[i]) {
                let out = to_request_json(&env, &s, you, "fuzz").unwrap();
                let back = Request::from_value(&out).unwrap();
                let _ = from_request(&back, Some(env.rules().num_snakes()), &mut rng);
            }
        }
    }
    assert!(accepted > 1000, "only {accepted} requests accepted");
}

/// A scripted [`EnvRandom`] (the trait invites overriding `below` to script choices) that breaks
/// the `0..n` contract must not corrupt the state: no food outside the spawn mask, no panic.
/// (Fixed: every choice goes through range-checked helpers.) It used to be that `pick_cell` turned `below(n) == n` into cell 128 (`nth_set_bit`), and `spawn_food` then
/// shifts `1u128 << 128`: a panic with overflow checks, food on cell 0 (under a snake here)
/// without them. `place_initial_food` indexes `allowed[below(k)]`, which panics at 4 and picks
/// the stale cell 0 for k..4.
#[test]
fn out_of_range_choices_do_not_corrupt_the_state() {
    struct OutOfRange;
    impl EnvRandom for OutOfRange {
        fn next_u64(&mut self) -> u64 {
            0
        }
        fn below(&mut self, n: u32) -> u32 {
            n
        }
    }
    let cfg = GameConfig::new(5, 5, 2, Ruleset::Standard).unwrap();
    let env = EnvConfig::new(cfg, MapKind::Standard)
        .unwrap()
        .with_food(0, 1);
    let mut s = GameState::new(&cfg);
    stacked(&cfg, &mut s, 0, 0, 0); // a snake on cell 0
    stacked(&cfg, &mut s, 1, 4, 4);
    s.refresh_done(&cfg);
    let mask = env.spawn_mask(&s);
    let r = std::panic::catch_unwind(move || {
        let mut t = s;
        env.spawn_food(&mut OutOfRange, &mut t);
        t
    });
    let t = r.expect("spawn_food panicked");
    assert_eq!(
        t.food & !mask,
        0,
        "food outside the spawn mask: {:#x}",
        t.food
    );
    let e = EnvConfig::ladder(Mode::Duel).unwrap();
    let r = std::panic::catch_unwind(move || e.reset(&mut OutOfRange));
    assert!(r.is_ok(), "reset panicked");
}
