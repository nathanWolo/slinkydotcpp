//! Adversarial parity review of the rules against slinky's jitted Python, on the hand-built and
//! fuzzed edge cases written by `native/golden/review_rules.py` (see its docstring for the file
//! format): 3- and 4-way head-ons on food, moves into tails that just stacked, necks, hazard
//! damage killing exactly at 0, healing hazards, eliminations on the same turn, last-two and
//! pile-up draws, royale shrink timing (shrink every 1 to 25 turns, boundary turns, weird side
//! values, turns that wrap), constrictor growth and starvation, malformed and extreme field
//! values, and the placement values for many draw values with ties.
//!
//! Every test skips with a message when the golden files are missing, so `cargo test` passes
//! without them. Plain unit tests at the bottom need no golden data.

use slinky_rules::{
    Cause, GameConfig, GameState, MAX_CELLS, MAX_SIDES, MAX_SNAKES, Move, MoveMask, NUM_ACTIONS,
    Ruleset, royale_hazards,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("review_rules")
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.bytes(1)[0]
    }
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.bytes(2).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
}

struct Meta {
    cfg: GameConfig,
    draws: Vec<f64>,
}

fn read_meta(dir: &Path) -> Meta {
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let kv: HashMap<&str, &str> = meta.lines().filter_map(|l| l.split_once(' ')).collect();
    let get = |k: &str| kv[k].parse::<i64>().unwrap();
    let cfg = GameConfig::new(
        get("width") as usize,
        get("height") as usize,
        get("num_snakes") as usize,
        Ruleset::from_name(kv["ruleset"]).unwrap(),
    )
    .unwrap()
    .with_hazard_damage(get("hazard_damage") as i32)
    .with_shrink_every(get("shrink_every") as i32)
    .unwrap();
    let draws = kv["draws"]
        .split(',')
        .map(|v| v.parse::<f64>().unwrap())
        .collect();
    Meta { cfg, draws }
}

fn read_state(r: &mut Reader, cfg: &GameConfig) -> GameState {
    let n = cfg.num_snakes();
    let cells = cfg.cells();
    let mut s = GameState::new(cfg);
    s.turn = r.i32();
    s.done = r.i32() != 0;
    for i in 0..n {
        s.head[i] = [r.i32(), r.i32()];
        s.length[i] = r.i32();
        s.health[i] = r.i32();
        s.alive[i] = r.i32() != 0;
        s.last_move[i] = r.i32() as u8;
        s.elim_cause[i] = r.i32() as i8;
        s.elim_turn[i] = r.i32();
    }
    for k in 0..cfg.num_sides() {
        s.sides[k] = r.u8() as i8;
    }
    for c in 0..cells {
        if r.u8() != 0 {
            s.food |= 1u128 << c;
        }
    }
    for c in 0..cells {
        s.hazard[c] = r.i16();
    }
    for i in 0..n {
        for c in 0..cells {
            s.body[i][c] = r.i16();
        }
    }
    s
}

#[derive(Debug, PartialEq)]
struct Derived {
    mask: MoveMask,
    legal: MoveMask,
    starving: [bool; 4],
    over: bool,
    blocked: u128,
    /// Placement values per draw value, compared bitwise.
    pv: Vec<[u32; MAX_SNAKES]>,
}

fn unbits(b: u8) -> [bool; NUM_ACTIONS] {
    [b & 1 != 0, b & 2 != 0, b & 4 != 0, b & 8 != 0]
}

fn read_derived(r: &mut Reader, n: usize, draws: usize) -> Derived {
    let mut d = Derived {
        mask: [[false; 4]; 4],
        legal: [[false; 4]; 4],
        starving: [false; 4],
        over: false,
        blocked: 0,
        pv: Vec::new(),
    };
    for i in 0..n {
        d.mask[i] = unbits(r.u8());
    }
    for i in 0..n {
        d.legal[i] = unbits(r.u8());
    }
    d.starving = unbits(r.u8());
    d.over = r.u8() != 0;
    d.blocked = u128::from_le_bytes(r.bytes(16).try_into().unwrap());
    for _ in 0..draws {
        let mut v = [0u32; MAX_SNAKES];
        for x in v.iter_mut().take(n) {
            *x = r.f32().to_bits();
        }
        d.pv.push(v);
    }
    d
}

fn derive(s: &GameState, m: &Meta) -> Derived {
    let cfg = &m.cfg;
    Derived {
        mask: s.action_mask(cfg),
        legal: s.legal_moves(cfg),
        starving: s.certainly_starving(cfg),
        over: s.is_game_over(cfg),
        blocked: s.blocked_cells(cfg),
        pv: m
            .draws
            .iter()
            .map(|&d| s.placement_values(cfg, d as f32).map(f32::to_bits))
            .collect(),
    }
}

fn describe(a: &GameState, b: &GameState, cfg: &GameConfig) -> String {
    let n = cfg.num_snakes();
    let mut out = Vec::new();
    macro_rules! cmp {
        ($f:ident) => {
            if a.$f[..n] != b.$f[..n] {
                out.push(format!(
                    "{}: rust {:?} python {:?}",
                    stringify!($f),
                    &a.$f[..n],
                    &b.$f[..n]
                ));
            }
        };
    }
    cmp!(head);
    cmp!(length);
    cmp!(health);
    cmp!(alive);
    cmp!(last_move);
    cmp!(elim_cause);
    cmp!(elim_turn);
    for i in 0..n {
        if a.body[i] != b.body[i] {
            out.push(format!("body[{i}] differs"));
        }
    }
    if a.food != b.food {
        out.push("food differs".into());
    }
    if a.hazard != b.hazard {
        out.push("hazard differs".into());
    }
    if (a.turn, a.done) != (b.turn, b.done) {
        out.push(format!(
            "turn/done: rust {:?} python {:?}",
            (a.turn, a.done),
            (b.turn, b.done)
        ));
    }
    out.join("; ")
}

/// Whether the placement values for `draw` with `n` seats are compared. The draw term used to be
/// rounded as `draw * tied * (1/3)`, one ulp off XLA's `tied * (draw * (1/3))` for 4 seats and
/// draws such as -0.45, so the per-group tests compared 4-seat values only for the deployed
/// -0.5; now fixed, `review_all_draws_bitwise` compares every draw.
fn draw_checked(n: usize, draw: f64, all: bool) -> bool {
    all || n < 4 || draw == -0.5
}

/// Compare one group directory; returns (states, transitions, mismatch descriptions).
fn check_group(dir: &Path, all_draws: bool) -> (usize, usize, Vec<String>) {
    let m = read_meta(dir);
    let cfg = m.cfg;
    let n = cfg.num_snakes();
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let sbuf = std::fs::read(dir.join("states.bin")).unwrap();
    let dbuf = std::fs::read(dir.join("derived.bin")).unwrap();
    let tbuf = std::fs::read(dir.join("trans.bin")).unwrap();
    let labels: HashMap<usize, String> = std::fs::read_to_string(dir.join("labels.txt"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(i, l)| (i.parse().unwrap(), l.to_string()))
        .collect();
    let label = |i: usize| labels.get(&i).cloned().unwrap_or_default();
    let (mut sr, mut dr) = (Reader::new(&sbuf), Reader::new(&dbuf));
    let mut states = Vec::new();
    let mut bad = Vec::new();
    while !sr.done() {
        let s = read_state(&mut sr, &cfg);
        let mut want = read_derived(&mut dr, n, m.draws.len());
        let mut got = derive(&s, &m);
        for (k, &d) in m.draws.iter().enumerate() {
            if !draw_checked(n, d, all_draws) {
                (got.pv[k], want.pv[k]) = ([0; MAX_SNAKES], [0; MAX_SNAKES]);
            }
        }
        if got != want {
            let mut what = Vec::new();
            if got.mask != want.mask {
                what.push(format!("mask rust {:?} python {:?}", got.mask, want.mask));
            }
            if got.legal != want.legal {
                what.push(format!(
                    "legal rust {:?} python {:?}",
                    got.legal, want.legal
                ));
            }
            if got.starving != want.starving {
                what.push(format!(
                    "starving rust {:?} python {:?}",
                    got.starving, want.starving
                ));
            }
            if got.over != want.over {
                what.push("game over".into());
            }
            if got.blocked != want.blocked {
                what.push("blocked".into());
            }
            for (k, (g, w)) in got.pv.iter().zip(&want.pv).enumerate() {
                if g != w {
                    let f = |v: &[u32; 4]| {
                        v[..n]
                            .iter()
                            .map(|&b| f32::from_bits(b))
                            .collect::<Vec<_>>()
                    };
                    what.push(format!(
                        "placement(draw {}) rust {:?} python {:?} (alive {:?} elim_turn {:?})",
                        m.draws[k],
                        f(g),
                        f(w),
                        &s.alive[..n],
                        &s.elim_turn[..n]
                    ));
                }
            }
            bad.push(format!(
                "{name}: state {} [{}] derived: {}",
                states.len(),
                label(states.len()),
                what.join("; ")
            ));
        }
        states.push(s);
    }
    assert!(dr.done(), "{name}: derived.bin longer than states.bin");
    let mut tr = Reader::new(&tbuf);
    let mut count = 0;
    while !tr.done() {
        let (p, c) = (tr.u32() as usize, tr.u32() as usize);
        let actions: [u8; 4] = tr.bytes(4).try_into().unwrap();
        let got = states[p].child(&cfg, &actions);
        if got != states[c] {
            bad.push(format!(
                "{name}: transition {count} (state {p} [{}] -> {c}, actions {actions:?}): {}",
                label(p),
                describe(&got, &states[c], &cfg)
            ));
        }
        count += 1;
    }
    (states.len(), count, bad)
}

fn group_dirs() -> Option<Vec<PathBuf>> {
    let rd = std::fs::read_dir(root()).ok()?;
    let mut dirs: Vec<_> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("meta.txt").exists())
        .collect();
    dirs.sort();
    Some(dirs)
}

fn run_groups(all_draws: bool, filter: impl Fn(&str) -> bool) {
    let Some(dirs) = group_dirs() else {
        eprintln!(
            "SKIP review rules: {} missing (run native/golden/review_rules.py)",
            root().display()
        );
        return;
    };
    let mut all_bad = Vec::new();
    let mut checked = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        if !filter(&name) {
            continue;
        }
        let (s, t, bad) = check_group(&dir, all_draws);
        eprintln!(
            "{name}: {s} states, {t} transitions, {} mismatches",
            bad.len()
        );
        checked += 1;
        all_bad.extend(bad);
    }
    if checked == 0 {
        eprintln!(
            "SKIP review rules: no matching groups in {}",
            root().display()
        );
        return;
    }
    for b in all_bad.iter().take(20) {
        eprintln!("{b}");
    }
    assert!(all_bad.is_empty(), "{} mismatches", all_bad.len());
}

#[test]
fn review_headon_stacked_neck_lasttwo() {
    run_groups(false, |n| {
        n.starts_with("headon")
            || n.starts_with("stacked")
            || n.starts_with("neck")
            || n.starts_with("lasttwo")
    });
}

#[test]
fn review_hazards() {
    run_groups(false, |n| n.starts_with("hazard"));
}

#[test]
fn review_crowds_and_constrictor() {
    run_groups(false, |n| {
        n.starts_with("crowd") || n.starts_with("constrictor")
    });
}

#[test]
fn review_royale() {
    run_groups(false, |n| n.starts_with("royale"));
}

#[test]
fn review_placement_values() {
    check_placement(false);
}

/// Fixed bug: `placement_values` added the draw term as `(draw * tied) * (1/(N-1))`; the jitted
/// Python (`draw_value * tied / (n - 1)`, mcts.py:562) is compiled by XLA into
/// `tied * (draw * (1/(N-1)))` with `draw * (1/(N-1))` constant-folded in f32. With N = 4 and
/// three or four seats tied at the end, the two differ by one ulp for draws such as -0.45
/// (Python -0.45000002), -0.9 and -0.123. The deployed -0.5 is exact either way.
#[test]
fn review_all_draws_bitwise() {
    check_placement(true);
    run_groups(true, |_| true);
}

fn check_placement(all_draws: bool) {
    let path = root().join("placement.bin");
    let Ok(buf) = std::fs::read(&path) else {
        eprintln!("SKIP placement review: {} missing", path.display());
        return;
    };
    let mut r = Reader::new(&buf);
    let (mut count, mut bad) = (0, Vec::new());
    while !r.done() {
        let n = r.i32() as usize;
        let draw = r.f64();
        let alive: Vec<bool> = r.bytes(4).iter().map(|&b| b != 0).collect();
        let elim: Vec<i32> = (0..4).map(|_| r.i32()).collect();
        let want: Vec<f32> = (0..4).map(|_| r.f32()).collect();
        let cfg = GameConfig::new(11, 11, n, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&cfg);
        s.alive[..n].copy_from_slice(&alive[..n]);
        s.elim_turn[..n].copy_from_slice(&elim[..n]);
        if !draw_checked(n, draw, all_draws) {
            continue;
        }
        let got = s.placement_values(&cfg, draw as f32);
        if got[..n]
            .iter()
            .map(|v| v.to_bits())
            .ne(want[..n].iter().map(|v| v.to_bits()))
        {
            bad.push(format!(
                "n {n} draw {draw} alive {:?} elim {:?}: rust {:?} python {:?}",
                &alive[..n],
                &elim[..n],
                &got[..n],
                &want[..n]
            ));
        }
        count += 1;
    }
    for b in bad.iter().take(20) {
        eprintln!("{b}");
    }
    eprintln!(
        "placement review: {count} records, {} mismatches",
        bad.len()
    );
    assert!(bad.is_empty(), "{} placement mismatches", bad.len());
}

#[test]
fn review_royale_hazards() {
    let path = root().join("royale_hazards.bin");
    let Ok(buf) = std::fs::read(&path) else {
        eprintln!("SKIP royale hazards review: {} missing", path.display());
        return;
    };
    let mut r = Reader::new(&buf);
    let mut count = 0;
    while !r.done() {
        let (w, h, shrink, turn) = (r.i32(), r.i32(), r.i32(), r.i32());
        let cfg = GameConfig::new(w as usize, h as usize, 2, Ruleset::Royale)
            .unwrap()
            .with_shrink_every(shrink)
            .unwrap();
        let mut sides = [0i8; MAX_SIDES];
        for s in sides.iter_mut().take(cfg.num_sides()) {
            *s = r.u8() as i8;
        }
        let want: Vec<i16> = r.bytes(cfg.cells()).iter().map(|&v| v as i16).collect();
        let got = royale_hazards(&cfg, &sides, turn);
        assert_eq!(
            &got[..cfg.cells()],
            &want[..],
            "{w}x{h} shrink {shrink} turn {turn} sides {:?}",
            &sides[..cfg.num_sides()]
        );
        count += 1;
    }
    eprintln!("royale hazards review: {count} records match");
}

// --- Plain unit tests (no golden data): the semantics the review checked by hand. -----------

const U: u8 = Move::Up as u8;
const D: u8 = Move::Down as u8;
const L: u8 = Move::Left as u8;
const R: u8 = Move::Right as u8;

fn build(cfg: &GameConfig, snakes: &[(&[(i32, i32)], i32)]) -> GameState {
    let mut s = GameState::new(cfg);
    for (i, &(b, h)) in snakes.iter().enumerate() {
        s.set_snake(cfg, i, b, h, None).unwrap();
    }
    s.refresh_done(cfg);
    s
}

/// Hazard damage that leaves exactly 0 health eliminates by hazard; 1 health survives.
#[test]
fn hazard_exact_zero_kills() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    for (health, dies) in [(15, true), (16, false), (14, true)] {
        let mut s = build(
            &cfg,
            &[
                (&[(5, 5), (5, 4), (5, 3)], health),
                (&[(9, 9), (9, 8)], 100),
            ],
        );
        s.add_hazard(&cfg, 6, 5);
        let c = s.child(&cfg, &[R, U, 0, 0]);
        assert_eq!(!c.alive[0], dies, "health {health}");
        if dies {
            assert_eq!(c.elim_cause[0], Cause::Hazard as i8);
            assert_eq!(c.health[0], 0);
        } else {
            assert_eq!(c.health[0], 1);
        }
    }
}

/// A snake that ate last turn has a stacked tail: moving into it is a body collision.
#[test]
fn tail_that_just_stacked_blocks() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    let mut s = build(
        &cfg,
        &[
            (&[(5, 5), (5, 4), (5, 3)], 100),
            (&[(6, 3), (7, 3), (8, 3)], 100),
        ],
    );
    s.add_food(&cfg, 5, 6).unwrap();
    let ate = s.child(&cfg, &[U, R, 0, 0]);
    assert_eq!(ate.length[0], 4);
    assert_eq!(ate.body_at(&cfg, 0, 5, 4), 2); // the stacked tail
    assert_eq!(ate.body_at(&cfg, 0, 5, 3), 0);
    // A snake whose tail is stacked, with another head next to that tail.
    let t = build(
        &cfg,
        &[
            (&[(5, 6), (5, 5), (5, 4), (5, 4)], 100),
            (&[(6, 4), (7, 4), (8, 4)], 100),
        ],
    );
    let c = t.child(&cfg, &[U, L, 0, 0]);
    assert!(!c.alive[1]);
    assert_eq!(c.elim_cause[1], Cause::Collision as i8);
    // Without the stack, the tail pops and the move is safe.
    let u = build(
        &cfg,
        &[
            (&[(5, 6), (5, 5), (5, 4)], 100),
            (&[(6, 4), (7, 4), (8, 4)], 100),
        ],
    );
    assert!(u.child(&cfg, &[U, L, 0, 0]).alive[1]);
}

/// A length-2 snake may turn back into its neck (the neck is its tail and pops); length 3 dies.
#[test]
fn neck_reversal() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    let s = build(&cfg, &[(&[(5, 5), (5, 4)], 100), (&[(9, 9), (9, 8)], 100)]);
    let c = s.child(&cfg, &[D, U, 0, 0]);
    assert!(c.alive[0]);
    let s = build(
        &cfg,
        &[(&[(5, 5), (5, 4), (5, 3)], 100), (&[(9, 9), (9, 8)], 100)],
    );
    let c = s.child(&cfg, &[D, U, 0, 0]);
    assert_eq!(c.elim_cause[0], Cause::SelfCollision as i8);
}

/// A starved snake's body does not block (pass 1 removes it before collisions).
#[test]
fn starved_body_does_not_block() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    let s = build(
        &cfg,
        &[
            (&[(5, 5), (5, 4), (5, 3), (5, 2)], 1),
            (&[(6, 4), (7, 4), (8, 4)], 50),
        ],
    );
    let c = s.child(&cfg, &[U, L, 0, 0]);
    assert_eq!(c.elim_cause[0], Cause::OutOfHealth as i8);
    assert!(c.alive[1]);
    assert!(c.done);
    assert_eq!(c.placement_values(&cfg, -0.5)[..2], [-1.0, 1.0]);
}

/// Royale: the zone of `turn + 1` is applied after the move; damage uses the pre-step zone.
#[test]
fn royale_shrink_timing() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Royale)
        .unwrap()
        .with_shrink_every(25)
        .unwrap();
    let mut s = build(
        &cfg,
        &[(&[(1, 5), (2, 5), (3, 5)], 50), (&[(9, 9), (9, 8)], 50)],
    );
    s.sides = [0; MAX_SIDES];
    s.turn = 24;
    s.hazard = royale_hazards(&cfg, &s.sides, 24);
    assert!(s.hazard.iter().all(|&v| v == 0));
    // Move into column 0 at turn 24: no damage this step, the zone appears in the child.
    let c = s.child(&cfg, &[L, U, 0, 0]);
    assert_eq!(c.health[0], 49);
    assert_eq!(c.hazard[5 * 11], 1);
    // The next step in column 0 is damaged.
    let c2 = c.child(&cfg, &[U, U, 0, 0]);
    assert_eq!(c2.health[0], 48 - 14);
    assert!(c2.hazard[..MAX_CELLS].iter().filter(|&&v| v > 0).count() == 11);
}

/// BUG (minor, unreachable from a request: `set_snake` and the step only store 0 to 3): an
/// out-of-range stored `last_move` with an invalid action. The jitted Python (checked by hand)
/// keeps the raw int8 `last_move` (7 stays 7) and indexes `_DELTAS` with JAX's gather, which
/// wraps negative int8 ids once (-1 -> RIGHT) and then clamps (-5 -> UP). Rust clamps with
/// `last_move.min(3)` (u8 251, i.e. int8 -5, moves RIGHT) and stores the clamped 3.
#[test]
#[ignore = "bug (minor, unreachable): out-of-range last_move handling differs from JAX gather"]
fn out_of_range_last_move_like_jax() {
    let cfg = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
    let base = build(&cfg, &[(&[(5, 5)], 50), (&[(2, 2)], 50)]);
    for (lm, head, stored) in [
        (7u8, [6, 5], 7u8),
        (127, [6, 5], 127),
        (255, [6, 5], 255),
        (251, [5, 6], 251),
    ] {
        let mut s = base;
        s.last_move[0] = lm;
        let c = s.child(&cfg, &[9, U, 0, 0]);
        assert_eq!(
            (c.head[0], c.last_move[0]),
            (head, stored),
            "last_move {lm}"
        );
    }
}
