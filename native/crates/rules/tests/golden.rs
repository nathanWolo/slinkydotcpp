//! Parity with slinky's jitted Python rules on golden vectors written by
//! `native/golden/rules_golden.py` (see its docstring for the file format). Every transition's
//! full next state, and every state's masks, forced moves, game-over flag and placement values,
//! must match exactly. Skips when the files are missing.

use slinky_rules::{
    GameConfig, GameState, MAX_SIDES, MoveMask, NUM_ACTIONS, Ruleset, royale_hazards,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn golden_root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("rules")
}

/// A little-endian reader over a byte slice.
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
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
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
        s.hazard[c] = r.u8() as i16;
    }
    for i in 0..n {
        for c in 0..cells {
            s.body[i][c] = r.u8() as i16;
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
    pv5: [f32; 4],
    pv3: [f32; 4],
}

fn unbits(b: u8) -> [bool; NUM_ACTIONS] {
    [b & 1 != 0, b & 2 != 0, b & 4 != 0, b & 8 != 0]
}

fn read_derived(r: &mut Reader, n: usize) -> Derived {
    let mut d = Derived {
        mask: [[false; 4]; 4],
        legal: [[false; 4]; 4],
        starving: [false; 4],
        over: false,
        blocked: 0,
        pv5: [0.0; 4],
        pv3: [0.0; 4],
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
    for i in 0..n {
        d.pv5[i] = r.f32();
    }
    for i in 0..n {
        d.pv3[i] = r.f32();
    }
    d
}

fn derive(s: &GameState, cfg: &GameConfig) -> Derived {
    Derived {
        mask: s.action_mask(cfg),
        legal: s.legal_moves(cfg),
        starving: s.certainly_starving(cfg),
        over: s.is_game_over(cfg),
        blocked: s.blocked_cells(cfg),
        pv5: s.placement_values(cfg, -0.5),
        pv3: s.placement_values(cfg, -0.3),
    }
}

fn read_config(dir: &Path) -> GameConfig {
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let kv: HashMap<&str, &str> = meta.lines().filter_map(|l| l.split_once(' ')).collect();
    let get = |k: &str| kv[k].parse::<i64>().unwrap();
    GameConfig::new(
        get("width") as usize,
        get("height") as usize,
        get("num_snakes") as usize,
        Ruleset::from_name(kv["ruleset"]).unwrap(),
    )
    .unwrap()
    .with_hazard_damage(get("hazard_damage") as i32)
    .with_shrink_every(get("shrink_every") as i32)
    .unwrap()
}

/// Field-by-field differences, for readable failure messages.
fn diff(a: &GameState, b: &GameState, cfg: &GameConfig) -> String {
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
    if a.sides != b.sides {
        out.push("sides differ".into());
    }
    out.join("; ")
}

/// What the transitions exercise, to show the golden data covers the rules' edge cases.
#[derive(Default)]
struct Coverage {
    causes: [usize; 7],
    multi_head_on: usize,
    eats: usize,
    double_eat: usize,
    stacked_tail_moves: usize,
    hazard_hits: usize,
    growth: usize,
    starving_parents: usize,
    invalid_actions: usize,
    zone_changes: usize,
    game_overs: usize,
    max_length: i32,
}

impl Coverage {
    #[allow(clippy::needless_range_loop)] // parallel per-seat arrays, as in the rules
    fn add(&mut self, p: &GameState, c: &GameState, actions: &[u8; 4], cfg: &GameConfig) {
        let n = cfg.num_snakes();
        if p.done {
            return;
        }
        let mut eaters = 0;
        for i in 0..n {
            if !p.alive[i] {
                continue;
            }
            if !c.alive[i] {
                self.causes[c.elim_cause[i] as usize] += 1;
            }
            let [x, y] = c.head[i];
            if p.food_at(cfg, x, y) {
                self.eats += 1;
                eaters += 1;
            }
            if p.hazard_at(cfg, x, y) > 0 {
                self.hazard_hits += 1;
            }
            if c.length[i] > p.length[i] && cfg.ruleset() == Ruleset::Constrictor {
                self.growth += 1;
            }
            let tail = p.body[i].iter().filter(|&&v| v > 0).min().copied();
            if tail.is_some_and(|t| t >= 2) {
                self.stacked_tail_moves += 1;
            }
            if actions[i] >= 4 {
                self.invalid_actions += 1;
            }
            let same = (0..n)
                .filter(|&j| p.alive[j] && c.head[j] == c.head[i])
                .count();
            if same >= 3 && (0..n).find(|&j| p.alive[j] && c.head[j] == c.head[i]) == Some(i) {
                self.multi_head_on += 1;
            }
            self.max_length = self.max_length.max(c.length[i]);
        }
        if eaters >= 2 {
            self.double_eat += 1;
        }
        if p.certainly_starving(cfg).iter().any(|&b| b) {
            self.starving_parents += 1;
        }
        if p.hazard != c.hazard {
            self.zone_changes += 1;
        }
        if c.done {
            self.game_overs += 1;
        }
    }

    fn report(&self) -> String {
        format!(
            "deaths by cause (health/wall/self/body/head/hazard) {:?}, 3+ way head-ons {}, \
             eats {}, 2+ eaters {}, stacked-tail moves {}, hazard hits {}, constrictor growth {}, \
             starving parents {}, invalid actions {}, zone changes {}, game overs {}, max length {}",
            &self.causes[1..],
            self.multi_head_on,
            self.eats,
            self.double_eat,
            self.stacked_tail_moves,
            self.hazard_hits,
            self.growth,
            self.starving_parents,
            self.invalid_actions,
            self.zone_changes,
            self.game_overs,
            self.max_length
        )
    }
}

/// Returns (states, transitions) checked.
fn check_dir(dir: &Path) -> (usize, usize) {
    let cfg = read_config(dir);
    let n = cfg.num_snakes();
    let sbuf = std::fs::read(dir.join("states.bin")).unwrap();
    let dbuf = std::fs::read(dir.join("derived.bin")).unwrap();
    let tbuf = std::fs::read(dir.join("trans.bin")).unwrap();
    let (mut sr, mut dr) = (Reader::new(&sbuf), Reader::new(&dbuf));
    let mut states = Vec::new();
    let mut derived_bad = 0;
    while !sr.done() {
        let s = read_state(&mut sr, &cfg);
        let want = read_derived(&mut dr, n);
        let got = derive(&s, &cfg);
        if got != want {
            if derived_bad < 5 {
                eprintln!(
                    "{}: state {} derived mismatch\n  rust   {got:?}\n  python {want:?}",
                    dir.display(),
                    states.len()
                );
            }
            derived_bad += 1;
        }
        states.push(s);
    }
    assert!(dr.done(), "derived.bin longer than states.bin");
    let mut tr = Reader::new(&tbuf);
    let (mut count, mut bad, mut stepped_over) = (0, 0, 0);
    let mut cov = Coverage::default();
    while !tr.done() {
        let (p, c) = (tr.u32() as usize, tr.u32() as usize);
        let actions: [u8; 4] = tr.bytes(4).try_into().unwrap();
        let parent = &states[p];
        let got = parent.child(&cfg, &actions);
        let want = &states[c];
        if got != *want {
            if bad < 5 {
                eprintln!(
                    "{}: transition {count} (state {p} -> {c}, actions {actions:?}): {}",
                    dir.display(),
                    diff(&got, want, &cfg)
                );
            }
            bad += 1;
        }
        // rules_step alone is the same except for turn and done.
        if !parent.done {
            let mut raw = *parent;
            raw.rules_step(&cfg, &actions);
            raw.turn = want.turn;
            raw.done = want.done;
            if raw != *want {
                bad += 1;
            }
        } else {
            stepped_over += 1;
        }
        cov.add(parent, want, &actions, &cfg);
        count += 1;
    }
    assert!(stepped_over > 0, "no done parents covered");
    eprintln!(
        "{}: {}",
        dir.file_name().unwrap().to_string_lossy(),
        cov.report()
    );
    assert_eq!(derived_bad, 0, "{}: derived mismatches", dir.display());
    assert_eq!(
        bad,
        0,
        "{}: {bad} of {count} transitions differ",
        dir.display()
    );
    (states.len(), count)
}

#[test]
fn transitions_match_python() {
    let root = golden_root();
    let Ok(rd) = std::fs::read_dir(&root) else {
        eprintln!(
            "SKIP golden parity: {} missing (run native/golden/rules_golden.py)",
            root.display()
        );
        return;
    };
    let mut dirs: Vec<_> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("meta.txt").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no golden configs in {}", root.display());
    for dir in dirs {
        let (s, t) = check_dir(&dir);
        eprintln!(
            "{}: {t} transitions and {s} states match exactly",
            dir.file_name().unwrap().to_string_lossy()
        );
    }
}

#[test]
fn royale_hazards_match_python() {
    let path = golden_root().join("royale_hazards.bin");
    let Ok(buf) = std::fs::read(&path) else {
        eprintln!("SKIP royale hazards: {} missing", path.display());
        return;
    };
    let mut r = Reader::new(&buf);
    let mut count = 0;
    while !r.done() {
        let (w, h, shrink, turn) = (r.i32(), r.i32(), r.i32(), r.i32());
        let cfg = GameConfig::new(w as usize, h as usize, 4, Ruleset::Royale)
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
            "{w}x{h} shrink {shrink} turn {turn} sides {sides:?}"
        );
        assert!(got[cfg.cells()..].iter().all(|&v| v == 0));
        count += 1;
    }
    eprintln!("royale_hazards: {count} cases match exactly");
}
