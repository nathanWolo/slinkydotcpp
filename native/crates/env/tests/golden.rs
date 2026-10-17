//! Exact transitions against slinky's jitted `BattlesnakeEnv.step`, on golden records written by
//! `native/golden/env_golden.py` (`--only trans`; see its docstring for the format).
//!
//! Each record holds a state, a joint action, the spawn roll slinky derived from its key, the
//! spawn mask of the post-rules state, the cells it spawned, `truncated`, both rewards and the
//! next state. The Rust step runs with a scripted [`EnvRandom`] that answers the roll and hands
//! out exactly those cells, so everything the map's randomness did not decide must match: the
//! mask, the number of cells asked for, the whole next state, `truncated`, `done` and the
//! rewards (f32 bits). Skips when the files are missing.

use slinky_env::{EnvConfig, EnvRandom, MapKind, nth_set_bit};
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, Ruleset};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn golden_root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("env").join("trans")
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
    fn u8(&mut self) -> u8 {
        self.bytes(1)[0]
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
        s.hazard[c] = i16::from(r.u8());
    }
    for i in 0..n {
        for c in 0..cells {
            s.body[i][c] = i16::from(r.u8());
        }
    }
    s
}

fn read_meta(dir: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(dir.join("meta.txt"))
        .unwrap()
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

fn env_of(meta: &HashMap<String, String>) -> EnvConfig {
    let num = |k: &str| -> i32 { meta[k].parse().unwrap() };
    let ruleset = Ruleset::from_name(&meta["ruleset"]).unwrap();
    let max_turns = num("max_turns");
    let rules = GameConfig::new(
        num("width") as usize,
        num("height") as usize,
        num("num_snakes") as usize,
        ruleset,
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

/// Answers slinky's roll and hands out slinky's spawned cells, recording what the env asked.
struct Scripted {
    roll: i32,
    cells: u128,
    rolls: u32,
    picks: u32,
    /// Picks asked for while cells were free but slinky had spawned no more.
    extra: u32,
    /// Masks the env picked from (the first one is the spawn mask).
    first_mask: Option<u128>,
    /// A slinky cell outside the mask the env picked from.
    outside: bool,
}

impl EnvRandom for Scripted {
    fn next_u64(&mut self) -> u64 {
        panic!("the step drew raw randomness outside the roll and the cell picks");
    }
    fn food_roll(&mut self) -> i32 {
        self.rolls += 1;
        self.roll
    }
    fn pick_cell(&mut self, mask: u128) -> Option<usize> {
        self.picks += 1;
        self.first_mask.get_or_insert(mask);
        if self.cells == 0 {
            if mask != 0 {
                self.extra += 1;
            }
            return None;
        }
        let c = nth_set_bit(self.cells, 0);
        self.cells &= !(1u128 << c);
        if mask & (1u128 << c) == 0 {
            self.outside = true;
        }
        Some(c)
    }
}

fn run_dir(dir: &Path) -> (usize, usize, Vec<String>) {
    let meta = read_meta(dir);
    let env = env_of(&meta);
    let cfg = *env.rules();
    let n = cfg.num_snakes();
    let buf = std::fs::read(dir.join("trans.bin")).unwrap();
    let mut r = Reader { buf: &buf, pos: 0 };
    let (mut count, mut spawns) = (0, 0);
    let mut problems = Vec::new();
    while r.pos < buf.len() {
        let pre = read_state(&mut r, &cfg);
        let mut actions = [0u8; MAX_SNAKES];
        for a in actions.iter_mut() {
            *a = r.u8();
        }
        let roll = r.i32();
        let mask = r.u128();
        let spawned = r.u128();
        let truncated = r.u8() != 0;
        let wl: Vec<f32> = (0..n).map(|_| r.f32()).collect();
        let pl: Vec<f32> = (0..n).map(|_| r.f32()).collect();
        let next = read_state(&mut r, &cfg);

        let mut rng = Scripted {
            roll,
            cells: spawned,
            rolls: 0,
            picks: 0,
            extra: 0,
            first_mask: None,
            outside: false,
        };
        let mut s = pre;
        let info = env.step(&mut rng, &mut s, &actions);
        let mut why = Vec::new();
        if s != next {
            why.push(format!(
                "next state differs:\n  rust   {s:?}\n  python {next:?}"
            ));
        }
        if rng.cells != 0 {
            why.push(format!(
                "python spawned {} more cells",
                rng.cells.count_ones()
            ));
        }
        if rng.extra > 0 || rng.outside {
            why.push(format!(
                "extra picks {} / cell outside mask {}",
                rng.extra, rng.outside
            ));
        }
        if let Some(m) = rng.first_mask
            && m != mask
        {
            why.push(format!("spawn mask {m:#x} != python {mask:#x}"));
        }
        // The mask the env would pick from, whether or not it spawned.
        if !pre.done && env.spawns_food() {
            let mut post = pre;
            post.rules_step(&cfg, &actions);
            if env.spawn_mask(&post) != mask {
                why.push("spawn_mask(post-rules state) differs".into());
            }
        }
        if info.truncated != truncated || info.done != next.done {
            why.push(format!(
                "truncated/done {}/{} != python {truncated}/{}",
                info.truncated, info.done, next.done
            ));
        }
        for i in 0..n {
            if info.win_loss[i].to_bits() != wl[i].to_bits()
                || info.placement[i].to_bits() != pl[i].to_bits()
            {
                why.push(format!(
                    "rewards {:?}/{:?} != python {wl:?}/{pl:?}",
                    &info.win_loss[..n],
                    &info.placement[..n]
                ));
                break;
            }
        }
        if info.spawned != spawned.count_ones() {
            why.push(format!(
                "spawned {} != {}",
                info.spawned,
                spawned.count_ones()
            ));
        }
        if !why.is_empty() && problems.len() < 5 {
            problems.push(format!(
                "{} #{count}: actions {actions:?} roll {roll}: {}",
                dir.display(),
                why.join("; ")
            ));
        } else if !why.is_empty() {
            problems.push(String::new());
        }
        spawns += usize::from(spawned != 0);
        count += 1;
    }
    (count, spawns, problems)
}

#[test]
fn transitions_match_slinky() {
    let root = golden_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        eprintln!("SKIP: no env golden transitions at {}", root.display());
        return;
    };
    let mut dirs: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
    dirs.sort();
    let mut total = 0;
    let mut failures = Vec::new();
    for dir in dirs.iter().filter(|d| d.join("trans.bin").exists()) {
        let (count, spawns, problems) = run_dir(dir);
        let name = dir.file_name().unwrap().to_string_lossy();
        eprintln!(
            "env golden {name}: {count} transitions ({spawns} with spawns), {} mismatches",
            problems.len()
        );
        total += count;
        failures.extend(problems.into_iter().filter(|p| !p.is_empty()));
        assert!(failures.len() < 20, "{}", failures.join("\n"));
    }
    if total == 0 {
        eprintln!("SKIP: no env golden transitions at {}", root.display());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
