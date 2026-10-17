//! Bit-exact parity with slinky's jitted Python network inputs on golden vectors written by
//! `native/golden/obs_golden.py` (see its docstring for the file format). For every state and
//! every seat, the board (`unified.policy_inputs`, food hidden in constrictor), the raw window
//! (`env.observe`, food shown) and the feature rows (`policy_inputs` and `modes.mode_features`
//! for all four told modes) must have identical f32 bit patterns. Skips when the files are missing.

use slinky_obs::{Encoder, NUM_CHANNELS, NUM_FEATURES, egocentric_into, mode_features};
use slinky_rules::{GameConfig, GameState, Mode, Ruleset};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn golden_root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("obs")
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
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.bytes(2).try_into().unwrap())
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes(2).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    /// A sparse f32 array: u32 count, u16 index[count], u32 bits[count].
    fn sparse(&mut self) -> Vec<(usize, u32)> {
        let k = self.u32() as usize;
        let idx: Vec<usize> = (0..k).map(|_| self.u16() as usize).collect();
        idx.into_iter().map(|i| (i, self.u32())).collect()
    }
}

fn read_meta(dir: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(dir.join("meta.txt"))
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
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
        s.elim_turn[i] = r.i32();
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Mismatch counts of one config directory.
#[derive(Default, Debug)]
struct Tally {
    states: usize,
    seats: usize,
    floats: usize,
    board_bad: usize,
    raw_bad: usize,
    feat_bad: usize,
    first: Option<String>,
}

fn check_dir(dir: &Path) -> Tally {
    let meta = read_meta(dir);
    let get = |k: &str| meta[k].parse::<usize>().unwrap();
    let ruleset = Ruleset::from_name(&meta["ruleset"]).unwrap();
    let cfg = GameConfig::new(get("width"), get("height"), get("num_snakes"), ruleset)
        .unwrap()
        .with_shrink_every(get("shrink_every") as i32)
        .unwrap();
    let n = cfg.num_snakes();
    let enc = Encoder::new(cfg, true);
    assert_eq!(
        enc.hides_food(),
        get("hide_food") == 1,
        "{dir:?}: hide_food"
    );
    let len = enc.obs_len();
    assert_eq!(len, get("rows") * get("cols") * NUM_CHANNELS);

    let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
    let (sb, bb, rb, fb) = (
        read("states.bin"),
        read("boards.bin"),
        read("raw_diff.bin"),
        read("feats.bin"),
    );
    let (mut rs, mut rbo, mut rr, mut rf) = (
        Reader::new(&sb),
        Reader::new(&bb),
        Reader::new(&rb),
        Reader::new(&fb),
    );
    let mut t = Tally::default();
    let mut boards = vec![0.0f32; n * len];
    let mut feats = vec![0.0f32; n * NUM_FEATURES];
    let mut raw = vec![0.0f32; len];
    let mut want = vec![0u32; len];
    while !rs.done() {
        let s = read_state(&mut rs, &cfg);
        let si = t.states;
        t.states += 1;
        enc.encode_into(&s, &mut boards, &mut feats).unwrap();
        for seat in 0..n {
            t.seats += 1;
            t.floats += len;
            // The board the network gets.
            want.fill(0);
            for (i, b) in rbo.sparse() {
                want[i] = b;
            }
            let got = bits(&boards[seat * len..][..len]);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            if bad > 0 {
                t.board_bad += 1;
                if t.first.is_none() {
                    let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                    t.first = Some(format!(
                        "state {si} seat {seat}: {bad} board floats differ, first at index {i} \
                         (row {}, col {}, channel {}): got {} want {}",
                        i / NUM_CHANNELS / (2 * cfg.width() - 1),
                        i / NUM_CHANNELS % (2 * cfg.width() - 1),
                        i % NUM_CHANNELS,
                        f32::from_bits(got[i]),
                        f32::from_bits(want[i])
                    ));
                }
            }
            // The raw window, food shown (env.observe).
            for (i, b) in rr.sparse() {
                want[i] = b;
            }
            egocentric_into(&s, &cfg, seat, false, &mut raw).unwrap();
            if bits(&raw) != want {
                t.raw_bad += 1;
                t.first
                    .get_or_insert_with(|| format!("state {si} seat {seat}: raw window differs"));
            }
        }
        // policy_inputs features, then mode_features for every told mode.
        let mut want_f: Vec<u32> = (0..n * NUM_FEATURES).map(|_| rf.u32()).collect();
        let mut got_f = bits(&feats);
        for told in Mode::ALL {
            want_f.extend((0..n * NUM_FEATURES).map(|_| rf.u32()));
            let rows = mode_features(&s, &cfg, told);
            got_f.extend(rows[..n].iter().flat_map(|r| bits(r)));
        }
        if got_f != want_f {
            t.feat_bad += 1;
            t.first.get_or_insert_with(|| {
                format!("state {si}: features differ: got {got_f:?} want {want_f:?}")
            });
        }
    }
    assert!(
        rbo.done() && rr.done() && rf.done(),
        "{dir:?}: trailing data"
    );
    assert_eq!(t.states, get("num_states"), "{dir:?}: state count");
    t
}

#[test]
fn golden_observations_and_features() {
    let root = golden_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        eprintln!(
            "SKIP: no golden vectors at {root:?}; run native/golden/obs_golden.py (see its docstring)"
        );
        return;
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.txt").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no config directories under {root:?}");
    let mut failures = Vec::new();
    for dir in &dirs {
        let t = check_dir(dir);
        let name = dir.file_name().unwrap().to_string_lossy();
        eprintln!(
            "{name}: {} states, {} seat boards ({} floats): board mismatches {}, raw mismatches {}, \
             feature mismatches {}",
            t.states, t.seats, t.floats, t.board_bad, t.raw_bad, t.feat_bad
        );
        if let Some(f) = t.first {
            failures.push(format!("{name}: {f}"));
        }
    }
    assert!(failures.is_empty(), "mismatches:\n{}", failures.join("\n"));
}
