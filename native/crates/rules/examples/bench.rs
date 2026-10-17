//! Throughput of the search's per-node rules work: `child` (copy + step + turn/done) and
//! `legal_moves`, over random-legal 4-snake and duel games on 11x11.
//!
//! Run: `cargo run --release -p slinky-rules --example bench`

use slinky_rules::{GameConfig, GameState, Mode};
use std::hint::black_box;
use std::time::Instant;

/// xorshift64*: a tiny deterministic RNG, enough to pick moves.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn start(cfg: &GameConfig) -> GameState {
    let mut s = GameState::new(cfg);
    let spots = [(1, 1), (9, 9), (1, 9), (9, 1)];
    for (i, &p) in spots.iter().take(cfg.num_snakes()).enumerate() {
        s.set_snake(cfg, i, &[p, p, p], 100, None).unwrap();
    }
    for (x, y) in [(5, 5), (3, 7), (7, 3), (2, 2), (8, 8)] {
        s.add_food(cfg, x, y).unwrap();
    }
    s.refresh_done(cfg);
    s
}

fn bench(mode: Mode) {
    let cfg = GameConfig::for_mode(mode, None).unwrap();
    let n = cfg.num_snakes();
    let root = start(&cfg);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut steps, mut games) = (0u64, 0u64);
    let t = Instant::now();
    let mut s = root;
    while steps < 2_000_000 {
        let legal = s.legal_moves(&cfg);
        let mut actions = [0u8; 4];
        for (i, a) in actions.iter_mut().enumerate().take(n) {
            let count = legal[i].iter().filter(|&&b| b).count() as u64;
            let pick = (rng.next() % count) as usize;
            *a = (0..4u8)
                .filter(|&m| legal[i][m as usize])
                .nth(pick)
                .unwrap_or(0);
        }
        s = black_box(s.child(&cfg, &actions));
        steps += 1;
        if s.done {
            black_box(s.placement_values(&cfg, -0.5));
            s = root;
            games += 1;
        }
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "{:>11}: {:.0} ns per legal_moves + child ({steps} steps, {games} games), state {} bytes",
        mode.name(),
        dt * 1e9 / steps as f64,
        std::mem::size_of::<GameState>()
    );
}

fn main() {
    for mode in Mode::ALL {
        bench(mode);
    }
}
