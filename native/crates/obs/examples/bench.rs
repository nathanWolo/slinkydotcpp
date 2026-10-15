//! Time `Encoder::encode_into` (every seat's board and features of one node) on mid-game states
//! of each mode. Run with `cargo run --release -p slinky-obs --example bench`.

use slinky_obs::{Encoder, NUM_FEATURES};
use slinky_rules::{GameConfig, GameState, Mode};
use std::hint::black_box;
use std::time::Instant;

/// A mid-game state: snakes of length 8 to 14 winding up the board, some food and hazards.
fn state(cfg: &GameConfig) -> GameState {
    let mut s = GameState::new(cfg);
    for seat in 0..cfg.num_snakes() {
        let x0 = 1 + 3 * seat as i32;
        let len = 8 + 2 * seat as i32;
        let segs: Vec<(i32, i32)> = (0..len)
            .map(|k| {
                let (col, row) = (k / 10, k % 10);
                let y = if col % 2 == 0 { 10 - row } else { 1 + row };
                (x0 + col, y)
            })
            .collect();
        s.set_snake(cfg, seat, &segs, 60 + seat as i32, None)
            .unwrap();
    }
    for (x, y) in [(0, 0), (5, 5), (10, 3)] {
        let _ = s.add_food(cfg, x, y);
    }
    for x in 0..11 {
        s.add_hazard(cfg, x, 0);
    }
    s.turn = 137;
    s.refresh_done(cfg);
    s
}

fn main() {
    for mode in Mode::ALL {
        let cfg = GameConfig::for_mode(mode, None).unwrap();
        let s = state(&cfg);
        let enc = Encoder::new(cfg, true);
        let n = cfg.num_snakes();
        let mut boards = vec![0.0f32; n * enc.obs_len()];
        let mut feats = vec![0.0f32; n * NUM_FEATURES];
        let iters = 200_000;
        for _ in 0..1000 {
            enc.encode_into(&s, &mut boards, &mut feats).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            enc.encode_into(black_box(&s), &mut boards, &mut feats)
                .unwrap();
            black_box(&boards);
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "{:<12} {n} seats: {:7.0} ns per node, {:6.0} ns per seat",
            mode.name(),
            ns,
            ns / n as f64
        );
    }
}
