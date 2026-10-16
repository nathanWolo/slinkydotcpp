//! Speed of the simulator: random-legal games per ladder mode, one thread.
//!
//! `cargo run --release -p slinky-env --example env_bench -- [games per mode]`
//!
//! Reports steps per second for the env step alone and with the random-legal policy.

use slinky_env::{EnvConfig, Rng, random_legal};
use slinky_rules::Mode;
use std::time::Instant;

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000);
    for mode in Mode::ALL {
        let env = EnvConfig::ladder(mode).expect("ladder config");
        // Record the action sequences once, then time the env alone on them.
        let mut rng = Rng::new(1);
        let mut actions = Vec::new();
        let mut starts = Vec::new();
        let t0 = Instant::now();
        let mut steps = 0u64;
        for g in 0..games {
            let mut game_rng = Rng::with_stream(7, g);
            let mut s = env.reset(&mut game_rng);
            starts.push((s, game_rng));
            while !s.done {
                let a = random_legal(&mut rng, env.rules(), &s);
                actions.push(a);
                env.step(&mut game_rng, &mut s, &a);
                steps += 1;
            }
        }
        let with_policy = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut k = 0;
        let mut turns = 0i64;
        for (s0, r0) in &starts {
            let (mut s, mut r) = (*s0, *r0);
            while !s.done {
                env.step(&mut r, &mut s, &actions[k]);
                k += 1;
            }
            turns += i64::from(s.turn);
        }
        let env_only = t1.elapsed().as_secs_f64();
        assert_eq!(k as u64, steps);
        println!(
            "{:<12} {games} games, mean {:.1} turns: env step {:.0} ns ({:.1} M steps/s); \
             with random_legal {:.0} ns/step",
            mode.name(),
            turns as f64 / games as f64,
            env_only * 1e9 / steps as f64,
            steps as f64 / env_only / 1e6,
            with_policy * 1e9 / steps as f64,
        );
    }
}
