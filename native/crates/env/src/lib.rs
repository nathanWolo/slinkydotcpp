//! slinky-env: a full Battlesnake game simulator for native self-play, the Rust counterpart of
//! slinky's `BattlesnakeEnv` around the rules the search uses.
//!
//! `slinky-rules` holds the deterministic rules step (bit-exact with slinky). This crate adds
//! everything a whole game needs on top, as slinky's Python does:
//!
//! * turn 0 (`maps.py` setups, `env.init_state`): the engine's start positions (corners or
//!   cardinals on square boards of side 7 or more, random even cells otherwise), starting food
//!   (one food diagonal to each head plus the centre), constrictor's food removal, and the royale
//!   shrink sides, drawn once at reset;
//! * every turn (`env.step`, `maps.spawn_food_standard`, `RoyaleMap.post_update`): the rules
//!   step, food spawning (top up to `minimum_food`, otherwise one food with probability
//!   `(food_spawn_chance - 1) / 100`, on a uniformly random cell that is free and not next to a
//!   head), the royale zone, the turn increment, game over and `max_turns` truncation, and the
//!   win/loss and placement rewards;
//! * the maps `standard`, `royale` and `empty` (no food: the official constrictor ladder's
//!   map), and the four ladder modes ([`EnvConfig::ladder`]) or slinky's training configs
//!   ([`EnvConfig::for_mode`], constrictor on the standard map where food spawns but is inert);
//! * positions from real games: [`from_request`] turns a Battlesnake request into an environment
//!   and state (through `slinky-api`), and [`to_request_json`] turns a state into a request.
//!
//! Randomness comes from a seedable [`Rng`] (xoshiro256++) through the [`EnvRandom`] trait, so a
//! game is fixed by a `u64` seed. Like slinky itself (which matches the official engine's
//! distributions, not its Go random stream), this does not reproduce JAX's stream; it matches
//! slinky's distributions. A step allocates nothing on the heap: [`GameState`] is `Copy`.
//!
//! ```
//! use slinky_env::{EnvConfig, Rng, random_legal};
//! use slinky_rules::Mode;
//!
//! let env = EnvConfig::ladder(Mode::Royale)?;      // 4 seats, royale map and ruleset
//! let mut rng = Rng::new(7);
//! let mut state = env.reset(&mut rng);             // turn 0
//! while !state.done {
//!     let actions = random_legal(&mut rng, env.rules(), &state);
//!     let info = env.step(&mut rng, &mut state, &actions);
//!     let _ = (info.placement, info.spawned);
//! }
//! # Ok::<(), slinky_env::EnvError>(())
//! ```
//!
//! # Verification
//!
//! `native/golden/env_golden.py` writes two kinds of golden data from slinky's jitted env:
//!
//! * transitions (`tests/golden.rs`): states from random-legal rollouts in 13 configs are
//!   stepped by `BattlesnakeEnv.step`; the Rust step, given the same state, actions, spawn roll
//!   and spawned cells (a scripted [`EnvRandom`]), must reproduce the spawn mask, the number of
//!   cells it asks for, the full next state, `truncated` and both rewards exactly;
//! * distributions (`tests/distributions.rs`, `examples/env_stats.rs`): thousands of whole
//!   random-legal games per config; start positions, starting food, food counts per turn, spawn
//!   rates, royale side frequencies and zone timing, game lengths, outcomes and elimination
//!   causes must agree within statistical tolerances.

mod config;
mod env;
mod policy;
mod request;
mod rng;
mod setup;

pub use config::{EnvConfig, EnvError, MapKind};
pub use env::{Game, StepInfo, placement_reward, win_loss_reward};
pub use policy::{random_legal, random_moves};
pub use request::{body_segments, from_request, to_request_json};
pub use rng::{EnvRandom, Rng, nth_set_bit};
pub use setup::{
    Placement, START_LENGTH, StartPoints, fixed_start_points, place_snakes, random_start_cells,
    spawn_mask,
};

pub use slinky_rules::{Actions, GameState};
