//! slinky-referee: a native match runner for strength tests between native agents
//! (`naz-match`), the Rust counterpart of slinky's `benchmarks/strength.py` for az_modes
//! agents.
//!
//! A matchup plays many games of agent A against agent B in one of the four ladder modes
//! ([`slinky_env::EnvConfig::ladder`]): in the duel A and B alternate seats; with 4 snakes A
//! holds one seat and B the other three, A's seat rotating with the game id (game `g` puts A
//! in seat `g % N`, as `evaluate.run_match` and `play_table` with line-up `(0, 1, 1, 1)` do).
//! Games run on a pool of threads, one game per thread, and each game is fixed by the seed and
//! its id ([`game`]), so results do not depend on the thread count. Output is resumable JSONL:
//! one line per game and per round, and a summary line per matchup ([`runner`]).
//!
//! Statistics ([`stats`], [`outcome`]): W/D/L and the score (a draw counts 1/2) with the Wald
//! half-width of `slinky.evaluate`, the Wilson interval of strength.py and Elo with its
//! interval for the duel; the share of the win and the placement with their intervals for 4
//! snakes; and, with `--sequential`, the eval-speed branch's anytime-valid betting test and
//! confidence sequence, ported to Rust. `--target-ci` stops at a fixed precision instead.
//! Each agent's search time per move is recorded, and the summary gives its p50, p90, p99 and
//! maximum per mode.
//!
//! The sequential test is ported rather than run in Python from the per-round file: it is
//! about a hundred lines of pure functions of the per-round W/D/L counts, and a port keeps
//! `naz-match` a single binary that decides its stop in-process between rounds, on machines
//! without slinky's Python stack (the laptops, AWS). The per-round file still has
//! strength.py's W/D/L columns first, so the Python can check a run, and
//! `native/golden/referee_golden.py` compares the port with the Python on random inputs.

#![recursion_limit = "256"]

pub mod game;
pub mod outcome;
pub mod player;
pub mod runner;
pub mod stats;
pub mod util;

pub use game::{GameRecord, MoveLog, Setup, View, Workspace, play_game};
pub use outcome::{GameScore, Outcome, score_game};
pub use player::{Player, PlayerSpec};
pub use runner::{MatchConfig, run_matchup};
