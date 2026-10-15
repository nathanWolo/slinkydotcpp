//! The error type of the crate. Nothing reachable from a request panics on bad input: every
//! check made when a configuration or a state is built returns a [`RulesError`].

use std::fmt;

/// Everything that can be refused while building a [`GameConfig`](crate::GameConfig) or a
/// [`GameState`](crate::GameState).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulesError {
    /// The board is empty or larger than [`MAX_WIDTH`](crate::MAX_WIDTH) x
    /// [`MAX_HEIGHT`](crate::MAX_HEIGHT).
    BoardSize { width: usize, height: usize },
    /// Fewer than 2 or more than [`MAX_SNAKES`](crate::MAX_SNAKES) seats (solo games and games
    /// with more than 4 snakes are not supported; the Python search refuses more than 4 too,
    /// `mcts._check_game`, mcts.py:481-483).
    NumSnakes(usize),
    /// `shrink_every_n_turns` below 1 (the engine rejects 0; types.py:109-110).
    ShrinkEvery(i32),
    /// A ruleset name this crate does not play (wrapped and solo are out of scope).
    Ruleset(String),
    /// A mode name outside `modes.MODE_NAMES`, or a seat count the mode cannot have
    /// (`modes.mode_config`, modes.py:84-87).
    Mode(String),
    /// A seat index at or beyond the config's `num_snakes`.
    Seat(usize),
    /// A snake body longer than the countdown grid can hold (more than `i16::MAX` segments).
    BodyTooLong(usize),
    /// A food cell outside the board, or listed twice (engine_json.py:124-128 raises too).
    Food { x: i32, y: i32 },
}

impl fmt::Display for RulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RulesError::BoardSize { width, height } => {
                write!(
                    f,
                    "unsupported board size {width}x{height} (1..=11 per side)"
                )
            }
            RulesError::NumSnakes(n) => write!(f, "unsupported snake count {n} (2..=4)"),
            RulesError::ShrinkEvery(s) => write!(f, "shrink_every_n_turns must be >= 1, got {s}"),
            RulesError::Ruleset(name) => write!(f, "unsupported ruleset {name:?}"),
            RulesError::Mode(m) => write!(f, "bad game mode: {m}"),
            RulesError::Seat(s) => write!(f, "seat {s} is out of range"),
            RulesError::BodyTooLong(n) => write!(f, "snake body of {n} segments is too long"),
            RulesError::Food { x, y } => {
                write!(f, "food at ({x}, {y}) is off the board or duplicated")
            }
        }
    }
}

impl std::error::Error for RulesError {}
