//! The error type of the crate. Every way a request can fail to become a game state is a value
//! here; nothing reachable from a request panics. The Lambda answers any `Err` with
//! [`safe_move`](crate::safe_move) (the Python server does the same: `Mover.move` falls back to
//! `safe_move` for a game it has no config for, and the HTTP handler for any exception,
//! server.py:419-432, 649-668).

use std::fmt;

use slinky_rules::RulesError;

/// Why a request could not be turned into a game.
#[derive(Debug, Clone, PartialEq)]
pub enum ApiError {
    /// The body is not a Battlesnake request at all: empty or over 64 KB, not JSON (in
    /// Python's dialect), not an object, or an object whose `game` is not one. The Python
    /// handler answers HTTP 400 and sends no move (server.py:649-663).
    Json(String),
    /// A well-formed request for a game the agent does not model: another ruleset or map
    /// (wrapped, solo, squad, arcade maze, constrictor on the royale map, ...), a board larger
    /// than 11x11, or more than 4 or fewer than 2 seats (`server.game_config` returning `None`,
    /// `modes.mode_of` raising, `mcts._check_game`).
    Unsupported(String),
    /// A request on which the Python conversion raises: a field it reads is missing or has a
    /// type it cannot use, `you` is not among the snakes, a food cell is off the board or
    /// listed twice, the lists are larger than the board can hold, the turn or a health does
    /// not fit in an `int32` (`server.check_sizes`, `request_to_engine`, `state_from_engine`).
    /// The message names the Python exception.
    BadRequest(String),
    /// The rules crate refused a configuration or a state.
    Rules(RulesError),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::Json(m) => write!(f, "bad JSON: {m}"),
            ApiError::Unsupported(m) => write!(f, "unsupported game: {m}"),
            ApiError::BadRequest(m) => write!(f, "bad request: {m}"),
            ApiError::Rules(e) => write!(f, "rules: {e}"),
        }
    }
}

impl std::error::Error for ApiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApiError::Rules(e) => Some(e),
            _ => None,
        }
    }
}

impl From<RulesError> for ApiError {
    fn from(e: RulesError) -> Self {
        ApiError::Rules(e)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::Json(e.to_string())
    }
}
