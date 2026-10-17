//! The error type of the crate. Nothing here panics on bad input: a seat out of range or a
//! buffer of the wrong size is reported as an [`ObsError`].

use std::fmt;

/// Everything the observation builders can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObsError {
    /// A seat index at or beyond the config's `num_snakes`.
    Seat { seat: usize, num_snakes: usize },
    /// An output buffer whose length is not the one the call writes.
    Buffer {
        /// What the buffer is (`"observation"`, `"boards"`, `"features"`).
        what: &'static str,
        /// The length passed in.
        got: usize,
        /// The length the call needs.
        need: usize,
    },
}

impl fmt::Display for ObsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObsError::Seat { seat, num_snakes } => {
                write!(f, "seat {seat} out of range for {num_snakes} snakes")
            }
            ObsError::Buffer { what, got, need } => {
                write!(f, "{what} buffer has {got} floats, need exactly {need}")
            }
        }
    }
}

impl std::error::Error for ObsError {}
