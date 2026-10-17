//! The error type of the crate: nothing here panics on bad input, every failure is a [`NetError`].

use std::fmt;

/// Everything that can go wrong while loading or running a network.
#[derive(Debug)]
pub enum NetError {
    /// Reading a file failed.
    Io(String, std::io::Error),
    /// `config.json` is not valid JSON or has the wrong field types.
    Json(String),
    /// `params.npz` (zip container or `.npy` entry) is malformed or uses an unsupported feature.
    Npz(String),
    /// A config the engine cannot run (unsupported dtype, features mode, algorithm, sizes).
    Unsupported(String),
    /// An array is missing or has the wrong shape for the config.
    Shape(String),
    /// The caller passed inputs of the wrong length.
    Input(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Io(path, e) => write!(f, "cannot read {path}: {e}"),
            NetError::Json(m) => write!(f, "bad config.json: {m}"),
            NetError::Npz(m) => write!(f, "bad params.npz: {m}"),
            NetError::Unsupported(m) => write!(f, "unsupported network: {m}"),
            NetError::Shape(m) => write!(f, "weight shape mismatch: {m}"),
            NetError::Input(m) => write!(f, "bad network input: {m}"),
        }
    }
}

impl std::error::Error for NetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NetError::Io(_, e) => Some(e),
            _ => None,
        }
    }
}
