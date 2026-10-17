//! The crate's error type.

use slinky_api::ApiError;
use slinky_net::NetError;
use slinky_obs::ObsError;
use std::fmt;

/// Why a search or an agent move could not be made. The caller answers with the safe move
/// ([`slinky_api::safe_move`]) in every case, as the Python server does.
#[derive(Debug)]
pub enum SearchError {
    /// A game the agent does not play: board size, seat count, or a mode outside the run's
    /// `game_modes` (`agents.check_game`, agents.py:300-336). Python plays some of these with
    /// its heuristic fallback agent, which is not ported (see the `agent` module docs).
    Unsupported(String),
    /// The noise source had no bits for this simulation.
    Noise {
        /// The simulation (0-based).
        sim: usize,
    },
    /// The request could not be turned into a game.
    Api(ApiError),
    /// The network failed (bad run dir, bad inputs).
    Net(NetError),
    /// The observation builder failed.
    Obs(ObsError),
    /// A custom evaluator failed.
    Evaluator(String),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::Unsupported(m) => write!(f, "unsupported game: {m}"),
            SearchError::Noise { sim } => write!(f, "no noise bits for simulation {sim}"),
            SearchError::Api(e) => write!(f, "request: {e}"),
            SearchError::Net(e) => write!(f, "network: {e}"),
            SearchError::Obs(e) => write!(f, "observation: {e}"),
            SearchError::Evaluator(m) => write!(f, "evaluator: {m}"),
        }
    }
}

impl std::error::Error for SearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SearchError::Api(e) => Some(e),
            SearchError::Net(e) => Some(e),
            SearchError::Obs(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ApiError> for SearchError {
    fn from(e: ApiError) -> Self {
        SearchError::Api(e)
    }
}

impl From<NetError> for SearchError {
    fn from(e: NetError) -> Self {
        SearchError::Net(e)
    }
}

impl From<ObsError> for SearchError {
    fn from(e: ObsError) -> Self {
        SearchError::Obs(e)
    }
}
