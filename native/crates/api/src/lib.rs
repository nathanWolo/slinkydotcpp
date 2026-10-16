//! slinky-api: the Battlesnake HTTP API (version 1) on top of [`slinky_rules`].
//!
//! It turns a `/move` request into what the agent plays on, exactly as the Python server does
//! (`src/slinky/server.py`, `src/slinky/engine_json.py`, SPEC.md section 6):
//!
//! * [`Request`]: a request body parsed as the Python server parses it (`json.loads`'s
//!   dialect, the HTTP handler's checks) and kept as a dynamically typed [`PyValue`], because
//!   the server reads its fields with Python's semantics (see [`py`]). Bad input is an
//!   [`ApiError`], never a panic. [`MoveResponse`], [`InfoResponse`]: the responses.
//! * [`game_from_request`]: request -> [`Game`] (the normalised `GameConfig`, the `GameState`
//!   with our snake in seat 0, and the [`Mode`](slinky_rules::Mode) the network is told).
//! * [`safe_move`] / [`safe_move_json`]: the fallback move for any error.
//!
//! # Snake count
//!
//! The API lists only living snakes and never says how many started. The Python server learns
//! it from `/start` and remembers it per game id; a stateless Lambda must do the same (or pass
//! `None`). [`game_from_request`] takes it as `starting_snakes: Option<usize>`: with `Some(n)`
//! the dead seats are padded back in as eliminated snakes, so a four-player game keeps four
//! seats (and its mode) to the end; with `None` the seat count is the number of snakes alive,
//! which reads a four-player standard game that is down to two snakes as a *duel* (the network is
//! told `duel` and the placement values change) and gives no game at all once one snake is left.
//!
//! ```
//! use slinky_api::{Request, game_from_request, safe_move_json};
//! use slinky_rules::Mode;
//!
//! let body = r#"{
//!   "game": {"id": "g1", "ruleset": {"name": "standard", "settings": {}}, "map": "standard"},
//!   "turn": 0,
//!   "board": {"width": 11, "height": 11, "food": [{"x": 0, "y": 1}], "hazards": [],
//!     "snakes": [
//!       {"id": "a", "health": 100, "body": [{"x":1,"y":1},{"x":1,"y":1},{"x":1,"y":1}]},
//!       {"id": "b", "health": 100, "body": [{"x":9,"y":9},{"x":9,"y":9},{"x":9,"y":9}]}]},
//!   "you": {"id": "b", "health": 100, "body": [{"x":9,"y":9},{"x":9,"y":9},{"x":9,"y":9}]}
//! }"#;
//! let req = Request::from_json(body)?;
//! let started_with = req.snake_count(); // remembered from /start
//! let game = game_from_request(&req, started_with)?;
//! assert_eq!(game.mode, Mode::Duel);
//! assert_eq!(game.state.body_at(&game.config, 0, 9, 9), 3); // `you` is seat 0
//! let fallback = safe_move_json(body); // what to answer if anything above fails
//! # let _ = fallback;
//! # Ok::<(), slinky_api::ApiError>(())
//! ```
//!
//! Parity: `tests/golden.rs` replays thousands of requests built from random slinky games in
//! all four modes plus hand-written edge cases through the Python server's own
//! `game_config`/`request_state`/`safe_move` (`native/golden/api_golden.py`) and requires the
//! config, the mode, the whole state and the fallback move to match exactly;
//! `tests/review_api_conversion.rs` does the same for hostile bodies and random edits of
//! realistic ones (`native/golden/review_api_conversion.py`), against what the whole server
//! answers.

mod convert;
mod error;
pub mod py;
mod safe;
mod types;

pub use convert::{
    Game, MAX_SETTING, check_sizes, game_config, game_from_request, request_state, royale_sides,
    setting,
};
pub use error::ApiError;
pub use py::{PyErr, PyValue};
pub use safe::{safe_move, safe_move_json};
pub use types::{Coord, InfoResponse, MAX_BODY, MoveResponse, Request};
