//! The Battlesnake API (version 1) as serde types: the `/start`, `/move` and `/end` request body
//! (all three have the same shape), the `/move` response and the `GET /` info response.
//!
//! Parsing is defensive: unknown fields are ignored (the API grows them), fields the conversion
//! needs must be present with the right type, and anything else is a [`ApiError::Json`]. The
//! `game` object is kept as raw JSON because its settings are read leniently (see
//! `convert::game_config`, ported from `server._setting`, server.py:149-158).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ApiError;

/// A board position. `(0, 0)` is the bottom-left cell and `y` grows upwards, the same convention
/// as the engine's `grid[y][x]` with row 0 at the bottom (types.py:1-16), so nothing is flipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

/// A snake as listed in `board.snakes` and in `you`.
///
/// Only `id`, `health` and `body` are needed to build a state. `head` and `length` are used by
/// [`safe_move`](crate::safe_move) when present (the API always sends them).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snake {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub health: i32,
    /// Head first. At turn 0 all segments share one cell; a snake that just ate repeats its tail.
    pub body: Vec<Coord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<Coord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<i64>,
}

/// The `board` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Board {
    pub width: i64,
    pub height: i64,
    pub food: Vec<Coord>,
    /// Absent in rulesets without hazards. Duplicates stack.
    #[serde(default)]
    pub hazards: Vec<Coord>,
    /// The living snakes only.
    pub snakes: Vec<Snake>,
}

/// The body of `POST /start`, `/move` and `/end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// The `game` object (id, ruleset with settings, map, timeout, source); `None` when absent,
    /// which reads as a standard game on the standard map, like the Python `req.get("game", {})`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game: Option<Value>,
    pub turn: i64,
    pub board: Board,
    pub you: Snake,
}

impl Request {
    /// Parses a request body.
    pub fn from_json(text: &str) -> Result<Request, ApiError> {
        Ok(serde_json::from_str(text)?)
    }

    /// `game.id`, the key of the caller's `/start` cache (`Mover._games`, server.py:459-470).
    pub fn game_id(&self) -> Option<&str> {
        self.game.as_ref()?.get("id")?.as_str()
    }

    /// The number of snakes listed. At `/start` this is the game's starting snake count, which
    /// the caller should remember (`Mover.start`, server.py:459-470) and pass as
    /// `starting_snakes` to [`game_from_request`](crate::game_from_request) on every `/move`.
    pub fn snake_count(&self) -> usize {
        self.board.snakes.len()
    }
}

/// The `GET /<snake>` response (server.py:812-819).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InfoResponse {
    pub apiversion: String,
    pub author: String,
    pub color: String,
    pub head: String,
    pub tail: String,
    pub version: String,
}

impl InfoResponse {
    /// Info for API version 1 with the given look.
    pub fn new(
        author: impl Into<String>,
        color: impl Into<String>,
        head: impl Into<String>,
        tail: impl Into<String>,
        version: impl Into<String>,
    ) -> InfoResponse {
        InfoResponse {
            apiversion: "1".to_string(),
            author: author.into(),
            color: color.into(),
            head: head.into(),
            tail: tail.into(),
            version: version.into(),
        }
    }
}

/// The `POST /move` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveResponse {
    /// `"up"`, `"down"`, `"left"` or `"right"`.
    #[serde(rename = "move")]
    pub mv: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shout: Option<String>,
}

impl MoveResponse {
    /// A response for `mv` without a shout.
    pub fn new(mv: slinky_rules::Move) -> MoveResponse {
        MoveResponse {
            mv: mv.name().to_string(),
            shout: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_serialize_like_the_api() {
        let m = MoveResponse::new(slinky_rules::Move::Left);
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"{"move":"left"}"#);
        let i = InfoResponse::new("a", "#fff", "default", "default", "v");
        let v = serde_json::to_value(&i).unwrap();
        assert_eq!(v["apiversion"], "1");
        assert_eq!(v["color"], "#fff");
    }

    #[test]
    fn bad_bodies_are_errors() {
        for text in [
            "",
            "{}",
            "[1]",
            r#"{"turn":"x"}"#,
            r#"{"turn":1,"board":{},"you":{}}"#,
        ] {
            assert!(Request::from_json(text).is_err(), "{text}");
        }
    }
}
