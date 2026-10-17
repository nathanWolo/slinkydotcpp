//! The Battlesnake API (version 1): the `/start`, `/move` and `/end` request body (all three
//! have the same shape), the `/move` response and the `GET /` info response.
//!
//! A request is kept as the dynamically typed value the Python server works on
//! ([`PyValue`], see `py.rs`): the server reads only the fields it needs, with Python's
//! semantics, so a typed schema would refuse requests Python plays (a `you` without `health`,
//! a numeric-string health) and accept ones it refuses (a coordinate written `[x, y]`).
//! [`Request::from_json`] applies the HTTP handler's own checks (server.py:649-663).

use serde::{Deserialize, Serialize};

use crate::ApiError;
use crate::py::{self, PyValue};

/// Largest body the Python handler reads (`server.MAX_BODY`, server.py:104): it answers 400
/// to an empty body or a longer one.
pub const MAX_BODY: usize = 1 << 16;

/// A board position. `(0, 0)` is the bottom-left cell and `y` grows upwards, the same convention
/// as the engine's `grid[y][x]` with row 0 at the bottom (types.py:1-16), so nothing is flipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

/// The body of `POST /start`, `/move` and `/end`, parsed as the Python server parses it.
///
/// It is a JSON object (a dict) whose `game`, if present, is an object too; everything else is
/// read when the request is converted ([`crate::game_from_request`]) or answered with the
/// fallback move ([`crate::safe_move`]).
pub struct Request {
    root: PyValue,
}

impl std::fmt::Debug for Request {
    /// A summary: the body may nest thousands of levels, too deep to print recursively.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Request")
            .field("game_id", &self.game_id())
            .field("snakes", &self.snake_count())
            .finish_non_exhaustive()
    }
}

impl Request {
    /// Parses a request body as the Python handler does (server.py:649-663): an empty body or
    /// one over [`MAX_BODY`] bytes, a body `json.loads` refuses (Python's dialect: `NaN` and
    /// `Infinity` are accepted, the last of duplicate keys wins, a UTF-8 byte order mark is
    /// skipped), a body that is not an object, and an object whose `game` is not an object
    /// are all [`ApiError::Json`] (the Python server answers 400 and sends no move).
    pub fn from_json(text: &str) -> Result<Request, ApiError> {
        if text.is_empty() || text.len() > MAX_BODY {
            return Err(ApiError::Json(format!(
                "body of {} bytes (1 to {MAX_BODY} accepted)",
                text.len()
            )));
        }
        // json.loads on bytes detects the encoding: a UTF-8 BOM selects utf-8-sig.
        let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
        let root = py::parse(text).map_err(ApiError::Json)?;
        if !root.is_dict() || root.get("game").is_some_and(|g| !g.is_dict()) {
            return Err(ApiError::Json("not a Battlesnake request".into()));
        }
        Ok(Request { root })
    }

    /// [`Request::from_json`] of a `serde_json` value (handy for building requests in code).
    pub fn from_value(value: &serde_json::Value) -> Result<Request, ApiError> {
        Request::from_json(&value.to_string())
    }

    /// The parsed body.
    pub fn root(&self) -> &PyValue {
        &self.root
    }

    /// `game.id` when it is a string: the key of the caller's `/start` cache
    /// (`Mover._games`, server.py:459-470).
    pub fn game_id(&self) -> Option<&str> {
        self.root.get("game")?.get("id")?.as_str()
    }

    /// The number of snakes listed (`len(req["board"]["snakes"])`), `None` when Python's `len`
    /// would raise. At `/start` this is the game's starting snake count, which the caller
    /// should remember (`Mover.start`, server.py:459-470) and pass as `starting_snakes` to
    /// [`game_from_request`](crate::game_from_request) on every `/move`.
    pub fn snake_count(&self) -> Option<usize> {
        self.root
            .item("board")
            .and_then(|b| b.item("snakes"))
            .and_then(PyValue::len)
            .ok()
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
            "[1]",
            "null",
            r#"{"game": []}"#,
            r#"{"game": null}"#,
            "{\"a\": 1,}",
        ] {
            assert!(
                matches!(Request::from_json(text), Err(ApiError::Json(_))),
                "{text}"
            );
        }
        let long = format!(r#"{{"x": "{}"}}"#, "a".repeat(MAX_BODY));
        assert!(Request::from_json(&long).is_err());
        // Accepted by the handler (the conversion may still refuse them).
        for text in [
            "{}",
            "\u{FEFF}{}",
            r#"{"game": {}, "turn": NaN}"#,
            r#"{"a": 1, "a": 2}"#,
        ] {
            assert!(Request::from_json(text).is_ok(), "{text}");
        }
        let r =
            Request::from_json(r#"{"game": {"id": "g"}, "board": {"snakes": [1, 2]}}"#).unwrap();
        assert_eq!(r.game_id(), Some("g"));
        assert_eq!(r.snake_count(), Some(2));
    }
}
