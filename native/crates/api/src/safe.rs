//! The fallback move for requests the engine cannot play (`server.safe_move`, server.py:277-309).
//!
//! A line-by-line port with Python's semantics (see `py.rs`): the request's fields are read
//! as Python reads them (`5.0` and `5` are the same cell, a string `length` cannot be compared
//! with a number, a ruleset name list containing `"wrapped"` makes the board wrap), and
//! wherever Python's `safe_move` raises, the server answers 0, "up" (server.py:672-677), so
//! [`safe_move`] does too. It never fails and never panics: integer arithmetic is checked.

use std::cmp::Ordering;
use std::collections::HashSet;

use slinky_rules::{DELTAS, Move};

use crate::Request;
use crate::py::{Key, Num, PyResult, PyValue, number, py_cmp, py_eq, py_ge};

/// A move that stays on the board and avoids body segments that stay put, preferring cells that
/// bigger-or-equal snakes cannot reach next turn; "up" wherever Python's `safe_move` raises.
pub fn safe_move(req: &Request) -> Move {
    python_safe_move(req.root()).unwrap_or(Move::Up)
}

/// [`safe_move`] on the request's JSON text. A body the Python server refuses with HTTP 400
/// (see [`Request::from_json`]) gets "up" as well: some move has to be sent.
pub fn safe_move_json(text: &str) -> Move {
    Request::from_json(text).map_or(Move::Up, |req| safe_move(&req))
}

/// `s["body"][:-1]` for a body whose items are then subscripted.
fn all_but_last(body: &PyValue) -> PyResult<&[PyValue]> {
    match body {
        PyValue::List(a) => Ok(&a[..a.len().saturating_sub(1)]),
        // A str slice is a str; its characters cannot be subscripted with "x".
        PyValue::Str(s) if s.chars().count() <= 1 => Ok(&[]),
        other => Err(crate::py::PyErr(format!(
            "TypeError: cannot take the segments of a {}",
            other.type_name()
        ))),
    }
}

/// `needle in haystack` for the ruleset name (`"wrapped" in name`).
fn contains(haystack: &PyValue, needle: &str) -> PyResult<bool> {
    match haystack {
        PyValue::Str(s) => Ok(s.contains(needle)),
        PyValue::List(items) => {
            let n = PyValue::Str(needle.to_string());
            for item in items {
                if py_eq(item, &n)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PyValue::Dict(_) => Ok(haystack.get(needle).is_some()),
        other => Err(crate::py::PyErr(format!(
            "TypeError: argument of type '{}' is not iterable",
            other.type_name()
        ))),
    }
}

/// A point's key in the server's sets; `None` when a NaN makes it unequal to every member.
fn point_key(x: Key, y: Key) -> Option<(Key, Key)> {
    (x != Key::NaN && y != Key::NaN).then_some((x, y))
}

/// `n < bound` for a computed coordinate and a request value.
fn less(n: &Num, bound: &PyValue) -> PyResult<bool> {
    let n = match n {
        Num::Int(i) => PyValue::Int(*i),
        Num::Float(f) => PyValue::Float(*f),
        Num::Big { negative, digits } => PyValue::BigInt {
            negative: *negative,
            digits: digits.clone(),
        },
    };
    Ok(py_cmp(&n, bound)? == Some(Ordering::Less))
}

/// `server.safe_move` (server.py:277-309), statement by statement.
fn python_safe_move(req: &PyValue) -> PyResult<Move> {
    let (board, you) = (req.item("board")?, req.item("you")?);
    let (w, h) = (board.item("width")?, board.item("height")?);
    let snakes = board.item("snakes")?.subscripted_items()?;
    // blocked = {(p["x"], p["y"]) for s in board["snakes"] for p in s["body"][:-1]}
    let mut blocked: HashSet<(Key, Key)> = HashSet::new();
    for s in snakes {
        for p in all_but_last(s.item("body")?)? {
            let (x, y) = (p.item("x")?.key()?, p.item("y")?.key()?);
            blocked.extend(point_key(x, y));
        }
    }
    // danger = {(head.x + dx, head.y + dy) for s in snakes if s["id"] != you["id"] and
    //           s["length"] >= you["length"] for dx, dy in ACTION_DELTAS}
    let mut danger: HashSet<(Key, Key)> = HashSet::new();
    for s in snakes {
        if !py_eq(s.item("id")?, you.item("id")?)? && py_ge(s.item("length")?, you.item("length")?)?
        {
            for (dx, dy) in DELTAS {
                let head = s.item("head")?;
                let x = number(head.item("x")?)?.add(&Num::Int(dx.into()))?;
                let y = number(head.item("y")?)?.add(&Num::Int(dy.into()))?;
                danger.extend(point_key(x.key(), y.key()));
            }
        }
    }
    let head = you.item("head")?;
    let (hx, hy) = (head.item("x")?, head.item("y")?);
    // "wrapped" in req.get("game", {}).get("ruleset", {}).get("name", "")
    let empty = PyValue::Dict(Vec::new());
    let ruleset = req
        .get("game")
        .unwrap_or(&empty)
        .dict_get("ruleset")?
        .unwrap_or(&empty);
    let wrapped = match ruleset.dict_get("name")? {
        None => false,
        Some(name) => contains(name, "wrapped")?,
    };
    let mut best = Move::Up;
    let mut best_score = -1;
    for (mv, (dx, dy)) in Move::ALL.into_iter().zip(DELTAS) {
        let mut px = number(hx)?.add(&Num::Int(dx.into()))?;
        let mut py = number(hy)?.add(&Num::Int(dy.into()))?;
        if wrapped {
            px = px.rem(&number(w)?)?;
            py = py.rem(&number(h)?)?;
        } else {
            let ge0 = |n: &Num| {
                matches!(
                    n.compare(&Num::Int(0)),
                    Some(Ordering::Greater | Ordering::Equal)
                )
            };
            let inside = ge0(&px) && less(&px, w)? && ge0(&py) && less(&py, h)?;
            if !inside {
                continue;
            }
        }
        let key = point_key(px.key(), py.key());
        let score = match &key {
            Some(k) if blocked.contains(k) => 0,
            Some(k) if danger.contains(k) => 1,
            _ => 2,
        };
        if score > best_score {
            best = mv;
            best_score = score;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn req(head: (i64, i64), body: Vec<(i64, i64)>, other: Value, name: &str) -> String {
        let seg = |(x, y): (i64, i64)| json!({"x": x, "y": y});
        json!({
            "game": {"ruleset": {"name": name}},
            "turn": 3,
            "board": {"width": 11, "height": 11, "food": [], "snakes": [
                {"id": "me", "health": 90, "length": body.len(), "head": seg(head),
                 "body": body.iter().copied().map(seg).collect::<Vec<_>>()},
                other]},
            "you": {"id": "me", "health": 90, "length": body.len(), "head": seg(head),
                    "body": body.iter().copied().map(seg).collect::<Vec<_>>()},
        })
        .to_string()
    }

    fn foe(head: (i64, i64), len: usize) -> Value {
        let seg = json!({"x": head.0, "y": head.1});
        json!({"id": "foe", "health": 90, "length": len, "head": seg, "body": [seg]})
    }

    #[test]
    fn avoids_walls_bodies_and_big_heads() {
        // Top-left corner: up and left leave the board; down is our own neck.
        let t = req(
            (0, 10),
            vec![(0, 10), (0, 9), (0, 8)],
            foe((9, 9), 3),
            "standard",
        );
        assert_eq!(safe_move_json(&t), Move::Right);
        // The only free cell is next to a longer snake's head: still taken over a body or wall.
        let t = req(
            (0, 10),
            vec![(0, 10), (0, 9), (0, 8)],
            foe((2, 10), 5),
            "standard",
        );
        assert_eq!(safe_move_json(&t), Move::Right);
        // Two open cells, one next to a longer head: the other wins.
        let t = req(
            (5, 5),
            vec![(5, 5), (5, 4), (5, 3)],
            foe((7, 5), 4),
            "standard",
        );
        assert_eq!(safe_move_json(&t), Move::Up);
        // The tail moves away, so its cell is free.
        let t = req((1, 0), vec![(1, 0), (0, 0)], foe((9, 9), 1), "standard");
        assert_eq!(safe_move_json(&t), Move::Up);
    }

    #[test]
    fn wrapped_boards_wrap() {
        let t = req((0, 0), vec![(0, 0), (1, 0)], foe((9, 9), 1), "wrapped");
        assert_eq!(safe_move_json(&t), Move::Up);
        let t = req(
            (0, 0),
            vec![(0, 0), (0, 1), (0, 2)],
            foe((9, 9), 1),
            "wrapped",
        );
        assert_eq!(safe_move_json(&t), Move::Down); // the cell below is across the edge
    }

    #[test]
    fn never_fails() {
        for t in [
            "",
            "null",
            "[]",
            "{}",
            "{\"board\": 3}",
            "{\"you\": {\"head\": 1}}",
        ] {
            assert_eq!(safe_move_json(t), Move::Up);
        }
        // The typed path agrees with the JSON path.
        let t = req(
            (5, 5),
            vec![(5, 5), (5, 4), (5, 3)],
            foe((7, 5), 4),
            "standard",
        );
        assert_eq!(
            safe_move(&Request::from_json(&t).unwrap()),
            safe_move_json(&t)
        );
    }
}
