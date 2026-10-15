//! The fallback move for requests the engine cannot play (`server.safe_move`, server.py:277-309).
//!
//! It never fails: whatever the JSON looks like, [`safe_move_json`] returns a move. It reads
//! only the board size, the snake bodies, `you` and the other snakes' heads and lengths, so it
//! works for games with rulesets the agent does not model and for requests the conversion
//! refused (a duplicate food, a missing turn, an unknown ruleset).

use std::collections::HashSet;

use serde_json::Value;
use slinky_rules::{DELTAS, Move};

use crate::Request;

/// What [`safe_move`] looks at.
struct View {
    width: i64,
    height: i64,
    wrapped: bool,
    /// Body segments that stay put next turn: all but the tail of every snake.
    blocked: HashSet<(i64, i64)>,
    /// `(head, length)` of every snake on the board.
    snakes: Vec<((i64, i64), i64, bool)>,
    head: (i64, i64),
    length: i64,
}

/// The move itself. Ties go to the lowest action id; with nothing to prefer it is `Up`.
fn choose(v: &View) -> Move {
    // Cells the other snakes (as long as us or longer) may step on next turn.
    let mut danger: HashSet<(i64, i64)> = HashSet::new();
    for &((hx, hy), len, is_you) in &v.snakes {
        if !is_you && len >= v.length {
            for (dx, dy) in DELTAS {
                danger.insert((hx + i64::from(dx), hy + i64::from(dy)));
            }
        }
    }
    let mut best = Move::Up;
    let mut best_score = -1;
    for (mv, (dx, dy)) in Move::ALL.into_iter().zip(DELTAS) {
        let mut p = (v.head.0 + i64::from(dx), v.head.1 + i64::from(dy));
        if v.wrapped {
            if v.width <= 0 || v.height <= 0 {
                continue;
            }
            p = (p.0.rem_euclid(v.width), p.1.rem_euclid(v.height));
        } else if !(0..v.width).contains(&p.0) || !(0..v.height).contains(&p.1) {
            continue;
        }
        let score = if v.blocked.contains(&p) {
            0
        } else {
            1 + i32::from(!danger.contains(&p))
        };
        if score > best_score {
            best = mv;
            best_score = score;
        }
    }
    best
}

/// A move that stays on the board and avoids body segments that stay put, preferring cells that
/// bigger-or-equal snakes cannot reach next turn. For requests that parsed.
pub fn safe_move(req: &Request) -> Move {
    let name = req
        .game
        .as_ref()
        .and_then(|g| g.get("ruleset")?.get("name")?.as_str());
    let snake_view = |s: &crate::Snake| {
        let head = s.head.or_else(|| s.body.first().copied());
        (
            head.map_or((0, 0), |c| (i64::from(c.x), i64::from(c.y))),
            s.length.unwrap_or(s.body.len() as i64),
        )
    };
    let mut blocked = HashSet::new();
    let mut snakes = Vec::new();
    for s in &req.board.snakes {
        let n = s.body.len().saturating_sub(1);
        blocked.extend(s.body[..n].iter().map(|c| (i64::from(c.x), i64::from(c.y))));
        let (head, len) = snake_view(s);
        snakes.push((head, len, s.id == req.you.id));
    }
    let (head, length) = snake_view(&req.you);
    choose(&View {
        width: req.board.width,
        height: req.board.height,
        wrapped: name.is_some_and(|n| n.contains("wrapped")),
        blocked,
        snakes,
        head,
        length,
    })
}

fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_i64)
}

fn point(v: Option<&Value>) -> Option<(i64, i64)> {
    let v = v?;
    Some((int(v.get("x"))?, int(v.get("y"))?))
}

/// [`safe_move`] on raw JSON, tolerant of anything: a missing or malformed field reads as
/// "nothing there" (no bodies, a head at the origin, a 0x0 board leaves only `Up`).
pub fn safe_move_json(text: &str) -> Move {
    let Ok(req) = serde_json::from_str::<Value>(text) else {
        return Move::Up;
    };
    let board = req.get("board");
    let you = req.get("you");
    let you_id = you.and_then(|y| y.get("id"));
    let mut blocked = HashSet::new();
    let mut snakes = Vec::new();
    let empty = Vec::new();
    for s in board
        .and_then(|b| b.get("snakes"))
        .and_then(Value::as_array)
        .unwrap_or(&empty)
    {
        let body: Vec<(i64, i64)> = s
            .get("body")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .filter_map(|c| point(Some(c)))
            .collect();
        blocked.extend(body[..body.len().saturating_sub(1)].iter().copied());
        let head = point(s.get("head"))
            .or(body.first().copied())
            .unwrap_or((0, 0));
        let len = int(s.get("length")).unwrap_or(body.len() as i64);
        snakes.push((head, len, s.get("id") == you_id && you_id.is_some()));
    }
    let head = point(you.and_then(|y| y.get("head")))
        .or_else(|| point(you.and_then(|y| y.get("body")?.get(0))))
        .unwrap_or((0, 0));
    let length = int(you.and_then(|y| y.get("length"))).unwrap_or(0);
    let name = req
        .get("game")
        .and_then(|g| g.get("ruleset")?.get("name")?.as_str());
    choose(&View {
        width: int(board.and_then(|b| b.get("width"))).unwrap_or(0),
        height: int(board.and_then(|b| b.get("height"))).unwrap_or(0),
        wrapped: name.is_some_and(|n| n.contains("wrapped")),
        blocked,
        snakes,
        head,
        length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
