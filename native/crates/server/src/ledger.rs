//! Whether the engine applied each answer, and a JSON line per answer: a port of the Python
//! server's `MoveLedger` (server.py:511-606).
//!
//! The engine never says that an answer came too late: it moves the snake on as if it had
//! repeated its last move. Each game's next request (`/move` or `/end`) shows where the
//! snake's head went, so a head that moved another way than the answer means the answer
//! missed the game's timeout, from compute or the network. A late answer that happens to be
//! the default move goes unnoticed, so the count is a lower bound. A turn that skips one means
//! a request that never reached us.
//!
//! Each answer appends one line to the move log (when there is one), with the same fields,
//! order and formatting as Python's: `t`, `snake`, `game`, `turn`, `ruleset`, `map`, `snakes`,
//! `source`, `timeout`, `agent`, `move`, `ms`, `prev` (the verdict on this game's previous
//! answer: `applied`, `missed`, `gap` or null), then `wall` and `colo` when known, and
//! `"end": true` on the `/end` line (whose `prev` is null when our snake died before the end).
//! Detected misses are also logged as warnings.
//!
//! The request is read with Python's semantics (`int()` of the turn and the coordinates, the
//! game id's `str()`), including where Python's `record` raises: an unhashable game id with a
//! readable turn, an infinite float where an integer is read, or a truthy non-container
//! `board.snakes` make [`MoveLedger::record`] return an [`Abort`] (the HTTP handler logs it, as
//! Python logs the exception), after the same partial bookkeeping as Python.
//!
//! Lines are written by a background thread in the order they were recorded (the file is
//! opened in append mode for each batch, so log rotation works as with Python's per-line
//! `open`); [`MoveLedger::flush`] waits for them.

use crate::info;
use crate::ordered::OrderedMap;
use crate::pyfmt::{dumps, json_string, py_str_opt, rounded_repr};
use slinky_api::PyValue;
use slinky_api::py::{Num, python_int};
use slinky_rules::Move;
use std::cmp::Ordering as CmpOrdering;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// Games remembered at most (`server.MAX_GAMES`): the oldest is forgotten first.
pub const MAX_GAMES: usize = 1000;

/// The fate of a game's previous answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The head moved the way we answered.
    Applied,
    /// The head moved another way: the engine did not use our answer.
    Missed,
    /// The turn skipped one: a request never reached us.
    Gap,
}

impl Verdict {
    /// `"applied"`, `"missed"` or `"gap"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Applied => "applied",
            Verdict::Missed => "missed",
            Verdict::Gap => "gap",
        }
    }
}

/// An exception that escapes Python's `MoveLedger.record` (module docs); the message names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abort(pub String);

impl std::fmt::Display for Abort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What `int(x)` does with a request value.
enum IntError {
    /// `TypeError` / `ValueError` / `KeyError`: caught by the ledger.
    Caught,
    /// `OverflowError` (an infinite float): not caught.
    Overflow,
}

impl From<slinky_api::PyErr> for IntError {
    fn from(_: slinky_api::PyErr) -> Self {
        IntError::Caught
    }
}

/// Python's `int(v)`.
fn py_int(v: &PyValue) -> Result<Num, IntError> {
    match v {
        PyValue::Bool(b) => Ok(Num::Int(i128::from(*b))),
        PyValue::Int(i) => Ok(Num::Int(*i)),
        PyValue::BigInt { negative, digits } => Ok(Num::Big {
            negative: *negative,
            digits: digits.clone(),
        }),
        PyValue::Float(f) if f.is_nan() => Err(IntError::Caught),
        PyValue::Float(f) if f.is_infinite() => Err(IntError::Overflow),
        PyValue::Float(f) => {
            let t = f.trunc();
            const LIMIT: f64 = 1.7014118346046923e38; // 2^127: below, `t` is an exact i128
            if (-LIMIT..LIMIT).contains(&t) {
                Ok(Num::Int(t as i128))
            } else {
                let digits = format!("{:.0}", t.abs());
                Ok(Num::Big {
                    negative: t < 0.0,
                    digits,
                })
            }
        }
        PyValue::Str(s) => python_int(s).ok_or(IntError::Caught),
        _ => Err(IntError::Caught),
    }
}

/// Python's floor division of `i128`s (`None` on overflow or a zero divisor).
fn floor_div(a: i128, b: i128) -> Option<i128> {
    let q = a.checked_div(b)?;
    Some(if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    })
}

/// `-n` of an integer (an `i128` that overflows becomes a [`Num::Big`]; [`Num::add`]
/// normalises it again).
fn negate(n: &Num) -> Num {
    match n {
        Num::Int(i) => i.checked_neg().map_or_else(
            || Num::Big {
                negative: false,
                digits: i.unsigned_abs().to_string(),
            },
            Num::Int,
        ),
        Num::Big { negative, digits } => Num::Big {
            negative: !negative,
            digits: digits.clone(),
        },
        Num::Float(f) => Num::Float(-f),
    }
}

/// The magnitude of an integer as little-endian 64-bit limbs, no high zero limbs (empty for 0
/// or a float, which `py_int` never returns).
fn limbs(n: &Num) -> Vec<u64> {
    let mut out = match n {
        Num::Int(i) => {
            let u = i.unsigned_abs();
            vec![u as u64, (u >> 64) as u64]
        }
        Num::Big { digits, .. } => {
            let mut out: Vec<u64> = Vec::new();
            for d in digits.bytes() {
                // out = out * 10 + d
                let mut carry = u128::from(d.wrapping_sub(b'0') % 10);
                for l in out.iter_mut() {
                    let v = u128::from(*l) * 10 + carry;
                    *l = v as u64;
                    carry = v >> 64;
                }
                if carry > 0 {
                    out.push(carry as u64);
                }
            }
            out
        }
        Num::Float(_) => Vec::new(),
    };
    while out.last() == Some(&0) {
        out.pop();
    }
    out
}

/// Whether the integer `m` (not 0) divides the integer `a`, exactly at any size (binary long
/// division of the magnitudes: about `bits(a) * limbs(m)` word operations, a few million for
/// the largest numbers a request can hold).
fn divides(m: &Num, a: &Num) -> bool {
    let (m, a) = (limbs(m), limbs(a));
    if a.is_empty() {
        return true;
    }
    if m.is_empty() {
        return false;
    }
    // r < m always; one extra limb holds r * 2 + 1 before it is reduced.
    let mut r = vec![0u64; m.len() + 1];
    let ge = |r: &[u64]| -> bool {
        if r[m.len()] != 0 {
            return true;
        }
        for k in (0..m.len()).rev() {
            if r[k] != m[k] {
                return r[k] > m[k];
            }
        }
        true
    };
    for bit in (0..a.len() * 64).rev() {
        let mut carry = (a[bit / 64] >> (bit % 64)) & 1;
        for l in r.iter_mut() {
            let next = *l >> 63;
            *l = (*l << 1) | carry;
            carry = next;
        }
        if ge(&r) {
            let mut borrow = 0u64;
            for (k, l) in r.iter_mut().enumerate() {
                let sub = m.get(k).copied().unwrap_or(0);
                let (v, b1) = l.overflowing_sub(sub);
                let (v, b2) = v.overflowing_sub(borrow);
                *l = v;
                borrow = u64::from(b1 || b2);
            }
        }
    }
    r.iter().all(|&l| l == 0)
}

/// Whether the shortest signed step from `p` to `x` on an axis of size `w` is `delta`, with
/// Python's exact integers: `(x - p + w // 2) % w - w // 2 if w else x - p` (a wrapped board
/// crosses an edge in one step). For `w != 0` that is `delta` exactly when `delta + w // 2` is
/// a possible value of `% w` and `w` divides `x - p - delta`. The arguments are integers
/// (`py_int` returns no floats).
fn steps_by(x: &Num, p: &Num, w: &Num, delta: i128) -> bool {
    let Ok(e) = x.add(&negate(p)).and_then(|d| d.add(&Num::Int(-delta))) else {
        return false;
    };
    match w {
        Num::Int(0) => matches!(e, Num::Int(0)),
        Num::Int(w) => {
            let Some(t) = floor_div(*w, 2).and_then(|h| h.checked_add(delta)) else {
                return false;
            };
            let in_range = if *w > 0 {
                (0..*w).contains(&t)
            } else {
                *w < t && t <= 0
            };
            in_range && divides(&Num::Int(*w), &e)
        }
        // |w| >= 2^127: `delta + w // 2` is always a possible value of `% w`.
        Num::Big { .. } => divides(w, &e),
        Num::Float(_) => false,
    }
}

/// What the ledger remembers of a game's last answer: turn, move, head.
#[derive(Debug, Clone)]
struct Last {
    turn: Num,
    mv: Move,
    x: Num,
    y: Num,
}

enum WriterMsg {
    Line(String),
    Flush(mpsc::SyncSender<()>),
}

/// The move ledger (module docs). Shared by all snakes and requests.
pub struct MoveLedger {
    path: Option<PathBuf>,
    last: Mutex<OrderedMap<(String, String), Last>>,
    writer: Option<Mutex<mpsc::Sender<WriterMsg>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl MoveLedger {
    /// A ledger writing its lines to `path` (appending), or only judging answers with `None`.
    pub fn new(path: Option<PathBuf>) -> MoveLedger {
        let writer = path.clone().map(|p| {
            let (tx, rx) = mpsc::channel();
            std::thread::Builder::new()
                .name("move-log".into())
                .spawn(move || write_lines(&p, &rx))
                .map(|_| Mutex::new(tx))
        });
        let writer = match writer {
            Some(Ok(w)) => Some(w),
            Some(Err(e)) => {
                info!("cannot start the move log writer: {e}");
                None
            }
            None => None,
        };
        MoveLedger {
            path,
            last: Mutex::new(OrderedMap::new()),
            writer,
        }
    }

    /// The move log file, if any.
    pub fn path(&self) -> Option<&PathBuf> {
        self.path.as_ref()
    }

    /// The fate of this game's previous answer, from the head in this request
    /// (`MoveLedger.verdict`).
    pub fn verdict(&self, snake: &str, req: &PyValue) -> Result<Option<Verdict>, Abort> {
        // game, turn = req["game"]["id"], int(req["turn"]); then the head and the board.
        let read = || -> Result<(&PyValue, Num, Num, Num, Num, Num), IntError> {
            let game = req.item("game")?.item("id")?;
            let turn = py_int(req.item("turn")?)?;
            let head = req.item("you")?.item("head")?;
            let board = req.item("board")?;
            let x = py_int(head.item("x")?)?;
            let y = py_int(head.item("y")?)?;
            let w = py_int(board.item("width")?)?;
            let h = py_int(board.item("height")?)?;
            Ok((game, turn, x, y, w, h))
        };
        let (game, turn, x, y, w, h) = match read() {
            Ok(v) => v,
            Err(IntError::Caught) => return Ok(None),
            Err(IntError::Overflow) => {
                return Err(Abort(
                    "OverflowError: cannot convert float infinity to integer".into(),
                ));
            }
        };
        // `self._last.get((snake, game))` with the raw id; the keys hold `str(id)`, so only a
        // string id can match, and a list or dict id is unhashable.
        let prev = match game {
            PyValue::Str(id) => lock(&self.last)
                .get(&(snake.to_string(), id.clone()))
                .cloned(),
            PyValue::List(_) | PyValue::Dict(_) => {
                return Err(Abort(format!(
                    "TypeError: unhashable type: '{}'",
                    game.type_name()
                )));
            }
            _ => None,
        };
        let Some(prev) = prev else {
            return Ok(None);
        };
        let next_turn = prev.turn.add(&Num::Int(1)).ok();
        if next_turn.and_then(|t| turn.compare(&t)) != Some(CmpOrdering::Equal) {
            return Ok(Some(Verdict::Gap));
        }
        let (dx, dy) = prev.mv.delta();
        Ok(Some(
            if steps_by(&x, &prev.x, &w, i128::from(dx))
                && steps_by(&y, &prev.y, &h, i128::from(dy))
            {
                Verdict::Applied
            } else {
                Verdict::Missed
            },
        ))
    }

    /// Judges the previous answer and remembers this one (`mv`; `None` at `/end`), then writes
    /// the line (`MoveLedger.record`). `ms` is our compute time, `wall_ms` the time from the
    /// request reaching the handler to the answer, `colo` the Cloudflare data centre.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        snake: &str,
        req: &PyValue,
        mv: Option<Move>,
        agent: &str,
        ms: f64,
        end: bool,
        wall_ms: Option<f64>,
        colo: Option<&str>,
    ) -> Result<Option<Verdict>, Abort> {
        let mut verdict = self.verdict(snake, req)?;
        if end && verdict == Some(Verdict::Gap) {
            verdict = None; // our snake died earlier; nothing was asked since
        }
        let empty = PyValue::Dict(Vec::new());
        let game = req.get("game").filter(|g| g.is_dict()).unwrap_or(&empty);
        let game_id = py_str_opt(game.get("id")).map_err(|e| Abort(e.to_string()))?;
        let key = (snake.to_string(), game_id);
        // head = req["you"]["head"]; last = (int(turn), move, int(x), int(y))
        let read = || -> Result<(Num, Num, Num), IntError> {
            let head = req.item("you")?.item("head")?;
            Ok((
                py_int(req.item("turn")?)?,
                py_int(head.item("x")?)?,
                py_int(head.item("y")?)?,
            ))
        };
        let last = match read() {
            Ok(v) => Some(v),
            Err(IntError::Caught) => None,
            Err(IntError::Overflow) => {
                return Err(Abort(
                    "OverflowError: cannot convert float infinity to integer".into(),
                ));
            }
        };
        {
            let mut map = lock(&self.last);
            match (end, last, mv) {
                (false, Some((turn, x, y)), Some(mv)) => {
                    if !map.contains_key(&key) && map.len() >= MAX_GAMES {
                        map.pop_first(); // games whose /end never came
                    }
                    map.insert(key.clone(), Last { turn, mv, x, y });
                }
                _ => {
                    map.remove(&key);
                }
            }
        }
        if let Some(v @ (Verdict::Missed | Verdict::Gap)) = verdict {
            info!(
                "{} game {} turn {}: our previous answer was not applied ({}; timeout {} ms)",
                snake,
                key.1,
                py_str_opt(req.get("turn")).unwrap_or_default(),
                v.as_str(),
                py_str_opt(game.get("timeout")).unwrap_or_default(),
            );
        }
        let Some(writer) = &self.writer else {
            return Ok(verdict);
        };
        let ruleset = game
            .get("ruleset")
            .filter(|r| r.is_dict())
            .unwrap_or(&empty);
        let board = req.get("board").filter(|b| b.is_dict()).unwrap_or(&empty);
        // len(board.get("snakes") or [])
        let snakes = match board.get("snakes") {
            Some(s) if s.truthy() => s.len().map_err(|e| Abort(e.0))?,
            _ => 0,
        };
        let dump = |v: Option<&PyValue>| -> Result<String, Abort> {
            v.map_or_else(
                || Ok("null".to_string()),
                |v| dumps(v).map_err(|e| Abort(e.to_string())),
            )
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let mut line = format!(
            "{{\"t\": {}, \"snake\": {}, \"game\": {}, \"turn\": {}, \"ruleset\": {}, \
             \"map\": {}, \"snakes\": {}, \"source\": {}, \"timeout\": {}, \"agent\": {}, \
             \"move\": {}, \"ms\": {}, \"prev\": {}",
            rounded_repr(now, 3),
            json_string(snake),
            json_string(&key.1),
            dump(req.get("turn"))?,
            dump(ruleset.get("name"))?,
            dump(game.get("map"))?,
            snakes,
            dump(game.get("source"))?,
            dump(game.get("timeout"))?,
            json_string(agent),
            mv.map_or_else(|| "null".to_string(), |m| json_string(m.name())),
            rounded_repr(ms, 1),
            verdict.map_or_else(|| "null".to_string(), |v| json_string(v.as_str())),
        );
        if let Some(w) = wall_ms {
            line.push_str(&format!(", \"wall\": {}", rounded_repr(w, 1)));
        }
        if let Some(c) = colo.filter(|c| !c.is_empty()) {
            line.push_str(&format!(", \"colo\": {}", json_string(c)));
        }
        if end {
            line.push_str(", \"end\": true");
        }
        line.push('}');
        let _ = lock(writer).send(WriterMsg::Line(line));
        Ok(verdict)
    }

    /// Waits until every line recorded so far is written.
    pub fn flush(&self) {
        if let Some(writer) = &self.writer {
            let (tx, rx) = mpsc::sync_channel(1);
            if lock(writer).send(WriterMsg::Flush(tx)).is_ok() {
                let _ = rx.recv();
            }
        }
    }
}

/// The writer thread: appends each batch of lines, answers flushes once all earlier lines are
/// written.
fn write_lines(path: &PathBuf, rx: &mpsc::Receiver<WriterMsg>) {
    while let Ok(first) = rx.recv() {
        let mut batch = Vec::new();
        let mut flushes = Vec::new();
        let mut take = |m: WriterMsg| match m {
            WriterMsg::Line(l) => batch.push(l),
            WriterMsg::Flush(f) => flushes.push(f),
        };
        take(first);
        while let Ok(m) = rx.try_recv() {
            take(m);
        }
        if !batch.is_empty() {
            let mut text = batch.join("\n");
            text.push('\n');
            let res = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| f.write_all(text.as_bytes()));
            if let Err(e) = res {
                info!("cannot write the move log {}: {e}", path.display());
            }
        }
        for f in flushes {
            let _ = f.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use slinky_api::py::parse;

    fn value(v: &Value) -> PyValue {
        parse(&v.to_string()).unwrap()
    }

    /// The parts of a request the ledger reads: our head at `turn`.
    fn ledger_request(turn: i64, x: i64, y: i64, game: &str, w: i64, h: i64) -> PyValue {
        value(&json!({
            "game": {"id": game, "timeout": 500, "source": "ladder", "map": "standard",
                     "ruleset": {"name": "standard"}},
            "turn": turn,
            "board": {"width": w, "height": h, "snakes": [{}, {}]},
            "you": {"head": {"x": x, "y": y}},
        }))
    }

    fn req(turn: i64, x: i64, y: i64) -> PyValue {
        ledger_request(turn, x, y, "g", 11, 11)
    }

    fn rec(
        l: &MoveLedger,
        snake: &str,
        r: &PyValue,
        mv: Option<Move>,
        ms: f64,
    ) -> Option<&'static str> {
        let end = mv.is_none();
        let agent = if end { "" } else { "az" };
        l.record(snake, r, mv, agent, ms, end, None, None)
            .unwrap()
            .map(Verdict::as_str)
    }

    fn temp_path(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("slinky-server-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        dir
    }

    /// test_server.py::test_move_ledger_judges_each_answer
    #[test]
    fn judges_each_answer() {
        crate::log::set_quiet(true);
        let path = temp_path("judge.jsonl");
        let l = MoveLedger::new(Some(path.clone()));
        use Move::*;
        assert_eq!(rec(&l, "s", &req(0, 5, 5), Some(Up), 30.0), None); // nothing before
        assert_eq!(
            rec(&l, "s", &req(1, 5, 6), Some(Left), 31.0),
            Some("applied")
        ); // went up
        assert_eq!(
            rec(&l, "s", &req(2, 4, 6), Some(Left), 32.0),
            Some("applied")
        );
        // We said left; the head went up instead: the engine used its default move.
        assert_eq!(
            rec(&l, "s", &req(3, 4, 7), Some(Down), 33.0),
            Some("missed")
        );
        // Turn 4's request never reached us: turn 5 arrives after turn 3.
        assert_eq!(rec(&l, "s", &req(5, 4, 5), Some(Up), 34.0), Some("gap"));
        // Games and snakes are judged apart.
        assert_eq!(rec(&l, "other", &req(6, 0, 0), Some(Up), 1.0), None);
        let h = ledger_request(0, 1, 1, "h", 11, 11);
        assert_eq!(rec(&l, "s", &h, Some(Right), 1.0), None);
        // /end judges the last answer (the snake ran into the wall as told), forgets the game.
        assert_eq!(rec(&l, "s", &req(6, 4, 6), None, 0.0), Some("applied"));
        assert_eq!(rec(&l, "s", &req(7, 4, 7), Some(Up), 1.0), None);
        // Our snake died after turn 7: /end comes at turn 20 and is not a gap.
        assert_eq!(rec(&l, "s", &req(20, 4, 8), None, 0.0), None);
        l.flush();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<Value> = text
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let prevs: Vec<Value> = lines.iter().map(|r| r["prev"].clone()).collect();
        let want = [
            json!(null),
            json!("applied"),
            json!("applied"),
            json!("missed"),
            json!("gap"),
            json!(null),
            json!(null),
            json!("applied"),
            json!(null),
            json!(null),
        ];
        assert_eq!(prevs, want);
        let mut third = lines[3].clone();
        third["t"] = json!(0);
        assert_eq!(
            third,
            json!({
                "t": 0, "snake": "s", "game": "g", "turn": 3, "ruleset": "standard",
                "map": "standard", "snakes": 2, "source": "ladder", "timeout": 500,
                "agent": "az", "move": "down", "ms": 33.0, "prev": "missed",
            })
        );
        assert_eq!(lines[7]["end"], json!(true));
        assert_eq!(lines[7]["move"], json!(null));
        // The text is Python's json.dumps form, key order included.
        let raw = text.lines().nth(3).unwrap();
        let tail = &raw[raw.find(", \"snake\"").unwrap()..];
        assert_eq!(
            tail,
            ", \"snake\": \"s\", \"game\": \"g\", \"turn\": 3, \"ruleset\": \"standard\", \
             \"map\": \"standard\", \"snakes\": 2, \"source\": \"ladder\", \"timeout\": 500, \
             \"agent\": \"az\", \"move\": \"down\", \"ms\": 33.0, \"prev\": \"missed\"}"
        );
    }

    /// test_server.py::test_move_ledger_wraps_and_survives_junk
    #[test]
    fn wraps_and_survives_junk() {
        crate::log::set_quiet(true);
        let l = MoveLedger::new(None);
        rec(&l, "s", &req(0, 10, 3), Some(Move::Right), 1.0);
        let v = |r: &PyValue| l.verdict("s", r).unwrap().map(Verdict::as_str);
        assert_eq!(v(&req(1, 0, 3)), Some("applied")); // across the wrapped edge
        assert_eq!(v(&req(1, 9, 3)), Some("missed"));
        rec(
            &l,
            "s",
            &ledger_request(0, 4, 0, "w", 11, 11),
            Some(Move::Down),
            1.0,
        );
        let w = ledger_request(1, 4, 10, "w", 11, 11);
        assert_eq!(v(&w), Some("applied")); // y 0 -> 10: down
        for junk in [
            json!({}),
            json!({"game": 3}),
            json!({"game": {"id": "g"}, "turn": "x"}),
            json!({"board": 3}),
        ] {
            assert_eq!(rec(&l, "s", &value(&junk), Some(Move::Up), 1.0), None);
        }
    }

    #[test]
    fn reads_requests_as_python_does() {
        crate::log::set_quiet(true);
        let l = MoveLedger::new(None);
        // A numeric-string turn and float coordinates are read with int().
        let r = value(
            &json!({"game": {"id": "g"}, "turn": "4", "you": {"head": {"x": 2.9, "y": 3}},
                              "board": {"width": 11, "height": 11}}),
        );
        assert_eq!(rec(&l, "s", &r, Some(Move::Up), 1.0), None);
        let r = value(
            &json!({"game": {"id": "g"}, "turn": 5.5, "you": {"head": {"x": 2, "y": 4}},
                              "board": {"width": "11", "height": 11}}),
        );
        assert_eq!(l.verdict("s", &r).unwrap(), Some(Verdict::Applied));
        // A non-string id never matches its own str() key: no verdict.
        let r = value(
            &json!({"game": {"id": 7}, "turn": 0, "you": {"head": {"x": 2, "y": 4}},
                              "board": {"width": 11, "height": 11}}),
        );
        rec(&l, "s", &r, Some(Move::Up), 1.0);
        let mut next = r;
        if let PyValue::Dict(items) = &mut next {
            items[1].1 = PyValue::Int(1);
        }
        assert_eq!(l.verdict("s", &next).unwrap(), None);
        // An unhashable id with a readable request raises, as in Python.
        let r = value(
            &json!({"game": {"id": [1]}, "turn": 0, "you": {"head": {"x": 2, "y": 4}},
                              "board": {"width": 11, "height": 11}}),
        );
        assert!(
            l.record("s", &r, Some(Move::Up), "az", 1.0, false, None, None)
                .is_err()
        );
        // Negative and zero board sizes follow Python's floor arithmetic.
        let i = Num::Int;
        assert!(steps_by(&i(0), &i(10), &i(11), 1));
        assert!(steps_by(&i(10), &i(0), &i(11), -1));
        assert!(steps_by(&i(3), &i(2), &i(0), 1) && !steps_by(&i(3), &i(1), &i(0), 1));
        assert!(!steps_by(&i(3), &i(1), &i(-4), 1)); // the step is 2
        assert!(steps_by(&i(3), &i(2), &i(-4), 1) && steps_by(&i(2), &i(3), &i(-4), -1));
        // (1 - 0 + 1) % 2 - 1 == -1 and (0 - 1 + 1) % 2 - 1 == -1: on a width of 2 every
        // step is 0 or -1, as in Python.
        assert!(!steps_by(&i(1), &i(0), &i(2), 1) && steps_by(&i(0), &i(1), &i(2), -1));
        assert!(steps_by(&i(1), &i(0), &i(2), -1));
        assert!(steps_by(&i(0), &i(0), &i(1), 0) && !steps_by(&i(1), &i(0), &i(1), 1));
        // Beyond i128, exactly (Python's integers).
        let big = |negative: bool, digits: &str| Num::Big {
            negative,
            digits: digits.into(),
        };
        let e40 = big(false, "10000000000000000000000000000000000000000");
        let e40p1 = big(false, "10000000000000000000000000000000000000001");
        assert!(steps_by(&i(4), &i(3), &e40, 1));
        assert!(steps_by(
            &i(4),
            &i(3),
            &big(true, "10000000000000000000000000000000000000000"),
            1
        ));
        assert!(steps_by(&e40p1, &e40, &i(11), 1) && !steps_by(&e40p1, &e40, &i(11), -1));
        assert!(steps_by(&e40, &e40p1, &e40, -1));
        // x - p = 10**40 + 1 on a width of 10**40 wraps to a step of 1.
        assert!(steps_by(&e40p1, &i(0), &e40, 1));
        assert!(!steps_by(&i(i128::MIN), &i(i128::MAX), &i(0), 1));
        // (d - 1) % -1 + 1 == 1: on a width of -1 every step is 1.
        assert!(steps_by(&i(i128::MAX), &i(i128::MIN), &i(-1), 1));
        assert!(!steps_by(&i(i128::MAX), &i(i128::MIN), &i(-1), 0));
        assert!(divides(
            &i(3),
            &big(false, "300000000000000000000000000000000000000000000000000")
        ));
        assert!(!divides(
            &e40p1,
            &big(false, "300000000000000000000000000000000000000000000000000")
        ));
        assert!(divides(
            &e40,
            &big(true, "300000000000000000000000000000000000000000000000000")
        ));
        // A truthy non-container snakes list raises after the bookkeeping (with a log file).
        let path = temp_path("junk.jsonl");
        let l = MoveLedger::new(Some(path.clone()));
        let r = value(
            &json!({"game": {"id": "g"}, "turn": 0, "you": {"head": {"x": 2, "y": 4}},
                              "board": {"width": 11, "height": 11, "snakes": 3}}),
        );
        assert!(
            l.record("s", &r, Some(Move::Up), "az", 1.0, false, None, None)
                .is_err()
        );
        let r = value(
            &json!({"game": {"id": "g"}, "turn": 1, "you": {"head": {"x": 2, "y": 5}},
                              "board": {"width": 11, "height": 11, "snakes": []}}),
        );
        assert_eq!(
            l.record(
                "s",
                &r,
                Some(Move::Up),
                "az",
                1.0,
                false,
                Some(2.25),
                Some("SEA")
            ),
            Ok(Some(Verdict::Applied))
        );
        l.flush();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(text.lines().count(), 1);
        assert!(
            text.ends_with(
                "\"ms\": 1.0, \"prev\": \"applied\", \"wall\": 2.2, \"colo\": \"SEA\"}\n"
            ),
            "{text}"
        );
    }
}
