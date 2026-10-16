//! Request -> game: the config, the mode and the state the search starts from.
//!
//! Ported from `src/slinky/server.py` (`game_config` 161-226, `_setting` 149-158,
//! `request_to_engine` 120-146, `royale_sides` 237-261, `request_state` 264-274, `check_sizes`
//! 347-356, `Mover.move` 476-496) and `src/slinky/engine_json.py` (`state_from_engine` 95-151,
//! `body_to_countdown` 35-42, `_direction` 73-92), see SPEC.md section 6. The rules crate's
//! `GameState::set_snake` does the body -> countdown step (stacked segments take the largest
//! countdown).
//!
//! What the Python does that differs from "the obvious reading" of the API:
//!
//! * The snake count of the config is `max(starting_snakes, snakes listed)`. The request lists
//!   only the living snakes, so a caller without the game's `/start` has to pass `None`, and
//!   then the count is the number of *living* snakes: a four-player standard game with two
//!   snakes left is read as a duel (and told the duel mode), and a royale or constrictor game
//!   keeps its mode but loses seats (SPEC.md section 8, point 1). The dead seats are appended
//!   as eliminated snakes (cause `snake-collision`, turn 0, empty body, health 0).
//! * `you` is seat 0 and the other snakes keep the API order (a stable sort on `id != you.id`,
//!   with Python's `==`: the ids `1`, `1.0` and `true` are equal).
//! * Every royale spelling (royale ruleset on the royale or standard map, standard ruleset on the
//!   royale map) becomes the royale ruleset; constrictor on the `empty` map becomes plain
//!   constrictor; constrictor on the royale map, the standard ruleset on the empty map, and any
//!   other ruleset or map have no config.
//! * Settings are read with [`setting`] (Python's `int()`: an integer, a truncated float, a
//!   numeric string in any script, within +-2^20; anything else keeps the default, but an
//!   infinite float raises). Only `hazardDamagePerTurn` (royale, or a board that lists
//!   hazards) and the royale shrink period (royale only; the nested
//!   `royale.shrinkEveryNTurns` wins over the flat one) reach the config. The food settings
//!   are never read, and `max_turns` is `None`.
//! * Only the fields the server reads are read: `you.id`, and the `id`, `health` and `body` of
//!   the board's snakes. `health` and `turn` are stored as numpy int32 (a float is truncated, a
//!   numeric string is parsed, a bool is 0 or 1). Off-board body segments and hazards are
//!   dropped whatever their type or size (Python compares before it indexes); on-board ones
//!   must be integers. A food cell is a numpy index: `-1` is the last column or row, beyond
//!   the board or listed twice is an error.
//! * Royale shrink sides are rebuilt from the listed hazards (see [`royale_sides`]).
//!
//! Remaining differences from Python, all on inputs no Battlesnake engine sends: a head mixing
//! a string with a float coordinate (numpy converts the pair as one array, which can fail
//! where the element-wise store here does not), and the few integer operations beyond 128
//! bits that `py.rs` does not do exactly.

use slinky_rules::{
    Cause, GameConfig, GameState, MAX_CELLS, MAX_SIDES, MAX_SNAKES, Mode, Move, Ruleset,
    UNKNOWN_SIDE,
};

use crate::py::{
    Num, PyErr, PyValue, in_range, int32, number, numpy_index, py_cmp, py_eq, python_int,
};
use crate::{ApiError, Coord, Request};

/// Settings outside this range keep their default (server.py:107, `MAX_SETTING`).
pub const MAX_SETTING: i64 = 1 << 20;

/// Largest board side and seat count the Python server makes a config for (server.py:111).
const PY_MAX_SIDE: i128 = 25;
const PY_MAX_SNAKES: usize = 8;

impl From<PyErr> for ApiError {
    fn from(e: PyErr) -> Self {
        ApiError::BadRequest(e.0)
    }
}

/// Everything the agent needs from a request.
#[derive(Debug, Clone, PartialEq)]
pub struct Game {
    /// The normalised game: board, seat count, ruleset, hazard damage, shrink period.
    pub config: GameConfig,
    /// The state at the request's turn, `you` in seat 0 (the other living snakes next, in API
    /// order, then the dead seats).
    pub state: GameState,
    /// The mode the network is told (`modes.mode_of`, `agents._told_mode`); `config.mode()`.
    pub mode: Mode,
}

impl Game {
    /// Whether the board is the 11x11 the trained networks read (`agents.check_game` refuses
    /// every other size). The conversion itself accepts any board the rules crate supports.
    pub fn is_network_board(&self) -> bool {
        self.config.width() == 11 && self.config.height() == 11
    }
}

/// The request's game as the agent plays it: config, mode and state.
///
/// `starting_snakes` is the number of snakes at the game's `/start` (see
/// [`Request::snake_count`]), or `None` when the caller does not know it, in which case the
/// seat count is the number of living snakes (module docs).
///
/// This is `server.Mover.move` up to the search: [`check_sizes`], the `/start` cache lookup by
/// `game.id` (which raises for a list or object id), [`game_config`], the mode (refused for
/// fewer than 2 seats), then [`request_state`]. Every request Python answers with
/// `safe_move` is an error here.
pub fn game_from_request(req: &Request, starting_snakes: Option<usize>) -> Result<Game, ApiError> {
    check_sizes(req)?;
    if let Some(id) = req.root().get("game").and_then(|g| g.get("id")) {
        id.key()?; // `self._games.get(id)` needs a hashable id
    }
    let config = game_config(req, starting_snakes)?;
    let state = request_state(req, &config)?;
    Ok(Game {
        mode: config.mode(),
        config,
        state,
    })
}

/// `len > factor * cells + offset` with Python's arithmetic on the board's cell count.
fn len_exceeds(len: usize, cells: &Num, factor: i128, offset: i128) -> bool {
    match cells {
        Num::Int(c) => {
            (len as i128) > c.saturating_mul(factor).saturating_add(offset) // saturates only far beyond any length
        }
        Num::Float(f) => (len as f64) > *f * factor as f64 + offset as f64,
        Num::Big { negative, .. } => *negative,
    }
}

/// Rejects a request that lists more than its board can hold (`server.check_sizes`,
/// server.py:347-356): more food than cells, more than four hazards per cell, or a body longer
/// than the board plus three. Also an error wherever Python's checks raise (no board, a
/// non-numeric size, lists that have no length).
pub fn check_sizes(req: &Request) -> Result<(), ApiError> {
    let board = req.root().item("board")?;
    let (w, h) = (board.item("width")?, board.item("height")?);
    let cells = match (w.num(), h.num()) {
        (Some(a), Some(b)) => a.mul(&b)?,
        _ => {
            return Err(ApiError::BadRequest(format!(
                "TypeError: board size {} x {}",
                w.type_name(),
                h.type_name()
            )));
        }
    };
    let too_much =
        || ApiError::BadRequest("more food, hazards or body than the board holds".into());
    if len_exceeds(board.item("food")?.len()?, &cells, 1, 0) {
        return Err(too_much());
    }
    let hazards = board.get("hazards").map_or(Ok(0), PyValue::len)?;
    if len_exceeds(hazards, &cells, 4, 0) {
        return Err(too_much());
    }
    for s in board.item("snakes")?.subscripted_items()? {
        if len_exceeds(s.item("body")?.len()?, &cells, 1, 3) {
            return Err(too_much());
        }
    }
    Ok(())
}

/// The integer `settings[key]`, or `default` (`server._setting`, server.py:149-158).
///
/// `int(value)` in Python: a JSON integer, a float (truncated toward zero, like `int(14.9)`) or
/// a numeric string (the engine CLI sends strings; any Unicode decimal digits, underscores
/// between digits); a boolean, `null`, a list, an object, junk, NaN or a value outside
/// +-[`MAX_SETTING`] keeps `default`. A `settings` that is not an object also keeps it. An
/// infinite float raises `OverflowError` in Python (not caught there), an error here.
pub fn setting(settings: Option<&PyValue>, key: &str, default: i32) -> Result<i32, ApiError> {
    let n = match settings.and_then(|s| s.get(key)) {
        Some(PyValue::Int(i)) => *i,
        Some(PyValue::Float(f)) if f.is_infinite() => {
            return Err(ApiError::BadRequest(format!(
                "OverflowError: setting {key} is an infinite float"
            )));
        }
        Some(PyValue::Float(f)) if f.is_finite() && f.abs() < 1e30 => f.trunc() as i128,
        Some(PyValue::Str(s)) => match python_int(s) {
            Some(Num::Int(i)) => i,
            _ => return Ok(default),
        },
        _ => return Ok(default),
    };
    Ok(match i32::try_from(n) {
        Ok(v) if i64::from(v).abs() <= MAX_SETTING => v,
        _ => default,
    })
}

/// `lo <= x <= hi`, evaluated as Python's chained comparison.
fn in_closed(x: &PyValue, lo: i128, hi: i128) -> Result<bool, ApiError> {
    Ok(in_range(x, lo, hi)? || (py_eq(x, &PyValue::Int(hi))? && hi >= lo))
}

/// The [`GameConfig`] of a request's game (`server.game_config`, server.py:161-226).
///
/// `starting_snakes` is the snake count at `/start` if known. Errors with
/// [`ApiError::Unsupported`] where Python has no config or no mode for the game: another
/// ruleset or map, a board outside 1..=25, a shrink period below 1, and also where the
/// rules crate or the search cannot play it (a board larger than 11x11, fewer than 2 or more
/// than 4 seats; Python plays some of those with its heuristic agent). Errors with
/// [`ApiError::BadRequest`] where Python raises.
pub fn game_config(req: &Request, starting_snakes: Option<usize>) -> Result<GameConfig, ApiError> {
    let root = req.root();
    let empty = PyValue::Dict(Vec::new());
    let game = root.get("game").unwrap_or(&empty);
    // `game.get("ruleset", {}) or {}`; a truthy non-dict then fails on `.get`.
    let ruleset = match game.get("ruleset") {
        Some(v) if v.truthy() => v,
        _ => &empty,
    };
    let name = match ruleset.dict_get("name")? {
        None => Some("standard"),
        Some(v) => v.as_str(),
    };
    let map = match game.get("map") {
        None => Some("standard"),
        Some(v) => v.as_str(),
    };
    let unsupported = |why: String| Err(ApiError::Unsupported(why));
    let (name, map) = match (name, map) {
        (
            Some(n @ ("standard" | "royale" | "constrictor")),
            Some(m @ ("standard" | "royale" | "empty")),
        ) => (n, m),
        (n, m) => return unsupported(format!("ruleset {n:?} on map {m:?}")),
    };
    if map == "empty" && name != "constrictor" {
        return unsupported("the empty map is only played with constrictor".into());
    }
    let royale = name == "royale" || map == "royale";
    if royale && name == "constrictor" {
        return unsupported("constrictor on the royale map".into());
    }
    let board = root.item("board")?;
    let (w, h) = (board.item("width")?, board.item("height")?);
    let n = starting_snakes
        .unwrap_or(0)
        .max(board.item("snakes")?.len()?);
    if !(in_closed(w, 1, PY_MAX_SIDE)?
        && in_closed(h, 1, PY_MAX_SIDE)?
        && (1..=PY_MAX_SNAKES).contains(&n))
    {
        return unsupported(format!(
            "board or seats outside Python's limits ({n} seats)"
        ));
    }
    // A float size passes Python's checks but fails when the state is built (np.zeros).
    let side = |v: &PyValue| match v {
        PyValue::Int(i) => Ok(*i as usize),
        PyValue::Bool(b) => Ok(usize::from(*b)),
        other => Err(ApiError::BadRequest(format!(
            "TypeError: board size of type {}",
            other.type_name()
        ))),
    };
    let (w, h) = (side(w)?, side(h)?);
    if n < 2 {
        return unsupported(format!("{n} snake: no game mode for fewer than 2 seats"));
    }
    if n > MAX_SNAKES {
        return unsupported(format!("{n} snakes: the search plays at most {MAX_SNAKES}"));
    }
    let ruleset_kind = if royale {
        Ruleset::Royale
    } else if name == "constrictor" {
        Ruleset::Constrictor
    } else {
        Ruleset::Standard
    };
    let mut config =
        GameConfig::new(w, h, n, ruleset_kind).map_err(|e| ApiError::Unsupported(e.to_string()))?;
    // `ruleset.get("settings") or {}`
    let settings = ruleset.get("settings").filter(|s| s.truthy());
    if royale || board.get("hazards").is_some_and(PyValue::truthy) {
        config = config.with_hazard_damage(setting(
            settings,
            "hazardDamagePerTurn",
            config.hazard_damage_per_turn(),
        )?);
    }
    if royale {
        // The API nests the period under "royale"; the engine CLI's flat name is read too.
        let flat = setting(settings, "shrinkEveryNTurns", config.shrink_every_n_turns())?;
        let nested = setting(
            settings.and_then(|s| s.get("royale")),
            "shrinkEveryNTurns",
            flat,
        )?;
        config = config
            .with_shrink_every(nested)
            .map_err(|e| ApiError::Unsupported(e.to_string()))?;
    }
    Ok(config)
}

/// The royale shrink sides that give the zone shown at `turn` (`server.royale_sides`,
/// server.py:237-261).
///
/// The zone at `turn` has had `turn / shrink_every_n_turns` shrinks (capped at `W + H - 1`); the
/// listed hazards say how many came from each side (the cells outside the bounding box of the
/// safe cells), and their order changes nothing. The first `shrinks` entries are those sides,
/// left (0), right (1), bottom (2), top (3), in that order; the rest are
/// [`UNKNOWN_SIDE`], which moves no edge, so a search assumes the zone stays put after the
/// shrinks already shown. A board with no safe cell is read as a left-column sweep
/// (`shrinks >= W`) or a bottom-row sweep. Entries past `W + H - 1` are also [`UNKNOWN_SIDE`].
/// Hazards off the board are ignored.
pub fn royale_sides(hazards: &[Coord], turn: i32, config: &GameConfig) -> [i8; MAX_SIDES] {
    let (w, h) = (config.width(), config.height());
    let mut hazard = [false; MAX_CELLS];
    for c in hazards {
        if let Some(cell) = config.cell(c.x, c.y) {
            hazard[cell] = true;
        }
    }
    let shrinks = ((turn.max(0) / config.shrink_every_n_turns()) as usize).min(config.num_sides());
    // Bounding box of the safe (non-hazard) cells.
    let mut bounds: Option<(usize, usize, usize, usize)> = None;
    for cell in (0..config.cells()).filter(|&c| !hazard[c]) {
        let (x, y) = (cell % w, cell / w);
        bounds = Some(match bounds {
            None => (x, x, y, y),
            Some((x0, x1, y0, y1)) => (x0.min(x), x1.max(x), y0.min(y), y1.max(y)),
        });
    }
    let counts = match bounds {
        Some((x0, x1, y0, y1)) => [x0, w - 1 - x1, y0, h - 1 - y1],
        None if shrinks >= w => [w, 0, 0, 0],
        None => [0, 0, h, 0],
    };
    let mut sides = [UNKNOWN_SIDE; MAX_SIDES];
    let seen = counts
        .iter()
        .enumerate()
        .flat_map(|(side, &count)| std::iter::repeat_n(side as i8, count));
    for (slot, side) in sides.iter_mut().zip(seen.take(shrinks)) {
        *slot = side;
    }
    sides
}

/// `0 <= x < w and 0 <= y < h` as Python evaluates it (short-circuit; comparing a non-number
/// raises), then the cell as numpy indexes it in `body_to_countdown` (an on-board coordinate
/// must be an integer: a float raises, and a bool selects a whole row, whose `max` raises).
fn on_board(p: &Point<'_>, config: &GameConfig) -> Result<Option<Coord>, ApiError> {
    let (x, y) = *p;
    let (w, h) = (config.width() as i128, config.height() as i128);
    if !(in_range(x, 0, w)? && in_range(y, 0, h)?) {
        return Ok(None);
    }
    match (x, y) {
        (PyValue::Int(x), PyValue::Int(y)) => Ok(Some(Coord {
            x: *x as i32,
            y: *y as i32,
        })),
        _ => Err(ApiError::BadRequest(format!(
            "IndexError/ValueError: a {} and a {} are not one board cell",
            x.type_name(),
            y.type_name()
        ))),
    }
}

/// The cells `hazard[y, x]` writes for a listed hazard that passed the board check: one cell,
/// or, for a coordinate that is a bool, what numpy's boolean-scalar indexing selects (a bool
/// adds an axis instead of consuming one: `hazard[3, True]` is row 3, `hazard[True, True]` the
/// whole board, and a `False` nothing).
fn hazard_cells(p: &Point<'_>, config: &GameConfig) -> Result<Vec<Coord>, ApiError> {
    let (x, y) = *p;
    let (w, h) = (config.width() as i128, config.height() as i128);
    if !(in_range(x, 0, w)? && in_range(y, 0, h)?) {
        return Ok(Vec::new());
    }
    let row = |r: i128| -> Result<Vec<Coord>, ApiError> {
        if !(0..h).contains(&r) {
            return Err(ApiError::BadRequest(format!("IndexError: row {r}")));
        }
        Ok((0..w)
            .map(|c| Coord {
                x: c as i32,
                y: r as i32,
            })
            .collect())
    };
    match (y, x) {
        (PyValue::Int(y), PyValue::Int(x)) => Ok(vec![Coord {
            x: *x as i32,
            y: *y as i32,
        }]),
        (PyValue::Bool(false), PyValue::Int(_) | PyValue::Bool(_))
        | (PyValue::Int(_) | PyValue::Bool(_), PyValue::Bool(false)) => Ok(Vec::new()),
        // The one integer index (whichever coordinate it is) selects a row.
        (PyValue::Bool(true), PyValue::Int(r)) | (PyValue::Int(r), PyValue::Bool(true)) => row(*r),
        (PyValue::Bool(true), PyValue::Bool(true)) => {
            Ok((0..h).flat_map(|r| row(r).unwrap_or_default()).collect())
        }
        _ => Err(ApiError::BadRequest(format!(
            "IndexError: a {} and a {} are not valid indices",
            y.type_name(),
            x.type_name()
        ))),
    }
}

/// `_direction(head, neck)` (engine_json.py:73-92) with Python's arithmetic and comparisons:
/// the engine's default move, from the neck to the head.
fn direction(head: &Point<'_>, neck: &Point<'_>) -> Result<Move, ApiError> {
    let ((hx, hy), (nx, ny)) = (*head, *neck);
    let eq_num = |a: &PyValue, b: Num| -> Result<bool, ApiError> {
        Ok(a.num()
            .is_some_and(|x| x.compare(&b) == Some(std::cmp::Ordering::Equal)))
    };
    let plus =
        |v: &PyValue, d: i128| -> Result<Num, ApiError> { Ok(number(v)?.add(&Num::Int(d))?) };
    let positive = |v: &PyValue| -> Result<bool, ApiError> {
        Ok(py_cmp(v, &PyValue::Int(0))? == Some(std::cmp::Ordering::Greater))
    };
    let zero = |v: &PyValue| py_eq(v, &PyValue::Int(0));
    Ok(if eq_num(hx, plus(nx, 1)?)? {
        Move::Right
    } else if eq_num(hx, plus(nx, -1)?)? {
        Move::Left
    } else if eq_num(hy, plus(ny, 1)?)? {
        Move::Up
    } else if eq_num(hy, plus(ny, -1)?)? {
        Move::Down
    } else if zero(hx)? && positive(nx)? {
        Move::Right
    } else if zero(nx)? && positive(hx)? {
        Move::Left
    } else if zero(hy)? && positive(ny)? {
        Move::Up
    } else if zero(ny)? && positive(hy)? {
        Move::Down
    } else {
        Move::Up
    })
}

type Point<'a> = (&'a PyValue, &'a PyValue);

/// Python's `float // float` (`float_floor_div`, Objects/floatobject.c).
fn float_floordiv(vx: f64, wx: f64) -> f64 {
    let m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 && ((wx < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        0.0f64.copysign(vx / wx)
    }
}

/// A turn that gives `royale_sides` the shrink count Python computes from the request's
/// `turn`: `shrinks = min(max(turn, 0) // shrink_every, num_sides)`, which must be an `int`
/// to slice with. An int (or bool) turn is that turn; a float works only where `max` or `min`
/// returns its int argument instead (a negative float: 0 shrinks; one past the last shrink:
/// all of them); a string or `None` cannot be compared. The state's turn is read separately.
fn royale_turn(turn: &PyValue, config: &GameConfig) -> Result<i32, ApiError> {
    let num = config.num_sides() as f64;
    let shrink = f64::from(config.shrink_every_n_turns());
    match turn {
        PyValue::Int(_) | PyValue::Bool(_) => Ok(int32(turn)?),
        PyValue::BigInt { negative, .. } => Ok(if *negative { 0 } else { i32::MAX }),
        PyValue::Float(f) if *f < 0.0 => Ok(0), // `max(f, 0)` is the int 0
        PyValue::Float(f) if float_floordiv(*f, shrink) > num => Ok(i32::MAX), // `min` is num
        other => Err(ApiError::BadRequest(format!(
            "TypeError: royale shrink count from a turn of type {}",
            other.type_name()
        ))),
    }
}

/// `[p["x"], p["y"]] for p in v` (server.py:139-144).
fn points(v: &PyValue) -> Result<Vec<Point<'_>>, ApiError> {
    v.subscripted_items()?
        .iter()
        .map(|p| Ok((p.item("x")?, p.item("y")?)))
        .collect()
}

/// The state of a request in a game with `config`, `you` in seat 0 (`server.request_state`,
/// server.py:264-274, through `request_to_engine` and `state_from_engine`).
///
/// The dead snakes are padded in as eliminated seats up to `config.num_snakes()`.
pub fn request_state(req: &Request, config: &GameConfig) -> Result<GameState, ApiError> {
    let root = req.root();
    // --- request_to_engine (server.py:120-146).
    let (board, you) = (root.item("board")?, root.item("you")?);
    let mut snakes = Vec::new();
    for s in board.item("snakes")?.subscripted_items()? {
        let other = !py_eq(s.item("id")?, you.item("id")?)?;
        snakes.push((other, s));
    }
    snakes.sort_by_key(|&(other, _)| other); // stable: `you` first, the rest in API order
    if snakes.first().is_none_or(|&(other, _)| other) {
        return Err(ApiError::BadRequest("'you' is not on the board".into()));
    }
    if snakes.len() > config.num_snakes() {
        return Err(ApiError::BadRequest(format!(
            "{} snakes listed for a {}-seat game",
            snakes.len(),
            config.num_snakes()
        )));
    }
    let turn = root.item("turn")?;
    let mut listed = Vec::with_capacity(snakes.len());
    for &(_, s) in &snakes {
        let body = points(s.item("body")?)?;
        listed.push((body, s.item("health")?));
    }
    let food = points(board.item("food")?)?;
    let hazards = match board.get("hazards") {
        Some(v) => points(v)?,
        None => Vec::new(),
    };

    // --- royale_sides (server.py:237-261).
    let mut sides = None;
    if config.ruleset() == Ruleset::Royale {
        let t = royale_turn(turn, config)?;
        let mut cells = Vec::new();
        for p in &hazards {
            cells.extend(hazard_cells(p, config)?);
        }
        sides = Some(royale_sides(&cells, t, config));
    }

    // --- state_from_engine (engine_json.py:95-151).
    let mut state = GameState::new(config);
    let mut segments: Vec<(i32, i32)> = Vec::new();
    for (seat, (body, health)) in listed.iter().enumerate() {
        let health = int32(health)?;
        segments.clear();
        for p in body {
            // Off the board: dropped from the countdown grid (as `body_to_countdown` does).
            segments.push(on_board(p, config)?.map_or((-1, -1), |c| (c.x, c.y)));
        }
        state.set_snake(config, seat, &segments, health, None)?;
        if let Some(&(x, y)) = body.first() {
            // `head[i] = segs[0]`: an int32 store, wherever the head is.
            state.head[seat] = [int32(x)?, int32(y)?];
        }
        if let [head, neck, ..] = body.as_slice() {
            state.last_move[seat] = direction(head, neck)? as u8;
        }
    }
    for seat in listed.len()..config.num_snakes() {
        state.set_snake(config, seat, &[], 0, Some((Cause::Collision, 0)))?;
    }
    for &(x, y) in &food {
        // `food[y, x]`: numpy indexing, negative indices count from the end.
        let y = numpy_index(y, config.height())?;
        let x = numpy_index(x, config.width())?;
        state.add_food(config, x as i32, y as i32)?;
    }
    for p in &hazards {
        for c in hazard_cells(p, config)? {
            state.add_hazard(config, c.x, c.y);
        }
    }
    state.turn = int32(turn)?;
    if let Some(sides) = sides {
        state.sides = sides;
    }
    state.refresh_done(config);
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn value(v: serde_json::Value) -> PyValue {
        crate::py::parse(&v.to_string()).unwrap()
    }

    #[test]
    fn settings_are_lenient() {
        let s = value(
            json!({"a": "20", "b": true, "c": 7.9, "d": 2_000_000, "e": null,
                             "f": [1], "g": -3, "h": 1048576, "i": "x", "j": "\u{661}\u{662}",
                             "k": -1048576.9}),
        );
        let get = |k: &str| setting(Some(&s), k, 99).unwrap();
        assert_eq!(get("a"), 20);
        assert_eq!(get("b"), 99);
        assert_eq!(get("c"), 7);
        assert_eq!(get("d"), 99);
        assert_eq!(get("e"), 99);
        assert_eq!(get("f"), 99);
        assert_eq!(get("g"), -3);
        assert_eq!(get("h"), 1 << 20);
        assert_eq!(get("i"), 99);
        assert_eq!(get("j"), 12);
        assert_eq!(get("k"), -(1 << 20));
        assert_eq!(get("missing"), 99);
        assert_eq!(setting(None, "a", 5).unwrap(), 5);
        assert_eq!(setting(Some(&value(json!([1]))), "a", 5).unwrap(), 5);
        // NaN keeps the default (ValueError), an infinity raises (OverflowError).
        let s = crate::py::parse(r#"{"n": NaN, "i": Infinity}"#).unwrap();
        assert_eq!(setting(Some(&s), "n", 5).unwrap(), 5);
        assert!(setting(Some(&s), "i", 5).is_err());
    }

    fn cfg() -> GameConfig {
        GameConfig::new(11, 11, 4, Ruleset::Royale).unwrap()
    }

    fn hazards_outside(x0: i32, x1: i32, y0: i32, y1: i32) -> Vec<Coord> {
        (0..11)
            .flat_map(|x| (0..11).map(move |y| Coord { x, y }))
            .filter(|c| c.x < x0 || c.x > x1 || c.y < y0 || c.y > y1)
            .collect()
    }

    #[test]
    fn sides_from_the_zone() {
        let c = cfg();
        // 3 shrinks shown (turn 75): left once, right once, bottom once, in that fixed order.
        let sides = royale_sides(&hazards_outside(1, 9, 1, 10), 75, &c);
        assert_eq!(&sides[..4], &[0, 1, 2, UNKNOWN_SIDE]);
        assert!(sides[3..].iter().all(|&s| s == UNKNOWN_SIDE));
        // Not yet shrunk, whatever the zone says.
        assert!(royale_sides(&[], 24, &c).iter().all(|&s| s == UNKNOWN_SIDE));
        // Everything is hazard: a column sweep once shrinks >= W, else a row sweep.
        let all = hazards_outside(5, 4, 5, 4);
        assert_eq!(all.len(), 121);
        assert_eq!(
            &royale_sides(&all, 25 * 11, &c)[..12],
            &[0; 11]
                .iter()
                .copied()
                .chain([UNKNOWN_SIDE])
                .collect::<Vec<_>>()[..]
        );
        assert_eq!(
            &royale_sides(&all, 25 * 3, &c)[..4],
            &[2, 2, 2, UNKNOWN_SIDE]
        );
        // Negative turns count no shrink.
        assert!(
            royale_sides(&all, -50, &c)
                .iter()
                .all(|&s| s == UNKNOWN_SIDE)
        );
    }

    fn duel_json() -> serde_json::Value {
        let body = json!([{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}]);
        let other = json!([{"x": 9, "y": 9}, {"x": 9, "y": 9}, {"x": 9, "y": 9}]);
        json!({
            "game": {"id": "g", "ruleset": {"name": "standard", "settings": {}}, "map": "standard"},
            "turn": 0,
            "board": {"width": 11, "height": 11, "food": [{"x": 0, "y": 1}], "hazards": [],
                "snakes": [
                    {"id": "b", "health": 100, "body": other, "head": {"x": 9, "y": 9}, "length": 3},
                    {"id": "a", "health": 100, "body": body, "head": {"x": 1, "y": 1}, "length": 3}]},
            "you": {"id": "a", "health": 100, "body": body, "head": {"x": 1, "y": 1}, "length": 3},
        })
    }

    fn game_of(v: &serde_json::Value, starting: Option<usize>) -> Result<Game, ApiError> {
        game_from_request(&Request::from_value(v)?, starting)
    }

    #[test]
    fn duel_at_turn_zero() {
        let game = game_of(&duel_json(), Some(2)).unwrap();
        assert_eq!(game.mode, Mode::Duel);
        let (c, s) = (&game.config, &game.state);
        assert_eq!(s.body_at(c, 0, 1, 1), 3); // you moved to seat 0
        assert_eq!(s.body_at(c, 1, 9, 9), 3);
        assert_eq!(s.length[..2], [3, 3]);
        assert!(s.food_at(c, 0, 1) && !s.done && s.alive[..2] == [true, true]);
        assert!(game.is_network_board());
    }

    #[test]
    fn missing_start_reads_living_snakes_only() {
        let mut v = duel_json();
        v["board"]["snakes"].as_array_mut().unwrap().remove(0); // the other snake died
        // Without /start there is a single living snake: no mode.
        assert!(matches!(game_of(&v, None), Err(ApiError::Unsupported(_))));
        // With the cache the dead seat is padded in and the game is over.
        let game = game_of(&v, Some(2)).unwrap();
        assert!(game.state.done);
        assert!(!game.state.alive[1] && game.state.elim_cause[1] == Cause::Collision as i8);
        assert_eq!(game.state.length[1], 0);
    }

    /// Python reads only what it needs, with Python's types (see the module docs).
    #[test]
    fn python_reads() {
        let mut v = duel_json();
        v["you"] = json!({"id": "a"}); // nothing else of `you` is read
        v["board"]["snakes"][0]["health"] = json!(89.7); // numpy truncates
        v["board"]["snakes"][0]["length"] = json!("junk"); // never read
        v["turn"] = json!("7"); // numpy parses
        v["board"]["food"] = json!([{"x": -1, "y": 0}]); // numpy wraps to x = 10
        let game = game_of(&v, Some(2)).unwrap();
        let (c, s) = (&game.config, &game.state);
        assert_eq!((s.health[1], s.turn), (89, 7));
        assert!(s.food_at(c, 10, 0));
        // An off-board tail segment of any size is dropped; the length still counts it.
        let mut v = duel_json();
        v["board"]["snakes"][1]["body"][2] = json!({"x": 1u64 << 63, "y": 0});
        let game = game_of(&v, Some(2)).unwrap();
        assert_eq!(game.state.length[0], 3);
        // Ids compare with Python's ==: 1 == 1.0.
        let mut v = duel_json();
        v["board"]["snakes"][1]["id"] = json!(1);
        v["you"]["id"] = json!(1.0);
        assert!(game_of(&v, Some(2)).is_ok());
    }

    #[test]
    fn refusals() {
        let mut v = duel_json();
        v["board"]["food"]
            .as_array_mut()
            .unwrap()
            .push(json!({"x": 0, "y": 1}));
        assert!(matches!(game_of(&v, Some(2)), Err(ApiError::Rules(_))));
        let mut v = duel_json();
        v["you"]["id"] = json!("zzz");
        assert!(matches!(game_of(&v, Some(2)), Err(ApiError::BadRequest(_))));
        let mut v = duel_json();
        v["game"]["ruleset"]["name"] = json!("wrapped");
        assert!(matches!(
            game_of(&v, Some(2)),
            Err(ApiError::Unsupported(_))
        ));
        let mut v = duel_json();
        v["turn"] = json!(1u64 << 40);
        assert!(matches!(game_of(&v, Some(2)), Err(ApiError::BadRequest(_))));
        assert!(matches!(
            game_of(&duel_json(), Some(5)),
            Err(ApiError::Unsupported(_))
        ));
        // A coordinate written as a list is not subscriptable (TypeError in Python).
        let mut v = duel_json();
        v["board"]["snakes"][0]["body"][1] = json!([9, 8]);
        assert!(matches!(game_of(&v, Some(2)), Err(ApiError::BadRequest(_))));
        // A non-object game is refused by the HTTP layer.
        let mut v = duel_json();
        v["game"] = json!([]);
        assert!(matches!(game_of(&v, Some(2)), Err(ApiError::Json(_))));
    }
}
