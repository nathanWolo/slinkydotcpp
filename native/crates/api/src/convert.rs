//! Request -> game: the config, the mode and the state the search starts from.
//!
//! Ported from `src/slinky/server.py` (`game_config` 161-226, `_setting` 149-158,
//! `request_to_engine` 120-146, `royale_sides` 237-261, `request_state` 264-274, `check_sizes`
//! 347-356, `Mover.move` 476-496) and `src/slinky/engine_json.py` (`state_from_engine` 95-151,
//! `body_to_countdown` 35-42), see SPEC.md section 6. The rules crate's `GameState::set_snake`
//! already does the body -> countdown step (stacked segments take the largest countdown).
//!
//! What the Python does that differs from "the obvious reading" of the API:
//!
//! * The snake count of the config is `max(starting_snakes, snakes listed)`. The request lists
//!   only the living snakes, so a caller without the game's `/start` has to pass `None`, and
//!   then the count is the number of *living* snakes: a four-player standard game with two
//!   snakes left is read as a duel (and told the duel mode), and a royale or constrictor game
//!   keeps its mode but loses seats (SPEC.md section 8, point 1). The dead seats are appended
//!   as eliminated snakes (cause `snake-collision`, turn 0, empty body, health 0).
//! * `you` is seat 0 and the other snakes keep the API order (a stable sort on `id != you.id`).
//! * Every royale spelling (royale ruleset on the royale or standard map, standard ruleset on the
//!   royale map) becomes the royale ruleset; constrictor on the `empty` map becomes plain
//!   constrictor; constrictor on the royale map, the standard ruleset on the empty map, and any
//!   other ruleset or map have no config.
//! * Settings are read with [`setting`]: an integer, an integral-or-truncated float, or a numeric
//!   string within +-2^20; anything else keeps the default. Only `hazardDamagePerTurn` (royale,
//!   or a board that lists hazards) and the royale shrink period (royale only; the nested
//!   `royale.shrinkEveryNTurns` wins over the flat one) reach the config. The food settings are
//!   never read, and `max_turns` is `None`.
//! * A food cell off the board or listed twice is an error (Python: `ValueError`/`IndexError`),
//!   hazards off the board are dropped, duplicate hazards stack.
//! * Royale shrink sides are rebuilt from the listed hazards (see [`royale_sides`]).
//!
//! Small deviations from Python, all on inputs the Battlesnake engine never sends: a food
//! coordinate that is negative wraps around in numpy but is an error here; unicode digits in a
//! numeric-string setting are not parsed; a `health`, `turn` or coordinate that is not an
//! integer fails to parse (Python truncates floats); an explicit `"game": null` reads like an
//! absent `game` (the Python HTTP layer answers 400).

use serde_json::{Map, Value};

use slinky_rules::{
    Cause, GameConfig, GameState, MAX_CELLS, MAX_SIDES, MAX_SNAKES, Mode, Ruleset, UNKNOWN_SIDE,
};

use crate::{ApiError, Coord, Request, Snake};

/// Settings outside this range keep their default (server.py:107, `MAX_SETTING`).
pub const MAX_SETTING: i64 = 1 << 20;

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
/// This is `server.Mover.move` up to the search: [`check_sizes`], [`game_config`], the mode
/// (refused for fewer than 2 seats), then [`request_state`].
pub fn game_from_request(req: &Request, starting_snakes: Option<usize>) -> Result<Game, ApiError> {
    check_sizes(req)?;
    let config = game_config(req, starting_snakes)?;
    let state = request_state(req, &config)?;
    Ok(Game {
        mode: config.mode(),
        config,
        state,
    })
}

/// Rejects a request that lists more than its board can hold (`server.check_sizes`,
/// server.py:347-356): more food than cells, more than four hazards per cell, or a body longer
/// than the board plus three.
pub fn check_sizes(req: &Request) -> Result<(), ApiError> {
    let board = &req.board;
    let cells = i128::from(board.width) * i128::from(board.height);
    let too_long = board
        .snakes
        .iter()
        .any(|s| s.body.len() as i128 > cells + 3);
    if board.food.len() as i128 > cells || board.hazards.len() as i128 > 4 * cells || too_long {
        return Err(ApiError::BadRequest(
            "more food, hazards or body than the board holds".into(),
        ));
    }
    Ok(())
}

/// Python truthiness of a JSON value (`x or {}` in the Python).
fn falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// The object `v` as a map: absent and falsy values read as `{}`, other non-objects are errors
/// (`AttributeError` on `.get` in the Python).
fn object_or_empty<'a>(
    v: Option<&'a Value>,
    what: &str,
) -> Result<&'a Map<String, Value>, ApiError> {
    static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
    match v {
        None => Ok(&EMPTY),
        Some(Value::Object(m)) => Ok(m),
        Some(other) if falsy(other) => Ok(&EMPTY),
        Some(_) => Err(ApiError::BadRequest(format!("{what} is not an object"))),
    }
}

/// `int(text)` of Python for ASCII: optional whitespace and sign, digits with single underscores
/// between digits. `None` for anything else or a value beyond `i64`.
fn parse_python_int(text: &str) -> Option<i64> {
    let t = text.trim();
    let (neg, digits) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let bytes = digits.as_bytes();
    if bytes.is_empty()
        || !bytes[0].is_ascii_digit()
        || !bytes[bytes.len() - 1].is_ascii_digit()
        || digits.contains("__")
    {
        return None;
    }
    let mut value: i128 = 0;
    for &b in bytes {
        match b {
            b'_' => {}
            b'0'..=b'9' => {
                value = value * 10 + i128::from(b - b'0');
                if value > i128::from(i64::MAX) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    i64::try_from(if neg { -value } else { value }).ok()
}

/// The integer `settings[key]`, or `default` (`server._setting`, server.py:149-158).
///
/// Accepts a JSON integer, a float (truncated toward zero, like `int(14.9)`) or a numeric string
/// (the engine CLI sends strings); a boolean, `null`, a list, an object, junk or a value outside
/// +-[`MAX_SETTING`] keeps `default`. A `settings` that is not an object also keeps it.
pub fn setting(settings: Option<&Value>, key: &str, default: i32) -> i32 {
    let value = settings.and_then(Value::as_object).and_then(|m| m.get(key));
    let n = match value {
        Some(Value::Number(n)) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite() && f.abs() < 9e18)
                .map(|f| f.trunc() as i64)
        }),
        Some(Value::String(s)) => parse_python_int(s),
        _ => None,
    };
    match n {
        Some(n) if (-MAX_SETTING..=MAX_SETTING).contains(&n) => n as i32,
        _ => default,
    }
}

/// The [`GameConfig`] of a request's game (`server.game_config`, server.py:161-226).
///
/// `starting_snakes` is the snake count at `/start` if known. Errors with
/// [`ApiError::Unsupported`] for a game the agent does not model: another ruleset or map, a board
/// larger than 11x11, fewer than 2 or more than 4 seats, a shrink period below 1.
pub fn game_config(req: &Request, starting_snakes: Option<usize>) -> Result<GameConfig, ApiError> {
    let game = object_or_empty(req.game.as_ref(), "game")?;
    let ruleset = object_or_empty(game.get("ruleset"), "game.ruleset")?;
    let name = match ruleset.get("name") {
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
    let board = &req.board;
    let n = starting_snakes.unwrap_or(0).max(board.snakes.len());
    let (Ok(w), Ok(h)) = (usize::try_from(board.width), usize::try_from(board.height)) else {
        return unsupported(format!("board {}x{}", board.width, board.height));
    };
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
    let settings = ruleset.get("settings");
    if royale || !board.hazards.is_empty() {
        config = config.with_hazard_damage(setting(
            settings,
            "hazardDamagePerTurn",
            config.hazard_damage_per_turn(),
        ));
    }
    if royale {
        // The API nests the period under "royale"; the engine CLI's flat name is read too.
        let flat = setting(settings, "shrinkEveryNTurns", config.shrink_every_n_turns());
        let nested = setting(
            settings.and_then(|s| s.get("royale")),
            "shrinkEveryNTurns",
            flat,
        );
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

/// The state of a request in a game with `config`, `you` in seat 0 (`server.request_state`,
/// server.py:264-274, through `request_to_engine` and `state_from_engine`).
///
/// The dead snakes are padded in as eliminated seats up to `config.num_snakes()`.
pub fn request_state(req: &Request, config: &GameConfig) -> Result<GameState, ApiError> {
    let turn = i32::try_from(req.turn)
        .map_err(|_| ApiError::BadRequest(format!("turn {} does not fit in 32 bits", req.turn)))?;
    let mut snakes: Vec<&Snake> = req.board.snakes.iter().collect();
    snakes.sort_by_key(|s| s.id != req.you.id); // stable: `you` first, the rest in API order
    if snakes.first().is_none_or(|s| s.id != req.you.id) {
        return Err(ApiError::BadRequest("'you' is not on the board".into()));
    }
    if snakes.len() > config.num_snakes() {
        return Err(ApiError::BadRequest(format!(
            "{} snakes listed for a {}-seat game",
            snakes.len(),
            config.num_snakes()
        )));
    }
    let mut state = GameState::new(config);
    let mut segments: Vec<(i32, i32)> = Vec::new();
    for (seat, snake) in snakes.iter().enumerate() {
        segments.clear();
        segments.extend(snake.body.iter().map(|c| (c.x, c.y)));
        state.set_snake(config, seat, &segments, snake.health, None)?;
    }
    for seat in snakes.len()..config.num_snakes() {
        state.set_snake(config, seat, &[], 0, Some((Cause::Collision, 0)))?;
    }
    for c in &req.board.food {
        state.add_food(config, c.x, c.y)?;
    }
    for c in &req.board.hazards {
        state.add_hazard(config, c.x, c.y);
    }
    state.turn = turn;
    if config.ruleset() == Ruleset::Royale {
        state.sides = royale_sides(&req.board.hazards, turn, config);
    }
    state.refresh_done(config);
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_ints() {
        for (text, want) in [
            ("20", Some(20)),
            (" 12 ", Some(12)),
            ("+8", Some(8)),
            ("-8", Some(-8)),
            ("007", Some(7)),
            ("1_0", Some(10)),
            ("1__0", None),
            ("_1", None),
            ("1_", None),
            ("", None),
            ("-", None),
            ("3.5", None),
            ("2e1", None),
            ("0x10", None),
            ("99999999999999999999", None),
        ] {
            assert_eq!(parse_python_int(text), want, "{text:?}");
        }
    }

    #[test]
    fn settings_are_lenient() {
        let s = json!({"a": "20", "b": true, "c": 7.9, "d": 2_000_000, "e": null, "f": [1],
                       "g": -3, "h": 1048576, "i": "x"});
        let get = |k: &str| setting(Some(&s), k, 99);
        assert_eq!(get("a"), 20);
        assert_eq!(get("b"), 99);
        assert_eq!(get("c"), 7);
        assert_eq!(get("d"), 99);
        assert_eq!(get("e"), 99);
        assert_eq!(get("f"), 99);
        assert_eq!(get("g"), -3);
        assert_eq!(get("h"), 1 << 20);
        assert_eq!(get("i"), 99);
        assert_eq!(get("missing"), 99);
        assert_eq!(setting(None, "a", 5), 5);
        assert_eq!(setting(Some(&json!([1])), "a", 5), 5);
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

    fn duel_request() -> Request {
        let body = json!([{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}]);
        let other = json!([{"x": 9, "y": 9}, {"x": 9, "y": 9}, {"x": 9, "y": 9}]);
        serde_json::from_value(json!({
            "game": {"id": "g", "ruleset": {"name": "standard", "settings": {}}, "map": "standard"},
            "turn": 0,
            "board": {"width": 11, "height": 11, "food": [{"x": 0, "y": 1}], "hazards": [],
                "snakes": [
                    {"id": "b", "health": 100, "body": other, "head": {"x": 9, "y": 9}, "length": 3},
                    {"id": "a", "health": 100, "body": body, "head": {"x": 1, "y": 1}, "length": 3}]},
            "you": {"id": "a", "health": 100, "body": body, "head": {"x": 1, "y": 1}, "length": 3},
        }))
        .unwrap()
    }

    #[test]
    fn duel_at_turn_zero() {
        let req = duel_request();
        let game = game_from_request(&req, Some(2)).unwrap();
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
        let mut req = duel_request();
        req.board.snakes.truncate(1); // the other snake died
        req.board.snakes[0].id = "a".into();
        // Without /start there is a single living snake: no mode.
        assert!(matches!(
            game_from_request(&req, None),
            Err(ApiError::Unsupported(_))
        ));
        // With the cache the dead seat is padded in and the game is over.
        let game = game_from_request(&req, Some(2)).unwrap();
        assert!(game.state.done);
        assert!(!game.state.alive[1] && game.state.elim_cause[1] == Cause::Collision as i8);
        assert_eq!(game.state.length[1], 0);
    }

    #[test]
    fn refusals() {
        let mut req = duel_request();
        req.board.food.push(Coord { x: 0, y: 1 });
        assert!(matches!(
            game_from_request(&req, Some(2)),
            Err(ApiError::Rules(_))
        ));
        let mut req = duel_request();
        req.you.id = "zzz".into();
        assert!(matches!(
            game_from_request(&req, Some(2)),
            Err(ApiError::BadRequest(_))
        ));
        let mut req = duel_request();
        req.game.as_mut().unwrap()["ruleset"]["name"] = json!("wrapped");
        assert!(matches!(
            game_from_request(&req, Some(2)),
            Err(ApiError::Unsupported(_))
        ));
        let mut req = duel_request();
        req.turn = 1 << 40;
        assert!(matches!(
            game_from_request(&req, Some(2)),
            Err(ApiError::BadRequest(_))
        ));
        assert!(matches!(
            game_from_request(&duel_request(), Some(5)),
            Err(ApiError::Unsupported(_))
        ));
    }
}
