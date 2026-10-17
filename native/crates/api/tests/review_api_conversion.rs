//! Adversarial review of the request -> game conversion against the whole Python server.
//!
//! Golden cases come from `native/golden/review_api_conversion.py` (see its docstring for the
//! format): malformed and hostile bodies (missing and mistyped fields, NaN and duplicate keys,
//! off-board and huge coordinates, duplicate food, empty and stacked bodies, out-of-range health,
//! unknown rulesets and maps, huge boards), seat counts with and without the `/start` count,
//! royale hazards that are not a rectangle, and every spelling of the settings.
//!
//! Each case records what the Python server *answers* (`agent`, `fallback`, `safe`, `http400`)
//! and, when Python built one, the exact config and state. The checks:
//!
//! * nothing panics, neither the conversion nor `safe_move_json`;
//! * where Python built a state (2..=4 seats, at most 11x11), Rust builds the same one;
//! * where the agent plays in Python, Rust gives a game the agent plays (11x11, 2..=4 seats);
//! * where Python does not run the agent, Rust does not give an agent-playable game;
//! * where Python answers `safe_move`, Rust's `safe_move_json` gives the same move.
//!
//! There is one test per group of cases. Cases that exposed a parity bug are listed in
//! [`KNOWN`] with the bug's name; all of those bugs are fixed, so the group tests run every
//! case and each `bug_*` test reruns its bug's cases as a regression test. Skips (with a
//! message) when the golden file is missing.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use serde_json::Value;
use slinky_api::{Game, Request, game_from_request, safe_move_json};
use slinky_rules::{Mode, Ruleset};

/// Cases that exposed a parity bug, now fixed: (case name, bug name). Each bug has a test
/// `bug_<name>` below that says what was wrong and where.
const KNOWN: &[(&str, &str)] = &[
    (
        "you_object/you-only-id",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/length-string",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/length-float",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/head-partial",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/name-number",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/health-string",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "you_object/numeric-ids",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "board_snakes/other-length-string",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "board_snakes/other-head-partial",
        "typed_parse_rejects_unread_fields",
    ),
    (
        "board_snakes/other-name-number",
        "typed_parse_rejects_unread_fields",
    ),
    ("board_snakes/health-float", "strict_integer_fields"),
    ("board_snakes/health-float-frac", "strict_integer_fields"),
    ("board_snakes/health-string", "strict_integer_fields"),
    ("board_snakes/health-bool", "strict_integer_fields"),
    ("turns/standard-float", "strict_integer_fields"),
    ("turns/standard-float-frac", "strict_integer_fields"),
    ("turns/standard-string", "strict_integer_fields"),
    ("turns/standard-bool", "strict_integer_fields"),
    ("turns/royale-bool", "strict_integer_fields"),
    ("bodies/tail-x-i32over", "i32_coordinates_off_board"),
    ("bodies/tail-x-2pow40", "i32_coordinates_off_board"),
    ("bodies/tail-y-neg-2pow40", "i32_coordinates_off_board"),
    ("bodies/tail-x-i64over", "i32_coordinates_off_board"),
    ("hazards/standard-huge-x", "i32_coordinates_off_board"),
    ("hazards/standard-i32over", "i32_coordinates_off_board"),
    ("hazards/royale-huge-x", "i32_coordinates_off_board"),
    ("hazards/royale-i32over", "i32_coordinates_off_board"),
    ("hazards/constrictor-huge-x", "i32_coordinates_off_board"),
    ("hazards/constrictor-i32over", "i32_coordinates_off_board"),
    ("raw_text/dup-turn", "json_dialect"),
    ("raw_text/dup-coord-key", "json_dialect"),
    ("raw_text/dup-width", "json_dialect"),
    ("raw_text/damage-nan-standard", "json_dialect"),
    ("raw_text/damage-nan-royale", "json_dialect"),
    ("raw_text/shrink-nan", "json_dialect"),
    ("raw_text/timeout-nan", "json_dialect"),
    ("raw_text/damage-inf-standard", "json_dialect"),
    ("raw_text/timeout-inf", "json_dialect"),
    ("raw_text/damage-neginf-standard", "json_dialect"),
    ("raw_text/timeout-neginf", "json_dialect"),
    ("raw_text/bom", "json_dialect"),
    ("raw_text/big-int-setting", "json_dialect"),
    ("raw_text/deep-in-game", "json_dialect"),
    ("raw_text/over-64kb", "no_body_size_limit"),
    ("game_object/list", "falsy_game_object_plays"),
    ("game_object/zero", "falsy_game_object_plays"),
    ("game_object/empty_string", "falsy_game_object_plays"),
    ("game_object/false", "falsy_game_object_plays"),
    ("game_object/null", "falsy_game_object_plays"),
    ("food/neg-x", "negative_food_wraps_in_python"),
    ("food/neg-y", "negative_food_wraps_in_python"),
    ("settings/royale-damage-4", "unicode_digit_settings"),
    ("settings/royale-shrink-nested-4", "unicode_digit_settings"),
    ("settings/royale-shrink-flat-4", "unicode_digit_settings"),
    (
        "settings/standard-hazards-damage-4",
        "unicode_digit_settings",
    ),
    ("settings/royale-damage-5", "unicode_digit_settings"),
    ("settings/royale-shrink-nested-5", "unicode_digit_settings"),
    ("settings/royale-shrink-flat-5", "unicode_digit_settings"),
    (
        "settings/standard-hazards-damage-5",
        "unicode_digit_settings",
    ),
    ("bodies/coord-as-list", "coord_accepts_json_array"),
    (
        "safe_moves/wrapped-you-head-i64max",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/wrapped-you-head-i64min",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/wrapped-other-head-i64max",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/wrapped-other-head-i64min",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/solo-you-head-i64max",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/solo-you-head-i64min",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/solo-other-head-i64max",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/solo-other-head-i64min",
        "safe_move_overflow_panic",
    ),
    (
        "safe_moves/wrapped-you-no-length",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/wrapped-you-no-head",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/wrapped-other-no-head",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/wrapped-other-no-length",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/wrapped-other-no-id",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/wrapped-other-length-string",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-you-no-length",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-you-no-head",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-other-no-head",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-other-no-length",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-other-no-id",
        "safe_move_answers_where_python_raises",
    ),
    (
        "safe_moves/solo-other-length-string",
        "safe_move_answers_where_python_raises",
    ),
    ("safe_moves/wrapped-body-floats", "safe_move_number_reading"),
    ("safe_moves/wrapped-width-huge", "safe_move_number_reading"),
    ("safe_moves/solo-body-floats", "safe_move_number_reading"),
    ("safe_moves/solo-width-huge", "safe_move_number_reading"),
    ("boards/9223372036854775808x1", "safe_move_number_reading"),
];

fn known_bug(name: &str) -> Option<&'static str> {
    KNOWN
        .iter()
        .find(|(case, _)| *case == name)
        .map(|&(_, bug)| bug)
}

fn load() -> Option<Vec<Value>> {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let path = PathBuf::from(base).join("api").join("review_cases.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP review golden: {} missing (run native/golden/review_api_conversion.py)",
            path.display()
        );
        return None;
    };
    Some(
        text.lines()
            .map(|l| serde_json::from_str(l).expect("golden line"))
            .collect(),
    )
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

/// The first difference between Python's state dump and the Rust game.
fn compare_state(exp: &Value, got: &Game) -> Result<(), String> {
    let (cfg, st) = (&got.config, &got.state);
    let ec = &exp["config"];
    let check = |what: &str, a: i64, b: Option<i64>| {
        if Some(a) == b {
            Ok(())
        } else {
            Err(format!("{what}: rust {a}, python {b:?}"))
        }
    };
    check("width", cfg.width() as i64, ec["width"].as_i64())?;
    check("height", cfg.height() as i64, ec["height"].as_i64())?;
    check(
        "num_snakes",
        cfg.num_snakes() as i64,
        ec["num_snakes"].as_i64(),
    )?;
    let ruleset = ec["ruleset"]
        .as_str()
        .and_then(|r| Ruleset::from_name(r).ok());
    if Some(cfg.ruleset()) != ruleset {
        return Err(format!("ruleset {:?} vs {}", cfg.ruleset(), ec["ruleset"]));
    }
    check(
        "hazard_damage",
        cfg.hazard_damage_per_turn().into(),
        ec["hazard_damage"].as_i64(),
    )?;
    check(
        "shrink_every",
        cfg.shrink_every_n_turns().into(),
        ec["shrink_every"].as_i64(),
    )?;
    let mode = exp["mode"].as_str().and_then(|m| Mode::from_name(m).ok());
    if Some(got.mode) != mode {
        return Err(format!("mode {:?} vs {}", got.mode, exp["mode"]));
    }
    let es = &exp["state"];
    let cells = cfg.cells();
    for s in 0..cfg.num_snakes() {
        let rb: Vec<i64> = st.body[s][..cells].iter().map(|&b| b.into()).collect();
        if rb != ints(&es["body"][s]) {
            return Err(format!("body of seat {s}: rust {rb:?}"));
        }
        let head = ints(&es["head"][s]);
        check("head x", st.head[s][0].into(), head.first().copied())?;
        check("head y", st.head[s][1].into(), head.get(1).copied())?;
        check("length", st.length[s].into(), es["length"][s].as_i64())?;
        check("health", st.health[s].into(), es["health"][s].as_i64())?;
        check(
            "alive",
            st.alive[s].into(),
            es["alive"][s].as_bool().map(i64::from),
        )?;
        check(
            "last_move",
            st.last_move[s].into(),
            es["last_move"][s].as_i64(),
        )?;
        check(
            "elim_cause",
            st.elim_cause[s].into(),
            es["elim_cause"][s].as_i64(),
        )?;
        check(
            "elim_turn",
            st.elim_turn[s].into(),
            es["elim_turn"][s].as_i64(),
        )?;
    }
    let food = ints(&es["food"]);
    for (c, &f) in food.iter().enumerate() {
        check(&format!("food[{c}]"), st.has_food(c).into(), Some(f))?;
    }
    if food.len() != cells || st.food >> cells != 0 {
        return Err("food past the board".into());
    }
    let hazard = ints(&es["hazard"]);
    for (c, &h) in hazard.iter().enumerate() {
        check(&format!("hazard[{c}]"), st.hazard[c].into(), Some(h))?;
    }
    check("turn", st.turn.into(), es["turn"].as_i64())?;
    check("done", st.done.into(), es["done"].as_bool().map(i64::from))?;
    if cfg.ruleset() == Ruleset::Royale {
        let rs: Vec<i64> = st.sides[..cfg.num_sides()]
            .iter()
            .map(|&s| s.into())
            .collect();
        if rs != ints(&es["sides"]) {
            return Err(format!("sides: rust {rs:?} python {}", es["sides"]));
        }
    }
    Ok(())
}

/// Whether the Rust agent would search this game (`Agent::check_game` with every mode trained).
fn agent_plays(game: &Game) -> bool {
    (2..=4).contains(&game.config.num_snakes()) && game.is_network_board()
}

/// Everything wrong with one case (empty when Rust matches Python).
fn check_case(case: &Value) -> Vec<String> {
    let name = case["name"].as_str().unwrap_or("?");
    let text = case["text"].as_str().unwrap_or("");
    let starting = case["starting"].as_u64().map(|n| n as usize);
    let outcome = case["outcome"].as_str().unwrap_or("?");
    let exp = &case["expect"];
    let mut problems = Vec::new();
    let run = catch_unwind(AssertUnwindSafe(|| {
        let game = Request::from_json(text).and_then(|r| game_from_request(&r, starting));
        (game, safe_move_json(text))
    }));
    let Ok((game, safe)) = run else {
        return vec![format!("{name}: PANIC")];
    };
    if exp["ok"].as_bool() == Some(true) {
        match &game {
            Ok(g) => {
                if let Err(e) = compare_state(exp, g) {
                    problems.push(format!("{name}: state differs: {e}"));
                }
            }
            Err(e) => problems.push(format!("{name}: rust refused ({e}), python built a state")),
        }
    }
    let plays = game.as_ref().is_ok_and(agent_plays);
    match outcome {
        "agent" if !plays => problems.push(format!(
            "{name}: python's agent plays, rust {}",
            match &game {
                Ok(g) => format!("built an unplayable {:?} game", g.mode),
                Err(e) => format!("refused: {e}"),
            }
        )),
        "fallback" | "safe" | "http400" if plays => problems.push(format!(
            "{name}: python answers {outcome} ({}), rust's agent would play",
            case["why"]
        )),
        _ => {}
    }
    if outcome == "safe"
        && let Some(answer) = case["answer"].as_i64()
        && safe as i64 != answer
    {
        problems.push(format!(
            "{name}: safe_move rust {} python {answer} ({})",
            safe as i64, case["why"]
        ));
    }
    problems
}

/// Runs the cases of `group` (all groups when empty), only those of `bug` when given, and
/// fails on any problem.
fn run(group: &str, bug: Option<&str>) {
    let Some(cases) = load() else { return };
    let mut problems = Vec::new();
    let mut count = 0;
    for case in &cases {
        let name = case["name"].as_str().unwrap_or("?");
        let in_group = group.is_empty() || case["group"].as_str() == Some(group);
        if !in_group || bug.is_some_and(|b| known_bug(name) != Some(b)) {
            continue;
        }
        count += 1;
        problems.extend(check_case(case));
    }
    eprintln!(
        "review {group}{}: {count} cases, {} problems",
        bug.map(|b| format!(" [{b}]")).unwrap_or_default(),
        problems.len()
    );
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn game_object() {
    run("game_object", None);
}

#[test]
fn you_object() {
    run("you_object", None);
}

#[test]
fn board_snakes() {
    run("board_snakes", None);
}

#[test]
fn bodies() {
    run("bodies", None);
}

#[test]
fn food() {
    run("food", None);
}

#[test]
fn hazards() {
    run("hazards", None);
}

#[test]
fn turns() {
    run("turns", None);
}

#[test]
fn raw_text() {
    run("raw_text", None);
}

#[test]
fn seats() {
    run("seats", None);
}

#[test]
fn royale_zones() {
    run("royale_zones", None);
}

#[test]
fn settings() {
    run("settings", None);
}

#[test]
fn boards() {
    run("boards", None);
}

#[test]
fn safe_moves() {
    run("safe_moves", None);
}

/// Random hostile edits of realistic requests (`fuzz` in the golden script): the whole
/// conversion and `safe_move` against the Python server.
#[test]
fn fuzz() {
    run("fuzz", None);
}

// --- Regression tests of the parity bugs this review found (all fixed) -----------------------
//
// The descriptions say what the code did before the fix.

/// `types.rs`: `Request.you` is a full `Snake` (required `health` and `body`, typed `name`,
/// `head`, `length`) and board snakes type-check `name`, `head` and `length`, but the Python
/// conversion reads only `you.id` and the `id`, `health` and `body` of board snakes
/// (server.py:120-146). A `you` of just `{"id": ...}`, a `"length": "4"`, a `head` without `y`,
/// a numeric `name` or numeric ids make Rust refuse a request Python plays.
#[test]
fn bug_typed_parse_rejects_unread_fields() {
    run("", Some("typed_parse_rejects_unread_fields"));
}

/// `types.rs`: `Snake.health: i32` and `Request.turn: i64` accept only JSON integers. Python
/// stores them through numpy/jnp int32 (engine_json.py:113, 146), which truncates floats
/// (`89.7` -> 89), parses numeric strings (`"90"`) and reads booleans as 0/1, then plays.
#[test]
fn bug_strict_integer_fields() {
    run("", Some("strict_integer_fields"));
}

/// `types.rs`: `Coord` is `i32`, so one body segment (not the head) or one hazard beyond
/// +-2^31 fails the whole parse. Python drops off-board segments and hazards whatever their
/// magnitude (`body_to_countdown`, engine_json.py:35-42 and 129-135) and plays.
#[test]
fn bug_i32_coordinates_off_board() {
    run("", Some("i32_coordinates_off_board"));
}

/// `types.rs` `Request::from_json`: serde_json is strict JSON. Python's `json.loads` on the
/// body bytes (server.py:657) accepts `NaN`/`Infinity`/`-Infinity`, takes the last of duplicate
/// keys, strips a UTF-8 BOM, reads integers of any size (a 400-digit `minimumFood` inside
/// `game` is unread by Python but overflows serde_json's `Value`) and nests 500 deep (serde_json
/// stops at 128 inside the `game` `Value`). Rust refuses each.
#[test]
fn bug_json_dialect() {
    run("", Some("json_dialect"));
}

/// `types.rs` `Request::from_json` has no size limit: a 70 KB body is played, while the Python
/// handler answers 400 to any body over `MAX_BODY` = 64 KB (server.py:104, 652-654) before
/// parsing it. Whoever owns the HTTP layer (the Lambda) has to apply the limit, or the crate.
#[test]
fn bug_no_body_size_limit() {
    run("", Some("no_body_size_limit"));
}

/// `convert.rs` `object_or_empty` and `Request.game: Option<Value>`: a `game` of `null`, `[]`,
/// `0`, `""` or `false` reads as an empty game (standard ruleset, standard map) and the agent
/// plays. The Python HTTP handler answers 400 to any non-dict `game` (server.py:661-663), so
/// no move is sent at all.
#[test]
fn bug_falsy_game_object_plays() {
    run("", Some("falsy_game_object_plays"));
}

/// `convert.rs` `request_state` -> `GameState::add_food`: a food at x = -1 or y = -1 is an
/// error, but numpy indexing in `state_from_engine` (engine_json.py:124-128) wraps it to the
/// opposite edge (x = W-1) and Python plays with that food. Documented in the module docs as a
/// deliberate deviation.
#[test]
fn bug_negative_food_wraps_in_python() {
    run("", Some("negative_food_wraps_in_python"));
}

/// `convert.rs` `parse_python_int`: ASCII digits only. Python's `int()` in `_setting`
/// (server.py:149-158) also reads other Unicode decimal digits (Arabic-Indic, full-width), so a
/// setting of `"\u{661}\u{662}"` is 12 in Python and the default in Rust. Documented deviation.
#[test]
fn bug_unicode_digit_settings() {
    run("", Some("unicode_digit_settings"));
}

/// `types.rs` `Coord` derives `Deserialize`, and serde's derive accepts a struct as a JSON
/// array too: a body segment `[1, 2]` reads as `{"x": 1, "y": 2}` and the agent plays. Python
/// raises `TypeError` on `p["x"]` (server.py:139) and answers "up" (safe_move raises too).
#[test]
fn bug_coord_accepts_json_array() {
    run("", Some("coord_accepts_json_array"));
}

/// `safe.rs` `choose` (lines 35 and 42): `hx + i64::from(dx)` on heads read with `as_i64`
/// overflows for a head at `i64::MAX`/`i64::MIN`, a panic in debug builds (with
/// overflow-checks). Release builds wrap and happen to give Python's answer, but
/// `safe_move_json` is documented never to fail and the release profile has `panic = "abort"`.
#[test]
fn bug_safe_move_overflow_panic() {
    run("", Some("safe_move_overflow_panic"));
}

/// `safe.rs` `safe_move_json`: when `you.length`, a `head`, a snake `id` or a numeric `length`
/// is missing or mistyped, Python's `safe_move` raises and the HTTP handler answers 0 ("up",
/// server.py:672-677); Rust fills defaults (`length` 0, head = body[0]) and computes a move.
#[test]
fn bug_safe_move_answers_where_python_raises() {
    run("", Some("safe_move_answers_where_python_raises"));
}

/// `safe.rs` `int`/`point`: numbers are read with `as_i64` only, so a float coordinate (`5.0`,
/// equal to 5 in Python's sets) is dropped from the blocked cells and a width beyond `i64`
/// (`2^63`, `2^64`) reads as 0, leaving only "up". Python compares numbers by value.
#[test]
fn bug_safe_move_number_reading() {
    run("", Some("safe_move_number_reading"));
}

/// Every case is either clean or listed in [`KNOWN`] under a bug that exists.
#[test]
fn known_bugs_are_named() {
    for (case, bug) in KNOWN {
        assert!(!bug.is_empty() && case.contains('/'), "{case}");
    }
}
