//! Review tests of the handler in process: every `/move` gets a legal-ish answer (a fuzz of
//! mutated requests), the tie-break seed policy, time budgets and their reference point,
//! settings parsing, and the `/start` cache.
//!
//! The run dir is `$NAZ_RUN` (default: the deployed M network); without it the tests SKIP.

use serde_json::{Value, json};
use slinky_lambda::App;
use slinky_lambda::engine::{By, Settings, SnakeEngine, SnakeSpec, tie_seed};
use slinky_search::MoveBudget;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn run_dir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("NAZ_RUN")
            .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into()),
    );
    if d.join("params.npz").exists() {
        Some(d)
    } else {
        eprintln!("SKIP: no run dir {}", d.display());
        None
    }
}

fn examples() -> Value {
    serde_json::from_str(include_str!("../../server/data/examples.json")).unwrap()
}

fn settings(run: &Path, extra: &[(&str, &str)]) -> Settings {
    let parent = run.parent().unwrap().to_str().unwrap().to_string();
    let sub = format!("m={}", run.file_name().unwrap().to_str().unwrap());
    let extra: Vec<(String, String)> = extra
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Settings::from_lookup(&move |k| match k {
        "WEIGHTS_DIR" => Some(parent.clone()),
        "SNAKES" => Some(sub.clone()),
        _ => extra.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()),
    })
    .unwrap()
}

fn engine(run: &Path, extra: &[(&str, &str)]) -> SnakeEngine {
    let s = settings(run, extra);
    SnakeEngine::load(&s, &s.snakes[0])
}

fn mv(app: &App, path: &str, body: &str) -> (u16, Value) {
    let (c, t) = app.handle("POST", path, body, Instant::now());
    (c, serde_json::from_str(&t).unwrap())
}

fn is_move(v: &Value) -> bool {
    ["up", "down", "left", "right"].contains(&v["move"].as_str().unwrap_or(""))
}

// ---------------------------------------------------------------------------------------------
// Every /move gets an answer.

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn odd_value(r: &mut Rng) -> Value {
    match r.below(16) {
        0 => Value::Null,
        1 => json!(true),
        2 => json!(-1),
        3 => json!(0),
        4 => json!(11),
        5 => json!(12),
        6 => json!(i64::MAX),
        7 => json!(u64::MAX),
        8 => json!(-1e300),
        9 => json!(2.5),
        10 => json!(""),
        11 => json!("standard"),
        12 => json!([]),
        13 => json!({}),
        14 => json!([{"x": 0, "y": 0}]),
        _ => json!({"x": 5, "y": 5}),
    }
}

/// One random edit somewhere in the tree.
fn mutate(v: &mut Value, r: &mut Rng, depth: usize) {
    let stop = depth > 0 && r.below(3) == 0;
    match v {
        Value::Object(m) if !m.is_empty() && !stop => {
            let keys: Vec<String> = m.keys().cloned().collect();
            let k = keys[r.below(keys.len())].clone();
            match r.below(6) {
                0 => {
                    m.remove(&k);
                }
                1 => {
                    m.insert(k, odd_value(r));
                }
                _ => mutate(m.get_mut(&k).unwrap(), r, depth + 1),
            }
        }
        Value::Array(a) if !a.is_empty() && !stop => {
            let i = r.below(a.len());
            match r.below(6) {
                0 => {
                    a.remove(i);
                }
                1 => {
                    let x = a[i].clone();
                    a.push(x);
                }
                2 => a[i] = odd_value(r),
                _ => mutate(&mut a[i], r, depth + 1),
            }
        }
        _ => *v = odd_value(r),
    }
}

/// Mutated ladder requests through the handler and through the engine: the handler never
/// panics, a parsable request with a string game id always gets 200 and a move, and the search
/// itself never panics (a caught panic would show as `By::Fallback` with "panicked").
#[test]
fn fuzzed_moves_always_answer() {
    let Some(run) = run_dir() else { return };
    let extra = [("SIMULATIONS", "8"), ("TIE_SEED", "1")];
    let app = App::load(&settings(&run, &extra));
    let eng = engine(&run, &extra);
    let ex = examples();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut engine_moves, mut fallbacks, mut slowest) = (0, 0, Duration::ZERO);
    for case in 0..4000 {
        let mode = ["standard", "duel", "royale", "constrictor"][case % 4];
        let mut req = ex["requests"][mode].clone();
        req["turn"] = json!(r.below(300));
        for _ in 0..1 + r.below(3) {
            mutate(&mut req, &mut r, 0);
        }
        let body = req.to_string();
        let t = Instant::now();
        let out = catch_unwind(AssertUnwindSafe(|| {
            app.handle("POST", "/m/move", &body, Instant::now())
        }));
        let (code, text) = out.unwrap_or_else(|_| panic!("handler panicked on {body}"));
        let has_id = req["game"]["id"].is_string();
        if has_id {
            assert_eq!(code, 200, "{body}");
        }
        if code == 200 {
            let v: Value = serde_json::from_str(&text).unwrap();
            assert!(is_move(&v), "{body}: {text}");
        } else {
            assert_eq!(code, 400, "{body}");
        }
        let c = eng.choose(&body, "g", "0", None, Instant::now());
        slowest = slowest.max(t.elapsed());
        if let Some(e) = &c.error {
            assert!(!e.contains("panicked"), "search panicked on {body}");
        }
        match c.by {
            By::Engine => engine_moves += 1,
            By::Fallback => fallbacks += 1,
        }
    }
    eprintln!("fuzz: {engine_moves} searched, {fallbacks} fallbacks, slowest {slowest:?}");
    assert!(engine_moves > 500, "the fuzz rarely reached the search");
    assert!(slowest < Duration::from_secs(2), "slowest case {slowest:?}");
}

/// Well-formed but odd positions (coordinates on and off the board, overlapping bodies, health
/// 0 to 200, food and hazards anywhere, turns up to 10^6): the search runs and never panics, and
/// every answer is a move.
#[test]
fn fuzzed_positions_always_answer() {
    let Some(run) = run_dir() else { return };
    let extra = [("SIMULATIONS", "16"), ("TIE_SEED", "2")];
    let eng = engine(&run, &extra);
    let ex = examples();
    let mut r = Rng(0x0123_4567_89ab_cdef);
    let (mut engine_moves, mut fallbacks) = (0, 0);
    let coord = |r: &mut Rng| {
        let v = r.below(15) as i64 - 2; // -2 ..= 12
        json!(v)
    };
    for case in 0..3000 {
        let mode = ["standard", "duel", "royale", "constrictor"][case % 4];
        let mut req = ex["requests"][mode].clone();
        req["turn"] = json!([0, 1, 7, 99, 150, 1_000_000][r.below(6)]);
        let n = req["board"]["snakes"].as_array().unwrap().len();
        for s in 0..n {
            let len = 1 + r.below(8);
            let (mut x, mut y) = (r.below(11) as i64, r.below(11) as i64);
            let mut body = Vec::new();
            for _ in 0..len {
                body.push(json!({"x": x, "y": y}));
                match r.below(5) {
                    0 => x += 1,
                    1 => x -= 1,
                    2 => y += 1,
                    3 => y -= 1,
                    _ => {} // stacked
                }
            }
            if r.below(20) == 0 {
                body[0] = json!({"x": coord(&mut r), "y": coord(&mut r)});
            }
            let sn = &mut req["board"]["snakes"][s];
            sn["body"] = json!(body.clone());
            sn["head"] = body[0].clone();
            sn["length"] = json!(len);
            sn["health"] = json!([0, 1, 50, 100, 200][r.below(5)]);
            if s == 0 {
                req["you"] = req["board"]["snakes"][0].clone();
            }
        }
        let food: Vec<Value> = (0..r.below(6))
            .map(|_| json!({"x": coord(&mut r), "y": coord(&mut r)}))
            .collect();
        req["board"]["food"] = json!(food);
        if mode == "royale" || r.below(4) == 0 {
            let hz: Vec<Value> = (0..r.below(40))
                .map(|_| json!({"x": coord(&mut r), "y": coord(&mut r)}))
                .collect();
            req["board"]["hazards"] = json!(hz);
        }
        let body = req.to_string();
        let c = eng.choose(&body, "g", "0", Some(n), Instant::now());
        if let Some(e) = &c.error {
            assert!(!e.contains("panicked"), "search panicked on {body}");
        }
        match c.by {
            By::Engine => engine_moves += 1,
            By::Fallback => fallbacks += 1,
        }
    }
    eprintln!("position fuzz: {engine_moves} searched, {fallbacks} fallbacks");
    assert!(
        engine_moves > 1500,
        "only {engine_moves} reached the search"
    );
}

/// Python's lambda handler takes any `game.id` that exists (`req["game"]["id"]`, then logs it);
/// the ladder server even plays without one. (Fixed: the native handler answered 400, no move,
/// when the id was not a string.)
#[test]
fn non_string_game_id_still_gets_a_move() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[("SIMULATIONS", "8")]));
    for id in [json!(12345), json!(null), json!(["a"])] {
        let mut req = examples()["requests"]["duel"].clone();
        req["game"]["id"] = id.clone();
        let (c, v) = mv(&app, "/m/move", &req.to_string());
        assert_eq!(c, 200, "game id {id}");
        assert!(is_move(&v));
    }
}

/// The handler used to pre-parse the body with `serde_json` to find the game id, and answered
/// 400 when that failed (fixed: it now reads the body as `slinky_api::Request`). `slinky_api::Request::from_json` reads requests with Python's `json.loads`
/// semantics, and Python's handlers answer these with a move: a request the engine itself would
/// read must not be refused by the pre-parse.
#[test]
fn pre_parse_accepts_what_the_engine_reads() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[("SIMULATIONS", "8")]));
    let req = examples()["requests"]["duel"].to_string();
    let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
    let bodies = [
        (
            "NaN health",
            req.replacen("\"health\": 100", "\"health\": NaN", 1)
                .replacen("\"health\":100", "\"health\":NaN", 1),
        ),
        (
            "lone surrogate in a name",
            req.replacen("\"name\":\"", "\"name\":\"\\ud800", 1),
        ),
        ("UTF-8 BOM", format!("\u{feff}{req}")),
        (
            "deep unused field",
            req.replacen('{', &format!("{{\"x\":{deep},"), 1),
        ),
    ];
    let mut refused = Vec::new();
    for (what, body) in &bodies {
        assert_ne!(body, &req, "{what}: the edit did not apply");
        let engine_reads = slinky_api::Request::from_json(body).is_ok();
        let (c, _) = app.handle("POST", "/m/move", body, Instant::now());
        if engine_reads && c != 200 {
            refused.push(*what);
        }
    }
    assert!(refused.is_empty(), "refused with 400: {refused:?}");
}

/// Lone snakes, 5+ snakes, other board sizes and rulesets all answer a legal-ish move, and a
/// move that has a free neighbour never walks into a wall or a body (the safe move).
#[test]
fn unsupported_games_answer_a_safe_move() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[]));
    let ex = examples();
    let base = ex["requests"]["standard"].clone();
    let mut cases = Vec::new();
    for (w, h) in [(7, 7), (19, 19), (25, 25), (11, 9)] {
        let mut r = base.clone();
        r["board"]["width"] = json!(w);
        r["board"]["height"] = json!(h);
        cases.push(r);
    }
    for ruleset in ["wrapped", "solo", "squad", "bogus"] {
        let mut r = base.clone();
        r["game"]["ruleset"]["name"] = json!(ruleset);
        cases.push(r);
    }
    let mut five = base.clone();
    let mut extra = base["board"]["snakes"][0].clone();
    extra["id"] = json!("s4");
    extra["body"] = json!([{"x": 5, "y": 1}, {"x": 5, "y": 1}, {"x": 5, "y": 1}]);
    extra["head"] = json!({"x": 5, "y": 1});
    five["board"]["snakes"].as_array_mut().unwrap().push(extra);
    cases.push(five);
    // Our snake in the corner (9,1 -> 10,0): up and left are free; down and right are walls.
    let mut corner = base.clone();
    let c = json!([{"x": 10, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 0}]);
    corner["board"]["snakes"][0]["body"] = c.clone();
    corner["board"]["snakes"][0]["head"] = json!({"x": 10, "y": 0});
    corner["you"]["body"] = c;
    corner["you"]["head"] = json!({"x": 10, "y": 0});
    corner["board"]["width"] = json!(13); // unsupported: the safe move decides
    corner["board"]["height"] = json!(13);
    for r in &cases {
        let (c, v) = mv(&app, "/m/move", &r.to_string());
        assert_eq!(c, 200, "{r}");
        assert!(is_move(&v), "{r}");
    }
    let (_, v) = mv(&app, "/m/move", &corner.to_string());
    assert!(
        ["up", "left"].contains(&v["move"].as_str().unwrap()),
        "corner: {v}"
    );
}

// ---------------------------------------------------------------------------------------------
// Tie-break seed policy.

/// The seed is `tie_seed(<raw game id>, <turn JSON>)` as the handler passes them: the same request
/// gets the same move in two independent containers, a retry gets the same move, and the seed
/// depends on the game and the turn only (not on the snake, the body or a counter).
#[test]
fn seed_policy_is_per_request_and_container_independent() {
    let Some(run) = run_dir() else { return };
    let ex = examples();
    let a = App::load(&settings(&run, &[]));
    let b = App::load(&settings(&run, &[]));
    let e = engine(&run, &[]);
    for mode in ["standard", "duel", "royale", "constrictor"] {
        for turn in [0u64, 1, 2, 3, 50] {
            let mut req = ex["requests"][mode].clone();
            req["turn"] = json!(turn);
            let body = req.to_string();
            // Interleave traffic so a hidden counter would show.
            let m1 = mv(&a, "/m/move", &body).1;
            let _ = mv(&a, "/m/move", &ex["requests"]["duel"].to_string());
            let m2 = mv(&b, "/m/move", &body).1;
            let m3 = mv(&a, "/m/move", &body).1;
            assert_eq!(m1, m2, "{mode} turn {turn}: containers disagree");
            assert_eq!(m1, m3, "{mode} turn {turn}: retry differs");
            let id = req["game"]["id"].as_str().unwrap();
            let c = e.choose(&body, id, &turn.to_string(), None, Instant::now());
            assert_eq!(c.seed, tie_seed(id, &turn.to_string()));
            assert_eq!(json!(c.mv.name()), m1["move"], "{mode} turn {turn}");
        }
    }
}

/// `tie_seed`'s documentation says `game_id` is "the id's JSON text" (quoted), but the handler
/// passes the raw string; the README's recipe (`naz --seed <tie_seed(game id, turn)>`) follows
/// the handler. The two disagree for every id; this pins the handler's convention.
#[test]
fn seed_uses_the_raw_game_id() {
    let Some(run) = run_dir() else { return };
    let e = engine(&run, &[]);
    let req = examples()["requests"]["duel"].clone();
    let id = req["game"]["id"].as_str().unwrap();
    let app = App::load(&settings(&run, &[]));
    let got = mv(&app, "/m/move", &req.to_string()).1;
    let raw = e.choose(&req.to_string(), id, "0", None, Instant::now());
    assert_eq!(json!(raw.mv.name()), got["move"]);
    assert_ne!(tie_seed(id, "0"), tie_seed(&format!("\"{id}\""), "0"));
}

/// The turn enters as its JSON text, so `5` and `5.0` (the same turn to Python's `int()`) get
/// different seeds; a missing turn hashes as `null`. Documented here, not a bug.
#[test]
fn seed_turn_text() {
    assert_ne!(tie_seed("g", "5"), tie_seed("g", "5.0"));
    assert_eq!(tie_seed("g", "null"), tie_seed("g", "null"));
    // Separator: an id ending in the turn's digits cannot collide with a shorter id.
    assert_ne!(tie_seed("g1", "2"), tie_seed("g", "12"));
}

// ---------------------------------------------------------------------------------------------
// Time budgets.

/// `BUDGET_MS` counts from the arrival the handler is given, not from the start of the search.
#[test]
fn budget_counts_from_arrival() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[("BUDGET_MS", "150")]));
    let body = examples()["requests"]["standard"].to_string();
    // Arrived 100 ms ago: about 50 ms of search remain.
    let t = Instant::now();
    let (c, v) = app.handle(
        "POST",
        "/m/move",
        &body,
        Instant::now() - Duration::from_millis(100),
    );
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(c, 200);
    assert!(v.contains("move"));
    assert!((30.0..150.0).contains(&ms), "searched {ms} ms");
    // Arrived after the deadline: one simulation, answered at once.
    let t = Instant::now();
    let (c, _) = app.handle(
        "POST",
        "/m/move",
        &body,
        Instant::now() - Duration::from_secs(2),
    );
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(c, 200);
    assert!(ms < 40.0, "late request took {ms} ms");
}

/// A time budget with two root-parallel trees holds for every mode.
#[test]
fn budget_with_threads_holds_in_every_mode() {
    let Some(run) = run_dir() else { return };
    let e = engine(&run, &[("BUDGET_MS", "120"), ("THREADS", "2")]);
    let ex = examples();
    for mode in ["standard", "duel", "royale", "constrictor"] {
        let body = ex["requests"][mode].to_string();
        for _ in 0..3 {
            let t = Instant::now();
            let c = e.choose(&body, "g", "0", None, t);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(c.by, By::Engine, "{mode}: {:?}", c.error);
            assert!(ms < 160.0, "{mode}: {ms} ms");
            assert!(c.simulations > 64, "{mode}: {} simulations", c.simulations);
        }
    }
}

/// Any accepted setting must answer within Battlesnake's 500 ms on this (fast) machine. (Fixed:
/// without `BUDGET_MS` nothing bounded the time: `SIMULATIONS` accepts up to 2^20 and the
/// game's own `timeout` was not read, so the move only ended at the tree's memory cap many
/// seconds later. Now every search stops at `game.timeout - DEADLINE_MARGIN_MS`.)
#[test]
fn accepted_simulations_answer_in_time() {
    let Some(run) = run_dir() else { return };
    let e = engine(&run, &[("SIMULATIONS", "1048576")]);
    let mut req = examples()["requests"]["duel"].clone();
    req["game"]["timeout"] = json!(500);
    let t = Instant::now();
    let c = e.choose(&req.to_string(), "g", "0", None, t);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!("{} simulations in {ms:.0} ms", c.simulations);
    assert!(ms < 500.0, "{} simulations took {ms:.0} ms", c.simulations);
}

/// A game with a shorter `timeout` than `BUDGET_MS` should be answered within the game's timeout.
#[test]
fn game_timeout_caps_the_budget() {
    let Some(run) = run_dir() else { return };
    let e = engine(&run, &[("BUDGET_MS", "400")]);
    let mut req = examples()["requests"]["duel"].clone();
    req["game"]["timeout"] = json!(200);
    let t = Instant::now();
    let _ = e.choose(&req.to_string(), "g", "0", None, t);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(ms < 200.0, "answered after {ms:.0} ms");
}

// ---------------------------------------------------------------------------------------------
// Settings.

fn from(pairs: &[(&str, &str)]) -> Result<Settings, String> {
    let pairs: Vec<(String, String)> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Settings::from_lookup(&move |k| pairs.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()))
}

#[test]
fn settings_whitespace_and_empty_values() {
    let s = from(&[
        (
            "SNAKES",
            " a = m , , b=m: simulations = 64 :budget_ms= 300 ,",
        ),
        ("SIMULATIONS", " 1024 "),
        ("THREADS", "  "), // empty: the default
        ("TIE_SEED", "4294967295"),
        ("MEMORY_LIMIT_MB", "4096"),
    ])
    .unwrap();
    assert_eq!(s.snakes.len(), 2);
    assert_eq!(
        (s.snakes[0].name.as_str(), s.snakes[0].subdir.as_str()),
        ("a", "m")
    );
    assert_eq!(s.budget_of(&s.snakes[0]), MoveBudget::Simulations(1024));
    assert_eq!(
        s.budget_of(&s.snakes[1]),
        MoveBudget::Time {
            ms: 300,
            max_simulations: 64
        }
    );
    assert_eq!(s.threads, 1);
    assert_eq!(s.tie_seed, Some(u32::MAX));
    assert_eq!(s.memory_limit, 4096 << 20);
    for bad in [
        ("TIE_SEED", "4294967296"),
        ("TIE_SEED", "-1"),
        ("MEMORY_LIMIT_MB", "7"),
        ("MEMORY_LIMIT_MB", "4097"),
        ("THREADS", "17"),
        ("SIMULATIONS", "1048577"),
        ("SIMULATIONS", "1e3"),
        ("BUDGET_MS", "0"),
        ("BUDGET_MS", "10001"),
        ("SNAKES", ",,,"),
        ("SNAKES", "=m"),
        ("SNAKES", "a=m:simulations"),
        ("SNAKES", "a=m:simulations=0"),
        ("SNAKES", "a=x/../../etc"),
    ] {
        assert!(from(&[bad]).is_err(), "{bad:?} accepted");
    }
}

/// The same option twice in one `SNAKES` item is silently resolved (the last wins), and an empty
/// subdir serves `WEIGHTS_DIR` itself. Pinned so a change is deliberate.
#[test]
fn settings_lenient_cases() {
    let s = SnakeSpec::parse("a=m:simulations=64:simulations=128").unwrap();
    assert_eq!(s.simulations, Some(128));
    let s = SnakeSpec::parse("a=").unwrap();
    assert_eq!(s.subdir, "");
}

/// A per-snake `simulations=` cannot opt out of a global `BUDGET_MS`: the snake becomes a timed
/// (non-deterministic) search capped at its simulations. Pinned as documented behaviour.
#[test]
fn per_snake_simulations_under_global_budget() {
    let s = from(&[("SNAKES", "a=m:simulations=256"), ("BUDGET_MS", "300")]).unwrap();
    assert_eq!(
        s.budget_of(&s.snakes[0]),
        MoveBudget::Time {
            ms: 300,
            max_simulations: 256
        }
    );
}

// ---------------------------------------------------------------------------------------------
// The /start cache.

/// The count remembered at `/start` is per game id, survives many other games, and an old game
/// is evicted only after 4096 newer ones.
#[test]
fn start_cache_is_per_game_and_bounded() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[("SIMULATIONS", "4"), ("TIE_SEED", "0")]));
    let e = engine(&run, &[("SIMULATIONS", "4"), ("TIE_SEED", "0")]);
    let ex = examples();
    let start = ex["requests"]["standard"].clone();
    let mut two = start.clone();
    let left: Vec<Value> = two["board"]["snakes"].as_array().unwrap()[..2].to_vec();
    two["board"]["snakes"] = json!(left);
    two["turn"] = json!(40);
    // The mode the engine plays a two-snake board with a known start count.
    let mode = |n: Option<usize>| {
        e.choose(&two.to_string(), "g", "40", n, Instant::now())
            .mode
            .unwrap()
    };
    assert_eq!(mode(Some(4)), "standard");
    assert_eq!(mode(None), "duel");
    // The handler: the move of a started game equals the engine with Some(4).
    let with = |id: &str| {
        let mut r = two.clone();
        r["game"]["id"] = json!(id);
        r.to_string()
    };
    let started = |app: &App, id: &str| {
        let mut s = start.clone();
        s["game"]["id"] = json!(id);
        assert_eq!(mv(app, "/m/start", &s.to_string()).0, 200);
    };
    started(&app, "first");
    for i in 0..4095 {
        started(&app, &format!("other-{i}"));
    }
    let want4 = e
        .choose(&with("first"), "first", "40", Some(4), Instant::now())
        .mv;
    let want_none = e
        .choose(&with("first"), "first", "40", None, Instant::now())
        .mv;
    let got = mv(&app, "/m/move", &with("first")).1;
    assert_eq!(got["move"], want4.name(), "still cached after 4095 others");
    started(&app, "one-more");
    let got = mv(&app, "/m/move", &with("first")).1;
    if want4 != want_none {
        assert_eq!(got["move"], want_none.name(), "evicted after 4096 others");
    }
    // A /start with no snakes list caches 0, which must not shrink a game below its living snakes.
    let mut empty = start.clone();
    empty["game"]["id"] = json!("no-snakes");
    empty["board"].as_object_mut().unwrap().remove("snakes");
    assert_eq!(mv(&app, "/m/start", &empty.to_string()).0, 200);
    let mut r = ex["requests"]["standard"].clone();
    r["game"]["id"] = json!("no-snakes");
    let (c, v) = mv(&app, "/m/move", &r.to_string());
    assert_eq!(c, 200);
    let want = e
        .choose(&r.to_string(), "no-snakes", "0", None, Instant::now())
        .mv;
    assert_eq!(v["move"], want.name());
}

// ---------------------------------------------------------------------------------------------
// Cold start: weights that do not load.

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "slinky-lambda-review-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Broken run dirs (no config, bad JSON, truncated or bit-flipped params.npz, an empty file, a
/// directory in place of the file) load as errors, never panics, and the snake still answers.
#[test]
fn broken_weights_load_as_errors() {
    let Some(run) = run_dir() else { return };
    let config = std::fs::read(run.join("config.json")).unwrap();
    let params = std::fs::read(run.join("params.npz")).unwrap();
    let root = scratch("weights");
    let mut cases: Vec<(String, Vec<u8>, Vec<u8>)> = vec![
        ("noparams".into(), config.clone(), Vec::new()),
        ("badjson".into(), b"{not json".to_vec(), params.clone()),
        ("emptyjson".into(), b"{}".to_vec(), params.clone()),
        ("emptynpz".into(), config.clone(), Vec::new()),
    ];
    for frac in [1, 10, 50, 90, 99] {
        let n = params.len() * frac / 100;
        cases.push((format!("trunc{frac}"), config.clone(), params[..n].to_vec()));
    }
    let mut r = Rng(42);
    for i in 0..40 {
        let mut p = params.clone();
        // Flip bytes in the first 4 KiB (zip and npy headers) and anywhere.
        for _ in 0..4 {
            let at = if r.below(2) == 0 {
                r.below(4096.min(p.len()))
            } else {
                r.below(p.len())
            };
            p[at] ^= 1 << r.below(8);
        }
        cases.push((format!("flip{i}"), config.clone(), p));
    }
    let body = examples()["requests"]["duel"].to_string();
    let mut errors = 0;
    for (name, cfg, npz) in &cases {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), cfg).unwrap();
        if name == "noparams" {
            std::fs::create_dir_all(d.join("params.npz")).unwrap(); // a directory
        } else {
            std::fs::write(d.join("params.npz"), npz).unwrap();
        }
        let s = Settings::from_lookup(&|k| match k {
            "WEIGHTS_DIR" => Some(root.to_str().unwrap().to_string()),
            "SNAKES" => Some(format!("m={name}")),
            "SIMULATIONS" => Some("8".into()),
            _ => None,
        })
        .unwrap();
        let loaded = catch_unwind(AssertUnwindSafe(|| App::load(&s)));
        let app = loaded.unwrap_or_else(|_| panic!("{name}: loading panicked"));
        let out = catch_unwind(AssertUnwindSafe(|| {
            app.handle("POST", "/m/move", &body, Instant::now())
        }));
        let (c, t) = out.unwrap_or_else(|_| panic!("{name}: move panicked"));
        assert_eq!(c, 200, "{name}");
        assert!(is_move(&serde_json::from_str(&t).unwrap()), "{name}");
        let e = SnakeEngine::load(&s, &s.snakes[0]);
        errors += e.load_error().is_some() as usize;
    }
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        errors >= 9,
        "only {errors} broken run dirs reported an error"
    );
}

/// One flipped bit in the weights' data (a failing disk, a bad copy) must be a load error.
/// (Fixed: the npz reader did not check the zip entries' CRC-32, so the snake played a
/// corrupted network and logged `"ok": true`.)
#[test]
fn corrupted_weight_data_is_a_load_error() {
    let Some(run) = run_dir() else { return };
    let config = std::fs::read(run.join("config.json")).unwrap();
    let params = std::fs::read(run.join("params.npz")).unwrap();
    let root = scratch("crc");
    let mut silent = Vec::new();
    for frac in [30usize, 50, 70] {
        let mut p = params.clone();
        let at = p.len() * frac / 100;
        p[at] ^= 0x40;
        let name = format!("flip{frac}");
        let d = root.join(&name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), &config).unwrap();
        std::fs::write(d.join("params.npz"), &p).unwrap();
        let s = Settings::from_lookup(&|k| match k {
            "WEIGHTS_DIR" => Some(root.to_str().unwrap().to_string()),
            "SNAKES" => Some(format!("m={name}")),
            _ => None,
        })
        .unwrap();
        if SnakeEngine::load(&s, &s.snakes[0]).load_error().is_none() {
            silent.push(at);
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        silent.is_empty(),
        "bit flips at {silent:?} loaded without an error"
    );
}

/// With `THREADS=2` and a fixed simulation count the move is still a function of the request:
/// two containers and a retry agree.
#[test]
fn threads_with_simulations_are_deterministic() {
    let Some(run) = run_dir() else { return };
    let ex = examples();
    let a = App::load(&settings(&run, &[("THREADS", "2")]));
    let b = App::load(&settings(&run, &[("THREADS", "2")]));
    for mode in ["standard", "duel", "royale", "constrictor"] {
        for turn in [0u64, 9] {
            let mut req = ex["requests"][mode].clone();
            req["turn"] = json!(turn);
            let body = req.to_string();
            let m1 = mv(&a, "/m/move", &body).1;
            let m2 = mv(&b, "/m/move", &body).1;
            let m3 = mv(&a, "/m/move", &body).1;
            assert!(is_move(&m1));
            assert_eq!((&m1, &m1), (&m2, &m3), "{mode} turn {turn}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Deadlines (the fix of the two time-budget bugs above).

/// The deadline is `game.timeout - DEADLINE_MARGIN_MS` from the arrival, 500 ms when the request
/// has no usable timeout; `BUDGET_MS` beyond it is cut; a request that arrived after its deadline
/// is answered at once with one simulation's move.
#[test]
fn deadline_follows_the_game_timeout() {
    let Some(run) = run_dir() else { return };
    let base = examples()["requests"]["duel"].clone();
    let with = |t: Value| {
        let mut r = base.clone();
        r["game"]["timeout"] = t;
        r.to_string()
    };
    let e = engine(
        &run,
        &[("SIMULATIONS", "1048576"), ("DEADLINE_MARGIN_MS", "100")],
    );
    for (t, want) in [
        (json!(500), 400u64),
        (json!(300), 200),
        (json!(80), 0),
        (json!(null), 400),
        (json!("500"), 400),
        (json!(-5), 400),
        (json!(250.7), 150),
    ] {
        let a = Instant::now();
        let c = e.choose(&with(t.clone()), "g", "0", None, a);
        let ms = a.elapsed().as_secs_f64() * 1e3;
        assert_eq!(c.by, By::Engine, "timeout {t}: {:?}", c.error);
        assert_eq!(c.deadline_ms, want, "timeout {t}");
        assert!(
            c.cut,
            "timeout {t}: not cut ({} simulations)",
            c.simulations
        );
        assert!(
            ms >= want as f64 - 1.0 && ms < want as f64 + 40.0,
            "timeout {t}: answered after {ms:.0} ms"
        );
    }
    // BUDGET_MS past the deadline is cut to it; below it, it applies.
    let b = engine(&run, &[("BUDGET_MS", "450")]);
    let c = b.choose(&with(json!(500)), "g", "0", None, Instant::now());
    assert_eq!(c.deadline_ms, 350);
    let c = b.choose(&with(json!(1000)), "g", "0", None, Instant::now());
    assert_eq!(c.deadline_ms, 450);
    // Arrived after the deadline: one simulation, the engine's move.
    let late = Instant::now() - Duration::from_secs(1);
    let t = Instant::now();
    let c = e.choose(&with(json!(500)), "g", "0", None, late);
    assert_eq!((c.by, c.simulations), (By::Engine, 1));
    assert!(t.elapsed() < Duration::from_millis(40));
}

/// A fixed simulation count that fits in the deadline plays exactly as the plain agent: the
/// deadline never changes a faithful move (every mode, 256 and 1024 simulations, two threads).
#[test]
fn deadline_keeps_the_faithful_search() {
    let Some(run) = run_dir() else { return };
    let ex = examples();
    for (extra, sims) in [
        (&[][..], 256usize),
        (&[("SIMULATIONS", "1024")][..], 1024),
        (&[("THREADS", "2")][..], 256),
    ] {
        let e = engine(&run, extra);
        for mode in ["standard", "duel", "royale", "constrictor"] {
            for turn in [0u64, 3, 17] {
                let mut req = ex["requests"][mode].clone();
                req["turn"] = json!(turn);
                let body = req.to_string();
                let c = e.choose(&body, "gid", &turn.to_string(), None, Instant::now());
                assert_eq!(
                    (c.by, c.simulations, c.cut),
                    (By::Engine, sims, false),
                    "{mode}"
                );
                // No deadline at all: an agent with the same budget.
                let s = settings(&run, extra);
                let agent = e.agent().unwrap();
                let game = slinky_api::game_from_request(
                    &slinky_api::Request::from_json(&body).unwrap(),
                    None,
                )
                .unwrap();
                let seed = tie_seed("gid", &turn.to_string());
                let mut ws: Vec<slinky_search::Search> = (0..s.threads)
                    .map(|_| slinky_search::Search::new())
                    .collect();
                let m = agent
                    .search_game_threads(
                        &mut ws,
                        &game,
                        slinky_search::Budget::simulations(sims),
                        seed,
                    )
                    .unwrap();
                assert_eq!(m.mv, c.mv, "{mode} turn {turn} {extra:?}");
            }
        }
    }
}

/// A panic in the search of one request answers the safe move and leaves the engine working
/// (in process: the test harness always unwinds; `tests/runtime.rs` checks the binary).
#[test]
fn search_panic_answers_the_safe_move() {
    let Some(run) = run_dir() else { return };
    let app = App::load(&settings(&run, &[("DEBUG_PANIC", "search:boom")]));
    let mut req = examples()["requests"]["duel"].clone();
    let ok = req.to_string();
    req["game"]["id"] = json!("boom");
    let (c, v) = mv(&app, "/m/move", &req.to_string());
    assert_eq!(c, 200);
    assert!(is_move(&v));
    let (c, v) = mv(&app, "/m/move", &ok);
    assert!(c == 200 && is_move(&v));
}
