//! End-to-end tests of the handler on the real weights.
//!
//! * the four modes' example requests (`/start`, `/move`, `/end`) against `Agent::play` with the
//!   documented seed;
//! * the 466 ladder-shaped requests of `review_ladder.jsonl` (golden vectors of the Python agent,
//!   `$NAZ_GOLDEN`, default `/dev/shm/naz-golden`) through `/start` + `/move` with `TIE_SEED`
//!   set to the golden seed: the move must equal the Python agent's;
//! * routing, error and fallback behaviour; a time budget.
//!
//! The run dir is `$NAZ_RUN` (default: the deployed M network). Without it the tests print SKIP.

use serde_json::{Value, json};
use slinky_lambda::App;
use slinky_lambda::engine::{Settings, tie_seed};
use slinky_search::Agent;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

/// An app serving snake `m` from `run` with extra environment settings.
fn app(run: &Path, extra: &[(&str, &str)]) -> App {
    let parent = run.parent().unwrap().to_str().unwrap().to_string();
    let sub = format!("m={}", run.file_name().unwrap().to_str().unwrap());
    let extra: Vec<(String, String)> = extra
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let settings = Settings::from_lookup(&move |k| match k {
        "WEIGHTS_DIR" => Some(parent.clone()),
        "SNAKES" => Some(sub.clone()),
        _ => extra.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()),
    })
    .unwrap();
    App::load(&settings)
}

fn call(app: &App, method: &str, path: &str, body: &Value) -> (u16, Value) {
    let (code, text) = app.handle(method, path, &body.to_string(), Instant::now());
    (code, serde_json::from_str(&text).unwrap())
}

fn examples() -> Value {
    serde_json::from_str(include_str!("../../server/data/examples.json")).unwrap()
}

#[test]
fn modes_match_agent_with_documented_seed() {
    let Some(run) = run_dir() else { return };
    let app = app(&run, &[]);
    let agent = Agent::load(&run).unwrap();
    let ex = examples();
    for mode in ["standard", "duel", "royale", "constrictor"] {
        let req = &ex["requests"][mode];
        let n = req["board"]["snakes"].as_array().unwrap().len();
        let (c, _) = call(&app, "POST", "/m/start", req);
        assert_eq!(c, 200);
        for turn in [0u64, 7, 40] {
            let mut r = req.clone();
            r["turn"] = json!(turn);
            let (c, v) = call(&app, "POST", "/m/move", &r);
            assert_eq!(c, 200, "{mode}");
            let id = r["game"]["id"].as_str().unwrap();
            let want = agent
                .play(&r.to_string(), Some(n), tie_seed(id, &turn.to_string()))
                .unwrap();
            assert_eq!(v["move"], want.mv.name(), "{mode} turn {turn}");
            assert_eq!(want.search.simulations, 256);
        }
        let (c, v) = call(&app, "POST", "/m/end", req);
        assert_eq!((c, v), (200, json!({})));
    }
}

#[test]
fn ladder_requests_match_python_golden() {
    let Some(run) = run_dir() else { return };
    let gold =
        PathBuf::from(std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into()))
            .join("search/review_ladder.jsonl");
    let Ok(text) = std::fs::read_to_string(&gold) else {
        eprintln!("SKIP: no golden vectors {}", gold.display());
        return;
    };
    let (mut total, mut agree, mut fallbacks) = (0, 0, 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).unwrap();
        if v.get("skip").is_some() {
            continue; // Python plays its heuristic there; the fallback is tested below
        }
        let seed = v["seed"].as_u64().unwrap().to_string();
        let app = app(&run, &[("TIE_SEED", &seed)]);
        let req = &v["request"];
        if let Some(n) = v["starting"].as_u64() {
            let mut start = req.clone();
            start["board"]["snakes"] = json!(vec![json!({}); n as usize]);
            assert_eq!(call(&app, "POST", "/m/start", &start).0, 200);
        }
        let (c, got) = call(&app, "POST", "/m/move", req);
        assert_eq!(c, 200);
        total += 1;
        let want =
            ["up", "down", "left", "right"][v["out"]["action"][0].as_u64().unwrap() as usize];
        if got["move"] == want {
            agree += 1;
        } else {
            eprintln!(
                "DISAGREE {}: lambda {} python {want}",
                v["name"], got["move"]
            );
        }
        fallbacks += (v["out"]["visits"].is_null()) as usize;
    }
    eprintln!("lambda vs python golden: {agree}/{total} moves agree ({fallbacks} without visits)");
    assert!(total > 400, "only {total} golden requests");
    assert_eq!(agree, total);
}

#[test]
fn routing_and_errors() {
    let Some(run) = run_dir() else { return };
    let app = app(&run, &[]);
    let ex = examples();
    let req = &ex["requests"]["duel"];
    let (c, v) = call(&app, "GET", "/m", &json!(null));
    assert_eq!(c, 200);
    assert_eq!(v["apiversion"], "1");
    assert_eq!(call(&app, "GET", "/nope", &json!(null)).0, 404);
    assert_eq!(call(&app, "POST", "/m/other", req).0, 404);
    assert_eq!(call(&app, "GET", "/m/move", req).0, 404);
    assert_eq!(call(&app, "POST", "/m/move/x", req).0, 404);
    assert_eq!(
        app.handle("POST", "/m/move", "not json", Instant::now()).0,
        400
    );
    assert_eq!(app.handle("POST", "/m/move", "{}", Instant::now()).0, 400);
    let big = "x".repeat(70_000);
    assert_eq!(app.handle("POST", "/m/move", &big, Instant::now()).0, 400);
}

#[test]
fn unplayable_games_get_the_safe_move() {
    let Some(run) = run_dir() else { return };
    let app = app(&run, &[]);
    let ex = examples();
    // A lone snake (no mode), a 9x9 board (the network is 11x11): both answer 200 with a move.
    let mut lone = ex["requests"]["duel"].clone();
    lone["board"]["snakes"] = json!([lone["board"]["snakes"][0].clone()]);
    let mut small = ex["requests"]["duel"].clone();
    small["board"]["width"] = json!(9);
    small["board"]["height"] = json!(9);
    for r in [lone, small] {
        let (c, v) = call(&app, "POST", "/m/move", &r);
        assert_eq!(c, 200);
        assert!(["up", "down", "left", "right"].contains(&v["move"].as_str().unwrap()));
    }
    // Missing weights: still a move.
    let bad = Settings::from_lookup(&|k| match k {
        "WEIGHTS_DIR" => Some("/nonexistent".into()),
        "SNAKES" => Some("m=x".into()),
        _ => None,
    })
    .unwrap();
    let bad = App::load(&bad);
    let (c, v) = call(&bad, "POST", "/m/move", &ex["requests"]["standard"]);
    assert_eq!(c, 200);
    assert!(v["move"].is_string());
}

#[test]
fn simulations_threads_and_time_budget() {
    let Some(run) = run_dir() else { return };
    let ex = examples();
    let req = &ex["requests"]["standard"];
    let agent = Agent::load(&run).unwrap();
    // SIMULATIONS=1024: same as the agent with 1024 simulations.
    let a1k = app(&run, &[("SIMULATIONS", "1024"), ("TIE_SEED", "5")]);
    let (_, v) = call(&a1k, "POST", "/m/move", req);
    let want = agent
        .with_simulations(1024)
        .play(&req.to_string(), None, 5)
        .unwrap();
    assert_eq!(v["move"], want.mv.name());
    assert_eq!(want.search.simulations, 1024);
    // BUDGET_MS=60: answers within the budget plus slack, well before 1024-sim time x many.
    let ab = app(&run, &[("BUDGET_MS", "60")]);
    let t = Instant::now();
    let (c, v) = call(&ab, "POST", "/m/move", req);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(c, 200);
    assert!(v["move"].is_string());
    assert!(ms < 120.0, "budgeted move took {ms} ms");
    // THREADS=2 plays a legal move too.
    let a2 = app(&run, &[("THREADS", "2")]);
    assert_eq!(call(&a2, "POST", "/m/move", req).0, 200);
}
