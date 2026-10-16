//! AWS Lambda handler: native Battlesnake API snakes behind one Function URL.
//!
//! ```text
//! GET  /<snake>          snake info
//! POST /<snake>/start    remembers the game's snake count (bounded per-process cache)
//! POST /<snake>/move     {"move": "up" | "down" | "left" | "right"}
//! POST /<snake>/end      (logged)
//! ```
//!
//! The routes and bodies are those of `deploy/lambda/lambda_function.py`, and so is what is
//! accepted: a body is read with Python's `json.loads` dialect (`slinky_api::Request`) and
//! needs a `game.id` of any type; anything else is 400. A move request always gets a move: on
//! any error the safe move. Every search stops at the request's deadline (see [`engine`]). Settings are environment variables
//! (see [`engine`] for the engine's and the README for all of them); the info response reads
//! `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL` and `SNAKE_COLOR_<NAME>`.

pub mod engine;

use engine::{By, Settings, SnakeEngine};
use serde_json::{Value, json};
use slinky_api::{PyValue, Request};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const MAX_BODY: usize = 1 << 16;
const MAX_CACHED_GAMES: usize = 4096;
const DEFAULT_COLOR: &str = "#8b5cf6";

/// Per-process cache of the snake count at `/start`, keyed by game id, bounded
/// (oldest entry evicted first). A Lambda execution environment serves many
/// games, and the board at `/move` no longer shows how many snakes started.
/// (Another container may get the `/move`; then the living snakes are used, as the
/// Python server does after a restart.)
#[derive(Default)]
struct StartCache {
    map: HashMap<String, usize>,
    order: VecDeque<String>,
}

/// The loaded function: every snake's weights and the `/start` cache.
pub struct App {
    snakes: Vec<SnakeEngine>,
    cache: Mutex<StartCache>,
    debug_panic: Option<String>,
}

/// A scalar as JSON for the logs; a list or dict as its type name (a body may nest
/// thousands of levels: nothing here recurses).
pub fn log_value(v: Option<&PyValue>) -> Value {
    match v {
        None | Some(PyValue::None) => Value::Null,
        Some(PyValue::Bool(b)) => json!(b),
        Some(PyValue::Int(i)) => {
            i64::try_from(*i).map_or_else(|_| json!(i.to_string()), |i| json!(i))
        }
        Some(PyValue::BigInt { negative, digits }) => {
            json!(format!("{}{digits}", if *negative { "-" } else { "" }))
        }
        Some(PyValue::Float(f)) => json!(f),
        Some(PyValue::Str(s)) => json!(s),
        Some(v) => json!(format!("<{}>", v.type_name())),
    }
}

/// The text a value enters the tie-break seed with: a string as itself for the game id
/// (`raw`), anything else as its JSON text (`5`, `null`, `"5"` for a string turn).
pub fn seed_text(v: Option<&PyValue>, raw: bool) -> String {
    match (v, raw) {
        (Some(PyValue::Str(s)), true) => s.clone(),
        (v, _) => log_value(v).to_string(),
    }
}

/// An HTTP answer: status code and JSON text.
pub type Reply = (u16, String);

static LOG_TO_STDERR: AtomicBool = AtomicBool::new(false);

/// Writes the JSON log lines to stderr instead of stdout (the engine process, whose stdout
/// carries its answers).
pub fn log_to_stderr() {
    LOG_TO_STDERR.store(true, Ordering::Relaxed);
}

fn log(v: Value) {
    // One JSON line per event in CloudWatch.
    if LOG_TO_STDERR.load(Ordering::Relaxed) {
        eprintln!("{v}");
    } else {
        println!("{v}");
    }
}

fn reply(code: u16, body: Value) -> Reply {
    (code, body.to_string())
}

fn info(name: &str) -> Value {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let color_key = format!(
        "SNAKE_COLOR_{}",
        name.to_uppercase().replace(['-', '.'], "_")
    );
    json!({
        "apiversion": "1",
        "author": env("SNAKE_AUTHOR", "slinky"),
        "color": env(&color_key, DEFAULT_COLOR),
        "head": env("SNAKE_HEAD", "default"),
        "tail": env("SNAKE_TAIL", "default"),
        "version": format!("slinky {name} (native)"),
    })
}

impl App {
    /// Loads every snake's weights (once per container).
    pub fn load(settings: &Settings) -> App {
        let snakes: Vec<SnakeEngine> = settings
            .snakes
            .iter()
            .map(|spec| {
                let t = Instant::now();
                let e = SnakeEngine::load(settings, spec);
                log(json!({
                    "event": "load", "snake": spec.name, "ok": e.load_error().is_none(),
                    "error": e.load_error(), "budget": format!("{:?}", settings.budget_of(spec)),
                    "threads": settings.threads,
                    "ms": (t.elapsed().as_secs_f64() * 1e5).round() / 100.0,
                }));
                e
            })
            .collect();
        App {
            snakes,
            cache: Mutex::new(StartCache::default()),
            debug_panic: settings
                .debug_panic
                .as_ref()
                .filter(|(w, _)| w == "handler")
                .map(|(_, id)| id.clone()),
        }
    }

    fn cache_put(&self, game: &str, snakes: usize) {
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.map.insert(game.to_string(), snakes).is_none() {
            c.order.push_back(game.to_string());
            while c.order.len() > MAX_CACHED_GAMES {
                if let Some(old) = c.order.pop_front() {
                    c.map.remove(&old);
                }
            }
        }
    }

    fn cache_get(&self, game: &str) -> Option<usize> {
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.map.get(game).copied()
    }

    /// Answers one request. `arrival` is when it reached the handler (a time budget counts
    /// from there).
    pub fn handle(&self, method: &str, path: &str, body: &str, arrival: Instant) -> Reply {
        self.handle_within(method, path, body, arrival, None)
    }

    /// [`App::handle`] with an outside deadline (the Lambda invocation's end): a search
    /// stops before it, less the deadline margin.
    pub fn handle_within(
        &self,
        method: &str,
        path: &str,
        body: &str,
        arrival: Instant,
        limit: Option<Instant>,
    ) -> Reply {
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let Some(snake) = parts
            .first()
            .and_then(|p| self.snakes.iter().find(|s| s.name == *p))
        else {
            return reply(404, json!({"error": "not found"}));
        };
        let name = snake.name.as_str();
        if parts.len() > 2 {
            return reply(404, json!({"error": "not found"}));
        }
        let action = parts.get(1).copied().unwrap_or("");
        if method == "GET" && action.is_empty() {
            return reply(200, info(name));
        }
        if method != "POST" || !matches!(action, "start" | "move" | "end") {
            return reply(404, json!({"error": "not found"}));
        }
        if body.len() > MAX_BODY {
            return reply(400, json!({"error": "request too large"}));
        }
        // json.loads(raw); game = req["game"]["id"]; turn = req.get("turn")
        let Ok(req) = Request::from_json(body) else {
            return reply(400, json!({"error": "not a Battlesnake request"}));
        };
        let root = req.root();
        let Some(game) = root.get("game").and_then(|g| g.get("id")) else {
            return reply(400, json!({"error": "not a Battlesnake request"}));
        };
        let game_log = log_value(Some(game));
        let game_seed = seed_text(Some(game), true);
        let turn = root.get("turn");
        let turn_log = log_value(turn);

        match action {
            "move" => {
                if self.debug_panic.as_deref() == Some(game_seed.as_str()) {
                    panic!("DEBUG_PANIC: a panic in the handler, game {game_seed}");
                }
                let starting = req.game_id().and_then(|id| self.cache_get(id));
                let c = snake.choose_request(
                    &req,
                    &game_seed,
                    &seed_text(turn, false),
                    starting,
                    arrival,
                    limit,
                );
                let ms = (arrival.elapsed().as_secs_f64() * 1e5).round() / 100.0;
                log(json!({
                    "snake": name, "game": game_log, "turn": turn_log, "move": c.mv.name(),
                    "by": if c.by == By::Engine { "engine" } else { "fallback" },
                    "sims": c.simulations, "mode": c.mode, "seed": c.seed, "ms": ms,
                    "deadline_ms": c.deadline_ms, "cut": c.cut,
                    "error": c.error.map(|e| e.chars().take(300).collect::<String>()),
                }));
                reply(200, json!({"move": c.mv.name()}))
            }
            "start" => {
                let snakes = req.snake_count().unwrap_or(0);
                if let Some(id) = req.game_id() {
                    self.cache_put(id, snakes);
                }
                let board = root.get("board");
                let g = root.get("game");
                log(json!({
                    "snake": name, "game": game_log, "event": "start",
                    "ruleset": log_value(g.and_then(|g| g.get("ruleset")).and_then(|r| r.get("name"))),
                    "size": format!("{}x{}",
                        log_value(board.and_then(|b| b.get("width"))),
                        log_value(board.and_then(|b| b.get("height")))),
                    "snakes": snakes, "source": log_value(g.and_then(|g| g.get("source"))),
                }));
                reply(200, json!({}))
            }
            _ => {
                let me = log_value(root.get("you").and_then(|y| y.get("id")));
                let won = match root.get("board").and_then(|b| b.get("snakes")) {
                    Some(PyValue::List(left)) => {
                        left.len() == 1 && log_value(left[0].get("id")) == me
                    }
                    _ => false,
                };
                log(
                    json!({"snake": name, "game": game_log, "event": "end", "turn": turn_log,
                           "result": if won { "won" } else { "lost or drew" }}),
                );
                reply(200, json!({}))
            }
        }
    }
}
