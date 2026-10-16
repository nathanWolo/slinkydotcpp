//! AWS Lambda handler: native Battlesnake API snakes behind one Function URL.
//!
//! ```text
//! GET  /<snake>          snake info
//! POST /<snake>/start    remembers the game's snake count (bounded per-process cache)
//! POST /<snake>/move     {"move": "up" | "down" | "left" | "right"}
//! POST /<snake>/end      (logged)
//! ```
//!
//! The routes and bodies are those of `deploy/lambda/lambda_function.py`. A move request
//! always gets a move: on any error the safe move. Settings are environment variables
//! (see [`engine`] for the engine's and the README for all of them); the info response reads
//! `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL` and `SNAKE_COLOR_<NAME>`.

pub mod engine;

use engine::{By, Settings, SnakeEngine};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
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
}

/// An HTTP answer: status code and JSON text.
pub type Reply = (u16, String);

fn log(v: Value) {
    println!("{v}"); // one JSON line per event in CloudWatch
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
        let req: Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(_) => return reply(400, json!({"error": "not a Battlesnake request"})),
        };
        let Some(game) = req["game"]["id"].as_str().map(str::to_string) else {
            return reply(400, json!({"error": "not a Battlesnake request"}));
        };
        let turn = req["turn"].clone();

        match action {
            "move" => {
                let c = snake.choose(
                    body,
                    &game,
                    &turn.to_string(),
                    self.cache_get(&game),
                    arrival,
                );
                let ms = (arrival.elapsed().as_secs_f64() * 1e5).round() / 100.0;
                log(json!({
                    "snake": name, "game": game, "turn": turn, "move": c.mv.name(),
                    "by": if c.by == By::Engine { "engine" } else { "fallback" },
                    "sims": c.simulations, "mode": c.mode, "seed": c.seed, "ms": ms,
                    "error": c.error.map(|e| e.chars().take(300).collect::<String>()),
                }));
                reply(200, json!({"move": c.mv.name()}))
            }
            "start" => {
                let snakes = req["board"]["snakes"].as_array().map_or(0, Vec::len);
                self.cache_put(&game, snakes);
                log(json!({
                    "snake": name, "game": game, "event": "start",
                    "ruleset": req["game"]["ruleset"]["name"],
                    "size": format!("{}x{}", req["board"]["width"], req["board"]["height"]),
                    "snakes": snakes, "source": req["game"]["source"],
                }));
                reply(200, json!({}))
            }
            _ => {
                let me = &req["you"]["id"];
                let left = req["board"]["snakes"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let won = left.len() == 1 && &left[0]["id"] == me;
                log(
                    json!({"snake": name, "game": game, "event": "end", "turn": turn,
                           "result": if won { "won" } else { "lost or drew" }}),
                );
                reply(200, json!({}))
            }
        }
    }
}
