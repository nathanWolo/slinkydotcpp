//! AWS Lambda handler: native Battlesnake API snakes behind one Function URL.
//!
//! Same routing as deploy/lambda/lambda_function.py:
//!
//! ```text
//! GET  /<snake>          snake info
//! POST /<snake>/start    remembers the game's snake count (logged)
//! POST /<snake>/move     {"move": "up" | "down" | "left" | "right"}
//! POST /<snake>/end      (logged)
//! ```
//!
//! Settings (environment variables): `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL`,
//! `SNAKE_COLOR_<NAME>` for the info responses; `WEIGHTS_DIR` (default
//! `/var/task/weights/az-modes`) for the model directory.
//!
//! A move request always gets a move: on any error it answers "up".

use lambda_http::{Body, Error, Request, RequestExt, Response, run, service_fn};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

const MAX_BODY: usize = 1 << 16;
const DEFAULT_WEIGHTS_DIR: &str = "/var/task/weights/az-modes";
/// Snake names served (URL path prefix) and their default colours.
const SNAKES: &[(&str, &str)] = &[("slinky", "#8b5cf6")];
const MAX_CACHED_GAMES: usize = 4096;

/// Per-process cache of the snake count at `/start`, keyed by game id, bounded
/// (oldest entry evicted first). A Lambda execution environment serves many
/// games, and the board at `/move` no longer shows how many snakes started.
struct StartCache {
    map: HashMap<String, usize>,
    order: VecDeque<String>,
}

static START_SNAKES: Mutex<Option<StartCache>> = Mutex::new(None);

fn cache_put(game: &str, snakes: usize) {
    let mut g = START_SNAKES.lock().unwrap_or_else(|e| e.into_inner());
    let c = g.get_or_insert_with(|| StartCache {
        map: HashMap::new(),
        order: VecDeque::new(),
    });
    if c.map.insert(game.to_string(), snakes).is_none() {
        c.order.push_back(game.to_string());
        while c.order.len() > MAX_CACHED_GAMES {
            if let Some(old) = c.order.pop_front() {
                c.map.remove(&old);
            }
        }
    }
}

fn cache_get(game: &str) -> Option<usize> {
    let g = START_SNAKES.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().and_then(|c| c.map.get(game).copied())
}

fn weights_dir() -> String {
    std::env::var("WEIGHTS_DIR").unwrap_or_else(|_| DEFAULT_WEIGHTS_DIR.to_string())
}

// ---------------------------------------------------------------------------
// ENGINE PLUG-IN POINT (the only place the real engine goes).
//
// Replace the body with
//     slinky_search::choose_move(request_json, start_snakes)
// i.e. `fn choose_move(request_json: &str, start_snakes: Option<usize>)
//          -> Result<String, Error>`, returning the move name. The engine loads
// its weights from `weights_dir()` (once per process, e.g. in a OnceLock).
// ---------------------------------------------------------------------------
fn engine_choose_move(request_json: &str, start_snakes: Option<usize>) -> Result<String, Error> {
    let _ = (request_json, start_snakes, weights_dir());
    Ok("up".to_string())
}

fn log(v: Value) {
    println!("{v}"); // one JSON line per event in CloudWatch
}

fn json_response(code: u16, body: Value) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(code)
        .header("Content-Type", "application/json")
        .body(Body::Text(body.to_string()))?)
}

fn info(name: &str, default_color: &str) -> Value {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let color_key = format!("SNAKE_COLOR_{}", name.to_uppercase().replace('-', "_"));
    json!({
        "apiversion": "1",
        "author": env("SNAKE_AUTHOR", "slinky"),
        "color": env(&color_key, default_color),
        "head": env("SNAKE_HEAD", "default"),
        "tail": env("SNAKE_TAIL", "default"),
        "version": format!("slinky {name} (native)"),
    })
}

async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let method = event.method().as_str().to_string();
    let path = event.raw_http_path().to_string();
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let Some(&(name, color)) = parts
        .first()
        .and_then(|p| SNAKES.iter().find(|(n, _)| n == p))
    else {
        return json_response(404, json!({"error": "not found"}));
    };
    if parts.len() > 2 {
        return json_response(404, json!({"error": "not found"}));
    }
    let action = parts.get(1).copied().unwrap_or("");
    if method == "GET" && action.is_empty() {
        return json_response(200, info(name, color));
    }
    if method != "POST" || !matches!(action, "start" | "move" | "end") {
        return json_response(404, json!({"error": "not found"}));
    }

    let raw = match event.body() {
        Body::Text(s) => s.clone(),
        Body::Binary(b) => String::from_utf8_lossy(b).into_owned(),
        _ => String::new(),
    };
    if raw.len() > MAX_BODY {
        return json_response(400, json!({"error": "request too large"}));
    }
    let req: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return json_response(400, json!({"error": "not a Battlesnake request"})),
    };
    let Some(game) = req["game"]["id"].as_str().map(str::to_string) else {
        return json_response(400, json!({"error": "not a Battlesnake request"}));
    };
    let turn = req["turn"].clone();

    match action {
        "move" => {
            let t = Instant::now();
            let (mv, by) = match engine_choose_move(&raw, cache_get(&game)) {
                Ok(m) if matches!(m.as_str(), "up" | "down" | "left" | "right") => (m, "engine"),
                Ok(m) => {
                    log(json!({"snake": name, "error": format!("bad move {m:?}")}));
                    ("up".to_string(), "fallback")
                }
                Err(e) => {
                    log(
                        json!({"snake": name, "error": format!("{e}").chars().take(300).collect::<String>()}),
                    );
                    ("up".to_string(), "fallback")
                }
            };
            let ms = (t.elapsed().as_secs_f64() * 1e5).round() / 100.0;
            log(json!({"snake": name, "game": game, "turn": turn, "move": mv, "by": by, "ms": ms}));
            json_response(200, json!({"move": mv}))
        }
        "start" => {
            let snakes = req["board"]["snakes"].as_array().map_or(0, Vec::len);
            cache_put(&game, snakes);
            log(json!({
                "snake": name, "game": game, "event": "start",
                "ruleset": req["game"]["ruleset"]["name"],
                "size": format!("{}x{}", req["board"]["width"], req["board"]["height"]),
                "snakes": snakes, "source": req["game"]["source"],
            }));
            json_response(200, json!({}))
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
            json_response(200, json!({}))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}
