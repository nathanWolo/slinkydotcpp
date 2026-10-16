//! The HTTP side: routes, body limits, timeouts and log lines of the Python server's
//! `make_handler` (server.py:625-756), on hyper.
//!
//! * `GET /` lists the snakes, `GET /<name>` (or `/<name>/`) is a snake's info; anything else is
//!   404 `{"error": "not found"}`.
//! * `POST /<name>/start|move|end` with a body of 1 to 64 KiB (`Content-Length`, read with
//!   Python's `int()`); a body that is not JSON, not an object, or has a `game` that is not an
//!   object is 400 `{"error": ...}`. A body must arrive within 5 s (Python's handler timeout).
//! * `/move` always answers 200 with a move: the agent's, or the safe move when the request
//!   cannot be searched, when the search is not done `answer_by` after the request arrived
//!   (the search threads are all busy, say), or when it fails.
//! * Response bodies are byte for byte what Python's `json.dumps` sends.
//! * Connections are closed after each response, as Python's HTTP/1.0 handler does (no idle
//!   keep-alive connection that a proxy could reuse just as the server closes it).

use crate::engine::{Choice, Snake, Source};
use crate::info;
use crate::ledger::MoveLedger;
use crate::pool::{Job, Pool};
use crate::pyfmt::{json_string, py_str, py_str_opt};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Response, StatusCode};
use slinky_api::py::{Num, py_eq, python_int};
use slinky_api::{MAX_BODY, PyValue, Request, safe_move};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a client may take to send its request (Python's `Handler.timeout`).
pub const BODY_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything the handler serves.
pub struct App {
    snakes: Vec<Arc<Snake>>,
    ledger: Arc<MoveLedger>,
    pool: Pool,
    answer_by: Option<Duration>,
    list_body: Bytes,
    info_bodies: Vec<Bytes>,
}

fn json_dict(items: &[(&str, &str)]) -> String {
    let fields: Vec<String> = items
        .iter()
        .map(|(k, v)| format!("{}: {}", json_string(k), json_string(v)))
        .collect();
    format!("{{{}}}", fields.join(", "))
}

impl App {
    /// Serves `snakes` (in this order at `GET /`), with `threads` search threads. A `/move`
    /// still searching `answer_by` after its request arrived is answered with the safe move
    /// (`None`: wait for the search).
    pub fn new(
        snakes: Vec<Snake>,
        ledger: Arc<MoveLedger>,
        threads: usize,
        answer_by: Option<Duration>,
    ) -> std::io::Result<App> {
        let names: Vec<String> = snakes
            .iter()
            .map(|s| json_string(&format!("/{}", s.name)))
            .collect();
        let list_body = Bytes::from(format!("{{\"snakes\": [{}]}}", names.join(", ")));
        let info_bodies = snakes
            .iter()
            .map(|s| {
                let i = &s.info;
                Bytes::from(json_dict(&[
                    ("apiversion", &i.apiversion),
                    ("author", &i.author),
                    ("color", &i.color),
                    ("head", &i.head),
                    ("tail", &i.tail),
                    ("version", &i.version),
                ]))
            })
            .collect();
        Ok(App {
            snakes: snakes.into_iter().map(Arc::new).collect(),
            ledger,
            pool: Pool::new(threads)?,
            answer_by,
            list_body,
            info_bodies,
        })
    }

    /// The served snakes.
    pub fn snakes(&self) -> &[Arc<Snake>] {
        &self.snakes
    }

    /// The move ledger.
    pub fn ledger(&self) -> &MoveLedger {
        &self.ledger
    }

    /// The search threads.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// `(snake index, action)` of `/<name>[/<action>]` (`Handler._route`).
    fn route<'a>(&self, path: &'a str) -> Option<(usize, &'a str)> {
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        if parts.is_empty() || parts.len() > 2 {
            return None;
        }
        let i = self.snakes.iter().position(|s| s.name == parts[0])?;
        Some((i, parts.get(1).copied().unwrap_or("")))
    }
}

/// A JSON response.
fn respond(code: StatusCode, body: impl Into<Bytes>) -> Response<Full<Bytes>> {
    let mut r = Response::new(Full::new(body.into()));
    *r.status_mut() = code;
    r.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

fn error(code: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    respond(code, json_dict(&[("error", msg)]))
}

fn not_found() -> Response<Full<Bytes>> {
    error(StatusCode::NOT_FOUND, "not found")
}

/// A header as Python's `http.server` reads it (latin-1).
fn header_text(v: &HeaderValue) -> String {
    v.as_bytes().iter().map(|&b| char::from(b)).collect()
}

/// Answers one request (`Handler.do_GET` / `do_POST`).
pub async fn handle(app: &App, req: hyper::Request<Incoming>) -> Response<Full<Bytes>> {
    let t_in = Instant::now();
    let path = req.uri().path().to_string();
    match *req.method() {
        Method::GET => {
            if path.is_empty() || path == "/" {
                return respond(StatusCode::OK, app.list_body.clone());
            }
            match app.route(&path) {
                Some((i, "")) => respond(StatusCode::OK, app.info_bodies[i].clone()),
                _ => not_found(),
            }
        }
        Method::POST => post(app, req, &path, t_in).await,
        _ => error(StatusCode::NOT_IMPLEMENTED, "unsupported method"),
    }
}

/// Reads and parses the body of a POST, or the 400 response.
async fn read_request(
    req: hyper::Request<Incoming>,
) -> Result<(Request, Option<String>), Box<Response<Full<Bytes>>>> {
    let bad = |msg: &str| Box::new(error(StatusCode::BAD_REQUEST, msg));
    // size = int(self.headers.get("Content-Length", 0)); 0 < size <= MAX_BODY
    let size = match req.headers().get(CONTENT_LENGTH) {
        None => 0,
        Some(v) => {
            let text = header_text(v);
            match python_int(&text) {
                Some(Num::Int(i)) => i,
                Some(_) => i128::MAX,
                None => {
                    return Err(bad(&format!(
                        "invalid literal for int() with base 10: {text:?}"
                    )));
                }
            }
        }
    };
    if !(1..=MAX_BODY as i128).contains(&size) {
        return Err(bad(&format!("bad Content-Length {size}")));
    }
    let ray = req.headers().get("cf-ray").map(header_text);
    let body = Limited::new(req.into_body(), MAX_BODY);
    let bytes = match tokio::time::timeout(BODY_TIMEOUT, body.collect()).await {
        Ok(Ok(c)) => c.to_bytes(),
        Ok(Err(e)) => return Err(bad(&format!("cannot read the body: {e}"))),
        Err(_) => return Err(bad("timed out reading the body")),
    };
    let text = String::from_utf8(bytes.to_vec()).map_err(|e| bad(&e.to_string()))?;
    let request = Request::from_json(&text).map_err(|e| bad(&e.to_string()))?;
    Ok((request, ray))
}

async fn post(
    app: &App,
    req: hyper::Request<Incoming>,
    path: &str,
    t_in: Instant,
) -> Response<Full<Bytes>> {
    let Some((i, action)) = app.route(path) else {
        return not_found();
    };
    if !matches!(action, "start" | "move" | "end") {
        return not_found();
    }
    let snake = Arc::clone(&app.snakes[i]);
    let (request, ray) = match read_request(req).await {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    let root = request.root();
    // game = req.get("game", {}).get("id", "?")
    let game = match root.get("game").and_then(|g| g.get("id")) {
        Some(id) => py_str(id).unwrap_or_default(),
        None => "?".to_string(),
    };
    let turn = py_str_opt(root.get("turn")).unwrap_or_default();
    match action {
        "move" => {
            let choice = play(app, &snake, request, t_in, &game, &turn, ray).await;
            respond(
                StatusCode::OK,
                format!("{{\"move\": {}}}", json_string(choice.mv.name())),
            )
        }
        "start" => {
            match snake.start(&request) {
                Ok(name) => match start_line(root) {
                    Ok(s) => info!(
                        "{} game {} start: {}, playing {}",
                        snake.name, game, s, name
                    ),
                    Err(e) => info!("{} game {}: bad /start request ({})", snake.name, game, e),
                },
                Err(e) => info!("{} game {}: bad /start request ({})", snake.name, game, e),
            }
            respond(StatusCode::OK, "{}")
        }
        _ => {
            if let Err(e) = app
                .ledger
                .record(&snake.name, root, None, "", 0.0, true, None, None)
            {
                info!("{} game {}: move ledger: {}", snake.name, game, e);
            }
            match snake.end(&request).and_then(|()| won(root)) {
                Ok(won) => info!(
                    "{} game {} end at turn {}: {}",
                    snake.name,
                    game,
                    turn,
                    if won { "won" } else { "lost or drew" }
                ),
                Err(e) => info!("{} game {}: bad /end request ({})", snake.name, game, e),
            }
            respond(StatusCode::OK, "{}")
        }
    }
}

/// `"{w}x{h}, {n} snakes, {ruleset} on map {map}"` of a `/start` log line.
fn start_line(root: &PyValue) -> Result<String, String> {
    let s = |v: &PyValue| py_str(v).map_err(|e| e.to_string());
    let b = root.item("board").map_err(|e| e.0)?;
    let (w, h) = (
        b.item("width").map_err(|e| e.0)?,
        b.item("height").map_err(|e| e.0)?,
    );
    let n = b.item("snakes").and_then(PyValue::len).map_err(|e| e.0)?;
    let game = root.item("game").map_err(|e| e.0)?;
    let empty = PyValue::Dict(Vec::new());
    let ruleset = game
        .dict_get("ruleset")
        .map_err(|e| e.0)?
        .unwrap_or(&empty)
        .dict_get("name")
        .map_err(|e| e.0)?;
    Ok(format!(
        "{}x{}, {} snakes, {} on map {}",
        s(w)?,
        s(h)?,
        n,
        py_str_opt(ruleset).map_err(|e| e.to_string())?,
        py_str_opt(game.get("map")).map_err(|e| e.to_string())?
    ))
}

/// Whether our snake is the only one left (`left == [req["you"]["id"]]`).
fn won(root: &PyValue) -> Result<bool, String> {
    let snakes = root
        .item("board")
        .and_then(|b| b.item("snakes"))
        .and_then(PyValue::subscripted_items)
        .map_err(|e| e.0)?;
    let mut ids = Vec::with_capacity(snakes.len());
    for s in snakes {
        ids.push(s.item("id").map_err(|e| e.0)?);
    }
    let you = root
        .item("you")
        .and_then(|y| y.item("id"))
        .map_err(|e| e.0)?;
    Ok(ids.len() == 1 && py_eq(ids[0], you).map_err(|e| e.0)?)
}

/// Answers a `/move`: the search on a pool thread, the safe move if it fails or is late; logs
/// the answer and records it in the move ledger.
async fn play(
    app: &App,
    snake: &Arc<Snake>,
    request: Request,
    t_in: Instant,
    game: &str,
    turn: &str,
    ray: Option<String>,
) -> Choice {
    let req = Arc::new(request);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let job = Job {
        snake: Arc::clone(snake),
        req: Arc::clone(&req),
        t_in,
        reply: tx,
    };
    let fallback = |why: String| Choice {
        mv: safe_move(&req),
        source: Source::SafeMove,
        ms: t_in.elapsed().as_secs_f64() * 1e3,
        simulations: 0,
        error: Some(why),
    };
    let choice = if app.pool.submit(job).is_err() {
        fallback("the search threads have stopped".into())
    } else {
        let answer = match app.answer_by {
            Some(d) => match tokio::time::timeout_at((t_in + d).into(), rx).await {
                Ok(r) => r.ok(),
                Err(_) => {
                    return finish(
                        app,
                        snake,
                        &req,
                        fallback(format!("no move after {} ms", d.as_millis())),
                        t_in,
                        game,
                        turn,
                        ray,
                    );
                }
            },
            None => rx.await.ok(),
        };
        answer.unwrap_or_else(|| fallback("the search failed".into()))
    };
    finish(app, snake, &req, choice, t_in, game, turn, ray)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    app: &App,
    snake: &Snake,
    req: &Request,
    choice: Choice,
    t_in: Instant,
    game: &str,
    turn: &str,
    ray: Option<String>,
) -> Choice {
    if let Some(e) = &choice.error {
        info!(
            "{} game {} turn {}: falling back to safe_move ({})",
            snake.name, game, turn, e
        );
    }
    let name = snake.source_name(choice.source);
    // A timed search also says how far it got (the faithful search always runs its count).
    let sims = if snake.has_budget() && choice.source == Source::Agent {
        format!(", {} simulations", choice.simulations)
    } else {
        String::new()
    };
    info!(
        "{} game {} turn {}: {} ({}, {:.1} ms{})",
        snake.name,
        game,
        turn,
        choice.mv.name(),
        name,
        choice.ms,
        sims
    );
    let wall = t_in.elapsed().as_secs_f64() * 1e3;
    // "<id>-<colo>" behind Cloudflare: ray.rpartition("-")[2] or None
    let colo = ray.as_deref().and_then(|r| r.rsplit('-').next());
    if let Err(e) = app.ledger.record(
        &snake.name,
        req.root(),
        Some(choice.mv),
        name,
        choice.ms,
        false,
        Some(wall),
        colo,
    ) {
        info!("{} game {}: move ledger: {}", snake.name, game, e);
    }
    choice
}
