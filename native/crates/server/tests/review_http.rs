//! Adversarial review of the server over real HTTP: slow and stalled clients, half-closed and
//! pipelined connections, odd request targets and methods, body-size edges, deeply nested
//! JSON, the info JSON's escaping, the `/start` cache under eviction and concurrency, the move
//! counter under concurrency, answering within the deadline under load, and the binary's
//! startup failures and shutdown.
//!
//! Needs the deployed run dir (`$NAZ_RUN`, default `/home/nathan/slinky-serve/az-modes-M-s0`);
//! without it each test prints SKIP and passes. Tests marked `#[ignore]` reproduce a bug named
//! in their comment.

use serde_json::{Value, json};
use slinky_api::{InfoResponse, Request};
use slinky_server_native::Background;
use slinky_server_native::engine::{MAX_GAMES, Snake, SnakeSpec};
use slinky_server_native::http::App;
use slinky_server_native::ledger::MoveLedger;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn run_dir() -> Option<String> {
    let dir = std::env::var("NAZ_RUN")
        .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".to_string());
    if PathBuf::from(&dir).join("params.npz").exists() {
        Some(dir)
    } else {
        eprintln!("SKIP: no run dir at {dir} (set NAZ_RUN)");
        None
    }
}

fn snake_with(spec: &str, info: InfoResponse) -> Snake {
    let spec = SnakeSpec::parse(spec).unwrap();
    Snake::load(&spec, info).unwrap()
}

fn snake(spec: &str) -> Snake {
    snake_with(
        spec,
        InfoResponse::new("test", "#123456", "default", "default", "v"),
    )
}

fn server_with(
    specs: &[String],
    log: Option<PathBuf>,
    threads: usize,
    answer_by: Option<Duration>,
) -> Background {
    slinky_server_native::log::set_quiet(true);
    let snakes = specs.iter().map(|s| snake(s)).collect();
    let ledger = Arc::new(MoveLedger::new(log));
    let app = App::new(snakes, ledger, threads, answer_by).unwrap();
    Background::start(app, "127.0.0.1:0").unwrap()
}

fn temp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("slinky-review-{}-{name}", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

/// Sends raw bytes, reads until the server closes (or `limit` passes); `(status, head, body)`.
fn raw(addr: SocketAddr, bytes: &[u8], limit: Duration) -> (u16, String, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(limit)).unwrap();
    s.write_all(bytes).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    split_response(&out)
}

fn split_response(out: &[u8]) -> (u16, String, String) {
    let text = String::from_utf8_lossy(out).to_string();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .map(|(h, b)| (h.to_string(), b.to_string()))
        .unwrap_or((text.clone(), String::new()));
    (status, head, body)
}

fn request_bytes(method: &str, path: &str, body: &[u8], extra: &str) -> Vec<u8> {
    let mut v = format!(
        "{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n{extra}\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    v.extend_from_slice(body);
    v
}

fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, String) {
    let (code, _, body) = raw(
        addr,
        &request_bytes("POST", path, body.as_bytes(), ""),
        Duration::from_secs(60),
    );
    (code, body)
}

fn get(addr: SocketAddr, path: &str) -> (u16, String) {
    let (code, _, body) = raw(
        addr,
        format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").as_bytes(),
        Duration::from_secs(60),
    );
    (code, body)
}

fn example(mode: &str) -> Value {
    let (_, body) = slinky_server_native::examples::requests()
        .into_iter()
        .find(|(m, _)| m == mode)
        .unwrap();
    serde_json::from_str(&body).unwrap()
}

const MOVES: [&str; 4] = ["up", "down", "left", "right"];

fn move_of(body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    v["move"].as_str().unwrap_or("").to_string()
}

fn log_lines(srv: &Background, path: &PathBuf) -> Vec<Value> {
    srv.app.ledger().flush();
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

// ---------------------------------------------------------------------------------------
// Slow and stalled clients
// ---------------------------------------------------------------------------------------

/// A connection that never sends a byte is closed by the header timeout (5 s), and so is one
/// that stalls in the middle of its headers. Neither holds the server.
#[test]
fn idle_and_half_header_connections_are_closed() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=8")], None, 2, None);
    let a = srv.addr;
    let probe = |bytes: &'static [u8]| {
        std::thread::spawn(move || {
            let t = Instant::now();
            let (code, _, _) = raw(a, bytes, Duration::from_secs(20));
            (code, t.elapsed())
        })
    };
    let idle = probe(b"");
    let half = probe(b"POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Le");
    // Meanwhile the server answers.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(get(a, "/").0, 200);
    for (what, h) in [("idle", idle), ("half headers", half)] {
        let (code, took) = h.join().unwrap();
        eprintln!("{what}: status {code} after {took:?}");
        assert!(
            took < Duration::from_secs(8),
            "{what}: held for {took:?} (status {code})"
        );
    }
}

/// Clients that send their headers and part of the body, then stall, get a 400 within the
/// 5 s body timeout; many of them at once do not delay other games' moves.
#[test]
fn stalled_bodies_do_not_delay_moves() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(
        &[format!("s={dir}:simulations=32")],
        None,
        2,
        Some(Duration::from_millis(400)),
    );
    let a = srv.addr;
    let mut stalled = Vec::new();
    for _ in 0..64 {
        let mut s = TcpStream::connect(a).unwrap();
        s.write_all(
            b"POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Length: 1000\r\n\r\n{\"game\": {",
        )
        .unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        stalled.push((s, Instant::now()));
    }
    let req = example("duel").to_string();
    for _ in 0..5 {
        let t = Instant::now();
        let (code, body) = post(a, "/s/move", &req);
        let took = t.elapsed();
        assert_eq!(code, 200);
        assert!(MOVES.contains(&move_of(&body).as_str()));
        assert!(took < Duration::from_millis(300), "a move took {took:?}");
    }
    for (mut s, t) in stalled {
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        let took = t.elapsed();
        let (code, _, _) = split_response(&out);
        assert_eq!(code, 400, "{}", String::from_utf8_lossy(&out));
        assert!(
            took > Duration::from_secs(4) && took < Duration::from_secs(8),
            "{took:?}"
        );
    }
}

/// Fixed (lib.rs `serve`: hyper's `http1::Builder::half_close` was left at its default,
/// false): a client that half-closes its connection after sending the whole request
/// (shutdown of its write side, as `nc -N`, `curl` through some proxies, or a load balancer
/// health check do) gets no answer at all: hyper sees the read EOF and drops the connection
/// while the handler runs. Python's `http.server` answers it.
#[test]
fn half_closed_client_gets_its_answer() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=64")], None, 2, None);
    let req = example("standard").to_string();
    let half_closed = |bytes: &[u8]| {
        let mut s = TcpStream::connect(srv.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        s.write_all(bytes).unwrap();
        s.shutdown(Shutdown::Write).unwrap();
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        split_response(&out)
    };
    let (get_code, _, _) = half_closed(b"GET / HTTP/1.0\r\n\r\n");
    let (move_code, _, body) = half_closed(&request_bytes("POST", "/s/move", req.as_bytes(), ""));
    eprintln!("half-closed: GET {get_code}, POST /move {move_code} {body}");
    assert_eq!((get_code, move_code), (200, 200));
    assert!(MOVES.contains(&move_of(&body).as_str()), "{body}");
}

/// Keep-alive requests (HTTP/1.1 without `Connection: close`) and pipelined requests get one
/// answer and a closed connection, never a hang; HTTP/1.0 works.
#[test]
fn keep_alive_and_pipelining() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=16")], None, 2, None);
    let a = srv.addr;
    let req = example("duel").to_string();
    let one = format!(
        "POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{req}",
        req.len()
    );
    let t = Instant::now();
    let (code, head, body) = raw(a, one.as_bytes(), Duration::from_secs(10));
    assert_eq!(code, 200);
    assert!(
        head.to_ascii_lowercase().contains("connection: close"),
        "{head}"
    );
    assert!(MOVES.contains(&move_of(&body).as_str()));
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    // Two pipelined requests: the first is answered, the connection closes.
    let two = format!("{one}{one}");
    let t = Instant::now();
    let (code, _, body) = raw(a, two.as_bytes(), Duration::from_secs(10));
    assert_eq!(code, 200);
    assert_eq!(body.matches("\"move\"").count(), 1, "{body}");
    assert!(t.elapsed() < Duration::from_secs(2));
    // HTTP/1.0.
    let ten = format!(
        "POST /s/move HTTP/1.0\r\nContent-Length: {}\r\n\r\n{req}",
        req.len()
    );
    let (code, _, body) = raw(a, ten.as_bytes(), Duration::from_secs(10));
    assert_eq!(code, 200);
    assert!(MOVES.contains(&move_of(&body).as_str()));
}

// ---------------------------------------------------------------------------------------
// Routes, methods, bodies
// ---------------------------------------------------------------------------------------

#[test]
fn odd_targets_and_methods() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=8")], None, 2, None);
    let a = srv.addr;
    let req = example("duel").to_string();
    // Python's _route: path.split("?")[0].split("/") without empty parts.
    for (path, code) in [
        ("/s/move/", 200),
        ("/s//move", 200),
        ("//s/move", 200),
        ("/s/move?x=1&y=/a/b", 200),
        ("/S/move", 404),
        ("/s/MOVE", 404),
        ("/s/%6Dove", 404),
        ("/s/move/x", 404),
        ("/", 404),
        ("/s", 404),
        ("/s/", 404),
        ("/s/info", 404),
    ] {
        assert_eq!(post(a, path, &req).0, code, "POST {path}");
    }
    for (path, code) in [("/?x", 200), ("/s?", 200), ("/s/x", 404), ("/x", 404)] {
        assert_eq!(get(a, path).0, code, "GET {path}");
    }
    // A long path is an ordinary 404 (or a 414/431), not a hang or a crash.
    let long = format!("/{}", "a/".repeat(20_000));
    let (code, _, _) = raw(
        a,
        format!("GET {long} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").as_bytes(),
        Duration::from_secs(10),
    );
    assert!([404, 414, 431, 400].contains(&code), "{code}");
    // Methods http.server has no do_X for: 501.
    for m in ["PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"] {
        let (code, _, _) = raw(
            a,
            format!("{m} /s HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").as_bytes(),
            Duration::from_secs(10),
        );
        assert_eq!(code, 501, "{m}");
    }
    // Garbage instead of HTTP: a 400 or a closed connection, never a hang.
    let t = Instant::now();
    let (code, _, _) = raw(
        a,
        b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03garbage\r\n\r\n",
        Duration::from_secs(10),
    );
    assert!(code == 400 || code == 0, "{code}");
    assert!(t.elapsed() < Duration::from_secs(6));
    assert_eq!(get(a, "/").0, 200);
}

/// The absolute-form request target (`POST http://host/s/move`): Python's `self.path` is the
/// whole URL, so `_route` sees `["http:", "host", "s", "move"]` and answers 404; hyper's
/// `uri().path()` is `/s/move` and the native server plays the move. Harmless for the ladder
/// (no proxy sends absolute-form), recorded as a parity difference. (Fixed: the handler routes
/// on the whole target, as Python does.)
#[test]
fn absolute_form_target_is_404_as_in_python() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=8")], None, 2, None);
    let req = example("duel").to_string();
    assert_eq!(post(srv.addr, "http://example.com/s/move", &req).0, 404);
}

#[test]
fn body_size_edges() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(&[format!("s={dir}:simulations=8")], None, 2, None);
    let a = srv.addr;
    let req = example("duel").to_string();
    // Exactly 64 KiB (the request padded with spaces) is accepted; one byte more is not.
    let mut exact = req.clone();
    exact.push_str(&" ".repeat((1 << 16) - req.len()));
    assert_eq!(exact.len(), 1 << 16);
    let (code, body) = post(a, "/s/move", &exact);
    assert_eq!(code, 200, "{body}");
    assert!(MOVES.contains(&move_of(&body).as_str()));
    exact.push(' ');
    assert_eq!(post(a, "/s/move", &exact).0, 400);
    // A huge Content-Length with a small body: refused at once, not after a 5 s wait.
    let t = Instant::now();
    let (code, _, _) = raw(
        a,
        b"POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Length: 99999999999999999999\r\n\r\n{}",
        Duration::from_secs(10),
    );
    assert!(code == 400 || code == 0, "{code}");
    assert!(t.elapsed() < Duration::from_secs(2));
    // Content-Length with leading zeros (int("000123") == 123).
    let padded = format!(
        "POST /s/move HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Length: 000{}\r\n\r\n{req}",
        req.len()
    );
    let (code, _, body) = raw(a, padded.as_bytes(), Duration::from_secs(10));
    assert_eq!(code, 200, "{body}");
    // Two different Content-Lengths: refused (request smuggling), never a hang.
    let t = Instant::now();
    let two = format!(
        "POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\nContent-Length: 5\r\n\r\n{req}",
        req.len()
    );
    let (code, _, _) = raw(a, two.as_bytes(), Duration::from_secs(10));
    assert!(code == 400 || code == 0, "{code}");
    assert!(t.elapsed() < Duration::from_secs(6));
    // A body shorter than its Content-Length, then EOF: 400, not a hang until the timeout.
    let mut s = TcpStream::connect(a).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(b"POST /s/move HTTP/1.1\r\nHost: t\r\nContent-Length: 500\r\n\r\n{\"a\": 1}")
        .unwrap();
    s.shutdown(Shutdown::Write).unwrap();
    let t = Instant::now();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    let (code, _, _) = split_response(&out);
    assert!(code == 400 || code == 0, "{code}");
    assert!(t.elapsed() < Duration::from_secs(6), "{:?}", t.elapsed());
    assert_eq!(get(a, "/").0, 200);
}

/// JSON nested as deep as `json.loads` accepts (9,997 containers) in every field the
/// handler, the `/start` cache, the move log and the safe move read: answered, no crash.
#[test]
fn deeply_nested_json_does_not_crash() {
    let Some(dir) = run_dir() else { return };
    let path = temp("deep.jsonl");
    let srv = server_with(
        &[format!("s={dir}:simulations=8")],
        Some(path.clone()),
        2,
        None,
    );
    let a = srv.addr;
    let deep = |n: usize| format!("{}1{}", "[".repeat(n), "]".repeat(n));
    let deep_obj = |n: usize| format!("{}1{}", "{\"a\": ".repeat(n), "}".repeat(n));
    let base = example("duel").to_string();
    // Splice a nested value in for a field of the example request.
    let with = |field: &str, value: &str| -> String {
        let mut v: Value = serde_json::from_str(&base).unwrap();
        let ptr = format!("/{}", field.replace('.', "/"));
        *v.pointer_mut(&ptr).unwrap() = json!("@@DEEP@@");
        v.to_string().replace("\"@@DEEP@@\"", value)
    };
    for n in [950, 9_000] {
        for field in [
            "game.id",
            "game.map",
            "game.ruleset.name",
            "game.timeout",
            "turn",
            "board.width",
            "board.snakes",
            "board.food",
            "you.id",
            "you.head",
            "you.head.x",
        ] {
            for value in [deep(n), deep_obj(n)] {
                let body = with(field, &value);
                assert!(body.len() <= 1 << 16);
                for action in ["start", "move", "end"] {
                    let (code, answer) = post(a, &format!("/s/{action}"), &body);
                    assert!(
                        code == 200 || code == 400,
                        "{action} {field} depth {n}: {code} {answer}"
                    );
                    if action == "move" {
                        assert!(MOVES.contains(&move_of(&answer).as_str()), "{answer}");
                    }
                }
            }
        }
    }
    // Deeper than json.loads accepts: 400.
    let too_deep = with("game.map", &deep(10_000));
    assert_eq!(post(a, "/s/move", &too_deep).0, 400);
    assert_eq!(get(a, "/").0, 200);
    let lines = log_lines(&srv, &path);
    assert!(!lines.is_empty());
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------------------
// Info JSON
// ---------------------------------------------------------------------------------------

/// The info and list bodies are `json.dumps` of the flags: quotes, backslashes, control
/// characters, non-ASCII and astral characters escaped as Python does (`ensure_ascii`).
#[test]
fn info_json_is_python_json_dumps() {
    let Some(dir) = run_dir() else { return };
    slinky_server_native::log::set_quiet(true);
    let info = InfoResponse::new(
        "nathan \"W\" \\o/",
        "#ef4444\u{7f}",
        "cosmic-horror\té",
        "😀\u{1}",
        "slinky-native az_modes:/r:simulations=256",
    );
    let snakes = vec![
        snake_with(&format!("az-256={dir}:simulations=8"), info),
        snake(&format!("a.b_c-9={dir}:simulations=8")),
    ];
    let app = App::new(snakes, Arc::new(MoveLedger::new(None)), 1, None).unwrap();
    let srv = Background::start(app, "127.0.0.1:0").unwrap();
    let a = srv.addr;
    // python3 -c 'import json; print(json.dumps({...}))'
    let want = r##"{"apiversion": "1", "author": "nathan \"W\" \\o/", "color": "#ef4444\u007f", "head": "cosmic-horror\t\u00e9", "tail": "\ud83d\ude00\u0001", "version": "slinky-native az_modes:/r:simulations=256"}"##;
    assert_eq!(get(a, "/az-256"), (200, want.to_string()));
    assert_eq!(
        get(a, "/"),
        (200, r#"{"snakes": ["/az-256", "/a.b_c-9"]}"#.to_string())
    );
    let (_, head, _) = raw(
        a,
        b"GET /az-256 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n",
        Duration::from_secs(10),
    );
    let head = head.to_ascii_lowercase();
    assert!(head.contains("content-type: application/json"), "{head}");
    assert!(
        head.contains(&format!("content-length: {}", want.len())),
        "{head}"
    );
}

// ---------------------------------------------------------------------------------------
// The /start cache and the move counter
// ---------------------------------------------------------------------------------------

fn start_request(id: &Value, snakes: usize) -> Request {
    let mut r = example("standard");
    r["game"]["id"] = id.clone();
    r["board"]["snakes"]
        .as_array_mut()
        .unwrap()
        .truncate(snakes);
    Request::from_value(&r).unwrap()
}

/// `Mover.start`: when 1,000 games are remembered, the oldest is deleted before the new one
/// is stored, even when the new one is already there (it then moves to the end).
#[test]
fn start_cache_evicts_like_python() {
    let Some(dir) = run_dir() else { return };
    slinky_server_native::log::set_quiet(true);
    let s = snake(&format!("s={dir}:simulations=8"));
    let id = |i: usize| json!(format!("g{i}"));
    for i in 0..MAX_GAMES {
        s.start(&start_request(&id(i), 2 + i % 3)).unwrap();
    }
    let has = |v: &Value| s.starting_snakes(&start_request(v, 4));
    assert_eq!(has(&id(0)), Some(2));
    assert_eq!(has(&id(998)), Some(4)); // 2 + 998 % 3
    // Full: re-starting g0 deletes the oldest (g0 itself) and stores it again, last.
    s.start(&start_request(&id(0), 4)).unwrap();
    assert_eq!(has(&id(0)), Some(4));
    assert_eq!(has(&id(1)), Some(3));
    // The next new game evicts g1 (now the oldest), not g0.
    s.start(&start_request(&json!("new"), 3)).unwrap();
    assert_eq!(has(&id(1)), None);
    assert_eq!(has(&id(0)), Some(4));
    assert_eq!(has(&json!("new")), Some(3));
    // 1 == 1.0 == true are one key in Python; "1" is another.
    s.start(&start_request(&json!(1), 4)).unwrap();
    assert_eq!(has(&json!(1.0)), Some(4));
    assert_eq!(has(&json!(true)), Some(4));
    assert_eq!(has(&json!("1")), None);
    // /end forgets by the same key; an unhashable id is an error (Python raises).
    s.end(&start_request(&json!(true), 4)).unwrap();
    assert_eq!(has(&json!(1)), None);
    assert!(s.end(&start_request(&json!([1]), 4)).is_err());
    assert!(s.start(&start_request(&json!({"a": 1}), 4)).is_err());
    // A null id is not remembered.
    s.start(&start_request(&Value::Null, 4)).unwrap();
    assert_eq!(has(&Value::Null), None);
}

/// Fixed (engine.rs gives each NaN id its own never-matching key). Was BUG (api `Key::NaN`
/// derives `Eq`; engine.rs keys the `/start` cache with it): a NaN game id
/// (`json.loads("NaN")`) is a new float object on every request, so Python's
/// `self._games.get(nan)` never finds it and every NaN `/start` adds an entry. Here all NaN
/// ids are one key: a `/move` with a NaN id gets the snake count stored at a NaN `/start`
/// (a 4-seat game where Python plays the living snakes only).
#[test]
fn start_cache_never_finds_a_nan_id() {
    let Some(dir) = run_dir() else { return };
    slinky_server_native::log::set_quiet(true);
    let s = snake(&format!("s={dir}:simulations=8"));
    let mut r = example("standard");
    r["game"]["id"] = json!("@@NAN@@");
    let text = r.to_string().replace("\"@@NAN@@\"", "NaN");
    let req = Request::from_json(&text).unwrap();
    s.start(&req).unwrap();
    assert_eq!(s.starting_snakes(&req), None);
}

/// Many games starting, moving and ending at once from many connections: every move counted
/// once (`moves()`), and the cache holds exactly `MAX_GAMES` of the games started.
#[test]
fn start_cache_and_counter_under_concurrency() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(
        &[format!("s={dir}:simulations=4")],
        None,
        4,
        Some(Duration::from_secs(30)),
    );
    let a = srv.addr;
    let threads = 12;
    let per = 110; // 1,320 games: more than MAX_GAMES
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                let mut moves = 0u64;
                for i in 0..per {
                    let mut r = example("duel");
                    r["game"]["id"] = json!(format!("t{t}-{i}"));
                    let body = r.to_string();
                    assert_eq!(post(a, "/s/start", &body).0, 200);
                    if i % 10 == 0 {
                        let (code, ans) = post(a, "/s/move", &body);
                        assert_eq!(code, 200);
                        assert!(MOVES.contains(&move_of(&ans).as_str()));
                        moves += 1;
                    }
                }
                moves
            })
        })
        .collect();
    let moves: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let s = &srv.app.snakes()[0];
    assert_eq!(s.moves(), moves);
    let mut present = 0;
    for t in 0..threads {
        for i in 0..per {
            let mut r = example("duel");
            r["game"]["id"] = json!(format!("t{t}-{i}"));
            if s.starting_snakes(&Request::from_value(&r).unwrap())
                .is_some()
            {
                present += 1;
            }
        }
    }
    assert_eq!(present, MAX_GAMES);
}

// ---------------------------------------------------------------------------------------
// Deadlines
// ---------------------------------------------------------------------------------------

/// Far more concurrent games than search threads, with a search slower than the deadline's
/// share: every `/move` is answered by `answer_by` (plus a small margin) at the client, the
/// late ones with the logged safe move, and the server is healthy afterwards.
#[test]
fn answers_within_the_deadline_under_load() {
    let Some(dir) = run_dir() else { return };
    let path = temp("load.jsonl");
    let answer_by = Duration::from_millis(250);
    let srv = server_with(
        &[format!("s={dir}:simulations=2048")],
        Some(path.clone()),
        2,
        Some(answer_by),
    );
    let a = srv.addr;
    let reqs: Vec<String> = ["standard", "duel", "royale", "constrictor"]
        .iter()
        .map(|m| example(m).to_string())
        .collect();
    for round in 0..2 {
        let handles: Vec<_> = (0..48)
            .map(|i| {
                let body = reqs[i % 4].clone();
                std::thread::spawn(move || {
                    let t = Instant::now();
                    let (code, ans) = post(a, "/s/move", &body);
                    (code, ans, t.elapsed())
                })
            })
            .collect();
        let mut worst = Duration::ZERO;
        for h in handles {
            let (code, ans, took) = h.join().unwrap();
            assert_eq!(code, 200);
            assert!(MOVES.contains(&move_of(&ans).as_str()));
            worst = worst.max(took);
        }
        eprintln!("round {round}: slowest answer {worst:?}");
        assert!(worst < answer_by + Duration::from_millis(150), "{worst:?}");
    }
    // Let the running searches finish; then a move is the agent's again.
    std::thread::sleep(Duration::from_millis(1500));
    let before = log_lines(&srv, &path).len();
    let (code, _) = post(a, "/s/move", &reqs[1]);
    assert_eq!(code, 200);
    let lines = log_lines(&srv, &path);
    assert_eq!(lines.len(), before + 1);
    let late = lines[..before]
        .iter()
        .filter(|l| l["agent"] == json!("safe_move"))
        .count();
    eprintln!("{late} of {before} answers under load were the safe move");
    assert!(late > 0);
    assert_ne!(
        lines[before]["agent"],
        json!("safe_move"),
        "{:?}",
        lines[before]
    );
    let _ = std::fs::remove_file(&path);
}

/// Fixed (main.rs / engine.rs: there was no check between `budget_ms` and `--answer-by-ms`):
/// a time budget past the answer-by deadline made every move the safe move. Now a time-budget
/// search stops 20 ms before `answer_by`. The search runs until
/// `t_in + budget` and the handler gives up at `t_in + answer_by` first, so a snake started
/// as `--snake x=RUN:budget_ms=450` (inside Battlesnake's 500 ms) with the default
/// `--answer-by-ms 400`, or any `budget_ms` from about 400 up to the accepted 10,000, never
/// plays an agent move, silently (each move logged as `safe_move`, `no move after 400 ms`).
#[test]
fn budget_at_the_deadline_still_plays_the_agent() {
    let Some(dir) = run_dir() else { return };
    let path = temp("budget.jsonl");
    let srv = server_with(
        &[format!("b={dir}:budget_ms=450")],
        Some(path.clone()),
        2,
        Some(Duration::from_millis(400)),
    );
    for _ in 0..3 {
        let (code, _) = post(srv.addr, "/b/move", &example("duel").to_string());
        assert_eq!(code, 200);
    }
    let lines = log_lines(&srv, &path);
    let _ = std::fs::remove_file(&path);
    let safe = lines
        .iter()
        .filter(|l| l["agent"] == json!("safe_move"))
        .count();
    assert_eq!(safe, 0, "{} of {} moves were safe_move", safe, lines.len());
}

/// Every search stops at the request's deadline, `game.timeout` less the margin (100 ms) less
/// 20 ms, and plays the agent's best move so far: a 2^20-simulation snake answers a 200 ms game
/// after about 80 ms and a game without a timeout after about 380 ms, with the agent's move.
#[test]
fn game_timeout_bounds_every_search() {
    let Some(dir) = run_dir() else { return };
    let path = temp("timeout.jsonl");
    let srv = server_with(
        &[format!("big={dir}:simulations=1048576")],
        Some(path.clone()),
        2,
        Some(Duration::from_millis(400)),
    );
    let mut req = example("duel");
    for (timeout, want_ms) in [(json!(200), 80u64), (json!(null), 380), (json!(150), 30)] {
        req["game"]["timeout"] = timeout.clone();
        let t = Instant::now();
        let (code, body) = post(srv.addr, "/big/move", &req.to_string());
        let took = t.elapsed().as_millis() as u64;
        assert_eq!(code, 200);
        assert!(MOVES.contains(&move_of(&body).as_str()), "{body}");
        assert!(
            took >= want_ms && took < want_ms + 40,
            "timeout {timeout}: answered after {took} ms"
        );
    }
    let lines = log_lines(&srv, &path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(lines.len(), 3);
    assert!(
        lines.iter().all(|l| l["agent"] != json!("safe_move")),
        "{lines:?}"
    );
}

/// A huge `--answer-by-ms` (the parser accepts any u64) must not overflow the deadline
/// (`t_in + d`) and panic in the handler; `u64::MAX` ms still fits a Linux `Instant`.
#[test]
fn huge_answer_by_does_not_panic() {
    let Some(dir) = run_dir() else { return };
    let srv = server_with(
        &[format!("s={dir}:simulations=8")],
        None,
        1,
        Some(Duration::from_millis(u64::MAX)),
    );
    let (code, body) = post(srv.addr, "/s/move", &example("duel").to_string());
    assert_eq!(code, 200, "{body}");
}

// ---------------------------------------------------------------------------------------
// The binary: startup failures, a clean start and a clean stop
// ---------------------------------------------------------------------------------------

const BIN: &str = env!("CARGO_BIN_EXE_slinky-native-server");

fn run_bin(args: &[&str]) -> (Option<i32>, String, Duration) {
    let t = Instant::now();
    let out = Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code(), text, t.elapsed())
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn startup_failures_exit_with_a_message() {
    let Some(dir) = run_dir() else { return };
    let scratch = temp("bad-runs");
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let make = |name: &str, config: Option<&[u8]>, params: Option<&[u8]>| -> String {
        let d = scratch.join(name);
        std::fs::create_dir_all(&d).unwrap();
        if let Some(c) = config {
            std::fs::write(d.join("config.json"), c).unwrap();
        }
        if let Some(p) = params {
            std::fs::write(d.join("params.npz"), p).unwrap();
        }
        d.to_string_lossy().to_string()
    };
    let config = std::fs::read(format!("{dir}/config.json")).unwrap();
    let params = std::fs::read(format!("{dir}/params.npz")).unwrap();
    let missing = scratch.join("nope").to_string_lossy().to_string();
    let empty = make("empty", None, None);
    let bad_config = make("bad-config", Some(b"{not json"), Some(&params));
    let wrong_config = make(
        "wrong-config",
        Some(b"{\"algorithm\": \"ppo\"}"),
        Some(&params),
    );
    let truncated = make(
        "truncated",
        Some(&config),
        Some(&params[..params.len() / 2]),
    );
    let junk_params = make("junk-params", Some(&config), Some(b"PK\x03\x04garbage"));
    let file_dir = scratch.join("a-file");
    std::fs::write(&file_dir, b"x").unwrap();
    let file_dir = file_dir.to_string_lossy().to_string();
    for run in [
        &missing,
        &empty,
        &bad_config,
        &wrong_config,
        &truncated,
        &junk_params,
        &file_dir,
    ] {
        let port = free_port().to_string();
        let spec = format!("s={run}");
        let (code, text, took) = run_bin(&["--snake", &spec, "--port", &port, "--no-benchmark"]);
        eprintln!("{run}: exit {code:?} in {took:?}: {}", text.trim());
        assert_eq!(code, Some(1), "{run}: {text}");
        assert!(text.contains(run.as_str()), "{run}: {text}");
        assert!(took < Duration::from_secs(10));
    }
    // Usage errors: exit 2.
    for args in [
        &[][..],
        &["--snake"],
        &["--snake", "a=/r:simulations=0"],
        &["--snake", "a b=/r"],
        &["--port", "70000", "--snake", "a=/r"],
        &["--threads", "-1", "--snake", "a=/r"],
    ] {
        let (code, text, _) = run_bin(args);
        assert_eq!(code, Some(2), "{args:?}: {text}");
    }
    // The port is taken: exit 1 with a message, after loading.
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    let spec = format!("s={dir}:simulations=8");
    let (code, text, _) = run_bin(&["--snake", &spec, "--port", &port, "--no-benchmark"]);
    assert_eq!(code, Some(1), "{text}");
    assert!(text.contains("cannot listen"), "{text}");
    let _ = std::fs::remove_dir_all(&scratch);
}

/// The binary starts (benchmark included), serves, writes the move log, and on SIGTERM
/// flushes the log and exits 0. A move log in a directory that does not exist does not stop
/// the server from answering.
#[test]
fn binary_serves_and_stops_cleanly() {
    let Some(dir) = run_dir() else { return };
    let log = temp("bin.jsonl");
    for (move_log, writable) in [
        (log.to_string_lossy().to_string(), true),
        ("/nonexistent-dir/moves.jsonl".to_string(), false),
    ] {
        let port = free_port();
        let spec = format!("az-256={dir}:simulations=32");
        let mut child = Command::new(BIN)
            .args([
                "--snake",
                &spec,
                "--port",
                &port.to_string(),
                "--move-log",
                &move_log,
                "--threads",
                "2",
                "--head",
                "cosmic-horror",
                "--tail",
                "cosmic-horror",
                "--color",
                "#ef4444",
                "--author",
                "nathanWolo",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = tx.send(line);
            }
        });
        let t = Instant::now();
        let mut lines = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(l) => {
                    let serving = l.contains("serving ");
                    lines.push(l);
                    if serving {
                        break;
                    }
                }
                Err(_) => panic!("no serving line: {lines:?}"),
            }
        }
        eprintln!("ready in {:?}", t.elapsed());
        // The benchmark lines: four modes.
        assert_eq!(
            lines.iter().filter(|l| l.contains("plays 11x11")).count(),
            4,
            "{lines:?}"
        );
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let (code, body) = get(addr, "/az-256");
        assert_eq!(code, 200);
        let info: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(info["head"], json!("cosmic-horror"));
        assert_eq!(info["color"], json!("#ef4444"));
        assert_eq!(info["author"], json!("nathanWolo"));
        let mut r = example("duel");
        r["game"]["id"] = json!("bin");
        let (code, ans) = post(addr, "/az-256/move", &r.to_string());
        assert_eq!(code, 200);
        assert!(MOVES.contains(&move_of(&ans).as_str()));
        let pid = child.id().to_string();
        assert!(
            Command::new("kill")
                .args(["-TERM", &pid])
                .status()
                .unwrap()
                .success()
        );
        let t = Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break s;
            }
            assert!(
                t.elapsed() < Duration::from_secs(10),
                "no exit after SIGTERM"
            );
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(status.code(), Some(0));
        if writable {
            let text = std::fs::read_to_string(&log).unwrap();
            let line: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
            assert_eq!(line["game"], json!("bin"));
            assert_eq!(line["move"], json!(move_of(&ans)));
            let _ = std::fs::remove_file(&log);
        }
    }
}
