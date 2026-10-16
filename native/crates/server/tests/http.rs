//! The server over real HTTP: routes, status codes and bodies (test_server.py's
//! `test_http_endpoints`, `test_routes_are_per_snake`, `test_concurrent_moves`,
//! `test_http_moves_are_logged`), the moves against the agent with the server's seeds, the
//! `/start` snake count, time budgets and the answer-by deadline.
//!
//! Needs the deployed run dir (`$NAZ_RUN`, default `/home/nathan/slinky-serve/az-modes-M-s0`);
//! without it each test prints SKIP and passes.

use serde_json::{Value, json};
use slinky_api::{InfoResponse, Request, game_from_request};
use slinky_search::{Budget, JaxKey, Search};
use slinky_server_native::Background;
use slinky_server_native::engine::{Snake, SnakeSpec};
use slinky_server_native::http::App;
use slinky_server_native::ledger::MoveLedger;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
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

fn snake(spec: &str) -> Snake {
    let spec = SnakeSpec::parse(spec).unwrap();
    let info = InfoResponse::new("test", "#123456", "default", "default", "v");
    Snake::load(&spec, info).unwrap()
}

fn server(specs: &[String], log: Option<PathBuf>, answer_by: Option<Duration>) -> Background {
    slinky_server_native::log::set_quiet(true);
    let snakes = specs.iter().map(|s| snake(s)).collect();
    let ledger = Arc::new(MoveLedger::new(log));
    let app = App::new(snakes, ledger, 4, answer_by).unwrap();
    Background::start(app, "127.0.0.1:0").unwrap()
}

/// One HTTP/1.1 request; `(status, body)`.
fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    headers: &[&str],
) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n");
    for h in headers {
        head.push_str(h);
        head.push_str("\r\n");
    }
    if let Some(b) = body {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        ));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes()).unwrap();
    if let Some(b) = body {
        s.write_all(b).unwrap();
    }
    let mut out = Vec::new();
    s.read_to_end(&mut out).unwrap();
    let text = String::from_utf8_lossy(&out).to_string();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, String) {
    http(addr, "POST", path, Some(body.as_bytes()), &[])
}

fn examples() -> Vec<(String, Value)> {
    slinky_server_native::examples::requests()
        .into_iter()
        .map(|(m, b)| (m, serde_json::from_str(&b).unwrap()))
        .collect()
}

fn example(mode: &str) -> Value {
    examples().into_iter().find(|(m, _)| m == mode).unwrap().1
}

const MOVES: [&str; 4] = ["up", "down", "left", "right"];

fn move_of(body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap();
    v["move"].as_str().unwrap().to_string()
}

#[test]
fn http_endpoints() {
    let Some(dir) = run_dir() else { return };
    let srv = server(&[format!("s={dir}:simulations=32")], None, None);
    let a = srv.addr;
    assert_eq!(
        http(a, "GET", "/", None, &[]),
        (200, r#"{"snakes": ["/s"]}"#.to_string())
    );
    let info = r##"{"apiversion": "1", "author": "test", "color": "#123456", "head": "default", "tail": "default", "version": "v"}"##;
    for path in ["/s", "/s/", "//s", "/s?x=1"] {
        assert_eq!(
            http(a, "GET", path, None, &[]),
            (200, info.to_string()),
            "{path}"
        );
    }
    let req = example("duel").to_string();
    assert_eq!(post(a, "/s/start", &req), (200, "{}".to_string()));
    let (code, body) = post(a, "/s/move", &req);
    assert_eq!(code, 200);
    assert!(MOVES.contains(&move_of(&body).as_str()));
    assert!(body.starts_with("{\"move\": \""), "{body}"); // json.dumps spacing
    assert_eq!(post(a, "/s/end", &req), (200, "{}".to_string()));
    assert_eq!(post(a, "/s/move", "{not json").0, 400);
    assert_eq!(post(a, "/s/nope", &req).0, 404);
    assert_eq!(post(a, "/s/move", "[1, 2]").0, 400);
    assert_eq!(post(a, "/s/move", r#"{"game": 3}"#).0, 400);
    assert_eq!(
        post(a, "/s/move", "\u{FEFF}{\"board\": 3}"),
        (200, r#"{"move": "up"}"#.into())
    );
    // Junk that parses: still a move, never an error or a hang.
    assert_eq!(
        post(a, "/s/move", r#"{"board": 3}"#),
        (200, r#"{"move": "up"}"#.to_string())
    );
    let mut huge = example("duel");
    huge["board"]["width"] = json!(99);
    huge["board"]["height"] = json!(99);
    let (code, body) = post(a, "/s/move", &huge.to_string());
    assert!(code == 200 && MOVES.contains(&move_of(&body).as_str()));
    // Body limits: no Content-Length (chunked), an empty body, more than 64 KiB.
    let chunked = format!("{:x}\r\n{req}\r\n0\r\n\r\n", req.len());
    let mut s = TcpStream::connect(a).unwrap();
    write!(s, "POST /s/move HTTP/1.1\r\nHost: t\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n{chunked}").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    assert_eq!(http(a, "POST", "/s/move", Some(b""), &[]).0, 400);
    let big = format!(r#"{{"x": "{}"}}"#, "a".repeat(1 << 16));
    assert_eq!(post(a, "/s/move", &big).0, 400);
    assert_eq!(http(a, "PUT", "/s/move", Some(b"{}"), &[]).0, 501);
    // Not UTF-8.
    assert_eq!(
        http(a, "POST", "/s/move", Some(b"{\"a\": \"\xff\"}"), &[]).0,
        400
    );
}

#[test]
fn routes_are_per_snake() {
    let Some(dir) = run_dir() else { return };
    let srv = server(
        &[
            format!("s={dir}:simulations=8"),
            format!("t={dir}:simulations=8"),
        ],
        None,
        None,
    );
    let a = srv.addr;
    assert_eq!(
        http(a, "GET", "/", None, &[]).1,
        r#"{"snakes": ["/s", "/t"]}"#
    );
    let req = example("duel").to_string();
    assert_eq!(post(a, "/move", &req).0, 404); // no snake at the root
    assert_eq!(post(a, "/dqn/move", &req).0, 404); // not a snake on this server
    assert_eq!(post(a, "/s/move/extra", &req).0, 404);
    assert_eq!(post(a, "/t/move", &req).0, 200);
    for path in ["/dqn", "/s/move", "/s/move/x"] {
        // GET only serves a snake's info.
        assert_eq!(http(a, "GET", path, None, &[]).0, 404, "{path}");
    }
}

#[test]
fn concurrent_moves() {
    let Some(dir) = run_dir() else { return };
    let srv = server(&[format!("s={dir}:simulations=64")], None, None);
    let a = srv.addr;
    let reqs: Vec<String> = examples().into_iter().map(|(_, r)| r.to_string()).collect();
    let t = Instant::now();
    let handles: Vec<_> = (0..32)
        .map(|i| {
            let body = reqs[i % reqs.len()].clone();
            std::thread::spawn(move || post(a, "/s/move", &body))
        })
        .collect();
    for h in handles {
        let (code, body) = h.join().unwrap();
        assert_eq!(code, 200);
        assert!(MOVES.contains(&move_of(&body).as_str()));
    }
    eprintln!("32 concurrent moves in {:?}", t.elapsed());
}

/// The server's move `i` (from 1) is the agent's search with `fold_in(key(0), i)`, using the
/// snake count remembered at `/start`.
#[test]
fn moves_are_the_agents_with_the_servers_seeds() {
    let Some(dir) = run_dir() else { return };
    let spec = format!("s={dir}:simulations=64");
    let srv = server(std::slice::from_ref(&spec), None, None);
    let agent = snake(&spec);
    let a = srv.addr;
    let mut ws = Search::new();
    let mut expect = |req: &Value, starting: Option<usize>, index: u32| -> String {
        let r = Request::from_json(&req.to_string()).unwrap();
        let game = game_from_request(&r, starting).unwrap();
        let mut noise = JaxKey::new(0).fold_in(index).search_noise();
        let m = agent
            .agent()
            .search_game(&mut ws, &game, Budget::simulations(64), &mut noise)
            .unwrap();
        m.mv.name().to_string()
    };
    let mut index = 0;
    for (_, req) in examples() {
        for _ in 0..3 {
            index += 1;
            let got = move_of(&post(a, "/s/move", &req.to_string()).1);
            assert_eq!(got, expect(&req, None, index), "move {index}");
        }
    }
    // A request the agent cannot read still counts when Python's would (an unknown ruleset)...
    let mut wrapped = example("duel");
    wrapped["game"]["ruleset"]["name"] = json!("wrapped");
    post(a, "/s/move", &wrapped.to_string());
    index += 1;
    // ... but one that fails check_sizes does not.
    let mut junk = example("duel");
    junk["board"]["food"] = json!(vec![json!({"x": 0, "y": 0}); 122]);
    post(a, "/s/move", &junk.to_string());
    // A four-player game that lost a snake keeps its four seats after /start.
    let mut four = example("standard");
    four["game"]["id"] = json!("four");
    assert_eq!(post(a, "/s/start", &four.to_string()).0, 200);
    four["board"]["snakes"].as_array_mut().unwrap().pop();
    index += 1;
    let got = move_of(&post(a, "/s/move", &four.to_string()).1);
    assert_eq!(got, expect(&four, Some(4), index));
    assert_eq!(srv.app.snakes()[0].moves(), u64::from(index));
    // /end forgets the game: the next move reads the living snakes only (a 3-seat game).
    assert_eq!(post(a, "/s/end", &four.to_string()).0, 200);
    index += 1;
    let got = move_of(&post(a, "/s/move", &four.to_string()).1);
    assert_eq!(got, expect(&four, None, index));
}

#[test]
fn moves_are_logged() {
    let Some(dir) = run_dir() else { return };
    let path =
        std::env::temp_dir().join(format!("slinky-server-http-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let srv = server(
        &[format!("ppo={dir}:simulations=16")],
        Some(path.clone()),
        None,
    );
    let a = srv.addr;
    let mut req = example("duel");
    req["game"]["id"] = json!("logged");
    let body = req.to_string();
    assert_eq!(post(a, "/ppo/start", &body), (200, "{}".to_string()));
    let (code, answer) = http(
        a,
        "POST",
        "/ppo/move",
        Some(body.as_bytes()),
        &["Cf-Ray: 8d0f2a7b1c3e4f56-SEA"],
    );
    assert_eq!(code, 200);
    // Play our answer: the next request's head is one step that way.
    let (dx, dy) = match move_of(&answer).as_str() {
        "up" => (0, 1),
        "down" => (0, -1),
        "left" => (-1, 0),
        _ => (1, 0),
    };
    let mut next = req.clone();
    next["turn"] = json!(1);
    let head = &next["you"]["head"];
    let (x, y) = (
        head["x"].as_i64().unwrap() + dx,
        head["y"].as_i64().unwrap() + dy,
    );
    next["you"]["head"] = json!({"x": x, "y": y});
    assert_eq!(
        post(a, "/ppo/end", &next.to_string()),
        (200, "{}".to_string())
    );
    srv.stop();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert_eq!(lines[0]["move"], json!(move_of(&answer)));
    assert_eq!(lines[0]["prev"], Value::Null);
    assert_eq!(lines[0].get("end"), None);
    assert_eq!(lines[1]["move"], Value::Null);
    assert_eq!(lines[1]["prev"], json!("applied"));
    assert_eq!(lines[1]["end"], json!(true));
    assert_eq!(
        lines[0]["agent"],
        json!(format!("az_modes:{dir}:simulations=16"))
    );
    assert!(lines[0]["ms"].as_f64().unwrap() > 0.0);
    assert_eq!(lines[0]["game"], json!("logged"));
    assert!(lines[0]["wall"].as_f64().unwrap() >= lines[0]["ms"].as_f64().unwrap());
    assert_eq!(lines[0]["colo"], json!("SEA"));
    let keys: Vec<&String> = lines[0].as_object().unwrap().keys().collect();
    assert_eq!(keys.len(), 15);
}

#[test]
fn time_budget_and_answer_by() {
    let Some(dir) = run_dir() else { return };
    // A 60 ms budget stops a search that would take seconds.
    let srv = server(&[format!("b={dir}:budget_ms=60")], None, None);
    let req = example("standard").to_string();
    let t = Instant::now();
    let (code, body) = post(srv.addr, "/b/move", &req);
    let took = t.elapsed();
    assert_eq!(code, 200);
    assert!(MOVES.contains(&move_of(&body).as_str()));
    assert!(took < Duration::from_millis(400), "{took:?}");
    srv.stop();
    // A search still running after answer_by gets the safe move in time.
    let path =
        std::env::temp_dir().join(format!("slinky-server-late-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let srv = server(
        &[format!("slow={dir}:simulations=40000")],
        Some(path.clone()),
        Some(Duration::from_millis(100)),
    );
    let t = Instant::now();
    let (code, _) = post(srv.addr, "/slow/move", &req);
    let took = t.elapsed();
    assert_eq!(code, 200);
    assert!(took < Duration::from_millis(400), "{took:?}");
    srv.app.ledger().flush();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(text.contains("\"agent\": \"safe_move\""), "{text}");
}
