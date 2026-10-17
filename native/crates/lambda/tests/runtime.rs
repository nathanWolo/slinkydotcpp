//! Review tests: the real `slinky-lambda` binary (`main.rs` + `lambda_http`) against a mock
//! Lambda Runtime API that hands out real Function URL (payload v2) events.
//!
//! These exercise what `tests/handler.rs` cannot, because it calls `App::handle` directly: the
//! event decoding (base64 bodies, `rawPath` with and without a trailing slash, query strings,
//! missing bodies), the response shape, the `/start` cache across invocations of one process,
//! a process that keeps serving after a bad event, and init with bad settings or no weights.
//!
//! The binary is the one cargo builds for this package (`CARGO_BIN_EXE_slinky-lambda`), with
//! the workspace's release profile (`panic = "abort"`): what ships unless the build overrides it.
//! The run dir is `$NAZ_RUN` (default: the deployed M network); without it the tests SKIP.

use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
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

fn b64(data: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A Function URL (payload format 2.0) event as Lambda delivers it.
fn url_event(method: &str, raw_path: &str, query: &str, body: Option<&str>, base64: bool) -> Value {
    let host = "abc.lambda-url.us-east-1.on.aws";
    json!({
        "version": "2.0", "routeKey": "$default", "rawPath": raw_path, "rawQueryString": query,
        "headers": {"content-type": "application/json", "host": host,
                    "user-agent": "BattlesnakeEngine/1.0"},
        "requestContext": {
            "accountId": "anonymous", "apiId": "abc", "domainName": host, "domainPrefix": "abc",
            "http": {"method": method, "path": raw_path, "protocol": "HTTP/1.1",
                     "sourceIp": "1.2.3.4", "userAgent": "BattlesnakeEngine/1.0"},
            "requestId": "r", "routeKey": "$default", "stage": "$default",
            "time": "15/Oct/2026:19:00:00 +0000", "timeEpoch": 1_792_090_800_000u64},
        "body": body, "isBase64Encoded": base64,
    })
}

/// What the runtime posted back for one invocation.
#[derive(Debug)]
enum Outcome {
    /// `POST .../response`: the proxy response object.
    Response(Value),
    /// `POST .../error` (or init error): the error body.
    Error(String),
    /// Nothing within the timeout, or the process died.
    Nothing,
}

struct Shared {
    events: Mutex<VecDeque<(String, String)>>,
    ready: Condvar,
}

/// A mock Runtime API on 127.0.0.1 and the bootstrap process attached to it.
struct Runtime {
    shared: Arc<Shared>,
    results: Receiver<(String, String)>,
    child: Child,
    logs: Arc<Mutex<Vec<String>>>,
    next_id: usize,
}

fn read_request(r: &mut BufReader<TcpStream>) -> Option<(String, String, String)> {
    let mut line = String::new();
    if r.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut it = line.split_whitespace();
    let (method, path) = (it.next()?.to_string(), it.next()?.to_string());
    let (mut len, mut chunked) = (0usize, false);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 {
            return None;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':')?;
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
        if k == "content-length" {
            len = v.parse().ok()?;
        } else if k == "transfer-encoding" && v.contains("chunked") {
            chunked = true;
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let mut sz = String::new();
            r.read_line(&mut sz).ok()?;
            let n = usize::from_str_radix(sz.trim().split(';').next()?, 16).ok()?;
            let mut chunk = vec![0; n + 2];
            r.read_exact(&mut chunk).ok()?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
    } else {
        body.resize(len, 0);
        r.read_exact(&mut body).ok()?;
    }
    Some((method, path, String::from_utf8_lossy(&body).into_owned()))
}

fn serve(stream: TcpStream, shared: Arc<Shared>, tx: Sender<(String, String)>) {
    let mut w = stream.try_clone().unwrap();
    let mut r = BufReader::new(stream);
    while let Some((method, path, body)) = read_request(&mut r) {
        if method == "GET" && path.ends_with("/invocation/next") {
            let (id, payload) = {
                let mut q = shared.events.lock().unwrap();
                loop {
                    if let Some(e) = q.pop_front() {
                        break e;
                    }
                    q = shared.ready.wait(q).unwrap();
                }
            };
            let deadline = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
                + 60_000;
            let head = format!(
                "HTTP/1.1 200 OK\r\nLambda-Runtime-Aws-Request-Id: {id}\r\n\
                 Lambda-Runtime-Deadline-Ms: {deadline}\r\n\
                 Lambda-Runtime-Invoked-Function-Arn: arn:aws:lambda:us-east-1:1:function:f\r\n\
                 Lambda-Runtime-Trace-Id: Root=1-0-0\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                payload.len()
            );
            if w.write_all(head.as_bytes()).is_err() || w.write_all(payload.as_bytes()).is_err() {
                return;
            }
        } else {
            let _ = tx.send((path, body));
            if w.write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 2\r\n\r\n{}")
                .is_err()
            {
                return;
            }
        }
    }
}

impl Runtime {
    fn start(env: &[(&str, String)]) -> Runtime {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let shared = Arc::new(Shared {
            events: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        });
        let (tx, results) = channel();
        {
            let shared = shared.clone();
            thread::spawn(move || {
                for s in listener.incoming().flatten() {
                    let (shared, tx) = (shared.clone(), tx.clone());
                    thread::spawn(move || serve(s, shared, tx));
                }
            });
        }
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_slinky-lambda"));
        cmd.env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("AWS_LAMBDA_RUNTIME_API", format!("127.0.0.1:{port}"))
            .env("AWS_LAMBDA_FUNCTION_NAME", "f")
            .env("AWS_LAMBDA_FUNCTION_VERSION", "1")
            .env("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1769")
            .env("AWS_LAMBDA_LOG_GROUP_NAME", "g")
            .env("AWS_LAMBDA_LOG_STREAM_NAME", "s")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        let logs = Arc::new(Mutex::new(Vec::new()));
        for pipe in [
            Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
            Box::new(child.stderr.take().unwrap()),
        ] {
            let logs = logs.clone();
            thread::spawn(move || {
                for l in BufReader::new(pipe).lines().map_while(Result::ok) {
                    logs.lock().unwrap().push(l);
                }
            });
        }
        Runtime {
            shared,
            results,
            child,
            logs,
            next_id: 0,
        }
    }

    /// Hands out `payload` (any JSON) and waits for what the runtime posts back.
    fn invoke_raw(&mut self, payload: &Value) -> Outcome {
        self.next_id += 1;
        let id = format!("req-{}", self.next_id);
        self.shared
            .events
            .lock()
            .unwrap()
            .push_back((id.clone(), payload.to_string()));
        self.shared.ready.notify_all();
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(30) {
            match self.results.recv_timeout(Duration::from_millis(100)) {
                Ok((path, body)) if path.contains(&id) && path.ends_with("/response") => {
                    return Outcome::Response(serde_json::from_str(&body).unwrap());
                }
                Ok((path, body)) if path.contains(&id) || path.contains("/init/error") => {
                    return Outcome::Error(body);
                }
                Ok(_) => {}
                Err(_) => {
                    if let Ok(Some(_)) = self.child.try_wait() {
                        return Outcome::Nothing; // the process died
                    }
                }
            }
        }
        Outcome::Nothing
    }

    /// A Function URL request: `(status, parsed JSON body, whole proxy response)`.
    fn call(
        &mut self,
        method: &str,
        raw_path: &str,
        query: &str,
        body: Option<&str>,
        base64: bool,
    ) -> (u16, Value, Value) {
        let body = if base64 {
            body.map(|b| b64(b.as_bytes()))
        } else {
            body.map(str::to_string)
        };
        match self.invoke_raw(&url_event(method, raw_path, query, body.as_deref(), base64)) {
            Outcome::Response(r) => {
                let code = r["statusCode"].as_u64().unwrap_or(0) as u16;
                let text = r["body"].as_str().unwrap_or("").to_string();
                let parsed = serde_json::from_str(&text).unwrap_or(Value::Null);
                (code, parsed, r)
            }
            Outcome::Error(e) => panic!("{method} {raw_path}: error {e}"),
            Outcome::Nothing => panic!("{method} {raw_path}: no answer: {:?}", self.logs()),
        }
    }

    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn logs(&self) -> Vec<String> {
        self.logs.lock().unwrap().clone()
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn weights_env(run: &std::path::Path) -> Vec<(&'static str, String)> {
    vec![
        (
            "WEIGHTS_DIR",
            run.parent().unwrap().to_str().unwrap().to_string(),
        ),
        (
            "SNAKES",
            format!("m={}", run.file_name().unwrap().to_str().unwrap()),
        ),
        ("TIE_SEED", "3".into()),
    ]
}

fn is_move(v: &Value) -> bool {
    ["up", "down", "left", "right"].contains(&v["move"].as_str().unwrap_or(""))
}

/// Paths with and without a trailing slash, query strings and base64 bodies all route.
#[test]
fn function_url_event_shapes() {
    let Some(run) = run_dir() else { return };
    let mut rt = Runtime::start(&weights_env(&run));
    let req = examples()["requests"]["duel"].to_string();

    for path in ["/m", "/m/", "//m//"] {
        for q in ["", "x=1&y=2"] {
            let (c, v, r) = rt.call("GET", path, q, None, false);
            assert_eq!(c, 200, "GET {path}?{q}");
            assert_eq!(v["apiversion"], "1", "GET {path}?{q}");
            assert_eq!(r["headers"]["content-type"], "application/json", "{r}");
        }
    }
    let mut seen = Vec::new();
    for path in ["/m/move", "/m/move/", "//m/move"] {
        for q in ["", "a=b"] {
            for base64 in [false, true] {
                let (c, v, _) = rt.call("POST", path, q, Some(&req), base64);
                assert_eq!(c, 200, "POST {path}?{q} base64={base64}");
                assert!(is_move(&v), "POST {path}?{q} base64={base64}: {v}");
                seen.push(v["move"].clone());
            }
        }
    }
    // One request, one fixed seed: every shape gets the same move.
    assert!(seen.windows(2).all(|w| w[0] == w[1]), "{seen:?}");
    for (path, base64) in [("/m/start", true), ("/m/end", true), ("/m/start/", false)] {
        let (c, v, _) = rt.call("POST", path, "", Some(&req), base64);
        assert_eq!((c, v), (200, json!({})), "{path} base64={base64}");
    }
    // Not routes.
    for (m, p) in [
        ("GET", "/"),
        ("GET", ""),
        ("POST", "/m"),
        ("GET", "/m/move"),
        ("PUT", "/m/move"),
        ("HEAD", "/m"),
        ("POST", "/m/move/extra"),
        ("POST", "/M/move"),
    ] {
        let (c, _, _) = rt.call(m, p, "", Some(&req), false);
        assert_eq!(c, 404, "{m} {p:?}");
    }
    // No body, an empty body, a non-JSON base64 body: 400 (as the Python handler).
    for (body, base64) in [(None, false), (Some(""), false), (Some("not json"), true)] {
        let (c, _, _) = rt.call("POST", "/m/move", "", body, base64);
        assert_eq!(c, 400, "{body:?} base64={base64}");
    }
    assert!(rt.alive());
}

/// The `/start` snake count is remembered by the process across invocations: a 4-snake standard
/// game seen at `/move` with two snakes left is still standard (4 seats), not a duel.
#[test]
fn start_cache_survives_across_invocations() {
    let Some(run) = run_dir() else { return };
    let mut rt = Runtime::start(&weights_env(&run));
    let ex = examples();
    let start = ex["requests"]["standard"].clone();
    let mut later = start.clone();
    later["turn"] = json!(30);
    let two: Vec<Value> = later["board"]["snakes"].as_array().unwrap()[..2].to_vec();
    later["board"]["snakes"] = json!(two);
    let mut other = later.clone();
    other["game"]["id"] = json!("never-started");

    let mut g = start.clone();
    g["game"]["id"] = json!("cached-game");
    assert_eq!(
        rt.call("POST", "/m/start", "", Some(&g.to_string()), false)
            .0,
        200
    );
    // Unrelated traffic in between.
    for _ in 0..3 {
        rt.call("GET", "/m", "", None, false);
    }
    let mut l = later.clone();
    l["game"]["id"] = json!("cached-game");
    assert_eq!(
        rt.call("POST", "/m/move", "", Some(&l.to_string()), true).0,
        200
    );
    assert_eq!(
        rt.call("POST", "/m/move", "", Some(&other.to_string()), false)
            .0,
        200
    );
    thread::sleep(Duration::from_millis(200)); // let stdout drain
    let logs = rt.logs();
    let mode_of = |game: &str| -> String {
        logs.iter()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|v| v["game"] == game && v["move"].is_string())
            .map(|v| v["mode"].as_str().unwrap_or("?").to_string())
            .unwrap_or_else(|| format!("no log line in {logs:?}"))
    };
    assert_eq!(mode_of("cached-game"), "standard");
    assert_eq!(mode_of("never-started"), "duel");
}

/// A non-HTTP event (a warmer ping, a console test event) must not take the function down:
/// the next real move is answered by the same process.
#[test]
fn survives_non_http_events() {
    let Some(run) = run_dir() else { return };
    let mut rt = Runtime::start(&weights_env(&run));
    let req = examples()["requests"]["royale"].to_string();
    for ev in [
        json!({}),
        json!({"warmer": true}),
        json!([1, 2]),
        json!("x"),
    ] {
        let out = rt.invoke_raw(&ev);
        assert!(
            !matches!(out, Outcome::Nothing),
            "{ev}: no answer, process alive: {}",
            rt.alive()
        );
        let (c, v, _) = rt.call("POST", "/m/move", "", Some(&req), false);
        assert_eq!(c, 200, "after {ev}");
        assert!(is_move(&v));
    }
}

/// An event whose base64 body does not decode. Lambda itself never sends one. (Fixed: a panic
/// inside `lambda_http`'s event conversion, `Body::from_maybe_encoded` calls `expect`, aborted
/// the process under `panic = "abort"`; the event is now read as plain JSON and such a body is
/// a 400.)
#[test]
fn bad_base64_body_does_not_kill_the_process() {
    let Some(run) = run_dir() else { return };
    let mut rt = Runtime::start(&weights_env(&run));
    let out = rt.invoke_raw(&url_event(
        "POST",
        "/m/move",
        "",
        Some("!!!not base64"),
        true,
    ));
    thread::sleep(Duration::from_millis(200));
    assert!(
        !matches!(out, Outcome::Nothing),
        "no answer: {:?}",
        rt.logs()
    );
    match out {
        Outcome::Response(r) => assert_eq!(r["statusCode"], 400, "{r}"),
        other => panic!("{other:?}"),
    }
    assert!(rt.alive(), "process died: {:?}", rt.logs());
    let req = examples()["requests"]["duel"].to_string();
    let (c, v, _) = rt.call("POST", "/m/move", "", Some(&req), false);
    assert!(c == 200 && is_move(&v));
}

/// A panic while answering a request (`DEBUG_PANIC`, in the search or in the handler around
/// it) answers a move and the process keeps serving. Needs the binary built with
/// `panic = "unwind"`, as the deploy build is (`CARGO_PROFILE_RELEASE_PANIC=unwind`); an
/// `abort` build (the workspace default) prints SKIP: there a panic ends the container.
#[test]
fn a_panic_in_a_request_does_not_kill_the_process() {
    let Some(run) = run_dir() else { return };
    for (hook, id) in [("search", "boom-s"), ("handler", "boom-h")] {
        let mut env = weights_env(&run);
        env.push(("DEBUG_PANIC", format!("{hook}:{id}")));
        let mut rt = Runtime::start(&env);
        let mut req = examples()["requests"]["royale"].clone();
        let ok = req.to_string();
        let (c, _, _) = rt.call("GET", "/m", "", None, false);
        assert_eq!(c, 200);
        thread::sleep(Duration::from_millis(100));
        let init = rt.logs().join("\n");
        if init.contains("\"panic\":\"abort\"") {
            eprintln!(
                "SKIP: the binary is built with panic = abort; run the tests with \
                 CARGO_PROFILE_RELEASE_PANIC=unwind (the deploy build's setting)"
            );
            return;
        }
        assert!(init.contains("\"panic\":\"unwind\""), "{init}");
        req["game"]["id"] = json!(id);
        for _ in 0..2 {
            let (c, v, _) = rt.call("POST", "/m/move", "", Some(&req.to_string()), false);
            assert!(c == 200 && is_move(&v), "{hook}: {c} {v}");
        }
        let (c, v, _) = rt.call("POST", "/m/move", "", Some(&ok), false);
        assert!(c == 200 && is_move(&v), "{hook} after the panic");
        assert!(rt.alive(), "{hook}: process died");
        thread::sleep(Duration::from_millis(100));
        assert!(
            rt.logs().iter().any(|l| l.contains("DEBUG_PANIC")),
            "{hook}: the panic did not happen"
        );
    }
}

/// Missing weights at cold start: the function still starts and every move is a legal-ish move.
#[test]
fn missing_weights_still_answer_moves() {
    let mut rt = Runtime::start(&[
        ("WEIGHTS_DIR", "/nonexistent-weights".into()),
        ("SNAKES", "a=x,b=y".into()),
    ]);
    let ex = examples();
    for snake in ["a", "b"] {
        for mode in ["standard", "duel", "royale", "constrictor"] {
            let req = ex["requests"][mode].to_string();
            let (c, v, _) = rt.call("POST", &format!("/{snake}/move"), "", Some(&req), false);
            assert_eq!(c, 200);
            assert!(is_move(&v), "{snake} {mode}: {v}");
        }
    }
    thread::sleep(Duration::from_millis(200));
    let loads = rt
        .logs()
        .iter()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["event"] == "load" && v["ok"] == false)
        .count();
    assert_eq!(loads, 2, "{:?}", rt.logs());
}

/// Bad settings fail the init: the process must exit (not hang waiting for events) and say why.
#[test]
fn bad_settings_fail_init() {
    for (k, v) in [
        ("SIMULATIONS", "0"),
        ("THREADS", "lots"),
        ("SNAKES", "a=m,a=n"),
        ("BUDGET_MS", "-5"),
    ] {
        let mut rt = Runtime::start(&[(k, v.into())]);
        let t = Instant::now();
        let status = loop {
            if let Ok(Some(s)) = rt.child.try_wait() {
                break Some(s);
            }
            if t.elapsed() > Duration::from_secs(10) {
                break None;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let status = status.unwrap_or_else(|| panic!("{k}={v}: init did not exit"));
        assert!(!status.success(), "{k}={v}");
        thread::sleep(Duration::from_millis(100));
        let logs = rt.logs().join("\n");
        assert!(
            logs.contains(k),
            "{k}={v}: init error does not name it: {logs}"
        );
    }
}

/// Paths `http::Uri` would refuse (space, quote, braces, non-ASCII, bad escapes): `lambda_http`
/// builds the request URI with an `expect`, so check that none of them reaches it and kills the
/// process (under `panic = "abort"`); each gets a 404 and the next move still works.
#[test]
fn odd_paths_do_not_kill_the_process() {
    let Some(run) = run_dir() else { return };
    let mut rt = Runtime::start(&weights_env(&run));
    let req = examples()["requests"]["duel"].to_string();
    for p in [
        "/m/ move",
        "/m/\"x\"",
        "/m/{move}",
        "/m/mo|ve",
        "/m/\u{e9}",
        "/m/%zz",
        "/m/move#x",
    ] {
        match rt.invoke_raw(&url_event("GET", p, "", None, false)) {
            Outcome::Response(r) => assert_eq!(r["statusCode"], 404, "{p:?}"),
            other => panic!("{p:?}: {other:?} {:?}", rt.logs()),
        }
        let (c, v, _) = rt.call("POST", "/m/move", "", Some(&req), false);
        assert!(c == 200 && is_move(&v), "after {p:?}");
    }
}

/// A request the Function URL front end received a while ago (its `requestContext.timeEpoch`)
/// searches only for what is left of the snake's time budget, counted from that receipt, and
/// is logged as late; a receipt time more than 3 s old is not believed (the old behaviour).
#[test]
fn time_budget_counts_from_the_front_end_receipt() {
    let Some(run) = run_dir() else { return };
    let subdir = run.file_name().unwrap().to_str().unwrap().to_string();
    let mut env = weights_env(&run);
    env.retain(|(k, _)| *k != "SNAKES");
    env.push(("SNAKES", format!("t={subdir}:budget_ms=300")));
    let mut rt = Runtime::start(&env);
    let body = examples()["requests"]["duel"].to_string();
    let now_ms = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    };
    let timed = |rt: &mut Runtime, received: u64| {
        let mut ev = url_event("POST", "/t/move", "", Some(&body), false);
        ev["requestContext"]["timeEpoch"] = json!(received);
        let t = Instant::now();
        let out = rt.invoke_raw(&ev);
        let ms = t.elapsed().as_millis();
        let Outcome::Response(r) = out else {
            panic!("no answer: {:?}", rt.logs())
        };
        assert!(is_move(
            &serde_json::from_str(r["body"].as_str().unwrap()).unwrap()
        ));
        ms
    };
    timed(&mut rt, now_ms()); // the first move loads nothing more but warms the caches
    let fresh = timed(&mut rt, now_ms());
    let late = timed(&mut rt, now_ms() - 250);
    let stale = timed(&mut rt, now_ms() - 10_000);
    assert!((280..450).contains(&fresh), "fresh request took {fresh} ms");
    assert!(
        late < 150,
        "a request received 250 ms ago took {late} ms more"
    );
    assert!(
        (280..450).contains(&stale),
        "a 10 s old receipt time was believed: {stale} ms"
    );
    thread::sleep(Duration::from_millis(200)); // let stdout drain
    let late_lines: Vec<Value> = rt
        .logs()
        .iter()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["event"] == "late")
        .collect();
    // the first request may also be late: it waited for the process to start
    let long: Vec<u64> = late_lines
        .iter()
        .filter_map(|v| v["lag_ms"].as_u64())
        .filter(|&l| l >= 200)
        .collect();
    assert_eq!(long.len(), 1, "{late_lines:?}");
    assert!(long[0] >= 250, "{late_lines:?}");
}
