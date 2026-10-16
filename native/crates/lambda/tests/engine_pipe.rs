//! The `slinky-engine` process (JSON lines on stdin and stdout) against the in-process
//! handler: the same moves, ids echoed, junk lines answered, the `/start` cache kept across
//! lines, logs on stderr only, and a 200 ms game answered within its deadline.
//!
//! The run dir is `$NAZ_RUN` (default: the deployed M network); without it the tests SKIP.

use serde_json::{Value, json};
use slinky_lambda::App;
use slinky_lambda::engine::Settings;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
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

fn examples() -> Value {
    serde_json::from_str(include_str!("../../server/data/examples.json")).unwrap()
}

fn env_of(run: &std::path::Path) -> Vec<(String, String)> {
    vec![
        (
            "WEIGHTS_DIR".into(),
            run.parent().unwrap().to_str().unwrap().into(),
        ),
        (
            "SNAKES".into(),
            format!("az-256={}", run.file_name().unwrap().to_str().unwrap()),
        ),
    ]
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    fn start(env: &[(String, String)]) -> Engine {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_slinky-engine"));
        cmd.env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Engine {
            child,
            stdin,
            stdout,
        }
    }

    fn send_raw(&mut self, line: &str) -> Value {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
        let mut out = String::new();
        self.stdout.read_line(&mut out).unwrap();
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out:?}"))
    }

    fn send(&mut self, id: u64, kind: &str, body: &str) -> Value {
        let line = json!({"id": id, "snake": "az-256", "kind": kind, "body": body,
                          "elapsed_ms": 0.0});
        let out = self.send_raw(&line.to_string());
        assert_eq!(out["id"], json!(id), "{out}");
        out
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn engine_process_matches_the_handler() {
    let Some(run) = run_dir() else { return };
    let env = env_of(&run);
    let settings =
        Settings::from_lookup(&|k| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()))
            .unwrap();
    let app = App::load(&settings);
    let mut e = Engine::start(&env);
    let ex = examples();
    let mut id = 0;
    for mode in ["standard", "duel", "royale", "constrictor"] {
        let req = ex["requests"][mode].to_string();
        id += 1;
        let s = e.send(id, "start", &req);
        assert_eq!(
            (s["code"].clone(), s["reply"].clone()),
            (json!(200), json!({}))
        );
        assert_eq!(
            app.handle("POST", "/az-256/start", &req, Instant::now()).0,
            200
        );
        for turn in [0u64, 5] {
            let mut r = ex["requests"][mode].clone();
            r["turn"] = json!(turn);
            let body = r.to_string();
            id += 1;
            let got = e.send(id, "move", &body);
            let (_, want) = app.handle("POST", "/az-256/move", &body, Instant::now());
            let want: Value = serde_json::from_str(&want).unwrap();
            assert_eq!(got["code"], 200);
            assert_eq!(got["reply"], want, "{mode} turn {turn}");
        }
        id += 1;
        assert_eq!(e.send(id, "end", &req)["code"], 200);
    }
    // Junk: not JSON, missing fields, a bad body, an unknown snake or kind.
    assert_eq!(e.send_raw("not json")["code"], 400);
    assert_eq!(e.send_raw("{\"id\": 3}")["code"], 400);
    assert_eq!(e.send(90, "move", "{nope")["code"], 400);
    assert_eq!(e.send(91, "move", "")["code"], 400);
    let other = json!({"id": 92, "snake": "nope", "kind": "move", "body": "{}"});
    assert_eq!(e.send_raw(&other.to_string())["code"], 404);
    // Still serving.
    let r = e.send(93, "move", &ex["requests"]["duel"].to_string());
    assert!(r["reply"]["move"].is_string(), "{r}");
}

/// A 200 ms game is answered by its deadline (200 less the 150 ms margin, from the arrival
/// the handler reports), even with a huge simulation count.
#[test]
fn engine_process_keeps_the_deadline() {
    let Some(run) = run_dir() else { return };
    let mut env = env_of(&run);
    env.push(("SIMULATIONS".into(), "1048576".into()));
    let mut e = Engine::start(&env);
    let mut req = examples()["requests"]["standard"].clone();
    // Warm up (the weights load at start).
    e.send(1, "move", &req.to_string());
    for (timeout, elapsed, want) in [(200, 0.0, 50.0), (500, 0.0, 350.0), (500, 100.0, 250.0)] {
        req["game"]["timeout"] = json!(timeout);
        let line = json!({"id": 2, "snake": "az-256", "kind": "move", "body": req.to_string(),
                          "elapsed_ms": elapsed});
        let t = Instant::now();
        let out = e.send_raw(&line.to_string());
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert!(out["reply"]["move"].is_string(), "{out}");
        assert!(
            ms > want - 5.0 && ms < want + 30.0,
            "timeout {timeout}, elapsed {elapsed}: {ms:.0} ms"
        );
    }
}
