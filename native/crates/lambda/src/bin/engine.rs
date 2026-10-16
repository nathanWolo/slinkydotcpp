//! `slinky-engine`: the native snakes as a child process of the Python Lambda handler
//! (`deploy/lambda/lambda_function.py`), which starts it once per container and talks to it
//! in JSON lines.
//!
//! ```text
//! stdin:  {"id": 7, "snake": "az-256", "kind": "start" | "move" | "end",
//!          "body": "<the request body, as received>", "elapsed_ms": 0.4}
//! stdout: {"id": 7, "code": 200, "reply": {"move": "up"}}
//! ```
//!
//! One answer line per request line, in order. `elapsed_ms` is how long ago the request
//! reached the Python handler: the move's deadline (`game.timeout` less `DEADLINE_MARGIN_MS`,
//! see `slinky_lambda::engine`) counts from there. The settings are the Lambda's environment
//! variables (`SNAKES`, `WEIGHTS_DIR`, `SIMULATIONS`, `BUDGET_MS`, `DEADLINE_MARGIN_MS`, ...):
//! the weights load once, at start, and the `/start` cache lives as long as the process. The
//! JSON log lines go to stderr (CloudWatch through the Python runtime). A line that is not a
//! request gets `{"id": null, "code": 400, ...}`; a panic while answering (built with
//! `panic = "unwind"`, as the deploy build is) gets the safe move for a `/move` and a 500
//! otherwise, and the process keeps reading. It exits at the end of its input. Bad settings
//! exit at once with status 2 and a message on stderr.

use serde_json::{Value, json};
use slinky_lambda::engine::Settings;
use slinky_lambda::{App, log_to_stderr};
use std::io::{BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// The answer to one request line.
fn answer(app: &App, line: &[u8]) -> Value {
    let arrival = Instant::now();
    let text = String::from_utf8_lossy(line);
    let Ok(msg) = serde_json::from_str::<Value>(&text) else {
        return json!({"id": null, "code": 400, "reply": {"error": "not a request line"}});
    };
    let id = msg["id"].clone();
    let (Some(snake), Some(kind), Some(body)) = (
        msg["snake"].as_str(),
        msg["kind"].as_str(),
        msg["body"].as_str(),
    ) else {
        return json!({"id": id, "code": 400, "reply": {"error": "not a request line"}});
    };
    let elapsed = msg["elapsed_ms"]
        .as_f64()
        .unwrap_or(0.0)
        .clamp(0.0, 10_000.0);
    let arrival = arrival
        .checked_sub(Duration::from_secs_f64(elapsed / 1e3))
        .unwrap_or(arrival);
    let path = format!("/{snake}/{kind}");
    let (code, reply) = match catch_unwind(AssertUnwindSafe(|| {
        app.handle("POST", &path, body, arrival)
    })) {
        Ok((code, text)) => (code, serde_json::from_str(&text).unwrap_or(Value::Null)),
        Err(_) => {
            eprintln!("{}", json!({"event": "panic", "path": path}));
            if kind == "move" {
                let mv = slinky_api::safe_move_json(body);
                (200, json!({"move": mv.name()}))
            } else {
                (500, json!({"error": "internal error"}))
            }
        }
    };
    json!({"id": id, "code": code, "reply": reply})
}

fn main() -> ExitCode {
    log_to_stderr();
    let settings = match Settings::from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("slinky-engine: bad settings: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "{}",
        json!({"event": "init", "panic": if cfg!(panic = "unwind") { "unwind" } else { "abort" },
               "deadline_margin_ms": settings.deadline_margin_ms, "pid": std::process::id()})
    );
    let app = App::load(&settings);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut line = Vec::new();
    loop {
        line.clear();
        match input.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => return ExitCode::SUCCESS,
            Ok(_) => {}
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let out = answer(&app, &line);
        let mut w = stdout.lock();
        if writeln!(w, "{out}").and_then(|()| w.flush()).is_err() {
            return ExitCode::SUCCESS; // the handler is gone
        }
    }
}
