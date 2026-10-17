//! The Lambda entry point: weights load once at init, then each invocation is one request.
//!
//! The event is read as plain JSON (a Function URL / API Gateway payload 2.0 event, or a 1.0
//! one), not through `lambda_http`, whose event conversion `expect`s a valid base64 body and a
//! valid URI: no request content can panic here. A body that does not decode is a 400, an
//! event that is not an HTTP request a 404 (as the Python handler). A panic anywhere in the
//! handler (only possible in an unwinding build, which the deploy build is) answers a `/move`
//! with the safe move and anything else with a 500, and the process keeps serving.

use base64::Engine as _;
use lambda_runtime::{Error, LambdaEvent, run, service_fn};
use serde_json::{Value, json};
use slinky_lambda::App;
use slinky_lambda::engine::Settings;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A request that reached the handler this many ms after the front end received it is logged.
const LATE_MS: u64 = 25;

/// `(method, raw path, body)` of a Function URL / API Gateway event; `Err` is the reply for a
/// body that does not decode.
fn request_of(event: &Value) -> Result<(String, String, String), (u16, String)> {
    let method = event["requestContext"]["http"]["method"]
        .as_str()
        .or_else(|| event["httpMethod"].as_str())
        .unwrap_or("GET")
        .to_string();
    let path = event["rawPath"]
        .as_str()
        .or_else(|| event["path"].as_str())
        .unwrap_or("/")
        .to_string();
    let raw = event["body"].as_str().unwrap_or("");
    let body = if event["isBase64Encoded"].as_bool() == Some(true) {
        let compact: String = raw.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&compact)
            .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&compact));
        match decoded {
            Ok(b) => String::from_utf8_lossy(&b).into_owned(),
            Err(_) => {
                return Err((
                    400,
                    json!({"error": "not a Battlesnake request"}).to_string(),
                ));
            }
        }
    } else {
        raw.to_string()
    };
    Ok((method, path, body))
}

fn response(code: u16, body: String) -> Value {
    json!({
        "statusCode": code,
        "headers": {"content-type": "application/json"},
        "body": body,
        "isBase64Encoded": false,
    })
}

/// When the invocation must have answered (`deadline` is ms since the epoch); `None` when the
/// runtime gave no usable deadline.
fn invocation_limit(deadline_ms: u64, now: Instant) -> Option<Instant> {
    let epoch_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    if deadline_ms == 0 {
        return None;
    }
    Some(now + Duration::from_millis(deadline_ms.saturating_sub(epoch_ms)))
}

/// How long ago the Function URL front end received the request (`requestContext.timeEpoch`,
/// ms since the epoch); 0 when the event has no such time (an invoke without a URL) or one
/// more than 3 s old (longer than the function's timeout: not a real receipt time). The clocks
/// may differ by a few ms, so a receipt time in the future counts as 0.
fn front_end_lag(event: &Value) -> Duration {
    let Some(received) = event["requestContext"]["timeEpoch"].as_u64() else {
        return Duration::ZERO;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    match now.saturating_sub(received) {
        lag @ 0..=3000 => Duration::from_millis(lag),
        _ => Duration::ZERO,
    }
}

fn handle(app: &App, event: LambdaEvent<Value>) -> Value {
    let entry = Instant::now();
    // The time budget and the deadline count from the front end's receipt, so a request that
    // waited inside Lambda (a container start, a busy front end) searches less instead of
    // answering late.
    let lag = front_end_lag(&event.payload);
    let arrival = entry.checked_sub(lag).unwrap_or(entry);
    let limit = invocation_limit(event.context.deadline, entry);
    // A direct invocation's benchmark (an HTTP event never has this key).
    if let Some(spec) = event.payload.get("bench") {
        return app.bench(spec, limit);
    }
    let (method, path, body) = match request_of(&event.payload) {
        Ok(r) => r,
        Err((code, text)) => return response(code, text),
    };
    if lag >= Duration::from_millis(LATE_MS) {
        println!(
            "{}",
            json!({"event": "late", "path": path, "lag_ms": lag.as_millis() as u64})
        );
    }
    match catch_unwind(AssertUnwindSafe(|| {
        app.handle_within(&method, &path, &body, arrival, limit)
    })) {
        Ok((code, text)) => response(code, text),
        Err(_) => {
            let is_move = method == "POST"
                && path.split('/').filter(|p| !p.is_empty()).nth(1) == Some("move");
            println!(
                "{}",
                json!({"event": "panic", "path": path, "safe_move": is_move})
            );
            if is_move {
                let mv = slinky_api::safe_move_json(&body);
                response(200, json!({"move": mv.name()}).to_string())
            } else {
                response(500, json!({"error": "internal error"}).to_string())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let settings = Settings::from_env().map_err(|e| format!("bad settings: {e}"))?;
    println!(
        "{}",
        json!({"event": "init", "panic": if cfg!(panic = "unwind") { "unwind" } else { "abort" },
               "deadline_margin_ms": settings.deadline_margin_ms})
    );
    let app = Arc::new(App::load(&settings));
    run(service_fn(move |event: LambdaEvent<Value>| {
        let app = app.clone();
        async move { Ok::<Value, Error>(handle(&app, event)) }
    }))
    .await
}
