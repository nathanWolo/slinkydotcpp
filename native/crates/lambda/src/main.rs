//! The Lambda entry point: weights load once at init, then each invocation is one request.

use lambda_http::{Body, Error, Request, RequestExt, Response, run, service_fn};
use slinky_lambda::App;
use slinky_lambda::engine::Settings;
use std::sync::Arc;
use std::time::Instant;

async fn handler(app: Arc<App>, event: Request) -> Result<Response<Body>, Error> {
    let arrival = Instant::now();
    let method = event.method().as_str().to_string();
    let path = event.raw_http_path().to_string();
    let body = match event.body() {
        Body::Text(s) => s.clone(),
        Body::Binary(b) => String::from_utf8_lossy(b).into_owned(),
        _ => String::new(),
    };
    let (code, text) = app.handle(&method, &path, &body, arrival);
    Ok(Response::builder()
        .status(code)
        .header("Content-Type", "application/json")
        .body(Body::Text(text))?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let settings = Settings::from_env().map_err(|e| format!("bad settings: {e}"))?;
    let app = Arc::new(App::load(&settings));
    run(service_fn(move |event| handler(app.clone(), event))).await
}
