//! slinky-server-native: the Battlesnake HTTP server of the native az_modes engine, a drop-in
//! replacement for the Python ladder server (`python -m slinky.server`, `src/slinky/server.py`).
//!
//! The binary `slinky-native-server` serves one or more snakes, each a run dir of the multi-mode
//! AlphaZero agent with its own search budget:
//!
//! ```text
//! slinky-native-server --snake az-256=/path/az-modes-M-s0:simulations=256 --port 8000 \
//!     --author nathanWolo --color "#ef4444" --move-log moves.jsonl
//! ```
//!
//! What it keeps from the Python server (see the modules for the details):
//!
//! * the routes, response bodies (byte for byte), status codes and body limits ([`http`]);
//! * the `/start` cache of each game's snake count, bounded at 1,000 games ([`engine`]);
//! * the move counter and its tie-noise keys, `fold_in(key(0), i)`, so the same requests in
//!   the same order give the same moves as a Python server ([`engine`]);
//! * the move log with the same JSONL fields and the applied/missed/gap verdicts, and the
//!   warnings for misses ([`ledger`]);
//! * the log lines, and the startup self-benchmark (ms per move per mode) on the same
//!   positions ([`engine::Snake::benchmark`]).
//!
//! What is new: the searches run on a pool of threads (one per core by default), each request's
//! search on one thread; a snake can search for a time budget (`:budget_ms=T`) instead of a
//! fixed simulation count; every search stops at the request's deadline (`game.timeout`
//! less `--margin-ms`, so 400 ms by default; a time budget also before `--answer-by-ms`) and
//! plays its best move so far; and a `/move` whose search has not answered `--answer-by-ms`
//! (400 ms by default) after its arrival, or by the request's deadline, is answered with the
//! safe move. The HTTP stack is hyper 1 (HTTP/1.1 only) on a
//! two-thread tokio runtime: header and body timeouts are built in, so a slow client never
//! holds a search thread.

pub mod engine;
pub mod examples;
pub mod http;
pub mod ledger;
pub mod log;
pub mod ordered;
pub mod pool;
pub mod pyfmt;

use http::App;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Serves `app` on `listener` until `shutdown` completes.
pub async fn serve(
    listener: std::net::TcpListener,
    app: Arc<App>,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let _ = stream.set_nodelay(true);
                    let app = Arc::clone(&app);
                    tokio::spawn(async move {
                        let svc = service_fn(move |req| {
                            let app = Arc::clone(&app);
                            async move { Ok::<_, Infallible>(http::handle(&app, req).await) }
                        });
                        let mut builder = http1::Builder::new();
                        builder
                            .timer(TokioTimer::new())
                            .header_read_timeout(http::BODY_TIMEOUT)
                            .keep_alive(false)
                            // Answer a client that shut its write side after the request
                            // (`nc -N`, some proxies and health checks), as Python does.
                            .half_close(true);
                        let _ = builder.serve_connection(TokioIo::new(stream), svc).await;
                    });
                }
                Err(e) => {
                    // Out of file descriptors, say: wait a little rather than spin.
                    info!("accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            },
            () = &mut shutdown => return Ok(()),
        }
    }
}

/// A server running on a background thread (tests and tools).
pub struct Background {
    /// The address it listens on.
    pub addr: SocketAddr,
    /// The app it serves.
    pub app: Arc<App>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Background {
    /// Binds `addr` (port 0: any free port) and serves `app` on a new thread.
    pub fn start(app: App, addr: &str) -> std::io::Result<Background> {
        let listener = std::net::TcpListener::bind(addr)?;
        let local = listener.local_addr()?;
        let app = Arc::new(app);
        let (stop, rx) = tokio::sync::oneshot::channel::<()>();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let served = Arc::clone(&app);
        let thread = std::thread::spawn(move || {
            let _ = runtime.block_on(serve(listener, served, async {
                let _ = rx.await;
            }));
        });
        Ok(Background {
            addr: local,
            app,
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    /// Stops the server and waits for its thread.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.app.ledger().flush();
    }
}

impl Drop for Background {
    fn drop(&mut self) {
        self.shutdown();
    }
}
