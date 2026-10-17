//! `slinky-native-server`: the Battlesnake ladder server on the native az_modes engine.
//!
//! ```text
//! slinky-native-server --snake NAME=RUN_DIR[:simulations=N][:budget_ms=T] [--snake ...]
//!     [--host 127.0.0.1] [--port 8000] [--author slinky] [--color "#3b82f6"]
//!     [--head default] [--tail default] [--move-log FILE] [--threads N]
//!     [--answer-by-ms 400] [--margin-ms 100] [--no-benchmark]
//! ```
//!
//! `--agent NAME=az_modes:RUN_DIR[:simulations=N]` (the Python server's form) is read as
//! `--snake`. See the crate README.

use slinky_api::InfoResponse;
use slinky_search::Search;
use slinky_server_native::engine::{Snake, SnakeSpec};
use slinky_server_native::http::{App, SEARCH_SLACK};
use slinky_server_native::ledger::MoveLedger;
use slinky_server_native::{info, serve};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const USAGE: &str = "\
usage: slinky-native-server --snake NAME=RUN_DIR[:simulations=N][:budget_ms=T] [--snake ...]
       [--host HOST] [--port PORT] [--author A] [--color C] [--head H] [--tail T]
       [--move-log FILE] [--threads N] [--answer-by-ms MS] [--margin-ms MS]
       [--no-benchmark]

A Battlesnake API server for the native multi-mode AlphaZero engine (az_modes run dirs).

  --snake NAME=RUN_DIR[:OPTS]  a snake served at /NAME (repeat for several). RUN_DIR holds
                               config.json and params.npz. OPTS: simulations=N (default 256:
                               the faithful ladder search) and/or budget_ms=T (stop T ms after
                               the request arrived; up to 16384 simulations unless N is given).
                               --agent NAME=az_modes:RUN_DIR[:OPTS] is the same.
  --host HOST                  address to bind (default 127.0.0.1)
  --port PORT                  port (default 8000)
  --author, --color, --head, --tail
                               the snake info (defaults: slinky, #3b82f6, default, default)
  --move-log FILE              append a JSON line per answer, with whether the engine applied
                               the game's previous answer (the Python server's MoveLedger)
  --threads N                  search threads (default: the number of cores)
  --answer-by-ms MS            answer the safe move if a search is not done MS ms after its
                               request arrived (default 400; 0: always wait). A
                               budget_ms search stops 20 ms before it.
  --margin-ms MS               every search stops by the request's game.timeout (500 if
                               missing) less MS, less 20 ms, and plays its best move so far;
                               the safe move is sent then at the latest (default 100)
  --no-benchmark               skip the startup self-benchmark (it also advances the move
                               counter that seeds the tie noise, as in the Python server)";

struct Options {
    host: String,
    port: u16,
    author: String,
    color: String,
    head: String,
    tail: String,
    snakes: Vec<SnakeSpec>,
    move_log: Option<PathBuf>,
    threads: usize,
    answer_by_ms: u64,
    margin_ms: u64,
    benchmark: bool,
}

const MAX_MARGIN_MS: u64 = 10_000;

fn parse_args(args: Vec<String>) -> Result<Options, String> {
    let mut o = Options {
        host: "127.0.0.1".into(),
        port: 8000,
        author: "slinky".into(),
        color: "#3b82f6".into(),
        head: "default".into(),
        tail: "default".into(),
        snakes: Vec::new(),
        move_log: None,
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        answer_by_ms: 400,
        margin_ms: 100,
        benchmark: true,
    };
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        // --flag=value or --flag value
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| it.next())
                .ok_or(format!("{flag} needs a value"))
        };
        let number = |v: String, flag: &str| -> Result<u64, String> {
            v.trim()
                .parse::<u64>()
                .map_err(|_| format!("{flag}: not a number: {v:?}"))
        };
        match flag.as_str() {
            "--snake" | "--agent" => o.snakes.push(SnakeSpec::parse(&value()?)?),
            "--host" => o.host = value()?,
            "--port" => {
                o.port = u16::try_from(number(value()?, "--port")?)
                    .map_err(|_| "--port: out of range".to_string())?;
            }
            "--author" => o.author = value()?,
            "--color" => o.color = value()?,
            "--head" => o.head = value()?,
            "--tail" => o.tail = value()?,
            "--move-log" => o.move_log = Some(PathBuf::from(value()?)),
            "--threads" => {
                o.threads = number(value()?, "--threads")?.clamp(1, 1024) as usize;
            }
            "--answer-by-ms" => o.answer_by_ms = number(value()?, "--answer-by-ms")?,
            "--margin-ms" => {
                o.margin_ms = number(value()?, "--margin-ms")?;
                if o.margin_ms > MAX_MARGIN_MS {
                    return Err(format!("--margin-ms: at most {MAX_MARGIN_MS}"));
                }
            }
            "--no-benchmark" => o.benchmark = false,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if o.snakes.is_empty() {
        return Err("give at least one --snake NAME=RUN_DIR".into());
    }
    let mut names: Vec<&str> = o.snakes.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err("two snakes have the same name".into());
    }
    Ok(o)
}

async fn shutdown_signal() {
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        () = term => {},
    }
}

fn main() -> ExitCode {
    let o = match parse_args(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(e) if e.is_empty() => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("{USAGE}\n\nslinky-native-server: error: {e}");
            return ExitCode::from(2);
        }
    };
    let mut snakes = Vec::new();
    let mut workspace = Search::new();
    for spec in &o.snakes {
        let t = Instant::now();
        let info = InfoResponse::new(
            &o.author,
            &o.color,
            &o.head,
            &o.tail,
            format!("slinky-native {}", spec.agent),
        );
        let snake = match Snake::load(spec, info) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "slinky-native-server: /{}: {}: {e}",
                    spec.name, spec.run_dir
                );
                return ExitCode::FAILURE;
            }
        };
        let budget = spec
            .budget_ms
            .map_or(String::new(), |b| format!(", stop after {b} ms"));
        if let Some(b) = spec.budget_ms {
            let cap = o
                .answer_by_ms
                .saturating_sub(SEARCH_SLACK.as_millis() as u64);
            if o.answer_by_ms > 0 && b > cap {
                info!(
                    "/{}: budget_ms={} is past --answer-by-ms {} less {} ms: its searches \
                     stop {} ms after the request",
                    spec.name,
                    b,
                    o.answer_by_ms,
                    SEARCH_SLACK.as_millis(),
                    cap
                );
            }
        }
        info!(
            "/{}: loaded {} ({} simulations per move{}) in {:.1} ms",
            spec.name,
            spec.run_dir,
            spec.simulations(),
            budget,
            t.elapsed().as_secs_f64() * 1e3
        );
        if o.benchmark {
            snake.benchmark(&mut workspace);
        }
        snakes.push(snake);
    }
    drop(workspace);
    let ledger = Arc::new(MoveLedger::new(o.move_log.clone()));
    let answer_by = (o.answer_by_ms > 0).then(|| Duration::from_millis(o.answer_by_ms));
    let app = match App::new(snakes, Arc::clone(&ledger), o.threads, answer_by) {
        Ok(a) => Arc::new(a.with_margin(Duration::from_millis(o.margin_ms))),
        Err(e) => {
            eprintln!("slinky-native-server: cannot start the search threads: {e}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match std::net::TcpListener::bind((o.host.as_str(), o.port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "slinky-native-server: cannot listen on {}:{}: {e}",
                o.host, o.port
            );
            return ExitCode::FAILURE;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("slinky-native-server: cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    for spec in &o.snakes {
        info!(
            "serving {} on http://{}:{}/{} ({} search threads)",
            spec.agent,
            o.host,
            o.port,
            spec.name,
            app.threads()
        );
    }
    let res = runtime.block_on(serve(listener, app, shutdown_signal()));
    ledger.flush();
    match res {
        Ok(()) => {
            info!("stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("slinky-native-server: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_the_command_line() {
        let o = parse_args(args(&[
            "--snake",
            "az-256=/r:simulations=256",
            "--port=8123",
            "--author",
            "nathanWolo",
            "--color",
            "#ef4444",
            "--agent",
            "b=az_modes:/r:budget_ms=300",
            "--move-log",
            "/tmp/m.jsonl",
            "--threads",
            "3",
            "--answer-by-ms",
            "0",
            "--margin-ms",
            "120",
            "--no-benchmark",
        ]))
        .unwrap();
        assert_eq!(o.port, 8123);
        assert_eq!(o.snakes.len(), 2);
        assert_eq!(o.snakes[1].budget_ms, Some(300));
        assert_eq!((o.threads, o.answer_by_ms, o.benchmark), (3, 0, false));
        assert_eq!(o.margin_ms, 120);
        assert_eq!(o.color, "#ef4444");
        assert_eq!(o.host, "127.0.0.1");
        for bad in [
            &["--port", "x", "--snake", "a=/r"][..],
            &["--snake", "a=/r", "--snake", "a=/s"],
            &[],
            &["--snake"],
            &["--bogus"],
            &["--snake", "a=/r", "--margin-ms", "99999"],
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(parse_args(args(&["--help"])).err().as_deref(), Some(""));
    }
}
