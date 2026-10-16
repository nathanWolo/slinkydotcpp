//! `naz`: answer one Battlesnake `/move` request with the native az_modes agent.
//!
//! Reads the request JSON on stdin and prints the `/move` response JSON on stdout. On any error
//! (bad JSON, a game the agent does not play) it prints the safe move, as the server would, and
//! says why on stderr.
//!
//! ```text
//! naz --run DIR [--sims N] [--seed S] [--starting N] [--time-ms T] [-v] < move.json
//! ```
//!
//! * `--run DIR` (or env `NAZ_RUN`): the run dir with `config.json` and `params.npz`.
//! * `--sims N`: simulations (default 256, the ladder agent).
//! * `--seed S`: the tie noise of `jax.random.key(S)` (default 0).
//! * `--starting N`: snakes at the game's start (default: the living snakes in the request).
//! * `--time-ms T`: also stop the search after T ms (not the faithful mode).
//! * `-v`: print the root visits, mean values and timings on stderr.

use slinky_api::{MoveResponse, Request, game_from_request, safe_move_json};
use slinky_search::{Agent, Budget, JaxNoise, Search};
use std::io::Read;
use std::process::ExitCode;
use std::time::{Duration, Instant};

struct Args {
    run: Option<String>,
    sims: usize,
    seed: u32,
    starting: Option<usize>,
    time_ms: Option<u64>,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        run: std::env::var("NAZ_RUN").ok(),
        sims: 256,
        seed: 0,
        starting: None,
        time_ms: None,
        verbose: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match flag.as_str() {
            "--run" => a.run = Some(value("--run")?),
            "--sims" => {
                a.sims = value("--sims")?
                    .parse()
                    .map_err(|e| format!("--sims: {e}"))?
            }
            "--seed" => {
                a.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--starting" => {
                a.starting = Some(
                    value("--starting")?
                        .parse()
                        .map_err(|e| format!("--starting: {e}"))?,
                )
            }
            "--time-ms" => {
                a.time_ms = Some(
                    value("--time-ms")?
                        .parse()
                        .map_err(|e| format!("--time-ms: {e}"))?,
                )
            }
            "-v" | "--verbose" => a.verbose = true,
            "-h" | "--help" => {
                return Err("usage: naz --run DIR [--sims N] [--seed S] [--starting N] \
                            [--time-ms T] [-v] < move.json"
                    .into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let Some(run) = args.run.clone() else {
        eprintln!("naz: give the run dir with --run DIR or NAZ_RUN");
        return ExitCode::from(2);
    };
    let mut body = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut body) {
        eprintln!("naz: reading stdin: {e}");
        return ExitCode::from(2);
    }
    let t_load = Instant::now();
    let agent = match Agent::load(&run) {
        Ok(a) => a.with_simulations(args.sims),
        Err(e) => {
            eprintln!("naz: {e}");
            return ExitCode::from(1);
        }
    };
    let load = t_load.elapsed();
    let result = Request::from_json(&body)
        .map_err(slinky_search::SearchError::from)
        .and_then(|req| {
            let game = game_from_request(&req, args.starting)?;
            let budget = match args.time_ms {
                Some(ms) => Budget::until(args.sims, Instant::now() + Duration::from_millis(ms)),
                None => Budget::simulations(args.sims),
            };
            agent.search_game(
                &mut Search::new(),
                &game,
                budget,
                &mut JaxNoise::from_seed(args.seed),
            )
        });
    let mv = match &result {
        Ok(m) => m.mv,
        Err(e) => {
            eprintln!("naz: {e}; answering the safe move");
            safe_move_json(&body)
        }
    };
    if let (true, Ok(m)) = (args.verbose, &result) {
        let s = &m.search;
        eprintln!(
            "mode {} ({} seats), {} simulations, {} nodes, depth {}, search {:.1} ms, load {:.1} ms",
            m.mode.name(),
            m.num_snakes,
            s.simulations,
            s.nodes_used,
            s.depth,
            m.elapsed.as_secs_f64() * 1e3,
            load.as_secs_f64() * 1e3
        );
        for p in 0..m.num_snakes {
            eprintln!(
                "seat {p}: move {} visits {:?} q {:?} prior {:?}",
                s.action[p], s.visits[p], s.q[p], s.root_prior[p]
            );
        }
    }
    match serde_json::to_string(&MoveResponse::new(mv)) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("naz: {e}");
            ExitCode::from(1)
        }
    }
}
