//! Move quality of longer and root-parallel searches against a long reference search, and how
//! much of a tree the next move could reuse. A proxy for strength that needs no games.
//!
//! ```text
//! cargo run --release -p slinky-search --example quality -- \
//!     [RUN_DIR] [CASES_PER_CONFIG] [REFERENCE_SIMS] [WORKER_THREADS]
//! ```
//!
//! For each position (the agent golden cases, see `examples/common`) a reference search of
//! `REFERENCE_SIMS` simulations (default 16384, its own noise seed) gives seat 0's mean value
//! per root move. Each method then picks a move; the report gives how often it picks the
//! reference's move and its mean regret, `q_ref(reference move) - q_ref(chosen move)` in value
//! units (the value scale is -1 to 1). Methods: single trees of 256 to 4096 simulations and
//! root-parallel searches of `k` trees (`k x n`: `n` simulations each, the wall time of `n`).
//! Reuse: the share of a 4096-simulation search that went through the child the next move
//! would start from if every seat played the searched move (an upper bound for tree reuse:
//! real opponents and food spawns make it smaller). Positions are spread over
//! `WORKER_THREADS` threads (default 4); every search is single-threaded except the
//! root-parallel ones (deterministic, so the thread timing does not change their result).

#[path = "common/mod.rs"]
mod common;

use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, NUM_ACTIONS};
use slinky_search::{
    Budget, JaxNoise, NetEvaluator, Search, SearchConfig, SearchOutput, run_root_parallel,
    thread_noise,
};
use std::sync::Mutex;

/// (label, trees, simulations per tree)
const METHODS: [(&str, usize, usize); 9] = [
    ("1x256", 1, 256),
    ("1x512", 1, 512),
    ("1x1024", 1, 1024),
    ("1x2048", 1, 2048),
    ("1x4096", 1, 4096),
    ("2x512", 2, 512),
    ("4x256", 4, 256),
    ("2x2048", 2, 2048),
    ("4x1024", 4, 1024),
];

struct Row {
    config: String,
    agree: [bool; METHODS.len()],
    regret: [f64; METHODS.len()],
    reuse: f64,
}

#[allow(clippy::too_many_arguments)]
fn search(
    net: &Network,
    settings: &SearchConfig,
    cfg: &GameConfig,
    s: &GameState,
    trees: usize,
    sims: usize,
    seed: u32,
    ws: &mut Search,
) -> SearchOutput {
    if trees == 1 {
        let mut eval = NetEvaluator::for_game(net, *cfg).unwrap();
        return ws
            .run(
                cfg,
                s,
                settings,
                Budget::simulations(sims),
                &mut eval,
                &mut JaxNoise::from_seed(seed),
            )
            .unwrap();
    }
    let mut wss: Vec<Search> = (0..trees).map(|_| Search::new()).collect();
    let mut evals: Vec<NetEvaluator> = (0..trees)
        .map(|_| NetEvaluator::for_game(net, *cfg).unwrap())
        .collect();
    let mut noises: Vec<JaxNoise> = (0..trees).map(|t| thread_noise(seed, t)).collect();
    run_root_parallel(
        &mut wss,
        cfg,
        s,
        settings,
        Budget::simulations(trees * sims),
        &mut evals,
        &mut noises,
    )
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let run = args
        .next()
        .unwrap_or_else(|| "/home/nathan/slinky-serve/az-modes-M-s0".into());
    let per: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(60);
    let ref_sims: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(16384);
    let workers: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(4);
    let net = Network::load(&run)?;
    let settings = SearchConfig::from_net(net.config());
    let by = common::positions(per)?;
    let jobs: Vec<(String, GameConfig, GameState, u32)> = by
        .iter()
        .flat_map(|(name, cases)| {
            cases
                .iter()
                .enumerate()
                .map(|(k, (c, s))| (name.clone(), *c, *s, k as u32))
        })
        .collect();
    let next = Mutex::new(0usize);
    let rows = Mutex::new(Vec::<Row>::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut ws = Search::new();
                loop {
                    let j = {
                        let mut g = next.lock().unwrap();
                        *g += 1;
                        *g - 1
                    };
                    let Some((name, cfg, s, k)) = jobs.get(j) else {
                        break;
                    };
                    let r = search(&net, &settings, cfg, s, 1, ref_sims, 1_000_000 + k, &mut ws);
                    let q = r.q[0];
                    let best = r.action[0] as usize;
                    let mut row = Row {
                        config: name.clone(),
                        agree: [false; METHODS.len()],
                        regret: [0.0; METHODS.len()],
                        reuse: 0.0,
                    };
                    for (i, &(_, trees, sims)) in METHODS.iter().enumerate() {
                        let o = search(&net, &settings, cfg, s, trees, sims, *k, &mut ws);
                        let a = o.action[0] as usize;
                        row.agree[i] = a == best;
                        // An unvisited move in the reference has no estimate: count it as the
                        // worst visited move.
                        let worst = (0..NUM_ACTIONS)
                            .filter(|&m| r.visits[0][m] > 0)
                            .map(|m| q[m])
                            .fold(f32::INFINITY, f32::min);
                        let qa = if r.visits[0][a] > 0 { q[a] } else { worst };
                        row.regret[i] = (q[best] - qa).max(0.0) as f64;
                        if trees == 1 && sims == 4096 {
                            row.reuse = ws.subtree_simulations(&o.action) as f64 / sims as f64;
                        }
                    }
                    rows.lock().unwrap().push(row);
                }
            });
        }
    });
    let rows = rows.into_inner()?;
    let mut names: Vec<String> = by.keys().cloned().collect();
    names.push("all".into());
    print!("{:>12} | moves", "config");
    for (label, _, _) in METHODS {
        print!(" | {label:>7}");
    }
    println!(" | reuse 1x4096");
    for name in &names {
        let sel: Vec<&Row> = rows
            .iter()
            .filter(|r| name == "all" || &r.config == name)
            .collect();
        let m = sel.len() as f64;
        print!("{name:>12} | {:>5}", sel.len());
        for i in 0..METHODS.len() {
            let agree = sel.iter().filter(|r| r.agree[i]).count() as f64 / m;
            print!(" | {:>6.1}%", agree * 100.0);
        }
        let reuse = sel.iter().map(|r| r.reuse).sum::<f64>() / m;
        println!(" | {:.1}%", reuse * 100.0);
        print!("{:>12} | {:>5}", "regret x1000", "");
        for i in 0..METHODS.len() {
            let reg = sel.iter().map(|r| r.regret[i]).sum::<f64>() / m;
            print!(" | {:>7.2}", reg * 1000.0);
        }
        println!();
    }
    Ok(())
}
