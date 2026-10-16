//! Time-budget and long searches on one core: simulations reached, time used, deadline
//! overshoot and tree memory per move.
//!
//! ```text
//! taskset -c 7 cargo run --release -p slinky-search --example budget -- \
//!     [RUN_DIR] [CASES_PER_CONFIG] [BUDGETS_MS] [SIMS]
//! ```
//!
//! `BUDGETS_MS` is a comma list (default `50,100,200,300,400`), `SIMS` a comma list of
//! deterministic simulation counts (default `256,1024,4096`; `-` for none). Each move is
//! searched in a fresh workspace, as [`slinky_search::Agent`] does, with the deadline measured
//! from just before the evaluator is built (the request's arrival, less the ~0.1 ms of request
//! parsing). Positions are the agent golden cases (see `examples/common`).

#[path = "common/mod.rs"]
mod common;

use slinky_net::Network;
use slinky_search::{Budget, JaxNoise, MAX_SIMULATIONS, NetEvaluator, Search, SearchConfig};
use std::time::Instant;

fn list(arg: Option<String>, default: &str) -> Vec<u64> {
    let s = arg.unwrap_or_else(|| default.into());
    if s == "-" {
        return Vec::new();
    }
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let run = args
        .next()
        .unwrap_or_else(|| "/home/nathan/slinky-serve/az-modes-M-s0".into());
    let per: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(40);
    let budgets = list(args.next(), "50,100,200,300,400");
    let sims_list = list(args.next(), "256,1024,4096");
    let net = Network::load(&run)?;
    let settings = SearchConfig::from_net(net.config());
    let by = common::positions(per)?;
    println!(
        "config | budget | moves | sims p50 / min / max (mean) | time p50 / p100 ms | over budget max ms \
         | nodes mean / max | tree MB max | allocated MB max"
    );
    // Warm up the code and the allocator.
    if let Some((_, cases)) = by.iter().next() {
        for (k, (cfg, s)) in cases.iter().take(5).enumerate() {
            let mut eval = NetEvaluator::for_game(&net, *cfg)?;
            Search::new().run(
                cfg,
                s,
                &settings,
                Budget::simulations(256),
                &mut eval,
                &mut JaxNoise::from_seed(k as u32),
            )?;
        }
    }
    let mut runs: Vec<(String, Option<u64>, usize)> = Vec::new();
    for &ms in &budgets {
        runs.push((format!("{ms} ms"), Some(ms), MAX_SIMULATIONS));
    }
    for &n in &sims_list {
        runs.push((format!("{n} sims"), None, n as usize));
    }
    for (name, cases) in &by {
        for (label, ms, cap) in &runs {
            let (mut sims, mut times, mut nodes) = (Vec::new(), Vec::new(), Vec::new());
            let (mut tree, mut alloc, mut over) = (0usize, 0usize, f64::NEG_INFINITY);
            for (k, (cfg, s)) in cases.iter().enumerate() {
                let arrival = Instant::now();
                let budget = match ms {
                    Some(ms) => Budget::millis_after(arrival, *ms, *cap),
                    None => Budget::simulations(*cap),
                };
                let mut ws = Search::new();
                let mut eval = NetEvaluator::for_game(&net, *cfg)?;
                let out = ws.run(
                    cfg,
                    s,
                    &settings,
                    budget,
                    &mut eval,
                    &mut JaxNoise::from_seed(k as u32),
                )?;
                let dt = arrival.elapsed().as_secs_f64() * 1e3;
                let st = ws.stats();
                sims.push(out.simulations as f64);
                times.push(dt);
                nodes.push(st.nodes as f64);
                tree = tree.max(st.tree_bytes);
                alloc = alloc.max(st.allocated_bytes);
                if let Some(ms) = ms {
                    over = over.max(dt - *ms as f64);
                }
            }
            let m = sims.len() as f64;
            let smean = sims.iter().sum::<f64>() / m;
            let nmean = nodes.iter().sum::<f64>() / m;
            println!(
                "{name:>12} | {label:>9} | {} | {:.0} / {:.0} / {:.0} ({:.0}) | {:.1} / {:.1} | {} | \
                 {:.0} / {:.0} | {:.1} | {:.1}",
                sims.len(),
                common::quantile(&mut sims, 0.5),
                common::quantile(&mut sims, 0.0),
                common::quantile(&mut sims, 1.0),
                smean,
                common::quantile(&mut times, 0.5),
                common::quantile(&mut times, 1.0),
                if over.is_finite() {
                    format!("{over:+.2}")
                } else {
                    "-".into()
                },
                nmean,
                common::quantile(&mut nodes, 1.0),
                tree as f64 / 1048576.0,
                alloc as f64 / 1048576.0,
            );
        }
    }
    Ok(())
}
