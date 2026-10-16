//! Speed of the faithful agent search on one core: ms per move at 256 simulations.
//!
//! ```text
//! cargo run --release -p slinky-search --example bench -- [RUN_DIR] [CASES_PER_CONFIG]
//! ```
//!
//! Positions are the agent golden cases (`$NAZ_GOLDEN/search/agent.jsonl`, written by
//! `native/golden/search_golden.py --only agent`), so they are realistic mid-game states. Each
//! config's positions are searched once to warm up and then timed; the report gives the mean,
//! median and 90th percentile search time per move, and the network calls, rows and tree nodes
//! per move. Run it pinned to one core (`taskset -c 2 ...`).

use serde_json::Value;
use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, Ruleset};
use slinky_search::{Budget, JaxNoise, NetEvaluator, Search, SearchConfig};
use std::collections::BTreeMap;
use std::time::Instant;

fn u(v: &Value) -> i64 {
    v.as_i64().unwrap()
}

fn parse(v: &Value) -> (GameConfig, GameState) {
    let g = &v["game"];
    let cfg = GameConfig::new(
        u(&g["width"]) as usize,
        u(&g["height"]) as usize,
        u(&g["num_snakes"]) as usize,
        Ruleset::from_name(g["ruleset"].as_str().unwrap()).unwrap(),
    )
    .unwrap()
    .with_hazard_damage(u(&g["hazard_damage"]) as i32)
    .with_shrink_every(u(&g["shrink_every"]) as i32)
    .unwrap();
    let st = &v["state"];
    let mut s = GameState::new(&cfg);
    s.turn = u(&st["turn"]) as i32;
    s.done = u(&st["done"]) != 0;
    for p in 0..cfg.num_snakes() {
        s.head[p] = [u(&st["head"][p][0]) as i32, u(&st["head"][p][1]) as i32];
        s.length[p] = u(&st["length"][p]) as i32;
        s.health[p] = u(&st["health"][p]) as i32;
        s.alive[p] = u(&st["alive"][p]) != 0;
        s.elim_turn[p] = u(&st["elim_turn"][p]) as i32;
        for (c, x) in st["body"][p].as_array().unwrap().iter().enumerate() {
            s.body[p][c] = u(x) as i16;
        }
    }
    for (k, x) in st["sides"].as_array().unwrap().iter().enumerate() {
        s.sides[k] = u(x) as i8;
    }
    for c in st["food"].as_array().unwrap() {
        s.food |= 1u128 << u(c);
    }
    for (c, x) in st["hazard"].as_array().unwrap().iter().enumerate() {
        s.hazard[c] = u(x) as i16;
    }
    (cfg, s)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let run = args
        .next()
        .unwrap_or_else(|| "/home/nathan/slinky-serve/az-modes-M-s0".into());
    let per: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(100);
    let golden = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let text = std::fs::read_to_string(format!("{golden}/search/agent.jsonl"))?;
    let net = Network::load(&run)?;
    let settings = SearchConfig::from_net(net.config());
    let mut by: BTreeMap<String, Vec<(GameConfig, GameState)>> = BTreeMap::new();
    for line in text.lines() {
        let v: Value = serde_json::from_str(line)?;
        let list = by
            .entry(v["config"].as_str().unwrap().to_string())
            .or_default();
        if list.len() < per {
            list.push(parse(&v));
        }
    }
    let mut search = Search::new();
    for (name, cases) in &by {
        let mut times = Vec::new();
        let (mut calls, mut rows, mut nodes) = (0usize, 0usize, 0usize);
        for warm in [true, false] {
            for (k, (cfg, s)) in cases.iter().enumerate() {
                let mut eval = NetEvaluator::for_game(&net, *cfg)?;
                let t = Instant::now();
                let out = search.run(
                    cfg,
                    s,
                    &settings,
                    Budget::simulations(256),
                    &mut eval,
                    &mut JaxNoise::from_seed(k as u32),
                )?;
                let dt = t.elapsed().as_secs_f64() * 1e3;
                if !warm {
                    times.push(dt);
                    calls += eval.calls;
                    rows += eval.rows;
                    nodes += out.nodes_used;
                }
            }
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let m = times.len() as f64;
        let mean = times.iter().sum::<f64>() / m;
        println!(
            "{name:>13}: {} moves, {:.2} ms/move mean, {:.2} median, {:.2} p90 | {:.0} net calls, \
             {:.0} rows, {:.0} nodes per move | {:.1} us per row",
            times.len(),
            mean,
            sorted[sorted.len() / 2],
            sorted[(sorted.len() * 9) / 10],
            calls as f64 / m,
            rows as f64 / m,
            nodes as f64 / m,
            mean * 1e3 / (rows as f64 / m)
        );
    }
    Ok(())
}
