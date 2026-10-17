//! Adversarial review of the time-budget and long searches (`Budget::millis_after`,
//! `MoveBudget`, the memory limit, `SearchStats`, `run_root_parallel`).
//!
//! * Parity of the long fixed-simulation mode (golden vectors of
//!   `native/golden/review_budget.py`, SKIP when missing): `long_mechanics_bit_exact` and
//!   `long_agent_replay_bit_exact` check that 777 to 4096 simulations are still exactly
//!   `mcts.search(simulations=N)`; `long_agent_network_moves` reports the move agreement with the
//!   Rust network.
//! * Determinism of the fixed-simulation mode: across workspace histories (other seat counts,
//!   memory limits, deadlines, failed runs), across threads sharing one `Agent`, and a far
//!   deadline against the same simulation count.
//! * Deadlines under load in every mode (duel, standard, royale, constrictor, standard3, royale2,
//!   constrictor2), with plain and root-parallel searches; `NAZ_REVIEW_LOAD=1` runs the heavy
//!   version (2x the cores, more rounds).
//! * Memory per move and the memory limit, node counts at large simulation counts.
//! * Confirmed bugs were `#[ignore]`d tests that name the bug; all are fixed now.
//!
//! Run in release: `cargo test --release -p slinky-search --test review_budget -- --nocapture`
//! (`--include-ignored` to see the bugs).

use serde_json::Value;
use slinky_api::{Game, Request, game_from_request};
use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, Mode, NUM_ACTIONS, Ruleset};
use slinky_search::{
    Agent, Budget, Evaluator, JaxNoise, MAX_DEPTH, MoveBudget, NetEvaluator, NodeEval, NoiseTable,
    Prior, Search, SearchConfig, SearchError, SearchOutput, StopReason, UniformEvaluator,
    node_bytes, run_root_parallel, thread_noise,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

// --- Shared helpers ----------------------------------------------------------------------------

fn golden_dir() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("search")
}

fn run_dir() -> PathBuf {
    std::env::var("NAZ_RUN")
        .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into())
        .into()
}

fn load_net() -> Option<Network> {
    match Network::load(run_dir()) {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("SKIP: run dir {} not loadable: {e}", run_dir().display());
            None
        }
    }
}

fn heavy() -> bool {
    std::env::var("NAZ_REVIEW_LOAD").is_ok_and(|v| v == "1")
}

fn read_lines(name: &str) -> Option<Vec<Value>> {
    let path = golden_dir().join(name);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP: {} missing (run native/golden/review_budget.py)",
            path.display()
        );
        return None;
    };
    Some(
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("bad golden line"))
            .collect(),
    )
}

fn u(v: &Value) -> u64 {
    v.as_u64().expect("unsigned int")
}

fn i(v: &Value) -> i64 {
    v.as_i64().expect("int")
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().expect("list").iter().map(i).collect()
}

fn f32_of(v: &Value) -> f32 {
    f32::from_bits(u(v) as u32)
}

fn bits_list(v: &Value) -> Vec<u32> {
    fn walk(v: &Value, out: &mut Vec<u32>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            x => out.push(u(x) as u32),
        }
    }
    let mut out = Vec::new();
    walk(v, &mut out);
    out
}

fn parse_game(v: &Value) -> GameConfig {
    let rs = Ruleset::from_name(v["ruleset"].as_str().unwrap()).unwrap();
    GameConfig::new(
        u(&v["width"]) as usize,
        u(&v["height"]) as usize,
        u(&v["num_snakes"]) as usize,
        rs,
    )
    .unwrap()
    .with_hazard_damage(i(&v["hazard_damage"]) as i32)
    .with_shrink_every(i(&v["shrink_every"]) as i32)
    .unwrap()
}

fn parse_state(v: &Value, cfg: &GameConfig) -> GameState {
    let n = cfg.num_snakes();
    let mut s = GameState::new(cfg);
    s.turn = i(&v["turn"]) as i32;
    s.done = i(&v["done"]) != 0;
    for p in 0..n {
        let h = ints(&v["head"][p]);
        s.head[p] = [h[0] as i32, h[1] as i32];
        s.length[p] = i(&v["length"][p]) as i32;
        s.health[p] = i(&v["health"][p]) as i32;
        s.alive[p] = i(&v["alive"][p]) != 0;
        s.last_move[p] = i(&v["last_move"][p]) as u8;
        s.elim_cause[p] = i(&v["elim_cause"][p]) as i8;
        s.elim_turn[p] = i(&v["elim_turn"][p]) as i32;
        for (c, x) in ints(&v["body"][p]).into_iter().enumerate() {
            s.body[p][c] = x as i16;
        }
    }
    for (k, x) in ints(&v["sides"]).into_iter().enumerate() {
        s.sides[k] = x as i8;
    }
    for c in ints(&v["food"]) {
        s.food |= 1u128 << c;
    }
    for (c, x) in ints(&v["hazard"]).into_iter().enumerate() {
        s.hazard[c] = x as i16;
    }
    s
}

fn parse_settings(v: &Value) -> (SearchConfig, usize) {
    let cfg = SearchConfig {
        c_puct: f32_of(&v["c_puct"]),
        fpu_reduction: f32_of(&v["fpu"]),
        draw_value: f32_of(&v["draw"]),
        ..SearchConfig::default()
    };
    (cfg, u(&v["sims"]) as usize)
}

/// Every output field against a golden `out`, bit for bit; the names of the fields that differ.
fn diff(out: &SearchOutput, gold: &Value, n: usize) -> Vec<&'static str> {
    let flat = |a: &[[f32; NUM_ACTIONS]; MAX_SNAKES]| -> Vec<u32> {
        a[..n].iter().flatten().map(|x| x.to_bits()).collect()
    };
    let visits: Vec<i64> = out.visits[..n]
        .iter()
        .flatten()
        .map(|&v| v as i64)
        .collect();
    let gv: Vec<i64> = gold["visits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(ints)
        .collect();
    let value: Vec<u32> = out.value[..n].iter().map(|x| x.to_bits()).collect();
    let action: Vec<i64> = out.action[..n].iter().map(|&a| a as i64).collect();
    let mut bad = Vec::new();
    if visits != gv {
        bad.push("visits");
    }
    if flat(&out.q) != bits_list(&gold["q"]) {
        bad.push("q");
    }
    if flat(&out.policy) != bits_list(&gold["policy"]) {
        bad.push("policy");
    }
    if value != bits_list(&gold["value"]) {
        bad.push("value");
    }
    if flat(&out.root_prior) != bits_list(&gold["root_prior"]) {
        bad.push("root_prior");
    }
    if action != ints(&gold["action"]) {
        bad.push("action");
    }
    if out.nodes_used as u64 != u(&gold["nodes_used"]) {
        bad.push("nodes_used");
    }
    if out.depth as u64 != u(&gold["depth"]) {
        bad.push("depth");
    }
    bad
}

/// xorshift64*, enough for picking random legal moves.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// The seven ladder configs: (name, mode, seats).
const CONFIGS: [(&str, Mode, Option<usize>); 7] = [
    ("duel", Mode::Duel, None),
    ("standard", Mode::Standard, None),
    ("royale", Mode::Royale, None),
    ("constrictor", Mode::Constrictor, None),
    ("standard3", Mode::Standard, Some(3)),
    ("royale2", Mode::Royale, Some(2)),
    ("constrictor2", Mode::Constrictor, Some(2)),
];

/// Mid-game positions of a config from random legal play (never finished).
fn positions(mode: Mode, seats: Option<usize>, count: usize, seed: u64) -> Vec<Game> {
    let config = GameConfig::for_mode(mode, seats).unwrap();
    let n = config.num_snakes();
    let starts = [(1, 1), (9, 9), (1, 9), (9, 1)];
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    while out.len() < count {
        let mut s = GameState::new(&config);
        for (p, &(x, y)) in starts.iter().enumerate().take(n) {
            s.set_snake(&config, p, &[(x, y), (x, y), (x, y)], 100, None)
                .unwrap();
        }
        if mode != Mode::Constrictor {
            for c in [config.cell(5, 5), config.cell(2, 2), config.cell(8, 8)] {
                s.food |= 1u128 << c.unwrap();
            }
        }
        s.refresh_done(&config);
        let turns = 6 + (rng.next() % 30) as usize;
        for _ in 0..turns {
            let legal = s.legal_moves(&config);
            let mut actions = [0u8; MAX_SNAKES];
            for p in 0..n {
                let ok: Vec<u8> = (0..NUM_ACTIONS as u8)
                    .filter(|&a| legal[p][a as usize])
                    .collect();
                actions[p] = ok[(rng.next() % ok.len() as u64) as usize];
            }
            let next = s.child(&config, &actions);
            if next.done {
                break;
            }
            s = next;
        }
        if !s.done {
            out.push(Game {
                config,
                state: s,
                mode: config.mode(),
            });
        }
    }
    out
}

fn duel() -> (GameConfig, GameState) {
    let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
    let mut s = GameState::new(&cfg);
    s.set_snake(&cfg, 0, &[(1, 1), (1, 1), (1, 1)], 100, None)
        .unwrap();
    s.set_snake(&cfg, 1, &[(9, 9), (9, 9), (9, 9)], 100, None)
        .unwrap();
    s.refresh_done(&cfg);
    (cfg, s)
}

fn four() -> (GameConfig, GameState) {
    let cfg = GameConfig::for_mode(Mode::Standard, None).unwrap();
    let mut s = GameState::new(&cfg);
    for (p, (x, y)) in [(1, 1), (9, 9), (1, 9), (9, 1)].into_iter().enumerate() {
        s.set_snake(&cfg, p, &[(x, y), (x, y), (x, y)], 100, None)
            .unwrap();
    }
    s.refresh_done(&cfg);
    (cfg, s)
}

fn run_uniform(
    ws: &mut Search,
    (cfg, s): &(GameConfig, GameState),
    budget: Budget,
    seed: u32,
) -> SearchOutput {
    ws.run(
        cfg,
        s,
        &SearchConfig::default(),
        budget,
        &mut UniformEvaluator,
        &mut JaxNoise::from_seed(seed),
    )
    .unwrap()
}

// --- Parity of long fixed-simulation searches --------------------------------------------------

#[test]
fn long_mechanics_bit_exact() {
    let Some(lines) = read_lines("review_long_mech.jsonl") else {
        return;
    };
    let mut ws = Search::new();
    let mut bad = 0;
    for (k, v) in lines.iter().enumerate() {
        let cfg = parse_game(&v["game"]);
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let out = ws
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        let d = diff(&out, &v["out"], cfg.num_snakes());
        if !d.is_empty() {
            bad += 1;
            eprintln!("case {k} ({} {sims}): differs in {d:?}", v["config"]);
        }
    }
    eprintln!("long mechanics: {} cases, {bad} differ", lines.len());
    assert_eq!(bad, 0);
}

/// Node outputs replayed from Python's tree, in creation order.
struct Replay<'a> {
    recs: &'a [f32],
    n: usize,
}

impl Evaluator for Replay<'_> {
    fn evaluate(
        &mut self,
        node: usize,
        _state: &GameState,
        _seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        let n = self.n;
        let rec = self
            .recs
            .get(node * n * 5..(node + 1) * n * 5)
            .ok_or(SearchError::Evaluator(format!("no record for node {node}")))?;
        let mut values = [0.0; MAX_SNAKES];
        let mut probs = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        for p in 0..n {
            values[p] = rec[p];
            probs[p].copy_from_slice(&rec[n + 4 * p..n + 4 * p + 4]);
        }
        Ok(NodeEval {
            values,
            prior: Prior::Probs(probs),
        })
    }
}

fn long_agent_cases() -> Option<(Vec<Value>, Vec<f32>)> {
    let lines = read_lines("review_long_agent.jsonl")?;
    let path = golden_dir().join("review_long_agent_nodes.bin");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("SKIP: {} missing", path.display());
        return None;
    };
    let recs = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Some((lines, recs))
}

#[test]
fn long_agent_replay_bit_exact() {
    let Some((lines, recs)) = long_agent_cases() else {
        return;
    };
    let mut ws = Search::new();
    let mut bad = 0;
    for (k, v) in lines.iter().enumerate() {
        let cfg = parse_game(&v["game"]);
        let n = cfg.num_snakes();
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let off = u(&v["node_offset"]) as usize / 4;
        let len = u(&v["num_nodes"]) as usize * n * 5;
        let mut replay = Replay {
            recs: &recs[off..off + len],
            n,
        };
        let out = ws
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut replay,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        let d = diff(&out, &v["out"], n);
        if !d.is_empty() {
            bad += 1;
            eprintln!("case {k} ({} {sims}): differs in {d:?}", v["config"]);
        }
    }
    eprintln!("long agent replay: {} cases, {bad} differ", lines.len());
    assert_eq!(bad, 0);
}

#[test]
fn long_agent_network_moves() {
    let Some((lines, _)) = long_agent_cases() else {
        return;
    };
    let Some(net) = load_net() else {
        return;
    };
    let mut ws = Search::new();
    let (mut agree, mut visits_eq) = (0, 0);
    for v in &lines {
        let cfg = parse_game(&v["game"]);
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let mut ev = NetEvaluator::for_game(&net, cfg).unwrap();
        let out = ws
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut ev,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        if out.action[0] as i64 == ints(&v["out"]["action"])[0] {
            agree += 1;
        }
        let gv: Vec<i64> = ints(&v["out"]["visits"][0]);
        if out.visits[0].iter().map(|&x| x as i64).collect::<Vec<_>>() == gv {
            visits_eq += 1;
        }
    }
    eprintln!(
        "long agent with the Rust network: {} cases, seat-0 move agrees {agree}, seat-0 visits equal {visits_eq}",
        lines.len()
    );
    // Long searches magnify 1e-6 network differences less than 256 does, not more: the move
    // must agree in nearly every case.
    assert!(agree * 100 >= lines.len() * 95, "{agree}/{}", lines.len());
}

// --- Determinism of the fixed-simulation mode --------------------------------------------------

#[test]
fn fixed_simulations_independent_of_workspace_history() {
    let d = duel();
    let f = four();
    let want_d = run_uniform(&mut Search::new(), &d, Budget::simulations(3000), 21);
    let want_f = run_uniform(&mut Search::new(), &f, Budget::simulations(1500), 22);
    let mut ws = Search::new();
    // A big 4-seat tree, a duel deadline run, a memory-limited run, a failed run.
    let _ = run_uniform(&mut ws, &f, Budget::simulations(4000), 1);
    assert_eq!(
        run_uniform(&mut ws, &d, Budget::simulations(3000), 21),
        want_d
    );
    let _ = run_uniform(
        &mut ws,
        &d,
        Budget::millis_after(Instant::now(), 3, usize::MAX),
        2,
    );
    assert_eq!(
        run_uniform(&mut ws, &f, Budget::simulations(1500), 22),
        want_f
    );
    ws.set_memory_limit(node_bytes(4) * 10);
    let _ = run_uniform(&mut ws, &f, Budget::simulations(1500), 22);
    assert_eq!(ws.stats().stop, StopReason::MemoryLimit);
    ws.set_memory_limit(slinky_search::DEFAULT_MEMORY_LIMIT);
    let err = ws.run(
        &f.0,
        &f.1,
        &SearchConfig::default(),
        Budget::simulations(50),
        &mut UniformEvaluator,
        &mut NoiseTable::default(),
    );
    assert!(err.is_err());
    assert_eq!(
        run_uniform(&mut ws, &d, Budget::simulations(3000), 21),
        want_d
    );
    assert_eq!(
        run_uniform(&mut ws, &f, Budget::simulations(1500), 22),
        want_f
    );
    assert_eq!(ws.stats().stop, StopReason::Simulations);
}

#[test]
fn far_deadline_equals_fixed_simulations() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    for (k, &(name, mode, seats)) in CONFIGS.iter().enumerate() {
        for (j, game) in positions(mode, seats, 2, 500 + k as u64).iter().enumerate() {
            let sims = 600 + 37 * j;
            let a = agent
                .search_game(
                    &mut Search::new(),
                    game,
                    Budget::simulations(sims),
                    &mut JaxNoise::from_seed(j as u32),
                )
                .unwrap();
            let b = agent
                .search_game(
                    &mut Search::new(),
                    game,
                    Budget::millis_after(Instant::now(), 600_000, sims),
                    &mut JaxNoise::from_seed(j as u32),
                )
                .unwrap();
            assert_eq!(a.search, b.search, "{name} case {j}");
        }
    }
}

const REQUEST4: &str = r#"{
  "game": {"id": "g4", "ruleset": {"name": "standard", "settings": {}}, "map": "standard", "timeout": 500},
  "turn": 12,
  "board": {"width": 11, "height": 11, "food": [{"x": 5, "y": 5}, {"x": 0, "y": 10}], "hazards": [],
    "snakes": [
      {"id": "a", "health": 88, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]},
      {"id": "b", "health": 90, "body": [{"x": 8, "y": 8}, {"x": 8, "y": 7}, {"x": 8, "y": 6}]},
      {"id": "c", "health": 70, "body": [{"x": 2, "y": 8}, {"x": 3, "y": 8}, {"x": 4, "y": 8}, {"x": 4, "y": 7}]},
      {"id": "d", "health": 95, "body": [{"x": 8, "y": 2}, {"x": 7, "y": 2}, {"x": 6, "y": 2}]}
    ]},
  "you": {"id": "a", "health": 88, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]}
}"#;

#[test]
fn fixed_simulations_identical_across_threads() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Arc::new(Agent::from_network(net).with_simulations(1024));
    fn assert_sync<T: Send + Sync>() {}
    assert_sync::<Agent>();
    let want = agent.play(REQUEST4, Some(4), 5).unwrap().search;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8);
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let agent = Arc::clone(&agent);
            std::thread::spawn(move || {
                (0..3)
                    .map(|_| agent.play(REQUEST4, Some(4), 5).unwrap().search)
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for got in h.join().unwrap() {
            assert_eq!(got, want);
        }
    }
    // Root-parallel with a fixed count: deterministic across calls and threads too.
    let par = Arc::new(
        Agent::from_network(load_net().unwrap())
            .with_simulations(1024)
            .with_threads(3),
    );
    let want = par.play(REQUEST4, Some(4), 5).unwrap().search;
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let par = Arc::clone(&par);
            std::thread::spawn(move || par.play(REQUEST4, Some(4), 5).unwrap().search)
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), want);
    }
}

// --- Deadlines under load ------------------------------------------------------------------------

/// Every config searched by `threads` concurrent workers, each move with a `budget_ms` time
/// budget from its arrival; returns the worst overrun (answer time - budget) per config, the
/// tree free included (the agent drops its workspace before returning).
fn overruns_under_load(
    agent: &Agent,
    threads: usize,
    rounds: usize,
    budget_ms: u64,
) -> Vec<(String, Duration, usize)> {
    let games: Vec<(String, Vec<Game>)> = CONFIGS
        .iter()
        .enumerate()
        .map(|(k, &(name, mode, seats))| {
            (
                name.to_string(),
                positions(mode, seats, rounds, 900 + k as u64),
            )
        })
        .collect();
    let games = &games;
    let mut worst: Vec<(String, Duration, usize)> = games
        .iter()
        .map(|(n, _)| (n.clone(), Duration::ZERO, 0))
        .collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut w = vec![(Duration::ZERO, 0usize); games.len()];
                    for r in 0..rounds {
                        // Each worker walks the configs in its own order, so all modes overlap.
                        for c in 0..games.len() {
                            let ci = (c + t) % games.len();
                            let game = &games[ci].1[r];
                            let arrival = Instant::now();
                            let budget = Budget::millis_after(arrival, budget_ms, usize::MAX);
                            let m = {
                                let mut ws = Search::new();
                                agent
                                    .search_game(
                                        &mut ws,
                                        game,
                                        budget,
                                        &mut JaxNoise::from_seed((t * 1000 + r) as u32),
                                    )
                                    .unwrap()
                            };
                            let over = arrival
                                .elapsed()
                                .saturating_sub(Duration::from_millis(budget_ms));
                            assert!(m.search.simulations >= 1);
                            w[ci].0 = w[ci].0.max(over);
                            w[ci].1 = w[ci].1.max(m.search.simulations);
                        }
                    }
                    w
                })
            })
            .collect();
        for h in handles {
            for (k, (o, s)) in h.join().unwrap().into_iter().enumerate() {
                worst[k].1 = worst[k].1.max(o);
                worst[k].2 = worst[k].2.max(s);
            }
        }
    });
    worst
}

#[test]
fn deadline_overrun_under_load_every_mode() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (threads, rounds, budget_ms) = if heavy() {
        (2 * cores, 6, 300)
    } else {
        (cores.min(4), 1, 40)
    };
    let worst = overruns_under_load(&agent, threads, rounds, budget_ms);
    let mut p100 = Duration::ZERO;
    for (name, o, sims) in &worst {
        eprintln!(
            "{name}: {threads} workers, {budget_ms} ms budget, worst overrun {:.2} ms, max sims {sims}",
            o.as_secs_f64() * 1e3
        );
        p100 = p100.max(*o);
    }
    // Battlesnake gives 500 ms; the ladder budget leaves about 150 ms for the network. An
    // overrun of 25 ms under 2x oversubscription is the most the server can absorb.
    assert!(p100 <= Duration::from_millis(25), "p100 overrun {p100:?}");
}

#[test]
fn root_parallel_deadline_under_load() {
    let Some(net) = load_net() else {
        return;
    };
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let k = if heavy() { 2 * cores } else { 4 };
    let agent = Agent::from_network(net)
        .with_budget(MoveBudget::time_ms(60))
        .with_threads(k);
    let mut worst = Duration::ZERO;
    for _ in 0..if heavy() { 10 } else { 2 } {
        let arrival = Instant::now();
        let m = agent.play_from(REQUEST4, Some(4), 3, arrival).unwrap();
        worst = worst.max(arrival.elapsed().saturating_sub(Duration::from_millis(60)));
        assert_eq!(
            m.search.visits[0].iter().sum::<i32>() as usize,
            m.search.simulations
        );
    }
    eprintln!(
        "root-parallel {k} trees: worst overrun {:.2} ms",
        worst.as_secs_f64() * 1e3
    );
    assert!(worst <= Duration::from_millis(25), "{worst:?}");
}

// --- Memory, node pool and depth at large simulation counts -------------------------------------

#[test]
fn memory_per_move_every_mode() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    let budget_ms = if heavy() { 400 } else { 60 };
    for (k, &(name, mode, seats)) in CONFIGS.iter().enumerate() {
        let game = &positions(mode, seats, 1, 1300 + k as u64)[0];
        let n = game.config.num_snakes();
        let mut ws = Search::new();
        let m = agent
            .search_game(
                &mut ws,
                game,
                Budget::millis_after(Instant::now(), budget_ms, usize::MAX),
                &mut JaxNoise::from_seed(1),
            )
            .unwrap();
        let st = ws.stats();
        eprintln!(
            "{name}: {} sims, {} nodes, tree {:.1} MB, allocated {:.1} MB",
            st.simulations,
            st.nodes,
            st.tree_bytes as f64 / 1048576.0,
            st.allocated_bytes as f64 / 1048576.0
        );
        assert_eq!(st.tree_bytes, st.nodes * node_bytes(n));
        assert!(st.allocated_bytes <= ws.memory_limit());
        assert!(st.nodes <= st.simulations + 1 && st.depth <= MAX_DEPTH);
        assert_eq!(m.search.nodes_used, st.nodes);
    }
}

#[test]
fn memory_limit_bounds_long_fixed_searches() {
    // A fixed budget far beyond the node pool: stops at the limit, never beyond, deterministic.
    let f = four();
    let limit = node_bytes(4) * 3000;
    let mut a = Search::with_memory_limit(limit);
    let x = run_uniform(&mut a, &f, Budget::simulations(1 << 30), 3);
    let st = a.stats();
    assert_eq!(st.stop, StopReason::MemoryLimit);
    assert_eq!(st.nodes, 3000);
    assert!(
        st.allocated_bytes <= limit,
        "{} > {limit}",
        st.allocated_bytes
    );
    assert!(x.simulations <= 1 << 24);
    let y = run_uniform(
        &mut Search::with_memory_limit(limit),
        &f,
        Budget::simulations(1 << 30),
        3,
    );
    assert_eq!(x, y);
    // Depth never exceeds the limit, whatever the count.
    assert!(x.depth <= MAX_DEPTH);
}

#[test]
fn deep_settings_do_not_break_the_tree() {
    // A max_depth beyond 32 (public setting) and many simulations: paths and noise stay sane.
    let d = duel();
    let set = SearchConfig {
        max_depth: 200,
        ..SearchConfig::default()
    };
    let out = Search::new()
        .run(
            &d.0,
            &d.1,
            &set,
            Budget::simulations(20_000),
            &mut UniformEvaluator,
            &mut JaxNoise::from_seed(8),
        )
        .unwrap();
    assert_eq!(out.visits[0].iter().sum::<i32>(), 20_000);
    assert!(out.depth <= 200);
}

// --- Confirmed bugs --------------------------------------------------------------------------------

/// Fixed (`run` now shrinks a reused pool to the limit). Was BUG (memory limit after workspace reuse): `Search::run` clears the node pool but keeps its
/// capacity, and `push` only checks the limit when `nodes.len() == nodes.capacity()`. A
/// workspace reused after a bigger tree (the server's pool and the Lambda reuse one per thread)
/// therefore holds more than `memory_limit` bytes: here a duel tree followed by a 4-seat search
/// under the same limit. `allocated_bytes` exceeds the limit although the docs say the pool
/// "never grows beyond the limit".
#[test]

fn memory_limit_holds_after_workspace_reuse() {
    let limit = node_bytes(2) * 4000;
    let mut ws = Search::with_memory_limit(limit);
    let _ = run_uniform(&mut ws, &duel(), Budget::simulations(1 << 20), 1);
    assert!(ws.stats().allocated_bytes <= limit);
    let _ = run_uniform(&mut ws, &four(), Budget::simulations(1 << 20), 1);
    let st = ws.stats();
    assert_eq!(st.stop, StopReason::MemoryLimit);
    assert!(
        st.allocated_bytes <= limit,
        "allocated {} > limit {limit} after reuse",
        st.allocated_bytes
    );
}

/// Fixed (`push` reserves exactly the node pool's slots). Was BUG (children table grows by doubling past the limit): when the node pool has spare capacity
/// left from an earlier run, `push` never reserves, and `children.resize` grows the children
/// table by Vec's amortised doubling. A reused workspace that first ran a large duel and then a
/// 4-seat search whose deadline estimate reserved few nodes ends with a children table up to
/// twice what the 4-seat tree needs.
#[test]

fn children_table_stays_within_the_limit() {
    let limit = node_bytes(4) * 2000;
    let mut ws = Search::with_memory_limit(limit);
    // A duel tree leaves room for many nodes but a small children table (16 slots per node).
    let _ = run_uniform(&mut ws, &duel(), Budget::simulations(1 << 20), 1);
    // A 4-seat search with a short deadline: the reservation is sized for the deadline, the
    // uniform evaluator expands far faster than the 15 us estimate.
    let _ = run_uniform(
        &mut ws,
        &four(),
        Budget::millis_after(Instant::now(), 1, usize::MAX),
        1,
    );
    let _ = run_uniform(&mut ws, &four(), Budget::simulations(1 << 20), 2);
    let st = ws.stats();
    let children_needed = st.nodes * NUM_ACTIONS.pow(4) * 4;
    assert!(
        st.allocated_bytes <= limit,
        "allocated {} > limit {limit} (children needed {children_needed})",
        st.allocated_bytes
    );
}

/// An evaluator with a constant leaf value and uniform priors.
struct Constant(f32);

impl Evaluator for Constant {
    fn evaluate(
        &mut self,
        _node: usize,
        _state: &GameState,
        _seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        Ok(NodeEval {
            values: [self.0; MAX_SNAKES],
            prior: Prior::Uniform,
        })
    }
}

/// Fixed (f64 sums past `F32_SUM_VISITS` visits per move). Was BUG (f32 value sums in long searches): the per-move value sums are f32, so once a move has
/// about a million visits each added leaf value is rounded to the sum's ulp (0.125 at 2^20). The
/// docs say budgets up to `MAX_SIMULATIONS` (2^24) are exact, but only the visit counts are: with
/// every leaf worth exactly 1/3 the reported mean values drift far from 1/3. Positions near the
/// end of a game reach millions of simulations under a time budget (finished nodes are cheap to
/// revisit), and 4-seat placement values such as 1/3 are not exact in f32. Measured with the M
/// network: random-play 4-seat constrictor endgames run 2.2 to 2.4 million simulations in a
/// 400 ms budget on a tree of 5 to 13 nodes, and report q -0.6596 and 0.3325 for seats whose
/// every line is worth exactly -2/3 and 1/3.
#[test]

fn value_sums_stay_accurate_up_to_max_simulations() {
    let d = duel();
    // Depth 1: every simulation after the 16 root children are expanded only revisits them,
    // as simulations into a finished endgame do.
    let set = SearchConfig {
        max_depth: 1,
        ..SearchConfig::default()
    };
    let v = 1.0f32 / 3.0;
    let out = Search::new()
        .run(
            &d.0,
            &d.1,
            &set,
            Budget::simulations(1 << 23),
            &mut Constant(v),
            &mut JaxNoise::from_seed(1),
        )
        .unwrap();
    for p in 0..2 {
        for a in 0..NUM_ACTIONS {
            if out.visits[p][a] > 0 {
                assert!(
                    (out.q[p][a] - v).abs() < 1e-3,
                    "seat {p} move {a}: q {} after {} visits, every leaf worth {v}",
                    out.q[p][a],
                    out.visits[p][a]
                );
            }
        }
    }
}

/// Fixed (a deadline budget runs at least one). Was BUG (a time budget with a zero cap runs no simulation): `MoveBudget::Time` documents "at
/// least one simulation always runs", but `max_simulations: 0` (or `Budget::until(0, ..)`) gives
/// `sims_cap` 0 and the loop never starts: the move is the first legal move with no search.
#[test]

fn time_budget_always_runs_one_simulation() {
    let out = run_uniform(
        &mut Search::new(),
        &duel(),
        MoveBudget::Time {
            ms: 50,
            max_simulations: 0,
        }
        .budget_from(Instant::now()),
        1,
    );
    assert!(out.simulations >= 1, "{} simulations", out.simulations);
}

/// Fixed (`StopReason::Solved`). Was BUG (perf: a finished root burns the whole time budget): with a time budget and a root whose
/// game is over, every "simulation" is a no-op (`continue`), but the loop still draws noise and
/// runs until the deadline or 2^24 iterations. A request listing only our snake with a /start
/// count of 2 is such a root: the agent answers only when the deadline passes (on Lambda that is
/// billed time). A fixed count of 256 answers at once.
#[test]

fn finished_root_answers_at_once_under_a_time_budget() {
    let (cfg, mut s) = duel();
    s.alive[1] = false;
    s.body[1] = [0; slinky_rules::MAX_CELLS];
    s.refresh_done(&cfg);
    assert!(s.done);
    let t = Instant::now();
    let mut ws = Search::new();
    let out = ws
        .run(
            &cfg,
            &s,
            &SearchConfig::default(),
            Budget::millis_after(t, 300, usize::MAX),
            &mut UniformEvaluator,
            &mut JaxNoise::from_seed(1),
        )
        .unwrap();
    let used = t.elapsed();
    eprintln!(
        "finished root: {} simulations in {:.1} ms",
        out.simulations,
        used.as_secs_f64() * 1e3
    );
    assert!(used < Duration::from_millis(50), "{used:?}");
}

/// The same through the API: a request with only our snake and a /start count of 2.
#[test]

fn lone_snake_request_answers_at_once_under_a_time_budget() {
    let Some(net) = load_net() else {
        return;
    };
    let body = r#"{
      "game": {"id": "g", "ruleset": {"name": "standard", "settings": {}}, "map": "standard", "timeout": 500},
      "turn": 40,
      "board": {"width": 11, "height": 11, "food": [], "hazards": [],
        "snakes": [{"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]}]},
      "you": {"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]}
    }"#;
    let game = game_from_request(&Request::from_json(body).unwrap(), Some(2)).unwrap();
    assert!(game.state.done, "the lone-snake game is over");
    let agent = Agent::from_network(net).with_budget(MoveBudget::time_ms(300));
    let t = Instant::now();
    let m = agent.play(body, Some(2), 0).unwrap();
    let used = t.elapsed();
    eprintln!(
        "lone snake: {} simulations in {:.1} ms",
        m.search.simulations,
        used.as_secs_f64() * 1e3
    );
    assert!(used < Duration::from_millis(50), "{used:?}");
}

// Root-parallel helpers used directly (k trees of one deadline) stay consistent.
#[test]
fn root_parallel_direct_call_sums_are_consistent() {
    let d = duel();
    let set = SearchConfig::default();
    let mut ws: Vec<Search> = (0..5).map(|_| Search::new()).collect();
    let mut evs = [UniformEvaluator; 5];
    let mut noises: Vec<JaxNoise> = (0..5).map(|t| thread_noise(4, t)).collect();
    let out = run_root_parallel(
        &mut ws,
        &d.0,
        &d.1,
        &set,
        Budget::millis_after(Instant::now(), 20, usize::MAX),
        &mut evs,
        &mut noises,
    )
    .unwrap();
    let per: usize = ws.iter().map(|w| w.stats().simulations).sum();
    assert_eq!(per, out.simulations);
    for p in 0..2 {
        assert_eq!(out.visits[p].iter().sum::<i32>() as usize, out.simulations);
    }
}
