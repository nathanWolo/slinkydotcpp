//! Time budgets and long searches with the real network (the deployed M run; the tests print
//! SKIP and pass when it is missing). No golden data needed: positions come from random play.
//!
//! * `deadline_respected_all_modes`: a 40 ms budget in duel, standard, royale and constrictor
//!   (4 seats) ends within a small margin of the deadline (the strict p100 measurement on one
//!   pinned core is `examples/budget.rs`; this test allows for a busy test machine).
//! * `agent_time_budget_from_arrival`: `Agent::respond_from` with a time budget answers within
//!   the budget measured from the given arrival, which can lie in the past.
//! * `long_simulation_budgets_are_deterministic`: 1024 and 4096 simulations through the agent
//!   give the same search twice, and a tree far beyond 256 simulations stays consistent.
//!
//! Run in release: `cargo test --release -p slinky-search --test budget -- --nocapture`.

use slinky_api::Game;
use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, Mode, NUM_ACTIONS};
use slinky_search::{Agent, Budget, JaxNoise, MAX_DEPTH, MoveBudget, Search, StopReason};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn run_dir() -> PathBuf {
    std::env::var("NAZ_RUN")
        .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into())
        .into()
}

fn load_agent() -> Option<Agent> {
    match Network::load(run_dir()) {
        Ok(n) => Some(Agent::from_network(n)),
        Err(e) => {
            eprintln!("SKIP: cannot load {}: {e}", run_dir().display());
            None
        }
    }
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

/// Mid-game positions of `mode` (4 seats, 2 in a duel) from random legal play.
fn positions(mode: Mode, count: usize, seed: u64) -> Vec<Game> {
    let config = GameConfig::for_mode(mode, None).unwrap();
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
        let turns = 8 + (rng.next() % 20) as usize;
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
                mode,
            });
        }
    }
    out
}

#[test]
fn deadline_respected_all_modes() {
    let Some(agent) = load_agent() else {
        return;
    };
    let budget_ms = 40u64;
    let margin = Duration::from_millis(15);
    for mode in [Mode::Duel, Mode::Standard, Mode::Royale, Mode::Constrictor] {
        let mut worst = Duration::ZERO;
        let mut sims = Vec::new();
        for (k, game) in positions(mode, 5, 11 + mode as u64).iter().enumerate() {
            let arrival = Instant::now();
            let mut ws = Search::new();
            let m = agent
                .search_game(
                    &mut ws,
                    game,
                    Budget::millis_after(arrival, budget_ms, usize::MAX),
                    &mut JaxNoise::from_seed(k as u32),
                )
                .unwrap();
            let used = arrival.elapsed();
            worst = worst.max(used);
            sims.push(m.search.simulations);
            let st = ws.stats();
            assert_eq!(st.simulations, m.search.simulations);
            assert_eq!(st.stop, StopReason::Deadline, "{mode:?} case {k}");
            assert!(st.depth <= MAX_DEPTH);
            assert_eq!(
                st.tree_bytes,
                st.nodes * slinky_search::node_bytes(game.config.num_snakes())
            );
            let total: i32 = m.search.visits[0].iter().sum();
            assert_eq!(total as usize, m.search.simulations);
            assert!(
                used <= Duration::from_millis(budget_ms) + margin,
                "{mode:?} case {k}: {used:?} for a {budget_ms} ms budget"
            );
        }
        eprintln!(
            "{}: {budget_ms} ms budget, simulations {sims:?}, slowest {:.2} ms",
            mode.name(),
            worst.as_secs_f64() * 1e3
        );
    }
}

const REQUEST: &str = r#"{
  "game": {"id": "g", "ruleset": {"name": "standard", "settings": {}}, "map": "standard", "timeout": 500},
  "turn": 5,
  "board": {"width": 11, "height": 11, "food": [{"x": 5, "y": 5}], "hazards": [],
    "snakes": [
      {"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]},
      {"id": "b", "health": 90, "body": [{"x": 8, "y": 8}, {"x": 8, "y": 7}, {"x": 8, "y": 6}]}
    ]},
  "you": {"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]}
}"#;

#[test]
fn agent_time_budget_from_arrival() {
    let Some(agent) = load_agent() else {
        return;
    };
    let agent = agent.with_budget(MoveBudget::time_ms(60));
    assert_eq!(agent.budget(), MoveBudget::time_ms(60));
    // Arrived 20 ms ago: 40 ms are left.
    let now = Instant::now();
    let arrival = now.checked_sub(Duration::from_millis(20)).unwrap_or(now);
    let (resp, res) = agent.respond_from(REQUEST, Some(2), 0, arrival);
    let used = arrival.elapsed();
    let m = res.unwrap();
    assert_eq!(resp.mv, m.mv.name());
    assert!(m.search.simulations > 1);
    assert!(used <= Duration::from_millis(60 + 15), "{used:?}");
    // Arrived long ago: one simulation still runs and a move comes back at once.
    let now = Instant::now();
    let arrival = now.checked_sub(Duration::from_secs(5)).unwrap_or(now);
    let m = agent.play_from(REQUEST, Some(2), 0, arrival).unwrap();
    assert_eq!(m.search.simulations, 1);
    // Two root-parallel trees share the same deadline.
    let agent = agent.with_threads(2);
    let arrival = Instant::now();
    let m = agent.play_from(REQUEST, Some(2), 0, arrival).unwrap();
    assert!(arrival.elapsed() <= Duration::from_millis(60 + 15));
    assert_eq!(
        m.search.visits[0].iter().sum::<i32>() as usize,
        m.search.simulations
    );
}

#[test]
fn long_simulation_budgets_are_deterministic() {
    if load_agent().is_none() {
        return;
    }
    for sims in [1024usize, 4096] {
        let ag = load_agent().unwrap().with_simulations(sims);
        let x = ag.play(REQUEST, Some(2), 7).unwrap();
        let y = ag.play(REQUEST, Some(2), 7).unwrap();
        assert_eq!(x.search, y.search);
        assert_eq!(x.search.simulations, sims);
        assert_eq!(x.search.visits[0].iter().sum::<i32>() as usize, sims);
        assert!(x.search.depth <= MAX_DEPTH);
        assert!(x.search.nodes_used > 256 && x.search.nodes_used <= sims + 1);
        // Root-parallel with a fixed simulation count is deterministic too.
        let ag = ag.with_threads(4);
        let x = ag.play(REQUEST, Some(2), 7).unwrap();
        let y = ag.play(REQUEST, Some(2), 7).unwrap();
        assert_eq!(x.search, y.search);
        assert_eq!(x.search.simulations, sims);
    }
}
