//! The two speed-ups that must not change a move: the early stop once a seat's move is decided
//! (`Search::set_stop_when_decided`) and the reuse of the previous tree's evaluations
//! (`Search::set_reuse`).
//!
//! The evaluator is a pure function of the state (peaked priors and values from a hash of it),
//! so the searches are those of a trained network as far as the tree is concerned, and the games
//! are played by the searches themselves, so consecutive positions follow the tree's own lines
//! (with food spawns) as in a real game.

use slinky_env::{EnvConfig, Game};
use slinky_rules::{GameState, MAX_SNAKES, Mode, NUM_ACTIONS};
use slinky_search::{
    Budget, Evaluator, JaxNoise, NodeEval, Prior, Search, SearchConfig, SearchError, SearchOutput,
    StopReason, UniformEvaluator,
};
use std::hash::{Hash, Hasher};

/// Priors and values from a hash of the state: deterministic, peaked (logits up to 6 apart).
#[derive(Default)]
struct HashEval {
    calls: usize,
    keyed: bool,
}

fn mix(state: &GameState, salt: u64) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (state, salt).hash(&mut h);
    h.finish()
}

impl Evaluator for HashEval {
    fn evaluate(
        &mut self,
        _node: usize,
        state: &GameState,
        seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        self.calls += 1;
        let mut logits = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        let mut values = [0.0f32; MAX_SNAKES];
        for p in 0..MAX_SNAKES {
            if !seats[p] {
                continue;
            }
            let h = mix(state, p as u64);
            for (a, l) in logits[p].iter_mut().enumerate() {
                *l = ((h >> (8 * a)) & 0xff) as f32 / 255.0 * 6.0;
            }
            values[p] = ((h >> 40) & 0xffff) as f32 / 65535.0 * 1.6 - 0.8;
        }
        Ok(NodeEval {
            values,
            prior: Prior::Logits(logits),
        })
    }

    fn reuse_key(&self) -> Option<u64> {
        self.keyed.then_some(7)
    }
}

const MODES: [Mode; 4] = [Mode::Standard, Mode::Duel, Mode::Royale, Mode::Constrictor];

fn search(
    ws: &mut Search,
    game: &Game,
    sims: usize,
    seed: u32,
    eval: &mut dyn Evaluator,
) -> SearchOutput {
    ws.run(
        game.config.rules(),
        &game.state,
        &SearchConfig::default(),
        Budget::simulations(sims),
        eval,
        &mut JaxNoise::from_seed(seed),
    )
    .unwrap()
}

/// Plays `turns` turns of a game of `mode` with every seat's move from a full search, calling
/// `check(turn, game)` on each position before the move.
fn play(mode: Mode, seed: u64, turns: usize, mut check: impl FnMut(usize, &Game)) {
    let mut game = Game::new(EnvConfig::for_mode(mode, None).unwrap(), seed);
    let mut ws = Search::new();
    for turn in 0..turns {
        if game.state.done {
            break;
        }
        check(turn, &game);
        let out = search(&mut ws, &game, 64, turn as u32, &mut HashEval::default());
        let actions = out.action;
        game.step(&actions);
    }
}

#[test]
fn early_stop_keeps_the_move_and_saves_simulations() {
    let (mut full_sims, mut early_sims, mut positions) = (0, 0, 0);
    for (m, &mode) in MODES.iter().enumerate() {
        for seed in 0..3u64 {
            play(mode, 100 * m as u64 + seed, 40, |turn, game| {
                let n = game.config.rules().num_snakes();
                let mut full = Search::new();
                let f = search(&mut full, game, 256, turn as u32, &mut HashEval::default());
                for seat in 0..n {
                    let mut ws = Search::new();
                    ws.set_stop_when_decided(Some(seat));
                    let e = search(&mut ws, game, 256, turn as u32, &mut HashEval::default());
                    assert_eq!(
                        e.action[seat], f.action[seat],
                        "{mode:?} seed {seed} turn {turn} seat {seat}"
                    );
                    if e.simulations < 256 {
                        assert_eq!(ws.stats().stop, StopReason::Decided);
                    }
                    full_sims += f.simulations;
                    early_sims += e.simulations;
                    positions += 1;
                }
            });
        }
    }
    eprintln!("early stop: {early_sims} of {full_sims} simulations in {positions} searches");
    assert!(positions > 300);
    assert!(early_sims * 10 < full_sims * 9, "expected savings");
}

#[test]
fn early_stop_with_one_legal_move_runs_no_simulation() {
    // A uniform evaluator never decides by visits; a seat with one legal move is decided at
    // once, and a seat with two never before the leader has more visits than are left.
    let mut seen_forced = false;
    for &mode in &MODES {
        play(mode, 9, 60, |turn, game| {
            let legal = game.state.legal_moves(game.config.rules());
            let count = legal[0].iter().filter(|&&l| l).count();
            let mut ws = Search::new();
            ws.set_stop_when_decided(Some(0));
            let e = search(&mut ws, game, 128, turn as u32, &mut UniformEvaluator);
            let f = search(
                &mut Search::new(),
                game,
                128,
                turn as u32,
                &mut UniformEvaluator,
            );
            assert_eq!(e.action[0], f.action[0]);
            if count <= 1 && !game.state.done {
                seen_forced = true;
                assert_eq!(e.simulations, 0);
            }
        });
    }
    assert!(seen_forced, "no forced move in the games");
}

#[test]
fn reuse_is_bit_identical_and_saves_evaluations() {
    let (mut fresh_calls, mut reuse_calls) = (0, 0);
    for (m, &mode) in MODES.iter().enumerate() {
        for seed in 0..3u64 {
            let mut game = Game::new(
                EnvConfig::for_mode(mode, None).unwrap(),
                50 + 10 * m as u64 + seed,
            );
            let mut kept = Search::new();
            kept.set_reuse(true);
            for turn in 0..60 {
                if game.state.done {
                    break;
                }
                let mut a = HashEval::default();
                let f = search(&mut Search::new(), &game, 256, turn, &mut a);
                let mut b = HashEval {
                    keyed: true,
                    ..HashEval::default()
                };
                let r = search(&mut kept, &game, 256, turn, &mut b);
                assert_eq!(f, r, "{mode:?} seed {seed} turn {turn}");
                // Every node is evaluated or reused, except finished ones (never evaluated).
                let covered = b.calls + kept.reused_nodes();
                assert!(covered >= a.calls && covered <= f.nodes_used);
                fresh_calls += a.calls;
                reuse_calls += b.calls;
                game.step(&f.action);
            }
        }
    }
    eprintln!("reuse: {reuse_calls} of {fresh_calls} evaluations");
    assert!(reuse_calls * 10 < fresh_calls * 8, "expected savings");
}

#[test]
fn reuse_needs_a_key_and_the_same_settings() {
    let game = Game::new(EnvConfig::for_mode(Mode::Duel, None).unwrap(), 3);
    let mut ws = Search::new();
    ws.set_reuse(true);
    // No key: nothing is reused, even for the same position.
    for _ in 0..2 {
        search(&mut ws, &game, 64, 0, &mut HashEval::default());
        assert_eq!(ws.reused_nodes(), 0);
    }
    // With a key the same position reuses its root (and more).
    let keyed = || HashEval {
        keyed: true,
        ..HashEval::default()
    };
    search(&mut ws, &game, 64, 0, &mut keyed());
    search(&mut ws, &game, 64, 1, &mut keyed());
    assert!(ws.reused_nodes() > 1);
    // Other settings: no reuse.
    let other = SearchConfig {
        c_puct: 1.0,
        ..SearchConfig::default()
    };
    let mut e = keyed();
    ws.run(
        game.config.rules(),
        &game.state,
        &other,
        Budget::simulations(64),
        &mut e,
        &mut JaxNoise::from_seed(0),
    )
    .unwrap();
    assert_eq!(ws.reused_nodes(), 0);
    // Turning it off drops the kept tree.
    ws.set_reuse(false);
    search(&mut ws, &game, 64, 0, &mut keyed());
    ws.set_reuse(true);
    search(&mut ws, &game, 64, 0, &mut keyed());
    assert_eq!(ws.reused_nodes(), 0);
}

#[test]
fn early_stop_and_reuse_together() {
    for (m, &mode) in MODES.iter().enumerate() {
        let mut game = Game::new(EnvConfig::for_mode(mode, None).unwrap(), 77 + m as u64);
        let mut kept = Search::new();
        kept.set_reuse(true);
        kept.set_stop_when_decided(Some(0));
        for turn in 0..60 {
            if game.state.done {
                break;
            }
            let f = search(
                &mut Search::new(),
                &game,
                256,
                turn,
                &mut HashEval::default(),
            );
            let mut b = HashEval {
                keyed: true,
                ..HashEval::default()
            };
            let r = search(&mut kept, &game, 256, turn, &mut b);
            assert_eq!(f.action[0], r.action[0], "{mode:?} turn {turn}");
            game.step(&f.action);
        }
    }
}
