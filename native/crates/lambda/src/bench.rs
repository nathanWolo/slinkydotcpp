//! A speed benchmark on the function's own hardware, for a direct invocation (not reachable
//! through the Function URL, whose events are always HTTP requests):
//!
//! ```text
//! aws lambda invoke --function-name slinky-native --payload '{"bench": {}}' \
//!     --cli-binary-format raw-in-base64-out out.json
//! ```
//!
//! `{"bench": {"snake": "az-256", "positions": 12, "seed": 1, "modes": ["duel"]}}` (all
//! optional; the first snake and every mode of its network by default). For each mode it makes
//! `positions` mid-game positions, in runs of consecutive turns (self-play with the network's
//! greedy policy and some random moves, fixed by `seed`), then measures:
//!
//! * the search of each position at 256 simulations: plain, with the early stop, and with the
//!   early stop and the reuse of the previous turn's tree; ms per move, simulations, and how
//!   often seat 0's move (the seat a request view plays) equals the plain search's;
//! * the network on every node those searches evaluated: the dense `forward_into`, the
//!   structured `forward_boards_into`, and the structured pass with bfloat16 hidden weights;
//!   microseconds per row (median of interleaved rounds), and the bfloat16 outputs' distance
//!   from the `f32` ones;
//! * the 256-simulation search with the bfloat16 network: seat 0's move against the `f32` one.
//!
//! It stops early, with what it has, 300 ms before the invocation's deadline: give the
//! function a longer timeout for the run (about 20 s at 1769 MB with the defaults).

use serde_json::{Value, json};
use slinky_env::{EnvConfig, EnvRandom, Game, Rng, random_legal};
use slinky_net::{BoardInput, Network};
use slinky_obs::{Encoder, NUM_FEATURES, OBS_LEN, SparseObs};
use slinky_rules::{GameState, MAX_SNAKES, Mode, NUM_ACTIONS};
use slinky_search::{
    Budget, Evaluator, JaxNoise, NetEvaluator, NodeEval, Search, SearchConfig, SearchError,
};
use std::time::{Duration, Instant};

/// Records what it evaluates, answering with the network.
struct Recorder<'a> {
    inner: NetEvaluator<'a>,
    seen: Vec<(GameState, [bool; MAX_SNAKES])>,
}

impl Evaluator for Recorder<'_> {
    fn evaluate(
        &mut self,
        node: usize,
        state: &GameState,
        seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        if self.seen.len() < 20_000 {
            self.seen.push((*state, *seats));
        }
        self.inner.evaluate(node, state, seats)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Positions of `mode`: runs of consecutive turns (a quarter of `count` each, from turn 10,
/// 35, 60 or 85) of self-play games with the network's greedy policy and 20% random moves.
fn positions(net: &Network, mode: Mode, count: usize, seed: u64) -> Vec<Vec<GameState>> {
    let env = EnvConfig::ladder(mode).expect("a ladder mode");
    let rules = *env.rules();
    let enc = Encoder::new(rules, net.config().hide_constrictor_food);
    let mut scratch = net.scratch();
    let mut rng = Rng::new(seed);
    let (mut games, mut total) = (Vec::new(), 0);
    let mut g = 0;
    while total < count && g < 100 {
        let mut game = Game::new(env, seed.wrapping_mul(1000) + g);
        let from = 10 + 25 * (g % 4) as i32;
        let run = count.div_ceil(4).min(count - total);
        g += 1;
        let mut line = Vec::new();
        while !game.state.done && line.len() < run {
            let n = rules.num_snakes();
            let mut obs = vec![0f32; n * OBS_LEN];
            let mut feats = vec![0f32; n * NUM_FEATURES];
            let f = enc.features(&game.state);
            for p in 0..n {
                let _ = enc.observe_into(&game.state, p, &mut obs[p * OBS_LEN..(p + 1) * OBS_LEN]);
                feats[p * NUM_FEATURES..(p + 1) * NUM_FEATURES].copy_from_slice(&f[p]);
            }
            let (mut lg, mut v) = ([[0f32; NUM_ACTIONS]; MAX_SNAKES], [0f32; MAX_SNAKES]);
            let _ = net.forward_into(&mut scratch, &obs, &feats, &mut lg[..n], &mut v[..n]);
            let legal = game.state.legal_moves(&rules);
            let mut actions = random_legal(&mut rng, &rules, &game.state);
            for p in 0..n {
                if rng.below(5) != 0 {
                    let best = (0..NUM_ACTIONS)
                        .filter(|&a| legal[p][a])
                        .max_by(|&a, &b| lg[p][a].total_cmp(&lg[p][b]));
                    if let Some(a) = best {
                        actions[p] = a as u8;
                    }
                }
            }
            if game.state.turn >= from {
                line.push(game.state);
            }
            game.step(&actions);
        }
        total += line.len();
        if !line.is_empty() {
            games.push(line);
        }
    }
    games
}

/// Runs the benchmark with the run dir's network (see the module docs).
pub fn run(run_dir: &std::path::Path, spec: &Value, limit: Option<Instant>) -> Value {
    let t_start = Instant::now();
    let stop_at = limit.map(|l| l.checked_sub(Duration::from_millis(300)).unwrap_or(l));
    let late = || stop_at.is_some_and(|s| Instant::now() >= s);
    let count = spec["positions"].as_u64().unwrap_or(12).clamp(1, 200) as usize;
    let seed = spec["seed"].as_u64().unwrap_or(1);
    let load = |bf16: bool| -> Result<Network, String> {
        let mut n = Network::load(run_dir).map_err(|e| e.to_string())?;
        n.store_hidden_bf16(bf16);
        Ok(n)
    };
    let (net, half) = match (load(false), load(true)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return json!({"error": e}),
    };
    let settings = SearchConfig::from_net(net.config());
    let wanted = |m: Mode| match spec["modes"].as_array() {
        Some(list) => list.iter().any(|x| x.as_str() == Some(m.name())),
        None => true,
    };
    let mut modes = Vec::new();
    for mode in [Mode::Duel, Mode::Standard, Mode::Royale, Mode::Constrictor] {
        if late() {
            break;
        }
        if !wanted(mode) || !net.config().game_modes.iter().any(|m| m == mode.name()) {
            continue;
        }
        let rules = *EnvConfig::ladder(mode).expect("a ladder mode").rules();
        let lines = positions(&net, mode, count, seed);
        let search = |ws: &mut Search, eval: &mut dyn Evaluator, s: &GameState, turn: u32| {
            let t = Instant::now();
            let out = ws
                .run(
                    &rules,
                    s,
                    &settings,
                    Budget::simulations(256),
                    eval,
                    &mut JaxNoise::from_seed(turn),
                )
                .ok();
            (out, t.elapsed().as_secs_f64() * 1e3)
        };
        // Searches: plain, early stop, early stop + reuse, bf16 (recording its evaluations,
        // which are the plain search's but for the bf16 rounding).
        let mut rec = Recorder {
            inner: NetEvaluator::for_game(&half, rules).expect("an 11x11 ladder board"),
            seen: Vec::new(),
        };
        let (mut ms, mut ms_e, mut ms_er, mut sims_e, mut sims_er) = (vec![], vec![], vec![], 0, 0);
        let (mut agree_e, mut agree_er, mut agree_h, mut searched) = (0, 0, 0, 0);
        for line in &lines {
            let mut kept = Search::new();
            kept.set_reuse(true);
            kept.set_stop_when_decided(Some(0));
            for s in line {
                if late() {
                    break;
                }
                let turn = s.turn as u32;
                let mut ev = NetEvaluator::for_game(&net, rules).expect("board");
                let (full, t0) = search(&mut Search::new(), &mut ev, s, turn);
                let mut early = Search::new();
                early.set_stop_when_decided(Some(0));
                let mut ev = NetEvaluator::for_game(&net, rules).expect("board");
                let (e, t1) = search(&mut early, &mut ev, s, turn);
                let mut ev = NetEvaluator::for_game(&net, rules).expect("board");
                let (er, t2) = search(&mut kept, &mut ev, s, turn);
                let (h, _) = search(&mut Search::new(), &mut rec, s, turn);
                let (Some(full), Some(e), Some(er), Some(h)) = (full, e, er, h) else {
                    continue;
                };
                searched += 1;
                ms.push(t0);
                ms_e.push(t1);
                ms_er.push(t2);
                sims_e += e.simulations;
                sims_er += er.simulations;
                agree_e += usize::from(e.action[0] == full.action[0]);
                agree_er += usize::from(er.action[0] == full.action[0]);
                agree_h += usize::from(h.action[0] == full.action[0]);
            }
        }
        // The network on the recorded nodes, three ways, in interleaved rounds.
        let enc = Encoder::new(rules, net.config().hide_constrictor_food);
        let mut obs = vec![0f32; MAX_SNAKES * OBS_LEN];
        let mut feats = [0f32; MAX_SNAKES * NUM_FEATURES];
        let mut sparse: Vec<SparseObs> = (0..MAX_SNAKES).map(|_| SparseObs::default()).collect();
        let mut scratch = net.scratch();
        let rows: usize = rec
            .seen
            .iter()
            .map(|(_, s)| s.iter().filter(|&&x| x).count())
            .sum();
        let (mut t_dense, mut t_struct, mut t_half) = (vec![], vec![], vec![]);
        let (mut max_dl, mut max_dv, mut flips) = (0f32, 0f32, 0usize);
        for round in 0..5 {
            if late() {
                break;
            }
            let (mut a, mut b, mut c) = (0f64, 0f64, 0f64);
            for (state, seats) in &rec.seen {
                let f = enc.features(state);
                let asked: Vec<usize> = (0..MAX_SNAKES).filter(|&p| seats[p]).collect();
                let n = asked.len();
                for (k, &p) in asked.iter().enumerate() {
                    feats[k * NUM_FEATURES..(k + 1) * NUM_FEATURES].copy_from_slice(&f[p]);
                }
                let t = Instant::now();
                for (k, &p) in asked.iter().enumerate() {
                    let _ = enc.observe_into(state, p, &mut obs[k * OBS_LEN..(k + 1) * OBS_LEN]);
                }
                let (mut lg, mut v) = ([[0f32; NUM_ACTIONS]; MAX_SNAKES], [0f32; MAX_SNAKES]);
                let _ = net.forward_into(
                    &mut scratch,
                    &obs[..n * OBS_LEN],
                    &feats[..n * NUM_FEATURES],
                    &mut lg[..n],
                    &mut v[..n],
                );
                let t1 = Instant::now();
                for (k, &p) in asked.iter().enumerate() {
                    let _ = enc.observe_sparse_into(state, p, &mut sparse[k]);
                }
                let boards: Vec<BoardInput> = sparse[..n]
                    .iter()
                    .map(|o| BoardInput {
                        r0: o.r0,
                        c0: o.c0,
                        health: o.health,
                        length: o.length,
                        entries: &o.entries,
                    })
                    .collect();
                let (mut lg2, mut v2) = ([[0f32; NUM_ACTIONS]; MAX_SNAKES], [0f32; MAX_SNAKES]);
                let _ = net.forward_boards_into(
                    &mut scratch,
                    &boards,
                    &feats[..n * NUM_FEATURES],
                    &mut lg2[..n],
                    &mut v2[..n],
                );
                let t2 = Instant::now();
                let (mut lg3, mut v3) = ([[0f32; NUM_ACTIONS]; MAX_SNAKES], [0f32; MAX_SNAKES]);
                let _ = half.forward_boards_into(
                    &mut scratch,
                    &boards,
                    &feats[..n * NUM_FEATURES],
                    &mut lg3[..n],
                    &mut v3[..n],
                );
                let t3 = Instant::now();
                a += (t1 - t).as_secs_f64();
                b += (t2 - t1).as_secs_f64();
                c += (t3 - t2).as_secs_f64();
                if round == 0 {
                    let argmax = |l: &[f32; NUM_ACTIONS]| {
                        (0..NUM_ACTIONS).fold(0, |m, x| if l[x] > l[m] { x } else { m })
                    };
                    for r in 0..n {
                        max_dv = max_dv.max((v3[r] - v[r]).abs());
                        flips += usize::from(argmax(&lg3[r]) != argmax(&lg[r]));
                        for x in 0..NUM_ACTIONS {
                            max_dl = max_dl.max((lg3[r][x] - lg[r][x]).abs());
                        }
                    }
                }
            }
            let per = |s: f64| s * 1e6 / rows.max(1) as f64;
            t_dense.push(per(a));
            t_struct.push(per(b));
            t_half.push(per(c));
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        modes.push(json!({
            "mode": mode.name(),
            "positions": searched,
            "search_ms_per_move": {"plain": mean(&ms), "early_stop": mean(&ms_e), "early_stop_reuse": mean(&ms_er)},
            "simulations_per_move": {"plain": 256, "early_stop": sims_e as f64 / searched.max(1) as f64,
                                     "early_stop_reuse": sims_er as f64 / searched.max(1) as f64},
            "moves_equal_plain": {"early_stop": agree_e, "early_stop_reuse": agree_er, "bf16": agree_h},
            "network_nodes": rec.seen.len(),
            "network_rows": rows,
            "network_us_per_row": {"dense": median(t_dense), "structured": median(t_struct), "structured_bf16": median(t_half)},
            "bf16_error": {"max_abs_logit": max_dl, "max_abs_value": max_dv, "argmax_flips": flips},
        }));
    }
    json!({
        "arch": std::env::consts::ARCH,
        "seconds": t_start.elapsed().as_secs_f64(),
        "cut_short": late(),
        "modes": modes,
    })
}
