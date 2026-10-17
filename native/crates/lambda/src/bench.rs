//! A speed benchmark on the function's own hardware, for a direct invocation (not reachable
//! through the Function URL, whose events are always HTTP requests):
//!
//! ```text
//! aws lambda invoke --function-name slinky-native --payload '{"bench": {}}' \
//!     --cli-binary-format raw-in-base64-out out.json
//! ```
//!
//! `{"bench": {"snake": "az-256", "positions": 12, "seed": 1, "modes": ["duel"], "budget_ms":
//! 200, "turns": 10, "int8_kernel": "scalar"}}` (all optional; the first snake and every mode
//! of its network by default; `int8_kernel` forces the plain Rust int8 kernel instead of the
//! CPU's dot-product one, reported as `int8_kernel`). For each mode it makes
//! `positions` mid-game positions, in runs of consecutive turns (self-play with the network's
//! greedy policy and some random moves, fixed by `seed`), then measures:
//!
//! * the search of each position at 256 simulations: plain, with the early stop, and with the
//!   early stop and the reuse of the previous turn's tree; ms per move, simulations, and how
//!   often seat 0's move (the seat a request view plays) equals the plain search's;
//! * the network on every node those searches evaluated: the dense `forward_into`, the
//!   structured `forward_boards_into`, and the structured pass with bfloat16 hidden weights and
//!   with the int8 network ([`Network::quantize_int8`]); microseconds per row (median of
//!   interleaved rounds), and the bfloat16 and int8 outputs' distance from the `f32` ones;
//! * the 256-simulation search with the bfloat16 and the int8 networks: seat 0's move against
//!   the `f32` one;
//! * the time-budget snake (`budget_ms`, default 200, cap 16,384): the simulations a search
//!   reaches with and without the reuse of the last tree, on `turns` turns (default 10; 0 skips
//!   it) of games where seat 0 plays the searched move, and with the reuse on the int8
//!   network. With a time budget the early stop only ends forced moves (the cap is out of
//!   reach), so these simulations are what speed buys.
//!
//! It stops early, with what it has, 300 ms before the invocation's deadline: give the
//! function a longer timeout for the run (about 50 s on the dev box's Xeon with the defaults;
//! give the Lambda 120 s).

use serde_json::{Value, json};
use slinky_env::{EnvConfig, EnvRandom, Game, Rng, random_legal};
use slinky_net::{BoardInput, Int8Kernel, Network, Scratch};
use slinky_obs::{Encoder, NUM_FEATURES, OBS_LEN, SparseObs};
use slinky_rules::{Actions, GameConfig, GameState, MAX_SNAKES, Mode, NUM_ACTIONS};
use slinky_search::{
    Budget, Evaluator, JaxNoise, NetEvaluator, NodeEval, Search, SearchConfig, SearchError,
};
use std::time::{Duration, Instant};

/// The simulation cap of a time-budget snake (the engine's default with `BUDGET_MS`).
const TIME_CAP: usize = 16_384;

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

/// The scripted players: the network's greedy policy, with a random legal move 20% of the time.
struct Scripted<'a> {
    net: &'a Network,
    rules: GameConfig,
    enc: Encoder,
    scratch: Scratch,
    rng: Rng,
}

impl<'a> Scripted<'a> {
    fn new(net: &'a Network, rules: GameConfig, seed: u64) -> Scripted<'a> {
        Scripted {
            net,
            rules,
            enc: Encoder::new(rules, net.config().hide_constrictor_food),
            scratch: net.scratch(),
            rng: Rng::new(seed),
        }
    }

    fn moves(&mut self, state: &GameState) -> Actions {
        let n = self.rules.num_snakes();
        let mut obs = vec![0f32; n * OBS_LEN];
        let mut feats = vec![0f32; n * NUM_FEATURES];
        let f = self.enc.features(state);
        for p in 0..n {
            let _ = self
                .enc
                .observe_into(state, p, &mut obs[p * OBS_LEN..(p + 1) * OBS_LEN]);
            feats[p * NUM_FEATURES..(p + 1) * NUM_FEATURES].copy_from_slice(&f[p]);
        }
        let (mut lg, mut v) = ([[0f32; NUM_ACTIONS]; MAX_SNAKES], [0f32; MAX_SNAKES]);
        let _ = self
            .net
            .forward_into(&mut self.scratch, &obs, &feats, &mut lg[..n], &mut v[..n]);
        let legal = state.legal_moves(&self.rules);
        let mut actions = random_legal(&mut self.rng, &self.rules, state);
        for p in 0..n {
            if self.rng.below(5) != 0 {
                let best = (0..NUM_ACTIONS)
                    .filter(|&a| legal[p][a])
                    .max_by(|&a, &b| lg[p][a].total_cmp(&lg[p][b]));
                if let Some(a) = best {
                    actions[p] = a as u8;
                }
            }
        }
        actions
    }
}

/// Positions of `mode`: runs of consecutive turns (a quarter of `count` each, from turn 10,
/// 35, 60 or 85) of games between [`Scripted`] players.
fn positions(net: &Network, mode: Mode, count: usize, seed: u64) -> Vec<Vec<GameState>> {
    let env = EnvConfig::ladder(mode).expect("a ladder mode");
    let mut players = Scripted::new(net, *env.rules(), seed);
    let (mut games, mut total) = (Vec::new(), 0);
    let mut g = 0;
    while total < count && g < 100 {
        let mut game = Game::new(env, seed.wrapping_mul(1000) + g);
        let from = 10 + 25 * (g % 4) as i32;
        let run = count.div_ceil(4).min(count - total);
        g += 1;
        let mut line = Vec::new();
        while !game.state.done && line.len() < run {
            let actions = players.moves(&game.state);
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

/// Simulations a `ms` search reaches with and without the reuse of the last tree, and with the
/// reuse on the int8 network, on `turns` turns (from turn 10) of games where seat 0 plays the
/// reusing `f32` search's move and the others are [`Scripted`]. The order of the searches
/// alternates.
fn time_budget(
    net: &Network,
    int8: &Network,
    mode: Mode,
    ms: u64,
    turns: usize,
    seed: u64,
    late: &dyn Fn() -> bool,
) -> Value {
    let env = EnvConfig::ladder(mode).expect("a ladder mode");
    let rules = *env.rules();
    let settings = SearchConfig::from_net(net.config());
    let mut players = Scripted::new(net, rules, seed);
    let mut kept = Search::with_memory_limit(256 << 20);
    kept.set_reuse(true);
    let mut kept8 = Search::with_memory_limit(256 << 20);
    kept8.set_reuse(true);
    let (mut fresh_sims, mut reuse_sims, mut reused, mut done) = (vec![], vec![], 0usize, 0usize);
    let mut int8_sims = vec![];
    let mut g = 0;
    while done < turns && g < 20 && !late() {
        let mut game = Game::new(env, seed.wrapping_mul(1000) + 500 + g);
        g += 1;
        while !game.state.done && game.state.turn < 10 {
            let a = players.moves(&game.state);
            game.step(&a);
        }
        while !game.state.done && done < turns && !late() {
            let noise = game.state.turn as u32;
            let go = |ws: &mut Search, net: &Network| {
                let mut ev = NetEvaluator::for_game(net, rules).expect("an 11x11 ladder board");
                let budget = Budget::millis_after(Instant::now(), ms, TIME_CAP);
                ws.run(
                    &rules,
                    &game.state,
                    &settings,
                    budget,
                    &mut ev,
                    &mut JaxNoise::from_seed(noise),
                )
                .ok()
            };
            let mut fresh = Search::with_memory_limit(256 << 20);
            let (f, r, q) = if done % 2 == 0 {
                let f = go(&mut fresh, net);
                let r = go(&mut kept, net);
                (f, r, go(&mut kept8, int8))
            } else {
                let q = go(&mut kept8, int8);
                let r = go(&mut kept, net);
                (go(&mut fresh, net), r, q)
            };
            let (Some(f), Some(r), Some(q)) = (f, r, q) else {
                break;
            };
            int8_sims.push(q.simulations as f64);
            fresh_sims.push(f.simulations as f64);
            reuse_sims.push(r.simulations as f64);
            reused += kept.reused_nodes();
            done += 1;
            let mut actions = players.moves(&game.state);
            actions[0] = r.action[0];
            game.step(&actions);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({
        "budget_ms": ms,
        "turns": done,
        "simulations_fresh": mean(&fresh_sims),
        "simulations_reuse": mean(&reuse_sims),
        "simulations_reuse_int8": mean(&int8_sims),
        "reused_nodes_per_move": reused as f64 / done.max(1) as f64,
    })
}

/// The network on the recorded nodes: the dense `forward_into` and the structured pass of `net`
/// and of each variant, in 5 interleaved rounds (microseconds per row, median), and each
/// variant's distance from `net`'s dense outputs.
fn network_rounds(
    net: &Network,
    variants: &[(&str, &Network)],
    rules: GameConfig,
    seen: &[(GameState, [bool; MAX_SNAKES])],
    late: &dyn Fn() -> bool,
) -> Value {
    let enc = Encoder::new(rules, net.config().hide_constrictor_food);
    let mut obs = vec![0f32; MAX_SNAKES * OBS_LEN];
    let mut feats = [0f32; MAX_SNAKES * NUM_FEATURES];
    let mut sparse: Vec<SparseObs> = (0..MAX_SNAKES).map(|_| SparseObs::default()).collect();
    let mut scratch = net.scratch();
    let rows: usize = seen
        .iter()
        .map(|(_, s)| s.iter().filter(|&&x| x).count())
        .sum();
    // Per path: seconds of each round; per variant: max and summed |dlogit|, |dvalue|, flips.
    let mut times = vec![vec![]; 2 + variants.len()];
    let mut err = vec![(0f32, 0f64, 0f32, 0f64, 0usize); variants.len()];
    let argmax =
        |l: &[f32; NUM_ACTIONS]| (0..NUM_ACTIONS).fold(0, |m, x| if l[x] > l[m] { x } else { m });
    type Out = ([[f32; NUM_ACTIONS]; MAX_SNAKES], [f32; MAX_SNAKES]);
    for round in 0..5 {
        if late() {
            break;
        }
        let mut t = vec![0f64; 2 + variants.len()];
        for (state, seats) in seen {
            let f = enc.features(state);
            let asked: Vec<usize> = (0..MAX_SNAKES).filter(|&p| seats[p]).collect();
            let n = asked.len();
            for (k, &p) in asked.iter().enumerate() {
                feats[k * NUM_FEATURES..(k + 1) * NUM_FEATURES].copy_from_slice(&f[p]);
            }
            let feats = &feats[..n * NUM_FEATURES];
            let start = Instant::now();
            for (k, &p) in asked.iter().enumerate() {
                let _ = enc.observe_into(state, p, &mut obs[k * OBS_LEN..(k + 1) * OBS_LEN]);
            }
            let mut want: Out = Default::default();
            let _ = net.forward_into(
                &mut scratch,
                &obs[..n * OBS_LEN],
                feats,
                &mut want.0[..n],
                &mut want.1[..n],
            );
            t[0] += start.elapsed().as_secs_f64();
            for (i, v) in std::iter::once(net)
                .chain(variants.iter().map(|v| v.1))
                .enumerate()
            {
                let start = Instant::now();
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
                let mut got: Out = Default::default();
                let _ = v.forward_boards_into(
                    &mut scratch,
                    &boards,
                    feats,
                    &mut got.0[..n],
                    &mut got.1[..n],
                );
                t[1 + i] += start.elapsed().as_secs_f64();
                if round == 0 && i > 0 {
                    let e = &mut err[i - 1];
                    for r in 0..n {
                        let dv = (got.1[r] - want.1[r]).abs();
                        e.2 = e.2.max(dv);
                        e.3 += f64::from(dv);
                        e.4 += usize::from(argmax(&got.0[r]) != argmax(&want.0[r]));
                        for x in 0..NUM_ACTIONS {
                            let dl = (got.0[r][x] - want.0[r][x]).abs();
                            e.0 = e.0.max(dl);
                            e.1 += f64::from(dl) / NUM_ACTIONS as f64;
                        }
                    }
                }
            }
        }
        for (all, s) in times.iter_mut().zip(t) {
            all.push(s * 1e6 / rows.max(1) as f64);
        }
    }
    let mut times = times.into_iter().map(median);
    let mut us = json!({"dense": times.next(), "structured": times.next()});
    let mut errors = json!({});
    for ((name, _), (t, e)) in variants.iter().zip(times.zip(&err)) {
        us[format!("structured_{name}")] = json!(t);
        let r = rows.max(1) as f64;
        errors[*name] = json!({"max_abs_logit": e.0, "mean_abs_logit": e.1 / r, "max_abs_value": e.2,
                               "mean_abs_value": e.3 / r, "argmax_flips": e.4});
    }
    json!({"nodes": seen.len(), "rows": rows, "us_per_row": us, "error_vs_f32": errors})
}

/// Runs the benchmark with the run dir's network (see the module docs).
pub fn run(run_dir: &std::path::Path, spec: &Value, limit: Option<Instant>) -> Value {
    let t_start = Instant::now();
    let stop_at = limit.map(|l| l.checked_sub(Duration::from_millis(300)).unwrap_or(l));
    let late = || stop_at.is_some_and(|s| Instant::now() >= s);
    let count = spec["positions"].as_u64().unwrap_or(12).clamp(1, 200) as usize;
    let seed = spec["seed"].as_u64().unwrap_or(1);
    let budget_ms = spec["budget_ms"].as_u64().unwrap_or(200).clamp(1, 2000);
    let turns = spec["turns"].as_u64().unwrap_or(10).min(100) as usize;
    let load = |kind: &str| -> Result<Network, String> {
        let mut n = Network::load(run_dir).map_err(|e| e.to_string())?;
        match kind {
            "bf16" => n.store_hidden_bf16(true),
            "int8" => {
                n.quantize_int8(true).map_err(|e| e.to_string())?;
                if spec["int8_kernel"].as_str() == Some("scalar") {
                    n.set_int8_kernel(Int8Kernel::Scalar)
                        .map_err(|e| e.to_string())?;
                }
            }
            _ => {}
        }
        Ok(n)
    };
    let (net, half, int8) = match (load("f32"), load("bf16"), load("int8")) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return json!({"error": e}),
    };
    // The approximate networks, timed and compared with `net`.
    let variants = [("bf16", &half), ("int8", &int8)];
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
        // Searches: plain (recording its evaluations), early stop, early stop + reuse, and
        // each approximate network.
        let mut rec = Recorder {
            inner: NetEvaluator::for_game(&net, rules).expect("an 11x11 ladder board"),
            seen: Vec::new(),
        };
        let (mut ms, mut ms_e, mut ms_er, mut sims_e, mut sims_er) = (vec![], vec![], vec![], 0, 0);
        let (mut agree_e, mut agree_er, mut searched) = (0, 0, 0);
        let mut agree_v = [0usize; 2];
        for line in &lines {
            let mut kept = Search::new();
            kept.set_reuse(true);
            kept.set_stop_when_decided(Some(0));
            for s in line {
                if late() {
                    break;
                }
                let turn = s.turn as u32;
                let (full, t0) = search(&mut Search::new(), &mut rec, s, turn);
                let mut early = Search::new();
                early.set_stop_when_decided(Some(0));
                let mut ev = NetEvaluator::for_game(&net, rules).expect("board");
                let (e, t1) = search(&mut early, &mut ev, s, turn);
                let mut ev = NetEvaluator::for_game(&net, rules).expect("board");
                let (er, t2) = search(&mut kept, &mut ev, s, turn);
                let (Some(full), Some(e), Some(er)) = (full, e, er) else {
                    continue;
                };
                for (agree, (_, v)) in agree_v.iter_mut().zip(&variants) {
                    let mut ev = NetEvaluator::for_game(v, rules).expect("board");
                    let (o, _) = search(&mut Search::new(), &mut ev, s, turn);
                    *agree += usize::from(o.is_some_and(|o| o.action[0] == full.action[0]));
                }
                searched += 1;
                ms.push(t0);
                ms_e.push(t1);
                ms_er.push(t2);
                sims_e += e.simulations;
                sims_er += er.simulations;
                agree_e += usize::from(e.action[0] == full.action[0]);
                agree_er += usize::from(er.action[0] == full.action[0]);
            }
        }
        let nets = network_rounds(&net, &variants, rules, &rec.seen, &late);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let timed =
            (turns > 0).then(|| time_budget(&net, &int8, mode, budget_ms, turns, seed, &late));
        let mut agree = json!({"early_stop": agree_e, "early_stop_reuse": agree_er});
        for ((name, _), a) in variants.iter().zip(agree_v) {
            agree[*name] = json!(a);
        }
        modes.push(json!({
            "mode": mode.name(),
            "positions": searched,
            "search_ms_per_move": {"plain": mean(&ms), "early_stop": mean(&ms_e), "early_stop_reuse": mean(&ms_er)},
            "simulations_per_move": {"plain": 256, "early_stop": sims_e as f64 / searched.max(1) as f64,
                                     "early_stop_reuse": sims_er as f64 / searched.max(1) as f64},
            "moves_equal_plain": agree,
            "network": nets,
            "time_budget": timed,
        }));
    }
    json!({
        "arch": std::env::consts::ARCH,
        "int8_kernel": format!("{:?}", int8.int8_kernel()),
        "seconds": t_start.elapsed().as_secs_f64(),
        "cut_short": late(),
        "modes": modes,
    })
}
