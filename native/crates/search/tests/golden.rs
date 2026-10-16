//! Parity with slinky's jitted Python search on golden vectors written by
//! `native/golden/search_golden.py` (see its docstring for the file formats). Skips each part
//! whose files are missing (and the network parts when the run dir is missing).
//!
//! * `noise_matches_jax`: `JaxKey`/`JaxNoise` against `jax.random.key`, `split`, `fold_in`
//!   and the bits `mcts.search` draws.
//! * `mechanics_bit_exact`: uniform priors and zero leaf values (SPEC.md 7.5): root visits,
//!   q, policy, value, priors, moves, node count and depth must be bit-identical.
//! * `agent_replay_bit_exact`: the real agent's search with every node's value and priors
//!   replayed from Python's tree: the same, bit-identical. This is the search with the network
//!   calls answered by Python, so any difference left in `agent_network` is network float noise.
//! * `agent_network`: the real agent with the Rust network (needs the run dir, `NAZ_RUN`,
//!   default /home/nathan/slinky-serve/az-modes-M-s0): move and visit agreement, and for every
//!   disagreement the largest difference between the Rust network and Python's outputs over
//!   Python's tree. `NAZ_AGENT_CASES` limits the number of cases (default all).
//! * `end_to_end_requests`: `Agent::play` on API requests against the Python server path.
//!
//! Run the network parts in release for speed: `cargo test --release -p slinky-search --test
//! golden -- --nocapture`.

use serde_json::Value;
use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, NUM_ACTIONS, Ruleset};
use slinky_search::{
    Agent, Budget, Evaluator, JaxKey, JaxNoise, NetEvaluator, NodeEval, NoiseSource, Prior, Search,
    SearchConfig, SearchError, SearchOutput, UniformEvaluator, masked_softmax,
};
use std::path::PathBuf;
use std::sync::Mutex;

fn golden_dir() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("search")
}

fn run_dir() -> PathBuf {
    std::env::var("NAZ_RUN")
        .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into())
        .into()
}

/// The lines of a golden JSONL file, or `None` (with a message) when it is missing.
fn read_lines(name: &str) -> Option<Vec<Value>> {
    let path = golden_dir().join(name);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP: {} missing (run native/golden/search_golden.py)",
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

fn load_net() -> Option<Network> {
    match Network::load(run_dir()) {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("SKIP: run dir {} not loadable: {e}", run_dir().display());
            None
        }
    }
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

/// A list of f32 bit patterns, flat.
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

/// The fields of one comparison, counted over a test.
#[derive(Default, Debug)]
struct Mismatch {
    visits: usize,
    q: usize,
    policy: usize,
    value: usize,
    prior: usize,
    action: usize,
    action0: usize,
    nodes: usize,
    depth: usize,
}

impl Mismatch {
    fn total(&self) -> usize {
        self.visits
            + self.q
            + self.policy
            + self.value
            + self.prior
            + self.action
            + self.nodes
            + self.depth
    }
}

/// Compare a Rust output with a golden `out`; returns whether everything matched bitwise.
fn compare(out: &SearchOutput, gold: &Value, n: usize, mm: &mut Mismatch) -> bool {
    let mut ok = true;
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
    let mut check = |cond: bool, counter: &mut usize| {
        if !cond {
            *counter += 1;
            ok = false;
        }
    };
    check(visits == gv, &mut mm.visits);
    check(flat(&out.q) == bits_list(&gold["q"]), &mut mm.q);
    check(
        flat(&out.policy) == bits_list(&gold["policy"]),
        &mut mm.policy,
    );
    let value: Vec<u32> = out.value[..n].iter().map(|x| x.to_bits()).collect();
    check(value == bits_list(&gold["value"]), &mut mm.value);
    check(
        flat(&out.root_prior) == bits_list(&gold["root_prior"]),
        &mut mm.prior,
    );
    let action: Vec<i64> = out.action[..n].iter().map(|&a| a as i64).collect();
    check(action == ints(&gold["action"]), &mut mm.action);
    check(action[0] == ints(&gold["action"])[0], &mut mm.action0);
    check(
        out.nodes_used as u64 == u(&gold["nodes_used"]),
        &mut mm.nodes,
    );
    check(out.depth as u64 == u(&gold["depth"]), &mut mm.depth);
    ok
}

#[test]
fn noise_matches_jax() {
    let Some(lines) = read_lines("noise.jsonl") else {
        return;
    };
    for v in &lines {
        let seed = u(&v["seed"]) as u32;
        let key = JaxKey::new(seed);
        assert_eq!(
            key.0.map(u64::from).to_vec(),
            ints(&v["key"])
                .iter()
                .map(|&x| x as u64)
                .collect::<Vec<_>>()
        );
        for (k, sp) in v["split"].as_array().unwrap().iter().enumerate() {
            let want: Vec<u32> = ints(sp).iter().map(|&x| x as u32).collect();
            assert_eq!(key.split_at(k as u64).0.to_vec(), want, "split seed {seed}");
        }
        for f in v["fold"].as_array().unwrap() {
            let f: Vec<u32> = ints(f).iter().map(|&x| x as u32).collect();
            assert_eq!(
                key.fold_in(f[0]).0.to_vec(),
                f[1..].to_vec(),
                "fold_in seed {seed}"
            );
        }
        let n = u(&v["n"]) as usize;
        let sims = u(&v["sims"]) as usize;
        let want = bits_list(&v["bits"]);
        let mut noise = JaxNoise::from_seed(seed);
        let mut got: Vec<u32> = Vec::new();
        for s in 0..sims {
            let b = noise.bits(s, n).unwrap();
            got.extend(b[..n].iter().flatten());
        }
        assert_eq!(got, want, "bits seed {seed} n {n}");
    }
    eprintln!("noise: {} key/bit streams match JAX", lines.len());
}

#[test]
fn mechanics_bit_exact() {
    let Some(lines) = read_lines("mech.jsonl") else {
        return;
    };
    let mut mm = Mismatch::default();
    let mut search = Search::new();
    let mut shown = 0;
    for (k, v) in lines.iter().enumerate() {
        let cfg = parse_game(&v["game"]);
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let out = search
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        if !compare(&out, &v["out"], cfg.num_snakes(), &mut mm) && shown < 5 {
            shown += 1;
            eprintln!(
                "case {k} ({}): rust visits {:?} q {:?} value {:?}\n  gold {}",
                v["config"],
                &out.visits[..cfg.num_snakes()],
                &out.q[..cfg.num_snakes()],
                &out.value[..cfg.num_snakes()],
                v["out"]
            );
        }
    }
    eprintln!("mechanics: {} cases, mismatches {:?}", lines.len(), mm);
    assert_eq!(mm.total(), 0, "{mm:?}");
}

/// Node outputs replayed from Python's tree (`agent_nodes.bin`), optionally checking the Rust
/// network against them on the same states.
struct Replay<'a> {
    recs: &'a [f32],
    n: usize,
    cfg: GameConfig,
    net: Option<NetEvaluator<'a>>,
    max_value_diff: f32,
    max_prior_diff: f32,
}

impl Evaluator for Replay<'_> {
    fn evaluate(
        &mut self,
        node: usize,
        state: &GameState,
        seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        let n = self.n;
        let rec = self
            .recs
            .get(node * n * 5..(node + 1) * n * 5)
            .ok_or(SearchError::Evaluator(format!("no record for node {node}")))?;
        let mut out = NodeEval {
            values: [0.0; MAX_SNAKES],
            prior: Prior::Uniform,
        };
        let mut probs = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        for p in 0..n {
            out.values[p] = rec[p];
            probs[p].copy_from_slice(&rec[n + 4 * p..n + 4 * p + 4]);
        }
        out.prior = Prior::Probs(probs);
        if let Some(net) = &mut self.net {
            let mine = net.evaluate(node, state, seats)?;
            let legal = state.legal_moves(&self.cfg);
            let Prior::Logits(l) = mine.prior else {
                unreachable!()
            };
            for p in (0..n).filter(|&p| seats[p]) {
                let dv = (mine.values[p] - out.values[p]).abs();
                self.max_value_diff = self.max_value_diff.max(dv);
                let pr = masked_softmax(&l[p], &legal[p]);
                for a in 0..NUM_ACTIONS {
                    self.max_prior_diff = self.max_prior_diff.max((pr[a] - probs[p][a]).abs());
                }
            }
        }
        Ok(out)
    }
}

/// The agent cases and their node records.
fn agent_cases() -> Option<(Vec<Value>, Vec<f32>)> {
    let lines = read_lines("agent.jsonl")?;
    let path = golden_dir().join("agent_nodes.bin");
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
    let limit = std::env::var("NAZ_AGENT_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    Some((lines.into_iter().take(limit).collect(), recs))
}

fn case_records<'a>(v: &Value, recs: &'a [f32], n: usize) -> &'a [f32] {
    let off = u(&v["node_offset"]) as usize / 4;
    let len = u(&v["num_nodes"]) as usize * n * 5;
    &recs[off..off + len]
}

#[test]
fn agent_replay_bit_exact() {
    let Some((lines, recs)) = agent_cases() else {
        return;
    };
    let mut mm = Mismatch::default();
    let mut search = Search::new();
    for (k, v) in lines.iter().enumerate() {
        let cfg = parse_game(&v["game"]);
        let n = cfg.num_snakes();
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let mut replay = Replay {
            recs: case_records(v, &recs, n),
            n,
            cfg,
            net: None,
            max_value_diff: 0.0,
            max_prior_diff: 0.0,
        };
        let out = search
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut replay,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        if !compare(&out, &v["out"], n, &mut mm) {
            eprintln!(
                "case {k} ({}): rust visits {:?}\n  gold {}",
                v["config"],
                &out.visits[..n],
                v["out"]["visits"]
            );
        }
    }
    eprintln!("agent replay: {} cases, mismatches {:?}", lines.len(), mm);
    assert_eq!(mm.total(), 0, "{mm:?}");
}

/// The masked softmax on Python's own root logits against Python's root priors: equal, or 1
/// or 2 ulp apart where XLA's polynomial `exp` rounds differently from `f32::exp`.
#[test]
fn prior_softmax_close() {
    let Some((lines, _)) = agent_cases() else {
        return;
    };
    let (mut rows, mut bad, mut max_ulp) = (0usize, 0usize, 0u32);
    for v in &lines {
        let cfg = parse_game(&v["game"]);
        let state = parse_state(&v["state"], &cfg);
        let legal = state.legal_moves(&cfg);
        let logits = bits_list(&v["root_logits"]);
        let want = bits_list(&v["out"]["root_prior"]);
        for p in 0..cfg.num_snakes() {
            let l: [f32; 4] = std::array::from_fn(|a| f32::from_bits(logits[p * 4 + a]));
            let got = masked_softmax(&l, &legal[p]);
            rows += 1;
            let ulp = (0..NUM_ACTIONS)
                .map(|a| got[a].to_bits().abs_diff(want[p * 4 + a]))
                .max()
                .unwrap_or(0);
            bad += (ulp > 0) as usize;
            max_ulp = max_ulp.max(ulp);
        }
    }
    eprintln!("prior softmax: {rows} root rows, {bad} differ, by at most {max_ulp} ulp");
    assert!(max_ulp <= 2, "softmax off by {max_ulp} ulp");
}

/// One agent case with the Rust network: (config, move agrees, all seats' visits agree,
/// seat 0's visits agree, a line describing a disagreement).
fn agent_case(
    k: usize,
    v: &Value,
    recs: &[f32],
    net: &Network,
    search: &mut Search,
) -> (String, bool, bool, bool, Option<String>) {
    let cfg = parse_game(&v["game"]);
    let n = cfg.num_snakes();
    let state = parse_state(&v["state"], &cfg);
    let (settings, sims) = parse_settings(&v["search"]);
    let mut eval = NetEvaluator::for_game(net, cfg).unwrap();
    let out = search
        .run(
            &cfg,
            &state,
            &settings,
            Budget::simulations(sims),
            &mut eval,
            &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
        )
        .unwrap();
    let gold = &v["out"];
    let gv: Vec<Vec<i64>> = gold["visits"]
        .as_array()
        .unwrap()
        .iter()
        .map(ints)
        .collect();
    let all_visits = (0..n).all(|p| {
        out.visits[p]
            .iter()
            .map(|&x| x as i64)
            .eq(gv[p].iter().copied())
    });
    let visits0 = out.visits[0]
        .iter()
        .map(|&x| x as i64)
        .eq(gv[0].iter().copied());
    let a0 = out.action[0] as i64 == ints(&gold["action"])[0];
    let name = v["config"].as_str().unwrap().to_string();
    let mut note = None;
    if !a0 || !all_visits {
        // Python's tree with the Rust network evaluated on every node: how far apart are the
        // two networks on the states this search saw?
        let mut replay = Replay {
            recs: case_records(v, recs, n),
            n,
            cfg,
            net: Some(NetEvaluator::for_game(net, cfg).unwrap()),
            max_value_diff: 0.0,
            max_prior_diff: 0.0,
        };
        let rep = Search::new()
            .run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut replay,
                &mut JaxNoise::from_seed(u(&v["seed"]) as u32),
            )
            .unwrap();
        let mut mm = Mismatch::default();
        let replay_exact = compare(&rep, gold, n, &mut mm);
        // Root logits: Rust vs Python.
        let rl = bits_list(&v["root_logits"]);
        let mut e = NetEvaluator::for_game(net, cfg).unwrap();
        let seats = std::array::from_fn(|p| p < n && state.alive[p]);
        let mine = e.evaluate(0, &state, &seats).unwrap();
        let Prior::Logits(l) = mine.prior else {
            unreachable!()
        };
        let mut dl = 0f32;
        for p in (0..n).filter(|&p| seats[p]) {
            for a in 0..NUM_ACTIONS {
                dl = dl.max((l[p][a] - f32::from_bits(rl[p * 4 + a])).abs());
            }
        }
        note = Some(format!(
            "case {k} {name} seat0 move rust {} python {} | rust visits {:?} python {:?} | \
             replay with Python's net outputs bit-exact: {replay_exact} | max |dvalue| {:.2e} \
             max |dprior| {:.2e} over Python's {} nodes, root |dlogit| {:.2e}",
            out.action[0],
            ints(&gold["action"])[0],
            &out.visits[..n],
            gv,
            replay.max_value_diff,
            replay.max_prior_diff,
            u(&v["num_nodes"]),
            dl
        ));
    }
    (name, a0, all_visits, visits0, note)
}

#[test]
fn agent_network() {
    let Some((lines, recs)) = agent_cases() else {
        return;
    };
    let Some(net) = load_net() else {
        return;
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1);
    let next = Mutex::new(0usize);
    let results = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut search = Search::new();
                loop {
                    let k = {
                        let mut g = next.lock().unwrap();
                        let k = *g;
                        *g += 1;
                        k
                    };
                    if k >= lines.len() {
                        break;
                    }
                    let r = agent_case(k, &lines[k], &recs, &net, &mut search);
                    results.lock().unwrap().push((k, r));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let mut per: std::collections::BTreeMap<String, [usize; 4]> = Default::default();
    for (_, (name, a0, all, v0, note)) in &results {
        let e = per.entry(name.clone()).or_default();
        e[0] += 1;
        e[1] += *a0 as usize;
        e[2] += *all as usize;
        e[3] += *v0 as usize;
        if let Some(n) = note {
            eprintln!("{n}");
        }
    }
    let mut tot = [0usize; 4];
    for (name, e) in &per {
        eprintln!(
            "agent {name}: {} cases, move agrees {}, seat-0 visits equal {}, all visits equal {}",
            e[0], e[1], e[3], e[2]
        );
        for i in 0..4 {
            tot[i] += e[i];
        }
    }
    eprintln!(
        "agent total: {} cases in {:.1}s, move agreement {:.2}% ({}), seat-0 visits equal {:.2}%, all visits equal {:.2}%",
        tot[0],
        t0.elapsed().as_secs_f64(),
        100.0 * tot[1] as f64 / tot[0] as f64,
        tot[0] - tot[1],
        100.0 * tot[3] as f64 / tot[0] as f64,
        100.0 * tot[2] as f64 / tot[0] as f64
    );
    assert!(
        tot[1] as f64 >= 0.99 * tot[0] as f64,
        "move agreement below 99%"
    );
}

#[test]
fn end_to_end_requests() {
    let Some(lines) = read_lines("e2e.jsonl") else {
        return;
    };
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    let (mut moves, mut visits) = (0, 0);
    for (k, v) in lines.iter().enumerate() {
        let body = v["request"].to_string();
        let starting = v["starting"].as_u64().map(|x| x as usize);
        let m = agent
            .play(&body, starting, u(&v["seed"]) as u32)
            .unwrap_or_else(|e| panic!("case {k}: {e}"));
        assert_eq!(m.num_snakes as u64, u(&v["num_snakes"]), "case {k}");
        assert_eq!(m.mode.name(), v["mode"].as_str().unwrap(), "case {k}");
        let gold = &v["out"];
        let a0 = m.mv as i64 == ints(&gold["action"])[0];
        let gv = ints(&gold["visits"][0]);
        let v0 = m.search.visits[0]
            .iter()
            .map(|&x| x as i64)
            .eq(gv.iter().copied());
        moves += a0 as usize;
        visits += v0 as usize;
        if !a0 || !v0 {
            eprintln!(
                "e2e case {k} ({}): rust {:?} {:?} python {} {:?}",
                v["mode"], m.mv, m.search.visits[0], gold["action"], gv
            );
        }
    }
    eprintln!(
        "e2e: {} requests, move agrees {moves}, seat-0 visits equal {visits}",
        lines.len()
    );
    assert!(moves as f64 >= 0.98 * lines.len() as f64);
}
