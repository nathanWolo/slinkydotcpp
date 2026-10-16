//! Adversarial review of the search against slinky's Python search, on the vectors of
//! `native/golden/review_search.py` (see its docstring for the formats), plus request-level
//! robustness checks that need no golden data.
//!
//! Where `tests/golden.rs` samples typical rollout positions, these cases aim at the parts most
//! likely to break: forced moves of dead and certainly-starving seats (ours included), finished
//! roots and terminal children at small depth, the depth cutoff (`max_depth` 1 to 6 and a sharp
//! evaluator that drives one line down to depth 32), exact ties (`tie_noise` 0, constant
//! evaluators: the final move's tie-breaks), 4-seat placement backups with dead seats, royale
//! stacked hazards, constrictor roots with food, uniformly random play, and API requests shaped
//! like the ladder's games.
//!
//! * `review_crafted_exact` / `review_random_exact`: `mech` lines with the uniform evaluator and
//!   `replay`/`agent` lines with every node's value and priors replayed from Python's tree must
//!   reproduce `mcts.search` bit for bit (visits, q, policy, value, moves, node count, depth).
//!   The root priors are checked separately (`root_prior_*`), see below.
//! * `review_crafted_network` / `review_random_network`: the same positions with the Rust
//!   network: seat 0's move must agree (float noise may move a few visits).
//! * `review_ladder_requests`: `Agent::play` on ladder-shaped requests against the Python
//!   server path, including requests the server would not search.
//! * `review_bad_requests_never_panic`: hostile request bodies through `Agent::respond`.
//!
//! Run in release: `cargo test --release -p slinky-search --test review_search -- --nocapture`.

use serde_json::{Value, json};
use slinky_net::Network;
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, NUM_ACTIONS, Ruleset};
use slinky_search::{
    Agent, Budget, Evaluator, JaxNoise, NetEvaluator, NodeEval, Prior, Search, SearchConfig,
    SearchError, SearchOutput, UniformEvaluator,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

// --- Golden data -------------------------------------------------------------------------------

fn golden_dir() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("search")
}

fn run_dir() -> PathBuf {
    std::env::var("NAZ_RUN")
        .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into())
        .into()
}

fn read_lines(name: &str) -> Option<Vec<Value>> {
    let path = golden_dir().join(name);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP: {} missing (run native/golden/review_search.py)",
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

fn read_records(name: &str) -> Option<Vec<f32>> {
    let path = golden_dir().join(name);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("SKIP: {} missing", path.display());
        return None;
    };
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
    let mut s = GameState::new(cfg);
    s.turn = i(&v["turn"]) as i32;
    s.done = i(&v["done"]) != 0;
    for p in 0..cfg.num_snakes() {
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
        tie_noise: f32_of(&v["tie_noise"]),
        max_depth: u(&v["max_depth"]) as usize,
    };
    (cfg, u(&v["sims"]) as usize)
}

/// Node outputs replayed from Python's tree, by creation index.
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
        let mut values = [0.0f32; MAX_SNAKES];
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

/// Which output fields differ from the golden `out` (empty when bit-identical). The root
/// priors are reported as `root_prior` and kept apart by the callers.
fn diff_fields(out: &SearchOutput, gold: &Value, n: usize) -> Vec<&'static str> {
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
    let checks = [
        ("visits", visits == gv),
        ("q", flat(&out.q) == bits_list(&gold["q"])),
        ("policy", flat(&out.policy) == bits_list(&gold["policy"])),
        ("value", value == bits_list(&gold["value"])),
        ("action", action == ints(&gold["action"])),
        (
            "nodes_used",
            out.nodes_used as u64 == u(&gold["nodes_used"]),
        ),
        ("depth", out.depth as u64 == u(&gold["depth"])),
        (
            "root_prior",
            flat(&out.root_prior) == bits_list(&gold["root_prior"]),
        ),
    ];
    for (name, ok) in checks {
        if !ok {
            bad.push(name);
        }
    }
    bad
}

fn case_records<'a>(v: &Value, recs: &'a [f32], n: usize) -> &'a [f32] {
    let off = u(&v["node_offset"]) as usize / 4;
    let len = u(&v["num_nodes"]) as usize * n * 5;
    &recs[off..off + len]
}

/// Bit-exact check of every line of one part: (searches, failures excluding root priors,
/// root-prior-only differences on finished roots, other root-prior differences).
fn exact_part(part: &str) -> Option<(usize, Vec<String>, usize, Vec<String>)> {
    let lines = read_lines(&format!("review_{part}.jsonl"))?;
    let recs = read_records(&format!("review_{part}_nodes.bin"))?;
    let mut search = Search::new();
    let mut failures = Vec::new();
    let mut prior_on_finished = 0usize;
    let mut prior_other = Vec::new();
    for (k, v) in lines.iter().enumerate() {
        let cfg = parse_game(&v["game"]);
        let n = cfg.num_snakes();
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let mut noise = JaxNoise::from_seed(u(&v["seed"]) as u32);
        let out = if v["kind"] == "mech" {
            search.run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut UniformEvaluator,
                &mut noise,
            )
        } else {
            let mut replay = Replay {
                recs: case_records(v, &recs, n),
                n,
            };
            search.run(
                &cfg,
                &state,
                &settings,
                Budget::simulations(sims),
                &mut replay,
                &mut noise,
            )
        }
        .unwrap_or_else(|e| panic!("line {k}: {e}"));
        let bad = diff_fields(&out, &v["out"], n);
        let label = format!(
            "line {k} {} [{} {} max_depth {} tie {}]",
            v["name"].as_str().unwrap(),
            v["kind"].as_str().unwrap(),
            v["evaluator"].as_str().unwrap(),
            settings.max_depth,
            settings.tie_noise
        );
        let rest: Vec<&str> = bad.iter().copied().filter(|&b| b != "root_prior").collect();
        if !rest.is_empty() {
            failures.push(format!(
                "{label}: {rest:?}\n    rust visits {:?} action {:?} nodes {} depth {}\n    \
                 python visits {} action {} nodes {} depth {}",
                &out.visits[..n],
                &out.action[..n],
                out.nodes_used,
                out.depth,
                v["out"]["visits"],
                v["out"]["action"],
                v["out"]["nodes_used"],
                v["out"]["depth"]
            ));
        }
        if bad.contains(&"root_prior") {
            if state.done {
                prior_on_finished += 1;
            } else {
                prior_other.push(label);
            }
        }
    }
    Some((lines.len(), failures, prior_on_finished, prior_other))
}

fn report_exact(part: &str) {
    let Some((total, failures, prior_finished, prior_other)) = exact_part(part) else {
        return;
    };
    for f in failures.iter().take(20) {
        eprintln!("{f}");
    }
    eprintln!(
        "review {part}: {total} searches, {} differ (root priors aside); root priors differ on \
         {prior_finished} finished roots and {} other searches",
        failures.len(),
        prior_other.len()
    );
    for p in prior_other.iter().take(5) {
        eprintln!("  root prior differs: {p}");
    }
    assert!(failures.is_empty(), "{} searches differ", failures.len());
    assert!(prior_other.is_empty(), "root priors differ on live roots");
}

#[test]
fn review_crafted_exact() {
    report_exact("crafted");
}

#[test]
fn review_random_exact() {
    report_exact("random");
}

/// The crafted positions with the search constants a retrained run may use (`c_puct` 0.3 to
/// 1.1, `fpu_reduction` 0 to 0.5, `draw_value` -0.7 to 0.25).
///
/// Fails on the 4-seat draws: when every seat is dead and several died on the last turn, the
/// jitted `placement_values` computes the draw term `draw_value * tied / (n - 1)` as
/// `tied * (f32(draw_value) * f32(1 / (n - 1)))` (XLA folds the two constants), while
/// `slinky_rules::GameState::placement_values` computes `(draw_value * tied) * recip`; with
/// `n - 1 = 3` the two round differently for some draw values (-0.45, -1/3 with 4 tied), so the
/// terminal values and every q above them differ by an ulp. The deployed run's -0.5 is exact
/// either way, so the ladder agent is not affected.
#[test]
#[ignore = "BUG (rules crate, retrained configs only): 4-seat draw term rounded as (draw*tied)*recip, \
            XLA computes tied*(draw*recip)"]
fn review_variants_exact() {
    report_exact("variants");
}

/// The root priors of a finished root: Python reports the masked softmax of the network's
/// logits there (the evaluator runs on the root regardless, mcts.py:1226, 873-878), the Rust
/// search reports the uniform prior because it skips the evaluator on finished nodes. Only the
/// reported `root_prior` differs (a finished root is never searched, so no move changes).
#[test]
#[ignore = "BUG (minor, reporting only): root_prior of a finished root is uniform in Rust, the \
            network's masked softmax in Python"]
fn root_prior_on_finished_root() {
    let Some((_, _, prior_finished, _)) = exact_part("crafted") else {
        return;
    };
    assert_eq!(prior_finished, 0, "root_prior differs on finished roots");
}

// --- The Rust network on the same positions ---------------------------------------------------

fn network_part(part: &str) {
    let Some(lines) = read_lines(&format!("review_{part}.jsonl")) else {
        return;
    };
    let Some(net) = load_net() else {
        return;
    };
    let mut search = Search::new();
    let mut per: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for (k, v) in lines
        .iter()
        .enumerate()
        .filter(|(_, v)| v["kind"] == "agent")
    {
        let cfg = parse_game(&v["game"]);
        let n = cfg.num_snakes();
        let state = parse_state(&v["state"], &cfg);
        let (settings, sims) = parse_settings(&v["search"]);
        let mut eval = NetEvaluator::for_game(&net, cfg).unwrap();
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
        let a0 = out.action[0] as i64 == ints(&gold["action"])[0];
        let v0 = out.visits[0]
            .iter()
            .map(|&x| x as i64)
            .eq(ints(&gold["visits"][0]));
        let e = per
            .entry(v["config"].as_str().unwrap().to_string())
            .or_default();
        e[0] += 1;
        e[1] += a0 as usize;
        e[2] += v0 as usize;
        if !a0 || !v0 {
            eprintln!(
                "{part} line {k} {}: rust move {} visits {:?} | python move {} visits {}",
                v["name"].as_str().unwrap(),
                out.action[0],
                &out.visits[..n],
                gold["action"][0],
                gold["visits"]
            );
        }
    }
    let mut tot = [0usize; 3];
    for (name, e) in &per {
        eprintln!(
            "network {part} {name}: {} searches, move agrees {}, seat-0 visits equal {}",
            e[0], e[1], e[2]
        );
        for j in 0..3 {
            tot[j] += e[j];
        }
    }
    assert!(
        tot[1] as f64 >= 0.98 * tot[0] as f64,
        "move agreement {}/{}",
        tot[1],
        tot[0]
    );
}

#[test]
fn review_crafted_network() {
    network_part("crafted");
}

#[test]
fn review_random_network() {
    network_part("random");
}

// --- Ladder-shaped requests ------------------------------------------------------------------

#[test]
fn review_ladder_requests() {
    let Some(lines) = read_lines("review_ladder.jsonl") else {
        return;
    };
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    let mut per: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut problems = Vec::new();
    for (k, v) in lines.iter().enumerate() {
        let body = v["request"].to_string();
        let starting = v["starting"].as_u64().map(|x| x as usize);
        let res = agent.play(&body, starting, u(&v["seed"]) as u32);
        let label = format!(
            "{} starting {:?}",
            v["name"].as_str().unwrap(),
            v["starting"].as_u64()
        );
        let e = per.entry(label.clone()).or_default();
        e[0] += 1;
        if let Some(why) = v.get("skip").and_then(Value::as_str) {
            // Python does not search: the server plays its fallback.
            if let Ok(m) = &res {
                problems.push(format!(
                    "line {k} {label}: Python skips ({why}) but Rust searched ({} seats)",
                    m.num_snakes
                ));
            } else {
                e[1] += 1;
                e[2] += 1;
            }
            continue;
        }
        let m = match res {
            Ok(m) => m,
            Err(err) => {
                problems.push(format!("line {k} {label}: Rust refused: {err}"));
                continue;
            }
        };
        if m.num_snakes as u64 != u(&v["num_snakes"]) || m.mode.name() != v["mode"] {
            problems.push(format!(
                "line {k} {label}: seats/mode rust {} {} python {} {}",
                m.num_snakes,
                m.mode.name(),
                v["num_snakes"],
                v["mode"]
            ));
        }
        let gold = &v["out"];
        let a0 = m.mv as i64 == ints(&gold["action"])[0];
        let v0 = m.search.visits[0]
            .iter()
            .map(|&x| x as i64)
            .eq(ints(&gold["visits"][0]));
        e[1] += a0 as usize;
        e[2] += v0 as usize;
        if !a0 || !v0 {
            eprintln!(
                "ladder line {k} {label}: rust {:?} {:?} python {} {}",
                m.mv, m.search.visits[0], gold["action"][0], gold["visits"][0]
            );
        }
    }
    let mut tot = [0usize; 3];
    for (name, e) in &per {
        eprintln!(
            "ladder {name}: {} requests, move agrees {}, seat-0 visits equal {}",
            e[0], e[1], e[2]
        );
        for j in 0..3 {
            tot[j] += e[j];
        }
    }
    for p in &problems {
        eprintln!("{p}");
    }
    assert!(problems.is_empty(), "{} problems", problems.len());
    assert!(tot[1] as f64 >= 0.98 * tot[0] as f64);
}

// --- Hostile requests ---------------------------------------------------------------------------

fn base_request() -> Value {
    json!({
        "game": {"id": "g", "ruleset": {"name": "standard", "settings": {}}, "map": "standard"},
        "turn": 5,
        "board": {
            "width": 11, "height": 11, "food": [{"x": 5, "y": 5}], "hazards": [],
            "snakes": [
                {"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]},
                {"id": "b", "health": 90, "body": [{"x": 8, "y": 8}, {"x": 8, "y": 7}, {"x": 8, "y": 6}]}
            ]
        },
        "you": {"id": "a", "health": 90, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]}
    })
}

/// Hostile and malformed bodies (from a public endpoint) must never panic: each gets a move.
#[test]
fn review_bad_requests_never_panic() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net).with_simulations(16);
    let mut bodies: Vec<(String, String)> = Vec::new();
    let mut add = |name: &str, v: Value| bodies.push((name.into(), v.to_string()));
    let b = base_request();
    add("base", b.clone());
    let mut v = b.clone();
    v["board"]["snakes"] = json!([]);
    add("no snakes", v);
    let mut v = b.clone();
    v["you"]["id"] = json!("zzz");
    add("you missing", v);
    let mut v = b.clone();
    v["board"]["snakes"][1]["id"] = json!("a");
    add("duplicate you id", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["body"] = json!([]);
    v["you"]["body"] = json!([]);
    add("empty body", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["body"] = json!([{"x": -3, "y": 50}, {"x": 99, "y": -7}]);
    add("off-board body", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["body"] =
        json!([{"x": 2147483647, "y": -2147483648}, {"x": 1, "y": 1}]);
    add("extreme coords", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["health"] = json!(-2147483648i64);
    add("min health", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["health"] = json!(2147483647);
    add("max health", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["health"] = json!(0);
    add("zero health", v);
    let mut v = b.clone();
    v["turn"] = json!(2147483647);
    add("max turn", v);
    let mut v = b.clone();
    v["turn"] = json!(-5);
    add("negative turn", v);
    let mut v = b.clone();
    v["turn"] = json!(9_000_000_000i64);
    add("turn beyond i32", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["name"] = json!("royale");
    v["turn"] = json!(2147483647);
    v["board"]["hazards"] = json!([{"x": 0, "y": 0}]);
    add("royale max turn", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["name"] = json!("royale");
    v["game"]["ruleset"]["settings"] = json!({"royale": {"shrinkEveryNTurns": 0}});
    add("royale shrink 0", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["name"] = json!("royale");
    v["game"]["ruleset"]["settings"] = json!({"royale": {"shrinkEveryNTurns": -25}});
    add("royale shrink negative", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["settings"] = json!({"hazardDamagePerTurn": -1048576});
    v["board"]["hazards"] = json!([{"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": 2, "y": 1}]);
    add("healing hazards", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["settings"] = json!({"hazardDamagePerTurn": 1048576});
    v["board"]["hazards"] = Value::Array(vec![json!({"x": 1, "y": 0}); 4]);
    add("huge hazard damage stacked", v);
    let mut v = b.clone();
    v["board"]["food"] = json!([{"x": 5, "y": 5}, {"x": 5, "y": 5}]);
    add("duplicate food", v);
    let mut v = b.clone();
    v["board"]["food"] = json!([{"x": 50, "y": 5}]);
    add("off-board food", v);
    let mut v = b.clone();
    v["board"]["width"] = json!(0);
    v["board"]["height"] = json!(0);
    add("zero board", v);
    let mut v = b.clone();
    v["board"]["width"] = json!(-11);
    add("negative width", v);
    let mut v = b.clone();
    v["board"]["width"] = json!(1_000_000_000i64);
    v["board"]["height"] = json!(1_000_000_000i64);
    add("huge board", v);
    let mut v = b.clone();
    v["board"]["width"] = json!(25);
    v["board"]["height"] = json!(25);
    add("25x25 board", v);
    let mut v = b.clone();
    v["board"]["width"] = json!(7);
    v["board"]["height"] = json!(7);
    add("7x7 board", v);
    let mut v = b.clone();
    let five: Vec<Value> = (0..5)
        .map(|k| {
            json!({"id": if k == 0 {"a".to_string()} else {format!("o{k}")}, "health": 50,
                        "body": [{"x": 2 * k, "y": 0}]})
        })
        .collect();
    v["board"]["snakes"] = Value::Array(five);
    add("five snakes", v);
    let mut v = b.clone();
    let long: Vec<Value> = (0..124)
        .map(|k| json!({"x": k % 11, "y": k / 11}))
        .collect();
    v["board"]["snakes"][0]["body"] = Value::Array(long);
    add("body longer than board", v);
    let mut v = b.clone();
    v["board"]["snakes"][0]["body"] = Value::Array(vec![json!({"x": 1, "y": 1}); 40]);
    add("40 stacked segments", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["name"] = json!("constrictor");
    v["game"]["map"] = json!("empty");
    add("constrictor empty", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["name"] = json!("wrapped");
    add("wrapped", v);
    let mut v = b.clone();
    v["game"] = json!("nonsense");
    add("game not an object", v);
    let mut v = b.clone();
    v["game"]["ruleset"]["settings"] = json!({"hazardDamagePerTurn": "1e9"});
    add("string setting", v);
    for (name, body) in &bodies {
        for starting in [
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(4),
            Some(9),
            Some(usize::MAX),
        ] {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                agent.respond(body, starting, 3)
            }));
            assert!(r.is_ok(), "panic on {name} (starting {starting:?})");
        }
    }
    for text in ["", "{", "null", "[]", "{\"turn\": 1}", "\u{0}"] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent.respond(text, None, 0)
        }));
        assert!(r.is_ok(), "panic on {text:?}");
    }
}

/// Our snake certainly starving (health 1, no food beside its head): the search forces move 0,
/// so the agent answers "up" whatever the board (SPEC.md 8.5), as Python does.
#[test]
fn review_own_starving_snake_answers_up() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net).with_simulations(64);
    let mut v = base_request();
    v["board"]["snakes"][0]["health"] = json!(1);
    v["you"]["health"] = json!(1);
    // Moving up runs into its own neck: the forced move is still 0.
    let m = agent.play(&v.to_string(), Some(2), 1).unwrap();
    assert_eq!(m.mv.name(), "up");
    assert_eq!(m.search.legal[0], [true, false, false, false]);
    assert_eq!(m.search.visits[0], [64, 0, 0, 0]);
}

/// Our snake alone (a 4-seat game whose other snakes died): a finished root, nothing is
/// searched, and the move is the lowest-index move of the action mask (`_output`'s uniform
/// policy and q = 0), as Python answers.
#[test]
fn review_lone_survivor_plays_first_safe_move() {
    let Some(net) = load_net() else {
        return;
    };
    let agent = Agent::from_network(net);
    let mut v = base_request();
    v["board"]["snakes"] = json!([v["board"]["snakes"][0].clone()]);
    // Head (1, 1), neck (1, 2): up is the neck (blocked), down is free.
    let m = agent.play(&v.to_string(), Some(4), 0).unwrap();
    assert_eq!(m.search.visits[0], [0; 4]);
    assert_eq!(m.search.nodes_used, 1);
    assert_eq!(m.mv.name(), "down");
}

/// The draw term of the placement values against the jitted Python's bits (from
/// `mcts.placement_values(state, standard 4p, draw)` with all four seats eliminated on turn 5):
/// XLA evaluates `draw * tied / (n - 1)` as `tied * (f32(draw) * f32(1/3))`.
#[test]
#[ignore = "BUG (rules crate, retrained configs only): 4-seat draw term rounded as (draw*tied)*recip, \
            XLA computes tied*(draw*recip)"]
fn placement_draw_term_matches_xla() {
    let cfg = GameConfig::new(11, 11, 4, Ruleset::Standard).unwrap();
    let mut s = GameState::new(&cfg);
    s.alive = [false; MAX_SNAKES];
    s.elim_turn = [5; MAX_SNAKES];
    s.turn = 6;
    s.done = true;
    // (draw value, Python's f32 bits for every seat)
    for (draw, want) in [
        (-0.45f64, 3_202_770_535u32),
        (-1.0 / 3.0, 3_198_855_852),
        (-0.5, 3_204_448_256),
    ] {
        let v = s.placement_values(&cfg, draw as f32);
        assert_eq!(
            v.map(f32::to_bits),
            [want; MAX_SNAKES],
            "draw {draw}: rust {:?}",
            v
        );
    }
}
