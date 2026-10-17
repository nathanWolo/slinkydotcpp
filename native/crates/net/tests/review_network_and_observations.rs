//! Adversarial review of the network (`slinky-net`) and observation (`slinky-obs`) crates.
//!
//! Golden vectors come from `native/golden/review_network_and_observations.py` (see its
//! docstring for the file format), written to `$NAZ_GOLDEN/review_netobs/` (default
//! `/dev/shm/naz-golden`). The golden tests skip with a message when the files are missing.
//!
//! * [`odd_architectures_match_jax`]: the forward pass on nets with odd strides (1, 3, 4, five
//!   stride-2 layers down to 1x1), odd channel counts and hidden widths, all three feature modes,
//!   and the S, M and L sizes, against the jitted `az_modes.network` on boards aimed at the
//!   padding arithmetic (single pixels in each window corner, the board in each window corner).
//! * [`e2e_boards_and_features_are_bitwise`]: [`Encoder::encode_into`] against
//!   `unified.policy_inputs` on hand-built edge states (corner heads, coiled snakes, turn 0 and
//!   late turns, royale shrink boundaries for several periods, deaths of every kind, constrictor
//!   food, tiny and rectangular boards), bit for bit.
//! * [`e2e_evaluator_matches_python`]: the encoder followed by the network against the whole
//!   Python evaluator (`az_modes.make_evaluator`) for every net, including the no_mode / none
//!   feature handling split between the two crates.
//! * Pure-Rust checks (no golden needed): window geometry for every head cell, layout constants
//!   agreeing between the crates, and robustness on bad input (no panics).

use serde_json::Value;
use slinky_net::{NUM_FEATURES, NetConfig, Network, OBS_LEN, npy};
use slinky_obs::{Encoder, NUM_CHANNELS, ch, col, egocentric_into, mode_features, obs_len};
use slinky_rules::{GameConfig, GameState, Mode, Ruleset};
use std::path::PathBuf;

fn root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("review_netobs")
}

fn skip(what: &str) -> bool {
    let r = root();
    if r.join(what).exists() {
        return false;
    }
    eprintln!(
        "SKIP: {} missing; run native/golden/review_network_and_observations.py (see its docstring)",
        r.join(what).display()
    );
    true
}

// --- Network on odd architectures ------------------------------------------------------------

#[test]
fn odd_architectures_match_jax() {
    if skip("nets") {
        return;
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("nets"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort();
    let mut bad = Vec::new();
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let net = Network::load(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        let cases = npy::parse_npz(&std::fs::read(dir.join("cases.npz")).unwrap()).unwrap();
        let (obs, feats, logits, value) = (
            &cases["obs"].data,
            &cases["feats"].data,
            &cases["logits"].data,
            &cases["value"].data,
        );
        let n = value.len();
        assert_eq!(obs.len(), n * OBS_LEN);
        // One row at a time and the whole batch must agree with each other and with JAX.
        let all = net.forward(obs, feats).unwrap();
        let (mut max_rel, mut max_v) = (0f32, 0f32);
        for r in 0..n {
            let one = net
                .forward(
                    &obs[r * OBS_LEN..][..OBS_LEN],
                    &feats[r * NUM_FEATURES..][..NUM_FEATURES],
                )
                .unwrap();
            assert_eq!(
                one.logits[0], all.logits[r],
                "{name}: row {r} batch vs single"
            );
            for a in 0..4 {
                let want = logits[r * 4 + a];
                let e = (all.logits[r][a] - want).abs() / want.abs().max(1.0);
                max_rel = max_rel.max(e);
            }
            max_v = max_v.max((all.values[r] - value[r]).abs());
        }
        eprintln!("{name}: {n} cases, max rel dlogit {max_rel:.2e}, max dvalue {max_v:.2e}");
        if !(max_rel < 1e-4 && max_v < 1e-4) {
            bad.push(format!("{name}: rel dlogit {max_rel:e}, dvalue {max_v:e}"));
        }
    }
    assert!(bad.is_empty(), "network mismatches: {bad:?}");
}

// --- End to end: observation + network -------------------------------------------------------

struct Case {
    name: String,
    cfg: GameConfig,
    hide: bool,
    state: GameState,
    rec: Value,
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn load_cases() -> Vec<Case> {
    let text = std::fs::read_to_string(root().join("e2e.jsonl")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let rec: Value = serde_json::from_str(l).unwrap();
            let u = |k: &str| rec[k].as_u64().unwrap() as usize;
            let ruleset = Ruleset::from_name(rec["ruleset"].as_str().unwrap()).unwrap();
            let cfg = GameConfig::new(u("width"), u("height"), u("num_snakes"), ruleset)
                .unwrap()
                .with_shrink_every(rec["shrink_every"].as_i64().unwrap() as i32)
                .unwrap();
            let n = cfg.num_snakes();
            let mut s = GameState::new(&cfg);
            s.turn = rec["turn"].as_i64().unwrap() as i32;
            s.done = rec["done"].as_i64().unwrap() != 0;
            let heads = rec["head"].as_array().unwrap();
            let (len, hp, alive, et) = (
                ints(&rec["length"]),
                ints(&rec["health"]),
                ints(&rec["alive"]),
                ints(&rec["elim_turn"]),
            );
            let body = rec["body"].as_array().unwrap();
            for i in 0..n {
                let h = ints(&heads[i]);
                s.head[i] = [h[0] as i32, h[1] as i32];
                s.length[i] = len[i] as i32;
                s.health[i] = hp[i] as i32;
                s.alive[i] = alive[i] != 0;
                s.elim_turn[i] = et[i] as i32;
                for (c, v) in ints(&body[i]).into_iter().enumerate() {
                    s.body[i][c] = v as i16;
                }
            }
            for c in ints(&rec["food"]) {
                s.food |= 1u128 << c;
            }
            for (c, v) in ints(&rec["hazard"]).into_iter().enumerate() {
                s.hazard[c] = v as i16;
            }
            Case {
                name: rec["name"].as_str().unwrap().to_string(),
                cfg,
                hide: rec["hide_food"].as_i64().unwrap() != 0,
                state: s,
                rec,
            }
        })
        .collect()
}

fn describe(i: usize, cols: usize) -> String {
    format!(
        "row {} col {} channel {}",
        i / NUM_CHANNELS / cols,
        i / NUM_CHANNELS % cols,
        i % NUM_CHANNELS
    )
}

#[test]
fn e2e_boards_and_features_are_bitwise() {
    if skip("e2e.jsonl") {
        return;
    }
    let cases = load_cases();
    assert!(!cases.is_empty());
    let mut bad = Vec::new();
    for c in &cases {
        let enc = Encoder::new(c.cfg, true);
        assert_eq!(enc.hides_food(), c.hide, "{}: hide_food", c.name);
        let n = c.cfg.num_snakes();
        let len = enc.obs_len();
        let mut boards = vec![f32::NAN; n * len];
        let mut feats = vec![f32::NAN; n * NUM_FEATURES];
        enc.encode_into(&c.state, &mut boards, &mut feats).unwrap();
        let cols = 2 * c.cfg.width() - 1;
        for seat in 0..n {
            let mut want = vec![0u32; len];
            for e in c.rec["boards"][seat].as_array().unwrap() {
                let e = ints(e);
                want[e[0] as usize] = e[1] as u32;
            }
            let got = &boards[seat * len..][..len];
            if let Some(i) = (0..len).find(|&i| got[i].to_bits() != want[i]) {
                let k = (0..len).filter(|&i| got[i].to_bits() != want[i]).count();
                bad.push(format!(
                    "{} seat {seat}: {k} board floats differ, first {}: got {} want {}",
                    c.name,
                    describe(i, cols),
                    got[i],
                    f32::from_bits(want[i])
                ));
            }
        }
        let want_f: Vec<u32> = ints(&c.rec["feats"]).iter().map(|&b| b as u32).collect();
        let got_f: Vec<u32> = feats.iter().map(|f| f.to_bits()).collect();
        if want_f != got_f {
            bad.push(format!(
                "{}: features differ: got {:?} want {:?}",
                c.name,
                feats,
                want_f
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect::<Vec<_>>()
            ));
        }
    }
    eprintln!("{} states checked", cases.len());
    assert!(
        bad.is_empty(),
        "observation mismatches:\n{}",
        bad.join("\n")
    );
}

#[test]
fn e2e_evaluator_matches_python() {
    if skip("e2e.jsonl") || skip("nets") {
        return;
    }
    let cases = load_cases();
    let mut nets = std::collections::BTreeMap::new();
    let mut bad = Vec::new();
    let mut checked = 0;
    for c in &cases {
        let Some(evals) = c.rec["eval"].as_object() else {
            continue;
        };
        for (name, want) in evals {
            let net = nets.entry(name.clone()).or_insert_with(|| {
                Network::load(root().join("nets").join(name)).unwrap_or_else(|e| panic!("{e}"))
            });
            // What the search does: the encoder for the game with the run's food hiding.
            let enc = Encoder::new(c.cfg, net.config().hide_constrictor_food);
            let n = c.cfg.num_snakes();
            let mut boards = vec![0.0; n * OBS_LEN];
            let mut feats = vec![0.0; n * NUM_FEATURES];
            enc.encode_into(&c.state, &mut boards, &mut feats).unwrap();
            let out = net.forward(&boards, &feats).unwrap();
            let wv: Vec<f64> = want["value"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            for seat in 0..n {
                let wl: Vec<f64> = want["logits"][seat]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                let rel = (0..4)
                    .map(|a| (out.logits[seat][a] as f64 - wl[a]).abs() / wl[a].abs().max(1.0))
                    .fold(0f64, f64::max);
                let dv = (out.values[seat] as f64 - wv[seat]).abs();
                checked += 1;
                if rel > 1e-4 || dv > 1e-4 {
                    bad.push(format!(
                        "{} net {name} seat {seat}: logits {:?} want {wl:?}, value {} want {}",
                        c.name, out.logits[seat], out.values[seat], wv[seat]
                    ));
                }
            }
        }
    }
    eprintln!("{checked} (state, net, seat) rows checked");
    assert!(checked > 0);
    assert!(bad.is_empty(), "evaluator mismatches:\n{}", bad.join("\n"));
}

// --- Pure Rust checks -------------------------------------------------------------------------

#[test]
fn layouts_agree_between_crates() {
    assert_eq!(slinky_obs::OBS_LEN, OBS_LEN);
    assert_eq!(slinky_obs::NUM_FEATURES, NUM_FEATURES);
    assert_eq!(slinky_obs::OBS_ROWS, slinky_net::OBS_ROWS);
    assert_eq!(slinky_obs::OBS_COLS, slinky_net::OBS_COLS);
    assert_eq!(slinky_obs::NUM_CHANNELS, slinky_net::OBS_CHANNELS);
}

/// For every head cell of several boards: the head sits at the window centre, the board
/// occupies exactly `cells` window pixels (on_board), and window pixel (r, c) is board cell
/// (hx + c - (W-1), hy + r - (H-1)).
#[test]
fn window_geometry_for_every_head_cell() {
    for (w, h) in [(11, 11), (7, 7), (9, 11), (11, 5), (1, 3), (3, 1), (1, 1)] {
        let Ok(cfg) = GameConfig::new(w, h, 2, Ruleset::Standard) else {
            continue;
        };
        let len = obs_len(&cfg);
        let (rows, cols) = (2 * h - 1, 2 * w - 1);
        let mut o = vec![f32::NAN; len];
        for hy in 0..h as i32 {
            for hx in 0..w as i32 {
                let mut s = GameState::new(&cfg);
                s.set_snake(&cfg, 0, &[(hx, hy)], 50, None).unwrap();
                s.set_snake(&cfg, 1, &[], 0, Some((slinky_rules::Cause::Collision, 0)))
                    .unwrap();
                // A hazard count equal to cell index + 1 identifies every cell.
                for c in 0..cfg.cells() {
                    s.hazard[c] = c as i16 + 1;
                }
                egocentric_into(&s, &cfg, 0, false, &mut o).unwrap();
                let centre = ((h - 1) * cols + (w - 1)) * NUM_CHANNELS;
                assert_eq!(o[centre + ch::OWN_HEAD], 1.0, "{w}x{h} head ({hx},{hy})");
                let mut on = 0;
                for r in 0..rows {
                    for c in 0..cols {
                        let p = (r * cols + c) * NUM_CHANNELS;
                        let x = hx + c as i32 - (w as i32 - 1);
                        let y = hy + r as i32 - (h as i32 - 1);
                        let inside = cfg.in_bounds(x, y);
                        assert_eq!(o[p + ch::ON_BOARD], f32::from(u8::from(inside)));
                        on += usize::from(inside);
                        let want = cfg.cell(x, y).map_or(0.0, |cell| cell as f32 + 1.0);
                        assert_eq!(
                            o[p + ch::HAZARD],
                            want,
                            "{w}x{h} head ({hx},{hy}) r{r} c{c}"
                        );
                    }
                }
                assert_eq!(on, cfg.cells());
            }
        }
    }
}

/// `opponents_alive` after deaths and with padded seats, and the shrink phase at the period
/// boundaries, from first principles (SPEC 3.3).
#[test]
fn feature_edge_values() {
    let cfg = GameConfig::for_mode(Mode::Royale, None).unwrap();
    let mut s = GameState::new(&cfg);
    for i in 0..4 {
        s.set_snake(&cfg, i, &[(i as i32 * 2, 0); 3], 100, None)
            .unwrap();
    }
    let third = (1.0f64 / 3.0) as f32;
    for dead in 0..=3 {
        for i in 0..4 {
            s.alive[i] = i >= dead;
        }
        let f = mode_features(&s, &cfg, Mode::Royale);
        let alive = (4 - dead) as f32;
        for (i, row) in f.iter().enumerate() {
            let own = if i >= dead { 1.0 } else { 0.0 };
            assert_eq!(
                row[col::OPPONENTS_ALIVE],
                (alive - own) * third,
                "dead {dead} seat {i}"
            );
        }
    }
    for (t, phase) in [
        (0, 0),
        (24, 24),
        (25, 0),
        (26, 1),
        (49, 24),
        (50, 0),
        (1003, 3),
    ] {
        s.turn = t;
        let f = mode_features(&s, &cfg, Mode::Royale);
        assert_eq!(
            f[0][col::SHRINK_PHASE],
            phase as f32 * (1.0f64 / 25.0) as f32
        );
        let turn = (t.min(1000)) as f32 * (1.0f64 / 1000.0) as f32;
        assert_eq!(f[3][col::TURN], turn);
    }
}

/// Garbage that could reach the encoder from a request (huge or negative fields, heads far
/// off the board, a dead own seat) must not panic and must give finite values.
#[test]
fn encoder_survives_garbage_states() {
    for mode in Mode::ALL {
        let cfg = GameConfig::for_mode(mode, None).unwrap();
        let enc = Encoder::new(cfg, true);
        let n = cfg.num_snakes();
        let mut s = GameState::new(&cfg);
        for i in 0..n {
            s.head[i] = [i32::MIN + i as i32, i32::MAX - i as i32];
            s.length[i] = i32::MAX;
            s.health[i] = i32::MIN;
            s.alive[i] = i % 2 == 0;
            s.body[i] = [i16::MIN; 121];
            s.body[i][7] = i16::MAX;
        }
        s.hazard = [i16::MAX; 121];
        s.food = u128::MAX;
        for turn in [i32::MIN, -1, 0, i32::MAX] {
            s.turn = turn;
            let mut boards = vec![0.0; n * OBS_LEN];
            let mut feats = vec![0.0; n * NUM_FEATURES];
            assert_eq!(enc.encode_into(&s, &mut boards, &mut feats), Ok(n));
            assert!(boards.iter().chain(&feats).all(|v| v.is_finite()));
        }
    }
}

/// Bad configs given through the public `NetConfig` fields, and inputs of the wrong size, must
/// be errors, not panics.
#[test]
fn network_rejects_bad_input_without_panicking() {
    for json in [
        r#"{"conv_channels":[],"conv_strides":[]}"#,
        r#"{"conv_channels":[16],"conv_strides":[0]}"#,
        r#"{"conv_channels":[16,4],"conv_strides":[2]}"#,
        r#"{"hidden":0}"#,
        r#"{"hidden":-3}"#,
        r#"{"conv_channels":[1e3]}"#,
        r#"{"features":null}"#,
        r#"{"algorithm":"az_modes","conv_channels":[99999]}"#,
        "",
        "[]",
    ] {
        assert!(NetConfig::from_json(json).is_err(), "{json}");
    }
    // Integer-valued floats in config.json (Python's json.dump writes 1 for a float field set to
    // the int 1) must parse.
    let c = NetConfig::from_json(
        r#"{"algorithm":"az_modes","c_puct":1,"fpu_reduction":0,"draw_value":-1}"#,
    )
    .unwrap();
    assert_eq!((c.c_puct, c.fpu_reduction, c.draw_value), (1.0, 0.0, -1.0));
}

/// Arrays of the right shapes for a net whose convolutions are `convs` (cout, stride), hidden
/// width `hid`, all features.
fn arrays_for(
    convs: &[(usize, usize)],
    hid: usize,
) -> std::collections::HashMap<String, npy::Array> {
    let mut m = std::collections::HashMap::new();
    let mut put = |k: String, shape: Vec<usize>| {
        let n = shape.iter().product();
        m.insert(
            k,
            npy::Array {
                shape,
                data: vec![0.01; n],
            },
        );
    };
    let (mut h, mut cin) = (21usize, 13usize);
    for (i, &(cout, stride)) in convs.iter().enumerate() {
        put(format!("conv{i}/w"), vec![3, 3, cin, cout]);
        put(format!("conv{i}/b"), vec![cout]);
        (h, cin) = (h.div_ceil(stride.max(1)), cout);
    }
    put("hidden/w".into(), vec![h * h * cin, hid]);
    put("hidden/b".into(), vec![hid]);
    put("feat/w".into(), vec![7, hid]);
    put("policy/w".into(), vec![hid, 4]);
    put("policy/b".into(), vec![4]);
    put("value/w".into(), vec![hid, 1]);
    put("value/b".into(), vec![1]);
    m
}

/// `NetConfig`'s fields are public, so a hand-built config bypasses `from_json`'s validation.
/// `Network::from_arrays` used to panic (divide by zero on a zero stride, `convs.len() - 1`
/// underflow on no convolutions) or silently drop layers (`zip` of channel and stride lists of
/// different lengths); it now validates the config again and returns an error.
/// (conv_channels, conv_strides, the (cout, stride) the arrays are shaped for).
type Spec = (Vec<usize>, Vec<usize>, Vec<(usize, usize)>);

#[test]
fn from_arrays_validates_hand_built_configs() {
    let base = NetConfig::from_json(r#"{"algorithm":"az_modes"}"#).unwrap();
    let cases: [Spec; 3] = [
        (vec![], vec![], vec![]),
        (vec![16], vec![0], vec![(16, 0)]),
        (vec![16, 32], vec![2], vec![(16, 2)]),
    ];
    for (channels, strides, arrays) in cases {
        let mut cfg = base.clone();
        cfg.conv_channels = channels.clone();
        cfg.conv_strides = strides.clone();
        let m = arrays_for(&arrays, cfg.hidden);
        let r = std::panic::catch_unwind(move || Network::from_arrays(cfg, m).is_err());
        assert!(
            matches!(r, Ok(true)),
            "from_arrays({channels:?}, {strides:?}): {}",
            if r.is_err() { "panicked" } else { "accepted" }
        );
    }
}

/// Byte-level corruption of a real `params.npz` never panics the loader.
#[test]
fn npz_loader_survives_corruption() {
    let path = root().join("nets").join("odd_a").join("params.npz");
    let Ok(good) = std::fs::read(&path) else {
        eprintln!("SKIP: {} missing", path.display());
        return;
    };
    assert!(npy::parse_npz(&good).is_ok());
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for k in 0..3000 {
        let mut b = good.clone();
        match k % 3 {
            0 => {
                let n = (next() % b.len() as u64) as usize;
                b.truncate(n);
            }
            1 => {
                for _ in 0..1 + next() % 4 {
                    let i = (next() % b.len() as u64) as usize;
                    b[i] = next() as u8;
                }
            }
            _ => {
                // Hit the headers: the first 400 bytes and the central directory at the end.
                let i = if next() % 2 == 0 {
                    (next() % 400) as usize
                } else {
                    b.len() - 1 - (next() % 300) as usize
                };
                b[i] = next() as u8;
            }
        }
        let r = std::panic::catch_unwind(|| {
            let _ = npy::parse_npz(&b);
        });
        assert!(r.is_ok(), "parse_npz panicked on mutation {k}");
    }
}

/// Non-finite and huge inputs reach the network without a panic.
#[test]
fn network_survives_non_finite_inputs() {
    let path = root().join("nets").join("odd_a");
    let Ok(net) = Network::load(&path) else {
        eprintln!("SKIP: {} missing", path.display());
        return;
    };
    for v in [f32::NAN, f32::INFINITY, -f32::INFINITY, f32::MAX, -0.0] {
        let obs = vec![v; 2 * OBS_LEN];
        let feats = vec![v; 2 * NUM_FEATURES];
        let out = net.forward(&obs, &feats).unwrap();
        assert_eq!(out.values.len(), 2);
    }
}
