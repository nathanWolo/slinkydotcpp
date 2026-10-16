//! End-to-end tests of the match runner: determinism across thread counts, resume, self-match
//! parity, the stop rules and the request view. The az_modes parts use the deployed run dir
//! (`$NAZ_RUN`, default `/home/nathan/slinky-serve/az-modes-M-s0`) and skip without it.

use std::path::{Path, PathBuf};

use serde_json::Value;
use slinky_env::EnvConfig;
use slinky_referee::runner::{MatchConfig, games_file, rounds_file, run_matchup};
use slinky_referee::stats::{SeqSpec, score_margin};
use slinky_referee::util::read_jsonl;
use slinky_referee::{Player, PlayerSpec, View};
use slinky_rules::Mode;

fn scratch(name: &str) -> PathBuf {
    let base = if Path::new("/dev/shm").is_dir() {
        PathBuf::from("/dev/shm")
    } else {
        std::env::temp_dir()
    };
    let dir = base.join(format!("naz-referee-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config(mode: Mode, games: u64, threads: usize, out: PathBuf) -> MatchConfig {
    MatchConfig {
        env: EnvConfig::ladder(mode).unwrap(),
        games,
        round_games: 16,
        seed: 3,
        max_turns: 500,
        view: View::State,
        early_stop: true,
        per_seat: false,
        threads,
        target_ci: None,
        target_ci_draws: false,
        seq: None,
        out,
        rerun: false,
        quiet: true,
        fingerprint: "test".into(),
    }
}

fn random() -> Player {
    Player::load(PlayerSpec::RandomLegal).unwrap()
}

fn run_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("NAZ_RUN")
            .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into()),
    );
    if dir.join("params.npz").is_file() {
        Some(dir)
    } else {
        println!("SKIP: no run dir {}", dir.display());
        None
    }
}

/// The games of a games file without the timing fields, sorted by id.
fn games_without_timing(path: &Path) -> Vec<Value> {
    let mut v: Vec<Value> = read_jsonl(path)
        .into_iter()
        .map(|mut g| {
            let o = g.as_object_mut().unwrap();
            o.remove("seconds");
            for p in ["a", "b"] {
                o[p].as_object_mut().unwrap().remove("us");
            }
            g
        })
        .collect();
    v.sort_by_key(|g| g["game"].as_u64().unwrap());
    v
}

#[test]
fn games_do_not_depend_on_the_thread_count() {
    let dir = scratch("threads");
    let (a, b) = (random(), random());
    for mode in [Mode::Duel, Mode::Standard, Mode::Royale, Mode::Constrictor] {
        let one = dir.join(format!("{}-1.jsonl", mode.name()));
        let four = dir.join(format!("{}-4.jsonl", mode.name()));
        let r1 = run_matchup(&config(mode, 48, 1, one.clone()), [&a, &b]).unwrap();
        let r4 = run_matchup(&config(mode, 48, 4, four.clone()), [&a, &b]).unwrap();
        let g1 = games_without_timing(&games_file(&one));
        let g4 = games_without_timing(&games_file(&four));
        assert_eq!(g1.len(), 48);
        assert_eq!(g1, g4, "{}", mode.name());
        for k in [
            "wins",
            "draws",
            "losses",
            "truncated",
            "share",
            "placement",
            "mean_turns",
        ] {
            assert_eq!(r1[k], r4[k], "{k}");
        }
        // A sits in seat g % N.
        let n = EnvConfig::ladder(mode).unwrap().rules().num_snakes() as u64;
        for g in &g1 {
            assert_eq!(
                g["a_seat"].as_u64().unwrap(),
                g["game"].as_u64().unwrap() % n
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_interrupted_or_continued_matchup_reuses_its_games() {
    let dir = scratch("resume");
    let (a, b) = (random(), random());
    let out = dir.join("r.jsonl");
    let first = run_matchup(&config(Mode::Standard, 32, 2, out.clone()), [&a, &b]).unwrap();
    assert_eq!(first["games_played_this_run"], 32);
    // The same command again: the summary is already there.
    let again = run_matchup(&config(Mode::Standard, 32, 2, out.clone()), [&a, &b]).unwrap();
    assert_eq!(again["record_id"], first["record_id"]);
    assert_eq!(read_jsonl(&out).len(), 1);
    // Continued to 64 games: only the new 32 are played.
    let more = run_matchup(&config(Mode::Standard, 64, 2, out.clone()), [&a, &b]).unwrap();
    assert_eq!(more["games_played_this_run"], 32);
    assert_eq!(more["games"], 64);
    // The same as a fresh 64-game run.
    let fresh_out = dir.join("fresh.jsonl");
    let fresh = run_matchup(&config(Mode::Standard, 64, 3, fresh_out.clone()), [&a, &b]).unwrap();
    for k in [
        "wins",
        "draws",
        "losses",
        "truncated",
        "share",
        "placement",
        "mean_turns",
    ] {
        assert_eq!(more[k], fresh[k], "{k}");
    }
    // A torn last line (a kill mid-write) is skipped, and the game is played again.
    let games = games_file(&out);
    let text = std::fs::read_to_string(&games).unwrap();
    let cut = text.trim_end().rfind('\n').unwrap() + 20;
    std::fs::write(&games, &text[..cut]).unwrap();
    let mut cfg = config(Mode::Standard, 64, 2, out.clone());
    cfg.rerun = true;
    let repaired = run_matchup(&cfg, [&a, &b]).unwrap();
    assert_eq!(repaired["games_played_this_run"], 1);
    assert_eq!(repaired["wins"], fresh["wins"]);
    // Rounds are written once each.
    let rounds = read_jsonl(&rounds_file(&out));
    assert_eq!(rounds.len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_self_match_scores_parity_within_its_interval() {
    let dir = scratch("self");
    let (a, b) = (random(), random());
    let duel = run_matchup(&config(Mode::Duel, 2048, 4, dir.join("d.jsonl")), [&a, &b]).unwrap();
    let f = |r: &Value, k: &str| r[k].as_f64().unwrap();
    assert!(
        f(&duel, "ci95_low") < 0.5 && 0.5 < f(&duel, "ci95_high"),
        "{duel}"
    );
    assert!(f(&duel, "elo_ci95_low") < 0.0 && 0.0 < f(&duel, "elo_ci95_high"));
    for mode in [Mode::Standard, Mode::Royale, Mode::Constrictor] {
        let r = run_matchup(
            &config(mode, 2048, 4, dir.join(format!("{}.jsonl", mode.name()))),
            [&a, &b],
        )
        .unwrap();
        assert!(
            (f(&r, "share") - 0.25).abs() < f(&r, "share_ci95"),
            "{}: {r}",
            mode.name()
        );
        assert!(
            (f(&r, "placement") - 2.5).abs() < f(&r, "placement_ci95"),
            "{}: {r}",
            mode.name()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn target_ci_stops_at_the_requested_precision() {
    let dir = scratch("target");
    let (a, b) = (random(), random());
    let mut cfg = config(Mode::Duel, 4096, 4, dir.join("t.jsonl"));
    cfg.round_games = 64;
    cfg.target_ci = Some(0.05);
    let r = run_matchup(&cfg, [&a, &b]).unwrap();
    let n = r["games"].as_u64().unwrap();
    assert!(r["stopped_early"].as_bool().unwrap());
    assert!(n < 4096 && n.is_multiple_of(64), "{n}");
    let half = (r["ci95_high"].as_f64().unwrap() - r["ci95_low"].as_f64().unwrap()) / 2.0;
    assert!(half <= 0.05);
    // One round fewer would not have been precise enough: a Wilson half-width near 0.5
    // shrinks like 1 / sqrt(n).
    assert!(0.98 / ((n - 64) as f64).sqrt() > 0.05, "{n}");
    assert_eq!(read_jsonl(&rounds_file(&cfg.out)).len() as u64, n / 64);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sequential_stops_a_lopsided_matchup_early() {
    let Some(run) = run_dir() else { return };
    let dir = scratch("seq");
    let spec = PlayerSpec::parse(&format!("az_modes:{}:simulations=8", run.display())).unwrap();
    let a = Player::load(spec).unwrap();
    let b = random();
    let mut cfg = config(Mode::Duel, 512, 4, dir.join("s.jsonl"));
    cfg.round_games = 32;
    cfg.seq = Some(SeqSpec {
        p0: 0.5,
        margin: score_margin(35.0),
        alpha: 0.05,
        beta: 0.05,
        duel: true,
    });
    let r = run_matchup(&cfg, [&a, &b]).unwrap();
    assert_eq!(r["seq_decision"], "A>B", "{r}");
    let n = r["games"].as_u64().unwrap();
    assert!(n <= 96 && n.is_multiple_of(32), "{n}");
    assert_eq!(r["rounds_played"].as_u64().unwrap() * 32, n);
    assert!(r["cs95_low"].as_f64().unwrap() > 0.5);
    assert!(r["elo"].as_f64().unwrap() > 200.0);
    // Games of later rounds that were still running were abandoned, not counted.
    let games = read_jsonl(&games_file(&cfg.out));
    assert!(games.len() as u64 >= n);
    // Timing: one search per turn for A, 8 simulations each.
    let t = &r["timing"]["a"];
    assert_eq!(t["sims_p50"], 8.0);
    assert!(t["p50_ms"].as_f64().unwrap() > 0.0);
    assert!(t["p99_ms"].as_f64().unwrap() >= t["p50_ms"].as_f64().unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn request_view_plays_every_mode_without_errors() {
    let Some(run) = run_dir() else { return };
    let dir = scratch("request");
    let spec = PlayerSpec::parse(&format!("az_modes:{}:simulations=4", run.display())).unwrap();
    let a = Player::load(spec.clone()).unwrap();
    let b = Player::load(spec).unwrap();
    for mode in [Mode::Duel, Mode::Standard, Mode::Royale, Mode::Constrictor] {
        let mut cfg = config(mode, 4, 4, dir.join(format!("{}.jsonl", mode.name())));
        cfg.view = View::Request;
        cfg.max_turns = 60;
        let r = run_matchup(&cfg, [&a, &b]).unwrap();
        assert_eq!(r["games"], 4);
        assert_eq!(r["timing"]["a"]["errors"], 0, "{r}");
        assert_eq!(r["timing"]["b"]["errors"], 0, "{r}");
        // B's seats are asked one by one: at least as many searches as A in 4-snake games.
        let n = EnvConfig::ladder(mode).unwrap().rules().num_snakes();
        if n > 2 {
            let moves = |p: &str| r["timing"][p]["moves"].as_u64().unwrap();
            assert!(moves("b") > moves("a"), "{r}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
