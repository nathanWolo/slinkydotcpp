//! Distribution tests against slinky's env: whole random-legal games of each config in
//! `$NAZ_GOLDEN/env/dist/*.json` (written by `native/golden/env_golden.py --only dist`) are
//! replayed natively, and every aggregate (start positions, starting food, food per turn, spawn
//! rates, royale sides and zone timing, game lengths, outcomes, elimination causes) must agree
//! within the tolerances of `common/dist.rs`. Skips when the files are missing.
//!
//! `NAZ_ENV_GAMES` sets the number of native games per config (default 40,000).

mod common;

use common::dist::{Dist, compare, env_of, table};
use std::path::PathBuf;

#[test]
fn distributions_match_slinky() {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let dir = PathBuf::from(base).join("env").join("dist");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!(
            "SKIP: no env distribution golden files at {}",
            dir.display()
        );
        return;
    };
    let games: u64 = std::env::var("NAZ_ENV_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40_000);
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!(
            "SKIP: no env distribution golden files at {}",
            dir.display()
        );
    }
    let mut failed = Vec::new();
    for (k, path) in files.iter().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let env = env_of(&v);
        let py = Dist::from_json(&v);
        let rs = Dist::simulate(&env, games, 1000 + k as u64);
        let rows = compare(&py, &rs, env.rules());
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        eprintln!("{}", table(&name, &rows, false));
        if rows.iter().any(|r| !r.ok()) {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "out of tolerance: {failed:?}");
}
