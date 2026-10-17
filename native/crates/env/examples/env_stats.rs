//! Side-by-side distribution report: slinky's env (golden JSON from
//! `native/golden/env_golden.py --only dist`) against slinky-env, config by config.
//!
//! `cargo run --release -p slinky-env --example env_stats -- [games per config] [--all] [name ...]`
//!
//! Without `--all`, rows within tolerance are summarised (per-turn rows only at turns 0, 25, 50
//! and 100; every row out of tolerance is printed). Reads `$NAZ_GOLDEN/env/dist` (default
//! `/dev/shm/naz-golden`).

#[path = "../tests/common/dist.rs"]
mod dist;

use dist::{Dist, compare, env_of, table};
use std::path::PathBuf;

fn main() {
    let mut games = 40_000u64;
    let mut all = false;
    let mut names = Vec::new();
    for a in std::env::args().skip(1) {
        if a == "--all" {
            all = true;
        } else if let Ok(g) = a.parse() {
            games = g;
        } else {
            names.push(a);
        }
    }
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let dir = PathBuf::from(base).join("env").join("dist");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no golden files at {}", dir.display());
        std::process::exit(2);
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let mut bad = 0;
    for (k, path) in files.iter().enumerate() {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if !names.is_empty() && !names.contains(&name) {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let env = env_of(&v);
        let t0 = std::time::Instant::now();
        let rs = Dist::simulate(&env, games, 1000 + k as u64);
        let secs = t0.elapsed().as_secs_f64();
        let rows = compare(&Dist::from_json(&v), &rs, env.rules());
        bad += rows.iter().filter(|r| !r.ok()).count();
        println!("{}", table(&name, &rows, all));
        println!("(native: {games} games in {secs:.2} s)\n");
    }
    if bad > 0 {
        println!("{bad} rows out of tolerance");
        std::process::exit(1);
    }
}
