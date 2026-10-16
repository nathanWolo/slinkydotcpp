//! Shared by the examples: the realistic mid-game positions of the agent golden cases
//! (`$NAZ_GOLDEN/search/agent.jsonl`, written by `native/golden/search_golden.py --only agent`).

#![allow(dead_code)]

use serde_json::Value;
use slinky_rules::{GameConfig, GameState, Ruleset};
use std::collections::BTreeMap;

fn u(v: &Value) -> i64 {
    v.as_i64().unwrap()
}

pub fn parse(v: &Value) -> (GameConfig, GameState) {
    let g = &v["game"];
    let cfg = GameConfig::new(
        u(&g["width"]) as usize,
        u(&g["height"]) as usize,
        u(&g["num_snakes"]) as usize,
        Ruleset::from_name(g["ruleset"].as_str().unwrap()).unwrap(),
    )
    .unwrap()
    .with_hazard_damage(u(&g["hazard_damage"]) as i32)
    .with_shrink_every(u(&g["shrink_every"]) as i32)
    .unwrap();
    let st = &v["state"];
    let mut s = GameState::new(&cfg);
    s.turn = u(&st["turn"]) as i32;
    s.done = u(&st["done"]) != 0;
    for p in 0..cfg.num_snakes() {
        s.head[p] = [u(&st["head"][p][0]) as i32, u(&st["head"][p][1]) as i32];
        s.length[p] = u(&st["length"][p]) as i32;
        s.health[p] = u(&st["health"][p]) as i32;
        s.alive[p] = u(&st["alive"][p]) != 0;
        s.elim_turn[p] = u(&st["elim_turn"][p]) as i32;
        for (c, x) in st["body"][p].as_array().unwrap().iter().enumerate() {
            s.body[p][c] = u(x) as i16;
        }
    }
    for (k, x) in st["sides"].as_array().unwrap().iter().enumerate() {
        s.sides[k] = u(x) as i8;
    }
    for c in st["food"].as_array().unwrap() {
        s.food |= 1u128 << u(c);
    }
    for (c, x) in st["hazard"].as_array().unwrap().iter().enumerate() {
        s.hazard[c] = u(x) as i16;
    }
    (cfg, s)
}

/// Up to `per` positions per config name (duel, standard, royale, constrictor, standard3,
/// royale2, constrictor2), in file order.
pub fn positions(per: usize) -> Result<BTreeMap<String, Vec<(GameConfig, GameState)>>, String> {
    let golden = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let path = format!("{golden}/search/agent.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut by: BTreeMap<String, Vec<(GameConfig, GameState)>> = BTreeMap::new();
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let list = by
            .entry(v["config"].as_str().unwrap_or("?").to_string())
            .or_default();
        if list.len() < per {
            list.push(parse(&v));
        }
    }
    Ok(by)
}

/// The value at quantile `q` (0 to 1) of `xs` (sorted in place).
pub fn quantile(xs: &mut [f64], q: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    if xs.is_empty() {
        return f64::NAN;
    }
    let i = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[i]
}
