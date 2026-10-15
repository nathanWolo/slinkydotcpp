//! Parity with the Python server on golden request cases written by `native/golden/api_golden.py`
//! (see its docstring for the format): the config, the mode, the whole game state and the
//! fallback move must match exactly, and requests the Python server cannot play must be errors.
//! Skips when the file is missing.

use serde_json::Value;
use slinky_api::{Request, game_from_request, safe_move_json};
use slinky_rules::{Mode, Ruleset};
use std::path::PathBuf;

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn mode_of(name: &str) -> Mode {
    Mode::from_name(name).unwrap()
}

/// Compares one expected-ok case; returns a description of the first difference.
fn compare(case: &Value, got: &slinky_api::Game) -> Result<(), String> {
    let exp = &case["expect"];
    let (cfg, st) = (&got.config, &got.state);
    let ec = &exp["config"];
    let eq = |what: &str, a: i64, b: i64| {
        if a == b {
            Ok(())
        } else {
            Err(format!("{what}: rust {a}, python {b}"))
        }
    };
    eq("width", cfg.width() as i64, ec["width"].as_i64().unwrap())?;
    eq(
        "height",
        cfg.height() as i64,
        ec["height"].as_i64().unwrap(),
    )?;
    eq(
        "num_snakes",
        cfg.num_snakes() as i64,
        ec["num_snakes"].as_i64().unwrap(),
    )?;
    if cfg.ruleset() != Ruleset::from_name(ec["ruleset"].as_str().unwrap()).unwrap() {
        return Err(format!("ruleset {:?} vs {}", cfg.ruleset(), ec["ruleset"]));
    }
    eq(
        "hazard_damage",
        cfg.hazard_damage_per_turn() as i64,
        ec["hazard_damage"].as_i64().unwrap(),
    )?;
    eq(
        "shrink_every",
        cfg.shrink_every_n_turns() as i64,
        ec["shrink_every"].as_i64().unwrap(),
    )?;
    if got.mode != mode_of(exp["mode"].as_str().unwrap()) {
        return Err(format!("mode {:?} vs {}", got.mode, exp["mode"]));
    }
    let es = &exp["state"];
    let (n, cells) = (cfg.num_snakes(), cfg.cells());
    for s in 0..n {
        let body: Vec<i64> = ints(&es["body"][s]);
        let rb: Vec<i64> = st.body[s][..cells].iter().map(|&b| i64::from(b)).collect();
        if body != rb {
            return Err(format!("body of seat {s}: rust {rb:?} python {body:?}"));
        }
        if st.body[s][cells..].iter().any(|&b| b != 0) {
            return Err(format!("body of seat {s} has cells past the board"));
        }
        let head = ints(&es["head"][s]);
        eq("head x", st.head[s][0].into(), head[0])?;
        eq("head y", st.head[s][1].into(), head[1])?;
        eq(
            "length",
            st.length[s].into(),
            es["length"][s].as_i64().unwrap(),
        )?;
        eq(
            "health",
            st.health[s].into(),
            es["health"][s].as_i64().unwrap(),
        )?;
        eq(
            "alive",
            st.alive[s].into(),
            es["alive"][s].as_bool().unwrap().into(),
        )?;
        eq(
            "last_move",
            st.last_move[s].into(),
            es["last_move"][s].as_i64().unwrap(),
        )?;
        eq(
            "elim_cause",
            st.elim_cause[s].into(),
            es["elim_cause"][s].as_i64().unwrap(),
        )?;
        eq(
            "elim_turn",
            st.elim_turn[s].into(),
            es["elim_turn"][s].as_i64().unwrap(),
        )?;
    }
    let food = ints(&es["food"]);
    for (c, &f) in food.iter().enumerate() {
        eq(&format!("food[{c}]"), st.has_food(c).into(), f)?;
    }
    if st.food >> cells != 0 {
        return Err("food past the board".into());
    }
    let hazard = ints(&es["hazard"]);
    for (c, &hz) in hazard.iter().enumerate() {
        eq(&format!("hazard[{c}]"), st.hazard[c].into(), hz)?;
    }
    if st.hazard[cells..].iter().any(|&h| h != 0) {
        return Err("hazard past the board".into());
    }
    eq("turn", st.turn.into(), es["turn"].as_i64().unwrap())?;
    eq("done", st.done.into(), es["done"].as_bool().unwrap().into())?;
    if cfg.ruleset() == Ruleset::Royale {
        let sides = ints(&es["sides"]);
        let rs: Vec<i64> = st.sides[..cfg.num_sides()]
            .iter()
            .map(|&s| s.into())
            .collect();
        if sides != rs {
            return Err(format!("sides: rust {rs:?} python {sides:?}"));
        }
    } else if !es["sides"].is_null() {
        return Err("python has sides outside royale".into());
    }
    Ok(())
}

#[test]
fn matches_the_python_server() {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let path = PathBuf::from(base).join("api").join("cases.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP golden parity: {} missing (run native/golden/api_golden.py)",
            path.display()
        );
        return;
    };
    let (mut total, mut oks, mut errs, mut safes) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for line in text.lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let name = case["name"].as_str().unwrap();
        let body = serde_json::to_string(&case["request"]).unwrap();
        let starting = case["starting"].as_u64().map(|n| n as usize);
        let exp = &case["expect"];
        total += 1;
        // Never panics, whatever the case says.
        let parsed = Request::from_json(&body);
        let result = parsed
            .as_ref()
            .map_err(|e| e.clone())
            .and_then(|r| game_from_request(r, starting));
        if exp["ok"].as_bool().unwrap() {
            oks += 1;
            match &result {
                Ok(game) => {
                    if let Err(e) = compare(&case, game) {
                        failures.push(format!("{name}: {e}"));
                    }
                }
                Err(e) => failures.push(format!("{name}: rust refused ({e}) but python played")),
            }
        } else {
            errs += 1;
            if let Ok(game) = &result {
                failures.push(format!(
                    "{name}: rust made a {:?} game but python refused ({})",
                    game.mode, exp["why"]
                ));
            }
        }
        if let Some(safe) = exp["safe"].as_i64() {
            safes += 1;
            let got = safe_move_json(&body) as i64;
            if got != safe {
                failures.push(format!("{name}: safe_move rust {got} python {safe}"));
            }
            if let Ok(r) = &parsed
                && slinky_api::safe_move(r) as i64 != got
            {
                failures.push(format!("{name}: typed and JSON safe_move differ"));
            }
        }
    }
    eprintln!("api golden: {total} cases ({oks} states, {errs} refusals), {safes} safe moves");
    assert!(
        failures.is_empty(),
        "{} failures, first ones:\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(total > 100);
}
