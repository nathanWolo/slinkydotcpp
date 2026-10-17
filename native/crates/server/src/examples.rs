//! The startup self-benchmark's requests: the Python server's
//! `example_request(MODE_CONFIGS[mode])` for each mode (`data/examples.json`, written by
//! `py/make_examples.py`), so both servers benchmark on the same positions and advance their
//! move counters by the same 35 moves.

const EXAMPLES: &str = include_str!("../data/examples.json");

/// `(mode, request JSON)` in `modes.MODE_NAMES` order (standard, duel, royale, constrictor).
pub fn requests() -> Vec<(String, String)> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(EXAMPLES) else {
        return Vec::new();
    };
    let Some(modes) = v["modes"].as_array() else {
        return Vec::new();
    };
    modes
        .iter()
        .filter_map(|m| {
            let mode = m.as_str()?;
            let req = v["requests"].get(mode)?;
            Some((mode.to_string(), req.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn has_the_four_modes() {
        let r = super::requests();
        let modes: Vec<&str> = r.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(modes, ["standard", "duel", "royale", "constrictor"]);
        for (_, body) in &r {
            assert!(slinky_api::Request::from_json(body).is_ok());
        }
    }
}
