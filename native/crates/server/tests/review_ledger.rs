//! Adversarial review: the move ledger against Python's `server.MoveLedger`, call by call.
//!
//! `py/ledger_golden.py` drives Python's ledger the way the HTTP handler does over a fuzz of
//! games (plain and wrapped boards, applied/missed/gap answers, `/end` after death, more than
//! `MAX_GAMES` games) and junk requests, and records each call's verdict, exception and the
//! move-log line (without `t`). This test replays the same calls into the Rust ledger and
//! compares everything. Without `$NAZ_GOLDEN/server/ledger.json` it prints SKIP and passes.

use serde_json::Value;
use slinky_api::py::parse;
use slinky_rules::Move;
use slinky_server_native::ledger::{MoveLedger, Verdict};
use std::path::PathBuf;

fn golden() -> Option<Value> {
    let dir = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    let path = PathBuf::from(dir).join("server/ledger.json");
    match std::fs::read_to_string(&path) {
        Ok(t) => Some(serde_json::from_str(&t).expect("ledger.json")),
        Err(_) => {
            eprintln!("SKIP: no {} (run py/ledger_golden.py)", path.display());
            None
        }
    }
}

fn mv(name: &str) -> Move {
    Move::ALL
        .into_iter()
        .find(|m| m.name() == name)
        .expect("a move name")
}

/// Strips the `"t": <t>, ` prefix of a move-log line, as the golden script does.
fn without_t(line: &str) -> String {
    let rest = line.strip_prefix("{\"t\": ").expect("a line starts with t");
    rest.split_once(", ")
        .expect("t, then the rest")
        .1
        .to_string()
}

#[derive(Debug, PartialEq)]
struct Outcome {
    verdict: Option<String>,
    raised: bool,
    line: Option<String>,
}

/// Replays the golden calls; returns the mismatches (index, what Python did, what Rust did).
fn replay(name: &str, calls: &[Value]) -> Vec<(usize, String, Outcome, Outcome)> {
    slinky_server_native::log::set_quiet(true);
    let path = std::env::temp_dir().join(format!(
        "slinky-review-ledger-{}-{name}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let ledger = MoveLedger::new(Some(path.clone()));
    let mut seen = 0usize;
    let mut bad = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let text = c["request"].as_str().unwrap();
        let req = parse(text).expect("the Rust parser reads json.dumps text");
        let res = ledger.record(
            c["snake"].as_str().unwrap(),
            &req,
            c["move"].as_str().map(mv),
            c["agent"].as_str().unwrap(),
            c["ms"].as_f64().unwrap(),
            c["end"].as_bool().unwrap(),
            c["wall"].as_f64(),
            c["colo"].as_str(),
        );
        ledger.flush();
        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect();
        let line = if lines.len() > seen {
            seen = lines.len();
            Some(without_t(lines.last().unwrap()))
        } else {
            None
        };
        let got = Outcome {
            verdict: res
                .as_ref()
                .ok()
                .and_then(|v| v.map(|v: Verdict| v.as_str().to_string())),
            raised: res.is_err(),
            line,
        };
        let want = Outcome {
            verdict: c["verdict"].as_str().map(str::to_string),
            raised: !c["raised"].is_null(),
            line: c["line"].as_str().map(str::to_string),
        };
        if got != want {
            bad.push((i, text.to_string(), want, got));
        }
    }
    let _ = std::fs::remove_file(&path);
    bad
}

/// A request with a board size or head coordinate of 10**40 (beyond i128).
fn huge(text: &str) -> bool {
    text.contains("10000000000000000000000000000000000000000") || text.contains("1e+40")
}

/// Every call of the fuzz except the huge-number cases (see the test below).
#[test]
fn ledger_matches_python_call_by_call() {
    let Some(g) = golden() else { return };
    let calls = g["calls"].as_array().unwrap();
    let bad = replay("all", calls);
    let (big, other): (Vec<_>, Vec<_>) = bad.into_iter().partition(|(_, text, _, _)| huge(text));
    for (i, text, want, got) in other.iter().take(20) {
        eprintln!("call {i}: {text}\n  python {want:?}\n  rust   {got:?}");
    }
    assert!(
        other.is_empty(),
        "{} of {} calls differ from Python",
        other.len(),
        calls.len()
    );
    // The huge-number cases are checked by the test below.
    eprintln!(
        "{} huge-number calls differ (see ledger_huge_numbers_match_python)",
        big.len()
    );
}

/// Fixed (exact integer arithmetic in `steps_by`). Was BUG (ledger.rs `step`): a board width/height or head coordinate outside i128 makes `step`
/// return None, so the verdict is `missed` where Python's exact integer arithmetic says
/// `applied` (width 10**40 with a one-step move, or x = 10**40 moving to 10**40 + 1).
#[test]
fn ledger_huge_numbers_match_python() {
    let Some(g) = golden() else { return };
    let calls = g["calls"].as_array().unwrap();
    let bad = replay("big", calls);
    let big: Vec<_> = bad.iter().filter(|(_, text, _, _)| huge(text)).collect();
    for (i, text, want, got) in &big {
        eprintln!("call {i}: {text}\n  python {want:?}\n  rust   {got:?}");
    }
    assert!(big.is_empty(), "{} huge-number calls differ", big.len());
}
