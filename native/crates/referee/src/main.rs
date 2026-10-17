//! `naz-match`: strength tests between native agents (see the crate docs and `--help`).

use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::Value;
use slinky_env::EnvConfig;
use slinky_referee::runner::{MatchConfig, describe, run_matchup};
use slinky_referee::stats::{SeqSpec, score_margin};
use slinky_referee::util::{exe_fingerprint, read_jsonl};
use slinky_referee::{Player, PlayerSpec, View};
use slinky_rules::Mode;

const HELP: &str = "\
naz-match: play native agents against each other in the ladder modes.

Usage:
  naz-match --a SPEC --b SPEC [options]
  naz-match --table FILE

Agents (SPEC):
  random_legal
  az_modes:<run dir>[:simulations=N | :time_ms=T[:max_simulations=N]][:threads=K][:memory_mb=M]
                [:early_stop=1][:reuse=1][:int8=1]
  <run dir>                              (= az_modes:<run dir>:simulations=256)

Options:
  --mode LIST         duel,standard,royale,constrictor or all (default all); ladder configs
  --games N           games per mode (default 256; the maximum with a stop rule), rounded up
                      to whole seat rotations
  --round-games N     games per round (default 64): progress, checkpoints and stop checks
  --seed S            match seed (default 0); game g is fixed by (seed, g)
  --threads N         games at once (default: all cores)
  --max-turns N       harness cut-off, a draw if A is alive with others (default 500)
  --view state|request
                      state: search the true state (default, as slinky's strength.py);
                      request: each seat gets the ladder's /move request (one search per seat)
  --per-agent         4 snakes: B searches once per turn for its three seats, as slinky's
                      run_match (biased against A: a self-match scores below parity). The
                      default (--per-seat) searches each seat separately with its own noise
  --full-games        play 4-snake games to the end (default: stop once A is dead or alone;
                      the statistics are the same)
  --out FILE          summary JSONL (default naz-match.jsonl); also FILE.games.jsonl and
                      FILE.rounds.jsonl. Rerunning the same command resumes.
  --rerun             play a matchup again even if FILE has its summary
  --sequential        stop when A>B, B>A or 'within the margin' is established (eval-speed
                      betting tests; alpha = beta = 0.05)
  --seq-elo E         duel margin in Elo (default 35)
  --seq-p0 P --seq-margin M
                      A's score at equal strength and the margin (required for 4 snakes)
  --seq-alpha A --seq-beta B
  --target-ci W       stop once the score's 95% Wilson half-width is at most W (>= 64 games)
  --target-ci-draws   use the draw-aware interval for --target-ci
  --quiet             fewer progress lines
";

struct Args {
    a: Option<String>,
    b: Option<String>,
    modes: Vec<Mode>,
    games: u64,
    round_games: u64,
    seed: u64,
    threads: usize,
    max_turns: i32,
    view: View,
    full_games: bool,
    per_seat: bool,
    out: PathBuf,
    rerun: bool,
    sequential: bool,
    seq_elo: f64,
    seq_p0: Option<f64>,
    seq_margin: Option<f64>,
    seq_alpha: f64,
    seq_beta: f64,
    target_ci: Option<f64>,
    target_ci_draws: bool,
    quiet: bool,
    table: Option<PathBuf>,
}

fn parse_modes(text: &str) -> Result<Vec<Mode>, String> {
    if text == "all" {
        return Ok(vec![
            Mode::Duel,
            Mode::Standard,
            Mode::Royale,
            Mode::Constrictor,
        ]);
    }
    let mut out = Vec::new();
    for name in text.split(',') {
        let m = Mode::from_name(name.trim()).map_err(|e| format!("--mode: {e}"))?;
        if !out.contains(&m) {
            out.push(m);
        }
    }
    Ok(out)
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        a: None,
        b: None,
        modes: parse_modes("all")?,
        games: 256,
        round_games: 64,
        seed: 0,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        max_turns: 500,
        view: View::State,
        full_games: false,
        per_seat: true,
        out: PathBuf::from("naz-match.jsonl"),
        rerun: false,
        sequential: false,
        seq_elo: 35.0,
        seq_p0: None,
        seq_margin: None,
        seq_alpha: 0.05,
        seq_beta: 0.05,
        target_ci: None,
        target_ci_draws: false,
        quiet: false,
        table: None,
    };
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
            v.parse().map_err(|_| format!("{name}: bad value {v:?}"))
        }
        match flag.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--a" => a.a = Some(value("--a")?),
            "--b" => a.b = Some(value("--b")?),
            "--mode" | "--modes" => a.modes = parse_modes(&value("--mode")?)?,
            "--games" => a.games = num("--games", &value("--games")?)?,
            "--round-games" => a.round_games = num("--round-games", &value("--round-games")?)?,
            "--seed" => a.seed = num("--seed", &value("--seed")?)?,
            "--threads" => a.threads = num::<usize>("--threads", &value("--threads")?)?.max(1),
            "--max-turns" => a.max_turns = num("--max-turns", &value("--max-turns")?)?,
            "--view" => {
                let v = value("--view")?;
                a.view =
                    View::from_name(&v).ok_or(format!("--view: {v:?} is not state or request"))?;
            }
            "--full-games" => a.full_games = true,
            "--per-seat" => a.per_seat = true,
            "--per-agent" => a.per_seat = false,
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--rerun" => a.rerun = true,
            "--sequential" => a.sequential = true,
            "--seq-elo" => a.seq_elo = num("--seq-elo", &value("--seq-elo")?)?,
            "--seq-p0" => a.seq_p0 = Some(num("--seq-p0", &value("--seq-p0")?)?),
            "--seq-margin" => a.seq_margin = Some(num("--seq-margin", &value("--seq-margin")?)?),
            "--seq-alpha" => a.seq_alpha = num("--seq-alpha", &value("--seq-alpha")?)?,
            "--seq-beta" => a.seq_beta = num("--seq-beta", &value("--seq-beta")?)?,
            "--target-ci" => a.target_ci = Some(num("--target-ci", &value("--target-ci")?)?),
            "--target-ci-draws" => a.target_ci_draws = true,
            "--quiet" => a.quiet = true,
            "--table" => a.table = Some(PathBuf::from(value("--table")?)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if a.table.is_some() {
        return Ok(a);
    }
    if a.a.is_none() || a.b.is_none() {
        return Err("--a and --b are required".into());
    }
    if a.games == 0 || a.round_games == 0 || a.max_turns < 1 {
        return Err("--games, --round-games and --max-turns must be positive".into());
    }
    if a.sequential && a.target_ci.is_some() {
        return Err("--sequential cannot be combined with --target-ci".into());
    }
    if a.target_ci_draws && a.target_ci.is_none() {
        return Err("--target-ci-draws needs --target-ci".into());
    }
    for (name, v) in [("--seq-alpha", a.seq_alpha), ("--seq-beta", a.seq_beta)] {
        if !(v > 0.0 && v < 1.0) {
            return Err(format!("{name} must lie in (0, 1)"));
        }
    }
    if let Some(t) = a.target_ci
        && (t.is_nan() || t <= 0.0)
    {
        return Err("--target-ci must be positive".into());
    }
    Ok(a)
}

/// The sequential spec for a game with `n` seats.
fn seq_spec(a: &Args, n: usize) -> Result<Option<SeqSpec>, String> {
    if !a.sequential {
        return Ok(None);
    }
    let (p0, margin, duel) = if n == 2 && a.seq_p0.is_none() && a.seq_margin.is_none() {
        if a.seq_elo.is_nan() || a.seq_elo <= 0.0 {
            return Err("--seq-elo must be positive".into());
        }
        (0.5, score_margin(a.seq_elo), true)
    } else {
        match (a.seq_p0, a.seq_margin) {
            (Some(p0), Some(m)) => (p0, m, false),
            _ => {
                return Err(format!(
                    "--sequential with {n} snakes needs --seq-p0 and --seq-margin (A's score at \
                     equal strength, a draw counting 1/2, and the margin in score units)"
                ));
            }
        }
    };
    if !(margin > 0.0 && p0 - margin > 0.0 && p0 + margin < 1.0) {
        return Err(format!(
            "--sequential needs 0 < p0 - margin and p0 + margin < 1 (p0 {p0}, margin {margin})"
        ));
    }
    Ok(Some(SeqSpec {
        p0,
        margin,
        alpha: a.seq_alpha,
        beta: a.seq_beta,
        duel,
    }))
}

fn table(records: &[Value]) -> String {
    let mut s = String::from(
        "| mode | A | B | games | W/D/L | score [95%] | Elo / share / placement | stop | A ms p50/p99 | B ms p50/p99 |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in records {
        let f = |k: &str| r.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
        let u = |k: &str| r.get(k).and_then(Value::as_u64).unwrap_or(0);
        let st = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("?").to_string();
        let strength = if r.get("elo").is_some_and(Value::is_number) {
            format!(
                "{:+.0} [{:+.0}, {:+.0}]",
                f("elo"),
                f("elo_ci95_low"),
                f("elo_ci95_high")
            )
        } else {
            format!(
                "{:.3} +- {:.3} / {:.2} +- {:.2}",
                f("share"),
                f("share_ci95"),
                f("placement"),
                f("placement_ci95")
            )
        };
        let stop = match r.get("seq_decision").and_then(Value::as_str) {
            Some(d) => format!("seq {d}"),
            None if r.get("stopped_early").and_then(Value::as_bool) == Some(true) => "ci".into(),
            None => String::new(),
        };
        let t = |p: &str, k: &str| {
            r["timing"][p]
                .get(k)
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN)
        };
        s += &format!(
            "| {} | {} | {} | {} | {}/{}/{} | {:.3} [{:.3}, {:.3}] | {} | {} | {:.1}/{:.1} | {:.1}/{:.1} |\n",
            st("mode"),
            st("a_short"),
            st("b_short"),
            u("games"),
            u("wins"),
            u("draws"),
            u("losses"),
            f("score"),
            f("ci95_low"),
            f("ci95_high"),
            strength,
            stop,
            t("a", "p50_ms"),
            t("a", "p99_ms"),
            t("b", "p50_ms"),
            t("b", "p99_ms"),
        );
    }
    s
}

fn run(args: &Args) -> Result<(), String> {
    if let Some(path) = &args.table {
        let records: Vec<Value> = read_jsonl(path)
            .into_iter()
            .filter(|r| r.get("tool").and_then(Value::as_str) == Some("naz-match"))
            .collect();
        print!("{}", table(&records));
        return Ok(());
    }
    let spec_a = PlayerSpec::parse(args.a.as_deref().unwrap_or_default())?;
    let spec_b = PlayerSpec::parse(args.b.as_deref().unwrap_or_default())?;
    let a = Player::load(spec_a)?;
    let b = Player::load(spec_b)?;
    let fingerprint = exe_fingerprint();
    // Check every mode's settings before playing any.
    let mut configs = Vec::new();
    for &mode in &args.modes {
        let env = EnvConfig::ladder(mode).map_err(|e| format!("{}: {e}", mode.name()))?;
        let n = env.rules().num_snakes();
        let seq = seq_spec(args, n)?;
        for p in [&a, &b] {
            if let Some(agent) = &p.agent {
                let mut rng = slinky_env::Rng::new(0);
                let state = env.reset(&mut rng);
                let g = slinky_api::Game {
                    config: *env.rules(),
                    state,
                    mode: env.mode(),
                };
                agent
                    .check_game(&g)
                    .map_err(|e| format!("{} cannot play {}: {e}", p.name(), mode.name()))?;
            }
        }
        configs.push(MatchConfig {
            env,
            games: args.games,
            round_games: args.round_games,
            seed: args.seed,
            max_turns: args.max_turns,
            view: args.view,
            early_stop: !args.full_games,
            per_seat: args.per_seat,
            threads: args.threads,
            target_ci: args.target_ci,
            target_ci_draws: args.target_ci_draws,
            seq,
            out: args.out.clone(),
            rerun: args.rerun,
            quiet: args.quiet,
            fingerprint: fingerprint.clone(),
        });
    }
    let mut records = Vec::new();
    for cfg in &configs {
        records.push(run_matchup(cfg, [&a, &b])?);
    }
    if records.len() > 1 {
        eprintln!();
        for r in &records {
            eprintln!(
                "{}: {}",
                r.get("mode").and_then(Value::as_str).unwrap_or("?"),
                describe(r)
            );
        }
    }
    print!("{}", table(&records));
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("naz-match: {e}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("naz-match: {e}");
            ExitCode::FAILURE
        }
    }
}
