//! A matchup: many games of A against B in one game config, on a pool of threads (one game per
//! thread at a time), in rounds of whole seat rotations, with resumable output and optional
//! early stopping.
//!
//! **Files.** `<out>` gets one summary line per finished matchup (as strength.py's results
//! file). `<out>.games.jsonl` gets one line per finished game as soon as it ends, and
//! `<out>.rounds.jsonl` one line per finished round (its W/D/L row, as the eval-speed branch's
//! `<out>.rounds.jsonl`). A matchup whose games file already holds games (same matchup id, same
//! code fingerprint) plays only the missing ones, so an interrupted run resumes by running the
//! same command, and a finished 1024-game matchup continued with `--games 4096` reuses its
//! games. A summary line whose `record_id` (matchup id + games + stop settings) is already in
//! `<out>` is not run again unless `--rerun`.
//!
//! **Rounds and stopping.** Games are numbered `0 .. games - 1` (rounded up to whole seat
//! rotations) and grouped into rounds of `--round-games` (also whole rotations). After each
//! complete round, in order, the stop rules run: `--target-ci` (Wilson half-width of the score,
//! or the draw-aware interval with `--target-ci-draws`, after at least 64 games) or
//! `--sequential` (the eval-speed betting tests, [`crate::stats::seq_test`]). On a stop, games
//! of later rounds still running are abandoned and the statistics cover exactly the rounds up
//! to the stop: games `0 .. n - 1`, the games a fixed run of `n` games plays.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use serde_json::{Value, json};
use slinky_env::{EnvConfig, MapKind};

use crate::game::{GameRecord, Setup, View, Workspace, play_game};
use crate::player::Player;
use crate::stats::{self, Decision, MIN_GAMES_TO_STOP, SEQ_TESTS, SEQ_VERSION, SeqSpec, SeqState};
use crate::util::{append_jsonl, fmt_duration, hash_hex, read_jsonl, utc_now};

/// Version of the records (games, rounds and summary lines).
pub const SCHEMA: u32 = 1;

/// Columns of a round row (`rounds_detail`); the first three are what the sequential test reads.
pub const ROUND_COLUMNS: [&str; 8] = [
    "wins",
    "draws",
    "losses",
    "truncated",
    "turns",
    "share_sum",
    "placement_sum",
    "game_seconds",
];

/// The settings of one matchup.
pub struct MatchConfig {
    /// The game.
    pub env: EnvConfig,
    /// Games wanted (rounded up to whole seat rotations); with a stop rule, the maximum.
    pub games: u64,
    /// Games per round (rounded up to whole seat rotations).
    pub round_games: u64,
    /// The match seed.
    pub seed: u64,
    /// Harness cut-off in turns.
    pub max_turns: i32,
    /// What the agents see.
    pub view: View,
    /// End games once decided between the agents.
    pub early_stop: bool,
    /// Search once per seat instead of once per agent ([`crate::Setup::per_seat`]).
    pub per_seat: bool,
    /// Worker threads (games at once).
    pub threads: usize,
    /// `--target-ci`: stop once the score's 95% half-width is at most this.
    pub target_ci: Option<f64>,
    /// `--target-ci-draws`: use the draw-aware interval for `--target-ci`.
    pub target_ci_draws: bool,
    /// `--sequential`.
    pub seq: Option<SeqSpec>,
    /// The summary file.
    pub out: PathBuf,
    /// Run even if the summary file has this matchup.
    pub rerun: bool,
    /// Fewer progress lines.
    pub quiet: bool,
    /// The code fingerprint written to the records.
    pub fingerprint: String,
}

/// The per-game file of a summary file.
pub fn games_file(out: &Path) -> PathBuf {
    out.with_file_name(format!(
        "{}.games.jsonl",
        out.file_name()
            .map_or("naz-match".into(), |f| f.to_string_lossy())
    ))
}

/// The per-round file of a summary file.
pub fn rounds_file(out: &Path) -> PathBuf {
    out.with_file_name(format!(
        "{}.rounds.jsonl",
        out.file_name()
            .map_or("naz-match".into(), |f| f.to_string_lossy())
    ))
}

/// The game config as JSON (part of the matchup identity).
pub fn env_json(env: &EnvConfig) -> Value {
    let r = env.rules();
    json!({
        "mode": env.mode().name(),
        "width": r.width(),
        "height": r.height(),
        "num_snakes": r.num_snakes(),
        "ruleset": r.ruleset().name(),
        "map": match env.map() { MapKind::Standard => "standard", MapKind::Royale => "royale", MapKind::Empty => "empty" },
        "food_spawn_chance": env.food_spawn_chance(),
        "minimum_food": env.minimum_food(),
        "hazard_damage_per_turn": r.hazard_damage_per_turn(),
        "shrink_every_n_turns": r.shrink_every_n_turns(),
    })
}

fn player_json(p: &Player) -> Value {
    json!({"name": p.name(), "weights": p.weights_hash, "deterministic": p.spec.is_deterministic()})
}

/// The matchup identity: the agents, their weights, the game, the cut-off, the seed, the view
/// and early stopping. Games of the same matchup are the same games.
pub fn matchup_id(cfg: &MatchConfig, players: [&Player; 2]) -> String {
    let mut desc = json!({
        "schema": SCHEMA,
        "a": player_json(players[0]),
        "b": player_json(players[1]),
        "game": env_json(&cfg.env),
        "max_turns": cfg.max_turns,
        "seed": cfg.seed,
        "view": cfg.view.name(),
        "early_stop": cfg.early_stop,
    });
    if cfg.per_seat && cfg.view == View::State {
        desc["per_seat"] = json!(true); // (absent otherwise: ids of earlier runs stay valid)
    }
    hash_hex(&desc.to_string())
}

fn seq_json(seq: &Option<SeqSpec>) -> Value {
    match seq {
        None => Value::Null,
        Some(s) => json!({
            "p0": s.p0, "margin": s.margin, "alpha": s.alpha, "beta": s.beta, "duel": s.duel,
            "version": SEQ_VERSION,
        }),
    }
}

/// The record identity: the matchup plus how many games and when it may stop.
pub fn record_id(cfg: &MatchConfig, matchup: &str) -> String {
    let n = cfg.env.rules().num_snakes() as u64;
    let desc = json!({
        "matchup_id": matchup,
        "games": cfg.games.div_ceil(n) * n,
        "round_games": round_size(cfg),
        "target_ci": cfg.target_ci,
        "target_ci_draws": cfg.target_ci_draws,
        "seq": seq_json(&cfg.seq),
    });
    hash_hex(&desc.to_string())
}

fn round_size(cfg: &MatchConfig) -> u64 {
    let n = cfg.env.rules().num_snakes() as u64;
    cfg.round_games.max(n).div_ceil(n) * n
}

/// The rounds `(first game, games)` of a matchup.
pub fn plan_rounds(cfg: &MatchConfig) -> Vec<(u64, u64)> {
    let n = cfg.env.rules().num_snakes() as u64;
    let games = cfg.games.max(1).div_ceil(n) * n;
    let size = round_size(cfg);
    (0..games.div_ceil(size))
        .map(|r| (r * size, size.min(games - r * size)))
        .collect()
}

/// Running totals over complete rounds.
#[derive(Default)]
struct Totals {
    wins: u64,
    draws: u64,
    losses: u64,
    truncated: u64,
    turns: u64,
    shares: Vec<f64>,
    placements: Vec<f64>,
    game_seconds: f64,
    rows: Vec<[f64; 8]>,
}

impl Totals {
    fn add_round(&mut self, games: &[&GameRecord]) -> [f64; 8] {
        let mut row = [0.0; 8];
        for g in games {
            match g.score.outcome {
                crate::outcome::Outcome::Win => row[0] += 1.0,
                crate::outcome::Outcome::Draw => row[1] += 1.0,
                crate::outcome::Outcome::Loss => row[2] += 1.0,
            }
            row[3] += f64::from(u8::from(g.score.truncated));
            row[4] += f64::from(g.turns);
            row[5] += g.score.share;
            row[6] += g.score.placement;
            row[7] += g.seconds;
            self.shares.push(g.score.share);
            self.placements.push(g.score.placement);
        }
        self.wins += row[0] as u64;
        self.draws += row[1] as u64;
        self.losses += row[2] as u64;
        self.truncated += row[3] as u64;
        self.turns += row[4] as u64;
        self.game_seconds += row[7];
        row[7] = (row[7] * 1e3).round() / 1e3;
        self.rows.push(row);
        row
    }

    fn n(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    fn wdl_rounds(&self) -> Vec<(u64, u64, u64)> {
        self.rows
            .iter()
            .map(|r| (r[0] as u64, r[1] as u64, r[2] as u64))
            .collect()
    }
}

fn log(msg: &str) {
    eprintln!("{msg}");
}

fn interval_text(w: u64, d: u64, l: u64) -> String {
    let (s, lo, hi) = stats::score_interval(w, d, l);
    format!("{s:.3} [{lo:.3}, {hi:.3}]")
}

fn elo_text(score: f64) -> String {
    format!("{:+.0}", stats::score_to_elo(score))
}

fn seq_line(spec: &SeqSpec, st: &SeqState, cs: (f64, f64)) -> String {
    let claim = (1.0 / spec.alpha).ln();
    let futile = (1.0 / spec.beta).ln();
    let unit = if spec.duel {
        format!("{} Elo", elo_text(spec.p0 + spec.margin))
    } else {
        format!("+{:.3}", spec.margin)
    };
    let part = |label: &str, i: usize, thr: f64, side: usize, kind: stats::SideEnd| {
        let done = if st.ended[side] == Some(kind) {
            " (accepted)"
        } else {
            ""
        };
        format!("{label} {:.2}/{thr:.2}{done}", st.log_e[i])
    };
    format!(
        "seq: {} | {} | {} | {} | CS95 [{:.3}, {:.3}]",
        part("A>B log-e", 0, claim, 0, stats::SideEnd::Claim),
        part("B>A", 2, claim, 1, stats::SideEnd::Claim),
        part(
            &format!("A not {unit}"),
            1,
            futile,
            0,
            stats::SideEnd::Futile
        ),
        part(
            &format!("B not {unit}"),
            3,
            futile,
            1,
            stats::SideEnd::Futile
        ),
        cs.0,
        cs.1
    )
}

/// Timing statistics of one agent's moves.
fn timing_json(games: &[&GameRecord], p: usize) -> Value {
    let mut us: Vec<f64> = games
        .iter()
        .flat_map(|g| g.moves[p].us.iter().map(|&x| f64::from(x) / 1e3))
        .collect();
    let mut sims: Vec<f64> = games
        .iter()
        .flat_map(|g| g.moves[p].sims.iter().map(|&x| f64::from(x)))
        .collect();
    let errors: u64 = games.iter().map(|g| u64::from(g.moves[p].errors)).sum();
    us.sort_by(f64::total_cmp);
    sims.sort_by(f64::total_cmp);
    let r3 = |x: f64| (x * 1e3).round() / 1e3;
    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let num = |x: f64| {
        if x.is_finite() {
            json!(r3(x))
        } else {
            Value::Null
        }
    };
    json!({
        "moves": us.len(),
        "errors": errors,
        "mean_ms": num(mean(&us)),
        "p50_ms": num(stats::quantile(&us, 0.5)),
        "p90_ms": num(stats::quantile(&us, 0.9)),
        "p99_ms": num(stats::quantile(&us, 0.99)),
        "max_ms": num(us.last().copied().unwrap_or(f64::NAN)),
        "sims_mean": num(mean(&sims)),
        "sims_p50": num(stats::quantile(&sims, 0.5)),
        "sims_min": num(sims.first().copied().unwrap_or(f64::NAN)),
    })
}

fn finite(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// Run (or resume) a matchup and append its summary line. Returns the summary.
pub fn run_matchup(cfg: &MatchConfig, players: [&Player; 2]) -> Result<Value, String> {
    let t_start = Instant::now();
    let n_seats = cfg.env.rules().num_snakes() as u64;
    let mid = matchup_id(cfg, players);
    let rid = record_id(cfg, &mid);
    let tag = format!(
        "[{} {} vs {}]",
        cfg.env.mode().name(),
        players[0].spec.short_name(),
        players[1].spec.short_name()
    );
    if !cfg.rerun
        && let Some(old) = read_jsonl(&cfg.out)
            .into_iter()
            .rev()
            .find(|r| r.get("record_id").and_then(Value::as_str) == Some(rid.as_str()))
    {
        log(&format!(
            "{tag} already in {} (record {rid}); --rerun to play it again",
            cfg.out.display()
        ));
        return Ok(old);
    }
    let rounds = plan_rounds(cfg);
    let total_games = rounds.last().map_or(0, |&(f, s)| f + s);
    let games_path = games_file(&cfg.out);
    let rounds_path = rounds_file(&cfg.out);

    // Games finished by earlier runs of this matchup (same code).
    let mut done: BTreeMap<u64, GameRecord> = BTreeMap::new();
    let mut other_code = 0usize;
    for v in read_jsonl(&games_path) {
        if v.get("matchup_id").and_then(Value::as_str) != Some(mid.as_str()) {
            continue;
        }
        if v.get("code_fingerprint").and_then(Value::as_str) != Some(cfg.fingerprint.as_str()) {
            other_code += 1;
            continue;
        }
        if let Some(g) = GameRecord::from_json(&v)
            && g.game < total_games
            && g.num_snakes as u64 == n_seats
        {
            done.insert(g.game, g);
        }
    }
    let written_rounds: HashSet<(u64, u64)> = read_jsonl(&rounds_path)
        .iter()
        .filter(|v| v.get("matchup_id").and_then(Value::as_str) == Some(mid.as_str()))
        .filter(|v| {
            v.get("code_fingerprint").and_then(Value::as_str) == Some(cfg.fingerprint.as_str())
        })
        .filter_map(|v| Some((v.get("first_game")?.as_u64()?, v.get("games")?.as_u64()?)))
        .collect();
    let deterministic = players.iter().all(|p| p.spec.is_deterministic());
    log(&format!(
        "{tag} matchup {mid}: {total_games} games max in {} rounds of {}, {} threads, view {}, \
         searches per {}, max {} turns{}{}",
        rounds.len(),
        round_size(cfg),
        cfg.threads,
        cfg.view.name(),
        if cfg.per_seat || cfg.view == View::Request {
            "seat"
        } else {
            "agent"
        },
        cfg.max_turns,
        if done.is_empty() {
            String::new()
        } else {
            format!(
                ", {} games loaded from {}",
                done.len(),
                games_path.display()
            )
        },
        if other_code > 0 {
            format!(" ({other_code} games of other code builds ignored)")
        } else {
            String::new()
        },
    ));
    if !deterministic {
        log(&format!(
            "{tag} note: a time budget is not deterministic; games depend on the machine's load"
        ));
    }

    let mut totals = Totals::default();
    let mut next_round = 0usize;
    let mut stop_after: Option<usize> = None; // index of the round the stop rule ended on
    let mut seq_state: Option<SeqState> = None;
    let mut last_progress = Instant::now();
    let play_start = Instant::now();
    let mut played_now = 0u64;

    // Close every complete round, in order; run the stop rules after each.
    let close_rounds = |done: &BTreeMap<u64, GameRecord>,
                        totals: &mut Totals,
                        next_round: &mut usize,
                        stop_after: &mut Option<usize>,
                        seq_state: &mut Option<SeqState>,
                        played_now: u64|
     -> Result<(), String> {
        while stop_after.is_none() && *next_round < rounds.len() {
            let (first, size) = rounds[*next_round];
            let games: Vec<&GameRecord> =
                (first..first + size).filter_map(|g| done.get(&g)).collect();
            if games.len() as u64 != size {
                break;
            }
            let row = totals.add_round(&games);
            if !written_rounds.contains(&(first, size)) {
                append_jsonl(
                    &rounds_path,
                    &json!({
                        "schema": SCHEMA, "matchup_id": mid, "record_id": rid,
                        "code_fingerprint": cfg.fingerprint, "seed": cfg.seed,
                        "first_game": first, "games": size, "row": row,
                        "row_columns": ROUND_COLUMNS, "date": utc_now(),
                    }),
                )
                .map_err(|e| format!("{}: {e}", rounds_path.display()))?;
            }
            let r = *next_round;
            *next_round += 1;
            let n = totals.n();
            let (w, d, l) = (totals.wins, totals.draws, totals.losses);
            let elapsed = play_start.elapsed().as_secs_f64();
            let left = total_games.saturating_sub(n);
            let eta = if played_now > 0 {
                fmt_duration(elapsed / played_now as f64 * left.min(total_games) as f64)
            } else {
                "?".into()
            };
            let extra = if n_seats > 2 {
                let (s, sh) = stats::mean_ci(&totals.shares);
                let (p, ph) = stats::mean_ci(&totals.placements);
                format!(" | share {s:.3} +- {sh:.3}, placement {p:.2} +- {ph:.2}")
            } else {
                format!(" | Elo {}", elo_text(stats::outcome_stats(w, d, l).1))
            };
            if !cfg.quiet || r + 1 == rounds.len() {
                log(&format!(
                    "{tag} round {}/{}: total {n} games, W/D/L {w}/{d}/{l} ({} truncated), \
                     score {}{extra} | {} | ETA {eta}",
                    r + 1,
                    rounds.len(),
                    totals.truncated,
                    interval_text(w, d, l),
                    fmt_duration(elapsed),
                ));
            }
            let last = r + 1 == rounds.len();
            if let Some(t) = cfg.target_ci
                && !last
                && n >= MIN_GAMES_TO_STOP
            {
                let (lo, hi) = if cfg.target_ci_draws {
                    stats::wilson_eff_interval(w, d, l)
                } else {
                    let (_, lo, hi) = stats::score_interval(w, d, l);
                    (lo, hi)
                };
                if (hi - lo) / 2.0 <= t {
                    log(&format!(
                        "{tag} stopping early: CI half-width {:.4} <= {t}",
                        (hi - lo) / 2.0
                    ));
                    *stop_after = Some(r);
                }
            }
            if let Some(spec) = &cfg.seq {
                let st = stats::seq_test(
                    &totals.wdl_rounds(),
                    spec.p0,
                    spec.margin,
                    spec.alpha,
                    spec.beta,
                );
                if !cfg.quiet {
                    log(&format!(
                        "{tag}   {}",
                        seq_line(spec, &st, stats::score_cs95(w, d, l))
                    ));
                }
                let decided = st.decision.is_some();
                *seq_state = Some(st);
                if decided && !last {
                    *stop_after = Some(r);
                }
            }
        }
        Ok(())
    };

    close_rounds(
        &done,
        &mut totals,
        &mut next_round,
        &mut stop_after,
        &mut seq_state,
        0,
    )?;

    if stop_after.is_none() && next_round < rounds.len() {
        let todo: Vec<u64> = (0..total_games).filter(|g| !done.contains_key(g)).collect();
        let next = AtomicUsize::new(0);
        let limit = AtomicU64::new(total_games);
        let setup = Setup {
            env: cfg.env,
            players,
            seed: cfg.seed,
            max_turns: cfg.max_turns,
            view: cfg.view,
            early_stop: cfg.early_stop,
            per_seat: cfg.per_seat,
        };
        let threads = cfg.threads.clamp(1, todo.len().max(1));
        let mut io_error: Option<String> = None;
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel::<GameRecord>();
            for _ in 0..threads {
                let tx = tx.clone();
                let (setup, next, limit, todo) = (&setup, &next, &limit, &todo);
                scope.spawn(move || {
                    let mut ws = Workspace::new(setup);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&g) = todo.get(i) else { break };
                        if g >= limit.load(Ordering::Relaxed) {
                            break;
                        }
                        let abort = || g >= limit.load(Ordering::Relaxed);
                        if let Some(rec) = play_game(setup, &mut ws, g, &abort)
                            && tx.send(rec).is_err()
                        {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            for rec in rx {
                if io_error.is_some() || rec.game >= limit.load(Ordering::Relaxed) {
                    continue;
                }
                if let Err(e) = append_jsonl(&games_path, &rec.to_json(&mid, &cfg.fingerprint)) {
                    io_error = Some(format!("{}: {e}", games_path.display()));
                    limit.store(0, Ordering::Relaxed);
                    continue;
                }
                done.insert(rec.game, rec);
                played_now += 1;
                if let Err(e) = close_rounds(
                    &done,
                    &mut totals,
                    &mut next_round,
                    &mut stop_after,
                    &mut seq_state,
                    played_now,
                ) {
                    io_error = Some(e);
                    limit.store(0, Ordering::Relaxed);
                    continue;
                }
                if let Some(r) = stop_after {
                    let (first, size) = rounds[r];
                    limit.store(first + size, Ordering::Relaxed);
                }
                if !cfg.quiet && last_progress.elapsed().as_secs() >= 60 {
                    last_progress = Instant::now();
                    log(&format!(
                        "{tag} ... {} of {total_games} games finished ({} this run) in {}",
                        done.len(),
                        played_now,
                        fmt_duration(play_start.elapsed().as_secs_f64())
                    ));
                }
            }
        });
        if let Some(e) = io_error {
            return Err(e);
        }
    }

    // The counted games: the rounds closed (all of them, or up to the stop).
    let counted_end = rounds[..next_round].last().map_or(0, |&(f, s)| f + s);
    let counted: Vec<&GameRecord> = (0..counted_end).filter_map(|g| done.get(&g)).collect();
    let (w, d, l) = (totals.wins, totals.draws, totals.losses);
    let (n, score, ci95) = stats::outcome_stats(w, d, l);
    let (_, lo, hi) = stats::score_interval(w, d, l);
    let (cs_lo, cs_hi) = stats::score_cs95(w, d, l);
    let (eff_lo, eff_hi) = stats::wilson_eff_interval(w, d, l);
    let (share, share_ci) = stats::mean_ci(&totals.shares);
    let share_wilson = if n > 0 {
        stats::wilson(share, n as f64)
    } else {
        (f64::NAN, f64::NAN)
    };
    let (placement, placement_ci) = stats::mean_ci(&totals.placements);
    let duel = n_seats == 2;
    let decision = cfg.seq.as_ref().map(|_| {
        seq_state
            .as_ref()
            .and_then(|s| s.decision)
            .map_or("inconclusive", Decision::name)
    });
    let seq_log_e = seq_state.as_ref().map(|s| {
        let mut m = serde_json::Map::new();
        for (k, v) in SEQ_TESTS.iter().zip(s.log_e) {
            m.insert((*k).into(), json!((v * 1e6).round() / 1e6));
        }
        Value::Object(m)
    });
    let seq_tests = seq_state
        .as_ref()
        .map(|s| json!({"a": s.ended[0].map(|e| e.name()), "b": s.ended[1].map(|e| e.name())}));
    let elo = |x: f64| {
        if duel && x.is_finite() {
            json!(stats::score_to_elo(x))
        } else {
            Value::Null
        }
    };
    let record = json!({
        "schema": SCHEMA,
        "tool": "naz-match",
        "record_id": rid,
        "matchup_id": mid,
        "a": players[0].name(),
        "b": players[1].name(),
        "a_short": players[0].spec.short_name(),
        "b_short": players[1].spec.short_name(),
        "a_config": player_json(players[0]),
        "b_config": player_json(players[1]),
        "mode": cfg.env.mode().name(),
        "game": env_json(&cfg.env),
        "view": cfg.view.name(),
        "early_stop": cfg.early_stop,
        "per_seat": cfg.per_seat || cfg.view == View::Request,
        "max_turns": cfg.max_turns,
        "seed": cfg.seed,
        "games_requested": cfg.games,
        "games_max": total_games,
        "games": n,
        "wins": w,
        "draws": d,
        "losses": l,
        "truncated": totals.truncated,
        "score": finite(score),
        "ci95": finite(ci95),
        "ci95_low": finite(lo),
        "ci95_high": finite(hi),
        "cs95_low": cs_lo,
        "cs95_high": cs_hi,
        "wilson_eff_low": eff_lo,
        "wilson_eff_high": eff_hi,
        "elo": elo(score),
        "elo_ci95_low": elo(lo),
        "elo_ci95_high": elo(hi),
        "elo_cs95_low": elo(cs_lo),
        "elo_cs95_high": elo(cs_hi),
        "share": finite(share),
        "share_ci95": finite(share_ci),
        "share_wilson_low": finite(share_wilson.0),
        "share_wilson_high": finite(share_wilson.1),
        "share_parity": 1.0 / n_seats as f64,
        "placement": finite(placement),
        "placement_ci95": finite(placement_ci),
        "placement_parity": (n_seats as f64 + 1.0) / 2.0,
        "mean_turns": if n > 0 { json!(totals.turns as f64 / n as f64) } else { Value::Null },
        "rounds": rounds.len(),
        "rounds_played": next_round,
        "round_games": round_size(cfg),
        "target_ci": cfg.target_ci,
        "target_ci_draws": cfg.target_ci_draws,
        "stopped_early": stop_after.is_some(),
        "seq": seq_json(&cfg.seq),
        "seq_decision": decision,
        "seq_stop_games": cfg.seq.as_ref().map(|_| n),
        "seq_log_e": seq_log_e,
        "seq_tests": seq_tests,
        "timing": {"a": timing_json(&counted, 0), "b": timing_json(&counted, 1)},
        "threads": cfg.threads,
        "games_played_this_run": played_now,
        "wall_seconds": (t_start.elapsed().as_secs_f64() * 1e3).round() / 1e3,
        "game_seconds": (totals.game_seconds * 1e3).round() / 1e3,
        "code_fingerprint": cfg.fingerprint,
        "cpu_count": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "date": utc_now(),
        "rounds_detail": totals.rows,
        "rounds_detail_columns": ROUND_COLUMNS,
    });
    append_jsonl(&cfg.out, &record).map_err(|e| format!("{}: {e}", cfg.out.display()))?;
    log(&format!("{tag} done: {}", describe(&record)));
    Ok(record)
}

/// A one-line description of a summary record.
pub fn describe(r: &Value) -> String {
    let u = |k: &str| r.get(k).and_then(Value::as_u64).unwrap_or(0);
    let f = |k: &str| r.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
    let mut s = format!(
        "score {:.3} [{:.3}, {:.3}] over {} games (W/D/L {}/{}/{}, {} truncated)",
        f("score"),
        f("ci95_low"),
        f("ci95_high"),
        u("games"),
        u("wins"),
        u("draws"),
        u("losses"),
        u("truncated")
    );
    if r.get("elo").is_some_and(Value::is_number) {
        s += &format!(
            ", Elo {:+.0} [{:+.0}, {:+.0}]",
            f("elo"),
            f("elo_ci95_low"),
            f("elo_ci95_high")
        );
    } else {
        s += &format!(
            ", share {:.3} +- {:.3} (parity {:.3}), placement {:.2} +- {:.2} (parity {:.1})",
            f("share"),
            f("share_ci95"),
            f("share_parity"),
            f("placement"),
            f("placement_ci95"),
            f("placement_parity")
        );
    }
    if let Some(dec) = r.get("seq_decision").and_then(Value::as_str) {
        s += &format!(
            "; sequential: {dec} (always-valid 95% CS [{:.3}, {:.3}])",
            f("cs95_low"),
            f("cs95_high")
        );
    }
    for (p, name) in [("a", "A"), ("b", "B")] {
        let t = &r["timing"][p];
        let g = |k: &str| t.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
        s += &format!(
            "; {name} ms/move p50 {:.1} p90 {:.1} p99 {:.1} max {:.1} (sims p50 {:.0})",
            g("p50_ms"),
            g("p90_ms"),
            g("p99_ms"),
            g("max_ms"),
            g("sims_p50")
        );
    }
    s
}
