//! The engine behind `/move`: the weights of each snake (loaded once per container), the
//! search budget and the tie-break seed policy.
//!
//! # Settings (environment variables)
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `WEIGHTS_DIR` | root of the weights; a snake's run dir is `WEIGHTS_DIR/<subdir>` | `/var/task/weights` |
//! | `SNAKES` | `name=subdir[:simulations=N][:budget_ms=T],...` | `slinky=az-modes` |
//! | `SIMULATIONS` | simulations per move (the cap, when `BUDGET_MS` is set) | 256 (16384 with `BUDGET_MS`) |
//! | `BUDGET_MS` | stop searching this many ms after the request reached the handler | none |
//! | `THREADS` | root-parallel search trees (use the vCPU count; 1 is the plain search) | 1 |
//! | `TIE_SEED` | fixed tie-break seed (testing only) | derived from game id and turn |
//! | `MEMORY_LIMIT_MB` | cap on one search tree | 256 |
//!
//! Without `BUDGET_MS` the search is the faithful 256-simulation search of the Python agent
//! (or `SIMULATIONS` of them): deterministic given the seed. With `BUDGET_MS` the number of
//! simulations depends on the machine's speed (not a parity mode).
//!
//! # Seed policy
//!
//! The Python server draws the tie noise of move `i` from `fold_in(key(0), i)` with a
//! per-process move counter. A Lambda function has many independent containers, so a counter
//! would make the move depend on which container is hit. The seed is instead
//! `tie_seed(game id, turn)`, a fixed hash (FNV-1a, 64 bit, folded to 32): the same request
//! always gets the same move (a retry cannot flip a near-tie), different games and turns get
//! different noise, and every container agrees. The noise is that of `jax.random.key(seed)`,
//! so `naz --seed <tie_seed>` reproduces a Lambda move exactly.
//!
//! # Fallback
//!
//! Any error (a game the network does not play, a bad request, missing weights) and any panic
//! in an unwinding build answers with the safe move (`slinky_api::safe_move`), never an HTTP
//! error.

use slinky_api::{Request, game_from_request, safe_move_json};
use slinky_rules::Move;
use slinky_search::{Agent, MoveBudget, Search, SearchError};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// Simulations per move of the faithful ladder agent.
pub const DEFAULT_SIMULATIONS: usize = slinky_search::LADDER_SIMULATIONS;
/// Simulation cap when only `BUDGET_MS` is set (the deadline is reached first).
pub const BUDGET_SIMULATIONS: usize = 16384;
/// Largest value accepted for `SIMULATIONS`.
pub const MAX_SIMULATIONS: usize = 1 << 20;
/// Largest value accepted for `BUDGET_MS`.
pub const MAX_BUDGET_MS: u64 = 10_000;
/// Largest value accepted for `THREADS`.
pub const MAX_THREADS: usize = 16;
/// Default weights root inside the Lambda package.
pub const DEFAULT_WEIGHTS_DIR: &str = "/var/task/weights";
/// Default `SNAKES`.
pub const DEFAULT_SNAKES: &str = "slinky=az-modes";

/// One snake of the `SNAKES` list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnakeSpec {
    /// URL path prefix.
    pub name: String,
    /// Run dir relative to the weights root.
    pub subdir: String,
    /// Per-snake `simulations=` override.
    pub simulations: Option<usize>,
    /// Per-snake `budget_ms=` override.
    pub budget_ms: Option<u64>,
}

/// All settings, read from the environment (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Weights root.
    pub weights_dir: PathBuf,
    /// Served snakes.
    pub snakes: Vec<SnakeSpec>,
    /// Global `SIMULATIONS`.
    pub simulations: Option<usize>,
    /// Global `BUDGET_MS`.
    pub budget_ms: Option<u64>,
    /// Root-parallel trees.
    pub threads: usize,
    /// Fixed tie-break seed.
    pub tie_seed: Option<u32>,
    /// Memory cap of one tree, bytes.
    pub memory_limit: usize,
}

fn number<T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display>(
    what: &str,
    text: &str,
    min: T,
    max: T,
) -> Result<T, String> {
    match text.trim().parse::<T>() {
        Ok(v) if v >= min && v <= max => Ok(v),
        _ => Err(format!("{what} must be {min} to {max}, not {text:?}")),
    }
}

impl SnakeSpec {
    /// Parses `name=subdir[:simulations=N][:budget_ms=T]`.
    pub fn parse(text: &str) -> Result<SnakeSpec, String> {
        let (name, rest) = text
            .split_once('=')
            .ok_or_else(|| format!("SNAKES item {text:?} is not name=subdir"))?;
        let name = name.trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return Err(format!("snake name {name:?}: use letters, digits . _ -"));
        }
        let mut items = rest.split(':');
        let subdir = items.next().unwrap_or("").trim().to_string();
        if subdir.split('/').any(|p| p == "..") || subdir.starts_with('/') {
            return Err(format!("snake {name}: subdir {subdir:?} must be relative"));
        }
        let mut spec = SnakeSpec {
            name: name.to_string(),
            subdir,
            simulations: None,
            budget_ms: None,
        };
        for item in items {
            let (k, v) = item
                .split_once('=')
                .ok_or_else(|| format!("snake {name}: option {item:?} is not key=value"))?;
            match k.trim() {
                "simulations" => {
                    spec.simulations = Some(number("simulations", v, 1, MAX_SIMULATIONS)?)
                }
                "budget_ms" => spec.budget_ms = Some(number("budget_ms", v, 1, MAX_BUDGET_MS)?),
                other => return Err(format!("snake {name}: unknown option {other:?}")),
            }
        }
        Ok(spec)
    }
}

impl Settings {
    /// Reads the settings through `get` (the environment in production).
    pub fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> Result<Settings, String> {
        let nonempty = |k: &str| get(k).filter(|v| !v.trim().is_empty());
        let snakes_text = nonempty("SNAKES").unwrap_or_else(|| DEFAULT_SNAKES.to_string());
        let snakes = snakes_text
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(SnakeSpec::parse)
            .collect::<Result<Vec<_>, _>>()?;
        if snakes.is_empty() {
            return Err("SNAKES is empty".into());
        }
        let mut names: Vec<&str> = snakes.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err("SNAKES names a snake twice".into());
        }
        Ok(Settings {
            weights_dir: PathBuf::from(
                nonempty("WEIGHTS_DIR").unwrap_or_else(|| DEFAULT_WEIGHTS_DIR.to_string()),
            ),
            snakes,
            simulations: nonempty("SIMULATIONS")
                .map(|v| number("SIMULATIONS", &v, 1, MAX_SIMULATIONS))
                .transpose()?,
            budget_ms: nonempty("BUDGET_MS")
                .map(|v| number("BUDGET_MS", &v, 1, MAX_BUDGET_MS))
                .transpose()?,
            threads: nonempty("THREADS")
                .map(|v| number("THREADS", &v, 1, MAX_THREADS))
                .transpose()?
                .unwrap_or(1),
            tie_seed: nonempty("TIE_SEED")
                .map(|v| number("TIE_SEED", &v, 0, u32::MAX))
                .transpose()?,
            memory_limit: nonempty("MEMORY_LIMIT_MB")
                .map(|v| number("MEMORY_LIMIT_MB", &v, 8usize, 4096))
                .transpose()?
                .unwrap_or(256)
                << 20,
        })
    }

    /// [`Settings::from_lookup`] on the process environment.
    pub fn from_env() -> Result<Settings, String> {
        Settings::from_lookup(&|k| std::env::var(k).ok())
    }

    /// The move budget of `spec`: its own options first, then the global ones.
    pub fn budget_of(&self, spec: &SnakeSpec) -> MoveBudget {
        let sims = spec.simulations.or(self.simulations);
        match spec.budget_ms.or(self.budget_ms) {
            None => MoveBudget::Simulations(sims.unwrap_or(DEFAULT_SIMULATIONS)),
            Some(ms) => MoveBudget::Time {
                ms,
                max_simulations: sims.unwrap_or(BUDGET_SIMULATIONS),
            },
        }
    }
}

/// The tie-break seed of a move (module docs): FNV-1a 64 over the game id and the turn,
/// folded to 32 bits. `game_id` is the id's JSON text, `turn` the turn's JSON text.
pub fn tie_seed(game_id: &str, turn: &str) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in game_id.bytes().chain([0xff]).chain(turn.bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h ^ (h >> 32)) as u32
}

/// Who chose a move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum By {
    /// The search.
    Engine,
    /// The safe-move picker (an error or a game the network does not play).
    Fallback,
}

/// A move and how it was chosen.
#[derive(Debug, Clone)]
pub struct Choice {
    /// The answer.
    pub mv: Move,
    /// Search or fallback.
    pub by: By,
    /// Simulations run (0 for a fallback).
    pub simulations: usize,
    /// Mode name the network was told.
    pub mode: Option<&'static str>,
    /// The seed used.
    pub seed: u32,
    /// Why the fallback was used.
    pub error: Option<String>,
}

/// One served snake.
pub struct SnakeEngine {
    /// URL path prefix.
    pub name: String,
    agent: Result<Agent, String>,
    threads: usize,
    memory_limit: usize,
    fixed_seed: Option<u32>,
    /// Reusable search trees (one per thread). A Lambda container serves one request at a
    /// time; a second concurrent caller just allocates its own.
    workspaces: Mutex<Vec<Search>>,
}

impl SnakeEngine {
    /// Loads the weights of `spec` (once). A load error is kept: the snake then answers every
    /// move with the safe move and says why.
    pub fn load(settings: &Settings, spec: &SnakeSpec) -> SnakeEngine {
        let dir = settings.weights_dir.join(&spec.subdir);
        let agent = Agent::load(&dir)
            .map(|a| {
                a.with_budget(settings.budget_of(spec))
                    .with_threads(settings.threads)
                    .with_memory_limit(settings.memory_limit)
            })
            .map_err(|e| format!("loading {}: {e}", dir.display()));
        SnakeEngine {
            name: spec.name.clone(),
            agent,
            threads: settings.threads,
            memory_limit: settings.memory_limit,
            fixed_seed: settings.tie_seed,
            workspaces: Mutex::new(Vec::new()),
        }
    }

    /// The weights' load error, if any.
    pub fn load_error(&self) -> Option<&str> {
        self.agent.as_ref().err().map(String::as_str)
    }

    /// The loaded agent.
    pub fn agent(&self) -> Option<&Agent> {
        self.agent.as_ref().ok()
    }

    fn search(
        &self,
        agent: &Agent,
        request_json: &str,
        starting: Option<usize>,
        seed: u32,
        arrival: Instant,
    ) -> Result<(Move, usize, &'static str), SearchError> {
        let req = Request::from_json(request_json)?;
        let game = game_from_request(&req, starting)?;
        let budget = agent.budget().budget_from(arrival);
        let mut ws = match self.workspaces.try_lock() {
            Ok(mut g) => std::mem::take(&mut *g),
            Err(_) => Vec::new(),
        };
        while ws.len() < self.threads {
            ws.push(Search::with_memory_limit(self.memory_limit));
        }
        let res = if self.threads <= 1 {
            agent.search_game(
                &mut ws[0],
                &game,
                budget,
                &mut slinky_search::JaxNoise::from_seed(seed),
            )
        } else {
            agent.search_game_threads(&mut ws, &game, budget, seed)
        };
        if let Ok(mut g) = self.workspaces.try_lock() {
            *g = ws;
        }
        let m = res?;
        Ok((m.mv, m.search.simulations, m.mode.name()))
    }

    /// Chooses a move for the request `body` (`game_id`/`turn` are its JSON texts, for the seed).
    /// `starting` is the snake count remembered at `/start`; `arrival` is when the request
    /// reached the handler. Never fails: errors and panics give the safe move.
    pub fn choose(
        &self,
        body: &str,
        game_id: &str,
        turn: &str,
        starting: Option<usize>,
        arrival: Instant,
    ) -> Choice {
        let seed = self.fixed_seed.unwrap_or_else(|| tie_seed(game_id, turn));
        let fallback = |error: String| Choice {
            mv: safe_move_json(body),
            by: By::Fallback,
            simulations: 0,
            mode: None,
            seed,
            error: Some(error),
        };
        let agent = match &self.agent {
            Ok(a) => a,
            Err(e) => return fallback(e.clone()),
        };
        match catch_unwind(AssertUnwindSafe(|| {
            self.search(agent, body, starting, seed, arrival)
        })) {
            Ok(Ok((mv, simulations, mode))) => Choice {
                mv,
                by: By::Engine,
                simulations,
                mode: Some(mode),
                seed,
                error: None,
            },
            Ok(Err(e)) => fallback(e.to_string()),
            Err(_) => {
                // A poisoned or half-filled workspace is dropped: the next move allocates anew.
                fallback("the search panicked".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn defaults() {
        let s = Settings::from_lookup(&lookup(&[])).unwrap();
        assert_eq!(s.weights_dir, PathBuf::from("/var/task/weights"));
        assert_eq!(s.snakes.len(), 1);
        assert_eq!(s.snakes[0].name, "slinky");
        assert_eq!(s.snakes[0].subdir, "az-modes");
        assert_eq!(s.budget_of(&s.snakes[0]), MoveBudget::Simulations(256));
        assert_eq!(s.threads, 1);
        assert_eq!(s.memory_limit, 256 << 20);
    }

    #[test]
    fn budgets_and_overrides() {
        let s = Settings::from_lookup(&lookup(&[
            ("SNAKES", "a=m,b=m:simulations=64,c=m:budget_ms=200"),
            ("SIMULATIONS", "1024"),
            ("THREADS", "2"),
            ("TIE_SEED", "7"),
        ]))
        .unwrap();
        assert_eq!(s.budget_of(&s.snakes[0]), MoveBudget::Simulations(1024));
        assert_eq!(s.budget_of(&s.snakes[1]), MoveBudget::Simulations(64));
        assert_eq!(
            s.budget_of(&s.snakes[2]),
            MoveBudget::Time {
                ms: 200,
                max_simulations: 1024
            }
        );
        assert_eq!((s.threads, s.tie_seed), (2, Some(7)));
        let t = Settings::from_lookup(&lookup(&[("BUDGET_MS", "300")])).unwrap();
        assert_eq!(
            t.budget_of(&t.snakes[0]),
            MoveBudget::Time {
                ms: 300,
                max_simulations: BUDGET_SIMULATIONS
            }
        );
    }

    #[test]
    fn bad_settings_are_errors() {
        for pairs in [
            &[("SIMULATIONS", "0")][..],
            &[("SIMULATIONS", "x")],
            &[("BUDGET_MS", "999999")],
            &[("THREADS", "0")],
            &[("SNAKES", "noequals")],
            &[("SNAKES", "a=../x")],
            &[("SNAKES", "a=/abs")],
            &[("SNAKES", "a=m,a=n")],
            &[("SNAKES", "a=m:bogus=1")],
            &[("SNAKES", "bad/name=m")],
        ] {
            let pairs: &'static [(&str, &str)] = Box::leak(pairs.to_vec().into_boxed_slice());
            assert!(Settings::from_lookup(&lookup(pairs)).is_err(), "{pairs:?}");
        }
    }

    #[test]
    fn seed_is_stable_and_spread() {
        assert_eq!(tie_seed("g1", "5"), tie_seed("g1", "5"));
        assert_ne!(tie_seed("g1", "5"), tie_seed("g1", "6"));
        assert_ne!(tie_seed("g1", "5"), tie_seed("g2", "5"));
        // "a"+"bc" and "ab"+"c" must differ (the separator).
        assert_ne!(tie_seed("a", "bc"), tie_seed("ab", "c"));
    }

    #[test]
    fn missing_weights_fall_back_to_safe_move() {
        let s = Settings::from_lookup(&lookup(&[("WEIGHTS_DIR", "/nonexistent")])).unwrap();
        let e = SnakeEngine::load(&s, &s.snakes[0]);
        assert!(e.load_error().is_some());
        let c = e.choose("{}", "\"g\"", "0", None, Instant::now());
        assert_eq!(c.by, By::Fallback);
        assert!(c.error.is_some());
    }
}
