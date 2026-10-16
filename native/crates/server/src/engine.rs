//! One served snake: the loaded agent, its search budget, the `/start` cache and the move
//! counter; the server's `Mover` (server.py:359-505) for the native engine.
//!
//! # Seeds
//!
//! The Python server searches move number `i` of a snake (counting from 1 over the snake's
//! life, the startup benchmark's 35 moves included) with the key
//! `jax.random.fold_in(jax.random.key(0), i)`. This server keeps the same counter with the
//! same rules (it advances on every `/move` whose request gets past `check_sizes`, the game id
//! hash and `game_config` without an exception, even when the answer is the safe move) and
//! the same key, so a native server and a Python server started with the same snakes and sent
//! the same requests in the same order search with the same tie noise.
//!
//! # Differences from the Python server
//!
//! * No compilation: every game the agent plays is searched from its first move. Python
//!   answers with `safe_move` until a game config it has not precompiled (a royale with a
//!   non-default hazard damage or shrink period, another board) is compiled.
//! * No heuristic fallback: games with a config but no mode or another board (Python's
//!   `heuristic` agent) get the safe move, as `slinky_search::Agent::respond` does.
//! * Where Python's `safe_move` itself raises, Python names the answer's agent `none`; here it
//!   is `safe_move` (the move is "up" in both).

use crate::info;
use crate::ordered::OrderedMap;
use slinky_api::py::Key;
use slinky_api::{
    ApiError, InfoResponse, PyValue, Request, check_sizes, game_config, game_from_request,
    safe_move,
};
use slinky_rules::Move;
use slinky_search::{Agent, Budget, JaxKey, Search, SearchError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Games whose `/start` snake count is remembered (`server.MAX_GAMES`); the oldest goes first.
pub const MAX_GAMES: usize = 1000;
/// Simulations per move when a snake has a time budget and no simulation count: a ceiling
/// that the deadline reaches first (about 5,000 simulations fit in 300 ms on one core).
pub const BUDGET_SIMULATIONS: usize = 16384;
/// The largest simulation count accepted.
pub const MAX_SIMULATIONS: usize = 1 << 20;
/// The largest time budget accepted, in ms.
pub const MAX_BUDGET_MS: u64 = 10_000;
/// `game.timeout` of a request that has none (Battlesnake's default).
pub const DEFAULT_GAME_TIMEOUT_MS: u64 = 500;
/// Largest `game.timeout` taken from a request (a larger one counts as this).
pub const MAX_GAME_TIMEOUT_MS: u64 = 60_000;

/// The `game.timeout` of a request in ms: [`DEFAULT_GAME_TIMEOUT_MS`] when it is missing or
/// not a positive number, at most [`MAX_GAME_TIMEOUT_MS`].
pub fn game_timeout_ms(req: &Request) -> u64 {
    let t = req.root().get("game").and_then(|g| g.get("timeout"));
    let ms = match t {
        Some(PyValue::Int(i)) if *i >= 1 => u64::try_from(*i).unwrap_or(u64::MAX),
        Some(PyValue::BigInt {
            negative: false, ..
        }) => u64::MAX,
        Some(PyValue::Float(f)) if f.is_finite() && *f >= 1.0 => f.min(u64::MAX as f64) as u64,
        _ => DEFAULT_GAME_TIMEOUT_MS,
    };
    ms.min(MAX_GAME_TIMEOUT_MS)
}

/// The `name=<run dir>[:simulations=N][:budget_ms=T]` value of `--snake`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnakeSpec {
    /// The URL path of the snake (`/<name>`).
    pub name: String,
    /// The run dir with `config.json` and `params.npz`.
    pub run_dir: String,
    /// Simulations per move (`None`: 256, or [`BUDGET_SIMULATIONS`] with a time budget).
    pub simulations: Option<usize>,
    /// Stop searching this many ms after the request arrived (`None`: the faithful search).
    pub budget_ms: Option<u64>,
    /// The agent text after `name=`, as given (`az_modes:` added if missing).
    pub agent: String,
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

impl SnakeSpec {
    /// Parses `name=<agent>` or `<agent>`, where `<agent>` is `[az_modes:]<run dir>` followed by
    /// `:simulations=N` and/or `:budget_ms=T`. As in the Python server, the snake's name must
    /// be letters, digits, `.`, `_` and `-`, so a run dir needs a `name=`.
    pub fn parse(text: &str) -> Result<SnakeSpec, String> {
        let (name, agent) = match text.split_once('=') {
            Some((n, a)) => (n, a),
            None => (text, text),
        };
        if !valid_name(name) {
            return Err(format!(
                "snake name {name:?} can't be a URL path: write --snake name={agent}, \
                 with letters, digits, '.', '_' and '-' in the name"
            ));
        }
        let body = agent.strip_prefix("az_modes:").unwrap_or(agent);
        let mut items = body.split(':');
        let run_dir = items.next().unwrap_or_default().to_string();
        if run_dir.is_empty() {
            return Err(format!("{text:?}: no run dir"));
        }
        let mut spec = SnakeSpec {
            name: name.to_string(),
            run_dir,
            simulations: None,
            budget_ms: None,
            agent: format!("az_modes:{body}"),
        };
        for item in items {
            let (key, value) = item
                .split_once('=')
                .ok_or(format!("{text:?}: option {item:?} is not key=value"))?;
            let number = |max: u64| -> Result<u64, String> {
                match value.trim().parse::<u64>() {
                    Ok(v) if (1..=max).contains(&v) => Ok(v),
                    _ => Err(format!("{text:?}: {key} must be 1 to {max}, not {value:?}")),
                }
            };
            match key {
                "simulations" if spec.simulations.is_none() => {
                    spec.simulations = Some(number(MAX_SIMULATIONS as u64)? as usize);
                }
                "budget_ms" if spec.budget_ms.is_none() => {
                    spec.budget_ms = Some(number(MAX_BUDGET_MS)?);
                }
                "simulations" | "budget_ms" => return Err(format!("{text:?}: {key} given twice")),
                _ => {
                    return Err(format!(
                        "{text:?}: unknown option {key:?} (simulations=N, budget_ms=T)"
                    ));
                }
            }
        }
        Ok(spec)
    }

    /// The simulations per move.
    pub fn simulations(&self) -> usize {
        self.simulations.unwrap_or(if self.budget_ms.is_some() {
            BUDGET_SIMULATIONS
        } else {
            slinky_search::LADDER_SIMULATIONS
        })
    }
}

/// Who chose a move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The agent's search.
    Agent,
    /// The safe-move picker: a game the agent does not play, or a request it cannot read.
    SafeMove,
}

/// A move and how it was made.
#[derive(Debug, Clone)]
pub struct Choice {
    /// The answer.
    pub mv: Move,
    /// The search or the safe-move picker.
    pub source: Source,
    /// Time spent choosing, in ms (parsing excluded, conversion and search included).
    pub ms: f64,
    /// Simulations run (0 for a safe move).
    pub simulations: usize,
    /// Why the safe move was played where Python would have logged an exception.
    pub error: Option<String>,
}

/// A `/start` cache key. Python's dict finds an id by equality, and a NaN id (a new float
/// object on every request) equals nothing, itself included: each NaN `/start` adds an entry
/// no request ever finds (and that only eviction removes), so a NaN gets a key of its own.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GameKey {
    Id(Key),
    NaN(u64),
}

impl GameKey {
    /// The key a lookup uses: `None` for a NaN, which is never found.
    fn lookup(key: Key) -> Option<GameKey> {
        (key != Key::NaN).then_some(GameKey::Id(key))
    }
}

#[derive(Debug, Default)]
struct SnakeState {
    /// game id -> snakes at `/start`.
    games: OrderedMap<GameKey, usize>,
    /// NaN ids stored so far (each gets its own [`GameKey::NaN`]).
    nan_ids: u64,
    /// Moves counted so far (the Python server's `Mover._moves`).
    moves: u64,
}

/// A served snake (module docs).
pub struct Snake {
    /// The URL path (`/<name>`).
    pub name: String,
    /// The agent's name in logs and the move log: the agent text (e.g.
    /// `az_modes:/path/run:simulations=256`), as the Python server names it.
    pub agent_name: String,
    /// The `GET /<name>` response.
    pub info: InfoResponse,
    agent: Agent,
    simulations: usize,
    budget: Option<Duration>,
    state: Mutex<SnakeState>,
}

fn lock(m: &Mutex<SnakeState>) -> std::sync::MutexGuard<'_, SnakeState> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether Python's `game_config(req, n)` returns (a config or `None`) rather than raising.
/// The api crate refuses a float board size that Python's checks let through.
fn python_config_returns(req: &Request, starting: Option<usize>) -> bool {
    match game_config(req, starting) {
        Ok(_) | Err(ApiError::Unsupported(_) | ApiError::Rules(_)) => true,
        Err(ApiError::BadRequest(_)) => {
            let board = req.root().get("board");
            let is_float =
                |k: &str| matches!(board.and_then(|b| b.get(k)), Some(PyValue::Float(_)));
            is_float("width") || is_float("height")
        }
        Err(ApiError::Json(_)) => false,
    }
}

impl Snake {
    /// Loads the run dir of `spec` (the weights are read once, here).
    pub fn load(spec: &SnakeSpec, info: InfoResponse) -> Result<Snake, SearchError> {
        let simulations = spec.simulations();
        let agent = Agent::load(&spec.run_dir)?.with_simulations(simulations);
        Ok(Snake {
            name: spec.name.clone(),
            agent_name: spec.agent.clone(),
            info,
            agent,
            simulations,
            budget: spec.budget_ms.map(Duration::from_millis),
            state: Mutex::new(SnakeState::default()),
        })
    }

    /// The loaded agent.
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Moves counted so far (the seed of the next move is this plus one).
    pub fn moves(&self) -> u64 {
        lock(&self.state).moves
    }

    /// The name of who answers `source` in logs.
    pub fn source_name(&self, source: Source) -> &str {
        match source {
            Source::Agent => &self.agent_name,
            Source::SafeMove => "safe_move",
        }
    }

    /// Whether the snake searches for a time budget.
    pub fn has_budget(&self) -> bool {
        self.budget.is_some()
    }

    /// The search budget of a move whose request arrived at `t_in`.
    pub fn budget(&self, t_in: Instant) -> Budget {
        match self.budget {
            Some(b) => Budget::until(self.simulations, t_in + b),
            None => Budget::simulations(self.simulations),
        }
    }

    /// `/start`: remembers the game's snake count (`Mover.start`). Returns who will play the
    /// game, or the exception Python's `start` raises.
    pub fn start(&self, req: &Request) -> Result<String, String> {
        let root = req.root();
        let game = root.item("game").map_err(|e| e.0)?;
        let id = game.get("id").filter(|v| !matches!(v, PyValue::None));
        let n = req
            .snake_count()
            .ok_or("TypeError: the board's snakes have no len()")?;
        if !python_config_returns(req, None) {
            return Err(game_config(req, None)
                .err()
                .map_or_else(String::new, |e| e.to_string()));
        }
        if let Some(id) = id {
            let mut st = lock(&self.state);
            if st.games.len() >= MAX_GAMES {
                st.games.pop_first(); // games whose /end never came
            }
            let key = match id.key().map_err(|e| e.0)? {
                Key::NaN => {
                    st.nan_ids += 1;
                    GameKey::NaN(st.nan_ids)
                }
                key => GameKey::Id(key),
            };
            st.games.insert(key, n);
        }
        let plays = game_from_request(req, Some(n))
            .ok()
            .is_some_and(|g| self.agent.check_game(&g).is_ok());
        Ok(if plays {
            self.agent_name.clone()
        } else {
            "safe_move".to_string()
        })
    }

    /// `/end`: forgets the game (`Mover.end`); an error where Python's raises.
    pub fn end(&self, req: &Request) -> Result<(), String> {
        if let Some(id) = req.root().get("game").and_then(|g| g.get("id"))
            && let Some(key) = GameKey::lookup(id.key().map_err(|e| e.0)?)
        {
            lock(&self.state).games.remove(&key);
        }
        Ok(())
    }

    /// The snake count remembered at `/start` for this game.
    pub fn starting_snakes(&self, req: &Request) -> Option<usize> {
        let id = req.root().get("game").and_then(|g| g.get("id"))?;
        let key = GameKey::lookup(id.key().ok()?)?;
        lock(&self.state).games.get(&key).copied()
    }

    /// [`Snake::budget`], stopped at `deadline` too (the request's, see `http`).
    pub fn budget_until(&self, t_in: Instant, deadline: Option<Instant>) -> Budget {
        let mut b = self.budget(t_in);
        if let Some(d) = deadline {
            b.deadline = Some(b.deadline.map_or(d, |own| own.min(d)));
        }
        b
    }

    /// `/move`: the agent's move, or the safe move (`Mover.move` plus the handler's fallback).
    /// `t_in` is when the request arrived (a time budget counts from there); `workspace` is
    /// the calling thread's search tree. No deadline but the snake's own budget: see
    /// [`Snake::choose_until`].
    pub fn choose(&self, workspace: &mut Search, req: &Request, t_in: Instant) -> Choice {
        self.choose_until(workspace, req, t_in, None)
    }

    /// [`Snake::choose`] whose search also stops at `deadline` (the best move so far is
    /// played; a search that ends before it is unchanged).
    pub fn choose_until(
        &self,
        workspace: &mut Search,
        req: &Request,
        t_in: Instant,
        deadline: Option<Instant>,
    ) -> Choice {
        let t0 = Instant::now();
        let id = req.root().get("game").and_then(|g| g.get("id"));
        let key = id.map(PyValue::key);
        let sizes_ok = check_sizes(req).is_ok();
        let (starting, index) = {
            let mut st = lock(&self.state);
            let starting = match &key {
                Some(Ok(k)) => GameKey::lookup(k.clone()).and_then(|k| st.games.get(&k).copied()),
                _ => None,
            };
            let counts =
                sizes_ok && !matches!(key, Some(Err(_))) && python_config_returns(req, starting);
            if counts {
                st.moves += 1;
            }
            (starting, st.moves)
        };
        let result = game_from_request(req, starting)
            .map_err(SearchError::from)
            .and_then(|game| {
                let mut noise = JaxKey::new(0).fold_in(index as u32).search_noise();
                self.agent.search_game(
                    workspace,
                    &game,
                    self.budget_until(t_in, deadline),
                    &mut noise,
                )
            });
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        match result {
            Ok(m) => Choice {
                mv: m.mv,
                source: Source::Agent,
                ms: ms(t0),
                simulations: m.search.simulations,
                error: None,
            },
            Err(e) => {
                // Python logs an exception where its conversion raises; a game it has no
                // config or agent for is played by safe_move silently.
                let error = match &e {
                    SearchError::Unsupported(_) | SearchError::Api(ApiError::Unsupported(_)) => {
                        None
                    }
                    other => Some(other.to_string()),
                };
                Choice {
                    mv: safe_move(req),
                    source: Source::SafeMove,
                    ms: ms(t0),
                    simulations: 0,
                    error,
                }
            }
        }
    }

    /// The startup self-benchmark (as the Python server's `main`): plays each mode's example
    /// request (20 duel moves, 5 of each other mode, in `modes.MODE_NAMES` order) and logs the
    /// median time. These moves advance the move counter, as in Python.
    pub fn benchmark(&self, workspace: &mut Search) -> Vec<(String, f64, usize)> {
        let mut out = Vec::new();
        for (mode, body) in crate::examples::requests() {
            let Ok(req) = Request::from_json(&body) else {
                continue;
            };
            let count = if mode == "duel" { 20 } else { 5 };
            let mut times = Vec::with_capacity(count);
            let mut name = String::new();
            for _ in 0..count {
                let t = Instant::now();
                let c = self.choose(workspace, &req, t);
                times.push(t.elapsed().as_secs_f64() * 1e3);
                name = self.source_name(c.source).to_string();
            }
            let median = median(&mut times);
            info!(
                "/{}: {} plays 11x11 {}: {:.1} ms per move (median of {})",
                self.name, name, mode, median, count
            );
            out.push((mode, median, count));
        }
        out
    }
}

/// `np.median`: the middle value, or the mean of the two middle values.
pub fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_snake_specs() {
        let s = SnakeSpec::parse("az-256=/runs/m:simulations=256").unwrap();
        assert_eq!(
            (
                s.name.as_str(),
                s.run_dir.as_str(),
                s.simulations,
                s.budget_ms
            ),
            ("az-256", "/runs/m", Some(256), None)
        );
        assert_eq!(s.agent, "az_modes:/runs/m:simulations=256");
        assert_eq!(s.simulations(), 256);
        // The Python server's --agent form works too.
        let p = SnakeSpec::parse("az-256=az_modes:/runs/m:simulations=256").unwrap();
        assert_eq!(p, s);
        let b = SnakeSpec::parse("fast=/runs/m:budget_ms=300").unwrap();
        assert_eq!((b.simulations, b.budget_ms), (None, Some(300)));
        assert_eq!(b.simulations(), BUDGET_SIMULATIONS);
        let both = SnakeSpec::parse("x=/r:simulations=1024:budget_ms=250").unwrap();
        assert_eq!((both.simulations(), both.budget_ms), (1024, Some(250)));
        assert_eq!(SnakeSpec::parse("plain=/r").unwrap().simulations(), 256);
        assert_eq!(SnakeSpec::parse("dir").unwrap().run_dir, "dir");
        for bad in [
            "/runs/m:simulations=256", // no name: the path is not a URL name
            "a b=/r",
            "=/r",
            "x=",
            "x=/r:simulations=0",
            "x=/r:simulations=abc",
            "x=/r:budget_ms=0",
            "x=/r:budget_ms=99999",
            "x=/r:sims=5",
            "x=/r:simulations=5:simulations=6",
            "x=/r:junk",
        ] {
            assert!(SnakeSpec::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn median_is_numpy_median() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&mut []).is_nan());
    }
}
