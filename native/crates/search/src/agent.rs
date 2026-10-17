//! The az_modes agent end to end: a `/move` request in, a move out.
//!
//! Ports what `python -m slinky.server` does for `az_modes:<run>:simulations=256`
//! (SPEC.md section 0): the request becomes a game ([`slinky_api::game_from_request`],
//! server.py:161-274), the agent checks it can play it (`agents.check_game`,
//! agents.py:300-336: an 11x11 board, 2 to 4 seats, a mode among the run's `game_modes`), and
//! `az_modes.make_policy(..., simulations=256)` searches every seat (az_modes.py:250-288);
//! the answer is seat 0's move (server.py:425-429).
//!
//! The search key: a move made with `seed` uses the noise of `jax.random.key(seed)`, so
//! [`Agent::play`] equals Python's `make_agent(name, config).policy(jax.random.key(seed),
//! state, ts)[0]` up to float rounding in the network (SPEC.md 2.6). (The Python server itself
//! uses `fold_in(key(0), move_counter)`, which [`crate::JaxKey::fold_in`] can reproduce.)
//!
//! # Known difference: no heuristic fallback
//!
//! Where the Python server builds a game config but its multi-mode agent will not play it,
//! `Mover._build` (server.py:416-433) plays its *fallback* agent, `heuristic`
//! (`slinky.heuristic`): a game with no mode (one seat, e.g. a lone snake listed without the
//! `/start` count; or more than 4 seats on a board other than 11x11), a board other than 11x11
//! (up to 25x25), or a mode the run was not trained on. This crate has no heuristic agent:
//! those games are [`SearchError::Unsupported`] (or an [`slinky_api::ApiError::Unsupported`]
//! for boards larger than the rules crate's 11x11) and [`Agent::respond`] answers with the
//! safe move, as Python does for games with no config at all. This is a deliberate gap, not a
//! conversion bug: the ladder's games (11x11, 2 to 4 snakes, the four modes every az_modes run
//! trains on) never reach the fallback, and the heuristic is a separate ~950-line JAX policy
//! (flood fills, Voronoi races, a duel joint search, random tie-breaks) on boards up to 25x25
//! that the 121-cell rules crate does not represent. Porting it is follow-up work.
//!
//! # Budgets beyond the faithful 256 simulations
//!
//! [`Agent::with_budget`] sets how much each move searches ([`MoveBudget`]): a fixed simulation
//! count (deterministic: 256 is the faithful ladder agent, 1024 or 4096 for reproducible
//! strength tests) or a time budget measured from the request's arrival
//! ([`MoveBudget::Time`]: search until `ms` after `arrival`, at most `max_simulations`).
//! The `*_from` methods take the arrival [`Instant`]; the older methods use the time of the
//! call. [`Agent::with_threads`] searches with several root-parallel trees (see
//! [`crate::run_root_parallel`]); one thread, the default, is the plain search.
//!
//! A body the Python handler refuses with HTTP 400 ([`slinky_api::ApiError::Json`]: not JSON,
//! over 64 KB, a non-object `game`) gets no move from Python; [`Agent::respond`] answers "up"
//! (the caller may answer 400 instead).

use crate::SearchError;
use crate::eval::NetEvaluator;
use crate::noise::{JaxNoise, NoiseSource};
use crate::parallel::{run_root_parallel, thread_noise};
use crate::search::{
    Budget, DEFAULT_MEMORY_LIMIT, MAX_SIMULATIONS, Search, SearchConfig, SearchOutput,
};
use slinky_api::{Game, MoveResponse, Request, game_from_request, safe_move_json};
use slinky_net::Network;
use slinky_rules::{MAX_SNAKES, Mode, Move};
use std::path::Path;
use std::time::{Duration, Instant};

/// The play-time simulation count of the ladder agent (`az_modes:...:simulations=256`).
pub const LADDER_SIMULATIONS: usize = 256;

/// How much an agent searches per move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveBudget {
    /// Exactly this many simulations (deterministic). 256 ([`LADDER_SIMULATIONS`]) is the
    /// faithful ladder agent.
    Simulations(usize),
    /// Search until `ms` milliseconds after the request's arrival, at most `max_simulations`
    /// (at least one simulation always runs). Not deterministic.
    Time {
        /// Milliseconds from the arrival to the end of the search.
        ms: u64,
        /// Simulation cap ([`MAX_SIMULATIONS`] for none).
        max_simulations: usize,
    },
}

impl MoveBudget {
    /// The faithful ladder agent: 256 simulations.
    pub const LADDER: MoveBudget = MoveBudget::Simulations(LADDER_SIMULATIONS);

    /// A time budget of `ms` milliseconds from the arrival, with no simulation cap.
    pub fn time_ms(ms: u64) -> MoveBudget {
        MoveBudget::Time {
            ms,
            max_simulations: MAX_SIMULATIONS,
        }
    }

    /// The search budget of a request that arrived at `arrival`.
    pub fn budget_from(self, arrival: Instant) -> Budget {
        match self {
            MoveBudget::Simulations(n) => Budget::simulations(n),
            MoveBudget::Time {
                ms,
                max_simulations,
            } => Budget::millis_after(arrival, ms, max_simulations),
        }
    }

    /// The simulation count, or the cap of a time budget.
    pub fn max_simulations(self) -> usize {
        match self {
            MoveBudget::Simulations(n) => n,
            MoveBudget::Time {
                max_simulations, ..
            } => max_simulations,
        }
    }
}

/// A loaded az_modes agent: the network and its search settings. Immutable, so one value can
/// serve many threads (each with its own [`Search`] workspace).
pub struct Agent {
    net: Network,
    settings: SearchConfig,
    budget: MoveBudget,
    threads: usize,
    memory_limit: usize,
    early_stop: bool,
    reuse: bool,
}

/// One move of the agent, with the root statistics of its search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentMove {
    /// The move for our snake (seat 0).
    pub mv: Move,
    /// The mode the network was told.
    pub mode: Mode,
    /// Seats in the searched game.
    pub num_snakes: usize,
    /// The search's root output (seat 0 is ours).
    pub search: SearchOutput,
    /// Wall time of the search (request parsing excluded).
    pub elapsed: Duration,
}

impl Agent {
    /// Load a run dir (`config.json` + `params.npz`), with the ladder's 256 simulations.
    pub fn load(run_dir: impl AsRef<Path>) -> Result<Agent, SearchError> {
        Ok(Agent::from_network(Network::load(run_dir)?))
    }

    /// An agent around a loaded network, its search constants from the run's config.
    pub fn from_network(net: Network) -> Agent {
        Agent {
            settings: SearchConfig::from_net(net.config()),
            net,
            budget: MoveBudget::LADDER,
            threads: 1,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            early_stop: false,
            reuse: false,
        }
    }

    /// The same agent with another simulation count (a deterministic [`MoveBudget::Simulations`]).
    pub fn with_simulations(mut self, simulations: usize) -> Agent {
        self.budget = MoveBudget::Simulations(simulations);
        self
    }

    /// The simulations per move (the cap of a time budget).
    pub fn simulations(&self) -> usize {
        self.budget.max_simulations()
    }

    /// The same agent with another per-move budget.
    pub fn with_budget(mut self, budget: MoveBudget) -> Agent {
        self.budget = budget;
        self
    }

    /// The per-move budget.
    pub fn budget(&self) -> MoveBudget {
        self.budget
    }

    /// The same agent searching with `threads` root-parallel trees (at least 1; 1, the
    /// default, is the plain search). A simulation budget is split between the trees.
    pub fn with_threads(mut self, threads: usize) -> Agent {
        self.threads = threads.max(1);
        self
    }

    /// The search threads per move.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The same agent with a tree memory cap per search thread (default
    /// [`DEFAULT_MEMORY_LIMIT`]; see [`Search::set_memory_limit`]).
    pub fn with_memory_limit(mut self, bytes: usize) -> Agent {
        self.memory_limit = bytes;
        self
    }

    /// The tree memory cap per search thread.
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// The same agent stopping each one-tree search once our move is decided
    /// ([`Search::set_stop_when_decided`] for seat 0): the same move as the full budget in less
    /// time, but the reported visits are those of the shorter search. Off by default. The
    /// methods that make their own workspace apply it; with a caller's workspace
    /// ([`Agent::search_game`]) the caller sets it on the workspace.
    pub fn with_early_stop(mut self, on: bool) -> Agent {
        self.early_stop = on;
        self
    }

    /// Whether searches stop once our move is decided ([`Agent::with_early_stop`]).
    pub fn early_stop(&self) -> bool {
        self.early_stop
    }

    /// The same agent asking callers that keep a workspace between moves to reuse the last
    /// tree's evaluations ([`Search::set_reuse`]; the same results, fewer network calls). Off by
    /// default. The methods that make a new workspace per move cannot reuse anything.
    pub fn with_reuse(mut self, on: bool) -> Agent {
        self.reuse = on;
        self
    }

    /// Whether workspaces kept between moves reuse evaluations ([`Agent::with_reuse`]).
    pub fn reuse(&self) -> bool {
        self.reuse
    }

    /// The network.
    pub fn network(&self) -> &Network {
        &self.net
    }

    /// The search constants.
    pub fn settings(&self) -> &SearchConfig {
        &self.settings
    }

    /// Whether the agent plays `game` (`agents.check_game` for a multi-mode agent searching):
    /// an 11x11 board, 2 to 4 seats, and the told mode among the run's `game_modes`.
    pub fn check_game(&self, game: &Game) -> Result<(), SearchError> {
        let n = game.config.num_snakes();
        if !(2..=MAX_SNAKES).contains(&n) {
            return Err(SearchError::Unsupported(format!(
                "the search plays 2 to {MAX_SNAKES} snakes, not {n}"
            )));
        }
        if !game.is_network_board() {
            return Err(SearchError::Unsupported(format!(
                "the network was trained on 11x11 boards, not {}x{}",
                game.config.width(),
                game.config.height()
            )));
        }
        let modes = &self.net.config().game_modes;
        if !modes.iter().any(|m| m == game.mode.name()) {
            return Err(SearchError::Unsupported(format!(
                "the run was trained on {} only, not {}",
                modes.join(", "),
                game.mode.name()
            )));
        }
        Ok(())
    }

    /// Search `game` with an explicit budget and noise source, reusing `workspace`.
    pub fn search_game(
        &self,
        workspace: &mut Search,
        game: &Game,
        budget: Budget,
        noise: &mut dyn NoiseSource,
    ) -> Result<AgentMove, SearchError> {
        self.check_game(game)?;
        let t0 = Instant::now();
        let mut eval = NetEvaluator::for_game(&self.net, game.config)?;
        let out = workspace.run(
            &game.config,
            &game.state,
            &self.settings,
            budget,
            &mut eval,
            noise,
        )?;
        let mv = Move::from_index(out.action[0] as usize).unwrap_or(Move::Up);
        Ok(AgentMove {
            mv,
            mode: game.mode,
            num_snakes: game.config.num_snakes(),
            search: out,
            elapsed: t0.elapsed(),
        })
    }

    /// Search `game` with `workspaces.len()` root-parallel trees (see
    /// [`crate::run_root_parallel`]), tree `t` drawing the noise of
    /// [`crate::thread_noise`]`(seed, t)`. One workspace is [`Agent::search_game`] with
    /// [`JaxNoise::from_seed`]`(seed)`.
    pub fn search_game_threads(
        &self,
        workspaces: &mut [Search],
        game: &Game,
        budget: Budget,
        seed: u32,
    ) -> Result<AgentMove, SearchError> {
        self.check_game(game)?;
        let t0 = Instant::now();
        let k = workspaces.len();
        let mut evals = (0..k)
            .map(|_| NetEvaluator::for_game(&self.net, game.config))
            .collect::<Result<Vec<_>, _>>()?;
        let mut noises: Vec<JaxNoise> = (0..k).map(|t| thread_noise(seed, t)).collect();
        let out = run_root_parallel(
            workspaces,
            &game.config,
            &game.state,
            &self.settings,
            budget,
            &mut evals,
            &mut noises,
        )?;
        let mv = Move::from_index(out.action[0] as usize).unwrap_or(Move::Up);
        Ok(AgentMove {
            mv,
            mode: game.mode,
            num_snakes: game.config.num_snakes(),
            search: out,
            elapsed: t0.elapsed(),
        })
    }

    /// The agent's move for a parsed request with the agent's budget, threads and the noise of
    /// `jax.random.key(seed)`. `starting_snakes` is the snake count at the game's `/start`
    /// (see [`slinky_api::game_from_request`]; `None` uses the living snakes). With the default
    /// budget (256 simulations, one thread) this is the faithful search.
    pub fn play_request(
        &self,
        req: &Request,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> Result<AgentMove, SearchError> {
        self.play_request_from(req, starting_snakes, seed, Instant::now())
    }

    /// [`Agent::play_request`] for a request that arrived at `arrival` (a time budget runs
    /// until `arrival + ms`).
    pub fn play_request_from(
        &self,
        req: &Request,
        starting_snakes: Option<usize>,
        seed: u32,
        arrival: Instant,
    ) -> Result<AgentMove, SearchError> {
        let game = game_from_request(req, starting_snakes)?;
        let budget = self.budget.budget_from(arrival);
        if self.threads <= 1 {
            let mut ws = Search::with_memory_limit(self.memory_limit);
            ws.set_stop_when_decided(self.early_stop.then_some(0));
            self.search_game(&mut ws, &game, budget, &mut JaxNoise::from_seed(seed))
        } else {
            let mut ws: Vec<Search> = (0..self.threads)
                .map(|_| Search::with_memory_limit(self.memory_limit))
                .collect();
            self.search_game_threads(&mut ws, &game, budget, seed)
        }
    }

    /// [`Agent::play_request_from`] in a caller's workspace `ws` (one tree; the agent's thread
    /// count is ignored), with the agent's early stop and reuse settings applied to it. Keep
    /// one workspace per snake and game so that [`Agent::with_reuse`] finds the last tree.
    pub fn play_request_in(
        &self,
        ws: &mut Search,
        req: &Request,
        starting_snakes: Option<usize>,
        seed: u32,
        arrival: Instant,
    ) -> Result<AgentMove, SearchError> {
        let game = game_from_request(req, starting_snakes)?;
        ws.set_stop_when_decided(self.early_stop.then_some(0));
        if ws.reuse() != self.reuse {
            ws.set_reuse(self.reuse);
        }
        self.search_game(
            ws,
            &game,
            self.budget.budget_from(arrival),
            &mut JaxNoise::from_seed(seed),
        )
    }

    /// [`Agent::play`] for a request that arrived at `arrival`.
    pub fn play_from(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
        arrival: Instant,
    ) -> Result<AgentMove, SearchError> {
        self.play_request_from(
            &Request::from_json(request_json)?,
            starting_snakes,
            seed,
            arrival,
        )
    }

    /// [`Agent::respond`] for a request that arrived at `arrival`: the server and the Lambda
    /// call this with the instant the request came in, so a time budget covers the parsing.
    pub fn respond_from(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
        arrival: Instant,
    ) -> (MoveResponse, Result<AgentMove, SearchError>) {
        let res = self.play_from(request_json, starting_snakes, seed, arrival);
        let mv = match &res {
            Ok(m) => m.mv,
            Err(_) => safe_move_json(request_json),
        };
        (MoveResponse::new(mv), res)
    }

    /// [`Agent::play_request`] from the request's JSON text.
    pub fn play(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> Result<AgentMove, SearchError> {
        self.play_from(request_json, starting_snakes, seed, Instant::now())
    }

    /// The `/move` response: the agent's move, or the safe move when the request cannot be
    /// played (the error is returned alongside for logging).
    pub fn respond(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> (MoveResponse, Result<AgentMove, SearchError>) {
        self.respond_from(request_json, starting_snakes, seed, Instant::now())
    }
}

/// One-shot convenience: load the run dir and play one request (see [`Agent::play`]). Loading
/// takes about a millisecond for the M net; keep an [`Agent`] to answer many requests.
pub fn play_once(
    run_dir: impl AsRef<Path>,
    request_json: &str,
    starting_snakes: Option<usize>,
    seed: u32,
) -> Result<AgentMove, SearchError> {
    Agent::load(run_dir)?.play(request_json, starting_snakes, seed)
}
