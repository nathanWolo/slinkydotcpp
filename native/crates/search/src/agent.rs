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
//! A body the Python handler refuses with HTTP 400 ([`slinky_api::ApiError::Json`]: not JSON,
//! over 64 KB, a non-object `game`) gets no move from Python; [`Agent::respond`] answers "up"
//! (the caller may answer 400 instead).

use crate::SearchError;
use crate::eval::NetEvaluator;
use crate::noise::{JaxNoise, NoiseSource};
use crate::search::{Budget, Search, SearchConfig, SearchOutput};
use slinky_api::{Game, MoveResponse, Request, game_from_request, safe_move_json};
use slinky_net::Network;
use slinky_rules::{MAX_SNAKES, Mode, Move};
use std::path::Path;
use std::time::{Duration, Instant};

/// The play-time simulation count of the ladder agent (`az_modes:...:simulations=256`).
pub const LADDER_SIMULATIONS: usize = 256;

/// A loaded az_modes agent: the network and its search settings. Immutable, so one value can
/// serve many threads (each with its own [`Search`] workspace).
pub struct Agent {
    net: Network,
    settings: SearchConfig,
    simulations: usize,
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
            simulations: LADDER_SIMULATIONS,
        }
    }

    /// The same agent with another simulation count.
    pub fn with_simulations(mut self, simulations: usize) -> Agent {
        self.simulations = simulations;
        self
    }

    /// The simulations per move.
    pub fn simulations(&self) -> usize {
        self.simulations
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

    /// The agent's move for a parsed request: the faithful search with the noise of
    /// `jax.random.key(seed)`. `starting_snakes` is the snake count at the game's `/start`
    /// (see [`slinky_api::game_from_request`]; `None` uses the living snakes).
    pub fn play_request(
        &self,
        req: &Request,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> Result<AgentMove, SearchError> {
        let game = game_from_request(req, starting_snakes)?;
        self.search_game(
            &mut Search::new(),
            &game,
            Budget::simulations(self.simulations),
            &mut JaxNoise::from_seed(seed),
        )
    }

    /// [`Agent::play_request`] from the request's JSON text.
    pub fn play(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> Result<AgentMove, SearchError> {
        self.play_request(&Request::from_json(request_json)?, starting_snakes, seed)
    }

    /// The `/move` response: the agent's move, or the safe move when the request cannot be
    /// played (the error is returned alongside for logging).
    pub fn respond(
        &self,
        request_json: &str,
        starting_snakes: Option<usize>,
        seed: u32,
    ) -> (MoveResponse, Result<AgentMove, SearchError>) {
        let res = self.play(request_json, starting_snakes, seed);
        let mv = match &res {
            Ok(m) => m.mv,
            Err(_) => safe_move_json(request_json),
        };
        (MoveResponse::new(mv), res)
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
