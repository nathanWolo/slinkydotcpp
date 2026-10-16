//! slinky-search: the search of slinky's multi-mode AlphaZero agent (`az_modes`), ported from
//! JAX to plain Rust with bit-exact mechanics.
//!
//! Sources (the Python of slinky v0.1.0): `src/slinky/mcts.py` (`search`, the single-game path
//! the server plays: `_init_tree`, `_puct_moves`, `_descent_body`, `_transition`, `_record`,
//! `_backup`, `_output`, `_depth_noise`, `_node_prior`, `_static_value`),
//! `baselines/az_modes.py` (`agent_search` 172-183, `make_evaluator` 209-223, `make_policy`
//! 250-288) and `src/slinky/agents.py` (`check_game` 300-336, `_told_mode` 724-731). See
//! `native-az/SPEC.md` section 4. Built on the `slinky-rules` (state, step, masks, placement
//! values), `slinky-obs` (network inputs), `slinky-net` (forward pass) and `slinky-api`
//! (request to game) crates.
//!
//! What the search does: decoupled PUCT over joint moves of all seats at once, priors and leaf
//! values from the network for every living seat of each new node, exact placement values for
//! dead seats and finished games (draws worth `draw_value`), `c_puct` 0.5 on the `[0, 1]` scale,
//! first-play urgency = node value - 0.2, depth limit 32, dead and certainly-starving seats forced
//! to move 0, no Dirichlet noise, no virtual loss, no food spawns; the final move is the most
//! visited, ties by mean value, then the lowest index; near-ties are broken by `0.01 * U(0, 1)`
//! from each simulation's noise bits hashed per depth.
//!
//! # Public API
//!
//! ```no_run
//! use slinky_search::{Agent, Budget, JaxNoise, Search};
//!
//! // The agent: run dir with config.json + params.npz (any S/M/L az_modes run), 256 simulations.
//! let agent = Agent::load("/path/to/az-modes-M-s0")?;
//! let body = std::fs::read_to_string("move.json")?;
//! // Request JSON, the snake count seen at /start (None: the living snakes), the seed of
//! // jax.random.key(seed) for the tie noise.
//! let m = agent.play(&body, Some(4), 0)?;
//! println!("{} visits {:?}", m.mv.name(), m.search.visits[0]);
//! // Never failing: the safe move when the request cannot be searched.
//! let (response, _result) = agent.respond(&body, None, 0);
//!
//! // Lower level: any game, budget (simulations and/or a deadline) and noise source.
//! let game = slinky_api::game_from_request(&slinky_api::Request::from_json(&body)?, None)?;
//! let mut ws = Search::new();
//! let m = agent.search_game(&mut ws, &game, Budget::simulations(256), &mut JaxNoise::from_seed(0))?;
//! # let _ = (m, response);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! * [`Agent`]: `load(run_dir)`, `from_network(Network)`, `with_simulations(n)`,
//!   `check_game(&Game)`, `search_game(&mut Search, &Game, Budget, &mut dyn NoiseSource) ->
//!   Result<AgentMove>`, `play_request(&Request, starting_snakes, seed)`, `play(json,
//!   starting_snakes, seed)`, `respond(json, starting_snakes, seed) -> (MoveResponse, Result)`.
//!   [`AgentMove`] holds the move, the told mode, the seat count, the [`SearchOutput`] and the
//!   search time. [`play_once`]`(run_dir, json, starting_snakes, seed)` loads and plays in one call.
//! * [`Search`] (a reusable tree workspace): `run(&GameConfig, &GameState, &SearchConfig,
//!   Budget, &mut dyn Evaluator, &mut dyn NoiseSource) -> Result<SearchOutput>`.
//!   [`SearchOutput`]: per seat `action`, `policy`, `visits`, `q`, `value`, `root_prior`,
//!   `legal`; `nodes_used`, `depth`, `simulations`.
//! * [`SearchConfig`] (`c_puct`, `fpu_reduction`, `draw_value`, `tie_noise`, `max_depth`;
//!   `from_net(&NetConfig)`), [`Budget`] (`simulations(n)` is the faithful mode,
//!   `until(n, deadline)` stops early at a deadline).
//! * Evaluators ([`Evaluator`] trait, returns a [`NodeEval`] of values and a [`Prior`]):
//!   [`NetEvaluator`] (the network), [`UniformEvaluator`] (uniform priors, zero values: the
//!   mechanics test of SPEC.md 7.5). Helpers [`masked_softmax`], [`uniform_prior`].
//! * Noise ([`NoiseSource`] trait, one `[N][4]` block of `u32` per simulation): [`JaxNoise`]
//!   (exactly `jax.random.bits(split(key, 4)[0], (sims, N, 4))`, generated on demand),
//!   [`NoiseTable`] (explicit bits); [`JaxKey`] (`new(seed)` = `jax.random.key`, `fold_in`,
//!   `split_at`, `bits_at`), [`threefry2x32`], [`depth_noise`].
//!
//! Parity (`tests/golden.rs`, vectors from `native/golden/search_golden.py`): with uniform
//! priors and zero leaf values the root visits, mean values, moves, node counts and depths
//! equal `mcts.search` bit for bit in all four modes (plus 3-seat and 2-seat variants); with
//! the network's per-node outputs replayed from Python the same holds for the real agent; and
//! with the Rust network the moves agree with the Python agent except where float rounding in
//! the network decides a near-tie.

mod agent;
mod error;
mod eval;
mod noise;
mod search;

pub use agent::{Agent, AgentMove, LADDER_SIMULATIONS, play_once};
pub use error::SearchError;
pub use eval::{
    Evaluator, NetEvaluator, NodeEval, Prior, UniformEvaluator, masked_softmax, uniform_prior,
};
pub use noise::{JaxKey, JaxNoise, NoiseSource, NoiseTable, SimBits, depth_noise, threefry2x32};
pub use search::{Budget, MAX_DEPTH, Search, SearchConfig, SearchOutput};
