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
//!   `until(n, deadline)` and `millis_after(start, ms, cap)` stop at a deadline,
//!   `is_deterministic()`).
//! * Evaluators ([`Evaluator`] trait, returns a [`NodeEval`] of values and a [`Prior`]):
//!   [`NetEvaluator`] (the network), [`UniformEvaluator`] (uniform priors, zero values: the
//!   mechanics test of SPEC.md 7.5). Helpers [`masked_softmax`], [`uniform_prior`].
//! * Noise ([`NoiseSource`] trait, one `[N][4]` block of `u32` per simulation): [`JaxNoise`]
//!   (exactly `jax.random.bits(split(key, 4)[0], (sims, N, 4))`, generated on demand),
//!   [`NoiseTable`] (explicit bits); [`JaxKey`] (`new(seed)` = `jax.random.key`, `fold_in`,
//!   `split_at`, `bits_at`), [`threefry2x32`], [`depth_noise`].
//!
//! # Time budgets, long searches and threads (not in the Python agent)
//!
//! The faithful mode stays the default: [`Agent::load`] searches 256 simulations on one thread
//! and every parity test runs through it unchanged. On top of it:
//!
//! ```no_run
//! use slinky_search::{Agent, MoveBudget};
//! use std::time::Instant;
//! # let body = String::new();
//! // Ladder: search until 350 ms after the request arrived (the server keeps the rest of the
//! // 500 ms for the network round trip). Not deterministic.
//! let agent = Agent::load("/path/to/az-modes-M-s0")?.with_budget(MoveBudget::time_ms(350));
//! let arrival = Instant::now(); // when the request came in, before reading the body
//! let (response, result) = agent.respond_from(&body, Some(4), 0, arrival);
//! if let Ok(m) = &result {
//!     println!("{} simulations in {:?}", m.search.simulations, arrival.elapsed());
//! }
//! // Strength tests: a fixed simulation count is deterministic and reproducible.
//! let agent = Agent::load("/path/to/az-modes-M-s0")?.with_simulations(1024);
//! # let _ = (response, agent);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! * [`MoveBudget`]: `Simulations(n)` (deterministic; [`MoveBudget::LADDER`] = 256) or
//!   `Time { ms, max_simulations }` (until `arrival + ms`; [`MoveBudget::time_ms`]`(ms)` has no
//!   cap), `budget_from(arrival) -> Budget`.
//! * [`Agent`]: `with_budget(MoveBudget)`, `budget()`, `with_threads(k)`, `threads()`,
//!   `with_memory_limit(bytes)`; `play_from` / `play_request_from` / `respond_from` take the
//!   request's arrival [`std::time::Instant`] (`play`, `play_request` and `respond` use the time
//!   of the call); `search_game_threads(&mut [Search], &Game, Budget, seed)` searches with one
//!   root-parallel tree per workspace.
//! * [`Search`]: `with_memory_limit(bytes)` / `set_memory_limit` / `memory_limit()` (default
//!   [`DEFAULT_MEMORY_LIMIT`], 256 MiB per workspace; the search stops before a simulation once
//!   the tree is full), `stats() -> `[`SearchStats`] (simulations, nodes, depth, elapsed,
//!   [`StopReason`] `Simulations | Deadline | MemoryLimit`, `tree_bytes`, `allocated_bytes`),
//!   `subtree_simulations(&moves)` (the part of the tree under a root child). [`node_bytes`]`(N)`
//!   is the tree memory per node; [`MAX_SIMULATIONS`] (2^24) caps any budget.
//! * [`run_root_parallel`]`(&mut [Search], config, state, settings, budget, &mut [E], &mut [N])`
//!   and [`thread_noise`]`(seed, t)`: independent trees on scoped threads, root statistics
//!   summed, the budget's simulations split between the trees, one deadline for all. One tree
//!   is exactly [`Search::run`].
//!
//! ## Measured: what extra search buys, threads and tree reuse
//!
//! `examples/quality.rs` (420 golden positions over the 7 configs, M network) compares each
//! method's move with a 16384-simulation reference search: agreement and mean regret (the
//! reference's value of its move minus that of the chosen move, value scale -1 to 1):
//!
//! | method | 1x256 | 1x512 | 1x1024 | 1x2048 | 1x4096 | 2x512 | 4x256 | 2x2048 | 4x1024 |
//! |---|---|---|---|---|---|---|---|---|---|
//! | agrees | 82.1% | 85.0% | 85.5% | 88.6% | 91.2% | 85.2% | 82.9% | 88.6% | 86.4% |
//! | regret x1000 | 5.42 | 4.12 | 3.16 | 2.71 | 2.18 | 4.09 | 4.92 | 2.75 | 3.28 |
//!
//! * More simulations keep improving the move (a proxy; game results are the real test).
//! * **Root parallelism does not pay**: `k x n` (k trees of n simulations, the wall time of n)
//!   plays like `1 x n`, not like `1 x kn`. It stays available ([`Agent::with_threads`]) for
//!   experiments but is off by default and not recommended. A shared tree with virtual loss
//!   could do better, at the price of determinism and a lock-free tree; its gain is bounded by
//!   the `1x1024` to `1x4096` gap above, and a 300 ms single-core search already reaches 4k to
//!   7k simulations.
//! * **No tree reuse between moves.** Only the subtree under the joint move actually played can
//!   be kept: on average 49% of a 4096-simulation tree lies under the child of the searched
//!   moves of *all* seats (33% in 4-seat standard, 37% royale, 56% duel, 63% constrictor), an
//!   upper bound since real opponents often play otherwise. The next request matches that child
//!   only if no food spawned (the search does not model spawns), the royale zone did not shrink
//!   onto unknown sides, and the seats are mapped back (the request puts living snakes first, so
//!   a death reorders them). It needs a per-game tree store (10 to 40 MB per game) that a
//!   stateless or multi-container Lambda does not have, and it makes moves depend on history.
//!   At best it is worth under 1.5x the simulations (about one row of the table above), so it
//!   is not done.
//!
//! ## Measured: deadlines, simulations and memory
//!
//! A deadline is checked before every simulation, so a search ends within one simulation
//! (0.1 ms or less) of it. `examples/budget.rs`, one pinned core of the PC (Ryzen 7 5800XT,
//! shared with other jobs), M network, 60 golden positions per config, a fresh workspace per
//! move: of 2,520 timed moves (budgets 50 to 500 ms) the slowest ended 0.63 ms after its
//! deadline and all others within 0.13 ms (an earlier run had one 3.5 ms outlier in 1,400, a
//! preemption on the shared machine). Freeing the tree takes up to about 1 ms more. Median
//! simulations per move (positions with a nearly finished game reach far more, up to millions,
//! as simulations that only revisit finished nodes are cheap):
//!
//! | budget | duel | standard 4p | royale 4p | constrictor 4p | standard3 | royale2 | constrictor2 |
//! |---|---|---|---|---|---|---|---|
//! | 50 ms | 1,197 | 856 | 819 | 752 | 832 | 1,192 | 1,328 |
//! | 100 ms | 2,394 | 1,680 | 1,591 | 1,576 | 1,672 | 2,368 | 2,649 |
//! | 200 ms | 4,756 | 3,362 | 3,200 | 3,427 | 3,326 | 4,769 | 5,269 |
//! | 300 ms | 7,161 | 5,031 | 4,831 | 5,273 | 5,017 | 7,156 | 7,761 |
//! | 400 ms | 9,412 | 6,639 | 6,286 | 7,058 | 6,681 | 9,578 | 10,182 |
//! | 500 ms | 11,832 | 8,312 | 7,855 | 9,557 | 8,353 | 11,727 | 12,749 |
//!
//! Tree memory per move at 500 ms: at most 18.7k nodes, 47 MB in a 4-seat game (2.6 KB per
//! node) and 20 MB in a duel (1.7 KB per node); 0.4 to 0.7 MB for the faithful 256
//! simulations. Fixed simulation counts on the same core: 1024 takes 44 ms (duel) to 60-73 ms
//! (4 seats) per move at the median, 4096 takes 174 ms to 245-284 ms (p100 up to 363 ms).
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
mod parallel;
mod search;

pub use agent::{Agent, AgentMove, LADDER_SIMULATIONS, MoveBudget, play_once};
pub use error::SearchError;
pub use eval::{
    Evaluator, NetEvaluator, NodeEval, Prior, UniformEvaluator, masked_softmax, uniform_prior,
};
pub use noise::{JaxKey, JaxNoise, NoiseSource, NoiseTable, SimBits, depth_noise, threefry2x32};
pub use parallel::{run_root_parallel, thread_noise};
pub use search::{
    Budget, DEFAULT_MEMORY_LIMIT, MAX_DEPTH, MAX_SIMULATIONS, Search, SearchConfig, SearchOutput,
    SearchStats, StopReason, node_bytes,
};
