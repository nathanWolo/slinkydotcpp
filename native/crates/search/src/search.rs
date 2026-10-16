//! The decoupled-PUCT simultaneous-move search, a faithful port of `slinky.mcts.search` as
//! `az_modes` plays it (SPEC.md section 4).
//!
//! Sources (`src/slinky/mcts.py` of slinky v0.1.0): `_init_tree` (842-890), `_puct_moves`
//! (786-802), `_descent_body` (913-949), `_expand`/`_transition` (1089-1106, 498-512),
//! `_record`/`_static_value` (1109-1146, 515-536), `_backup` (1040-1071), `_run` (1195-1228)
//! and `_output` (1332-1362); the settings are `az_modes.agent_search`
//! (baselines/az_modes.py:172-183).
//!
//! One simulation: every seat independently picks the legal move maximising
//! `0.5 * (q + 1) + (c_puct * P) * sqrt(max(N, 1)) / (1 + n) + 0.01 * u` (`q` the move's mean
//! value, or the node's own value minus `fpu_reduction` when unvisited; `u` the depth's tie
//! noise; ties to the lowest move), the joint move `sum_p move_p * 4^p` leads to the child, and
//! the walk stops at an unexpanded joint move (expand it: one rules step without food spawns,
//! evaluate the new node), at a finished node or at depth `min(32, simulations)`. Each seat
//! then adds its own component of the leaf value to its own move's statistics at every node on
//! the path. The final move of each seat is its most visited legal root move, ties by mean
//! value, then lowest index.
//!
//! Every f32 operation is in the jitted Python's order, so with the same evaluator outputs and
//! noise bits the visits, values and moves are bit-identical (`tests/golden.rs`).
//!
//! Differences that do not change results: the tree grows a node per expansion (Python fills
//! slot `sim + 1` and leaves it unused when nothing is expanded); the evaluator is not called on
//! simulations that expand nothing, on finished nodes other than the root, or for dead seats
//! (SPEC ambiguity 8).
//!
//! The search can also stop at a deadline ([`Budget::deadline`]); without one it runs exactly
//! [`Budget::simulations`] simulations, as Python does.
//!
//! # Longer searches
//!
//! Nothing in the tree is sized for 256 simulations: nodes and the children table grow with the
//! expansions, a descent's path is at most `min(max_depth, simulations)` (32) edges, and the
//! noise of simulation `s` does not depend on the total. Two limits keep a long search safe:
//!
//! * the simulation count is capped at [`MAX_SIMULATIONS`] (2^24), where the f32 visit totals of
//!   the PUCT formula stop being exact and an i32 visit count is still far from overflowing;
//!   the value sums are exact enough up to there too: each move's sum is kept both in f32, in
//!   Python's order (used while the move has at most [`F32_SUM_VISITS`] visits, so every search
//!   of up to that many simulations is bit-identical to Python), and in f64 (used beyond, where
//!   an f32 sum would round each added value to its ulp and the mean values would drift);
//! * the tree's memory is capped by [`Search::set_memory_limit`] (default
//!   [`DEFAULT_MEMORY_LIMIT`], 256 MiB): before each simulation the search stops when the tree
//!   holds `limit / node_bytes(N)` nodes ([`node_bytes`]: about 1.6 KB per node plus 4^N child
//!   slots of 4 bytes). The node pool grows in place, never beyond the limit; with a deadline it
//!   is preallocated (address space only) for the expansions the remaining time allows. A
//!   reused workspace first gives back what an earlier, bigger tree left beyond the limit.
//!
//! Under a deadline the search also stops early when more simulations cannot change anything
//! or teach anything ([`StopReason::Solved`]): at once when the root's game is over, and once a
//! root whose whole tree is solved (every line expanded down to a finished game) has run
//! [`SOLVED_SIMULATIONS`] simulations. A fixed simulation count always runs in full.
//!
//! A deadline is checked before every simulation (a monotonic [`Instant::now`] read, about 20 ns
//! against 20 to 100 us per simulation), so a search ends at most one simulation after its
//! deadline. Each run records [`SearchStats`] ([`Search::stats`]): simulations, nodes, depth,
//! time, why it stopped, and the tree's memory.

use crate::SearchError;
use crate::eval::{Evaluator, Prior, masked_softmax, uniform_prior};
use crate::noise::{NoiseSource, depth_noise};
use slinky_rules::{Actions, GameConfig, GameState, MAX_SNAKES, MoveMask, NUM_ACTIONS};
use std::time::{Duration, Instant};

/// The play-time `max_depth` (`MCTSConfig.max_depth`, mcts.py:348-367).
pub const MAX_DEPTH: usize = 32;

/// Marks an unexpanded joint action in the children table.
const NONE: u32 = u32::MAX;

/// The most simulations one search runs (2^24); larger budgets are clamped. Up to here the
/// f32 visit totals in the PUCT formula are exact.
pub const MAX_SIMULATIONS: usize = 1 << 24;

/// Visits of a move up to which its mean value is computed from the f32 sum, in Python's order
/// (bit-exact with Python); beyond, from the f64 sum.
pub const F32_SUM_VISITS: i32 = 1 << 12;

/// Under a deadline, a search whose root is solved (see the module docs) stops once it has run
/// this many simulations: enough for the visit shares of a finished tree to settle.
pub const SOLVED_SIMULATIONS: usize = 4096;

/// The default cap on a workspace's tree memory (256 MiB, about 100k nodes in a 4-seat game and
/// 160k in a duel; a 500 ms search on one core of the PC builds at most about 19k).
pub const DEFAULT_MEMORY_LIMIT: usize = 256 << 20;

/// The fastest a simulation that expands a node can be (one network row is about 20 us on the
/// PC); used only to preallocate the node pool for a deadline.
const MIN_EXPANSION: Duration = Duration::from_micros(15);

/// Bytes of tree per node in a game of `num_snakes` seats: the node (state, priors, values,
/// statistics) and its `4^N` child slots.
pub fn node_bytes(num_snakes: usize) -> usize {
    std::mem::size_of::<Node>()
        + NUM_ACTIONS.pow(num_snakes.min(MAX_SNAKES) as u32) * std::mem::size_of::<u32>()
}

/// Why a search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    /// It ran its [`Budget::simulations`] (always the case without a deadline or memory cap).
    #[default]
    Simulations,
    /// The [`Budget::deadline`] passed.
    Deadline,
    /// The tree reached the workspace's memory limit ([`Search::set_memory_limit`]).
    MemoryLimit,
    /// Under a deadline, the root's game is over or its whole tree is solved (module docs).
    Solved,
}

/// Statistics of the last [`Search::run`] of a workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchStats {
    /// Simulations run.
    pub simulations: usize,
    /// Nodes in the tree, the root included.
    pub nodes: usize,
    /// Depth of the deepest expansion.
    pub depth: usize,
    /// Wall time of the run (root evaluation included).
    pub elapsed: Duration,
    /// Why it stopped.
    pub stop: StopReason,
    /// Bytes of tree in use: `nodes * node_bytes(N)`.
    pub tree_bytes: usize,
    /// Bytes the workspace has allocated for nodes and child slots (its capacity; pages never
    /// written are not resident).
    pub allocated_bytes: usize,
}

/// The search constants (`az_modes.agent_search`; SPEC.md 4.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchConfig {
    /// PUCT constant on the `[0, 1]` value scale (`c_puct`, 0.5 in the deployed runs).
    pub c_puct: f32,
    /// First-play urgency: an unvisited move's value is the node's value minus this (0.2).
    pub fpu_reduction: f32,
    /// Value of a drawn finish in the placement values (-0.5).
    pub draw_value: f32,
    /// Scale of the tie-break noise (`tie_noise`, 0.01).
    pub tie_noise: f32,
    /// Depth limit of a descent before `min(max_depth, simulations)` (32).
    pub max_depth: usize,
}

impl Default for SearchConfig {
    /// The deployed az_modes settings: `c_puct` 0.5, FPU 0.2, draw -0.5, tie noise 0.01, depth 32.
    fn default() -> Self {
        SearchConfig {
            c_puct: 0.5,
            fpu_reduction: 0.2,
            draw_value: -0.5,
            tie_noise: 0.01,
            max_depth: MAX_DEPTH,
        }
    }
}

impl SearchConfig {
    /// The run's search constants from its `config.json` (`az_modes.agent_search`).
    pub fn from_net(cfg: &slinky_net::NetConfig) -> SearchConfig {
        SearchConfig {
            c_puct: cfg.c_puct,
            fpu_reduction: cfg.fpu_reduction,
            draw_value: cfg.draw_value,
            ..SearchConfig::default()
        }
    }
}

/// How long to search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Number of simulations (256 for the ladder agent); also sets the depth limit
    /// `min(max_depth, simulations)` as in Python.
    pub simulations: usize,
    /// Stop before starting a simulation once this instant has passed (at least one simulation
    /// always runs). `None`, the faithful mode, runs exactly `simulations`.
    pub deadline: Option<Instant>,
}

impl Budget {
    /// Exactly `n` simulations (the faithful mode).
    pub fn simulations(n: usize) -> Budget {
        Budget {
            simulations: n,
            deadline: None,
        }
    }

    /// At most `n` simulations, stopping early at `deadline`.
    pub fn until(n: usize, deadline: Instant) -> Budget {
        Budget {
            simulations: n,
            deadline: Some(deadline),
        }
    }

    /// Search until `ms` milliseconds after `start` (e.g. the request's arrival), at most
    /// `max_simulations` simulations ([`MAX_SIMULATIONS`] for no cap). A deadline beyond what
    /// [`Instant`] can represent means no deadline.
    pub fn millis_after(start: Instant, ms: u64, max_simulations: usize) -> Budget {
        Budget {
            simulations: max_simulations,
            deadline: start.checked_add(Duration::from_millis(ms)),
        }
    }

    /// Whether the result depends only on the inputs (no deadline): the same position, network,
    /// noise and budget give the same search.
    pub fn is_deterministic(&self) -> bool {
        self.deadline.is_none()
    }
}

/// The search result at the root (`mcts.SearchOutput`, mcts.py:1332-1362). Rows at or beyond
/// the game's seat count are zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOutput {
    /// Each seat's chosen move (seat 0 is ours).
    pub action: [u8; MAX_SNAKES],
    /// Visit shares over the legal root moves (uniform over them when nothing was visited).
    pub policy: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    /// Root visits per seat and move.
    pub visits: [[i32; NUM_ACTIONS]; MAX_SNAKES],
    /// Mean value per seat and root move (0 where unvisited).
    pub q: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    /// Mean root value per seat over all visits (the root's leaf value when unvisited).
    pub value: [f32; MAX_SNAKES],
    /// Root priors.
    pub root_prior: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    /// Root legal moves.
    pub legal: MoveMask,
    /// Nodes in the tree, the root included (`expanded + 1`).
    pub nodes_used: usize,
    /// Depth of the deepest expansion.
    pub depth: usize,
    /// Simulations run.
    pub simulations: usize,
}

/// One tree node (`mcts.Tree` row; SPEC.md 4.3).
#[derive(Debug, Clone, Copy)]
struct Node {
    state: GameState,
    legal: MoveMask,
    prior: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    /// Leaf value at expansion (exact for dead seats and finished games).
    value: [f32; MAX_SNAKES],
    visits: [[i32; NUM_ACTIONS]; MAX_SNAKES],
    /// Value sums in f32, added in Python's order.
    value_sum: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    /// The same sums in f64 (for moves past [`F32_SUM_VISITS`] visits).
    value_sum_wide: [[f64; NUM_ACTIONS]; MAX_SNAKES],
    terminal: bool,
    /// The parent node ([`NONE`] for the root).
    parent: u32,
    /// Joint moves (of the legal ones each seat can pick) whose child is not solved yet; the node
    /// is solved when it is terminal or this reaches 0.
    open: u16,
}

impl Node {
    fn solved(&self) -> bool {
        self.terminal || self.open == 0
    }
}

/// The mean value of a move from its sums and visits (`visits > 0`): the f32 sum in Python's
/// order up to [`F32_SUM_VISITS`] visits, the f64 sum beyond.
#[inline]
fn mean_value(sum: f32, wide: f64, visits: i32) -> f32 {
    if visits <= F32_SUM_VISITS {
        sum / visits as f32
    } else {
        (wide / f64::from(visits)) as f32
    }
}

/// The root statistics a result is computed from (one tree's, or the sum of several).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RootStats {
    pub legal: MoveMask,
    pub prior: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    pub value: [f32; MAX_SNAKES],
    pub visits: [[i32; NUM_ACTIONS]; MAX_SNAKES],
    pub value_sum: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    pub value_sum_wide: [[f64; NUM_ACTIONS]; MAX_SNAKES],
}

/// A reusable search workspace: the tree's nodes and children table. Keep one per thread to
/// avoid reallocating between moves.
#[derive(Debug)]
pub struct Search {
    nodes: Vec<Node>,
    /// `children[node * J + joint]`, [`NONE`] when not expanded.
    children: Vec<u32>,
    /// `(node, joint, moves)` per edge of the current descent.
    path: Vec<(u32, u16, [u8; MAX_SNAKES])>,
    /// Tree memory cap in bytes.
    memory_limit: usize,
    /// Node cap of the current run (from `memory_limit`).
    node_limit: usize,
    /// The last run's statistics.
    stats: SearchStats,
    /// Seats of the last run.
    seats: usize,
}

impl Default for Search {
    fn default() -> Self {
        Search {
            nodes: Vec::new(),
            children: Vec::new(),
            path: Vec::new(),
            memory_limit: DEFAULT_MEMORY_LIMIT,
            node_limit: usize::MAX,
            stats: SearchStats::default(),
            seats: 0,
        }
    }
}

impl Search {
    /// An empty workspace with the default memory limit ([`DEFAULT_MEMORY_LIMIT`]).
    pub fn new() -> Search {
        Search::default()
    }

    /// An empty workspace whose tree may use at most `bytes` (see [`Search::set_memory_limit`]).
    pub fn with_memory_limit(bytes: usize) -> Search {
        let mut s = Search::default();
        s.set_memory_limit(bytes);
        s
    }

    /// Cap the tree at `bytes`: a search stops before a simulation once it holds
    /// `bytes / node_bytes(N)` nodes (at least one simulation always runs). The 256-simulation
    /// faithful search needs about 0.7 MB in a 4-seat game, so the default never binds it.
    pub fn set_memory_limit(&mut self, bytes: usize) {
        self.memory_limit = bytes;
    }

    /// The tree memory cap in bytes.
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// Statistics of the last [`Search::run`] (zero before the first).
    pub fn stats(&self) -> SearchStats {
        self.stats
    }

    /// Simulations that went through the root's child reached by `moves` (one per seat) in the
    /// last run, 0 when that child was never expanded: the part of the tree a search of the
    /// next position could reuse if the game went that way.
    pub fn subtree_simulations(&self, moves: &[u8; MAX_SNAKES]) -> usize {
        let n = self.seats;
        if self.nodes.is_empty()
            || n == 0
            || moves.iter().take(n).any(|&m| m as usize >= NUM_ACTIONS)
        {
            return 0;
        }
        match self.children.get(joint_of(moves, n)) {
            Some(&c) if c != NONE => {
                let child = &self.nodes[c as usize];
                // The expanding simulation plus every simulation that descended through it
                // (those that stopped at the child, finished or at the depth limit, are not
                // counted: a slight undercount).
                1 + child.visits[0].iter().sum::<i32>().max(0) as usize
            }
            _ => 0,
        }
    }

    /// The root's statistics after a run.
    pub(crate) fn root_stats(&self) -> Option<RootStats> {
        self.nodes.first().map(|r| RootStats {
            legal: r.legal,
            prior: r.prior,
            value: r.value,
            visits: r.visits,
            value_sum: r.value_sum,
            value_sum_wide: r.value_sum_wide,
        })
    }

    /// Search `state` of a game with `config` (2 to 4 seats) for every seat at once.
    pub fn run(
        &mut self,
        config: &GameConfig,
        state: &GameState,
        settings: &SearchConfig,
        budget: Budget,
        evaluator: &mut dyn Evaluator,
        noise: &mut dyn NoiseSource,
    ) -> Result<SearchOutput, SearchError> {
        let n = config.num_snakes();
        if !(2..=MAX_SNAKES).contains(&n) {
            return Err(SearchError::Unsupported(format!(
                "the search plays 2 to {MAX_SNAKES} seats, not {n}"
            )));
        }
        let t0 = Instant::now();
        let joints = NUM_ACTIONS.pow(n as u32);
        let mut sims_cap = budget.simulations.min(MAX_SIMULATIONS);
        if budget.deadline.is_some() {
            sims_cap = sims_cap.max(1); // at least one simulation always runs
        }
        let max_depth = settings.max_depth.min(sims_cap);
        let per_node = node_bytes(n);
        self.node_limit = (self.memory_limit / per_node).max(1);
        self.seats = n;
        self.stats = SearchStats::default();
        self.nodes.clear();
        self.children.clear();
        // A reused workspace keeps the capacity of its biggest tree: give back what this game's
        // limit does not allow (`push` keeps both within it from here on).
        let max_nodes = self.node_limit.max(2);
        if self.nodes.capacity() > max_nodes {
            self.nodes.shrink_to(max_nodes);
        }
        if self.children.capacity() > max_nodes.saturating_mul(joints) {
            self.children.shrink_to(max_nodes.saturating_mul(joints));
        }
        // Room for every node the budget can expand (address space only: pages are touched as
        // nodes are made). With a deadline, the expansions the remaining time allows.
        let mut expect = sims_cap.saturating_add(1);
        if let Some(d) = budget.deadline {
            let left = d.saturating_duration_since(t0).as_nanos();
            let fit = left / MIN_EXPANSION.as_nanos() + 2;
            expect = expect.min(usize::try_from(fit).unwrap_or(usize::MAX));
        }
        let expect = expect.min(self.node_limit.max(2));
        self.nodes.reserve_exact(expect);
        self.children.reserve_exact(expect.saturating_mul(joints));

        let root = self.make_node(0, NONE, *state, config, settings, evaluator)?;
        self.push(root, joints);

        let mut deepest = 0usize;
        let mut sims = 0usize;
        let mut stop = StopReason::Simulations;
        for sim in 0..sims_cap {
            if sim > 0 {
                if budget.deadline.is_some_and(|d| Instant::now() >= d) {
                    stop = StopReason::Deadline;
                    break;
                }
                if budget.deadline.is_some()
                    && (self.nodes[0].terminal
                        || (self.nodes[0].open == 0 && sims >= SOLVED_SIMULATIONS))
                {
                    stop = StopReason::Solved;
                    break;
                }
                if self.nodes.len() >= self.node_limit {
                    stop = StopReason::MemoryLimit;
                    break;
                }
            }
            let bits = noise.bits(sim, n).ok_or(SearchError::Noise { sim })?;
            sims += 1;
            if self.nodes[0].terminal {
                continue; // a finished root: no simulation changes anything
            }
            // (b) Selection.
            self.path.clear();
            let mut node = 0usize;
            let mut depth = 0usize;
            let leaf_value = loop {
                let moves = self.select(node, n, &bits, depth as u32, settings);
                let joint = joint_of(&moves, n);
                self.path.push((node as u32, joint as u16, moves));
                depth += 1;
                let child = self.children[node * joints + joint];
                if child == NONE {
                    // (c) Expansion and (d) evaluation.
                    let actions: Actions = moves;
                    let next = self.nodes[node].state.child(config, &actions);
                    let id = self.nodes.len();
                    let new = self.make_node(id, node as u32, next, config, settings, evaluator)?;
                    self.push(new, joints);
                    self.children[node * joints + joint] = id as u32;
                    if new.solved() {
                        self.close(node);
                    }
                    deepest = deepest.max(depth);
                    break new.value;
                }
                node = child as usize;
                let nd = &self.nodes[node];
                if nd.terminal || depth >= max_depth {
                    break nd.value;
                }
            };
            // (e) Backup: each seat adds its own value to its own move.
            for &(pn, _, moves) in &self.path {
                let nd = &mut self.nodes[pn as usize];
                for p in 0..n {
                    let a = moves[p] as usize;
                    nd.visits[p][a] += 1;
                    nd.value_sum[p][a] += leaf_value[p];
                    nd.value_sum_wide[p][a] += f64::from(leaf_value[p]);
                }
            }
        }
        let out = self.output(n, deepest, sims);
        self.stats = SearchStats {
            simulations: sims,
            nodes: self.nodes.len(),
            depth: deepest,
            elapsed: t0.elapsed(),
            stop,
            tree_bytes: self.nodes.len() * per_node,
            allocated_bytes: self.nodes.capacity() * std::mem::size_of::<Node>()
                + self.children.capacity() * std::mem::size_of::<u32>(),
        };
        Ok(out)
    }

    fn push(&mut self, node: Node, joints: usize) {
        if self.nodes.len() == self.nodes.capacity() {
            // Grow by doubling, but never past the memory limit's node count.
            let len = self.nodes.len();
            let extra = len.max(16).min(self.node_limit.saturating_sub(len)).max(1);
            self.nodes.reserve_exact(extra);
        }
        self.nodes.push(node);
        // The children table holds exactly the slots of the node pool's capacity (not Vec's
        // amortised doubling, which could pass the memory limit).
        let need = self.children.len() + joints;
        if self.children.capacity() < need {
            let slots = self.nodes.capacity().saturating_mul(joints).max(need);
            self.children.reserve_exact(slots - self.children.len());
        }
        self.children.resize(need, NONE);
    }

    /// One more joint move of `node` leads to a solved child: count it down, and when `node`
    /// becomes solved, pass it on to its parent (and so on up the tree).
    fn close(&mut self, mut node: usize) {
        loop {
            let nd = &mut self.nodes[node];
            if nd.solved() {
                return; // a terminal node is never descended from
            }
            nd.open -= 1;
            if nd.open > 0 || nd.parent == NONE {
                return;
            }
            node = nd.parent as usize;
        }
    }

    /// A new node for `state` (`_init_tree` / `_record`): legal moves, priors and values.
    fn make_node(
        &self,
        id: usize,
        parent: u32,
        state: GameState,
        config: &GameConfig,
        settings: &SearchConfig,
        evaluator: &mut dyn Evaluator,
    ) -> Result<Node, SearchError> {
        let n = config.num_snakes();
        let legal = state.legal_moves(config);
        let exact = state.placement_values(config, settings.draw_value);
        let mut value = exact;
        let mut prior = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        let terminal = state.done;
        // A finished node is never searched from, so only the root's priors matter there: they
        // are reported as `root_prior`, which Python takes from the evaluator even when the game
        // is over (`_init_tree` always calls it, mcts.py:873-878). Its values stay exact.
        let mut seats = [false; MAX_SNAKES];
        if !terminal || id == 0 {
            for (s, a) in seats.iter_mut().zip(&state.alive).take(n) {
                *s = *a;
            }
        }
        let eval = if seats.iter().any(|&s| s) {
            Some(evaluator.evaluate(id, &state, &seats)?)
        } else {
            None
        };
        for p in 0..n {
            if let Some(e) = &eval
                && seats[p]
                && !terminal
            {
                // `jnp.clip(net_value, -1, 1)`; NaN stays NaN as in Python.
                value[p] = e.values[p].clamp(-1.0, 1.0);
            }
            prior[p] = match eval.map(|e| e.prior) {
                Some(Prior::Logits(l)) => masked_softmax(&l[p], &legal[p]),
                Some(Prior::Probs(pr)) => pr[p],
                Some(Prior::Uniform) | None => uniform_prior(&legal[p]),
            };
        }
        // `select` picks one of a seat's legal moves, or move 0 when it has none.
        let open = (0..n)
            .map(|p| legal[p].iter().filter(|&&l| l).count().max(1))
            .product::<usize>();
        Ok(Node {
            state,
            legal,
            prior,
            value,
            visits: [[0; NUM_ACTIONS]; MAX_SNAKES],
            value_sum: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            value_sum_wide: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            terminal,
            parent,
            open: open as u16,
        })
    }

    /// Every seat's decoupled PUCT move at `node` (`_puct_moves`, mcts.py:786-802).
    fn select(
        &self,
        node: usize,
        n: usize,
        bits: &crate::noise::SimBits,
        depth: u32,
        cfg: &SearchConfig,
    ) -> [u8; MAX_SNAKES] {
        let nd = &self.nodes[node];
        let mut moves = [0u8; MAX_SNAKES];
        for p in 0..n {
            let visits = &nd.visits[p];
            let total: f32 = visits.iter().map(|&v| v as f32).sum();
            let sqrt_n = total.max(1.0).sqrt();
            let fpu = nd.value[p] - cfg.fpu_reduction;
            let mut best = f32::NEG_INFINITY;
            let mut best_a = 0u8;
            for a in 0..NUM_ACTIONS {
                if !nd.legal[p][a] {
                    continue;
                }
                let nf = visits[a] as f32;
                let q = if visits[a] > 0 {
                    mean_value(nd.value_sum[p][a], nd.value_sum_wide[p][a], visits[a])
                } else {
                    fpu
                };
                let bonus = ((cfg.c_puct * nd.prior[p][a]) * sqrt_n) / (1.0 + nf);
                let u = depth_noise(bits[p][a], depth);
                let score = (0.5 * (q + 1.0) + bonus) + cfg.tie_noise * u;
                // jnp.argmax: the first maximum; a NaN counts as the maximum.
                if score > best || (score.is_nan() && !best.is_nan()) {
                    best = score;
                    best_a = a as u8;
                }
            }
            moves[p] = best_a;
        }
        moves
    }

    /// The root summary and final moves (`_output`, mcts.py:1332-1362, `final="max"`).
    fn output(&self, n: usize, deepest: usize, sims: usize) -> SearchOutput {
        let root = self
            .root_stats()
            .expect("run pushes the root before any output");
        summarize(&root, n, self.nodes.len(), deepest, sims)
    }
}

/// The root summary and final moves from root statistics (`_output`, mcts.py:1332-1362,
/// `final="max"`), in the jitted Python's f32 order.
pub(crate) fn summarize(
    r: &RootStats,
    n: usize,
    nodes_used: usize,
    deepest: usize,
    sims: usize,
) -> SearchOutput {
    let mut out = SearchOutput {
        action: [0; MAX_SNAKES],
        policy: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
        visits: [[0; NUM_ACTIONS]; MAX_SNAKES],
        q: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
        value: [0.0; MAX_SNAKES],
        root_prior: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
        legal: [[false; NUM_ACTIONS]; MAX_SNAKES],
        nodes_used,
        depth: deepest,
        simulations: sims,
    };
    for p in 0..n.min(MAX_SNAKES) {
        let (visits, legal) = (&r.visits[p], &r.legal[p]);
        let total: i32 = visits.iter().sum();
        let q = std::array::from_fn::<f32, NUM_ACTIONS, _>(|a| {
            if visits[a] > 0 {
                mean_value(r.value_sum[p][a], r.value_sum_wide[p][a], visits[a])
            } else {
                0.0
            }
        });
        let sum = ((r.value_sum[p][0] + r.value_sum[p][1]) + r.value_sum[p][2]) + r.value_sum[p][3];
        let wide = r.value_sum_wide[p].iter().sum::<f64>();
        out.value[p] = if total > 0 {
            mean_value(sum, wide, total)
        } else {
            r.value[p]
        };
        let weight = std::array::from_fn::<f32, NUM_ACTIONS, _>(|a| {
            if legal[a] { visits[a] as f32 } else { 0.0 }
        });
        let w_total = ((weight[0] + weight[1]) + weight[2]) + weight[3];
        let policy = if w_total > 0.0 {
            // XLA turns the division by the broadcast total into a multiplication by its
            // reciprocal; doing the same keeps the shares bit-exact.
            let r = 1.0 / w_total.max(1e-30);
            weight.map(|w| w * r)
        } else {
            uniform_prior(legal)
        };
        let top_p = policy.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 1e-6;
        let mut best = f32::NEG_INFINITY;
        let mut best_a = 0u8;
        for a in 0..NUM_ACTIONS {
            if legal[a] && policy[a] >= top_p && q[a] > best {
                best = q[a];
                best_a = a as u8;
            }
        }
        out.action[p] = best_a;
        out.policy[p] = policy;
        out.visits[p] = *visits;
        out.q[p] = q;
        out.root_prior[p] = r.prior[p];
        out.legal[p] = *legal;
    }
    out
}

/// `sum_p moves[p] * 4^p` (mcts.py:486-489).
fn joint_of(moves: &[u8; MAX_SNAKES], n: usize) -> usize {
    (0..n)
        .rev()
        .fold(0, |j, p| j * NUM_ACTIONS + moves[p] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::UniformEvaluator;
    use crate::noise::JaxNoise;
    use slinky_rules::Mode;

    fn duel() -> (GameConfig, GameState) {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(1, 1), (1, 1), (1, 1)], 100, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(9, 9), (9, 9), (9, 9)], 100, None)
            .unwrap();
        s.refresh_done(&cfg);
        (cfg, s)
    }

    #[test]
    fn joint_encoding() {
        assert_eq!(joint_of(&[1, 2, 3, 0], 3), 1 + 2 * 4 + 3 * 16);
        assert_eq!(joint_of(&[3, 3, 0, 0], 2), 15);
    }

    #[test]
    fn visits_sum_to_simulations() {
        let (cfg, s) = duel();
        let mut search = Search::new();
        let out = search
            .run(
                &cfg,
                &s,
                &SearchConfig::default(),
                Budget::simulations(64),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(1),
            )
            .unwrap();
        for p in 0..2 {
            assert_eq!(out.visits[p].iter().sum::<i32>(), 64);
            assert!(out.legal[p][out.action[p] as usize]);
        }
        assert!(out.nodes_used > 1 && out.nodes_used <= 65);
        assert_eq!(out.simulations, 64);
    }

    #[test]
    fn finished_root_is_untouched() {
        let (cfg, mut s) = duel();
        s.alive[1] = false;
        s.body[1] = [0; slinky_rules::MAX_CELLS];
        s.refresh_done(&cfg);
        assert!(s.done);
        let out = Search::new()
            .run(
                &cfg,
                &s,
                &SearchConfig::default(),
                Budget::simulations(16),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(1),
            )
            .unwrap();
        assert_eq!(out.visits, [[0; 4]; 4]);
        assert_eq!(out.nodes_used, 1);
        assert_eq!(out.value[0], 1.0);
    }

    #[test]
    fn deadline_stops_early() {
        let (cfg, s) = duel();
        let out = Search::new()
            .run(
                &cfg,
                &s,
                &SearchConfig::default(),
                Budget::until(1000, Instant::now()),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(1),
            )
            .unwrap();
        assert_eq!(out.simulations, 1);
        assert_eq!(out.visits[0].iter().sum::<i32>(), 1);
    }

    fn run_duel(ws: &mut Search, budget: Budget, seed: u32) -> SearchOutput {
        let (cfg, s) = duel();
        ws.run(
            &cfg,
            &s,
            &SearchConfig::default(),
            budget,
            &mut UniformEvaluator,
            &mut JaxNoise::from_seed(seed),
        )
        .unwrap()
    }

    #[test]
    fn long_search_is_deterministic_and_consistent() {
        let mut ws = Search::new();
        let a = run_duel(&mut ws, Budget::simulations(5000), 4);
        let st = ws.stats();
        assert_eq!(st.stop, StopReason::Simulations);
        assert_eq!(
            (st.simulations, st.nodes, st.depth),
            (5000, a.nodes_used, a.depth)
        );
        assert_eq!(st.tree_bytes, a.nodes_used * node_bytes(2));
        assert!(st.allocated_bytes >= st.tree_bytes);
        assert!(a.depth <= MAX_DEPTH && a.nodes_used > 1000 && a.nodes_used <= 5001);
        for p in 0..2 {
            assert_eq!(a.visits[p].iter().sum::<i32>(), 5000);
        }
        // A reused workspace (capacity left from a bigger tree) and a fresh one agree.
        let _ = run_duel(&mut ws, Budget::simulations(9000), 5);
        let b = run_duel(&mut ws, Budget::simulations(5000), 4);
        let c = run_duel(&mut Search::new(), Budget::simulations(5000), 4);
        assert_eq!(a, b);
        assert_eq!(a, c);
        // Every simulation went through exactly one root child (none ends at the root's
        // children here: no finished child, depth limit 32).
        let mut through = 0;
        for m0 in 0..4u8 {
            for m1 in 0..4u8 {
                through += ws.subtree_simulations(&[m0, m1, 0, 0]);
            }
        }
        assert!(through <= 5000 && through > 4900, "{through}");
        assert_eq!(ws.subtree_simulations(&[9, 0, 0, 0]), 0);
    }

    #[test]
    fn memory_limit_stops_the_search() {
        let mut ws = Search::with_memory_limit(node_bytes(2) * 50);
        assert_eq!(ws.memory_limit(), node_bytes(2) * 50);
        let out = run_duel(&mut ws, Budget::simulations(1000), 1);
        let st = ws.stats();
        assert_eq!(st.stop, StopReason::MemoryLimit);
        assert_eq!(st.nodes, 50);
        assert!(out.simulations < 1000 && out.simulations >= 49);
        // The pool never grows past the limit.
        assert!(st.allocated_bytes <= 50 * node_bytes(2));
        // A limit below one node still runs one simulation.
        let mut ws = Search::with_memory_limit(0);
        let out = run_duel(&mut ws, Budget::simulations(10), 1);
        assert_eq!(out.simulations, 1);
        assert_eq!(ws.stats().stop, StopReason::MemoryLimit);
    }

    #[test]
    fn budget_helpers() {
        let t = Instant::now();
        let b = Budget::millis_after(t, 250, 77);
        assert_eq!(b.simulations, 77);
        assert_eq!(b.deadline, Some(t + Duration::from_millis(250)));
        assert!(!b.is_deterministic());
        assert!(Budget::simulations(5).is_deterministic());
        // Never panics: a far deadline, or none when the Instant cannot hold it.
        assert!(
            Budget::millis_after(t, u64::MAX, 3)
                .deadline
                .is_none_or(|d| d > t)
        );
        // A deadline far away stops at the simulation cap.
        let mut ws = Search::new();
        let out = run_duel(&mut ws, Budget::millis_after(t, 60_000, 300), 2);
        assert_eq!(out.simulations, 300);
        assert_eq!(ws.stats().stop, StopReason::Simulations);
        // A short deadline ends the search.
        let out = run_duel(
            &mut ws,
            Budget::millis_after(Instant::now(), 5, usize::MAX),
            2,
        );
        assert_eq!(ws.stats().stop, StopReason::Deadline);
        assert!(out.simulations >= 1);
        assert!(ws.stats().elapsed < Duration::from_millis(500));
    }

    #[test]
    fn short_noise_table_is_an_error() {
        let cfg = GameConfig::for_mode(Mode::Standard, Some(3)).unwrap();
        let s = GameState::new(&cfg);
        let mut table = crate::noise::NoiseTable::default();
        let err = Search::new().run(
            &cfg,
            &s,
            &SearchConfig::default(),
            Budget::simulations(4),
            &mut UniformEvaluator,
            &mut table,
        );
        // No snake alive: the root is finished, but the empty noise table still runs out.
        assert!(matches!(err, Err(SearchError::Noise { sim: 0 })));
    }

    /// Both snakes starve next turn whatever they do: every joint move leads to a finished
    /// game, so once all are expanded the root is solved. Under a deadline the search stops
    /// after `SOLVED_SIMULATIONS`; a fixed count runs in full.
    #[test]
    fn solved_tree_stops_under_a_deadline() {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(1, 1), (1, 1), (1, 1)], 1, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(9, 9), (9, 9), (9, 9)], 1, None)
            .unwrap();
        s.refresh_done(&cfg);
        assert!(!s.done);
        let mut ws = Search::new();
        let t = Instant::now();
        let run = |ws: &mut Search, budget| {
            ws.run(
                &cfg,
                &s,
                &SearchConfig::default(),
                budget,
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(3),
            )
            .unwrap()
        };
        let out = run(&mut ws, Budget::millis_after(t, 60_000, usize::MAX));
        assert_eq!(ws.stats().stop, StopReason::Solved);
        assert_eq!(out.simulations, SOLVED_SIMULATIONS);
        // The root and one finished child per joint move `select` can pick (move 0 for a seat
        // with no legal move).
        let picks = |p: usize| out.legal[p].iter().filter(|&&l| l).count().max(1);
        assert_eq!(ws.stats().nodes, 1 + picks(0) * picks(1));
        assert!(t.elapsed() < Duration::from_secs(5));
        let fixed = run(&mut ws, Budget::simulations(SOLVED_SIMULATIONS + 10));
        assert_eq!(fixed.simulations, SOLVED_SIMULATIONS + 10);
        assert_eq!(ws.stats().stop, StopReason::Simulations);
        // Two turns from starving: solved through a level of unfinished nodes.
        let mut two = s;
        two.health[..2].copy_from_slice(&[2, 2]);
        let deep = ws
            .run(
                &cfg,
                &two,
                &SearchConfig::default(),
                Budget::millis_after(Instant::now(), 60_000, usize::MAX),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(4),
            )
            .unwrap();
        assert_eq!(ws.stats().stop, StopReason::Solved);
        assert_eq!(deep.simulations, SOLVED_SIMULATIONS);
        assert!(
            ws.stats().nodes > 2 && ws.stats().depth == 2,
            "{:?}",
            ws.stats()
        );
        // An unsolved tree (full health) does not stop early.
        let mut u = s;
        u.health[..2].copy_from_slice(&[100, 100]);
        let out = ws
            .run(
                &cfg,
                &u,
                &SearchConfig::default(),
                Budget::millis_after(Instant::now(), 60_000, SOLVED_SIMULATIONS + 10),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(3),
            )
            .unwrap();
        assert_eq!(out.simulations, SOLVED_SIMULATIONS + 10);
    }
}
