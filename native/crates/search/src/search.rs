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
//! simulations that expand nothing, on finished nodes or for dead seats (SPEC ambiguity 8).
//!
//! The search can also stop at a deadline ([`Budget::deadline`]); without one it runs exactly
//! [`Budget::simulations`] simulations, as Python does.

use crate::SearchError;
use crate::eval::{Evaluator, Prior, masked_softmax, uniform_prior};
use crate::noise::{NoiseSource, depth_noise};
use slinky_rules::{Actions, GameConfig, GameState, MAX_SNAKES, MoveMask, NUM_ACTIONS};
use std::time::Instant;

/// The play-time `max_depth` (`MCTSConfig.max_depth`, mcts.py:348-367).
pub const MAX_DEPTH: usize = 32;

/// Marks an unexpanded joint action in the children table.
const NONE: u32 = u32::MAX;

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
    value_sum: [[f32; NUM_ACTIONS]; MAX_SNAKES],
    terminal: bool,
}

/// A reusable search workspace: the tree's nodes and children table. Keep one per thread to
/// avoid reallocating between moves.
#[derive(Debug, Default)]
pub struct Search {
    nodes: Vec<Node>,
    /// `children[node * J + joint]`, [`NONE`] when not expanded.
    children: Vec<u32>,
    /// `(node, joint, moves)` per edge of the current descent.
    path: Vec<(u32, u16, [u8; MAX_SNAKES])>,
}

impl Search {
    /// An empty workspace.
    pub fn new() -> Search {
        Search::default()
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
        let joints = NUM_ACTIONS.pow(n as u32);
        let max_depth = settings.max_depth.min(budget.simulations);
        self.nodes.clear();
        self.children.clear();
        // Room for the faithful 256-simulation tree; a longer timed search grows as needed.
        let expect = budget.simulations.min(4096) + 1;
        self.nodes.reserve(expect);
        self.children.reserve(expect * joints);

        let root = self.make_node(0, *state, config, settings, evaluator)?;
        self.push(root, joints);

        let mut deepest = 0usize;
        let mut sims = 0usize;
        for sim in 0..budget.simulations {
            if sim > 0 && budget.deadline.is_some_and(|d| Instant::now() >= d) {
                break;
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
                    let new = self.make_node(id, next, config, settings, evaluator)?;
                    self.push(new, joints);
                    self.children[node * joints + joint] = id as u32;
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
                }
            }
        }
        Ok(self.output(n, deepest, sims))
    }

    fn push(&mut self, node: Node, joints: usize) {
        self.nodes.push(node);
        self.children.resize(self.children.len() + joints, NONE);
    }

    /// A new node for `state` (`_init_tree` / `_record`): legal moves, priors and values.
    fn make_node(
        &self,
        id: usize,
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
        let mut seats = [false; MAX_SNAKES];
        if !terminal {
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
        Ok(Node {
            state,
            legal,
            prior,
            value,
            visits: [[0; NUM_ACTIONS]; MAX_SNAKES],
            value_sum: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            terminal,
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
                    nd.value_sum[p][a] / nf.max(1.0)
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
        let r = &self.nodes[0];
        let mut out = SearchOutput {
            action: [0; MAX_SNAKES],
            policy: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            visits: [[0; NUM_ACTIONS]; MAX_SNAKES],
            q: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            value: [0.0; MAX_SNAKES],
            root_prior: [[0.0; NUM_ACTIONS]; MAX_SNAKES],
            legal: [[false; NUM_ACTIONS]; MAX_SNAKES],
            nodes_used: self.nodes.len(),
            depth: deepest,
            simulations: sims,
        };
        for p in 0..n {
            let (visits, legal) = (&r.visits[p], &r.legal[p]);
            let total: i32 = visits.iter().sum();
            let q = std::array::from_fn::<f32, NUM_ACTIONS, _>(|a| {
                if visits[a] > 0 {
                    r.value_sum[p][a] / visits[a] as f32
                } else {
                    0.0
                }
            });
            let sum =
                ((r.value_sum[p][0] + r.value_sum[p][1]) + r.value_sum[p][2]) + r.value_sum[p][3];
            out.value[p] = if total > 0 {
                sum / total as f32
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
}
