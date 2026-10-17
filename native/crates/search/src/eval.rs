//! Leaf evaluation: what the search asks of the network at a new node.
//!
//! Ported from `az_modes.make_evaluator` (baselines/az_modes.py:209-223: every seat's
//! egocentric board and mode features, one forward pass), `mcts._node_prior` (mcts.py:636-648:
//! masked softmax of the logits, or uniform for `prior="uniform"`) and the evaluator half of
//! `mcts._static_value` (mcts.py:515-536). See SPEC.md 3.6, 4.4, 4.5(d) and 4.6.
//!
//! The search only asks for the seats whose outputs can matter (SPEC ambiguity 8): living seats
//! of a node whose game is not over. A dead seat's value is its exact placement value and its
//! prior is forced to move 0, and a finished node is never selected from, so Python's rows for
//! them are computed and thrown away. Skipping them changes no result, because the network's
//! rows are independent of each other.

use crate::SearchError;
use slinky_net::{NUM_ACTIONS, Network, Scratch};
use slinky_obs::{Encoder, NUM_FEATURES, OBS_LEN};
use slinky_rules::{GameConfig, GameState, MAX_SNAKES};

/// Where a node's PUCT priors come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prior {
    /// `prior="uniform"`: `legal / sum(legal)`.
    Uniform,
    /// `prior="network"`: raw logits per seat; the search applies the masked softmax.
    Logits([[f32; NUM_ACTIONS]; MAX_SNAKES]),
    /// Final priors per seat, used as given (a replay of priors computed elsewhere).
    Probs([[f32; NUM_ACTIONS]; MAX_SNAKES]),
}

/// What an [`Evaluator`] returns for one node. Entries of seats that were not asked for are
/// ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeEval {
    /// Leaf value estimate per seat (clipped to `[-1, 1]` by the search).
    pub values: [f32; MAX_SNAKES],
    /// The prior source.
    pub prior: Prior,
}

impl Default for NodeEval {
    fn default() -> Self {
        NodeEval {
            values: [0.0; MAX_SNAKES],
            prior: Prior::Uniform,
        }
    }
}

/// A leaf evaluator for [`crate::Search`].
pub trait Evaluator {
    /// Evaluate `state`, the state of node `node` (numbered in creation order, the root is 0),
    /// for the seats with `seats[i]` set: the living seats of a node whose game is not over,
    /// or of a finished root (only its priors are used there, for `root_prior`; its values are
    /// the exact placement values). The search does not call it for other finished nodes.
    fn evaluate(
        &mut self,
        node: usize,
        state: &GameState,
        seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError>;

    /// A key such that two evaluators with the same key return bit-identical evaluations of
    /// the same state and seats, whatever the node: then a search may reuse the evaluations of
    /// its previous tree ([`crate::Search::set_reuse`]). `None` (the default) never reuses, for
    /// evaluators whose answers depend on the node number or on earlier calls.
    fn reuse_key(&self) -> Option<u64> {
        None
    }
}

/// The mechanics evaluator of SPEC.md 7.5 (`prior="uniform", leaf="none"`): uniform priors over
/// the legal moves and 0 for every living seat.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(
        &mut self,
        _node: usize,
        _state: &GameState,
        _seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        Ok(NodeEval::default())
    }

    fn reuse_key(&self) -> Option<u64> {
        Some(0x756e_6966_6f72_6d00) // any constant: the answers depend on nothing
    }
}

/// The network evaluator (`az_modes.make_evaluator`): the asked seats' boards and feature rows
/// in one batch through [`Network::forward_into`]. Holds its work buffers, so evaluating
/// allocates nothing.
pub struct NetEvaluator<'a> {
    net: &'a Network,
    encoder: Encoder,
    scratch: Scratch,
    obs: Vec<f32>,
    feats: Vec<f32>,
    /// Network evaluations made (one per call that had a seat to evaluate).
    pub calls: usize,
    /// Network rows evaluated in total.
    pub rows: usize,
}

impl<'a> NetEvaluator<'a> {
    /// An evaluator for games with `config`, telling the network `encoder.told_mode()`. The
    /// board must give the network's 21x21 window (an 11x11 board).
    pub fn new(net: &'a Network, encoder: Encoder) -> Result<NetEvaluator<'a>, SearchError> {
        if encoder.obs_len() != OBS_LEN {
            let c = encoder.config();
            return Err(SearchError::Unsupported(format!(
                "the network reads 11x11 boards, not {}x{}",
                c.width(),
                c.height()
            )));
        }
        Ok(NetEvaluator {
            net,
            encoder,
            scratch: net.scratch(),
            obs: vec![0.0; MAX_SNAKES * OBS_LEN],
            feats: vec![0.0; MAX_SNAKES * NUM_FEATURES],
            calls: 0,
            rows: 0,
        })
    }

    /// The evaluator of `az_modes.make_policy` for `config`: told `config.mode()`, food hidden
    /// in constrictor when the run's `hide_constrictor_food` is set.
    pub fn for_game(net: &'a Network, config: GameConfig) -> Result<NetEvaluator<'a>, SearchError> {
        let hide = net.config().hide_constrictor_food;
        NetEvaluator::new(net, Encoder::new(config, hide))
    }

    /// The encoder used for the network inputs.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }
}

impl Evaluator for NetEvaluator<'_> {
    fn reuse_key(&self) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.net.id(), self.encoder).hash(&mut h);
        Some(h.finish())
    }

    fn evaluate(
        &mut self,
        _node: usize,
        state: &GameState,
        seats: &[bool; MAX_SNAKES],
    ) -> Result<NodeEval, SearchError> {
        let n = self.encoder.config().num_snakes().min(MAX_SNAKES);
        let feats = self.encoder.features(state);
        let mut rows = [0usize; MAX_SNAKES];
        let mut b = 0;
        for seat in (0..n).filter(|&i| seats[i]) {
            self.encoder.observe_into(
                state,
                seat,
                &mut self.obs[b * OBS_LEN..(b + 1) * OBS_LEN],
            )?;
            self.feats[b * NUM_FEATURES..(b + 1) * NUM_FEATURES].copy_from_slice(&feats[seat]);
            rows[b] = seat;
            b += 1;
        }
        let mut out = NodeEval {
            values: [0.0; MAX_SNAKES],
            prior: Prior::Logits([[0.0; NUM_ACTIONS]; MAX_SNAKES]),
        };
        if b == 0 {
            return Ok(out);
        }
        let mut logits = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        let mut values = [0.0f32; MAX_SNAKES];
        self.net.forward_into(
            &mut self.scratch,
            &self.obs[..b * OBS_LEN],
            &self.feats[..b * NUM_FEATURES],
            &mut logits[..b],
            &mut values[..b],
        )?;
        self.calls += 1;
        self.rows += b;
        let mut full = [[0.0f32; NUM_ACTIONS]; MAX_SNAKES];
        for r in 0..b {
            out.values[rows[r]] = values[r];
            full[rows[r]] = logits[r];
        }
        out.prior = Prior::Logits(full);
        Ok(out)
    }
}

/// `mcts._node_prior` for `prior="network"` (mcts.py:636-648): logits clipped to `+-1e30`,
/// `-inf` on illegal moves, `e = exp(l - max)`, `p = e * (1 / sum(e))`, in f32. Illegal moves
/// get exactly 0 and a row with one legal move gets exactly 1 on it.
///
/// On the same logits this equals the jitted `jax.nn.softmax` in about 85% of rows and is
/// within 2 ulp in the rest: XLA's `exp` is a polynomial approximation, not the correctly
/// rounded `f32::exp` (`tests/golden.rs`, `prior_softmax_close`).
pub fn masked_softmax(
    logits: &[f32; NUM_ACTIONS],
    legal: &[bool; NUM_ACTIONS],
) -> [f32; NUM_ACTIONS] {
    let mut l = [f32::NEG_INFINITY; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            l[a] = logits[a].clamp(-1e30, 1e30);
        }
    }
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            e[a] = (l[a] - m).exp();
        }
    }
    // XLA multiplies by the reciprocal of the broadcast sum rather than dividing.
    let r = 1.0 / (((e[0] + e[1]) + e[2]) + e[3]);
    e.map(|x| x * r)
}

/// `prior="uniform"` (mcts.py:644-645): `legal / sum(legal)`.
pub fn uniform_prior(legal: &[bool; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let k = legal.iter().filter(|&&b| b).count() as f32;
    legal.map(|b| if b { 1.0 / k } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_forced_and_masked() {
        let p = masked_softmax(&[3.0, 9.0, -2.0, 0.5], &[true, false, false, false]);
        assert_eq!(p, [1.0, 0.0, 0.0, 0.0]);
        let p = masked_softmax(&[0.0, 0.0, 5.0, 0.0], &[true, true, false, true]);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[0], 1.0 * (1.0f32 / 3.0));
        let p = masked_softmax(&[f32::NEG_INFINITY; 4], &[true, true, false, false]);
        assert_eq!(p, [0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn uniform() {
        assert_eq!(uniform_prior(&[true, false, true, true])[3], 1.0 / 3.0);
        assert_eq!(
            uniform_prior(&[false, true, false, false]),
            [0.0, 1.0, 0.0, 0.0]
        );
    }
}
