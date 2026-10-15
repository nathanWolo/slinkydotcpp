//! What the search reads from a state besides the step: forced moves, legal moves and the exact
//! placement values.
//!
//! Ported from `rules.certainly_starving` (rules.py:239-252), `rules.blocked_cells`
//! (rules.py:255-263), `rules.action_mask` (rules.py:266-281), `mcts._legal` (mcts.py:492-495) and
//! `mcts.placement_values` (mcts.py:539-564). See SPEC sections 4.4 and 4.6.

// The rules are a pipeline over parallel per-seat arrays (as in the Python); indexing them by
// seat reads more clearly than zipping five iterators.
#![allow(clippy::needless_range_loop)]

use crate::GameState;
use crate::config::{DELTAS, GameConfig, MAX_SNAKES, NUM_ACTIONS};

/// Per-seat flags over the 4 moves, `[seat][action]`. Rows at seats `>= num_snakes` are all
/// false.
pub type MoveMask = [[bool; NUM_ACTIONS]; MAX_SNAKES];

impl GameState {
    /// Living snakes that will certainly starve before collisions are checked next turn
    /// (`rules.certainly_starving`, rules.py:239-252): `health <= 1` and no food on any of the 4
    /// neighbouring cells (nor a hazard cell there when the hazard damage is negative, i.e. heals).
    /// Their bodies block nobody and the search forces them to move 0.
    pub fn certainly_starving(&self, config: &GameConfig) -> [bool; MAX_SNAKES] {
        let heals = config.hazard_damage_per_turn() < 0;
        let mut out = [false; MAX_SNAKES];
        for (i, o) in out.iter_mut().enumerate().take(config.num_snakes()) {
            if !self.alive[i] || self.health[i] > 1 {
                continue;
            }
            let [x, y] = self.head[i];
            let rescue = DELTAS.iter().any(|&(dx, dy)| {
                let (nx, ny) = (x.wrapping_add(dx), y.wrapping_add(dy));
                self.food_at(config, nx, ny) || (heals && self.hazard_at(config, nx, ny) > 0)
            });
            *o = !rescue;
        }
        out
    }

    /// Cells (bitboard, bit `y * width + x`) that hold a body segment next turn whatever anyone
    /// does (`rules.blocked_cells`, rules.py:255-263): countdown above 1 of a living snake that is
    /// not certainly starving.
    pub fn blocked_cells(&self, config: &GameConfig) -> u128 {
        self.blocked_with(config, &self.certainly_starving(config))
    }

    fn blocked_with(&self, config: &GameConfig, starving: &[bool; MAX_SNAKES]) -> u128 {
        let mut blocked = 0u128;
        for i in 0..config.num_snakes() {
            if self.alive[i] && !starving[i] {
                for (c, &v) in self.body[i][..config.cells()].iter().enumerate() {
                    if v > 1 {
                        blocked |= 1u128 << c;
                    }
                }
            }
        }
        blocked
    }

    /// Moves that do not certainly die to a wall or a body next turn (`rules.action_mask`,
    /// rules.py:266-281). Rows of dead or certainly-starving snakes, and rows with no such move,
    /// are all true. This is the mask the environment reports; the search uses
    /// [`legal_moves`](GameState::legal_moves).
    pub fn action_mask(&self, config: &GameConfig) -> MoveMask {
        self.mask_with(config, &self.certainly_starving(config))
    }

    fn mask_with(&self, config: &GameConfig, starving: &[bool; MAX_SNAKES]) -> MoveMask {
        let n = config.num_snakes();
        // Only the cells next to heads matter: test those instead of building the bitboard.
        let blocked_at =
            |c: usize| (0..n).any(|j| self.alive[j] && !starving[j] && self.body[j][c] > 1);
        let mut mask = [[false; NUM_ACTIONS]; MAX_SNAKES];
        for (i, row) in mask.iter_mut().enumerate().take(n) {
            let [x, y] = self.head[i];
            for (ok, &(dx, dy)) in row.iter_mut().zip(DELTAS.iter()) {
                *ok = config
                    .cell(x.wrapping_add(dx), y.wrapping_add(dy))
                    .is_some_and(|c| !blocked_at(c));
            }
            if !self.alive[i] || starving[i] || !row.iter().any(|&b| b) {
                *row = [true; NUM_ACTIONS];
            }
        }
        mask
    }

    /// The moves the search may choose at this state (`mcts._legal(state, action_mask(state))`,
    /// mcts.py:492-495): the action mask, except that dead and certainly-starving seats get move 0
    /// (UP) only.
    pub fn legal_moves(&self, config: &GameConfig) -> MoveMask {
        let starving = self.certainly_starving(config);
        let mut legal = self.mask_with(config, &starving);
        for (i, row) in legal.iter_mut().enumerate().take(config.num_snakes()) {
            if self.forced_with(i, &starving) {
                *row = [true, false, false, false];
            }
        }
        legal
    }

    /// Seats whose move the search fixes to 0: dead or certainly starving (mcts.py:494).
    pub fn forced(&self, config: &GameConfig) -> [bool; MAX_SNAKES] {
        let starving = self.certainly_starving(config);
        let mut out = [false; MAX_SNAKES];
        for (i, o) in out.iter_mut().enumerate().take(config.num_snakes()) {
            *o = self.forced_with(i, &starving);
        }
        out
    }

    fn forced_with(&self, seat: usize, starving: &[bool; MAX_SNAKES]) -> bool {
        !self.alive[seat] || starving[seat]
    }

    /// Placement values `f32[N]` (`mcts.placement_values`, mcts.py:539-564):
    /// `(outlasted - outlasted_by) / (N - 1)`, where a living snake outlasts every dead one and
    /// snakes eliminated on the same turn tie. When nobody is alive, the snakes eliminated last
    /// also get `draw_value * (ties - 1) / (N - 1)`. Exact for dead seats always and for every
    /// seat once the game is over; the search replaces living seats' values with the network's
    /// before then. Entries at seats `>= num_snakes` are 0.
    ///
    /// The f32 arithmetic matches the jitted Python bit for bit: the division by the constant
    /// `N - 1` is a multiplication by its f32 reciprocal, as XLA rewrites it.
    pub fn placement_values(&self, config: &GameConfig, draw_value: f32) -> [f32; MAX_SNAKES] {
        let n = config.num_snakes();
        let mut out = [i32::MAX; MAX_SNAKES];
        for (i, o) in out.iter_mut().enumerate().take(n) {
            if !self.alive[i] {
                *o = self.elim_turn[i];
            }
        }
        let out = &out[..n];
        let recip = 1.0f32 / (n - 1) as f32;
        let last = out.iter().copied().max().unwrap_or(i32::MAX);
        let nobody = !self.alive[..n].iter().any(|&a| a);
        let tied = out.iter().filter(|&&o| o == last).count() as f32 - 1.0;
        let draw = draw_value * tied * recip;
        let mut values = [0f32; MAX_SNAKES];
        for (i, v) in values.iter_mut().enumerate().take(n) {
            let outlasted = out.iter().filter(|&&o| out[i] > o).count() as i32;
            let outlasted_by = out.iter().filter(|&&o| o > out[i]).count() as i32;
            *v = (outlasted - outlasted_by) as f32 * recip;
            if nobody && out[i] == last {
                *v += draw;
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cause, GameConfig, GameState, MAX_SNAKES, Ruleset};

    fn four() -> GameConfig {
        GameConfig::new(11, 11, 4, Ruleset::Standard).unwrap()
    }

    #[test]
    fn placement_examples() {
        let c = four();
        let mut s = GameState::new(&c);
        for i in 0..4 {
            s.set_snake(&c, i, &[(i as i32, 0)], 100, None).unwrap();
        }
        // Places 1st to 4th: +1, +1/3, -1/3, -1.
        s.alive = [true, false, false, false];
        s.elim_turn = [0, 9, 5, 2];
        let v = s.placement_values(&c, -0.5);
        assert_eq!(v, [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0]);
        // All four die on the same turn: -0.5 each.
        s.alive = [false; MAX_SNAKES];
        s.elim_turn = [7; MAX_SNAKES];
        assert_eq!(s.placement_values(&c, -0.5), [-0.5; 4]);
        // The last two die together after the others: 2/3 - 1/6 = 0.5 each.
        s.elim_turn = [7, 7, 3, 1];
        let v = s.placement_values(&c, -0.5);
        assert_eq!(v[..2], [0.5, 0.5]);
        assert_eq!(v[2], -1.0 / 3.0);
        // Duel draw.
        let d = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&d);
        s.elim_turn = [4, 4, 0, 0];
        assert_eq!(s.placement_values(&d, -0.5), [-0.5, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn starving_snake_is_forced_and_does_not_block() {
        let c = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&c);
        s.set_snake(&c, 0, &[(5, 5), (5, 4), (5, 3), (5, 2)], 1, None)
            .unwrap();
        s.set_snake(&c, 1, &[(6, 4), (7, 4), (8, 4)], 50, None)
            .unwrap();
        s.refresh_done(&c);
        assert_eq!(s.certainly_starving(&c)[..2], [true, false]);
        assert_eq!(s.forced(&c)[..2], [true, false]);
        assert_eq!(s.legal_moves(&c)[0], [true, false, false, false]);
        assert_eq!(s.action_mask(&c)[0], [true; 4]);
        // Snake 1 may move left onto (5, 4): the starving body does not block.
        assert_eq!(s.action_mask(&c)[1], [true, true, true, false]);
        // Food next to the head rescues it.
        s.add_food(&c, 4, 5).unwrap();
        assert!(!s.certainly_starving(&c)[0]);
        assert_eq!(s.legal_moves(&c)[0], [true, false, true, true]);
        assert_eq!(s.action_mask(&c)[1], [true, true, false, false]);
        // A healing hazard next to the head rescues too, only when damage is negative.
        let mut t = s;
        t.food = 0;
        t.add_hazard(&c, 6, 5);
        assert!(t.certainly_starving(&c)[0]);
        let heal = c.with_hazard_damage(-5);
        assert!(!t.certainly_starving(&heal)[0]);
        // Dead seats: forced, all-true mask.
        t.set_snake(&c, 1, &[], 0, Some((Cause::Collision, 0)))
            .unwrap();
        assert_eq!(t.legal_moves(&c)[1], [true, false, false, false]);
        assert_eq!(t.action_mask(&c)[1], [true; 4]);
    }

    #[test]
    fn no_safe_move_gives_all_true_row() {
        let c = GameConfig::new(3, 3, 2, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&c);
        // Snake 0 in the corner, boxed in by snake 1's body.
        s.set_snake(&c, 0, &[(0, 0), (0, 0), (0, 0)], 100, None)
            .unwrap();
        s.set_snake(
            &c,
            1,
            &[(2, 0), (2, 1), (1, 1), (0, 1), (0, 2)][..],
            100,
            None,
        )
        .unwrap();
        // (1, 0) is free and (0, 1) has countdown 2 -> blocked.
        assert_eq!(s.action_mask(&c)[0], [false, false, false, true]);
        s.set_snake(
            &c,
            1,
            &[(1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2)][..],
            100,
            None,
        )
        .unwrap();
        assert_eq!(s.action_mask(&c)[0], [true; 4]);
        assert_eq!(s.legal_moves(&c)[0], [true; 4]);
    }
}
