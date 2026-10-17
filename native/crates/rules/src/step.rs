//! One turn of the rules, as the search simulates it.
//!
//! Ported from `rules.rules_step` (rules.py:111-228), `rules.next_heads` (rules.py:74-87),
//! `rules.royale_hazards` (rules.py:90-108) and the search's transition `mcts._transition` with
//! `spawn_food=False` (mcts.py:498-512). The stage order is the engine's (SPEC section 5.2):
//!
//! 0. game-over guard (at most one snake alive: nothing changes);
//! 1. move: pop every living tail (countdown - 1), then write the old length at the new head;
//! 2. starvation (health - 1);
//! 3. hazard damage from the pre-step hazard grid, skipped on food, per stacked layer;
//! 4. feed: every living snake on food eats it (health 100, length + 1, tail stacks);
//! 5. eliminate pass 1: out of health, then out of bounds;
//! 6. eliminate pass 2, simultaneously among the survivors: self, body, head-to-head (the
//!    shorter or equal snake dies), against the post-pop, pre-feed bodies;
//! 7. constrictor: no food, full health, growth unless the tail is already stacked;
//! 8. royale: the zone for `turn + 1`, regenerated from the shrink sides;
//! 9. bodies of eliminated snakes are cleared.
//!
//! No food spawns here (`spawn_food=False`; SPEC 4.10) and wrapped boards are not supported.
//! Integer arithmetic wraps like XLA's int32 so hostile values cannot panic.

// The rules are a pipeline over parallel per-seat arrays (as in the Python); indexing them by
// seat reads more clearly than zipping five iterators.
#![allow(clippy::needless_range_loop)]

use crate::config::{DELTAS, GameConfig, MAX_CELLS, MAX_HEALTH, MAX_SIDES, MAX_SNAKES, Ruleset};
use crate::{Cause, GameState};

/// Joint action: one action id per seat (entries at seats `>= num_snakes` are ignored). Ids 0 to 3
/// are UP, DOWN, LEFT, RIGHT; any other value is an invalid move and repeats the seat's
/// `last_move`, the engine's default (rules.py:77-82).
pub type Actions = [u8; MAX_SNAKES];

/// Row of `DELTAS` that `_DELTAS[move]` reads in the jitted Python (rules.py:83) for a stored
/// move id. `last_move` is int8 there: a negative id wraps once (`id + 4`, numpy-style), and
/// the gather then clamps to `[0, 3]`. Only reachable with a hand-set out-of-range `last_move`
/// (the step and `set_snake` store 0 to 3), but it keeps the port exact for any state.
fn jax_delta_index(mv: u8) -> usize {
    let id = i32::from(mv as i8);
    let id = if id < 0 { id + 4 } else { id };
    id.clamp(0, 3) as usize
}

/// The royale hazard grid (0 or 1 per cell) of the zone shown at `turn`
/// (`rules.royale_hazards`, rules.py:90-108): `turn / shrink_every_n_turns` shrinks have happened
/// (floor division), and the first that many `sides` say which edge each one moved
/// (0 left, 1 right, 2 bottom, 3 top; other values move nothing).
pub fn royale_hazards(config: &GameConfig, sides: &[i8; MAX_SIDES], turn: i32) -> [i16; MAX_CELLS] {
    let shrinks = turn.div_euclid(config.shrink_every_n_turns());
    let mut c = [0i32; 4];
    for (k, &side) in sides[..config.num_sides()].iter().enumerate() {
        if (k as i64) < shrinks as i64 && (0..4).contains(&side) {
            c[side as usize] += 1;
        }
    }
    let (w, h) = (config.width() as i32, config.height() as i32);
    let mut grid = [0i16; MAX_CELLS];
    for y in 0..h {
        for x in 0..w {
            let outside = x < c[0] || x > w - 1 - c[1] || y < c[2] || y > h - 1 - c[3];
            grid[(y * w + x) as usize] = outside as i16;
        }
    }
    grid
}

impl GameState {
    /// Apply one turn of the ruleset pipeline in place (`rules.rules_step`, rules.py:111-228):
    /// no food spawning and **no turn increment**. If at most one snake is alive the state is left
    /// unchanged, as the engine does.
    pub fn rules_step(&mut self, config: &GameConfig, actions: &Actions) {
        let n = config.num_snakes();
        if self.is_game_over(config) {
            return;
        }
        let alive0 = self.alive;

        // --- Move: next heads (rules.py:74-87, 131-137).
        let mut moves = [0u8; MAX_SNAKES];
        let mut head_cell: [Option<usize>; MAX_SNAKES] = [None; MAX_SNAKES];
        for i in 0..n {
            if !alive0[i] {
                continue;
            }
            let a = actions[i];
            // An invalid action repeats the stored last_move, kept raw as Python keeps it.
            let mv = if a < 4 { a } else { self.last_move[i] };
            moves[i] = mv;
            let (dx, dy) = DELTAS[jax_delta_index(mv)];
            let [x, y] = self.head[i];
            let nh = [x.wrapping_add(dx), y.wrapping_add(dy)];
            self.head[i] = nh;
            head_cell[i] = config.cell(nh[0], nh[1]);
        }
        // Pop the tails: body_dec = max(body - 1, 0) for living snakes.
        for i in 0..n {
            if alive0[i] {
                for v in self.body[i].iter_mut() {
                    *v = v.wrapping_sub(1).max(0);
                }
            }
        }
        // hits[i][j]: head i lands on a cell snake j still occupies after its tail pop.
        let mut hits = [[false; MAX_SNAKES]; MAX_SNAKES];
        for i in 0..n {
            if let Some(c) = head_cell[i] {
                for j in 0..n {
                    hits[i][j] = self.body[j][c] > 0;
                }
            }
        }
        // Write the old length at the new head (living snakes on the board only).
        for i in 0..n {
            if let (true, Some(c)) = (alive0[i], head_cell[i]) {
                self.body[i][c] = self.length[i] as i16;
            }
        }

        // --- Starvation (rules.py:140).
        for i in 0..n {
            if alive0[i] {
                self.health[i] = self.health[i].wrapping_sub(1);
            }
        }

        // --- Hazard damage from the pre-step grid, skipped on food (rules.py:143-153).
        let damage = config
            .hazard_damage_per_turn()
            .clamp(-MAX_HEALTH, MAX_HEALTH);
        let mut food_at_head = [false; MAX_SNAKES];
        let mut hazard_elim = [false; MAX_SNAKES];
        let mut alive1 = [false; MAX_SNAKES];
        for i in 0..n {
            let (food, layers) = match head_cell[i] {
                // Dead seats keep their old head; only living ones are read below.
                Some(c) => (self.has_food(c), self.hazard[c] as i32),
                None => (false, 0),
            };
            food_at_head[i] = food;
            if alive0[i] && layers > 0 && !food {
                let hit = self.health[i]
                    .wrapping_sub(layers.min(MAX_HEALTH + 1).wrapping_mul(damage))
                    .clamp(0, MAX_HEALTH);
                self.health[i] = hit;
                hazard_elim[i] = hit <= 0;
            }
            alive1[i] = alive0[i] && !hazard_elim[i];
        }

        // --- Feed (rules.py:156-161).
        let mut eats = [false; MAX_SNAKES];
        for i in 0..n {
            if alive1[i] && food_at_head[i] {
                eats[i] = true;
                if let Some(c) = head_cell[i] {
                    self.food &= !(1u128 << c);
                }
                self.health[i] = MAX_HEALTH;
                self.length[i] = self.length[i].wrapping_add(1);
                grow(&mut self.body[i]);
            }
        }

        // --- Eliminate, pass 1: out of health, then out of bounds (rules.py:164-166).
        let mut starved = [false; MAX_SNAKES];
        let mut walled = [false; MAX_SNAKES];
        let mut alive2 = [false; MAX_SNAKES];
        for i in 0..n {
            starved[i] = alive1[i] && self.health[i] <= 0;
            walled[i] = alive1[i] && !starved[i] && head_cell[i].is_none();
            alive2[i] = alive1[i] && !starved[i] && !walled[i];
        }

        // --- Eliminate, pass 2: collisions among the pass-1 survivors (rules.py:171-179).
        let mut self_hit = [false; MAX_SNAKES];
        let mut body_hit = [false; MAX_SNAKES];
        let mut h2h = [false; MAX_SNAKES];
        let mut alive3 = [false; MAX_SNAKES];
        for i in 0..n {
            if !alive2[i] {
                continue;
            }
            self_hit[i] = hits[i][i];
            body_hit[i] = !self_hit[i] && (0..n).any(|j| j != i && alive2[j] && hits[i][j]);
            h2h[i] = !self_hit[i]
                && !body_hit[i]
                && (0..n).any(|j| {
                    j != i
                        && alive2[j]
                        && self.head[j] == self.head[i]
                        && self.length[i] <= self.length[j]
                });
            alive3[i] = !self_hit[i] && !body_hit[i] && !h2h[i];
        }

        // Causes, in precedence order, and the turn of elimination (rules.py:181-195).
        let next_turn = self.turn.wrapping_add(1);
        for i in 0..n {
            if alive0[i] && !alive3[i] {
                let cause = if hazard_elim[i] {
                    Cause::Hazard
                } else if starved[i] {
                    Cause::OutOfHealth
                } else if walled[i] {
                    Cause::OutOfBounds
                } else if self_hit[i] {
                    Cause::SelfCollision
                } else if body_hit[i] {
                    Cause::Collision
                } else {
                    Cause::HeadCollision
                };
                self.elim_cause[i] = cause as i8;
                self.elim_turn[i] = next_turn;
            }
        }

        // --- Constrictor (rules.py:199-205): no food, full health for every seat, and growth
        // for survivors whose tail (smallest positive countdown) is a single segment.
        if config.ruleset() == Ruleset::Constrictor {
            self.food = 0;
            for h in self.health[..n].iter_mut() {
                *h = MAX_HEALTH;
            }
            for i in 0..n {
                if alive3[i] {
                    let tail = self.body[i]
                        .iter()
                        .map(|&v| if v > 0 { v } else { i16::MAX })
                        .fold(i16::MAX, i16::min);
                    if tail == 1 {
                        self.length[i] = self.length[i].wrapping_add(1);
                        grow(&mut self.body[i]);
                    }
                }
            }
        }

        // --- Royale: the zone of the next turn, from scratch (rules.py:209-211).
        if config.ruleset() == Ruleset::Royale {
            self.hazard = royale_hazards(config, &self.sides, next_turn);
        }

        // --- Clear the bodies of every seat not alive now (rules.py:213).
        for i in 0..n {
            if !alive3[i] {
                self.body[i] = [0; MAX_CELLS];
            }
            if alive0[i] {
                self.last_move[i] = moves[i];
            }
        }
        self.alive = alive3;
    }

    /// The search's transition in place (`mcts._transition` with `spawn_food=False`,
    /// mcts.py:498-512): [`rules_step`](GameState::rules_step), then `turn + 1` and
    /// `done = is_game_over` (or `turn >= max_turns` when the config truncates). A state that is
    /// already `done` is left entirely unchanged (turn included).
    pub fn advance(&mut self, config: &GameConfig, actions: &Actions) {
        if self.done {
            return;
        }
        self.rules_step(config, actions);
        self.turn = self.turn.wrapping_add(1);
        let truncated = config.max_turns().is_some_and(|m| self.turn >= m);
        self.done = self.is_game_over(config) || truncated;
    }

    /// The child state the search expands for a joint action: a copy of `self` after
    /// [`advance`](GameState::advance).
    pub fn child(&self, config: &GameConfig, actions: &Actions) -> GameState {
        let mut next = *self;
        next.advance(config, actions);
        next
    }
}

/// Eating or constrictor growth: every occupied cell's countdown + 1, which stacks the tail.
#[inline]
fn grow(grid: &mut [i16; MAX_CELLS]) {
    for v in grid.iter_mut() {
        *v = v.wrapping_add((*v > 0) as i16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Move;

    const U: u8 = Move::Up as u8;
    const D: u8 = Move::Down as u8;
    const L: u8 = Move::Left as u8;
    const R: u8 = Move::Right as u8;

    fn duel() -> GameConfig {
        GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap()
    }

    fn state(c: &GameConfig, snakes: &[&[(i32, i32)]]) -> GameState {
        let mut s = GameState::new(c);
        for (i, b) in snakes.iter().enumerate() {
            s.set_snake(c, i, b, 100, None).unwrap();
        }
        s.refresh_done(c);
        s
    }

    #[test]
    fn move_pops_tail_and_writes_head() {
        let c = duel();
        let mut s = state(&c, &[&[(1, 1), (1, 1), (1, 1)], &[(9, 9), (9, 8), (9, 7)]]);
        s.advance(&c, &[U, U, 0, 0]);
        assert_eq!(s.head[0], [1, 2]);
        assert_eq!(s.body_at(&c, 0, 1, 2), 3);
        assert_eq!(s.body_at(&c, 0, 1, 1), 2);
        assert_eq!(s.body_at(&c, 1, 9, 10), 3);
        assert_eq!(s.body_at(&c, 1, 9, 7), 0);
        assert_eq!(s.health, [99, 99, 0, 0]);
        assert_eq!(s.turn, 1);
        assert!(!s.done);
    }

    #[test]
    fn equal_head_to_head_kills_both_and_draws() {
        let c = duel();
        let mut s = state(&c, &[&[(4, 5), (3, 5), (2, 5)], &[(6, 5), (7, 5), (8, 5)]]);
        s.advance(&c, &[R, L, 0, 0]);
        assert_eq!(s.alive[..2], [false, false]);
        assert_eq!(s.elim_cause[..2], [Cause::HeadCollision as i8; 2]);
        assert_eq!(s.elim_turn[..2], [1, 1]);
        assert!(s.done);
        assert_eq!(s.body[0], [0; MAX_CELLS]);
        assert_eq!(s.placement_values(&c, -0.5)[..2], [-0.5, -0.5]);
    }

    #[test]
    fn eating_stacks_tail_and_wins_head_to_head() {
        let c = duel();
        let mut s = state(&c, &[&[(4, 5), (3, 5), (2, 5)], &[(6, 5), (7, 5), (8, 5)]]);
        s.add_food(&c, 5, 5).unwrap();
        s.add_food(&c, 0, 0).unwrap();
        // Both reach the same food: both eat, equal lengths, both die.
        let both = s.child(&c, &[R, L, 0, 0]);
        assert_eq!(both.alive[..2], [false, false]);
        assert!(!both.food_at(&c, 5, 5));
        // Only snake 0 eats.
        s.add_food(&c, 4, 6).unwrap();
        let one = s.child(&c, &[U, D, 0, 0]);
        assert_eq!(one.length[..2], [4, 3]);
        assert_eq!(one.health[..2], [100, 99]);
        assert_eq!(one.body_at(&c, 0, 4, 6), 4);
        assert_eq!(one.body_at(&c, 0, 3, 5), 2); // stacked tail
        // Next move: the stacked tail does not pop away.
        let two = one.child(&c, &[U, D, 0, 0]);
        assert_eq!(two.body_at(&c, 0, 3, 5), 1);
    }

    #[test]
    fn walls_self_and_tail_chasing() {
        let c = duel();
        let mut s = state(
            &c,
            &[&[(0, 5), (0, 4), (0, 3)], &[(5, 5), (5, 6), (6, 6), (6, 5)]],
        );
        let walled = s.child(&c, &[L, 0, 0, 0]);
        assert_eq!(walled.elim_cause[0], Cause::OutOfBounds as i8);
        assert_eq!(walled.head[0], [-1, 5]);
        assert!(walled.done);
        // Snake 1 may move onto its own tail (countdown 1 pops first).
        let chase = s.child(&c, &[U, R, 0, 0]);
        assert!(chase.alive[1]);
        assert_eq!(chase.body_at(&c, 1, 6, 5), 4);
        // ... but not onto its neck.
        let selfhit = s.child(&c, &[U, U, 0, 0]);
        assert_eq!(selfhit.elim_cause[1], Cause::SelfCollision as i8);
        s.done = true;
        assert_eq!(s.child(&c, &[L, L, 0, 0]), s);
    }

    #[test]
    fn royale_zone() {
        let c = GameConfig::new(11, 11, 2, Ruleset::Royale)
            .unwrap()
            .with_shrink_every(25)
            .unwrap();
        let mut sides = [crate::UNKNOWN_SIDE; MAX_SIDES];
        sides[..3].copy_from_slice(&[0, 3, 0]);
        assert_eq!(royale_hazards(&c, &sides, 24), [0; MAX_CELLS]);
        let z = royale_hazards(&c, &sides, 50);
        assert_eq!(z[0], 1); // (0, 0): left column
        assert_eq!(z[1], 0);
        assert_eq!(z[10 * 11 + 5], 1); // top row
        assert_eq!(z.iter().map(|&v| v as i32).sum::<i32>(), 21);
        // A third shrink on the left side.
        assert_eq!(royale_hazards(&c, &sides, 75)[1], 1);
        // Unknown sides move nothing.
        assert_eq!(
            royale_hazards(&c, &sides, 1000),
            royale_hazards(&c, &sides, 75)
        );
        assert_eq!(royale_hazards(&c, &sides, -3), [0; MAX_CELLS]);
    }

    #[test]
    fn constrictor_grows_every_turn() {
        let c = GameConfig::new(11, 11, 2, Ruleset::Constrictor).unwrap();
        let mut s = state(&c, &[&[(1, 1), (1, 1), (1, 1)], &[(9, 9), (9, 9), (9, 9)]]);
        s.health = [50, 7, 0, 0];
        // The turn-0 stack still holds a stacked tail after one move: no growth yet.
        s.advance(&c, &[U, D, 0, 0]);
        assert_eq!(s.length[..2], [3, 3]);
        assert_eq!(s.health[..2], [100, 100]);
        assert_eq!(s.body_at(&c, 0, 1, 1), 2);
        s.advance(&c, &[U, D, 0, 0]);
        assert_eq!(s.length[..2], [4, 4]);
        assert_eq!(s.body_at(&c, 0, 1, 1), 2); // tail restacked
        assert_eq!(s.body_at(&c, 0, 1, 3), 4);
        s.advance(&c, &[U, D, 0, 0]);
        assert_eq!(s.length[..2], [5, 5]);
    }

    #[test]
    fn invalid_action_repeats_last_move() {
        let c = duel();
        let mut s = state(&c, &[&[(4, 5), (3, 5), (2, 5)], &[(6, 2), (6, 1), (6, 0)]]);
        s.advance(&c, &[9, 4, 0, 0]);
        assert_eq!(s.head[0], [5, 5]);
        assert_eq!(s.head[1], [6, 3]);
    }
}
