//! The game state: countdown-grid bodies, food, hazards and per-seat scalars, in fixed-size arrays.
//!
//! Ported from `types.State` (types.py:118-138) and `types.RoyaleState` (types.py:141-151), with
//! the builders of `engine_json.state_from_engine` (engine_json.py:35-42 `body_to_countdown`,
//! 73-92 `_direction`, 95-151). Grids are `[y][x]` flattened to `y * width + x`, row 0 at the
//! bottom, exactly as the API and Python store them (no flips).
//!
//! # Representation
//!
//! [`GameState`] is `Copy` and about 1.3 KB: one `[i16; 121]` countdown grid per seat, a `u128`
//! food bitboard, an `[i16; 121]` hazard-count grid and a few scalars per seat. It is cloned per
//! search node and stepped in place with no heap allocation. Boards smaller than 11x11 use the
//! first `width * height` entries; the rest stay zero, so derived equality compares exactly the
//! board contents.
//!
//! A snake's countdown at a cell is `length - (smallest body index on that cell)`: the head holds
//! `length`, the tail 1, or 2 when the tail is stacked after eating (types.py:7-16). Dead seats
//! have an all-zero grid.

use crate::config::{GameConfig, MAX_CELLS, MAX_SIDES, MAX_SNAKES, Move};
use crate::{Cause, RulesError};

/// Full game state (`types.State`). Indexing is by seat `0..config.num_snakes()`; entries at
/// higher seats are unused and stay at their [`GameState::new`] values.
///
/// The fields are public so the observation and API crates can read and build states. The step
/// functions never panic whatever the field values: every grid index comes from the validated
/// [`GameConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    /// Countdown grid per seat, `body[seat][y * width + x]` (`State.body`, int16).
    pub body: [[i16; MAX_CELLS]; MAX_SNAKES],
    /// Head `(x, y)` per seat. Off the board for a snake that just hit a wall; dead seats keep the
    /// head they had (`State.head`, int32).
    pub head: [[i32; 2]; MAX_SNAKES],
    /// Body length (segments, stacked ones included) per seat (`State.length`).
    pub length: [i32; MAX_SNAKES],
    /// Health per seat (`State.health`).
    pub health: [i32; MAX_SNAKES],
    /// Whether each seat is still in the game (`State.alive`).
    pub alive: [bool; MAX_SNAKES],
    /// Last applied move per seat (action id 0 to 3), the fallback for invalid actions
    /// (`State.last_move`). Unused by the search.
    pub last_move: [u8; MAX_SNAKES],
    /// Food bitboard: bit `y * width + x` (`State.food`).
    pub food: u128,
    /// Hazard count per cell, stacked hazards counting once each (`State.hazard`, int16).
    pub hazard: [i16; MAX_CELLS],
    /// Engine turn number (`State.turn`).
    pub turn: i32,
    /// [`Cause`] code per seat, 0 while alive (`State.elim_cause`). Reporting only.
    pub elim_cause: [i8; MAX_SNAKES],
    /// Turn of elimination per seat, 0 while alive (`State.elim_turn`). Placement values use it.
    pub elim_turn: [i32; MAX_SNAKES],
    /// Game over (`State.done`).
    pub done: bool,
    /// Royale shrink sides, first `config.num_sides()` entries used: 0 left, 1 right, 2 bottom,
    /// 3 top, anything else ([`UNKNOWN_SIDE`](crate::UNKNOWN_SIDE)) moves no edge
    /// (`RoyaleState.sides`, int8). Ignored outside the royale ruleset.
    pub sides: [i8; MAX_SIDES],
}

impl GameState {
    /// The all-empty state of `types.empty_state` (types.py:173-190): no snakes alive, no food or
    /// hazards, turn 0, `last_move` UP, royale sides all 0 (the "blank" sides of
    /// `maps._blank_sides`).
    pub fn new(_config: &GameConfig) -> GameState {
        GameState {
            body: [[0; MAX_CELLS]; MAX_SNAKES],
            head: [[0; 2]; MAX_SNAKES],
            length: [0; MAX_SNAKES],
            health: [0; MAX_SNAKES],
            alive: [false; MAX_SNAKES],
            last_move: [Move::Up as u8; MAX_SNAKES],
            food: 0,
            hazard: [0; MAX_CELLS],
            turn: 0,
            elim_cause: [0; MAX_SNAKES],
            elim_turn: [0; MAX_SNAKES],
            done: false,
            sides: [0; MAX_SIDES],
        }
    }

    /// Fill `seat` from a head-first list of `(x, y)` segments, as `engine_json.state_from_engine`
    /// does for one engine snake (engine_json.py:108-120):
    ///
    /// * `length = segments.len()`, `head = segments[0]` (or `(0, 0)` when empty), `health` as given;
    /// * `elim = None` makes a living snake whose countdown grid is
    ///   `grid[y][x] = max(grid[y][x], n - i)` over its on-board segments (`body_to_countdown`);
    ///   `elim = Some((cause, turn))` makes an eliminated seat with an all-zero grid;
    /// * `last_move` is the neck-to-head direction, UP for a stacked or single segment (`_direction`).
    ///
    /// Call [`GameState::refresh_done`] once every seat is placed.
    pub fn set_snake(
        &mut self,
        config: &GameConfig,
        seat: usize,
        segments: &[(i32, i32)],
        health: i32,
        elim: Option<(Cause, i32)>,
    ) -> Result<(), RulesError> {
        if seat >= config.num_snakes() {
            return Err(RulesError::Seat(seat));
        }
        let n = segments.len();
        if n > i16::MAX as usize {
            return Err(RulesError::BodyTooLong(n));
        }
        let mut grid = [0i16; MAX_CELLS];
        if elim.is_none() {
            for (i, &(x, y)) in segments.iter().enumerate() {
                if let Some(c) = config.cell(x, y) {
                    grid[c] = grid[c].max((n - i) as i16);
                }
            }
        }
        self.body[seat] = grid;
        self.length[seat] = n as i32;
        self.health[seat] = health;
        self.head[seat] = segments.first().map_or([0, 0], |&(x, y)| [x, y]);
        self.last_move[seat] = match segments {
            [head, neck, ..] => default_move(*head, *neck) as u8,
            _ => Move::Up as u8,
        };
        let (cause, turn) = elim.unwrap_or((Cause::None, 0));
        self.alive[seat] = cause == Cause::None;
        self.elim_cause[seat] = cause as i8;
        self.elim_turn[seat] = turn;
        Ok(())
    }

    /// Put food on `(x, y)`. Errors if the cell is off the board or already has food (the
    /// engine-json loader refuses duplicate food, engine_json.py:124-128).
    pub fn add_food(&mut self, config: &GameConfig, x: i32, y: i32) -> Result<(), RulesError> {
        match config.cell(x, y) {
            Some(c) if self.food & (1u128 << c) == 0 => {
                self.food |= 1u128 << c;
                Ok(())
            }
            _ => Err(RulesError::Food { x, y }),
        }
    }

    /// Add one hazard layer on `(x, y)`; off-board hazards are dropped (engine_json.py:129-135).
    /// Returns whether it was on the board.
    pub fn add_hazard(&mut self, config: &GameConfig, x: i32, y: i32) -> bool {
        match config.cell(x, y) {
            Some(c) => {
                self.hazard[c] = self.hazard[c].saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// Set `done` to the engine's game-over condition, at most one snake alive
    /// (`rules.is_game_over`, rules.py:44-47), as the state builders do (engine_json.py:149).
    pub fn refresh_done(&mut self, config: &GameConfig) {
        self.done = self.is_game_over(config);
    }

    /// Whether at most one snake is alive (`rules.is_game_over` on `self.alive`; never solo).
    #[inline]
    pub fn is_game_over(&self, config: &GameConfig) -> bool {
        self.num_alive(config) <= 1
    }

    /// Number of living seats.
    #[inline]
    pub fn num_alive(&self, config: &GameConfig) -> usize {
        self.alive[..config.num_snakes()]
            .iter()
            .filter(|&&a| a)
            .count()
    }

    /// Whether the cell index `cell` has food.
    #[inline]
    pub fn has_food(&self, cell: usize) -> bool {
        cell < 128 && self.food >> cell & 1 == 1
    }

    /// Food at `(x, y)`; false off the board (`rules.value_at`, rules.py:56-64).
    #[inline]
    pub fn food_at(&self, config: &GameConfig, x: i32, y: i32) -> bool {
        config.cell(x, y).is_some_and(|c| self.has_food(c))
    }

    /// Hazard count at `(x, y)`; 0 off the board.
    #[inline]
    pub fn hazard_at(&self, config: &GameConfig, x: i32, y: i32) -> i16 {
        config.cell(x, y).map_or(0, |c| self.hazard[c])
    }

    /// Countdown of `seat` at `(x, y)`; 0 off the board or for a seat out of range.
    #[inline]
    pub fn body_at(&self, config: &GameConfig, seat: usize, x: i32, y: i32) -> i16 {
        match (config.cell(x, y), self.body.get(seat)) {
            (Some(c), Some(grid)) => grid[c],
            _ => 0,
        }
    }

    /// The cell index of `seat`'s head, or `None` when it is off the board.
    #[inline]
    pub fn head_cell(&self, config: &GameConfig, seat: usize) -> Option<usize> {
        let [x, y] = *self.head.get(seat)?;
        config.cell(x, y)
    }
}

/// The engine's `getDefaultMove`: the neck-to-head direction (`engine_json._direction`,
/// engine_json.py:73-92), UP when the two segments are stacked. The wrap-around branches are kept
/// for fidelity; on a non-wrapped board they only fire for malformed bodies.
pub fn default_move(head: (i32, i32), neck: (i32, i32)) -> Move {
    let ((hx, hy), (nx, ny)) = (head, neck);
    if hx == nx.wrapping_add(1) {
        Move::Right
    } else if hx == nx.wrapping_sub(1) {
        Move::Left
    } else if hy == ny.wrapping_add(1) {
        Move::Up
    } else if hy == ny.wrapping_sub(1) {
        Move::Down
    } else if hx == 0 && nx > 0 {
        Move::Right
    } else if nx == 0 && hx > 0 {
        Move::Left
    } else if hy == 0 && ny > 0 {
        Move::Up
    } else if ny == 0 && hy > 0 {
        Move::Down
    } else {
        Move::Up
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ruleset;

    #[test]
    fn body_to_countdown() {
        let c = GameConfig::new(11, 11, 2, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&c);
        // Turn-0 snake: three segments stacked.
        s.set_snake(&c, 0, &[(1, 1), (1, 1), (1, 1)], 100, None)
            .unwrap();
        assert_eq!(s.body_at(&c, 0, 1, 1), 3);
        assert_eq!(s.length[0], 3);
        assert_eq!(s.last_move[0], Move::Up as u8);
        // A snake that just ate: tail stacked, countdown 2; off-board segments skipped.
        s.set_snake(&c, 1, &[(5, 5), (4, 5), (4, 4), (4, 4)], 90, None)
            .unwrap();
        assert_eq!(s.body_at(&c, 1, 5, 5), 4);
        assert_eq!(s.body_at(&c, 1, 4, 5), 3);
        assert_eq!(s.body_at(&c, 1, 4, 4), 2);
        assert_eq!(s.last_move[1], Move::Right as u8);
        s.refresh_done(&c);
        assert!(!s.done);
        s.set_snake(&c, 1, &[], 0, Some((Cause::Collision, 0)))
            .unwrap();
        assert_eq!(s.body[1], [0; MAX_CELLS]);
        assert!(!s.alive[1]);
        s.refresh_done(&c);
        assert!(s.done);
        assert_eq!(s.set_snake(&c, 2, &[], 0, None), Err(RulesError::Seat(2)));
    }

    #[test]
    fn food_and_hazards() {
        let c = GameConfig::new(7, 5, 2, Ruleset::Standard).unwrap();
        let mut s = GameState::new(&c);
        s.add_food(&c, 6, 4).unwrap();
        assert!(s.food_at(&c, 6, 4));
        assert!(s.add_food(&c, 6, 4).is_err());
        assert!(s.add_food(&c, 7, 0).is_err());
        assert!(s.add_hazard(&c, 2, 2));
        assert!(s.add_hazard(&c, 2, 2));
        assert!(!s.add_hazard(&c, -1, 2));
        assert_eq!(s.hazard_at(&c, 2, 2), 2);
        assert_eq!(s.hazard_at(&c, 20, 2), 0);
    }
}
