//! Turn 0: snake placement, starting food and the royale shrink sides.
//!
//! Ported from `maps.py`: `placement_kind` (maps.py:139-151), `_place_fixed` (maps.py:76-86),
//! `_random_cells`/`_random_start_cells` (maps.py:101-133), `snakes_at` (maps.py:173-182),
//! `_fixed_food_candidates`/`place_food_fixed` (maps.py:222-322), `place_initial_food`
//! (maps.py:325-336), the maps' `setup` (maps.py:359-440) and `env.init_state` (env.py:104-112).
//! Choices are uniform as in slinky (a uniform permutation where slinky argsorts random scores,
//! a uniform pick where it takes the argmax of random scores over the allowed cells), drawn from
//! an [`EnvRandom`] instead of JAX's stream.
//!
//! The engine's "distributed" placement needs more than 8 snakes and never applies here (at
//! most 4 seats).

use slinky_rules::{GameConfig, GameState, MAX_SNAKES};

use crate::EnvConfig;
use crate::rng::{EnvRandom, pick_in, side_of, uniform};

/// Body length of a new snake (types.py:36).
pub const START_LENGTH: usize = 3;

/// Which of the engine's `PlaceSnakesAutomatically` strategies a board uses
/// (`maps.placement_kind`, with at most 4 snakes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Square boards of side 7 or more: the 4 corner and 4 cardinal start points.
    Fixed,
    /// Every other board: distinct random cells of even `x + y`, not the centre.
    Random,
}

impl Placement {
    /// The strategy of `config`'s board.
    pub fn of(config: &GameConfig) -> Placement {
        if config.width() == config.height() && config.width() >= 7 {
            Placement::Fixed
        } else {
            Placement::Random
        }
    }
}

/// Bitboard of the board's cells (`width * height` low bits).
#[inline]
pub(crate) fn board_mask(config: &GameConfig) -> u128 {
    (1u128 << config.cells()) - 1
}

/// Four `(x, y)` start points.
pub type StartPoints = [(i32, i32); 4];

/// The 4 corner and 4 cardinal start points of a square board (`_fixed_start_points`).
pub fn fixed_start_points(config: &GameConfig) -> (StartPoints, StartPoints) {
    let w = config.width() as i32;
    let (mn, md, mx) = (1, (w - 1) / 2, w - 2);
    (
        [(mn, mn), (mn, mx), (mx, mn), (mx, mx)],
        [(mn, md), (md, mn), (md, mx), (mx, md)],
    )
}

/// Cells random placement may use: even `x + y`, not the centre (`_random_start_cells`).
pub fn random_start_cells(config: &GameConfig) -> u128 {
    let (w, h) = (config.width() as i32, config.height() as i32);
    let mut mask = 0u128;
    for y in 0..h {
        for x in 0..w {
            if (x + y) % 2 == 0 && (x, y) != ((w - 1) / 2, (h - 1) / 2) {
                mask |= 1u128 << (y * w + x);
            }
        }
    }
    mask
}

fn shuffle<R: EnvRandom + ?Sized>(rng: &mut R, items: &mut [(i32, i32)]) {
    for i in (1..items.len()).rev() {
        let j = uniform(rng, i as u32 + 1) as usize;
        items.swap(i, j);
    }
}

/// Start points of seats `0..num_snakes` (`maps.place_snakes`). Entries past the seat count are
/// `(0, 0)`, and so are the seats random placement has no room for (a board
/// [`EnvConfig::new`] refuses, see [`EnvConfig::can_reset`]).
pub fn place_snakes<R: EnvRandom + ?Sized>(
    config: &GameConfig,
    rng: &mut R,
) -> [(i32, i32); MAX_SNAKES] {
    place_seats(config, rng).0
}

/// [`place_snakes`] and the number of seats placed (seats `0..placed`; fewer than the seat
/// count only when random placement runs out of room).
fn place_seats<R: EnvRandom + ?Sized>(
    config: &GameConfig,
    rng: &mut R,
) -> ([(i32, i32); MAX_SNAKES], usize) {
    let n = config.num_snakes();
    let mut out = [(0, 0); MAX_SNAKES];
    let mut placed = n;
    match Placement::of(config) {
        Placement::Fixed => {
            let (mut corners, mut cardinals) = fixed_start_points(config);
            shuffle(rng, &mut corners);
            shuffle(rng, &mut cardinals);
            // Corners first or cardinals first, 50/50; with at most 4 seats only the first
            // group is used.
            let first = if uniform(rng, 2) == 1 {
                corners
            } else {
                cardinals
            };
            out[..n].copy_from_slice(&first[..n]);
        }
        Placement::Random => {
            let mut free = random_start_cells(config);
            for (seat, slot) in out.iter_mut().enumerate().take(n) {
                // EnvConfig::new checked there is room for every seat; a position's config
                // (EnvConfig::for_position) may have less.
                let Some(c) = pick_in(rng, free) else {
                    placed = seat;
                    break;
                };
                free &= !(1u128 << c);
                *slot = config.xy(c);
            }
        }
    }
    (out, placed)
}

/// The diagonal neighbours of `head` allowed for starting food (`_fixed_food_candidates`): on
/// the board, on the far side of the head from the centre, not a corner, not the centre.
fn fixed_food_candidates(config: &GameConfig, head: (i32, i32)) -> [Option<usize>; 4] {
    let (w, h) = (config.width() as i32, config.height() as i32);
    let (cx, cy) = ((w - 1) / 2, (h - 1) / 2);
    let (hx, hy) = head;
    let mut out = [None; 4];
    for (slot, (dx, dy)) in out.iter_mut().zip([(-1, -1), (-1, 1), (1, -1), (1, 1)]) {
        let (px, py) = (hx + dx, hy + dy);
        let away = (px < hx && hx < cx)
            || (cx < hx && hx < px)
            || (py < hy && hy < cy)
            || (cy < hy && hy < py);
        let corner = (px == 0 || px == w - 1) && (py == 0 || py == h - 1);
        let centre = px == cx && py == cy;
        if away && !corner && !centre {
            *slot = config.cell(px, py);
        }
    }
    out
}

/// Spawn-allowed cells of `state` (`maps.spawn_mask`): not food, not a living snake's body and,
/// with `exclude_head_moves`, not orthogonally next to a living snake's head (no wrapping).
/// Hazards are allowed.
pub fn spawn_mask(config: &GameConfig, state: &GameState, exclude_head_moves: bool) -> u128 {
    let cells = config.cells();
    let mut occupied = state.food;
    for seat in 0..config.num_snakes() {
        if !state.alive[seat] {
            continue;
        }
        for (c, &v) in state.body[seat][..cells].iter().enumerate() {
            if v > 0 {
                occupied |= 1u128 << c;
            }
        }
        if exclude_head_moves {
            let [x, y] = state.head[seat];
            if config.in_bounds(x, y) {
                for (dx, dy) in [(0, 1), (0, -1), (-1, 0), (1, 0)] {
                    if let Some(c) = config.cell(x + dx, y + dy) {
                        occupied |= 1u128 << c;
                    }
                }
            }
        }
    }
    board_mask(config) & !occupied
}

/// Starting food (`place_initial_food`): on square boards of side 7 or more, one food diagonal
/// to each head (seat order, uniform among the allowed cells not already food) and one in the
/// centre (`place_food_fixed`); elsewhere one food per snake on random spawn-allowed cells.
fn place_initial_food<R: EnvRandom + ?Sized>(
    config: &GameConfig,
    state: &mut GameState,
    rng: &mut R,
) {
    let n = config.num_snakes();
    if config.width() == config.height() && config.width() >= 7 {
        for seat in 0..n {
            let [hx, hy] = state.head[seat];
            let mut allowed = [0usize; 4];
            let mut k = 0;
            for c in fixed_food_candidates(config, (hx, hy))
                .into_iter()
                .flatten()
            {
                if !state.has_food(c) {
                    allowed[k] = c;
                    k += 1;
                }
            }
            if k > 0 {
                let c = allowed[uniform(rng, k as u32) as usize];
                state.food |= 1u128 << c;
            }
        }
        let (w, h) = (config.width() as i32, config.height() as i32);
        if let Some(c) = config.cell((w - 1) / 2, (h - 1) / 2) {
            state.food |= 1u128 << c;
        }
    } else {
        let mut free = spawn_mask(config, state, true);
        for _ in 0..n.min(config.cells()) {
            match pick_in(rng, free) {
                Some(c) => {
                    state.food |= 1u128 << c;
                    free &= !(1u128 << c);
                }
                None => break,
            }
        }
    }
}

impl EnvConfig {
    /// A new game's turn-0 state (`BattlesnakeEnv.init_state`/`reset`): snakes placed with
    /// length 3 stacked on their start cell and full health, starting food unless the map is
    /// `empty` (removed again under constrictor, which the engine runs once at setup), random
    /// shrink sides under the royale ruleset, no hazards, `done` if at most one snake is alive.
    ///
    /// On a config from [`EnvConfig::for_position`] whose board has no room for every seat
    /// (see [`EnvConfig::can_reset`]), the seats that do not fit stay dead (slinky raises).
    pub fn reset<R: EnvRandom + ?Sized>(&self, rng: &mut R) -> GameState {
        let config = self.rules();
        let mut state = GameState::new(config);
        let (points, placed) = place_seats(config, rng);
        for (seat, &p) in points.iter().enumerate().take(placed) {
            // Cannot fail: the seat is in range and the body is 3 segments.
            let _ = state.set_snake(
                config,
                seat,
                &[p; START_LENGTH],
                slinky_rules::MAX_HEALTH,
                None,
            );
        }
        if self.spawns_food() {
            place_initial_food(config, &mut state, rng);
        }
        if config.ruleset() == slinky_rules::Ruleset::Constrictor {
            state.food = 0;
        }
        if self.has_sides() {
            for s in state.sides[..config.num_sides()].iter_mut() {
                *s = side_of(rng);
            }
        }
        state.refresh_done(config);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MapKind, Rng};
    use slinky_rules::{Mode, Ruleset};

    #[test]
    fn fixed_points_and_candidates_on_11x11() {
        let cfg = GameConfig::for_mode(Mode::Standard, None).unwrap();
        let (corners, cardinals) = fixed_start_points(&cfg);
        assert_eq!(corners, [(1, 1), (1, 9), (9, 1), (9, 9)]);
        assert_eq!(cardinals, [(1, 5), (5, 1), (5, 9), (9, 5)]);
        let cells = |h| -> Vec<(i32, i32)> {
            fixed_food_candidates(&cfg, h)
                .into_iter()
                .flatten()
                .map(|c| cfg.xy(c))
                .collect()
        };
        assert_eq!(cells((1, 1)), vec![(0, 2), (2, 0)]);
        assert_eq!(cells((1, 5)), vec![(0, 4), (0, 6)]);
        assert_eq!(cells((9, 9)), vec![(8, 10), (10, 8)]);
        assert_eq!(cells((5, 9)), vec![(4, 10), (6, 10)]);
    }

    #[test]
    fn reset_standard_duel() {
        let env = EnvConfig::for_mode(Mode::Duel, None).unwrap();
        let cfg = *env.rules();
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let s = env.reset(&mut rng);
            assert_eq!(s.turn, 0);
            assert!(!s.done);
            assert_eq!(s.food.count_ones(), 3);
            assert!(s.food_at(&cfg, 5, 5));
            for seat in 0..2 {
                let [x, y] = s.head[seat];
                assert_eq!(s.body_at(&cfg, seat, x, y), 3);
                assert_eq!((s.length[seat], s.health[seat]), (3, 100));
            }
            let corner = |[x, y]: [i32; 2]| x != 5 && y != 5;
            assert_eq!(corner(s.head[0]), corner(s.head[1]));
            assert_ne!(s.head[0], s.head[1]);
            assert_eq!(s.hazard, [0; slinky_rules::MAX_CELLS]);
            assert_eq!(s.sides, [0; slinky_rules::MAX_SIDES]);
        }
    }

    #[test]
    fn reset_without_room_leaves_the_extra_seats_dead() {
        // 2x2, 3 seats: random placement has one cell, (1, 1).
        let cfg = GameConfig::new(2, 2, 3, Ruleset::Standard).unwrap();
        let env = EnvConfig::for_position(cfg, MapKind::Standard).unwrap();
        assert!(!env.can_reset());
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            let s = env.reset(&mut rng);
            assert_eq!(s.alive[..3], [true, false, false]);
            assert_eq!(s.head[0], [1, 1]);
            assert!(s.done);
        }
    }

    #[test]
    fn reset_constrictor_royale_and_random_boards() {
        let mut rng = Rng::new(4);
        let c = EnvConfig::for_mode(Mode::Constrictor, None).unwrap();
        let s = c.reset(&mut rng);
        assert_eq!(s.food, 0);
        let e = EnvConfig::ladder(Mode::Constrictor).unwrap();
        assert_eq!(e.reset(&mut rng).food, 0);
        let r = EnvConfig::ladder(Mode::Royale).unwrap();
        let s = r.reset(&mut rng);
        assert!(s.sides[..21].iter().all(|&v| (0..4).contains(&v)));
        assert_eq!(s.food.count_ones(), 5);
        // A 9x11 board places randomly on even cells, then random food away from heads.
        let rect = GameConfig::new(9, 11, 3, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(rect, MapKind::Standard).unwrap();
        for _ in 0..200 {
            let s = env.reset(&mut rng);
            let heads: Vec<_> = (0..3).map(|i| s.head[i]).collect();
            for &[x, y] in &heads {
                assert_eq!((x + y) % 2, 0);
                assert_ne!((x, y), (4, 5));
            }
            assert!(heads[0] != heads[1] && heads[1] != heads[2] && heads[0] != heads[2]);
            assert_eq!(s.food.count_ones(), 3);
            for c in 0..rect.cells() {
                if s.has_food(c) {
                    let (fx, fy) = rect.xy(c);
                    for &[x, y] in &heads {
                        assert!((fx - x).abs() + (fy - y).abs() > 1);
                    }
                }
            }
        }
    }
}
