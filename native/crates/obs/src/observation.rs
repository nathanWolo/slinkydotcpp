//! The egocentric board observation of one seat: `f32[2H-1][2W-1][13]`, channels last.
//!
//! Ported from `observations.allocentric_obs` (src/slinky/observations.py:51-98) and
//! `observations.egocentric_obs` (observations.py:101-115), with the constrictor food hiding of
//! `modes.hide_food` (modes.py:161-173) as applied by `unified.policy_inputs` (unified.py:277-292).
//! See `native-az/SPEC.md` sections 3.1, 3.2 and 3.4.
//!
//! # Layout
//!
//! `out[(r * cols + c) * 13 + ch]` with `rows = 2H-1`, `cols = 2W-1` (21 x 21 on 11x11): row `r`
//! is the y offset `r - (H-1)` from the (clamped) head, increasing upwards, and column `c` the x
//! offset `c - (W-1)`, increasing rightwards. No flip, no rotation. Off-board cells are zero in
//! every channel, `on_board` included. This is the NHWC row the net crate's `forward` takes.
//!
//! Because the head is clamped onto the board and the window spans `H-1` cells on both sides of
//! it, the whole board always lies inside the window: the builder writes every board cell once
//! and zero-fills the margins, so each output float is written exactly once.
//!
//! # Floats
//!
//! XLA rewrites the Python's `x / 100.0` and `x / cells` into multiplications by the float32
//! reciprocal, so this module multiplies by [`RECIP_100`] and `1/cells` (an f32 reciprocal) to be
//! bit-exact with the jitted Python (SPEC 3.1, "Float detail").

use crate::error::ObsError;
use crate::{NUM_CHANNELS, ch};
use slinky_rules::{GameConfig, GameState, MAX_SNAKES};

/// `f32(1/100)`: XLA's rewrite of `health / 100.0` (observations.py:76, 78).
pub const RECIP_100: f32 = 1.0 / 100.0;

/// The observation shape `(rows, cols, channels)` = `(2H-1, 2W-1, 13)` of a (non-wrapped) game
/// (`observations.obs_shape`, observations.py:118-125).
pub fn obs_shape(config: &GameConfig) -> (usize, usize, usize) {
    (
        2 * config.height() - 1,
        2 * config.width() - 1,
        NUM_CHANNELS,
    )
}

/// Floats in one seat's observation: `(2H-1) * (2W-1) * 13`, 5,733 on 11x11.
pub fn obs_len(config: &GameConfig) -> usize {
    let (r, c, k) = obs_shape(config);
    r * c * k
}

/// `f32(1/cells)`, the factor of the countdown and length planes (observations.py:54, 79, 88,
/// 92). Equal to `f32(1.0/cells)` computed in f64 for every board size (tested).
#[inline]
pub(crate) fn recip_cells(config: &GameConfig) -> f32 {
    1.0 / config.cells() as f32
}

/// Write seat `seat`'s egocentric observation of `state` into `out` (exactly
/// [`obs_len`]`(config)` floats), with the food plane zeroed when `hide_food` is set.
///
/// `hide_food` should be `hide_constrictor_food && ruleset == Constrictor` (the *real* ruleset,
/// not the told mode; unified.py:291); [`crate::Encoder`] decides it for you. Dead seats get the
/// Python's well-defined but meaningless window: their head is clamped onto the board and every
/// own channel is zero. No allocation; errors only on a bad seat or buffer length.
pub fn egocentric_into(
    state: &GameState,
    config: &GameConfig,
    seat: usize,
    hide_food: bool,
    out: &mut [f32],
) -> Result<(), ObsError> {
    let (w, h, n) = (config.width(), config.height(), config.num_snakes());
    if seat >= n {
        return Err(ObsError::Seat {
            seat,
            num_snakes: n,
        });
    }
    let need = obs_len(config);
    if out.len() != need {
        return Err(ObsError::Buffer {
            what: "observation",
            got: out.len(),
            need,
        });
    }
    const C: usize = NUM_CHANNELS;
    let (rows, cols) = (2 * h - 1, 2 * w - 1);
    let rc = recip_cells(config);

    // The agent's own scalars (observations.py:60, 70, 78-79): zero when the seat is dead.
    let own_alive = state.alive[seat];
    let own_body = &state.body[seat];
    let own_health = if own_alive {
        state.health[seat] as f32 * RECIP_100
    } else {
        0.0
    };
    let own_length = if own_alive {
        state.length[seat] as f32 * rc
    } else {
        0.0
    };

    // Living opponents (observations.py:59): dead seats are ignored entirely.
    let mut opp_buf = [0usize; MAX_SNAKES];
    let mut k = 0;
    for j in 0..n {
        if j != seat && state.alive[j] {
            opp_buf[k] = j;
            k += 1;
        }
    }
    let opps = &opp_buf[..k];

    // The clamped head (observations.py:107-108); the board's corner (0, 0) lands at window
    // position (r0, c0), and the board fills rows r0..r0+h, columns c0..c0+w.
    let hx = state.head[seat][0].clamp(0, w as i32 - 1) as usize;
    let hy = state.head[seat][1].clamp(0, h as i32 - 1) as usize;
    let (r0, c0) = (h - 1 - hy, w - 1 - hx);
    let row_len = cols * C;

    for (r, line) in out.chunks_exact_mut(row_len).enumerate().take(rows) {
        if r < r0 || r >= r0 + h {
            line.fill(0.0);
            continue;
        }
        let y = r - r0;
        let (left, rest) = line.split_at_mut(c0 * C);
        let (board, right) = rest.split_at_mut(w * C);
        left.fill(0.0);
        right.fill(0.0);
        for (x, px) in board.as_chunks_mut::<C>().0.iter_mut().enumerate() {
            let cell = y * w + x;
            let food = if !hide_food && state.has_food(cell) {
                1.0
            } else {
                0.0
            };
            let ob = if own_alive { own_body[cell] } else { 0 };
            let mut opp_any = false;
            let mut opp_max: i16 = 0; // max over seats of where(opp, body, 0): 0 from non-opponents
            for &j in opps {
                let b = state.body[j][cell];
                opp_any |= b > 0;
                opp_max = opp_max.max(b);
            }
            *px = [
                1.0,                         // on_board
                food,                        // food (hidden in constrictor)
                state.hazard[cell] as f32,   // hazard: raw stacked count
                0.0,                         // own_head: set below
                f32::from(u8::from(ob > 0)), // own_body
                ob as f32 * rc,              // own_countdown
                0.0,                         // opp_head_ge: set below
                0.0,                         // opp_head_lt: set below
                f32::from(u8::from(opp_any)),
                opp_max as f32 * rc, // opp_countdown
                0.0,                 // opp_head_health: set below
                own_health,
                own_length,
            ];
        }
    }

    // Head pixels (observations.py:66, 71, 74-76): only for heads on the board.
    let at = |cell: usize| {
        let (x, y) = (cell % w, cell / w);
        ((y + r0) * cols + x + c0) * C
    };
    if own_alive && let Some(cell) = state.head_cell(config, seat) {
        out[at(cell) + ch::OWN_HEAD] = 1.0;
    }
    let own_len = state.length[seat];
    for &j in opps {
        if let Some(cell) = state.head_cell(config, j) {
            let i = at(cell);
            let plane = if state.length[j] >= own_len {
                ch::OPP_HEAD_GE
            } else {
                ch::OPP_HEAD_LT
            };
            out[i + plane] = 1.0;
            // where(opp_head, health / 100, 0).max(seats): the 0.0 of the other seats is a floor.
            let hp = state.health[j] as f32 * RECIP_100;
            let slot = &mut out[i + ch::OPP_HEAD_HEALTH];
            *slot = slot.max(hp);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use slinky_rules::{Cause, Mode, Ruleset};

    fn duel() -> (GameConfig, GameState) {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(1, 1), (1, 2), (1, 3)], 90, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(9, 9), (9, 8), (9, 7), (9, 7)], 55, None)
            .unwrap();
        s.add_food(&cfg, 5, 5).unwrap();
        s.add_hazard(&cfg, 0, 0);
        s.add_hazard(&cfg, 0, 0);
        s.refresh_done(&cfg);
        (cfg, s)
    }

    fn get(o: &[f32], cfg: &GameConfig, r: usize, c: usize, k: usize) -> f32 {
        o[(r * (2 * cfg.width() - 1) + c) * NUM_CHANNELS + k]
    }

    #[test]
    fn reciprocal_matches_f64_rounding() {
        for c in 1..=10_000u32 {
            assert_eq!(1.0f32 / c as f32, (1.0f64 / c as f64) as f32, "1/{c}");
        }
        assert_eq!(RECIP_100, (1.0f64 / 100.0) as f32);
    }

    #[test]
    fn head_centred_window() {
        let (cfg, s) = duel();
        let mut o = vec![f32::NAN; obs_len(&cfg)];
        egocentric_into(&s, &cfg, 0, false, &mut o).unwrap();
        assert!(o.iter().all(|v| v.is_finite()));
        // Own head at the centre, own body above it (y up = row up).
        assert_eq!(get(&o, &cfg, 10, 10, ch::OWN_HEAD), 1.0);
        // A multiplication by the reciprocal, which differs from 3/121 in the last bit.
        assert_eq!(
            get(&o, &cfg, 10, 10, ch::OWN_COUNTDOWN),
            3.0 * (1.0 / 121.0f32)
        );
        assert_ne!(get(&o, &cfg, 10, 10, ch::OWN_COUNTDOWN), 3.0 / 121.0f32);
        assert_eq!(get(&o, &cfg, 12, 10, ch::OWN_BODY), 1.0);
        // Head (1,1): cell (0,0) is at offset (-1,-1); the stacked hazard counts 2.
        assert_eq!(get(&o, &cfg, 9, 9, ch::HAZARD), 2.0);
        assert_eq!(get(&o, &cfg, 8, 8, ch::ON_BOARD), 0.0);
        assert_eq!(get(&o, &cfg, 9, 9, ch::ON_BOARD), 1.0);
        // Food at (5,5) = offset (+4,+4).
        assert_eq!(get(&o, &cfg, 14, 14, ch::FOOD), 1.0);
        // Opponent: head (9,9), longer, health 55; stacked tail countdown 2 at (9,7).
        assert_eq!(get(&o, &cfg, 18, 18, ch::OPP_HEAD_GE), 1.0);
        assert_eq!(get(&o, &cfg, 18, 18, ch::OPP_HEAD_LT), 0.0);
        assert_eq!(get(&o, &cfg, 18, 18, ch::OPP_HEAD_HEALTH), 55.0 * 0.01f32);
        assert_eq!(
            get(&o, &cfg, 16, 18, ch::OPP_COUNTDOWN),
            2.0 * (1.0 / 121.0f32)
        );
        assert_eq!(get(&o, &cfg, 0, 0, ch::OWN_HEALTH), 0.0);
        assert_eq!(get(&o, &cfg, 9, 9, ch::OWN_HEALTH), 0.9f32);
        assert_eq!(get(&o, &cfg, 9, 9, ch::OWN_LENGTH), 3.0 * (1.0 / 121.0f32));

        egocentric_into(&s, &cfg, 0, true, &mut o).unwrap();
        assert_eq!(get(&o, &cfg, 14, 14, ch::FOOD), 0.0);
    }

    #[test]
    fn dead_seat_is_clamped_and_blank() {
        let (cfg, mut s) = duel();
        s.set_snake(&cfg, 1, &[], 0, Some((Cause::OutOfBounds, 3)))
            .unwrap();
        s.head[1] = [-4, 30];
        let mut o = vec![f32::NAN; obs_len(&cfg)];
        egocentric_into(&s, &cfg, 1, false, &mut o).unwrap();
        // Clamped head (0, 10): the board fills rows 0..11, columns 10..21.
        assert_eq!(get(&o, &cfg, 0, 10, ch::ON_BOARD), 1.0);
        assert_eq!(get(&o, &cfg, 11, 10, ch::ON_BOARD), 0.0);
        assert_eq!(get(&o, &cfg, 0, 9, ch::ON_BOARD), 0.0);
        for k in [ch::OWN_HEAD, ch::OWN_BODY, ch::OWN_HEALTH, ch::OWN_LENGTH] {
            assert!((0..o.len() / NUM_CHANNELS).all(|p| o[p * NUM_CHANNELS + k] == 0.0));
        }
        // Seat 0 sees no opponent at all.
        egocentric_into(&s, &cfg, 0, false, &mut o).unwrap();
        for k in [ch::OPP_HEAD_GE, ch::OPP_BODY, ch::OPP_HEAD_HEALTH] {
            assert!((0..o.len() / NUM_CHANNELS).all(|p| o[p * NUM_CHANNELS + k] == 0.0));
        }
    }

    #[test]
    fn bad_input_is_an_error() {
        let (cfg, s) = duel();
        let mut o = vec![0.0; obs_len(&cfg)];
        assert!(matches!(
            egocentric_into(&s, &cfg, 2, false, &mut o),
            Err(ObsError::Seat { .. })
        ));
        assert!(matches!(
            egocentric_into(&s, &cfg, 0, false, &mut o[1..]),
            Err(ObsError::Buffer { .. })
        ));
    }

    #[test]
    fn small_boards() {
        let cfg = GameConfig::new(1, 3, 2, Ruleset::Standard).unwrap();
        assert_eq!(obs_shape(&cfg), (5, 1, 13));
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(0, 0)], 1, None).unwrap();
        s.set_snake(&cfg, 1, &[(0, 2)], 1, None).unwrap();
        let mut o = vec![0.0; obs_len(&cfg)];
        egocentric_into(&s, &cfg, 1, false, &mut o).unwrap();
        assert_eq!(get(&o, &cfg, 0, 0, ch::OPP_HEAD_LT), 0.0);
        assert_eq!(get(&o, &cfg, 0, 0, ch::OPP_HEAD_GE), 1.0);
        assert_eq!(get(&o, &cfg, 2, 0, ch::OWN_HEAD), 1.0);
        assert_eq!(get(&o, &cfg, 3, 0, ch::ON_BOARD), 0.0);
    }
}
