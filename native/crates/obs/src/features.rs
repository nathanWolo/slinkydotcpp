//! The side features `f32[N][7]` of one state: the told mode's one-hot, the turn, the royale
//! shrink phase and the fraction of opponents alive.
//!
//! Ported from `modes.mode_features` (src/slinky/modes.py:124-158), with the reciprocal of
//! `modes._reciprocal` (modes.py:63-64), as called by `unified.policy_features`
//! (unified.py:260-274) with `features = "all"`. The `"no_mode"` and `"none"` variants are applied
//! by the network (the net crate masks or drops the features itself), so the raw vector is what
//! goes in. See `native-az/SPEC.md` section 3.3.

use crate::NUM_FEATURES;
use slinky_rules::{GameConfig, GameState, MAX_SNAKES, Mode};

/// The turn feature is `min(turn, TURN_SCALE) / TURN_SCALE` (modes.py:57).
pub const TURN_SCALE: i32 = 1000;

/// Feature column of each name, in `modes.FEATURE_NAMES` order (modes.py:46-54).
pub mod col {
    /// One-hot: told mode standard (columns 0-3 are indexed by `Mode as usize`).
    pub const IS_STANDARD: usize = 0;
    /// One-hot: told mode duel.
    pub const IS_DUEL: usize = 1;
    /// One-hot: told mode royale.
    pub const IS_ROYALE: usize = 2;
    /// One-hot: told mode constrictor.
    pub const IS_CONSTRICTOR: usize = 3;
    /// `min(turn, 1000) * f32(1/1000)`.
    pub const TURN: usize = 4;
    /// Royale only: `(turn mod S) * f32(1/S)`, S = `shrink_every_n_turns`.
    pub const SHRINK_PHASE: usize = 5;
    /// `(alive count - own alive) * f32(1/max(N-1, 1))`.
    pub const OPPONENTS_ALIVE: usize = 6;
}

/// `modes._reciprocal`: `np.float32(1.0 / x)`, the f64 reciprocal rounded to f32.
#[inline]
fn reciprocal(x: i32) -> f32 {
    (1.0f64 / f64::from(x)) as f32
}

/// The feature rows of `state`, as the network is told `told` (normally `config.mode()`).
/// Rows `>= config.num_snakes()` are zero.
///
/// Every row shares the one-hot, turn and phase; `opponents_alive` uses the row's own alive flag,
/// and `N` counts every seat, dead padded ones included (SPEC 3.3). The phase follows the told
/// mode (royale) but the period comes from the config, as in the Python.
pub fn mode_features(
    state: &GameState,
    config: &GameConfig,
    told: Mode,
) -> [[f32; NUM_FEATURES]; MAX_SNAKES] {
    let n = config.num_snakes();
    // min(turn, 1000).astype(f32) * f32(1/1000) (modes.py:147).
    let turn = state.turn.min(TURN_SCALE) as f32 * reciprocal(TURN_SCALE);
    // (turn % S).astype(f32) * f32(1/S): jnp's % is a floor modulo, rem_euclid for S >= 1.
    let phase = if told == Mode::Royale {
        let period = config.shrink_every_n_turns().max(1);
        state.turn.rem_euclid(period) as f32 * reciprocal(period)
    } else {
        0.0
    };
    let alive = |i: usize| f32::from(u8::from(state.alive[i]));
    // jnp.sum over the f32 alive flags (small integers, exact in any order).
    let total: f32 = (0..n).map(alive).sum();
    let per_opp = reciprocal((n as i32 - 1).max(1));

    let mut out = [[0.0f32; NUM_FEATURES]; MAX_SNAKES];
    for (i, row) in out.iter_mut().enumerate().take(n) {
        row[told as usize] = 1.0;
        row[col::TURN] = turn;
        row[col::SHRINK_PHASE] = phase;
        row[col::OPPONENTS_ALIVE] = (total - alive(i)) * per_opp;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use slinky_rules::Cause;

    fn royale4() -> (GameConfig, GameState) {
        let cfg = GameConfig::for_mode(Mode::Royale, None).unwrap();
        let mut s = GameState::new(&cfg);
        for (i, x) in [1, 3, 5, 7].into_iter().enumerate() {
            s.set_snake(&cfg, i, &[(x, 1), (x, 1), (x, 1)], 100, None)
                .unwrap();
        }
        s.set_snake(&cfg, 3, &[], 0, Some((Cause::Collision, 0)))
            .unwrap();
        (cfg, s)
    }

    #[test]
    fn values() {
        let (cfg, mut s) = royale4();
        s.turn = 37;
        let f = mode_features(&s, &cfg, Mode::Royale);
        assert_eq!(&f[0][..4], &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(f[0][col::TURN], 37.0 * (1.0f64 / 1000.0) as f32);
        assert_eq!(f[1][col::SHRINK_PHASE], 12.0 * (1.0f64 / 25.0) as f32);
        assert_eq!(f[0][col::OPPONENTS_ALIVE], 2.0 * (1.0f64 / 3.0) as f32);
        assert_eq!(f[3][col::OPPONENTS_ALIVE], 1.0);

        // Told another mode: no phase. Turns past the scale saturate; negative turns floor-mod.
        s.turn = 4321;
        let f = mode_features(&s, &cfg, Mode::Standard);
        assert_eq!(f[2][col::IS_STANDARD], 1.0);
        assert_eq!(f[2][col::TURN], 1.0);
        assert_eq!(f[2][col::SHRINK_PHASE], 0.0);
        s.turn = -3;
        let f = mode_features(&s, &cfg, Mode::Royale);
        assert_eq!(f[0][col::SHRINK_PHASE], 22.0 * (1.0f64 / 25.0) as f32);
        assert_eq!(f[0][col::TURN], -3.0 * (1.0f64 / 1000.0) as f32);
    }

    #[test]
    fn unused_rows_are_zero() {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let s = GameState::new(&cfg);
        let f = mode_features(&s, &cfg, Mode::Duel);
        assert_eq!(f[0][col::IS_DUEL], 1.0);
        assert_eq!(f[2], [0.0; NUM_FEATURES]);
        assert_eq!(f[3], [0.0; NUM_FEATURES]);
    }
}
