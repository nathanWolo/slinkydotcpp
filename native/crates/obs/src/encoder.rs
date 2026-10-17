//! [`Encoder`]: the network inputs of one search node, every seat at once.
//!
//! Ported from `az_modes.make_evaluator` (baselines/az_modes.py:209-223) up to the network call:
//! `unified.policy_inputs(env.observe(state), state, game, cfg, mode)` (unified.py:277-292), i.e.
//! one egocentric window per seat (food hidden in constrictor) and the feature rows. See
//! `native-az/SPEC.md` section 3.6.

use crate::error::ObsError;
use crate::features::mode_features;
use crate::observation::{egocentric_into, obs_len};
use crate::{NUM_FEATURES, ObsShape};
use slinky_rules::{GameConfig, GameState, Mode, Ruleset};

/// Builds the network inputs of states of one game. Cheap to construct and `Copy`; holds the
/// config, the mode the network is told and whether food is hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Encoder {
    config: GameConfig,
    told: Mode,
    hide_food: bool,
}

impl Encoder {
    /// An encoder for `config`, telling the network the config's own mode (`modes.mode_of`, as
    /// the server does: agents.py:724-731). `hide_constrictor_food` is the run's config.json
    /// field; food is hidden when it is set and the *real* ruleset is constrictor
    /// (unified.py:291, modes.py:161-173).
    pub fn new(config: GameConfig, hide_constrictor_food: bool) -> Encoder {
        Encoder {
            config,
            told: config.mode(),
            hide_food: hide_constrictor_food && config.ruleset() == Ruleset::Constrictor,
        }
    }

    /// Tell the network another mode (`game_mode=` override). Food hiding still follows the
    /// real ruleset.
    pub fn with_told_mode(mut self, told: Mode) -> Encoder {
        self.told = told;
        self
    }

    /// The game config.
    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    /// The mode the network is told.
    pub fn told_mode(&self) -> Mode {
        self.told
    }

    /// Whether the food plane is zeroed.
    pub fn hides_food(&self) -> bool {
        self.hide_food
    }

    /// The `(rows, cols, channels)` of one seat's observation.
    pub fn obs_shape(&self) -> ObsShape {
        crate::obs_shape(&self.config)
    }

    /// Floats in one seat's observation ([`crate::OBS_LEN`] on 11x11).
    pub fn obs_len(&self) -> usize {
        obs_len(&self.config)
    }

    /// Seat `seat`'s board into `out` (exactly [`Encoder::obs_len`] floats).
    pub fn observe_into(
        &self,
        state: &GameState,
        seat: usize,
        out: &mut [f32],
    ) -> Result<(), ObsError> {
        egocentric_into(state, &self.config, seat, self.hide_food, out)
    }

    /// The feature rows of every seat (rows beyond `num_snakes` are zero).
    pub fn features(&self, state: &GameState) -> [[f32; NUM_FEATURES]; slinky_rules::MAX_SNAKES] {
        mode_features(state, &self.config, self.told)
    }

    /// The whole batch of a node: `boards` gets `N` observations back to back (`N *
    /// obs_len` floats, the net's `[B][21][21][13]`) and `feats` gets `N * 7` floats
    /// (`[B][7]`), with `N = num_snakes`, seat `i` in row `i`. Returns `N`. No allocation.
    pub fn encode_into(
        &self,
        state: &GameState,
        boards: &mut [f32],
        feats: &mut [f32],
    ) -> Result<usize, ObsError> {
        let n = self.config.num_snakes();
        let len = self.obs_len();
        if boards.len() != n * len {
            return Err(ObsError::Buffer {
                what: "boards",
                got: boards.len(),
                need: n * len,
            });
        }
        if feats.len() != n * NUM_FEATURES {
            return Err(ObsError::Buffer {
                what: "features",
                got: feats.len(),
                need: n * NUM_FEATURES,
            });
        }
        for (seat, out) in boards.chunks_exact_mut(len).enumerate() {
            self.observe_into(state, seat, out)?;
        }
        let rows = self.features(state);
        for (out, row) in feats
            .as_chunks_mut::<NUM_FEATURES>()
            .0
            .iter_mut()
            .zip(&rows)
        {
            *out = *row;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OBS_LEN, ch};

    #[test]
    fn hides_food_by_real_ruleset() {
        let con = GameConfig::for_mode(Mode::Constrictor, None).unwrap();
        assert!(Encoder::new(con, true).hides_food());
        assert!(!Encoder::new(con, false).hides_food());
        let std4 = GameConfig::for_mode(Mode::Standard, None).unwrap();
        let e = Encoder::new(std4, true).with_told_mode(Mode::Constrictor);
        assert!(!e.hides_food());
        assert_eq!(e.told_mode(), Mode::Constrictor);
    }

    #[test]
    fn batch_layout() {
        let cfg = GameConfig::for_mode(Mode::Constrictor, Some(2)).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(2, 2), (2, 2), (2, 2)], 100, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(8, 8), (8, 8), (8, 8)], 100, None)
            .unwrap();
        s.add_food(&cfg, 2, 3).unwrap();
        let e = Encoder::new(cfg, true);
        assert_eq!(e.obs_len(), OBS_LEN);
        let mut boards = vec![f32::NAN; 2 * OBS_LEN];
        let mut feats = vec![f32::NAN; 2 * NUM_FEATURES];
        assert_eq!(e.encode_into(&s, &mut boards, &mut feats), Ok(2));
        assert!(boards.iter().chain(&feats).all(|v| v.is_finite()));
        let centre = (10 * 21 + 10) * 13;
        assert_eq!(boards[centre + ch::OWN_HEAD], 1.0);
        assert_eq!(boards[OBS_LEN + centre + ch::OWN_HEAD], 1.0);
        assert!(
            boards
                .as_chunks::<13>()
                .0
                .iter()
                .all(|p| p[ch::FOOD] == 0.0)
        );
        assert_eq!(&feats[..7], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert!(
            e.encode_into(&s, &mut boards[1..], &mut feats).is_err()
                && e.encode_into(&s, &mut boards, &mut feats[1..]).is_err()
        );
    }
}
