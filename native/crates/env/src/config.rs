//! The environment's configuration: a rules [`GameConfig`] plus the map and its food settings,
//! and the error type.
//!
//! Ported from `types.GameConfig` (types.py:82-115: `map`, `food_spawn_chance`, `minimum_food`,
//! `max_turns`, the fields the rules crate leaves out), `modes.mode_config` (modes.py:72-92) and
//! `maps.get_map` (maps.py:441-460).

use std::fmt;

use slinky_api::ApiError;
use slinky_rules::{GameConfig, MAX_SNAKES, Mode, RulesError, Ruleset};

use crate::setup::{Placement, random_start_cells};

/// The map: where snakes start, where food appears, and (royale) the shrinking zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapKind {
    /// `standard`: engine placement, starting food, food spawns every turn.
    Standard,
    /// `royale`: the standard map plus the royale zone. Requires the royale ruleset (the royale
    /// map under the standard ruleset is the same game, and the server reads it as royale).
    Royale,
    /// `empty`: engine placement, no food ever (the official constrictor ladder's map).
    Empty,
}

impl MapKind {
    /// The API map name.
    pub fn name(self) -> &'static str {
        match self {
            MapKind::Standard => "standard",
            MapKind::Royale => "royale",
            MapKind::Empty => "empty",
        }
    }

    /// Parse an API map name (`standard`, `royale`, `empty`).
    pub fn from_name(name: &str) -> Result<MapKind, EnvError> {
        match name {
            "standard" => Ok(MapKind::Standard),
            "royale" => Ok(MapKind::Royale),
            "empty" => Ok(MapKind::Empty),
            other => Err(EnvError::Config(format!("unknown map {other:?}"))),
        }
    }
}

/// Why an environment could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvError {
    /// The rules crate refused the game (board size, seat count, ...).
    Rules(RulesError),
    /// The request could not be converted (see [`ApiError`]).
    Api(ApiError),
    /// A map and ruleset combination, or settings, this environment does not simulate.
    Config(String),
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::Rules(e) => write!(f, "rules: {e}"),
            EnvError::Api(e) => write!(f, "api: {e}"),
            EnvError::Config(m) => write!(f, "env config: {m}"),
        }
    }
}

impl std::error::Error for EnvError {}

impl From<RulesError> for EnvError {
    fn from(e: RulesError) -> Self {
        EnvError::Rules(e)
    }
}

impl From<ApiError> for EnvError {
    fn from(e: ApiError) -> Self {
        EnvError::Api(e)
    }
}

/// A whole game's static configuration (`types.GameConfig` with its map): the rules config
/// (board, seats, ruleset, hazard damage, shrink period, `max_turns`), the map, and the standard
/// map's food settings. Always valid: built only through [`EnvConfig::new`] and the mode helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EnvConfig {
    rules: GameConfig,
    map: MapKind,
    food_spawn_chance: i32,
    minimum_food: i32,
}

impl EnvConfig {
    /// A game of `rules` on `map` with the default food settings (`food_spawn_chance = 15`,
    /// `minimum_food = 1`, types.py:97-98).
    ///
    /// Errors on the royale map without the royale ruleset (normalise it as the server does), and
    /// on a board where the engine's random placement has no room for every snake.
    pub fn new(rules: GameConfig, map: MapKind) -> Result<EnvConfig, EnvError> {
        if map == MapKind::Royale && rules.ruleset() != Ruleset::Royale {
            return Err(EnvError::Config(format!(
                "the royale map is played with the royale ruleset, not {:?}",
                rules.ruleset().name()
            )));
        }
        if Placement::of(&rules) == Placement::Random
            && (random_start_cells(&rules).count_ones() as usize) < rules.num_snakes()
        {
            return Err(EnvError::Config("not enough room to place snakes".into()));
        }
        debug_assert!(rules.num_snakes() <= MAX_SNAKES);
        Ok(EnvConfig {
            rules,
            map,
            food_spawn_chance: 15,
            minimum_food: 1,
        })
    }

    /// slinky's training config of a mode (`modes.mode_config`): 11x11, `num_snakes` seats
    /// (`None`: 2 for duel, else 4), royale on the royale map, everything else (constrictor
    /// included) on the standard map, where constrictor's food is spawned but inert.
    pub fn for_mode(mode: Mode, num_snakes: Option<usize>) -> Result<EnvConfig, EnvError> {
        let rules = GameConfig::for_mode(mode, num_snakes)?;
        let map = if mode == Mode::Royale {
            MapKind::Royale
        } else {
            MapKind::Standard
        };
        EnvConfig::new(rules, map)
    }

    /// The official ladder's game of a mode: as [`EnvConfig::for_mode`] with the default seat
    /// count, except constrictor, which the ladder plays on the `empty` map (no food at all).
    pub fn ladder(mode: Mode) -> Result<EnvConfig, EnvError> {
        let rules = GameConfig::for_mode(mode, None)?;
        let map = match mode {
            Mode::Royale => MapKind::Royale,
            Mode::Constrictor => MapKind::Empty,
            Mode::Standard | Mode::Duel => MapKind::Standard,
        };
        EnvConfig::new(rules, map)
    }

    /// Set the food settings: the spawn chance in percent (the engine spawns with probability
    /// `(chance - 1) / 100`; 0 or less never) and the minimum food kept on the board.
    pub fn with_food(mut self, food_spawn_chance: i32, minimum_food: i32) -> EnvConfig {
        self.food_spawn_chance = food_spawn_chance;
        self.minimum_food = minimum_food;
        self
    }

    /// Set the training-only truncation `max_turns` (`None`: play until the rules end the game).
    pub fn with_max_turns(mut self, max_turns: Option<i32>) -> EnvConfig {
        self.rules = self.rules.with_max_turns(max_turns);
        self
    }

    /// The rules config (what `slinky-rules`, `slinky-obs` and the search take).
    pub fn rules(&self) -> &GameConfig {
        &self.rules
    }

    /// The map.
    pub fn map(&self) -> MapKind {
        self.map
    }

    /// `food_spawn_chance` in percent.
    pub fn food_spawn_chance(&self) -> i32 {
        self.food_spawn_chance
    }

    /// `minimum_food`.
    pub fn minimum_food(&self) -> i32 {
        self.minimum_food
    }

    /// The leaderboard mode (`rules().mode()`).
    pub fn mode(&self) -> Mode {
        self.rules.mode()
    }

    /// Whether the map places starting food and spawns food each turn (not the `empty` map).
    pub fn spawns_food(&self) -> bool {
        self.map != MapKind::Empty
    }

    /// Whether the game draws royale shrink sides at reset (the royale ruleset, any map).
    pub fn has_sides(&self) -> bool {
        self.rules.ruleset() == Ruleset::Royale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_and_ladder() {
        let duel = EnvConfig::for_mode(Mode::Duel, None).unwrap();
        assert_eq!(duel.rules().num_snakes(), 2);
        assert_eq!(duel.map(), MapKind::Standard);
        assert_eq!((duel.food_spawn_chance(), duel.minimum_food()), (15, 1));
        let royale = EnvConfig::ladder(Mode::Royale).unwrap();
        assert_eq!(royale.map(), MapKind::Royale);
        assert!(royale.has_sides());
        let c = EnvConfig::ladder(Mode::Constrictor).unwrap();
        assert_eq!(c.map(), MapKind::Empty);
        assert!(!c.spawns_food());
        assert_eq!(c.rules().num_snakes(), 4);
        let ct = EnvConfig::for_mode(Mode::Constrictor, Some(2)).unwrap();
        assert_eq!(ct.map(), MapKind::Standard);
        assert_eq!(ct.mode(), Mode::Constrictor);
        assert!(EnvConfig::for_mode(Mode::Duel, Some(3)).is_err());
    }

    #[test]
    fn refuses_bad_combinations() {
        let std4 = GameConfig::new(11, 11, 4, Ruleset::Standard).unwrap();
        assert!(EnvConfig::new(std4, MapKind::Royale).is_err());
        // Random placement needs 4 even-parity cells other than the centre: 3x3 has 4.
        let tiny = GameConfig::new(3, 3, 4, Ruleset::Standard).unwrap();
        assert!(EnvConfig::new(tiny, MapKind::Standard).is_ok());
        let tinier = GameConfig::new(2, 2, 3, Ruleset::Standard).unwrap();
        assert!(EnvConfig::new(tinier, MapKind::Standard).is_err());
        assert_eq!(MapKind::from_name("empty").unwrap(), MapKind::Empty);
        assert!(MapKind::from_name("arcade_maze").is_err());
    }
}
