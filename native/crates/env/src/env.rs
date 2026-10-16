//! One turn of a full game: the rules step, the map's update (food spawns, the royale zone), the
//! turn increment, game end, `max_turns` truncation and rewards.
//!
//! Ported from `env.BattlesnakeEnv.step` / `step_autoreset` (env.py:121-169), `win_loss_reward`
//! and `placement_reward` (env.py:32-76), `maps.spawn_food_standard` (maps.py:339-356) and the
//! maps' `post_update` (maps.py:375-440). The deterministic part is [`GameState::rules_step`]
//! from `slinky-rules`.

use slinky_rules::{Actions, GameState, MAX_SNAKES, royale_hazards};

use crate::rng::{EnvRandom, Rng};
use crate::setup::spawn_mask;
use crate::{EnvConfig, MapKind};

/// What one [`EnvConfig::step`] did (`TimeStep` without the observations).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepInfo {
    /// The game is over after this step (or was already): at most one snake alive, or truncated.
    pub done: bool,
    /// The step ended the game by `max_turns` rather than by the rules.
    pub truncated: bool,
    /// Food spawned by the map this step.
    pub spawned: u32,
    /// `win_loss_reward` of the transition (0 when the game was already over).
    pub win_loss: [f32; MAX_SNAKES],
    /// `placement_reward` of the transition (0 when the game was already over).
    pub placement: [f32; MAX_SNAKES],
}

/// Sparse win/loss reward (`env.win_loss_reward`): -1 to a snake on the turn it is eliminated,
/// +1 to the last snake standing when the game ends, 0 to everyone eliminated on the last turn if
/// nobody survives (a draw). Seats past the seat count are 0.
pub fn win_loss_reward(
    config: &EnvConfig,
    prev: &GameState,
    next: &GameState,
) -> [f32; MAX_SNAKES] {
    rewards(config, &prev.alive, prev.done, next).0
}

/// Zero-sum finishing-order reward (`env.placement_reward`): each living snake gains
/// `1 / (N - 1)` per snake eliminated this turn, each eliminated snake loses `1 / (N - 1)` per
/// snake still alive. Computed as XLA does, `(gain - loss)` times the f32 reciprocal of `N - 1`.
pub fn placement_reward(
    config: &EnvConfig,
    prev: &GameState,
    next: &GameState,
) -> [f32; MAX_SNAKES] {
    rewards(config, &prev.alive, prev.done, next).1
}

/// `(win_loss, placement)` rewards of a transition from a state with `prev_alive` and
/// `prev_done` to `next` (the rewards themselves, not masked for an already finished game).
fn rewards(
    config: &EnvConfig,
    prev_alive: &[bool; MAX_SNAKES],
    prev_done: bool,
    next: &GameState,
) -> ([f32; MAX_SNAKES], [f32; MAX_SNAKES]) {
    let rules = config.rules();
    let n = rules.num_snakes();
    let alive = next.num_alive(rules) as i32;
    let over = next.is_game_over(rules) && !prev_done;
    let draw = over && alive == 0;
    let newly_dead = (0..n).filter(|&i| prev_alive[i] && !next.alive[i]).count() as i32;
    let recip = 1.0f32 / (n - 1) as f32;
    let mut win_loss = [0.0; MAX_SNAKES];
    let mut placement = [0.0; MAX_SNAKES];
    for i in 0..n {
        let died = prev_alive[i] && !next.alive[i];
        win_loss[i] = if over && next.alive[i] {
            1.0
        } else if died && !draw {
            -1.0
        } else {
            0.0
        };
        let gain = if next.alive[i] { newly_dead } else { 0 };
        let loss = if died { alive } else { 0 };
        placement[i] = (gain - loss) as f32 * recip;
    }
    (win_loss, placement)
}

impl EnvConfig {
    /// Cells where the map may spawn food in `state` (`maps.spawn_mask`): free of food and
    /// living bodies, and not next to a living head.
    pub fn spawn_mask(&self, state: &GameState) -> u128 {
        spawn_mask(self.rules(), state, true)
    }

    /// The standard map's per-turn spawn (`maps.spawn_food_standard`) on a post-rules state:
    /// tops the food up to `minimum_food`, otherwise spawns one food when the roll
    /// ([`EnvRandom::food_roll`], drawn only then) is below `food_spawn_chance`. New food goes on
    /// distinct uniformly random cells of [`spawn_mask`](EnvConfig::spawn_mask), at most
    /// `max(minimum_food, 1)` and as many as there are free cells. Returns the number placed.
    /// Does nothing on the `empty` map.
    pub fn spawn_food<R: EnvRandom + ?Sized>(&self, rng: &mut R, state: &mut GameState) -> u32 {
        if !self.spawns_food() {
            return 0;
        }
        let count = state.food.count_ones() as i32;
        let min = self.minimum_food();
        let chance = self.food_spawn_chance();
        let need = if count < min {
            min - count
        } else if chance > 0 && rng.food_roll() < chance {
            1
        } else {
            0
        };
        let max_n = min.max(1).min(self.rules().cells() as i32);
        let need = need.min(max_n);
        if need <= 0 {
            return 0;
        }
        let mut free = self.spawn_mask(state);
        let mut placed = 0;
        for _ in 0..need {
            match rng.pick_cell(free) {
                Some(c) => {
                    state.food |= 1u128 << c;
                    free &= !(1u128 << c);
                    placed += 1;
                }
                None => break,
            }
        }
        placed
    }

    /// Advance one turn in place (`BattlesnakeEnv.step`): the rules (`rules_step`), the map's
    /// update (food spawn; the royale map also regenerates the zone for the next turn), `turn + 1`,
    /// then `done` = at most one snake alive, or `turn >= max_turns` (truncated). Stepping a
    /// finished game changes nothing and pays no reward.
    ///
    /// `actions[i]` is seat `i`'s move id (0 up, 1 down, 2 left, 3 right; anything else repeats
    /// the seat's last move). Dead seats' entries are ignored. No heap allocation.
    pub fn step<R: EnvRandom + ?Sized>(
        &self,
        rng: &mut R,
        state: &mut GameState,
        actions: &Actions,
    ) -> StepInfo {
        if state.done {
            return StepInfo {
                done: true,
                ..StepInfo::default()
            };
        }
        let config = self.rules();
        let prev_alive = state.alive;
        let prev_turn = state.turn;
        // The rules step leaves a state with at most one snake alive untouched; otherwise the
        // royale ruleset (which the royale map requires) has already drawn next turn's zone.
        let rules_skipped = state.is_game_over(config);
        state.rules_step(config, actions);
        let spawned = self.spawn_food(rng, state);
        if self.map() == MapKind::Royale && rules_skipped {
            state.hazard = royale_hazards(config, &state.sides, prev_turn.wrapping_add(1));
        }
        state.turn = prev_turn.wrapping_add(1);
        let over = state.is_game_over(config);
        let truncated = !over && config.max_turns().is_some_and(|m| state.turn >= m);
        state.done = over || truncated;

        let (win_loss, placement) = rewards(self, &prev_alive, false, state);
        StepInfo {
            done: state.done,
            truncated,
            spawned,
            win_loss,
            placement,
        }
    }

    /// [`step`](EnvConfig::step), then replace a finished game by a fresh one
    /// (`BattlesnakeEnv.step_autoreset`). Returns the step's info and, when the game ended, the
    /// final state it reached (the `final_obs` state) while `state` becomes the new game.
    pub fn step_autoreset<R: EnvRandom + ?Sized>(
        &self,
        rng: &mut R,
        state: &mut GameState,
        actions: &Actions,
    ) -> (StepInfo, Option<GameState>) {
        let info = self.step(rng, state, actions);
        if info.done {
            let finished = *state;
            *state = self.reset(rng);
            (info, Some(finished))
        } else {
            (info, None)
        }
    }
}

/// A game in progress with its own generator: the convenient way to simulate one game.
///
/// ```
/// use slinky_env::{EnvConfig, Game, random_legal};
/// use slinky_rules::Mode;
///
/// let mut game = Game::new(EnvConfig::ladder(Mode::Duel)?, 42);
/// while !game.state.done {
///     let actions = random_legal(&mut game.rng, game.config.rules(), &game.state);
///     game.step(&actions);
/// }
/// assert!(game.state.turn > 0);
/// # Ok::<(), slinky_env::EnvError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Game {
    /// The game's configuration.
    pub config: EnvConfig,
    /// The current state.
    pub state: GameState,
    /// The game's generator (map randomness; policies may draw from it too).
    pub rng: Rng,
}

impl Game {
    /// A new game fixed by `seed`.
    pub fn new(config: EnvConfig, seed: u64) -> Game {
        let mut rng = Rng::new(seed);
        let state = config.reset(&mut rng);
        Game { config, state, rng }
    }

    /// Continue from a given state (e.g. one built from a request) with generator `seed`.
    pub fn from_state(config: EnvConfig, state: GameState, seed: u64) -> Game {
        Game {
            config,
            state,
            rng: Rng::new(seed),
        }
    }

    /// Advance one turn ([`EnvConfig::step`]).
    pub fn step(&mut self, actions: &Actions) -> StepInfo {
        self.config.step(&mut self.rng, &mut self.state, actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_legal;
    use slinky_rules::{GameConfig, Mode, Ruleset};

    fn stacked(cfg: &GameConfig, s: &mut GameState, seat: usize, x: i32, y: i32) {
        s.set_snake(cfg, seat, &[(x, y); 3], 100, None).unwrap();
    }

    #[test]
    fn done_games_are_frozen() {
        let env = EnvConfig::for_mode(Mode::Duel, None).unwrap();
        let mut rng = Rng::new(1);
        let mut s = env.reset(&mut rng);
        s.done = true;
        let before = s;
        let info = env.step(&mut rng, &mut s, &[0, 0, 0, 0]);
        assert_eq!(s, before);
        assert!(info.done && !info.truncated && info.spawned == 0);
        assert_eq!(info.win_loss, [0.0; 4]);
    }

    #[test]
    fn minimum_food_is_topped_up_and_spawns_avoid_heads() {
        let cfg = GameConfig::new(5, 5, 2, Ruleset::Standard).unwrap();
        let env = EnvConfig::new(cfg, MapKind::Standard)
            .unwrap()
            .with_food(0, 30);
        let mut rng = Rng::new(9);
        let mut s = GameState::new(&cfg);
        stacked(&cfg, &mut s, 0, 0, 0);
        stacked(&cfg, &mut s, 1, 4, 4);
        s.refresh_done(&cfg);
        // 25 cells minus 2 bodies minus 4 head neighbours: 19 free cells, all filled.
        let mut t = s;
        let placed = env.spawn_food(&mut rng, &mut t);
        assert_eq!(placed, 19);
        assert_eq!(t.food, env.spawn_mask(&s));
        // Chance 0 with the minimum met: nothing.
        let env0 = env.with_food(0, 1);
        let mut u = s;
        u.food = 1 << 12;
        assert_eq!(env0.spawn_food(&mut rng, &mut u), 0);
        // The empty map never spawns.
        let empty = EnvConfig::new(cfg, MapKind::Empty).unwrap();
        assert_eq!(empty.spawn_food(&mut rng, &mut s.clone()), 0);
    }

    #[test]
    fn truncation_and_rewards() {
        let env = EnvConfig::for_mode(Mode::Standard, None)
            .unwrap()
            .with_max_turns(Some(5));
        let mut rng = Rng::new(2);
        let mut s = env.reset(&mut rng);
        let mut last = StepInfo::default();
        for _ in 0..5 {
            let a = random_legal(&mut rng, env.rules(), &s);
            last = env.step(&mut rng, &mut s, &a);
        }
        assert!(s.done && last.done);
        assert_eq!(s.turn, 5);
        assert!(last.truncated || s.num_alive(env.rules()) <= 1);
        // A head-on of equal lengths in a duel: both die, a draw with zero rewards.
        let cfg = *EnvConfig::for_mode(Mode::Duel, None).unwrap().rules();
        let duel = EnvConfig::new(cfg, MapKind::Standard).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(4, 5), (3, 5), (2, 5)], 100, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(6, 5), (7, 5), (8, 5)], 100, None)
            .unwrap();
        s.refresh_done(&cfg);
        let prev = s;
        let info = duel.step(&mut rng, &mut s, &[3, 2, 0, 0]);
        assert!(info.done && !info.truncated);
        assert_eq!(s.num_alive(&cfg), 0);
        assert_eq!(info.win_loss, [0.0; 4]);
        assert_eq!(info.placement, [0.0; 4]);
        assert_eq!(info.win_loss, win_loss_reward(&duel, &prev, &s));
        assert_eq!(info.placement, placement_reward(&duel, &prev, &s));
    }

    #[test]
    fn autoreset_and_game() {
        let env = EnvConfig::ladder(Mode::Constrictor).unwrap();
        let mut rng = Rng::new(5);
        let mut s = env.reset(&mut rng);
        let mut resets = 0;
        for _ in 0..2000 {
            let a = random_legal(&mut rng, env.rules(), &s);
            let (info, fin) = env.step_autoreset(&mut rng, &mut s, &a);
            if info.done {
                resets += 1;
                let fin = fin.unwrap();
                assert!(fin.done);
                assert_eq!(s.turn, 0);
            }
            assert_eq!(s.food, 0);
        }
        assert!(resets > 5);
        let a = Game::new(env, 77);
        let b = Game::new(env, 77);
        assert_eq!(a.state, b.state);
    }
}
