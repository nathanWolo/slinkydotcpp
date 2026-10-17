//! Simple policies for self-play baselines and the distribution tests.

use slinky_rules::{Actions, GameConfig, GameState, NUM_ACTIONS};

use crate::rng::{EnvRandom, uniform};

/// `policies.random_legal_policy`: each seat picks uniformly among its
/// [`action_mask`](GameState::action_mask) moves (moves that do not certainly hit a wall or a
/// body; dead seats, seats certain to starve and seats with no such move pick among all 4).
/// Seats past the seat count get 0.
pub fn random_legal<R: EnvRandom + ?Sized>(
    rng: &mut R,
    config: &GameConfig,
    state: &GameState,
) -> Actions {
    let mask = state.action_mask(config);
    let mut out = [0u8; 4];
    for (seat, a) in out.iter_mut().enumerate().take(config.num_snakes()) {
        let row = mask[seat];
        let k = row.iter().filter(|&&b| b).count() as u32;
        let mut pick = uniform(rng, k.max(1));
        for (m, &ok) in row.iter().enumerate().take(NUM_ACTIONS) {
            if ok {
                if pick == 0 {
                    *a = m as u8;
                    break;
                }
                pick -= 1;
            }
        }
    }
    out
}

/// `policies.random_policy`: a uniformly random move id for every seat, ignoring the state.
pub fn random_moves<R: EnvRandom + ?Sized>(rng: &mut R, config: &GameConfig) -> Actions {
    let mut out = [0u8; 4];
    for a in out.iter_mut().take(config.num_snakes()) {
        *a = uniform(rng, NUM_ACTIONS as u32) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use slinky_rules::Mode;

    #[test]
    fn random_legal_keeps_to_the_mask() {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let mut s = GameState::new(&cfg);
        // Seat 0 in the bottom-left corner moving up: down and left are walls.
        s.set_snake(&cfg, 0, &[(0, 1), (0, 0), (1, 0)], 100, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(5, 5); 3], 100, None).unwrap();
        s.refresh_done(&cfg);
        let mut rng = Rng::new(0);
        let mut seen = [0u32; 4];
        for _ in 0..4000 {
            let a = random_legal(&mut rng, &cfg, &s);
            assert!(a[0] == 0 || a[0] == 3, "{a:?}");
            seen[a[1] as usize] += 1;
            assert_eq!(&a[2..], &[0, 0]);
        }
        assert!(seen.iter().all(|&c| c > 800));
        let r = random_moves(&mut rng, &cfg);
        assert!(r[0] < 4 && r[1] < 4 && r[2] == 0);
    }
}
