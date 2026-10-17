//! Scoring a finished game from A's point of view, with slinky's conventions.
//!
//! * W/D/L (`evaluate._compiled_match.outcome`, the score of strength.py): A **wins** when it
//!   is the last snake alive; it **draws** when the game was cut off by `max_turns` while A and
//!   at least one other snake were alive (`truncated`), or when A was eliminated on the final
//!   turn together with every other snake; any other elimination is a **loss**. The score is
//!   `(wins + draws / 2) / games`.
//! * Share and placement (`evaluate._compiled_table.outcome`, the 4-snake table statistics):
//!   the snakes sharing the win are the survivors, or, if nobody survived, those eliminated on
//!   the final turn; a policy's **share** is the fraction of that group it holds (parity
//!   `seats / N`). A seat's **placement** is `1 + (snakes that outlasted it) + 0.5 * (other
//!   snakes eliminated on the same turn)`, a living snake counting as eliminated at infinity;
//!   a policy's placement is the mean over its seats (parity `(N + 1) / 2`).
//!
//! With early stopping (`play_table(early_stop=True)`, the referee's default) a game ends as
//! soon as it is decided between the two agents: A is dead, or A is the only snake left. All
//! three statistics are then already fixed: a dead A has lost (unless every other snake died
//! with it, which the rules end anyway), its placement counts the snakes still alive as
//! outlasting it, and its share is 0.

use slinky_rules::MAX_SNAKES;

/// A game's result for A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// A was the last snake alive.
    Win,
    /// Cut off with A alive, or A eliminated together with every other snake on the last turn.
    Draw,
    /// Any other elimination of A.
    Loss,
}

impl Outcome {
    /// `"W"`, `"D"` or `"L"`.
    pub fn code(self) -> &'static str {
        match self {
            Outcome::Win => "W",
            Outcome::Draw => "D",
            Outcome::Loss => "L",
        }
    }

    /// Parse [`Outcome::code`].
    pub fn from_code(code: &str) -> Option<Outcome> {
        match code {
            "W" => Some(Outcome::Win),
            "D" => Some(Outcome::Draw),
            "L" => Some(Outcome::Loss),
            _ => None,
        }
    }
}

/// The scores of one finished (or decided) game for the agent in seat `a_seat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameScore {
    /// W/D/L from A's point of view.
    pub outcome: Outcome,
    /// Cut off by `max_turns` with A and another snake alive (counted as a draw too).
    pub truncated: bool,
    /// A's share of the win (0 to 1).
    pub share: f64,
    /// A's placement (1 = best, `N` = worst).
    pub placement: f64,
}

/// Whether the game is still contested between A (seat `a_seat`) and the other agent: A and
/// at least one other snake are alive.
pub fn contested(alive: &[bool], a_seat: usize) -> bool {
    alive[a_seat] && alive.iter().enumerate().any(|(i, &x)| x && i != a_seat)
}

/// Score a final state: `alive` and `elim_turn` per seat (`elim_turn` is the turn count after
/// the step that eliminated a seat, 0 while alive, as in `slinky.types.State`), A in `a_seat`.
/// `alive.len()` is the seat count (2 to 4).
pub fn score_game(alive: &[bool], elim_turn: &[i32], a_seat: usize) -> GameScore {
    let n = alive.len();
    debug_assert!((2..=MAX_SNAKES).contains(&n) && elim_turn.len() == n && a_seat < n);
    let remaining = alive.iter().filter(|&&x| x).count();
    let a_alive = alive[a_seat];
    // run_match's outcome code.
    let cut_off = a_alive && remaining >= 2;
    let max_elim = elim_turn.iter().copied().max().unwrap_or(0);
    let all_died = !a_alive && remaining == 0 && elim_turn[a_seat] == max_elim;
    let outcome = if a_alive && remaining == 1 {
        Outcome::Win
    } else if cut_off || all_died {
        Outcome::Draw
    } else {
        Outcome::Loss
    };
    // play_table's share: the group sharing the win.
    let last = (0..n)
        .map(|i| if alive[i] { 0 } else { elim_turn[i] })
        .max()
        .unwrap_or(0);
    let in_group = |i: usize| {
        if remaining > 0 {
            alive[i]
        } else {
            elim_turn[i] == last
        }
    };
    let group = (0..n).filter(|&i| in_group(i)).count().max(1);
    let share = if in_group(a_seat) {
        1.0 / group as f64
    } else {
        0.0
    };
    // Placement: a living seat counts as eliminated at infinity.
    let out = |i: usize| {
        if alive[i] {
            i64::MAX
        } else {
            i64::from(elim_turn[i])
        }
    };
    let mine = out(a_seat);
    let outlasted = (0..n).filter(|&j| out(j) > mine).count();
    let tied = (0..n).filter(|&j| j != a_seat && out(j) == mine).count();
    let placement = 1.0 + outlasted as f64 + 0.5 * tied as f64;
    GameScore {
        outcome,
        truncated: cut_off,
        share,
        placement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(alive: &[bool], elim: &[i32], a: usize) -> (Outcome, bool, f64, f64) {
        let s = score_game(alive, elim, a);
        (s.outcome, s.truncated, s.share, s.placement)
    }

    // Each case below was worked out by hand from evaluate.py's outcome code
    // (_compiled_match.outcome) and the table's outcome (_compiled_table.outcome).

    #[test]
    fn duel_cases() {
        // A (seat 0) survives, B died on turn 57: win, share 1, first place.
        assert_eq!(
            score(&[true, false], &[0, 57], 0),
            (Outcome::Win, false, 1.0, 1.0)
        );
        // A in seat 1 died on turn 57: loss.
        assert_eq!(
            score(&[true, false], &[0, 57], 1),
            (Outcome::Loss, false, 0.0, 2.0)
        );
        // Head-on: both eliminated on turn 80: draw, half the win, placement 1.5 each.
        assert_eq!(
            score(&[false, false], &[80, 80], 0),
            (Outcome::Draw, false, 0.5, 1.5)
        );
        assert_eq!(
            score(&[false, false], &[80, 80], 1),
            (Outcome::Draw, false, 0.5, 1.5)
        );
        // Cut off at max_turns with both alive: draw, truncated.
        assert_eq!(
            score(&[true, true], &[0, 0], 0),
            (Outcome::Draw, true, 0.5, 1.5)
        );
    }

    #[test]
    fn four_snake_cases() {
        // A (seat 2) is the last snake: B seats died on turns 10, 30 and 31.
        assert_eq!(
            score(&[false, false, true, false], &[10, 30, 0, 31], 2),
            (Outcome::Win, false, 1.0, 1.0)
        );
        // A (seat 0) died first on turn 10; the game is decided (early stop) with 3 B alive:
        // loss, share 0, placement 4.
        assert_eq!(
            score(&[false, true, true, true], &[10, 0, 0, 0], 0),
            (Outcome::Loss, false, 0.0, 4.0)
        );
        // A (seat 1) died on turn 40 together with seat 3; seats 0 and 2 alive (decided):
        // loss, placement 1 + 2 + 0.5 = 3.5.
        assert_eq!(
            score(&[true, false, true, false], &[0, 40, 0, 40], 1),
            (Outcome::Loss, false, 0.0, 3.5)
        );
        // The last three snakes (A in seat 3 among them) all die on turn 99, seat 0 died
        // earlier: draw, a third of the win, placement 1 + 0 + 0.5 * 2 = 2.
        assert_eq!(
            score(&[false, false, false, false], &[12, 99, 99, 99], 3),
            (Outcome::Draw, false, 1.0 / 3.0, 2.0)
        );
        // Everyone died, A (seat 0) before the last two: loss, share 0, placement
        // 1 + 2 + 0.5 = 3.5 (seat 1 died with A on turn 50).
        assert_eq!(
            score(&[false, false, false, false], &[50, 50, 70, 70], 0),
            (Outcome::Loss, false, 0.0, 3.5)
        );
        // Cut off with A (seat 1) and two B alive: draw (truncated), a third of the win,
        // placement 1 + 0 + 0.5 * 2 = 2.
        assert_eq!(
            score(&[false, true, true, true], &[20, 0, 0, 0], 1),
            (Outcome::Draw, true, 1.0 / 3.0, 2.0)
        );
        // Cut off with A dead and two B alive: loss, not truncated (as run_match).
        assert_eq!(
            score(&[true, false, true, false], &[0, 20, 0, 5], 1),
            (Outcome::Loss, false, 0.0, 3.0)
        );
    }

    #[test]
    fn contested_needs_a_and_another_snake() {
        assert!(contested(&[true, true, false, false], 0));
        assert!(!contested(&[false, true, true, false], 0));
        assert!(!contested(&[true, false, false, false], 0));
        assert!(contested(&[false, true, true, false], 2));
    }
}
