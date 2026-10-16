//! The randomness of the environment: a seedable PRNG and the [`EnvRandom`] trait every random
//! choice goes through.
//!
//! slinky draws its map randomness from JAX's threefry stream and only matches the official
//! engine's *distributions* (maps.py:15-18). This crate does the same with its own generator:
//! [`Rng`] is xoshiro256++ seeded through SplitMix64, so a `u64` seed fixes a whole game, and every
//! choice is an unbiased uniform integer (Lemire's method), as slinky's `argsort`/`argmax` of
//! random scores is uniform.
//!
//! The environment never calls the generator directly: it asks an [`EnvRandom`] for a uniform
//! integer, the food-spawn roll, a free cell or a shrink side. The defaults of those methods are
//! the sampling rules; a test can override them to replay the exact choices slinky made
//! (`tests/golden.rs` does this with the roll and the spawned cells of each Python transition).

/// A source of the environment's random choices.
///
/// Only [`next_u64`](EnvRandom::next_u64) is required. The provided methods define how each
/// choice is sampled; override them to script choices (tests) without changing the env.
pub trait EnvRandom {
    /// 64 uniformly random bits.
    fn next_u64(&mut self) -> u64;

    /// A uniform integer in `0..n` (0 when `n == 0`), without modulo bias (Lemire's method).
    fn below(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let mut m = u64::from((self.next_u64() >> 32) as u32) * u64::from(n);
        if (m as u32) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u32) < threshold {
                m = u64::from((self.next_u64() >> 32) as u32) * u64::from(n);
            }
        }
        (m >> 32) as u32
    }

    /// The standard map's spawn roll, uniform in `1..=100`. The engine spawns a food when
    /// `roll < food_spawn_chance` (`100 - rand.Intn(100) < chance`, maps.py:344-356).
    fn food_roll(&mut self) -> i32 {
        100 - uniform(self, 100) as i32
    }

    /// A uniformly random set bit of `mask` (a cell index), or `None` when `mask` is 0.
    fn pick_cell(&mut self, mask: u128) -> Option<usize> {
        let n = mask.count_ones();
        if n == 0 {
            return None;
        }
        Some(nth_set_bit(mask, uniform(self, n)))
    }

    /// A royale shrink side, uniform in `0..4` (0 left, 1 right, 2 bottom, 3 top;
    /// `maps._random_sides`, maps.py:379-381).
    fn side(&mut self) -> i8 {
        uniform(self, 4) as i8
    }
}

/// [`EnvRandom::below`] kept inside `0..n` (0 when `n == 0`) even when an overriding
/// implementation breaks the contract (a value `k >= n` becomes `k % n`), so a scripted
/// generator can never make the env index or shift out of range.
#[inline]
pub(crate) fn uniform<R: EnvRandom + ?Sized>(rng: &mut R, n: u32) -> u32 {
    let k = rng.below(n);
    if k < n {
        k
    } else {
        k.checked_rem(n).unwrap_or(0)
    }
}

/// [`EnvRandom::pick_cell`] kept to the set bits of `mask`: `None` when the generator answers
/// `None` (no more picks) or a cell that is not in `mask` (an out-of-contract override).
#[inline]
pub(crate) fn pick_in<R: EnvRandom + ?Sized>(rng: &mut R, mask: u128) -> Option<usize> {
    rng.pick_cell(mask)
        .filter(|&c| c < 128 && (mask >> c) & 1 == 1)
}

/// [`EnvRandom::side`] kept in `0..4` (an out-of-range side is reduced modulo 4).
#[inline]
pub(crate) fn side_of<R: EnvRandom + ?Sized>(rng: &mut R) -> i8 {
    rng.side().rem_euclid(4)
}

/// The index of the `k`-th set bit of `mask` (0-based, from the least significant bit).
/// `k` must be below `mask.count_ones()`; otherwise the result is 128.
#[inline]
pub fn nth_set_bit(mask: u128, k: u32) -> usize {
    let lo = mask as u64;
    let c = lo.count_ones();
    let (mut word, base, k) = if k < c {
        (lo, 0, k)
    } else {
        ((mask >> 64) as u64, 64, k - c)
    };
    if k >= word.count_ones() {
        return 128;
    }
    for _ in 0..k {
        word &= word - 1;
    }
    base + word.trailing_zeros() as usize
}

/// xoshiro256++ (Blackman and Vigna), seeded from a `u64` with SplitMix64. Fast, small
/// (32 bytes, `Copy`) and statistically strong; not cryptographic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

/// One SplitMix64 step: advances `state` and returns the next output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// A generator fixed by `seed` (any value, 0 included).
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        // SplitMix64 never yields four zero words, but keep the invariant explicit.
        if s == [0; 4] {
            return Rng { s: [1, 2, 3, 4] };
        }
        Rng { s }
    }

    /// The generator of stream `stream` under `seed`: independent seeds for parallel games
    /// (`Rng::new` of a SplitMix64 hash of both).
    pub fn with_stream(seed: u64, stream: u64) -> Rng {
        let mut sm = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Rng::new(splitmix64(&mut sm) ^ stream)
    }

    /// A uniform `f64` in `[0, 1)` (53 bits).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl EnvRandom for Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nth_set_bit_walks_both_words() {
        let mask: u128 = (1 << 3) | (1 << 63) | (1 << 64) | (1 << 120);
        assert_eq!(nth_set_bit(mask, 0), 3);
        assert_eq!(nth_set_bit(mask, 1), 63);
        assert_eq!(nth_set_bit(mask, 2), 64);
        assert_eq!(nth_set_bit(mask, 3), 120);
        assert_eq!(nth_set_bit(mask, 4), 128);
        assert_eq!(nth_set_bit(0, 0), 128);
        assert_eq!(nth_set_bit(u128::MAX, 127), 127);
    }

    #[test]
    fn deterministic_and_seed_sensitive() {
        let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let zs: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
        assert_ne!(Rng::with_stream(1, 0), Rng::with_stream(1, 1));
        assert_ne!(Rng::with_stream(1, 0), Rng::new(1));
    }

    #[test]
    fn below_is_uniform() {
        let mut r = Rng::new(1);
        for n in [1u32, 2, 3, 7, 100] {
            let mut counts = vec![0u32; n as usize];
            let draws = 20_000 * n;
            for _ in 0..draws {
                let v = r.below(n);
                assert!(v < n);
                counts[v as usize] += 1;
            }
            for &c in &counts {
                // 20,000 expected per bucket; 6 sigma is about 850.
                assert!((c as i64 - 20_000).abs() < 900, "n={n} counts={counts:?}");
            }
        }
        assert_eq!(r.below(0), 0);
        let roll_ok = (0..10_000)
            .map(|_| r.food_roll())
            .all(|v| (1..=100).contains(&v));
        assert!(roll_ok);
        assert_eq!(r.pick_cell(0), None);
        for _ in 0..1000 {
            let c = r.pick_cell(0b1010_0000).unwrap();
            assert!(c == 5 || c == 7);
        }
    }
}
