//! The search's only randomness: per-simulation noise bits and the per-depth tie noise.
//!
//! Ported from `slinky.mcts` (`src/slinky/mcts.py`): `_depth_noise` (mcts.py:703-715) and
//! `_search_keys` (mcts.py:1074-1086, `noise = jax.random.bits(split(key, 4)[0],
//! (num_simulations, N, 4), uint32)`), see SPEC.md 4.5(a) and 4.9.
//!
//! Each simulation gets a `u32[N][4]` block of bits ([`SimBits`]); at depth `d` of its descent
//! the bits go through [`depth_noise`] to give `U(0, 1)` floats that break near-ties in the PUCT
//! scores (`0.01 * u`). A [`NoiseSource`] hands out those blocks:
//!
//! * [`JaxNoise`]: the exact stream of `jax.random.bits` for a JAX key (threefry2x32 with
//!   `jax_threefry_partitionable=True`, the default of the JAX 0.11 the agent was served with).
//!   Generated on demand per simulation, so it also serves a search whose simulation count is
//!   not known in advance (a time budget): the bits of simulation `s` do not depend on the
//!   total.
//! * [`NoiseTable`]: explicit bits, e.g. dumped from Python for a golden test.
//!
//! [`JaxKey`] reproduces the JAX key operations the agent uses (`jax.random.key(seed)`,
//! `split`, `fold_in`), so `JaxKey::new(seed)` searched in Rust draws the same noise as
//! `policy(jax.random.key(seed), state, ...)` in Python.

use slinky_rules::{MAX_SNAKES, NUM_ACTIONS};

/// One simulation's noise bits, `[seat][move]` (rows at or beyond `N` are unused).
pub type SimBits = [[u32; NUM_ACTIONS]; MAX_SNAKES];

/// Tie-break noise `U(0, 1)` for depth `depth` of a descent (`mcts._depth_noise`,
/// mcts.py:703-715): `bits + depth * 0x9E3779B9` through Wellons' lowbias32 hash, top 24 bits
/// scaled by `2^-24`. Bit-exact with the jitted Python.
#[inline]
pub fn depth_noise(bits: u32, depth: u32) -> f32 {
    let mut x = bits.wrapping_add(depth.wrapping_mul(0x9E37_79B9));
    x = (x ^ (x >> 16)).wrapping_mul(0x7FEB_352D);
    x = (x ^ (x >> 15)).wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Supplies each simulation's noise bits.
pub trait NoiseSource {
    /// The bits of simulation `sim` (0-based) of a search over `num_snakes` seats, or `None`
    /// when the source has no bits for it (a table that is too short).
    fn bits(&mut self, sim: usize, num_snakes: usize) -> Option<SimBits>;
}

/// Explicit bits, `table[sim][seat][move]` (seats beyond the game's are ignored).
#[derive(Debug, Clone, Default)]
pub struct NoiseTable {
    /// One block per simulation.
    pub table: Vec<SimBits>,
}

impl NoiseTable {
    /// A table from flat bits laid out as numpy's `uint32[sims, n, 4]` (C order).
    pub fn from_flat(flat: &[u32], num_snakes: usize) -> Option<NoiseTable> {
        let per = num_snakes * NUM_ACTIONS;
        if num_snakes == 0 || num_snakes > MAX_SNAKES || !flat.len().is_multiple_of(per) {
            return None;
        }
        let table = flat
            .chunks_exact(per)
            .map(|c| {
                let mut b = [[0u32; NUM_ACTIONS]; MAX_SNAKES];
                for (p, row) in c.as_chunks::<NUM_ACTIONS>().0.iter().enumerate() {
                    b[p] = *row;
                }
                b
            })
            .collect();
        Some(NoiseTable { table })
    }
}

impl NoiseSource for NoiseTable {
    fn bits(&mut self, sim: usize, _num_snakes: usize) -> Option<SimBits> {
        self.table.get(sim).copied()
    }
}

/// The Threefry-2x32 block cipher (20 rounds), as `jax._src.random.threefry2x32`
/// (`_threefry2x32_lowering`): `key` is the 64-bit key as two words, `x` the two count words.
pub fn threefry2x32(key: [u32; 2], x: [u32; 2]) -> [u32; 2] {
    const ROT: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    let ks = [key[0], key[1], key[0] ^ key[1] ^ 0x1BD1_1BDA];
    let (mut x0, mut x1) = (x[0].wrapping_add(ks[0]), x[1].wrapping_add(ks[1]));
    for i in 0..5u32 {
        for &r in &ROT[(i % 2) as usize] {
            x0 = x0.wrapping_add(x1);
            x1 = x1.rotate_left(r) ^ x0;
        }
        let j = i as usize;
        x0 = x0.wrapping_add(ks[(j + 1) % 3]);
        x1 = x1.wrapping_add(ks[(j + 2) % 3]).wrapping_add(i + 1);
    }
    [x0, x1]
}

/// A raw JAX threefry PRNG key (`jax.random.key_data`), with the operations of
/// `jax._src.random.threefry2x32` under `jax_threefry_partitionable=True`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JaxKey(pub [u32; 2]);

impl JaxKey {
    /// `jax.random.key(seed)` for a seed in `0..2^32` (`threefry_seed`: high word 0).
    pub fn new(seed: u32) -> JaxKey {
        JaxKey([0, seed])
    }

    /// `jax.random.fold_in(key, data)`: threefry of the count pair `(0, data)`.
    pub fn fold_in(self, data: u32) -> JaxKey {
        JaxKey(threefry2x32(self.0, [0, data]))
    }

    /// Key `i` of `jax.random.split(key, num)` (partitionable: threefry of the 64-bit index
    /// `i` as `(hi, lo)`; independent of `num`).
    pub fn split_at(self, i: u64) -> JaxKey {
        JaxKey(threefry2x32(self.0, [(i >> 32) as u32, i as u32]))
    }

    /// Element `i` (flat C-order index) of `jax.random.bits(key, shape, uint32)`.
    pub fn bits_at(self, i: u64) -> u32 {
        let [a, b] = threefry2x32(self.0, [(i >> 32) as u32, i as u32]);
        a ^ b
    }

    /// The noise stream of `mcts.search(key, ...)`: `split(key, 4)[0]` drawn as
    /// `bits((sims, N, 4))`.
    pub fn search_noise(self) -> JaxNoise {
        JaxNoise {
            key: self.split_at(0),
        }
    }
}

/// The noise bits `mcts.search` draws from its key, generated per simulation on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JaxNoise {
    /// The noise key, `split(search_key, 4)[0]`.
    pub key: JaxKey,
}

impl JaxNoise {
    /// The stream of `mcts.search(JaxKey::new(seed), ...)`.
    pub fn from_seed(seed: u32) -> JaxNoise {
        JaxKey::new(seed).search_noise()
    }
}

impl NoiseSource for JaxNoise {
    fn bits(&mut self, sim: usize, num_snakes: usize) -> Option<SimBits> {
        let mut b = [[0u32; NUM_ACTIONS]; MAX_SNAKES];
        let base = (sim as u64).checked_mul((num_snakes * NUM_ACTIONS) as u64)?;
        for (p, row) in b.iter_mut().enumerate().take(num_snakes.min(MAX_SNAKES)) {
            for (a, v) in row.iter_mut().enumerate() {
                *v = self.key.bits_at(base + (p * NUM_ACTIONS + a) as u64);
            }
        }
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefry_known_answer() {
        // Random123 known-answer vectors for threefry2x32_20.
        assert_eq!(threefry2x32([0, 0], [0, 0]), [0x6b20_0159, 0x99ba_4efe]);
        assert_eq!(
            threefry2x32([0xffff_ffff, 0xffff_ffff], [0xffff_ffff, 0xffff_ffff]),
            [0x1cb9_96fc, 0xbb00_2be7]
        );
        assert_eq!(
            threefry2x32([0x1319_8a2e, 0x0370_7344], [0x243f_6a88, 0x85a3_08d3]),
            [0xc492_3a9c, 0x483d_f7a0]
        );
    }

    #[test]
    fn depth_noise_is_unit_interval() {
        for bits in [0u32, 1, 0xdead_beef, u32::MAX] {
            for d in 0..40 {
                let u = depth_noise(bits, d);
                assert!((0.0..1.0).contains(&u));
            }
        }
    }

    #[test]
    fn jax_noise_matches_flat_index() {
        let mut n = JaxNoise::from_seed(7);
        let b = n.bits(3, 2).unwrap();
        let key = JaxKey::new(7).split_at(0);
        assert_eq!(b[1][2], key.bits_at(3 * 8 + 4 + 2));
        assert_eq!(b[2], [0; 4]);
    }

    #[test]
    fn table_from_flat() {
        let flat: Vec<u32> = (0..16).collect();
        let mut t = NoiseTable::from_flat(&flat, 2).unwrap();
        assert_eq!(t.bits(1, 2).unwrap()[1], [12, 13, 14, 15]);
        assert!(t.bits(2, 2).is_none());
        assert!(NoiseTable::from_flat(&flat[..5], 2).is_none());
    }
}
