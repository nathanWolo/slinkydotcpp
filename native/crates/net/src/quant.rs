//! The int8 network body ([`Network::quantize_int8`](crate::Network::quantize_int8)): every
//! convolution after the first and the dense hidden layer in 8-bit integers.
//!
//! Weights are rounded to int8 with one symmetric scale per output channel. Activations (the
//! ReLU outputs, so never negative) are rounded to `0..=127` with one scale per board and
//! layer, the layer's largest value over 127: nothing to calibrate, and every board uses its
//! full range. Sums are exact in `i32`, then scaled back to `f32` for the bias and the ReLU, so
//! the result does not depend on the kernel: the plain Rust one, `sdot` on aarch64 CPUs with
//! the dot-product extension (Graviton2 and later), or AVX-VNNI on x86-64, chosen at run time.
//! The first convolution stays `f32` (its input is a sparse list, see [`BoardInput`]), and so
//! does the share of every pixel that only sees zero padding (copied from the all-zero board
//! in `f32`, as in the `f32` pass).
//!
//! A dot-product instruction multiplies four int8 pairs per 32-bit lane, so one 128-bit
//! instruction does 16 multiply-adds where an `f32` one does 4. The kernels keep enough
//! independent accumulators that consecutive dot products do not wait on each other: two banks
//! (even and odd input groups) for a convolution pixel, whose three tap rows are summed in
//! registers, or several boards sharing each weight load in the hidden layer. A convolution
//! reads every input group, zero or not (four in five are not, and a test per group costs more
//! in mispredicted branches than it saves); the hidden layer first lists the groups that are
//! not zero in a chunk of boards (the pixels that are not live are zero there), so its loop has
//! no tests either.
//!
//! [`BoardInput`]: crate::BoardInput

use crate::NetError;

/// The dot-product kernel in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Kernel {
    /// Plain Rust (any CPU).
    Scalar,
    /// aarch64 `sdot` (the `dotprod` extension).
    Sdot,
    /// x86-64 AVX-VNNI `vpdpbusd`.
    Vnni,
}

impl Int8Kernel {
    /// The fastest kernel this CPU runs.
    pub fn detect() -> Int8Kernel {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Int8Kernel::Sdot;
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return Int8Kernel::Vnni;
        }
        Int8Kernel::Scalar
    }

    /// Whether this CPU runs the kernel.
    pub fn supported(self) -> bool {
        match self {
            Int8Kernel::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Int8Kernel::Sdot => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Int8Kernel::Vnni => std::arch::is_x86_feature_detected!("avxvnni"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// The four values of a group as one word (zero when all are).
#[inline(always)]
pub(crate) fn bits(g: &[i8; 4]) -> u32 {
    u32::from_ne_bytes(g.map(|v| v as u8))
}

/// A run of consecutive input groups against consecutive weight groups: `x[x0 + j]` with
/// `w[w0 + j]` for `j < len` (a convolution's tap row: neighbouring input pixels against
/// neighbouring taps).
#[derive(Clone, Copy, Default)]
pub(crate) struct Run {
    pub x0: usize,
    pub w0: usize,
    pub len: usize,
}

/// One row: `acc[o] += sum over runs, j < len of dot(x[x0 + j], w[(w0 + j) * n + o])` for
/// `o < n` (`n = acc.len()`, a multiple of 16). `x` values are in `0..=127`; `w` is
/// `[groups][n]` groups of four int8. The sums of all runs stay in registers, and zero groups
/// are not skipped: four in five are not zero after a ReLU here, and a test per group costs more
/// in mispredicted branches than it saves.
pub(crate) fn dot_runs(k: Int8Kernel, acc: &mut [i32], x: &[[i8; 4]], runs: &[Run], w: &[[i8; 4]]) {
    let n = acc.len();
    assert!(n.is_multiple_of(16));
    assert!(
        runs.iter()
            .all(|r| r.x0 + r.len <= x.len() && (r.w0 + r.len) * n <= w.len())
    );
    match k {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: only selected when the CPU has the dot-product extension (`supported`); the
        // runs were checked above.
        Int8Kernel::Sdot => unsafe { neon::dot_runs(acc, x, runs, w) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: only selected when the CPU has AVX-VNNI (`supported`); the runs were checked
        // above.
        Int8Kernel::Vnni => unsafe { vnni::dot_runs(acc, x, runs, w) },
        _ => {
            for r in runs {
                for (xg, j) in x[r.x0..r.x0 + r.len].iter().zip(r.w0..) {
                    for (a, wg) in acc.iter_mut().zip(&w[j * n..][..n]) {
                        *a += (0..4)
                            .map(|t| i32::from(xg[t]) * i32::from(wg[t]))
                            .sum::<i32>();
                    }
                }
            }
        }
    }
}

/// Several rows sharing `w`: `acc` is `[rows][n]`, `x` is `[rows][groups]`, `w` is
/// `[groups][n]`, and `acc[r][o] += sum over g of dot(x[r][g], w[g][o])`. Each weight vector
/// loaded serves a chunk of up to four rows, and the groups that are zero in all of a chunk's
/// rows (most of the hidden layer's input: the pixels that are not live) are left out of one
/// list, so the loop has no tests.
pub(crate) fn dot_rows(k: Int8Kernel, acc: &mut [i32], n: usize, x: &[[i8; 4]], w: &[[i8; 4]]) {
    let rows = acc.len() / n;
    if rows == 0 {
        return;
    }
    let groups = x.len() / rows;
    assert!(n.is_multiple_of(16) && acc.len() == rows * n && x.len() == rows * groups);
    assert!(groups * n <= w.len() && groups <= usize::from(u16::MAX));
    let mut list = Vec::with_capacity(groups);
    let mut r = 0;
    while r < rows {
        let take = match rows - r {
            1 => 1,
            2 | 3 => 2,
            _ => 4,
        };
        let a = &mut acc[r * n..(r + take) * n];
        let xr = &x[r * groups..(r + take) * groups];
        list.clear();
        list.extend(
            (0..groups as u16)
                .filter(|&g| (0..take).any(|i| bits(&xr[i * groups + usize::from(g)]) != 0)),
        );
        match k {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: only selected when the CPU has the dot-product extension (`supported`);
            // the listed groups are below `groups`, checked against `w` above.
            Int8Kernel::Sdot => unsafe { neon::dot_rows(a, n, xr, &list, w, take) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: only selected when the CPU has AVX-VNNI (`supported`); the listed groups
            // are below `groups`, checked against `w` above.
            Int8Kernel::Vnni => unsafe { vnni::dot_rows(a, n, xr, &list, w, take) },
            _ => {
                for (ar, xg) in a.chunks_exact_mut(n).zip(xr.chunks_exact(groups)) {
                    for &g in &list {
                        let (xg, wg) = (xg[usize::from(g)], &w[usize::from(g) * n..][..n]);
                        for (a, wv) in ar.iter_mut().zip(wg) {
                            *a += (0..4)
                                .map(|t| i32::from(xg[t]) * i32::from(wv[t]))
                                .sum::<i32>();
                        }
                    }
                }
            }
        }
        r += take;
    }
}

// The register loops index several arrays (accumulators, weights, rows) together.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// `acc + dot4(a, b)` per lane (`sdot`; stable Rust has no intrinsic for it yet).
    #[inline(always)]
    fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        // SAFETY: register-only arithmetic; the callers run only on CPUs with `dotprod`.
        unsafe {
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        acc
    }

    /// Group `g` broadcast to every lane.
    #[inline(always)]
    fn splat(g: &[i8; 4]) -> int8x16_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vreinterpretq_s8_u32(vdupq_n_u32(super::bits(g))) }
    }

    /// Outputs `t0 .. t0 + 4 K` of one row in two banks of accumulators (even and odd groups of
    /// each run), so consecutive `sdot`s do not wait on each other.
    #[inline(always)]
    unsafe fn tile_runs<const K: usize>(
        acc: &mut [i32],
        x: &[[i8; 4]],
        runs: &[super::Run],
        w: &[[i8; 4]],
        t0: usize,
    ) {
        let n = acc.len();
        // SAFETY (all loads and stores): `t0 + 4 K <= n == acc.len()`, and the runs are inside
        // `x` and `w` (checked by `super::dot_runs`).
        unsafe {
            let ap = acc.as_mut_ptr().add(t0);
            let mut a: [int32x4_t; K] = core::array::from_fn(|k| vld1q_s32(ap.add(4 * k)));
            let mut b: [int32x4_t; K] = [vdupq_n_s32(0); K];
            for r in runs {
                let xr = x.get_unchecked(r.x0..r.x0 + r.len);
                let wp = w.as_ptr().add(r.w0 * n + t0) as *const i8;
                let (pairs, rest) = xr.as_chunks::<2>();
                for (p, xp) in pairs.iter().enumerate() {
                    let (x0, x1) = (splat(&xp[0]), splat(&xp[1]));
                    let w0 = wp.add(8 * p * n);
                    let w1 = w0.add(4 * n);
                    for k in 0..K {
                        a[k] = sdot(a[k], vld1q_s8(w0.add(16 * k)), x0);
                        b[k] = sdot(b[k], vld1q_s8(w1.add(16 * k)), x1);
                    }
                }
                if let Some(xg) = rest.first() {
                    let (x0, w0) = (splat(xg), wp.add(8 * pairs.len() * n));
                    for k in 0..K {
                        a[k] = sdot(a[k], vld1q_s8(w0.add(16 * k)), x0);
                    }
                }
            }
            for k in 0..K {
                vst1q_s32(ap.add(4 * k), vaddq_s32(a[k], b[k]));
            }
        }
    }

    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn dot_runs(
        acc: &mut [i32],
        x: &[[i8; 4]],
        runs: &[super::Run],
        w: &[[i8; 4]],
    ) {
        let n = acc.len();
        let mut t0 = 0;
        // SAFETY: the tiles cover `0..n` without passing it.
        unsafe {
            while t0 + 32 <= n {
                tile_runs::<8>(acc, x, runs, w, t0);
                t0 += 32;
            }
            if t0 < n {
                tile_runs::<4>(acc, x, runs, w, t0);
            }
        }
    }

    /// Outputs `t0 .. t0 + 4 K` of `R` rows: each weight vector feeds the `R` rows.
    #[inline(always)]
    unsafe fn tile_rows<const R: usize, const K: usize>(
        acc: &mut [i32],
        n: usize,
        x: &[[i8; 4]],
        list: &[u16],
        w: &[[i8; 4]],
        t0: usize,
    ) {
        let groups = x.len() / R;
        // SAFETY (all loads and stores): `acc` is `[R][n]` with `t0 + 4 K <= n`, and the listed
        // groups are below `groups` with their rows of `w` inside it (checked by the caller).
        unsafe {
            let ap = acc.as_mut_ptr();
            let mut a: [[int32x4_t; K]; R] = core::array::from_fn(|r| {
                core::array::from_fn(|k| vld1q_s32(ap.add(r * n + t0 + 4 * k)))
            });
            for &g in list {
                let g = usize::from(g);
                let xb: [int8x16_t; R] =
                    core::array::from_fn(|r| splat(x.get_unchecked(r * groups + g)));
                let wp = w.as_ptr().add(g * n + t0) as *const i8;
                for k in 0..K {
                    let wv = vld1q_s8(wp.add(16 * k));
                    for r in 0..R {
                        a[r][k] = sdot(a[r][k], wv, xb[r]);
                    }
                }
            }
            for r in 0..R {
                for k in 0..K {
                    vst1q_s32(ap.add(r * n + t0 + 4 * k), a[r][k]);
                }
            }
        }
    }

    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn dot_rows(
        acc: &mut [i32],
        n: usize,
        x: &[[i8; 4]],
        list: &[u16],
        w: &[[i8; 4]],
        rows: usize,
    ) {
        // SAFETY: the tiles cover `0..n` without passing it; 16 accumulators either way.
        unsafe {
            match rows {
                1 if n.is_multiple_of(64) => {
                    for t0 in (0..n).step_by(64) {
                        tile_rows::<1, 16>(acc, n, x, list, w, t0);
                    }
                }
                1 => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<1, 4>(acc, n, x, list, w, t0);
                    }
                }
                2 if n.is_multiple_of(32) => {
                    for t0 in (0..n).step_by(32) {
                        tile_rows::<2, 8>(acc, n, x, list, w, t0);
                    }
                }
                2 => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<2, 4>(acc, n, x, list, w, t0);
                    }
                }
                _ => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<4, 4>(acc, n, x, list, w, t0);
                    }
                }
            }
        }
    }
}

// The register loops index several arrays (accumulators, weights, rows) together.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "x86_64")]
mod vnni {
    use core::arch::x86_64::*;

    /// Group `g` broadcast to every lane. Activations are `0..=127`, so the unsigned reading
    /// `vpdpbusd` gives them is their value.
    #[inline(always)]
    fn splat(g: &[i8; 4]) -> __m256i {
        // SAFETY: AVX is enabled in every caller (AVX-VNNI implies it).
        unsafe { _mm256_set1_epi32(super::bits(g) as i32) }
    }

    /// Outputs `t0 .. t0 + 8 K` of one row in two banks of accumulators, as on aarch64.
    #[inline(always)]
    unsafe fn tile_runs<const K: usize>(
        acc: &mut [i32],
        x: &[[i8; 4]],
        runs: &[super::Run],
        w: &[[i8; 4]],
        t0: usize,
    ) {
        let n = acc.len();
        // SAFETY (all loads and stores): `t0 + 8 K <= n == acc.len()`, and the runs are inside
        // `x` and `w` (checked by `super::dot_runs`).
        unsafe {
            let ap = acc.as_mut_ptr().add(t0) as *mut __m256i;
            let mut a: [__m256i; K] = core::array::from_fn(|k| _mm256_loadu_si256(ap.add(k)));
            let mut b: [__m256i; K] = [_mm256_setzero_si256(); K];
            for r in runs {
                let xr = x.get_unchecked(r.x0..r.x0 + r.len);
                let wp = w.as_ptr().add(r.w0 * n + t0);
                let (pairs, rest) = xr.as_chunks::<2>();
                for (p, xp) in pairs.iter().enumerate() {
                    let (x0, x1) = (splat(&xp[0]), splat(&xp[1]));
                    let w0 = wp.add(2 * p * n) as *const __m256i;
                    let w1 = wp.add((2 * p + 1) * n) as *const __m256i;
                    for k in 0..K {
                        a[k] = _mm256_dpbusd_avx_epi32(a[k], x0, _mm256_loadu_si256(w0.add(k)));
                        b[k] = _mm256_dpbusd_avx_epi32(b[k], x1, _mm256_loadu_si256(w1.add(k)));
                    }
                }
                if let Some(xg) = rest.first() {
                    let x0 = splat(xg);
                    let w0 = wp.add(2 * pairs.len() * n) as *const __m256i;
                    for k in 0..K {
                        a[k] = _mm256_dpbusd_avx_epi32(a[k], x0, _mm256_loadu_si256(w0.add(k)));
                    }
                }
            }
            for k in 0..K {
                _mm256_storeu_si256(ap.add(k), _mm256_add_epi32(a[k], b[k]));
            }
        }
    }

    #[target_feature(enable = "avxvnni")]
    pub(super) unsafe fn dot_runs(
        acc: &mut [i32],
        x: &[[i8; 4]],
        runs: &[super::Run],
        w: &[[i8; 4]],
    ) {
        let n = acc.len();
        let mut t0 = 0;
        // SAFETY: the tiles cover `0..n` without passing it.
        unsafe {
            while t0 + 32 <= n {
                tile_runs::<4>(acc, x, runs, w, t0);
                t0 += 32;
            }
            if t0 < n {
                tile_runs::<2>(acc, x, runs, w, t0);
            }
        }
    }

    /// Outputs `t0 .. t0 + 8 K` of `R` rows: each weight vector feeds the `R` rows.
    #[inline(always)]
    unsafe fn tile_rows<const R: usize, const K: usize>(
        acc: &mut [i32],
        n: usize,
        x: &[[i8; 4]],
        list: &[u16],
        w: &[[i8; 4]],
        t0: usize,
    ) {
        let groups = x.len() / R;
        // SAFETY (all loads and stores): `acc` is `[R][n]` with `t0 + 8 K <= n`, and the listed
        // groups are below `groups` with their rows of `w` inside it (checked by the caller).
        unsafe {
            let ap = acc.as_mut_ptr();
            let mut a: [[__m256i; K]; R] = core::array::from_fn(|r| {
                core::array::from_fn(|k| {
                    _mm256_loadu_si256(ap.add(r * n + t0 + 8 * k) as *const __m256i)
                })
            });
            for &g in list {
                let g = usize::from(g);
                let xb: [__m256i; R] =
                    core::array::from_fn(|r| splat(x.get_unchecked(r * groups + g)));
                let wp = w.as_ptr().add(g * n + t0) as *const __m256i;
                for k in 0..K {
                    let wv = _mm256_loadu_si256(wp.add(k));
                    for r in 0..R {
                        a[r][k] = _mm256_dpbusd_avx_epi32(a[r][k], xb[r], wv);
                    }
                }
            }
            for r in 0..R {
                for k in 0..K {
                    _mm256_storeu_si256(ap.add(r * n + t0 + 8 * k) as *mut __m256i, a[r][k]);
                }
            }
        }
    }

    #[target_feature(enable = "avxvnni")]
    pub(super) unsafe fn dot_rows(
        acc: &mut [i32],
        n: usize,
        x: &[[i8; 4]],
        list: &[u16],
        w: &[[i8; 4]],
        rows: usize,
    ) {
        // SAFETY: the tiles cover `0..n` without passing it; 8 accumulators either way.
        unsafe {
            match rows {
                1 if n.is_multiple_of(64) => {
                    for t0 in (0..n).step_by(64) {
                        tile_rows::<1, 8>(acc, n, x, list, w, t0);
                    }
                }
                1 => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<1, 2>(acc, n, x, list, w, t0);
                    }
                }
                2 if n.is_multiple_of(32) => {
                    for t0 in (0..n).step_by(32) {
                        tile_rows::<2, 4>(acc, n, x, list, w, t0);
                    }
                }
                2 => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<2, 2>(acc, n, x, list, w, t0);
                    }
                }
                _ => {
                    for t0 in (0..n).step_by(16) {
                        tile_rows::<4, 2>(acc, n, x, list, w, t0);
                    }
                }
            }
        }
    }
}

/// Symmetric int8 rounding of `w` (`[rows][cols]`, row-major) with one scale per column:
/// `(q, scale)`, `w ~ q * scale`.
pub(crate) fn quantize_columns(w: &[f32], cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut scale = vec![0f32; cols];
    for row in w.chunks_exact(cols) {
        for (s, &v) in scale.iter_mut().zip(row) {
            *s = s.max(v.abs());
        }
    }
    for s in scale.iter_mut() {
        *s = if *s > 0.0 { *s / 127.0 } else { 1.0 };
    }
    let q = w
        .chunks_exact(cols)
        .flat_map(|row| {
            row.iter()
                .zip(&scale)
                .map(|(&v, &s)| (v / s).round().clamp(-127.0, 127.0) as i8)
        })
        .collect();
    (q, scale)
}

/// Reorders `[rows][cols]` int8 (rows a multiple of 4) into `[rows / 4][cols]` groups of the
/// four rows' values, the layout the kernels read.
pub(crate) fn group_rows(q: &[i8], cols: usize) -> Vec<[i8; 4]> {
    let rows = q.len() / cols;
    let mut out = vec![[0i8; 4]; rows / 4 * cols];
    for (r, row) in q.chunks_exact(cols).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[r / 4 * cols + c][r % 4] = v;
        }
    }
    out
}

/// The scale of a layer's activations whose largest value is `max`, `max / 127` (1 when all
/// are zero), and its inverse.
pub(crate) fn act_scale(max: f32) -> (f32, f32) {
    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
    (scale, 1.0 / scale)
}

/// A non-negative activation at scale `1 / inv`, rounded half up into `0..=127` by truncation
/// (one instruction, where `f32::round` is a library call on baseline x86-64); the same
/// operations as [`V4::quantize`](crate::simd::V4::quantize).
#[inline(always)]
pub(crate) fn act(v: f32, inv: f32) -> i8 {
    ((v * inv + 0.5) as i32).min(127) as i8
}

/// Why the int8 body cannot be built for a network, if it cannot.
pub(crate) fn check_shapes(channels: &[usize], flat: usize, hidden: usize) -> Result<(), NetError> {
    let bad = channels.len() < 2
        || channels.iter().any(|&c| !c.is_multiple_of(16))
        || !flat.is_multiple_of(4)
        || flat / 4 > usize::from(u16::MAX)
        || !hidden.is_multiple_of(16);
    if bad {
        return Err(NetError::Shape(format!(
            "int8 needs two or more convolutions with multiples of 16 channels and a hidden \
             width that is a multiple of 16 (conv channels {channels:?}, hidden {hidden})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNELS: [Int8Kernel; 3] = [Int8Kernel::Scalar, Int8Kernel::Sdot, Int8Kernel::Vnni];

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> i32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (self.0 >> 16) as i32
        }
        fn act(&mut self) -> [i8; 4] {
            std::array::from_fn(|_| (self.next() % 128) as i8)
        }
        fn weight(&mut self) -> [i8; 4] {
            std::array::from_fn(|_| (self.next() % 255 - 127) as i8)
        }
    }

    /// The plain sum `acc[o] += sum dot(x[i], w[j][o])` over the `(i, j)` pairs.
    fn reference(acc: &mut [i32], x: &[[i8; 4]], pairs: &[(usize, usize)], w: &[[i8; 4]]) {
        let n = acc.len();
        for &(i, j) in pairs {
            for (o, a) in acc.iter_mut().enumerate() {
                let (xg, wg) = (x[i], w[j * n + o]);
                *a += (0..4).map(|t| xg[t] as i32 * wg[t] as i32).sum::<i32>();
            }
        }
    }

    #[test]
    fn run_kernels_agree_exactly() {
        let mut rng = Lcg(12345);
        for n in [16usize, 32, 48, 64, 256] {
            for len in [0usize, 1, 3, 8, 71] {
                let x: Vec<[i8; 4]> = (0..90).map(|_| rng.act()).collect();
                let w: Vec<[i8; 4]> = (0..100 * n).map(|_| rng.weight()).collect();
                // Up to three runs, as a convolution's tap rows.
                let runs: Vec<Run> = (0..len % 3 + 1)
                    .map(|_| Run {
                        x0: (rng.next() % 19) as usize,
                        w0: (rng.next() % 29) as usize,
                        len,
                    })
                    .collect();
                let pairs: Vec<(usize, usize)> = runs
                    .iter()
                    .flat_map(|r| (0..r.len).map(|j| (r.x0 + j, r.w0 + j)))
                    .collect();
                let init: Vec<i32> = (0..n).map(|_| rng.next() % 1000 - 500).collect();
                let mut want = init.clone();
                reference(&mut want, &x, &pairs, &w);
                for k in KERNELS.into_iter().filter(|k| k.supported()) {
                    let mut got = init.clone();
                    dot_runs(k, &mut got, &x, &runs, &w);
                    assert_eq!(got, want, "{k:?} n {n} len {len}");
                }
            }
        }
    }

    #[test]
    fn row_kernels_agree_exactly() {
        let mut rng = Lcg(777);
        for n in [16usize, 64] {
            let groups = 40;
            let w: Vec<[i8; 4]> = (0..groups * n).map(|_| rng.weight()).collect();
            for rows in 1..=7usize {
                let x: Vec<[i8; 4]> = (0..rows * groups)
                    .map(|i| {
                        if i % 5 == 2 || (i % groups) % 7 == 0 {
                            [0; 4]
                        } else {
                            rng.act()
                        }
                    })
                    .collect();
                let all: Vec<(usize, usize)> = (0..groups).map(|g| (g, g)).collect();
                let mut want = vec![3i32; rows * n];
                for r in 0..rows {
                    reference(
                        &mut want[r * n..][..n],
                        &x[r * groups..][..groups],
                        &all,
                        &w,
                    );
                }
                for k in KERNELS.into_iter().filter(|k| k.supported()) {
                    let mut got = vec![3i32; rows * n];
                    dot_rows(k, &mut got, n, &x, &w);
                    assert_eq!(got, want, "{k:?} n {n} rows {rows}");
                }
            }
        }
    }

    #[test]
    fn grouping_and_rounding() {
        let w: Vec<f32> = (0..8 * 3).map(|i| i as f32 - 10.0).collect();
        let (q, s) = quantize_columns(&w, 3);
        assert_eq!(q[0], (-10.0 / s[0]).round() as i8);
        assert!(q.iter().all(|&v| v >= -127));
        let g = group_rows(&q, 3);
        assert_eq!(g.len(), 2 * 3);
        assert_eq!(g[3 + 1], [q[13], q[16], q[19], q[22]]);
        let (sc, inv) = act_scale(2.0);
        assert_eq!(sc, 2.0 / 127.0);
        assert_eq!([0.0, 1.0, 2.0, 0.5].map(|v| act(v, inv)), [0, 64, 127, 32]);
        assert_eq!(act_scale(0.0), (1.0, 1.0));
    }
}
