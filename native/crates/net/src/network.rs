//! The network forward pass, config-driven and in plain `f32`.
//!
//! Ported from `az_modes.network` (`baselines/az_modes.py:198-206`), `unified.conv_features`
//! (`src/slinky/unified.py:431-445`), `unified.torso` (`unified.py:461-470`) and the SAME-padding
//! convolution `unified.conv3x3` (`unified.py:338-348`, mirroring XLA). See `SPEC.md` section 2.
//!
//! # How it is fast
//!
//! The activations are sparse: the 21x21 egocentric window is mostly zero padding around the
//! 11x11 board, the board planes are near one-hot, and after each ReLU most channels are zero
//! (about 61%, 79% and 63% after the three convolutions of the M net on real states). So every
//! layer works on a compressed list of the non-zero activations of its input ([`Sparse`]): for
//! each pixel the `(channel, value)` pairs that are not zero. A layer gathers, for one output
//! pixel, the non-zero inputs of its in-bounds 3x3 taps and accumulates `value * weight row`
//! into a register-resident accumulator (a fixed-size array the compiler keeps in vector
//! registers: `fmla` by element on aarch64), then adds the bias, applies the ReLU and appends
//! the output's non-zero channels to the next list, fused, so no layer materialises a dense
//! activation. The dense hidden layer does the same over the non-zero entries of the flattened
//! torso, streaming each needed weight row once, in groups of [`HG`] rows per accumulator pass.
//!
//! On top of that, the padding is propagated: a pixel whose whole receptive field (through all
//! earlier layers) is zero input has exactly the value it has for an all-zero board, which is
//! computed once at load time ([`Conv::zero`], [`Network::hidden_zero`]). Each layer tracks which
//! of its pixels depend on non-zero input ([`Sparse::live`]); the others are copied, not
//! computed. On real states (M net) that skips about 45% of the second convolution's
//! multiply-adds and 20% of the dense layer's.
//!
//! The search feeds structured boards ([`BoardInput`], [`Network::forward_boards_into`]): of
//! the ~400 non-zero inputs of a board, ~360 are three planes that are constant over the 121
//! board cells (`on_board`, own health, own length). Their contribution to a first-layer output
//! pixel depends only on which of its 3x3 taps fall on the board, one of 64 row-tap x
//! column-tap classes, so it is precomputed per class at load time (`board_base`, per unit
//! value) and the first layer gathers only the ~40 content entries. The rest of the pass is
//! the dense one's.
//!
//! The gather kernels accumulate in [`V4`] lanes (NEON / SSE2 / plain arrays), `K = cout / 4`
//! vectors per output pixel, each lane rounding exactly as the scalar [`madd`].
//!
//! Only multiply-adds by an exact zero activation, and recomputations of all-zero-input pixels,
//! are skipped, so the result is exact for any input up to float rounding (summation order, and
//! fused multiply-adds on aarch64): it agrees with XLA to about 1e-6, not bitwise.

use crate::config::{
    FeatureMode, NUM_ACTIONS, NUM_FEATURES, NetConfig, OBS_CHANNELS, OBS_COLS, OBS_LEN, OBS_ROWS,
};
use crate::error::NetError;
use crate::npy::{Array, parse_npz};
use crate::simd::V4;
use std::collections::HashMap;
use std::path::Path;

/// `unified._NO_MODE_KEEP`: features kept for `features="no_mode"` (`turn`, `opponents_alive`).
const NO_MODE_KEEP: [f32; NUM_FEATURES] = [0., 0., 0., 0., 1., 0., 1.];

/// Floats of accumulator a gather loop may keep in vector registers: 16 x 128-bit registers on
/// x86-64 (SSE2 baseline) leave room for 8 accumulators, NEON's 32 for 16.
#[cfg(target_arch = "aarch64")]
const ACC_FLOATS: usize = 64;
#[cfg(not(target_arch = "aarch64"))]
const ACC_FLOATS: usize = 32;

/// Hidden-layer column tile (two accumulator chains of `HT` floats).
const HT: usize = ACC_FLOATS / 2;

/// Entries of the dense layer whose weight rows are streamed together: each `[hidden]` row is
/// read once, contiguously, and the accumulators are loaded and stored once per group.
const HG: usize = 8;

/// Boards whose convolutions run back to back before their dense layers, which keeps each
/// layer's weights in cache across the boards of a batch.
const MAX_ROWS: usize = 8;

/// `c + a * b`, fused where the hardware has a vector FMA (`fmla` on aarch64), so it costs one
/// instruction; plain multiply-add elsewhere (a non-native `fma` would be a slow libm call).
#[inline(always)]
fn madd(c: f32, a: f32, b: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        c + a * b
    }
}

struct Conv {
    cin: usize,
    cout: usize,
    stride: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    /// Zeros added before the first row / column (`pad_total / 2`).
    pad_lo: usize,
    /// HWIO, flattened `[dy][dx][cin][cout]`: row `(3 * dy + dx) * cin + c` holds the `cout`
    /// weights of input channel `c` at tap `(dy, dx)`.
    w: Vec<f32>,
    b: Vec<f32>,
    /// This layer's output for an all-zero board (every pixel computed), in the same
    /// compressed form [`Conv::apply`] writes.
    zero: Sparse,
}

/// The in-bounds taps `lo..hi` along one axis of output coordinate `o`, and the input
/// coordinate tap `lo` reads.
#[inline(always)]
fn tap_range(o: usize, stride: usize, pad: usize, len: usize) -> (usize, usize, usize) {
    let first = o * stride;
    let lo = pad.saturating_sub(first);
    let hi = (len + pad).saturating_sub(first).min(3);
    (lo, hi.max(lo), (first + lo).wrapping_sub(pad))
}

impl Conv {
    fn out_pixels(&self) -> usize {
        self.out_h * self.out_w
    }

    /// `relu(conv(x) + b)` of one board, from and to compressed form ([`Sparse`]). Output
    /// indices are `ox * cout + channel`, or for the `last` layer the flat
    /// `(oy * out_w + ox) * cout + channel` index of the flattened torso; the `last` layer
    /// leaves out the pixels that are not live (the dense layer adds their precomputed part).
    fn apply(&self, x: &Sparse, out: &mut Sparse, last: bool, acc: &mut [f32]) {
        match self.cout {
            8 => self.apply_k::<2>(x, out, last),
            16 => self.apply_k::<4>(x, out, last),
            32 => self.apply_k::<8>(x, out, last),
            64 => self.apply_k::<16>(x, out, last),
            _ => self.apply_dyn(x, out, last, acc),
        }
    }

    /// Calls `f(row_offset, idx, val)` for every in-bounds kernel row of output pixel
    /// `(oy, ox)`, with the entries of the consecutive input pixels its taps read: entry `idx`
    /// multiplies weight row `row_offset + idx` (wrapping).
    #[inline(always)]
    fn for_taps(&self, x: &Sparse, oy: usize, ox: usize, mut f: impl FnMut(usize, &[u32], &[f32])) {
        let (dy0, dy1, iy0) = tap_range(oy, self.stride, self.pad_lo, self.in_h);
        let (dx0, dx1, ix0) = tap_range(ox, self.stride, self.pad_lo, self.in_w);
        if dx0 == dx1 {
            return;
        }
        // Entry index ix * cin + c of tap dx reads weight row (3 dy + dx) cin + c, and
        // dx = ix - ix0 + dx0, so the row is (3 dy + dx0 - ix0) cin + idx.
        let shift = (dx0 * self.cin).wrapping_sub(ix0 * self.cin);
        for (dy, iy) in (dy0..dy1).zip(iy0..) {
            let p = iy * self.in_w + ix0;
            let (s, e) = (x.start[p] as usize, x.start[p + dx1 - dx0] as usize);
            f(
                (3 * dy * self.cin).wrapping_add(shift),
                &x.idx[s..e],
                &x.val[s..e],
            );
        }
    }

    /// The pixel loop shared by the kernels: `pixel(oy, ox, base, idx, val)` computes one live
    /// output pixel into `idx` / `val` (room for `cout` entries, indices offset by `base`) and
    /// returns how many it wrote. Pixels that read no live input get the all-zero-board output.
    /// `K` is `cout / 4` when it is a compile-time constant (so that copy is inline), else 0.
    #[inline(always)]
    fn drive<const K: usize>(
        &self,
        x: &Sparse,
        out: &mut Sparse,
        last: bool,
        mut pixel: impl FnMut(usize, usize, usize, &mut [u32], &mut [f32]) -> usize,
    ) {
        let co = self.cout;
        let mut n = 0;
        out.start[0] = 0;
        for oy in 0..self.out_h {
            let (dy0, dy1, iy0) = tap_range(oy, self.stride, self.pad_lo, self.in_h);
            let rows_live = if dy0 < dy1 {
                x.live[iy0..iy0 + dy1 - dy0].iter().fold(0, |m, &r| m | r)
            } else {
                0
            };
            let mut live = 0u32;
            for ox in 0..self.out_w {
                let (dx0, dx1, ix0) = tap_range(ox, self.stride, self.pad_lo, self.in_w);
                let window = ((1u32 << (dx1 - dx0)) - 1).wrapping_shl(ix0 as u32);
                let po = oy * self.out_w + ox;
                if rows_live & window != 0 {
                    live |= 1 << ox;
                    let base = if last { po * co } else { ox * co };
                    n += pixel(
                        oy,
                        ox,
                        base,
                        &mut out.idx[n..n + co],
                        &mut out.val[n..n + co],
                    );
                } else if !last {
                    let z = &self.zero;
                    let (s, e) = (z.start[po] as usize, z.start[po + 1] as usize);
                    // A pixel has at most `cout` entries, and the zero list has room for `cout`
                    // after each pixel's start, so a fixed-length copy is in bounds.
                    let len = if K > 0 { 4 * K } else { e - s };
                    out.idx[n..n + len].copy_from_slice(&z.idx[s..s + len]);
                    out.val[n..n + len].copy_from_slice(&z.val[s..s + len]);
                    n += e - s;
                }
                out.start[po + 1] = n as u32;
            }
            out.live[oy] = live;
        }
    }

    /// [`Conv::apply`] for `cout = 4 * K`, accumulating in registers.
    fn apply_k<const K: usize>(&self, x: &Sparse, out: &mut Sparse, last: bool) {
        self.drive::<K>(x, out, last, |oy, ox, base, oi, ov| {
            let y = self.pixel::<K>(x, oy, ox);
            compact(y.as_flattened(), base, oi, ov)
        });
    }

    /// `relu(conv + b)` at output pixel `(oy, ox)` for `cout = 4 * K`. Not inlined, so the
    /// accumulators get the registers to themselves.
    #[inline(never)]
    fn pixel<const K: usize>(&self, x: &Sparse, oy: usize, ox: usize) -> [[f32; 4]; K] {
        let rows = self.w.as_chunks::<4>().0.as_chunks::<K>().0;
        let bias = &self.b.as_chunks::<4>().0[..K];
        let mut a = [[V4::zero(); K]; 2];
        self.for_taps(x, oy, ox, |off, idx, val| {
            gather(&mut a, rows, off, idx, val)
        });
        std::array::from_fn(|k| relu4(a[0][k].add(a[1][k]).add(V4::load(&bias[k]))))
    }

    /// [`Conv::apply`] of the first convolution on a structured board ([`BoardInput`]): `x`
    /// holds the board's listed entries only, and the three whole-board planes come from
    /// `base` (see [`Network::board_base`]). `None` when `cout` has no register kernel (the
    /// caller then expands the board).
    fn apply_board(
        &self,
        x: &Sparse,
        board: &BoardConsts,
        base: &[f32],
        out: &mut Sparse,
        last: bool,
    ) -> Option<()> {
        match self.cout {
            8 => self.apply_board_k::<2>(x, board, base, out, last),
            16 => self.apply_board_k::<4>(x, board, base, out, last),
            32 => self.apply_board_k::<8>(x, board, base, out, last),
            64 => self.apply_board_k::<16>(x, board, base, out, last),
            _ => return None,
        }
        Some(())
    }

    fn apply_board_k<const K: usize>(
        &self,
        x: &Sparse,
        board: &BoardConsts,
        base: &[f32],
        out: &mut Sparse,
        last: bool,
    ) {
        self.drive::<K>(x, out, last, |oy, ox, b, oi, ov| {
            let y = self.pixel_board::<K>(x, board, base, oy, ox);
            compact(y.as_flattened(), b, oi, ov)
        });
    }

    /// [`Conv::pixel`] on a structured board: the listed entries gathered as usual, plus the
    /// whole-board planes' share for this pixel's on-board taps.
    #[inline(never)]
    fn pixel_board<const K: usize>(
        &self,
        x: &Sparse,
        board: &BoardConsts,
        base: &[f32],
        oy: usize,
        ox: usize,
    ) -> [[f32; 4]; K] {
        let rows = self.w.as_chunks::<4>().0.as_chunks::<K>().0;
        let bias = &self.b.as_chunks::<4>().0[..K];
        let mut a = [[V4::zero(); K]; 2];
        self.for_taps(x, oy, ox, |off, idx, val| {
            gather(&mut a, rows, off, idx, val)
        });
        let mask = |o: usize, lo: usize| {
            (0..3).fold(0, |m, d| {
                let i = (o * self.stride + d).wrapping_sub(self.pad_lo);
                m | (usize::from(i.wrapping_sub(lo) < BOARD) << d)
            })
        };
        let class = mask(oy, board.r0) * 8 + mask(ox, board.c0);
        let planes = base[class * 3 * 4 * K..][..3 * 4 * K].as_chunks::<4>().0;
        std::array::from_fn(|k| {
            let whole = V4::load(&planes[k])
                .fma(V4::load(&planes[K + k]), board.health)
                .fma(V4::load(&planes[2 * K + k]), board.length);
            relu4(a[0][k].add(a[1][k]).add(whole).add(V4::load(&bias[k])))
        })
    }

    /// [`Conv::apply`] for any `cout`, accumulating in memory.
    fn apply_dyn(&self, x: &Sparse, out: &mut Sparse, last: bool, acc: &mut [f32]) {
        let co = self.cout;
        let acc = &mut acc[..co];
        self.drive::<0>(x, out, last, |oy, ox, base, oi, ov| {
            acc.fill(0.0);
            self.for_taps(x, oy, ox, |off, idx, val| {
                for (&i, &v) in idx.iter().zip(val) {
                    let row = &self.w[off.wrapping_add(i as usize) * co..][..co];
                    for (o, &w) in acc.iter_mut().zip(row) {
                        *o = madd(*o, v, w);
                    }
                }
            });
            let mut m = 0;
            for (j, (&a, &b)) in acc.iter().zip(&self.b).enumerate() {
                let y = (a + b).max(0.0);
                if y != 0.0 {
                    oi[m] = (base + j) as u32;
                    ov[m] = y;
                    m += 1;
                }
            }
            m
        });
    }
}

/// `relu` of four lanes, as the scalar `x.max(0.0)` (a NaN becomes 0).
#[inline(always)]
fn relu4(x: V4) -> [f32; 4] {
    x.to_array().map(|v| v.max(0.0))
}

/// Append the non-zero values of `y` with indices `base + j` to `oi` / `ov`; returns how many.
#[inline(always)]
fn compact(y: &[f32], base: usize, oi: &mut [u32], ov: &mut [f32]) -> usize {
    let (oi, ov) = (&mut oi[..y.len()], &mut ov[..y.len()]);
    let mut m = 0;
    for (j, &y) in y.iter().enumerate() {
        oi[m] = (base + j) as u32;
        ov[m] = y;
        m += (y != 0.0) as usize;
    }
    m
}

/// `a += sum_k val[k] * rows[off + idx[k]]` (wrapping offset), in registers: with two
/// independent accumulator chains (even and odd `k`) when both fit, so the multiply-adds are not
/// latency bound.
#[inline(always)]
fn gather<const K: usize>(
    a: &mut [[V4; K]; 2],
    rows: &[[[f32; 4]; K]],
    off: usize,
    idx: &[u32],
    val: &[f32],
) {
    // Work on local copies: accumulators behind a `&mut` would be stored back on every step
    // (a bounds-check panic could observe them).
    let row = |i: u32| &rows[off.wrapping_add(i as usize)];
    if 8 * K <= ACC_FLOATS {
        let [mut a0, mut a1] = *a;
        let (ip, ir) = idx.as_chunks::<2>();
        let (vp, vr) = val.as_chunks::<2>();
        for (i, v) in ip.iter().zip(vp) {
            let (r0, r1) = (row(i[0]), row(i[1]));
            for k in 0..K {
                a0[k] = a0[k].fma(V4::load(&r0[k]), v[0]);
                a1[k] = a1[k].fma(V4::load(&r1[k]), v[1]);
            }
        }
        if let (Some(&i), Some(&v)) = (ir.first(), vr.first()) {
            let r = row(i);
            for k in 0..K {
                a0[k] = a0[k].fma(V4::load(&r[k]), v);
            }
        }
        *a = [a0, a1];
    } else {
        let mut a0 = a[0];
        for (&i, &v) in idx.iter().zip(val) {
            let r = row(i);
            for k in 0..K {
                a0[k] = a0[k].fma(V4::load(&r[k]), v);
            }
        }
        a[0] = a0;
    }
}

/// The round-to-nearest-even bfloat16 of `x` (the high half of its bits). NaN stays NaN.
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return (b >> 16) as u16 | 0x40;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A weight type the vector hidden layer reads: `f32`, or bfloat16 bits.
trait Weight: Copy {
    fn v4(x: &[Self; 4]) -> V4;
    fn f32(self) -> f32;
}

impl Weight for f32 {
    #[inline(always)]
    fn v4(x: &[f32; 4]) -> V4 {
        V4::load(x)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        self
    }
}

impl Weight for u16 {
    #[inline(always)]
    fn v4(x: &[u16; 4]) -> V4 {
        V4::load_bf16(x)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        f32::from_bits(u32::from(self) << 16)
    }
}

/// [`hidden_group`] in registers ([`V4`]); the same operations per column.
#[inline(always)]
fn hidden_group_v<T: Weight, const G: usize>(
    w: &[T],
    h: usize,
    i: &[u32; G],
    v: &[f32; G],
    out: &mut [f32],
) {
    const KT: usize = HT / 4;
    let tiles = h / HT;
    let rows: [&[T]; G] = std::array::from_fn(|g| &w[i[g] as usize * h..][..h]);
    let tiled: [&[[[T; 4]; KT]]; G] =
        std::array::from_fn(|g| rows[g].as_chunks::<4>().0.as_chunks::<KT>().0);
    let (ot, orest) = out.as_chunks_mut::<HT>();
    for (t, o) in ot.iter_mut().enumerate() {
        let o4 = o.as_chunks_mut::<4>().0;
        let mut a0: [V4; KT] = std::array::from_fn(|k| V4::load(&o4[k]));
        let mut a1 = [V4::zero(); KT];
        for g in (0..G - 1).step_by(2) {
            let (r0, r1) = (&tiled[g][t], &tiled[g + 1][t]);
            for k in 0..KT {
                a0[k] = a0[k].fma(T::v4(&r0[k]), v[g]);
                a1[k] = a1[k].fma(T::v4(&r1[k]), v[g + 1]);
            }
        }
        if G % 2 == 1 {
            let r = &tiled[G - 1][t];
            for k in 0..KT {
                a0[k] = a0[k].fma(T::v4(&r[k]), v[G - 1]);
            }
        }
        for k in 0..KT {
            a0[k].add(a1[k]).store(&mut o4[k]);
        }
    }
    for (g, r) in rows.iter().enumerate() {
        for (o, &x) in orest.iter_mut().zip(&r[tiles * HT..]) {
            *o = madd(*o, v[g], x.f32());
        }
    }
}

/// `out += sum_g v[g] * w[i[g]]` for `w` = `[flat][h]`, one group of `G` rows. The weights are
/// `f32`, or `i8` for the experimental quantised layer ([`Network::quantize_hidden_int8`]).
#[inline(always)]
fn hidden_group<T: Copy + Into<f32>, const G: usize>(
    w: &[T],
    h: usize,
    i: &[u32; G],
    v: &[f32; G],
    out: &mut [f32],
) {
    let tiles = h / HT;
    let rows: [&[T]; G] = std::array::from_fn(|g| &w[i[g] as usize * h..][..h]);
    let tiled: [&[[T; HT]]; G] = std::array::from_fn(|g| rows[g].as_chunks::<HT>().0);
    let (ot, orest) = out.as_chunks_mut::<HT>();
    for (t, o) in ot.iter_mut().enumerate() {
        // Two accumulator chains (even and odd g), so the multiply-adds are not latency bound.
        let mut a0 = *o;
        let mut a1 = [0f32; HT];
        for g in (0..G - 1).step_by(2) {
            let (r0, r1) = (&tiled[g][t], &tiled[g + 1][t]);
            for j in 0..HT {
                a0[j] = madd(a0[j], v[g], r0[j].into());
                a1[j] = madd(a1[j], v[g + 1], r1[j].into());
            }
        }
        if G % 2 == 1 {
            let r = &tiled[G - 1][t];
            for j in 0..HT {
                a0[j] = madd(a0[j], v[G - 1], r[j].into());
            }
        }
        for j in 0..HT {
            o[j] = a0[j] + a1[j];
        }
    }
    for (g, r) in rows.iter().enumerate() {
        for (o, &x) in orest.iter_mut().zip(&r[tiles * HT..]) {
            *o = madd(*o, v[g], x.into());
        }
    }
}

/// [`hidden_add`] in registers ([`hidden_group_v`]).
fn hidden_add_v<T: Weight>(w: &[T], h: usize, idx: &[u32], val: &[f32], out: &mut [f32]) {
    let (ig, ir) = idx.as_chunks::<HG>();
    let (vg, vr) = val.as_chunks::<HG>();
    for (i, v) in ig.iter().zip(vg) {
        hidden_group_v(w, h, i, v, out);
    }
    let (i4, ir) = ir.as_chunks::<4>();
    let (v4, vr) = vr.as_chunks::<4>();
    if let (Some(i), Some(v)) = (i4.first(), v4.first()) {
        hidden_group_v(w, h, i, v, out);
    }
    let (i2, ir) = ir.as_chunks::<2>();
    let (v2, vr) = vr.as_chunks::<2>();
    if let (Some(i), Some(v)) = (i2.first(), v2.first()) {
        hidden_group_v(w, h, i, v, out);
    }
    if let (Some(&i), Some(&v)) = (ir.first(), vr.first()) {
        hidden_group_v(w, h, &[i], &[v], out);
    }
}

/// `out += sum_k val[k] * w[idx[k]]` for `w` = `[flat][h]` (rows of `h` floats).
fn hidden_add<T: Copy + Into<f32>>(w: &[T], h: usize, idx: &[u32], val: &[f32], out: &mut [f32]) {
    let (ig, ir) = idx.as_chunks::<HG>();
    let (vg, vr) = val.as_chunks::<HG>();
    for (i, v) in ig.iter().zip(vg) {
        hidden_group(w, h, i, v, out);
    }
    // The last < 8 entries in groups of 4, 2 and 1.
    let (i4, ir) = ir.as_chunks::<4>();
    let (v4, vr) = vr.as_chunks::<4>();
    if let (Some(i), Some(v)) = (i4.first(), v4.first()) {
        hidden_group(w, h, i, v, out);
    }
    let (i2, ir) = ir.as_chunks::<2>();
    let (v2, vr) = vr.as_chunks::<2>();
    if let (Some(i), Some(v)) = (i2.first(), v2.first()) {
        hidden_group(w, h, i, v, out);
    }
    if let (Some(&i), Some(&v)) = (ir.first(), vr.first()) {
        hidden_group(w, h, &[i], &[v], out);
    }
}

/// Rows (and columns) of the board inside the 21x21 window.
const BOARD: usize = OBS_ROWS.div_ceil(2);

/// One board's input in structured form: the observation that is 1 in [`CH_ON_BOARD`], `health`
/// in [`CH_OWN_HEALTH`] and `length` in [`CH_OWN_LENGTH`] on every cell of the 11x11 board (rows
/// `r0..r0 + 11`, columns `c0..c0 + 11` of the window), 0 off it, plus `entries`. This is how
/// the egocentric observation is built (`slinky_obs::SparseObs`), and it lets the first
/// convolution add those three planes from a table instead of gathering about 360 inputs.
#[derive(Debug, Clone, Copy)]
pub struct BoardInput<'a> {
    /// Window row of the board's first row.
    pub r0: usize,
    /// Window column of the board's first column.
    pub c0: usize,
    /// The value of [`CH_OWN_HEALTH`] on the board.
    pub health: f32,
    /// The value of [`CH_OWN_LENGTH`] on the board.
    pub length: f32,
    /// `(window pixel, channel, value)` of every other non-zero input, by pixel and then channel,
    /// all on the board and none in the three planes above.
    pub entries: &'a [(u16, u8, f32)],
}

impl BoardInput<'_> {
    /// The dense `[21][21][13]` observation this board stands for.
    pub fn expand_into(&self, out: &mut [f32]) -> Result<(), NetError> {
        if out.len() != OBS_LEN || self.r0 + BOARD > OBS_ROWS || self.c0 + BOARD > OBS_COLS {
            return Err(NetError::Input("a board outside the window".into()));
        }
        out.fill(0.0);
        for y in self.r0..self.r0 + BOARD {
            for x in self.c0..self.c0 + BOARD {
                let px = &mut out[(y * OBS_COLS + x) * OBS_CHANNELS..][..OBS_CHANNELS];
                px[CH_ON_BOARD] = 1.0;
                px[CH_OWN_HEALTH] = self.health;
                px[CH_OWN_LENGTH] = self.length;
            }
        }
        for &(p, c, v) in self.entries {
            let i = p as usize * OBS_CHANNELS + c as usize;
            if (c as usize) >= OBS_CHANNELS || i >= OBS_LEN {
                return Err(NetError::Input(format!(
                    "entry ({p}, {c}) outside the window"
                )));
            }
            out[i] = v;
        }
        Ok(())
    }
}

/// Channel 1 on every board cell of a [`BoardInput`].
pub const CH_ON_BOARD: usize = 0;
/// Channel equal to [`BoardInput::health`] on every board cell.
pub const CH_OWN_HEALTH: usize = 11;
/// Channel equal to [`BoardInput::length`] on every board cell.
pub const CH_OWN_LENGTH: usize = 12;

/// The scalars of a [`BoardInput`] the first convolution needs.
struct BoardConsts {
    r0: usize,
    c0: usize,
    health: f32,
    length: f32,
}

/// The non-zero activations of one board in compressed form: entries
/// `start[p] .. start[p + 1]` of `idx` / `val` are the non-zero channels `c` of pixel
/// `p = y * width + x` and their values, with `idx = x * channels + c` (so the entries of a run of
/// consecutive pixels of one row index consecutive weight rows), or for the last convolution the
/// flat `p * channels + c` index of the flattened torso.
#[derive(Clone)]
struct Sparse {
    start: Vec<u32>,
    idx: Vec<u32>,
    val: Vec<f32>,
    /// Per row, bit `x` set when pixel `x` depends on a non-zero input. A pixel that does not
    /// has its all-zero-board value. (Rows are at most 21 pixels wide.)
    live: Vec<u32>,
}

impl Sparse {
    fn new(h: usize, w: usize, channels: usize) -> Sparse {
        Sparse {
            start: vec![0; h * w + 1],
            idx: vec![0; h * w * channels],
            val: vec![0.0; h * w * channels],
            live: vec![0; h],
        }
    }

    fn fits(&self, h: usize, w: usize, channels: usize) -> bool {
        self.start.len() > h * w && self.idx.len() >= h * w * channels && self.live.len() >= h
    }

    /// Compress one observation (`[21][21][13]`), skipping all-zero rows and pixels quickly.
    fn load_obs(&mut self, board: &[f32]) {
        // Zero test on the bits without the sign, so it vectorises (and -0.0 counts as zero).
        let nonzero = |xs: &[f32]| xs.iter().fold(0u32, |a, &x| a | (x.to_bits() << 1)) != 0;
        let mut n = 0;
        self.start[0] = 0;
        let row_len = OBS_COLS * OBS_CHANNELS;
        for (y, row) in board.chunks_exact(row_len).enumerate() {
            let starts = &mut self.start[y * OBS_COLS + 1..][..OBS_COLS];
            let mut live = 0u32;
            if nonzero(row) {
                let pixels = row.as_chunks::<OBS_CHANNELS>().0;
                for (x, (px, st)) in pixels.iter().zip(starts).enumerate() {
                    if nonzero(px) {
                        live |= 1 << x;
                        let (oi, ov) = (
                            &mut self.idx[n..n + OBS_CHANNELS],
                            &mut self.val[n..n + OBS_CHANNELS],
                        );
                        let mut m = 0;
                        for (c, &a) in px.iter().enumerate() {
                            oi[m] = (x * OBS_CHANNELS + c) as u32;
                            ov[m] = a;
                            m += (a != 0.0) as usize;
                        }
                        n += m;
                    }
                    *st = n as u32;
                }
            } else {
                starts.fill(n as u32);
            }
            self.live[y] = live;
        }
    }

    /// Load a structured board's listed entries; every board pixel is live (the whole-board
    /// planes are not zero there). `None` when an entry is off the board or out of order.
    fn load_board(&mut self, b: &BoardInput) -> Option<()> {
        if b.r0 + BOARD > OBS_ROWS || b.c0 + BOARD > OBS_COLS {
            return None;
        }
        let pixels = OBS_ROWS * OBS_COLS;
        let mut p = 0usize;
        self.start[0] = 0;
        for (n, &(px, c, v)) in b.entries.iter().enumerate() {
            let (px, c) = (px as usize, c as usize);
            let (y, x) = (px / OBS_COLS, px % OBS_COLS);
            if px < p
                || c >= OBS_CHANNELS
                || y.wrapping_sub(b.r0) >= BOARD
                || x.wrapping_sub(b.c0) >= BOARD
                || n >= self.idx.len()
            {
                return None;
            }
            while p < px {
                p += 1;
                self.start[p] = n as u32;
            }
            self.idx[n] = (x * OBS_CHANNELS + c) as u32;
            self.val[n] = v;
        }
        let n = b.entries.len() as u32;
        while p < pixels {
            p += 1;
            self.start[p] = n;
        }
        let cols = ((1u32 << BOARD) - 1) << b.c0;
        for (y, l) in self.live[..OBS_ROWS].iter_mut().enumerate() {
            *l = if y.wrapping_sub(b.r0) < BOARD {
                cols
            } else {
                0
            };
        }
        Some(())
    }

    /// The `(idx, val)` entries of pixels `p0..p1`.
    fn entries(&self, p0: usize, p1: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.start[p0] as usize, self.start[p1] as usize);
        (&self.idx[s..e], &self.val[s..e])
    }
}

/// Source of [`Network::id`].
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A loaded network. Immutable, so one value can serve many threads; use one [`Scratch`] each.
pub struct Network {
    /// Unique per network and per change of its outputs ([`Network::id`]).
    id: u64,
    cfg: NetConfig,
    convs: Vec<Conv>,
    /// The first convolution's share of the three whole-board planes of a [`BoardInput`], per
    /// tap class: `[rows mask * 8 + columns mask][plane][cout]`, where bit `d` of a mask is set
    /// when tap `d` of that axis is on the board, and the planes are [`CH_ON_BOARD`] (times 1),
    /// [`CH_OWN_HEALTH`] and [`CH_OWN_LENGTH`] (times their value).
    board_base: Vec<f32>,
    /// `hidden/w`, `[flat][hidden]`.
    hidden_w: Vec<f32>,
    /// Experimental int8 copy of `hidden/w` and its per-column scales, used instead when set.
    hidden_q: Option<(Vec<i8>, Vec<f32>)>,
    /// bfloat16 copy of `hidden/w` (rounded to nearest even), used instead when set.
    hidden_bf16: Option<Vec<u16>>,
    /// `[last conv pixels][hidden]`: each last-convolution pixel's share of `flat @ W` for an
    /// all-zero board, added for the pixels that are not live.
    hidden_zero: Vec<f32>,
    hidden_b: Vec<f32>,
    feat_w: Option<Vec<f32>>,
    policy_w: Vec<f32>,
    policy_b: [f32; NUM_ACTIONS],
    value_w: Vec<f32>,
    value_b: f32,
}

/// Reusable work buffers for [`Network::forward_into`], so an evaluation allocates nothing.
pub struct Scratch {
    ping: Sparse,
    pong: Sparse,
    /// The last convolution's output of each board of a group of up to [`MAX_ROWS`].
    torso: Vec<Sparse>,
    /// `[MAX_ROWS][hidden]` hidden layers.
    hid: Vec<f32>,
    term: Vec<f32>,
    acc: Vec<f32>,
}

/// Policy logits and value of a batch, as returned by [`Network::forward`].
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Raw policy logits per row, in the order UP, DOWN, LEFT, RIGHT (no masking, no softmax).
    pub logits: Vec<[f32; NUM_ACTIONS]>,
    /// `tanh` value per row, in (-1, 1).
    pub values: Vec<f32>,
}

fn take(m: &mut HashMap<String, Array>, key: &str, shape: &[usize]) -> Result<Vec<f32>, NetError> {
    let a = m
        .remove(key)
        .ok_or_else(|| NetError::Shape(format!("missing array {key}")))?;
    if a.shape != shape {
        return Err(NetError::Shape(format!(
            "{key}: file has shape {:?}, config needs {shape:?}",
            a.shape
        )));
    }
    Ok(a.data)
}

impl Network {
    /// Load `config.json` and `params.npz` from a run dir.
    pub fn load(run_dir: impl AsRef<Path>) -> Result<Network, NetError> {
        let dir = run_dir.as_ref();
        let read = |name: &str| {
            let p = dir.join(name);
            std::fs::read(&p).map_err(|e| NetError::Io(p.display().to_string(), e))
        };
        let cfg = NetConfig::from_json(&String::from_utf8_lossy(&read("config.json")?))?;
        Network::from_npz_bytes(cfg, &read("params.npz")?)
    }

    /// Build from a parsed config and the bytes of a `params.npz`.
    pub fn from_npz_bytes(cfg: NetConfig, npz: &[u8]) -> Result<Network, NetError> {
        Network::from_arrays(cfg, parse_npz(npz)?)
    }

    /// Build from a config and named arrays (`"conv0/w"`, ...), validating every shape.
    ///
    /// The config is validated again ([`NetConfig::validate`]): its fields are public, so it
    /// may not come from [`NetConfig::from_json`].
    pub fn from_arrays(cfg: NetConfig, mut m: HashMap<String, Array>) -> Result<Network, NetError> {
        cfg.validate()?;
        let (mut h, mut w, mut cin) = (OBS_ROWS, OBS_COLS, OBS_CHANNELS);
        let mut convs = Vec::new();
        for (i, (&cout, &stride)) in cfg.conv_channels.iter().zip(&cfg.conv_strides).enumerate() {
            let (oh, ow) = (h.div_ceil(stride), w.div_ceil(stride));
            let pad_total = ((oh - 1) * stride + 3).saturating_sub(h);
            convs.push(Conv {
                cin,
                cout,
                stride,
                in_h: h,
                in_w: w,
                out_h: oh,
                out_w: ow,
                pad_lo: pad_total / 2,
                w: take(&mut m, &format!("conv{i}/w"), &[3, 3, cin, cout])?,
                b: take(&mut m, &format!("conv{i}/b"), &[cout])?,
                zero: Sparse::new(oh, ow, cout),
            });
            (h, w, cin) = (oh, ow, cout);
        }
        let flat = h * w * cin;
        let hid = cfg.hidden;
        let hidden_w = take(&mut m, "hidden/w", &[flat, hid])?;
        let hidden_b = take(&mut m, "hidden/b", &[hid])?;
        let feat_w = match cfg.features {
            FeatureMode::None => None,
            _ => Some(take(&mut m, "feat/w", &[NUM_FEATURES, hid])?),
        };
        let policy_w = take(&mut m, "policy/w", &[hid, NUM_ACTIONS])?;
        let mut policy_b = [0.0; NUM_ACTIONS];
        policy_b.copy_from_slice(&take(&mut m, "policy/b", &[NUM_ACTIONS])?);
        let value_w = take(&mut m, "value/w", &[hid, 1])?;
        let value_b = take(&mut m, "value/b", &[1])?[0];

        // The all-zero board through every layer, with every pixel marked live so all of them
        // are computed by the same kernels the forward pass uses.
        let mut x = Sparse::new(OBS_ROWS, OBS_COLS, 0);
        x.live.fill(u32::MAX);
        let last = convs.len() - 1;
        let mut acc = vec![0.0; cfg.conv_channels.iter().copied().max().unwrap_or(0)];
        for (i, c) in convs.iter_mut().enumerate() {
            let mut out = Sparse::new(c.out_h, c.out_w, c.cout);
            c.apply(&x, &mut out, i == last, &mut acc);
            c.zero = out.clone();
            x = out;
        }
        let pixels = convs[last].out_pixels();
        let mut hidden_zero = vec![0.0; pixels * hid];
        for (p, out) in hidden_zero.chunks_exact_mut(hid).enumerate() {
            let (idx, val) = x.entries(p, p + 1);
            hidden_add_v(&hidden_w[..], hid, idx, val, out);
        }
        let c0 = &convs[0];
        let mut board_base = vec![0f32; 64 * 3 * c0.cout];
        for class in 0..64 {
            let (rm, cm) = (class / 8, class % 8);
            for (k, &ch) in [CH_ON_BOARD, CH_OWN_HEALTH, CH_OWN_LENGTH]
                .iter()
                .enumerate()
            {
                let out = &mut board_base[(class * 3 + k) * c0.cout..][..c0.cout];
                for dy in (0..3).filter(|d| rm >> d & 1 == 1) {
                    for dx in (0..3).filter(|d| cm >> d & 1 == 1) {
                        let row = &c0.w[((3 * dy + dx) * c0.cin + ch) * c0.cout..][..c0.cout];
                        for (o, &w) in out.iter_mut().zip(row) {
                            *o += w;
                        }
                    }
                }
            }
        }
        Ok(Network {
            id: next_id(),
            board_base,
            cfg,
            convs,
            hidden_w,
            hidden_q: None,
            hidden_bf16: None,
            hidden_zero,
            hidden_b,
            feat_w,
            policy_w,
            policy_b,
            value_w,
            value_b,
        })
    }

    /// An id no other network of this process has, renewed whenever this one's outputs change
    /// ([`Network::quantize_hidden_int8`]): equal ids give bit-identical outputs, so a cache of
    /// outputs can be keyed by it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The config this network was built from.
    pub fn config(&self) -> &NetConfig {
        &self.cfg
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        self.convs
            .iter()
            .map(|c| c.w.len() + c.b.len())
            .sum::<usize>()
            + self.hidden_w.len()
            + self.hidden_b.len()
            + self.feat_w.as_ref().map_or(0, Vec::len)
            + self.policy_w.len()
            + NUM_ACTIONS
            + self.value_w.len()
            + 1
    }

    /// **Experimental, off by default.** Quantise the dense hidden layer's weights (93% of the
    /// M net's parameters) to int8 with one symmetric scale per output column, and use them
    /// (converted on the fly; activations stay `f32`) in every later forward pass. This cuts that
    /// layer's weight memory 4x (1.2 MB to 0.3 MB for M, so it fits a 1 MB L2) at the cost of
    /// exactness: outputs move by up to about 1e-1 instead of 1e-6. `on = false` restores `f32`.
    pub fn quantize_hidden_int8(&mut self, on: bool) {
        self.id = next_id();
        self.hidden_bf16 = None;
        if !on {
            self.hidden_q = None;
            return;
        }
        let h = self.cfg.hidden;
        let mut scale = vec![0f32; h];
        for row in self.hidden_w.chunks_exact(h) {
            for (s, &w) in scale.iter_mut().zip(row) {
                *s = s.max(w.abs());
            }
        }
        for s in scale.iter_mut() {
            *s = if *s > 0.0 { *s / 127.0 } else { 1.0 };
        }
        let q = self
            .hidden_w
            .chunks_exact(h)
            .flat_map(|row| row.iter().zip(&scale).map(|(&w, &s)| (w / s).round() as i8))
            .collect();
        self.hidden_q = Some((q, scale));
    }

    /// **Off by default; changes the outputs.** Store the dense hidden layer's weights (93% of
    /// the M net's parameters) as bfloat16, rounded to nearest even, and widen them to `f32` as
    /// they are used (activations and sums stay `f32`). The layer's weights take half the memory
    /// (1.2 MB to 0.6 MB for M: inside a 1 MB L2, as on Graviton2), each weight moves by at most
    /// 2^-9 of itself, and outputs by about 1e-3. `on = false` restores the `f32` weights.
    pub fn store_hidden_bf16(&mut self, on: bool) {
        self.id = next_id();
        self.hidden_q = None;
        self.hidden_bf16 = on.then(|| self.hidden_w.iter().map(|&w| bf16_bits(w)).collect());
    }

    /// Whether the hidden layer uses bfloat16 weights ([`Network::store_hidden_bf16`]).
    pub fn hidden_is_bf16(&self) -> bool {
        self.hidden_bf16.is_some()
    }

    /// Rows, columns and channels of the largest compressed activation (the input or a conv
    /// output) a forward pass holds.
    fn widest(&self) -> (usize, usize, usize) {
        self.convs
            .iter()
            .map(|c| (c.out_h, c.out_w, c.cout))
            .chain([(OBS_ROWS, OBS_COLS, OBS_CHANNELS)])
            .fold((0, 0, 0), |(h, w, c), (i, j, k)| {
                (h.max(i), w.max(j), c.max(k))
            })
    }

    /// Fresh work buffers sized for this network.
    pub fn scratch(&self) -> Scratch {
        let (h, w, c) = self.widest();
        let last = &self.convs[self.convs.len() - 1];
        Scratch {
            ping: Sparse::new(h, w, c),
            pong: Sparse::new(h, w, c),
            torso: (0..MAX_ROWS)
                .map(|_| Sparse::new(last.out_h, last.out_w, last.cout))
                .collect(),
            hid: vec![0.0; MAX_ROWS * self.cfg.hidden],
            term: vec![0.0; self.cfg.hidden],
            acc: vec![0.0; c],
        }
    }

    /// Whether `s` is big enough for this network (it may come from another one).
    fn scratch_fits(&self, s: &Scratch) -> bool {
        let (h, w, c) = self.widest();
        let last = &self.convs[self.convs.len() - 1];
        s.hid.len() == MAX_ROWS * self.cfg.hidden
            && s.term.len() == self.cfg.hidden
            && s.acc.len() >= c
            && s.ping.fits(h, w, c)
            && s.pong.fits(h, w, c)
            && s.torso.len() == MAX_ROWS
            && s.torso
                .iter()
                .all(|t| t.fits(last.out_h, last.out_w, last.cout))
    }

    /// Evaluate a batch. `obs` is `[B][21][21][13]` row-major (`OBS_LEN` floats per row) and
    /// `feats` is `[B][7]`. Allocates; hot loops should use [`Network::forward_into`].
    pub fn forward(&self, obs: &[f32], feats: &[f32]) -> Result<Output, NetError> {
        let b = feats.len() / NUM_FEATURES;
        let mut out = Output {
            logits: vec![[0.0; NUM_ACTIONS]; b],
            values: vec![0.0; b],
        };
        self.forward_into(
            &mut self.scratch(),
            obs,
            feats,
            &mut out.logits,
            &mut out.values,
        )?;
        Ok(out)
    }

    /// Evaluate a batch into caller-provided buffers without allocating. The batch size is
    /// `values.len()`; `logits` must have the same length. Rows are independent.
    pub fn forward_into(
        &self,
        scratch: &mut Scratch,
        obs: &[f32],
        feats: &[f32],
        logits: &mut [[f32; NUM_ACTIONS]],
        values: &mut [f32],
    ) -> Result<(), NetError> {
        let b = values.len();
        if logits.len() != b || obs.len() != b * OBS_LEN || feats.len() != b * NUM_FEATURES {
            return Err(NetError::Input(format!(
                "batch {b}: got {} obs floats (need {}), {} feature floats (need {}), {} logit rows",
                obs.len(),
                b * OBS_LEN,
                feats.len(),
                b * NUM_FEATURES,
                logits.len()
            )));
        }
        if !self.scratch_fits(scratch) {
            *scratch = self.scratch();
        }
        let hdim = self.cfg.hidden;
        for first in (0..b).step_by(MAX_ROWS) {
            let rows = (b - first).min(MAX_ROWS);
            for r in 0..rows {
                self.torso(scratch, &obs[(first + r) * OBS_LEN..][..OBS_LEN], r);
            }
            self.hidden(scratch, rows);
            for r in 0..rows {
                let feat = &feats[(first + r) * NUM_FEATURES..][..NUM_FEATURES];
                let h = &mut scratch.hid[r * hdim..][..hdim];
                self.finish_hidden(h, &mut scratch.term, feat);
                let (lg, v) = self.heads(h);
                logits[first + r] = lg;
                values[first + r] = v;
            }
        }
        Ok(())
    }

    /// [`Network::forward_into`] on structured boards ([`BoardInput`], one per row): the same
    /// outputs up to float rounding (the whole-board planes are added in another order), without
    /// building or scanning the dense 21x21x13 window.
    pub fn forward_boards_into(
        &self,
        scratch: &mut Scratch,
        boards: &[BoardInput],
        feats: &[f32],
        logits: &mut [[f32; NUM_ACTIONS]],
        values: &mut [f32],
    ) -> Result<(), NetError> {
        let b = values.len();
        if logits.len() != b || boards.len() != b || feats.len() != b * NUM_FEATURES {
            return Err(NetError::Input(format!(
                "batch {b}: got {} boards, {} feature floats (need {}), {} logit rows",
                boards.len(),
                feats.len(),
                b * NUM_FEATURES,
                logits.len()
            )));
        }
        if !self.scratch_fits(scratch) {
            *scratch = self.scratch();
        }
        let hdim = self.cfg.hidden;
        for first in (0..b).step_by(MAX_ROWS) {
            let rows = (b - first).min(MAX_ROWS);
            for r in 0..rows {
                if self.torso_board(scratch, &boards[first + r], r).is_none() {
                    let mut dense = vec![0f32; OBS_LEN];
                    boards[first + r].expand_into(&mut dense)?;
                    self.torso(scratch, &dense, r);
                }
            }
            self.hidden(scratch, rows);
            for r in 0..rows {
                let feat = &feats[(first + r) * NUM_FEATURES..][..NUM_FEATURES];
                let h = &mut scratch.hid[r * hdim..][..hdim];
                self.finish_hidden(h, &mut scratch.term, feat);
                let (lg, v) = self.heads(h);
                logits[first + r] = lg;
                values[first + r] = v;
            }
        }
        Ok(())
    }

    /// [`Network::torso`] of a structured board; `None` (nothing done) when the first
    /// convolution has no board kernel or the board is malformed.
    fn torso_board(&self, scratch: &mut Scratch, board: &BoardInput, r: usize) -> Option<()> {
        let Scratch {
            ping,
            pong,
            torso,
            acc,
            ..
        } = scratch;
        ping.load_board(board)?;
        let consts = BoardConsts {
            r0: board.r0,
            c0: board.c0,
            health: board.health,
            length: board.length,
        };
        let last = self.convs.len() - 1;
        let first = &self.convs[0];
        if last == 0 {
            return first.apply_board(ping, &consts, &self.board_base, &mut torso[r], true);
        }
        first.apply_board(ping, &consts, &self.board_base, pong, false)?;
        let (mut cur, mut next) = (pong, ping);
        for (i, conv) in self.convs.iter().enumerate().skip(1) {
            if i == last {
                conv.apply(cur, &mut torso[r], true, acc);
            } else {
                conv.apply(cur, next, false, acc);
                std::mem::swap(&mut cur, &mut next);
            }
        }
        Some(())
    }

    /// The convolutions of one board, leaving the compressed output in `scratch.torso[r]`.
    fn torso(&self, scratch: &mut Scratch, board: &[f32], r: usize) {
        let Scratch {
            ping,
            pong,
            torso,
            acc,
            ..
        } = scratch;
        ping.load_obs(board);
        let (mut cur, mut next) = (ping, pong);
        let last = self.convs.len() - 1;
        for (i, conv) in self.convs.iter().enumerate() {
            if i == last {
                conv.apply(cur, &mut torso[r], true, acc);
            } else {
                conv.apply(cur, next, false, acc);
                std::mem::swap(&mut cur, &mut next);
            }
        }
    }

    /// `flat @ W` of the boards in `scratch.torso[..rows]` into `scratch.hid`: the live
    /// entries times their weight rows, plus the precomputed share of every pixel that is not
    /// live.
    fn hidden(&self, scratch: &mut Scratch, rows: usize) {
        let hdim = self.cfg.hidden;
        let last = &self.convs[self.convs.len() - 1];
        let hid = &mut scratch.hid[..rows * hdim];
        hid.fill(0.0);
        for (t, out) in scratch.torso[..rows].iter().zip(hid.chunks_exact_mut(hdim)) {
            let (idx, val) = t.entries(0, last.out_pixels());
            match &self.hidden_q {
                None if self.hidden_bf16.is_some() => {
                    let w = self.hidden_bf16.as_deref().unwrap_or_default();
                    hidden_add_v(w, hdim, idx, val, out);
                }
                None => hidden_add_v(&self.hidden_w[..], hdim, idx, val, out),
                Some((q, scale)) => {
                    hidden_add(q, hdim, idx, val, out);
                    for (o, &s) in out.iter_mut().zip(scale) {
                        *o *= s;
                    }
                }
            }
            for (oy, &live) in t.live[..last.out_h].iter().enumerate() {
                for ox in (0..last.out_w).filter(|&ox| live & (1 << ox) == 0) {
                    let z = &self.hidden_zero[(oy * last.out_w + ox) * hdim..][..hdim];
                    for (o, &zv) in out.iter_mut().zip(z) {
                        *o += zv;
                    }
                }
            }
        }
    }

    /// `h = relu((h + b) + feats @ F)`, in that order of float additions.
    fn finish_hidden(&self, h: &mut [f32], term: &mut [f32], feat: &[f32]) {
        for (o, &bv) in h.iter_mut().zip(&self.hidden_b) {
            *o += bv;
        }
        if let Some(fw) = &self.feat_w {
            // feats @ F, accumulated in feature order, then added.
            term.fill(0.0);
            for (k, (&fv, frow)) in feat.iter().zip(fw.chunks_exact(h.len())).enumerate() {
                let fv = if self.cfg.features == FeatureMode::NoMode {
                    fv * NO_MODE_KEEP[k]
                } else {
                    fv
                };
                for (t, &wv) in term.iter_mut().zip(frow) {
                    *t += fv * wv;
                }
            }
            for (o, &t) in h.iter_mut().zip(term.iter()) {
                *o += t;
            }
        }
        for o in h.iter_mut() {
            *o = o.max(0.0);
        }
    }

    /// Policy logits and `tanh` value from a hidden layer.
    fn heads(&self, h: &[f32]) -> ([f32; NUM_ACTIONS], f32) {
        let mut lg = self.policy_b;
        for (&hv, wrow) in h.iter().zip(self.policy_w.as_chunks::<NUM_ACTIONS>().0) {
            for (l, &wv) in lg.iter_mut().zip(wrow) {
                *l += hv * wv;
            }
        }
        let v: f32 = h
            .iter()
            .zip(&self.value_w)
            .map(|(&a, &w)| a * w)
            .sum::<f32>()
            + self.value_b;
        (lg, v.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        }
    }

    fn random_net(channels: &[usize], strides: &[usize], hidden: usize, features: &str) -> Network {
        let cfg = NetConfig::from_json(&format!(
            r#"{{"algorithm":"az_modes","conv_channels":{channels:?},"conv_strides":{strides:?},"hidden":{hidden},"features":"{features}"}}"#
        ))
        .unwrap();
        let mut rng = Rng(channels.iter().sum::<usize>() as u64 + hidden as u64);
        let mut m = HashMap::new();
        let mut put = |name: &str, shape: Vec<usize>, scale: f32| {
            let n = shape.iter().product();
            let data = (0..n).map(|_| rng.next() * scale).collect();
            m.insert(name.to_string(), Array { shape, data });
        };
        let (mut h, mut cin) = (OBS_ROWS, OBS_CHANNELS);
        for (i, (&c, &s)) in channels.iter().zip(strides).enumerate() {
            // Some negative biases, so whole channels and the padding's output can be zero.
            put(&format!("conv{i}/w"), vec![3, 3, cin, c], 0.4);
            put(&format!("conv{i}/b"), vec![c], 0.3);
            (h, cin) = (h.div_ceil(s), c);
        }
        put("hidden/w", vec![h * h * cin, hidden], 0.1);
        put("hidden/b", vec![hidden], 0.1);
        put("feat/w", vec![NUM_FEATURES, hidden], 0.3);
        put("policy/w", vec![hidden, NUM_ACTIONS], 0.3);
        put("policy/b", vec![NUM_ACTIONS], 0.1);
        put("value/w", vec![hidden, 1], 0.2);
        put("value/b", vec![1], 0.1);
        Network::from_arrays(cfg, m).unwrap()
    }

    /// The textbook forward pass in f64: dense SAME convolutions on `[h][w][c]` images.
    fn reference(net: &Network, board: &[f32], feat: &[f32]) -> ([f64; NUM_ACTIONS], f64) {
        let mut x: Vec<f64> = board.iter().map(|&v| v as f64).collect();
        for c in &net.convs {
            let mut y = vec![0f64; c.out_pixels() * c.cout];
            for oy in 0..c.out_h {
                for ox in 0..c.out_w {
                    for o in 0..c.cout {
                        let mut a = c.b[o] as f64;
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let iy = (oy * c.stride + dy) as isize - c.pad_lo as isize;
                                let ix = (ox * c.stride + dx) as isize - c.pad_lo as isize;
                                if iy < 0
                                    || ix < 0
                                    || iy >= c.in_h as isize
                                    || ix >= c.in_w as isize
                                {
                                    continue;
                                }
                                for i in 0..c.cin {
                                    let xv = x[(iy as usize * c.in_w + ix as usize) * c.cin + i];
                                    let wv = c.w[((dy * 3 + dx) * c.cin + i) * c.cout + o];
                                    a += xv * wv as f64;
                                }
                            }
                        }
                        y[(oy * c.out_w + ox) * c.cout + o] = a.max(0.0);
                    }
                }
            }
            x = y;
        }
        let hd = net.cfg.hidden;
        let mut h: Vec<f64> = net.hidden_b.iter().map(|&b| b as f64).collect();
        for (&xv, row) in x.iter().zip(net.hidden_w.chunks_exact(hd)) {
            for (hv, &w) in h.iter_mut().zip(row) {
                *hv += xv * w as f64;
            }
        }
        if let Some(fw) = &net.feat_w {
            for k in 0..NUM_FEATURES {
                let keep = if net.cfg.features == FeatureMode::NoMode {
                    NO_MODE_KEEP[k]
                } else {
                    1.0
                };
                for (hv, &w) in h.iter_mut().zip(&fw[k * hd..][..hd]) {
                    *hv += (feat[k] * keep) as f64 * w as f64;
                }
            }
        }
        let mut lg = [0f64; NUM_ACTIONS];
        let mut v = net.value_b as f64;
        for (j, &hv) in h.iter().enumerate() {
            let hv = hv.max(0.0);
            for (a, l) in lg.iter_mut().enumerate() {
                *l += hv * net.policy_w[j * NUM_ACTIONS + a] as f64;
            }
            v += hv * net.value_w[j] as f64;
        }
        for (a, l) in lg.iter_mut().enumerate() {
            *l += net.policy_b[a] as f64;
        }
        (lg, v.tanh())
    }

    /// Boards of every kind: all zero, an 11x11 board somewhere in the padding (sparse planes,
    /// some -0.0), a corner pixel only, and dense random values of both signs.
    fn boards(rng: &mut Rng, n: usize) -> Vec<f32> {
        let mut obs = vec![0f32; n * OBS_LEN];
        for (b, board) in obs.as_chunks_mut::<OBS_LEN>().0.iter_mut().enumerate() {
            match b % 4 {
                0 => {}
                1 => {
                    let (y0, x0) = (b % 11, (3 * b) % 11);
                    for y in y0..y0 + 11 {
                        for x in x0..x0 + 11 {
                            for c in 0..OBS_CHANNELS {
                                let r = rng.next();
                                let v = if c == 0 {
                                    1.0
                                } else if r > 0.6 {
                                    r
                                } else if r < -0.9 {
                                    -0.0
                                } else {
                                    0.0
                                };
                                board[(y * OBS_COLS + x) * OBS_CHANNELS + c] = v;
                            }
                        }
                    }
                }
                2 => board[OBS_LEN - 1] = 0.7,
                _ => board.iter_mut().for_each(|v| *v = rng.next()),
            }
        }
        obs
    }

    #[test]
    fn matches_reference_on_odd_shapes() {
        let nets = [
            random_net(&[5, 12, 7], &[2, 1, 3], 37, "all"),
            random_net(&[8, 16, 32], &[2, 2, 1], 40, "no_mode"),
            random_net(&[64, 6], &[1, 4], 70, "none"),
            random_net(&[16, 32, 32], &[2, 2, 1], 256, "all"),
        ];
        let mut rng = Rng(7);
        // One scratch shared by all the nets (it must be replaced when it does not fit).
        let mut scratch = nets[0].scratch();
        for net in &nets {
            for b in [1, 3, 11] {
                let obs = boards(&mut rng, b);
                let feats: Vec<f32> = (0..b * NUM_FEATURES).map(|_| rng.next()).collect();
                let (mut lg, mut v) = (vec![[0f32; NUM_ACTIONS]; b], vec![0f32; b]);
                net.forward_into(&mut scratch, &obs, &feats, &mut lg, &mut v)
                    .unwrap();
                for r in 0..b {
                    let (wl, wv) = reference(
                        net,
                        &obs[r * OBS_LEN..][..OBS_LEN],
                        &feats[r * NUM_FEATURES..][..NUM_FEATURES],
                    );
                    let scale = wl.iter().fold(1f64, |m, l| m.max(l.abs()));
                    for a in 0..NUM_ACTIONS {
                        let e = (lg[r][a] as f64 - wl[a]).abs() / scale;
                        assert!(
                            e < 1e-5,
                            "logit {a} of row {r}/{b}: {} vs {}",
                            lg[r][a],
                            wl[a]
                        );
                    }
                    assert!((v[r] as f64 - wv).abs() < 1e-5, "value {} vs {wv}", v[r]);
                }
            }
        }
    }

    #[test]
    fn structured_boards_match_the_reference() {
        let nets = [
            random_net(&[5, 12, 7], &[2, 1, 3], 37, "all"),
            random_net(&[8, 16, 32], &[2, 2, 1], 40, "no_mode"),
            random_net(&[64, 6], &[1, 4], 70, "none"),
            random_net(&[16, 32, 32], &[2, 2, 1], 256, "all"),
            random_net(&[16], &[2], 30, "all"),
            random_net(&[32, 8], &[3, 1], 20, "all"),
        ];
        let mut rng = Rng(11);
        let mut scratch = nets[0].scratch();
        for net in &nets {
            for b in [1, 2, 4, 9] {
                let mut lists = Vec::new();
                let mut heads = Vec::new();
                for i in 0..b {
                    let (r0, c0) = ((3 * i + 1) % 11, (5 * i + 2) % 11);
                    let mut e = Vec::new();
                    for y in r0..r0 + BOARD {
                        for x in c0..c0 + BOARD {
                            for c in 1..CH_OWN_HEALTH {
                                let r = rng.next();
                                if r > 0.8 {
                                    e.push(((y * OBS_COLS + x) as u16, c as u8, r));
                                }
                            }
                        }
                    }
                    let (h, l) = if i % 3 == 2 {
                        (0.0, 0.0)
                    } else {
                        (rng.next(), rng.next())
                    };
                    heads.push((r0, c0, h, l));
                    lists.push(e);
                }
                let boards: Vec<BoardInput> = (0..b)
                    .map(|i| BoardInput {
                        r0: heads[i].0,
                        c0: heads[i].1,
                        health: heads[i].2,
                        length: heads[i].3,
                        entries: &lists[i],
                    })
                    .collect();
                let mut obs = vec![0f32; b * OBS_LEN];
                for (i, bd) in boards.iter().enumerate() {
                    bd.expand_into(&mut obs[i * OBS_LEN..][..OBS_LEN]).unwrap();
                }
                let feats: Vec<f32> = (0..b * NUM_FEATURES).map(|_| rng.next()).collect();
                let (mut lg, mut v) = (vec![[0f32; NUM_ACTIONS]; b], vec![0f32; b]);
                net.forward_boards_into(&mut scratch, &boards, &feats, &mut lg, &mut v)
                    .unwrap();
                for r in 0..b {
                    let (wl, wv) = reference(
                        net,
                        &obs[r * OBS_LEN..][..OBS_LEN],
                        &feats[r * NUM_FEATURES..][..NUM_FEATURES],
                    );
                    let scale = wl.iter().fold(1f64, |m, l| m.max(l.abs()));
                    for a in 0..NUM_ACTIONS {
                        let e = (lg[r][a] as f64 - wl[a]).abs() / scale;
                        assert!(e < 1e-5, "logit {a} row {r}/{b}: {} vs {}", lg[r][a], wl[a]);
                    }
                    assert!((v[r] as f64 - wv).abs() < 1e-5, "value {} vs {wv}", v[r]);
                }
            }
        }
        // A malformed board is an error or falls back, never a panic.
        let bad = [(0u16, 1u8, 1.0f32)];
        let b = BoardInput {
            r0: 5,
            c0: 5,
            health: 0.5,
            length: 0.1,
            entries: &bad,
        };
        let (mut lg, mut v) = ([[0f32; NUM_ACTIONS]; 1], [0f32; 1]);
        let _ =
            nets[3].forward_boards_into(&mut scratch, &[b], &[0.0; NUM_FEATURES], &mut lg, &mut v);
    }

    #[test]
    fn int8_is_close_and_reversible() {
        let mut net = random_net(&[8, 16, 16], &[2, 2, 1], 64, "all");
        let mut rng = Rng(3);
        let obs = boards(&mut rng, 8);
        let feats: Vec<f32> = (0..8 * NUM_FEATURES).map(|_| rng.next()).collect();
        let want = net.forward(&obs, &feats).unwrap();
        net.quantize_hidden_int8(true);
        let got = net.forward(&obs, &feats).unwrap();
        assert_ne!(got, want);
        for (g, w) in got.logits.iter().zip(&want.logits) {
            for (a, b) in g.iter().zip(w) {
                assert!((a - b).abs() < 0.05 * b.abs().max(1.0), "{a} vs {b}");
            }
        }
        net.quantize_hidden_int8(false);
        assert_eq!(net.forward(&obs, &feats).unwrap(), want);
    }

    #[test]
    fn bad_inputs_are_errors() {
        let net = random_net(&[8], &[2], 16, "all");
        let mut s = net.scratch();
        let (mut lg, mut v) = (vec![[0f32; NUM_ACTIONS]; 2], vec![0f32; 2]);
        let obs = vec![0f32; 2 * OBS_LEN];
        let feats = vec![0f32; 2 * NUM_FEATURES];
        assert!(
            net.forward_into(&mut s, &obs[1..], &feats, &mut lg, &mut v)
                .is_err()
        );
        assert!(
            net.forward_into(&mut s, &obs, &feats[1..], &mut lg, &mut v)
                .is_err()
        );
        assert!(
            net.forward_into(&mut s, &obs, &feats, &mut lg[1..], &mut v)
                .is_err()
        );
        assert!(
            net.forward_into(&mut s, &obs, &feats, &mut lg, &mut v)
                .is_ok()
        );
    }
}
