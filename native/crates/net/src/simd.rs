//! Four `f32` lanes: NEON on aarch64, SSE on x86-64, plain arrays elsewhere.
//!
//! The network's kernels keep their accumulators in registers as `[V4; K]`. Written as `f32`
//! arrays, LLVM leaves those loops scalar (it does not vectorise the loop-carried arrays), so the
//! lanes are spelled out here. Every lane does exactly what the scalar code did, in the same
//! order: [`V4::fma`] is a fused multiply-add where the scalar `madd` was fused (aarch64, or x86
//! built with the `fma` feature) and a multiply then an add elsewhere, so the results are
//! bit-identical to the scalar kernels.
//!
//! With `quant`'s int8 kernels, the only `unsafe` code of the crate: loads and stores through
//! references to four lanes, which are always valid.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

/// Four `f32` lanes.
#[derive(Clone, Copy)]
pub(crate) struct V4(
    #[cfg(target_arch = "aarch64")] arch::float32x4_t,
    #[cfg(target_arch = "x86_64")] arch::__m128,
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))] [f32; 4],
);

impl V4 {
    /// All zero.
    #[inline(always)]
    pub(crate) fn zero() -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vdupq_n_f32(0.0) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE is part of the x86-64 baseline.
        return V4(unsafe { arch::_mm_setzero_ps() });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4([0.0; 4]);
    }

    /// The four floats of `x`.
    #[inline(always)]
    pub(crate) fn load(x: &[f32; 4]) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `x` is four readable floats.
        return V4(unsafe { arch::vld1q_f32(x.as_ptr()) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `x` is four readable floats (unaligned load).
        return V4(unsafe { arch::_mm_loadu_ps(x.as_ptr()) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(*x);
    }

    /// Write the lanes to `out`.
    #[inline(always)]
    pub(crate) fn store(self, out: &mut [f32; 4]) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `out` is four writable floats.
        unsafe {
            arch::vst1q_f32(out.as_mut_ptr(), self.0)
        };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `out` is four writable floats (unaligned store).
        unsafe {
            arch::_mm_storeu_ps(out.as_mut_ptr(), self.0)
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            *out = self.0;
        }
    }

    /// Four bfloat16 values (the high halves of `f32` bit patterns), widened exactly.
    #[inline(always)]
    pub(crate) fn load_bf16(x: &[u16; 4]) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `x` is four readable u16; NEON is part of the aarch64 baseline.
        return V4(unsafe {
            arch::vreinterpretq_f32_u32(arch::vshll_n_u16::<16>(arch::vld1_u16(x.as_ptr())))
        });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `x` is eight readable bytes; SSE2 is part of the x86-64 baseline.
        return V4(unsafe {
            let h = arch::_mm_loadl_epi64(x.as_ptr().cast());
            arch::_mm_castsi128_ps(arch::_mm_unpacklo_epi16(arch::_mm_setzero_si128(), h))
        });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(x.map(|b| f32::from_bits(u32::from(b) << 16)));
    }

    /// The lanes as an array.
    #[inline(always)]
    pub(crate) fn to_array(self) -> [f32; 4] {
        let mut a = [0.0; 4];
        self.store(&mut a);
        a
    }

    /// `self + a * s` per lane, as the scalar `madd(self, s, a)`.
    #[inline(always)]
    pub(crate) fn fma(self, a: V4, s: f32) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vfmaq_n_f32(self.0, a.0, s) });
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        // SAFETY: the `fma` feature is enabled at compile time.
        return V4(unsafe { arch::_mm_fmadd_ps(a.0, arch::_mm_set1_ps(s), self.0) });
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        // SAFETY: SSE is part of the x86-64 baseline.
        return V4(unsafe {
            arch::_mm_add_ps(self.0, arch::_mm_mul_ps(a.0, arch::_mm_set1_ps(s)))
        });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(std::array::from_fn(|j| self.0[j] + a.0[j] * s));
    }

    /// `self + b` per lane.
    #[inline(always)]
    pub(crate) fn add(self, b: V4) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vaddq_f32(self.0, b.0) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE is part of the x86-64 baseline.
        return V4(unsafe { arch::_mm_add_ps(self.0, b.0) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(std::array::from_fn(|j| self.0[j] + b.0[j]));
    }

    /// Four `i32` converted to `f32` (rounded to nearest, as `v as f32`).
    #[inline(always)]
    pub(crate) fn from_i32(x: &[i32; 4]) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `x` is four readable i32; NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vcvtq_f32_s32(arch::vld1q_s32(x.as_ptr())) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `x` is four readable i32 (unaligned load); SSE2 is part of the baseline.
        return V4(unsafe { arch::_mm_cvtepi32_ps(arch::_mm_loadu_si128(x.as_ptr().cast())) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(x.map(|v| v as f32));
    }

    /// `self * s + b` per lane, rounded after the multiply and after the add (never fused).
    #[inline(always)]
    pub(crate) fn mul_add_unfused(self, s: V4, b: V4) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vaddq_f32(arch::vmulq_f32(self.0, s.0), b.0) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE is part of the x86-64 baseline.
        return V4(unsafe { arch::_mm_add_ps(arch::_mm_mul_ps(self.0, s.0), b.0) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(std::array::from_fn(|j| self.0[j] * s.0[j] + b.0[j]));
    }

    /// `x.max(0.0)` per lane (a NaN becomes 0).
    #[inline(always)]
    pub(crate) fn relu(self) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline (`fmaxnm`: the number when one is NaN).
        return V4(unsafe { arch::vmaxnmq_f32(self.0, arch::vdupq_n_f32(0.0)) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE is part of the x86-64 baseline (`maxps` gives the second operand, 0, when
        // the first is NaN).
        return V4(unsafe { arch::_mm_max_ps(self.0, arch::_mm_setzero_ps()) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(self.0.map(|v| v.max(0.0)));
    }

    /// The larger of each lane pair, for lanes that are not NaN.
    #[inline(always)]
    pub(crate) fn max(self, b: V4) -> V4 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return V4(unsafe { arch::vmaxnmq_f32(self.0, b.0) });
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE is part of the x86-64 baseline.
        return V4(unsafe { arch::_mm_max_ps(self.0, b.0) });
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return V4(std::array::from_fn(|j| self.0[j].max(b.0[j])));
    }

    /// `((x * inv + 0.5) as i32).min(127) as i8` per lane, for lanes with `0 <= x * inv <=
    /// 127` (activations at their own scale): round half up into `0..=127`.
    #[inline(always)]
    #[allow(clippy::needless_return)] // one `return` per target, as in the other methods
    pub(crate) fn quantize(self, inv: f32) -> [i8; 4] {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline; the narrowing saturates.
        unsafe {
            let v = arch::vaddq_f32(arch::vmulq_n_f32(self.0, inv), arch::vdupq_n_f32(0.5));
            let w = arch::vqmovn_s16(arch::vcombine_s16(
                arch::vqmovn_s32(arch::vcvtq_s32_f32(v)),
                arch::vdup_n_s16(0),
            ));
            let b = arch::vget_lane_u32::<0>(arch::vreinterpret_u32_s8(w));
            return b.to_ne_bytes().map(|x| x as i8);
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is part of the x86-64 baseline; the packs saturate.
        unsafe {
            let v = arch::_mm_add_ps(
                arch::_mm_mul_ps(self.0, arch::_mm_set1_ps(inv)),
                arch::_mm_set1_ps(0.5),
            );
            let i = arch::_mm_cvttps_epi32(v);
            let z = arch::_mm_setzero_si128();
            let b = arch::_mm_packs_epi16(arch::_mm_packs_epi32(i, z), z);
            return (arch::_mm_cvtsi128_si32(b) as u32)
                .to_ne_bytes()
                .map(|x| x as i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return self.0.map(|v| ((v * inv + 0.5) as i32).min(127) as i8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_match_scalar() {
        let (a, b, c) = (
            [1.5f32, -2.25, 3e-8, 7.0],
            [0.1f32, 0.2, -0.3, 1e30],
            0.7f32,
        );
        let fused = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
        let got = V4::load(&a).fma(V4::load(&b), c).to_array();
        for j in 0..4 {
            let want = if fused {
                b[j].mul_add(c, a[j])
            } else {
                a[j] + b[j] * c
            };
            assert_eq!(got[j].to_bits(), want.to_bits());
        }
        let h = [0x3f80u16, 0xc040, 0x0001, 0x7f7f];
        let w = V4::load_bf16(&h).to_array();
        for j in 0..4 {
            assert_eq!(w[j].to_bits(), u32::from(h[j]) << 16);
        }
        let s = V4::load(&a).add(V4::load(&b)).add(V4::zero()).to_array();
        for j in 0..4 {
            assert_eq!(s[j], a[j] + b[j]);
        }
        let i = [7i32, -3, 16_777_217, 0];
        let d = V4::from_i32(&i)
            .mul_add_unfused(V4::load(&[c; 4]), V4::load(&b))
            .to_array();
        let r = V4::load(&a).relu().max(V4::load(&[0.5; 4])).to_array();
        for j in 0..4 {
            assert_eq!(d[j].to_bits(), (i[j] as f32 * c + b[j]).to_bits());
            assert_eq!(r[j], a[j].max(0.0).max(0.5));
        }
        let x = [0.0f32, 0.49, 63.5, 127.0];
        let q = V4::load(&x).quantize(1.0);
        assert_eq!(q, x.map(|v| ((v + 0.5) as i32).min(127) as i8));
        assert_eq!(q, [0, 0, 64, 127]);
    }
}
