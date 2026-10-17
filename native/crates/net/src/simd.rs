//! Four `f32` lanes: NEON on aarch64, SSE on x86-64, plain arrays elsewhere.
//!
//! The network's kernels keep their accumulators in registers as `[V4; K]`. Written as `f32`
//! arrays, LLVM leaves those loops scalar (it does not vectorise the loop-carried arrays), so the
//! lanes are spelled out here. Every lane does exactly what the scalar code did, in the same
//! order: [`V4::fma`] is a fused multiply-add where the scalar `madd` was fused (aarch64, or x86
//! built with the `fma` feature) and a multiply then an add elsewhere, so the results are
//! bit-identical to the scalar kernels.
//!
//! The only `unsafe` code of the crate: loads and stores through references to four floats,
//! which are always valid.

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
    }
}
