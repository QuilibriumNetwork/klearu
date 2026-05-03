//! Single-precision GEMM (SGEMM) wrapper.
//!
//! Default backend: `matrixmultiply` crate (portable, NEON on M1, AVX2 on x86).
//! With `accelerate` feature on macOS: `cblas_sgemm` from Apple's Accelerate
//! framework, which uses the AMX matrix coprocessor on M-series for ~2-3× over
//! pure NEON on large matmuls.
//!
//! Convention: row-major. Computes
//!
//!   C[m × n] = α · A[m × k] · B[k × n] + β · C[m × n]

#[cfg(feature = "accelerate")]
mod accelerate_backend {
    // We declare the cblas_sgemm symbol manually so we don't take a `cblas` dep.
    // Accelerate.framework links it via the `accelerate-src` build script.
    extern crate accelerate_src;

    #[repr(i32)]
    #[allow(non_camel_case_types, dead_code)]
    enum Order { RowMajor = 101, ColMajor = 102 }

    #[repr(i32)]
    #[allow(non_camel_case_types, dead_code)]
    enum Trans { NoTrans = 111, Trans = 112, ConjTrans = 113 }

    extern "C" {
        fn cblas_sgemm(
            order: i32, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: f32,
            a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32,
            c: *mut f32, ldc: i32,
        );
    }

    pub fn sgemm_row_major(
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: &[f32], lda: usize,
        b: &[f32], ldb: usize,
        beta: f32,
        c: &mut [f32], ldc: usize,
    ) {
        unsafe {
            cblas_sgemm(
                Order::RowMajor as i32,
                Trans::NoTrans as i32,
                Trans::NoTrans as i32,
                m as i32, n as i32, k as i32,
                alpha,
                a.as_ptr(), lda as i32,
                b.as_ptr(), ldb as i32,
                beta,
                c.as_mut_ptr(), ldc as i32,
            );
        }
    }
}

#[cfg(not(feature = "accelerate"))]
mod matrixmultiply_backend {
    pub fn sgemm_row_major(
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: &[f32], lda: usize,
        b: &[f32], ldb: usize,
        beta: f32,
        c: &mut [f32], ldc: usize,
    ) {
        // matrixmultiply uses (rsa, csa) row/col stride pairs. For row-major:
        // A is m×k, row stride = lda, col stride = 1.
        // B is k×n, row stride = ldb, col stride = 1.
        // C is m×n, row stride = ldc, col stride = 1.
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                alpha,
                a.as_ptr(), lda as isize, 1,
                b.as_ptr(), ldb as isize, 1,
                beta,
                c.as_mut_ptr(), ldc as isize, 1,
            );
        }
    }
}

// Backend dispatch: metal > accelerate > matrixmultiply.
//
// At compile time we pick exactly one. `metal` and `accelerate` can be
// combined; in that case Metal wins for sgemm_row_major and Accelerate is
// available as a fallback (callers can opt in via crate::blas::sgemm_*
// directly if they want CPU-side matmul on a Metal-enabled build).
#[cfg(all(feature = "metal", not(feature = "accelerate")))]
pub fn sgemm_row_major(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &[f32], _lda: usize,
    b: &[f32], _ldb: usize,
    beta: f32,
    c: &mut [f32], _ldc: usize,
) {
    // Metal kernel only supports alpha=1, beta=0 (the only pattern our pipeline uses).
    if alpha != 1.0 || beta != 0.0 {
        // Fall back to matrixmultiply for non-default scaling.
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, alpha,
                a.as_ptr(), _lda as isize, 1,
                b.as_ptr(), _ldb as isize, 1,
                beta, c.as_mut_ptr(), _ldc as isize, 1,
            );
        }
        return;
    }
    crate::metal_backend::sgemm_metal(m, n, k, a, b, c);
}

#[cfg(all(feature = "accelerate", not(feature = "metal")))]
pub use accelerate_backend::sgemm_row_major;
#[cfg(all(feature = "accelerate", feature = "metal"))]
pub fn sgemm_row_major(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SDXL VAE convs hit `sgemm_row_major` and produce activations that grow
    // into the thousands by the upper decoder stages — past fp16's 65504
    // limit, causing overflow → NaN. Always use Accelerate fp32 here.
    //
    // The UNet GPU-residence hot path uses `sgemm_f16_buf*` variants directly
    // (which take pre-uploaded GPU buffers, not f32 slices), so they aren't
    // affected by this. Set `KLEARU_FORCE_MPS_FP16_SYNC=1` to opt back into
    // the sync MPS path for benchmarking.
    if std::env::var_os("KLEARU_FORCE_MPS_FP16_SYNC").is_some()
        && std::env::var_os("KLEARU_DISABLE_MPS_FP16").is_none()
        && alpha == 1.0 && beta == 0.0 && (m * n * k) >= (1 << 28)
        && lda == k && ldb == n && ldc == n
    {
        crate::metal_backend::sgemm_metal_f16(m, n, k, a, b, c);
        return;
    }
    accelerate_backend::sgemm_row_major(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#[cfg(not(any(feature = "accelerate", feature = "metal")))]
pub use matrixmultiply_backend::sgemm_row_major;

/// C = A · Bᵀ (a common pattern: linear-layer forward where W is [out, in]).
/// A is [m × k], B is [n × k] (so we treat it as transposed). Output [m × n].
pub fn sgemm_a_btrans(
    m: usize, n: usize, k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    // SDXL VAE attention's `to_q/to_k/to_v` Linear layers hit this path with
    // values that overflow fp16 in the decoder. Always use Accelerate fp32.
    // Hot path (UNet GPU residence) uses `sgemm_f16_a_btrans_buf` directly
    // (GPU-buffer interface) and isn't affected. Set
    // `KLEARU_FORCE_MPS_FP16_SYNC=1` to re-enable the sync MPS path.
    #[cfg(all(feature = "metal", feature = "accelerate"))]
    if std::env::var_os("KLEARU_FORCE_MPS_FP16_SYNC").is_some()
        && std::env::var_os("KLEARU_DISABLE_MPS_FP16").is_none()
        && (m * n * k) >= (1 << 28)
    {
        crate::metal_backend::sgemm_metal_f16_a_btrans(m, n, k, a, b, c);
        return;
    }

    // C = A · Bᵀ : equivalent to row-major sgemm with B implicitly transposed.
    // Since we don't expose the trans flag through our wrapper in the
    // matrixmultiply path, we use the col-stride trick: re-index B as a
    // k×n matrix by treating its row-stride as 1 and col-stride as k.
    #[cfg(not(feature = "accelerate"))]
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,        // A row-major [m,k]
            b.as_ptr(), 1, k as isize,        // Bᵀ: original [n,k] viewed as [k,n] col-major
            0.0,
            c.as_mut_ptr(), n as isize, 1,    // C row-major [m,n]
        );
    }
    #[cfg(feature = "accelerate")]
    {
        // For Accelerate, set transb=Trans and pass B as-is.
        use std::os::raw::c_int;
        const ROW_MAJOR: c_int = 101;
        const NO_TRANS: c_int = 111;
        const TRANS: c_int = 112;
        extern "C" {
            fn cblas_sgemm(
                order: c_int, transa: c_int, transb: c_int,
                m: c_int, n: c_int, k: c_int,
                alpha: f32,
                a: *const f32, lda: c_int,
                b: *const f32, ldb: c_int,
                beta: f32,
                c: *mut f32, ldc: c_int,
            );
        }
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS,
                m as c_int, n as c_int, k as c_int,
                1.0,
                a.as_ptr(), k as c_int,
                b.as_ptr(), k as c_int,
                0.0,
                c.as_mut_ptr(), n as c_int,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_identity() {
        // C = I · I = I
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        sgemm_row_major(2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 0.0).abs() < 1e-6);
        assert!((c[2] - 0.0).abs() < 1e-6);
        assert!((c[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sgemm_a_btrans_basic() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]; C = A · Bᵀ = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        // = [[17, 23], [39, 53]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_a_btrans(2, 2, 2, &a, &b, &mut c);
        assert!((c[0] - 17.0).abs() < 1e-5);
        assert!((c[1] - 23.0).abs() < 1e-5);
        assert!((c[2] - 39.0).abs() < 1e-5);
        assert!((c[3] - 53.0).abs() < 1e-5);
    }

    /// MPS fp16 sgemm with B transposed (the path used by Linear.forward_batch).
    #[cfg(all(feature = "metal", feature = "accelerate"))]
    #[test]
    fn sgemm_metal_f16_a_btrans_matches_reference() {
        let m = 64; let n = 32; let k = 48;
        let a: Vec<f32> = (0..m*k).map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5).collect();
        let b: Vec<f32> = (0..n*k).map(|i| ((i * 23) % 19) as f32 / 19.0 - 0.5).collect();

        let mut c_ref = vec![0.0f32; m * n];
        sgemm_a_btrans(m, n, k, &a, &b, &mut c_ref);

        let mut c_f16 = vec![0.0f32; m * n];
        crate::metal_backend::sgemm_metal_f16_a_btrans(m, n, k, &a, &b, &mut c_f16);

        let mut max_diff = 0.0f32;
        for (got, want) in c_f16.iter().zip(c_ref.iter()) {
            let diff = (got - want).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(max_diff < 0.1,
            "fp16 sgemm_a_btrans diverges from fp32 reference: max_diff={max_diff}");
    }

    /// MPS fp16 sgemm should match the f32 reference within fp16 quantisation
    /// error. Tolerance reflects ~3.5 decimal digits of fp16 mantissa precision.
    #[cfg(feature = "metal")]
    #[test]
    fn sgemm_metal_f16_matches_reference() {
        // 32×32 · 32×32 matmul with deterministic random values in [-1, 1].
        let m = 32; let n = 32; let k = 32;
        let a: Vec<f32> = (0..m*k).map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i * 23) % 19) as f32 / 19.0 - 0.5).collect();

        // Reference (Accelerate or matrixmultiply, fp32 accumulator).
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_row_major(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c_ref, n);

        // fp16 MPS path.
        let mut c_f16 = vec![0.0f32; m * n];
        crate::metal_backend::sgemm_metal_f16(m, n, k, &a, &b, &mut c_f16);

        let mut max_diff = 0.0f32;
        for (got, want) in c_f16.iter().zip(c_ref.iter()) {
            let diff = (got - want).abs();
            if diff > max_diff { max_diff = diff; }
        }
        // fp16 quantization at this scale (values bounded by ±32) gives ~0.03
        // absolute error after k=32 accumulations; allow 0.1 as headroom.
        assert!(max_diff < 0.1,
            "fp16 sgemm diverges from fp32 reference: max_diff={max_diff}");
    }
}
