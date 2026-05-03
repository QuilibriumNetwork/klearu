//! Element-wise activations: SiLU, quick_gelu, GELU.
//!
//! For tensors above the dispatch threshold we route to Metal kernels.
//! Below the threshold we run a small inline scalar loop (no rayon) — the
//! sub-threshold path is only hit for time-embeddings (~1280 elements).

#[cfg(feature = "metal")]
const METAL_ACT_THRESHOLD: usize = 1 << 14;

/// SiLU(x) = x · σ(x). Used everywhere in SD.
#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    #[cfg(feature = "metal")]
    if x.len() >= METAL_ACT_THRESHOLD {
        crate::metal_backend::silu_metal(x);
        return;
    }
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// CLIP's quick_gelu(x) = x · σ(1.702 · x). Cheaper than exact GELU.
/// CLIPTextModel uses this for the MLP activation (CLIP-L; CLIP-G uses true GELU).
#[inline]
pub fn quick_gelu_inplace(x: &mut [f32]) {
    #[cfg(feature = "metal")]
    if x.len() >= METAL_ACT_THRESHOLD {
        crate::metal_backend::quick_gelu_metal(x);
        return;
    }
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-1.702 * *v).exp());
    }
}

/// True GELU(x) = 0.5 x · (1 + erf(x/√2)). CLIP-G's MLP uses this; we use a
/// tanh-based approximation that's standard in transformer libraries.
#[inline]
pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(feature = "metal")]
    if x.len() >= METAL_ACT_THRESHOLD {
        crate::metal_backend::gelu_metal(x);
        return;
    }
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let x = *v;
        *v = 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_basic() {
        let mut x = vec![0.0, 1.0, -1.0, 2.0];
        silu_inplace(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6);                       // 0 * σ(0) = 0
        assert!((x[1] - 0.7310586).abs() < 1e-5);                 // 1 * σ(1)
        assert!((x[2] - (-0.26894143)).abs() < 1e-5);             // -1 * σ(-1)
    }

    #[test]
    fn quick_gelu_basic() {
        let mut x = vec![0.0, 1.0];
        quick_gelu_inplace(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6);
        // 1 · σ(1.702) ≈ 0.84579
        assert!((x[1] - 0.84579).abs() < 1e-3);
    }

    #[test]
    fn gelu_basic() {
        let mut x = vec![0.0, 1.0];
        gelu_inplace(&mut x);
        assert!(x[0].abs() < 1e-6);
        // GELU(1) ≈ 0.8413
        assert!((x[1] - 0.8413).abs() < 1e-3);
    }
}
