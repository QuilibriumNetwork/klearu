//! GroupNorm — normalises per (group of channels, batch). SD uses 32 groups
//! everywhere with ε=1e-5.
//!
//! Input: NCHW flattened. For each (n, group), compute mean/var across the
//! `(channels_per_group × H × W)` elements; normalise; scale by γ_c + shift by β_c.

pub struct GroupNorm {
    pub gamma: Vec<f32>, // [num_channels]
    pub beta: Vec<f32>,  // [num_channels]
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize, eps: f32) -> Self {
        assert!(num_channels.is_multiple_of(num_groups),
            "num_channels {num_channels} must be divisible by num_groups {num_groups}");
        Self {
            gamma: vec![1.0; num_channels],
            beta: vec![0.0; num_channels],
            num_groups,
            num_channels,
            eps,
        }
    }

    /// In-place: x has layout [N, C, H, W] flattened.
    /// One-pass mean+var via E[X²] − E[X]² (faster than two-pass mean-then-var
    /// at the cost of slight numerical instability, fine for our value ranges).
    /// Groups are independent so we parallelise across them with rayon.
    pub fn forward_inplace(&self, x: &mut [f32], n: usize, h: usize, w: usize) {
        use rayon::prelude::*;

        let c = self.num_channels;

        // Metal path for any non-trivial input. UNet/VAE GroupNorms operate
        // on 100K+ elements which amortise dispatch easily.
        #[cfg(feature = "metal")]
        {
            if x.len() >= 1 << 14 {
                crate::metal_backend::groupnorm_metal(
                    x, &self.gamma, &self.beta,
                    n, c, h, w, self.num_groups, self.eps,
                );
                return;
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = n;

        let cg = c / self.num_groups;
        let hw = h * w;
        let group_size = cg * hw;
        let num_groups = self.num_groups;
        let eps = self.eps;
        let gamma = &self.gamma;
        let beta = &self.beta;

        // (n × num_groups) independent groups; chunk the buffer accordingly.
        // Each chunk is cg*hw contiguous floats. Use par_chunks_mut for safe
        // disjoint mutation.
        x.par_chunks_mut(group_size).enumerate().for_each(|(idx, group_slice)| {
            let g = idx % num_groups;
            // Fused one-pass mean and sum-of-squares.
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &v in group_slice.iter() {
                sum += v;
                sum_sq += v * v;
            }
            let inv_gs = 1.0 / group_size as f32;
            let mean = sum * inv_gs;
            let var = (sum_sq * inv_gs - mean * mean).max(0.0);
            let inv_std = 1.0 / (var + eps).sqrt();

            for ci in 0..cg {
                let channel_idx = g * cg + ci;
                let scale = gamma[channel_idx];
                let shift = beta[channel_idx];
                let chan_off = ci * hw;
                for j in 0..hw {
                    let v = group_slice[chan_off + j];
                    group_slice[chan_off + j] = (v - mean) * inv_std * scale + shift;
                }
            }
        });
        let _ = c; // silence unused warning when refactoring further
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gn_normalises_each_group() {
        let gn = GroupNorm::new(2, 4, 1e-5);
        // [N=1, C=4, H=2, W=2] = 16 floats
        let mut x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        gn.forward_inplace(&mut x, 1, 2, 2);
        // Group 0 = channels [0,1] = elements 0..8.
        // After normalisation each group has mean 0, var 1.
        let g0: f32 = x[..8].iter().sum::<f32>() / 8.0;
        let g1: f32 = x[8..].iter().sum::<f32>() / 8.0;
        assert!(g0.abs() < 1e-4);
        assert!(g1.abs() < 1e-4);
    }
}
