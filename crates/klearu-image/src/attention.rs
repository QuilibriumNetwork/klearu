//! Dense training SDPA. BLAS performs the contractions; softmax stays on CPU.
use crate::model::ImageTransformerConfig;
use klearu_diffusion::blas::{sgemm_a_btrans, sgemm_row_major};

fn head(x: &[f32], n: usize, d: usize, dh: usize, hi: usize, out: &mut [f32]) {
    for t in 0..n {
        out[t * dh..(t + 1) * dh].copy_from_slice(&x[t * d + hi * dh..t * d + (hi + 1) * dh]);
    }
}
fn scatter(x: &[f32], n: usize, d: usize, dh: usize, hi: usize, out: &mut [f32]) {
    for t in 0..n {
        out[t * d + hi * dh..t * d + (hi + 1) * dh].copy_from_slice(&x[t * dh..(t + 1) * dh]);
    }
}
fn transpose(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = x[i * cols + j];
        }
    }
}
fn mm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    sgemm_row_major(m, n, k, 1.0, a, k, b, n, 0.0, out, n);
}

pub(crate) fn forward(
    cfg: &ImageTransformerConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let dh = cfg.head_dim();
    let mut weights = vec![0.0; cfg.num_heads * n * n];
    let mut out = vec![0.0; n * d];
    let (mut qh, mut kh, mut vh, mut oh) = (
        vec![0.0; n * dh],
        vec![0.0; n * dh],
        vec![0.0; n * dh],
        vec![0.0; n * dh],
    );
    for hi in 0..cfg.num_heads {
        head(q, n, d, dh, hi, &mut qh);
        head(k, n, d, dh, hi, &mut kh);
        head(v, n, d, dh, hi, &mut vh);
        let aw = &mut weights[hi * n * n..(hi + 1) * n * n];
        sgemm_a_btrans(n, n, dh, &qh, &kh, aw);
        for i in 0..n {
            let row = &mut aw[i * n..(i + 1) * n];
            for x in &mut row[..=i] {
                *x /= (dh as f32).sqrt();
            }
            let max = row[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in &mut row[..=i] {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in &mut row[..=i] {
                *x /= sum;
            }
            row[i + 1..].fill(0.0);
        }
        mm(n, dh, n, aw, &vh, &mut oh);
        scatter(&oh, n, d, dh, hi, &mut out);
    }
    (weights, out)
}

pub(crate) fn backward(
    cfg: &ImageTransformerConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    weights: &[f32],
    grad_out: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let dh = cfg.head_dim();
    let (mut gq, mut gk, mut gv) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
    let (mut qh, mut kh, mut vh, mut gh, mut result) = (
        vec![0.0; n * dh],
        vec![0.0; n * dh],
        vec![0.0; n * dh],
        vec![0.0; n * dh],
        vec![0.0; n * dh],
    );
    let (mut scores_grad, mut transposed) = (vec![0.0; n * n], vec![0.0; n * n]);
    for hi in 0..cfg.num_heads {
        head(q, n, d, dh, hi, &mut qh);
        head(k, n, d, dh, hi, &mut kh);
        head(v, n, d, dh, hi, &mut vh);
        head(grad_out, n, d, dh, hi, &mut gh);
        let aw = &weights[hi * n * n..(hi + 1) * n * n];
        transpose(aw, n, n, &mut transposed);
        mm(n, dh, n, &transposed, &gh, &mut result);
        scatter(&result, n, d, dh, hi, &mut gv);
        sgemm_a_btrans(n, n, dh, &gh, &vh, &mut scores_grad);
        for i in 0..n {
            let row = &mut scores_grad[i * n..(i + 1) * n];
            let a = &aw[i * n..(i + 1) * n];
            let dot: f32 = (0..=i).map(|j| row[j] * a[j]).sum();
            for j in 0..=i {
                row[j] = a[j] * (row[j] - dot) / (dh as f32).sqrt();
            }
            row[i + 1..].fill(0.0);
        }
        mm(n, dh, n, &scores_grad, &kh, &mut result);
        scatter(&result, n, d, dh, hi, &mut gq);
        transpose(&scores_grad, n, n, &mut transposed);
        mm(n, dh, n, &transposed, &qh, &mut result);
        scatter(&result, n, d, dh, hi, &mut gk);
    }
    (gq, gk, gv)
}
