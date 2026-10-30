//! The tensor inventory a Wan DiT checkpoint must supply.
//!
//! Generated from config alone, so it can be checked against a checkpoint
//! before any weights are read — and, more usefully, asserted against the real
//! checkpoint's tensor count as a structural test. Wan2.2-TI2V-5B's
//! `transformer/` index lists **825** tensors across five shards.
//!
//! Naming notes, all from the reference module tree:
//!   * `to_out` is an `nn.ModuleList` whose linear is index **0** (index 1 is
//!     dropout), so `attn1.to_out.0.weight`.
//!   * `ffn` is a `FeedForward` whose `net` is `[GELU(proj), Dropout, Linear]`,
//!     giving `ffn.net.0.proj.*` and `ffn.net.2.*` — note the gap at index 1.
//!   * `norm1` and `norm3` have no affine parameters and contribute nothing;
//!     only `norm2` does, because `cross_attn_norm` is true.
//!   * `scale_shift_table` is a bare `nn.Parameter`, both per block and once
//!     at model level — no `.weight` suffix.

use super::config::WanDitConfig;

fn linear(out: &mut Vec<String>, p: &str) {
    out.push(format!("{p}.weight"));
    out.push(format!("{p}.bias"));
}

/// Names contributed by one attention module.
fn attention(out: &mut Vec<String>, p: &str, added_kv: bool) {
    // RMSNorm over the whole inner_dim, applied before the head split.
    out.push(format!("{p}.norm_q.weight"));
    out.push(format!("{p}.norm_k.weight"));
    linear(out, &format!("{p}.to_q"));
    linear(out, &format!("{p}.to_k"));
    linear(out, &format!("{p}.to_v"));
    linear(out, &format!("{p}.to_out.0"));
    if added_kv {
        linear(out, &format!("{p}.add_k_proj"));
        linear(out, &format!("{p}.add_v_proj"));
        out.push(format!("{p}.norm_added_k.weight"));
    }
}

/// Every tensor name the model needs, in a stable order.
pub fn tensor_names(cfg: &WanDitConfig) -> Vec<String> {
    let mut out = Vec::new();
    linear(&mut out, "patch_embedding");

    // Conditioning: sinusoidal timestep -> MLP -> six modulation vectors,
    // plus a two-layer projection of the text encoder's hidden states.
    linear(&mut out, "condition_embedder.time_embedder.linear_1");
    linear(&mut out, "condition_embedder.time_embedder.linear_2");
    linear(&mut out, "condition_embedder.time_proj");
    linear(&mut out, "condition_embedder.text_embedder.linear_1");
    linear(&mut out, "condition_embedder.text_embedder.linear_2");

    for i in 0..cfg.num_layers {
        let b = format!("blocks.{i}");
        out.push(format!("{b}.scale_shift_table"));
        attention(&mut out, &format!("{b}.attn1"), false);
        attention(&mut out, &format!("{b}.attn2"), cfg.added_kv_proj_dim.is_some());
        if cfg.cross_attn_norm {
            linear(&mut out, &format!("{b}.norm2"));
        }
        linear(&mut out, &format!("{b}.ffn.net.0.proj"));
        linear(&mut out, &format!("{b}.ffn.net.2"));
    }

    out.push("scale_shift_table".to_string());
    linear(&mut out, "proj_out");
    out
}

/// Expected element count for each named tensor, so a checkpoint can be
/// validated without loading it.
pub fn tensor_numel(cfg: &WanDitConfig, name: &str) -> Option<usize> {
    let d = cfg.inner_dim();
    let (pt, ph, pw) = cfg.patch();
    let strip = |s: &str| -> Option<String> {
        // blocks.<n>.rest -> rest
        let r = s.strip_prefix("blocks.")?;
        let (_, rest) = r.split_once('.')?;
        Some(rest.to_string())
    };

    if let Some(rest) = strip(name) {
        return Some(match rest.as_str() {
            "scale_shift_table" => 6 * d,
            "norm2.weight" | "norm2.bias" => d,
            s if s.ends_with("norm_q.weight") || s.ends_with("norm_k.weight") => d,
            s if s.ends_with("to_q.weight") || s.ends_with("to_k.weight")
                || s.ends_with("to_v.weight") || s.ends_with("to_out.0.weight") => d * d,
            s if s.ends_with("to_q.bias") || s.ends_with("to_k.bias")
                || s.ends_with("to_v.bias") || s.ends_with("to_out.0.bias") => d,
            "ffn.net.0.proj.weight" => cfg.ffn_dim * d,
            "ffn.net.0.proj.bias" => cfg.ffn_dim,
            "ffn.net.2.weight" => d * cfg.ffn_dim,
            "ffn.net.2.bias" => d,
            _ => return None,
        });
    }

    Some(match name {
        // Conv3d patch embed: [dim, in_channels, p_t, p_h, p_w].
        "patch_embedding.weight" => d * cfg.in_channels * pt * ph * pw,
        "patch_embedding.bias" => d,
        "condition_embedder.time_embedder.linear_1.weight" => d * cfg.freq_dim,
        "condition_embedder.time_embedder.linear_1.bias" => d,
        "condition_embedder.time_embedder.linear_2.weight" => d * d,
        "condition_embedder.time_embedder.linear_2.bias" => d,
        "condition_embedder.time_proj.weight" => cfg.time_proj_dim() * d,
        "condition_embedder.time_proj.bias" => cfg.time_proj_dim(),
        "condition_embedder.text_embedder.linear_1.weight" => d * cfg.text_dim,
        "condition_embedder.text_embedder.linear_1.bias" => d,
        "condition_embedder.text_embedder.linear_2.weight" => d * d,
        "condition_embedder.text_embedder.linear_2.bias" => d,
        "scale_shift_table" => 2 * d,
        "proj_out.weight" => pt * ph * pw * cfg.out_channels * d,
        "proj_out.bias" => pt * ph * pw * cfg.out_channels,
        _ => return None,
    })
}
