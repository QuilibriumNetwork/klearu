//! Model diagnostic tool: validates model loading and plaintext forward pass.
//!
//! Usage: diagnose <model_dir>
//!
//! This tool helps verify that a model loads correctly and produces
//! reasonable output BEFORE testing it with MPC. If the plaintext model
//! doesn't work, the MPC model won't either.

use std::path::PathBuf;

use klearu_llm::generate::pipeline::detect_eos_tokens;
use klearu_llm::generate::sampler::{SamplerConfig, sample};
use klearu_llm::model::block::AttentionLayer;
use klearu_llm::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [--prompt <text>]", args[0]);
        eprintln!();
        eprintln!("Diagnoses model loading and plaintext inference.");
        eprintln!("Run this BEFORE testing with MPC to verify the base model works.");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let mut prompt = "Hello, how are you?".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // 1. Load config
    eprintln!("=== Model Diagnostics ===\n");
    eprintln!("Model directory: {}", model_dir.display());

    let config_path = model_dir.join("config.json");
    let config = klearu_llm::config::LlmConfig::from_file(&config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to load config.json: {e}");
        std::process::exit(1);
    });

    eprintln!("\n--- Config ---");
    eprintln!("  vocab_size:       {}", config.vocab_size);
    eprintln!("  hidden_size:      {}", config.hidden_size);
    eprintln!("  num_heads:        {}", config.num_heads);
    eprintln!("  num_kv_heads:     {}", config.num_kv_heads);
    eprintln!("  head_dim:         {}", config.head_dim);
    eprintln!("  intermediate_size:{}", config.intermediate_size);
    eprintln!("  num_layers:       {}", config.num_layers);
    eprintln!("  max_seq_len:      {}", config.max_seq_len);
    eprintln!("  rope_theta:       {}", config.rope_theta);
    eprintln!("  rms_norm_eps:     {}", config.rms_norm_eps);
    eprintln!("  tie_word_embeddings: {}", config.tie_word_embeddings);
    eprintln!("  GQA group_size:   {}", config.gqa_group_size());
    if config.is_qwen35() {
        eprintln!("  model_type:       Qwen3.5");
        eprintln!("  attn_output_gate: {}", config.attn_output_gate);
        eprintln!("  linear_heads:     {}", config.linear_num_key_heads);
        eprintln!("  linear_key_dim:   {}", config.linear_key_head_dim);
        eprintln!("  linear_value_dim: {}", config.linear_value_head_dim);
        eprintln!("  partial_rotary:   {}", config.partial_rotary_factor());
        let linear_count = (0..config.num_layers)
            .filter(|&i| config.layer_type(i) == klearu_llm::config::LayerType::LinearAttention)
            .count();
        eprintln!("  linear_attn layers: {}/{}", linear_count, config.num_layers);
    }

    // 2. Detect EOS tokens
    let eos = detect_eos_tokens(&model_dir);
    eprintln!("\n--- EOS Tokens ---");
    eprintln!("  Detected: {:?}", eos);

    // 3. Load model
    eprintln!("\n--- Loading Model ---");
    let mut model = klearu_llm::weight::load_model(&model_dir).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to load model: {e}");
        std::process::exit(1);
    });
    eprintln!("  Model loaded successfully.");

    // 4. Check weight statistics
    eprintln!("\n--- Weight Statistics ---");

    // Embedding
    let emb_weights = model.embedding.weights.as_raw_slice();
    let emb_nonzero = emb_weights.iter().filter(|&&v| v != 0.0).count();
    let emb_total = emb_weights.len();
    eprintln!("  embedding: {}/{} non-zero ({:.1}%)", emb_nonzero, emb_total, 100.0 * emb_nonzero as f64 / emb_total as f64);

    // Final norm
    let norm_nonzero = model.final_norm.weight.iter().filter(|&&v| v != 0.0).count();
    eprintln!("  final_norm: {}/{} non-zero", norm_nonzero, model.final_norm.weight.len());

    // Per-layer stats
    let mut total_layer_nonzero = 0usize;
    let mut total_layer_weights = 0usize;
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let mut layer_nonzero = 0usize;
        let mut layer_total = 0usize;

        // Attention norms
        layer_nonzero += layer.attn_norm.weight.iter().filter(|&&v| v != 0.0).count();
        layer_total += layer.attn_norm.weight.len();
        layer_nonzero += layer.mlp_norm.weight.iter().filter(|&&v| v != 0.0).count();
        layer_total += layer.mlp_norm.weight.len();

        // Attention projections (handle both layer types)
        match &layer.attention {
            AttentionLayer::Standard(attn) => {
                for proj in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj] {
                    let w = proj.weights.as_raw_slice();
                    layer_nonzero += w.iter().filter(|&&v| v != 0.0).count();
                    layer_total += w.len();
                }
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                for proj in [&dn.in_proj_qkv, &dn.in_proj_z, &dn.in_proj_a,
                             &dn.in_proj_b, &dn.out_proj] {
                    let w = proj.weights.as_raw_slice();
                    layer_nonzero += w.iter().filter(|&&v| v != 0.0).count();
                    layer_total += w.len();
                }
                layer_nonzero += dn.conv_weight.iter().filter(|&&v| v != 0.0).count();
                layer_total += dn.conv_weight.len();
                layer_nonzero += dn.dt_bias.iter().filter(|&&v| v != 0.0).count();
                layer_total += dn.dt_bias.len();
                layer_nonzero += dn.a_log.iter().filter(|&&v| v != 0.0).count();
                layer_total += dn.a_log.len();
                layer_nonzero += dn.norm_weight.iter().filter(|&&v| v != 0.0).count();
                layer_total += dn.norm_weight.len();
            }
        }

        // MLP projections
        for proj in [&layer.mlp.gate_proj, &layer.mlp.up_proj, &layer.mlp.down_proj] {
            let w = proj.weights.as_raw_slice();
            layer_nonzero += w.iter().filter(|&&v| v != 0.0).count();
            layer_total += w.len();
        }

        let layer_kind = match &layer.attention {
            AttentionLayer::Standard(_) => "full",
            AttentionLayer::GatedDeltaNet(_) => "linear",
        };

        if layer_idx < 3 || layer_idx == config.num_layers - 1 {
            eprintln!("  layer {} ({}): {}/{} non-zero ({:.1}%)",
                layer_idx, layer_kind, layer_nonzero, layer_total,
                100.0 * layer_nonzero as f64 / layer_total as f64);
        } else if layer_idx == 3 {
            eprintln!("  ... (skipping middle layers) ...");
        }

        total_layer_nonzero += layer_nonzero;
        total_layer_weights += layer_total;
    }
    eprintln!("  TOTAL layers: {}/{} non-zero ({:.1}%)",
        total_layer_nonzero, total_layer_weights,
        100.0 * total_layer_nonzero as f64 / total_layer_weights as f64);

    if total_layer_nonzero == 0 {
        eprintln!("\nERROR: All layer weights are zero! Model weights were not loaded correctly.");
        std::process::exit(1);
    }

    // 5. Load tokenizer
    eprintln!("\n--- Tokenizer ---");
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to load tokenizer: {e}");
        std::process::exit(1);
    });
    eprintln!("  Tokenizer loaded.");

    // 6. Tokenize prompt
    let tokens = tokenizer.encode(&prompt).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to tokenize prompt: {e}");
        std::process::exit(1);
    });
    eprintln!("\n--- Prompt ---");
    eprintln!("  Text: \"{}\"", prompt);
    eprintln!("  Tokens: {:?}", tokens);
    eprintln!("  Token count: {}", tokens.len());

    // 7. Run forward pass
    eprintln!("\n--- Forward Pass ---");
    model.reset_kv_caches();

    // Prefill
    if tokens.len() > 1 {
        let _ = model.forward_prefill(&tokens[..tokens.len() - 1]);
        eprintln!("  Prefilled {} tokens.", tokens.len() - 1);
    }

    let last_token = *tokens.last().unwrap_or(&0);
    let position = if tokens.is_empty() { 0 } else { tokens.len() - 1 };
    eprintln!("\n--- Per-Layer Norms (last token) ---");
    let logits = model.forward_decode_debug(last_token, position);

    eprintln!("  Logits: {} values", logits.len());

    // Check for NaN/Inf
    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    let inf_count = logits.iter().filter(|v| v.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        eprintln!("  WARNING: {} NaN, {} Inf values in logits!", nan_count, inf_count);
    }

    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logit_mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    eprintln!("  Logit range: [{:.4}, {:.4}], mean: {:.4}", logit_min, logit_max, logit_mean);

    // Top-5 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n--- Top 10 Predicted Tokens ---");
    for (rank, &(idx, logit)) in indexed.iter().take(10).enumerate() {
        let text = tokenizer.decode(&[idx as u32]).unwrap_or_else(|_| format!("<id:{idx}>"));
        eprintln!("  #{}: token {} ({:?}) logit={:.4}", rank + 1, idx, text, logit);
    }

    // 8. Generate a few tokens
    eprintln!("\n--- Generation (greedy, 20 tokens) ---");
    let sampler = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let mut all_ids: Vec<u32> = tokens.clone();
    let mut logits = logits;
    let mut output = String::new();

    for step in 0..20 {
        let next_token = sample(&mut logits, &sampler, &all_ids, &mut rand::thread_rng());

        if eos.contains(&next_token) {
            eprintln!("  [EOS at step {}]", step);
            break;
        }

        all_ids.push(next_token);
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            output.push_str(&text);
        }

        let pos = tokens.len() + step;
        logits = model.forward_decode(next_token, pos);
    }

    eprintln!("  Output: \"{}\"", output);

    // 9. Summary
    eprintln!("\n=== Diagnosis Complete ===");
    if total_layer_nonzero == 0 {
        eprintln!("FAIL: No weights loaded. Check .safetensors files and tensor names.");
    } else if nan_count > 0 || inf_count > 0 {
        eprintln!("FAIL: NaN/Inf in logits. Possible numerical instability.");
    } else if logit_max == logit_min {
        eprintln!("WARN: All logits identical. Model may not be working correctly.");
    } else {
        eprintln!("OK: Plaintext model appears to work. If MPC produces garbage,");
        eprintln!("    the issue may be in the MPC protocol, not the base model.");
        eprintln!();
        eprintln!("Compare this output with MPC output to diagnose further.");
    }
}
