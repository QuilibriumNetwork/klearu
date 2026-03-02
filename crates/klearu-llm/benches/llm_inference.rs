/// Benchmarks for klearu-llm inference.
///
/// All benchmarks use a synthetic in-memory model (zeroed weights with norms
/// set to 1.0) so no HuggingFace download is required.  The numbers reflect
/// the raw compute cost of the forward pass; real-model numbers will be
/// similar once weights are loaded from disk.
///
/// To benchmark against a real HuggingFace model, set the environment
/// variable `KLEARU_MODEL_DIR` to the path of a downloaded model directory:
///
///   KLEARU_MODEL_DIR=./SmolLM-135M-Instruct \
///     cargo bench --bench llm_inference -p klearu-llm
///
/// Without that variable the benchmarks fall back to the synthetic model.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use klearu_llm::config::LlmConfig;
use klearu_llm::model::Model;

// ---------------------------------------------------------------------------
// Synthetic model helpers
// ---------------------------------------------------------------------------

/// A small but non-trivial config that exercises the full transformer stack
/// without being too slow for CI.
fn small_config() -> LlmConfig {
    LlmConfig {
        vocab_size: 512,
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_size: 512,
        num_layers: 4,
        max_seq_len: 512,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
    }
}

/// A medium config that approximates a ~135M-parameter SmolLM model.
fn medium_config() -> LlmConfig {
    LlmConfig {
        vocab_size: 49_152,
        hidden_size: 576,
        num_heads: 9,
        num_kv_heads: 3,
        head_dim: 64,
        intermediate_size: 1_536,
        num_layers: 30,
        max_seq_len: 2_048,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
    }
}

/// Build a synthetic model with norm weights initialised to 1.0 so that
/// the forward pass produces finite (though meaningless) logits.
fn build_synthetic(config: LlmConfig) -> Model {
    let mut model = Model::new(config);
    for w in model.final_norm.weight.iter_mut() {
        *w = 1.0;
    }
    for layer in &mut model.layers {
        for w in layer.attn_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for w in layer.mlp_norm.weight.iter_mut() {
            *w = 1.0;
        }
    }
    model
}

/// Try to load a real model from `KLEARU_MODEL_DIR`.  Returns `None` when the
/// variable is unset or the directory cannot be loaded.
fn maybe_load_real_model() -> Option<Model> {
    let dir = std::env::var("KLEARU_MODEL_DIR").ok()?;
    let path = std::path::Path::new(&dir);
    klearu_llm::weight::load_model(path).ok()
}

// ---------------------------------------------------------------------------
// 1. Single-token decode latency  (time-to-first-token proxy)
// ---------------------------------------------------------------------------

fn bench_decode_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_single_token");

    // Small synthetic
    {
        let mut model = build_synthetic(small_config());
        group.bench_function("small_synthetic", |b| {
            b.iter(|| {
                model.reset_kv_caches();
                black_box(model.forward_decode(black_box(1u32), 0));
            })
        });
    }

    // Medium synthetic
    {
        let mut model = build_synthetic(medium_config());
        group.bench_function("medium_synthetic", |b| {
            b.iter(|| {
                model.reset_kv_caches();
                black_box(model.forward_decode(black_box(1u32), 0));
            })
        });
    }

    // Real model (optional)
    if let Some(mut model) = maybe_load_real_model() {
        group.bench_function("real_model", |b| {
            b.iter(|| {
                model.reset_kv_caches();
                black_box(model.forward_decode(black_box(1u32), 0));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Prefill throughput across prompt lengths
// ---------------------------------------------------------------------------

fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill_throughput");

    let prompt_lengths: &[usize] = &[8, 32, 128, 256];

    for &len in prompt_lengths {
        let tokens: Vec<u32> = (0..len as u32).map(|i| i % 512).collect();

        // Small synthetic
        {
            let mut model = build_synthetic(small_config());
            group.bench_with_input(
                BenchmarkId::new("small_synthetic", len),
                &len,
                |b, _| {
                    b.iter(|| {
                        model.reset_kv_caches();
                        black_box(model.forward_prefill(black_box(&tokens)));
                    })
                },
            );
        }

        // Real model (optional)
        if let Some(mut model) = maybe_load_real_model() {
            let real_tokens: Vec<u32> = (0..len as u32)
                .map(|i| i % model.config.vocab_size as u32)
                .collect();
            group.bench_with_input(
                BenchmarkId::new("real_model", len),
                &len,
                |b, _| {
                    b.iter(|| {
                        model.reset_kv_caches();
                        black_box(model.forward_prefill(black_box(&real_tokens)));
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Autoregressive decode throughput (N sequential tokens)
// ---------------------------------------------------------------------------

fn bench_decode_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_sequential");

    let decode_steps: &[usize] = &[16, 64, 128];

    for &steps in decode_steps {
        // Small synthetic
        {
            let mut model = build_synthetic(small_config());
            group.bench_with_input(
                BenchmarkId::new("small_synthetic", steps),
                &steps,
                |b, &n| {
                    b.iter(|| {
                        model.reset_kv_caches();
                        let mut logits = model.forward_decode(1u32, 0);
                        for step in 1..n {
                            // greedy: pick argmax
                            let next = logits
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .map(|(i, _)| i as u32)
                                .unwrap_or(0);
                            logits = model.forward_decode(black_box(next), step);
                        }
                        black_box(logits);
                    })
                },
            );
        }

        // Real model (optional)
        if let Some(mut model) = maybe_load_real_model() {
            let vocab = model.config.vocab_size as u32;
            group.bench_with_input(
                BenchmarkId::new("real_model", steps),
                &steps,
                |b, &n| {
                    b.iter(|| {
                        model.reset_kv_caches();
                        let mut logits = model.forward_decode(1u32, 0);
                        for step in 1..n {
                            let next = logits
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .map(|(i, _)| (i as u32) % vocab)
                                .unwrap_or(0);
                            logits = model.forward_decode(black_box(next), step);
                        }
                        black_box(logits);
                    })
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. KV-cache warm vs cold decode
//    Measures the cost of a single decode step at increasing context lengths.
// ---------------------------------------------------------------------------

fn bench_decode_at_context_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_at_context_length");

    let context_lengths: &[usize] = &[0, 32, 128, 256];

    for &ctx_len in context_lengths {
        let mut model = build_synthetic(small_config());

        group.bench_with_input(
            BenchmarkId::new("small_synthetic_ctx", ctx_len),
            &ctx_len,
            |b, &ctx| {
                b.iter(|| {
                    // Each decode appends to the KV cache, so we must reset and
                    // prefill before every iteration to avoid overflow.
                    model.reset_kv_caches();
                    for pos in 0..ctx {
                        let tok = (pos % 512) as u32;
                        let _ = model.forward_decode(tok, pos);
                    }
                    black_box(model.forward_decode(black_box(1u32), ctx));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_decode_single,
    bench_prefill,
    bench_decode_sequential,
    bench_decode_at_context_length,
);
criterion_main!(benches);
