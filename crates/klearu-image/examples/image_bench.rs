//! Bounded compute benchmark, no dataset or pretrained weights required.
use clap::Parser;
use klearu_image::{
    backward::train_step,
    grad::Gradients,
    model::{ImageTransformer, ImageTransformerConfig},
    optim::{AdamW, AdamWConfig},
    sample::{sample_image_tokens, SampleConfig},
    train::{assemble_batch, TrainExample},
};
use std::time::Instant;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    baseline: bool,
    #[arg(long)]
    uncached: bool,
    #[arg(long, default_value_t = 5)]
    train_steps: usize,
}
fn main() {
    let args = Args::parse();
    assert!(args.train_steps > 0, "--train-steps must be positive");
    let baseline = args.baseline;
    let full = args.uncached;
    let mut cfg = if baseline {
        ImageTransformerConfig::baseline_50m()
    } else {
        ImageTransformerConfig::pilot()
    };
    cfg.vocab_text = 8192;
    cfg.bos_token = 8192;
    cfg.sep_image_token = 8193;
    cfg.eos_token = 8194;
    let mut m = ImageTransformer::from_config(cfg.clone());
    m.init_random(42);
    let ex = TrainExample {
        text_tokens: vec![10; 64],
        image_tokens: (0..256).map(|i| (i * 31 % 8192) as u32).collect(),
    };
    let batch = assemble_batch(&cfg, &ex).unwrap();
    let mut opt = AdamW::new(&m, AdamWConfig::default());
    let mut g = Gradients::zeros_for(&m);
    train_step(&mut m, &mut opt, &mut g, &batch).unwrap(); // warm up
    let t = Instant::now();
    for _ in 0..args.train_steps {
        std::hint::black_box(train_step(&mut m, &mut opt, &mut g, &batch).unwrap());
    }
    let train = t.elapsed().as_secs_f64() / args.train_steps as f64;
    let t = Instant::now();
    let tokens = sample_image_tokens(
        &m,
        &ex.text_tokens,
        &SampleConfig {
            top_k: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let decode = t.elapsed().as_secs_f64();
    let mut uncached = None;
    if full {
        let mut seq = klearu_image::sample::build_prefix(&cfg, &ex.text_tokens).unwrap();
        let t = Instant::now();
        for token in &tokens {
            std::hint::black_box(m.forward(&seq).unwrap());
            seq.push(m.image_token_to_id(*token));
        }
        uncached = Some(t.elapsed().as_secs_f64());
    }
    println!(
        "{}",
        serde_json::json!({"preset":if baseline{"baseline"}else{"pilot"},"params":m.param_count(),
        "text_vocab":cfg.vocab_text,"sequence_length":batch.token_ids.len(),"train_seconds_per_example":train,
        "cached_generation_seconds":decode,"uncached_generation_seconds":uncached,
        "parameter_gradient_adam_bytes":m.param_count()*16,"warmup_steps":1,"measured_steps":args.train_steps})
    );
}
