//! sd_inventory — walk a Stable Diffusion checkpoint directory and
//! print every safetensors tensor with shape and dtype, plus a per-
//! component summary. Sanity-check before any further work on a
//! downloaded checkpoint.
//!
//! Usage:
//!   sd_inventory --checkpoint PATH/TO/sd-1-5
//!     [--component unet|vae|text_encoder]   (filter to one component)
//!     [--names-only]                          (don't print shape/dtype)
//!     [--top-by-bytes N]                      (top N largest tensors)

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_diffusion::config::{CheckpointConfig, SDVariant};
use klearu_diffusion::weight::{inventory_checkpoint, summarise, SingleFileLoader};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint: Option<PathBuf> = None;
    let mut single_file: Option<PathBuf> = None;
    let mut component_filter: Option<String> = None;
    let mut names_only = false;
    let mut top_by_bytes: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => { i += 1; checkpoint = Some(PathBuf::from(&args[i])); }
            "--single-file" => { i += 1; single_file = Some(PathBuf::from(&args[i])); }
            "--component" => { i += 1; component_filter = Some(args[i].clone()); }
            "--names-only" => { names_only = true; }
            "--top-by-bytes" => { i += 1; top_by_bytes = Some(args[i].parse().expect("usize")); }
            "--help" | "-h" => {
                eprintln!("usage:");
                eprintln!("  sd_inventory --checkpoint DIR [--component NAME] [--names-only] [--top-by-bytes N]");
                eprintln!("  sd_inventory --single-file FILE.safetensors [--top-by-bytes N]");
                return ExitCode::SUCCESS;
            }
            other => { eprintln!("unknown flag: {other}"); return ExitCode::FAILURE; }
        }
        i += 1;
    }

    // Single-file mode: print modelspec metadata, variant, and per-namespace tensor summary.
    if let Some(sf) = &single_file {
        let loader = match SingleFileLoader::open(sf) {
            Ok(l) => l,
            Err(e) => { eprintln!("single-file: {e}"); return ExitCode::FAILURE; }
        };
        println!("file: {}", sf.display());
        println!("variant (detected): {:?}", loader.variant);
        println!();
        println!("modelspec metadata:");
        let m = &loader.metadata;
        if let Some(t) = &m.title { println!("  title:           {t:?}"); }
        if let Some(a) = &m.architecture { println!("  architecture:    {a:?}"); }
        if let Some(i) = &m.implementation { println!("  implementation:  {i:?}"); }
        if let Some(au) = &m.author { println!("  author:          {au:?}"); }
        if let Some(d) = &m.date { println!("  date:            {d:?}"); }
        if let Some(l) = &m.license { println!("  license:         {l:?}"); }
        if let Some(p) = &m.prediction_type { println!("  prediction_type: {p:?}"); }
        if let Some(r) = &m.resolution { println!("  resolution:      {r:?}"); }
        if let Some(t) = &m.timestep_range { println!("  timestep_range:  {t:?}"); }
        if let Some(u) = &m.usage_hint { println!("  usage_hint:      {u:?}"); }
        if let Some(d) = &m.description {
            let truncated: String = d.chars().take(120).collect();
            println!("  description:     {truncated:?}");
        }
        // Other modelspec.* keys not promoted to typed fields.
        let other_keys: Vec<&String> = m.raw.keys()
            .filter(|k| k.starts_with("modelspec.") &&
                !matches!(k.as_str(),
                    "modelspec.architecture" | "modelspec.implementation" |
                    "modelspec.title" | "modelspec.author" | "modelspec.date" |
                    "modelspec.license" | "modelspec.prediction_type" |
                    "modelspec.resolution" | "modelspec.timestep_range" |
                    "modelspec.usage_hint" | "modelspec.description"))
            .collect();
        if !other_keys.is_empty() {
            println!("  other modelspec.* keys: {} (use --names-only to dump)", other_keys.len());
            if names_only {
                for k in &other_keys {
                    println!("    {k}: {:?}", m.raw.get(k.as_str()).unwrap());
                }
            }
        }
        println!();

        // Group tensors by their top-level namespace (= first dotted segment).
        let mut by_ns: std::collections::BTreeMap<&str, (usize, usize)> =
            std::collections::BTreeMap::new();
        for (name, r) in &loader.raw_tensors {
            let ns = name.split('.').next().unwrap_or("?");
            let elems: usize = r.shape.iter().product();
            let bpe = match r.dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::F64 => 8,
                _ => 4,
            };
            let entry = by_ns.entry(ns).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += elems * bpe;
        }
        println!("tensors grouped by top-level namespace:");
        let mut total_count = 0usize;
        let mut total_bytes = 0usize;
        for (ns, (count, bytes)) in &by_ns {
            println!("  {ns:<30}  {count:>5} tensors  {:>10} MB", bytes / (1024*1024));
            total_count += count;
            total_bytes += bytes;
        }
        println!("  {:<30}  {total_count:>5} tensors  {:>10} MB total", "(grand total)",
            total_bytes / (1024*1024));

        if let Some(n) = top_by_bytes {
            let mut items: Vec<(&String, usize)> = loader.raw_tensors.iter().map(|(name, r)| {
                let elems: usize = r.shape.iter().product();
                let bpe = match r.dtype {
                    safetensors::Dtype::F32 => 4,
                    safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                    safetensors::Dtype::F64 => 8,
                    _ => 4,
                };
                (name, elems * bpe)
            }).collect();
            items.sort_by(|a, b| b.1.cmp(&a.1));
            items.truncate(n);
            println!();
            println!("=== top {n} tensors by bytes ===");
            for (name, b) in items {
                let r = &loader.raw_tensors[name.as_str()];
                println!("  {:>10} MB  [{:?}]  {name}", b / (1024*1024), r.shape);
            }
        }
        return ExitCode::SUCCESS;
    }

    let checkpoint = checkpoint.expect("--checkpoint or --single-file required");

    // Try to parse the configs first (gives a meaningful error if the
    // checkpoint is the wrong shape).
    match CheckpointConfig::from_dir(&checkpoint) {
        Ok(cfg) => {
            println!("checkpoint: {}", checkpoint.display());
            println!("class: {}", cfg.model_index.class_name);
            let variant = cfg.variant();
            println!("detected variant: {:?}", variant);
            println!("components in model_index: {} keys", cfg.model_index.components.len());
            println!();
            println!("UNet:");
            println!("  in_channels={}, out_channels={}", cfg.unet.in_channels, cfg.unet.out_channels);
            println!("  block_out_channels={:?}", cfg.unet.block_out_channels);
            println!("  cross_attention_dim={}", cfg.unet.cross_attention_dim);
            println!("  transformer_layers_per_block={:?}", cfg.unet.transformer_layers_expanded());
            if matches!(variant, SDVariant::Sdxl) {
                println!("  addition_embed_type={:?}", cfg.unet.addition_embed_type);
                println!("  addition_time_embed_dim={:?}", cfg.unet.addition_time_embed_dim);
                println!("  projection_class_embeddings_input_dim={:?}",
                    cfg.unet.projection_class_embeddings_input_dim);
            }
            println!("VAE:");
            println!("  in/out_channels={}/{}, latent_channels={}",
                cfg.vae.in_channels, cfg.vae.out_channels, cfg.vae.latent_channels);
            println!("  block_out_channels={:?}, scaling_factor={}",
                cfg.vae.block_out_channels, cfg.vae.scaling_factor);
            println!("Text encoder (CLIP-L):");
            println!("  hidden_size={}, num_layers={}, num_heads={}, max_pos={}",
                cfg.text_encoder.hidden_size, cfg.text_encoder.num_hidden_layers,
                cfg.text_encoder.num_attention_heads, cfg.text_encoder.max_position_embeddings);
            if let Some(t2) = &cfg.text_encoder_2 {
                println!("Text encoder 2 (CLIP-G, SDXL):");
                println!("  hidden_size={}, num_layers={}, num_heads={}, max_pos={}",
                    t2.hidden_size, t2.num_hidden_layers,
                    t2.num_attention_heads, t2.max_position_embeddings);
            }
            println!("Scheduler:");
            println!("  class={}, T={}, β=[{}, {}], schedule={}, prediction={}",
                cfg.scheduler.class_name, cfg.scheduler.num_train_timesteps,
                cfg.scheduler.beta_start, cfg.scheduler.beta_end,
                cfg.scheduler.beta_schedule, cfg.scheduler.prediction_type);
            println!();
        }
        Err(e) => {
            eprintln!("could not parse configs ({e}); proceeding to raw safetensors inventory");
        }
    }

    let infos = match inventory_checkpoint(&checkpoint) {
        Ok(i) => i,
        Err(e) => { eprintln!("inventory failed: {e}"); return ExitCode::FAILURE; }
    };

    let stats = summarise(&infos);
    println!("=== component summary ===");
    let mut total_b = 0usize;
    for (name, s) in &stats {
        println!("  {name:<20}  {} tensors  {:>14} elems  {:>10} MB",
            s.tensor_count, s.total_elements, s.total_bytes / (1024*1024));
        total_b += s.total_bytes;
    }
    println!("  {:<20}  {:>10} MB total", "(grand total)", total_b / (1024*1024));
    println!();

    let filtered: Vec<&_> = if let Some(ref c) = component_filter {
        infos.iter().filter(|i| i.component == *c).collect()
    } else {
        infos.iter().collect()
    };

    if let Some(n) = top_by_bytes {
        let mut items: Vec<(&_, usize)> = filtered.iter().map(|info| {
            let elems: usize = info.shape.iter().product();
            let bytes_per = match info.dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::F64 => 8,
                _ => 4,
            };
            (*info, elems * bytes_per)
        }).collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(n);
        println!("=== top {n} tensors by bytes ===");
        for (info, b) in items {
            println!("  {:>10} MB  [{:?}]  {}::{}",
                b / (1024*1024), info.shape, info.component, info.name);
        }
        return ExitCode::SUCCESS;
    }

    if names_only {
        for info in filtered {
            println!("{}::{}", info.component, info.name);
        }
    } else {
        for info in filtered {
            println!("{:<20} {:?} dtype={:?} {}",
                info.component, info.shape, info.dtype, info.name);
        }
    }
    ExitCode::SUCCESS
}
