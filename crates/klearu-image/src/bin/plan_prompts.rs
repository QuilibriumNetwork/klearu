//! plan_prompts — expand seed prompts through klearu-llm's ReasoningPlanner.
//!
//! Reads prompts from stdin (one per line) or `--in FILE`. For each line,
//! runs the planner LLM and emits one JSON object per line on stdout:
//!
//!     {"original": "...", "expanded": "<plan> ... </plan> ..."}
//!
//! The `expanded` field is the planner's structured prefix concatenated
//! with the original prompt — exactly what `assemble_planned_prefix`
//! expects callers to BPE-encode as the image transformer's text slot.
//!
//! Designed to be pipe-friendly so `distill_from_sdxl.py` can shell out
//! to it without managing the LLM lifecycle itself.
//!
//! Usage:
//!     plan_prompts \
//!         --llm-model-dir Qwen3.5-4B/ \
//!         [--in seeds.txt] [--out expanded.jsonl] \
//!         [--max-new 96] [--temperature 0.2] [--seed 42]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use klearu_image::error::Result as ImgResult;
use klearu_image::planner::{ReasoningPlanner, Plan};
use klearu_llm::generate::pipeline::{GenerateConfig, Pipeline};
use klearu_llm::generate::sampler::SamplerConfig;

struct Args {
    llm_model_dir: PathBuf,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    max_new: usize,
    temperature: f32,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args {
        llm_model_dir: PathBuf::new(),
        input: None,
        output: None,
        max_new: 96,
        temperature: 0.2,
        seed: 42,
    };
    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            raw.get(*i).cloned().ok_or_else(|| format!("flag {arg} expects a value"))
        };
        match arg.as_str() {
            "--llm-model-dir" => args.llm_model_dir = PathBuf::from(next(&mut i)?),
            "--in" => args.input = Some(PathBuf::from(next(&mut i)?)),
            "--out" => args.output = Some(PathBuf::from(next(&mut i)?)),
            "--max-new" => args.max_new = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--temperature" => args.temperature = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => args.seed = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => return Err(format!("unknown flag: {other:?}")),
        }
        i += 1;
    }
    if args.llm_model_dir.as_os_str().is_empty() {
        return Err("--llm-model-dir is required".into());
    }
    Ok(args)
}

fn print_help() {
    eprintln!("plan_prompts — expand seed prompts via klearu-llm planner");
    eprintln!();
    eprintln!("  --llm-model-dir DIR   klearu-llm checkpoint folder (required)");
    eprintln!("  --in FILE             Read prompts from FILE (default stdin)");
    eprintln!("  --out FILE            Write JSONL to FILE (default stdout)");
    eprintln!("  --max-new N           Max tokens generated per plan (default 96)");
    eprintln!("  --temperature F       Planner sampling temperature (default 0.2)");
    eprintln!("  --seed N              RNG seed (default 42)");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::from(2); }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}

fn run(args: Args) -> ImgResult<()> {
    eprintln!("[plan_prompts] loading LLM from {}", args.llm_model_dir.display());
    let pipeline = Pipeline::from_dir(&args.llm_model_dir)
        .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
            format!("pipeline load: {e}")))?;
    eprintln!("[plan_prompts]   ✓ loaded");

    let mut planner = ReasoningPlanner::new(pipeline);
    planner.gen_config = GenerateConfig {
        max_new_tokens: args.max_new,
        sampler: SamplerConfig {
            temperature: args.temperature,
            top_k: 16,
            top_p: 0.95,
            repetition_penalty: 1.05,
        },
        ..GenerateConfig::default()
    };

    // Input: file or stdin.
    let stdin = std::io::stdin();
    let reader: Box<dyn BufRead> = match &args.input {
        Some(p) => Box::new(BufReader::new(std::fs::File::open(p)?)),
        None => Box::new(stdin.lock()),
    };
    // Output: file or stdout.
    let stdout = std::io::stdout();
    let mut writer: Box<dyn Write> = match &args.output {
        Some(p) => Box::new(std::fs::File::create(p)?),
        None => Box::new(stdout.lock()),
    };

    let mut n_done = 0usize;
    let mut seed = args.seed;
    for line in reader.lines() {
        let line = line?;
        let prompt = line.trim();
        if prompt.is_empty() || prompt.starts_with('#') { continue; }

        let plan = match planner.plan(prompt, seed) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[plan_prompts] plan failed for {prompt:?}: {e}");
                Plan::default()
            }
        };
        let plan_text = plan.to_text();
        // Expanded = plan prefix + the original user prompt. The image
        // transformer's BPE will tokenize the whole thing.
        let expanded = format!("{plan_text} {prompt}");

        let record = serde_json::json!({
            "original": prompt,
            "expanded": expanded,
        });
        writeln!(writer, "{}", record)?;
        writer.flush()?;
        n_done += 1;
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        if n_done % 20 == 0 {
            eprintln!("[plan_prompts] {n_done} prompts expanded");
        }
    }
    eprintln!("[plan_prompts] done — {n_done} prompts processed");
    Ok(())
}
