use std::path::PathBuf;

use clap::Parser;
use klearu_llm::similarity::{cosine_similarity, TokenEmbeddings};
use klearu_llm::tokenizer::Tokenizer;

/// Compare input token embeddings (mean-pooled for multi-token combinations).
#[derive(Parser)]
struct Args {
    /// Local model directory or safetensors file
    model: PathBuf,
    /// Text for combination A (encoded without adding special tokens)
    #[arg(long, required_unless_present = "a_ids", conflicts_with = "a_ids")]
    a: Option<String>,
    /// Text for combination B
    #[arg(long, required_unless_present = "b_ids", conflicts_with = "b_ids")]
    b: Option<String>,
    /// Comma-separated token IDs for combination A
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    a_ids: Option<Vec<u32>>,
    /// Comma-separated token IDs for combination B
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    b_ids: Option<Vec<u32>>,
    /// Explicit embedding tensor name, for other model architectures
    #[arg(long)]
    tensor: Option<String>,
    /// Override tokenizer.json path (only needed for text inputs)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Emit a JSON object
    #[arg(long)]
    json: bool,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = if args.a.is_some() || args.b.is_some() {
        let dir = if args.model.is_dir() {
            args.model.as_path()
        } else {
            args.model
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
        };
        Some(Tokenizer::from_file(
            &args.tokenizer.unwrap_or_else(|| dir.join("tokenizer.json")),
        )?)
    } else {
        None
    };
    let resolve = |text: Option<String>, ids: Option<Vec<u32>>| match (text, ids) {
        (Some(text), _) => tokenizer.as_ref().unwrap().encode(&text),
        (_, Some(ids)) => Ok(ids),
        _ => unreachable!("clap requires text or IDs"),
    };
    let a_ids = resolve(args.a, args.a_ids)?;
    let b_ids = resolve(args.b, args.b_ids)?;
    let embeddings = TokenEmbeddings::open(&args.model, args.tensor.as_deref())?;
    let a = embeddings.mean_embedding(&a_ids)?;
    let b = embeddings.mean_embedding(&b_ids)?;
    let score = cosine_similarity(&a, &b)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "cosine_similarity": score,
                "a_ids": a_ids,
                "b_ids": b_ids,
                "tensor": embeddings.tensor_name,
            "hidden_size": embeddings.hidden_size(),
                "pooling": "mean",
                "representation": "input_embeddings"
            }))?
        );
    } else {
        println!("A token IDs: {a_ids:?}\nB token IDs: {b_ids:?}");
        println!(
            "Tensor: {} ({} dimensions; mean pooling)",
            embeddings.tensor_name,
            embeddings.hidden_size()
        );
        println!("Cosine similarity: {score:.10}");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
