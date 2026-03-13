use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use rand::SeedableRng;

use klearu_llm::generate::chat_template::{ChatMessage, ChatTemplate};
use klearu_llm::generate::pipeline::{GenerateConfig, Pipeline, detect_eos_tokens};
use klearu_llm::generate::sampler::SamplerConfig;
#[cfg(feature = "sparse")]
use klearu_llm::generate::sparse_pipeline::SparsePipeline;

macro_rules! run_chat_loop {
    ($pipeline:expr, $template:expr, $config:expr, $system_msg:expr) => {{
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut history: Vec<ChatMessage> = Vec::new();

        if let Some(sys) = &$system_msg {
            history.push(ChatMessage::system(sys.clone()));
            eprintln!("System: {sys}");
        }

        eprintln!("Ready. Type a message (Ctrl-D to quit).\n");

        let stdin = io::stdin();
        loop {
            eprint!("> ");
            io::stderr().flush().unwrap();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).unwrap() == 0 {
                eprintln!();
                break;
            }
            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            history.push(ChatMessage::user(input));
            let prompt = $template.apply(&history);

            let mut response = String::new();
            match $pipeline.generate_streaming(&prompt, &$config, &mut rng, |token_text| {
                print!("{token_text}");
                io::stdout().flush().unwrap();
                response.push_str(token_text);
                true
            }) {
                Ok(_) => {
                    println!();
                    history.push(ChatMessage::assistant(response));
                }
                Err(e) => {
                    eprintln!("\nGeneration error: {e}");
                }
            }
        }
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [--temp <f32>] [--top-k <n>] [--top-p <f32>] [--max-tokens <n>] [--template <name>] [--system <msg>] [--sparse] [--head-sparsity <f32>] [--neuron-sparsity <f32>]", args[0]);
        eprintln!();
        eprintln!("Templates: zephyr, chatml, llama2, llama3, mistral, raw, auto (default)");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);

    // Parse CLI args
    let mut temperature = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut max_tokens = 512usize;
    let mut rep_penalty = 1.1f32;
    let mut template_name = "auto".to_string();
    let mut system_msg: Option<String> = None;
    let mut use_sparse = false;
    let mut head_sparsity = 0.5f32;
    let mut neuron_sparsity = 0.5f32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" | "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("Invalid temperature");
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse().expect("Invalid top-k");
            }
            "--top-p" => {
                i += 1;
                top_p = args[i].parse().expect("Invalid top-p");
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("Invalid max-tokens");
            }
            "--rep-penalty" => {
                i += 1;
                rep_penalty = args[i].parse().expect("Invalid rep-penalty");
            }
            "--template" => {
                i += 1;
                template_name = args[i].clone();
            }
            "--system" => {
                i += 1;
                system_msg = Some(args[i].clone());
            }
            "--sparse" => {
                use_sparse = true;
            }
            "--head-sparsity" => {
                i += 1;
                head_sparsity = args[i].parse().expect("Invalid head-sparsity");
            }
            "--neuron-sparsity" => {
                i += 1;
                neuron_sparsity = args[i].parse().expect("Invalid neuron-sparsity");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Detect or use specified template
    let template = match template_name.as_str() {
        "auto" => {
            let detected = ChatTemplate::detect(&model_dir).unwrap_or(ChatTemplate::Raw);
            eprintln!("Auto-detected template: {detected:?}");
            detected
        }
        "zephyr" => ChatTemplate::Zephyr,
        "chatml" => ChatTemplate::ChatML,
        "llama2" => ChatTemplate::Llama2,
        "llama3" => ChatTemplate::Llama3,
        "mistral" => ChatTemplate::MistralInstruct,
        "raw" => ChatTemplate::Raw,
        other => {
            eprintln!("Unknown template: {other}");
            std::process::exit(1);
        }
    };

    // Detect EOS token from model config
    let eos_token_ids = detect_eos_tokens(&model_dir);
    eprintln!("EOS token IDs: {eos_token_ids:?}");

    let config = GenerateConfig {
        max_new_tokens: max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: rep_penalty,
        },
        eos_token_ids,
    };

    if use_sparse {
        #[cfg(not(feature = "sparse"))]
        {
            eprintln!("Error: --sparse requires the 'sparse' feature. Rebuild with --features sparse");
            std::process::exit(1);
        }

        #[cfg(feature = "sparse")]
        {
            eprintln!(
                "Loading sparse model from {} (head_sparsity={head_sparsity}, neuron_sparsity={neuron_sparsity})...",
                model_dir.display()
            );
            let mut pipeline =
                SparsePipeline::from_dir(&model_dir, head_sparsity, neuron_sparsity)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to load model: {e}");
                        std::process::exit(1);
                    });
            eprintln!(
                "Model loaded: {} layers, hidden_size={}, vocab_size={}",
                pipeline.model.model.config.num_layers,
                pipeline.model.model.config.hidden_size,
                pipeline.model.model.config.vocab_size
            );
            run_chat_loop!(pipeline, template, config, system_msg);
        }
    } else {
        eprintln!("Loading model from {}...", model_dir.display());
        let mut pipeline = Pipeline::from_dir(&model_dir).unwrap_or_else(|e| {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "Model loaded: {} layers, hidden_size={}, vocab_size={}",
            pipeline.model.config.num_layers,
            pipeline.model.config.hidden_size,
            pipeline.model.config.vocab_size
        );
        run_chat_loop!(pipeline, template, config, system_msg);
    }
}
