use axum::http::{HeaderValue, header};
use axum::middleware::{self, Next};
use axum::{Router, routing::{get, post}};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

mod api;
mod session;
pub mod two_party;
mod ws;

use klearu_llm::generate::pipeline::{detect_eos_token, detect_eos_tokens, Pipeline};

pub struct AppState {
    pub pipeline: Mutex<Pipeline>,
    pub model_dir: PathBuf,
    pub eos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

#[derive(Parser)]
#[command(name = "klearu-server", about = "Private 2PC inference server")]
struct Args {
    /// Path to the model directory (must contain config.json, tokenizer.json, *.safetensors)
    model_dir: PathBuf,

    /// Port to listen on
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Path to static files to serve (built web app)
    #[arg(long)]
    static_dir: Option<PathBuf>,
}

/// Middleware to add Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy
/// headers, required for SharedArrayBuffer support in the browser.
async fn coop_coep_headers(
    request: axum::extract::Request,
    next: Next,
) -> axum::response::Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::HeaderName::from_static("cross-origin-opener-policy"),
        HeaderValue::from_static("same-origin"),
    );
    headers.insert(
        header::HeaderName::from_static("cross-origin-embedder-policy"),
        HeaderValue::from_static("credentialless"),
    );
    response
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("klearu_server=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Loading model from {:?}", args.model_dir);
    let pipeline = Pipeline::from_dir(&args.model_dir)
        .expect("Failed to load model");
    let eos_token_ids = detect_eos_tokens(&args.model_dir);
    let eos_token_id = detect_eos_token(&args.model_dir);

    tracing::info!(
        "Model loaded: vocab={}, hidden={}, layers={}, heads={}",
        pipeline.model.config.vocab_size,
        pipeline.model.config.hidden_size,
        pipeline.model.config.num_layers,
        pipeline.model.config.num_heads,
    );

    // Write weight caches to disk so they can be streamed via ServeFile
    // (avoids holding hundreds of MB in memory as a single Bytes frame)
    let weights_dir = std::env::temp_dir().join("klearu-weights");
    std::fs::create_dir_all(&weights_dir).expect("Failed to create weights cache dir");

    tracing::info!("Serializing embedding weights to disk...");
    let embedding_path = weights_dir.join("embeddings.bin");
    let embedding_data = api::serialize_embedding_weights(&pipeline.model);
    tracing::info!("Embedding weights: {} bytes", embedding_data.len());
    std::fs::write(&embedding_path, &embedding_data).expect("Failed to write embedding weights");
    drop(embedding_data);

    tracing::info!("Serializing model weights to disk...");
    let model_path = weights_dir.join("model.bin");
    let model_data = api::serialize_model_weights(&pipeline.model);
    tracing::info!("Model weights: {} bytes", model_data.len());
    std::fs::write(&model_path, &model_data).expect("Failed to write model weights");
    drop(model_data);

    let state = Arc::new(AppState {
        pipeline: Mutex::new(pipeline),
        model_dir: args.model_dir,
        eos_token_id,
        eos_token_ids,
    });

    let mut app = Router::new()
        .route("/api/health", get(api::health))
        .route("/api/model", get(api::model_info))
        .route("/api/models/{filename}", get(api::model_file))
        .route_service("/api/weights/embeddings", ServeFile::new(&embedding_path))
        .route_service("/api/weights/model", ServeFile::new(&model_path))
        .route("/api/generate", post(api::generate))
        .route("/api/generate_from_embedding", post(api::generate_from_embedding))
        .route("/api/top_tokens", post(api::top_tokens))
        .route("/api/embed_tokens", post(api::embed_tokens))
        .route("/api/embed_text", post(api::embed_text))
        .route("/api/embed_and_generate", post(api::embed_and_generate))
        .route("/ws", get(ws::ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    if let Some(static_dir) = args.static_dir {
        app = app.fallback_service(ServeDir::new(static_dir));
    }

    // Apply COOP/COEP headers AFTER fallback_service so static files also get them
    let app = app.layer(middleware::from_fn(coop_coep_headers));

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
