# Klearu

A native Rust implementation of the SLIDE paper family (Sub-LInear Deep learning Engine), with extensions for LLM inference, transformer sparsity prediction, and private two-party computation.

## License

AGPL-3.0 with additional terms. See [LICENSE](LICENSE) for details. Commercial use is restricted to the Quilibrium mainnet. Automated reproduction (including LLM-assisted "clean room" reimplementation) for commercial substitutes is expressly prohibited.

## Workspace Overview

Klearu is organized as a Cargo workspace with 11 crates:

| Crate | Description |
|---|---|
| **klearu-core** | Foundation: LSH hash families, sparse tensors, SLIDE network training |
| **klearu-accel** | SIMD vectorization (AVX2/NEON/scalar), BF16 quantization, cache-aligned memory |
| **klearu-mongoose** | Learnable hash functions, adaptive rebuild scheduling with drift detection |
| **klearu-bolt** | LSH hyperparameter autotuning, sparse inference optimizations |
| **klearu-dejavu** | Deja Vu transformer sparsity prediction (attention heads + MLP neurons) |
| **klearu-vision** | Vision transformers: DaViT, ViT, Swin, ConvNeXt, Hiera, EVA-02, Qwen Vision, SigLIP, DINOv2 |
| **klearu-llm** | LLaMA-compatible LLM inference with optional sparsity and VLM support |
| **klearu** | Facade crate with feature-gated re-exports |
| **klearu-dpf** | Distributed Point Functions (AES-based BGI construction) and DCF |
| **klearu-mpc** | 2PC building blocks: Q16.16/Q32.32 fixed-point, Beaver triples, additive sharing |
| **klearu-private** | Private LLM inference via 2PC with Ferret OT and Ristretto255 OPRF |

## Building

The core crates build standalone and only require this repo to be cloned.

### Building klearu-private
The `klearu-private` crate depends on the `ferret` crate from the [Quilibrium monorepo](https://github.com/quilibriumnetwork/monorepo) via a relative path.

To build the full workspace including private inference, clone both repositories as siblings:

```
your-workspace/
  klearu/       # this repository
  monorepo/     # git clone https://github.com/quilibriumnetwork/monorepo
```
#### Installing Ferret Dependencies
To be able to build `ferret`, the EMP tools in the `monorepo/` directory must be installed:

```bash
# Inside the monorepo dir/
sudo ./install-emp.sh
```

At this point, `ferret` can be built, but will be built via the cargo commands in `klearu/`.

### Building KLEARU
Navigate to the `klearu/` directory:

| Feature                  | Description                              | Build Command                                              | Is `monorepo` Required? |
|--------------------------|------------------------------------------|------------------------------------------------------------|--------------------------|
| Full                     | Build all features                       | `cargo build --release`                                    | `true`                   |
| Full (facade)            | Build all features via facade crate      | `cargo build --release -p klearu --features full`         | `true`                   |
| LLM Inference            | LLM inference only                       | `cargo build --release -p klearu-llm`                      | `false`                  |
| LLM + Sparse             | LLM with sparse inference support        | `cargo build --release -p klearu-llm --features sparse`   | `false`                  |
| Vision models            | Vision model inference                   | `cargo build --release -p klearu-vision`                   | `false`                  |
| Vision + Sparse          | Vision models with sparse inference      | `cargo build --release -p klearu-vision --features sparse`| `false`                  |
| Vision CLI               | Vision CLI (image classification)        | `cargo build --release -p klearu-vision --features cli`   | `false`                  |
| Private inference        | Private / proprietary inference features | `cargo build --release -p klearu-private`                  | `true`                   |

## Testing

```bash
cargo test --workspace
```

## Crate Details

### klearu-core — SLIDE Primitives

The foundation crate provides LSH-based sub-linear training and inference.

**Hash families** (`HashFamily` trait): SimHash, WtaHash, DwtaHash, MinHash, SparseRandomProjection

**LSH index** (`LshIndexTrait`): `query()`, `query_union()`, `query_with_counts()` — with FIFO or reservoir-sampled buckets

**Network**: Full SLIDE training loop with configurable layers, optimizers, and sampling strategies.

```rust
use klearu_core::config::*;
use klearu_core::network::Network;

let config = SlideConfig {
    network: NetworkConfig {
        layers: vec![
            LayerConfig::hidden(784, 1024),
            LayerConfig::output(1024, 10),
        ],
        optimizer: OptimizerType::Adam,
        learning_rate: 0.001,
        batch_size: 128,
        num_threads: 4,
    },
    seed: 42,
    hogwild: true,
};

let mut network = Network::new(config);
```

#### Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_tables` (L) | 50 | Number of LSH hash tables |
| `num_hashes` (K) | 6 | Hash bits per table |
| `bucket_capacity` | 128 | Max neurons per bucket |
| `bucket_type` | FIFO | FIFO or Reservoir sampling |
| `hash_function` | SimHash | SimHash, WtaHash, DwtaHash, MinHash, SRP |
| `rebuild_interval_base` | 100 | Steps between LSH rebuilds |
| `rebuild_decay` | 0.1 | Exponential decay for rebuild interval |
| `optimizer` | Adam | Adam or SGD |
| `activation` | ReLU | ReLU, Sigmoid, Tanh, Softmax |
| `sampling` | Vanilla | Vanilla, TopK, Threshold |
| `hogwild` | false | Lock-free parallel training |

### klearu-accel — Hardware Acceleration

Platform-adaptive SIMD (AVX2 on x86, NEON on ARM, scalar fallback) for dot products and scatter-add. BF16 quantization with two modes: full BF16 or BF16-storage/FP32-gradient. `ContiguousWeightStore` provides cache-line-aligned (64-byte) weight layouts.

### klearu-mongoose — Learnable Hashing

Trainable hash functions that adapt to data distribution, plus an `AdaptiveScheduler` that monitors hash-bucket drift via EMA and triggers rebuilds only when needed.

| Parameter | Default | Description |
|---|---|---|
| `min_interval` | — | Minimum steps between rebuild checks |
| `max_interval` | — | Forced rebuild interval |
| `sample_fraction` | — | Fraction of neurons to sample for drift |
| `drift_threshold` | — | Drift level that triggers a rebuild |
| `ema_alpha` | 0.3 | Exponential moving average smoothing |

### klearu-bolt — Autotuning

Automatic LSH hyperparameter search over K and L to hit a target recall while minimizing query cost.

```rust
use klearu_bolt::autotune::LshAutotuner;

let tuner = LshAutotuner::new(0.9)   // target 90% recall
    .with_k_range(4, 16)
    .with_l_range(10, 200)
    .with_num_samples(100)
    .with_speedup_ratio(0.1);

let result = tuner.autotune(&neurons, &queries, 42);
// result.best_k, result.best_l, result.recall, result.query_cost
```

### klearu-dejavu — Transformer Sparsity

Implementation of the [Deja Vu](https://arxiv.org/abs/2310.17157) paper: lightweight MLP predictors that identify which attention heads and FFN neurons are important for each token, enabling sparse transformer inference.

### klearu-llm — LLM Inference

A LLaMA-compatible inference engine supporting GQA, RoPE, RMSNorm, and SwiGLU. Works with any HuggingFace-format model that uses the LLaMA architecture.

#### LLM Configuration

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.7 | Sampling temperature (0.0 = greedy) |
| `top_k` | 40 | Top-k filtering (0 = disabled) |
| `top_p` | 0.9 | Nucleus sampling (1.0 = disabled) |
| `repetition_penalty` | 1.1 | Penalize repeated tokens (1.0 = disabled) |
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `template` | auto | Chat template (auto, zephyr, chatml, llama2, llama3, mistral, raw) |

#### Sparse Inference (feature: `sparse`)

| Parameter | Default | Description |
|---|---|---|
| `head_sparsity` | 0.5 | Fraction of attention heads to keep |
| `neuron_sparsity` | 0.5 | Fraction of FFN neurons to keep |

### klearu-vision — Vision Transformers

A pure-Rust vision encoder library supporting multiple architectures. All models load from [timm](https://huggingface.co/timm) safetensors format with automatic preprocessing config detection.

**Supported architectures:**

| Architecture | Loader | Notes |
|---|---|---|
| DaViT | `load_davit_model()` | 4-stage dual-attention (spatial window + channel) |
| ViT | `load_vit_model()` | Standard Vision Transformer (CLS/mean pool) |
| Swin | — | Shifted-window attention with relative position bias |
| ConvNeXt | — | Pure-convolution "modernized ResNet" |
| Hiera | — | Hierarchical ViT with masked-unit attention |
| EVA-02 | `load_eva02_model()` | ViT with SwiGLU MLP and RoPE |
| DINOv2 | `load_dinov2_model()` | Self-supervised ViT feature extractor |
| SigLIP | `load_siglip_model()` | Sigmoid-loss contrastive vision encoder |
| Qwen Vision | `load_qwen_vision_from_dir()` | Conv2d patch embed → ViT blocks → PatchMerger |

**Features:**
- Preprocessing: resize (bicubic/bilinear), center crop, ImageNet normalization — auto-detected from timm `pretrained_cfg`
- INT8 quantization: `QuantizedLinear` (W8A32) with per-channel symmetric quantization
- 2D RoPE for position-aware attention (EVA-02)
- Test-time augmentation: horizontal flip, five-crop, ten-crop
- Sparse inference (feature: `sparse`): per-block Deja Vu sparsity prediction for all architectures

#### Running DaViT Inference

```bash
# Download a DaViT model from timm
huggingface-cli download timm/davit_tiny.msft_in1k --local-dir davit_tiny

# Run inference on an image (requires `cli` feature)
cargo run --release -p klearu-vision --features cli --bin davit-infer -- \
    ./davit_tiny path/to/image.jpg

# With horizontal-flip TTA
cargo run --release -p klearu-vision --features cli --bin davit-infer -- \
    ./davit_tiny path/to/image.jpg --tta
```

#### Programmatic Usage

```rust
use klearu_vision::weight::load_vit_model;
use klearu_vision::preprocess::PreprocessConfig;

// Load a ViT model from a timm model directory
let model = load_vit_model("./vit_tiny")?;

// Preprocess: [C, H, W] f32 tensor, ImageNet-normalized
let image: Vec<f32> = /* your preprocessing */;
let logits = model.forward(&image);
```

#### Vision-Language Model (VLM) Bridge

The `klearu-llm` crate includes a `VlmBridge` that connects the Qwen Vision encoder to LLM inference, replacing `<image>` placeholder tokens with vision encoder outputs:

```rust
use klearu_llm::vlm::{VlmBridge, VlmImage};

let bridge = VlmBridge::new(vision_encoder, image_token_id,
    vision_start_token_id, vision_end_token_id);

// Encode image and inject into text embeddings
let image = VlmImage { data: chw_tensor, height: 448, width: 448 };
let merged = bridge.inject_vision_tokens(
    &token_ids, &text_embeddings, &[image], hidden_size,
);
```

#### Downloading Vision Models

```bash
# DaViT tiny (~44 MB)
huggingface-cli download timm/davit_tiny.msft_in1k --local-dir davit_tiny

# ViT tiny (~22 MB)
huggingface-cli download timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k --local-dir vit_tiny

# EVA-02 tiny (~22 MB)
huggingface-cli download timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k --local-dir eva02_tiny

# DINOv2 small (~84 MB)
huggingface-cli download timm/vit_small_patch14_dinov2.lvd142m --local-dir dinov2_small

# SigLIP base (~354 MB)
huggingface-cli download timm/vit_base_patch16_siglip_224.webli --local-dir siglip_base

# Qwen3.5-0.8B VLM (~1.8 GB, includes both vision encoder and LLM)
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B
```

### klearu-dpf — Distributed Point Functions

AES-based DPF using the BGI construction, plus DCF (Distributed Comparison Functions) via prefix decomposition into DPFs. Used as a building block for the MPC protocols.

### klearu-mpc — Two-Party Computation

Fixed-point arithmetic in Q16.16 (u32 shares) and Q32.32 (u64 shares), additive secret sharing, Beaver triple multiplication, polynomial SiLU approximation, and reveal-and-correct RMSNorm. Provides a `Transport` trait for abstracting communication.

### klearu-private — Private LLM Inference

End-to-end private inference combining Ferret COT (Correlated Oblivious Transfer), Ristretto255 OPRF, and the MPC building blocks. Two security levels:

| Level | Communication | Privacy | Speed |
|---|---|---|---|
| **Lower** | ~4.6 KB/token | Server learns nothing; client embedding revealed then plaintext forward | Fast |
| **High** | ~2 MB/token, ~34K triples | Only norms, queries, and gate values revealed | Slower |

## Running the LLM Demo

### 1. Download a Model

Klearu works with any HuggingFace LLaMA-architecture model in safetensors format. [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) models are a good starting point for testing:

```bash
# Install the HuggingFace CLI if you don't have it
pip install huggingface-hub

# Download SmolLM-135M-Instruct (~270 MB)
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct \
    --local-dir SmolLM-135M-Instruct

# Or a larger model — SmolLM-360M-Instruct (~720 MB)
huggingface-cli download HuggingFaceTB/SmolLM-360M-Instruct \
    --local-dir SmolLM-360M-Instruct

# Or SmolLM-1.7B-Instruct (~3.4 GB)
huggingface-cli download HuggingFaceTB/SmolLM-1.7B-Instruct \
    --local-dir SmolLM-1.7B-Instruct
```

The model directory should contain at minimum:
- `config.json` — HuggingFace model configuration
- `tokenizer.json` — Tokenizer
- `*.safetensors` — Model weights

### 2. Run the Chat Interface

```bash
# Basic chat (auto-detects chat template)
cargo run --release --bin chat -- ./SmolLM-135M-Instruct

# With custom sampling parameters
cargo run --release --bin chat -- ./SmolLM-135M-Instruct \
    --temp 0.8 --top-k 50 --top-p 0.95 --max-tokens 256

# With a system prompt
cargo run --release --bin chat -- ./SmolLM-135M-Instruct \
    --system "You are a helpful coding assistant."

# Force a specific chat template
cargo run --release --bin chat -- ./SmolLM-135M-Instruct \
    --template chatml
```

The chat binary starts an interactive loop — type your message and press Enter. Use Ctrl-D to quit.

### 3. Sparse Inference (Optional)

First calibrate sparsity predictors, then run with `--sparse`:

```bash
# Train predictors (requires sparse feature)
cargo run --release --features sparse --bin calibrate -- ./SmolLM-135M-Instruct \
    --samples 16 --epochs 100

# Chat with sparse inference
cargo run --release --features sparse --bin chat -- ./SmolLM-135M-Instruct \
    --sparse --head-sparsity 0.5 --neuron-sparsity 0.5
```

### 4. Model Diagnostics

Validate that a model loads and runs correctly:

```bash
cargo run --release --bin diagnose -- ./SmolLM-135M-Instruct
```

This checks config parsing, weight loading, tokenizer functionality, forward pass sanity, and greedy generation.

### 5. Private Two-Party Inference

Run inference where the server holds the model weights and the client's input tokens remain private:

```bash
# Terminal 1 — start the server
cargo run --release --bin private-server -- ./SmolLM-135M-Instruct \
    --port 9000 --security lower

# Terminal 2 — connect the client
cargo run --release --bin private-client -- ./SmolLM-135M-Instruct \
    --host localhost:9000 --security lower
```

For development and testing, add `--dummy-triples` to both sides to skip Ferret OT setup. For real security, omit this flag to use actual oblivious transfer.

## Feature Flags

The facade crate (`klearu`) provides feature-gated access to all functionality:

| Feature | Enables |
|---|---|
| `simd` | SIMD-accelerated dot products and scatter-add |
| `bf16` | BF16 quantization |
| `mongoose` | Learnable hashing and adaptive scheduling |
| `bolt` | LSH autotuning |
| `deja-vu` | Transformer sparsity prediction |
| `llm` | LLM inference engine |
| `full` | All of the above |

The `sparse` feature on `klearu-llm` enables Deja Vu sparse inference and the `calibrate` binary. The `sparse` feature on `klearu-vision` enables per-block sparsity prediction for all vision architectures. The `cli` feature on `klearu-vision` enables the `davit-infer` binary for image classification.
