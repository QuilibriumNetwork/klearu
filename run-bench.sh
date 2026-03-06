#!/usr/bin/env bash
#
# Run klearu-llm chat benchmark.
#
# Usage:
#   ./run-bench.sh <model_dir> [options]
#
# Options:
#   --sparse              Enable sparse inference (requires 'sparse' feature)
#   --head-sparsity N     Fraction of heads to keep (default: 0.5)
#   --neuron-sparsity N   Fraction of neurons to keep (default: 0.5)
#   --system "msg"        System prompt
#   --temp N              Temperature (default: 0.7)
#   --top-k N             Top-k sampling (default: 40)
#   --top-p N             Top-p sampling (default: 0.9)
#   --max-tokens N        Max tokens to generate (default: 512)
#   --template NAME       Chat template (auto, chatml, llama3, etc.)
#
# Examples:
#   ./run-bench.sh ./Qwen3.5-0.8B
#   ./run-bench.sh ./Qwen3.5-0.8B --sparse --head-sparsity 0.5 --neuron-sparsity 0.5
#   ./run-bench.sh ./Qwen3.5-0.8B --system "You are a helpful assistant."

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_dir> [options]" >&2
    echo "  Options: --sparse --head-sparsity N --neuron-sparsity N --system 'msg'" >&2
    echo "           --temp N --top-k N --top-p N --max-tokens N --template NAME" >&2
    exit 1
fi

MODEL_DIR="$1"
shift

# Check if --sparse is among the flags to determine features
FEATURES=""
for arg in "$@"; do
    if [ "$arg" = "--sparse" ]; then
        FEATURES="--features sparse"
        break
    fi
done

cargo run -p klearu-llm --bin chat --release $FEATURES -- "$MODEL_DIR" "$@"
