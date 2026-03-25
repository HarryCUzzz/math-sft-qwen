#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B-Base}"

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "Model not found: $QWEN35_MODEL_PATH"
    exit 1
fi

python src_qwen35/evaluation.py "$@"
