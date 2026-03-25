#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B-Base}"
TARGET_DIR="${TARGET_DIR:-/home/lyl/models/Qwen/Qwen3.5-4B-Base}"
USE_MODELSCOPE="${USE_MODELSCOPE:-false}"

echo "Model: $MODEL_NAME"
echo "Target dir: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

if [ "$USE_MODELSCOPE" = "true" ]; then
    if ! python -c "import modelscope" >/dev/null 2>&1; then
        pip install modelscope
    fi
    python -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen3.5-4B-Base', cache_dir='/tmp/qwen35_base_cache'))"
else
    if [ -z "$HF_ENDPOINT" ]; then
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
    python -m huggingface_hub.cli download "$MODEL_NAME" --local-dir "$TARGET_DIR" --local-dir-use-symlinks False --endpoint "$HF_ENDPOINT"
fi
