#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"
export USE_4BIT="${USE_4BIT:-false}"
export LOGGER="${LOGGER:-tensorboard}"
export EPOCHS="${EPOCHS:-1.0}"
export MAX_STEPS="${MAX_STEPS:-2200}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export LEARNING_RATE="${LEARNING_RATE:-8e-5}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export QWEN35_EXPERIMENT_TAG="${QWEN35_EXPERIMENT_TAG:-}"
export SFT_STAGE="${SFT_STAGE:-main}"

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "Model not found: $QWEN35_MODEL_PATH"
    exit 1
fi

DATA_ROOT="$PROJECT_DIR/data/qwen35_v2"
OUTPUT_ROOT="$PROJECT_DIR/outputs_qwen35"
if [ -n "$QWEN35_EXPERIMENT_TAG" ]; then
    DATA_ROOT="$DATA_ROOT/$QWEN35_EXPERIMENT_TAG"
    OUTPUT_ROOT="$OUTPUT_ROOT/$QWEN35_EXPERIMENT_TAG"
fi

TRAIN_FILE="$DATA_ROOT/processed/sft_train.jsonl"
if [ "$SFT_STAGE" = "calibration" ]; then
    TRAIN_FILE="$DATA_ROOT/processed/sft_calibration_train.jsonl"
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Cleaned SFT dataset not found: $TRAIN_FILE"
    echo "Run scripts/run_prepare_data_qwen35.sh first."
    exit 1
fi

echo "Experiment tag: ${QWEN35_EXPERIMENT_TAG:-default}"
echo "SFT stage: ${SFT_STAGE}"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"

accelerate launch --config_file "$PROJECT_DIR/configs/accelerate/single_gpu.yaml" src_qwen35/sft_training.py --mode single
