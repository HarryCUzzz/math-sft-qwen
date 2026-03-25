#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B-Base}"
export USE_4BIT="${USE_4BIT:-false}"
export LOGGER="${LOGGER:-tensorboard}"
export MAX_STEPS="${MAX_STEPS:-300}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
export LEARNING_RATE="${LEARNING_RATE:-8e-7}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
export BETA="${BETA:-0.04}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export SAVE_STEPS="${SAVE_STEPS:-100}"

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/outputs_qwen35/sft_model/adapter_config.json" ]; then
    echo "SFT adapter not found. Run SFT first."
    exit 1
fi
if [ ! -f "$PROJECT_DIR/data/qwen35_v2/processed/rl_train.jsonl" ]; then
    echo "Cleaned RL dataset not found. Run scripts/run_prepare_data_qwen35.sh first."
    exit 1
fi

accelerate launch --config_file "$PROJECT_DIR/configs/accelerate/single_gpu.yaml" src_qwen35/grpo_training.py --mode single
