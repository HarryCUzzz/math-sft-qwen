#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"
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
export NUMERIC_CLOSE_REWARD="${NUMERIC_CLOSE_REWARD:-0.35}"
export PARSED_WRONG_REWARD="${PARSED_WRONG_REWARD:-0.10}"
export PARSE_BONUS="${PARSE_BONUS:-0.03}"
export INVALID_ANSWER_PENALTY="${INVALID_ANSWER_PENALTY:--0.10}"
export FINAL_STRUCTURE_BONUS="${FINAL_STRUCTURE_BONUS:-0.02}"
export LENGTH_PENALTY_START_TOKENS="${LENGTH_PENALTY_START_TOKENS:-320}"
export LENGTH_PENALTY_WEIGHT="${LENGTH_PENALTY_WEIGHT:-0.10}"
export QWEN35_EXPERIMENT_TAG="${QWEN35_EXPERIMENT_TAG:-}"

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

DATA_ROOT="$PROJECT_DIR/data/qwen35_v2"
OUTPUT_ROOT="$PROJECT_DIR/outputs_qwen35"
if [ -n "$QWEN35_EXPERIMENT_TAG" ]; then
    DATA_ROOT="$DATA_ROOT/$QWEN35_EXPERIMENT_TAG"
    OUTPUT_ROOT="$OUTPUT_ROOT/$QWEN35_EXPERIMENT_TAG"
fi

if [ ! -f "$OUTPUT_ROOT/sft_model/adapter_config.json" ]; then
    echo "SFT adapter not found: $OUTPUT_ROOT/sft_model/adapter_config.json"
    echo "Run SFT first."
    exit 1
fi
if [ ! -f "$DATA_ROOT/processed/rl_train.jsonl" ]; then
    echo "Cleaned RL dataset not found: $DATA_ROOT/processed/rl_train.jsonl"
    echo "Run scripts/run_prepare_data_qwen35.sh first."
    exit 1
fi

echo "Experiment tag: ${QWEN35_EXPERIMENT_TAG:-default}"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"

accelerate launch --config_file "$PROJECT_DIR/configs/accelerate/single_gpu.yaml" src_qwen35/grpo_training.py --mode single
