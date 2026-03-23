#!/bin/bash
# =============================================================================
# Qwen3.5-4B GRPO 训练启动脚本
# =============================================================================
# 自动检测 GPU 数量，选择单卡或多卡配置
# 使用方法:
#   单卡: CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
#   双卡: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_grpo_qwen35.sh
#   4-bit 量化: USE_4BIT=true CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
# =============================================================================

set -e

# 配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"
export USE_4BIT="${USE_4BIT:-false}"  # 4-bit 量化选项

# 项目根目录
PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

# 检查 GPU 数量
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Qwen3.5-4B GRPO Training"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS 卡)"
echo "模型路径: $QWEN35_MODEL_PATH"
echo "4-bit 量化: $USE_4BIT"
echo "=========================================="

# 检查模型是否存在
if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "错误: 模型未找到: $QWEN35_MODEL_PATH"
    echo "请先运行: bash scripts/download_qwen35.sh"
    exit 1
fi

# 检查 SFT 模型是否存在
SFT_MODEL_DIR="$PROJECT_DIR/outputs_qwen35/sft_model"
if [ ! -f "$SFT_MODEL_DIR/adapter_config.json" ]; then
    echo "警告: SFT adapter 未找到: $SFT_MODEL_DIR"
    echo "将直接在基座模型上进行 GRPO 训练"
fi

# 检查 RL 数据是否存在
if [ ! -f "$PROJECT_DIR/data/rl_prompts/rl_train.json" ]; then
    echo "错误: RL 训练数据未找到"
    echo "请确保 data/rl_prompts/rl_train.json 存在"
    exit 1
fi

# 根据 GPU 数量选择配置
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "使用多卡训练配置..."
    MODE="multi"
    CONFIG_FILE="$PROJECT_DIR/configs/accelerate/multi_gpu.yaml"
else
    echo "使用单卡训练配置..."
    MODE="single"
    CONFIG_FILE="$PROJECT_DIR/configs/accelerate/single_gpu.yaml"
fi

# 启动训练
echo ""
echo "启动 GRPO 训练..."
echo ""

accelerate launch \
    --config_file "$CONFIG_FILE" \
    src_qwen35/grpo_training.py --mode "$MODE"

echo ""
echo "=========================================="
echo "GRPO 训练完成!"
echo "模型保存至: $PROJECT_DIR/outputs_qwen35/grpo_model/"
echo "=========================================="
