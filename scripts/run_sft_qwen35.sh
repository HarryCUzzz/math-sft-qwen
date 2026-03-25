#!/bin/bash
# =============================================================================
# Qwen3.5-4B SFT 训练启动脚本（增强版）
# =============================================================================
# 自动检测 GPU 数量，选择单卡或多卡配置
# 使用方法:
#   基础: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_sft_qwen35.sh
#   快速: EPOCHS=1.5 LORA_RANK=32 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_sft_qwen35.sh
#   量化: USE_4BIT=true CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
#   SwanLab: LOGGER=swanlab CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
#   自定义: EPOCHS=2.0 LEARNING_RATE=2e-5 MAX_LENGTH=1536 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_sft_qwen35.sh
# =============================================================================

set -e

# ==================== 基础配置 ====================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"
export USE_4BIT="${USE_4BIT:-false}"
export LOGGER="${LOGGER:-tensorboard}"  # 日志工具: tensorboard/swanlab/wandb/none

# ==================== 训练参数（可通过环境变量覆盖）====================
export EPOCHS="${EPOCHS:-2.0}"                    # 训练轮数 (1.0-3.0, 推荐 2.0)
export LORA_RANK="${LORA_RANK:-64}"               # LoRA 秩 (16/32/64, 推荐 64)
export LORA_ALPHA="${LORA_ALPHA:-128}"            # LoRA alpha (建议 2×rank)
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"    # 学习率 (1e-6 到 5e-5)
export MAX_LENGTH="${MAX_LENGTH:-2048}"           # 最大序列长度 (512-2048)
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"  # 梯度累积步数（留空则自动）
export SAVE_STEPS="${SAVE_STEPS:-200}"           # 保存检查点频率
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"      # 预热比例

# 项目根目录
PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

# 检查 GPU 数量
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 自动设置梯度累积步数（如果未指定）
if [ -z "$GRAD_ACCUM_STEPS" ]; then
    if [ "$NUM_GPUS" -gt 1 ]; then
        GRAD_ACCUM_STEPS=16  # 多卡：每卡 batch=1, 累积 16 步 → 有效 batch=32
    else
        GRAD_ACCUM_STEPS=32  # 单卡：batch=1, 累积 32 步 → 有效 batch=32
    fi
    export GRAD_ACCUM_STEPS
fi

echo "=========================================="
echo "Qwen3.5-4B SFT Training (增强配置)"
echo "=========================================="
echo "🖥️  硬件配置:"
echo "  GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS 卡)"
echo "  模型路径: $QWEN35_MODEL_PATH"
echo "  4-bit 量化: $USE_4BIT"
echo "  日志工具: $LOGGER"
echo ""
echo "📊 训练参数:"
echo "  Epochs: $EPOCHS"
echo "  LoRA Rank: $LORA_RANK (alpha=$LORA_ALPHA)"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Length: $MAX_LENGTH tokens"
echo "  Gradient Accumulation: $GRAD_ACCUM_STEPS steps"
echo "  Save Every: $SAVE_STEPS steps"
echo "  Warmup Ratio: $WARMUP_RATIO"
echo "=========================================="

# 检查模型是否存在
if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "错误: 模型未找到: $QWEN35_MODEL_PATH"
    echo "请先运行: bash scripts/download_qwen35.sh"
    exit 1
fi

# 检查数据是否存在
if [ ! -f "$PROJECT_DIR/data/filtered/sft_train.json" ]; then
    echo "错误: 训练数据未找到"
    echo "请确保 data/filtered/sft_train.json 存在"
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
echo "启动 SFT 训练..."
echo ""

accelerate launch \
    --config_file "$CONFIG_FILE" \
    src_qwen35/sft_training.py --mode "$MODE"

echo ""
echo "=========================================="
echo "SFT 训练完成!"
echo "模型保存至: $PROJECT_DIR/outputs_qwen35/sft_model/"
echo "=========================================="
