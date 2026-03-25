#!/bin/bash
# =============================================================================
# Qwen3.5-4B GRPO 训练启动脚本（增强版）
# =============================================================================
# 自动检测 GPU 数量，选择单卡或多卡配置
# 使用方法:
#   基础: CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
#   快速: MAX_STEPS=300 NUM_GENERATIONS=6 CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
#   量化: USE_4BIT=true CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
#   SwanLab: LOGGER=swanlab CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
#   自定义: MAX_STEPS=400 TEMPERATURE=0.8 BETA=0.02 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_grpo_qwen35.sh
# =============================================================================

set -e

# ==================== 基础配置 ====================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"
export USE_4BIT="${USE_4BIT:-false}"
export LOGGER="${LOGGER:-tensorboard}"  # 日志工具: tensorboard/swanlab/wandb/none

# ==================== 训练参数（可通过环境变量覆盖）====================
export MAX_STEPS="${MAX_STEPS:-500}"                    # 训练步数 (200-600, 推荐 400-500)
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"          # 每步生成数 (4-8, 推荐 6-8)
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"  # 生成最大长度
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"          # 学习率 (1e-8 到 1e-6)
export TEMPERATURE="${TEMPERATURE:-0.7}"                # 生成温度 (0.6-0.9)
export BETA="${BETA:-0.04}"                            # KL 散度惩罚系数 (0.01-0.1)
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"        # 梯度累积（留空自动）
export SAVE_STEPS="${SAVE_STEPS:-100}"                 # 保存检查点频率

# 项目根目录
PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

# 检查 GPU 数量
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 自动设置梯度累积步数（如果未指定）
if [ -z "$GRAD_ACCUM_STEPS" ]; then
    if [ "$NUM_GPUS" -gt 1 ]; then
        GRAD_ACCUM_STEPS=4   # 多卡：较少累积
    else
        GRAD_ACCUM_STEPS=8   # 单卡：较多累积
    fi
    export GRAD_ACCUM_STEPS
fi

echo "=========================================="
echo "Qwen3.5-4B GRPO Training (增强配置)"
echo "=========================================="
echo "🖥️  硬件配置:"
echo "  GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS 卡)"
echo "  模型路径: $QWEN35_MODEL_PATH"
echo "  4-bit 量化: $USE_4BIT"
echo "  日志工具: $LOGGER"
echo ""
echo "📊 训练参数:"
echo "  Max Steps: $MAX_STEPS"
echo "  Generations per Step: $NUM_GENERATIONS"
echo "  Max Completion Length: $MAX_COMPLETION_LENGTH tokens"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Temperature: $TEMPERATURE"
echo "  Beta (KL penalty): $BETA"
echo "  Gradient Accumulation: $GRAD_ACCUM_STEPS steps"
echo "  Save Every: $SAVE_STEPS steps"
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
