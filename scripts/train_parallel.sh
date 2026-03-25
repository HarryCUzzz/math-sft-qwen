#!/bin/bash
# =============================================================================
# Qwen3.5-4B 双卡并行训练脚本
# =============================================================================
# 使用两张 A6000 并行运行 SFT 和 GRPO 训练
# 使用方法:
#   bash scripts/train_parallel.sh [sft|grpo|both]
# =============================================================================

set -e

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

# 配置
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"

# 两张 A6000 的 GPU ID
GPU_A6000_1=1
GPU_A6000_2=3

# 操作类型
OPERATION="${1:-both}"  # sft, grpo, or both

echo "=========================================="
echo "Qwen3.5-4B 双卡并行训练"
echo "=========================================="
echo "GPU 配置: GPU $GPU_A6000_1 + GPU $GPU_A6000_2 (两张 A6000)"
echo "操作: $OPERATION"
echo "=========================================="
echo ""

# 检查模型
if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "❌ 错误: 模型未找到: $QWEN35_MODEL_PATH"
    exit 1
fi

# 检查数据
if [ ! -f "$PROJECT_DIR/data/filtered/sft_train.json" ]; then
    echo "❌ 错误: SFT 训练数据不存在"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/data/rl_prompts/rl_train.json" ]; then
    echo "❌ 错误: RL 训练数据不存在"
    exit 1
fi

# 启动 SFT 训练函数
run_sft() {
    echo "🔵 [$(date +%H:%M:%S)] 启动 SFT 训练 (GPU $GPU_A6000_1, $GPU_A6000_2)..."
    CUDA_VISIBLE_DEVICES=$GPU_A6000_1,$GPU_A6000_2 bash scripts/run_sft_qwen35.sh
    echo "✅ [$(date +%H:%M:%S)] SFT 训练完成"
}

# 启动 GRPO 训练函数
run_grpo() {
    echo "🔵 [$(date +%H:%M:%S)] 启动 GRPO 训练 (GPU $GPU_A6000_1, $GPU_A6000_2)..."
    echo "⚠️  等待 SFT 模型就绪..."

    # 等待 SFT 模型生成
    while [ ! -d "$PROJECT_DIR/outputs_qwen35/sft_model" ]; do
        echo "   等待 SFT 模型检查点..."
        sleep 10
    done

    echo "✅ SFT 模型已准备就绪，启动 GRPO..."
    CUDA_VISIBLE_DEVICES=$GPU_A6000_1,$GPU_A6000_2 bash scripts/run_grpo_qwen35.sh
    echo "✅ [$(date +%H:%M:%S)] GRPO 训练完成"
}

# 执行操作
case "$OPERATION" in
    sft)
        run_sft
        ;;
    grpo)
        run_grpo
        ;;
    both)
        echo "📊 启动顺序训练: SFT → GRPO"
        echo ""
        run_sft
        echo ""
        run_grpo
        ;;
    *)
        echo "❌ 未知操作: $OPERATION"
        echo "使用方法: bash scripts/train_parallel.sh [sft|grpo|both]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo "输出位置:"
echo "  - SFT 模型: $PROJECT_DIR/outputs_qwen35/sft_model/"
echo "  - GRPO 模型: $PROJECT_DIR/outputs_qwen35/grpo_model/"
echo "  - 训练日志: $PROJECT_DIR/outputs_qwen35/{sft,grpo}_logs/"
echo "=========================================="
