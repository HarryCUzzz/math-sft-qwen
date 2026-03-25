#!/bin/bash
# =============================================================================
# Qwen3.5-4B 真正并行训练脚本
# =============================================================================
# 同时运行 SFT 和 GRPO 训练，各自占用一张 A6000
# - 进程 1: SFT 训练 (GPU 1)
# - 进程 2: GRPO 训练 (GPU 3) - 后台等待 SFT checkpoint
# 使用方法:
#   bash scripts/train_concurrent.sh
# =============================================================================

set -e

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

# 配置
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3.5-4B}"

# 两张 A6000 的 GPU ID
GPU_SFT=1    # SFT 单卡训练
GPU_GRPO=3   # GRPO 单卡训练

echo "=========================================="
echo "Qwen3.5-4B 真正并行训练"
echo "=========================================="
echo "配置:"
echo "  SFT 训练: GPU $GPU_SFT (A6000)"
echo "  GRPO 训练: GPU $GPU_GRPO (A6000)"
echo "  运行方式: 并行运行"
echo "=========================================="
echo ""

# 检查模型和数据
echo "📋 检查环境..."
if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "❌ 模型未找到"
    exit 1
fi
if [ ! -f "$PROJECT_DIR/data/filtered/sft_train.json" ]; then
    echo "❌ SFT 数据不存在"
    exit 1
fi
if [ ! -f "$PROJECT_DIR/data/rl_prompts/rl_train.json" ]; then
    echo "❌ RL 数据不存在"
    exit 1
fi
echo "✅ 环境检查通过"
echo ""

# 清理之前的输出
echo "🧹 清理之前的运行输出..."
rm -rf "$PROJECT_DIR/outputs_qwen35/sft_model"/*
rm -rf "$PROJECT_DIR/outputs_qwen35/grpo_model"/*
mkdir -p "$PROJECT_DIR/outputs_qwen35/"{sft_logs,grpo_logs}

# 启动 SFT 训练（前台）
run_sft() {
    echo "[SFT] $(date +%H:%M:%S) 启动 SFT 训练..."
    CUDA_VISIBLE_DEVICES=$GPU_SFT timeout 72h bash scripts/run_sft_qwen35.sh 2>&1 | sed 's/^/[SFT] /'
    echo "[SFT] $(date +%H:%M:%S) 训练完成"
}

# 启动 GRPO 训练（后台，带等待逻辑）
run_grpo() {
    echo "[GRPO] $(date +%H:%M:%S) GRPO 进程启动..."

    # 等待 SFT checkpoint 生成
    echo "[GRPO] 等待 SFT checkpoint..."
    TIMEOUT=300  # 5 分钟超时
    ELAPSED=0
    while [ ! -d "$PROJECT_DIR/outputs_qwen35/sft_model" ] || [ -z "$(ls -A $PROJECT_DIR/outputs_qwen35/sft_model 2>/dev/null)" ]; do
        if [ $ELAPSED -gt $TIMEOUT ]; then
            echo "[GRPO] ❌ SFT checkpoint 生成超时"
            return 1
        fi
        echo "[GRPO] $(date +%H:%M:%S) 等待中... ($ELAPSED/$TIMEOUT s)"
        sleep 10
        ELAPSED=$((ELAPSED + 10))
    done

    echo "[GRPO] ✅ SFT checkpoint 已准备，启动 GRPO..."
    CUDA_VISIBLE_DEVICES=$GPU_GRPO timeout 96h bash scripts/run_grpo_qwen35.sh 2>&1 | sed 's/^/[GRPO] /'
    echo "[GRPO] $(date +%H:%M:%S) 训练完成"
}

# 以后台进程启动 GRPO
run_grpo &
GRPO_PID=$!
echo "🟢 GRPO 进程 PID: $GRPO_PID (后台运行)"

# 前台运行 SFT
echo "🟡 SFT 进程启动（前台）..."
run_sft
SFT_EXIT_CODE=$?

# 等待 GRPO 完成
echo ""
echo "⏳ 等待 GRPO 训练完成..."
wait $GRPO_PID
GRPO_EXIT_CODE=$?

# 检查结果
echo ""
echo "=========================================="
echo "📊 训练结果"
echo "=========================================="

if [ $SFT_EXIT_CODE -eq 0 ]; then
    echo "✅ SFT 训练成功"
else
    echo "❌ SFT 训练失败 (退出码: $SFT_EXIT_CODE)"
fi

if [ $GRPO_EXIT_CODE -eq 0 ]; then
    echo "✅ GRPO 训练成功"
else
    echo "❌ GRPO 训练失败 (退出码: $GRPO_EXIT_CODE)"
fi

echo ""
echo "📁 输出位置:"
echo "  - SFT 模型: $PROJECT_DIR/outputs_qwen35/sft_model/"
echo "  - GRPO 模型: $PROJECT_DIR/outputs_qwen35/grpo_model/"
echo "  - 训练日志: $PROJECT_DIR/outputs_qwen35/{sft,grpo}_logs/"
echo "=========================================="

# 返回最终状态
if [ $SFT_EXIT_CODE -eq 0 ] && [ $GRPO_EXIT_CODE -eq 0 ]; then
    echo "✅ 所有训练完成！"
    exit 0
else
    echo "❌ 部分训练失败"
    exit 1
fi
