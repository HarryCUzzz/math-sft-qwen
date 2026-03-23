#!/bin/bash
# =============================================================================
# Qwen3.5-4B 评估启动脚本
# =============================================================================
# 在 MATH-500、GSM8K、TheoremQA 上评估 base/sft/grpo 模型
# 使用方法:
#   完整评估: bash scripts/run_eval_qwen35.sh
#   仅评估 base 模型: bash scripts/run_eval_qwen35.sh --models base
#   跳过某个数据集: bash scripts/run_eval_qwen35.sh --skip-datasets theoremqa
# =============================================================================

set -e

# 配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/home/lyl/models/Qwen/Qwen3___5-4B}"

# 项目根目录
PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Qwen3.5-4B Evaluation"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "模型路径: $QWEN35_MODEL_PATH"
echo "=========================================="

# 检查模型是否存在
if [ ! -f "$QWEN35_MODEL_PATH/config.json" ]; then
    echo "错误: 模型未找到: $QWEN35_MODEL_PATH"
    echo "请先运行: bash scripts/download_qwen35.sh"
    exit 1
fi

# 启动评估
echo ""
echo "启动评估..."
echo ""

python src_qwen35/evaluation.py "$@"

echo ""
echo "=========================================="
echo "评估完成!"
echo "结果保存至: $PROJECT_DIR/outputs_qwen35/eval_results/"
echo "=========================================="
