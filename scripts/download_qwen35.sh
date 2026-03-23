#!/bin/bash
# =============================================================================
# Qwen3.5-4B 模型下载脚本 (最终精准版)
# =============================================================================
# 下载 HuggingFace 官方 Qwen/Qwen3.5-4B 模型
# 使用方法: bash scripts/download_qwen35.sh
# =============================================================================

set -e

# 配置：精准指向 Qwen3.5-4B 官方名称
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"  # 你指出的正确 HF 模型名
TARGET_DIR="${TARGET_DIR:-/home/lyl/models/Qwen/Qwen3.5-4B}"  # 规范目录名
USE_MODELSCOPE="${USE_MODELSCOPE:-false}"

echo "=========================================="
echo "Qwen 模型下载脚本"
echo "=========================================="
echo "模型: $MODEL_NAME (官方 Qwen3.5-4B)"
echo "目标目录: $TARGET_DIR"
echo "=========================================="

# 创建目标目录
mkdir -p "$TARGET_DIR"

if [ "$USE_MODELSCOPE" = "true" ]; then
    # ModelScope 下载（国内网络推荐，名称也对应 Qwen3.5-4B）
    echo "使用 ModelScope 下载..."
    if ! python -c "import modelscope" 2>/dev/null; then
        echo "安装 ModelScope 依赖..."
        pip install modelscope
    fi
    MS_MODEL_NAME="qwen/Qwen3.5-4B"  # ModelScope 官方名称
    python -c "
from modelscope import snapshot_download
import os
os.environ['MODELSCOPE_CACHE'] = '$TARGET_DIR'
snapshot_download('$MS_MODEL_NAME', cache_dir='$TARGET_DIR')
print('下载完成!')
"
else
    # HuggingFace 下载（核心修复 + 精准模型名）
    echo "使用 HuggingFace 下载..."
    # 强制使用 HF 镜像（国内访问必备）
    if [ -z "$HF_ENDPOINT" ]; then
        export HF_ENDPOINT="https://hf-mirror.com"
        echo "使用镜像: $HF_ENDPOINT"
    fi

    # 兼容所有环境的下载命令（解决 command not found）
    python -m huggingface_hub.cli download \
        "$MODEL_NAME" \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False \
        --endpoint "$HF_ENDPOINT" \
        --skip-cached-files  # 跳过已下载文件，断点续传
fi

echo "=========================================="
echo "下载完成!"
echo "模型路径: $TARGET_DIR"
echo "=========================================="

# 严格验证 Qwen3.5-4B 完整性
if [ -f "$TARGET_DIR/config.json" ]; then
    echo "✅ 验证通过: config.json 存在"
else
    echo "❌ 验证失败: config.json 不存在，下载不完整"
    exit 1
fi

# 检查权重文件（Qwen3.5-4B 主要用 .safetensors 格式）
if ls "$TARGET_DIR"/*.safetensors 1> /dev/null 2>&1; then
    echo "✅ 验证通过: .safetensors 权重文件存在"
elif ls "$TARGET_DIR"/*.bin 1> /dev/null 2>&1; then
    echo "✅ 验证通过: .bin 权重文件存在"
else
    echo "❌ 验证失败: 未找到任何权重文件"
    exit 1
fi
