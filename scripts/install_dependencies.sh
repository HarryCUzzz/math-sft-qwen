#!/bin/bash
# =============================================================================
# 快速安装所有依赖（用于训练 Qwen3.5-4B）
# =============================================================================

set -e

echo "=========================================="
echo "检查环境"
echo "=========================================="
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  未检测到 NVIDIA GPU"
fi
echo ""

echo "=========================================="
echo "安装 PyTorch (CUDA 12.1)"
echo "=========================================="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=========================================="
echo "安装核心依赖"
echo "=========================================="
pip install transformers>=4.44.0
pip install datasets>=2.19.0
pip install accelerate>=0.30.0
pip install peft>=0.11.0
pip install trl>=0.9.0

echo ""
echo "=========================================="
echo "安装辅助工具"
echo "=========================================="
pip install pandas numpy matplotlib
pip install sentencepiece protobuf safetensors
pip install tensorboard tqdm scipy

echo ""
echo "=========================================="
echo "安装 SwanLab（可选，用于可视化）"
echo "=========================================="
pip install swanlab

echo ""
echo "=========================================="
echo "安装 BitsAndBytes（4-bit 量化支持）"
echo "=========================================="
pip install bitsandbytes

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
echo "检查关键包..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import accelerate; print(f'✅ Accelerate {accelerate.__version__}')"
python -c "import peft; print(f'✅ PEFT {peft.__version__}')"
python -c "import trl; print(f'✅ TRL {trl.__version__}')"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo "现在可以开始训练了："
echo "  cd /home/lyl/mathRL"
echo "  bash scripts/start_sft_tmux.sh"
echo "=========================================="
