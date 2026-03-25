#!/bin/bash
# =============================================================================
# A100 上使用 tmux 后台运行 SFT 训练的完整脚本
# =============================================================================
# 使用方法: bash scripts/start_sft_tmux.sh
# 或者直接运行下面的命令
# =============================================================================

echo "=========================================="
echo "检查 tmux 是否安装"
echo "=========================================="
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux 未安装，正在安装..."
    # Ubuntu/Debian
    sudo apt-get install -y tmux
    # 或 CentOS/RHEL
    # sudo yum install -y tmux
fi

echo ""
echo "=========================================="
echo "准备在 A100 上启动 SFT 训练"
echo "=========================================="
echo "环境: mathRL conda 环境"
echo "GPU: A100"
echo "训练类型: SFT (Supervised Fine-Tuning)"
echo "预计时间: 11-13 小时"
echo "检查点保存: 每 50 步"
echo "日志工具: SwanLab"
echo "=========================================="
echo ""

# 创建或连接到 tmux 会话
SESSION_NAME="sft_training"

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  tmux 会话 '$SESSION_NAME' 已存在"
    echo "选项："
    echo "  1. 连接到现有会话: tmux attach -t $SESSION_NAME"
    echo "  2. 杀掉旧会话重新开始: tmux kill-session -t $SESSION_NAME"
    read -p "是否杀掉旧会话并重新开始？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
        echo "✅ 已杀掉旧会话"
    else
        echo "连接到现有会话..."
        tmux attach -t $SESSION_NAME
        exit 0
    fi
fi

echo ""
echo "🚀 创建新的 tmux 会话: $SESSION_NAME"
echo ""
echo "提示："
echo "  - 断开会话: Ctrl+B 然后按 D"
echo "  - 重新连接: tmux attach -t $SESSION_NAME"
echo "  - 查看日志: tail -f /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log"
echo ""
echo "3 秒后启动训练..."
sleep 3

# 创建 tmux 会话并在其中运行训练
tmux new-session -d -s $SESSION_NAME

# 在 tmux 会话中执行训练命令
tmux send-keys -t $SESSION_NAME "cd /home/lyl/mathRL" C-m
tmux send-keys -t $SESSION_NAME "conda activate mathRL" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t $SESSION_NAME "export SAVE_STEPS=50" C-m
tmux send-keys -t $SESSION_NAME "export LOGGER=swanlab" C-m
tmux send-keys -t $SESSION_NAME "export EPOCHS=2.0" C-m
tmux send-keys -t $SESSION_NAME "export LORA_RANK=64" C-m
tmux send-keys -t $SESSION_NAME "export LEARNING_RATE=1e-5" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo '开始 A100 SFT 训练'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "date" C-m
tmux send-keys -t $SESSION_NAME "nvidia-smi" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "bash scripts/run_sft_qwen35.sh" C-m

echo ""
echo "✅ tmux 会话 '$SESSION_NAME' 已创建并启动训练"
echo ""
echo "=========================================="
echo "常用命令"
echo "=========================================="
echo "连接到会话:    tmux attach -t $SESSION_NAME"
echo "查看会话列表:  tmux ls"
echo "杀掉会话:      tmux kill-session -t $SESSION_NAME"
echo ""
echo "查看训练日志:  tail -f /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log"
echo "查看 GPU:      watch -n 1 nvidia-smi"
echo "查看检查点:    ls -lh /home/lyl/mathRL/outputs_qwen35/sft_model/"
echo "=========================================="
echo ""
echo "现在连接到 tmux 会话查看训练进度..."
sleep 2
tmux attach -t $SESSION_NAME
