# A100 后台运行 SFT 训练操作指南

## ✅ 路径检查结果

已为你创建软链接，代码路径**无需修改**：

```bash
/home/lyl/mathRL -> /home/data5/lyl/mathRL
/home/lyl/models -> /home/data5/lyl/models
```

所有原有代码路径 `/home/lyl/...` 会自动指向新位置 `/home/data5/lyl/...`

---

## 🚀 方案 1: 一键启动（推荐）

```bash
cd /home/lyl/mathRL
bash scripts/start_sft_tmux.sh
```

**这个脚本会自动：**
1. ✅ 检查并安装 tmux（如需要）
2. ✅ 创建 tmux 会话 `sft_training`
3. ✅ 配置所有环境变量（A100, 50步保存, SwanLab）
4. ✅ 启动 SFT 训练
5. ✅ 自动连接到会话查看进度

**断开会话保持训练继续运行**：按 `Ctrl+B` 然后按 `D`

**重新连接查看进度**：`tmux attach -t sft_training`

---

## 🛠️ 方案 2: 手动操作（完整控制）

### Step 1: 创建 tmux 会话

```bash
# 创建名为 sft_training 的会话
tmux new -s sft_training
```

### Step 2: 在 tmux 会话中配置并启动训练

```bash
# 进入项目目录
cd /home/lyl/mathRL

# 配置环境变量（A100 单卡）
export CUDA_VISIBLE_DEVICES=0
export SAVE_STEPS=50           # 每 50 步保存，防止中断丢失进度
export LOGGER=swanlab           # 使用 SwanLab 实时监控
export EPOCHS=2.0               # 2 轮训练
export LORA_RANK=64             # LoRA 秩
export LEARNING_RATE=1e-5       # 学习率

# 确认配置
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "保存间隔: 每 $SAVE_STEPS 步"
echo "日志工具: $LOGGER"
nvidia-smi

# 启动训练
bash scripts/run_sft_qwen35.sh
```

### Step 3: 断开会话（训练继续）

在 tmux 会话中按：
- `Ctrl+B` （先按下 Ctrl 和 B）
- 然后按 `D` （松开前面的键后，单独按 D）

你会看到：`[detached (from session sft_training)]`

此时训练在后台继续运行！

### Step 4: 重新连接查看进度

```bash
# 重新连接到 tmux 会话
tmux attach -t sft_training

# 或简写
tmux a -t sft_training
```

---

## 📊 监控训练进度

### 方式 1: 查看实时日志

```bash
# 新开一个终端窗口
tail -f /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log
```

### 方式 2: 查看 SwanLab 网页

训练启动后，日志中会显示 SwanLab 实验链接：
```
SwanLab 初始化成功: 项目=Qwen3.5-4B-MathRL
实验链接: https://swanlab.cn/@your-workspace/Qwen3.5-4B-MathRL/runs/xxx
```

复制链接到浏览器，实时查看训练曲线！

### 方式 3: 查看 GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi
```

### 方式 4: 查看检查点

```bash
# 列出已保存的检查点
ls -lh /home/lyl/mathRL/outputs_qwen35/sft_model/

# 实时查看检查点变化
watch -n 10 "ls -lht /home/lyl/mathRL/outputs_qwen35/sft_model/ | head -10"
```

---

## 🔧 常用 tmux 命令

| 命令 | 说明 |
|------|------|
| `tmux new -s NAME` | 创建新会话 |
| `tmux ls` | 列出所有会话 |
| `tmux attach -t NAME` | 连接到会话 |
| `tmux kill-session -t NAME` | 杀掉会话 |
| **在会话内** | |
| `Ctrl+B` 然后 `D` | 断开会话（训练继续） |
| `Ctrl+B` 然后 `[` | 进入滚动模式（查看历史输出） |
| `Ctrl+B` 然后 `C` | 在会话中创建新窗口 |
| `Ctrl+B` 然后 `0-9` | 切换到第N个窗口 |

---

## 🎯 完整训练流程示例

```bash
# ============ 方式 1: 自动脚本 ============
cd /home/lyl/mathRL
bash scripts/start_sft_tmux.sh
# 按 Ctrl+B 然后 D 断开，去干其他事

# ============ 方式 2: 手动操作 ============
# Step 1: 创建 tmux 会话
tmux new -s sft_training

# Step 2: 在 tmux 中运行训练
cd /home/lyl/mathRL
export CUDA_VISIBLE_DEVICES=0
export SAVE_STEPS=50
export LOGGER=swanlab
bash scripts/run_sft_qwen35.sh

# Step 3: 断开会话（按键操作）
# Ctrl+B, D

# Step 4: 在其他终端查看日志
tail -f /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log

# Step 5: 11-13 小时后，重新连接查看结果
tmux attach -t sft_training
```

---

## ⚠️ 重要提示

### 1. 如果训练再次中断

由于设置了 `SAVE_STEPS=50`，最多只会丢失 50 步进度。重新运行训练脚本会**自动从最近的检查点恢复**：

```bash
# 直接重新运行，会自动恢复
tmux new -s sft_training
cd /home/lyl/mathRL
export CUDA_VISIBLE_DEVICES=0
export SAVE_STEPS=50
export LOGGER=swanlab
bash scripts/run_sft_qwen35.sh
```

日志会显示：
```
从 checkpoint 恢复训练: /home/lyl/mathRL/outputs_qwen35/sft_model/checkpoint-250
```

### 2. 检查点占用空间

每 50 步保存一次，每个检查点约 3-5GB，保留最近 5 个（约 20GB）

### 3. 预计训练时间

- **A100 单卡**: 11-13 小时
- **训练集大小**: 320,723 条数据
- **总步数**: 约 20,045 步（2 epochs）
- **检查点数量**: 约 400 个（但只保留最近 5 个）

### 4. 完成后的操作

```bash
# 训练完成后，关闭 tmux 会话
tmux kill-session -t sft_training

# 或者在 tmux 会话内直接退出
exit
```

---

## 📋 故障排查

### Q1: tmux 命令不存在

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tmux

# CentOS/RHEL
sudo yum install -y tmux
```

### Q2: 无法连接到会话

```bash
# 查看所有会话
tmux ls

# 如果会话名不对，使用正确的名字
tmux attach -t 实际会话名
```

### Q3: 训练没有启动

```bash
# 重新连接到 tmux
tmux attach -t sft_training

# 查看错误信息
tail -100 /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log
```

### Q4: 想停止训练

```bash
# 方式 1: 连接到 tmux 并按 Ctrl+C
tmux attach -t sft_training
# 然后按 Ctrl+C

# 方式 2: 直接杀掉会话
tmux kill-session -t sft_training
```

### Q5: SwanLab 未安装

```bash
pip install swanlab
swanlab login
```

---

## ✅ 快速开始（复制粘贴）

```bash
# 方案 1: 一键启动（推荐）
cd /home/lyl/mathRL && bash scripts/start_sft_tmux.sh

# 方案 2: 手动启动（5 行命令）
tmux new -s sft_training
cd /home/lyl/mathRL
export CUDA_VISIBLE_DEVICES=0 SAVE_STEPS=50 LOGGER=swanlab
bash scripts/run_sft_qwen35.sh
# 按 Ctrl+B 然后 D 断开

# 重新连接
tmux attach -t sft_training

# 查看日志
tail -f /home/lyl/mathRL/outputs_qwen35/sft_logs/sft_training.log
```

---

## 🎓 总结

1. ✅ **路径已修复** - 软链接已建立，无需改代码
2. ✅ **自动脚本** - `bash scripts/start_sft_tmux.sh` 一键启动
3. ✅ **断点续传** - 每 50 步保存，自动恢复
4. ✅ **实时监控** - SwanLab 网页查看训练曲线
5. ✅ **后台运行** - tmux 保护训练不受终端关闭影响

**现在就开始吧！** 🚀
