# 训练日志与可视化指南

## 📊 支持的日志工具

| 工具 | 说明 | 优势 | 安装 |
|------|------|------|------|
| **TensorBoard** ⭐ | PyTorch 官方工具 | 本地运行，无需注册 | 已内置 |
| **SwanLab** | 国内开发的实验跟踪平台 | 中文支持，云端存储，实时查看 | `pip install swanlab` |
| **WandB** | Weights & Biases | 功能最强，团队协作 | `pip install wandb` |
| **None** | 禁用可视化 | 节省开销 | - |

---

## 🚀 快速开始

### 方案 1: 使用 TensorBoard（默认）⭐

**无需任何配置，开箱即用！**

```bash
# 启动训练（默认使用 TensorBoard）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
```

**查看可视化**：
```bash
# 新终端窗口
tensorboard --logdir outputs_qwen35/sft_model --port 6006

# 浏览器访问: http://localhost:6006
```

**可以看到的内容**：
- 📉 Loss 曲线
- 📈 Learning Rate 变化
- 🔥 Gradient Norm
- ⏱️ 训练速度 (samples/s)

---

### 方案 2: 使用 SwanLab（推荐，更现代）✨

**优势**：
- ✅ 中文界面
- ✅ 云端自动保存
- ✅ 实时查看（无需启动服务）
- ✅ 实验对比功能
- ✅ 支持手机查看

#### Step 1: 安装 SwanLab

```bash
pip install swanlab
```

#### Step 2: 登录/注册（首次使用）

```bash
swanlab login
```

或访问 [SwanLab官网](https://swanlab.cn) 注册账号后，在设置中获取 API Key，然后：

```bash
swanlab login --api-key YOUR_API_KEY
```

#### Step 3: 启动训练

```bash
# SFT 训练
LOGGER=swanlab CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh

# GRPO 训练
LOGGER=swanlab CUDA_VISIBLE_DEVICES=0 bash scripts/run_grpo_qwen35.sh
```

#### Step 4: 查看可视化

训练开始后，终端会显示实验链接，点击即可在浏览器查看实时训练曲线！

```
SwanLab 初始化成功: 项目=Qwen3.5-4B-MathRL
实验链接: https://swanlab.cn/@your-workspace/Qwen3.5-4B-MathRL/runs/xxx
```

**可以看到的内容**：
- 📉 训练/验证 Loss
- 📊 所有超参数对比
- 💾 模型检查点管理
- 🎯 自定义指标（Reward, KL Divergence 等）
- 📝 实验笔记
- 🔄 多实验对比

---

### 方案 3: 使用 WandB

```bash
# 安装
pip install wandb

# 登录
wandb login

# 训练
LOGGER=wandb CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
```

---

### 方案 4: 禁用日志（调试时）

```bash
LOGGER=none CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
```

---

## 📈 各工具对比详解

### TensorBoard

**适合场景**：
- 本地开发
- 不想联网
- 只需基础曲线

**使用示例**：
```bash
# 启动训练（默认）
bash scripts/run_sft_qwen35.sh

# 查看日志
tensorboard --logdir outputs_qwen35/sft_model
```

**可视化内容**：
```
Scalars:
├── train/loss                    # 训练 loss
├── train/learning_rate           # 学习率变化
├── train/grad_norm              # 梯度范数
├── train/epoch                   # 当前 epoch
└── eval/loss                     # 验证 loss

Graphs:
└── 模型计算图
```

---

### SwanLab

**适合场景**：
- 需要中文界面
- 多次实验对比
- 团队协作
- 移动端查看

**高级配置**：
```bash
# 自定义项目名和工作空间
export SWANLAB_PROJECT="My-MathRL-Experiment"
export SWANLAB_WORKSPACE="your-org"
LOGGER=swanlab bash scripts/run_sft_qwen35.sh
```

**可视化内容**：
```
训练指标:
├── train_loss                    # 训练损失
├── learning_rate                 # 学习率
├── grad_norm                     # 梯度范数
├── train_samples_per_second     # 训练速度
└── train_steps_per_second       # 步数速度

GRPO 专属:
├── rewards/mean                  # 平均奖励
├── rewards/correctness           # 正确性奖励
├── rewards/format                # 格式奖励
├── policy_loss                   # 策略损失
└── kl_divergence                 # KL 散度

系统监控:
├── GPU 显存占用
├── GPU 利用率
└── CPU/内存占用

超参数:
└── 所有训练配置（自动记录）
```

**实验对比**：
```bash
# 第一次实验：标准配置
LOGGER=swanlab EPOCHS=2.0 bash scripts/run_sft_qwen35.sh

# 第二次实验：快速配置
LOGGER=swanlab EPOCHS=1.5 LORA_RANK=32 bash scripts/run_sft_qwen35.sh

# 在 SwanLab 网页界面可直接对比两次实验的所有指标！
```

---

## 🔧 环境变量配置

### 日志工具选择

```bash
export LOGGER=swanlab      # 使用 SwanLab
export LOGGER=tensorboard  # 使用 TensorBoard（默认）
export LOGGER=wandb        # 使用 WandB
export LOGGER=none         # 禁用日志
```

### SwanLab 配置

```bash
export SWANLAB_PROJECT="Qwen3.5-4B-MathRL"  # 项目名
export SWANLAB_WORKSPACE="your-team"        # 工作空间（可选）
```

---

## 📊 实际使用示例

### 示例 1: 基础训练（TensorBoard）

```bash
cd /home/lyl/mathRL

# 启动 SFT 训练
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh

# 新终端查看日志
tensorboard --logdir outputs_qwen35/sft_model
```

---

### 示例 2: 完整训练（SwanLab）✨

```bash
cd /home/lyl/mathRL

# 安装 SwanLab（首次）
pip install swanlab
swanlab login

# SFT 训练
LOGGER=swanlab \
EPOCHS=2.0 \
LORA_RANK=64 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh

# GRPO 训练（SFT 完成后）
LOGGER=swanlab \
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_grpo_qwen35.sh
```

**实验链接**会在训练开始时自动显示，打开即可实时查看训练进度！

---

### 示例 3: 消融实验对比

```bash
# 实验 1: 标准 LoRA Rank 64
LOGGER=swanlab LORA_RANK=64 bash scripts/run_sft_qwen35.sh

# 实验 2: 小 LoRA Rank 32
LOGGER=swanlab LORA_RANK=32 bash scripts/run_sft_qwen35.sh

# 实验 3: 快速训练 1.5 epochs
LOGGER=swanlab EPOCHS=1.5 bash scripts/run_sft_qwen35.sh
```

在 SwanLab 界面可以一键对比三次实验的：
- Loss 曲线
- 超参数差异
- 最终性能

---

## 🎯 SwanLab 高级功能

### 1. 实验笔记

在 SwanLab 界面可以添加 Markdown 笔记：

```markdown
## SFT-Rank64 实验

**目标**: 测试 LoRA Rank 64 的效果

**观察**:
- 训练 loss 稳定下降
- 第 500 步后收敛
- 显存占用 58GB

**结论**: Rank 64 效果好，但需要大显存
```

### 2. 自定义可视化

SwanLab 自动记录所有 Trainer 输出的指标，无需手动配置！

### 3. 模型版本管理

SwanLab 会自动记录每个实验的：
- 代码版本（Git commit）
- 超参数配置
- 检查点路径

---

## 🐛 常见问题

### Q1: SwanLab 登录失败

**解决**:
```bash
# 方案 1: 使用 API Key
swanlab login --api-key YOUR_API_KEY

# 方案 2: 重新登录
swanlab logout
swanlab login
```

### Q2: TensorBoard 端口被占用

**解决**:
```bash
# 使用其他端口
tensorboard --logdir outputs_qwen35/sft_model --port 6007
```

### Q3: SwanLab 提示未安装

**解决**:
```bash
pip install swanlab

# 或在 requirements.txt 中添加
echo "swanlab>=0.3.0" >> requirements.txt
pip install -r requirements.txt
```

### Q4: 想同时使用 TensorBoard 和 SwanLab

**解决**:
目前只能选择一个。推荐优先使用 SwanLab，功能更强大。

---

## 📋 完整实验流程（推荐）

### 使用 SwanLab 的完整训练流程

```bash
# Step 1: 环境准备
cd /home/lyl/mathRL
pip install swanlab
swanlab login

# Step 2: SFT 训练（11-13h on A100）
LOGGER=swanlab \
EPOCHS=2.0 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh

# 训练开始后，终端会显示实验链接，可实时查看

# Step 3: GRPO 训练（SFT 完成后，8-10h）
LOGGER=swanlab \
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_grpo_qwen35.sh

# Step 4: 评估
python src_qwen35/evaluation.py --models base sft grpo

# Step 5: 在 SwanLab 界面查看完整训练历史和对比
```

---

## 🎨 可视化效果预览

### TensorBoard
```
Scalars 页面:
┌─────────────────────────────┐
│ train/loss                  │
│ ▼▼▼▼▼▼▼__________          │  ← Loss 曲线
│                             │
│ train/learning_rate         │
│ /▔▔▔▔▔\                     │  ← 学习率曲线
└─────────────────────────────┘
```

### SwanLab
```
Dashboard:
┌────────────────────────────────────┐
│ 📊 实验概览                        │
│ ├─ SFT-single (进行中)            │
│ ├─ SFT-multi (已完成)             │
│ └─ GRPO-single (队列中)           │
├────────────────────────────────────┤
│ 📈 Loss 对比                       │
│ [多条曲线实时更新]                 │
├────────────────────────────────────┤
│ 🔍 超参数对比                      │
│ Rank: 32 vs 64                    │
│ LR: 1e-5 vs 2e-5                  │
└────────────────────────────────────┘
```

---

## ✅ 推荐配置

### 本地实验
```bash
export LOGGER=tensorboard  # 简单快速
```

### 重要训练 ⭐
```bash
export LOGGER=swanlab      # 完整记录，云端保存
```

### 团队协作
```bash
export LOGGER=swanlab      # 或 wandb，支持共享
export SWANLAB_WORKSPACE=your-team
```

### 调试测试
```bash
export LOGGER=none         # 禁用日志，专注调试
```

---

**现在就试试 SwanLab，享受更好的实验跟踪体验！** 🚀

有问题？查看官方文档：
- SwanLab: https://docs.swanlab.cn
- TensorBoard: https://www.tensorflow.org/tensorboard
