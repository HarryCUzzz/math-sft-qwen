# Qwen3.5-4B 训练时间分析与配置建议

## 📊 训练时间预估

### SFT 训练时间

| 配置 | GPU 数量 | Epochs | Batch Size | 预估时间 | 是否满足 <12h |
|------|----------|--------|------------|----------|---------------|
| 保守配置 | 1 张 A6000 | 2.0 | 32 (grad_accum) | 20-24h | ❌ |
| 标准配置 | 2 张 A6000 | 2.0 | 64 (grad_accum) | 10-12h | ✅ |
| 快速配置 | 2 张 A6000 | 1.5 | 64 (grad_accum) | 7-9h | ✅ |
| 极速配置 | 2 张 A6000 | 1.0 | 64 (grad_accum) | 5-6h | ✅ |

### GRPO 训练时间

| 配置 | GPU 数量 | Max Steps | Generations | 预估时间 | 是否满足 <12h |
|------|----------|-----------|-------------|----------|---------------|
| 完整训练 | 1 张 A6000 | 500 | 8 | 10-12h | ✅ |
| 标准配置 | 1 张 A6000 | 400 | 8 | 8-10h | ✅ |
| 快速配置 | 1 张 A6000 | 300 | 6 | 6-8h | ✅ |
| 双卡加速 | 2 张 A6000 | 400 | 8 | 4-5h | ✅ |

---

## 🎯 推荐方案

### 方案 A: 满足 <12h 约束（推荐）

**SFT**: 双卡训练 (GPU 1+3)
```bash
EPOCHS=2.0 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh
```
- 预计时间: **10-12 小时**
- 显存需求: 2 × 40GB = 80GB
- 效果: 完整 2 epochs 训练

**GRPO**: 单卡训练 (GPU 1 或 3)
```bash
MAX_STEPS=400 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh
```
- 预计时间: **8-10 小时**
- 显存需求: 45GB
- 效果: 400 步强化学习

**总耗时**: SFT 和 GRPO 顺序执行约 **18-22 小时**

---

### 方案 B: 追求 ~9h 单阶段（激进）

**SFT**: 双卡 + 减少 epochs
```bash
EPOCHS=1.5 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh
```
- 预计时间: **7-9 小时**
- 权衡: 略微减少训练轮数

**GRPO**: 单卡 + 减少步数
```bash
MAX_STEPS=300 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh
```
- 预计时间: **6-8 小时**

**总耗时**: 顺序执行约 **13-17 小时**

---

### 方案 C: 极致速度（实验性）

**SFT + GRPO 并行**: 使用 `train_concurrent.sh`
- SFT (GPU 1): 单卡 20-24h
- GRPO (GPU 3): 单卡 8-12h（等待 SFT checkpoint 后启动）
- **总耗时**: ~**24-30 小时**（并行运行，但 SFT 单卡较慢）

⚠️ **不推荐**: 虽然并行，但 SFT 单卡训练时间太长，不如方案 A 的双卡顺序执行。

---

## 🔧 可配置参数清单

### SFT 训练参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| 训练轮数 | `EPOCHS` | 2.0 | 减少可加速训练 |
| LoRA Rank | `LORA_RANK` | 64 | 降低至 32 可节省显存 |
| 学习率 | `LEARNING_RATE` | 1e-5 | 调整收敛速度 |
| 梯度累积 | `GRAD_ACCUM_STEPS` | 32/16 | 单卡用 32，双卡用 16 |
| 最大长度 | `MAX_LENGTH` | 2048 | 减少可加速，但影响长问题 |
| 4-bit 量化 | `USE_4BIT` | false | 启用可节省 ~50% 显存 |

### GRPO 训练参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| 最大步数 | `MAX_STEPS` | 500 | 减少至 300-400 可加速 |
| 生成数量 | `NUM_GENERATIONS` | 8 | 减少至 4-6 可加速 |
| 生成长度 | `MAX_COMPLETION_LENGTH` | 1024 | 减少可加速 |
| 学习率 | `LEARNING_RATE` | 5e-7 | 调整训练稳定性 |
| 温度 | `TEMPERATURE` | 0.7 | 控制生成多样性 |
| Beta (KL 惩罚) | `BETA` | 0.01 | 控制策略偏离程度 |

---

## 📋 实际使用示例

### 示例 1: 平衡速度与效果（推荐）

```bash
# Step 1: SFT 双卡训练 (10-12h)
EPOCHS=2.0 \
LORA_RANK=64 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# Step 2: GRPO 单卡训练 (8-10h)
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**总耗时**: 18-22h
**预期效果**: MATH-500 准确率 25-30%

---

### 示例 2: 快速实验（牺牲效果）

```bash
# SFT 快速训练 (7-9h)
EPOCHS=1.5 \
LORA_RANK=32 \
MAX_LENGTH=1536 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO 快速训练 (6-8h)
MAX_STEPS=300 \
NUM_GENERATIONS=6 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**总耗时**: 13-17h
**预期效果**: MATH-500 准确率 20-25%（略低）

---

### 示例 3: 节省显存（启用量化）

```bash
# 单卡 + 4-bit 量化（显存 35GB → 20GB）
USE_4BIT=true \
EPOCHS=2.0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_sft_qwen35.sh
```

**时间**: 22-26h（单卡但量化略慢）
**显存**: 仅需 ~25GB

---

## 💡 决策建议

### 如果你的目标是...

1. **每阶段 <12h，效果优先** → **方案 A**
   - SFT 双卡 2.0 epochs (10-12h) + GRPO 单卡 400 steps (8-10h)
   - 总时间: 18-22h
   - 预期准确率: 25-30%

2. **每阶段 ~9h，接受效果损失** → **方案 B**
   - SFT 双卡 1.5 epochs (7-9h) + GRPO 单卡 300 steps (6-8h)
   - 总时间: 13-17h
   - 预期准确率: 20-25%

3. **显存受限** → **启用 4-bit 量化**
   - 接受单卡训练时间延长至 22-26h
   - 显存需求降至 ~25GB

4. **不在乎时间，追求最优效果** → **完整配置**
   - SFT 双卡 3.0 epochs + GRPO 单卡 600 steps
   - 总时间: 30-40h
   - 预期准确率: 30-35%

---

## 🚀 快速开始

### 1. 检查清单
```bash
# 确认模型已下载
ls /home/lyl/models/Qwen/Qwen3.5-4B/config.json

# 确认数据已准备
ls data/filtered/sft_train.json
ls data/rl_prompts/rl_train.json

# 检查 GPU 可用性
nvidia-smi
```

### 2. 执行训练（方案 A - 推荐）
```bash
cd /home/lyl/mathRL

# SFT 双卡训练
EPOCHS=2.0 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh

# 等待 SFT 完成后，运行 GRPO
MAX_STEPS=400 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh
```

### 3. 监控训练
```bash
# 实时日志
tail -f outputs_qwen35/sft_logs/sft_training.log
tail -f outputs_qwen35/grpo_logs/grpo_training.log

# GPU 监控
watch -n 1 nvidia-smi
```

---

## 📈 时间计算公式

### SFT 时间估算
```
训练时间 (小时) ≈ (样本数 × Epochs × 平均长度) / (有效 Batch Size × GPU 数 × 吞吐量)

其中:
- 样本数: ~8000
- 平均长度: ~800 tokens
- 有效 Batch Size: 32-64
- 吞吐量: ~800 tokens/s/GPU (A6000 + bf16)

单卡: (8000 × 2.0 × 800) / (32 × 1 × 800) ≈ 20h
双卡: (8000 × 2.0 × 800) / (64 × 2 × 800) ≈ 10h
```

### GRPO 时间估算
```
训练时间 (小时) ≈ (Max Steps × Num Generations × 平均生成长度) / (吞吐量 × GPU 数)

其中:
- Max Steps: 300-500
- Num Generations: 4-8
- 平均生成长度: ~512 tokens
- 吞吐量: ~600 tokens/s/GPU (生成较慢)

单卡 400 步: (400 × 8 × 512) / (600 × 1) / 3600 ≈ 9h
双卡 400 步: (400 × 8 × 512) / (600 × 2) / 3600 ≈ 4.5h
```

---

## ⚠️ 注意事项

1. **时间为粗略估算**: 实际时间受数据复杂度、硬件状态、系统负载影响，可能有 ±20% 波动
2. **双卡 GRPO 收益有限**: GRPO 瓶颈在生成阶段，双卡加速约 1.5-1.8x（非 2x）
3. **量化会略微降速**: 4-bit 量化虽节省显存，但推理速度降低 ~10-15%
4. **检查点开销**: 每次保存检查点需 1-2 分钟，已包含在估算中

---

**最终推荐**: 使用 **方案 A**（SFT 双卡 + GRPO 单卡），每阶段均在 12h 内完成，总时间 18-22h，效果最优。
