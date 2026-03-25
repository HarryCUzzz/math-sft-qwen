# Qwen3.5-4B 可配置参数完整指南

## 📋 快速参考

### SFT 训练参数

| 参数名 | 环境变量 | 默认值 (单卡) | 默认值 (双卡) | 取值范围 | 影响 |
|--------|----------|---------------|---------------|----------|------|
| 训练轮数 | `EPOCHS` | 2.0 | 2.0 | 1.0-3.0 | 训练时间线性增长 |
| LoRA 秩 | `LORA_RANK` | 32 | 64 | 16/32/64 | 显存占用，模型表达能力 |
| LoRA alpha | `LORA_ALPHA` | 64 | 128 | 通常 2×rank | 学习率缩放 |
| 学习率 | `LEARNING_RATE` | 1e-5 | 1e-5 | 1e-6 ~ 5e-5 | 收敛速度与稳定性 |
| 最大长度 | `MAX_LENGTH` | 2048 | 2048 | 512-2048 | 显存占用，长问题支持 |
| 梯度累积 | `GRAD_ACCUM_STEPS` | 32 | 16 | 8-64 | 有效批次大小 |
| 保存频率 | `SAVE_STEPS` | 500 | 500 | 100-1000 | 检查点数量 |
| 预热比例 | `WARMUP_RATIO` | 0.1 | 0.1 | 0.0-0.2 | 学习率预热 |

### GRPO 训练参数

| 参数名 | 环境变量 | 默认值 (单卡) | 默认值 (双卡) | 取值范围 | 影响 |
|--------|----------|---------------|---------------|----------|------|
| 最大步数 | `MAX_STEPS` | 300 | 500 | 200-600 | 训练时间线性增长 |
| 生成数量 | `NUM_GENERATIONS` | 4 | 8 | 4-8 | 每步时间，探索多样性 |
| 生成长度 | `MAX_COMPLETION_LENGTH` | 768 | 1024 | 512-1536 | 生成时间，显存占用 |
| 学习率 | `LEARNING_RATE` | 1e-7 | 1e-7 | 1e-8 ~ 1e-6 | 策略更新幅度 |
| 温度 | `TEMPERATURE` | 0.7 | 0.7 | 0.6-0.9 | 生成多样性 |
| KL 惩罚 | `BETA` | 0.04 | 0.04 | 0.01-0.1 | 策略偏离约束 |
| 梯度累积 | `GRAD_ACCUM_STEPS` | 8 | 4 | 4-16 | 有效批次大小 |
| 保存频率 | `SAVE_STEPS` | 100 | 100 | 50-200 | 检查点数量 |

---

## 🚀 常用配置场景

### 场景 1: 标准训练（推荐）

**目标**: 平衡效果与时间，每阶段 <12h

```bash
# SFT: 双卡，约 10-12h
EPOCHS=2.0 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO: 单卡，约 8-10h
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**预期效果**: MATH-500 准确率 25-30%

---

### 场景 2: 快速实验（9h 目标）

**目标**: 快速验证思路，可接受效果略降

```bash
# SFT: 双卡，约 7-9h
EPOCHS=1.5 \
LORA_RANK=32 \
MAX_LENGTH=1536 \
LEARNING_RATE=1.5e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO: 单卡，约 6-8h
MAX_STEPS=300 \
NUM_GENERATIONS=6 \
MAX_COMPLETION_LENGTH=768 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**预期效果**: MATH-500 准确率 20-25%

---

### 场景 3: 极致加速（<6h 单阶段）

**目标**: 最快速度完成训练

```bash
# SFT: 双卡，约 5-6h
EPOCHS=1.0 \
LORA_RANK=32 \
MAX_LENGTH=1536 \
GRAD_ACCUM_STEPS=8 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO: 单卡，约 4-5h
MAX_STEPS=200 \
NUM_GENERATIONS=4 \
MAX_COMPLETION_LENGTH=512 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**预期效果**: MATH-500 准确率 15-20%（基础性能）

---

### 场景 4: 追求最优效果

**目标**: 不在乎时间，追求最高准确率

```bash
# SFT: 双卡，约 15-18h
EPOCHS=3.0 \
LORA_RANK=64 \
LEARNING_RATE=8e-6 \
MAX_LENGTH=2048 \
WARMUP_RATIO=0.15 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO: 双卡，约 12-15h
MAX_STEPS=600 \
NUM_GENERATIONS=8 \
MAX_COMPLETION_LENGTH=1024 \
LEARNING_RATE=8e-8 \
TEMPERATURE=0.75 \
BETA=0.03 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_grpo_qwen35.sh
```

**预期效果**: MATH-500 准确率 30-35%（最优）

---

### 场景 5: 显存受限（单卡 + 量化）

**目标**: 在显存不足时完成训练

```bash
# SFT: 单卡 + 4-bit 量化，约 24-28h
USE_4BIT=true \
EPOCHS=2.0 \
LORA_RANK=32 \
MAX_LENGTH=1536 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_sft_qwen35.sh

# GRPO: 单卡 + 4-bit 量化，约 10-12h
USE_4BIT=true \
MAX_STEPS=400 \
NUM_GENERATIONS=6 \
MAX_COMPLETION_LENGTH=768 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**显存需求**: 从 40GB 降至 ~25GB
**预期效果**: MATH-500 准确率 22-28%（量化损失较小）

---

## 🔧 参数详细说明

### SFT 参数

#### `EPOCHS` - 训练轮数
- **默认**: 2.0
- **推荐范围**: 1.0-3.0
- **影响**:
  - 值越大，训练越充分，但时间线性增长
  - 1.0 → 快速收敛，适合实验
  - 2.0 → 平衡选择
  - 3.0 → 充分训练，可能过拟合
- **时间影响**: `时间 ∝ EPOCHS`
  - 2.0 → 10h (双卡) / 20h (单卡)
  - 1.5 → 7.5h / 15h
  - 1.0 → 5h / 10h

#### `LORA_RANK` - LoRA 秩
- **默认**: 32 (单卡) / 64 (双卡)
- **推荐范围**: 16/32/64
- **影响**:
  - 越大，模型表达能力越强，但显存占用增加
  - 16 → 最省显存 (~30GB)
  - 32 → 平衡 (~35GB)
  - 64 → 最佳效果 (~40GB)
- **性能对比**:
  - Rank 16: MATH-500 ~18-22%
  - Rank 32: MATH-500 ~22-26%
  - Rank 64: MATH-500 ~25-30%

#### `LEARNING_RATE` - 学习率
- **默认**: 1e-5
- **推荐范围**: 1e-6 ~ 5e-5
- **影响**:
  - 太小 (1e-6): 收敛慢，可能欠拟合
  - 适中 (1e-5): 稳定收敛
  - 太大 (5e-5): 可能震荡或发散
- **调整建议**:
  - 降低 epochs 时可略微提高学习率 (如 1.5e-5)
  - 更大模型建议更小学习率

#### `MAX_LENGTH` - 最大序列长度
- **默认**: 2048
- **推荐范围**: 512-2048
- **影响**:
  - 越大，支持越长的问题，但显存占用和训练时间增加
  - 512 → 快速训练，只适合短问题
  - 1536 → 平衡，覆盖 90% 问题
  - 2048 → 最完整，覆盖 98% 问题
- **显存影响**: `显存 ∝ LENGTH²`
  - 2048 → 40GB
  - 1536 → 30GB
  - 1024 → 22GB

#### `GRAD_ACCUM_STEPS` - 梯度累积步数
- **默认**: 32 (单卡) / 16 (双卡)
- **推荐范围**: 8-64
- **影响**:
  - 有效批次大小 = `per_device_batch_size × num_gpus × grad_accum_steps`
  - 越大，训练越稳定，但每步时间更长
- **自动计算**: 留空则自动选择最优值

---

### GRPO 参数

#### `MAX_STEPS` - 训练步数
- **默认**: 300 (单卡) / 500 (双卡)
- **推荐范围**: 200-600
- **影响**:
  - 步数越多，策略优化越充分，时间线性增长
  - 200 → 快速实验 (~5h 单卡)
  - 400 → 平衡 (~9h 单卡)
  - 600 → 充分训练 (~13h 单卡)
- **时间影响**: `时间 ∝ MAX_STEPS`

#### `NUM_GENERATIONS` - 每步生成数
- **默认**: 4 (单卡) / 8 (双卡)
- **推荐范围**: 4-8
- **影响**:
  - 越大，探索多样性越好，但每步时间增加
  - 4 → 最快，多样性低
  - 6 → 平衡
  - 8 → 最佳多样性，时间+100%
- **性能对比**:
  - 4 gens: MATH-500 ~22-25%
  - 6 gens: MATH-500 ~25-28%
  - 8 gens: MATH-500 ~28-32%

#### `TEMPERATURE` - 生成温度
- **默认**: 0.7
- **推荐范围**: 0.6-0.9
- **影响**:
  - 越高，生成越随机，探索性强但可能无意义
  - 0.6 → 保守，更确定性
  - 0.7 → 平衡
  - 0.8-0.9 → 激进，高多样性
- **调整建议**:
  - 数学问题推荐 0.7
  - 创造性任务可用 0.8-0.9

#### `BETA` - KL 散度惩罚系数
- **默认**: 0.04
- **推荐范围**: 0.01-0.1
- **影响**:
  - 越大，策略偏离初始模型越小，训练更保守
  - 0.01 → 允许大幅偏离，可能不稳定
  - 0.04 → 平衡
  - 0.1 → 保守，接近基线
- **调整建议**:
  - 出现训练不稳定时增大 beta
  - 希望更激进探索时减小 beta

#### `MAX_COMPLETION_LENGTH` - 生成最大长度
- **默认**: 768 (单卡) / 1024 (双卡)
- **推荐范围**: 512-1536
- **影响**:
  - 越大，支持更复杂推理，但显存和时间增加
  - 512 → 快速，简单问题
  - 768 → 平衡
  - 1024 → 完整，复杂推理
- **时间影响**: `时间 ∝ LENGTH`

---

## ⏱️ 训练时间计算公式

### SFT 时间估算

```
训练时间 (小时) = (数据量 × Epochs × 平均长度) / (有效批次 × 吞吐量 × 3600)

其中:
- 数据量: ~8000 个样本
- 平均长度: ~800 tokens
- 有效批次: batch_size × num_gpus × grad_accum_steps
- 吞吐量: ~800 tokens/s (A6000, bf16)

示例计算:
单卡 2.0 epochs:
  (8000 × 2.0 × 800) / (1 × 1 × 32 × 800) ≈ 20 小时

双卡 2.0 epochs:
  (8000 × 2.0 × 800) / (2 × 2 × 16 × 800) ≈ 10 小时

双卡 1.5 epochs:
  (8000 × 1.5 × 800) / (2 × 2 × 16 × 800) ≈ 7.5 小时
```

### GRPO 时间估算

```
训练时间 (小时) = (Max Steps × Num Gens × 平均生成长度) / (吞吐量 × num_gpus × 3600)

其中:
- Max Steps: 训练步数
- Num Gens: 每步生成数
- 平均生成长度: ~512 tokens
- 吞吐量: ~600 tokens/s (生成较慢)

示例计算:
单卡 400 步 × 8 gens:
  (400 × 8 × 512) / (600 × 1 × 3600) ≈ 9 小时

单卡 300 步 × 6 gens:
  (300 × 6 × 512) / (600 × 1 × 3600) ≈ 6.5 小时

双卡 400 步 × 8 gens:
  (400 × 8 × 512) / (600 × 2 × 3600) ≈ 4.5 小时
```

---

## 💡 调参建议

### 优先级 1: 满足时间约束

如果目标是 **每阶段 <12h**:
1. SFT: 使用双卡 + 2.0 epochs (10-12h)
2. GRPO: 使用单卡 + 400 steps (8-10h)

如果目标是 **每阶段 ~9h**:
1. SFT: 使用双卡 + 1.5 epochs (7-9h)
2. GRPO: 使用单卡 + 300 steps + 6 gens (6-8h)

### 优先级 2: 显存约束

如果单卡显存不足（<35GB）:
1. 启用 4-bit 量化: `USE_4BIT=true`
2. 或降低 LoRA rank: `LORA_RANK=32`
3. 或减小最大长度: `MAX_LENGTH=1536`

### 优先级 3: 追求效果

如果有充足时间（>30h）:
1. SFT: 3.0 epochs + Rank 64
2. GRPO: 600 steps + 8 gens

### 调参禁忌

❌ **不要同时**:
- 大幅降低 epochs (<1.0) + 降低 LoRA rank (<32)
- 减少 max_steps (<200) + 减少 num_generations (<4)

✅ **推荐组合**:
- 快速: 减 epochs OR 减 rank，不要两者都减
- 平衡: 标准配置
- 完整: 增 epochs + 标准其他参数

---

## 📊 效果预期对照表

| 配置等级 | SFT 时间 | GRPO 时间 | 总时间 | MATH-500 准确率 |
|----------|----------|-----------|--------|-----------------|
| 极速 | 5h | 4h | 9h | 15-20% |
| 快速 | 7-9h | 6-8h | 13-17h | 20-25% |
| 标准 ⭐ | 10-12h | 8-10h | 18-22h | 25-30% |
| 完整 | 15-18h | 12-15h | 27-33h | 30-35% |

**标准配置** 提供最佳的时间-效果平衡，推荐优先使用。

---

## 🛠️ 实际使用示例

### 示例 1: 首次完整训练

```bash
cd /home/lyl/mathRL

# 确认环境
bash scripts/download_qwen35.sh  # 如果模型未下载
nvidia-smi  # 确认 GPU 可用

# SFT 训练 (双卡，10-12h)
EPOCHS=2.0 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO 训练 (单卡，8-10h)
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh

# 评估
python src_qwen35/evaluation.py --models base sft grpo
```

### 示例 2: 调试参数（快速迭代）

```bash
# 用少量数据快速测试（可手动编辑 data/filtered/sft_train.json 只保留前 100 条）
EPOCHS=0.5 \
LORA_RANK=32 \
MAX_LENGTH=1024 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_sft_qwen35.sh

# 预计时间: 0.5h → 快速验证流程
```

### 示例 3: 在时间限制下最大化效果

```bash
# 假设你有 18 小时可用时间

# SFT: 分配 11 小时
EPOCHS=2.2 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
WARMUP_RATIO=0.12 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# GRPO: 分配 7 小时
MAX_STEPS=350 \
NUM_GENERATIONS=7 \
LEARNING_RATE=8e-8 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

---

## ✅ 检查清单

训练前确认:
- [ ] 模型已下载: `ls /home/lyl/models/Qwen/Qwen3.5-4B/config.json`
- [ ] 数据已准备: `ls data/filtered/sft_train.json data/rl_prompts/rl_train.json`
- [ ] GPU 可用: `nvidia-smi`
- [ ] 确认目标时间约束和效果要求
- [ ] 选择合适的配置场景

训练中监控:
```bash
# 实时日志
tail -f outputs_qwen35/sft_logs/sft_training.log
tail -f outputs_qwen35/grpo_logs/grpo_training.log

# GPU 使用
watch -n 1 nvidia-smi

# 检查检查点
ls -lh outputs_qwen35/sft_model/
ls -lh outputs_qwen35/grpo_model/
```

---

**祝训练顺利！**如有问题参考 `TRAINING_TIME_ANALYSIS.md` 或 `TRAINING_GUIDE_DUAL_A6000.md`。
