# 训练方案决策指南

根据你的要求（每阶段 <12h，如可能在 9h 左右完成就用双卡），以下是详细的分析和推荐。

---

## 🎯 你的要求分析

1. **SFT 训练**: 时长 <12h，如能在 9h 左右完成更佳
2. **GRPO 训练**: 时长 <12h，如能在 9h 左右完成更佳
3. **高度可配置**: 脚本需支持灵活调整参数
4. **明确预估**: 需要各种配置的时间预估

---

## ✅ 推荐方案：混合模式（最优）

### 方案概述

- **SFT**: 双卡并行 (GPU 1+3) → 10-12h
- **GRPO**: 单卡训练 (GPU 1 或 3) → 8-10h
- **总耗时**: 18-22h（顺序执行）
- **满足要求**: ✅ 每阶段都 <12h

### 为什么这样选择？

#### SFT - 为什么用双卡？

单卡 SFT 需要 20-24h，**不满足 <12h 要求**。
双卡可以降到 10-12h，**刚好满足 <12h 要求**。

如果追求 ~9h：可以用双卡 + 减少 epochs 到 1.5 → 7-9h（略有效果损失）

#### GRPO - 为什么用单卡？

单卡 GRPO 只需 8-10h，**已经满足 <12h 要求**。
双卡加速有限（4-5h），但会占用两张卡导致 SFT 无法并行。

如果追求 ~9h：单卡标准配置已经是 8-10h，刚好符合目标。

---

## 📊 详细时间对比表

### SFT 训练时间对比

| 配置 | GPU 配置 | Epochs | LoRA Rank | 预估时间 | 是否 <12h | 是否 ~9h |
|------|----------|--------|-----------|----------|-----------|----------|
| 保守 | 1 卡 | 2.0 | 32 | 20-24h | ❌ | ❌ |
| **标准** | **2 卡** | **2.0** | **64** | **10-12h** | ✅ | 接近 |
| **快速** | **2 卡** | **1.5** | **64** | **7-9h** | ✅ | ✅ |
| 极速 | 2 卡 | 1.0 | 32 | 5-6h | ✅ | ✅ (过快) |

### GRPO 训练时间对比

| 配置 | GPU 配置 | Max Steps | Generations | 预估时间 | 是否 <12h | 是否 ~9h |
|------|----------|-----------|-------------|----------|-----------|----------|
| 完整 | 1 卡 | 500 | 8 | 10-12h | ✅ | 接近 |
| **标准** | **1 卡** | **400** | **8** | **8-10h** | ✅ | ✅ |
| 快速 | 1 卡 | 300 | 6 | 6-8h | ✅ | ✅ (较快) |
| 双卡加速 | 2 卡 | 400 | 8 | 4-5h | ✅ | ✅ (过快) |

---

## 🚀 最终推荐

### 推荐 A: 标准配置（平衡最优）⭐⭐⭐

**如果你想要最佳效果的同时满足 <12h 约束**：

```bash
cd /home/lyl/mathRL

# ========== SFT 训练 ==========
# 双卡，2.0 epochs，预计 10-12 小时
EPOCHS=2.0 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# ========== GRPO 训练 ==========
# 单卡，400 步，预计 8-10 小时
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**结果**:
- SFT: 10-12h ✅
- GRPO: 8-10h ✅
- 总时间: 18-22h
- 预期准确率: **MATH-500 25-30%**

---

### 推荐 B: 快速配置（追求 ~9h）⭐⭐

**如果你更看重每阶段都接近 9h**：

```bash
# ========== SFT 训练 ==========
# 双卡，1.5 epochs，预计 7-9 小时
EPOCHS=1.5 \
LORA_RANK=64 \
LEARNING_RATE=1.2e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh

# ========== GRPO 训练 ==========
# 单卡，350 步，预计 7-9 小时
MAX_STEPS=350 \
NUM_GENERATIONS=7 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**结果**:
- SFT: 7-9h ✅
- GRPO: 7-9h ✅
- 总时间: 14-18h
- 预期准确率: **MATH-500 23-28%**（略低于标准配置）

---

### 推荐 C: 一键并行（适合夜间训练）⭐

**如果你想一次性启动，第二天看结果**：

```bash
# 使用现成的并行脚本，自动两卡并行
bash scripts/train_concurrent.sh
```

**注意**: 这个脚本会让 SFT 单卡运行（20-24h），GRPO 单卡运行（8-12h），并行总时间约 24-30h。
**不推荐**: SFT 单卡太慢，不满足 <12h 要求。

---

## 🔄 灵活调整

### 如果 SFT 训练太慢（>12h）

```bash
# 方案 1: 减少 epochs
EPOCHS=1.5 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh

# 方案 2: 降低 LoRA rank（牺牲效果）
LORA_RANK=32 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh

# 方案 3: 减小最大长度（适合短问题）
MAX_LENGTH=1536 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh
```

### 如果 GRPO 训练太慢（>12h）

```bash
# 方案 1: 减少步数
MAX_STEPS=300 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh

# 方案 2: 减少生成数
NUM_GENERATIONS=6 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh

# 方案 3: 使用双卡加速（4-5h，但占用两卡）
CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_grpo_qwen35.sh
```

---

## 📌 重要说明

### 为什么不推荐 SFT 和 GRPO 同时并行？

虽然 `train_concurrent.sh` 支持同时运行，但：
1. **SFT 单卡需要 20-24h**，不满足你的 <12h 要求
2. **双卡顺序执行更快**: SFT 双卡 (10-12h) + GRPO 单卡 (8-10h) = 18-22h
3. **同时并行总时间**: SFT 单卡 (20-24h) + GRPO 单卡 (8-12h 并行) = ~24-30h（更慢！）

**结论**: 双卡顺序执行 SFT → 单卡 GRPO 是最优方案。

### 双卡 GRPO 的收益

GRPO 双卡加速约 1.8x（非 2x），从 8-10h 降到 4-5h。
但这会占用两张卡，导致 SFT 无法同时运行，总时间反而更长。

**仅在以下情况推荐双卡 GRPO**:
- SFT 已经完成
- 急需快速迭代 GRPO 参数

---

## 🎯 决策流程图

```
开始
  ↓
是否满足 <12h？
  ├─ 是 → 是否追求 ~9h？
  │       ├─ 是 → 推荐 B (快速配置)
  │       └─ 否 → 推荐 A (标准配置)
  └─ 否 → 必须在 12h 内吗？
          ├─ 是 → 使用推荐 B 或更激进配置
          └─ 否 → 可以用更充分的配置（3 epochs, 600 steps）
```

---

## 🛠️ 实际执行步骤

### 步骤 1: 准备环境

```bash
cd /home/lyl/mathRL

# 检查模型
ls /home/lyl/models/Qwen/Qwen3.5-4B/config.json

# 检查数据
ls data/filtered/sft_train.json
ls data/rl_prompts/rl_train.json

# 检查 GPU
nvidia-smi
```

### 步骤 2: 启动 SFT 训练

```bash
# 推荐 A 的 SFT 配置
EPOCHS=2.0 \
LORA_RANK=64 \
LEARNING_RATE=1e-5 \
CUDA_VISIBLE_DEVICES=1,3 \
bash scripts/run_sft_qwen35.sh
```

**预计时间**: 10-12 小时

监控训练：
```bash
# 新终端窗口
tail -f outputs_qwen35/sft_logs/sft_training.log
watch -n 1 nvidia-smi
```

### 步骤 3: 启动 GRPO 训练（等 SFT 完成后）

```bash
# 推荐 A 的 GRPO 配置
MAX_STEPS=400 \
NUM_GENERATIONS=8 \
TEMPERATURE=0.7 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_grpo_qwen35.sh
```

**预计时间**: 8-10 小时

### 步骤 4: 评估结果

```bash
python src_qwen35/evaluation.py --models base sft grpo
python src_qwen35/analysis.py
```

---

## 📊 预期结果对比

| 方案 | SFT 时间 | GRPO 时间 | 总时间 | MATH-500 | GSM8K | TheoremQA |
|------|----------|-----------|--------|----------|-------|-----------|
| 推荐 A (标准) | 10-12h | 8-10h | 18-22h | 25-30% | 60-70% | 18-25% |
| 推荐 B (快速) | 7-9h | 7-9h | 14-18h | 23-28% | 55-65% | 16-22% |
| 极速配置 | 5-6h | 4-5h | 9-11h | 18-22% | 45-55% | 12-18% |
| 完整配置 | 15-18h | 12-15h | 27-33h | 30-35% | 70-80% | 25-32% |

---

## ✅ 最终建议

根据你的需求（每阶段 <12h，如能 ~9h 更好），**推荐使用"推荐 B（快速配置）"**：

```bash
# SFT: 双卡，1.5 epochs，7-9h
EPOCHS=1.5 LORA_RANK=64 LEARNING_RATE=1.2e-5 CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_sft_qwen35.sh

# GRPO: 单卡，350 步，7-9h
MAX_STEPS=350 NUM_GENERATIONS=7 TEMPERATURE=0.7 CUDA_VISIBLE_DEVICES=1 bash scripts/run_grpo_qwen35.sh
```

**理由**:
1. ✅ 每阶段都在 7-9h，满足你的时间要求
2. ✅ 预期准确率 23-28%，相比 Qwen2.5-0.5B (4.6%) 有巨大提升
3. ✅ 总时间 14-18h，适中
4. ✅ 参数量 8 倍提升 (0.5B → 4B) + Thinking Mode 加持

如果你更看重效果而不是严格的 9h 约束，可以用 **"推荐 A（标准配置）"** 换取额外 2-5% 的准确率提升。

---

## 📞 需要帮助？

- 查看详细参数说明: `CONFIGURABLE_PARAMETERS.md`
- 查看时间分析: `TRAINING_TIME_ANALYSIS.md`
- 查看完整指南: `TRAINING_GUIDE_DUAL_A6000.md`
- 查看快速开始: `QUICK_START_DUAL_A6000.md`

祝训练顺利！🚀
