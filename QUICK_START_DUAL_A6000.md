# Qwen3.5-4B 双卡 A6000 快速参考

## 🚀 一键启动（推荐）

```bash
cd /home/lyl/mathRL

# 方案 B: 并行运行 SFT (GPU 1) + GRPO (GPU 3) [RECOMMENDED]
bash scripts/train_concurrent.sh

# 或方案 A: 顺序运行 SFT → GRPO
bash scripts/train_parallel.sh both
```

## 📍 GPU 分配

| 进程 | GPU ID | 显存 | 状态 |
|------|--------|------|------|
| SFT | GPU 1 | 48GB | 20-24h |
| GRPO | GPU 3 | 48GB | 8-12h |
| **总耗时** | **并行** | **96GB** | **~30h** |

## 📂 查看输出

```bash
# 监控实时日志
tail -f outputs_qwen35/sft_logs/sft_training.log
tail -f outputs_qwen35/grpo_logs/grpo_training.log

# GPU 监控
watch -n 1 nvidia-smi
```

## ✅ 完整流程

### 1️⃣ 启动并行训练
```bash
bash scripts/train_concurrent.sh
```
- SFT 和 GRPO 同时运行，各占一张 A6000
- 约 30 小时完成

### 2️⃣ 评估模型
```bash
python src_qwen35/evaluation.py --models base sft grpo
```
- 评估 MATH-500、GSM8K、TheoremQA
- 生成准确率对比表

### 3️⃣ 分析结果
```bash
python src_qwen35/analysis.py
```
- 生成训练曲线
- 失败案例分析

## 🔧 高级用法

### 启用 4-bit 量化（节省显存）
```bash
# 修改 sft_training.py/grpo_training.py
USE_4BIT=true CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh
```

### 单独运行 SFT
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh
```

### 单独运行 GRPO
```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/run_grpo_qwen35.sh
```

### 手动两终端并行
```bash
# 终端 1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh

# 终端 2（等待 SFT 生成 checkpoint 后启动）
CUDA_VISIBLE_DEVICES=3 bash scripts/run_grpo_qwen35.sh
```

## 📊 性能预期

```
Qwen2.5-0.5B (原始):    MATH-500: 4.6%
                        ↓↓↓ (升级 + 训练)
Qwen3.5-4B (SFT):      MATH-500: ~20-24%
                        ↓↓↓ (GRPO 强化)
Qwen3.5-4B (GRPO):     MATH-500: ~28-35%
```

**关键优势**：
- ✅ 参数量 8 倍提升
- ✅ Thinking Mode 支持
- ✅ 双卡时间减半
- ✅ 近无损 4-bit 量化可选

## 🛑 停止训练

```bash
# 如果使用 train_concurrent.sh
Ctrl+C  # 会终止两个进程

# 如果单独运行，在各终端按 Ctrl+C
```

下次运行时会从最新 checkpoint 自动恢复。

## 📋 检查清单

- [ ] 模型已下载：`/home/lyl/models/Qwen/Qwen3.5-4B/`
- [ ] 数据已准备：`data/filtered/` 和 `data/rl_prompts/`
- [ ] 输出目录已创建：`outputs_qwen35/`
- [ ] 两张 A6000 可用
- [ ] 显存充足：~96GB 总显存

准备好了？运行：
```bash
bash scripts/train_concurrent.sh
```

更多详情见 `TRAINING_GUIDE_DUAL_A6000.md`
