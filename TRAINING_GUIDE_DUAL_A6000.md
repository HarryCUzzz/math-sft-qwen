# Qwen3.5-4B 双卡 A6000 并行训练指南

## 快速启动

### 方案 A: 顺序训练 (SFT → GRPO)
```bash
# 各用一张 A6000 顺序运行
bash scripts/train_parallel.sh both
```

### 方案 B: 真正并行训练 (同时运行 SFT 和 GRPO) ⭐ 推荐
```bash
# 同时启动 SFT (GPU 1) 和 GRPO (GPU 3)，在后台并行运行
bash scripts/train_concurrent.sh
```

### 方案 C: 手动控制 (灵活性最高)

**终端 1 - SFT 训练**
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh
# 或使用 4-bit 量化
USE_4BIT=true CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh
```

**终端 2 - GRPO 训练**（等待 SFT-model 生成后启动）
```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/run_grpo_qwen35.sh
```

## GPU 分配

你的两张 A6000：
- **GPU 1** (ID: 00000000:1B:00.0) → SFT 训练
- **GPU 3** (ID: 00000000:88:00.0) → GRPO 训练

## 配置细节

### SFT 训练 (单卡模式)
```
GPU: GPU 1 (A6000 48GB)
配置:
  - per_device_train_batch_size: 1
  - gradient_accumulation_steps: 32
  - 有效 batch_size: 32
  - 学习率: 1e-5
  - epochs: 2.0
训练时间: ~20-24 小时
```

### GRPO 训练 (单卡模式)
```
GPU: GPU 3 (A6000 48GB)
配置:
  - per_device_train_batch_size: 1
  - num_generations: 4-8
  - max_completion_length: 768-1024
  - max_steps: 300-500
训练时间: ~8-12 小时
```

## 显存监控

监控 GPU 使用：
```bash
watch -n 1 nvidia-smi
```

预期显存占用：
- **SFT**: 35-40 GB / 48 GB
- **GRPO**: 40-45 GB / 48 GB

## 输出文件

训练完成后的文件位置：
```
outputs_qwen35/
├── sft_model/              # SFT LoRA 模型
├── sft_logs/               # SFT 训练日志
├── grpo_model/             # GRPO 完整模型
├── grpo_logs/              # GRPO 训练日志
└── eval_results/           # 评估结果
```

## 高级选项

### 1. 启用 4-bit 量化 (节省 ~50% 显存)
```bash
USE_4BIT=true CUDA_VISIBLE_DEVICES=1 bash scripts/run_sft_qwen35.sh
```

### 2. 调整训练参数
编辑 `src_qwen35/config.py`：
- `SFT_CONFIG_SINGLE/MULTI` - SFT 超参数
- `GRPO_CONFIG_SINGLE/MULTI` - GRPO 超参数

### 3. 断点恢复
如果训练中断，再次运行会自动从最新的 checkpoint 恢复。

## 推荐工作流

### 第一次完整训练
```bash
# 方案 B: 并行训练（最快）
bash scripts/train_concurrent.sh
```

### 本地测试/小数据训练
```bash
# 方案 A: 顺序训练（更容易监控）
bash scripts/train_parallel.sh both
```

### 调试/参数优化
```bash
# 方案 C: 手动两个终端分别运行，可独立控制
```

## 📊 预期性能

对比 Qwen2.5-0.5B (之前训练结果：4.6%):

| 阶段 | 双卡并行 耗时 | 预期准确率提升 |
|------|-------------|--------------|
| SFT | ~20h | Base → +15-20% |
| GRPO | ~10h | SFT → +10-15% |
| 总计 | ~30h | 最终 → 25-35% |

**关键优势**：
- ✅ Thinking Mode 启用（内部推理）
- ✅ 参数量 8 倍 (0.5B → 4B)
- ✅ 262k 上下文支持
- ✅ 双卡并行（显著加速）

## 常见问题

### Q: 显存不足怎么办？
A: 启用 4-bit 量化：`USE_4BIT=true`

### Q: 如何只运行 GRPO？
A: `bash scripts/train_parallel.sh grpo`

### Q: 训练被中断如何继续？
A: 直接再次运行脚本，会自动从 checkpoint 恢复

### Q: 如何更改评估数据集？
A: 编辑 `src_qwen35/config.py` 中的 `EVAL_DATASETS`

## 后续步骤

1. **评估**
```bash
python src_qwen35/evaluation.py --models base sft grpo
```

2. **结果分析**
```bash
python src_qwen35/analysis.py
```

3. **消融实验** (可选)
修改参数后重新训练，对比效果。
