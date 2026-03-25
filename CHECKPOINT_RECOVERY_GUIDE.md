# 训练断点续传指南

## 📌 当前问题诊断

你的训练在 **step 180** 被中断，但 **没有检查点可以恢复**，原因：

- 默认配置：`SAVE_STEPS=200`（每 200 步保存一次）
- 训练进度：180 步（还没到第一个检查点）
- 结果：`outputs_qwen35/sft_model/` 目录为空

**结论**：❌ 无法从当前状态恢复，需要重新开始训练（但可以调整策略避免再次发生）

---

## ✅ 自动断点续传功能说明

你的代码**已经支持自动断点续传**！下次训练只要有检查点就会自动恢复：

```python
# src_qwen35/sft_training.py 自动检测检查点
checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    resume_from_checkpoint = str(latest_checkpoint)
    logger.info(f"从 checkpoint 恢复训练: {resume_from_checkpoint}")

trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

**使用方法**：直接重新运行训练脚本，会自动检测并恢复！

```bash
# 如果有检查点，会自动恢复
bash scripts/run_sft_qwen35.sh
```

---

## 🚀 推荐方案：减小保存间隔（防止再次丢失）

### 方案 1: 每 50 步保存（推荐）⭐

```bash
SAVE_STEPS=50 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh
```

**特点**：
- ✅ 最多损失 50 步进度（约 5-10 分钟）
- ✅ 检查点占用可控（自动保留最近 5 个）
- ✅ 平衡了安全性和磁盘占用

**计算**：
- 训练集：320,723 条
- Batch size：32
- 总步数：320,723 ÷ 32 × 2 epochs ≈ 20,045 步
- 检查点数量：20,045 ÷ 50 = 401 个（但只保留最近 5 个）
- 磁盘占用：5 × 4GB ≈ 20GB

---

### 方案 2: 每 100 步保存（平衡）

```bash
SAVE_STEPS=100 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh
```

**特点**：
- 最多损失 100 步进度（约 10-20 分钟）
- 磁盘占用更少

---

### 方案 3: 每 20 步保存（极致安全，调试用）

```bash
SAVE_STEPS=20 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh
```

**特点**：
- 几乎不会丢失进度
- 适合调试阶段或不稳定环境
- 磁盘 I/O 开销略大

---

## 📊 检查点文件结构

训练时会自动在 `outputs_qwen35/sft_model/` 下保存检查点：

```
outputs_qwen35/sft_model/
├── checkpoint-50/              # 第 50 步
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (~3-5GB)
│   ├── optimizer.pt
│   ├── rng_state.pth
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-100/             # 第 100 步
├── checkpoint-150/             # 第 150 步
├── checkpoint-200/             # 第 200 步
└── checkpoint-250/             # 第 250 步（最新）
```

**自动清理**：配置了 `save_total_limit=5`，会自动删除旧检查点，只保留最近 5 个。

---

## 🔄 如何手动恢复训练

### 情况 1: 自动恢复（推荐）

只需重新运行脚本，会自动检测检查点：

```bash
# 自动检测 outputs_qwen35/sft_model/checkpoint-* 并恢复
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_qwen35.sh
```

**日志输出**：
```
从 checkpoint 恢复训练: /home/lyl/mathRL/outputs_qwen35/sft_model/checkpoint-250
```

---

### 情况 2: 手动指定检查点（不常用）

如果需要从特定检查点恢复，修改 `sft_training.py`：

```python
# 手动指定检查点路径
resume_from_checkpoint = "/home/lyl/mathRL/outputs_qwen35/sft_model/checkpoint-200"
train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

---

## 🛡️ 避免训练中断的建议

### 1. 使用 tmux/screen（强烈推荐）

```bash
# 创建新会话
tmux new -s training
 
# 在 tmux 中运行训练
SAVE_STEPS=50 bash scripts/run_sft_qwen35.sh

# 断开会话（训练继续）：Ctrl+B 然后按 D
# 重新连接：tmux attach -t training
```

---

### 2. 使用 nohup（后台运行）

```bash
nohup bash scripts/run_sft_qwen35.sh > training.log 2>&1 &

# 查看日志
tail -f training.log

# 查看进程
ps aux | grep sft_training
```

---

### 3. 调整 SAVE_STEPS + 监控

```bash
# 设置频繁保存 + SwanLab 实时监控
SAVE_STEPS=50 \
LOGGER=swanlab \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh
```

在 SwanLab 界面可以实时查看训练进度，提前发现问题。

---

## 📋 完整训练流程（含容错）

### Step 1: 启动训练（tmux + 频繁保存）

```bash
# 创建 tmux 会话
tmux new -s sft_training

# 配置环境变量
export CUDA_VISIBLE_DEVICES=0
export SAVE_STEPS=50           # 每 50 步保存
export LOGGER=swanlab           # 实时监控

# 启动训练
bash scripts/run_sft_qwen35.sh
```

---

### Step 2: 监控训练

**方式 1: 查看日志**
```bash
# 新终端窗口
tail -f outputs_qwen35/sft_logs/sft_training.log
```

**方式 2: 查看 SwanLab**
- 训练启动后会显示实验链接
- 浏览器打开实时查看

**方式 3: 监控 GPU**
```bash
watch -n 1 nvidia-smi
```

---

### Step 3: 如果中断，自动恢复

如果训练中断（电源、网络、OOM 等），只需重新运行脚本：

```bash
# 重新连接 tmux（如果断开）
tmux attach -t sft_training

# 或直接重新运行（会自动恢复）
bash scripts/run_sft_qwen35.sh
```

**日志会显示**：
```
从 checkpoint 恢复训练: /home/lyl/mathRL/outputs_qwen35/sft_model/checkpoint-350
继续训练从 step 351 开始...
```

---

## 🔍 检查当前检查点状态

### 查看保存的检查点

```bash
# 列出所有检查点
ls -lh outputs_qwen35/sft_model/

# 查看最新检查点
ls -lt outputs_qwen35/sft_model/ | head -5

# 查看检查点数量
ls outputs_qwen35/sft_model/ | grep checkpoint | wc -l
```

### 查看检查点详情

```bash
# 查看某个检查点的训练状态
cat outputs_qwen35/sft_model/checkpoint-250/trainer_state.json
```

---

## ⚠️ 注意事项

### 1. 检查点占用磁盘空间

每个检查点约 **3-5GB**（LoRA adapter），保留 5 个约 **15-25GB**。

如果磁盘空间有限，可以减少保留数量：

```bash
# 修改 config.py 的 save_total_limit
"save_total_limit": 3,  # 只保留 3 个检查点
```

---

### 2. 保存频率 vs 训练速度

保存检查点会暂停训练 1-2 秒，但影响很小：

| SAVE_STEPS | 保存次数 | 总暂停时间 | 影响 |
|------------|---------|-----------|------|
| 200 | ~100 次 | ~3 分钟 | 可忽略 |
| 100 | ~200 次 | ~6 分钟 | 可忽略 |
| 50 | ~400 次 | ~12 分钟 | 约 1-2% |
| 20 | ~1000 次 | ~30 分钟 | 约 3-5% |

**推荐**：`SAVE_STEPS=50` 是最佳平衡点。

---

### 3. 不要手动删除检查点

训练过程中不要手动删除 `outputs_qwen35/sft_model/` 下的检查点，可能导致恢复失败。

如果需要清理，等训练完成后再删除。

---

## 🎯 你的当前情况解决方案

### 推荐操作流程

```bash
# Step 1: 进入 tmux（避免再次中断）
tmux new -s sft_training

# Step 2: 重新开始训练，每 50 步保存
SAVE_STEPS=50 \
LOGGER=swanlab \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_sft_qwen35.sh

# Step 3: 断开 tmux 会话（训练继续）
# 按 Ctrl+B 然后按 D

# Step 4: 随时重新连接查看进度
tmux attach -t sft_training
```

### 预期效果

- ✅ 每 50 步自动保存检查点
- ✅ 如果再次中断，自动从最近检查点恢复
- ✅ SwanLab 实时监控训练曲线
- ✅ tmux 保护进程不受终端关闭影响

---

## 📊 恢复训练的验证

训练恢复后，检查日志确认：

```bash
tail -f outputs_qwen35/sft_logs/sft_training.log
```

**正确的恢复日志**：
```
2026-03-23 20:30:15 [INFO] 从 checkpoint 恢复训练: .../checkpoint-250
2026-03-23 20:30:20 [INFO] 加载 optimizer 状态
2026-03-23 20:30:25 [INFO] 加载 scheduler 状态
2026-03-23 20:30:30 [INFO] 从 step 251 继续训练
{'loss': 2.345, 'learning_rate': 9.8e-06, 'epoch': 0.025}
```

---

## 🆘 常见问题

### Q1: 训练中断后，检查点损坏怎么办？

**解决**：从上一个检查点恢复
```bash
# 删除损坏的最新检查点
rm -rf outputs_qwen35/sft_model/checkpoint-250

# 重新运行，会自动使用 checkpoint-200
bash scripts/run_sft_qwen35.sh
```

---

### Q2: 想清理所有检查点重新开始

```bash
# 清空所有检查点
rm -rf outputs_qwen35/sft_model/*

# 重新开始训练
bash scripts/run_sft_qwen35.sh
```

---

### Q3: 检查点占用太多空间

```bash
# 方案 1: 减少保留数量（修改 config.py）
"save_total_limit": 2,

# 方案 2: 手动删除旧检查点（保留最新的）
cd outputs_qwen35/sft_model
ls -t | tail -n +6 | xargs rm -rf  # 保留最新 5 个
```

---

### Q4: 如何查看恢复后的学习率/epoch 是否正确？

查看恢复后的第一条日志，应该显示正确的 epoch 和 learning_rate：

```bash
grep "epoch\|learning_rate" outputs_qwen35/sft_logs/sft_training.log | tail -5
```

---

## ✅ 总结

1. **当前无法恢复** - 需要重新训练（因为 step 180 < SAVE_STEPS 200）
2. **代码已支持断点续传** - 下次有检查点会自动恢复
3. **推荐配置** - `SAVE_STEPS=50` + `tmux` + `SwanLab`
4. **预防措施** - 使用 tmux/screen 避免终端关闭导致中断

**立即行动**：
```bash
tmux new -s training
SAVE_STEPS=50 LOGGER=swanlab bash scripts/run_sft_qwen35.sh
```

祝训练顺利！如果再次中断，现在可以无缝恢复了 🚀
