# AIMS5740 Final Project - 完整实现指南

## Topic 1: Data Selection + RL for LLMs (Math / STEM)

---

## 项目概述

本项目旨在通过 **数据筛选 + 强化学习（RL）后训练** 的方式，提升小型语言模型在数学和 STEM 推理任务上的准确率。整体流水线参考 DeepSeek-R1 训练范式，包含：数据筛选 → SFT 冷启动 → GRPO 强化学习 → 评估与分析。

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 基座模型 | Qwen2.5-0.5B |
| 数据集 | DeepMath-103K / Big-Math-RL-Verified |
| SFT 框架 | LLaMA-Factory |
| RL 框架 | TRL (trl) + GRPO |
| 评估框架 | lm-eval-harness |
| 深度学习框架 | PyTorch + Transformers + PEFT |
| 数据处理 | datasets (HuggingFace) + pandas |

---

## Stage A - 数据筛选 (Data Selection)

### 目标
从大规模数学数据集中筛选出高质量、格式规范、可验证的训练样本。

### 步骤

#### A1. 下载数据集
```bash
# 使用 HuggingFace datasets 下载
python src/data_selection.py --step download
```

**数据集来源：**
- `zwhe99/DeepMath-103K` — 103K 条数学推理数据
- `SynthLabsAI/Big-Math-RL-Verified` — 大规模 RL 验证数学题

#### A2. 数据清洗与过滤规则
1. **移除模糊问题**：去除问题文本长度 < 10 字符或 > 2000 字符的样本
2. **答案格式规范化**：强制要求最终答案以 `\boxed{}` 格式呈现
3. **难度控制**：过滤掉过于简单（答案为纯数字且 < 10）或过于复杂的样本
4. **去重**：基于问题文本的 hash 去重
5. **质量筛选**：移除包含乱码、非英文、或格式严重损坏的样本

#### A3. 数据格式转换
将筛选后的数据转换为 LLaMA-Factory 所需的 alpaca 格式：
```json
{
  "instruction": "Solve the following math problem step by step.",
  "input": "<math problem>",
  "output": "<step-by-step solution>\n\nThe answer is \\boxed{<answer>}"
}
```

#### A4. 输出统计
- 筛选前/后的数据量对比
- 问题长度分布
- 答案类型分布

---

## Stage B - 有监督微调 (Supervised Fine-Tuning)

### 目标
使用筛选后的数据对基座模型进行 SFT，使其学会数学推理的基本格式和逻辑。

### 步骤

#### B1. 配置 LLaMA-Factory 数据集
在 `LLaMA-Factory/data/dataset_info.json` 中注册数据集。

#### B2. 启动 SFT 训练
```bash
python src/sft_training.py
```

**关键超参数：**
- 学习率：2e-5
- Batch size：4 (gradient accumulation: 8)
- Epochs：3
- LoRA rank：64, alpha：128
- 最大序列长度：2048
- 优化器：AdamW + cosine scheduler

#### B3. 评估 SFT 效果
- 对比基础模型 vs SFT 模型的 loss 曲线
- 在 MATH-500 上做 few-shot 测试
- 输出样例对比

---

## Stage C - 强化学习 (Reinforcement Learning with GRPO)

### 目标
使用 GRPO（Group Relative Policy Optimization）进一步提升模型的推理正确性。

### 步骤

#### C1. 准备 RL 数据集
从筛选后的数据中提取 prompt（仅问题，不含答案），同时保留参考答案用于奖励计算。

#### C2. 设计奖励函数（至少 2 个组成部分）
1. **正确性奖励（Correctness Reward）**：
   - 精确匹配 `\boxed{}` 中的答案与参考答案
   - 正确 → +1.0, 错误 → 0.0

2. **格式奖励（Format Reward）**：
   - 输出是否包含 `\boxed{}` 格式 → +0.5
   - 是否包含逐步推理过程（含 "Step" 或分行推理）→ +0.3
   - 输出长度是否合理（50-1500 tokens）→ +0.2

#### C3. GRPO 训练
```bash
python src/grpo_training.py
```

**关键超参数：**
- 学习率：5e-7
- Group size：8（每个 prompt 生成 8 个回答）
- KL 系数 (beta)：0.04
- 最大生成长度：1024
- 训练步数：500-1000
- 温度：0.7

#### C4. 训练监控
- 奖励曲线（正确性奖励、格式奖励、总奖励）
- KL 散度变化
- 生成样本质量抽查

---

## Stage D - 评估 (Evaluation)

### 目标
在标准 benchmark 上对比基座模型、SFT 模型、GRPO 模型的表现。

### 步骤

#### D1. 评估数据集
- **MATH-500** (`HuggingFaceH4/MATH-500`)：500 道竞赛级数学题
- **GSM8K** (`openai/gsm8k`)：小学数学应用题
- **TheoremQA** (`TIGER-Lab/TheoremQA`)：定理推理

#### D2. 运行评估
```bash
python src/evaluation.py
```

#### D3. 输出指标
- 各数据集的精确匹配准确率 (Exact Match Accuracy)
- 各模型（Base / SFT / GRPO）对比表格
- 样例输出对比

---

## Stage E - 分析 (Analysis)

### 分析方向

#### E1. RL 对推理正确性 vs 通用能力的影响
- 对比 GRPO 模型在数学 benchmark 和通用问答上的表现
- 分析 RL 是否导致"能力遗忘"

#### E2. 数据过滤策略的敏感性
- 对比不同过滤策略（严格 vs 宽松）训练出的模型效果
- 消融实验：分别去除某一过滤规则，观察影响

#### E3. 失效模式分析
- **格式偏移**：模型是否学会了奖励 hacking（只输出 \boxed{} 而无推理）
- **奖励利用**：模型是否在格式上得分高但实际答案错误
- **通用性能变化**：长文本推理 vs 短文本推理的不同表现

---

## 硬件需求与时间估算

> 以下数据基于 **Qwen2.5-0.5B** 模型，在单卡环境下的估算，供参考。

### GPU 显存需求

| 阶段 | 任务类型 | 最低显存 | 推荐显存 | 说明 |
|------|---------|---------|---------|------|
| Stage A - 数据筛选 | CPU 任务 | — | — | 无需 GPU，纯 Python 数据处理 |
| Stage B - SFT (LoRA) | 训练 | 8 GB | 16 GB | LoRA rank=64，batch=4，seq=2048 |
| Stage C - GRPO | 训练 | 16 GB | 24 GB | 需同时生成 8 个候选，显存占用约 SFT 的 2x |
| Stage D - 评估 | 推理 | 4 GB | 8 GB | 贪心解码，batch=1 |
| Stage E - 分析 | CPU 任务 | — | — | 无需 GPU，仅绘图 |

### 各阶段预计耗时

| 阶段 | RTX 3090/4090 (24GB) | A100 (40GB) | 备注 |
|------|---------------------|-------------|------|
| Stage A - 数据筛选 | 30–60 min | 30–60 min | 受网络和数据集大小影响 |
| Stage B - SFT | 4–6 h | 2–3 h | 3 epoch，~50K 样本，LoRA |
| Stage C - GRPO | 6–10 h | 3–5 h | 500 step，num_generations=8 |
| Stage D - 评估 | 1–2 h | 0.5–1 h | 三个数据集全量评估 |
| Stage E - 分析 | < 5 min | < 5 min | 读取已有结果，CPU 绘图 |
| **总计** | **≈ 12–18 h** | **≈ 6–10 h** | |

> **提示**：若显存不足，可在 `src/sft_training.py` 中将 `lora_rank` 调低至 32，或在 `src/grpo_training.py` 中将 `num_generations` 调低至 4，以减少显存占用。

---

## 预期实验结果

> 以下为基于 Qwen2.5-0.5B 规模和同类工作经验的合理预期区间，实际结果因数据过滤策略、训练轮数等超参数而有所浮动。

### Benchmark 准确率预测（Exact Match Accuracy %）

| 模型 | MATH-500 | GSM8K | TheoremQA | 备注 |
|------|---------|-------|-----------|------|
| Base (Qwen2.5-0.5B) | 30–38% | 40–52% | 14–20% | 0-shot，无 chat template |
| SFT | 38–47% | 52–65% | 18–26% | 预期提升：格式稳定，推理链更完整 |
| GRPO | 43–53% | 57–70% | 20–29% | 预期提升：正确性进一步提高，但提升幅度小于 SFT |

> **说明**：TheoremQA 为分布外 (OOD) 数据集，提升幅度通常小于 GSM8K；GRPO 对 GSM8K 这类有明确数值答案的任务提升最显著。

### 各阶段典型提升趋势

```
MATH-500 准确率（示意）
                         ┌───────────────────────────────────────────────┐
  SFT vs Base            │  Base: ~34%  →  SFT: ~43%  (+9%)             │
  GRPO vs SFT            │  SFT:  ~43%  →  GRPO: ~48%  (+5%)            │
  GRPO vs Base (总提升)   │  Base: ~34%  →  GRPO: ~48%  (+14%)           │
                         └───────────────────────────────────────────────┘
```

### 预期观察到的现象

#### 正向现象
- **SFT 效果明显**：模型从零散输出学会 `\boxed{}` 格式和逐步推理，正确率跳升最大
- **GRPO 持续优化**：奖励信号引导模型在推理链质量上进一步细化，GSM8K 提升尤为明显
- **格式一致性提升**：SFT 后 `\boxed{}` 出现率从 Base 的 ~30% 提升到 ~90% 以上

#### 需关注的风险
| 风险 | 描述 | 判断方法 |
|------|------|---------|
| 奖励利用 (Reward Hacking) | 模型只输出 `\boxed{42}` 无任何推理过程 | 检查生成文本长度分布，平均长度不应 < 50 tokens |
| 格式偏移 | GRPO 后格式奖励高但正确性低 | 对比格式奖励分 vs 正确性奖励分的相关性 |
| 通用能力遗忘 | 非数学问题回答能力下降 | 可在 TriviaQA 或 HellaSwag 上做对比评测 |
| 训练不稳定 | GRPO loss 出现 NaN 或奖励不收敛 | 检查 KL 散度，若 > 0.5 建议调低学习率 |

---

## 快速运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 克隆 LLaMA-Factory（SFT 使用）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

# 3. 数据筛选
python src/data_selection.py

# 4. SFT 训练
python src/sft_training.py

# 5. GRPO 强化学习
python src/grpo_training.py

# 6. 评估
python src/evaluation.py

# 7. 分析
python src/analysis.py
```

---

## 项目结构

```
5740project/
├── README.md                  # 项目说明
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略规则
├── docs/
│   └── PROJECT_GUIDE.md       # 本文档 - 完整实现指南
├── configs/                   # 训练配置文件（由代码自动生成）
├── src/
│   ├── __init__.py
│   ├── data_selection.py      # Stage A: 数据筛选
│   ├── sft_training.py        # Stage B: SFT 训练
│   ├── grpo_training.py       # Stage C: GRPO 强化学习
│   ├── evaluation.py          # Stage D: 评估
│   └── analysis.py            # Stage E: 分析
├── data/
│   ├── raw/                   # 原始数据
│   ├── filtered/              # 筛选后数据
│   └── rl_prompts/            # RL 训练用 prompts
├── outputs/
│   ├── sft_model/             # SFT 模型检查点
│   ├── grpo_model/            # GRPO 模型检查点
│   ├── eval_results/          # 评估结果
│   └── analysis/              # 分析图表
└── LLaMA-Factory/             # LLaMA-Factory 框架（git clone）
```
