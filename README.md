# AIMS5740 Final Project

## Topic 1: Data Selection + RL for LLMs (Math / STEM)

通过 **数据筛选 + GRPO 强化学习** 提升 Qwen2.5-0.5B 在数学推理任务上的准确率。

如果你更关心这个项目下一步应该如何补全 ablation、奖励分析与失败模式分析，可以直接看 [docs/PROJECT_UPGRADE_NOTES.md](docs/PROJECT_UPGRADE_NOTES.md)。

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 克隆 LLaMA-Factory（SFT 备选方案）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

# 按顺序执行 pipeline
python src/data_selection.py    # Stage A: 数据筛选
python src/sft_training.py      # Stage B: SFT 微调
python src/grpo_training.py     # Stage C: GRPO 强化学习
python src/evaluation.py        # Stage D: 评估
python src/analysis.py          # Stage E: 分析
```

### 项目结构

```
5740project/
├── README.md                  # 项目说明
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略规则
├── docs/
│   └── PROJECT_GUIDE.md       # 完整实现指南（可作为 prompt 复用）
├── configs/                   # 训练配置文件（由代码自动生成）
├── src/
│   ├── __init__.py
│   ├── data_selection.py      # Stage A: 数据筛选与清洗
│   ├── sft_training.py        # Stage B: 有监督微调 (LoRA)
│   ├── grpo_training.py       # Stage C: GRPO 强化学习
│   ├── evaluation.py          # Stage D: 多 benchmark 评估
│   └── analysis.py            # Stage E: 可视化分析与报告
├── data/                      # 数据目录（运行时生成）
│   ├── raw/
│   ├── filtered/
│   └── rl_prompts/
├── outputs/                   # 输出目录（运行时生成）
│   ├── sft_model/
│   ├── grpo_model/
│   ├── eval_results/
│   └── analysis/
└── LLaMA-Factory/             # LLaMA-Factory 框架（git clone）
```

### 技术栈

| 组件 | 选型 |
|------|------|
| 基座模型 | Qwen2.5-0.5B |
| SFT 微调 | Transformers + PEFT (LoRA) / LLaMA-Factory |
| 强化学习 | TRL GRPOTrainer |
| 评估 | MATH-500 / GSM8K / TheoremQA |
| 可视化 | matplotlib |

详细步骤见 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)。
