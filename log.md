# 项目日志

---

## Stage A — 数据处理阶段总结

### 一、数据来源

从 HuggingFace 下载两个数学推理数据集：

- **DeepMath-103K** (`zwhe99/DeepMath-103K`)：大规模数学推理题目与解答
- **Big-Math-RL-Verified** (`SynthLabsAI/Big-Math-RL-Verified`)：经过强化学习验证的高质量数学数据

---

### 二、格式统一（Normalization）

两个数据集字段名不统一，分别通过 `normalize_deepmath()` 和 `normalize_bigmath()` 转换为统一结构：

```json
{
  "question": "...",
  "answer": "...",
  "solution": "...",
  "source": "DeepMath-103K | Big-Math-RL-Verified"
}
```

字段映射策略：优先取标准字段名，若不存在则 fallback 到备选字段名（如 `problem`/`input`/`output` 等）。

---

### 三、过滤流水线（按顺序执行，统计每步剔除数量）

| 步骤                | 规则                        | 具体操作                                                                                        |
| ------------------- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| **1. 长度过滤**     | `10 ≤ len(question) ≤ 2000` | 问题过短（<10字符）视为不完整；过长（>2000字符）视为含多余信息                                  |
| **2. 答案格式过滤** | 必须有可用答案来源          | 检查 `answer` 字段是否非空，**或** `solution` 中是否包含 `\boxed{}`（支持嵌套大括号的手工解析） |
| **3. 乱码检测**     | 有效字符占比 ≥ 70%          | 统计 ASCII + 中文 + 日文 + LaTeX符号 的占比，低于阈值则丢弃                                     |
| **4. 哈希去重**     | 基于问题文本 MD5            | 对 `question` 计算 MD5，已见过的 hash 直接跳过                                                  |

---

### 四、格式转换

**SFT 格式（Alpaca）**：转换为 LLaMA-Factory 可直接使用的格式：

```json
{
  "instruction": "Solve the following math problem step by step. Put your final answer in \\boxed{}.",
  "input": "<question>",
  "output": "<step-by-step solution with \\boxed{answer}>"
}
```

- 若 `solution` 已含 `\boxed{}`，原样保留
- 若无 `\boxed{}`，在末尾追加 `\boxed{answer}`
- 若无 `solution` 只有 `answer`，生成 `The answer is \boxed{answer}`

**RL 格式（GRPO 用）**：

```json
{
  "prompt": "Solve... \n\n<question>",
  "reference_answer": "<boxed答案 或 规范化answer>"
}
```

参考答案优先从 `solution` 中提取 `\boxed{}` 内的内容，其次才用 `answer` 字段。

---

### 五、数据划分

使用固定随机种子（`seed=42`）打乱后按 **95% / 5%** 划分：

- `sft_train.json`：SFT 训练集
- `sft_eval.json`：SFT 验证集
- `rl_train.json`：RL 训练 prompts（全量过滤后数据，不做划分）

---

### 六、输出产物

```
data/filtered/sft_train.json   # SFT 训练集（alpaca格式）
data/filtered/sft_eval.json    # SFT 验证集（alpaca格式）
data/rl_prompts/rl_train.json  # RL训练prompts + 参考答案
data/filtered/stats.json       # 各步骤过滤统计 + 问题长度分布 + 数据来源分布
```
