"""
Stage C - GRPO 强化学习 (Group Relative Policy Optimization)
=============================================================
使用 TRL 库的 GRPOTrainer 对 SFT 模型进行强化学习后训练。

奖励函数设计（2 个组成部分）：
1. 正确性奖励 (Correctness Reward): 精确匹配 \\boxed{} 中的答案
2. 格式/结构奖励 (Format Reward): 检查推理格式是否规范

参考: https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
"""

import os
import re
import json
import logging
import torch
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _add_file_handler(log_dir: Path) -> None:
    """将日志同时写入文件，避免重复添加 handler。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "grpo_training.log"
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_file.resolve())
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.info(f"日志文件: {log_file}")


# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RL = PROJECT_ROOT / "data" / "rl_prompts"
SFT_MODEL_DIR = PROJECT_ROOT / "outputs" / "sft_model"
GRPO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grpo_model"
GRPO_LOG_DIR = PROJECT_ROOT / "outputs" / "grpo_logs"

# ============================================================
# GRPO 训练超参数
# ============================================================
GRPO_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",       # 基座模型（SFT 模型基于此）
    "sft_adapter_path": str(SFT_MODEL_DIR),     # SFT LoRA adapter 路径
    "learning_rate": 5e-7,                       # GRPO 学习率（较 SFT 更低）
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_generations": 8,                        # 每个 prompt 生成 8 个答案（Group size）
    "max_prompt_length": 512,                    # prompt 最大长度
    "max_completion_length": 1024,               # 生成的最大长度
    "beta": 0.04,                                # KL 散度惩罚系数
    "temperature": 0.7,                          # 生成温度
    "logging_steps": 5,
    "save_steps": 100,
    "max_steps": 500,                            # 最大训练步数
    "seed": 42,
}


# ============================================================
# 1. 答案提取工具函数
# ============================================================
def extract_boxed_answer(text):
    """
    从模型输出中提取 \\boxed{} 内的答案。
    支持嵌套大括号的情况。
    返回: 答案字符串，如果找不到则返回 None
    """
    if not text:
        return None

    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # 取最后一个 \boxed{}（通常最终答案在最后）
    match = matches[-1]
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i-1].strip()
    return None


def normalize_answer(answer_str):
    """
    规范化答案用于比较：
    - 去除空格和 $ 符号
    - 统一分数表示
    - 处理常见数学等价形式
    """
    if not answer_str:
        return ""
    ans = answer_str.strip()
    ans = ans.replace("$", "").replace(" ", "")
    # 去除尾部的句号
    ans = ans.rstrip(".")
    return ans


def answers_match(predicted, reference):
    """
    判断预测答案和参考答案是否匹配。
    先进行规范化，再做精确匹配。
    """
    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    return pred == ref


# ============================================================
# 2. 奖励函数定义
# ============================================================
def correctness_reward_fn(completions, reference_answers, **kwargs):
    """
    正确性奖励 (Correctness Reward):
    - 从模型输出的 \\boxed{} 中提取答案
    - 与参考答案做精确匹配
    - 正确 → 1.0, 错误 → 0.0

    参数:
        completions: 模型生成的回答列表  [str, ...]
        reference_answers: 参考答案列表    [str, ...]
    返回:
        奖励列表 [float, ...]
    """
    rewards = []
    for completion, ref_answer in zip(completions, reference_answers):
        # 从模型输出中提取答案
        predicted = extract_boxed_answer(completion)
        # 判断是否匹配
        if predicted and answers_match(predicted, ref_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_fn(completions, **kwargs):
    """
    格式/结构奖励 (Format Reward):
    检查模型输出是否遵循良好的推理格式，由三部分组成：
    1. 是否包含 \\boxed{} 格式 → 0.5 分
    2. 是否包含逐步推理（含 Step/step 字样或多行推理）→ 0.3 分
    3. 输出长度是否合理（50-1500 字符）→ 0.2 分

    参数:
        completions: 模型生成的回答列表
    返回:
        奖励列表
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # 部分 1: 是否包含 \\boxed{} 格式（0.5 分）
        if r'\boxed{' in completion:
            reward += 0.5

        # 部分 2: 是否包含逐步推理过程（0.3 分）
        # 检查是否有 "Step" 关键词或多行推理（至少 3 行非空内容）
        has_step_keyword = bool(re.search(r'(?i)(step\s*\d|first|then|next|therefore|thus|hence|so,|finally)', completion))
        lines = [l.strip() for l in completion.split('\n') if l.strip()]
        has_multi_line = len(lines) >= 3

        if has_step_keyword or has_multi_line:
            reward += 0.3

        # 部分 3: 输出长度是否合理（0.2 分）
        # 50-1500 字符为合理范围，防止过短（空洞回答）或过长（冗余输出）
        comp_len = len(completion)
        if 50 <= comp_len <= 1500:
            reward += 0.2

        rewards.append(reward)
    return rewards


# ============================================================
# 3. 数据准备
# ============================================================
def load_rl_dataset():
    """
    加载 RL 训练数据集。
    每条数据包含:
    - prompt: 数学问题（已格式化为对话模板）
    - reference_answer: 参考答案（用于正确性奖励计算）
    """
    rl_data_path = DATA_RL / "rl_train.json"
    if not rl_data_path.exists():
        raise FileNotFoundError(
            f"RL 训练数据未找到: {rl_data_path}\n"
            "请先运行 Stage A (data_selection.py) 完成数据筛选。"
        )

    with open(rl_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"加载 RL 训练数据: {len(data)} 条")
    return data


def prepare_dataset_for_grpo(raw_data, tokenizer, max_samples=None):
    """
    将原始 RL 数据转换为 GRPOTrainer 所需格式。

    GRPOTrainer 需要数据集包含:
    - "prompt": 经过 chat template 格式化的 prompt 文本
    - 自定义列用于奖励函数

    参数:
        raw_data: 从 rl_train.json 加载的原始数据
        tokenizer: 分词器（用于 apply_chat_template）
        max_samples: 限制样本数（调试用）
    """
    from datasets import Dataset

    if max_samples:
        raw_data = raw_data[:max_samples]

    processed = []
    for item in raw_data:
        # 构建对话格式的 prompt
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and put your final answer in \\boxed{}."},
            {"role": "user", "content": item["prompt"]},
        ]
        # 使用 tokenizer 的 chat template 格式化 prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processed.append({
            "prompt": formatted_prompt,
            "reference_answer": item["reference_answer"],
        })

    dataset = Dataset.from_list(processed)
    logger.info(f"GRPO 数据集准备完成: {len(dataset)} 条")
    return dataset


# ============================================================
# 4. 合并奖励函数
# ============================================================
def combined_reward_fn(completions, **kwargs):
    """
    综合奖励函数：结合正确性奖励和格式奖励。
    总奖励 = 正确性奖励 + 格式奖励
    最大可能奖励 = 1.0 + 1.0 = 2.0

    这个函数会被 GRPOTrainer 调用，
    completions 是一个 batch 中所有生成的文本。
    """
    # 从 kwargs 中获取参考答案
    reference_answers = kwargs.get("reference_answer", [])

    # 计算正确性奖励
    corr_rewards = correctness_reward_fn(completions, reference_answers)

    # 计算格式奖励
    fmt_rewards = format_reward_fn(completions)

    # 合并奖励
    total_rewards = [c + f for c, f in zip(corr_rewards, fmt_rewards)]

    return total_rewards


# ============================================================
# 5. GRPO 训练主流程
# ============================================================
def run_grpo_training():
    """
    GRPO 训练主流程:
    1. 加载 SFT 模型（基座模型 + LoRA adapter）
    2. 准备 RL 数据集
    3. 配置 GRPOTrainer
    4. 启动训练
    5. 保存模型
    """
    _add_file_handler(GRPO_LOG_DIR)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from trl import GRPOTrainer, GRPOConfig

    # ---- 步骤 1: 加载分词器 ----
    logger.info(f"加载分词器: {GRPO_CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        GRPO_CONFIG["model_name"],
        trust_remote_code=True,
        padding_side="left",  # GRPO 生成时需要左 padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 步骤 2: 加载 SFT 模型 ----
    logger.info(f"加载基座模型: {GRPO_CONFIG['model_name']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        GRPO_CONFIG["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 如果 SFT adapter 存在，加载 LoRA adapter
    sft_adapter_path = Path(GRPO_CONFIG["sft_adapter_path"])
    if sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists():
        logger.info(f"加载 SFT LoRA adapter: {sft_adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_path))
        # 合并 LoRA 权重到基座模型（GRPO 需要完整模型）
        model = model.merge_and_unload()
        logger.info("LoRA adapter 已合并到基座模型")
    else:
        logger.warning(f"SFT adapter 未找到: {sft_adapter_path}, 将直接在基座模型上进行 GRPO")
        model = base_model

    # ---- 步骤 3: 准备数据集 ----
    raw_data = load_rl_dataset()
    train_dataset = prepare_dataset_for_grpo(raw_data, tokenizer)

    # ---- 步骤 4: 配置 GRPOTrainer ----
    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRPO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(GRPO_OUTPUT_DIR),
        logging_dir=str(GRPO_LOG_DIR),
        learning_rate=GRPO_CONFIG["learning_rate"],
        num_train_epochs=GRPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        num_generations=GRPO_CONFIG["num_generations"],
        max_prompt_length=GRPO_CONFIG["max_prompt_length"],
        max_completion_length=GRPO_CONFIG["max_completion_length"],
        beta=GRPO_CONFIG["beta"],
        logging_steps=GRPO_CONFIG["logging_steps"],
        save_steps=GRPO_CONFIG["save_steps"],
        max_steps=GRPO_CONFIG["max_steps"],
        seed=GRPO_CONFIG["seed"],
        bf16=True,
        report_to=["tensorboard", "swanlab"],
        remove_unused_columns=False,  # 保留 reference_answer 列用于奖励计算
    )

    # ---- 步骤 5: 创建 GRPOTrainer 并训练 ----
    logger.info("=" * 60)
    logger.info("开始 GRPO 训练")
    logger.info(f"  Group size (num_generations): {GRPO_CONFIG['num_generations']}")
    logger.info(f"  KL 系数 (beta): {GRPO_CONFIG['beta']}")
    logger.info(f"  学习率: {GRPO_CONFIG['learning_rate']}")
    logger.info(f"  最大步数: {GRPO_CONFIG['max_steps']}")
    logger.info(f"  生成温度: {GRPO_CONFIG['temperature']}")
    logger.info("=" * 60)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=combined_reward_fn,
        tokenizer=tokenizer,
    )

    # 启动训练
    train_result = trainer.train()

    # ---- 步骤 6: 保存模型和指标 ----
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"GRPO 训练完成！模型已保存至: {GRPO_OUTPUT_DIR}")
    logger.info(f"训练指标: {metrics}")

    return metrics


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage C: GRPO 强化学习训练")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="覆盖默认的最大训练步数")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="覆盖默认的 group size")
    args = parser.parse_args()

    # 允许通过命令行覆盖默认配置
    if args.max_steps:
        GRPO_CONFIG["max_steps"] = args.max_steps
    if args.num_generations:
        GRPO_CONFIG["num_generations"] = args.num_generations

    logger.info("=" * 60)
    logger.info("Stage C - GRPO 强化学习训练开始")
    logger.info("=" * 60)

    metrics = run_grpo_training()

    logger.info("Stage C 完成！")


if __name__ == "__main__":
    main()
