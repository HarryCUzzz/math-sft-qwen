"""
Stage C - GRPO 强化学习 (Group Relative Policy Optimization) - Qwen3.5-4B 版本
===============================================================================
使用 TRL 库的 GRPOTrainer 对 SFT 模型进行强化学习后训练。

奖励函数设计（2 个组成部分）：
1. 正确性奖励 (Correctness Reward): 精确匹配 \\boxed{} 中的答案
2. 格式/结构奖励 (Format Reward): 检查推理格式是否规范

适配 A6000 (48GB) 显存优化：
- 减少 num_generations (从 8 降到 4)
- 缩短 max_completion_length (从 1024 降到 768)
- 启用 gradient_checkpointing

输出：
- outputs_qwen35/grpo_model/    (GRPO 模型检查点)
- outputs_qwen35/grpo_logs/     (训练日志)
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from config import (
    DATA_RL,
    OUTPUT_BASE,
    QUANTIZATION_CONFIG,
    THINKING_SYSTEM_PROMPT,
    LOGGER_TYPE,
    SWANLAB_PROJECT,
    SWANLAB_WORKSPACE,
    DEFAULT_REPORT_TO,
    get_grpo_config,
    print_config_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 目录配置
SFT_MODEL_DIR = OUTPUT_BASE / "sft_model"
GRPO_OUTPUT_DIR = OUTPUT_BASE / "grpo_model"
GRPO_LOG_DIR = OUTPUT_BASE / "grpo_logs"


def _add_file_handler(log_dir: Path) -> None:
    """将日志同时写入文件"""
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
# 答案提取工具函数
# ============================================================
def extract_boxed_answer(text):
    """
    从模型输出中提取 \\boxed{} 内的答案。
    支持嵌套大括号的情况。
    """
    if not text:
        return None

    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    match = matches[-1]
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()
    return None


def normalize_answer(answer_str):
    """规范化答案用于比较"""
    if not answer_str:
        return ""
    ans = answer_str.strip()
    ans = ans.replace("$", "").replace(" ", "")
    ans = ans.rstrip(".")
    return ans


def answers_match(predicted, reference):
    """判断预测答案和参考答案是否匹配"""
    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    return pred == ref


# ============================================================
# 奖励函数定义
# ============================================================
def correctness_reward_fn(completions, reference_answers, **kwargs):
    """
    正确性奖励 (Correctness Reward):
    - 从模型输出的 \\boxed{} 中提取答案
    - 与参考答案做精确匹配
    - 正确 → 1.0, 错误 → 0.0
    """
    rewards = []
    for completion, ref_answer in zip(completions, reference_answers):
        predicted = extract_boxed_answer(completion)
        if predicted and answers_match(predicted, ref_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_fn(completions, **kwargs):
    """
    格式/结构奖励 (Format Reward) - 适配 Qwen3.5-4B Thinking Mode:
    1. 是否包含 \\boxed{} 格式 → 0.4 分
    2. 是否使用 Thinking Mode (<think>...</think>) → 0.3 分 (新增)
    3. 是否包含逐步推理 → 0.2 分
    4. 输出长度是否合理（100-2000 字符）→ 0.1 分
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # 部分 1: 是否包含 \boxed{} 格式（0.4 分）
        if r"\boxed{" in completion:
            reward += 0.4

        # 部分 2: 是否使用 Thinking Mode（0.3 分）- 新增
        has_thinking = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        if has_thinking:
            reward += 0.3

        # 部分 3: 是否包含逐步推理过程（0.2 分）
        has_step_keyword = bool(
            re.search(
                r"(?i)(step\s*\d|first|then|next|therefore|thus|hence|so,|finally)",
                completion,
            )
        )
        lines = [line.strip() for line in completion.split("\n") if line.strip()]
        has_multi_line = len(lines) >= 3

        if has_step_keyword or has_multi_line:
            reward += 0.2

        # 部分 4: 输出长度是否合理（0.1 分）- 调整范围以适应 Thinking Mode
        comp_len = len(completion)
        if 100 <= comp_len <= 2000:
            reward += 0.1

        rewards.append(reward)
    return rewards


def combined_reward_fn(completions, **kwargs):
    """
    综合奖励函数：结合正确性奖励和格式奖励。
    总奖励 = 正确性奖励 + 格式奖励
    最大可能奖励 = 1.0 + 1.0 = 2.0
    """
    reference_answers = kwargs.get("reference_answer", [])
    corr_rewards = correctness_reward_fn(completions, reference_answers)
    fmt_rewards = format_reward_fn(completions)
    total_rewards = [c + f for c, f in zip(corr_rewards, fmt_rewards)]
    return total_rewards


# ============================================================
# 数据准备
# ============================================================
def load_rl_dataset():
    """加载 RL 训练数据集"""
    rl_data_path = DATA_RL / "rl_train.json"
    if not rl_data_path.exists():
        raise FileNotFoundError(
            f"RL 训练数据未找到: {rl_data_path}\n"
            "请确保数据文件存在于 data/rl_prompts/ 目录中"
        )

    with open(rl_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"加载 RL 训练数据: {len(data)} 条")
    return data


def prepare_dataset_for_grpo(raw_data, tokenizer, max_samples=None):
    """将原始 RL 数据转换为 GRPOTrainer 所需格式"""
    if max_samples:
        raw_data = raw_data[:max_samples]

    processed = []
    for item in raw_data:
        messages = [
            {
                "role": "system",
                "content": THINKING_SYSTEM_PROMPT,
            },
            {"role": "user", "content": item["prompt"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processed.append(
            {
                "prompt": formatted_prompt,
                "reference_answer": item["reference_answer"],
            }
        )

    dataset = Dataset.from_list(processed)
    logger.info(f"GRPO 数据集准备完成: {len(dataset)} 条")
    return dataset


# ============================================================
# GRPO 训练主流程
# ============================================================
def run_grpo_training(mode=None):
    """
    GRPO 训练主流程

    Args:
        mode: "single" 或 "multi"，如果为 None 则自动检测
    """
    _add_file_handler(GRPO_LOG_DIR)

    # 获取配置
    config = get_grpo_config(mode)
    print_config_summary(config, "GRPO Training Config")

    # 初始化 SwanLab（如果启用）
    if LOGGER_TYPE == "swanlab":
        try:
            import swanlab
            swanlab.init(
                project=SWANLAB_PROJECT,
                workspace=SWANLAB_WORKSPACE,
                experiment_name=f"GRPO-{mode or 'auto'}",
                description=f"Qwen3.5-4B GRPO Training ({mode or 'auto'} GPU)",
                config=config,
            )
            logger.info(f"SwanLab 初始化成功: 项目={SWANLAB_PROJECT}")
        except ImportError:
            logger.warning("SwanLab 未安装，将退回到 TensorBoard")
        except Exception as e:
            logger.warning(f"SwanLab 初始化失败: {e}，将退回到 TensorBoard")

    logger.info("=" * 60)
    logger.info("Stage C - Qwen3.5-4B GRPO 训练开始")
    logger.info("=" * 60)

    # ---- 步骤 1: 加载分词器 ----
    logger.info(f"加载分词器: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 步骤 2: 加载 SFT 模型 ----
    logger.info(f"加载基座模型: {config['model_name']}")

    # 支持 4-bit 量化或 bf16 精度
    if config.get("use_4bit", False):
        logger.info("启用 4-bit 量化加载模型...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        logger.info("使用 bf16 精度加载模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # 如果 SFT adapter 存在，加载 LoRA adapter
    if SFT_MODEL_DIR.exists() and (SFT_MODEL_DIR / "adapter_config.json").exists():
        logger.info(f"加载 SFT LoRA adapter: {SFT_MODEL_DIR}")
        model = PeftModel.from_pretrained(base_model, str(SFT_MODEL_DIR))
        model = model.merge_and_unload()
        logger.info("LoRA adapter 已合并到基座模型")
    else:
        logger.warning(
            f"SFT adapter 未找到: {SFT_MODEL_DIR}, 将直接在基座模型上进行 GRPO"
        )
        model = base_model

    # 修复 TRL 库兼容性问题
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # 启用梯度检查点
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("启用 gradient_checkpointing")

    # ---- 步骤 3: 准备数据集 ----
    raw_data = load_rl_dataset()
    train_dataset = prepare_dataset_for_grpo(raw_data, tokenizer)

    # ---- 步骤 4: 配置 GRPOTrainer ----
    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRPO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(GRPO_OUTPUT_DIR),
        logging_dir=str(GRPO_LOG_DIR),
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        beta=config["beta"],
        temperature=config["temperature"],  # 修复: 添加 temperature 参数
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        max_steps=config["max_steps"],
        seed=config["seed"],
        bf16=config["bf16"],
        report_to=DEFAULT_REPORT_TO,
        remove_unused_columns=False,
    )

    # ---- 步骤 5: 创建 GRPOTrainer 并训练 ----
    logger.info("=" * 60)
    logger.info("开始 GRPO 训练")
    logger.info(f"  Group size (num_generations): {config['num_generations']}")
    logger.info(f"  KL 系数 (beta): {config['beta']}")
    logger.info(f"  学习率: {config['learning_rate']}")
    logger.info(f"  最大步数: {config['max_steps']}")
    logger.info(f"  生成温度: {config['temperature']}")
    logger.info("=" * 60)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=combined_reward_fn,
        processing_class=tokenizer,
    )

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


def main():
    parser = argparse.ArgumentParser(description="Stage C: Qwen3.5-4B GRPO 训练")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["single", "multi"],
        help="训练模式: single (单卡) 或 multi (多卡)，默认自动检测",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="覆盖默认的最大训练步数",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="覆盖默认的 group size",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage C - GRPO 强化学习训练开始")
    logger.info("=" * 60)

    run_grpo_training(mode=args.mode)
    logger.info("Stage C 完成！")


if __name__ == "__main__":
    main()
