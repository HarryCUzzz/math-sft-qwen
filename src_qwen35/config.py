"""
Qwen3.5-4B 训练配置管理
========================
集中管理所有训练、评估相关的配置参数。
支持单卡/双卡 A6000 自动切换。
"""

import os
from pathlib import Path

import torch

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILTERED = PROJECT_ROOT / "data" / "filtered"
DATA_RL = PROJECT_ROOT / "data" / "rl_prompts"
DATA_EVAL = PROJECT_ROOT / "data" / "eval_datasets"
OUTPUT_BASE = PROJECT_ROOT / "outputs_qwen35"

# 模型路径 - 支持环境变量覆盖
MODEL_PATH = os.environ.get(
    "QWEN35_MODEL_PATH",
    "/home/lyl/models/Qwen/Qwen3.5-4B"
)

# ============================================================
# 实验跟踪配置 (Experiment Tracking)
# ============================================================
# 支持的日志工具: tensorboard, swanlab, wandb, none
# 可通过环境变量 LOGGER 指定: tensorboard/swanlab/wandb/none
LOGGER_TYPE = os.environ.get("LOGGER", "tensorboard").lower()

# SwanLab 配置
SWANLAB_PROJECT = os.environ.get("SWANLAB_PROJECT", "Qwen3.5-4B-MathRL")
SWANLAB_WORKSPACE = os.environ.get("SWANLAB_WORKSPACE", None)  # 留空使用默认

# 根据 LOGGER_TYPE 生成 report_to 列表
def get_report_to():
    """获取日志报告目标"""
    if LOGGER_TYPE == "none":
        return []
    elif LOGGER_TYPE == "swanlab":
        return ["swanlab"]
    elif LOGGER_TYPE == "wandb":
        return ["wandb"]
    else:  # 默认 tensorboard
        return ["tensorboard"]

DEFAULT_REPORT_TO = get_report_to()

# ============================================================
# Qwen3.5-4B 特性配置 - Thinking Mode
# ============================================================
THINKING_MODE_ENABLED = True

# Thinking Mode System Prompt - 引导模型使用 <think>...</think> 标签
THINKING_SYSTEM_PROMPT = (
    "You are a helpful math assistant. For complex problems, "
    "use <think>...</think> to show your internal reasoning process, "
    "then provide your final answer in \\boxed{}."
)

# 非 Thinking Mode System Prompt (备用)
SIMPLE_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve problems step by step "
    "and put your final answer in \\boxed{}."
)

# ============================================================
# 4-bit 量化配置 (BitsAndBytes)
# ============================================================
USE_4BIT_QUANTIZATION = os.environ.get("USE_4BIT", "false").lower() == "true"

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}

# ============================================================
# SFT 配置 - 单卡 A6000 (48GB)
# ============================================================
SFT_CONFIG_SINGLE = {
    # 模型配置
    "model_name_or_path": MODEL_PATH,
    "trust_remote_code": True,
    "use_4bit": USE_4BIT_QUANTIZATION,  # 4-bit 量化选项

    # LoRA 配置 (适配 4B 模型，显存优化)
    "lora_rank": 32,              # 从 64 降到 32，节省显存
    "lora_alpha": 64,             # alpha = 2 * rank
    "lora_target": "all",         # 对所有线性层应用 LoRA
    "lora_dropout": 0.05,

    # 数据配置
    "cutoff_len": 2048,           # 最大序列长度
    "preprocessing_num_workers": 4,

    # 训练超参数 (显存优化)
    "per_device_train_batch_size": 1,    # 从 2 降到 1
    "gradient_accumulation_steps": 32,   # 从 16 增到 32，保持有效 batch=32
    "learning_rate": 1e-5,               # 从 2e-5 降到 1e-5 (大模型用更小学习率)
    "num_train_epochs": 2.0,             # 从 3 降到 2 (大模型收敛更快)
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    # 精度与优化
    "bf16": True,
    "gradient_checkpointing": True,      # 梯度检查点节省显存
    "optim": "adamw_torch_fused",        # 使用 fused AdamW 提升速度

    # 日志与保存
    "logging_steps": 10,
    "save_steps": 500,                   # 增加保存间隔
    "save_total_limit": 5,               # 保留最近 5 个检查点（增加容错）
    "eval_strategy": "steps",
    "eval_steps": 500,

    # 其他
    "seed": 42,
    "report_to": None,  # 将在 get_sft_config() 中动态设置
}

# ============================================================
# SFT 配置 - 双卡 A6000 (48GB x 2 = 96GB)
# ============================================================
SFT_CONFIG_MULTI = {
    **SFT_CONFIG_SINGLE,

    # 双卡可以用更大的 batch
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,   # 有效 batch = 2 * 2 * 16 = 64

    # LoRA 可以用更大的 rank
    "lora_rank": 64,
    "lora_alpha": 128,
}

# ============================================================
# GRPO 配置 - 单卡 A6000
# ============================================================
GRPO_CONFIG_SINGLE = {
    "model_name": MODEL_PATH,
    "use_4bit": USE_4BIT_QUANTIZATION,  # 4-bit 量化选项

    # 训练参数
    "learning_rate": 1e-7,               # GRPO 用更小学习率
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,

    # 生成配置 (显存关键)
    "num_generations": 4,                # 从 8 降到 4，显存优化
    "max_prompt_length": 512,
    "max_completion_length": 768,        # 从 1024 降到 768

    # GRPO 参数
    "beta": 0.04,                        # KL 散度惩罚系数
    "temperature": 0.7,

    # 其他
    "logging_steps": 5,
    "save_steps": 100,
    "max_steps": 300,                    # 从 500 降到 300
    "seed": 42,

    # 优化
    "gradient_checkpointing": True,
    "bf16": True,
}

# ============================================================
# GRPO 配置 - 双卡 A6000
# ============================================================
GRPO_CONFIG_MULTI = {
    **GRPO_CONFIG_SINGLE,

    "per_device_train_batch_size": 2,
    "num_generations": 8,                # 双卡可以用更多生成
    "max_completion_length": 1024,
    "max_steps": 500,
}

# ============================================================
# 评估配置
# ============================================================
EVAL_CONFIG = {
    "base_model": MODEL_PATH,
    "max_new_tokens": 1024,
    "batch_size": 8,                     # 从 32 降到 8 (4B 模型)
    "temperature": 0.0,
    "do_sample": False,
    "torch_compile": False,              # 4B 模型不建议 compile (编译慢)
    "use_flash_attention": True,
    "save_details": True,
}

# 评估数据集配置
EVAL_DATASETS = {
    "math500": {
        "name_hf": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_key": "problem",
        "answer_key": "answer",
    },
    "gsm8k": {
        "name_hf": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
    },
    "theoremqa": {
        "name_hf": "TIGER-Lab/TheoremQA",
        "split": "test",
        "question_key": "Question",
        "answer_key": "Answer",
    },
}


# ============================================================
# 自动配置选择
# ============================================================
def get_num_gpus():
    """获取可用 GPU 数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_sft_config(mode=None):
    """
    获取 SFT 配置（支持环境变量覆盖）

    Args:
        mode: "single" 或 "multi"，如果为 None 则自动检测

    Returns:
        SFT 配置字典
    """
    if mode is None:
        mode = "multi" if get_num_gpus() >= 2 else "single"

    if mode == "multi":
        config = SFT_CONFIG_MULTI.copy()
    else:
        config = SFT_CONFIG_SINGLE.copy()

    # 环境变量覆盖（优先级最高）
    if "EPOCHS" in os.environ:
        config["num_train_epochs"] = float(os.environ["EPOCHS"])
    if "LORA_RANK" in os.environ:
        config["lora_rank"] = int(os.environ["LORA_RANK"])
    if "LORA_ALPHA" in os.environ:
        config["lora_alpha"] = int(os.environ["LORA_ALPHA"])
    if "LEARNING_RATE" in os.environ:
        config["learning_rate"] = float(os.environ["LEARNING_RATE"])
    if "MAX_LENGTH" in os.environ:
        config["cutoff_len"] = int(os.environ["MAX_LENGTH"])
    if "GRAD_ACCUM_STEPS" in os.environ:
        config["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM_STEPS"])
    if "SAVE_STEPS" in os.environ:
        config["save_steps"] = int(os.environ["SAVE_STEPS"])
    if "WARMUP_RATIO" in os.environ:
        config["warmup_ratio"] = float(os.environ["WARMUP_RATIO"])

    # 设置日志工具
    config["report_to"] = DEFAULT_REPORT_TO

    return config


def get_grpo_config(mode=None):
    """
    获取 GRPO 配置（支持环境变量覆盖）

    Args:
        mode: "single" 或 "multi"，如果为 None 则自动检测

    Returns:
        GRPO 配置字典
    """
    if mode is None:
        mode = "multi" if get_num_gpus() >= 2 else "single"

    if mode == "multi":
        config = GRPO_CONFIG_MULTI.copy()
    else:
        config = GRPO_CONFIG_SINGLE.copy()

    # 环境变量覆盖（优先级最高）
    if "MAX_STEPS" in os.environ:
        config["max_steps"] = int(os.environ["MAX_STEPS"])
    if "NUM_GENERATIONS" in os.environ:
        config["num_generations"] = int(os.environ["NUM_GENERATIONS"])
    if "MAX_COMPLETION_LENGTH" in os.environ:
        config["max_completion_length"] = int(os.environ["MAX_COMPLETION_LENGTH"])
    if "LEARNING_RATE" in os.environ:
        config["learning_rate"] = float(os.environ["LEARNING_RATE"])
    if "TEMPERATURE" in os.environ:
        config["temperature"] = float(os.environ["TEMPERATURE"])
    if "BETA" in os.environ:
        config["beta"] = float(os.environ["BETA"])
    if "GRAD_ACCUM_STEPS" in os.environ:
        config["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM_STEPS"])
    if "SAVE_STEPS" in os.environ:
        config["save_steps"] = int(os.environ["SAVE_STEPS"])

    return config


def print_config_summary(config, name="Config"):
    """打印配置摘要"""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
