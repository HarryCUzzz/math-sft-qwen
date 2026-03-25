
"""Centralized configuration for the Qwen3.5-4B-Base math post-training stack."""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_EVAL = PROJECT_ROOT / "data" / "eval_datasets"
DATA_QWEN35 = PROJECT_ROOT / "data" / "qwen35_v2"
DATA_QWEN35_PROCESSED = DATA_QWEN35 / "processed"
DATA_QWEN35_SMOKE = DATA_QWEN35 / "smoke"
DATA_QWEN35_MANIFESTS = DATA_QWEN35 / "manifests"
SFT_TRAIN_PATH = DATA_QWEN35_PROCESSED / "sft_train.jsonl"
SFT_EVAL_PATH = DATA_QWEN35_PROCESSED / "sft_eval.jsonl"
RL_TRAIN_PATH = DATA_QWEN35_PROCESSED / "rl_train.jsonl"
SMOKE_OVERFIT_PATH = DATA_QWEN35_SMOKE / "overfit_50.jsonl"
SMOKE_SFT_PATH = DATA_QWEN35_SMOKE / "sft_pilot_200.jsonl"
OUTPUT_BASE = PROJECT_ROOT / "outputs_qwen35"

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "/home/lyl/models/Qwen/Qwen3.5-4B-Base")

LOGGER_TYPE = os.environ.get("LOGGER", "tensorboard").lower()
SWANLAB_PROJECT = os.environ.get("SWANLAB_PROJECT", "Qwen3.5-4B-Base-MathRL")
SWANLAB_WORKSPACE = os.environ.get("SWANLAB_WORKSPACE")

THINKING_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Think through the problem carefully inside "
    "<think>...</think> tags, then provide the final answer as `Final answer: \\boxed{...}`."
)

USE_4BIT_QUANTIZATION = os.environ.get("USE_4BIT", "false").lower() == "true"
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
}


def get_report_to():
    if LOGGER_TYPE == "none":
        return []
    if LOGGER_TYPE == "swanlab":
        return ["swanlab"]
    if LOGGER_TYPE == "wandb":
        return ["wandb"]
    return ["tensorboard"]


DEFAULT_REPORT_TO = get_report_to()

SFT_CONFIG_SINGLE = {
    "model_name_or_path": MODEL_PATH,
    "trust_remote_code": True,
    "use_4bit": USE_4BIT_QUANTIZATION,
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_dropout": 0.05,
    "cutoff_len": 3072,
    "preprocessing_num_workers": 4,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 8e-5,
    "num_train_epochs": 1.0,
    "max_steps": 2200,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "seed": 42,
    "report_to": None,
}

SFT_CONFIG_MULTI = {
    **SFT_CONFIG_SINGLE,
    "gradient_accumulation_steps": 4,
}

GRPO_CONFIG_SINGLE = {
    "model_name": MODEL_PATH,
    "use_4bit": USE_4BIT_QUANTIZATION,
    "learning_rate": 8e-7,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_generations": 4,
    "max_prompt_length": 1024,
    "max_completion_length": 512,
    "beta": 0.04,
    "temperature": 0.9,
    "logging_steps": 5,
    "save_steps": 100,
    "max_steps": 300,
    "seed": 42,
    "gradient_checkpointing": True,
    "bf16": True,
}

GRPO_CONFIG_MULTI = {
    **GRPO_CONFIG_SINGLE,
    "gradient_accumulation_steps": 2,
    "max_steps": 350,
}

EVAL_CONFIG = {
    "base_model": MODEL_PATH,
    "max_new_tokens": 1024,
    "batch_size": 16,
    "temperature": 0.0,
    "do_sample": False,
    "torch_compile": False,
    "use_flash_attention": True,
    "save_details": True,
    "num_eval_samples": None,
}

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


def get_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_sft_config(mode=None):
    if mode is None:
        mode = "multi" if get_num_gpus() >= 2 else "single"

    config = SFT_CONFIG_MULTI.copy() if mode == "multi" else SFT_CONFIG_SINGLE.copy()

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
    if "MAX_STEPS" in os.environ:
        config["max_steps"] = int(os.environ["MAX_STEPS"])
    if "GRAD_ACCUM_STEPS" in os.environ:
        config["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM_STEPS"])
    if "SAVE_STEPS" in os.environ:
        config["save_steps"] = int(os.environ["SAVE_STEPS"])
    if "WARMUP_RATIO" in os.environ:
        config["warmup_ratio"] = float(os.environ["WARMUP_RATIO"])
    if "USE_4BIT" in os.environ:
        config["use_4bit"] = os.environ["USE_4BIT"].lower() == "true"

    config["report_to"] = DEFAULT_REPORT_TO
    return config


def get_grpo_config(mode=None):
    if mode is None:
        mode = "multi" if get_num_gpus() >= 2 else "single"

    config = GRPO_CONFIG_MULTI.copy() if mode == "multi" else GRPO_CONFIG_SINGLE.copy()

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
    if "USE_4BIT" in os.environ:
        config["use_4bit"] = os.environ["USE_4BIT"].lower() == "true"

    return config


def print_config_summary(config, name="Config"):
    print(f"\n{'=' * 60}")
    print(f"{name} Summary")
    print(f"{'=' * 60}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")
