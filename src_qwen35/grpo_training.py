"""Stage C - lightweight GRPO continuation on top of the best SFT LoRA adapter."""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from answer_utils import answer_match_grade, extract_candidate_answer, has_valid_final_structure
from config import (
    DEFAULT_REPORT_TO,
    LOGGER_TYPE,
    OUTPUT_BASE,
    QUANTIZATION_CONFIG,
    RL_TRAIN_PATH,
    SWANLAB_PROJECT,
    SWANLAB_WORKSPACE,
    THINKING_SYSTEM_PROMPT,
    get_grpo_config,
    get_sft_output_dirs,
    print_config_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRPO_OUTPUT_DIR = OUTPUT_BASE / "grpo_model"
GRPO_LOG_DIR = OUTPUT_BASE / "grpo_logs"
REWARD_CONFIG = None
REWARD_TOKENIZER = None


def _resolve_parent_sft_dir() -> Path:
    requested_stage = os.environ.get("GRPO_PARENT_SFT_STAGE", "").strip().lower()
    if requested_stage:
        parent_dir, _ = get_sft_output_dirs(requested_stage)
        return parent_dir
    calibration_dir, _ = get_sft_output_dirs("calibration")
    if (calibration_dir / "adapter_config.json").exists():
        return calibration_dir
    main_dir, _ = get_sft_output_dirs("main")
    return main_dir


def _add_file_handler(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "grpo_training.log"
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == str(log_file.resolve())
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)


def _current_device_map():
    if not torch.cuda.is_available():
        return None
    return {"": int(os.environ.get("LOCAL_RANK", "0"))}


def _completion_token_count(completion: str) -> int:
    if REWARD_TOKENIZER is None:
        return max(1, len(completion.split()))
    return len(REWARD_TOKENIZER(completion, add_special_tokens=False)["input_ids"])


def correctness_reward_fn(completions, reference_answer, parser_type=None, **kwargs):
    parser_type = parser_type or ["default"] * len(completions)
    rewards = []
    for completion, reference, current_parser in zip(completions, reference_answer, parser_type):
        grade = answer_match_grade(completion, reference, current_parser)
        if grade == "exact":
            rewards.append(1.0)
        elif grade == "numeric_close":
            rewards.append(REWARD_CONFIG["numeric_close_reward"])
        elif grade == "parsed_wrong":
            rewards.append(REWARD_CONFIG["parsed_wrong_reward"])
        else:
            rewards.append(0.0)
    return rewards


def format_reward_fn(completions, reasoning_style=None, task_type=None, **kwargs):
    reasoning_style = reasoning_style or ["full_cot"] * len(completions)
    task_type = task_type or ["general_math"] * len(completions)
    rewards = []
    for completion, current_style, current_task in zip(completions, reasoning_style, task_type):
        reward = 0.0
        candidate = extract_candidate_answer(completion)
        if candidate:
            reward += REWARD_CONFIG["parse_bonus"]
        else:
            reward += REWARD_CONFIG["invalid_answer_penalty"]

        if has_valid_final_structure(completion):
            reward += REWARD_CONFIG["final_structure_bonus"]

        completion_tokens = _completion_token_count(completion)
        if current_style == "concise_cot" or current_task == "arithmetic_word_problem":
            start_tokens = REWARD_CONFIG["concise_length_penalty_start_tokens"]
        else:
            start_tokens = REWARD_CONFIG["full_length_penalty_start_tokens"]
        if completion_tokens > start_tokens:
            reward -= REWARD_CONFIG["length_penalty_weight"] * ((completion_tokens - start_tokens) / start_tokens)

        rewards.append(reward)
    return rewards


def combined_reward_fn(completions, **kwargs):
    correctness = correctness_reward_fn(completions, kwargs.get("reference_answer", []), kwargs.get("parser_type"))
    formatting = format_reward_fn(completions)
    return [left + right for left, right in zip(correctness, formatting)]


def load_rl_dataset():
    if not RL_TRAIN_PATH.exists():
        raise FileNotFoundError("Cleaned RL data not found. Run src_qwen35/data_prepare.py first.")
    with RL_TRAIN_PATH.open("r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    logger.info("Loaded RL prompts: %s", len(data))
    return data


def prepare_dataset_for_grpo(raw_data):
    processed = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        prompt = REWARD_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed.append(
            {
                "prompt": prompt,
                "reference_answer": item["reference_answer"],
                "parser_type": item.get("parser_type", "math"),
                "source_dataset": item.get("source_dataset", "unknown"),
                "reasoning_style": item.get("reasoning_style", "full_cot"),
                "task_type": item.get("task_type", "general_math"),
            }
        )
    return Dataset.from_list(processed)


def run_grpo_training(mode=None):
    global REWARD_CONFIG, REWARD_TOKENIZER

    _add_file_handler(GRPO_LOG_DIR)
    config = get_grpo_config(mode)
    REWARD_CONFIG = config
    print_config_summary(config, "GRPO Training Config")

    if LOGGER_TYPE == "swanlab":
        try:
            import swanlab

            swanlab.init(
                project=SWANLAB_PROJECT,
                workspace=SWANLAB_WORKSPACE,
                experiment_name=f"GRPO-{config['experiment_tag']}",
                description=f"Qwen3.5-4B-Base GRPO ({mode or 'auto'})",
                config=config,
            )
        except Exception as exc:  # pragma: no cover - optional service
            logger.warning("SwanLab init failed, continuing with default logger: %s", exc)

    parent_sft_dir = _resolve_parent_sft_dir()
    if not (parent_sft_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"SFT adapter not found for GRPO parent stage: {parent_sft_dir}")
    logger.info("Using parent SFT adapter for GRPO: %s", parent_sft_dir)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    REWARD_TOKENIZER = tokenizer

    if config.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=_current_device_map(),
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model = PeftModel.from_pretrained(base_model, str(parent_sft_dir), is_trainable=True)
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()

    train_dataset = prepare_dataset_for_grpo(load_rl_dataset())
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
        temperature=config["temperature"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        max_steps=config["max_steps"],
        seed=config["seed"],
        bf16=config["bf16"],
        report_to=DEFAULT_REPORT_TO,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=combined_reward_fn,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    logger.info("GRPO complete: %s", train_result.metrics)
    return train_result.metrics


def main():
    parser = argparse.ArgumentParser(description="Stage C: Qwen3.5-4B-Base GRPO training")
    parser.add_argument("--mode", type=str, default=None, choices=["single", "multi"])
    args = parser.parse_args()
    run_grpo_training(mode=args.mode)


if __name__ == "__main__":
    main()
