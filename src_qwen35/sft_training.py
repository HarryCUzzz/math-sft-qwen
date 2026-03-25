
"""Stage B - Qwen3.5-4B-Base SFT on cleaned qwen35_v2 datasets."""

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import (
    LOGGER_TYPE,
    OUTPUT_BASE,
    QUANTIZATION_CONFIG,
    SFT_EVAL_PATH,
    SFT_TRAIN_PATH,
    SWANLAB_PROJECT,
    SWANLAB_WORKSPACE,
    get_sft_config,
    print_config_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = OUTPUT_BASE / "sft_model"
LOG_DIR = OUTPUT_BASE / "sft_logs"


def _add_file_handler(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sft_training.log"
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
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def _load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config["trust_remote_code"],
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.get("use_4bit", False):
        logger.info("Loading model with QLoRA 4-bit quantization")
        quantization_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            trust_remote_code=config["trust_remote_code"],
            quantization_config=quantization_config,
            device_map=_current_device_map(),
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model in bf16 for A100 80GB")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            trust_remote_code=config["trust_remote_code"],
            torch_dtype=torch.bfloat16,
        )

    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _preprocess_dataset(dataset, tokenizer, max_length, num_proc, desc):
    def preprocess_function(examples):
        input_ids_list = []
        attention_masks = []
        labels_list = []
        label_token_counts = []

        for messages in examples["messages"]:
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            assistant_text = messages[-1]["content"] + tokenizer.eos_token
            full_text = prompt_text + assistant_text

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            tokenized = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            labels = tokenized["input_ids"].copy()
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])
            labels_list.append(labels)
            label_token_counts.append(sum(token != -100 for token in labels))

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_masks,
            "labels": labels_list,
            "label_token_count": label_token_counts,
        }

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=desc,
    )
    dataset = dataset.filter(lambda row: row["label_token_count"] > 0)
    return dataset.remove_columns(["label_token_count"])


def run_sft_training(mode=None):
    _add_file_handler(LOG_DIR)
    config = get_sft_config(mode)
    print_config_summary(config, "SFT Training Config")

    if LOGGER_TYPE == "swanlab":
        try:
            import swanlab

            swanlab.init(
                project=SWANLAB_PROJECT,
                workspace=SWANLAB_WORKSPACE,
                experiment_name=f"SFT-{mode or 'auto'}",
                description=f"Qwen3.5-4B-Base SFT ({mode or 'auto'})",
                config=config,
            )
        except Exception as exc:  # pragma: no cover - optional service
            logger.warning("SwanLab init failed, falling back to tensorboard: %s", exc)
            config["report_to"] = ["tensorboard"]

    if not SFT_TRAIN_PATH.exists() or not SFT_EVAL_PATH.exists():
        raise FileNotFoundError(
            "Cleaned qwen35_v2 data not found. Run src_qwen35/data_prepare.py first."
        )

    model, tokenizer = _load_model_and_tokenizer(config)
    train_dataset = hf_load_dataset("json", data_files=str(SFT_TRAIN_PATH), split="train")
    eval_dataset = hf_load_dataset("json", data_files=str(SFT_EVAL_PATH), split="train")

    logger.info("Loaded SFT datasets: train=%s eval=%s", len(train_dataset), len(eval_dataset))
    train_dataset = _preprocess_dataset(
        train_dataset,
        tokenizer,
        config["cutoff_len"],
        config["preprocessing_num_workers"],
        "Tokenizing train split",
    )
    eval_dataset = _preprocess_dataset(
        eval_dataset,
        tokenizer,
        config["cutoff_len"],
        config["preprocessing_num_workers"],
        "Tokenizing eval split",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOG_DIR),
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        bf16=config["bf16"],
        tf32=config.get("tf32", False),
        logging_steps=config["logging_steps"],
        logging_first_step=True,
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        report_to=config["report_to"],
        run_name="sft_qwen35_4b_base",
        seed=config["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        optim=config["optim"],
        save_safetensors=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoints:
        resume_from_checkpoint = str(max(checkpoints, key=lambda item: int(item.name.split("-")[1])))
        logger.info("Resuming from checkpoint %s", resume_from_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()

    train_metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("SFT complete. train_loss=%s eval_loss=%s", train_metrics.get("train_loss"), eval_metrics.get("eval_loss"))
    return train_metrics, eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage B: Qwen3.5-4B-Base SFT training")
    parser.add_argument("--mode", type=str, default=None, choices=["single", "multi"])
    args = parser.parse_args()
    run_sft_training(mode=args.mode)


if __name__ == "__main__":
    main()
