"""
Stage B - 有监督微调 (Supervised Fine-Tuning) - Qwen3.5-4B 版本
================================================================
使用 Transformers + PEFT 对 Qwen3.5-4B 进行 LoRA SFT 微调。

适配 A6000 (48GB) 显存优化：
- 启用 gradient_checkpointing
- 使用较小的 batch_size 配合梯度累积
- LoRA rank 适度降低

输出：
- outputs_qwen35/sft_model/     (SFT 模型检查点)
- outputs_qwen35/sft_logs/      (训练日志)
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import (
    DATA_FILTERED,
    OUTPUT_BASE,
    QUANTIZATION_CONFIG,
    THINKING_SYSTEM_PROMPT,
    get_sft_config,
    print_config_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = OUTPUT_BASE / "sft_model"
LOG_DIR = OUTPUT_BASE / "sft_logs"


def _add_file_handler(log_dir: Path) -> None:
    """将日志同时写入文件"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sft_training.log"
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


def run_sft_training(mode=None):
    """
    SFT 训练主流程

    Args:
        mode: "single" 或 "multi"，如果为 None 则自动检测 GPU 数量
    """
    _add_file_handler(LOG_DIR)

    # 获取配置
    config = get_sft_config(mode)
    print_config_summary(config, "SFT Training Config")

    logger.info("=" * 60)
    logger.info("Stage B - Qwen3.5-4B SFT 训练开始")
    logger.info("=" * 60)

    # ---- 加载分词器和模型 ----
    model_name = config["model_name_or_path"]
    logger.info(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 - 支持 4-bit 量化或 bf16 精度
    if config.get("use_4bit", False):
        logger.info("启用 4-bit 量化加载模型...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        logger.info("使用 bf16 精度加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # 启用梯度检查点
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("启用 gradient_checkpointing")

    # ---- 配置 LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 加载筛选后的数据 ----
    sft_train_path = str(DATA_FILTERED / "sft_train.json")
    sft_eval_path = str(DATA_FILTERED / "sft_eval.json")

    logger.info(f"加载训练数据: {sft_train_path}")
    train_dataset = hf_load_dataset("json", data_files=sft_train_path, split="train")

    logger.info(f"加载验证数据: {sft_eval_path}")
    eval_dataset = hf_load_dataset("json", data_files=sft_eval_path, split="train")

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(eval_dataset)}")

    # ---- 数据预处理 ----
    def preprocess_function(examples):
        """
        将 alpaca 格式转换为模型可用的 token 序列
        格式: <|im_start|>system\n...<|im_end|>\n
              <|im_start|>user\n{instruction}\n{input}<|im_end|>\n
              <|im_start|>assistant\n{output}<|im_end|>
        """
        inputs = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            messages = [
                {
                    "role": "system",
                    "content": THINKING_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{input_text}"
                    if input_text
                    else instruction,
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + output + tokenizer.eos_token
            inputs.append(full_text)

        model_inputs = tokenizer(
            inputs,
            max_length=config["cutoff_len"],
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    logger.info("正在预处理训练数据...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=config.get("preprocessing_num_workers", 4),
        desc="预处理训练集",
    )

    logger.info("正在预处理验证数据...")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=config.get("preprocessing_num_workers", 4),
        desc="预处理验证集",
    )

    # ---- 训练参数 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOG_DIR),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        report_to=config["report_to"],
        run_name="sft_qwen35_4b",
        seed=config["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        optim=config.get("optim", "adamw_torch"),
    )

    # ---- 数据整理器 ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=config["cutoff_len"],
    )

    # ---- 创建 Trainer 并训练 ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("开始训练...")
    logger.info(f"  模型: {model_name}")
    logger.info(f"  LoRA rank: {config['lora_rank']}, alpha: {config['lora_alpha']}")
    logger.info(
        f"  Batch size: {config['per_device_train_batch_size']} x {config['gradient_accumulation_steps']} = "
        f"{config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}"
    )
    logger.info(f"  学习率: {config['learning_rate']}")
    logger.info(f"  训练轮数: {config['num_train_epochs']}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")

    # 检查是否有 checkpoint 可以恢复
    checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        resume_from_checkpoint = str(latest_checkpoint)
        logger.info(f"从 checkpoint 恢复训练: {resume_from_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ---- 保存模型和训练指标 ----
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 评估
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"训练完成！模型已保存至: {OUTPUT_DIR}")
    logger.info(f"训练 loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"验证 loss: {eval_metrics.get('eval_loss', 'N/A')}")

    return metrics, eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage B: Qwen3.5-4B SFT 训练")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["single", "multi"],
        help="训练模式: single (单卡) 或 multi (多卡)，默认自动检测",
    )
    args = parser.parse_args()

    # 检查数据是否存在
    sft_train_path = DATA_FILTERED / "sft_train.json"
    if not sft_train_path.exists():
        logger.error(f"SFT 训练数据未找到: {sft_train_path}")
        logger.error("请确保数据文件存在于 data/filtered/ 目录中")
        return

    run_sft_training(mode=args.mode)
    logger.info("Stage B 完成！")


if __name__ == "__main__":
    main()
