"""
Stage B - 有监督微调 (Supervised Fine-Tuning)
==============================================
使用 LLaMA-Factory 对 Qwen2.5-0.5B 进行 SFT 微调。

主要步骤：
1. 注册数据集到 LLaMA-Factory
2. 生成 SFT 训练配置
3. 启动训练
4. 对比基座模型与 SFT 模型的效果

输出：
- outputs/sft_model/     (SFT 模型检查点)
- outputs/sft_logs/      (训练日志)
"""

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _add_file_handler(log_dir: Path) -> None:
    """将日志同时写入文件，避免重复添加 handler。"""
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


# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_FACTORY_DIR = PROJECT_ROOT / "LLaMA-Factory"
DATA_FILTERED = PROJECT_ROOT / "data" / "filtered"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft_model"
LOG_DIR = PROJECT_ROOT / "outputs" / "sft_logs"

# ============================================================
# 模型和训练超参数配置
# ============================================================
CONFIG = {
    # 模型配置
    "model_name_or_path": "Qwen/Qwen2.5-0.5B",  # 基座模型
    "trust_remote_code": True,
    # 训练方法
    "stage": "sft",  # 有监督微调阶段
    "do_train": True,
    "finetuning_type": "lora",  # 使用 LoRA 微调以节省显存
    # LoRA 配置
    "lora_rank": 64,  # LoRA 秩
    "lora_alpha": 128,  # LoRA alpha
    "lora_target": "all",  # 对所有线性层应用 LoRA
    "lora_dropout": 0.05,
    # 数据配置
    "dataset": "math_sft_train",  # 数据集名称（需注册到 dataset_info.json）
    "template": "qwen",  # Qwen 模型的 chat template
    "cutoff_len": 2048,  # 最大序列长度
    "preprocessing_num_workers": 4,
    # 训练超参数
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # 有效 batch size = 4 * 8 = 32
    "learning_rate": 2e-5,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",  # 余弦退火学习率
    "warmup_ratio": 0.1,  # 前 10% 步做 warmup
    "weight_decay": 0.01,
    # 精度与优化
    "bf16": True,  # 使用 bfloat16 混合精度训练
    "optim": "adamw_torch",  # AdamW 优化器
    # 日志与保存
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,  # 最多保留 3 个检查点
    "report_to": "tensorboard",  # 使用 TensorBoard 记录训练日志
    # 评估
    "val_size": 0.05,  # 从训练集中取 5% 做验证
    "per_device_eval_batch_size": 4,
    "eval_strategy": "steps",
    "eval_steps": 200,
    # 其他
    "seed": 42,
}


# ============================================================
# 1. 注册数据集到 LLaMA-Factory
# ============================================================
def register_dataset():
    """
    将筛选后的数据集注册到 LLaMA-Factory 的 dataset_info.json 中。
    LLaMA-Factory 通过此文件识别可用的数据集。
    """
    dataset_info_path = LLAMA_FACTORY_DIR / "data" / "dataset_info.json"

    # 读取现有配置（如果存在）
    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    # 注册 SFT 训练集
    sft_train_path = str(DATA_FILTERED / "sft_train.json")
    dataset_info["math_sft_train"] = {
        "file_name": sft_train_path,
        "formatting": "alpaca",  # alpaca 格式: instruction + input + output
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }

    # 注册 SFT 验证集
    sft_eval_path = str(DATA_FILTERED / "sft_eval.json")
    dataset_info["math_sft_eval"] = {
        "file_name": sft_eval_path,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }

    # 确保 LLaMA-Factory/data 目录存在
    (LLAMA_FACTORY_DIR / "data").mkdir(parents=True, exist_ok=True)

    # 写入配置
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    logger.info(f"数据集已注册到: {dataset_info_path}")
    logger.info(f"  - math_sft_train -> {sft_train_path}")
    logger.info(f"  - math_sft_eval  -> {sft_eval_path}")


# ============================================================
# 2. 生成训练配置文件
# ============================================================
def generate_training_config():
    """
    生成 LLaMA-Factory 的 YAML 训练配置文件。
    也可以直接使用命令行参数，但配置文件更易于管理和复现。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 构建完整配置
    full_config = {
        **CONFIG,
        "output_dir": str(OUTPUT_DIR),
        "logging_dir": str(LOG_DIR),
        "dataset_dir": str(LLAMA_FACTORY_DIR / "data"),
    }

    # 保存为 YAML 格式
    config_path = PROJECT_ROOT / "configs" / "sft_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # 手动写 YAML（避免额外依赖）
    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in full_config.items():
            if isinstance(value, bool):
                f.write(f"{key}: {'true' if value else 'false'}\n")
            elif isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info(f"训练配置已生成: {config_path}")
    return config_path


# ============================================================
# 3. 启动 SFT 训练
# ============================================================
def run_sft_training(config_path):
    """
    使用 LLaMA-Factory 的 CLI 启动 SFT 训练。
    等价于: llamafactory-cli train configs/sft_config.yaml
    """
    logger.info("=" * 60)
    logger.info("开始 SFT 训练")
    logger.info("=" * 60)
    logger.info(f"基座模型: {CONFIG['model_name_or_path']}")
    logger.info(
        f"微调方法: LoRA (rank={CONFIG['lora_rank']}, alpha={CONFIG['lora_alpha']})"
    )
    logger.info(
        f"Batch size: {CONFIG['per_device_train_batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}"
    )
    logger.info(f"学习率: {CONFIG['learning_rate']}")
    logger.info(f"训练轮数: {CONFIG['num_train_epochs']}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    # 使用 llamafactory-cli 启动训练
    cmd = [
        "llamafactory-cli",
        "train",
        str(config_path),
    ]

    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        process = subprocess.run(
            cmd,
            cwd=str(LLAMA_FACTORY_DIR),
            check=True,
            text=True,
        )
        logger.info("SFT 训练完成！")
    except FileNotFoundError:
        logger.warning(
            "llamafactory-cli 未找到，尝试使用 python -m llamafactory.cli 方式..."
        )
        # 备选方案：直接调用 Python 模块
        cmd_alt = [
            "python",
            "-m",
            "llamafactory.cli",
            "train",
            str(config_path),
        ]
        subprocess.run(cmd_alt, cwd=str(LLAMA_FACTORY_DIR), check=True, text=True)
        logger.info("SFT 训练完成！")
    except subprocess.CalledProcessError as e:
        logger.error(f"SFT 训练失败: {e}")
        raise


# ============================================================
# 4. 使用 Transformers 直接训练（备选方案）
# ============================================================
def run_sft_with_transformers():
    """
    备选方案：如果 LLaMA-Factory 安装有问题，可直接使用 Transformers + PEFT 进行 SFT。
    这个方案更灵活，但需要手动处理数据加载和模板。
    """
    import torch
    from datasets import load_dataset as hf_load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    _add_file_handler(LOG_DIR)
    logger.info("使用 Transformers + PEFT 直接进行 SFT 训练...")

    # ---- 加载分词器和模型 ----
    model_name = CONFIG["model_name_or_path"]
    logger.info(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ---- 配置 LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG.get("lora_dropout", 0.05),
        target_modules="all-linear",  # 对所有线性层应用 LoRA
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数量

    # ---- 加载筛选后的数据 ----
    sft_train_path = str(DATA_FILTERED / "sft_train.json")
    sft_eval_path = str(DATA_FILTERED / "sft_eval.json")

    train_dataset = hf_load_dataset("json", data_files=sft_train_path, split="train")
    eval_dataset = hf_load_dataset("json", data_files=sft_eval_path, split="train")

    # ---- 数据预处理：将 instruction + input + output 转为模型输入 ----
    def preprocess_function(examples):
        """
        将 alpaca 格式转换为模型可用的 token 序列。
        格式: <|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n
              <|im_start|>user\n{instruction}\n{input}<|im_end|>\n
              <|im_start|>assistant\n{output}<|im_end|>
        """
        inputs = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # 构建 Qwen chat 格式
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful math assistant. Solve problems step by step.",
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{input_text}"
                    if input_text
                    else instruction,
                },
            ]
            # 使用 tokenizer 的 apply_chat_template 方法
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + output + tokenizer.eos_token
            inputs.append(full_text)

        # 分词
        model_inputs = tokenizer(
            inputs,
            max_length=CONFIG["cutoff_len"],
            truncation=True,
            padding=False,
        )

        # 设置 labels（与 input_ids 相同，因为我们要训练生成）
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    logger.info("正在预处理训练数据...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="预处理训练集",
    )

    logger.info("正在预处理验证数据...")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        desc="预处理验证集",
    )

    # ---- 训练参数 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOG_DIR),
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        bf16=CONFIG["bf16"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        report_to=["tensorboard", "wandb"],
        run_name="sft_qwen2.5_0.5b",
        seed=CONFIG["seed"],
        remove_unused_columns=False,
    )

    # ---- 数据整理器 ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=CONFIG["cutoff_len"],
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
    train_result = trainer.train(resume_from_checkpoint=True)

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


# ============================================================
# 主函数
# ============================================================
def main():
    """
    SFT 训练主流程：
    1. 检查数据是否已筛选完成
    2. 注册数据集（LLaMA-Factory 方案用）
    3. 启动训练
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage B: SFT 训练")
    parser.add_argument(
        "--method",
        type=str,
        default="llamafactory",
        choices=["llamafactory", "transformers"],
        help="训练方法: llamafactory (推荐) 或 transformers (备选)",
    )
    args = parser.parse_args()

    # 检查筛选后的数据是否存在
    sft_train_path = DATA_FILTERED / "sft_train.json"
    if not sft_train_path.exists():
        logger.error(f"SFT 训练数据未找到: {sft_train_path}")
        logger.error("请先运行 Stage A (data_selection.py) 完成数据筛选。")
        return

    logger.info("=" * 60)
    logger.info("Stage B - SFT 训练开始")
    logger.info("=" * 60)

    if args.method == "llamafactory":
        # 方案一：使用 LLaMA-Factory
        register_dataset()
        config_path = generate_training_config()
        run_sft_training(config_path)
    else:
        # 方案二：使用 Transformers + PEFT 直接训练
        run_sft_with_transformers()

    logger.info("Stage B 完成！")


if __name__ == "__main__":
    main()
