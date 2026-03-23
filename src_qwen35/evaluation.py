"""
Stage D - 评估 (Evaluation) - Qwen3.5-4B 版本
==============================================
在标准 benchmark 上对比三个模型的表现:
1. Base 模型 (Qwen3.5-4B)
2. SFT 模型
3. GRPO 模型

评估数据集:
- MATH-500 (HuggingFaceH4/MATH-500) - 竞赛级数学题
- GSM8K (openai/gsm8k) - 小学数学应用题
- TheoremQA (TIGER-Lab/TheoremQA) - 定理推理

输出:
- outputs_qwen35/eval_results/ 目录下的 JSON 结果文件和对比表格
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DATA_EVAL,
    EVAL_CONFIG,
    EVAL_DATASETS,
    OUTPUT_BASE,
)

# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 目录配置
SFT_MODEL_DIR = OUTPUT_BASE / "sft_model"
GRPO_MODEL_DIR = OUTPUT_BASE / "grpo_model"
EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"


# ============================================================
# 答案提取和匹配工具
# ============================================================
def extract_boxed_answer(text):
    """从模型输出中提取 \\boxed{} 内的答案"""
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


def extract_gsm8k_answer(text):
    """从 GSM8K 答案中提取数字"""
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "")
        return ans
    return text.strip()


def normalize_for_comparison(answer_str):
    """规范化答案用于比较"""
    if not answer_str:
        return ""
    ans = str(answer_str).strip()
    ans = ans.replace("$", "").replace(",", "").replace(" ", "")
    ans = ans.rstrip(".")
    try:
        num = float(ans)
        if num == int(num):
            ans = str(int(num))
        else:
            ans = str(num)
    except (ValueError, OverflowError):
        pass
    return ans


def check_answer(predicted, reference, dataset_name="default"):
    """检查预测答案是否正确 - 支持 Thinking Mode"""
    # 移除 <think>...</think> 内容，只检查最终答案
    clean_predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL)

    pred_answer = extract_boxed_answer(clean_predicted)

    # 如果 boxed 提取失败，尝试多种模式
    if pred_answer is None:
        answer_patterns = [
            r"(?:Answer|answer|答案)[:\s]+([^\n]+)",
            r"(?:The answer is|the answer is)[:\s]+([^\n]+)",
            r"(?:Therefore|therefore)[,\s]+(?:the answer is)?[:\s]*([^\n]+)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, clean_predicted)
            if match:
                pred_answer = match.group(1).strip()
                break

    # 提取最后一行的数字
    if pred_answer is None:
        lines = clean_predicted.strip().split("\n")
        for line in reversed(lines):
            nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", line)
            if nums:
                pred_answer = nums[-1]
                break

    if dataset_name == "gsm8k":
        ref_answer = extract_gsm8k_answer(str(reference))
    else:
        ref_answer = str(reference)

    pred_norm = normalize_for_comparison(pred_answer)
    ref_norm = normalize_for_comparison(ref_answer)

    return pred_norm == ref_norm and pred_norm != ""


# ============================================================
# 模型加载
# ============================================================
def load_model_and_tokenizer(model_type="base"):
    """加载模型和分词器"""
    base_name = EVAL_CONFIG["base_model"]
    base_path = Path(base_name).resolve()

    # 检查模型路径
    if not base_path.exists():
        logger.error(f"模型路径不存在: {base_path}")
        raise FileNotFoundError(f"模型路径不存在: {base_path}")

    logger.info(f"加载 {model_type} 模型...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型加载参数
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # 启用 Flash Attention 2 (如果配置启用)
    if EVAL_CONFIG.get("use_flash_attention", True):
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("启用 Flash Attention 2")
        except Exception:
            logger.info("Flash Attention 2 不可用，使用默认实现")

    if model_type == "base":
        model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
    elif model_type == "sft":
        base_model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
        if (SFT_MODEL_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(SFT_MODEL_DIR))
            model = model.merge_and_unload()
        else:
            logger.warning("SFT adapter 未找到，将使用基座模型代替")
            model = base_model
    elif model_type == "grpo":
        if GRPO_MODEL_DIR.exists() and (GRPO_MODEL_DIR / "config.json").exists():
            model = AutoModelForCausalLM.from_pretrained(
                str(GRPO_MODEL_DIR), **load_kwargs
            )
        else:
            logger.warning("GRPO 模型未找到，将使用基座模型代替")
            model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.eval()
    logger.info(f"{model_type} 模型加载完成")
    return model, tokenizer


# ============================================================
# 数据集加载
# ============================================================
def load_eval_dataset(dataset_key):
    """加载评估数据集"""
    config = EVAL_DATASETS[dataset_key]
    dataset = None

    # 优先从本地加载
    local_path = DATA_EVAL / dataset_key
    if local_path.exists():
        try:
            from datasets import load_from_disk
            logger.info(f"从本地加载数据集: {local_path}")
            dataset = load_from_disk(str(local_path))
            logger.info("  本地加载成功")
        except Exception as e:
            logger.warning(f"  本地加载失败: {e}")

    # 回退到 HuggingFace
    if dataset is None:
        try:
            from datasets import load_dataset
            logger.info(f"从 HuggingFace 加载数据集: {config['name_hf']}")
            if "subset" in config:
                dataset = load_dataset(
                    config["name_hf"], config["subset"], split=config["split"]
                )
            else:
                dataset = load_dataset(config["name_hf"], split=config["split"])
            logger.info("  HuggingFace 加载成功")
        except Exception as e:
            logger.error(f"  HuggingFace 加载失败: {e}")
            return []

    if dataset is None:
        logger.error(f"数据集 {dataset_key} 加载失败")
        return []

    eval_data = []
    for item in dataset:
        question = item.get(config["question_key"], "")
        answer = item.get(config["answer_key"], "")
        if question and answer:
            eval_data.append(
                {
                    "question": str(question),
                    "reference_answer": str(answer),
                }
            )

    logger.info(f"  加载了 {len(eval_data)} 条评估数据")
    return eval_data


# ============================================================
# 模型推理与评估
# ============================================================
@torch.no_grad()
def evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key):
    """在数据集上评估模型"""
    correct = 0
    total = len(eval_data)
    results = []
    batch_size = EVAL_CONFIG.get("batch_size", 8)

    logger.info(f"开始评估 ({dataset_key}), 共 {total} 条...")
    logger.info(f"配置: max_new_tokens={EVAL_CONFIG['max_new_tokens']}, batch_size={batch_size}")

    for idx in range(0, total, batch_size):
        batch_data = eval_data[idx : idx + batch_size]
        if not batch_data:
            continue

        prompts = []
        for item in batch_data:
            system_prompt = "You are a helpful math assistant. Solve problems step by step and put your final answer in \\boxed{}."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=EVAL_CONFIG["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            use_cache=True,
            do_sample=False,
        )

        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for item, prediction in zip(batch_data, generated_texts):
            is_correct = check_answer(prediction, item["reference_answer"], dataset_key)
            if is_correct:
                correct += 1

            results.append(
                {
                    "question": item["question"][:200],
                    "reference_answer": item["reference_answer"],
                    "prediction": prediction[:500],
                    "is_correct": is_correct,
                }
            )

        current_idx = idx + len(batch_data)
        if current_idx % 50 == 0:
            acc = correct / current_idx * 100
            logger.info(f"  进度: {current_idx}/{total}, 当前准确率: {acc:.1f}%")

    accuracy = correct / max(total, 1) * 100

    return {
        "dataset": dataset_key,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "details": results if EVAL_CONFIG.get("save_details", True) else [],
    }


def run_full_evaluation(model_types=None, datasets=None, skip_datasets=None):
    """运行完整评估流程"""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_types = model_types or ["base", "sft", "grpo"]
    all_results = {}
    skip_datasets = skip_datasets or []
    datasets = datasets or list(EVAL_DATASETS.keys())

    for model_type in model_types:
        logger.info("=" * 60)
        logger.info(f"评估模型: {model_type}")
        logger.info("=" * 60)

        try:
            model, tokenizer = load_model_and_tokenizer(model_type)
        except Exception as e:
            logger.error(f"加载 {model_type} 模型失败: {e}")
            continue

        model_results = {}
        for dataset_key in datasets:
            if dataset_key in skip_datasets:
                logger.info(f"\n--- 数据集: {dataset_key} (跳过) ---")
                continue

            logger.info(f"\n--- 数据集: {dataset_key} ---")
            eval_data = load_eval_dataset(dataset_key)
            if not eval_data:
                logger.warning(f"数据集 {dataset_key} 为空，跳过")
                continue

            result = evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key)
            model_results[dataset_key] = result
            logger.info(f"  {dataset_key} 准确率: {result['accuracy']}%")

            result_path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                payload = result
                if not EVAL_CONFIG.get("save_details", True):
                    payload = {k: v for k, v in result.items() if k != "details"}
                json.dump(payload, f, ensure_ascii=False, indent=2)

        all_results[model_type] = model_results
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    generate_comparison_table(all_results)
    return all_results


def generate_comparison_table(all_results):
    """生成对比表格"""
    comparison = {}
    for model_type, model_results in all_results.items():
        comparison[model_type] = {}
        for dataset_key, result in model_results.items():
            comparison[model_type][dataset_key] = result["accuracy"]

    comparison_path = EVAL_RESULTS_DIR / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    table_path = EVAL_RESULTS_DIR / "comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Qwen3.5-4B 模型评估结果对比表 (Accuracy %)\n")
        f.write("=" * 70 + "\n\n")

        datasets = list(EVAL_DATASETS.keys())
        header = f"{'Model':<15}" + "".join(f"{d:<15}" for d in datasets)
        f.write(header + "\n")
        f.write("-" * 60 + "\n")

        for model_type in ["base", "sft", "grpo"]:
            if model_type in comparison:
                row = f"{model_type:<15}"
                for d in datasets:
                    acc = comparison[model_type].get(d, "N/A")
                    row += f"{acc:<15}"
                f.write(row + "\n")

        f.write("-" * 60 + "\n")

    logger.info(f"对比表格已保存至: {comparison_path}")
    logger.info(f"可读表格已保存至: {table_path}")

    with open(table_path, "r") as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Stage D: Qwen3.5-4B 模型评估")
    parser.add_argument(
        "--models", type=str, nargs="+", default=["base", "sft", "grpo"]
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument(
        "--skip-datasets",
        type=str,
        nargs="+",
        default=[],
        help="要跳过的数据集列表",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="仅保存汇总结果，不保存逐样本 details",
    )
    args = parser.parse_args()

    if args.num_samples:
        EVAL_CONFIG["num_eval_samples"] = args.num_samples
    if args.summary_only:
        EVAL_CONFIG["save_details"] = False

    selected_models = args.models or ["base", "sft", "grpo"]
    selected_datasets = args.datasets or list(EVAL_DATASETS.keys())

    logger.info("=" * 60)
    logger.info("Stage D - Qwen3.5-4B 评估开始")
    logger.info("=" * 60)
    logger.info(f"模型: {selected_models}")
    logger.info(f"数据集: {selected_datasets}")

    run_full_evaluation(
        model_types=selected_models,
        datasets=selected_datasets,
        skip_datasets=args.skip_datasets,
    )
    logger.info("Stage D 完成！")


if __name__ == "__main__":
    main()
