"""
Stage D - 评估 (Evaluation)
============================
在标准 benchmark 上对比三个模型的表现:
1. Base 模型 (Qwen2.5-0.5B)
2. SFT 模型
3. GRPO 模型

评估数据集:
- MATH-500 (HuggingFaceH4/MATH-500) - 竞赛级数学题
- GSM8K (openai/gsm8k) - 小学数学应用题
- TheoremQA (TIGER-Lab/TheoremQA) - 定理推理

输出:
- outputs/eval_results/ 目录下的 JSON 结果文件和对比表格
"""

import os
import re
import json
import logging
import torch
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_MODEL_DIR = PROJECT_ROOT / "outputs" / "sft_model"
GRPO_MODEL_DIR = PROJECT_ROOT / "outputs" / "grpo_model"
EVAL_RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_results"

# ============================================================
# 评估配置
# ============================================================
EVAL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "max_new_tokens": 1024,          # 生成的最大 token 数
    "temperature": 0.0,               # 评估时使用贪心解码（temperature=0）
    "do_sample": False,
    "num_eval_samples": None,         # None 表示全量评估，可设为整数进行快速测试
}

# 评估数据集列表
EVAL_DATASETS = {
    "math500": {
        "name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_key": "problem",
        "answer_key": "answer",
    },
    "gsm8k": {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
    },
    "theoremqa": {
        "name": "TIGER-Lab/TheoremQA",
        "split": "test",
        "question_key": "Question",
        "answer_key": "Answer",
    },
}


# ============================================================
# 1. 答案提取和匹配工具
# ============================================================
def extract_boxed_answer(text):
    """
    从模型输出中提取 \\boxed{} 内的答案。
    """
    if not text:
        return None
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

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


def extract_gsm8k_answer(text):
    """
    GSM8K 的答案格式特殊: 答案在 '####' 之后。
    例如: "... #### 42"
    """
    if "####" in text:
        ans = text.split("####")[-1].strip()
        # 去除逗号（如 1,000 → 1000）
        ans = ans.replace(",", "")
        return ans
    return text.strip()


def normalize_for_comparison(answer_str):
    """
    规范化答案字符串用于对比：
    - 去除空格、$、逗号
    - 统一小数格式
    """
    if not answer_str:
        return ""
    ans = str(answer_str).strip()
    ans = ans.replace("$", "").replace(",", "").replace(" ", "")
    ans = ans.rstrip(".")
    # 如果是纯数字，尝试统一格式
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
    """
    检查模型预测与参考答案是否匹配。
    针对不同数据集可能有不同的提取逻辑。
    """
    # 从模型输出中提取答案
    pred_answer = extract_boxed_answer(predicted)

    # 如果没找到 boxed，尝试提取最后一行的数字
    if pred_answer is None:
        # 尝试从最后一行提取数字
        lines = predicted.strip().split('\n')
        for line in reversed(lines):
            nums = re.findall(r'-?\d+\.?\d*', line)
            if nums:
                pred_answer = nums[-1]
                break

    # 处理参考答案
    if dataset_name == "gsm8k":
        ref_answer = extract_gsm8k_answer(str(reference))
    else:
        ref_answer = str(reference)

    # 规范化后对比
    pred_norm = normalize_for_comparison(pred_answer)
    ref_norm = normalize_for_comparison(ref_answer)

    return pred_norm == ref_norm and pred_norm != ""


# ============================================================
# 2. 模型加载
# ============================================================
def load_model_and_tokenizer(model_type="base"):
    """
    加载指定类型的模型和分词器。

    参数:
        model_type: "base" | "sft" | "grpo"
    返回:
        (model, tokenizer) 元组
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_name = EVAL_CONFIG["base_model"]

    logger.info(f"加载 {model_type} 模型...")

    # 加载分词器（所有模型共用同一个分词器）
    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "base":
        # 加载基座模型
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif model_type == "sft":
        # 加载基座模型 + SFT LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if (SFT_MODEL_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(SFT_MODEL_DIR))
            model = model.merge_and_unload()
        else:
            logger.warning(f"SFT adapter 未找到，将使用基座模型代替")
            model = base_model
    elif model_type == "grpo":
        # 加载 GRPO 模型
        if GRPO_MODEL_DIR.exists():
            model = AutoModelForCausalLM.from_pretrained(
                str(GRPO_MODEL_DIR),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            logger.warning(f"GRPO 模型未找到，将使用基座模型代替")
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.eval()
    logger.info(f"{model_type} 模型加载完成")
    return model, tokenizer


# ============================================================
# 3. 数据集加载
# ============================================================
def load_eval_dataset(dataset_key):
    """
    加载评估数据集。

    参数:
        dataset_key: 数据集标识 ("math500" | "gsm8k" | "theoremqa")
    返回:
        包含 (question, reference_answer) 的列表
    """
    from datasets import load_dataset

    config = EVAL_DATASETS[dataset_key]
    logger.info(f"加载评估数据集: {config['name']}")

    try:
        if "subset" in config:
            dataset = load_dataset(config["name"], config["subset"], split=config["split"])
        else:
            dataset = load_dataset(config["name"], split=config["split"])
    except Exception as e:
        logger.error(f"数据集 {config['name']} 加载失败: {e}")
        return []

    # 提取问题和答案
    eval_data = []
    for item in dataset:
        question = item.get(config["question_key"], "")
        answer = item.get(config["answer_key"], "")
        if question and answer:
            eval_data.append({
                "question": str(question),
                "reference_answer": str(answer),
            })

    # 限制评估样本数（调试用）
    if EVAL_CONFIG["num_eval_samples"]:
        eval_data = eval_data[:EVAL_CONFIG["num_eval_samples"]]

    logger.info(f"  加载了 {len(eval_data)} 条评估数据")
    return eval_data


# ============================================================
# 4. 模型推理
# ============================================================
@torch.no_grad()
def generate_answer(model, tokenizer, question, system_prompt=None):
    """
    使用模型生成数学问题的答案。

    参数:
        model: 语言模型
        tokenizer: 分词器
        question: 数学问题文本
        system_prompt: 系统提示词
    返回:
        生成的答案文本
    """
    if system_prompt is None:
        system_prompt = "You are a helpful math assistant. Solve problems step by step and put your final answer in \\boxed{}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # 应用 chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=EVAL_CONFIG["max_new_tokens"],
        temperature=EVAL_CONFIG["temperature"],
        do_sample=EVAL_CONFIG["do_sample"],
        pad_token_id=tokenizer.pad_token_id,
    )

    # 只取新生成的 token（去掉 prompt 部分）
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer


# ============================================================
# 5. 评估主流程
# ============================================================
def evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key):
    """
    在指定数据集上评估模型。

    参数:
        model: 语言模型
        tokenizer: 分词器
        eval_data: 评估数据列表
        dataset_key: 数据集标识
    返回:
        评估结果字典
    """
    correct = 0
    total = len(eval_data)
    results = []

    logger.info(f"开始评估 ({dataset_key}), 共 {total} 条...")

    for idx, item in enumerate(eval_data):
        # 生成答案
        prediction = generate_answer(model, tokenizer, item["question"])

        # 检查正确性
        is_correct = check_answer(prediction, item["reference_answer"], dataset_key)
        if is_correct:
            correct += 1

        results.append({
            "question": item["question"][:200],  # 截断以节省空间
            "reference_answer": item["reference_answer"],
            "prediction": prediction[:500],
            "is_correct": is_correct,
        })

        # 每 50 条打印一次进度
        if (idx + 1) % 50 == 0:
            acc = correct / (idx + 1) * 100
            logger.info(f"  进度: {idx+1}/{total}, 当前准确率: {acc:.1f}%")

    accuracy = correct / max(total, 1) * 100

    return {
        "dataset": dataset_key,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "details": results,
    }


def run_full_evaluation():
    """
    完整评估流程：
    对三个模型（base / sft / grpo）在三个数据集上分别评估，
    生成对比表格。
    """
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 需要评估的模型类型
    model_types = ["base", "sft", "grpo"]

    # 存储所有结果
    all_results = {}

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

        for dataset_key in EVAL_DATASETS:
            logger.info(f"\n--- 数据集: {dataset_key} ---")

            # 加载评估数据
            eval_data = load_eval_dataset(dataset_key)
            if not eval_data:
                logger.warning(f"数据集 {dataset_key} 为空，跳过")
                continue

            # 评估
            result = evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key)
            model_results[dataset_key] = result

            logger.info(f"  {dataset_key} 准确率: {result['accuracy']}%")

            # 保存单个结果
            result_path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            # 保存时不含 details（太大）
            summary = {k: v for k, v in result.items() if k != "details"}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        all_results[model_type] = model_results

        # 释放模型显存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 生成对比表格 ----
    generate_comparison_table(all_results)

    return all_results


def generate_comparison_table(all_results):
    """
    生成模型对比表格，保存为 JSON 和可读的文本格式。
    """
    # 构建对比表
    comparison = {}
    for model_type, model_results in all_results.items():
        comparison[model_type] = {}
        for dataset_key, result in model_results.items():
            comparison[model_type][dataset_key] = result["accuracy"]

    # 保存 JSON
    comparison_path = EVAL_RESULTS_DIR / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 生成可读文本表格
    table_path = EVAL_RESULTS_DIR / "comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("模型评估结果对比表 (Accuracy %)\n")
        f.write("=" * 70 + "\n\n")

        # 表头
        datasets = list(EVAL_DATASETS.keys())
        header = f"{'Model':<15}" + "".join(f"{d:<15}" for d in datasets)
        f.write(header + "\n")
        f.write("-" * 60 + "\n")

        # 每个模型一行
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

    # 打印表格
    with open(table_path, "r") as f:
        print(f.read())


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage D: 模型评估")
    parser.add_argument("--models", type=str, nargs="+", default=["base", "sft", "grpo"],
                        help="要评估的模型类型")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="要评估的数据集")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="每个数据集的评估样本数（调试用）")
    args = parser.parse_args()

    if args.num_samples:
        EVAL_CONFIG["num_eval_samples"] = args.num_samples

    logger.info("=" * 60)
    logger.info("Stage D - 评估开始")
    logger.info("=" * 60)

    run_full_evaluation()

    logger.info("Stage D 完成！")


if __name__ == "__main__":
    main()
