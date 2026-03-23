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

import json
import logging
import os
import re
from pathlib import Path

import torch
# 新增：导入4bit量化配置类
from transformers import BitsAndBytesConfig

# 设置 Hugging Face 镜像源（用于国内网络环境）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
# 使用本地模型路径（如果网络不可达）
# 可以通过环境变量 BASE_MODEL_PATH 覆盖，或者直接修改此处的路径
EVAL_CONFIG = {
    "base_model": os.environ.get(
        "BASE_MODEL_PATH", "/home/lyl/models/Qwen/Qwen2___5-0___5B"
    ),
    "max_new_tokens": 256,  # 生成速度快4倍
    "temperature": 0.0,
    "do_sample": False,
    "num_eval_samples": None,
}

# 新增：4bit量化配置（封装成Config类，解决参数不识别问题）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 评估数据集列表
EVAL_DATASETS = {
    "math500": {
        "name_modelscope": "MATH-500",
        "name_hf": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_key": "problem",
        "answer_key": "answer",
    },
    "gsm8k": {
        "name_modelscope": "modelscope/gsm8k",
        "name_hf": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
    },
    "theoremqa": {
        "name_modelscope": "TIGER-Lab/TheoremQA",
        "name_hf": "TIGER-Lab/TheoremQA",
        "split": "test",
        "question_key": "Question",
        "answer_key": "Answer",
    },
}


# ============================================================
# 1. 答案提取和匹配工具
# ============================================================
def extract_boxed_answer(text):
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
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "")
        return ans
    return text.strip()


def normalize_for_comparison(answer_str):
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
    pred_answer = extract_boxed_answer(predicted)
    if pred_answer is None:
        lines = predicted.strip().split("\n")
        for line in reversed(lines):
            nums = re.findall(r"-?\d+\.?\d*", line)
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
# 2. 模型加载（核心修复）
# ============================================================
def load_model_and_tokenizer(model_type="base"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = EVAL_CONFIG["base_model"]
    base_path = Path(base_name).resolve()
    if not base_path.exists():
        fallback_path = Path("/home/lyl/models/Qwen/Qwen2___5-0___5B").resolve()
        if fallback_path.exists():
            base_name = str(fallback_path)
            logger.info(f"使用备用模型路径: {base_name}")
        else:
            raise FileNotFoundError(f"模型路径不存在: {base_path} 和 {fallback_path}")

    logger.info(f"加载 {model_type} 模型...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 核心修复：用 quantization_config 传4bit配置，torch_dtype改dtype
    if model_type == "base":
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,  # 修复：torch_dtype → dtype
            device_map="auto",
            local_files_only=True,
            quantization_config=bnb_config,  # 修复：用配置类传4bit参数
        )
    elif model_type == "sft":
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            quantization_config=bnb_config,
        )
        if (SFT_MODEL_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(SFT_MODEL_DIR))
            model = model.merge_and_unload()
        else:
            logger.warning("SFT adapter 未找到，将使用基座模型代替")
            model = base_model
    elif model_type == "grpo":
        if GRPO_MODEL_DIR.exists():
            model = AutoModelForCausalLM.from_pretrained(
                str(GRPO_MODEL_DIR),
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                quantization_config=bnb_config,
            )
        else:
            logger.warning("GRPO 模型未找到，将使用基座模型代替")
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                quantization_config=bnb_config,
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
    config = EVAL_DATASETS[dataset_key]
    dataset = None

    try:
        from modelscope.msdatasets import MsDataset
        logger.info(f"尝试从 ModelScope 加载数据集: {config['name_modelscope']}")
        if "subset" in config:
            dataset = MsDataset.load(
                config["name_modelscope"],
                subset_name=config["subset"],
                split=config["split"]
            )
        else:
            dataset = MsDataset.load(config["name_modelscope"], split=config["split"])
        logger.info(f"  ✓ ModelScope 加载成功")
    except Exception as e:
        logger.warning(f"  ✗ ModelScope 加载失败: {e}")
        try:
            from datasets import load_dataset
            logger.info(f"尝试从 HuggingFace 镜像加载数据集: {config['name_hf']}")
            if "subset" in config:
                dataset = load_dataset(
                    config["name_hf"],
                    config["subset"],
                    split=config["split"]
                )
            else:
                dataset = load_dataset(config["name_hf"], split=config["split"])
            logger.info(f"  ✓ HuggingFace 加载成功")
        except Exception as e2:
            logger.error(f"  ✗ HuggingFace 加载也失败: {e2}")
            return []

    if dataset is None:
        logger.error(f"数据集 {dataset_key} 加载失败")
        return []

    eval_data = []
    for item in dataset:
        question = item.get(config["question_key"], "")
        answer = item.get(config["answer_key"], "")
        if question and answer:
            eval_data.append({
                "question": str(question),
                "reference_answer": str(answer),
            })

    if EVAL_CONFIG["num_eval_samples"]:
        eval_data = eval_data[: EVAL_CONFIG["num_eval_samples"]]

    logger.info(f"  加载了 {len(eval_data)} 条评估数据")
    return eval_data


# ============================================================
# 4. 模型推理
# ============================================================
@torch.no_grad()
def generate_answer(model, tokenizer, question, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are a helpful math assistant. Solve problems step by step and put your final answer in \\boxed{}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=EVAL_CONFIG["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        use_cache=True,
        max_length=None,
        repetition_penalty=1.0,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer


# ============================================================
# 5. 评估主流程
# ============================================================
def evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key):
    correct = 0
    total = len(eval_data)
    results = []
    batch_size = 4

    logger.info(f"开始评估 ({dataset_key}), 共 {total} 条...")

    for idx in range(0, total, batch_size):
        batch_data = eval_data[idx:idx+batch_size]
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
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=EVAL_CONFIG["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            use_cache=True,
            max_length=None,
            repetition_penalty=1.0,
        )

        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        for i, (item, prediction) in enumerate(zip(batch_data, generated_texts)):
            is_correct = check_answer(prediction, item["reference_answer"], dataset_key)
            if is_correct:
                correct += 1

            results.append({
                "question": item["question"][:200],
                "reference_answer": item["reference_answer"],
                "prediction": prediction[:500],
                "is_correct": is_correct,
            })

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
        "details": results,
    }


def run_full_evaluation():
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_types = ["base", "sft", "grpo"]
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
            eval_data = load_eval_dataset(dataset_key)
            if not eval_data:
                logger.warning(f"数据集 {dataset_key} 为空，跳过")
                continue

            result = evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key)
            model_results[dataset_key] = result
            logger.info(f"  {dataset_key} 准确率: {result['accuracy']}%")

            result_path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            summary = {k: v for k, v in result.items() if k != "details"}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        all_results[model_type] = model_results
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    generate_comparison_table(all_results)
    return all_results


def generate_comparison_table(all_results):
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
        f.write("模型评估结果对比表 (Accuracy %)\n")
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
    import argparse
    parser = argparse.ArgumentParser(description="Stage D: 模型评估")
    parser.add_argument("--models", type=str, nargs="+", default=["base", "sft", "grpo"])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
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
