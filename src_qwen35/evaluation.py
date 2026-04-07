"""Stage D - evaluate Base, SFT, and GRPO with shared answer parsing logic."""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_utils import answers_equivalent, extract_candidate_answer
from config import (
    DATA_EVAL,
    EVAL_CONFIG,
    EVAL_DATASETS,
    OUTPUT_BASE,
    THINKING_SYSTEM_PROMPT,
    build_conditioned_user_prompt,
    get_eval_condition,
    get_sft_output_dirs,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRPO_MODEL_DIR = OUTPUT_BASE / "grpo_model"
EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"


def _resolve_eval_sft_dir() -> Path:
    stage = os.environ.get("EVAL_SFT_STAGE", "main").strip().lower()
    model_dir, _ = get_sft_output_dirs(stage)
    return model_dir


def load_model_and_tokenizer(model_type="base"):
    base_name = EVAL_CONFIG["base_model"]
    base_path = Path(base_name).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {base_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if EVAL_CONFIG.get("use_flash_attention", True):
        load_kwargs["attn_implementation"] = "flash_attention_2"

    if model_type == "base":
        model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
    elif model_type == "sft":
        base_model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
        sft_model_dir = _resolve_eval_sft_dir()
        if (sft_model_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(sft_model_dir)).merge_and_unload()
        else:
            raise FileNotFoundError(f"SFT adapter missing: {sft_model_dir}")
    elif model_type == "grpo":
        base_model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
        if (GRPO_MODEL_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(GRPO_MODEL_DIR)).merge_and_unload()
        elif (GRPO_MODEL_DIR / "config.json").exists():
            model = AutoModelForCausalLM.from_pretrained(str(GRPO_MODEL_DIR), **load_kwargs)
        else:
            raise FileNotFoundError(f"GRPO model missing: {GRPO_MODEL_DIR}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model, tokenizer


def load_eval_dataset(dataset_key):
    config = EVAL_DATASETS[dataset_key]
    dataset = None
    local_path = DATA_EVAL / dataset_key
    if local_path.exists():
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(str(local_path))
        except Exception as exc:
            logger.warning("Failed to load local dataset %s: %s", local_path, exc)

    if dataset is None:
        from datasets import load_dataset

        if "subset" in config:
            dataset = load_dataset(config["name_hf"], config["subset"], split=config["split"])
        else:
            dataset = load_dataset(config["name_hf"], split=config["split"])

    if hasattr(dataset, "keys") and not isinstance(dataset, list) and not hasattr(dataset, "column_names"):
        split = config["split"] if config["split"] in dataset else list(dataset.keys())[0]
        dataset = dataset[split]

    eval_rows = []
    for item in dataset:
        question = item.get(config["question_key"], "")
        answer = item.get(config["answer_key"], "")
        if question and answer:
            eval_rows.append({"question": str(question), "reference_answer": str(answer)})

    if EVAL_CONFIG.get("num_eval_samples"):
        rng = random.Random(EVAL_CONFIG.get("sample_seed", 42))
        if len(eval_rows) > EVAL_CONFIG["num_eval_samples"]:
            eval_rows = rng.sample(eval_rows, EVAL_CONFIG["num_eval_samples"])
    return eval_rows


@torch.no_grad()
def evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key):
    correct = 0
    parsed = 0
    think_compliant = 0
    total_completion_tokens = 0
    results = []
    batch_size = EVAL_CONFIG["batch_size"]
    condition = get_eval_condition(dataset_key)

    for offset in range(0, len(eval_data), batch_size):
        batch = eval_data[offset : offset + batch_size]
        prompts = []
        for item in batch:
            conditioned_user_prompt = build_conditioned_user_prompt(
                item["question"],
                condition["task_type"],
                condition["domain"],
                condition["difficulty"],
                condition["reasoning_style"],
            )
            messages = [
                {"role": "system", "content": THINKING_SYSTEM_PROMPT},
                {"role": "user", "content": conditioned_user_prompt},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        generate_kwargs = {
            "max_new_tokens": EVAL_CONFIG["max_new_tokens"],
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": EVAL_CONFIG["do_sample"],
            "use_cache": True,
        }
        if EVAL_CONFIG["do_sample"]:
            generate_kwargs["temperature"] = EVAL_CONFIG["temperature"]

        outputs = model.generate(**inputs, **generate_kwargs)
        generated = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        for item, prediction in zip(batch, generated):
            candidate = extract_candidate_answer(prediction)
            is_correct = answers_equivalent(prediction, item["reference_answer"], dataset_key)
            if candidate:
                parsed += 1
            if "<think>" in prediction and "</think>" in prediction and "Final answer:" in prediction:
                think_compliant += 1
            if is_correct:
                correct += 1
            total_completion_tokens += len(tokenizer(prediction, add_special_tokens=False)["input_ids"])
            results.append(
                {
                    "question": item["question"][:200],
                    "reference_answer": item["reference_answer"],
                    "prediction": prediction[:800],
                    "parsed_answer": candidate,
                    "is_correct": is_correct,
                }
            )

    total = max(len(eval_data), 1)
    parse_rate = parsed / total * 100
    return {
        "dataset": dataset_key,
        "total": len(eval_data),
        "correct": correct,
        "accuracy": round(correct / total * 100, 2),
        "parse_rate": round(parse_rate, 2),
        "invalid_extraction_rate": round(100 - parse_rate, 2),
        "avg_completion_tokens": round(total_completion_tokens / total, 2),
        "think_tag_compliance": round(think_compliant / total * 100, 2),
        "details": results if EVAL_CONFIG.get("save_details", True) else [],
    }


def generate_comparison_table(all_results):
    comparison = {}
    for model_type, model_results in all_results.items():
        comparison[model_type] = {dataset_key: result["accuracy"] for dataset_key, result in model_results.items()}

    comparison_path = EVAL_RESULTS_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    table_lines = [
        "=" * 80,
        "Qwen3.5-4B-Base evaluation summary",
        "=" * 80,
        f"{'Model':<12}{'MATH-500':<12}{'GSM8K':<12}{'TheoremQA':<12}",
        "-" * 80,
    ]
    for model_type in ("base", "sft", "grpo"):
        row = comparison.get(model_type, {})
        table_lines.append(
            f"{model_type:<12}{str(row.get('math500', 'N/A')):<12}{str(row.get('gsm8k', 'N/A')):<12}{str(row.get('theoremqa', 'N/A')):<12}"
        )
    (EVAL_RESULTS_DIR / "comparison_table.txt").write_text("\n".join(table_lines) + "\n", encoding="utf-8")


def run_full_evaluation(model_types=None, datasets=None, skip_datasets=None):
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_types = model_types or ["base", "sft", "grpo"]
    datasets = datasets or list(EVAL_DATASETS.keys())
    skip_datasets = set(skip_datasets or [])
    all_results = {}

    for model_type in model_types:
        model, tokenizer = load_model_and_tokenizer(model_type)
        model_results = {}
        for dataset_key in datasets:
            if dataset_key in skip_datasets:
                continue
            eval_data = load_eval_dataset(dataset_key)
            result = evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key)
            model_results[dataset_key] = result
            (EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("%s on %s: %s", model_type, dataset_key, result)
        all_results[model_type] = model_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generate_comparison_table(all_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage D: Qwen3.5-4B-Base evaluation")
    parser.add_argument("--models", type=str, nargs="+", default=["base", "sft", "grpo"])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--skip-datasets", type=str, nargs="+", default=[])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.num_samples:
        EVAL_CONFIG["num_eval_samples"] = args.num_samples
    if "EVAL_SAMPLE_SEED" in os.environ:
        EVAL_CONFIG["sample_seed"] = int(os.environ["EVAL_SAMPLE_SEED"])
    if args.summary_only:
        EVAL_CONFIG["save_details"] = False

    run_full_evaluation(model_types=args.models, datasets=args.datasets, skip_datasets=args.skip_datasets)


if __name__ == "__main__":
    main()
