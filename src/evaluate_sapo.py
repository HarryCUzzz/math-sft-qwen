"""
Evaluate the SAPO-trained model without changing the original evaluation flow.
"""

import json
import logging
from pathlib import Path

import torch

from evaluation import (
    EVAL_CONFIG,
    EVAL_DATASETS,
    check_answer,
    generate_answer,
    load_eval_dataset,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAPO_MODEL_DIR = PROJECT_ROOT / "outputs" / "sapo_model"
EVAL_RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_results"


def load_sapo_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(SAPO_MODEL_DIR) if SAPO_MODEL_DIR.exists() else EVAL_CONFIG["base_model"]
    if not SAPO_MODEL_DIR.exists():
        logger.warning("SAPO model directory not found, falling back to base model.")

    tokenizer = AutoTokenizer.from_pretrained(
        EVAL_CONFIG["base_model"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key):
    correct = 0
    total = len(eval_data)
    results = []

    logger.info("Evaluating SAPO on %s with %s samples", dataset_key, total)
    for idx, item in enumerate(eval_data):
        prediction = generate_answer(model, tokenizer, item["question"])
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

        if (idx + 1) % 50 == 0:
            logger.info(
                "  Progress: %s/%s, accuracy %.1f%%",
                idx + 1,
                total,
                correct / (idx + 1) * 100,
            )

    accuracy = correct / max(total, 1) * 100
    return {
        "dataset": dataset_key,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "details": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the SAPO model on math benchmarks.")
    parser.add_argument("--datasets", nargs="+", default=list(EVAL_DATASETS.keys()))
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    if args.num_samples:
        EVAL_CONFIG["num_eval_samples"] = args.num_samples

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_sapo_model_and_tokenizer()
    summary = {}

    try:
        for dataset_key in args.datasets:
            eval_data = load_eval_dataset(dataset_key)
            if not eval_data:
                logger.warning("Skipping empty dataset: %s", dataset_key)
                continue
            result = evaluate_model_on_dataset(model, tokenizer, eval_data, dataset_key)
            summary[dataset_key] = result["accuracy"]
            result_path = EVAL_RESULTS_DIR / f"sapo_{dataset_key}.json"
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            logger.info("Saved SAPO evaluation to %s", result_path)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = EVAL_RESULTS_DIR / "sapo_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("Saved SAPO summary to %s", summary_path)


if __name__ == "__main__":
    main()
