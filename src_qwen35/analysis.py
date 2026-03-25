
"""Stage E - summarize qwen35 evaluation artifacts."""

import argparse
import json
import logging
from pathlib import Path

from config import EVAL_DATASETS, OUTPUT_BASE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"
ANALYSIS_DIR = OUTPUT_BASE / "analysis"


def load_all_results():
    results = {}
    for model_type in ("base", "sft", "grpo"):
        results[model_type] = {}
        for dataset_key in EVAL_DATASETS.keys():
            path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            if path.exists():
                results[model_type][dataset_key] = json.loads(path.read_text(encoding="utf-8"))
    return results


def generate_summary_report(results):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3.5-4B-Base Math Post-Training Report",
        "",
        "## Accuracy",
        "",
        "| Model | MATH-500 | GSM8K | TheoremQA |",
        "| --- | --- | --- | --- |",
    ]
    for model_type in ("base", "sft", "grpo"):
        row = results.get(model_type, {})
        lines.append(
            f"| {model_type} | {row.get('math500', {}).get('accuracy', 'N/A')} | {row.get('gsm8k', {}).get('accuracy', 'N/A')} | {row.get('theoremqa', {}).get('accuracy', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Parse Quality",
        "",
        "| Model | Dataset | Parse Rate | Invalid Extraction | Avg Completion Tokens | Think Compliance |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for model_type in ("base", "sft", "grpo"):
        for dataset_key in ("math500", "gsm8k", "theoremqa"):
            metrics = results.get(model_type, {}).get(dataset_key)
            if not metrics:
                continue
            lines.append(
                f"| {model_type} | {dataset_key} | {metrics.get('parse_rate', 'N/A')} | {metrics.get('invalid_extraction_rate', 'N/A')} | {metrics.get('avg_completion_tokens', 'N/A')} | {metrics.get('think_tag_compliance', 'N/A')} |"
            )

    report_path = ANALYSIS_DIR / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved summary report to %s", report_path)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Stage E: summarize qwen35 evaluation")
    parser.parse_args()
    results = load_all_results()
    if not any(results.values()):
        raise FileNotFoundError("No evaluation results found. Run evaluation first.")
    generate_summary_report(results)


if __name__ == "__main__":
    main()
