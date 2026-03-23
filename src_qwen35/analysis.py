"""
Stage E - 结果分析与可视化 - Qwen3.5-4B 版本
============================================
分析评估结果，生成统计报告和可视化图表。

分析维度:
1. 准确率对比 (Base vs SFT vs GRPO)
2. 逐数据集分析
3. 失败案例分析

输出:
- outputs_qwen35/analysis/   分析报告和图表
"""

import argparse
import json
import logging
from pathlib import Path

from config import EVAL_DATASETS, OUTPUT_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"
ANALYSIS_DIR = OUTPUT_BASE / "analysis"


def load_all_results():
    """加载所有评估结果"""
    results = {}

    for model_type in ["base", "sft", "grpo"]:
        results[model_type] = {}
        for dataset_key in EVAL_DATASETS.keys():
            result_file = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            if result_file.exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    results[model_type][dataset_key] = json.load(f)
            else:
                logger.warning(f"结果文件不存在: {result_file}")

    return results


def generate_summary_report(results):
    """生成汇总报告"""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        "Qwen3.5-4B 数学推理训练评估报告",
        "=" * 70,
        "",
        "## 1. 准确率对比",
        "",
        "| 模型 | MATH-500 | GSM8K | TheoremQA | 平均 |",
        "|------|----------|-------|-----------|------|",
    ]

    for model_type in ["base", "sft", "grpo"]:
        if model_type not in results:
            continue

        model_results = results[model_type]
        accs = []

        row = f"| {model_type} |"
        for dataset_key in ["math500", "gsm8k", "theoremqa"]:
            if dataset_key in model_results:
                acc = model_results[dataset_key].get("accuracy", "N/A")
                row += f" {acc}% |"
                if isinstance(acc, (int, float)):
                    accs.append(acc)
            else:
                row += " N/A |"

        avg_acc = sum(accs) / len(accs) if accs else 0
        row += f" {avg_acc:.2f}% |"
        report_lines.append(row)

    report_lines.extend([
        "",
        "## 2. 改进幅度",
        "",
    ])

    # 计算改进幅度
    base_results = results.get("base", {})
    for model_type in ["sft", "grpo"]:
        if model_type not in results:
            continue

        report_lines.append(f"### {model_type.upper()} vs Base")
        report_lines.append("")

        model_results = results[model_type]
        for dataset_key in ["math500", "gsm8k", "theoremqa"]:
            if dataset_key in model_results and dataset_key in base_results:
                base_acc = base_results[dataset_key].get("accuracy", 0)
                model_acc = model_results[dataset_key].get("accuracy", 0)
                improvement = model_acc - base_acc
                sign = "+" if improvement >= 0 else ""
                report_lines.append(
                    f"- {dataset_key}: {base_acc}% -> {model_acc}% ({sign}{improvement:.2f}%)"
                )

        report_lines.append("")

    # 保存报告
    report_path = ANALYSIS_DIR / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"汇总报告已保存至: {report_path}")
    print("\n".join(report_lines))

    return report_path


def analyze_failure_cases(results, dataset_key="math500", model_type="grpo"):
    """分析失败案例"""
    if model_type not in results or dataset_key not in results[model_type]:
        logger.warning(f"无法分析 {model_type} 在 {dataset_key} 上的失败案例")
        return

    result = results[model_type][dataset_key]
    details = result.get("details", [])

    if not details:
        logger.warning("没有详细结果可供分析")
        return

    # 统计失败案例
    failures = [d for d in details if not d.get("is_correct", False)]
    logger.info(f"失败案例数: {len(failures)} / {len(details)}")

    # 分析失败原因
    failure_reasons = {
        "no_boxed": 0,        # 没有使用 \boxed{} 格式
        "wrong_answer": 0,    # 答案错误
        "truncated": 0,       # 输出被截断
    }

    for failure in failures:
        pred = failure.get("prediction", "")
        if r"\boxed{" not in pred:
            failure_reasons["no_boxed"] += 1
        elif len(pred) > 900:  # 接近截断长度
            failure_reasons["truncated"] += 1
        else:
            failure_reasons["wrong_answer"] += 1

    # 生成失败分析报告
    analysis_lines = [
        f"# {model_type.upper()} 在 {dataset_key} 上的失败案例分析",
        "",
        f"总失败数: {len(failures)} / {len(details)} ({len(failures)/len(details)*100:.1f}%)",
        "",
        "## 失败原因分布",
        "",
        f"- 未使用 \\boxed{{}} 格式: {failure_reasons['no_boxed']}",
        f"- 答案计算错误: {failure_reasons['wrong_answer']}",
        f"- 输出可能被截断: {failure_reasons['truncated']}",
        "",
        "## 失败案例样例 (前 5 个)",
        "",
    ]

    for i, failure in enumerate(failures[:5]):
        analysis_lines.extend([
            f"### 案例 {i+1}",
            "",
            f"**问题**: {failure.get('question', 'N/A')[:300]}...",
            "",
            f"**参考答案**: {failure.get('reference_answer', 'N/A')}",
            "",
            f"**模型输出**: {failure.get('prediction', 'N/A')[:500]}...",
            "",
            "---",
            "",
        ])

    # 保存分析报告
    analysis_path = ANALYSIS_DIR / f"failure_analysis_{model_type}_{dataset_key}.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))

    logger.info(f"失败案例分析已保存至: {analysis_path}")


def generate_accuracy_chart(results):
    """生成准确率对比图表 (ASCII 版本)"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        datasets = ["math500", "gsm8k", "theoremqa"]
        models = ["base", "sft", "grpo"]
        colors = ["#4CAF50", "#2196F3", "#FF9800"]

        data = []
        for model in models:
            model_data = []
            for ds in datasets:
                if model in results and ds in results[model]:
                    model_data.append(results[model][ds].get("accuracy", 0))
                else:
                    model_data.append(0)
            data.append(model_data)

        x = np.arange(len(datasets))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (model, color) in enumerate(zip(models, colors)):
            bars = ax.bar(x + i * width, data[i], width, label=model.upper(), color=color)
            ax.bar_label(bars, fmt='%.1f%%', padding=3)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Qwen3.5-4B Math Reasoning Evaluation')
        ax.set_xticks(x + width)
        ax.set_xticklabels([ds.upper() for ds in datasets])
        ax.legend()
        ax.set_ylim(0, 100)

        chart_path = ANALYSIS_DIR / "accuracy_comparison.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()

        logger.info(f"准确率对比图已保存至: {chart_path}")

    except ImportError:
        logger.warning("matplotlib 未安装，跳过图表生成")
        # 生成 ASCII 版本的对比图
        ascii_chart_lines = [
            "",
            "准确率对比 (ASCII 版本)",
            "=" * 50,
        ]

        for model_type in ["base", "sft", "grpo"]:
            if model_type not in results:
                continue

            ascii_chart_lines.append(f"\n{model_type.upper()}:")
            for dataset_key in ["math500", "gsm8k", "theoremqa"]:
                if dataset_key in results[model_type]:
                    acc = results[model_type][dataset_key].get("accuracy", 0)
                    bar_len = int(acc / 2)
                    bar = "#" * bar_len
                    ascii_chart_lines.append(f"  {dataset_key:12s} [{bar:<50s}] {acc:.1f}%")

        ascii_chart_path = ANALYSIS_DIR / "accuracy_comparison.txt"
        with open(ascii_chart_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ascii_chart_lines))

        logger.info(f"ASCII 准确率对比图已保存至: {ascii_chart_path}")
        print("\n".join(ascii_chart_lines))


def main():
    parser = argparse.ArgumentParser(description="Stage E: 结果分析")
    parser.add_argument(
        "--failure-analysis",
        action="store_true",
        help="执行失败案例分析",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage E - 结果分析开始")
    logger.info("=" * 60)

    # 加载结果
    results = load_all_results()

    if not results or all(not v for v in results.values()):
        logger.error("没有找到评估结果，请先运行评估脚本")
        return

    # 生成汇总报告
    generate_summary_report(results)

    # 生成准确率图表
    generate_accuracy_chart(results)

    # 失败案例分析
    if args.failure_analysis:
        for model_type in ["sft", "grpo"]:
            for dataset_key in ["math500", "gsm8k"]:
                analyze_failure_cases(results, dataset_key, model_type)

    logger.info("Stage E 完成！")


if __name__ == "__main__":
    main()
