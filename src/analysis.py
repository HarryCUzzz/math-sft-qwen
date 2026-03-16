"""
Stage E - 分析 (Analysis)
==========================
对训练和评估结果进行深入分析，生成可视化图表和分析报告。

分析方向:
1. RL 对推理正确性 vs 通用能力的影响
2. 数据过滤策略的敏感性
3. 失效模式分析（格式偏移、奖励利用、通用性能变化）
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免需要 GUI
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_results"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
DATA_FILTERED = PROJECT_ROOT / "data" / "filtered"


def ensure_dirs():
    """创建分析输出目录"""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 评估结果对比可视化
# ============================================================
def plot_accuracy_comparison():
    """
    绘制三个模型在各评估数据集上的准确率对比柱状图。
    这是最核心的分析图表，展示 SFT 和 GRPO 各自带来的提升。
    """
    comparison_path = EVAL_RESULTS_DIR / "comparison.json"
    if not comparison_path.exists():
        logger.warning("对比结果文件不存在，跳过准确率对比图")
        return

    with open(comparison_path, "r") as f:
        comparison = json.load(f)

    # 提取数据
    models = ["base", "sft", "grpo"]
    datasets = ["math500", "gsm8k", "theoremqa"]
    dataset_labels = ["MATH-500", "GSM8K", "TheoremQA"]

    # 构建数据矩阵
    data = []
    for model in models:
        row = []
        for ds in datasets:
            acc = comparison.get(model, {}).get(ds, 0)
            row.append(acc)
        data.append(row)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25
    colors = ["#5B9BD5", "#70AD47", "#ED7D31"]  # 蓝、绿、橙
    model_labels = ["Base (Qwen2.5-0.5B)", "SFT", "GRPO"]

    for i, (model_data, color, label) in enumerate(zip(data, colors, model_labels)):
        bars = ax.bar(x + i * width, model_data, width, label=label, color=color)
        # 在柱状图上方标注数值
        for bar, val in zip(bars, model_data):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Evaluation Dataset", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Performance Comparison: Base vs SFT vs GRPO", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_labels)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = ANALYSIS_DIR / "accuracy_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"准确率对比图已保存: {save_path}")


# ============================================================
# 2. 训练曲线可视化
# ============================================================
def plot_training_curves():
    """
    从 TensorBoard 日志或 trainer_state.json 中提取训练曲线数据，
    绘制 loss 曲线和奖励曲线。
    """
    # 尝试读取 SFT 训练指标
    sft_state_path = PROJECT_ROOT / "outputs" / "sft_model" / "trainer_state.json"
    grpo_state_path = PROJECT_ROOT / "outputs" / "grpo_model" / "trainer_state.json"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 子图 1: SFT Loss 曲线 ----
    if sft_state_path.exists():
        with open(sft_state_path, "r") as f:
            sft_state = json.load(f)

        log_history = sft_state.get("log_history", [])
        # 提取 training loss
        train_steps = [e["step"] for e in log_history if "loss" in e]
        train_loss = [e["loss"] for e in log_history if "loss" in e]
        # 提取 eval loss
        eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
        eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]

        axes[0].plot(train_steps, train_loss, label="Training Loss", color="#5B9BD5")
        if eval_steps:
            axes[0].plot(eval_steps, eval_loss, label="Eval Loss", color="#ED7D31", marker="o")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("SFT Training Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "SFT training logs\nnot available",
                     ha="center", va="center", fontsize=12)
        axes[0].set_title("SFT Training Loss")

    # ---- 子图 2: GRPO 奖励曲线 ----
    if grpo_state_path.exists():
        with open(grpo_state_path, "r") as f:
            grpo_state = json.load(f)

        log_history = grpo_state.get("log_history", [])
        # 提取奖励相关指标
        steps = [e["step"] for e in log_history if "reward" in e or "rewards/combined" in e]
        rewards = [e.get("reward", e.get("rewards/combined", 0)) for e in log_history if "reward" in e or "rewards/combined" in e]

        if steps:
            axes[1].plot(steps, rewards, label="Total Reward", color="#70AD47")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Reward")
            axes[1].set_title("GRPO Training Reward")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "GRPO reward logs\nnot available",
                         ha="center", va="center", fontsize=12)
            axes[1].set_title("GRPO Training Reward")
    else:
        axes[1].text(0.5, 0.5, "GRPO training logs\nnot available",
                     ha="center", va="center", fontsize=12)
        axes[1].set_title("GRPO Training Reward")

    plt.tight_layout()
    save_path = ANALYSIS_DIR / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线图已保存: {save_path}")


# ============================================================
# 3. 失效模式分析
# ============================================================
def analyze_failure_modes():
    """
    分析模型的典型失效模式：
    1. 格式偏移: 模型只输出 \\boxed{} 而无推理过程
    2. 奖励利用: 格式正确但答案错误
    3. 长/短文本差异: 不同长度问题的表现差异

    从评估详情中读取每条预测结果进行分析。
    """
    analysis_results = {}

    # 遍历所有模型的评估结果文件
    for model_type in ["base", "sft", "grpo"]:
        for dataset_key in ["math500", "gsm8k", "theoremqa"]:
            detail_path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            if not detail_path.exists():
                continue

            with open(detail_path, "r") as f:
                result = json.load(f)

            # 如果有详细结果
            details = result.get("details", [])
            if not details:
                continue

            key = f"{model_type}_{dataset_key}"
            stats = {
                "total": len(details),
                "correct": sum(1 for d in details if d.get("is_correct")),
                "has_boxed": 0,          # 包含 \boxed{} 的数量
                "boxed_no_reasoning": 0,  # 有 boxed 但无推理过程
                "format_correct_answer_wrong": 0,  # 格式对但答案错
                "invalid_answer_extraction": 0,  # 无法从输出中提取有效答案
                "avg_prediction_length": 0.0,
                "avg_question_length": 0.0,
                "short_question_acc": 0,  # 短问题 (<100 字符) 的准确率
                "long_question_acc": 0,   # 长问题 (>500 字符) 的准确率
            }

            short_q_total, short_q_correct = 0, 0
            long_q_total, long_q_correct = 0, 0
            total_pred_len, total_q_len = 0, 0

            for d in details:
                pred = d.get("prediction", "")
                total_pred_len += len(pred)
                total_q_len += len(d.get("question", ""))

                # 检查是否含 \boxed{}
                if r'\boxed{' in pred:
                    stats["has_boxed"] += 1

                    # 检查是否有推理过程（少于 3 行认为无推理）
                    lines = [l.strip() for l in pred.split('\n') if l.strip()]
                    if len(lines) < 3:
                        stats["boxed_no_reasoning"] += 1

                    # 格式正确但答案错误
                    if not d.get("is_correct"):
                        stats["format_correct_answer_wrong"] += 1
                else:
                    # 没有 boxed 且末尾也没有明显数字时，记为答案提取失败
                    if not re.findall(r'-?\d+\.?\d*', pred):
                        stats["invalid_answer_extraction"] += 1

                # 按问题长度分析
                q_len = len(d.get("question", ""))
                if q_len < 100:
                    short_q_total += 1
                    if d.get("is_correct"):
                        short_q_correct += 1
                elif q_len > 500:
                    long_q_total += 1
                    if d.get("is_correct"):
                        long_q_correct += 1

            stats["short_question_acc"] = round(
                short_q_correct / max(short_q_total, 1) * 100, 1
            )
            stats["long_question_acc"] = round(
                long_q_correct / max(long_q_total, 1) * 100, 1
            )
            stats["boxed_rate"] = round(stats["has_boxed"] / max(stats["total"], 1) * 100, 1)
            stats["format_error_rate"] = round(stats["boxed_no_reasoning"] / max(stats["total"], 1) * 100, 1)
            stats["boxed_but_wrong_rate"] = round(
                stats["format_correct_answer_wrong"] / max(stats["total"], 1) * 100, 1
            )
            stats["invalid_answer_rate"] = round(
                stats["invalid_answer_extraction"] / max(stats["total"], 1) * 100, 1
            )
            stats["avg_prediction_length"] = round(total_pred_len / max(stats["total"], 1), 1)
            stats["avg_question_length"] = round(total_q_len / max(stats["total"], 1), 1)

            analysis_results[key] = stats

    # 保存分析结果
    if analysis_results:
        analysis_path = ANALYSIS_DIR / "failure_mode_analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        logger.info(f"失效模式分析已保存: {analysis_path}")

    return analysis_results


def collect_failure_examples(max_examples=3):
    """
    按模型与数据集收集典型失败案例，方便人工分析：
    1. 格式正确但答案错误
    2. 无法提取答案
    3. 输出过短/疑似空洞回答
    """
    example_results = {}

    for model_type in ["base", "sft", "grpo"]:
        for dataset_key in ["math500", "gsm8k", "theoremqa"]:
            detail_path = EVAL_RESULTS_DIR / f"{model_type}_{dataset_key}.json"
            if not detail_path.exists():
                continue

            with open(detail_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            details = result.get("details", [])
            if not details:
                continue

            grouped = {
                "boxed_but_wrong": [],
                "no_extractable_answer": [],
                "too_short_or_shallow": [],
            }

            for d in details:
                pred = d.get("prediction", "")
                question = d.get("question", "")
                reference = d.get("reference_answer", "")
                is_correct = d.get("is_correct", False)
                has_boxed = r"\boxed{" in pred
                has_number = bool(re.findall(r"-?\d+\.?\d*", pred))
                non_empty_lines = [line.strip() for line in pred.split("\n") if line.strip()]

                if has_boxed and not is_correct and len(grouped["boxed_but_wrong"]) < max_examples:
                    grouped["boxed_but_wrong"].append({
                        "question": question,
                        "reference_answer": reference,
                        "prediction": pred,
                    })

                if (not has_boxed and not has_number
                        and len(grouped["no_extractable_answer"]) < max_examples):
                    grouped["no_extractable_answer"].append({
                        "question": question,
                        "reference_answer": reference,
                        "prediction": pred,
                    })

                if (len(non_empty_lines) < 3 or len(pred) < 80) and not is_correct \
                        and len(grouped["too_short_or_shallow"]) < max_examples:
                    grouped["too_short_or_shallow"].append({
                        "question": question,
                        "reference_answer": reference,
                        "prediction": pred,
                    })

                if all(len(v) >= max_examples for v in grouped.values()):
                    break

            example_results[f"{model_type}_{dataset_key}"] = grouped

    if example_results:
        save_path = ANALYSIS_DIR / "failure_examples.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(example_results, f, ensure_ascii=False, indent=2)
        logger.info(f"典型失败案例已保存: {save_path}")

    return example_results


# ============================================================
# 4. 数据过滤敏感性分析
# ============================================================
def analyze_data_filtering_sensitivity():
    """
    分析数据过滤策略对模型效果的影响。
    读取 Stage A 的统计信息，并生成可视化。
    """
    stats_path = DATA_FILTERED / "stats.json"
    if not stats_path.exists():
        logger.warning("数据筛选统计文件不存在，跳过过滤敏感性分析")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # 绘制过滤漏斗图
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = [
        ("Original", stats.get("total_before", 0)),
        ("After Length Filter", stats.get("total_before", 0) - stats.get("removed_by_length", 0)),
        ("After Answer Filter", stats.get("total_before", 0) - stats.get("removed_by_length", 0) - stats.get("removed_by_answer", 0)),
        ("After Garbled Filter", stats.get("total_before", 0) - stats.get("removed_by_length", 0) - stats.get("removed_by_answer", 0) - stats.get("removed_by_garbled", 0)),
        ("After Dedup", stats.get("total_after", 0)),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    bars = ax.barh(labels[::-1], values[::-1], color=["#70AD47", "#5B9BD5", "#5B9BD5", "#5B9BD5", "#C0C0C0"])
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2.,
                f"{val:,}", ha="left", va="center", fontsize=10)

    ax.set_xlabel("Number of Samples")
    ax.set_title("Data Filtering Funnel: Impact of Each Filter Stage")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = ANALYSIS_DIR / "data_filtering_funnel.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"数据过滤漏斗图已保存: {save_path}")


# ============================================================
# 5. 生成分析报告
# ============================================================
def generate_analysis_report(failure_analysis):
    """
    生成完整的分析报告（Markdown 格式）。
    """
    report_path = ANALYSIS_DIR / "analysis_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AIMS5740 Project - 分析报告\n\n")
        f.write("## 1. 模型性能对比\n\n")

        # 读取对比结果
        comparison_path = EVAL_RESULTS_DIR / "comparison.json"
        if comparison_path.exists():
            with open(comparison_path, "r") as cf:
                comparison = json.load(cf)

            f.write("| Model | MATH-500 | GSM8K | TheoremQA |\n")
            f.write("|-------|----------|-------|-----------|\n")
            for model in ["base", "sft", "grpo"]:
                if model in comparison:
                    math = comparison[model].get("math500", "N/A")
                    gsm = comparison[model].get("gsm8k", "N/A")
                    thm = comparison[model].get("theoremqa", "N/A")
                    f.write(f"| {model} | {math}% | {gsm}% | {thm}% |\n")

            f.write("\n")

        # RL 对推理正确性 vs 通用能力的影响
        f.write("## 2. RL 对推理正确性 vs 通用能力的影响\n\n")
        f.write("通过对比 SFT 和 GRPO 模型在不同类型数学任务上的表现，分析 RL 训练的效果：\n\n")
        f.write("- **数学推理能力**: GRPO 是否在 MATH-500 和 GSM8K 上带来显著提升\n")
        f.write("- **泛化能力**: GRPO 在 TheoremQA（分布外测试集）上的表现如何\n")
        f.write("- **能力遗忘**: 若有通用问答评测，需检查 RL 是否损害了非数学任务的表现\n\n")

        # 数据过滤敏感性
        f.write("## 3. 数据过滤策略敏感性\n\n")
        stats_path = DATA_FILTERED / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as sf:
                stats = json.load(sf)
            f.write(f"- 原始数据量: {stats.get('total_before', 'N/A')}\n")
            f.write(f"- 筛选后数据量: {stats.get('total_after', 'N/A')}\n")
            f.write(f"- 过滤比例: {(1 - stats.get('total_after', 0) / max(stats.get('total_before', 1), 1)) * 100:.1f}%\n")
            f.write(f"- 长度过滤移除: {stats.get('removed_by_length', 'N/A')}\n")
            f.write(f"- 答案格式移除: {stats.get('removed_by_answer', 'N/A')}\n")
            f.write(f"- 去重移除: {stats.get('removed_by_dedup', 'N/A')}\n\n")

        # 失效模式分析
        f.write("## 4. 失效模式分析\n\n")
        if failure_analysis:
            for key, stats in failure_analysis.items():
                f.write(f"### {key}\n")
                f.write(f"- 总样本数: {stats['total']}\n")
                f.write(f"- 正确率: {stats['correct']}/{stats['total']} = {stats['correct']/max(stats['total'],1)*100:.1f}%\n")
                f.write(f"- 包含 \\\\boxed{{}}: {stats['has_boxed']}\n")
                f.write(f"- \\\\boxed{{}} 命中率: {stats['boxed_rate']}%\n")
                f.write(f"- 格式偏移（有 boxed 但无推理）: {stats['boxed_no_reasoning']}\n")
                f.write(f"- 格式偏移率: {stats['format_error_rate']}%\n")
                f.write(f"- 奖励利用（格式对答案错）: {stats['format_correct_answer_wrong']}\n")
                f.write(f"- boxed 但错误比例: {stats['boxed_but_wrong_rate']}%\n")
                f.write(f"- 答案提取失败数: {stats['invalid_answer_extraction']}\n")
                f.write(f"- 答案提取失败率: {stats['invalid_answer_rate']}%\n")
                f.write(f"- 平均输出长度: {stats['avg_prediction_length']}\n")
                f.write(f"- 平均问题长度: {stats['avg_question_length']}\n")
                f.write(f"- 短问题准确率: {stats['short_question_acc']}%\n")
                f.write(f"- 长问题准确率: {stats['long_question_acc']}%\n\n")

        f.write("## 5. 结论与建议\n\n")
        f.write("根据以上分析，总结以下关键发现：\n\n")
        f.write("1. **数据筛选的重要性**: 高质量、格式规范的数据是 SFT 效果的基础\n")
        f.write("2. **GRPO 的效果**: RL 后训练可以进一步提升推理正确性\n")
        f.write("3. **奖励设计**: 多组成部分的奖励函数有助于平衡正确性和格式规范性\n")
        f.write("4. **失效模式**: 需要关注格式偏移、答案提取失败和奖励利用等问题\n")
        f.write("5. **后续实验**: 推荐补充 Raw SFT vs Filtered SFT vs GRPO 的对照实验，进一步拆分数据质量与 RL 的贡献\n")

    logger.info(f"分析报告已生成: {report_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Stage E - 分析开始")
    logger.info("=" * 60)

    ensure_dirs()

    # 1. 评估结果对比可视化
    logger.info("1. 生成准确率对比图...")
    plot_accuracy_comparison()

    # 2. 训练曲线可视化
    logger.info("2. 生成训练曲线图...")
    plot_training_curves()

    # 3. 失效模式分析
    logger.info("3. 分析失效模式...")
    failure_analysis = analyze_failure_modes()
    collect_failure_examples()

    # 4. 数据过滤敏感性分析
    logger.info("4. 分析数据过滤敏感性...")
    analyze_data_filtering_sensitivity()

    # 5. 生成完整分析报告
    logger.info("5. 生成分析报告...")
    generate_analysis_report(failure_analysis)

    logger.info("=" * 60)
    logger.info("Stage E 完成！所有分析结果已保存至 outputs/analysis/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
