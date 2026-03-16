"""
Stage A - 数据筛选 (Data Selection)
=====================================
从 DeepMath-103K 和 Big-Math-RL-Verified 数据集中筛选高质量数学推理样本。

过滤规则：
1. 移除问题文本过短(<10字符)或过长(>2000字符)的样本
2. 规范化最终答案格式为 \\boxed{}
3. 基于 hash 去重
4. 移除乱码和格式损坏的样本
5. 控制难度范围

输出：
- data/filtered/sft_train.json  (SFT 训练集, alpaca 格式)
- data/filtered/sft_eval.json   (SFT 验证集)
- data/rl_prompts/rl_train.json (RL 训练用 prompts)
- data/filtered/stats.json      (数据统计信息)
"""

import os
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter

from datasets import load_dataset
import pandas as pd

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FILTERED = PROJECT_ROOT / "data" / "filtered"
DATA_RL = PROJECT_ROOT / "data" / "rl_prompts"


def ensure_dirs():
    """创建必要的目录结构"""
    for d in [DATA_RAW, DATA_FILTERED, DATA_RL]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 数据下载
# ============================================================
def download_datasets():
    """
    从 HuggingFace 下载数据集：
    - DeepMath-103K: 包含大量数学推理题目和解答
    - Big-Math-RL-Verified: 经过 RL 验证的数学数据集
    """
    logger.info("正在下载 DeepMath-103K 数据集...")
    try:
        deepmath = load_dataset("zwhe99/DeepMath-103K", split="train")
        logger.info(f"DeepMath-103K 下载完成, 共 {len(deepmath)} 条数据")
    except Exception as e:
        logger.warning(f"DeepMath-103K 下载失败: {e}, 将跳过此数据集")
        deepmath = None

    logger.info("正在下载 Big-Math-RL-Verified 数据集...")
    try:
        bigmath = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
        logger.info(f"Big-Math-RL-Verified 下载完成, 共 {len(bigmath)} 条数据")
    except Exception as e:
        logger.warning(f"Big-Math-RL-Verified 下载失败: {e}, 将跳过此数据集")
        bigmath = None

    return deepmath, bigmath


# ============================================================
# 2. 数据统一格式
# ============================================================
def normalize_deepmath(dataset):
    """
    将 DeepMath-103K 统一为标准格式:
    [{"question": ..., "answer": ..., "solution": ...}, ...]

    DeepMath-103K 的典型字段: question, answer, solution 等
    """
    samples = []
    for item in dataset:
        # 尝试提取字段（不同版本字段名可能不同）
        question = item.get("question", item.get("problem", item.get("input", "")))
        answer = item.get("answer", item.get("final_answer", ""))
        solution = item.get("solution", item.get("output", item.get("response", "")))

        if question and (answer or solution):
            samples.append({
                "question": str(question).strip(),
                "answer": str(answer).strip() if answer else "",
                "solution": str(solution).strip() if solution else "",
                "source": "DeepMath-103K",
            })
    logger.info(f"DeepMath-103K 标准化后: {len(samples)} 条")
    return samples


def normalize_bigmath(dataset):
    """
    将 Big-Math-RL-Verified 统一为标准格式。
    该数据集经过 RL 验证，质量较高。
    """
    samples = []
    for item in dataset:
        question = item.get("problem", item.get("question", item.get("input", "")))
        answer = item.get("answer", item.get("expected_answer", ""))
        solution = item.get("solution", item.get("output", item.get("response", "")))

        if question and (answer or solution):
            samples.append({
                "question": str(question).strip(),
                "answer": str(answer).strip() if answer else "",
                "solution": str(solution).strip() if solution else "",
                "source": "Big-Math-RL-Verified",
            })
    logger.info(f"Big-Math-RL-Verified 标准化后: {len(samples)} 条")
    return samples


# ============================================================
# 3. 过滤规则
# ============================================================
def extract_boxed_answer(text):
    """
    从文本中提取 \\boxed{} 内的答案。
    支持嵌套大括号。
    """
    # 查找 \boxed{ 的位置
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # 取最后一个 \boxed{} （通常最终答案在最后）
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


def normalize_answer(answer_str):
    """
    规范化答案字符串：去除多余空格、统一格式。
    """
    if not answer_str:
        return ""
    # 去除首尾空白
    ans = answer_str.strip()
    # 去除 $ 符号
    ans = ans.replace("$", "")
    # 去除多余空格
    ans = re.sub(r'\s+', ' ', ans)
    return ans


def is_garbled(text):
    """
    检测文本是否包含乱码。
    如果非 ASCII 且非常见 Unicode（中文、LaTeX 等）字符占比过高，则认为是乱码。
    """
    if not text:
        return True
    # 统计不可识别字符的比例
    total = len(text)
    # 允许 ASCII、中文、日文、韩文、LaTeX 常用符号
    valid_count = sum(
        1 for c in text
        if ord(c) < 128  # ASCII
        or '\u4e00' <= c <= '\u9fff'  # 中文
        or '\u3040' <= c <= '\u30ff'  # 日文
        or c in '\\{}[]()^_$|&'  # LaTeX 符号
    )
    # 如果有效字符占比低于 70%，认为是乱码
    return (valid_count / max(total, 1)) < 0.7


def filter_by_length(sample, min_q_len=10, max_q_len=2000):
    """
    规则 1: 过滤问题长度异常的样本
    - 问题文本过短 (<10 字符) 可能是不完整的题目
    - 问题文本过长 (>2000 字符) 可能包含冗余信息
    """
    q_len = len(sample["question"])
    return min_q_len <= q_len <= max_q_len


def filter_by_answer_format(sample):
    """
    规则 2: 确保答案存在且非空。
    对于 solution 字段，优先检查是否包含 \\boxed{}。
    如果 solution 中没有 boxed，但 answer 字段有值，仍然保留。
    """
    # 至少要有一个有效答案来源
    has_answer = bool(sample.get("answer", "").strip())
    has_boxed_in_solution = (
        bool(sample.get("solution", ""))
        and extract_boxed_answer(sample["solution"]) is not None
    )
    return has_answer or has_boxed_in_solution


def filter_garbled(sample):
    """
    规则 3: 移除包含乱码的样本
    """
    return not is_garbled(sample["question"])


def compute_hash(text):
    """计算文本的 MD5 哈希值，用于去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def filter_and_deduplicate(samples):
    """
    主过滤流水线：
    1. 长度过滤
    2. 答案格式检查
    3. 乱码检测
    4. 基于问题文本 hash 去重
    """
    stats = {
        "total_before": len(samples),
        "removed_by_length": 0,
        "removed_by_answer": 0,
        "removed_by_garbled": 0,
        "removed_by_dedup": 0,
    }

    # 步骤 1: 长度过滤
    filtered = []
    for s in samples:
        if filter_by_length(s):
            filtered.append(s)
        else:
            stats["removed_by_length"] += 1
    logger.info(f"长度过滤后: {len(filtered)} 条 (移除 {stats['removed_by_length']} 条)")

    # 步骤 2: 答案格式过滤
    temp = []
    for s in filtered:
        if filter_by_answer_format(s):
            temp.append(s)
        else:
            stats["removed_by_answer"] += 1
    filtered = temp
    logger.info(f"答案格式过滤后: {len(filtered)} 条 (移除 {stats['removed_by_answer']} 条)")

    # 步骤 3: 乱码过滤
    temp = []
    for s in filtered:
        if filter_garbled(s):
            temp.append(s)
        else:
            stats["removed_by_garbled"] += 1
    filtered = temp
    logger.info(f"乱码过滤后: {len(filtered)} 条 (移除 {stats['removed_by_garbled']} 条)")

    # 步骤 4: 基于问题文本 hash 去重
    seen_hashes = set()
    deduped = []
    for s in filtered:
        h = compute_hash(s["question"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(s)
        else:
            stats["removed_by_dedup"] += 1
    logger.info(f"去重后: {len(deduped)} 条 (移除 {stats['removed_by_dedup']} 条)")

    stats["total_after"] = len(deduped)
    return deduped, stats


# ============================================================
# 4. 格式转换
# ============================================================
def format_solution_with_boxed(sample):
    """
    将样本转换为带 \\boxed{} 格式的规范 solution。
    如果原始 solution 已包含 \\boxed{}，则保持原样。
    否则在 solution 末尾追加 \\boxed{answer}。
    """
    solution = sample.get("solution", "")
    answer = sample.get("answer", "")

    # 如果 solution 已含 boxed，直接使用
    if solution and extract_boxed_answer(solution) is not None:
        return solution

    # 如果 solution 存在但没有 boxed，追加
    if solution and answer:
        return f"{solution}\n\nThe answer is \\boxed{{{answer}}}"

    # 只有 answer 没有 solution
    if answer:
        return f"The answer is \\boxed{{{answer}}}"

    return solution


def convert_to_alpaca_format(samples):
    """
    将筛选后的数据转换为 LLaMA-Factory 的 alpaca 格式:
    {
        "instruction": "...",
        "input": "<math problem>",
        "output": "<step-by-step solution with \\boxed{answer}>"
    }
    """
    alpaca_data = []
    for s in samples:
        formatted_solution = format_solution_with_boxed(s)
        if not formatted_solution:
            continue

        alpaca_data.append({
            "instruction": "Solve the following math problem step by step. Put your final answer in \\boxed{}.",
            "input": s["question"],
            "output": formatted_solution,
        })
    return alpaca_data


def convert_to_rl_prompts(samples):
    """
    为 GRPO 强化学习准备 prompt 数据。
    每条数据包含：问题 prompt + 参考答案（用于奖励计算）。
    """
    rl_data = []
    for s in samples:
        # 提取参考答案（优先从 solution 的 boxed 中提取）
        ref_answer = None
        if s.get("solution"):
            ref_answer = extract_boxed_answer(s["solution"])
        if not ref_answer and s.get("answer"):
            ref_answer = normalize_answer(s["answer"])

        if not ref_answer:
            continue

        rl_data.append({
            "prompt": f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n{s['question']}",
            "reference_answer": ref_answer,
        })
    return rl_data


# ============================================================
# 5. 数据集划分
# ============================================================
def split_dataset(data, eval_ratio=0.05, seed=42):
    """
    将数据划分为训练集和验证集。
    eval_ratio: 验证集占比 (默认 5%)
    """
    import random
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    eval_size = max(1, int(len(data) * eval_ratio))
    eval_indices = set(indices[:eval_size])

    train_data = [data[i] for i in range(len(data)) if i not in eval_indices]
    eval_data = [data[i] for i in eval_indices]

    return train_data, eval_data


# ============================================================
# 6. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Stage A: 数据筛选")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制每个数据集的最大样本数（调试用）")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="验证集占比")
    args = parser.parse_args()

    # 创建目录
    ensure_dirs()

    # ---- 步骤 1: 下载数据集 ----
    logger.info("=" * 60)
    logger.info("Stage A - 数据筛选开始")
    logger.info("=" * 60)

    deepmath, bigmath = download_datasets()

    # ---- 步骤 2: 统一格式 ----
    all_samples = []
    if deepmath is not None:
        dm_samples = normalize_deepmath(deepmath)
        if args.max_samples:
            dm_samples = dm_samples[:args.max_samples]
        all_samples.extend(dm_samples)

    if bigmath is not None:
        bm_samples = normalize_bigmath(bigmath)
        if args.max_samples:
            bm_samples = bm_samples[:args.max_samples]
        all_samples.extend(bm_samples)

    if not all_samples:
        logger.error("没有成功加载任何数据集，请检查网络连接。")
        return

    logger.info(f"合并后总数据量: {len(all_samples)} 条")

    # ---- 步骤 3: 过滤和去重 ----
    filtered_samples, filter_stats = filter_and_deduplicate(all_samples)

    # ---- 步骤 4: 转换为 alpaca 格式 (SFT 用) ----
    alpaca_data = convert_to_alpaca_format(filtered_samples)
    logger.info(f"转换为 alpaca 格式: {len(alpaca_data)} 条")

    # ---- 步骤 5: 划分训练集和验证集 ----
    train_data, eval_data = split_dataset(alpaca_data, eval_ratio=args.eval_ratio)
    logger.info(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # ---- 步骤 6: 转换为 RL prompts ----
    rl_prompts = convert_to_rl_prompts(filtered_samples)
    logger.info(f"RL prompts: {len(rl_prompts)} 条")

    # ---- 步骤 7: 保存数据 ----
    # 保存 SFT 训练集
    sft_train_path = DATA_FILTERED / "sft_train.json"
    with open(sft_train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"SFT 训练集已保存至: {sft_train_path}")

    # 保存 SFT 验证集
    sft_eval_path = DATA_FILTERED / "sft_eval.json"
    with open(sft_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    logger.info(f"SFT 验证集已保存至: {sft_eval_path}")

    # 保存 RL prompts
    rl_path = DATA_RL / "rl_train.json"
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl_prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"RL prompts 已保存至: {rl_path}")

    # ---- 步骤 8: 统计信息 ----
    # 计算问题长度分布
    q_lengths = [len(s["question"]) for s in filtered_samples]
    source_counts = Counter(s["source"] for s in filtered_samples)

    stats = {
        **filter_stats,
        "source_distribution": dict(source_counts),
        "question_length": {
            "min": min(q_lengths) if q_lengths else 0,
            "max": max(q_lengths) if q_lengths else 0,
            "mean": round(sum(q_lengths) / max(len(q_lengths), 1), 1),
        },
        "sft_train_size": len(train_data),
        "sft_eval_size": len(eval_data),
        "rl_prompts_size": len(rl_prompts),
    }

    stats_path = DATA_FILTERED / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存至: {stats_path}")

    # 打印统计摘要
    logger.info("=" * 60)
    logger.info("数据筛选统计摘要:")
    logger.info(f"  原始数据量: {stats['total_before']}")
    logger.info(f"  长度过滤移除: {stats['removed_by_length']}")
    logger.info(f"  答案格式移除: {stats['removed_by_answer']}")
    logger.info(f"  乱码移除: {stats['removed_by_garbled']}")
    logger.info(f"  去重移除: {stats['removed_by_dedup']}")
    logger.info(f"  最终数据量: {stats['total_after']}")
    logger.info(f"  数据来源分布: {dict(source_counts)}")
    logger.info("=" * 60)
    logger.info("Stage A 完成！")


if __name__ == "__main__":
    main()
