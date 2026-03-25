"""Stage A for src_qwen35: build cleaned SFT/RL datasets without touching legacy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset, load_from_disk

from config import (
    DATA_EVAL,
    DATA_QWEN35_MANIFESTS,
    DATA_QWEN35_PROCESSED,
    DATA_QWEN35_SMOKE,
    RL_TRAIN_PATH,
    SFT_EVAL_PATH,
    SFT_TRAIN_PATH,
    SMOKE_OVERFIT_PATH,
    SMOKE_SFT_PATH,
    THINKING_SYSTEM_PROMPT,
)
from answer_utils import extract_boxed_answer, extract_reference_answer, normalize_answer, strip_think_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIGMATH_LOCAL_PARQUET = PROJECT_ROOT / "data" / "Big-Math-RL-Verified" / "data" / "train-00000-of-00001.parquet"
DEEPMATH_LOCAL_PATH = Path(
    __import__("os").environ.get("DEEPMATH_LOCAL_PATH", PROJECT_ROOT / "data" / "DeepMath-103K")
)

MIN_QUESTION_CHARS = 10
MAX_QUESTION_CHARS = 3500


def ensure_dirs() -> None:
    for directory in (DATA_QWEN35_PROCESSED, DATA_QWEN35_SMOKE, DATA_QWEN35_MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)


def _load_eval_questions() -> set[str]:
    normalized = set()
    for dataset_key in ("math500", "gsm8k", "theoremqa"):
        local_dir = DATA_EVAL / dataset_key
        try:
            if local_dir.exists():
                dataset = load_from_disk(str(local_dir))
            elif dataset_key == "math500":
                dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
            elif dataset_key == "gsm8k":
                dataset = load_dataset("openai/gsm8k", "main", split="test")
            else:
                dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")
        except Exception:
            continue

        if hasattr(dataset, "keys") and not isinstance(dataset, Dataset):
            split = "test" if "test" in dataset else list(dataset.keys())[0]
            dataset = dataset[split]

        question_key = {
            "math500": "problem",
            "gsm8k": "question",
            "theoremqa": "Question",
        }[dataset_key]
        for item in dataset:
            question = str(item.get(question_key, "")).strip()
            if question:
                normalized.add(_normalize_question(question))
    return normalized


def _load_deepmath() -> Dataset:
    if DEEPMATH_LOCAL_PATH.exists():
        if DEEPMATH_LOCAL_PATH.is_dir() and (DEEPMATH_LOCAL_PATH / "dataset_info.json").exists():
            return load_from_disk(str(DEEPMATH_LOCAL_PATH))
        if DEEPMATH_LOCAL_PATH.suffix in {".json", ".jsonl", ".parquet"}:
            return load_dataset(DEEPMATH_LOCAL_PATH.suffix.lstrip("."), data_files=str(DEEPMATH_LOCAL_PATH), split="train")
    return load_dataset("zwhe99/DeepMath-103K", split="train")


def _load_bigmath() -> Dataset:
    if BIGMATH_LOCAL_PARQUET.exists():
        return load_dataset("parquet", data_files=str(BIGMATH_LOCAL_PARQUET), split="train")
    return load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower()


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_garbled(text: str) -> bool:
    if not text:
        return True
    valid_count = sum(
        1
        for char in text
        if ord(char) < 128 or "\u4e00" <= char <= "\u9fff" or char in "\\{}[]()^_$|&<>"
    )
    return (valid_count / max(len(text), 1)) < 0.7


def _pick_first(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _guess_domain(item: dict, question: str) -> str:
    metadata = " ".join(str(item.get(key, "")) for key in ("domain", "topic", "category", "source"))
    text = f"{metadata} {question}".lower()
    keywords = {
        "algebra": ["equation", "polynomial", "factor", "algebra"],
        "geometry": ["triangle", "circle", "angle", "geometry"],
        "number_theory": ["prime", "integer", "mod", "divisible"],
        "calculus": ["derivative", "integral", "limit"],
        "probability": ["probability", "random", "expected"],
    }
    for domain, domain_keywords in keywords.items():
        if any(keyword in text for keyword in domain_keywords):
            return domain
    return "general"


def _guess_difficulty(item: dict) -> str:
    for key in ("difficulty", "level"):
        if item.get(key):
            return str(item[key]).strip().lower()
    solve_rate = item.get("llama8b_solve_rate")
    if solve_rate is None:
        return "unknown"
    try:
        value = float(solve_rate)
    except (TypeError, ValueError):
        return "unknown"
    if value < 0.2:
        return "hard"
    if value < 0.6:
        return "medium"
    return "easy"


def _build_assistant_text(solution: str, final_answer: str) -> str:
    cleaned_solution = strip_think_tags(solution).strip()
    final_line = f"Final answer: \\boxed{{{final_answer}}}"
    if cleaned_solution:
        return f"<think>\n{cleaned_solution}\n</think>\n\n{final_line}"
    return final_line


def _sample_from_records(records: Sequence[dict], quota: int, seed: int) -> List[dict]:
    if quota <= 0 or not records:
        return []
    if len(records) <= quota:
        return list(records)
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[:quota]


def _drop_overlaps(records: Sequence[dict], eval_questions: set[str]) -> List[dict]:
    filtered = []
    for record in records:
        if record["normalized_question"] in eval_questions:
            continue
        filtered.append(record)
    return filtered


def _normalize_deepmath(dataset: Dataset) -> List[dict]:
    records = []
    for item in dataset:
        question = _pick_first(item, "question", "problem", "input")
        answer = _pick_first(item, "answer", "final_answer")
        solution = _pick_first(item, "solution", "output", "response")
        if not question or not (answer or solution):
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "solution": solution,
                "source_dataset": "DeepMath-103K",
                "raw_source": _pick_first(item, "source", "dataset", "category"),
                "difficulty": _guess_difficulty(item),
                "domain": _guess_domain(item, question),
            }
        )
    return records


def _normalize_bigmath(dataset: Dataset) -> List[dict]:
    records = []
    for item in dataset:
        question = _pick_first(item, "problem", "question", "input")
        answer = _pick_first(item, "answer", "expected_answer", "final_answer")
        solution = _pick_first(item, "solution", "output", "response")
        raw_source = _pick_first(item, "source", "dataset", "origin")
        if "gsm8k" in raw_source.lower():
            continue
        if not question or not (answer or solution):
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "solution": solution,
                "source_dataset": "Big-Math-RL-Verified",
                "raw_source": raw_source,
                "difficulty": _guess_difficulty(item),
                "domain": _guess_domain(item, question),
            }
        )
    return records


def _clean_records(records: Sequence[dict], eval_questions: set[str]) -> Tuple[List[dict], Dict[str, int]]:
    stats = Counter(total_before=len(records))
    deduped = []
    seen_hashes = set()
    for record in records:
        question = record["question"].strip()
        if len(question) < MIN_QUESTION_CHARS or len(question) > MAX_QUESTION_CHARS:
            stats["removed_by_length"] += 1
            continue
        if _is_garbled(question):
            stats["removed_by_garbled"] += 1
            continue

        final_answer = extract_reference_answer(record.get("solution", "")) or normalize_answer(record.get("answer", ""))
        final_answer = normalize_answer(final_answer)
        if not final_answer:
            stats["removed_by_answer"] += 1
            continue

        normalized_question = _normalize_question(question)
        question_hash = _hash_text(normalized_question)
        if question_hash in seen_hashes:
            stats["removed_by_dedup"] += 1
            continue
        seen_hashes.add(question_hash)

        cleaned_solution = record.get("solution", "").strip()
        assistant_text = _build_assistant_text(cleaned_solution, final_answer)
        overlap = normalized_question in eval_questions
        deduped.append(
            {
                "question": question,
                "normalized_question": normalized_question,
                "final_answer": final_answer,
                "assistant_text": assistant_text,
                "source_dataset": record["source_dataset"],
                "raw_source": record.get("raw_source", ""),
                "difficulty": record.get("difficulty", "unknown"),
                "domain": record.get("domain", "general"),
                "benchmark_overlap_flag": overlap,
                "reasoning_style": "cot",
                "parser_type": "gsm8k" if "gsm8k" in record.get("raw_source", "").lower() else "math",
            }
        )
    stats["total_after"] = len(deduped)
    return deduped, dict(stats)


def _to_sft_record(record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["assistant_text"]},
        ],
        "final_answer": record["final_answer"],
        "source_dataset": record["source_dataset"],
        "difficulty": record["difficulty"],
        "domain": record["domain"],
        "benchmark_overlap_flag": record["benchmark_overlap_flag"],
        "reasoning_style": record["reasoning_style"],
        "parser_type": record["parser_type"],
    }


def _to_rl_record(record: dict) -> dict:
    return {
        "prompt": record["question"],
        "reference_answer": record["final_answer"],
        "source_dataset": record["source_dataset"],
        "difficulty": record["difficulty"],
        "domain": record["domain"],
        "benchmark_overlap_flag": record["benchmark_overlap_flag"],
        "reasoning_style": record["reasoning_style"],
        "parser_type": record["parser_type"],
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_datasets(seed: int, deepmath_quota: int, bigmath_sft_quota: int, bigmath_rl_quota: int, eval_ratio: float) -> Dict[str, int]:
    ensure_dirs()
    eval_questions = _load_eval_questions()

    logger.info("Loading DeepMath-103K...")
    deepmath_records = _normalize_deepmath(_load_deepmath())
    logger.info("Loading Big-Math-RL-Verified...")
    bigmath_records = _normalize_bigmath(_load_bigmath())

    deepmath_records, deepmath_stats = _clean_records(deepmath_records, eval_questions)
    bigmath_records, bigmath_stats = _clean_records(bigmath_records, eval_questions)

    deepmath_records = _drop_overlaps(deepmath_records, eval_questions)
    bigmath_records = _drop_overlaps(bigmath_records, eval_questions)

    deepmath_selected = _sample_from_records(deepmath_records, deepmath_quota, seed)
    bigmath_sft_selected = _sample_from_records(bigmath_records, bigmath_sft_quota, seed + 1)
    bigmath_rl_selected = _sample_from_records(bigmath_records, bigmath_rl_quota, seed + 2)

    sft_records = [_to_sft_record(record) for record in deepmath_selected + bigmath_sft_selected]
    rng = random.Random(seed)
    rng.shuffle(sft_records)
    eval_size = max(1, int(len(sft_records) * eval_ratio))
    sft_eval_records = sft_records[:eval_size]
    sft_train_records = sft_records[eval_size:]
    rl_records = [_to_rl_record(record) for record in bigmath_rl_selected]

    overfit_records = sft_train_records[:50]
    pilot_records = sft_train_records[:200]

    stats = {
        "seed": seed,
        "deepmath_quota": deepmath_quota,
        "bigmath_sft_quota": bigmath_sft_quota,
        "bigmath_rl_quota": bigmath_rl_quota,
        "deepmath_cleaned": len(deepmath_records),
        "bigmath_cleaned": len(bigmath_records),
        "sft_train_size": _write_jsonl(SFT_TRAIN_PATH, sft_train_records),
        "sft_eval_size": _write_jsonl(SFT_EVAL_PATH, sft_eval_records),
        "rl_train_size": _write_jsonl(RL_TRAIN_PATH, rl_records),
        "smoke_overfit_size": _write_jsonl(SMOKE_OVERFIT_PATH, overfit_records),
        "smoke_sft_size": _write_jsonl(SMOKE_SFT_PATH, pilot_records),
        "deepmath_stats": deepmath_stats,
        "bigmath_stats": bigmath_stats,
    }

    manifest_path = DATA_QWEN35_MANIFESTS / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned Qwen3.5 math datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepmath-quota", type=int, default=24000)
    parser.add_argument("--bigmath-sft-quota", type=int, default=8000)
    parser.add_argument("--bigmath-rl-quota", type=int, default=8000)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()

    stats = build_datasets(
        seed=args.seed,
        deepmath_quota=args.deepmath_quota,
        bigmath_sft_quota=args.bigmath_sft_quota,
        bigmath_rl_quota=args.bigmath_rl_quota,
        eval_ratio=args.eval_ratio,
    )
    logger.info("Prepared cleaned qwen35 datasets: %s", stats)


if __name__ == "__main__":
    main()
