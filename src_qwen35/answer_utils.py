"""Shared answer parsing and equivalence helpers for Qwen3.5 math post-training."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Optional

try:
    import sympy as sp
except Exception:  # pragma: no cover - optional dependency at runtime
    sp = None

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{")
FRACTION_TOKEN_RE = re.compile(r"[-+]?\\frac\{[^{}]+\}\{[^{}]+\}|[-+]?\d+\s*/\s*[-+]?\d+")
NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
FINAL_PATTERNS = [
    re.compile(r"(?:Final answer|final answer|Answer|answer)\s*[:?]\s*([^\n]+)"),
    re.compile(r"(?:The answer is|the answer is)\s*[:?]?\s*([^\n]+)"),
    re.compile(r"(?:Therefore|therefore|Thus|thus|Hence|hence)\s*[:,]?\s*([^\n]+)"),
]


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return THINK_RE.sub("", text).strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None

    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None

    match = matches[-1]
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1

    if depth == 0:
        return text[start:index - 1].strip()
    return None


def extract_reference_answer(reference: str, dataset_name: str = "default") -> str:
    if reference is None:
        return ""
    text = str(reference).strip()
    if dataset_name == "gsm8k" and "####" in text:
        text = text.split("####")[-1].strip()
    boxed = extract_boxed_answer(text)
    return boxed or text


def extract_candidate_answer(text: str) -> Optional[str]:
    if not text:
        return None

    clean_text = strip_think_tags(text)
    boxed = extract_boxed_answer(clean_text)
    if boxed:
        return boxed

    for pattern in FINAL_PATTERNS:
        match = pattern.search(clean_text)
        if match:
            return match.group(1).strip().rstrip(".")

    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    for line in reversed(lines):
        fraction_matches = FRACTION_TOKEN_RE.findall(line)
        if fraction_matches:
            return fraction_matches[-1].replace(" ", "")
        number_matches = NUMBER_TOKEN_RE.findall(line)
        if number_matches:
            return number_matches[-1]
    return None


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    normalized = str(text).strip()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    if normalized.startswith("{") and normalized.endswith("}"):
        normalized = normalized[1:-1]
    return normalized


def _coerce_numeric(text: str) -> Optional[float]:
    if not text:
        return None

    normalized = normalize_answer(text)
    try:
        return float(normalized)
    except ValueError:
        pass

    fraction_match = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", normalized)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            return float(Fraction(numerator, denominator))

    latex_fraction = re.fullmatch(r"([-+]?)\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", normalized)
    if latex_fraction:
        sign = -1.0 if latex_fraction.group(1) == "-" else 1.0
        numerator = int(latex_fraction.group(2))
        denominator = int(latex_fraction.group(3))
        if denominator != 0:
            return sign * float(Fraction(numerator, denominator))

    return None


def _sympy_equivalent(left: str, right: str) -> bool:
    if sp is None:
        return False

    try:
        left_expr = sp.sympify(_latex_fraction_to_plain(normalize_answer(left)))
        right_expr = sp.sympify(_latex_fraction_to_plain(normalize_answer(right)))
        difference = sp.simplify(left_expr - right_expr)
        return difference == 0
    except Exception:
        return False


def _latex_fraction_to_plain(text: str) -> str:
    return re.sub(r"([-+]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1(\2)/(\3)", text)


def answers_equivalent(predicted_text: str, reference_text: str, dataset_name: str = "default") -> bool:
    candidate = extract_candidate_answer(predicted_text)
    if candidate is None:
        candidate = predicted_text

    predicted = normalize_answer(candidate)
    reference = normalize_answer(extract_reference_answer(reference_text, dataset_name))
    if not predicted or not reference:
        return False
    if predicted == reference:
        return True

    predicted_num = _coerce_numeric(predicted)
    reference_num = _coerce_numeric(reference)
    if predicted_num is not None and reference_num is not None:
        return math.isclose(predicted_num, reference_num, rel_tol=1e-9, abs_tol=1e-9)

    return _sympy_equivalent(predicted, reference)


def has_valid_final_structure(text: str) -> bool:
    if not text:
        return False
    clean_text = strip_think_tags(text)
    return "Final answer:" in clean_text or "\\boxed{" in clean_text
