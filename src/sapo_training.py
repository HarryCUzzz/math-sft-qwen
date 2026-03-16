"""
Stage C+ - SAPO reinforcement learning training
==============================================

This module adds SAPO (Soft Adaptive Policy Optimization) as a parallel
post-training option alongside the existing GRPO pipeline. It intentionally
does not modify the original GRPO implementation.

Design goals:
1. Reuse the current project's RL data format and reward logic.
2. Keep model loading and SFT-adapter merging aligned with grpo_training.py.
3. Change only the policy-update rule by subclassing GRPOTrainer.
"""

import json
import logging
import re
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RL = PROJECT_ROOT / "data" / "rl_prompts"
SFT_MODEL_DIR = PROJECT_ROOT / "outputs" / "sft_model"
SAPO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sapo_model"
SAPO_LOG_DIR = PROJECT_ROOT / "outputs" / "sapo_logs"


SAPO_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "sft_adapter_path": str(SFT_MODEL_DIR),
    "learning_rate": 5e-7,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_generations": 8,
    "max_prompt_length": 512,
    "max_completion_length": 1024,
    "beta": 0.04,
    "temperature": 0.7,
    "logging_steps": 5,
    "save_steps": 100,
    "max_steps": 500,
    "seed": 42,
    "tau_pos": 1.0,
    "tau_neg": 1.5,
}


def extract_boxed_answer(text):
    if not text:
        return None

    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1
    idx = start
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1

    if depth == 0:
        return text[start : idx - 1].strip()
    return None


def normalize_answer(answer_str):
    if not answer_str:
        return ""
    ans = answer_str.strip().replace("$", "").replace(" ", "")
    return ans.rstrip(".")


def answers_match(predicted, reference):
    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    return bool(pred and ref and pred == ref)


def correctness_reward_fn(completions, reference_answer, **kwargs):
    rewards = []
    for completion, ref_answer in zip(completions, reference_answer):
        predicted = extract_boxed_answer(completion)
        rewards.append(1.0 if predicted and answers_match(predicted, ref_answer) else 0.0)
    return rewards


def format_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        if r"\boxed{" in completion:
            reward += 0.5

        has_step_keyword = bool(
            re.search(
                r"(?i)(step\s*\d|first|then|next|therefore|thus|hence|so,|finally)",
                completion,
            )
        )
        lines = [line.strip() for line in completion.split("\n") if line.strip()]
        if has_step_keyword or len(lines) >= 3:
            reward += 0.3

        if 50 <= len(completion) <= 1500:
            reward += 0.2

        rewards.append(reward)
    return rewards


def combined_reward_fn(completions, **kwargs):
    references = kwargs.get("reference_answer", [])
    corr_rewards = correctness_reward_fn(completions, references)
    fmt_rewards = format_reward_fn(completions)
    return [corr + fmt for corr, fmt in zip(corr_rewards, fmt_rewards)]


def load_rl_dataset():
    rl_data_path = DATA_RL / "rl_train.json"
    if not rl_data_path.exists():
        raise FileNotFoundError(
            f"RL training data not found: {rl_data_path}\n"
            "Run src/data_selection.py first to prepare rl_train.json."
        )

    with open(rl_data_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    logger.info("Loaded %s RL samples", len(data))
    return data


def prepare_dataset_for_rl(raw_data, tokenizer, max_samples=None):
    from datasets import Dataset

    if max_samples:
        raw_data = raw_data[:max_samples]

    processed = []
    for item in raw_data:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful math assistant. Solve problems step by step "
                    "and put your final answer in \\boxed{}."
                ),
            },
            {"role": "user", "content": item["prompt"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processed.append(
            {
                "prompt": formatted_prompt,
                "reference_answer": item["reference_answer"],
            }
        )

    dataset = Dataset.from_list(processed)
    logger.info("Prepared %s samples for SAPO", len(dataset))
    return dataset


class SAPOMetricsCallback:
    """
    A light callback that records trainer logs to a JSON file after training.
    """

    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step}
            entry.update(logs)
            self.history.append(entry)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "sapo_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)
        logger.info("Saved SAPO metrics to %s", metrics_path)


def run_sapo_training():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    class SAPOTrainer(GRPOTrainer):
        """
        SAPO replaces GRPO's clipped ratio objective with a soft adaptive gate:
        f(r) = sigmoid(tau * (r - 1)) * 4 / tau
        """

        def __init__(self, tau_pos=1.0, tau_neg=1.5, **kwargs):
            super().__init__(**kwargs)
            self.tau_pos = tau_pos
            self.tau_neg = tau_neg

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if return_outputs:
                raise ValueError("return_outputs=True is not supported in SAPOTrainer.")

            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]

            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.shape[1]

            per_token_logps = self._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
            )

            old_per_token_logps = inputs["old_per_token_logps"]
            ratio = torch.exp(per_token_logps - old_per_token_logps)

            advantages = inputs["advantages"]
            tau = torch.where(
                advantages > 0,
                torch.full_like(ratio, self.tau_pos),
                torch.full_like(ratio, self.tau_neg),
            )
            soft_gate = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)

            per_token_loss = -soft_gate * advantages
            loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1).clamp_min(1)
            ).mean()

            if self.beta > 0.0:
                ref_per_token_logps = inputs.get("ref_per_token_logps")
                if ref_per_token_logps is not None:
                    per_token_kl = torch.clamp(
                        per_token_logps.detach() - ref_per_token_logps,
                        min=0,
                    )
                    kl_penalty = (
                        (per_token_kl * completion_mask).sum(dim=1)
                        / completion_mask.sum(dim=1).clamp_min(1)
                    )
                    loss = loss + self.beta * kl_penalty.mean()

            with torch.no_grad():
                if not hasattr(self, "_metrics") or self._metrics is None:
                    self._metrics = {}
                mean_ratio = (
                    (ratio * completion_mask).sum(dim=1)
                    / completion_mask.sum(dim=1).clamp_min(1)
                ).mean()
                mean_gate = (
                    (soft_gate * completion_mask).sum(dim=1)
                    / completion_mask.sum(dim=1).clamp_min(1)
                ).mean()
                self._metrics["loss/sapo_policy"] = loss.detach().item()
                self._metrics["val/ratio_mean"] = mean_ratio.item()
                self._metrics["val/soft_gate_mean"] = mean_gate.item()
                self._metrics["completion_length"] = (
                    completion_mask.sum(dim=1).float().mean().item()
                )

            return loss

    class CallbackAdapter(TrainerCallback):
        def __init__(self, sink):
            self.sink = sink

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.sink.on_log(args, state, control, logs=logs, **kwargs)

    logger.info("Loading tokenizer: %s", SAPO_CONFIG["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        SAPO_CONFIG["model_name"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", SAPO_CONFIG["model_name"])
    base_model = AutoModelForCausalLM.from_pretrained(
        SAPO_CONFIG["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    sft_adapter_path = Path(SAPO_CONFIG["sft_adapter_path"])
    if sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists():
        logger.info("Loading SFT adapter: %s", sft_adapter_path)
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_path))
        model = model.merge_and_unload()
    else:
        logger.warning("SFT adapter not found. SAPO will run on the base model.")
        model = base_model

    raw_data = load_rl_dataset()
    train_dataset = prepare_dataset_for_rl(raw_data, tokenizer)

    SAPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAPO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(SAPO_OUTPUT_DIR),
        logging_dir=str(SAPO_LOG_DIR),
        learning_rate=SAPO_CONFIG["learning_rate"],
        num_train_epochs=SAPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=SAPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=SAPO_CONFIG["gradient_accumulation_steps"],
        num_generations=SAPO_CONFIG["num_generations"],
        max_prompt_length=SAPO_CONFIG["max_prompt_length"],
        max_completion_length=SAPO_CONFIG["max_completion_length"],
        beta=SAPO_CONFIG["beta"],
        logging_steps=SAPO_CONFIG["logging_steps"],
        save_steps=SAPO_CONFIG["save_steps"],
        max_steps=SAPO_CONFIG["max_steps"],
        seed=SAPO_CONFIG["seed"],
        bf16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    callback_sink = SAPOMetricsCallback()

    logger.info("=" * 60)
    logger.info("Starting SAPO training")
    logger.info("  tau_pos: %s", SAPO_CONFIG["tau_pos"])
    logger.info("  tau_neg: %s", SAPO_CONFIG["tau_neg"])
    logger.info("  num_generations: %s", SAPO_CONFIG["num_generations"])
    logger.info("  beta: %s", SAPO_CONFIG["beta"])
    logger.info("=" * 60)

    trainer = SAPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=combined_reward_fn,
        tokenizer=tokenizer,
        tau_pos=SAPO_CONFIG["tau_pos"],
        tau_neg=SAPO_CONFIG["tau_neg"],
        callbacks=[CallbackAdapter(callback_sink)],
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    callback_sink.save(SAPO_OUTPUT_DIR)

    logger.info("SAPO training complete. Model saved to %s", SAPO_OUTPUT_DIR)
    logger.info("Training metrics: %s", metrics)
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage C+: SAPO reinforcement learning")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Override the number of sampled completions per prompt.",
    )
    parser.add_argument("--tau_pos", type=float, default=None, help="Positive-advantage SAPO tau.")
    parser.add_argument("--tau_neg", type=float, default=None, help="Negative-advantage SAPO tau.")
    args = parser.parse_args()

    if args.max_steps is not None:
        SAPO_CONFIG["max_steps"] = args.max_steps
    if args.num_generations is not None:
        SAPO_CONFIG["num_generations"] = args.num_generations
    if args.tau_pos is not None:
        SAPO_CONFIG["tau_pos"] = args.tau_pos
    if args.tau_neg is not None:
        SAPO_CONFIG["tau_neg"] = args.tau_neg

    logger.info("=" * 60)
    logger.info("Stage C+ - SAPO training start")
    logger.info("=" * 60)
    run_sapo_training()


if __name__ == "__main__":
    main()
