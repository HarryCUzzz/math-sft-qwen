# SAPO Integration Notes

This project can support SAPO because the existing GRPO pipeline already has the hard parts that SAPO needs:

- a post-SFT causal language model
- prompt-based rollout generation
- reward functions over generated completions
- `trl` GRPO-style batch inputs with old-policy log probabilities and advantages

## What Was Added

- `src/sapo_training.py`
  A parallel RL training entry point that subclasses `GRPOTrainer` and replaces the clipped GRPO objective with SAPO's soft adaptive gate.
- `src/evaluate_sapo.py`
  A standalone evaluator for the SAPO-trained model, so the original evaluation script does not need to be changed.

## Why This Is Feasible

SAPO is compatible with the current project structure because it changes the policy update rule, not the surrounding pipeline. The following pieces are reused as-is in spirit:

- the current `rl_train.json` data format
- the Qwen2.5-0.5B base model and merged SFT adapter flow
- the math reasoning prompt template
- the existing correctness and format rewards

## Outputs

- SAPO checkpoints: `outputs/sapo_model/`
- SAPO logs: `outputs/sapo_logs/`
- SAPO eval results: `outputs/eval_results/sapo_*.json`

## Run

```bash
python src/sapo_training.py
python src/evaluate_sapo.py --num_samples 100
```

Optional SAPO hyperparameter overrides:

```bash
python src/sapo_training.py --max_steps 200 --num_generations 4 --tau_pos 1.0 --tau_neg 1.5
```

## Assumptions

- `trl`, `transformers`, `datasets`, `peft`, and `torch` are installed in the environment.
- The installed `trl` version still exposes the same `GRPOTrainer` internals used by the custom SAPO loss override.
- `data/rl_prompts/rl_train.json` already exists.
