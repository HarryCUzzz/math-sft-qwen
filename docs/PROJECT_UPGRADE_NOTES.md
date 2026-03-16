# Project Upgrade Notes

## Why This Document Exists

This project already includes a full math reasoning post-training pipeline:

- data selection
- LoRA SFT
- GRPO
- benchmark evaluation
- analysis

The next step is not to add more frameworks, but to make the project easier to defend in interviews and more convincing as an experiment-driven project.

## Current High-Value Upgrade Directions

### 1. Separate the contribution of data quality and RL

The most important missing comparison is:

- Base
- Raw-data SFT
- Filtered-data SFT
- Filtered-data SFT + GRPO

Without this ablation, it is hard to answer whether the gain mainly comes from:

- cleaner data
- better supervised warm-start
- or RL itself

### 2. Track failure modes, not only final accuracy

For math reasoning, exact-match accuracy is necessary but not sufficient.

Recommended tracking dimensions:

- boxed answer hit rate
- boxed but wrong rate
- invalid answer extraction rate
- average completion length
- short-question vs long-question accuracy
- representative failure examples

The analysis script has been extended in this direction so that the project can produce:

- richer failure statistics
- failure case dumps
- a more interview-ready report

### 3. Reward design should stay correctness-first

The current reward design uses:

- correctness reward
- format reward

This is a good starting point, but the key principle should remain:

> format reward is only a weak regularizer, not the primary optimization target.

Otherwise the model may learn to output `\boxed{}` reliably without actually improving reasoning quality.

### 4. Recommended next experiments

#### Experiment A: group size sensitivity

- `num_generations = 4 / 8 / 12`

Question:

- does a larger comparison group improve relative optimization?
- or does it mainly increase variance and cost?

#### Experiment B: beta sensitivity

- `beta = 0.01 / 0.04 / 0.1`

Question:

- what is the tradeoff between exploration and KL stability?
- when does the model become too conservative?

#### Experiment C: reward ablation

- correctness only
- correctness + weak format reward
- correctness + stronger format reward

Question:

- how much does format shaping help?
- when does it start to create reward hacking?

## Suggested One-Sentence Project Positioning

This project is best presented as:

> A small-model math reasoning post-training study that investigates how data filtering and GRPO reward design affect reasoning accuracy, output format stability, and failure modes.
