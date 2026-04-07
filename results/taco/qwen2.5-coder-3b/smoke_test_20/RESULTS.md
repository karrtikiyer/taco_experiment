# Smoke Test Results: 20 Problems

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-Coder-3B-Instruct |
| Temperature | 0.7 |
| top_p | 0.95 |
| num_samples | 10 per problem |
| max_new_tokens | 2048 |
| Problems | 20 (stratified across 5 difficulty levels) |
| Seed | 42 |
| Execution timeout | Per-test-case 4s (TACO default) with early_stop on first failure |

## Difficulty Distribution

| Difficulty | Count |
|---|---|
| EASY | 4 |
| MEDIUM | 4 |
| MEDIUM_HARD | 4 |
| HARD | 4 |
| VERY_HARD | 4 |

## pass@k Results (Execution-Based)

| Metric | Value |
|---|---|
| pass@1 | 0.0100 |
| pass@3 | 0.0267 |
| pass@5 | 0.0389 |
| pass@8 | 0.0489 |
| pass@10 | 0.0500 |

Only **1 out of 20 problems** had any correct generations (task_id=959, MEDIUM difficulty, 2/10 correct).
All other 19 problems had 0/10 correct across all difficulty levels.

### Per-Problem Breakdown

| task_id | Difficulty | Correct/10 | pass@1 |
|---|---|---|---|
| 959 | MEDIUM | 2/10 | 0.200 |
| 761 | MEDIUM_HARD | 0/10 | 0.000 |
| 878 | MEDIUM | 0/10 | 0.000 |
| 955 | MEDIUM | 0/10 | 0.000 |
| 530 | MEDIUM_HARD | 0/10 | 0.000 |
| 137 | EASY | 0/10 | 0.000 |
| 710 | MEDIUM_HARD | 0/10 | 0.000 |
| 335 | HARD | 0/10 | 0.000 |
| 125 | VERY_HARD | 0/10 | 0.000 |
| 29 | EASY | 0/10 | 0.000 |
| 46 | VERY_HARD | 0/10 | 0.000 |
| 362 | HARD | 0/10 | 0.000 |
| 133 | MEDIUM | 0/10 | 0.000 |
| 289 | VERY_HARD | 0/10 | 0.000 |
| 937 | EASY | 0/10 | 0.000 |
| 106 | MEDIUM_HARD | 0/10 | 0.000 |
| 290 | HARD | 0/10 | 0.000 |
| 820 | EASY | 0/10 | 0.000 |
| 50 | VERY_HARD | 0/10 | 0.000 |
| 166 | HARD | 0/10 | 0.000 |

## CodeBLEU Diversity Results

| Metric | Value | Interpretation |
|---|---|---|
| Quality vs GT | 0.2069 | Low similarity to ground truth solutions |
| Self-CodeBLEU | 0.4222 | Moderate inter-sample diversity (lower = more diverse) |
| GT Coverage | 0.4019 | ~40% of ground-truth solution space covered |

### Interpretation

- **Quality vs Ground Truth (0.207):** Generated code has low structural similarity to reference solutions. This aligns with the very low pass@k -- the model's solutions diverge significantly from correct approaches.
- **Self-CodeBLEU (0.422):** Temperature 0.7 produces moderately diverse samples. A score of ~0.42 means generated samples share less than half their structure, indicating reasonable diversity in generation strategy even when solutions are incorrect.
- **GT Coverage (0.402):** Despite low correctness, the 10 samples per problem collectively cover about 40% of the ground-truth solution space (measured by best-match CodeBLEU to each GT solution).

## Timing

| Phase | Duration |
|---|---|
| Generation (10 samples x 20 problems) | ~87 min (~4.4 min/problem) |
| Execution (with early_stop) | 98.5s (~4.9s/problem) |
| Diversity (CodeBLEU) | 28.7s |
| **Total (excl. generation)** | **~2.1 min** |

## Key Observations

1. **Very low pass@k**: The 3B model struggles with TACO problems at temperature 0.7. Only 1/20 problems (an MEDIUM-level one) saw any correct solutions, and even that had only 2/10.
2. **TACO is hard**: Even EASY problems got 0/10. TACO problems are competitive programming problems which are significantly harder than typical code generation benchmarks like HumanEval.
3. **Diversity is reasonable**: Self-CodeBLEU of 0.42 shows temperature 0.7 does produce structurally varied samples, even if they're all incorrect.
4. **early_stop optimization critical**: Enabling early_stop reduced execution from estimated 3+ hours to under 2 minutes for 20 problems. Essential for the 100-problem run.
5. **Memory stable**: RSS peaked at ~1003MB during execution and settled to ~380MB after GC. No memory issues.
