# Full Experiment Results: 100 Problems

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-Coder-3B-Instruct |
| Temperature | 0.7 |
| top_p | 0.95 |
| num_samples | 10 per problem |
| max_new_tokens | 2048 |
| Problems | 100 (stratified: 20 per difficulty level) |
| Seed | 42 |
| Execution | Per-test-case 4s timeout (TACO default), early_stop on first failure |

## pass@k Results

| Metric | Value |
|---|---|
| **pass@1** | **0.0100** |
| pass@3 | 0.0257 |
| pass@5 | 0.0375 |
| pass@8 | 0.0518 |
| **pass@10** | **0.0600** |

Only **6 out of 100 problems** had any correct generations.

### pass@1 by Difficulty

| Difficulty | pass@1 | Problems Solved |
|---|---|---|
| EASY | 0.0250 | 4 / 20 |
| MEDIUM | 0.0200 | 1 / 20 |
| MEDIUM_HARD | 0.0050 | 1 / 20 |
| HARD | 0.0000 | 0 / 20 |
| VERY_HARD | 0.0000 | 0 / 20 |

### Problems with Correct Solutions

| task_id | Difficulty | Correct / 10 | pass@1 |
|---|---|---|---|
| 770 | MEDIUM | 4/10 | 0.400 |
| 29 | EASY | 2/10 | 0.200 |
| 937 | EASY | 1/10 | 0.100 |
| 132 | EASY | 1/10 | 0.100 |
| 285 | EASY | 1/10 | 0.100 |
| 483 | MEDIUM_HARD | 1/10 | 0.100 |

## CodeBLEU Diversity Results

| Metric | Value | Interpretation |
|---|---|---|
| Quality vs GT | 0.1874 | Low structural similarity to ground truth |
| Self-CodeBLEU | 0.4086 | Moderate diversity (lower = more diverse) |
| GT Coverage | 0.4136 | ~41% of ground-truth solution space covered |

## Comparison: 20-Problem Smoke Test vs 100-Problem Full Run

| Metric | 20 Problems | 100 Problems |
|---|---|---|
| pass@1 | 0.0100 | 0.0100 |
| pass@3 | 0.0267 | 0.0257 |
| pass@5 | 0.0389 | 0.0375 |
| pass@8 | 0.0489 | 0.0518 |
| pass@10 | 0.0500 | 0.0600 |
| Quality vs GT | 0.2069 | 0.1874 |
| Self-CodeBLEU | 0.4222 | 0.4086 |
| GT Coverage | 0.4019 | 0.4136 |

The 20-problem smoke test was a good predictor of the full run -- metrics are consistent within expected variance.

## Timing

| Phase | Duration |
|---|---|
| Generation (resumed from 53/100) | ~2.6 hours (~93s/problem avg) |
| Generation (total incl. first attempt) | ~7 hours |
| Execution (100 problems, early_stop) | 688s (~11.5 min) |
| Diversity (CodeBLEU) | 137s (~2.3 min) |

## Key Findings

1. **pass@1 = 1%**: Qwen2.5-Coder-3B-Instruct at temperature 0.7 solves only 1 in 100 TACO competitive programming problems on the first try.

2. **pass@10 = 6%**: Even with 10 attempts, only 6 problems out of 100 are solved. The monotonic increase from pass@1 to pass@10 (1% -> 6%) shows sampling does help, but the model fundamentally lacks the capability for most TACO problems.

3. **Difficulty gradient is clear**: EASY problems have 4/20 solved vs 0/20 for HARD and VERY_HARD. This validates both the model's relative capability and the difficulty labels.

4. **Diversity is moderate and consistent**: Self-CodeBLEU of 0.41 means temperature 0.7 produces structurally varied samples (not identical copies). GT coverage of 41% shows the generated solutions explore a meaningful portion of the solution space, even when incorrect.

5. **Quality vs GT is low (0.19)**: Generated code looks structurally different from reference solutions, reflecting that the model often takes wrong algorithmic approaches for these competitive programming problems.

6. **The model gets close but can't finish**: Several solved problems had only 1/10 correct -- the model can sometimes produce a correct solution but isn't reliable. The best performer (task 770, MEDIUM) achieved 4/10.
