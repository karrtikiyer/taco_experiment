# TACO Experiment Results

## 1. Experiment Overview

### Dataset

[BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) test set — a competitive programming benchmark with multiple ground truth solutions per problem. We sampled 100 problems using stratified sampling across 5 difficulty levels (~20 per level, seed=42). 8 problems containing `<image>` tags were excluded from metric calculation since text-only models cannot interpret them, leaving **92 evaluated problems**.

### Models

| Model | Parameters |
|---|---|
| Qwen/Qwen2.5-Coder-3B-Instruct | 3B |
| Qwen/Qwen2.5-Coder-7B-Instruct | 7B |

### Decoding Methods

| Method | Temperature | top_p | Description |
|---|---|---|---|
| **top_p** | 0.7 | 0.95 | Standard nucleus sampling |
| **p-less** | 1.0 | None | Adaptive threshold: keep tokens where p_i >= sum(p^2). From [ryttry/p-less-sampling](https://github.com/ryttry/p-less-sampling) |
| **p-less-norm** | 1.0 | None | Normalized variant: keep tokens where p_i >= (V * sum(p^2) - 1) / (V - 1) |

All runs: 10 samples per problem, max_new_tokens=2048.

### Evaluation Protocol

- **Execution**: TACO's per-test-case 4-second timeout (no outer timeout), early stop on first failure, 512 MB memory limit per solution.
- **pass@k**: Standard unbiased estimator from Chen et al. (2021), "Evaluating Large Language Models Trained on Code." A sample is correct iff it passes all test cases.
- **Diversity**: CodeBLEU-based metrics (defined below).

### Difficulty Distribution (after image exclusion)

| Difficulty | Count |
|---|---|
| EASY | 19 |
| MEDIUM | 19 |
| MEDIUM_HARD | 17 |
| HARD | 19 |
| VERY_HARD | 18 |
| **Total** | **92** |

---

## 2. Metric Definitions

### 2.1 pass@k

The unbiased estimator of the probability that at least one of k samples is correct (Chen et al., 2021):

$$\text{pass@k} = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where n=10 samples are generated and c is the number of correct samples. This is the standard metric for code generation evaluation.

### 2.2 Quality vs Ground Truth (CodeBLEU)

For each generated sample, compute CodeBLEU (Ren et al., 2020) against all ground truth solutions as a multi-reference set. Average across all 10 samples and then across all problems.

CodeBLEU combines four equally weighted components: n-gram match, weighted n-gram match (emphasizing keywords), syntax match (AST subtree matching), and data-flow match (semantic analysis).

Multi-reference CodeBLEU follows the standard BLEU formulation where modified precision credits matching **any** reference — this is appropriate because multiple valid solutions exist for each problem.

**Interpretation**: Higher means generations are structurally closer to known correct solutions.

**Caveat**: This measures structural similarity, not semantic correctness. A correct solution using a novel algorithmic approach absent from the ground truth set will score low. Conversely, a syntactically similar but buggy solution may score high.

**Reference**: Ren et al. (2020), "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis" (arXiv:2009.10297).

### 2.3 Self-CodeBLEU

An adaptation of Self-BLEU (Zhu et al., 2018) to code using CodeBLEU. For each generated sample, the remaining 9 samples serve as multi-references. Average across all 10 samples per problem, then across problems.

**Interpretation**: Lower means generations are more diverse (structurally different from each other). Higher means the model tends to produce similar solutions.

**Caveat**: The multi-reference formulation measures similarity to the **closest** other sample (via max n-gram counts), not average pairwise similarity. This is consistent with the original Self-BLEU definition but means a single near-duplicate is sufficient to inflate a sample's score.

**Reference**: Zhu et al. (2018), "Texygen: A Benchmarking Platform for Text Generation Models" (arXiv:1802.01886).

### 2.4 GT Max-Recall

For each ground truth solution, compute the maximum CodeBLEU score against any of the 10 generations. Average across all ground truth solutions, then across problems. This is analogous to recall in information retrieval: "how well is each reference solution covered by the best-matching generation?"

**Interpretation**: Higher means the generated samples collectively cover the ground truth solution space well — for each known correct approach, at least one generation is structurally close to it.

Unlike the count-based coverage metric it replaces, GT Max-Recall is a continuous score not bounded by the ratio of generations to ground truth solutions. A problem with 50 GT solutions gets a meaningful score even with only 10 generations.

**Caveat**: Still CodeBLEU-based, so it captures structural (n-gram, AST, data-flow) similarity rather than true algorithmic equivalence. For deeper algorithmic diversity analysis, clustering-based approaches such as those proposed by Lee et al. (EMNLP 2025, "How Diversely Can Language Models Solve Problems?") may be more appropriate.

**Note**: The 3B baseline (Section 3) uses the older count-based GT Coverage metric and was not recomputed with GT Max-Recall. The 7B comparison (Section 4) uses GT Max-Recall throughout.

---

## 3. Results: 3B Baseline

**Model**: Qwen/Qwen2.5-Coder-3B-Instruct, top_p decoding (T=0.7, top_p=0.95)

### pass@k

| k | pass@k |
|---|---|
| 1 | 1.09% |
| 3 | 2.79% |
| 5 | 4.08% |
| 8 | 5.63% |
| 10 | 6.52% |

6 of 92 problems solved (at least 1 correct generation out of 10).

### Diversity

| Metric | Value |
|---|---|
| Quality vs GT | 0.196 |
| Self-CodeBLEU | 0.410 |
| GT Coverage (count-based) | 0.423 |

*Note: The 3B run uses the older count-based GT Coverage metric (see Section 2.4).*

### By Difficulty

| Difficulty | Solved | Approx. pass@1 |
|---|---|---|
| EASY | 4 / 19 | 2.63% |
| MEDIUM | 1 / 19 | 2.11% |
| MEDIUM_HARD | 1 / 17 | 0.59% |
| HARD | 0 / 19 | 0.00% |
| VERY_HARD | 0 / 18 | 0.00% |

---

## 4. Results: 7B Three-Way Decoding Comparison

**Model**: Qwen/Qwen2.5-Coder-7B-Instruct across three decoding methods.

### 4.1 pass@k

| k | top_p (T=0.7) | p-less (T=1.0) | p-less-norm (T=1.0) |
|---|---|---|---|
| **1** | 3.59% | **4.57%** | 3.91% |
| **3** | 5.82% | **6.63%** | 5.37% |
| **5** | 6.94% | **7.49%** | 5.89% |
| **8** | 8.04% | **8.26%** | 6.30% |
| **10** | **8.70%** | **8.70%** | 6.52% |

Problems solved (at least 1/10 correct):
- top_p: 8 / 92
- p-less: 8 / 92
- p-less-norm: 6 / 92

### 4.2 Diversity

| Metric | top_p | p-less | p-less-norm | Best |
|---|---|---|---|---|
| Quality vs GT | 0.206 | **0.209** | 0.208 | p-less (marginal) |
| Self-CodeBLEU | **0.535** | 0.780 | 0.784 | top_p (lower = more diverse) |
| GT Max-Recall | **0.223** | 0.207 | 0.205 | top_p (higher = broader) |

### 4.3 By Difficulty (problems solved)

| Difficulty | top_p | p-less | p-less-norm |
|---|---|---|---|
| EASY (19) | 5 | 4 | 3 |
| MEDIUM (19) | 3 | 2 | 2 |
| MEDIUM_HARD (17) | 0 | 0 | 0 |
| HARD (19) | 0 | 0 | 0 |
| VERY_HARD (18) | 0 | 2 | 1 |

### 4.4 3B vs 7B Scaling (top_p decoding)

| Metric | 3B | 7B | Change |
|---|---|---|---|
| pass@1 | 1.09% | 3.59% | +2.50 pp (+3.3x) |
| pass@10 | 6.52% | 8.70% | +2.18 pp (+1.3x) |
| Quality vs GT | 0.196 | 0.206 | +0.010 |
| Self-CodeBLEU | 0.410 | 0.535 | +0.125 (less diverse) |

*GT Coverage/Max-Recall not compared across model sizes due to different metric versions (3B uses count-based, 7B uses max-recall).*

---

## 5. Key Findings

### 5.1 p-less improves accuracy at the cost of diversity

p-less achieves the best pass@k across nearly all values of k, with a +27% relative improvement in pass@1 over top_p (4.57% vs 3.59%). However, its Self-CodeBLEU is substantially higher (0.780 vs 0.535), indicating that the 10 generated samples are much more similar to each other. GT Max-Recall also drops (0.207 vs 0.223), confirming that top_p's generations cover the ground truth solution space more broadly.

**Interpretation**: p-less concentrates probability mass on high-quality tokens, producing more correct but less varied code. This is consistent with the method's design — it adaptively narrows the sampling distribution without a fixed top_p cutoff.

### 5.2 p-less-norm underperforms on both axes

p-less-norm yields lower pass@k than both top_p and p-less, while providing no diversity advantage over plain p-less (Self-CodeBLEU: 0.784 vs 0.780). The normalized threshold appears too aggressive, filtering out tokens that occasionally produce correct solutions.

### 5.3 top_p produces the most diverse generations

top_p at T=0.7 produces Self-CodeBLEU of 0.535 — meaning samples are moderately different from each other. Both p-less variants cluster around 0.78, indicating significantly less inter-sample variation. top_p also achieves the highest GT Max-Recall (0.223 vs ~0.206), confirming its samples cover more of the ground truth solution space.

### 5.4 Quality vs GT is stable across methods

All three methods produce nearly identical Quality vs GT scores (~0.21). The structural similarity of generated code to ground truth solutions is primarily determined by the model, not the decoding method. This makes sense: all three methods decode from the same learned distribution, differing only in how they shape the tail.

### 5.5 Scaling from 3B to 7B improves accuracy but reduces diversity

Moving from 3B to 7B (both with top_p) yields a 3.3x improvement in pass@1 (1.09% to 3.59%), but Self-CodeBLEU increases from 0.410 to 0.535. The larger model is more confident in its outputs, producing more correct but less varied code.

### 5.6 TACO remains very difficult for models of this scale

Even the best configuration (7B + p-less) solves only 8 of 92 problems and achieves pass@1 under 5%. No configuration solves any HARD problems. The two VERY_HARD problems solved by p-less are notable exceptions but not statistically meaningful.

---

## 6. Limitations

### Metric Limitations

1. **Quality vs GT measures similarity, not correctness.** A generation can be correct yet score low (novel algorithm) or incorrect yet score high (syntactically similar but buggy). It should be interpreted alongside pass@k, not as a substitute.

2. **Self-CodeBLEU uses closest-match semantics.** The multi-reference formulation means a sample's score reflects similarity to its nearest neighbor among the other 9 samples, not average similarity to all. This is consistent with Self-BLEU (Zhu et al., 2018) but differs from pairwise average approaches.

3. **GT Max-Recall is CodeBLEU-based, not algorithmic.** It measures structural similarity (n-gram, AST, data-flow) between generations and ground truth solutions. Two algorithmically equivalent implementations with different syntax may score low, and vice versa. For deeper algorithmic diversity analysis, clustering-based approaches (Lee et al., EMNLP 2025) may be more principled.

### Experimental Limitations

4. **Sample size**: 92 problems (after image exclusion) with 10 samples each. Confidence intervals are wide. The difference between pass@1 of 3.59% (top_p) and 4.57% (p-less) is +0.98 percentage points on 92 problems — suggestive but not statistically conclusive.

5. **Single model family**: All results use Qwen2.5-Coder. The accuracy-diversity trade-off of p-less may differ for other architectures or model scales.

6. **Temperature confound**: top_p uses T=0.7 while p-less methods use T=1.0 (as designed by the p-less authors). The diversity difference partly reflects the temperature gap, not solely the decoding algorithm. A fairer comparison would include top_p at T=1.0 as a control.

7. **Image problems**: 8 of 100 sampled problems contained `<image>` references that text-only models cannot interpret. These were excluded from metric calculation but not from the original sampling, meaning the evaluated set of 92 is not a perfectly balanced stratified sample.

---

## 7. Reproducibility

All results were generated on Apple Silicon (MPS) using the code in this repository. To reproduce:

```bash
# 3B baseline
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --n-problems 100 --run-name full_100

# 7B + top_p
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --decoding-method top_p --n-problems 100 --run-name full_100_7b

# 7B + p-less
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --decoding-method pless --n-problems 100 --run-name full_100_7b_pless

# 7B + p-less-norm
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --decoding-method pless_norm --n-problems 100 --run-name full_100_7b_pless_norm

# Recalculate excluding image problems
PYTHONPATH=src uv run python scripts/recalc_exclude_image.py --run-name <run_name>

# Side-by-side comparison viewer
PYTHONPATH=src uv run python scripts/view_comparison.py --left full_100_7b --right full_100_7b_pless
```

Raw results are stored in `results/<dataset>/<model_short>/<run_name>/` with `report.json`, `pass_at_k.json`, `diversity_metrics.json`, `execution.jsonl`, and `generations.jsonl`.
