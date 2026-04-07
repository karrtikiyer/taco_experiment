# TACO Decoding Experiment

Measuring **pass@k** accuracy and **code diversity** of different decoding strategies on competitive programming problems from the [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) dataset.

## Key Finding

On 92 TACO problems with Qwen2.5-Coder-7B-Instruct, **p-less decoding improves correctness (+27% relative pass@1) at the cost of diversity** compared to standard top_p sampling.

| k | top_p (T=0.7) | p-less (T=1.0) | p-less-norm (T=1.0) |
|---|---|---|---|
| pass@1 | 3.59% | **4.57%** | 3.91% |
| pass@10 | **8.70%** | **8.70%** | 6.52% |

| Diversity Metric | top_p | p-less | p-less-norm |
|---|---|---|---|
| Self-CodeBLEU (lower = more diverse) | **0.535** | 0.780 | 0.784 |
| GT Max-Recall (higher = broader) | **0.223** | 0.207 | 0.205 |

Full results and metric definitions in [RESULTS.md](RESULTS.md).

## Setup

Requires Python 3.11+ and [UV](https://docs.astral.sh/uv/).

```bash
uv sync
```

For p-less decoding, clone the reference implementation into `src/`:

```bash
git clone https://github.com/ryttry/p-less-sampling.git src/p-less-sampling
```

## Usage

### Run an experiment

```bash
# 7B model with top_p decoding, 100 problems
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --decoding-method top_p \
    --n-problems 100 \
    --run-name full_100_7b

# 7B model with p-less decoding
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --decoding-method pless \
    --n-problems 100 \
    --run-name full_100_7b_pless

# 14B model on GPU with bfloat16 and FlashAttention-2
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --decoding-method top_p \
    --n-problems 100 \
    --run-name run_14b_top_p \
    --dtype bfloat16 \
    --attn-implementation flash_attention_2
```

The pipeline runs five phases: dataset loading, generation (with checkpointing), execution, pass@k computation, and CodeBLEU diversity metrics. It supports `--skip-generation`, `--skip-execution`, and `--skip-diversity` flags to resume or recompute individual phases.

### Run all 5 decoding methods

```bash
# Usage: ./scripts/run_all_methods.sh <model> <n_problems> <prefix> <dtype> <attn>
./scripts/run_all_methods.sh Qwen/Qwen2.5-Coder-14B-Instruct 100 run_14b bfloat16 flash_attention_2
```

This runs `top_p`, `temp_only`, `top_p_only`, `pless`, and `pless_norm` sequentially with incremental checkpointing. See [RUNPOD.md](RUNPOD.md) for GPU setup instructions.

### Recalculate metrics excluding image problems

8 of the 100 sampled problems contain `<image>` tags that text-only models cannot interpret. To recalculate metrics with these excluded:

```bash
PYTHONPATH=src uv run python scripts/recalc_exclude_image.py --run-name full_100_7b
```

### Browse problems and generations

```bash
# Single-run viewer (problem statements, GT solutions, generations with PASS/FAIL)
PYTHONPATH=src uv run python scripts/view_problems.py --run-name full_100_7b

# Side-by-side comparison of two decoding methods
PYTHONPATH=src uv run python scripts/view_comparison.py \
    --left full_100_7b --right full_100_7b_pless
```

### Run tests

```bash
uv run pytest
```

## Project Structure

```
src/taco_experiment/
    config.py           # Constants (model, temperatures, k values)
    data.py             # TACO dataset loading and stratified sampling
    generate.py         # Prompt building, model loading, sample generation
    execute.py          # Code execution, pass@k computation
    diversity.py        # CodeBLEU-based diversity metrics
    p_less_processors.py # HuggingFace LogitsProcessor wrappers for p-less
    pipeline.py         # End-to-end orchestration and CLI
    metrics/
        testing_util.py # Vendored from TACO for sandboxed code execution

scripts/
    run_all_methods.sh      # Batch runner for all 5 decoding methods
    view_problems.py        # HTML viewer for a single run
    view_comparison.py      # Side-by-side HTML comparison of two runs
    recalc_exclude_image.py # Recompute metrics excluding <image> problems

results/                    # Experiment outputs (included in repo)
    full_100/               # 3B baseline
    full_100_7b/            # 7B + top_p
    full_100_7b_pless/      # 7B + p-less
    full_100_7b_pless_norm/ # 7B + p-less-norm
    smoke_test_20*/         # 20-problem smoke tests

tests/                      # pytest suite
```

## Metrics

### pass@k
Unbiased estimator from Chen et al. (2021). A sample is correct iff it passes **all** test cases. Answers: *"What is the probability that at least 1 of k samples is correct?"*

### Quality vs GT (precision-like)
CodeBLEU (Ren et al., 2020) of each generation against all ground truth solutions as a multi-reference set. Averaged across all 10 generations and all problems.

Answers: *"On average, does each generation resemble at least one correct solution?"* A model that always produces the same correct-looking approach scores high. This is a per-generation **quality** signal.

### Self-CodeBLEU (diversity)
Adaptation of Self-BLEU (Zhu et al., 2018). Each sample is scored against the other 9 as multi-references. Lower = more diverse.

Answers: *"How similar are the 10 generations to each other?"* High means the model keeps producing near-identical code; low means it explores different implementations.

### GT Max-Recall (coverage)
For each ground truth solution, find the maximum CodeBLEU against any of the 10 generations. Average across all GT solutions.

Answers: *"For each known correct approach, did the model generate something structurally close to it?"* You need **diverse** generations to score high — a model that only produces greedy solutions will miss GT solutions based on dynamic programming or brute force, pulling the average down.

### Quality vs GT and GT Max-Recall differ in direction

Both involve CodeBLEU between generations and ground truth, but they iterate over different outer loops:

| | Quality vs GT | GT Max-Recall |
|---|---|---|
| Outer loop | Each generation | Each GT solution |
| Reference set | All GTs (multi-ref) | Single GT, max across gens |
| Analogy | Precision | Recall |

If all 10 generations use the same algorithm, Quality vs GT stays high (each one matches some GT), but GT Max-Recall drops (GT solutions using other algorithms are uncovered).

See [RESULTS.md](RESULTS.md) for full metric definitions, caveats, and analysis.

## Decoding Methods

| Method | Temperature | Truncation | Description |
|---|---|---|---|
| top_p | 0.7 | top_p=0.95 | Standard nucleus sampling with temperature shaping |
| temp_only | 0.7 | None (top_p=1.0) | Pure temperature sampling, no nucleus truncation |
| top_p_only | 1.0 | top_p=0.95 | Pure nucleus sampling, no temperature shaping |
| pless | 1.0 | Adaptive: p_i >= sum(p^2) | [ryttry/p-less-sampling](https://github.com/ryttry/p-less-sampling) |
| pless_norm | 1.0 | Normalized: p_i >= (V*sum(p^2)-1)/(V-1) | [ryttry/p-less-sampling](https://github.com/ryttry/p-less-sampling) |

## Hardware

Local experiments (3B, 7B) were run on Apple Silicon (MPS). The pipeline supports CUDA GPUs with automatic multi-GPU sharding via `device_map="auto"`. Use `--dtype bfloat16` and `--attn-implementation flash_attention_2` for optimal GPU performance with larger models. See [RUNPOD.md](RUNPOD.md) for cloud GPU setup.
