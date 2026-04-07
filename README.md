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

# 3B model (default)
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --n-problems 100 \
    --run-name full_100
```

The pipeline runs five phases: dataset loading, generation (with checkpointing), execution, pass@k computation, and CodeBLEU diversity metrics. It supports `--skip-generation`, `--skip-execution`, and `--skip-diversity` flags to resume or recompute individual phases.

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

- **pass@k**: Unbiased estimator (Chen et al., 2021). Execution-based; correct iff all test cases pass.
- **Quality vs GT**: CodeBLEU (Ren et al., 2020) of each generation against all ground truth solutions as multi-reference.
- **Self-CodeBLEU**: Adaptation of Self-BLEU (Zhu et al., 2018). Each sample scored against the other 9 as references. Lower = more diverse.
- **GT Max-Recall**: For each ground truth solution, max CodeBLEU against any generation. Measures how well the generation set covers the known solution space.

See [RESULTS.md](RESULTS.md) for full metric definitions, caveats, and analysis.

## Decoding Methods

| Method | Temperature | Truncation | Source |
|---|---|---|---|
| top_p | 0.7 | top_p=0.95 | Standard nucleus sampling |
| p-less | 1.0 | Adaptive: p_i >= sum(p^2) | [ryttry/p-less-sampling](https://github.com/ryttry/p-less-sampling) |
| p-less-norm | 1.0 | Normalized: p_i >= (V*sum(p^2)-1)/(V-1) | [ryttry/p-less-sampling](https://github.com/ryttry/p-less-sampling) |

## Hardware

All experiments were run on Apple Silicon (MPS) with the 7B model in bfloat16. Generation takes ~1.5 hours per 100 problems; execution ~20 minutes; diversity ~2 minutes.
