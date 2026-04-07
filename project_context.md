# TACO Experiment - Project Context

## Goal
Measure pass@k and diversity of code generation solutions on the TACO dataset test set, comparing standard top_p sampling against p-less adaptive decoding across model scales (3B, 7B).

## Dataset
- **TACO** (Topics in Algorithmic COde generation): https://huggingface.co/datasets/BAAI/TACO
- Test set: 1000 problems, each with multiple ground-truth Python solutions
- 5 difficulty levels: EASY, MEDIUM, MEDIUM_HARD, HARD, VERY_HARD
- Problems use either call-based (fn_name) or stdin-based I/O
- 8 problems contain `<image>` tags and are excluded from metric calculation

## Models
- Qwen2.5-Coder-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct
- Qwen2.5-Coder-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

## Decoding Methods
- **top_p**: Standard nucleus sampling (T=0.7, top_p=0.95)
- **p-less**: Adaptive threshold from ryttry/p-less-sampling (T=1.0, no top_p)
- **p-less-norm**: Normalized variant of p-less (T=1.0, no top_p)

## Metrics
1. **pass@k** (k=1,3,5,8,10): Execution-based correctness using TACO's testing framework
2. **Quality (CodeBLEU vs GT)**: Multi-reference CodeBLEU of generated samples against ground-truth solutions
3. **Self-CodeBLEU**: Inter-sample diversity (lower = more diverse)
4. **GT Max-Recall**: For each GT solution, max CodeBLEU against any generation; measures solution space coverage

## Experiment Design
- Stratified sample of 100 problems across difficulty levels (seed=42)
- 10 samples per problem
- Smoke tests (20 problems) followed by full runs (100 problems)
- Results exclude 8 image problems (92 evaluated problems)

## Tech Stack
- UV for project/dependency management
- transformers for model inference (MPS/Apple Silicon)
- TACO's vendored testing_util for code execution
- codebleu PyPI package for CodeBLEU metrics
- pytest for testing

## Code Execution Safety
TACO's testing_util uses reliability_guard() to disable dangerous builtins. 512 MB memory limit per solution, 4-second per-test-case timeout.
