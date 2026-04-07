# Running on RunPod GPU

Guide for running the TACO decoding experiment on RunPod GPU pods.

## GPU Sizing

VRAM rule of thumb: ~2 GB per billion parameters at full precision; ~1 GB/B at float16/bfloat16.

| Model | VRAM needed (fp16) | Recommended GPU | RunPod cost (approx.) |
|-------|-------------------|-----------------|----------------------|
| 7B | ~14 GB | 1x RTX 4090 (24 GB) | ~$0.44/hr spot |
| 14B | ~28 GB | 1x A100-40GB | ~$1.39/hr on-demand |
| 32B | ~64 GB | 1x A100-80GB or 2x A100-40GB | ~$1.49-2.78/hr |

Spot instances work well for this workload -- our incremental checkpointing handles preemption gracefully.

## Pod Creation

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods) and click **Deploy**
2. Select your GPU (see sizing table above)
3. Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
4. Storage:
   - **Container disk**: 20 GB (default, for OS/temp files)
   - **Volume disk**: 50 GB (persists at `/workspace` across pod stops)
5. Enable **SSH** with a public IP for file transfers (`scp`/`rsync`)
6. Click **Deploy On-Demand** (or **Deploy Spot** for cost savings)

## Connecting

**SSH** (recommended for long-running work):
- Add your public SSH key in RunPod account settings
- Copy the SSH command from your pod's **Connect** tab
- Use `tmux` or `screen` inside SSH for long runs

**Web Terminal**: Quick checks only -- disconnects on browser close.

## One-Time Setup

Run these once in `/workspace` (persists across pod stop/restart):

```bash
cd /workspace

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the experiment repo
git clone https://github.com/karrtikiyer/taco_experiment.git
cd taco_experiment

# Install dependencies
uv sync

# Clone p-less sampling (external dependency)
git clone https://github.com/ryttry/p-less-sampling.git src/p-less-sampling

# Verify GPU is visible
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## Running Experiments

### All 5 decoding methods (recommended)

```bash
cd /workspace/taco_experiment

# Usage: ./scripts/run_all_methods.sh <model> <n_problems> <prefix> <dtype> <attn>
# Runs: top_p, temp_only, top_p_only, pless, pless_norm sequentially

# 14B model, 100 problems, bfloat16
tmux new -s experiment
./scripts/run_all_methods.sh Qwen/Qwen2.5-Coder-14B-Instruct 100 run_14b bfloat16

# 14B with FlashAttention-2 (faster, requires Ampere+ GPU)
./scripts/run_all_methods.sh Qwen/Qwen2.5-Coder-14B-Instruct 100 run_14b bfloat16 flash_attention_2

# 32B model, full 1000 problems
./scripts/run_all_methods.sh Qwen/Qwen2.5-Coder-32B-Instruct 1000 run_32b bfloat16 flash_attention_2
```

### Single method

```bash
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --decoding-method top_p \
    --n-problems 100 \
    --run-name run_14b_top_p \
    --dtype bfloat16 \
    --attn-implementation flash_attention_2
```

### Smoke test first

Run 20 problems with one method to verify everything works:

```bash
PYTHONPATH=src uv run python -m taco_experiment.pipeline \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --decoding-method top_p \
    --n-problems 20 \
    --run-name smoke_14b \
    --dtype bfloat16
```

## Monitoring

In a separate `tmux` pane or SSH session:

```bash
# GPU utilization
nvidia-smi -l 5

# Generation progress (count of completed problems)
watch -n 30 'wc -l /workspace/taco_experiment/results/*/generations.jsonl'

# Execution progress
watch -n 30 'wc -l /workspace/taco_experiment/results/*/execution.jsonl'

# Disk usage
df -h /workspace
```

## Resumption

If the pod is preempted (spot) or stopped, restart it and re-run the same command. The pipeline automatically resumes from the last checkpoint -- completed problems in `generations.jsonl` and `execution.jsonl` are skipped.

## Downloading Results

### Via rsync/scp (requires public IP SSH)

```bash
# From your local machine:
rsync -avz root@<POD_IP>:/workspace/taco_experiment/results/ ./results/

# Or specific run:
scp -r root@<POD_IP>:/workspace/taco_experiment/results/run_14b_top_p/ ./results/
```

### Via git push

```bash
# From within the pod:
cd /workspace/taco_experiment
git add results/
git commit -m "Add GPU run results"
git push
```

## Cost Estimation

Rough estimates for 100 problems, 10 samples each:

| Model | Generation time | Execution time | Total wall time | Cost (A100 on-demand) |
|-------|----------------|---------------|-----------------|----------------------|
| 14B | ~2-3 hrs | ~1 hr | ~4 hrs per method | ~$5.50/method |
| 32B | ~5-8 hrs | ~1 hr | ~8 hrs per method | ~$12/method |

For all 5 methods: multiply by 5. Using spot instances can reduce cost by ~35-50%.

## Troubleshooting

**OOM during generation**: Try a smaller `--dtype` (`float16` instead of `auto`), or use a GPU with more VRAM. The `device_map="auto"` will automatically shard across multiple GPUs if available.

**FlashAttention-2 not available**: Install it explicitly: `pip install flash-attn --no-build-isolation`. Requires Ampere+ GPU (A100, H100, RTX 30xx/40xx). If unavailable, simply omit the `--attn-implementation` flag.

**Slow execution phase**: Code execution runs on CPU -- GPU is idle during this phase. This is expected; the bottleneck is per-test-case timeouts (4s each).
