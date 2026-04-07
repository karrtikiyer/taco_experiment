"""End-to-end experiment pipeline with incremental checkpointing and resume."""

import gc
import json
import time
from pathlib import Path

from .config import (
    RESULTS_DIR, MODEL_NAME, NUM_SAMPLES, TEMPERATURE, TOP_P,
    SMOKE_TEST_SIZE, FULL_TEST_SIZE, SEED,
)
from .data import load_dataset_split, stratified_sample, get_difficulty_distribution, model_short_name, SUPPORTED_DATASETS
from .generate import load_model, generate_all, load_existing_generations, DECODING_METHODS
from .execute import run_evaluation
from .diversity import compute_diversity_metrics


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline(n_problems=SMOKE_TEST_SIZE, run_name="smoke_test",
                 model_name=MODEL_NAME, decoding_method="top_p",
                 dataset_name="taco",
                 dtype="auto", attn_implementation=None,
                 skip_generation=False, skip_execution=False,
                 skip_diversity=False):
    """Run the full experiment pipeline with incremental checkpointing.

    Each stage saves results incrementally (JSONL, one line per problem).
    On re-run, already-completed problems are skipped automatically.

    Args:
        n_problems: number of problems to evaluate
        run_name: name for this run (used for output directory)
        model_name: HuggingFace model identifier
        decoding_method: one of "top_p", "temp_only", "top_p_only", "pless", "pless_norm"
        skip_generation: if True, skip generation entirely (use existing checkpoint)
        skip_execution: if True, skip execution entirely
        skip_diversity: if True, skip diversity computation
    """
    model_dir = model_short_name(model_name)
    output_dir = RESULTS_DIR / dataset_name / model_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_checkpoint = output_dir / "generations.jsonl"
    exec_checkpoint = output_dir / "execution.jsonl"

    if decoding_method in ("pless", "pless_norm"):
        effective_temp = 1.0
        effective_top_p = None
    elif decoding_method == "temp_only":
        effective_temp = TEMPERATURE
        effective_top_p = 1.0
    elif decoding_method == "top_p_only":
        effective_temp = 1.0
        effective_top_p = TOP_P
    else:
        effective_temp = TEMPERATURE
        effective_top_p = TOP_P

    print(f"=== {dataset_name.upper()} Experiment: {run_name} ({n_problems} problems) ===")
    print(f"  Dataset: {dataset_name}")
    print(f"  Decoding: {decoding_method}  temp={effective_temp}  top_p={effective_top_p}")
    print(f"  Output dir: {output_dir}")

    exclude_no_solutions = (dataset_name == "apps")

    # Step 1: Load dataset and sample
    print(f"\n[1/5] Loading {dataset_name.upper()} test set...")
    dataset = load_dataset_split(dataset_name)
    samples = stratified_sample(dataset, n_problems,
                                exclude_no_solutions=exclude_no_solutions)
    dist = get_difficulty_distribution(samples)
    print(f"  Difficulty distribution: {dict(dist)}")

    sample_meta = [{"task_id": tid, "difficulty": s["difficulty"]} for tid, s in samples]
    save_json(sample_meta, output_dir / "sample_meta.json")

    sample_lookup = {tid: dict(sample) for tid, sample in samples}
    del dataset
    gc.collect()
    print(f"  Freed full dataset; keeping {len(sample_lookup)} samples in memory")

    # Step 2: Generate solutions (incremental)
    if skip_generation:
        print("\n[2/5] Loading cached generations...")
        generation_results = list(load_existing_generations(gen_checkpoint).values())
        print(f"  Loaded {len(generation_results)} cached generation results")
    else:
        print("\n[2/5] Loading model and generating solutions...")
        model, tokenizer = load_model(
            model_name=model_name, dtype=dtype,
            attn_implementation=attn_implementation,
        )
        start = time.time()
        generation_results = generate_all(
            model, tokenizer, samples,
            n_samples=NUM_SAMPLES, temperature=effective_temp, top_p=effective_top_p,
            checkpoint_path=gen_checkpoint, decoding_method=decoding_method,
        )
        elapsed = time.time() - start
        print(f"  Generation complete: {elapsed:.1f}s ({elapsed/max(len(generation_results),1):.1f}s/problem)")

        del model, tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Step 3: Execute and compute pass@k (incremental)
    if skip_execution:
        print("\n[3/5] Skipping execution (--skip-execution)")
        passk_path = output_dir / "pass_at_k.json"
        passk_metrics = load_json(passk_path) if passk_path.exists() else {}
    else:
        print("\n[3/5] Executing generated code and computing pass@k...")
        start = time.time()
        execution_results, passk_metrics = run_evaluation(
            generation_results, sample_lookup,
            checkpoint_path=exec_checkpoint,
        )
        elapsed = time.time() - start
        print(f"  Execution complete: {elapsed:.1f}s")
        save_json(passk_metrics, output_dir / "pass_at_k.json")

    print("\n  pass@k results:")
    for k, v in passk_metrics.get("summary", {}).items():
        print(f"    {k}: {v:.4f}")

    # Step 4: Compute diversity metrics
    if skip_diversity:
        print("\n[4/5] Skipping diversity computation (--skip-diversity)")
        diversity_metrics = {}
    else:
        print("\n[4/5] Computing CodeBLEU diversity metrics...")
        start = time.time()
        diversity_metrics = compute_diversity_metrics(generation_results, sample_lookup)
        elapsed = time.time() - start
        print(f"  Diversity computation complete: {elapsed:.1f}s")
        save_json(diversity_metrics, output_dir / "diversity_metrics.json")

        summary = diversity_metrics.get("summary", {})
        print(f"  Quality (CodeBLEU vs GT): {summary.get('quality_vs_gt', 0):.4f}")
        print(f"  Self-CodeBLEU (lower=more diverse): {summary.get('self_codebleu', 0):.4f}")
        print(f"  GT Coverage: {summary.get('gt_coverage', 0):.4f}")

    # Step 5: Combined report
    print("\n[5/5] Generating combined report...")
    report = {
        "config": {
            "dataset": dataset_name,
            "model": model_name,
            "decoding_method": decoding_method,
            "temperature": effective_temp,
            "top_p": effective_top_p,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "num_samples": NUM_SAMPLES,
            "n_problems": n_problems,
            "seed": SEED,
        },
        "difficulty_distribution": dict(dist),
        "pass_at_k": passk_metrics.get("summary", {}),
        "diversity": diversity_metrics.get("summary", {}),
    }
    save_json(report, output_dir / "report.json")
    print(f"\n  Full report saved to {output_dir / 'report.json'}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TACO experiment")
    parser.add_argument("--dataset", type=str, default="taco",
                        choices=list(SUPPORTED_DATASETS),
                        help="Dataset to use (default: taco)")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--decoding-method", type=str, default="top_p",
                        choices=list(DECODING_METHODS))
    parser.add_argument("--n-problems", type=int, default=SMOKE_TEST_SIZE)
    parser.add_argument("--run-name", type=str, default="smoke_test")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16"],
                        help="Model dtype (default: auto)")
    parser.add_argument("--attn-implementation", type=str, default=None,
                        help="Attention implementation, e.g. flash_attention_2")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--skip-diversity", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        n_problems=args.n_problems,
        run_name=args.run_name,
        model_name=args.model,
        decoding_method=args.decoding_method,
        dataset_name=args.dataset,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        skip_generation=args.skip_generation,
        skip_execution=args.skip_execution,
        skip_diversity=args.skip_diversity,
    )
