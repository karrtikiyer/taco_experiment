"""Recalculate pass@k and diversity metrics excluding problems with <image> tags.

Reads existing execution.jsonl and generations.jsonl, filters out image problems,
recomputes metrics, and saves updated report.json (overwrites diversity_metrics.json
and pass_at_k.json with filtered versions).

Usage:
    PYTHONPATH=src uv run python scripts/recalc_exclude_image.py --run-name full_100_7b
"""

import ast
import json
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taco_experiment.data import load_dataset_split, _has_image, model_short_name, SUPPORTED_DATASETS
from taco_experiment.execute import compute_pass_at_k
from taco_experiment.diversity import (
    parse_ground_truth_solutions, quality_vs_ground_truth,
    self_codebleu, gt_max_recall,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="taco",
                        choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="HuggingFace model id (used to locate results directory)")
    args = parser.parse_args()

    model_dir = model_short_name(args.model)
    results_dir = Path(__file__).parent.parent / "results" / args.dataset / model_dir / args.run_name

    with open(results_dir / "sample_meta.json") as f:
        meta = json.load(f)

    print(f"Loading {args.dataset.upper()} dataset to identify image problems...")
    dataset = load_dataset_split(args.dataset)

    image_task_ids = set()
    for item in meta:
        tid = item["task_id"]
        if _has_image(dataset[tid]["question"]):
            image_task_ids.add(tid)

    print(f"Found {len(image_task_ids)} image problems to exclude: {sorted(image_task_ids)}")
    print(f"Keeping {len(meta) - len(image_task_ids)} / {len(meta)} problems")

    # --- Recalculate pass@k ---
    exec_path = results_dir / "execution.jsonl"
    if exec_path.exists():
        execution_results = {}
        with open(exec_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item["task_id"] not in image_task_ids:
                        execution_results[item["task_id"]] = item["results"]

        passk = compute_pass_at_k(execution_results)
        with open(results_dir / "pass_at_k.json", "w") as f:
            json.dump(passk, f, indent=2)

        print(f"\npass@k (excluding image problems):")
        for k, v in passk["summary"].items():
            print(f"  {k}: {v:.4f}")
    else:
        print("No execution.jsonl found, skipping pass@k")
        passk = {}

    # --- Recalculate diversity ---
    gen_path = results_dir / "generations.jsonl"
    if gen_path.exists():
        gen_results = []
        with open(gen_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item["task_id"] not in image_task_ids:
                        gen_results.append(item)

        sample_lookup = {}
        for item in meta:
            tid = item["task_id"]
            if tid not in image_task_ids:
                sample_lookup[tid] = dict(dataset[tid])

        all_quality = []
        all_self_cb = []
        all_coverage = []
        detail = {}

        for gen_item in gen_results:
            tid = gen_item["task_id"]
            generations = gen_item["output"]
            sample = sample_lookup[tid]
            gt = parse_ground_truth_solutions(sample)

            q = quality_vs_ground_truth(generations, gt)
            s = self_codebleu(generations)
            c = gt_max_recall(generations, gt)

            detail[tid] = {"quality": q, "self_codebleu": s, "gt_coverage": c}
            all_quality.append(q["mean"])
            all_self_cb.append(s["mean"])
            all_coverage.append(c["mean"])

        diversity = {
            "summary": {
                "quality_vs_gt": float(np.mean(all_quality)) if all_quality else 0.0,
                "self_codebleu": float(np.mean(all_self_cb)) if all_self_cb else 0.0,
                "gt_coverage": float(np.mean(all_coverage)) if all_coverage else 0.0,
            },
            "detail": detail,
        }

        with open(results_dir / "diversity_metrics.json", "w") as f:
            json.dump(diversity, f, indent=2, default=str)

        print(f"\nDiversity (excluding image problems):")
        for k, v in diversity["summary"].items():
            print(f"  {k}: {v:.4f}")
    else:
        print("No generations.jsonl found, skipping diversity")
        diversity = {}

    # --- Update report ---
    report_path = results_dir / "report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {}

    report["image_problems_excluded"] = sorted(image_task_ids)
    report["n_problems_after_exclusion"] = len(meta) - len(image_task_ids)
    if passk:
        report["pass_at_k"] = passk.get("summary", {})
    if diversity:
        report["diversity"] = diversity.get("summary", {})

    # Recalc difficulty distribution excluding image problems
    dist = {}
    for item in meta:
        if item["task_id"] not in image_task_ids:
            d = item["difficulty"]
            dist[d] = dist.get(d, 0) + 1
    report["difficulty_distribution"] = dist

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nUpdated report saved to {report_path}")


if __name__ == "__main__":
    main()
