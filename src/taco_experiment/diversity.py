"""CodeBLEU-based diversity and quality metrics."""

import ast
import json
import numpy as np
from itertools import combinations
from codebleu import calc_codebleu

from .config import CODEBLEU_WEIGHTS


def codebleu_score(predictions, references, weights=CODEBLEU_WEIGHTS):
    """Compute CodeBLEU score. references can be list[list[str]] for multi-ref."""
    try:
        result = calc_codebleu(
            references=references,
            predictions=predictions,
            lang="python",
            weights=weights,
        )
        return result["codebleu"]
    except Exception:
        return 0.0


def quality_vs_ground_truth(generations, ground_truth_solutions, weights=CODEBLEU_WEIGHTS):
    """Metric 1: Average CodeBLEU of each generated sample vs all ground truths (multi-reference).

    Higher = generations are closer to known correct solutions.
    """
    if not ground_truth_solutions:
        return {"mean": 0.0, "per_sample": []}

    per_sample = []
    for gen in generations:
        score = codebleu_score(
            predictions=[gen],
            references=[ground_truth_solutions],
            weights=weights,
        )
        per_sample.append(score)

    return {
        "mean": float(np.mean(per_sample)),
        "per_sample": per_sample,
    }


def self_codebleu(generations, weights=CODEBLEU_WEIGHTS):
    """Metric 2: Self-CodeBLEU — inter-sample similarity.

    For each sample, treat the other samples as references.
    Lower = more diverse (samples differ from each other).
    """
    if len(generations) < 2:
        return {"mean": 0.0, "per_sample": []}

    per_sample = []
    for i, gen in enumerate(generations):
        others = [g for j, g in enumerate(generations) if j != i]
        score = codebleu_score(
            predictions=[gen],
            references=[others],
            weights=weights,
        )
        per_sample.append(score)

    return {
        "mean": float(np.mean(per_sample)),
        "per_sample": per_sample,
    }


def gt_max_recall(generations, ground_truth_solutions, weights=CODEBLEU_WEIGHTS):
    """Metric 3: GT Max-Recall — how well generations collectively cover each GT solution.

    For each GT solution, find the max CodeBLEU against any generation.
    Average across all GTs. Higher = generations cover more of the GT space.
    Analogous to recall: "how well is each reference matched by the best generation?"
    """
    if not ground_truth_solutions or not generations:
        return {"mean": 0.0, "per_gt": [], "n_gt": 0}

    per_gt = []
    for gt in ground_truth_solutions:
        best = max(
            codebleu_score(predictions=[gen], references=[[gt]], weights=weights)
            for gen in generations
        )
        per_gt.append(best)

    return {
        "mean": float(np.mean(per_gt)),
        "per_gt": per_gt,
        "n_gt": len(ground_truth_solutions),
    }


def parse_ground_truth_solutions(sample, max_solutions=20):
    """Parse ground-truth solutions from a TACO sample.

    Limits to max_solutions to keep CodeBLEU computation tractable.
    Falls back to ast.literal_eval for TACO entries with malformed JSON
    (some contain unescaped single quotes).
    """
    raw = sample.get("solutions", "")
    for parser in (json.loads, ast.literal_eval):
        try:
            solutions = parser(raw)
            if isinstance(solutions, list):
                return solutions[:max_solutions]
        except Exception:
            continue
    return []


def compute_diversity_metrics(generation_results, dataset):
    """Compute all 3 diversity metrics for all problems.

    Args:
        generation_results: list of dicts with 'task_id' and 'output'
        dataset: TACO dataset

    Returns:
        dict with per-problem and aggregated metrics
    """
    all_quality = []
    all_self_codebleu = []
    all_coverage = []
    per_problem = {}

    for gen_item in generation_results:
        task_id = gen_item["task_id"]
        generations = gen_item["output"]
        sample = dataset[task_id]

        gt_solutions = parse_ground_truth_solutions(sample)

        q = quality_vs_ground_truth(generations, gt_solutions)
        s = self_codebleu(generations)
        c = gt_max_recall(generations, gt_solutions)

        per_problem[task_id] = {
            "quality": q,
            "self_codebleu": s,
            "gt_coverage": c,
        }

        all_quality.append(q["mean"])
        all_self_codebleu.append(s["mean"])
        all_coverage.append(c["mean"])

    summary = {
        "quality_vs_gt": float(np.mean(all_quality)) if all_quality else 0.0,
        "self_codebleu": float(np.mean(all_self_codebleu)) if all_self_codebleu else 0.0,
        "gt_coverage": float(np.mean(all_coverage)) if all_coverage else 0.0,
    }

    return {"summary": summary, "detail": per_problem}
