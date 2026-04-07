"""Load coding problem datasets (TACO, APPS) and create stratified samples."""

import json
import random
from collections import Counter
from datasets import load_dataset as hf_load_dataset

from .config import SEED


DATASET_REGISTRY = {
    "taco": {
        "hf_id": "BAAI/TACO",
        "split": "test",
        "trust_remote_code": True,
        "difficulties": ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"],
    },
    "apps": {
        "hf_id": "codeparrot/apps",
        "split": "test",
        "trust_remote_code": True,
        "difficulties": ["introductory", "interview", "competition"],
    },
}

SUPPORTED_DATASETS = tuple(DATASET_REGISTRY.keys())


def _has_image(question: str) -> bool:
    """Check if a problem references images that text-only models cannot see."""
    return "<image>" in question.lower()


def _has_solutions(sample: dict) -> bool:
    """Check if a problem has ground-truth solutions for diversity metrics."""
    raw = sample.get("solutions", "")
    if not raw or raw.strip() in ("", "[]"):
        return False
    try:
        parsed = json.loads(raw)
        return isinstance(parsed, list) and len(parsed) > 0
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def load_dataset_split(dataset_name="taco", difficulties=None):
    """Load a dataset split from the registry.

    Args:
        dataset_name: one of SUPPORTED_DATASETS ("taco", "apps").
        difficulties: optional list to filter by. None loads all.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. "
                         f"Must be one of {SUPPORTED_DATASETS}")

    info = DATASET_REGISTRY[dataset_name]
    kwargs = {"split": info["split"]}
    if info["trust_remote_code"]:
        kwargs["trust_remote_code"] = True

    dataset = hf_load_dataset(info["hf_id"], **kwargs)

    if difficulties and difficulties != ["ALL"]:
        dataset = dataset.filter(lambda x: x["difficulty"] in difficulties)

    return dataset


def load_taco_test(difficulties=None):
    """Backward-compatible wrapper: loads the TACO test set."""
    return load_dataset_split("taco", difficulties=difficulties)


def stratified_sample(dataset, n_total, seed=SEED,
                      exclude_image=True, exclude_no_solutions=False):
    """Sample n_total problems proportionally across difficulty levels.

    Returns list of (original_index, sample) tuples preserving the index
    for downstream task_id mapping.

    Args:
        exclude_image: if True, skip problems whose question contains <image>
            tags (unsolvable by text-only models).
        exclude_no_solutions: if True, skip problems without ground-truth
            solutions (needed for diversity metrics; relevant for APPS which
            has ~1235 test samples without GT solutions).
    """
    random.seed(seed)

    by_difficulty = {}
    n_image_excluded = 0
    n_nosol_excluded = 0
    for idx, sample in enumerate(dataset):
        if exclude_image and _has_image(sample["question"]):
            n_image_excluded += 1
            continue
        if exclude_no_solutions and not _has_solutions(sample):
            n_nosol_excluded += 1
            continue
        diff = sample["difficulty"]
        by_difficulty.setdefault(diff, []).append((idx, sample))

    if n_image_excluded:
        print(f"  Excluded {n_image_excluded} problems with <image> tags")
    if n_nosol_excluded:
        print(f"  Excluded {n_nosol_excluded} problems without GT solutions")

    total_count = sum(len(pool) for pool in by_difficulty.values())
    selected = []

    remaining = n_total
    difficulties_sorted = sorted(by_difficulty.keys())

    for i, diff in enumerate(difficulties_sorted):
        pool = by_difficulty[diff]
        if i == len(difficulties_sorted) - 1:
            count = remaining
        else:
            count = max(1, round(n_total * len(pool) / total_count))
            count = min(count, remaining)

        sampled = random.sample(pool, min(count, len(pool)))
        selected.extend(sampled)
        remaining -= len(sampled)

    random.shuffle(selected)
    return selected


def get_difficulty_distribution(samples):
    """Return a Counter of difficulty levels in the sample set."""
    return Counter(s["difficulty"] for _, s in samples)
