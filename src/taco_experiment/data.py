"""Load TACO test set and create stratified samples across difficulty levels."""

import random
from collections import Counter
from datasets import load_dataset

from .config import DIFFICULTY_LEVELS, SEED


def _has_image(question: str) -> bool:
    """Check if a problem references images that text-only models cannot see."""
    return "<image>" in question.lower()


def load_taco_test(difficulties=None):
    """Load TACO test set, optionally filtered by difficulties."""
    dataset = load_dataset("BAAI/TACO", split="test", trust_remote_code=True)
    if difficulties and difficulties != ["ALL"]:
        dataset = dataset.filter(lambda x: x["difficulty"] in difficulties)
    return dataset


def stratified_sample(dataset, n_total, seed=SEED, exclude_image=True):
    """Sample n_total problems proportionally across difficulty levels.

    Returns list of (original_index, sample) tuples preserving the index
    for downstream task_id mapping.

    Args:
        exclude_image: if True, skip problems whose question contains <image>
            tags (unsolvable by text-only models).
    """
    random.seed(seed)

    by_difficulty = {}
    n_excluded = 0
    for idx, sample in enumerate(dataset):
        if exclude_image and _has_image(sample["question"]):
            n_excluded += 1
            continue
        diff = sample["difficulty"]
        by_difficulty.setdefault(diff, []).append((idx, sample))

    if n_excluded:
        print(f"  Excluded {n_excluded} problems with <image> tags")

    total_count = len(dataset)
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
