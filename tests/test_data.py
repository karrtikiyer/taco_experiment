"""Tests for data loading and stratified sampling."""

import pytest
from collections import Counter


class TestStratifiedSample:
    """Test stratified sampling logic without requiring the full TACO download."""

    def _make_fake_dataset(self):
        """Create a minimal fake dataset mimicking TACO structure."""
        difficulties = (
            ["EASY"] * 40
            + ["MEDIUM"] * 30
            + ["MEDIUM_HARD"] * 15
            + ["HARD"] * 10
            + ["VERY_HARD"] * 5
        )
        return [{"difficulty": d, "question": f"q{i}", "solutions": "[]", "input_output": "{}"} for i, d in enumerate(difficulties)]

    def test_sample_size(self):
        from src.taco_experiment.data import stratified_sample
        fake = self._make_fake_dataset()
        samples = stratified_sample(fake, 20, seed=42)
        assert len(samples) == 20

    def test_all_difficulties_represented(self):
        from src.taco_experiment.data import stratified_sample
        fake = self._make_fake_dataset()
        samples = stratified_sample(fake, 20, seed=42)
        diffs = {s["difficulty"] for _, s in samples}
        assert len(diffs) == 5, f"Expected 5 difficulty levels, got {diffs}"

    def test_proportionality(self):
        from src.taco_experiment.data import stratified_sample
        fake = self._make_fake_dataset()
        samples = stratified_sample(fake, 20, seed=42)
        dist = Counter(s["difficulty"] for _, s in samples)
        # EASY is 40% of 100, so ~8 of 20; should be largest group
        assert dist["EASY"] >= dist["VERY_HARD"], (
            f"EASY ({dist['EASY']}) should have more samples than VERY_HARD ({dist['VERY_HARD']})"
        )

    def test_reproducibility(self):
        from src.taco_experiment.data import stratified_sample
        fake = self._make_fake_dataset()
        s1 = stratified_sample(fake, 20, seed=42)
        s2 = stratified_sample(fake, 20, seed=42)
        ids1 = [tid for tid, _ in s1]
        ids2 = [tid for tid, _ in s2]
        assert ids1 == ids2

    def test_returns_index_and_sample(self):
        from src.taco_experiment.data import stratified_sample
        fake = self._make_fake_dataset()
        samples = stratified_sample(fake, 5, seed=42)
        for idx, sample in samples:
            assert isinstance(idx, int)
            assert "difficulty" in sample
            assert "question" in sample


class TestDifficultyDistribution:
    def test_counts(self):
        from src.taco_experiment.data import get_difficulty_distribution
        samples = [
            (0, {"difficulty": "EASY"}),
            (1, {"difficulty": "EASY"}),
            (2, {"difficulty": "HARD"}),
        ]
        dist = get_difficulty_distribution(samples)
        assert dist["EASY"] == 2
        assert dist["HARD"] == 1
