"""Tests for CodeBLEU diversity metrics."""

import pytest


class TestCodebleuScore:
    def test_identical_code_scores_high(self):
        from src.taco_experiment.diversity import codebleu_score
        code = "def add(a, b):\n    return a + b"
        score = codebleu_score(predictions=[code], references=[[code]])
        assert score > 0.9, f"Identical code should score >0.9, got {score}"

    def test_different_code_scores_lower(self):
        from src.taco_experiment.diversity import codebleu_score
        code1 = "def add(a, b):\n    return a + b"
        code2 = "import sys\nfor line in sys.stdin:\n    print(int(line))"
        score = codebleu_score(predictions=[code1], references=[[code2]])
        assert score < 0.9, f"Different code should score <0.9, got {score}"

    def test_score_in_range(self):
        from src.taco_experiment.diversity import codebleu_score
        code1 = "x = 1"
        code2 = "y = 2"
        score = codebleu_score(predictions=[code1], references=[[code2]])
        assert 0.0 <= score <= 1.0


class TestQualityVsGroundTruth:
    def test_returns_mean_and_per_sample(self):
        from src.taco_experiment.diversity import quality_vs_ground_truth
        gens = ["def f():\n    return 1", "def g():\n    return 2"]
        gts = ["def f():\n    return 1"]
        result = quality_vs_ground_truth(gens, gts)
        assert "mean" in result
        assert "per_sample" in result
        assert len(result["per_sample"]) == 2

    def test_empty_ground_truth(self):
        from src.taco_experiment.diversity import quality_vs_ground_truth
        result = quality_vs_ground_truth(["code"], [])
        assert result["mean"] == 0.0


class TestSelfCodebleu:
    def test_identical_samples_high_self_bleu(self):
        from src.taco_experiment.diversity import self_codebleu
        code = "def solve(n):\n    return n * 2"
        result = self_codebleu([code, code, code])
        assert result["mean"] > 0.8, f"Identical samples should have high self-CodeBLEU, got {result['mean']}"

    def test_single_sample_returns_zero(self):
        from src.taco_experiment.diversity import self_codebleu
        result = self_codebleu(["def f(): pass"])
        assert result["mean"] == 0.0


class TestGtMaxRecall:
    def test_identical_gen_scores_high(self):
        from src.taco_experiment.diversity import gt_max_recall
        code = "def f():\n    return 1"
        result = gt_max_recall([code], [code])
        assert result["mean"] > 0.9, f"Identical code should score >0.9, got {result['mean']}"
        assert result["n_gt"] == 1

    def test_multiple_gts(self):
        from src.taco_experiment.diversity import gt_max_recall
        gens = ["def f():\n    return 1"] * 5
        gts = ["def f():\n    return 1", "def g():\n    return 2", "def h():\n    return 3"]
        result = gt_max_recall(gens, gts)
        assert 0.0 <= result["mean"] <= 1.0
        assert len(result["per_gt"]) == 3
        assert result["n_gt"] == 3

    def test_empty_inputs(self):
        from src.taco_experiment.diversity import gt_max_recall
        result = gt_max_recall([], [])
        assert result["mean"] == 0.0


class TestParseGroundTruth:
    def test_valid_json_solutions(self):
        from src.taco_experiment.diversity import parse_ground_truth_solutions
        import json
        solutions = ["def a(): pass", "def b(): pass"]
        sample = {"solutions": json.dumps(solutions)}
        parsed = parse_ground_truth_solutions(sample)
        assert len(parsed) == 2

    def test_max_solutions_cap(self):
        from src.taco_experiment.diversity import parse_ground_truth_solutions
        import json
        solutions = [f"def f{i}(): pass" for i in range(50)]
        sample = {"solutions": json.dumps(solutions)}
        parsed = parse_ground_truth_solutions(sample, max_solutions=10)
        assert len(parsed) == 10

    def test_invalid_json_returns_empty(self):
        from src.taco_experiment.diversity import parse_ground_truth_solutions
        sample = {"solutions": "not json"}
        parsed = parse_ground_truth_solutions(sample)
        assert parsed == []
