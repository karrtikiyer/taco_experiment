"""Tests for pass@k computation and check_correctness."""

import json
import multiprocessing
import pytest
import numpy as np


class TestCheckCorrectness:
    """Tests for the Pipe-based check_correctness function."""

    def _make_call_based_sample(self, fn_name="add", inputs=None, outputs=None):
        """Helper to build a minimal call-based TACO sample dict."""
        if inputs is None:
            inputs = [[1, 2], [3, 4], [0, 0]]
        if outputs is None:
            outputs = [3, 7, 0]
        return {
            "input_output": json.dumps({
                "fn_name": fn_name,
                "inputs": inputs,
                "outputs": outputs,
            })
        }

    def test_correct_solution_returns_all_true(self):
        from src.taco_experiment.execute import check_correctness
        sample = self._make_call_based_sample()
        code = "def add(a, b):\n    return a + b"
        result = check_correctness(sample, code)
        assert all(r is True for r in result), f"Expected all True, got {result}"

    def test_wrong_solution_returns_false(self):
        from src.taco_experiment.execute import check_correctness
        sample = self._make_call_based_sample()
        code = "def add(a, b):\n    return a - b"
        result = check_correctness(sample, code)
        assert not all(r is True for r in result)

    def test_timeout_returns_negative_ones(self):
        """Per-test-case TIMEOUT=4s inside testing_util kills sleeping code."""
        from src.taco_experiment.execute import check_correctness
        sample = self._make_call_based_sample()
        code = "import time\ndef add(a, b):\n    time.sleep(60)\n    return a + b"
        result = check_correctness(sample, code)
        assert not any(r is True for r in result), f"Expected no True results for sleeping code, got {result}"

    def test_no_zombie_processes(self):
        """Verify that check_correctness doesn't leak child processes."""
        from src.taco_experiment.execute import check_correctness
        import psutil

        proc = psutil.Process()
        children_before = len(proc.children(recursive=True))

        sample = self._make_call_based_sample()
        code = "def add(a, b):\n    return a + b"
        for _ in range(5):
            check_correctness(sample, code)

        children_after = len(proc.children(recursive=True))
        leaked = children_after - children_before
        assert leaked == 0, f"Leaked {leaked} child processes after 5 runs"


class TestEstimatePassAtK:
    def test_all_correct(self):
        from src.taco_experiment.execute import estimate_pass_at_k
        # 10 samples, 10 correct → pass@1 = 1.0
        assert estimate_pass_at_k(10, 10, 1) == 1.0
        assert estimate_pass_at_k(10, 10, 10) == 1.0

    def test_none_correct(self):
        from src.taco_experiment.execute import estimate_pass_at_k
        # 10 samples, 0 correct → pass@k = 0.0 for any k
        assert estimate_pass_at_k(10, 0, 1) == 0.0
        assert estimate_pass_at_k(10, 0, 10) == 0.0

    def test_half_correct_pass_at_1(self):
        from src.taco_experiment.execute import estimate_pass_at_k
        # 10 samples, 5 correct → pass@1 = 0.5
        result = estimate_pass_at_k(10, 5, 1)
        assert abs(result - 0.5) < 1e-6

    def test_one_correct_pass_at_10(self):
        from src.taco_experiment.execute import estimate_pass_at_k
        # 10 samples, 1 correct → pass@10 = 1.0 (guaranteed to pick it)
        assert estimate_pass_at_k(10, 1, 10) == 1.0

    def test_monotonic_in_k(self):
        from src.taco_experiment.execute import estimate_pass_at_k
        # pass@k should increase with k
        scores = [estimate_pass_at_k(10, 3, k) for k in [1, 3, 5, 8, 10]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


class TestComputePassAtK:
    def test_perfect_results(self):
        from src.taco_experiment.execute import compute_pass_at_k
        # Two problems, all tests pass for all generations
        exec_results = {
            0: [[True, True], [True, True], [True, True]],
            1: [[True], [True], [True]],
        }
        metrics = compute_pass_at_k(exec_results, k_list=[1, 3])
        assert metrics["summary"]["pass@1"] == 1.0
        assert metrics["summary"]["pass@3"] == 1.0

    def test_no_correct_results(self):
        from src.taco_experiment.execute import compute_pass_at_k
        exec_results = {
            0: [[False, True], [False, False]],
            1: [[-1], [-2]],
        }
        metrics = compute_pass_at_k(exec_results, k_list=[1, 2])
        assert metrics["summary"]["pass@1"] == 0.0
        assert metrics["summary"]["pass@2"] == 0.0

    def test_detail_per_problem(self):
        from src.taco_experiment.execute import compute_pass_at_k
        exec_results = {
            0: [[True, True], [True, True]],
            1: [[False], [False]],
        }
        metrics = compute_pass_at_k(exec_results, k_list=[1])
        detail = metrics["detail"]["pass@1"]
        assert detail[0] == 1.0
        assert detail[1] == 0.0
