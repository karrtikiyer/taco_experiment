"""Code execution and pass@k computation using TACO's testing framework."""

import gc
import json
import itertools
import multiprocessing
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from .metrics.testing_util import run_test
from .config import K_VALUES, MEMORY_WARNING_MB

multiprocessing.set_start_method("fork", force=True)


def _run_test_worker(sample, generation, debug, conn):
    """Run test in a child process, send result back via Pipe."""
    try:
        result = run_test(sample, test=generation, debug=debug)
        conn.send(result)
    except Exception:
        conn.send(None)
    finally:
        conn.close()


def check_correctness(sample, generation, debug=False):
    """Check correctness of a single generation against all test cases.

    Uses Pipe instead of Manager to avoid leaking server processes.
    No global timeout -- relies on TACO's per-test-case TIMEOUT=4s
    (signal.alarm for call-based, subprocess timeout for stdin-based).
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_run_test_worker,
        args=(sample, generation, debug, child_conn),
    )
    p.start()
    child_conn.close()

    parent_conn.poll(None)
    result = parent_conn.recv()

    if p.is_alive():
        p.kill()
    p.join(timeout=5)
    parent_conn.close()

    if not result:
        try:
            in_outs = json.loads(sample["input_output"])
            result = [-1 for _ in range(len(in_outs["inputs"]))]
        except Exception:
            result = [-1]
        if debug:
            print("child returned no result")
    return result


def evaluate_problem(task_id, sample, generations, debug=False):
    """Evaluate all generations for a single problem. Returns list of per-generation results."""
    results = []
    for gen in generations:
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, gen, debug=debug)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            if debug:
                print(f"Task {task_id} exception: {repr(e)}")
        results.append(curr_res)
    return results


def estimate_pass_at_k(n, c, k):
    """Standard Codex pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(execution_results, k_list=K_VALUES):
    """Compute pass@k across all problems.

    Args:
        execution_results: dict mapping task_id -> list of per-generation result lists
        k_list: list of k values

    Returns:
        dict with 'summary' (averaged pass@k) and 'detail' (per-problem pass@k)
    """
    totals = []
    corrects = []
    task_ids = []

    for task_id, gen_results in execution_results.items():
        all_correct = []
        for gen_result in gen_results:
            gen_arr = np.array(gen_result)
            all_correct.append(bool(np.all(gen_arr > 0)))
        task_ids.append(task_id)
        totals.append(len(all_correct))
        corrects.append(sum(all_correct))

    totals = np.array(totals)
    corrects = np.array(corrects)

    summary = {}
    detail = {}
    for k in k_list:
        if (totals >= k).all():
            per_problem = np.array([
                estimate_pass_at_k(int(n), int(c), k)
                for n, c in zip(totals, corrects)
            ])
            summary[f"pass@{k}"] = float(per_problem.mean())
            detail[f"pass@{k}"] = dict(zip(task_ids, per_problem.tolist()))

    return {"summary": summary, "detail": detail}


def load_existing_execution(exec_path):
    """Load already-completed execution results from JSONL checkpoint."""
    completed = {}
    exec_path = Path(exec_path)
    if exec_path.exists():
        with open(exec_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    completed[item["task_id"]] = item["results"]
    return completed


def _check_memory():
    """Log RSS and trigger gc.collect if above warning threshold."""
    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        if rss_mb > MEMORY_WARNING_MB:
            print(f"  WARNING: RSS={rss_mb:.0f}MB exceeds {MEMORY_WARNING_MB}MB, running gc.collect()")
            gc.collect()
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            print(f"  After gc: RSS={rss_mb:.0f}MB")
        return rss_mb
    except Exception:
        return 0.0


def run_evaluation(generation_results, dataset, debug=False, checkpoint_path=None):
    """Run execution-based evaluation on all generated code with incremental checkpointing.

    Args:
        generation_results: list of dicts with 'task_id' and 'output' keys
        dataset: dict or HF dataset mapping task_id -> sample
        checkpoint_path: path to JSONL file for incremental saving

    Returns:
        execution_results dict and pass@k metrics dict
    """
    execution_results = {}
    if checkpoint_path:
        execution_results = load_existing_execution(checkpoint_path)
        if execution_results:
            print(f"  Resuming: {len(execution_results)} problems already evaluated")

    for gen_item in tqdm(generation_results, desc="Evaluating"):
        task_id = gen_item["task_id"]
        if task_id in execution_results:
            continue

        sample = dataset[task_id]
        generations = gen_item["output"]

        results = evaluate_problem(task_id, sample, generations, debug=debug)
        execution_results[task_id] = results

        n_correct = sum(1 for r in results if all(x is True or (isinstance(x, (int, float)) and x > 0) for x in r))
        rss_mb = _check_memory()
        print(f"  [exec] task_id={task_id} correct={n_correct}/{len(results)} rss={rss_mb:.0f}MB")

        if checkpoint_path:
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "results": results}, default=str) + "\n")

    metrics = compute_pass_at_k(execution_results)
    return execution_results, metrics
