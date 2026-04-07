"""Generate a browsable HTML viewer for TACO problems used in our experiments.

Shows each problem's question, difficulty, test cases, ground truth solutions,
and (optionally) our model's generated solutions side-by-side.

Usage:
    uv run python scripts/view_problems.py --run-name full_100_7b
    open results/full_100_7b/problem_viewer.html
"""

import ast
import json
import html
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taco_experiment.data import load_taco_test


def load_generations(run_dir: Path) -> dict:
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        return {}
    gens = {}
    with open(gen_path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                gens[item["task_id"]] = item
    return gens


def load_execution_results(run_dir: Path) -> dict:
    """Load execution results. Returns {task_id: list of per-generation result lists}."""
    exec_path = run_dir / "execution.jsonl"
    if not exec_path.exists():
        return {}
    results = {}
    with open(exec_path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                results[item["task_id"]] = item["results"]
    return results


def is_generation_correct(gen_result: list) -> bool:
    """A generation is correct if ALL test case results are positive (True or > 0)."""
    return all(
        (x is True) or (isinstance(x, (int, float)) and x > 0)
        for x in gen_result
    )


def escape(text: str) -> str:
    return html.escape(text)


def build_html(problems: list, generations: dict, execution: dict, run_name: str) -> str:
    cards = []
    for i, (task_id, sample) in enumerate(problems):
        question = sample["question"]
        difficulty = sample["difficulty"]
        starter_code = sample.get("starter_code", "")

        try:
            solutions = json.loads(sample["solutions"])
        except (json.JSONDecodeError, TypeError):
            try:
                solutions = ast.literal_eval(sample["solutions"])
            except Exception:
                solutions = []

        try:
            io_data = json.loads(sample["input_output"])
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])
            fn_name = io_data.get("fn_name", None)
        except (json.JSONDecodeError, TypeError):
            inputs, outputs, fn_name = [], [], None

        io_type = "Call-based" if fn_name else "Standard I/O"

        gen_data = generations.get(task_id, {})
        gen_solutions = gen_data.get("output", [])
        exec_results = execution.get(task_id, [])

        diff_colors = {
            "EASY": "#22c55e",
            "MEDIUM": "#eab308",
            "MEDIUM_HARD": "#f97316",
            "HARD": "#ef4444",
            "VERY_HARD": "#dc2626",
        }
        badge_color = diff_colors.get(difficulty, "#6b7280")

        test_cases_html = ""
        for j, (inp, out) in enumerate(zip(inputs[:5], outputs[:5])):
            test_cases_html += f"""
            <div class="test-case">
                <div class="test-label">Test {j+1}</div>
                <div class="test-io">
                    <div><strong>Input:</strong><pre>{escape(str(inp))}</pre></div>
                    <div><strong>Expected:</strong><pre>{escape(str(out))}</pre></div>
                </div>
            </div>"""
        if len(inputs) > 5:
            test_cases_html += f'<div class="test-case muted">... and {len(inputs)-5} more test cases</div>'

        gt_solutions_html = ""
        for j, sol in enumerate(solutions[:10]):
            gt_solutions_html += f"""
            <details class="solution-block">
                <summary>Ground Truth #{j+1} <span class="muted">({len(sol.splitlines())} lines)</span></summary>
                <pre><code>{escape(sol)}</code></pre>
            </details>"""
        if len(solutions) > 10:
            gt_solutions_html += f'<div class="muted">... and {len(solutions)-10} more solutions</div>'

        gen_solutions_html = ""
        n_correct = 0
        if gen_solutions:
            for j, sol in enumerate(gen_solutions):
                if j < len(exec_results):
                    correct = is_generation_correct(exec_results[j])
                    if correct:
                        n_correct += 1
                    status_badge = ('<span class="pass-badge">PASS</span>' if correct
                                    else '<span class="fail-badge">FAIL</span>')
                    n_pass = sum(1 for x in exec_results[j]
                                 if (x is True) or (isinstance(x, (int, float)) and x > 0))
                    n_total = len(exec_results[j])
                    test_detail = f'<span class="muted">({n_pass}/{n_total} tests)</span>'
                    border_class = "gen correct" if correct else "gen incorrect"
                else:
                    status_badge = '<span class="muted">no exec</span>'
                    test_detail = ""
                    border_class = "gen"
                gen_solutions_html += f"""
                <details class="solution-block {border_class}">
                    <summary>Generated #{j+1} {status_badge} {test_detail} <span class="muted">({len(sol.splitlines())} lines)</span></summary>
                    <pre><code>{escape(sol)}</code></pre>
                </details>"""
        else:
            gen_solutions_html = '<div class="muted">No generations available for this problem</div>'

        correctness_summary = ""
        if exec_results:
            correctness_summary = f' &middot; <span style="color:#22c55e">{n_correct} correct</span> / {len(gen_solutions)} generations'

        starter_html = ""
        if starter_code and starter_code.strip():
            starter_html = f"""
            <div class="section">
                <h3>Starter Code</h3>
                <pre><code>{escape(starter_code)}</code></pre>
            </div>"""

        cards.append(f"""
        <div class="card" id="problem-{task_id}">
            <div class="card-header">
                <div class="card-title">
                    <span class="problem-num">#{i+1}</span>
                    Task ID: {task_id}
                    <span class="badge" style="background:{badge_color}">{difficulty}</span>
                    <span class="badge io-badge">{io_type}</span>
                    {f'<span class="badge fn-badge">fn: {escape(fn_name)}</span>' if fn_name else ''}
                </div>
                <div class="card-meta">
                    {len(solutions)} GT solutions &middot; {len(inputs)} test cases
                    {correctness_summary if exec_results else (f' &middot; {len(gen_solutions)} generations' if gen_solutions else '')}
                </div>
            </div>
            <div class="section">
                <h3>Problem Statement</h3>
                <div class="question">{escape(question)}</div>
            </div>
            {starter_html}
            <div class="section">
                <h3>Test Cases <span class="muted">(showing up to 5)</span></h3>
                {test_cases_html}
            </div>
            <div class="columns">
                <div class="col">
                    <h3>Ground Truth Solutions <span class="muted">({len(solutions)} total)</span></h3>
                    {gt_solutions_html}
                </div>
                <div class="col">
                    <h3>Model Generations <span class="muted">({len(gen_solutions)} total)</span></h3>
                    {gen_solutions_html}
                </div>
            </div>
        </div>""")

    difficulty_counts = {}
    for _, s in problems:
        d = s["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    nav_items = ""
    for i, (task_id, sample) in enumerate(problems):
        badge_color = diff_colors.get(sample["difficulty"], "#6b7280")
        nav_items += f'<a href="#problem-{task_id}" class="nav-item"><span class="nav-dot" style="background:{badge_color}"></span>#{i+1} (ID {task_id})</a>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TACO Problems — {run_name}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#0f172a; color:#e2e8f0; line-height:1.6; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}
.header {{ text-align:center; padding:40px 20px; border-bottom:1px solid #1e293b; margin-bottom:30px; }}
.header h1 {{ font-size:2em; color:#f8fafc; margin-bottom:8px; }}
.header .subtitle {{ color:#94a3b8; font-size:1.1em; }}
.stats {{ display:flex; gap:20px; justify-content:center; margin-top:20px; flex-wrap:wrap; }}
.stat {{ background:#1e293b; padding:8px 16px; border-radius:8px; }}
.sidebar {{ position:fixed; left:0; top:0; width:220px; height:100vh; overflow-y:auto; background:#1e293b; padding:16px 12px; border-right:1px solid #334155; z-index:100; }}
.sidebar h3 {{ color:#94a3b8; font-size:0.8em; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px; }}
.nav-item {{ display:flex; align-items:center; gap:6px; padding:4px 8px; color:#cbd5e1; text-decoration:none; font-size:0.85em; border-radius:4px; margin-bottom:2px; }}
.nav-item:hover {{ background:#334155; }}
.nav-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.main {{ margin-left:230px; }}
.filter-bar {{ background:#1e293b; padding:12px 20px; border-radius:8px; margin-bottom:20px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
.filter-btn {{ background:#334155; border:none; color:#cbd5e1; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.9em; }}
.filter-btn:hover, .filter-btn.active {{ background:#4f46e5; color:white; }}
.card {{ background:#1e293b; border-radius:12px; margin-bottom:24px; overflow:hidden; border:1px solid #334155; }}
.card-header {{ padding:20px 24px; background:#0f172a; border-bottom:1px solid #334155; }}
.card-title {{ font-size:1.2em; font-weight:600; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.card-meta {{ color:#64748b; font-size:0.9em; margin-top:6px; }}
.problem-num {{ color:#6366f1; font-weight:700; }}
.badge {{ padding:3px 10px; border-radius:12px; font-size:0.75em; font-weight:600; color:white; }}
.io-badge {{ background:#3b82f6; }}
.fn-badge {{ background:#8b5cf6; }}
.section {{ padding:16px 24px; border-bottom:1px solid #1e293b; }}
.section h3 {{ color:#94a3b8; font-size:0.9em; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px; }}
.question {{ white-space:pre-wrap; font-size:0.95em; background:#0f172a; padding:16px; border-radius:8px; max-height:400px; overflow-y:auto; }}
pre {{ background:#0f172a; padding:12px; border-radius:6px; overflow-x:auto; font-size:0.85em; white-space:pre-wrap; word-break:break-word; }}
code {{ font-family:'SF Mono', 'Fira Code', 'Cascadia Code', monospace; }}
.test-case {{ margin-bottom:8px; }}
.test-label {{ font-weight:600; color:#6366f1; font-size:0.85em; margin-bottom:4px; }}
.test-io {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.test-io pre {{ font-size:0.8em; max-height:100px; overflow-y:auto; }}
.columns {{ display:grid; grid-template-columns:1fr 1fr; gap:0; }}
.col {{ padding:16px 24px; }}
.col:first-child {{ border-right:1px solid #334155; }}
.col h3 {{ color:#94a3b8; font-size:0.9em; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px; }}
.solution-block {{ margin-bottom:6px; }}
.solution-block summary {{ cursor:pointer; padding:6px 10px; background:#0f172a; border-radius:6px; font-size:0.9em; }}
.solution-block summary:hover {{ background:#1a2744; }}
.solution-block.gen summary {{ border-left:3px solid #6366f1; }}
.solution-block.correct summary {{ border-left:3px solid #22c55e; background:#0f2918; }}
.solution-block.incorrect summary {{ border-left:3px solid #ef4444; }}
.pass-badge {{ background:#166534; color:#86efac; padding:1px 8px; border-radius:8px; font-size:0.75em; font-weight:700; letter-spacing:0.03em; }}
.fail-badge {{ background:#7f1d1d; color:#fca5a5; padding:1px 8px; border-radius:8px; font-size:0.75em; font-weight:700; letter-spacing:0.03em; }}
.solution-block pre {{ margin-top:6px; max-height:400px; overflow-y:auto; }}
.muted {{ color:#64748b; font-size:0.85em; }}
@media (max-width:1200px) {{
    .sidebar {{ display:none; }}
    .main {{ margin-left:0; }}
    .columns {{ grid-template-columns:1fr; }}
    .col:first-child {{ border-right:none; border-bottom:1px solid #334155; }}
}}
</style>
</head>
<body>
<div class="sidebar">
    <h3>Problems</h3>
    {nav_items}
</div>
<div class="main">
<div class="container">
    <div class="header">
        <h1>TACO Problem Viewer</h1>
        <div class="subtitle">Run: <strong>{escape(run_name)}</strong> &middot; {len(problems)} problems</div>
        <div class="stats">
            {''.join(f'<div class="stat"><span style="color:{diff_colors.get(d,"#6b7280")}">{d}</span>: {c}</div>' for d, c in sorted(difficulty_counts.items()))}
        </div>
    </div>
    <div class="filter-bar">
        <span class="muted">Filter:</span>
        <button class="filter-btn active" onclick="filterProblems('all')">All</button>
        {''.join('<button class="filter-btn" onclick="filterProblems(' + "'" + d + "'" + ')">' + d + '</button>' for d in sorted(difficulty_counts.keys()))}
    </div>
    <div id="problems">
        {''.join(cards)}
    </div>
</div>
</div>
<script>
function filterProblems(difficulty) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.card').forEach(card => {{
        if (difficulty === 'all') {{ card.style.display = ''; return; }}
        const badge = card.querySelector('.badge');
        card.style.display = badge && badge.textContent.trim() === difficulty ? '' : 'none';
    }});
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML viewer for TACO experiment problems")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the run directory under results/")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: results/<run-name>/problem_viewer.html)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results" / args.run_name
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    meta_path = results_dir / "sample_meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist")
        sys.exit(1)

    with open(meta_path) as f:
        sample_meta = json.load(f)

    task_ids = {item["task_id"] for item in sample_meta}

    print(f"Loading TACO dataset...")
    dataset = load_taco_test()

    problems = []
    for meta in sample_meta:
        tid = meta["task_id"]
        sample = dataset[tid]
        problems.append((tid, sample))

    print(f"Loaded {len(problems)} problems")

    generations = load_generations(results_dir)
    print(f"Loaded generations for {len(generations)} problems")

    execution = load_execution_results(results_dir)
    print(f"Loaded execution results for {len(execution)} problems")

    output_path = Path(args.output) if args.output else results_dir / "problem_viewer.html"
    html_content = build_html(problems, generations, execution, args.run_name)

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Viewer saved to: {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    main()
