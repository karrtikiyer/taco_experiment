"""Generate a side-by-side comparison viewer for different decoding methods.

Shows each problem with ground truth solutions and generations from
top_p vs p-less (or p-less-norm) side by side.

Usage:
    PYTHONPATH=src uv run python scripts/view_comparison.py \
        --left full_100_7b --right full_100_7b_pless
    open results/comparison_full_100_7b_vs_full_100_7b_pless.html
"""

import ast
import json
import html
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taco_experiment.data import load_dataset_split, _has_image, SUPPORTED_DATASETS


def load_jsonl(path):
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data[item["task_id"]] = item
    return data


def load_execution(path):
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data[item["task_id"]] = item["results"]
    return data


def is_correct(gen_result):
    return all(
        (x is True) or (isinstance(x, (int, float)) and x > 0)
        for x in gen_result
    )


def escape(text):
    return html.escape(text)


def run_label(run_name, report):
    method = report.get("config", {}).get("decoding_method", "top_p")
    temp = report.get("config", {}).get("temperature", "?")
    top_p = report.get("config", {}).get("top_p", "?")
    if method == "top_p":
        return f"{run_name} (temp={temp}, top_p={top_p})"
    return f"{run_name} ({method}, temp={temp})"


def build_gen_html(solutions, exec_results, label):
    if not solutions:
        return '<div class="muted">No generations</div>'
    n_correct = 0
    items = ""
    for j, sol in enumerate(solutions):
        if j < len(exec_results):
            correct = is_correct(exec_results[j])
            if correct:
                n_correct += 1
            badge = '<span class="pass-badge">PASS</span>' if correct else '<span class="fail-badge">FAIL</span>'
            n_pass = sum(1 for x in exec_results[j] if (x is True) or (isinstance(x, (int, float)) and x > 0))
            n_total = len(exec_results[j])
            test_info = f'<span class="muted">({n_pass}/{n_total})</span>'
            border = "correct" if correct else "incorrect"
        else:
            badge = ""
            test_info = ""
            border = ""
        items += f"""<details class="solution-block {border}">
            <summary>#{j+1} {badge} {test_info} <span class="muted">({len(sol.splitlines())}L)</span></summary>
            <pre><code>{escape(sol)}</code></pre>
        </details>"""
    header = f'<span style="color:#22c55e">{n_correct}</span>/{len(solutions)} correct'
    return f'<div class="gen-header">{header}</div>{items}'


def main():
    parser = argparse.ArgumentParser(description="Side-by-side decoding comparison viewer")
    parser.add_argument("--left", type=str, required=True, help="Left run name (e.g. full_100_7b)")
    parser.add_argument("--right", type=str, required=True, help="Right run name (e.g. full_100_7b_pless)")
    parser.add_argument("--dataset", type=str, default="taco",
                        choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_root = Path(__file__).parent.parent / "results"
    left_dir = results_root / args.left
    right_dir = results_root / args.right

    with open(left_dir / "sample_meta.json") as f:
        meta = json.load(f)

    left_report = {}
    right_report = {}
    if (left_dir / "report.json").exists():
        with open(left_dir / "report.json") as f:
            left_report = json.load(f)
    if (right_dir / "report.json").exists():
        with open(right_dir / "report.json") as f:
            right_report = json.load(f)

    left_label = run_label(args.left, left_report)
    right_label = run_label(args.right, right_report)

    print(f"Loading {args.dataset.upper()} dataset...")
    dataset = load_dataset_split(args.dataset)

    left_gens = load_jsonl(left_dir / "generations.jsonl")
    right_gens = load_jsonl(right_dir / "generations.jsonl")
    left_exec = load_execution(left_dir / "execution.jsonl")
    right_exec = load_execution(right_dir / "execution.jsonl")

    print(f"Left:  {args.left} — {len(left_gens)} generations, {len(left_exec)} executed")
    print(f"Right: {args.right} — {len(right_gens)} generations, {len(right_exec)} executed")

    diff_colors = {
        "EASY": "#22c55e", "MEDIUM": "#eab308", "MEDIUM_HARD": "#f97316",
        "HARD": "#ef4444", "VERY_HARD": "#dc2626",
        "introductory": "#22c55e", "interview": "#eab308",
        "competition": "#ef4444",
    }

    cards = []
    for i, item in enumerate(meta):
        tid = item["task_id"]
        sample = dataset[tid]
        difficulty = sample["difficulty"]
        has_img = _has_image(sample["question"])
        badge_color = diff_colors.get(difficulty, "#6b7280")

        question = sample["question"]
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

        test_cases_html = ""
        for j, (inp, out) in enumerate(zip(inputs[:5], outputs[:5])):
            test_cases_html += f"""<div class="test-case">
                <div class="test-label">Test {j+1}</div>
                <div class="test-io">
                    <div><strong>Input:</strong><pre>{escape(str(inp))}</pre></div>
                    <div><strong>Expected:</strong><pre>{escape(str(out))}</pre></div>
                </div>
            </div>"""
        if len(inputs) > 5:
            test_cases_html += f'<div class="test-case muted">... and {len(inputs)-5} more test cases</div>'

        gt_html = ""
        for j, sol in enumerate(solutions[:5]):
            gt_html += f"""<details class="solution-block gt">
                <summary>GT #{j+1} <span class="muted">({len(sol.splitlines())}L)</span></summary>
                <pre><code>{escape(sol)}</code></pre>
            </details>"""
        if len(solutions) > 5:
            gt_html += f'<div class="muted">... +{len(solutions)-5} more</div>'

        starter_html = ""
        if starter_code and starter_code.strip():
            starter_html = f"""<div class="section">
                <h3>Starter Code</h3>
                <pre><code>{escape(starter_code)}</code></pre>
            </div>"""

        left_sols = left_gens.get(tid, {}).get("output", [])
        right_sols = right_gens.get(tid, {}).get("output", [])
        left_ex = left_exec.get(tid, [])
        right_ex = right_exec.get(tid, [])

        left_html = build_gen_html(left_sols, left_ex, args.left)
        right_html = build_gen_html(right_sols, right_ex, args.right)

        img_tag = ' <span class="badge img-badge">HAS IMAGE</span>' if has_img else ''
        fn_badge = f'<span class="badge fn-badge">fn: {escape(fn_name)}</span>' if fn_name else ''

        cards.append(f"""
        <div class="card {'has-image' if has_img else ''}" id="p-{tid}" data-difficulty="{difficulty}">
            <div class="card-header">
                <div class="card-title">
                    <span class="problem-num">#{i+1}</span> Task {tid}
                    <span class="badge" style="background:{badge_color}">{difficulty}</span>
                    <span class="badge io-badge">{io_type}</span>
                    {fn_badge}{img_tag}
                </div>
                <div class="card-meta">{len(solutions)} GT solutions &middot; {len(inputs)} test cases</div>
            </div>
            <div class="section">
                <h3>Problem Statement</h3>
                <div class="question">{escape(question)}</div>
            </div>
            {starter_html}
            <details class="section collapsible-section">
                <summary class="section-toggle">Test Cases <span class="muted">(showing up to 5)</span></summary>
                <div class="section-body">{test_cases_html}</div>
            </details>
            <details class="section collapsible-section">
                <summary class="section-toggle">Ground Truth ({len(solutions)} solutions)</summary>
                <div class="section-body">{gt_html}</div>
            </details>
            <div class="compare-columns">
                <div class="col">
                    <h3>{escape(left_label)}</h3>
                    {left_html}
                </div>
                <div class="col">
                    <h3>{escape(right_label)}</h3>
                    {right_html}
                </div>
            </div>
        </div>""")

    nav = ""
    for i, item in enumerate(meta):
        tid = item["task_id"]
        d = item["difficulty"]
        c = diff_colors.get(d, "#6b7280")
        nav += f'<a href="#p-{tid}" class="nav-item"><span class="nav-dot" style="background:{c}"></span>#{i+1}</a>\n'

    diff_counts = {}
    for item in meta:
        d = item["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1

    filter_btns = '<button class="filter-btn active" onclick="filterCards(\'all\')">All</button>'
    filter_btns += '<button class="filter-btn" onclick="filterCards(\'no-image\')">No Image</button>'
    for d in sorted(diff_counts.keys()):
        filter_btns += '<button class="filter-btn" onclick="filterCards(' + "'" + d + "'" + ')">' + d + '</button>'

    output_path = Path(args.output) if args.output else results_root / f"comparison_{args.left}_vs_{args.right}.html"

    html_content = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comparison: {escape(args.left)} vs {escape(args.right)}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; line-height:1.6; }}
.sidebar {{ position:fixed; left:0; top:0; width:180px; height:100vh; overflow-y:auto; background:#1e293b; padding:12px 8px; border-right:1px solid #334155; z-index:100; font-size:0.8em; }}
.sidebar h3 {{ color:#94a3b8; font-size:0.75em; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px; }}
.nav-item {{ display:flex; align-items:center; gap:4px; padding:2px 6px; color:#cbd5e1; text-decoration:none; border-radius:3px; }}
.nav-item:hover {{ background:#334155; }}
.nav-dot {{ width:6px; height:6px; border-radius:50%; flex-shrink:0; }}
.main {{ margin-left:190px; max-width:1600px; padding:20px; }}
.header {{ text-align:center; padding:30px 20px; border-bottom:1px solid #1e293b; margin-bottom:20px; }}
.header h1 {{ font-size:1.8em; color:#f8fafc; margin-bottom:6px; }}
.header .sub {{ color:#94a3b8; }}
.filter-bar {{ background:#1e293b; padding:10px 16px; border-radius:8px; margin-bottom:16px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.filter-btn {{ background:#334155; border:none; color:#cbd5e1; padding:5px 12px; border-radius:6px; cursor:pointer; font-size:0.85em; }}
.filter-btn:hover,.filter-btn.active {{ background:#4f46e5; color:white; }}
.card {{ background:#1e293b; border-radius:10px; margin-bottom:20px; overflow:hidden; border:1px solid #334155; }}
.card.has-image {{ border-left:3px solid #f59e0b; }}
.card-header {{ padding:14px 20px; background:#0f172a; border-bottom:1px solid #334155; }}
.card-title {{ font-size:1.1em; font-weight:600; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.card-meta {{ color:#64748b; font-size:0.85em; margin-top:4px; }}
.problem-num {{ color:#6366f1; font-weight:700; }}
.badge {{ padding:2px 8px; border-radius:10px; font-size:0.7em; font-weight:600; color:white; }}
.io-badge {{ background:#3b82f6; }}
.fn-badge {{ background:#8b5cf6; }}
.img-badge {{ background:#f59e0b; color:#000; }}
.section {{ padding:14px 20px; border-bottom:1px solid #334155; }}
.section h3 {{ color:#94a3b8; font-size:0.85em; margin-bottom:8px; }}
.question {{ white-space:pre-wrap; font-size:0.95em; background:#0f172a; padding:16px; border-radius:8px; max-height:400px; overflow-y:auto; }}
.collapsible-section {{ padding:0; }}
.section-toggle {{ padding:10px 20px; cursor:pointer; color:#94a3b8; font-size:0.85em; display:block; }}
.section-toggle:hover {{ background:#0f172a; }}
.section-body {{ padding:8px 20px 12px; }}
.test-case {{ background:#0f172a; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:0.85em; }}
.test-label {{ font-weight:600; color:#6366f1; margin-bottom:4px; }}
.test-io {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.test-io pre {{ margin:4px 0; font-size:0.9em; }}
.compare-columns {{ display:grid; grid-template-columns:1fr 1fr; }}
.col {{ padding:12px 16px; }}
.col:first-child {{ border-right:1px solid #334155; }}
.col h3 {{ color:#94a3b8; font-size:0.8em; text-transform:uppercase; letter-spacing:0.03em; margin-bottom:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.gen-header {{ font-size:0.9em; margin-bottom:8px; font-weight:600; }}
.solution-block {{ margin-bottom:4px; }}
.solution-block summary {{ cursor:pointer; padding:4px 8px; background:#0f172a; border-radius:4px; font-size:0.85em; border-left:3px solid #334155; }}
.solution-block summary:hover {{ background:#1a2744; }}
.solution-block.correct summary {{ border-left-color:#22c55e; background:#0f2918; }}
.solution-block.incorrect summary {{ border-left-color:#ef4444; }}
.solution-block.gt summary {{ border-left-color:#6366f1; }}
.solution-block pre {{ margin-top:4px; max-height:350px; overflow-y:auto; }}
pre {{ background:#0f172a; padding:10px; border-radius:4px; overflow-x:auto; font-size:0.8em; white-space:pre-wrap; word-break:break-word; }}
code {{ font-family:'SF Mono','Fira Code','Cascadia Code',monospace; }}
.pass-badge {{ background:#166534; color:#86efac; padding:1px 6px; border-radius:6px; font-size:0.7em; font-weight:700; }}
.fail-badge {{ background:#7f1d1d; color:#fca5a5; padding:1px 6px; border-radius:6px; font-size:0.7em; font-weight:700; }}
.muted {{ color:#64748b; font-size:0.85em; }}
@media(max-width:1200px) {{
    .sidebar {{ display:none; }}
    .main {{ margin-left:0; }}
    .compare-columns {{ grid-template-columns:1fr; }}
    .col:first-child {{ border-right:none; border-bottom:1px solid #334155; }}
}}
</style></head><body>
<div class="sidebar"><h3>Problems</h3>{nav}</div>
<div class="main">
<div class="header">
    <h1>Decoding Method Comparison</h1>
    <div class="sub"><strong>{escape(left_label)}</strong> vs <strong>{escape(right_label)}</strong></div>
</div>
<div class="filter-bar"><span class="muted">Filter:</span>{filter_btns}</div>
<div id="cards">{''.join(cards)}</div>
</div>
<script>
function filterCards(f) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.card').forEach(c => {{
        if (f === 'all') {{ c.style.display = ''; return; }}
        if (f === 'no-image') {{ c.style.display = c.classList.contains('has-image') ? 'none' : ''; return; }}
        c.style.display = c.dataset.difficulty === f ? '' : 'none';
    }});
}}
</script></body></html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\nSaved to: {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    main()
