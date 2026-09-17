"""Standalone, self-contained HTML evaluation report generator."""

from __future__ import annotations

import html

from urdu_eval.models import RunResult


def generate_html_report(run_result: RunResult) -> str:
    """Generate a responsive, standalone, self-contained HTML evaluation report."""
    meta = run_result.metadata

    # Metric score cards HTML
    metric_cards_html = ""
    for m_name, score in run_result.metrics.items():
        pct = f"{score * 100:.1f}%" if 0.0 <= score <= 1.0 else f"{score:.2f}"
        ci_str = ""
        if m_name in run_result.confidence_intervals:
            low, high = run_result.confidence_intervals[m_name]
            ci_str = f'<div class="metric-ci" style="font-size:11px; color:#38bdf8; margin-top:4px; font-weight:600;">95% CI: [{low * 100:.1f}% - {high * 100:.1f}%]</div>'
        metric_cards_html += f"""
        <div class="metric-card">
            <div class="metric-title">{html.escape(m_name.upper())}</div>
            <div class="metric-score">{pct}</div>
            {ci_str}
            <div class="metric-raw">Scalar: {score:.4f}</div>
        </div>
        """

    # Failure summary bars HTML
    total_samples = run_result.total_samples or 1
    failure_bars_html = ""
    for cat, count in sorted(run_result.failure_summary.items(), key=lambda x: -x[1]):
        pct_float = (count / total_samples) * 100.0
        bar_color = "#10b981" if cat == "correct" else "#f59e0b" if cat == "partial" else "#ef4444"
        failure_bars_html += f"""
        <div class="failure-row">
            <div class="failure-label">
                <span class="badge" style="background: {bar_color}20; color: {bar_color}; border: 1px solid {bar_color}50;">{html.escape(cat)}</span>
                <span class="failure-count">{count} ({pct_float:.1f}%)</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: {pct_float}%; background: {bar_color};"></div>
            </div>
        </div>
        """

    # Samples table rows HTML
    sample_rows_html = ""
    for s in run_result.samples:
        cat = s.failure_category.value if s.failure_category else "error"
        badge_cls = (
            "badge-correct"
            if cat == "correct"
            else "badge-partial"
            if cat == "partial"
            else "badge-error"
        )
        ref_text = s.reference if isinstance(s.reference, str) else " | ".join(s.reference)
        metrics_str = ", ".join(f"{k}: {v:.2f}" for k, v in s.metrics.items())

        sample_rows_html += f"""
        <tr class="sample-row" data-category="{html.escape(cat)}" data-search="{html.escape((s.prompt + " " + ref_text + " " + s.prediction).lower())}">
            <td class="id-col">{html.escape(s.sample_id)}</td>
            <td class="urdu-col">{html.escape(s.prompt)}</td>
            <td class="urdu-col">{html.escape(ref_text)}</td>
            <td class="urdu-col">{html.escape(s.prediction)}</td>
            <td><span class="badge {badge_cls}">{html.escape(cat)}</span></td>
            <td class="metrics-col">{html.escape(metrics_str)}</td>
            <td class="latency-col">{s.latency_ms:.0f}ms</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UrduEval Report - {html.escape(meta.model_settings.model)} on {html.escape(meta.benchmark.name)}</title>
    <style>
        :root {{
            --bg-base: #0f172a;
            --bg-card: #1e293b;
            --bg-elevated: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --border-color: #334155;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Nastaliq Urdu", "Noto Sans Arabic", sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 2rem 1.5rem;
        }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        .logo-title {{ display: flex; align-items: center; gap: 1rem; }}
        .logo-badge {{
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-emerald));
            color: #0f172a;
            font-weight: 800;
            font-size: 1.25rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }}
        .title h1 {{ font-size: 1.75rem; font-weight: 700; color: #fff; }}
        .title p {{ color: var(--text-muted); font-size: 0.875rem; }}
        .meta-pill {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        .meta-pill strong {{ color: #fff; }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-2px); border-color: var(--accent-cyan); }}
        .metric-title {{ color: var(--text-muted); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; }}
        .metric-score {{ font-size: 2.25rem; font-weight: 800; color: var(--accent-emerald); }}
        .metric-raw {{ font-size: 0.75rem; color: var(--text-muted); }}
        .section-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }}
        .section-header h2 {{ font-size: 1.25rem; font-weight: 600; }}
        .failure-row {{ margin-bottom: 0.75rem; }}
        .failure-label {{ display: flex; justify-content: space-between; font-size: 0.875rem; margin-bottom: 0.25rem; }}
        .progress-track {{ background: var(--bg-elevated); height: 8px; border-radius: 9999px; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 9999px; }}
        .badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}
        .badge-correct {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-partial {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-error {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .controls {{ display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .search-input {{
            flex: 1;
            min-width: 250px;
            background: var(--bg-base);
            border: 1px solid var(--border-color);
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
        }}
        .search-input:focus {{ outline: none; border-color: var(--accent-cyan); }}
        .filter-btn {{
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.875rem;
        }}
        .filter-btn.active {{ background: var(--accent-cyan); color: #0f172a; font-weight: 700; }}
        .table-responsive {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: var(--bg-elevated); color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        .urdu-col {{ font-family: "Noto Nastaliq Urdu", "Noto Sans Arabic", var(--font-family); font-size: 1rem; }}
        .id-col {{ font-family: monospace; color: var(--accent-cyan); font-size: 0.75rem; }}
        .metrics-col {{ font-family: monospace; font-size: 0.75rem; color: var(--text-muted); }}
        .latency-col {{ font-family: monospace; font-size: 0.75rem; color: var(--text-muted); }}
        footer {{ text-align: center; color: var(--text-muted); font-size: 0.75rem; margin-top: 3rem; border-top: 1px solid var(--border-color); padding-top: 1.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-title">
                <div class="logo-badge">اردو EVAL</div>
                <div class="title">
                    <h1>UrduEval Benchmark Report</h1>
                    <p>Model: <strong>{html.escape(meta.model_settings.model)}</strong> | Benchmark: <strong>{html.escape(meta.benchmark.name)}</strong></p>
                </div>
            </div>
            <div class="meta-pill">
                Run ID: <strong>{html.escape(run_result.run_id)}</strong> | Version: <strong>{html.escape(meta.urdu_eval_version)}</strong>
            </div>
        </header>

        <!-- Metrics Cards -->
        <div class="grid-cards">
            {metric_cards_html}
            <div class="metric-card">
                <div class="metric-title">TOTAL SAMPLES</div>
                <div class="metric-score" style="color: var(--accent-cyan);">{run_result.total_samples}</div>
                <div class="metric-raw">Failed: {run_result.failed_samples}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">MEAN LATENCY</div>
                <div class="metric-score" style="color: var(--accent-amber);">{run_result.mean_latency_ms:.0f}<span style="font-size: 1rem;">ms</span></div>
                <div class="metric-raw">Platform: {html.escape(meta.platform)}</div>
            </div>
        </div>

        <!-- Failure Analysis -->
        <div class="section-box">
            <div class="section-header">
                <h2>Error & Linguistic Analysis</h2>
            </div>
            {failure_bars_html}
        </div>

        <!-- Sample Inspector -->
        <div class="section-box">
            <div class="section-header">
                <h2>Sample-Level Inspector ({run_result.total_samples} records)</h2>
            </div>
            <div class="controls">
                <input type="text" id="searchInput" class="search-input" placeholder="Search prompts, references, or outputs...">
                <button class="filter-btn active" onclick="filterCategory('all')">All</button>
                <button class="filter-btn" onclick="filterCategory('correct')">Correct</button>
                <button class="filter-btn" onclick="filterCategory('incorrect')">Incorrect / Failed</button>
            </div>
            <div class="table-responsive">
                <table id="samplesTable">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Prompt</th>
                            <th>Reference</th>
                            <th>Model Prediction</th>
                            <th>Category</th>
                            <th>Metrics</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sample_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            Generated by UrduEval v{html.escape(meta.urdu_eval_version)} on {html.escape(meta.timestamp)} • Zero Telemetry Local Evaluation
        </footer>
    </div>

    <script>
        let currentFilter = 'all';

        function filterCategory(cat) {{
            currentFilter = cat;
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent.toLowerCase().includes(cat));
            }});
            applyFilters();
        }}

        function applyFilters() {{
            const search = document.getElementById('searchInput').value.toLowerCase().trim();
            const rows = document.querySelectorAll('.sample-row');

            rows.forEach(row => {{
                const cat = row.getAttribute('data-category');
                const text = row.getAttribute('data-search');

                const matchesCat = (currentFilter === 'all') ||
                    (currentFilter === 'correct' && cat === 'correct') ||
                    (currentFilter === 'incorrect' && cat !== 'correct');

                const matchesSearch = !search || text.includes(search);

                row.style.display = (matchesCat && matchesSearch) ? '' : 'none';
            }});
        }}

        document.getElementById('searchInput').addEventListener('input', applyFilters);
    </script>
</body>
</html>
"""
    return html_content
