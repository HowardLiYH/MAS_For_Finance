"""NeurIPS Export Utilities.

Generates publication-ready figures, tables, and reasoning traces
for the PopAgent paper submission.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None

from .experiment_logger import load_experiment, list_experiments


# =============================================================================
# Configuration
# =============================================================================

# NeurIPS paper style
NEURIPS_STYLE = {
    'figure.figsize': (6.0, 4.0),  # NeurIPS column width
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'serif',
    'text.usetex': False,  # Set True if LaTeX is available
}

ROLE_COLORS = {
    'analyst': '#22d3ee',
    'researcher': '#a78bfa',
    'trader': '#10b981',
    'risk': '#f59e0b',
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResults:
    """Processed results for export."""
    experiment_id: str
    iterations: List[Dict[str, Any]]
    summary: Dict[str, Any]

    # Computed metrics
    pnl_series: List[float]
    cumulative_pnl: List[float]
    method_popularity: Dict[str, Dict[str, float]]
    transfer_events: List[int]
    diversity_over_time: Dict[str, List[float]]


# =============================================================================
# Data Processing
# =============================================================================

def process_experiment(log_dir: str, experiment_id: str) -> ExperimentResults:
    """Load and process experiment data for export."""
    data = load_experiment(log_dir, experiment_id)

    iterations = data.get("iterations", [])
    summary = data.get("summary", {})

    # Extract PnL series
    pnl_series = [it.get("best_pnl", 0) for it in iterations]

    # Calculate cumulative PnL
    cumulative_pnl = []
    total = 0
    for pnl in pnl_series:
        total += pnl
        cumulative_pnl.append(total)

    # Find transfer events
    transfer_events = [
        it.get("iteration", i)
        for i, it in enumerate(iterations)
        if it.get("knowledge_transfer") is not None
    ]

    # Calculate method popularity over time
    method_popularity: Dict[str, Dict[str, float]] = {}
    for role in ['analyst', 'researcher', 'trader', 'risk']:
        method_popularity[role] = {}

    for it in iterations:
        for decision in it.get("agent_decisions", []):
            role = decision.get("role")
            if role in method_popularity:
                for method in decision.get("methods_selected", []):
                    method_popularity[role][method] = method_popularity[role].get(method, 0) + 1

    # Normalize to percentages
    for role in method_popularity:
        total = sum(method_popularity[role].values())
        if total > 0:
            method_popularity[role] = {m: c / total for m, c in method_popularity[role].items()}

    # Extract diversity over time
    diversity_over_time: Dict[str, List[float]] = {role: [] for role in ['analyst', 'researcher', 'trader', 'risk']}

    for it in iterations:
        dm = it.get("diversity_metrics", {})
        for role in diversity_over_time:
            if role in dm:
                diversity_over_time[role].append(dm[role].get("selection_diversity", 0))
            else:
                diversity_over_time[role].append(0)

    return ExperimentResults(
        experiment_id=experiment_id,
        iterations=iterations,
        summary=summary,
        pnl_series=pnl_series,
        cumulative_pnl=cumulative_pnl,
        method_popularity=method_popularity,
        transfer_events=transfer_events,
        diversity_over_time=diversity_over_time,
    )


# =============================================================================
# Figure Generation
# =============================================================================

def generate_learning_curve_figure(
    results: ExperimentResults,
    output_path: str,
    include_transfers: bool = True,
    window_size: int = 5,
) -> str:
    """
    Generate learning curve figure (Figure 1 in paper).

    Shows:
    - Best PnL over iterations
    - Moving average
    - Knowledge transfer markers
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for figure generation")

    plt.rcParams.update(NEURIPS_STYLE)

    fig, ax = plt.subplots()

    x = list(range(1, len(results.pnl_series) + 1))
    y = [p * 100 for p in results.pnl_series]  # Convert to percentage

    # Raw PnL line
    ax.plot(x, y, 'o-', markersize=2, alpha=0.4, color='#22d3ee', label='Best PnL per iteration')

    # Moving average
    if len(y) >= window_size:
        ma = []
        for i in range(len(y)):
            start = max(0, i - window_size + 1)
            ma.append(np.mean(y[start:i+1]))
        ax.plot(x, ma, '-', linewidth=2, color='#22d3ee', label=f'{window_size}-iteration MA')

    # Transfer markers
    if include_transfers and results.transfer_events:
        for te in results.transfer_events:
            if 0 < te <= len(y):
                ax.axvline(x=te, color='#a78bfa', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=-1, color='#a78bfa', linestyle='--', alpha=0.5, linewidth=1, label='Knowledge transfer')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best PnL (%)')
    ax.set_title('PopAgent Learning Curve')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_method_popularity_figure(
    results: ExperimentResults,
    output_path: str,
    top_n: int = 5,
) -> str:
    """
    Generate method popularity bar chart (Figure 2 in paper).

    Shows top N methods by usage for each role.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for figure generation")

    plt.rcParams.update(NEURIPS_STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    roles = ['analyst', 'researcher', 'trader', 'risk']

    for idx, role in enumerate(roles):
        ax = axes[idx]

        popularity = results.method_popularity.get(role, {})
        if not popularity:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(role.capitalize())
            continue

        # Get top N methods
        sorted_methods = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:top_n]
        methods = [m for m, _ in sorted_methods]
        values = [v * 100 for _, v in sorted_methods]  # Convert to percentage

        colors = [ROLE_COLORS[role]] * len(methods)

        bars = ax.barh(methods, values, color=colors, alpha=0.8)
        ax.set_xlabel('Usage (%)')
        ax.set_title(role.capitalize())
        ax.set_xlim(0, max(values) * 1.2 if values else 100)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=7)

    plt.suptitle('Method Popularity by Role', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_cumulative_returns_figure(
    results: ExperimentResults,
    output_path: str,
) -> str:
    """
    Generate cumulative returns figure (Figure 3 in paper).
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for figure generation")

    plt.rcParams.update(NEURIPS_STYLE)

    fig, ax = plt.subplots()

    x = list(range(1, len(results.cumulative_pnl) + 1))
    y = [p * 100 for p in results.cumulative_pnl]

    # Fill area
    ax.fill_between(x, 0, y, alpha=0.3, color='#10b981')
    ax.plot(x, y, '-', linewidth=2, color='#10b981')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('PopAgent Cumulative Performance')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_diversity_figure(
    results: ExperimentResults,
    output_path: str,
) -> str:
    """
    Generate selection diversity over time figure (Figure 4 in paper).
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for figure generation")

    plt.rcParams.update(NEURIPS_STYLE)

    fig, ax = plt.subplots()

    for role, color in ROLE_COLORS.items():
        diversity = results.diversity_over_time.get(role, [])
        if diversity:
            x = list(range(1, len(diversity) + 1))
            ax.plot(x, diversity, '-', linewidth=1.5, color=color, label=role.capitalize(), alpha=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Selection Diversity')
    ax.set_title('Method Selection Diversity Over Time')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# Table Generation
# =============================================================================

def generate_performance_table(
    results: ExperimentResults,
    output_path: str,
    format: str = "latex",
) -> str:
    """
    Generate performance metrics table (Table 1 in paper).

    Args:
        results: Processed experiment results
        output_path: Output file path
        format: "latex" or "csv"
    """
    if not results.pnl_series:
        return ""

    pnl = np.array(results.pnl_series)

    # Calculate metrics
    total_return = sum(pnl) * 100
    avg_return = np.mean(pnl) * 100
    std_return = np.std(pnl) * 100
    win_rate = (pnl > 0).sum() / len(pnl) * 100

    # Sharpe (annualized, assuming 4h bars)
    periods_per_year = 2190
    sharpe = np.sqrt(periods_per_year) * np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = np.min(drawdowns) * 100

    # First/last 10 comparison
    first_10_avg = np.mean(pnl[:10]) * 100 if len(pnl) >= 10 else avg_return
    last_10_avg = np.mean(pnl[-10:]) * 100 if len(pnl) >= 10 else avg_return
    improvement = last_10_avg - first_10_avg

    metrics = [
        ("Total Return", f"{total_return:.2f}%"),
        ("Avg Return/Iteration", f"{avg_return:.3f}%"),
        ("Std Dev", f"{std_return:.3f}%"),
        ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
        ("Win Rate", f"{win_rate:.1f}%"),
        ("Max Drawdown", f"{max_drawdown:.2f}%"),
        ("First 10 Avg", f"{first_10_avg:.3f}%"),
        ("Last 10 Avg", f"{last_10_avg:.3f}%"),
        ("Improvement", f"{improvement:.3f}%"),
        ("Knowledge Transfers", f"{len(results.transfer_events)}"),
        ("Total Iterations", f"{len(pnl)}"),
    ]

    if format == "latex":
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{PopAgent Performance Metrics}",
            "\\label{tab:performance}",
            "\\begin{tabular}{lr}",
            "\\toprule",
            "Metric & Value \\\\",
            "\\midrule",
        ]
        for name, value in metrics:
            lines.append(f"{name} & {value} \\\\")
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        content = "\n".join(lines)
    else:  # CSV
        content = "Metric,Value\n"
        content += "\n".join([f"{name},{value}" for name, value in metrics])

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def generate_method_usage_table(
    results: ExperimentResults,
    output_path: str,
    format: str = "latex",
    top_n: int = 5,
) -> str:
    """
    Generate method usage table (Table 2 in paper).
    """
    if format == "latex":
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Top Methods by Role}",
            "\\label{tab:methods}",
            "\\begin{tabular}{llr}",
            "\\toprule",
            "Role & Method & Usage \\\\",
            "\\midrule",
        ]

        for role in ['analyst', 'researcher', 'trader', 'risk']:
            popularity = results.method_popularity.get(role, {})
            sorted_methods = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:top_n]

            for i, (method, usage) in enumerate(sorted_methods):
                role_label = role.capitalize() if i == 0 else ""
                lines.append(f"{role_label} & {method} & {usage*100:.1f}\\% \\\\")

            if role != 'risk':
                lines.append("\\midrule")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        content = "\n".join(lines)
    else:  # CSV
        lines = ["Role,Method,Usage"]
        for role in ['analyst', 'researcher', 'trader', 'risk']:
            popularity = results.method_popularity.get(role, {})
            sorted_methods = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:top_n]
            for method, usage in sorted_methods:
                lines.append(f"{role},{method},{usage*100:.1f}%")
        content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


# =============================================================================
# Reasoning Traces Export
# =============================================================================

def export_reasoning_traces(
    results: ExperimentResults,
    output_path: str,
    max_traces: int = 10,
) -> str:
    """
    Export sample agent reasoning traces for paper appendix.
    """
    traces = []

    for it in results.iterations[:max_traces]:
        iteration_traces = {
            "iteration": it.get("iteration"),
            "timestamp": it.get("timestamp"),
            "market_context": it.get("market_context"),
            "agent_decisions": [],
        }

        for decision in it.get("agent_decisions", []):
            if decision.get("reasoning"):
                iteration_traces["agent_decisions"].append({
                    "agent_id": decision.get("agent_id"),
                    "role": decision.get("role"),
                    "methods_selected": decision.get("methods_selected"),
                    "reasoning": decision.get("reasoning"),
                    "exploration_used": decision.get("exploration_used"),
                })

        if iteration_traces["agent_decisions"]:
            traces.append(iteration_traces)

    with open(output_path, "w") as f:
        json.dump(traces, f, indent=2)

    return output_path


# =============================================================================
# Main Export Function
# =============================================================================

def export_for_neurips(
    log_dir: str,
    experiment_id: str,
    output_dir: str,
    formats: List[str] = ["pdf", "latex", "csv"],
) -> Dict[str, str]:
    """
    Generate all publication-ready outputs for NeurIPS submission.

    Args:
        log_dir: Directory containing experiment logs
        experiment_id: Experiment to export
        output_dir: Output directory for figures and tables
        formats: List of formats to generate

    Returns:
        Dict mapping output type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process experiment
    results = process_experiment(log_dir, experiment_id)

    outputs = {}

    # Generate figures
    if "pdf" in formats or "png" in formats:
        ext = "pdf" if "pdf" in formats else "png"

        try:
            outputs["learning_curve"] = generate_learning_curve_figure(
                results, str(output_path / f"figure1_learning_curve.{ext}")
            )
        except Exception as e:
            print(f"Warning: Could not generate learning curve: {e}")

        try:
            outputs["method_popularity"] = generate_method_popularity_figure(
                results, str(output_path / f"figure2_method_popularity.{ext}")
            )
        except Exception as e:
            print(f"Warning: Could not generate method popularity: {e}")

        try:
            outputs["cumulative_returns"] = generate_cumulative_returns_figure(
                results, str(output_path / f"figure3_cumulative_returns.{ext}")
            )
        except Exception as e:
            print(f"Warning: Could not generate cumulative returns: {e}")

        try:
            outputs["diversity"] = generate_diversity_figure(
                results, str(output_path / f"figure4_diversity.{ext}")
            )
        except Exception as e:
            print(f"Warning: Could not generate diversity figure: {e}")

    # Generate tables
    table_format = "latex" if "latex" in formats else "csv"

    try:
        outputs["performance_table"] = generate_performance_table(
            results, str(output_path / f"table1_performance.{table_format}"), format=table_format
        )
    except Exception as e:
        print(f"Warning: Could not generate performance table: {e}")

    try:
        outputs["method_table"] = generate_method_usage_table(
            results, str(output_path / f"table2_methods.{table_format}"), format=table_format
        )
    except Exception as e:
        print(f"Warning: Could not generate method table: {e}")

    # Export reasoning traces
    try:
        outputs["reasoning_traces"] = export_reasoning_traces(
            results, str(output_path / "appendix_reasoning_traces.json")
        )
    except Exception as e:
        print(f"Warning: Could not export reasoning traces: {e}")

    print(f"\n✅ NeurIPS export complete!")
    print(f"   Output directory: {output_path}")
    print(f"   Files generated: {len(outputs)}")
    for name, path in outputs.items():
        print(f"   - {name}: {path}")

    return outputs
