from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import ensure_dir


PUBLICATION_COLORS = [
    "#1f3b73",
    "#2a6f97",
    "#5390d9",
    "#52b69a",
    "#76c893",
    "#f4a261",
    "#e76f51",
    "#b56576",
]

METHOD_PALETTE = {
    "few_shot": "#1f3b73",
    "zero_shot": "#5390d9",
    "hardware_agnostic": "#f4a261",
    "random_search": "#9AA0A6",
    "blackbox_cost_mlp": "#e76f51",
    "standard_cnn": "#2a9d8f",
    "shallow_wide": "#b56576",
    "deep_narrow": "#6a4c93",
    "depthwise_heavy_mixed_precision": "#ef476f",
    "mobilenet_like": "#52b69a",
}


def _apply_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.prop_cycle": matplotlib.cycler(color=PUBLICATION_COLORS),
        }
    )


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output


def _style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")
    ax.tick_params(colors="#344054")
    ax.title.set_color("#101828")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")


def _method_color(name: str) -> str:
    return METHOD_PALETTE.get(name, PUBLICATION_COLORS[0])


def plot_training_curve(frame: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str) -> Path:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame[x], frame[y], marker="o", linewidth=2.2, markersize=5.5, color=PUBLICATION_COLORS[0])
    ax.fill_between(frame[x], frame[y], alpha=0.12, color=PUBLICATION_COLORS[0])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _style_axis(ax)
    return _save_figure(fig, output_path)


def plot_grouped_bars(frame: pd.DataFrame, category: str, value: str, group: str, output_path: str | Path, title: str) -> Path:
    _apply_publication_style()
    pivot = frame.pivot_table(index=category, columns=group, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    pivot.plot(kind="bar", ax=ax, width=0.82, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(value)
    ax.set_xlabel("")
    _style_axis(ax)
    ax.legend(frameon=False, title=group, ncol=min(3, max(1, len(pivot.columns))))
    ax.tick_params(axis="x", rotation=18)
    return _save_figure(fig, output_path)


def plot_metric_bars(frame: pd.DataFrame, category: str, value: str, output_path: str | Path, title: str, ylabel: str) -> Path:
    _apply_publication_style()
    ordered = frame.sort_values(value, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    colors = [PUBLICATION_COLORS[idx % len(PUBLICATION_COLORS)] for idx in range(len(ordered))]
    ax.bar(ordered[category], ordered[value], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    _style_axis(ax)
    for idx, val in enumerate(ordered[value].tolist()):
        ax.text(idx, val, f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    return _save_figure(fig, output_path)


def plot_pareto_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    title: str,
    group: str | None = None,
    label_col: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Path:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    if group and group in frame.columns:
        groups = list(frame[group].dropna().unique())
        for idx, group_name in enumerate(groups):
            part = frame[frame[group] == group_name]
            ax.scatter(part[x], part[y], s=85, alpha=0.9, label=str(group_name), color=PUBLICATION_COLORS[idx % len(PUBLICATION_COLORS)], edgecolors="white", linewidths=0.8)
    else:
        ax.scatter(frame[x], frame[y], s=85, alpha=0.9, color=PUBLICATION_COLORS[0], edgecolors="white", linewidths=0.8)
    if label_col and label_col in frame.columns:
        x_span = float(frame[x].max() - frame[x].min()) if len(frame) > 1 else 1.0
        y_span = float(frame[y].max() - frame[y].min()) if len(frame) > 1 else 1.0
        for _, row in frame.iterrows():
            ax.text(row[x] + 0.01 * x_span, row[y] + 0.01 * y_span, str(row[label_col]), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)
    _style_axis(ax, grid_axis="both")
    if group and group in frame.columns:
        ax.legend(frameon=False)
    return _save_figure(fig, output_path)


def plot_tradeoff_scatter(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    x: str = "latency_ms",
    y: str = "accuracy",
    size: str = "peak_sram_bytes",
) -> Path:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    size_values = frame[size].astype(float).to_numpy()
    size_min = float(size_values.min()) if len(size_values) else 1.0
    size_max = float(size_values.max()) if len(size_values) else 1.0
    denom = max(size_max - size_min, 1e-6)
    bubble_sizes = 100.0 + 260.0 * (size_values - size_min) / denom
    for method in frame["method"].dropna().unique():
        subset = frame[frame["method"] == method]
        method_mask = frame["method"].to_numpy() == method
        ax.scatter(
            subset[x],
            subset[y],
            s=bubble_sizes[method_mask],
            alpha=0.84,
            color=_method_color(str(method)),
            edgecolors="white",
            linewidths=0.9,
            label=str(method),
        )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, ncol=2, loc="best")
    return _save_figure(fig, output_path)


def plot_publication_bar_panels(
    frame: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    output_path: str | Path,
    title: str,
    ylabel_map: dict[str, str] | None = None,
) -> Path:
    _apply_publication_style()
    ylabel_map = ylabel_map or {}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.1))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(methods))
    colors = [_method_color(method) for method in methods]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = []
        for method in methods:
            subset = frame[frame["method"] == method]
            values.append(float(subset[metric].mean()) if not subset.empty else np.nan)
        bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.8)
        ax.set_title(chr(ord("a") + idx) + f") {ylabel_map.get(metric, metric)}", loc="left", pad=8)
        ax.set_xticks(x, methods, rotation=18, ha="right")
        ax.set_ylabel(ylabel_map.get(metric, metric))
        _style_axis(ax)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, y=1.02, fontsize=14, color="#101828")
    return _save_figure(fig, output_path)


def plot_delta_vs_standard(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    metric_columns: list[str],
    label_map: dict[str, str] | None = None,
) -> Path:
    _apply_publication_style()
    label_map = label_map or {}
    fig, axes = plt.subplots(len(metric_columns), 1, figsize=(8.6, 3.0 * len(metric_columns)))
    if len(metric_columns) == 1:
        axes = [axes]
    devices = frame["device_name"].tolist()
    x = np.arange(len(devices))
    for idx, metric in enumerate(metric_columns):
        ax = axes[idx]
        values = frame[metric].astype(float).to_numpy()
        colors = ["#1f3b73" if value >= 0 else "#e76f51" for value in values]
        ax.bar(x, values, color=colors, width=0.7)
        ax.axhline(0.0, color="#667085", linewidth=1.0)
        ax.set_xticks(x, devices, rotation=18, ha="right")
        ax.set_ylabel(label_map.get(metric, metric))
        ax.set_title(chr(ord("a") + idx) + f") {label_map.get(metric, metric)}", loc="left", pad=8)
        _style_axis(ax)
    fig.suptitle(title, y=1.02, fontsize=14, color="#101828")
    return _save_figure(fig, output_path)


def write_markdown_summary(lines: list[str], output_path: str | Path, title: str) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")
    return output
