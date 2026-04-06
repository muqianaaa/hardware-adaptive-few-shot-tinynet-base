from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .io import ensure_dir

PAUL_TOL = {
    "deep_blue": "#4477AA",
    "cyan": "#66CCEE",
    "teal": "#228833",
    "green": "#44AA99",
    "sand": "#CCBB44",
    "rose": "#EE6677",
    "wine": "#AA3377",
    "grey": "#BBBBBB",
    "orange": "#EE7733",
}

METHOD_NAME_MAP = {
    "few_shot": "本文方法",
    "zero_shot": "零样本迁移",
    "hardware_agnostic": "无硬件搜索",
    "random_search": "随机搜索",
    "blackbox_cost_mlp": "黑盒代价回归",
    "standard_cnn": "标准 TinyNet 基线",
    "shallow_wide": "宽浅基线",
    "deep_narrow": "深窄基线",
    "depthwise_heavy_mixed_precision": "混合精度深度可分离基线",
    "mobilenet_like": "轻量深度可分离基线",
}

ABLATION_NAME_MAP = {
    "full": "完整方案",
    "no_probes": "去除微算子探针",
    "no_refs": "去除参考网络探针",
    "static_only": "去除全部探针，仅保留静态硬件特征",
    "no_generator": "去除生成器",
    "no_response": "去除响应解码层",
    "no_calibration": "去除设备校准",
    "no_adaptation": "去除元学习适配",
    "no_local_refine": "去除局部精修",
    "no_feasibility_loss": "去除可部署性判别损失",
    "no_generator_loss": "去除网络参数生成损失",
    "no_response_aux_loss": "去除硬件响应辅助监督项",
}

DATASET_NAME_MAP = {
    "cifar10": "CIFAR-10",
    "synthetic_cifar10": "CIFAR-10 合成对照集",
    "cifar100": "CIFAR-100",
}

FAMILY_NAME_MAP = {
    "low_memory_mcu": "低内存 MCU",
    "balanced_mcu": "均衡型 MCU",
    "high_performance_mcu": "高性能 MCU",
    "low_bit_friendly_mcu": "低比特友好 MCU",
    "depthwise_unfriendly_mcu": "深度可分离不友好 MCU",
    "memory_bottleneck_mcu": "访存瓶颈 MCU",
    "stm32f405rgt6_real": "STM32F405RGT6 实物板",
}

COLUMN_NAME_MAP = {
    "dataset_name": "数据集",
    "device_name": "设备名称",
    "family": "设备类别",
    "method": "方法",
    "ablation_mode": "消融设置",
    "support_size": "支持样本数",
    "accuracy": "准确率",
    "latency_ms": "延迟/ms",
    "peak_sram_bytes": "峰值SRAM/byte",
    "flash_bytes": "Flash/byte",
    "latency_mae": "延迟预测误差",
    "sram_mae": "SRAM预测误差",
    "flash_mae": "Flash预测误差",
    "feasible_acc": "可行性判别准确率",
    "accuracy_mae": "准确率预测误差",
    "arch_name": "架构名称",
    "arch_repr": "架构编码",
    "accuracy_delta": "准确率增益",
    "latency_reduction_ratio": "延迟下降比例",
    "sram_reduction_ratio": "SRAM下降比例",
    "flash_reduction_ratio": "Flash下降比例",
    "few_shot_dominates_standard": "本文方法是否支配标准基线",
}

METHOD_ORDER = [
    "标准 TinyNet 基线",
    "零样本迁移",
    "无硬件搜索",
    "随机搜索",
    "黑盒代价回归",
    "本文方法",
]


def _format_device_label(name: str, family: str | None = None) -> str:
    if family:
        family_label = _translate(str(family), FAMILY_NAME_MAP)
    else:
        family_label = None
    parts = str(name).split("_")
    suffix = parts[-1] if parts and parts[-1].isdigit() else None
    if family_label and suffix is not None:
        return f"{family_label}-{int(suffix) + 1:02d}"
    if family_label and suffix is None:
        return family_label
    if suffix is not None:
        return f"设备-{int(suffix) + 1:02d}"
    return str(name).replace("_", "-")


def _apply_paper_theme() -> None:
    plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "lines.linewidth": 2.1,
            "axes.prop_cycle": matplotlib.cycler(
                color=[
                    PAUL_TOL["deep_blue"],
                    PAUL_TOL["cyan"],
                    PAUL_TOL["sand"],
                    PAUL_TOL["grey"],
                    PAUL_TOL["rose"],
                    PAUL_TOL["teal"],
                ]
            ),
        }
    )


def _save_figure(fig: plt.Figure, output_path: str | Path, tight: bool = True) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    if tight:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def _translate(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.12, linewidth=0.7)
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")
    ax.tick_params(colors="#344054")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")


def _ordered_method_labels(labels: list[str]) -> list[str]:
    known = [name for name in METHOD_ORDER if name in labels]
    extra = [name for name in labels if name not in known]
    return known + extra


def _board_budget_title(frame: pd.DataFrame, board_name: str) -> str:
    return board_name


def _over_budget_style(tag: str) -> tuple[str, str, str]:
    return "#FFFFFF", "#FFFFFF", ""


def format_table_for_paper(frame: pd.DataFrame, rename_columns: bool = True) -> pd.DataFrame:
    out = frame.copy()
    if "method" in out.columns:
        out["method"] = out["method"].map(lambda x: _translate(str(x), METHOD_NAME_MAP))
    if "dataset_name" in out.columns:
        out["dataset_name"] = out["dataset_name"].map(lambda x: _translate(str(x), DATASET_NAME_MAP))
    if "ablation_mode" in out.columns:
        out["ablation_mode"] = out["ablation_mode"].map(lambda x: _translate(str(x), ABLATION_NAME_MAP))
    if "family" in out.columns:
        out["family"] = out["family"].map(lambda x: _translate(str(x), FAMILY_NAME_MAP))
    if "device_name" in out.columns:
        if "family" in out.columns:
            out["device_name"] = [
                _format_device_label(device_name, family)
                for device_name, family in zip(out["device_name"], out["family"])
            ]
        else:
            out["device_name"] = out["device_name"].map(lambda x: _format_device_label(str(x)))
    if rename_columns:
        rename_map = {column: COLUMN_NAME_MAP[column] for column in out.columns if column in COLUMN_NAME_MAP}
        out = out.rename(columns=rename_map)
    return out


def build_method_overview_diagram(output_path: str | Path) -> Path:
    _apply_paper_theme()
    fig, ax = plt.subplots(figsize=(16.2, 8.6))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    colors = {
        "lane": "#F8FAFC",
        "lane_edge": "#D0D5DD",
        "task": "#DCEBFA",
        "task_edge": "#4477AA",
        "device": "#DDEFE3",
        "device_edge": "#228833",
        "search": "#FCE7D6",
        "search_edge": "#EE7733",
        "text": "#111827",
        "sub": "#475467",
        "line": "#667085",
    }

    def rounded_box(x, y, w, h, fc, ec, title, body, number=None, badge_fc=None):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.22",
            fc=fc,
            ec=ec,
            lw=1.4,
        )
        ax.add_patch(patch)
        if number is not None:
            badge = Circle((x + 0.34, y + h - 0.38), 0.24, fc=badge_fc or ec, ec="white", lw=1.0, zorder=5)
            ax.add_patch(badge)
            ax.text(x + 0.34, y + h - 0.38, str(number), ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=6)
            title_x = x + 0.68
        else:
            title_x = x + 0.20
        ax.text(title_x, y + h - 0.32, title, ha="left", va="top", fontsize=11.5, color=colors["text"], fontweight="bold")
        ax.text(title_x, y + h - 0.92, body, ha="left", va="top", fontsize=9.7, color=colors["sub"], linespacing=1.42)

    def pill(x, y, w, h, text, fc, ec):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.16,rounding_size=0.3", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=colors["text"], fontweight="bold")

    def arrow(p0, p1, style="-|>", lw=1.35, color=None, ls="-", rad=0.0, ms=12):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle=style,
                mutation_scale=ms,
                lw=lw,
                color=color or colors["line"],
                linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    # Lanes
    top_lane = FancyBboxPatch((0.5, 6.55), 19.0, 4.55, boxstyle="round,pad=0.12,rounding_size=0.22", fc=colors["lane"], ec=colors["lane_edge"], lw=1.2)
    bottom_lane = FancyBboxPatch((0.5, 0.85), 19.0, 4.85, boxstyle="round,pad=0.12,rounding_size=0.22", fc=colors["lane"], ec=colors["lane_edge"], lw=1.2)
    ax.add_patch(top_lane)
    ax.add_patch(bottom_lane)

    # Soft region blocks
    region_specs = [
        (0.8, 7.0, 5.0, 3.75, "#F2F7FD"),
        (6.05, 1.25, 6.9, 9.5, "#F3FAF5"),
        (13.2, 1.25, 5.9, 9.5, "#FFF5EE"),
    ]
    for x, y, w, h, fc in region_specs:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18", fc=fc, ec="none", lw=0))

    pill(0.85, 10.45, 2.15, 0.55, "训练阶段", "#EAF2FB", colors["task_edge"])
    pill(0.85, 5.10, 2.15, 0.55, "部署阶段", "#EAF2FB", colors["task_edge"])
    pill(1.15, 9.65, 2.25, 0.42, "任务与网络空间", "#DCEBFA", colors["task_edge"])
    pill(7.85, 9.65, 2.25, 0.42, "MCU 硬件建模", "#DDEFE3", colors["device_edge"])
    pill(15.15, 9.65, 2.35, 0.42, "参数生成与部署", "#FCE7D6", colors["search_edge"])

    # Top row
    rounded_box(
        1.1, 7.35, 4.2, 2.65, colors["task"], colors["task_edge"],
        "任务侧数据与 TinyNet-Base",
        "公开图像分类数据集\nTinyNet-Base 与 5 个可搜索块\n子网络采样与结构编码 x(a)\n任务准确率数据与 Aω",
        number=1, badge_fc=colors["task_edge"]
    )
    rounded_box(
        6.5, 7.35, 3.6, 2.65, colors["device"], colors["device_edge"],
        "已知 MCU 元学习数据",
        "静态硬件特征\n9 个微算子探针\n3 个参考网络探针\n多设备性能测量样本",
        number=2, badge_fc=colors["device_edge"]
    )
    rounded_box(
        10.45, 7.35, 3.2, 2.65, colors["device"], colors["device_edge"],
        "共享硬件建模主干",
        "硬件编码器 Eφ\n响应解码器 Rψ\n多目标预测器 Cη\n可行性判别头 Fχ",
        number=3, badge_fc=colors["device_edge"]
    )
    rounded_box(
        14.05, 7.35, 2.5, 2.65, colors["device"], colors["device_edge"],
        "元学习训练",
        "设备-预算元任务\n支持集 / 查询集\n内循环更新设备状态\n外循环更新共享参数",
        number=4, badge_fc=colors["device_edge"]
    )
    rounded_box(
        16.95, 7.35, 2.05, 2.65, colors["search"], colors["search_edge"],
        "生成器训练",
        "Gθ 学习\n硬件条件到网络参数的映射\n形成部署先验",
        number=5, badge_fc=colors["search_edge"]
    )

    # Bottom row
    rounded_box(
        1.1, 1.55, 4.2, 2.95, colors["task"], colors["task_edge"],
        "未知 MCU 观测输入",
        "目标 MCU 静态特征\n微算子探针结果\n参考网络探针结果\n少量完整网络支持样本",
        number=6, badge_fc=colors["task_edge"]
    )
    rounded_box(
        6.5, 1.55, 3.6, 2.95, colors["device"], colors["device_edge"],
        "少样本设备适配",
        "共享主干读取 v_h\n更新设备嵌入 e_h\n更新设备校准 c_h\n形成目标 MCU 状态",
        number=7, badge_fc=colors["device_edge"]
    )
    rounded_box(
        10.45, 1.55, 3.2, 2.95, colors["device"], colors["device_edge"],
        "目标设备性能恢复",
        "Aω 估计分类准确率\nCη 预测延迟 / 峰值 SRAM / Flash\nFχ 判断预算可行性",
        badge_fc=colors["device_edge"]
    )
    rounded_box(
        14.05, 1.55, 2.5, 2.95, colors["search"], colors["search_edge"],
        "网络参数生成",
        "Gθ(e_h+c_h, b)\n直接输出 5 块 TinyNet-Base 参数\n得到候选部署网络",
        badge_fc=colors["search_edge"]
    )
    rounded_box(
        16.95, 1.55, 2.05, 2.95, colors["search"], colors["search_edge"],
        "局部精修与部署",
        "邻域内小范围校正\ntop-k 复测与重排序\n输出最终部署网络",
        badge_fc=colors["search_edge"]
    )

    # Shared-state callout
    state_box = FancyBboxPatch((11.1, 5.35), 2.4, 0.72, boxstyle="round,pad=0.14,rounding_size=0.18", fc="white", ec=colors["device_edge"], lw=1.1)
    ax.add_patch(state_box)
    ax.text(12.3, 5.71, "共享模型参数", ha="center", va="center", fontsize=9.8, color=colors["text"], fontweight="bold")
    ax.text(12.3, 5.43, "Eφ / Rψ / Aω / Cη / Fχ / Gθ", ha="center", va="center", fontsize=8.7, color=colors["sub"])

    budget_box = FancyBboxPatch((14.6, 5.35), 1.75, 0.72, boxstyle="round,pad=0.14,rounding_size=0.18", fc="white", ec=colors["search_edge"], lw=1.1)
    ax.add_patch(budget_box)
    ax.text(15.48, 5.72, "预算向量 b", ha="center", va="center", fontsize=9.7, color=colors["text"], fontweight="bold")
    ax.text(15.48, 5.43, "Tmax / Mmax / Fmax", ha="center", va="center", fontsize=8.6, color=colors["sub"])

    result_box = FancyBboxPatch((16.55, 5.18), 2.65, 0.90, boxstyle="round,pad=0.14,rounding_size=0.20", fc="white", ec=colors["search_edge"], lw=1.1)
    ax.add_patch(result_box)
    ax.text(17.88, 5.71, "最终输出", ha="center", va="center", fontsize=9.8, color=colors["text"], fontweight="bold")
    ax.text(17.88, 5.37, "准确率 / 延迟 / 峰值 SRAM / Flash", ha="center", va="center", fontsize=8.4, color=colors["sub"])

    # Top arrows
    arrow((5.3, 8.67), (6.5, 8.67))
    arrow((10.1, 8.67), (10.45, 8.67))
    arrow((13.65, 8.67), (14.05, 8.67))
    arrow((16.55, 8.67), (16.95, 8.67))

    # Bottom arrows
    arrow((5.3, 3.0), (6.5, 3.0))
    arrow((10.1, 3.0), (10.45, 3.0))
    arrow((13.65, 3.0), (14.05, 3.0))
    arrow((16.55, 3.0), (16.95, 3.0))

    # Cross-lane / shared arrows
    arrow((11.55, 7.35), (12.0, 6.08), style="-", lw=1.1, ms=1)
    arrow((12.0, 5.35), (12.0, 4.5), style="-", lw=1.1, ms=1)
    arrow((15.0, 7.35), (15.1, 6.08), style="-", lw=1.1, ms=1)
    arrow((15.5, 5.35), (15.35, 4.5), style="-", lw=1.1, ms=1)

    # Meta-learning feedback arrows
    arrow((15.0, 7.1), (11.95, 7.1), style="-|>", lw=1.1, ls="--", rad=0.12, ms=11)
    arrow((10.05, 2.0), (7.7, 2.0), style="-|>", lw=1.1, ls="--", rad=-0.10, ms=11)

    # Small annotations
    ax.text(3.2, 6.45, "任务侧：固定同一图像分类任务", ha="center", va="center", fontsize=9.2, color=colors["sub"])
    ax.text(8.3, 6.45, "硬件侧：从已知 MCU 学习“如何适配新 MCU”", ha="center", va="center", fontsize=9.2, color=colors["sub"])
    ax.text(16.15, 6.45, "部署侧：直接生成网络参数，再做少量工程修正", ha="center", va="center", fontsize=9.2, color=colors["sub"])

    # Final deployment arrow / footer
    arrow((19.0, 3.0), (19.35, 3.0))
    ax.text(19.08, 2.4, "部署到未知 MCU", ha="center", va="center", fontsize=9.8, color=colors["text"], fontweight="bold")

    return _save_figure(fig, output_path, tight=False)


def plot_kshot_curve(frame: pd.DataFrame, output_path: str | Path, metric: str = "latency_mae") -> Path:
    _apply_paper_theme()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for method in frame["method"].dropna().unique():
        part = frame[frame["method"] == method].sort_values("support_size")
        ax.plot(part["support_size"], part[metric], marker="o", label=_translate(str(method), METHOD_NAME_MAP))
        if {"metric_std"} <= set(part.columns):
            ax.fill_between(part["support_size"], part[metric] - part["metric_std"], part[metric] + part["metric_std"], alpha=0.08)
    ax.set_xlabel("支持样本数 K")
    ax.set_ylabel("延迟预测误差 / ms")
    ax.set_xticks(sorted(frame["support_size"].dropna().astype(int).unique().tolist()))
    _style_axis(ax)
    ax.legend(frameon=False)
    return _save_figure(fig, output_path)


def plot_main_result_panels(frame: pd.DataFrame, output_path: str | Path, metric_columns: list[str], ylabel_map: dict[str, str]) -> Path:
    _apply_paper_theme()
    paper_frame = format_table_for_paper(frame, rename_columns=False)
    methods = _ordered_method_labels(paper_frame["method"].dropna().astype(str).tolist())
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(4.8 * len(metric_columns), 4.7))
    if len(metric_columns) == 1:
        axes = [axes]
    for idx, metric in enumerate(metric_columns):
        ax = axes[idx]
        values = [float(paper_frame.loc[paper_frame["method"] == method, metric].mean()) for method in methods]
        positions = np.arange(len(methods))
        colors = [PAUL_TOL["deep_blue"] if method == "本文方法" else PAUL_TOL["grey"] if method == "标准 TinyNet 基线" else PAUL_TOL["cyan"] for method in methods]
        ax.hlines(positions, xmin=min(values) * 0.98 if min(values) > 0 else min(values) * 1.02, xmax=values, color="#D0D5DD", linewidth=1.4)
        ax.scatter(values, positions, s=56, color=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_yticks(positions, methods)
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.set_xlabel(ylabel_map.get(metric, metric))
        _style_axis(ax)
        ax.grid(axis="x", alpha=0.16, linewidth=0.7)
        ax.set_ylabel("")
        for value, y in zip(values, positions):
            label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
            ax.text(value, y + 0.13, label, fontsize=8.3, color="#344054")
    return _save_figure(fig, output_path)


def plot_board_improvement(frame: pd.DataFrame, output_path: str | Path) -> Path:
    _apply_paper_theme()
    paper_frame = format_table_for_paper(frame, rename_columns=False).copy()
    if "family" in paper_frame.columns:
        family_order = [
            "低内存 MCU",
            "均衡型 MCU",
            "高性能 MCU",
            "低比特友好 MCU",
            "深度可分离不友好 MCU",
            "访存瓶颈 MCU",
        ]
        family_rank = {name: idx for idx, name in enumerate(family_order)}
        paper_frame["_family_rank"] = paper_frame["family"].map(lambda x: family_rank.get(str(x), 99))
        paper_frame = paper_frame.sort_values(["_family_rank", "device_name"]).reset_index(drop=True)
        paper_frame = paper_frame.drop(columns=["_family_rank"])
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 5.2), sharey=True)
    metrics = [
        ("accuracy_delta", "准确率增益"),
        ("latency_reduction_ratio", "延迟下降比例"),
        ("sram_reduction_ratio", "峰值 SRAM 下降比例"),
        ("flash_reduction_ratio", "Flash 下降比例"),
    ]
    x = np.arange(len(paper_frame))
    labels = paper_frame["device_name"].tolist()
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = paper_frame[metric].astype(float).to_numpy()
        ax.scatter(values, x, s=55, color=PAUL_TOL["deep_blue"])
        for idx, value in enumerate(values):
            ax.plot([0.0, value], [idx, idx], color=PAUL_TOL["grey"], linewidth=1.0)
        ax.axvline(0.0, color="#475467", linewidth=1.0)
        ax.set_xlabel(ylabel)
        ax.set_ylim(-0.5, len(x) - 0.5)
        _style_axis(ax)
    axes[0].set_yticks(x, labels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.18, top=0.97, wspace=0.10)
    return _save_figure(fig, output_path, tight=False)


def plot_board_method_panels(frame: pd.DataFrame, output_path: str | Path, metric_columns: list[str], xlabel_map: dict[str, str]) -> Path:
    _apply_paper_theme()
    paper_frame = format_table_for_paper(frame, rename_columns=False).copy()
    board_names = paper_frame["device_name"].drop_duplicates().tolist()
    board_count = len(board_names)
    fig, axes = plt.subplots(len(metric_columns), board_count, figsize=(3.7 * board_count, 2.3 * len(metric_columns)), sharey="row")
    if len(metric_columns) == 1:
        axes = np.asarray([axes])
    if board_count == 1:
        axes = np.asarray([[ax] for ax in axes])
    for col_idx, board_name in enumerate(board_names):
        board_frame = paper_frame[paper_frame["device_name"] == board_name].copy()
        methods = _ordered_method_labels(board_frame["method"].dropna().astype(str).tolist())
        positions = np.arange(len(methods))
        for row_idx, metric in enumerate(metric_columns):
            ax = axes[row_idx, col_idx]
            values = [float(board_frame.loc[board_frame["method"] == method, metric].mean()) for method in methods]
            colors = [PAUL_TOL["deep_blue"] if method == "本文方法" else PAUL_TOL["grey"] if method == "标准 TinyNet 基线" else PAUL_TOL["cyan"] for method in methods]
            xmin = min(values) * 0.96 if min(values) > 0 else min(values) * 1.04
            ax.hlines(positions, xmin=xmin, xmax=values, color="#D0D5DD", linewidth=1.25)
            over_tags = [
                str(board_frame.loc[board_frame["method"] == method, "over_budget_dims"].iloc[0])
                if "over_budget_dims" in board_frame.columns and not board_frame.loc[board_frame["method"] == method].empty
                else "none"
                for method in methods
            ]
            for value, y, color, over_tag in zip(values, positions, colors, over_tags):
                edge, _, _ = _over_budget_style(over_tag)
                ax.scatter([value], [y], s=42, color=color, edgecolors=edge, linewidths=1.4 if over_tag != "none" else 0.8, marker="D" if over_tag != "none" else "o", zorder=3)
            if row_idx == 0:
                ax.set_title(_board_budget_title(board_frame, board_name), fontsize=9.8, color="#101828", pad=8)
            if col_idx == 0:
                ax.set_yticks(positions, methods)
                ax.set_ylabel(xlabel_map.get(metric, metric))
            else:
                ax.set_yticks(positions)
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel(xlabel_map.get(metric, metric))
            ax.grid(axis="x", alpha=0.16, linewidth=0.7)
            _style_axis(ax)
            for value, y, over_tag in zip(values, positions, over_tags):
                label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                ax.text(value, y + 0.10, label, fontsize=7.3, color="#344054")
                _, text_color, marker_text = _over_budget_style(over_tag)
                if marker_text:
                    ax.text(value, y - 0.18, marker_text, fontsize=7.0, color=text_color)
    fig.subplots_adjust(left=0.18 if board_count > 1 else 0.24, right=0.99, bottom=0.10, top=0.93, hspace=0.25, wspace=0.14)
    return _save_figure(fig, output_path, tight=False)


def plot_ablation_panels(frame: pd.DataFrame, output_path: str | Path, metric_columns: list[str], ylabel_map: dict[str, str]) -> Path:
    _apply_paper_theme()
    paper_frame = format_table_for_paper(frame, rename_columns=False)
    order = ["完整方案", "去除微算子探针", "去除参考网络探针", "去除全部探针，仅保留静态硬件特征", "去除响应解码层", "去除设备校准", "去除元学习适配", "去除局部精修", "去除预算可行性损失", "去除网络参数生成损失", "去除硬件响应辅助监督项"]
    groups = [name for name in order if name in paper_frame["ablation_mode"].tolist()]
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(4.5 * len(metric_columns), 4.4))
    if len(metric_columns) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_columns):
        values = [float(paper_frame.loc[paper_frame["ablation_mode"] == group, metric].mean()) for group in groups]
        positions = np.arange(len(groups))
        colors = [PAUL_TOL["deep_blue"] if group == "完整方案" else PAUL_TOL["orange"] for group in groups]
        ax.hlines(positions, xmin=min(values) * 0.98 if min(values) > 0 else min(values) * 1.02, xmax=values, color="#D0D5DD", linewidth=1.25)
        ax.scatter(values, positions, s=46, color=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_yticks(positions, groups)
        ax.set_xlabel(ylabel_map.get(metric, metric))
        ax.grid(axis="x", alpha=0.16, linewidth=0.7)
        _style_axis(ax)
        for value, y in zip(values, positions):
            label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
            ax.text(value, y + 0.12, label, fontsize=7.8, color="#344054")
    return _save_figure(fig, output_path)


def plot_board_ablation_panels(frame: pd.DataFrame, output_path: str | Path, metric_columns: list[str], xlabel_map: dict[str, str]) -> Path:
    _apply_paper_theme()
    paper_frame = format_table_for_paper(frame, rename_columns=False).copy()
    order = [
        "完整方案",
        "去除微算子探针",
        "去除参考网络探针",
        "去除全部探针，仅保留静态硬件特征",
        "去除元学习适配",
        "去除局部精修",
    ]
    board_names = paper_frame["device_name"].drop_duplicates().tolist()
    board_count = len(board_names)
    fig, axes = plt.subplots(len(metric_columns), board_count, figsize=(3.7 * board_count, 2.3 * len(metric_columns)), sharey="row")
    if len(metric_columns) == 1:
        axes = np.asarray([axes])
    if board_count == 1:
        axes = np.asarray([[ax] for ax in axes])
    for col_idx, board_name in enumerate(board_names):
        board_frame = paper_frame[paper_frame["device_name"] == board_name].copy()
        groups = [name for name in order if name in board_frame["ablation_mode"].tolist()]
        positions = np.arange(len(groups))
        for row_idx, metric in enumerate(metric_columns):
            ax = axes[row_idx, col_idx]
            values = [float(board_frame.loc[board_frame["ablation_mode"] == group, metric].mean()) for group in groups]
            colors = [PAUL_TOL["deep_blue"] if group == "完整方案" else PAUL_TOL["orange"] for group in groups]
            xmin = min(values) * 0.96 if min(values) > 0 else min(values) * 1.04
            ax.hlines(positions, xmin=xmin, xmax=values, color="#D0D5DD", linewidth=1.25)
            over_tags = [
                str(board_frame.loc[board_frame["ablation_mode"] == group, "over_budget_dims"].iloc[0])
                if "over_budget_dims" in board_frame.columns and not board_frame.loc[board_frame["ablation_mode"] == group].empty
                else "none"
                for group in groups
            ]
            for value, y, color, over_tag in zip(values, positions, colors, over_tags):
                edge, _, _ = _over_budget_style(over_tag)
                ax.scatter([value], [y], s=42, color=color, edgecolors=edge, linewidths=1.4 if over_tag != "none" else 0.8, marker="D" if over_tag != "none" else "o", zorder=3)
            if row_idx == 0:
                ax.set_title(_board_budget_title(board_frame, board_name), fontsize=9.8, color="#101828", pad=8)
            if col_idx == 0:
                ax.set_yticks(positions, groups)
                ax.set_ylabel(xlabel_map.get(metric, metric))
            else:
                ax.set_yticks(positions)
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel(xlabel_map.get(metric, metric))
            ax.grid(axis="x", alpha=0.16, linewidth=0.7)
            _style_axis(ax)
            for value, y, over_tag in zip(values, positions, over_tags):
                label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                ax.text(value, y + 0.10, label, fontsize=7.3, color="#344054")
                _, text_color, marker_text = _over_budget_style(over_tag)
                if marker_text:
                    ax.text(value, y - 0.18, marker_text, fontsize=7.0, color=text_color)
    fig.subplots_adjust(left=0.22 if board_count > 1 else 0.28, right=0.99, bottom=0.10, top=0.93, hspace=0.25, wspace=0.14)
    return _save_figure(fig, output_path, tight=False)
