#!/usr/bin/env python3
"""Validate the archived Figure 14 data and reproduce the publication plot."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data" / "figure14_results.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "figure14"
DEFAULT_SUMMARY = SCRIPT_DIR / "output" / "figure14_summary.tsv"

TOPK_ORDER = ["k5", "k10", "k100"]
TOPK_LABELS = {"k5": "Recall@5", "k10": "Recall@10", "k100": "Recall@100"}
DATASET_ORDER = [
    ("deep10m", "DP"),
    ("glove2m", "GV"),
    ("sift1m", "SF"),
    ("t2i1m", "T2I"),
    ("w2v1m", "W2V"),
    ("wiki1m", "WK"),
    ("pubmed", "PM"),
]
DESIGN_ORDER = [
    "cpu",
    "gpu_cagra",
    "bang",
    "ansmet",
    "ndp_base",
    "ndp_fpma",
    "ndp_et",
    "mfnns",
]
DESIGN_LABELS = {
    "cpu": "CPU",
    "gpu_cagra": "CAGRA",
    "bang": "BANG",
    "ansmet": "ANSMET",
    "ndp_base": "NMP-Base",
    "ndp_fpma": "NMP-FPMA/FPSA",
    "ndp_et": "NMP-FPSA-ET",
    "mfnns": "MFNNS",
}
DESIGN_COLORS = {
    "cpu": "#f4f4f4",
    "gpu_cagra": "#c9d693",
    "bang": "#9fc7e8",
    "ansmet": "#6ab3b5",
    "ndp_base": "#476c9c",
    "ndp_fpma": "#8d8d8d",
    "ndp_et": "#d28a45",
    "mfnns": "#ebd3a7",
}

FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 1.2
Y_AXIS_MAX = 50.0
BAR_WIDTH = 0.073
BAR_GROUP_SPAN = 0.70
BAR_OFFSETS = np.linspace(-BAR_GROUP_SPAN / 2, BAR_GROUP_SPAN / 2, len(DESIGN_ORDER))
OVERFLOW_LABEL_Y = Y_AXIS_MAX - 7
OVERFLOW_LABEL_SIDE = {"gpu_cagra": -1.0, "ndp_et": -1.0, "mfnns": 1.0}
TOPK_TEXT_POS = {"k5": (0.02, 0.96), "k10": (0.02, 0.96), "k100": (0.22, 0.96)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Write <prefix>.pdf and <prefix>.png.",
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data and print metrics without writing figures.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def build_lookup(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["top_k"], row["dataset_key"], row["design"])
        if key in lookup:
            raise ValueError(f"duplicate row: {key}")
        lookup[key] = row
    return lookup


def validate(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup = build_lookup(rows)
    expected = {
        (top_k, dataset, design)
        for top_k in TOPK_ORDER
        for dataset, _ in DATASET_ORDER
        for design in DESIGN_ORDER
    }
    actual = set(lookup)
    if actual != expected:
        raise ValueError(
            f"Figure 14 matrix differs: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )

    for key, row in lookup.items():
        qps = float(row["qps"])
        cpu_qps = float(row["cpu_qps"])
        speedup = float(row["qps_speedup_vs_cpu"])
        if not all(math.isfinite(value) and value > 0 for value in (qps, cpu_qps, speedup)):
            raise ValueError(f"non-positive/non-finite QPS value: {key}")
        calculated = qps / cpu_qps
        if not math.isclose(speedup, calculated, rel_tol=7e-4, abs_tol=1e-9):
            raise ValueError(
                f"speedup mismatch for {key}: exported={speedup}, calculated={calculated}"
            )
        for field in ("stats_ref", "run_ref"):
            reference = row.get(field, "")
            if reference and Path(reference).is_absolute():
                raise ValueError(f"absolute {field} for {key}: {reference}")

    measured = lookup[("k100", "glove2m", "ndp_base")]
    copied = lookup[("k100", "glove2m", "ndp_fpma")]
    expected_qps = 3418.924211
    expected_recall = 0.90294
    if not math.isclose(float(measured["qps"]), expected_qps, rel_tol=0, abs_tol=1e-9):
        raise ValueError("GloVe2M k100 NMP-Base does not contain the measured QPS")
    if not math.isclose(float(measured["recall"]), expected_recall, rel_tol=0, abs_tol=1e-12):
        raise ValueError("GloVe2M k100 NMP-Base does not contain the measured recall")
    if measured["data_status"] != "measured":
        raise ValueError("GloVe2M k100 NMP-Base must be marked measured")
    if copied["data_status"] != "derived_copy_no_direct_test":
        raise ValueError("GloVe2M k100 NMP-FPMA must remain marked as a derived copy")
    return lookup


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summary_metrics(
    lookup: dict[tuple[str, str, str], dict[str, str]]
) -> list[tuple[str, str, float]]:
    datasets = [dataset for dataset, _ in DATASET_ORDER]
    metrics: list[tuple[str, str, float]] = []
    for denominator in ("cpu", "bang", "gpu_cagra", "ansmet"):
        ratios = [
            float(lookup[(top_k, dataset, "mfnns")]["qps_speedup_vs_cpu"])
            / float(lookup[(top_k, dataset, denominator)]["qps_speedup_vs_cpu"])
            for top_k in TOPK_ORDER
            for dataset in datasets
        ]
        metrics.append((f"mfnns_vs_{denominator}", "all_21_points", geometric_mean(ratios)))
    for top_k in TOPK_ORDER:
        et_over_fpma = [
            float(lookup[(top_k, dataset, "ndp_et")]["qps"])
            / float(lookup[(top_k, dataset, "ndp_fpma")]["qps"])
            for dataset in datasets
        ]
        mfnns_over_et = [
            float(lookup[(top_k, dataset, "mfnns")]["qps"])
            / float(lookup[(top_k, dataset, "ndp_et")]["qps"])
            for dataset in datasets
        ]
        metrics.append(("ndp_et_vs_ndp_fpma", top_k, geometric_mean(et_over_fpma)))
        metrics.append(("mfnns_vs_ndp_et", top_k, geometric_mean(mfnns_over_et)))
    return metrics


def write_summary(path: Path, metrics: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "scope", "value"])
        for metric, scope, value in metrics:
            writer.writerow([metric, scope, f"{value:.12f}"])


def set_plot_style() -> None:
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for family in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def darken(color: str, factor: float = 0.68) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0, 1))


def style_axis(axis: plt.Axes) -> None:
    axis.set_ylim(0, Y_AXIS_MAX)
    axis.yaxis.set_major_locator(MultipleLocator(10))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    axis.grid(axis="y", which="major", color="#d9d9d9", linestyle="--", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=5.8, length=2.5, width=0.7, pad=1.0)
    axis.tick_params(axis="y", labelsize=5.8, length=2.5, width=0.7, pad=1.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.95)
    axis.spines["bottom"].set_linewidth(0.95)
    axis.axhline(0, color="black", linewidth=0.95, zorder=8, clip_on=False)


def plot(
    lookup: dict[tuple[str, str, str], dict[str, str]], output_prefix: Path
) -> None:
    set_plot_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(DATASET_ORDER), dtype=float)
    fig, axes = plt.subplots(
        1,
        len(TOPK_ORDER),
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=300,
        sharey=True,
    )
    x_left = x[0] + BAR_OFFSETS[0] - BAR_WIDTH * 0.65
    x_right = x[-1] + BAR_OFFSETS[-1] + BAR_WIDTH * 0.65

    for axis_index, top_k in enumerate(TOPK_ORDER):
        axis = axes[axis_index]
        for design_index, design in enumerate(DESIGN_ORDER):
            values = np.asarray(
                [
                    float(lookup[(top_k, dataset, design)]["qps_speedup_vs_cpu"])
                    for dataset, _ in DATASET_ORDER
                ],
                dtype=float,
            )
            bar_x = x + BAR_OFFSETS[design_index]
            axis.bar(
                bar_x,
                np.minimum(values, Y_AXIS_MAX),
                width=BAR_WIDTH,
                color=DESIGN_COLORS[design],
                edgecolor="black",
                linewidth=0.28,
                alpha=0.98,
                zorder=3,
            )
            for x_position, value in zip(bar_x, values):
                if value <= Y_AXIS_MAX:
                    continue
                side = OVERFLOW_LABEL_SIDE.get(design, 1.0)
                axis.text(
                    float(x_position) + side * BAR_WIDTH * 0.95,
                    OVERFLOW_LABEL_Y,
                    f"{value:.1f}x",
                    ha="left" if side > 0 else "right",
                    va="center",
                    rotation=90,
                    fontsize=5.8,
                    color=darken(DESIGN_COLORS[design]),
                    fontweight="bold",
                    path_effects=[path_effects.withStroke(linewidth=0, foreground="black")],
                    zorder=9,
                )

        style_axis(axis)
        axis.set_xlim(x_left, x_right)
        axis.set_xticks(x)
        axis.set_xticklabels([short for _, short in DATASET_ORDER], fontweight="bold")
        topk_x, topk_y = TOPK_TEXT_POS[top_k]
        axis.text(
            topk_x,
            topk_y,
            TOPK_LABELS[top_k],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            color="#222222",
            fontweight="bold",
        )
        if axis_index > 0:
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)
            axis.spines["left"].set_visible(False)

    fig.text(
        0.03,
        0.48,
        "Normalized QPS",
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=7.6,
        fontweight="bold",
    )
    handles = [
        Patch(
            facecolor=DESIGN_COLORS[design],
            edgecolor="black",
            linewidth=0.45,
            label=DESIGN_LABELS[design],
        )
        for design in DESIGN_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.92),
        ncol=len(DESIGN_ORDER),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.8,
        borderaxespad=0.1,
        prop={"size": 6.0, "weight": "bold"},
    )
    fig.subplots_adjust(left=0.055, right=0.998, bottom=0.18, top=0.82, wspace=0.045)
    for left_axis, right_axis in zip(axes[:-1], axes[1:]):
        left_box = left_axis.get_position()
        right_box = right_axis.get_position()
        x_separator = (left_box.x1 + right_box.x0) / 2
        fig.add_artist(
            Line2D(
                [x_separator, x_separator],
                [min(left_box.y0, right_box.y0), max(left_box.y1, right_box.y1)],
                transform=fig.transFigure,
                color="#8a8a8a",
                linewidth=0.55,
                linestyle=(0, (2.0, 2.2)),
                zorder=5,
            )
        )

    for suffix in ("pdf", "png"):
        output = Path(f"{output_prefix}.{suffix}")
        fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {output}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.data)
    lookup = validate(rows)
    metrics = summary_metrics(lookup)
    print("DATA_OK rows=168 measured_glove2m_k100_ndp_base_qps=3418.924211")
    for metric, scope, value in metrics:
        print(f"{metric}[{scope}]={value:.6f}")
    if not args.check_only:
        write_summary(args.summary_out, metrics)
        plot(lookup, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
