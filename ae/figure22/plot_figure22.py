#!/usr/bin/env python3
"""Validate the frozen Recall@10 data and reproduce paper Figure 22."""

from __future__ import print_function

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data/figure22_latency_breakdown.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/figure22"
DEFAULT_SUMMARY = SCRIPT_DIR / "output/figure22_summary.tsv"

DATASET_ORDER = [
    ("deep10m", "DP"),
    ("glove2m", "GV"),
    ("sift1m", "SF"),
    ("t2i1m", "T2I"),
    ("w2v1m", "W2V"),
    ("wiki1m", "WK"),
    ("pubmed", "PM"),
]
METHODS = [("ansmet", "ANSMET"), ("mfnns", "MFNNS")]
STACK_SEGMENTS = [
    ("distance_hidden", "Data Movement"),
    ("distance_visible", "Distance Computation"),
    ("disgather", "Result Collection"),
    ("index", "Index Traversal"),
]
STACK_COLORS = {
    "distance_hidden": "#b288a9",
    "distance_visible": "#6ab3b5",
    "disgather": "#c9d693",
    "index": "#ebd3a7",
}
METHOD_HATCHES = {"ansmet": r"\\\\", "mfnns": "////"}

FIG_SIZE = (7.2, 3.0)
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 10
MAX_Y_TICKS = 6
BAR_EDGE_COLOR = "#000000"
BAR_LINEWIDTH = 0.5
BAR_ALPHA = 0.98
BAR_WIDTH = 0.20
BAR_GAP = 0.06
GROUP_GAP = 0.42
HATCH_COLOR = "#FFFFFF"
HATCH_LINEWIDTH = 0.9
LEGEND_BBOX = (0.48, 1.25)
LAYOUT_RECT = (0, 0, 1, 0.81)

EXPECTED_METRICS = {
    "arithmetic_mean_total_latency_reduction_pct": 30.514425412371388,
    "geometric_mean_speedup": 1.4915766241142772,
    "arithmetic_mean_distance_computation_reduction_pct": 55.38103491704568,
    "arithmetic_mean_data_movement_reduction_pct": 18.342930319829417,
    "arithmetic_mean_result_collection_reduction_pct": 2.4353076034143464,
    "arithmetic_mean_index_traversal_reduction_pct": 4.476814918717736,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data and headline metrics without rewriting outputs.",
    )
    return parser.parse_args()


def close(actual, expected, tolerance=1e-12):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def load_rows(path):
    required = {
        "dataset_key",
        "dataset_short",
        "method",
        "ef_search",
        "queue_size",
        "recall",
        "total_memory_cycle",
        "stats_ref",
        "stats_sha256",
        "source_slurm_ref",
        "total_average_latency",
        "average_index_latency",
        "average_disgather_latency",
        "fmac_raw_cycles_sum",
        "fmac_hidden_by_memory_cycles_sum",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing data columns: {}".format(sorted(missing)))
        raw_rows = list(reader)

    expected_keys = {
        (dataset, method)
        for dataset, _ in DATASET_ORDER
        for method, _ in METHODS
    }
    observed_keys = {
        (row["dataset_key"], row["method"]) for row in raw_rows
    }
    if len(raw_rows) != 14 or observed_keys != expected_keys:
        raise ValueError(
            "Expected 14 unique dataset/method rows; missing={}, extra={}".format(
                sorted(expected_keys - observed_keys),
                sorted(observed_keys - expected_keys),
            )
        )

    label_lookup = dict(DATASET_ORDER)
    rows = []
    for raw in raw_rows:
        dataset = raw["dataset_key"]
        method = raw["method"]
        if raw["dataset_short"] != label_lookup[dataset]:
            raise ValueError("Dataset label mismatch for {}".format(dataset))
        recall = float(raw["recall"])
        if not recall > 0.895:
            raise ValueError(
                "Recall policy violation for {}/{}: {}".format(
                    dataset, method, recall
                )
            )
        if Path(raw["stats_ref"]).is_absolute() or Path(
            raw["source_slurm_ref"]
        ).is_absolute():
            raise ValueError("Absolute provenance path in portable data")
        if len(raw["stats_sha256"]) != 64:
            raise ValueError("Invalid stats SHA-256 for {}/{}".format(dataset, method))

        total = float(raw["total_average_latency"])
        index = float(raw["average_index_latency"])
        disgather = float(raw["average_disgather_latency"])
        raw_cycles = float(raw["fmac_raw_cycles_sum"])
        hidden_cycles = float(raw["fmac_hidden_by_memory_cycles_sum"])
        distance = total - index - disgather
        if min(total, index, disgather, raw_cycles, hidden_cycles, distance) < 0:
            raise ValueError("Negative latency/cycle value for {}/{}".format(dataset, method))
        if raw_cycles == 0 or hidden_cycles > raw_cycles:
            raise ValueError("Invalid FMAC overlap ratio for {}/{}".format(dataset, method))

        hidden_ratio = hidden_cycles / raw_cycles
        rows.append(
            {
                "dataset_key": dataset,
                "dataset_short": raw["dataset_short"],
                "method": method,
                "recall": recall,
                "total": total,
                "index": index,
                "disgather": disgather,
                "distance_visible": distance * (1.0 - hidden_ratio),
                "distance_hidden": distance * hidden_ratio,
            }
        )

    lookup = {(row["dataset_key"], row["method"]): row for row in rows}
    for dataset, _ in DATASET_ORDER:
        baseline = lookup[(dataset, "ansmet")]["total"]
        for method, _ in METHODS:
            row = lookup[(dataset, method)]
            row["normalized_total"] = row["total"] / baseline
            for segment, _ in STACK_SEGMENTS:
                row["normalized_" + segment] = row[segment] / baseline
            stack_sum = sum(
                row["normalized_" + segment] for segment, _ in STACK_SEGMENTS
            )
            if not close(stack_sum, row["normalized_total"]):
                raise ValueError("Stack sum mismatch for {}/{}".format(dataset, method))
        if not close(lookup[(dataset, "ansmet")]["normalized_total"], 1.0):
            raise ValueError("ANSMET baseline mismatch for {}".format(dataset))
    return rows


def compute_metrics(rows):
    lookup = {(row["dataset_key"], row["method"]): row for row in rows}
    reductions = []
    speedups = []
    component_reductions = {segment: [] for segment, _ in STACK_SEGMENTS}
    dataset_speedups = {}
    for dataset, _ in DATASET_ORDER:
        ansmet = lookup[(dataset, "ansmet")]
        mfnns = lookup[(dataset, "mfnns")]
        normalized = mfnns["normalized_total"]
        reductions.append(1.0 - normalized)
        speedups.append(1.0 / normalized)
        dataset_speedups[dataset] = 1.0 / normalized
        for segment, _ in STACK_SEGMENTS:
            component_reductions[segment].append(
                1.0
                - mfnns["normalized_" + segment]
                / ansmet["normalized_" + segment]
            )

    metrics = {
        "arithmetic_mean_total_latency_reduction_pct": 100.0
        * sum(reductions)
        / len(reductions),
        "geometric_mean_speedup": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "arithmetic_mean_distance_computation_reduction_pct": 100.0
        * sum(component_reductions["distance_visible"])
        / len(reductions),
        "arithmetic_mean_data_movement_reduction_pct": 100.0
        * sum(component_reductions["distance_hidden"])
        / len(reductions),
        "arithmetic_mean_result_collection_reduction_pct": 100.0
        * sum(component_reductions["disgather"])
        / len(reductions),
        "arithmetic_mean_index_traversal_reduction_pct": 100.0
        * sum(component_reductions["index"])
        / len(reductions),
    }
    for name, expected in EXPECTED_METRICS.items():
        if not close(metrics[name], expected):
            raise ValueError(
                "Headline metric mismatch for {}: expected {}, got {}".format(
                    name, expected, metrics[name]
                )
            )
    return metrics, dataset_speedups


def set_style():
    available = {
        font.name for font in matplotlib.font_manager.fontManager.ttflist
    }
    matplotlib.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.color": HATCH_COLOR,
            "hatch.linewidth": HATCH_LINEWIDTH,
        }
    )


def draw_stacked_bar(ax, x_pos, row):
    bottom = 0.0
    for segment_name, _ in STACK_SEGMENTS:
        value = row["normalized_" + segment_name]
        ax.bar(
            x_pos,
            value,
            BAR_WIDTH,
            bottom=bottom,
            color=STACK_COLORS[segment_name],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            alpha=BAR_ALPHA,
        )
        ax.bar(
            x_pos,
            value,
            BAR_WIDTH,
            bottom=bottom,
            color="none",
            edgecolor=HATCH_COLOR,
            linewidth=0.0,
            hatch=METHOD_HATCHES[row["method"]],
        )
        bottom += value


def plot(rows, output_prefix):
    lookup = {(row["dataset_key"], row["method"]): row for row in rows}
    x_values = np.arange(len(DATASET_ORDER)) * (
        len(METHODS) * (BAR_WIDTH + BAR_GAP) + GROUP_GAP
    )
    offsets = [-0.5 * (BAR_WIDTH + BAR_GAP), 0.5 * (BAR_WIDTH + BAR_GAP)]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for dataset_index, (dataset, _) in enumerate(DATASET_ORDER):
        for method_index, (method, _) in enumerate(METHODS):
            draw_stacked_bar(
                ax,
                x_values[dataset_index] + offsets[method_index],
                lookup[(dataset, method)],
            )

    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS))
    ax.set_ylabel("Normalized Latency", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_xticks(x_values)
    ax.set_xticklabels([label for _, label in DATASET_ORDER])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    stack_handles = {
        label: Patch(
            facecolor=STACK_COLORS[name],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
        )
        for name, label in STACK_SEGMENTS
    }
    method_handles = []
    for method, _ in METHODS:
        method_handles.append(
            (
                Patch(
                    facecolor="#6f6f6f",
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_LINEWIDTH,
                ),
                Patch(
                    facecolor="none",
                    edgecolor=HATCH_COLOR,
                    linewidth=0.0,
                    hatch=METHOD_HATCHES[method],
                ),
            )
        )
    labels = [
        "Data Movement",
        "Index Traversal",
        "Distance Computation",
        "Result Collection",
        "ANSMET",
        "MFNNS",
    ]
    handles = [
        stack_handles["Data Movement"],
        stack_handles["Index Traversal"],
        stack_handles["Distance Computation"],
        stack_handles["Result Collection"],
    ] + method_handles
    legend = ax.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=LEGEND_BBOX,
        handlelength=1.6,
        handleheight=0.7,
        handletextpad=0.45,
        labelspacing=0.1,
        columnspacing=2.0,
        borderaxespad=0.0,
        borderpad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=LAYOUT_RECT)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def write_summary(path, metrics, dataset_speedups):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        for name in EXPECTED_METRICS:
            writer.writerow([name, "{:.15f}".format(metrics[name])])
        for dataset, _ in DATASET_ORDER:
            writer.writerow(
                ["{}_speedup".format(dataset), "{:.15f}".format(dataset_speedups[dataset])]
            )


def main():
    args = parse_args()
    rows = load_rows(args.data.resolve())
    metrics, dataset_speedups = compute_metrics(rows)
    print(
        "CHECK_OK rows={} mean_reduction={:.12f}% geomean_speedup={:.12f}x".format(
            len(rows),
            metrics["arithmetic_mean_total_latency_reduction_pct"],
            metrics["geometric_mean_speedup"],
        )
    )
    if args.check_only:
        return

    set_style()
    pdf_path, png_path = plot(rows, args.output_prefix.resolve())
    write_summary(args.summary.resolve(), metrics, dataset_speedups)
    print("Wrote:")
    print("  {}".format(pdf_path))
    print("  {}".format(png_path))
    print("  {}".format(args.summary.resolve()))


if __name__ == "__main__":
    main()
