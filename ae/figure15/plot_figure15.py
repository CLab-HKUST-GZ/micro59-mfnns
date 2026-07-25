#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = SCRIPT_DIR / "data" / "figure15_area_efficiency.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

DATASET_ORDER = [
    ("deep10m", "DP"),
    ("glove2m", "GV"),
    ("sift1m", "SF"),
    ("t2i1m", "T2I"),
    ("w2v1m", "W2V"),
    ("wiki1m", "WK"),
    ("pubmed", "PM"),
]
METHOD_ORDER = [
    "ANSMET",
    "NMP-Base",
    "NMP-FPMA",
    "NMP-FPSA",
    "NMP-Base-ET",
    "MFNNS",
]
METHOD_COLORS = {
    "ANSMET": "#6ab3b5",
    "NMP-Base": "#c9d693",
    "NMP-FPMA": "#b288a9",
    "NMP-FPSA": "#d28a45",
    "NMP-Base-ET": "#476c9c",
    "MFNNS": "#ebd3a7",
}

FIG_SIZE = (8.0, 2.6)
BAR_WIDTH = 0.135
BAR_GAP = 0.005
GROUP_GAP = 0.24
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 10.6
Y_MAX_SCALE = 1.18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and plot the portable Figure 15 area-efficiency CSV."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data and metrics without rendering output files.",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 15 data: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_keys = {
        (dataset, method)
        for dataset, _ in DATASET_ORDER
        for method in METHOD_ORDER
    }
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset_key"], row["method"])
        if key in lookup:
            raise ValueError(f"Duplicate Figure 15 row: {key}")
        lookup[key] = row
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))
        extra = sorted(set(lookup) - expected_keys)
        raise ValueError(f"Figure 15 matrix mismatch: missing={missing}, extra={extra}")

    for dataset, _ in DATASET_ORDER:
        ansmet_efficiency = float(
            lookup[(dataset, "ANSMET")]["area_efficiency_kqps_per_mm2"]
        )
        for method in METHOD_ORDER:
            row = lookup[(dataset, method)]
            qps = float(row["qps"])
            area = float(row["area_mm2"])
            efficiency = float(row["area_efficiency_kqps_per_mm2"])
            normalized = float(row["normalized_area_efficiency_vs_ansmet"])
            expected_efficiency = (qps / 1000.0) / area
            expected_normalized = expected_efficiency / ansmet_efficiency
            if not math.isclose(efficiency, expected_efficiency, rel_tol=1e-12):
                raise ValueError(f"Area-efficiency mismatch for {(dataset, method)}")
            if not math.isclose(normalized, expected_normalized, rel_tol=1e-12):
                raise ValueError(f"Normalization mismatch for {(dataset, method)}")
            for field in ("figure14_stats_ref", "figure14_config_ref"):
                value = row[field].strip()
                if value and Path(value).is_absolute():
                    raise ValueError(f"Absolute {field} for {(dataset, method)}")

    return rows


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(rows: list[dict[str, str]]) -> tuple[float, list[str]]:
    lookup = {
        (row["dataset_key"], row["method"]): row
        for row in rows
    }
    ratios = [
        float(lookup[(dataset, "MFNNS")]["normalized_area_efficiency_vs_ansmet"])
        for dataset, _ in DATASET_ORDER
    ]
    labels = [f"{ratio:.1f}x" for ratio in ratios]
    expected_labels = ["2.3x", "4.2x", "2.3x", "3.0x", "3.9x", "2.6x", "4.8x"]
    if labels != expected_labels:
        raise ValueError(f"Unexpected Figure 15 labels: {labels}")
    geomean = geometric_mean(ratios)
    if not math.isclose(geomean, 3.1848204102510804, rel_tol=1e-12):
        raise ValueError(f"Unexpected MFNNS/ANSMET geometric mean: {geomean}")
    return geomean, labels


def configure_plot_style() -> None:
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    matplotlib.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE + 1,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.labelweight": "bold",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    configure_plot_style()
    lookup = {
        (row["dataset_key"], row["method"]): row
        for row in rows
    }
    values = {
        method: [
            float(
                lookup[(dataset, method)][
                    "normalized_area_efficiency_vs_ansmet"
                ]
            )
            for dataset, _ in DATASET_ORDER
        ]
        for method in METHOD_ORDER
    }

    group_span = len(METHOD_ORDER) * (BAR_WIDTH + BAR_GAP)
    x = np.arange(len(DATASET_ORDER)) * (group_span + GROUP_GAP)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_axisbelow(True)

    for index, method in enumerate(METHOD_ORDER):
        offset = (index - (len(METHOD_ORDER) - 1) / 2.0) * (BAR_WIDTH + BAR_GAP)
        ax.bar(
            x + offset,
            values[method],
            BAR_WIDTH,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.7,
            zorder=2,
        )

    ansmet_index = METHOD_ORDER.index("ANSMET")
    mfnns_index = METHOD_ORDER.index("MFNNS")
    ansmet_offset = (
        ansmet_index - (len(METHOD_ORDER) - 1) / 2.0
    ) * (BAR_WIDTH + BAR_GAP)
    mfnns_offset = (
        mfnns_index - (len(METHOD_ORDER) - 1) / 2.0
    ) * (BAR_WIDTH + BAR_GAP)

    for dataset_index in range(len(DATASET_ORDER)):
        x_start = x[dataset_index] + ansmet_offset
        x_end = x[dataset_index] + mfnns_offset
        y_start = values["ANSMET"][dataset_index]
        y_end = values["MFNNS"][dataset_index]
        ax.annotate(
            "",
            xy=(x_start, y_end),
            xytext=(x_start, y_start),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1},
        )
        x_right = x_start - BAR_WIDTH * 0.6
        ax.plot(
            [x_end, x_right],
            [y_end, y_end],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.text(
            0.5 * (x_end + x_right),
            y_end * 1.05,
            f"{y_end / y_start:.1f}x",
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_SIZE,
            fontweight="bold",
        )

    all_values = np.array(
        [value for method_values in values.values() for value in method_values],
        dtype=float,
    )
    ax.set_ylim(0, all_values.max() * Y_MAX_SCALE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylabel("Area Efficiency", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASET_ORDER])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, zorder=0)

    legend = ax.legend(
        ncol=6,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1,
        handlelength=0.9,
        handletextpad=0.25,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.2),
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        output_dir / "figure15.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_dir / "figure15.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_and_validate(args.data)
    geomean, labels = summarize(rows)
    if not args.check_only:
        plot(rows, args.output_dir)
    print(
        "CHECK_OK "
        f"rows={len(rows)} "
        f"mfnns_vs_ansmet={geomean:.15f} "
        f"labels={','.join(labels)}"
    )


if __name__ == "__main__":
    main()
