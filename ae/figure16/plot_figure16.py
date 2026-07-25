#!/usr/bin/env python3
"""Validate and plot the portable Figure 16 energy-efficiency data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data/figure16_energy_efficiency.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/figure16"

DATASETS = [("deep10m", "DP"), ("glove2m", "GV"), ("sift1m", "SF"),
            ("t2i1m", "T2I"), ("w2v1m", "W2V"), ("wiki1m", "WK"),
            ("pubmed", "PM")]
METHODS = ["CPU", "CAGRA", "BANG", "ANSMET", "NMP-Base", "NMP-FPMA",
           "NMP-FPSA", "NMP-Base-ET", "NMP-FPSA-ET", "MFNNS"]
COLORS = {
    "CPU": "#9e9e9e", "CAGRA": "#4f7cac", "BANG": "#59a14f",
    "ANSMET": "#6fb1b3", "NMP-Base": "#c8d596", "NMP-FPMA": "#ad86a8",
    "NMP-FPSA": "#d78d42", "NMP-Base-ET": "#90b45f",
    "NMP-FPSA-ET": "#c96552", "MFNNS": "#ead09e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["dataset_key"], row["method"])
        if key in lookup:
            raise ValueError(f"Duplicate Figure 16 row: {key}")
        value = float(row["energy_efficiency_norm_to_cpu"])
        if value <= 0:
            raise ValueError(f"Invalid normalized QPS/W: {key}")
        lookup[key] = value
    expected = {(dataset, method) for dataset, _ in DATASETS for method in METHODS}
    if set(lookup) != expected:
        raise ValueError("Figure 16 matrix is not 7 datasets x 10 methods")
    return lookup


def plot(lookup: dict[tuple[str, str], float], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.05, 2.57), dpi=300)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.075
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
    for index, method in enumerate(METHODS):
        values = [lookup[(dataset, method)] for dataset, _ in DATASETS]
        ax.bar(x + offsets[index], values, width=width, label=method,
               color=COLORS[method], edgecolor="black", linewidth=0.55, zorder=3)

    for index, (dataset, _) in enumerate(DATASETS):
        ratio = lookup[(dataset, "MFNNS")] / lookup[(dataset, "ANSMET")]
        y0 = lookup[(dataset, "ANSMET")]
        y1 = lookup[(dataset, "MFNNS")]
        ax.annotate("", xy=(x[index] + offsets[3], y1),
                    xytext=(x[index] + offsets[3], y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"))
        ax.plot([x[index] + offsets[3] - 0.04, x[index] + offsets[9] + 0.04],
                [y1, y1], color="black", linestyle="--", linewidth=0.8)
        ax.text(x[index] + (offsets[3] + offsets[9]) / 2, y1 + 0.35,
                f"{ratio:.1f}x", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#888888",
            alpha=0.65, zorder=0)
    ax.set_ylabel("Normalized QPS/W", fontsize=14, labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels([short for _, short in DATASETS], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ymax = max(lookup.values())
    ax.set_ylim(0, ymax * 1.18)
    legend = ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                       frameon=False, fontsize=9.5, columnspacing=1.15,
                       handlelength=1.6)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02,
                dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lookup = load(args.input)
    if not args.check_only:
        plot(lookup, args.output)


if __name__ == "__main__":
    main()
