#!/usr/bin/env python3
"""Validate and plot the portable Figure 17 normalized-energy breakdown."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data/figure17_energy_breakdown.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/figure17"
DATASETS = [("deep10m", "DP"), ("glove2m", "GV"), ("sift1m", "SF"),
            ("t2i1m", "T2I"), ("w2v1m", "W2V"), ("wiki1m", "WK"),
            ("pubmed", "PM")]
METHODS = ["NMP-Base", "ANSMET", "NMP-FPMA", "NMP-FPSA", "NMP-Base-ET",
           "NMP-FPSA-ET", "MFNNS"]
COLORS = {
    "NMP-Base": "#c8d596", "ANSMET": "#6fb1b3", "NMP-FPMA": "#ad86a8",
    "NMP-FPSA": "#d78d42", "NMP-Base-ET": "#90b45f",
    "NMP-FPSA-ET": "#c96552", "MFNNS": "#ead09e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        key = (row["dataset_key"], row["method"])
        if key in lookup:
            raise ValueError(f"Duplicate Figure 17 row: {key}")
        compute = float(row["compute_energy_norm_to_nmp_base"])
        total = float(row["total_energy_norm_to_nmp_base"])
        if compute < 0 or total <= 0 or compute > total:
            raise ValueError(f"Invalid Figure 17 energy components: {key}")
        lookup[key] = (compute, total)
    expected = {(dataset, method) for dataset, _ in DATASETS for method in METHODS}
    if set(lookup) != expected:
        raise ValueError("Figure 17 matrix is not 7 datasets x 7 methods")
    return lookup


def plot(
    lookup: dict[tuple[str, str], tuple[float, float]], output: Path
) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.19, 2.53), dpi=300)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.11
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
    for index, method in enumerate(METHODS):
        compute = [lookup[(dataset, method)][0] for dataset, _ in DATASETS]
        total = [lookup[(dataset, method)][1] for dataset, _ in DATASETS]
        ax.bar(x + offsets[index], total, width=width, label=method,
               color=COLORS[method], edgecolor="black", linewidth=0.55, zorder=2)
        ax.bar(x + offsets[index], compute, width=width, color="none",
               edgecolor="#3d3d3d", linewidth=0.8, hatch="////", zorder=3,
               label="Compute" if index == len(METHODS) - 1 else None)

    ax.set_ylabel("Normalized Energy", fontsize=14, labelpad=4)
    ax.set_xticks(x)
    ax.set_xticklabels([short for _, short in DATASETS], fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, ncol=4, loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=10,
                       columnspacing=1.2, handlelength=1.7)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
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
