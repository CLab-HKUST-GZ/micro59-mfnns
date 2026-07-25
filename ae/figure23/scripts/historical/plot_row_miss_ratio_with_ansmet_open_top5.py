#!/usr/bin/env python3
"""Plot the top-5 row-miss ratio for ANSMET open-row, NMP-ET, and MFNNS."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

FIG_SS = (8.6, 2.3)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

POWER_DIR = SCRIPT_DIR.parent / "power"
if str(POWER_DIR) not in sys.path:
    sys.path.insert(0, str(POWER_DIR))

import extract_and_plot_memory_efficiency_with_ansmet_rowpolicy_singlecol as source
import plot_energy_power_breakdown as power_style


DEFAULT_OUTPUT_STEM = "row_miss_ratio_with_ansmet_open_top5_k10"
DEFAULT_DATASET_ORDER = [
    ("deep10m", "normalized", "DP"),
    ("sift1m", "normalized", "SF"),
    ("t2i1m", "normalized", "T2I"),
    ("w2v1m", "normalized", "W2V"),
    ("wiki1m", "normalized", "WK"),
]
METHOD_SPECS = [
    {"design": "ansmet_open", "label": "ANSMET", "color": "#6ab3b5"},
    {"design": "ndp_et", "label": "NMP-FPSA-ET", "color": "#c96552"},
    {"design": "mfnns", "label": "MFNNS", "color": "#ebd3a7"},
]
LEGEND_NCOL = 3
LEGEND_BBOX = (0.5, 1.23)
LAYOUT_RECT = (0.0, 0.0, 0.992, 0.93)
Y_MAX_SCALE = 1.06
TIGHT_LAYOUT_PAD = 0.12
SAVE_PAD_INCHES = 0.03
DEFAULT_YLABEL_PAD = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=source.DEFAULT_BASE_DIR,
        help=f"Base dir for NMP-ET and MFNNS. Default: {source.DEFAULT_BASE_DIR}",
    )
    parser.add_argument(
        "--ansmet-open-dir",
        type=Path,
        default=source.DEFAULT_ANSMET_OPEN_DIR,
        help=f"Dir for ANSMET open-row runs. Default: {source.DEFAULT_ANSMET_OPEN_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=f"Output dir for the figure and extracted data. Default: {SCRIPT_DIR}",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=f"Stem for CSV/JSON/PNG/PDF outputs. Default: {DEFAULT_OUTPUT_STEM}",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("default", "all-complete"),
        default="all-complete",
        help="Use the default five datasets or every dataset with complete three-bar data.",
    )
    parser.add_argument(
        "--ylabel-pad",
        type=float,
        default=DEFAULT_YLABEL_PAD,
        help="Matplotlib labelpad for the left y-axis title. Smaller values move the title closer to the axis.",
    )
    return parser.parse_args()


def output_paths(output_dir: Path, output_stem: str) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / f"{output_stem}.csv",
        output_dir / f"{output_stem}.json",
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
    )


def collect_available_rows(
    base_dir: Path,
    ansmet_open_dir: Path,
) -> tuple[list[source.MetricRow], list[str]]:
    rows: list[source.MetricRow] = []
    skipped: list[str] = []

    for dataset, variant, dataset_short in source.DATASET_ORDER:
        dataset_rows: list[source.MetricRow] = []
        for spec in METHOD_SPECS:
            source_dir = ansmet_open_dir if spec["design"] == "ansmet_open" else base_dir
            try:
                dataset_rows.append(
                    source.collect_metric_row(
                        source_dir,
                        dataset,
                        variant,
                        dataset_short,
                        spec["design"],
                    )
                )
            except Exception as exc:
                skipped.append(
                    f"{dataset}_{variant}: skip because {spec['design']} is unavailable ({exc})"
                )
                dataset_rows = []
                break
        rows.extend(dataset_rows)

    return rows, skipped


def resolve_dataset_specs(
    rows: list[source.MetricRow],
    dataset_mode: str,
) -> list[tuple[str, str, str]]:
    lookup = source.build_lookup(rows)
    if dataset_mode == "default":
        return list(DEFAULT_DATASET_ORDER)

    dataset_specs: list[tuple[str, str, str]] = []
    for dataset, variant, dataset_short in source.DATASET_ORDER:
        if all((dataset, variant, spec["design"]) in lookup for spec in METHOD_SPECS):
            dataset_specs.append((dataset, variant, dataset_short))
    return dataset_specs


def collect_plot_rows(
    base_dir: Path,
    ansmet_open_dir: Path,
    dataset_mode: str,
) -> tuple[list[source.MetricRow], list[str], list[tuple[str, str, str]]]:
    rows, skipped = collect_available_rows(base_dir, ansmet_open_dir)
    dataset_specs = resolve_dataset_specs(rows, dataset_mode)
    lookup = source.build_lookup(rows)

    selected_rows: list[source.MetricRow] = []
    for dataset, variant, _ in dataset_specs:
        for spec in METHOD_SPECS:
            key = (dataset, variant, spec["design"])
            if key not in lookup:
                raise ValueError(f"Missing required record for {key}")
            selected_rows.append(lookup[key])
    return selected_rows, skipped, dataset_specs


def write_csv(rows: list[source.MetricRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(
    rows: list[source.MetricRow],
    skipped: list[str],
    dataset_specs: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    payload = {
        "records": [asdict(row) for row in rows],
        "skipped": list(skipped),
        "datasets": [
            {"dataset": dataset, "variant": variant, "short": dataset_short}
            for dataset, variant, dataset_short in dataset_specs
        ],
        "legend_labels": [spec["label"] for spec in METHOD_SPECS],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(
    rows: list[source.MetricRow],
    dataset_specs: list[tuple[str, str, str]],
    output_dir: Path,
    output_stem: str,
    ylabel_pad: float,
) -> tuple[Path, Path]:
    power_style.set_times_new_roman(power_style.USER_TIMES_TTF, power_style.FONT_FAMILY)
    power_style.set_global_rc()
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup = source.build_lookup(rows)
    dataset_labels = [dataset_short for _, _, dataset_short in dataset_specs]
    group_span = len(METHOD_SPECS) * (power_style.TOP10_BAR_WIDTH + power_style.TOP10_BAR_GAP)
    x_values = np.arange(len(dataset_specs)) * (group_span + power_style.TOP10_GROUP_GAP)

    fig, ax = plt.subplots(figsize=FIG_SS)

    for idx, spec in enumerate(METHOD_SPECS):
        offset = (idx - (len(METHOD_SPECS) - 1) / 2.0) * (
            power_style.TOP10_BAR_WIDTH + power_style.TOP10_BAR_GAP
        )
        values = [
            lookup[(dataset, variant, spec["design"])].row_miss_per_read
            for dataset, variant, _ in dataset_specs
        ]
        ax.bar(
            x_values + offset,
            values,
            power_style.TOP10_BAR_WIDTH,
            color=spec["color"],
            edgecolor=power_style.BAR_EDGE_COLOR,
            linewidth=power_style.BAR_LINEWIDTH,
            alpha=power_style.BAR_ALPHA,
            zorder=2.0,
        )

    flat_values = np.array(
        [
            lookup[(dataset, variant, spec["design"])].row_miss_per_read
            for dataset, variant, _ in dataset_specs
            for spec in METHOD_SPECS
        ],
        dtype=float,
    )
    ax.set_ylim(0.0, flat_values.max() * Y_MAX_SCALE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Row Misses Ratio", labelpad=ylabel_pad)
    ax.set_xticks(x_values)
    ax.set_xticklabels(dataset_labels, rotation=0, ha="center")
    ax.tick_params(axis="both", labelsize=power_style.TICK_LABEL_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35, zorder=0.0)
    ax.set_axisbelow(True)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    legend_handles = [
        Patch(
            facecolor=spec["color"],
            edgecolor=power_style.BAR_EDGE_COLOR,
            linewidth=power_style.BAR_LINEWIDTH,
            label=spec["label"],
        )
        for spec in METHOD_SPECS
    ]
    legend = ax.legend(
        handles=legend_handles,
        ncol=LEGEND_NCOL,
        fontsize=power_style.LEGEND_FONT_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=LEGEND_BBOX,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=LAYOUT_RECT, pad=TIGHT_LAYOUT_PAD)
    _, _, png_path, pdf_path = output_paths(output_dir, output_stem)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    rows, skipped, dataset_specs = collect_plot_rows(
        args.base_dir, args.ansmet_open_dir, args.dataset_mode
    )
    csv_path, json_path, _, _ = output_paths(args.output_dir, args.output_stem)
    write_csv(rows, csv_path)
    write_json(rows, skipped, dataset_specs, json_path)
    png_path, pdf_path = plot(
        rows,
        dataset_specs,
        args.output_dir,
        args.output_stem,
        args.ylabel_pad,
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")


if __name__ == "__main__":
    main()
