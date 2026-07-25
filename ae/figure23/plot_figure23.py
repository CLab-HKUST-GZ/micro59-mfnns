#!/usr/bin/env python3
"""Validate the frozen historical data and reproduce paper Figure 23."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data" / "figure23_row_miss_ratio.csv"
DEFAULT_PROVENANCE = SCRIPT_DIR / "data" / "config_provenance.tsv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "figure23"
DEFAULT_SUMMARY = SCRIPT_DIR / "output" / "figure23_summary.tsv"

DATASET_ORDER = [
    ("deep10m", "DP"),
    ("glove2m", "GV"),
    ("sift1m", "SF"),
    ("t2i1m", "T2I"),
    ("w2v1m", "W2V"),
    ("wiki1m", "WK"),
    ("pubmed", "PM"),
]
METHOD_SPECS = [
    {"design": "ansmet_open", "label": "ANSMET", "color": "#6ab3b5"},
    {"design": "ndp_et", "label": "NMP-FPSA-ET", "color": "#c96552"},
    {"design": "mfnns", "label": "MFNNS", "color": "#ebd3a7"},
]

FIG_SIZE = (8.6, 2.3)
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
BAR_EDGE_COLOR = "#000000"
BAR_LINEWIDTH = 0.5
BAR_ALPHA = 0.98
BAR_WIDTH = 0.115
BAR_GAP = 0.005
GROUP_GAP = 0.24
LEGEND_NCOL = 3
LEGEND_BBOX = (0.5, 1.23)
LAYOUT_RECT = (0.0, 0.0, 0.992, 0.93)
Y_MAX_SCALE = 1.06
TIGHT_LAYOUT_PAD = 0.12
SAVE_PAD_INCHES = 0.03
YLABEL_PAD = 2.0

EXPECTED_DATA_SHA256 = (
    "ad5af39e426a12b5a7cd1dc49605590e2c47fa17f174dd268846fdccb78aac59"
)
EXPECTED_METRICS = {
    "arithmetic_mean_mfnns_reduction_vs_ndp_pct": 73.44427317134844,
    "geometric_mean_ndp_over_mfnns": 5.408054394294337,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate frozen data and YAML provenance without rewriting outputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def relative_author_ref(path_text: str) -> str:
    marker = "/MFANNS/"
    if marker not in path_text:
        raise ValueError(f"Historical path does not contain {marker}: {path_text}")
    return path_text.split(marker, 1)[1]


def load_provenance(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    required = {
        "dataset",
        "dataset_short",
        "design",
        "config_ref",
        "config_sha256",
        "source_yaml_ref",
        "stats_ref",
        "stats_sha256",
        "source_slurm_ref",
        "source_slurm_sha256",
        "row_policy",
        "ef_search",
        "queue_size",
        "recall",
        "n_query",
        "gt_k",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing provenance columns: {sorted(missing)}")
        rows = list(reader)

    expected_keys = {
        (dataset, method["design"])
        for dataset, _ in DATASET_ORDER
        for method in METHOD_SPECS
    }
    observed_keys = {(row["dataset"], row["design"]) for row in rows}
    if len(rows) != 21 or observed_keys != expected_keys:
        raise ValueError(
            "Expected 21 unique provenance rows; "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )

    dataset_labels = dict(DATASET_ORDER)
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["design"])
        if row["dataset_short"] != dataset_labels[row["dataset"]]:
            raise ValueError(f"Dataset label mismatch for {key}")
        config_ref = Path(row["config_ref"])
        if config_ref.is_absolute() or ".." in config_ref.parts:
            raise ValueError(f"Non-portable config_ref for {key}: {config_ref}")
        config_path = SCRIPT_DIR / config_ref
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing archived YAML for {key}: {config_path}")
        if sha256(config_path) != row["config_sha256"]:
            raise ValueError(f"Archived YAML digest mismatch for {key}")
        config_text = config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(config_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"YAML root is not a mapping for {key}")
        if f"impl: {row['row_policy']}" not in config_text:
            raise ValueError(f"Row-policy mismatch in archived YAML for {key}")
        if int(row["n_query"]) != 1000 or int(row["gt_k"]) != 10:
            raise ValueError(f"Unexpected query/k configuration for {key}")
        for field in (
            "source_yaml_ref",
            "stats_ref",
            "source_slurm_ref",
        ):
            if Path(row[field]).is_absolute():
                raise ValueError(f"Absolute historical reference in {field} for {key}")
        for field in (
            "config_sha256",
            "stats_sha256",
            "source_slurm_sha256",
        ):
            if len(row[field]) != 64:
                raise ValueError(f"Invalid SHA-256 in {field} for {key}")
        lookup[key] = row
    return lookup


def load_rows(
    path: Path,
    provenance: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    required = {
        "dataset",
        "variant",
        "dataset_short",
        "design",
        "source_dir",
        "run_dir",
        "slurm_file",
        "occurrences",
        "sum_num_read_reqs_0",
        "sum_service_cycles_0",
        "sum_row_misses_0",
        "service_cycles_per_read",
        "row_miss_per_read",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing frozen-data columns: {sorted(missing)}")
        raw_rows = list(reader)

    if path.resolve() == DEFAULT_DATA.resolve() and sha256(path) != EXPECTED_DATA_SHA256:
        raise ValueError("Canonical frozen-data SHA-256 mismatch")

    expected_keys = {
        (dataset, method["design"])
        for dataset, _ in DATASET_ORDER
        for method in METHOD_SPECS
    }
    observed_keys = {(row["dataset"], row["design"]) for row in raw_rows}
    if len(raw_rows) != 21 or observed_keys != expected_keys:
        raise ValueError(
            "Expected 21 unique plot rows; "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )

    labels = dict(DATASET_ORDER)
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        key = (raw["dataset"], raw["design"])
        if raw["dataset_short"] != labels[raw["dataset"]]:
            raise ValueError(f"Dataset label mismatch for {key}")
        if int(raw["occurrences"]) != 32:
            raise ValueError(f"Expected 32 memory instances for {key}")

        reads = int(raw["sum_num_read_reqs_0"])
        service = int(raw["sum_service_cycles_0"])
        misses = int(raw["sum_row_misses_0"])
        if min(reads, service, misses) < 0 or reads == 0:
            raise ValueError(f"Invalid frozen counters for {key}")
        service_per_read = service / reads
        row_miss_per_read = misses / reads
        if not close(service_per_read, float(raw["service_cycles_per_read"])):
            raise ValueError(f"service_cycles_per_read mismatch for {key}")
        if not close(row_miss_per_read, float(raw["row_miss_per_read"])):
            raise ValueError(f"row_miss_per_read mismatch for {key}")

        provenance_row = provenance[key]
        if relative_author_ref(raw["slurm_file"]) != provenance_row["source_slurm_ref"]:
            raise ValueError(f"Slurm provenance mismatch for {key}")
        rows.append(
            {
                "dataset": raw["dataset"],
                "dataset_short": raw["dataset_short"],
                "design": raw["design"],
                "row_miss_per_read": row_miss_per_read,
            }
        )
    return rows


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    lookup = {
        (str(row["dataset"]), str(row["design"])): float(
            row["row_miss_per_read"]
        )
        for row in rows
    }
    reductions = []
    ratios = []
    for dataset, _ in DATASET_ORDER:
        ndp = lookup[(dataset, "ndp_et")]
        mfnns = lookup[(dataset, "mfnns")]
        reductions.append((1.0 - mfnns / ndp) * 100.0)
        ratios.append(ndp / mfnns)
    metrics = {
        "arithmetic_mean_mfnns_reduction_vs_ndp_pct": sum(reductions)
        / len(reductions),
        "geometric_mean_ndp_over_mfnns": math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        ),
    }
    for key, expected in EXPECTED_METRICS.items():
        if not close(metrics[key], expected):
            raise ValueError(
                f"Headline metric mismatch for {key}: "
                f"expected {expected}, got {metrics[key]}"
            )
    return metrics


def set_style() -> None:
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
            "axes.titleweight": "bold",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.25,
        }
    )


def plot(rows: list[dict[str, object]], output_prefix: Path) -> tuple[Path, Path]:
    set_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    lookup = {
        (str(row["dataset"]), str(row["design"])): float(
            row["row_miss_per_read"]
        )
        for row in rows
    }
    dataset_labels = [short for _, short in DATASET_ORDER]
    group_span = len(METHOD_SPECS) * (BAR_WIDTH + BAR_GAP)
    x_values = np.arange(len(DATASET_ORDER)) * (group_span + GROUP_GAP)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for index, method in enumerate(METHOD_SPECS):
        offset = (index - (len(METHOD_SPECS) - 1) / 2.0) * (
            BAR_WIDTH + BAR_GAP
        )
        values = [
            lookup[(dataset, method["design"])] for dataset, _ in DATASET_ORDER
        ]
        ax.bar(
            x_values + offset,
            values,
            BAR_WIDTH,
            color=method["color"],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            alpha=BAR_ALPHA,
            zorder=2.0,
        )

    flat_values = np.array(
        [
            lookup[(dataset, method["design"])]
            for dataset, _ in DATASET_ORDER
            for method in METHOD_SPECS
        ],
        dtype=float,
    )
    ax.set_ylim(0.0, flat_values.max() * Y_MAX_SCALE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Row Misses Ratio", labelpad=YLABEL_PAD)
    ax.set_xticks(x_values)
    ax.set_xticklabels(dataset_labels, rotation=0, ha="center")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
        zorder=0.0,
    )
    ax.set_axisbelow(True)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    legend_handles = [
        Patch(
            facecolor=method["color"],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label=method["label"],
        )
        for method in METHOD_SPECS
    ]
    legend = ax.legend(
        handles=legend_handles,
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=LEGEND_BBOX,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=LAYOUT_RECT, pad=TIGHT_LAYOUT_PAD)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return png_path, pdf_path


def write_summary(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("metric\tvalue\n")
        for key in (
            "arithmetic_mean_mfnns_reduction_vs_ndp_pct",
            "geometric_mean_ndp_over_mfnns",
        ):
            handle.write(f"{key}\t{metrics[key]:.15g}\n")


def main() -> None:
    args = parse_args()
    provenance = load_provenance(args.provenance)
    rows = load_rows(args.data, provenance)
    metrics = compute_metrics(rows)
    print(
        "CHECK_OK rows={} yamls={} mean_reduction={:.12f}% "
        "geomean_ratio={:.12f}x".format(
            len(rows),
            len(provenance),
            metrics["arithmetic_mean_mfnns_reduction_vs_ndp_pct"],
            metrics["geometric_mean_ndp_over_mfnns"],
        )
    )
    if args.check_only:
        return
    png_path, pdf_path = plot(rows, args.output_prefix)
    write_summary(args.summary, metrics)
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
