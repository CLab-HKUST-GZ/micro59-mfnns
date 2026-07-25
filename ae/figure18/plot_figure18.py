#!/usr/bin/env python3
"""Validate and plot Figure 18 billion-scale recall-QPS curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FormatStrFormatter,
    LogFormatterMathtext,
    LogLocator,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data/figure18_recall_qps.csv"
DEFAULT_PROVENANCE = SCRIPT_DIR / "data/simulator_provenance.tsv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/figure18"
DEFAULT_SUMMARY = SCRIPT_DIR / "output/figure18_summary.tsv"
DEFAULT_RECALL_DIGITS = 2
FREQ_HZ = 2_400_000_000.0
NQ = 100

PANEL_ORDER = [
    ("t2i1b", "r10"),
    ("deep1b", "r10"),
    ("t2i1b", "r100"),
    ("deep1b", "r100"),
]
PANEL_TITLES = {
    ("t2i1b", "r10"): "T2I1B Recall@10",
    ("deep1b", "r10"): "DP1B Recall@10",
    ("t2i1b", "r100"): "T2I1B Recall@100",
    ("deep1b", "r100"): "DP1B Recall@100",
}
METHOD_ORDER = ["cpu", "bang", "ansmet", "mfnns"]
METHOD_LABELS = {
    "cpu": "CPU",
    "bang": "BANG",
    "ansmet": "ANSMET",
    "mfnns": "MFNNS",
}
EXPECTED_COUNTS = {
    ("t2i1b", "r10", "cpu"): 8,
    ("t2i1b", "r10", "bang"): 12,
    ("t2i1b", "r10", "ansmet"): 14,
    ("t2i1b", "r10", "mfnns"): 15,
    ("deep1b", "r10", "cpu"): 8,
    ("deep1b", "r10", "bang"): 10,
    ("deep1b", "r10", "ansmet"): 14,
    ("deep1b", "r10", "mfnns"): 16,
    ("t2i1b", "r100", "cpu"): 8,
    ("t2i1b", "r100", "bang"): 8,
    ("t2i1b", "r100", "ansmet"): 11,
    ("t2i1b", "r100", "mfnns"): 14,
    ("deep1b", "r100", "cpu"): 7,
    ("deep1b", "r100", "bang"): 8,
    ("deep1b", "r100", "ansmet"): 11,
    ("deep1b", "r100", "mfnns"): 13,
}
EXPECTED_CONFIG_STATUS = {
    "verified_original_yaml_stats": 88,
    "original_yaml_cycle_mismatch": 1,
    "reconstructed_from_same_panel_template": 19,
}
METHOD_STYLES = {
    "cpu": {
        "color": "#444444",
        "marker": "^",
        "linewidth": 0.82,
        "markersize": 2.00,
        "markerfacecolor": "white",
        "markeredgewidth": 0.55,
        "linestyle": "--",
        "alpha": 0.86,
        "zorder": 3,
    },
    "bang": {
        "color": "#6f58a8",
        "marker": "D",
        "linewidth": 0.82,
        "markersize": 1.90,
        "markerfacecolor": "white",
        "markeredgewidth": 0.55,
        "linestyle": "-.",
        "alpha": 0.86,
        "zorder": 3,
    },
    "ansmet": {
        "color": "#4c78a8",
        "marker": "o",
        "linewidth": 0.92,
        "markersize": 2.05,
        "markerfacecolor": "white",
        "markeredgewidth": 0.58,
        "linestyle": "-",
        "alpha": 0.90,
        "zorder": 4,
    },
    "mfnns": {
        "color": "#bf1d2d",
        "marker": "s",
        "linewidth": 1.32,
        "markersize": 2.25,
        "markerfacecolor": "white",
        "markeredgewidth": 0.65,
        "linestyle": "-",
        "alpha": 1.00,
        "zorder": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--recall-digits",
        type=int,
        default=DEFAULT_RECALL_DIGITS,
        help="Round plotted recall with decimal ROUND_HALF_UP (default: 2).",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_recall(text: str, digits: int) -> Decimal:
    if digits < 0:
        raise ValueError("recall digits must be non-negative")
    quantum = Decimal(1).scaleb(-digits)
    return Decimal(text).quantize(quantum, rounding=ROUND_HALF_UP)


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def parse_yaml_int(text: str, key: str) -> int:
    match = re.search(rf"^  {re.escape(key)}:\s*([0-9]+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing top-level YAML integer: {key}")
    return int(match.group(1))


def validate_config(row: dict[str, str]) -> None:
    path = REPO_ROOT / row["portable_config_ref"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != row["portable_config_sha256"]:
        raise ValueError(f"Portable config SHA mismatch: {path}")
    text = path.read_text(encoding="utf-8")
    expected_k = 10 if row["recall_tag"] == "r10" else 100
    if parse_yaml_int(text, "k_neighbors") != expected_k:
        raise ValueError(f"k_neighbors mismatch: {row['point_id']}")
    if parse_yaml_int(text, "gt_k") < expected_k:
        raise ValueError(f"gt_k is below k_neighbors: {row['point_id']}")
    if parse_yaml_int(text, "nQueryLimit") != NQ:
        raise ValueError(f"nQueryLimit mismatch: {row['point_id']}")
    if parse_yaml_int(text, "nParallelQuery") != NQ:
        raise ValueError(f"nParallelQuery mismatch: {row['point_id']}")
    if parse_yaml_int(text, "ef_search") != int(row["ef"]):
        raise ValueError(f"ef_search mismatch: {row['point_id']}")
    if row["method"] == "mfnns":
        if parse_yaml_int(text, "dualQueueLowerBoundQueueSize") != int(row["lbq"]):
            raise ValueError(f"LBQueue mismatch: {row['point_id']}")
        if not re.search(r"^  mfnnsEnable:\s*true\s*$", text, re.MULTILINE):
            raise ValueError(f"MFNNS is not enabled: {row['point_id']}")
    elif not re.search(r"^  mfnnsEnable:\s*false\s*$", text, re.MULTILINE):
        raise ValueError(f"ANSMET config does not disable MFNNS: {row['point_id']}")
    stat_match = re.search(r"^  stat_path:\s*(.+?)\s*$", text, re.MULTILINE)
    if not stat_match or Path(stat_match.group(1)).is_absolute():
        raise ValueError(f"stat_path is not portable: {row['point_id']}")
    model_match = re.search(r"^  model_path:\s*(.+?)\s*$", text, re.MULTILINE)
    if not model_match:
        raise ValueError(f"Missing model path: {row['point_id']}")
    model = model_match.group(1).lower()
    if row["dataset"] == "deep1b" and "deep1b" not in model:
        raise ValueError(f"Deep1B model mismatch: {row['point_id']}")
    if row["dataset"] == "t2i1b" and not (
        "text2img1b" in model or "t2i1b" in model
    ):
        raise ValueError(f"T2I1B model mismatch: {row['point_id']}")


def data_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["dataset"],
        row["recall_tag"],
        row["method"],
        row["cycle"],
        row["ef"],
        row["lbq"],
    )


def provenance_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["dataset"],
        row["recall_tag"],
        row["method"],
        row["plot_cycle"],
        row["ef"],
        row["lbq"],
    )


def validate(
    data_rows: list[dict[str, str]],
    provenance_rows: list[dict[str, str]],
    recall_digits: int,
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    actual_counts = Counter(
        (row["dataset"], row["recall_tag"], row["method"]) for row in data_rows
    )
    if dict(actual_counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"Figure 18 point matrix differs: {dict(actual_counts)}"
        )

    provenance = {provenance_key(row): row for row in provenance_rows}
    if len(provenance) != 108:
        raise ValueError(
            f"Expected 108 unique simulator provenance rows, found {len(provenance)}"
        )
    status_counts = Counter(row["config_status"] for row in provenance_rows)
    if dict(status_counts) != EXPECTED_CONFIG_STATUS:
        raise ValueError(f"Unexpected config status counts: {dict(status_counts)}")

    series: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    simulator_rows = 0
    for row in data_rows:
        recall = rounded_recall(row["recall_raw"], recall_digits)
        if recall_digits == 2 and recall != Decimal(row["recall_2dp"]):
            raise ValueError(f"Stored two-decimal recall mismatch: {data_key(row)}")
        qps = float(row["qps"])
        if not math.isfinite(qps) or qps <= 0:
            raise ValueError(f"Invalid QPS: {data_key(row)}")
        if row["method"] in {"ansmet", "mfnns"}:
            simulator_rows += 1
            if data_key(row) not in provenance:
                raise ValueError(f"Missing simulator provenance: {data_key(row)}")
            cycle = int(row["cycle"])
            expected_qps = NQ * FREQ_HZ / cycle
            if not math.isclose(qps, expected_qps, rel_tol=0, abs_tol=1e-6):
                raise ValueError(
                    f"Simulator QPS/cycle mismatch: {data_key(row)}, "
                    f"{qps} != {expected_qps}"
                )
        series[(row["dataset"], row["recall_tag"], row["method"])].append(
            {
                "recall": float(recall),
                "recall_raw": float(row["recall_raw"]),
                "qps": qps,
            }
        )
    if simulator_rows != 108:
        raise ValueError(f"Expected 108 simulator rows, found {simulator_rows}")
    for values in series.values():
        values.sort(key=lambda item: (float(item["recall"]), float(item["recall_raw"])))
    for row in provenance_rows:
        validate_config(row)
    return series


def set_plot_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "serif"
    for candidate in (
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ):
        if candidate in available:
            family = candidate
            break
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 7.0,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 0.68,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 5.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def choose_ylim(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    low_power = math.floor(math.log10(low * 0.72))
    target_high = high * 1.18
    high_power = math.floor(math.log10(target_high))
    high_base = 10**high_power
    mantissa = target_high / high_base
    for step in (1.0, 2.0, 3.0, 5.0, 10.0):
        if mantissa <= step:
            return 10**low_power, step * high_base
    return 10**low_power, 10.0 * high_base


def legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        style = dict(METHOD_STYLES[method])
        marker = style.pop("marker")
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                label=METHOD_LABELS[method],
                **style,
            )
        )
    return handles


def plot(
    series: dict[tuple[str, str, str], list[dict[str, object]]],
    output: Path,
    recall_digits: int,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.33, 1.70), dpi=220)
    for index, (axis, panel) in enumerate(zip(axes.ravel(), PANEL_ORDER)):
        dataset, recall_tag = panel
        visible_qps: list[float] = []
        for method in METHOD_ORDER:
            values = series[(dataset, recall_tag, method)]
            style = dict(METHOD_STYLES[method])
            marker = style.pop("marker")
            axis.plot(
                [float(item["recall"]) for item in values],
                [float(item["qps"]) for item in values],
                marker=marker,
                label=METHOD_LABELS[method],
                solid_capstyle="round",
                dash_capstyle="round",
                **style,
            )
            visible_qps.extend(
                float(item["qps"])
                for item in values
                if 0.50 <= float(item["recall"]) <= 1.01
            )
        axis.set_yscale("log")
        axis.set_xlim(0.50, 1.01)
        axis.set_ylim(*choose_ylim(visible_qps))
        axis.set_xticks([0.60, 0.80, 1.00])
        axis.xaxis.set_major_formatter(FormatStrFormatter(f"%.{recall_digits}f"))
        low, high = axis.get_ylim()
        major = [
            10**power
            for power in range(
                math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1
            )
        ]
        axis.yaxis.set_major_locator(FixedLocator(major))
        axis.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=10)
        )
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.grid(
            True,
            which="major",
            linewidth=0.32,
            alpha=0.25,
            color="#bdbdbd",
            zorder=0,
        )
        axis.grid(
            True,
            which="minor",
            axis="y",
            linewidth=0.20,
            alpha=0.14,
            color="#bdbdbd",
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", which="major", length=2.1, width=0.58, pad=0.8)
        axis.tick_params(axis="y", which="major", length=2.1, width=0.58, pad=0.7)
        axis.text(
            0.975,
            0.985,
            PANEL_TITLES[panel],
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#b21f1f",
            fontsize=5.8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.4},
        )
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.515, 0.93),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.58,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    fig.text(0.55, 0.028, "Recall", ha="center", va="center", fontsize=6.0)
    fig.text(
        0.070,
        0.465,
        "QPS",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
    )
    fig.subplots_adjust(
        left=0.125,
        right=0.994,
        bottom=0.105,
        top=0.845,
        wspace=0.140,
        hspace=0.165,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.010,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        output.with_suffix(".png"),
        bbox_inches="tight",
        pad_inches=0.010,
        dpi=400,
    )
    plt.close(fig)


def write_summary(
    path: Path,
    data_rows: list[dict[str, str]],
    provenance_rows: list[dict[str, str]],
    recall_digits: int,
) -> None:
    status = Counter(row["config_status"] for row in provenance_rows)
    lines = [
        "metric\tvalue",
        f"plot_rows\t{len(data_rows)}",
        f"simulator_rows\t{len(provenance_rows)}",
        f"recall_digits\t{recall_digits}",
        "recall_rounding\tROUND_HALF_UP",
    ]
    for name in sorted(status):
        lines.append(f"config_status:{name}\t{status[name]}")
    for key in sorted(EXPECTED_COUNTS):
        dataset, recall_tag, method = key
        lines.append(
            f"points:{dataset}:{recall_tag}:{method}\t{EXPECTED_COUNTS[key]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_rows = read_rows(args.data)
    provenance_rows = read_rows(args.provenance, delimiter="\t")
    series = validate(data_rows, provenance_rows, args.recall_digits)
    if args.check_only:
        print(
            f"Validated {len(data_rows)} Figure 18 rows, "
            f"{len(provenance_rows)} simulator configs, "
            f"recall_digits={args.recall_digits}"
        )
        return
    plot(series, args.output_prefix, args.recall_digits)
    write_summary(
        args.summary_out,
        data_rows,
        provenance_rows,
        args.recall_digits,
    )
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
