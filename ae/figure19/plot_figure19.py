#!/usr/bin/env python3
"""Validate and plot the portable Figure 19 RTC/JUNO++ comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import defaultdict
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
DEFAULT_INPUT = SCRIPT_DIR / "data/figure19_plot_data.tsv"
DEFAULT_JUNO_RAW = SCRIPT_DIR / "data/juno_fig8_designs.tsv"
DEFAULT_PROVENANCE = SCRIPT_DIR / "data/figure19_mfnns_provenance.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/figure19"
CHECKSUM_PATH = SCRIPT_DIR / "data/SHA256SUMS"

FIGURE_CONFIG = {
    "width_in": 3.33,
    "height_in": 1.7,
    "figure_dpi": 220,
    "savefig_dpi": 400,
    "save_pad_inches": 0.010,
}
FONT_CONFIG = {
    "body_pt": 7.0,
    "axis_label_pt": 6.0,
    "tick_pt": 5.0,
    "legend_pt": 5.4,
    "panel_label_pt": 5.8,
    "font_weight": "bold",
}
LAYOUT_CONFIG = {
    "subplots_left": 0.125,
    "subplots_right": 0.994,
    "subplots_bottom": 0.105,
    "subplots_top": 0.860,
    "subplots_wspace": 0.135,
    "subplots_hspace": 0.155,
    "global_xlabel_x": 0.55,
    "global_xlabel_y": 0.028,
    "global_ylabel_x": 0.07,
    "global_ylabel_y": 0.475,
}
AXIS_CONFIG = {
    "x_tick_pad": 0.8,
    "y_tick_pad": 0.7,
    "major_tick_length": 2.1,
    "major_tick_width": 0.58,
    "minor_tick_length": 1.35,
    "minor_tick_width": 0.46,
    "axes_linewidth": 0.68,
    "grid_major_width": 0.32,
    "grid_minor_width": 0.20,
    "grid_major_alpha": 0.25,
    "grid_minor_alpha": 0.14,
}

PANEL_ORDER = [
    ("sift1m", "r1"),
    ("t2i1m", "r1"),
    ("sift1m", "r100"),
    ("t2i1m", "r100"),
]
PANEL_TITLES = {
    ("sift1m", "r1"): "SIFT1M Recall@1",
    ("t2i1m", "r1"): "T2I1M Recall@1",
    ("sift1m", "r100"): "SIFT1M Recall@100",
    ("t2i1m", "r100"): "T2I1M Recall@100",
}
SERIES_ORDER = [
    "+HNSW",
    "JUNO-H",
    "JUNO-M",
    "JUNO-L",
    "JUNO++ frontier",
    "MFNNS frontier",
]
SERIES_LABELS = {
    "+HNSW": "HNSW",
    "JUNO-H": "JUNO-H",
    "JUNO-M": "JUNO-M",
    "JUNO-L": "JUNO-L",
    "JUNO++ frontier": "JUNO++",
    "MFNNS frontier": "MFNNS",
}
SERIES_STYLES = {
    "MFNNS frontier": {
        "color": "#bf1d2d",
        "marker": "o",
        "linewidth": 1.35,
        "markersize": 2.35,
        "markerfacecolor": "white",
        "markeredgewidth": 0.65,
        "linestyle": "-",
        "alpha": 1.0,
        "zorder": 6,
    },
    "JUNO++ frontier": {
        "color": "#222222",
        "marker": None,
        "linewidth": 1.55,
        "markersize": 0.0,
        "linestyle": "-",
        "alpha": 0.95,
        "zorder": 5,
    },
    "JUNO-H": {
        "color": "#4f4f4f",
        "marker": "o",
        "linewidth": 0.82,
        "markersize": 2.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.55,
        "linestyle": "-",
        "alpha": 0.88,
        "zorder": 4,
    },
    "JUNO-M": {
        "color": "#4f4f4f",
        "marker": "D",
        "linewidth": 0.82,
        "markersize": 1.9,
        "markerfacecolor": "white",
        "markeredgewidth": 0.55,
        "linestyle": "--",
        "alpha": 0.86,
        "zorder": 4,
    },
    "JUNO-L": {
        "color": "#4f4f4f",
        "marker": "x",
        "linewidth": 0.82,
        "markersize": 2.1,
        "markeredgewidth": 0.6,
        "linestyle": "-.",
        "alpha": 0.86,
        "zorder": 4,
    },
    "+HNSW": {
        "color": "#6f58a8",
        "marker": "P",
        "linewidth": 0.78,
        "markersize": 1.95,
        "markerfacecolor": "white",
        "markeredgewidth": 0.5,
        "linestyle": "-",
        "alpha": 0.82,
        "zorder": 3,
    },
}
EXPECTED_COUNTS = {
    ("sift1m", "r1", "+HNSW"): 10,
    ("sift1m", "r1", "JUNO++ frontier"): 14,
    ("sift1m", "r1", "JUNO-H"): 14,
    ("sift1m", "r1", "JUNO-L"): 7,
    ("sift1m", "r1", "JUNO-M"): 5,
    ("sift1m", "r1", "MFNNS frontier"): 37,
    ("sift1m", "r100", "+HNSW"): 9,
    ("sift1m", "r100", "JUNO++ frontier"): 14,
    ("sift1m", "r100", "JUNO-H"): 14,
    ("sift1m", "r100", "JUNO-L"): 7,
    ("sift1m", "r100", "JUNO-M"): 5,
    ("sift1m", "r100", "MFNNS frontier"): 71,
    ("t2i1m", "r1", "+HNSW"): 10,
    ("t2i1m", "r1", "JUNO++ frontier"): 11,
    ("t2i1m", "r1", "JUNO-H"): 6,
    ("t2i1m", "r1", "JUNO-L"): 6,
    ("t2i1m", "r1", "MFNNS frontier"): 37,
    ("t2i1m", "r100", "+HNSW"): 9,
    ("t2i1m", "r100", "JUNO++ frontier"): 11,
    ("t2i1m", "r100", "JUNO-H"): 6,
    ("t2i1m", "r100", "JUNO-L"): 6,
    ("t2i1m", "r100", "MFNNS frontier"): 44,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--juno-raw", type=Path, default=DEFAULT_JUNO_RAW)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksum_files(output: Path) -> list[Path]:
    return [
        SCRIPT_DIR / "data/juno_fig8_designs.tsv",
        SCRIPT_DIR / "data/figure19_plot_data.tsv",
        SCRIPT_DIR / "data/figure19_mfnns_provenance.csv",
        SCRIPT_DIR / "data/figure19_source_experiments.csv",
        SCRIPT_DIR / "data/figure19_frontier_better_ranges.tsv",
        SCRIPT_DIR / "data/figure19_frontier_speedup_samples.tsv",
        SCRIPT_DIR / "output/figure19_provenance_summary.tsv",
        output.with_suffix(".pdf"),
        output.with_suffix(".png"),
    ]


def write_checksums(output: Path) -> None:
    lines = [
        f"{sha256(path)}  {os.path.relpath(path, CHECKSUM_PATH.parent)}"
        for path in checksum_files(output)
    ]
    CHECKSUM_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_checksums(output: Path) -> None:
    expected: dict[str, str] = {}
    for line in CHECKSUM_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    files = checksum_files(output)
    actual_names = {os.path.relpath(path, CHECKSUM_PATH.parent) for path in files}
    if set(expected) != actual_names:
        raise ValueError("Figure 19 checksum inventory differs from expected files")
    for path in files:
        relative = os.path.relpath(path, CHECKSUM_PATH.parent)
        if sha256(path) != expected[relative]:
            raise ValueError(f"Figure 19 checksum mismatch: {relative}")


def load_and_validate(
    plot_path: Path, juno_raw_path: Path, provenance_path: Path
) -> dict[tuple[str, str, str], list[dict[str, float | str]]]:
    with plot_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(raw_rows) != 353:
        raise ValueError(f"Expected 353 Figure 19 rows, found {len(raw_rows)}")

    with juno_raw_path.open("r", encoding="utf-8", newline="") as handle:
        juno_raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(juno_raw_rows) != 277:
        raise ValueError(
            f"Expected 277 JUNO++ Fig. 8 raw rows, found {len(juno_raw_rows)}"
        )

    juno_selected: list[dict[str, str]] = []
    for row in juno_raw_rows:
        if row["source"] != "juno++_fig8_vector":
            raise ValueError(f"Unexpected JUNO++ raw source: {row['source']}")
        if row["pdf_page_1based"] != "17" or row["pdf_page_0based"] != "16":
            raise ValueError("JUNO++ raw data does not point to Fig. 8 PDF page 17")
        if not math.isclose(
            10 ** float(row["log10_qps"]),
            float(row["qps"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "JUNO++ raw log10(QPS) and QPS differ at "
                f"drawing={row['drawing_index']},point={row['point_index']}"
            )
        if row["plot_default"] == "1":
            juno_selected.append(
                {
                    "source": row["source"],
                    "dataset": row["dataset"],
                    "paper_dataset": row["paper_dataset"],
                    "recall_tag": row["recall_tag"],
                    "paper_metric": row["paper_metric"],
                    "series": row["series"],
                    "recall": row["recall"],
                    "qps": row["qps"],
                    "source_detail": (
                        f"pdf_page=17,drawing={row['drawing_index']}"
                    ),
                }
            )
        elif row["plot_default"] != "0":
            raise ValueError(f"Invalid JUNO++ plot_default: {row['plot_default']}")
    if len(juno_selected) != 164:
        raise ValueError(
            f"Expected 164 selected JUNO++/HNSW rows, found {len(juno_selected)}"
        )
    juno_selected.sort(
        key=lambda row: (
            row["dataset"],
            row["recall_tag"],
            row["series"],
            float(row["recall"]),
        )
    )
    plot_juno_rows = [
        row for row in raw_rows if row["series"] != "MFNNS frontier"
    ]
    if juno_selected != plot_juno_rows:
        raise ValueError(
            "JUNO++ raw plot_default rows do not exactly reproduce Figure 19 inputs"
        )

    rows: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    mfnns_plot_keys: set[tuple[str, str, str]] = set()
    for row in raw_rows:
        series = row["series"]
        if series not in SERIES_ORDER:
            raise ValueError(f"Unexpected Figure 19 series: {series}")
        recall = float(row["recall"])
        qps = float(row["qps"])
        if not (0.0 <= recall <= 1.01) or qps <= 0:
            raise ValueError(f"Invalid Figure 19 point: {row}")
        key = (row["dataset"], row["recall_tag"], series)
        rows[key].append(
            {
                "recall": recall,
                "qps": qps,
                "source": row["source"],
                "source_detail": row["source_detail"],
            }
        )
        if series == "MFNNS frontier":
            mfnns_plot_keys.add((row["dataset"], row["recall_tag"], row["source_detail"]))

    actual_counts = {key: len(value) for key, value in rows.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError("Figure 19 panel/series point counts changed")
    for values in rows.values():
        values.sort(key=lambda row: float(row["recall"]))

    with provenance_path.open("r", encoding="utf-8", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    if len(provenance) != 189:
        raise ValueError(f"Expected 189 MFNNS provenance rows, found {len(provenance)}")
    provenance_keys: set[tuple[str, str, str]] = set()
    within_xlim = 0
    for row in provenance:
        if row["data_status"] != "measured_completed_simulator":
            raise ValueError(f"Unexpected MFNNS data status: {row['data_status']}")
        for field in (
            "test_record_ref",
            "yaml_generator_ref",
            "result_summarizer_ref",
            "case_manifest_ref",
            "yaml_ref",
            "stats_ref",
        ):
            if Path(row[field]).is_absolute():
                raise ValueError(f"Absolute Figure 19 provenance ref: {row[field]}")
        recall = float(row["recall"])
        expected_visible = int(0.4 <= recall <= 1.01)
        if int(row["point_within_xlim"]) != expected_visible:
            raise ValueError(f"Incorrect xlim visibility for {row['case_name']}")
        within_xlim += expected_visible
        provenance_keys.add((row["dataset"], row["recall_tag"], row["case_name"]))
    if within_xlim != 160:
        raise ValueError(f"Expected 160 MFNNS points within xlim, found {within_xlim}")
    if provenance_keys != mfnns_plot_keys:
        raise ValueError("MFNNS plot/provenance case sets differ")
    return rows


def set_global_style() -> None:
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
            "font.size": FONT_CONFIG["body_pt"],
            "font.weight": FONT_CONFIG["font_weight"],
            "axes.labelweight": FONT_CONFIG["font_weight"],
            "axes.titleweight": FONT_CONFIG["font_weight"],
            "axes.linewidth": AXIS_CONFIG["axes_linewidth"],
            "xtick.labelsize": FONT_CONFIG["tick_pt"],
            "ytick.labelsize": FONT_CONFIG["tick_pt"],
            "legend.fontsize": FONT_CONFIG["legend_pt"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def visible_qps(
    rows: dict[tuple[str, str, str], list[dict[str, float | str]]],
    dataset: str,
    recall_tag: str,
) -> list[float]:
    return [
        float(row["qps"])
        for series in SERIES_ORDER
        for row in rows.get((dataset, recall_tag, series), [])
        if 0.4 <= float(row["recall"]) <= 1.01
    ]


def choose_ylim(qps_values: list[float]) -> tuple[float, float]:
    low_power = math.floor(math.log10(min(qps_values) * 0.72))
    target_high = max(qps_values) * 1.18
    high_power = math.floor(math.log10(target_high))
    high_base = 10**high_power
    high_mantissa = target_high / high_base
    for step in (1.0, 2.0, 3.0, 5.0, 10.0):
        if high_mantissa <= step:
            return 10**low_power, step * high_base
    raise AssertionError("Unreachable y-limit branch")


def style_axis(ax: plt.Axes, y_limits: tuple[float, float]) -> None:
    ax.grid(
        True,
        which="major",
        linewidth=AXIS_CONFIG["grid_major_width"],
        alpha=AXIS_CONFIG["grid_major_alpha"],
        color="#bdbdbd",
        zorder=0,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linewidth=AXIS_CONFIG["grid_minor_width"],
        alpha=AXIS_CONFIG["grid_minor_alpha"],
        color="#bdbdbd",
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        which="major",
        length=AXIS_CONFIG["major_tick_length"],
        width=AXIS_CONFIG["major_tick_width"],
    )
    ax.tick_params(
        which="minor",
        length=AXIS_CONFIG["minor_tick_length"],
        width=AXIS_CONFIG["minor_tick_width"],
    )
    ax.tick_params(axis="x", which="major", pad=AXIS_CONFIG["x_tick_pad"])
    ax.tick_params(axis="y", which="major", pad=AXIS_CONFIG["y_tick_pad"])
    first_exp = math.ceil(math.log10(y_limits[0]))
    last_exp = math.floor(math.log10(y_limits[1]))
    ax.yaxis.set_major_locator(
        FixedLocator([10**exp for exp in range(first_exp, last_exp + 1)])
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=10)
    )
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))


def legend_handles() -> list[Line2D]:
    handles = []
    for series in SERIES_ORDER:
        style = dict(SERIES_STYLES[series])
        marker = style.pop("marker")
        handles.append(
            Line2D([0], [0], marker=marker, label=SERIES_LABELS[series], **style)
        )
    return handles


def plot(
    rows: dict[tuple[str, str, str], list[dict[str, float | str]]],
    output: Path,
) -> None:
    set_global_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_CONFIG["width_in"], FIGURE_CONFIG["height_in"]),
        dpi=FIGURE_CONFIG["figure_dpi"],
    )
    for index, (ax, (dataset, recall_tag)) in enumerate(
        zip(axes.ravel(), PANEL_ORDER)
    ):
        for series in SERIES_ORDER:
            points = rows.get((dataset, recall_tag, series), [])
            if not points:
                continue
            style = dict(SERIES_STYLES[series])
            marker = style.pop("marker")
            ax.plot(
                [float(row["recall"]) for row in points],
                [float(row["qps"]) for row in points],
                marker=marker,
                solid_capstyle="round",
                dash_capstyle="round",
                **style,
            )
        ax.set_yscale("log")
        ax.set_xlim(0.4, 1.01)
        y_limits = choose_ylim(visible_qps(rows, dataset, recall_tag))
        ax.set_ylim(*y_limits)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.text(
            0.975,
            0.985,
            PANEL_TITLES[(dataset, recall_tag)],
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#b21f1f",
            fontsize=FONT_CONFIG["panel_label_pt"],
            fontweight=FONT_CONFIG["font_weight"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 0.40,
            },
        )
        style_axis(ax, y_limits)
        if index < 2:
            ax.tick_params(axis="x", labelbottom=False)

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.515, 0.928),
        ncol=6,
        frameon=False,
        handlelength=0.92,
        columnspacing=0.42,
        handletextpad=0.22,
        borderaxespad=0.0,
    )
    fig.text(
        LAYOUT_CONFIG["global_xlabel_x"],
        LAYOUT_CONFIG["global_xlabel_y"],
        "Recall",
        ha="center",
        va="center",
        fontsize=FONT_CONFIG["axis_label_pt"],
        fontweight=FONT_CONFIG["font_weight"],
    )
    fig.text(
        LAYOUT_CONFIG["global_ylabel_x"],
        LAYOUT_CONFIG["global_ylabel_y"],
        "QPS",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FONT_CONFIG["axis_label_pt"],
        fontweight=FONT_CONFIG["font_weight"],
    )
    fig.subplots_adjust(
        left=LAYOUT_CONFIG["subplots_left"],
        right=LAYOUT_CONFIG["subplots_right"],
        bottom=LAYOUT_CONFIG["subplots_bottom"],
        top=LAYOUT_CONFIG["subplots_top"],
        wspace=LAYOUT_CONFIG["subplots_wspace"],
        hspace=LAYOUT_CONFIG["subplots_hspace"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=FIGURE_CONFIG["save_pad_inches"],
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        output.with_suffix(".png"),
        bbox_inches="tight",
        pad_inches=FIGURE_CONFIG["save_pad_inches"],
        dpi=FIGURE_CONFIG["savefig_dpi"],
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_and_validate(args.input, args.juno_raw, args.provenance)
    if args.check_only:
        validate_checksums(args.output)
        return
    plot(rows, args.output)
    write_checksums(args.output)


if __name__ == "__main__":
    main()
