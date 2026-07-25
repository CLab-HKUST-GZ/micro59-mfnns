#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SOURCE_TSV = (
    REPO_ROOT
    / "simulator"
    / "memory"
    / "20260329"
    / "mfnns_t2i_k10_ef20_30_40_dualq_q20_100"
    / "summary_latest.tsv"
)
DEFAULT_ANSMET_STATS_DIR = (
    REPO_ROOT
    / "simulator"
    / "memory"
    / "20260329"
    / "ansmet_recall09_k10_efsearch"
    / "logs"
)
DEFAULT_SAVE_PREFIX = "t2i_lbq_recall_throughput_singlecol"
DEFAULT_DATA_TSV = SCRIPT_DIR / f"{DEFAULT_SAVE_PREFIX}_data.tsv"

ACM_COLUMN_WIDTH_IN = 3.33
FIGURE_HEIGHT_IN = 1
BODY_FONT_PT = 9.0
AXES_LABEL_PT = 7.0
TICK_FONT_PT = 5.9
LEGEND_FONT_PT = 5.9
USER_TIMES_TTF = ""

DEFAULT_COLORS = {
    "recall": "#bf1d2d",
    "throughput": "#4c78a8",
    "grid": "#bdbdbd",
    "target": "#767676",
    "separator": "#7a7a7a",
}

EF_ORDER = [20, 30, 40]
LBQ_MIN = 20
LBQ_MAX = 100
LBQ_SPAN = LBQ_MAX - LBQ_MIN
BLOCK_GAP = 24
BLOCK_TICK_VALUES = [20, 60, 100]
MARK_EVERY = [0, 20, 40, 60, 80]
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single-axis ACM single-column figure for the t2i LBQ sweep. "
            "The figure uses one shared left/right y-axis pair and vertical separators "
            "to split ef_search={20,30,40} into three horizontal blocks."
        )
    )
    parser.add_argument(
        "--source-tsv",
        type=Path,
        default=DEFAULT_SOURCE_TSV,
        help="Input TSV generated from the t2i LBQ sweep.",
    )
    parser.add_argument(
        "--save-prefix",
        default=DEFAULT_SAVE_PREFIX,
        help="Output figure prefix, relative to this script unless absolute.",
    )
    parser.add_argument(
        "--ansmet-stats-dir",
        type=Path,
        default=DEFAULT_ANSMET_STATS_DIR,
        help="Directory containing ANSMET stats yml files used as per-panel recall upper bounds.",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=DEFAULT_DATA_TSV,
        help="Optional TSV export with plotted recall and normalized throughput.",
    )
    parser.add_argument(
        "--recall-color",
        default=DEFAULT_COLORS["recall"],
        help="Line color for recall.",
    )
    parser.add_argument(
        "--throughput-color",
        default=DEFAULT_COLORS["throughput"],
        help="Line color for throughput.",
    )
    parser.add_argument(
        "--grid-color",
        default=DEFAULT_COLORS["grid"],
        help="Grid color.",
    )
    parser.add_argument(
        "--separator-color",
        default=DEFAULT_COLORS["separator"],
        help="Vertical separator color between ef blocks.",
    )
    parser.add_argument(
        "--upper-bound-color",
        "--target-color",
        dest="upper_bound_color",
        default=DEFAULT_COLORS["target"],
        help="Reference-line color for the per-panel ANSMET recall upper bounds.",
    )
    parser.add_argument(
        "--show-upper-bound-line",
        "--show-target-line",
        dest="show_upper_bound_line",
        action="store_true",
        help="Enable the per-panel ANSMET recall upper-bound reference lines.",
    )
    parser.add_argument(
        "--hide-upper-bound-line",
        "--hide-target-line",
        dest="show_upper_bound_line",
        action="store_false",
        help="Disable the per-panel ANSMET recall upper-bound reference lines.",
    )
    parser.set_defaults(show_upper_bound_line=False)
    return parser.parse_args()


def resolve_output_prefix(save_prefix: str) -> Path:
    prefix_path = Path(save_prefix)
    if prefix_path.is_absolute():
        return prefix_path
    return SCRIPT_DIR / prefix_path


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def set_times_new_roman(ttf_path: str = "") -> None:
    if ttf_path and Path(ttf_path).is_file():
        try:
            matplotlib.font_manager.fontManager.addfont(ttf_path)
        except Exception:
            pass

    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for family in [
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]:
        if family in available:
            plt.rcParams["font.family"] = family
            return
    plt.rcParams["font.family"] = "serif"


def set_global_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": BODY_FONT_PT,
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "axes.labelsize": AXES_LABEL_PT,
            "axes.titlesize": AXES_LABEL_PT,
            "xtick.labelsize": TICK_FONT_PT,
            "ytick.labelsize": TICK_FONT_PT,
            "legend.fontsize": LEGEND_FONT_PT,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(source_tsv: Path) -> list[dict[str, float]]:
    if not source_tsv.is_file():
        raise FileNotFoundError(f"Missing source TSV: {source_tsv}")

    rows: list[dict[str, float]] = []
    with source_tsv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            if row.get("final_status") != "PASS":
                continue
            rows.append(
                {
                    "ef_search": int(row["ef_search"]),
                    "queue_size": int(row["queue_size"]),
                    "recall": float(row["recall"]),
                    "s_mem_cycle": float(row["s_mem_cycle"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No PASS rows found in {source_tsv}")
    return rows


def extract_scalar_from_stats(stats_path: Path, key: str) -> float:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing ANSMET stats file: {stats_path}")

    match = re.search(
        rf"^\s*{re.escape(key)}:\s*({FLOAT_PATTERN})\s*$",
        stats_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not find `{key}` in {stats_path}")
    return float(match.group(1))


def load_ansmet_upper_bounds(stats_dir: Path) -> dict[int, float]:
    bounds: dict[int, float] = {}
    for ef_search in EF_ORDER:
        stats_path = stats_dir / f"t2i1m_normalized_k10_ef{ef_search}_stats.yml"
        bounds[ef_search] = extract_scalar_from_stats(stats_path, "s_recall_rate")
    return bounds


def block_start(panel_index: int) -> int:
    return panel_index * (LBQ_SPAN + BLOCK_GAP)


def x_position(panel_index: int, queue_size: int) -> float:
    return float(block_start(panel_index) + (queue_size - LBQ_MIN))


def build_plot_rows(rows: list[dict[str, float]]) -> tuple[dict[int, list[dict[str, float]]], float]:
    global_max_cycle = max(row["s_mem_cycle"] for row in rows)
    grouped: dict[int, list[dict[str, float]]] = {ef: [] for ef in EF_ORDER}

    for row in rows:
        ef_search = int(row["ef_search"])
        if ef_search not in grouped:
            continue
        panel_index = EF_ORDER.index(ef_search)
        entry = dict(row)
        entry["norm_throughput"] = global_max_cycle / row["s_mem_cycle"]
        entry["plot_x"] = x_position(panel_index, int(row["queue_size"]))
        grouped[ef_search].append(entry)

    expected = list(range(LBQ_MIN, LBQ_MAX + 1))
    for ef_search in EF_ORDER:
        grouped[ef_search].sort(key=lambda item: item["queue_size"])
        queues = [int(item["queue_size"]) for item in grouped[ef_search]]
        if queues != expected:
            raise RuntimeError(
                f"Unexpected LBQ sweep for ef_search={ef_search}: expected {expected[0]}..{expected[-1]}, got {queues[:5]} ... {queues[-5:]}"
            )

    return grouped, global_max_cycle


def export_data_tsv(
    path: Path,
    grouped: dict[int, list[dict[str, float]]],
    global_max_cycle: float,
    ansmet_upper_bounds: dict[int, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(
            [
                "ef_search",
                "panel_index",
                "queue_size",
                "plot_x",
                "recall",
                "s_mem_cycle",
                "norm_throughput",
                "throughput_unit",
                "global_max_s_mem_cycle",
                "ansmet_upper_bound_recall",
            ]
        )
        for panel_index, ef_search in enumerate(EF_ORDER):
            for row in grouped[ef_search]:
                writer.writerow(
                    [
                        ef_search,
                        panel_index,
                        int(row["queue_size"]),
                        f"{row['plot_x']:.1f}",
                        f"{row['recall']:.6f}",
                        f"{row['s_mem_cycle']:.0f}",
                        f"{row['norm_throughput']:.6f}",
                        "1/max_s_mem_cycle",
                        f"{global_max_cycle:.0f}",
                        f"{ansmet_upper_bounds[ef_search]:.6f}",
                    ]
                )


def compute_axis_limits(
    grouped: dict[int, list[dict[str, float]]],
    ansmet_upper_bounds: dict[int, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_recalls = [row["recall"] for rows in grouped.values() for row in rows]
    all_thrputs = [row["norm_throughput"] for rows in grouped.values() for row in rows]
    all_recalls.extend(ansmet_upper_bounds.values())

    recall_lower = max(0.0, np.floor((min(all_recalls) - 0.005) * 20.0) / 20.0)
    recall_upper = min(1.0, np.ceil((max(all_recalls) + 0.005) * 100.0) / 100.0)
    thr_lower = max(0.0, float(np.floor(min(all_thrputs))))
    thr_upper = float(np.ceil(max(all_thrputs)))
    return (recall_lower, recall_upper), (thr_lower, thr_upper)


def recall_ticks(recall_lower: float, recall_upper: float) -> np.ndarray:
    start = np.ceil(recall_lower / 0.1) * 0.1
    end = np.floor(recall_upper / 0.1) * 0.1
    return np.arange(start, end + 1e-9, 0.1)


def throughput_ticks(thr_lower: float, thr_upper: float) -> np.ndarray:
    start = np.ceil(thr_lower)
    end = np.floor(thr_upper)
    return np.arange(start, end + 1e-9, 1.0)


def save_figure(fig: plt.Figure, save_prefix: Path) -> None:
    pdf_path = Path(f"{save_prefix}.pdf")
    png_path = Path(f"{save_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def build_xticks() -> tuple[list[float], list[str]]:
    ticks: list[float] = []
    labels: list[str] = []
    for panel_index, _ in enumerate(EF_ORDER):
        for value in BLOCK_TICK_VALUES:
            ticks.append(x_position(panel_index, value))
            labels.append(str(value))
    return ticks, labels


def separator_positions() -> list[float]:
    positions = []
    for panel_index in range(len(EF_ORDER) - 1):
        left_end = x_position(panel_index, LBQ_MAX)
        right_start = x_position(panel_index + 1, LBQ_MIN)
        positions.append((left_end + right_start) / 2.0)
    return positions


def block_centers() -> list[float]:
    centers = []
    for panel_index in range(len(EF_ORDER)):
        centers.append((x_position(panel_index, LBQ_MIN) + x_position(panel_index, LBQ_MAX)) / 2.0)
    return centers


def plot_figure(
    args: argparse.Namespace,
    grouped: dict[int, list[dict[str, float]]],
    ansmet_upper_bounds: dict[int, float],
) -> None:
    set_times_new_roman(USER_TIMES_TTF)
    set_global_rc()

    save_prefix = resolve_output_prefix(args.save_prefix)
    (recall_lower, recall_upper), (thr_lower, thr_upper) = compute_axis_limits(
        grouped, ansmet_upper_bounds
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(ACM_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=300,
    )
    ax_r = ax.twinx()

    if args.show_upper_bound_line:
        for panel_index, ef_search in enumerate(EF_ORDER):
            ax.hlines(
                ansmet_upper_bounds[ef_search],
                x_position(panel_index, LBQ_MIN),
                x_position(panel_index, LBQ_MAX),
                color=args.upper_bound_color,
                linewidth=0.95,
                linestyle=(0, (3.0, 1.8)),
                zorder=2,
            )

    legend_handles = None
    for panel_index, ef_search in enumerate(EF_ORDER):
        rows = [grouped[ef_search][idx] for idx in MARK_EVERY]
        x = np.array([row["plot_x"] for row in rows], dtype=float)
        recall = np.array([row["recall"] for row in rows], dtype=float)
        throughput = np.array([row["norm_throughput"] for row in rows], dtype=float)

        recall_line, = ax.plot(
            x,
            recall,
            color=args.recall_color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.35,
            drawstyle="default",
            solid_joinstyle="miter",
            solid_capstyle="butt",
            label="Recall",
            zorder=4,
        )
        throughput_line, = ax_r.plot(
            x,
            throughput,
            color=args.throughput_color,
            linestyle="-",
            marker="s",
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.2,
            drawstyle="default",
            solid_joinstyle="miter",
            solid_capstyle="butt",
            label="Throughput",
            zorder=4,
        )
        if legend_handles is None:
            legend_handles = [recall_line, throughput_line]

    for xpos in separator_positions():
        ax.plot(
            [xpos, xpos],
            [-0.18, 1.0],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            color=args.separator_color,
            linewidth=1.05,
            linestyle="-",
            alpha=0.98,
            zorder=3,
        )

    total_end = x_position(len(EF_ORDER) - 1, LBQ_MAX)
    ax.set_xlim(-2.0, total_end + 2.0)
    ax.set_ylim(recall_lower, recall_upper)
    ax_r.set_ylim(thr_lower, thr_upper)

    xticks, xticklabels = build_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("LBQueue Size", labelpad=0.9)
    ax.set_ylabel("Recall", labelpad=1.0)
    ax_r.set_ylabel("Throughput", labelpad=3.2)

    ax.set_yticks(recall_ticks(recall_lower, recall_upper))
    ax_r.set_yticks(throughput_ticks(thr_lower, thr_upper))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_r.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_r.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax.grid(True, which="major", axis="y", linewidth=0.42, alpha=0.34, color=args.grid_color, zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", which="major", length=2.5, width=0.7, pad=0.8)
    ax.tick_params(axis="y", which="major", length=2.5, width=0.7, pad=1.0)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.6)
    ax_r.tick_params(axis="y", which="major", length=2.5, width=0.7, pad=1.0)
    ax_r.tick_params(axis="y", which="minor", length=1.8, width=0.6)

    for center, ef_search in zip(block_centers(), EF_ORDER):
        ax.text(
            center-20,
            0.94,
            f"ef={ef_search}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=AXES_LABEL_PT,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax_r.spines["top"].set_visible(False)

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.7,
        handletextpad=0.45,
        borderaxespad=0.1,
    )

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.24, top=0.80)
    save_figure(fig, save_prefix)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ansmet_upper_bounds = load_ansmet_upper_bounds(args.ansmet_stats_dir)
    grouped, global_max_cycle = build_plot_rows(load_rows(args.source_tsv))
    export_data_tsv(resolve_output_path(args.data_out), grouped, global_max_cycle, ansmet_upper_bounds)
    plot_figure(args, grouped, ansmet_upper_bounds)

    print(f"Global max s_mem_cycle: {global_max_cycle:.0f}")
    for ef_search in EF_ORDER:
        rows = grouped[ef_search]
        best_recall = max(rows, key=lambda row: row["recall"])
        best_thr = max(rows, key=lambda row: row["norm_throughput"])
        print(
            f"ef_search={ef_search}: max_recall=(q={int(best_recall['queue_size'])}, recall={best_recall['recall']:.3f}), "
            f"max_norm_thr=(q={int(best_thr['queue_size'])}, thr={best_thr['norm_throughput']:.3f}), "
            f"ansmet_upper_bound={ansmet_upper_bounds[ef_search]:.3f}"
        )


if __name__ == "__main__":
    main()
