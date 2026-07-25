#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "figure20_area_power_breakdown.tsv"
DEFAULT_SAVE_PREFIX = SCRIPT_DIR / "output" / "figure20"
DEFAULT_DATA_OUT = SCRIPT_DIR / "output" / "figure20_normalized.csv"
DEFAULT_SUMMARY_OUT = SCRIPT_DIR / "output" / "figure20_summary.tsv"

BASELINE_DESIGN = "ANSMET"
METRIC_ORDER = ["Area", "Power"]
DESIGN_ORDER = ["ANSMET", "NMP-FPMA", "MFNNS"]
STACK_COMPONENT_ORDER = ["MACs", "Adder Tree", "Scale Unit", "SRAM"]

ACM_COLUMN_WIDTH_IN = 3.33
FIGURE_HEIGHT_IN = 1.18
BODY_FONT_PT = 8.0
AXES_LABEL_PT = 6.6
TICK_FONT_PT = 5.8
YTICK_FONT_PT = 5.3
LEGEND_FONT_PT = 5.2
USER_TIMES_TTF = ""
BAR_EDGE_COLOR = "#000000"
BAR_LINEWIDTH = 0.42
BAR_ALPHA = 0.98

COMPONENT_COLORS = {
    "SRAM": "#F58518",
    "MACs": "#4C78A8",
    "Adder Tree": "#72B7B2",
    "Scale Unit": "#B279A2",
}

COMPONENT_DISPLAY_LABELS = {
    "SRAM": "Others",
}

# modify: tune the horizontal `ANSMET -> MFNNS` main arrow and ratio text here.
MAIN_RATIO_STYLE = {
    "start_design": "ANSMET",
    "end_design": "MFNNS",
    "color": "#222222",
    "linewidth": 0.8,
    "linestyle": "--",
    "arrowstyle": "->",
    "arrow_mutation_scale": 10.0,
    "text_fmt": "{ratio:.1f}x",
    "line_x_offset": 0.02,
    "line_start_y_offset": 0.0,
    "line_end_y_offset": 0.0,
    "arrow_y_offset": 0.0,
    "arrow_end_x_offset": 0.0,
    "text_x_offset": -0.03,
    "text_y_fraction": 0.18,
    "text_y_offset": 0.0,
    "text_ha": "right",
    "text_va": "center",
}

# modify: use per-metric overrides when the two panels need different placement.
MAIN_RATIO_OVERRIDES = {
    "Area": {
        "line_x_offset": 0.0,
        "line_start_y_offset": 0.0,
        "line_end_y_offset": 0.0,
        "arrow_y_offset": 0.0,
        "arrow_end_x_offset": 0.0,
        "text_x_offset": -0.03,
        "text_y_fraction": 0.18,
        "text_y_offset": 1.35,
    },
    "Power": {
        "line_x_offset": 0.0,
        "line_start_y_offset": 0.0,
        "line_end_y_offset": 0.0,
        "arrow_y_offset": 0.0,
        "arrow_end_x_offset": 0.0,
        "text_x_offset": -0.03,
        "text_y_fraction": 0.18,
        "text_y_offset": 1.35,
    },
}

# modify: tune the small horizontal `NMP-FPMA -> MFNNS` curve and the ratio text here.
CURVE_RATIO_STYLE = {
    "start_design": "NMP-FPMA",
    "end_design": "MFNNS",
    "color": "#222222",
    "linewidth": 0.8,
    "linestyle": "-",
    "arrowstyle": "->",
    "arrow_mutation_scale": 12.0,
    "text_fmt": "{ratio:.1f}x",
    "start_x_offset": 0.0,
    "end_x_offset": 0.0,
    "start_y_offset": 0.0,
    "end_y_offset": 0.0,
    "curve_x_offset": 0.0,
    "curve_rad": -0.35,
    "text_x_offset": 0.0,
    "text_y_offset": -0.03,
}

# modify: use per-metric overrides when the two panels need different curve geometry.
CURVE_RATIO_OVERRIDES = {
    "Area": {
        "start_x_offset": 0.0,
        "end_x_offset": 0.0,
        "start_y_offset": 0.0,
        "end_y_offset": 0.0,
        "curve_x_offset": 0.0,
        "curve_rad": -0.35,
        "text_x_offset": 0.1,
        "text_y_offset": -0.03,
    },
    "Power": {
        "start_x_offset": 0.0,
        "end_x_offset": 0.0,
        "start_y_offset": 0.0,
        "end_y_offset": 0.0,
        "curve_x_offset": 0.0,
        "curve_rad": -0.35,
        "text_x_offset": 0.1,
        "text_y_offset": -0.03,
    },
}

DESIGN_TICK_LABELS = {
    "ANSMET": "ANSMET",
    "NMP-FPMA": "NMP-FPMA",
    "MFNNS": "MFNNS",
}

LAYOUT_WINDOW = {
    "legend_anchor_x": 0.52,
    "legend_anchor_y": 0.985,
    "subplots_left": 0.18,
    "subplots_right": 0.995,
    "subplots_bottom": 0.18,
    "subplots_top": 0.73,
    "subplots_wspace": 0.08,
    "bar_height": 0.46,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a horizontal-bar version of the side-by-side area/power "
            "breakdown plots normalized to ANSMET = 1."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input TSV-style text file containing the area/power breakdown tables.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=DEFAULT_SAVE_PREFIX,
        help="Output prefix for the generated PDF and PNG.",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=DEFAULT_DATA_OUT,
        help="CSV export with raw and normalized component values.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="TSV summary of totals and the two ratios annotated in each panel.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the frozen data and derived metrics without writing outputs.",
    )
    return parser.parse_args()


def resolve_output_path(path_like: Path) -> Path:
    if path_like.is_absolute():
        return path_like
    return SCRIPT_DIR / path_like


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
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_sections(input_path: Path) -> dict[str, dict[str, object]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    sections: dict[str, dict[str, object]] = {}
    current_metric: str | None = None
    current_designs: list[str] = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]
            if not any(row):
                current_metric = None
                current_designs = []
                continue

            if current_metric is None:
                current_metric = row[0]
                current_designs = [cell for cell in row[1:] if cell]
                if not current_designs:
                    raise ValueError(f"Malformed header row: {raw_row}")
                sections[current_metric] = {
                    "designs": current_designs,
                    "components": [],
                    "values": {},
                }
                continue

            component = row[0]
            value_cells = [cell for cell in row[1:] if cell]
            if len(value_cells) != len(current_designs):
                raise ValueError(
                    f"Expected {len(current_designs)} values for {current_metric}/{component}, "
                    f"got {len(value_cells)}"
                )

            sections[current_metric]["components"].append(component)
            for design, value_text in zip(current_designs, value_cells):
                sections[current_metric]["values"][(component, design)] = float(value_text)

    if set(sections) != set(METRIC_ORDER):
        raise ValueError(
            f"Unexpected metric sections: expected={METRIC_ORDER}, got={list(sections)}"
        )

    for metric in METRIC_ORDER:
        if metric not in sections:
            raise KeyError(f"Missing section: {metric}")
        designs = list(sections[metric]["designs"])
        components = list(sections[metric]["components"])
        if designs != DESIGN_ORDER:
            raise ValueError(
                f"Unexpected design order for {metric}: "
                f"expected={DESIGN_ORDER}, got={designs}"
            )
        if components != STACK_COMPONENT_ORDER:
            raise ValueError(
                f"Unexpected component order for {metric}: "
                f"expected={STACK_COMPONENT_ORDER}, got={components}"
            )
        values = dict(sections[metric]["values"])
        expected_keys = {
            (component, design)
            for component in STACK_COMPONENT_ORDER
            for design in DESIGN_ORDER
        }
        if set(values) != expected_keys:
            raise ValueError(f"Incomplete Figure 20 matrix for {metric}")
        if any(float(value) < 0.0 for value in values.values()):
            raise ValueError(f"Negative component value in {metric}")

    return sections


def ordered_components(components: list[str]) -> list[str]:
    known = [component for component in STACK_COMPONENT_ORDER if component in components]
    extras = [component for component in components if component not in STACK_COMPONENT_ORDER]
    return known + extras


def display_component_label(component: str) -> str:
    return COMPONENT_DISPLAY_LABELS.get(component, component)


def build_export_rows(sections: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRIC_ORDER:
        section = sections[metric]
        designs = list(section["designs"])
        components = ordered_components(list(section["components"]))
        values = dict(section["values"])

        baseline_total = sum(float(values[(component, BASELINE_DESIGN)]) for component in components)
        if baseline_total <= 0.0:
            raise ValueError(f"Baseline total must be positive for section {metric}")

        design_totals = {
            design: sum(float(values[(component, design)]) for component in components)
            for design in designs
        }

        for design in designs:
            for component in components:
                raw_value = float(values[(component, design)])
                rows.append(
                    {
                        "metric": metric,
                        "design": design,
                        "component": component,
                        "raw_value": raw_value,
                        "normalized_value": raw_value / baseline_total,
                        "design_total": design_totals[design],
                        "design_total_normalized": design_totals[design] / baseline_total,
                        "baseline_total": baseline_total,
                    }
                )
    return rows


def export_data_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "design",
        "component",
        "raw_value",
        "normalized_value",
        "design_total",
        "design_total_normalized",
        "baseline_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], float] = {}
    for row in rows:
        totals[(str(row["metric"]), str(row["design"]))] = float(
            row["design_total"]
        )

    summary: list[dict[str, object]] = []
    expected_labels = {
        "Area": ("2.2x", "1.2x"),
        "Power": ("2.7x", "1.2x"),
    }
    for metric in METRIC_ORDER:
        baseline_total = totals[(metric, "ANSMET")]
        nmp_total = totals[(metric, "NMP-FPMA")]
        mfnns_total = totals[(metric, "MFNNS")]
        if min(baseline_total, nmp_total, mfnns_total) <= 0.0:
            raise ValueError(f"Non-positive total in {metric}")
        baseline_ratio = baseline_total / mfnns_total
        nmp_ratio = nmp_total / mfnns_total
        labels = (f"{baseline_ratio:.1f}x", f"{nmp_ratio:.1f}x")
        if labels != expected_labels[metric]:
            raise ValueError(
                f"Unexpected Figure 20 ratio labels for {metric}: "
                f"expected={expected_labels[metric]}, got={labels}"
            )
        summary.append(
            {
                "metric": metric,
                "ansmet_total": baseline_total,
                "nmp_fpma_total": nmp_total,
                "mfnns_total": mfnns_total,
                "mfnns_normalized_to_ansmet": mfnns_total / baseline_total,
                "ansmet_over_mfnns": baseline_ratio,
                "nmp_fpma_over_mfnns": nmp_ratio,
                "paper_labels": f"{labels[1]};{labels[0]}",
            }
        )
    return summary


def export_summary_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "ansmet_total",
        "nmp_fpma_total",
        "mfnns_total",
        "mfnns_normalized_to_ansmet",
        "ansmet_over_mfnns",
        "nmp_fpma_over_mfnns",
        "paper_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        (str(row["metric"]), str(row["design"]), str(row["component"])): row
        for row in rows
    }


def merged_style(
    base_style: dict[str, object],
    metric_overrides: dict[str, dict[str, object]],
    metric: str,
) -> dict[str, object]:
    style = dict(base_style)
    style.update(metric_overrides.get(metric, {}))
    return style


def style_axis(ax: plt.Axes, x_right: float) -> None:
    ax.set_xlim(0.0, x_right)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="major", length=2.5, width=0.7, pad=1.0)
    ax.tick_params(axis="x", which="minor", length=1.7, width=0.55)
    ax.tick_params(axis="y", which="major", length=2.4, width=0.7, pad=1.0)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.spines["left"].set_zorder(8)
    ax.spines["bottom"].set_zorder(8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0.0, color="black", linewidth=0.95, zorder=8, clip_on=False)


def add_main_ratio_annotation(
    ax: plt.Axes,
    metric: str,
    y_positions: dict[str, float],
    totals: dict[str, float],
) -> None:
    style = merged_style(MAIN_RATIO_STYLE, MAIN_RATIO_OVERRIDES, metric)
    start_design = str(style["start_design"])
    end_design = str(style["end_design"])
    start_total = float(totals[start_design])
    end_total = float(totals[end_design])
    if end_total <= 0.0:
        return

    x_line = start_total + float(style["line_x_offset"])
    y_line_start = float(y_positions[start_design]) + float(style["line_start_y_offset"])
    y_line_end = float(y_positions[end_design]) + float(style["line_end_y_offset"])
    y_arrow = float(y_positions[end_design]) + float(style["arrow_y_offset"])
    x_arrow_end = end_total + float(style["arrow_end_x_offset"])

    ax.plot(
        [x_line, x_line],
        [y_line_start, y_line_end],
        color=str(style["color"]),
        linestyle=str(style["linestyle"]),
        linewidth=float(style["linewidth"]),
        zorder=5,
    )

    ax.annotate(
        "",
        xy=(x_arrow_end, y_arrow),
        xytext=(x_line, y_arrow),
        arrowprops=dict(
            arrowstyle=str(style["arrowstyle"]),
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            mutation_scale=float(style["arrow_mutation_scale"]),
        ),
        zorder=5,
    )

    ratio = start_total / end_total
    x_text = x_line + float(style["text_x_offset"])
    y_text = (
        y_line_start
        + float(style["text_y_fraction"]) * (y_line_end - y_line_start)
        + float(style["text_y_offset"])
    )
    ax.text(
        x_text,
        y_text,
        str(style["text_fmt"]).format(ratio=ratio),
        ha=str(style["text_ha"]),
        va=str(style["text_va"]),
        fontsize=TICK_FONT_PT,
        fontweight="bold",
        color=str(style["color"]),
        zorder=6,
    )


def add_curve_ratio_annotation(
    ax: plt.Axes,
    metric: str,
    y_positions: dict[str, float],
    totals: dict[str, float],
) -> None:
    style = merged_style(CURVE_RATIO_STYLE, CURVE_RATIO_OVERRIDES, metric)
    start_design = str(style["start_design"])
    end_design = str(style["end_design"])
    start_total = float(totals[start_design])
    end_total = float(totals[end_design])
    if end_total <= 0.0:
        return

    x_start = start_total + float(style["start_x_offset"])
    x_end = end_total + float(style["end_x_offset"])
    x_curve = max(start_total, end_total) + float(style["curve_x_offset"])
    y_start = float(y_positions[start_design]) + float(style["start_y_offset"])
    y_end = float(y_positions[end_design]) + float(style["end_y_offset"])
    curve_rad = float(style["curve_rad"])

    curve = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle=str(style["arrowstyle"]),
        connectionstyle=f"arc3,rad={curve_rad}",
        linewidth=float(style["linewidth"]),
        linestyle=str(style["linestyle"]),
        mutation_scale=float(style["arrow_mutation_scale"]),
        color=str(style["color"]),
        zorder=5,
    )
    ax.add_patch(curve)

    ratio = start_total / end_total
    x_text = 0.5 * (x_start + x_end) + 0.5 * (x_curve - max(x_start, x_end)) + float(style["text_x_offset"])
    y_text = 0.5 * (y_start + y_end) + float(style["text_y_offset"])
    ax.text(
        x_text,
        y_text,
        str(style["text_fmt"]).format(ratio=ratio),
        ha="center",
        va="top",
        fontsize=TICK_FONT_PT,
        fontweight="bold",
        color=str(style["color"]),
        zorder=6,
    )


def save_figure(fig: plt.Figure, save_prefix: Path) -> None:
    pdf_path = Path(f"{save_prefix}.pdf")
    png_path = Path(f"{save_prefix}.png")
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_figure(save_prefix: Path, sections: dict[str, dict[str, object]], rows: list[dict[str, object]]) -> None:
    set_times_new_roman(USER_TIMES_TTF)
    set_global_rc()

    lookup = build_lookup(rows)
    x_right = max(float(row["design_total_normalized"]) for row in rows) * 1.03

    fig, axes = plt.subplots(
        1,
        len(METRIC_ORDER),
        figsize=(ACM_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=300,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    for axis_index, metric in enumerate(METRIC_ORDER):
        ax = axes[axis_index]
        section = sections[metric]
        designs = list(section["designs"])
        components = ordered_components(list(section["components"]))
        y = np.arange(len(designs), dtype=float)
        y_positions = {design: float(y_value) for design, y_value in zip(designs, y)}
        lefts = np.zeros(len(designs), dtype=float)

        for component in components:
            values = np.asarray(
                [
                    float(lookup[(metric, design, component)]["normalized_value"])
                    for design in designs
                ],
                dtype=float,
            )
            ax.barh(
                y,
                values,
                height=LAYOUT_WINDOW["bar_height"],
                left=lefts,
                color=COMPONENT_COLORS.get(component, "#bbbbbb"),
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_LINEWIDTH,
                alpha=BAR_ALPHA,
                zorder=3,
                label=display_component_label(component) if axis_index == 0 else None,
            )
            lefts += values

        design_totals = {
            design: float(total_value)
            for design, total_value in zip(designs, lefts)
        }

        add_main_ratio_annotation(ax, metric, y_positions, design_totals)
        add_curve_ratio_annotation(ax, metric, y_positions, design_totals)

        style_axis(ax, x_right)
        ax.set_yticks(y)
        ax.set_yticklabels([DESIGN_TICK_LABELS.get(design, design) for design in designs])
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontsize(YTICK_FONT_PT)
        if axis_index > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

    axes[0].invert_yaxis()

    legend_handles = [
        Patch(
            facecolor=COMPONENT_COLORS.get(component, "#bbbbbb"),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label=display_component_label(component),
        )
        for component in ordered_components(
            list(sections[METRIC_ORDER[0]]["components"])
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            LAYOUT_WINDOW["legend_anchor_x"],
            LAYOUT_WINDOW["legend_anchor_y"],
        ),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=0.75,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.1,
    )

    fig.subplots_adjust(
        left=LAYOUT_WINDOW["subplots_left"],
        right=LAYOUT_WINDOW["subplots_right"],
        bottom=LAYOUT_WINDOW["subplots_bottom"],
        top=LAYOUT_WINDOW["subplots_top"],
        wspace=LAYOUT_WINDOW["subplots_wspace"],
    )
    save_figure(fig, save_prefix)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sections = load_sections(args.input)
    export_rows = build_export_rows(sections)
    summary_rows = build_summary_rows(export_rows)
    if args.check_only:
        print(
            "Validated Figure 20: "
            "2 metrics x 3 designs x 4 components; "
            "paper labels Area=1.2x/2.2x, Power=1.2x/2.7x."
        )
        return
    export_data_csv(resolve_output_path(args.data_out), export_rows)
    export_summary_tsv(resolve_output_path(args.summary_out), summary_rows)
    plot_figure(resolve_output_path(args.save_prefix), sections, export_rows)


if __name__ == "__main__":
    main()
