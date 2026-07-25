#!/usr/bin/env python3
"""Audit and export Figure 19 MFNNS-to-memory/YAML provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR.parents[2] / "MFANNS"
DEFAULT_JUNO_OUTPUT = SCRIPT_DIR / "data/juno_fig8_designs.tsv"
DEFAULT_PLOT_OUTPUT = SCRIPT_DIR / "data/figure19_plot_data.tsv"
DEFAULT_PROVENANCE_OUTPUT = SCRIPT_DIR / "data/figure19_mfnns_provenance.csv"
DEFAULT_EXPERIMENT_OUTPUT = SCRIPT_DIR / "data/figure19_source_experiments.csv"
DEFAULT_RANGE_OUTPUT = SCRIPT_DIR / "data/figure19_frontier_better_ranges.tsv"
DEFAULT_SPEEDUP_OUTPUT = SCRIPT_DIR / "data/figure19_frontier_speedup_samples.tsv"
DEFAULT_SUMMARY_OUTPUT = SCRIPT_DIR / "output/figure19_provenance_summary.tsv"
DEFAULT_SHA_OUTPUT = SCRIPT_DIR / "data/SHA256SUMS"

PANEL_ORDER = [
    ("sift1m", "r1"),
    ("t2i1m", "r1"),
    ("sift1m", "r100"),
    ("t2i1m", "r100"),
]
PANEL_LABELS = {
    ("sift1m", "r1"): "SIFT1M Recall@1",
    ("t2i1m", "r1"): "T2I1M Recall@1",
    ("sift1m", "r100"): "SIFT1M Recall@100",
    ("t2i1m", "r100"): "T2I1M Recall@100",
}
XLIM_LOW = 0.4
XLIM_HIGH = 1.01
QPS_FREQUENCY_HZ = 2_400_000_000.0

FINAL_SELECTION_SCRIPT = (
    "simulator/memory/20260613/006_mfnns_ansmet_lbq_juno_focused/"
    "scripts/summarize_lbq_tune.py"
)
RTC_EXTRACTION_SCRIPT = "figure/evaluation/RTC/extract_rtc_data.py"
SOURCE_OUTER_FRONTIER = (
    "simulator/memory/20260613/006_mfnns_ansmet_lbq_juno_focused/"
    "tables/mfnns_outer_frontier.tsv"
)

EXPERIMENTS = {
    "20260612/002_sift_t2i_ansmet_mfnns_r1_r100_frontier": {
        "test_record": "TEST_RECORD.md",
        "yaml_generator": "scripts/generate_submit_frontier.py",
        "result_summarizer": "scripts/summarize_frontier.py",
        "submission_helper": "",
        "recorded_total_cases": 128,
        "recorded_completed_cases": 128,
        "recorded_failed_cases": 0,
        "completion_markers": [
            "Final completion: 128 / 128 cases completed, 0 failed."
        ],
        "role": "broad SIFT1M/T2I1M r1/r100 ANSMET+MFNNS grid",
    },
    "20260612/003_sift_mfnns_lowef_subef_pareto": {
        "test_record": "TEST_RECORD.md",
        "yaml_generator": "scripts/generate_submit_sift_mfnns_refine.py",
        "result_summarizer": "scripts/summarize_sift_mfnns_refine.py",
        "submission_helper": "",
        "recorded_total_cases": 124,
        "recorded_completed_cases": 124,
        "recorded_failed_cases": 0,
        "completion_markers": [
            "Final completion: 124 / 124 new cases completed, 0 failed."
        ],
        "role": "SIFT1M low-ef/sub-ef/high-ef refinement",
    },
    "20260613/004_sift_t2i_mfnns_lowef_subef_frontier": {
        "test_record": "TEST_RECORD.md",
        "yaml_generator": "scripts/generate_submit_t2i_mfnns_refine.py",
        "result_summarizer": "scripts/summarize_mfnns_frontier.py",
        "submission_helper": "scripts/submit_accelerate_remaining.py",
        "recorded_total_cases": 124,
        "recorded_completed_cases": 124,
        "recorded_failed_cases": 0,
        "completion_markers": [
            "New T2I completed cases: 124 / 124.",
            "Failed status entries: 0.",
        ],
        "role": "T2I1M low-ef/sub-ef/high-ef refinement",
    },
    "20260613/006_mfnns_ansmet_lbq_juno_focused": {
        "test_record": "TEST_RECORD.md",
        "yaml_generator": "scripts/generate_submit_lbq_tune.py",
        "result_summarizer": "scripts/summarize_lbq_tune.py",
        "submission_helper": "",
        "recorded_total_cases": 228,
        "recorded_completed_cases": 228,
        "recorded_failed_cases": 0,
        "completion_markers": [
            "Completed new cases: 228 / 228.",
            "Failed cases: 0.",
        ],
        "role": "JUNO-focused ANSMET anchors and MFNNS LBQ-fill refinement",
    },
}

PROVENANCE_FIELDS = [
    "panel",
    "dataset",
    "recall_tag",
    "paper_metric",
    "frontier_rank",
    "case_name",
    "group_tag",
    "recall",
    "qps_2p4ghz",
    "s_mem_cycle",
    "plot_input",
    "point_within_xlim",
    "data_status",
    "source_experiment",
    "test_record_ref",
    "yaml_generator_ref",
    "result_summarizer_ref",
    "final_selection_script_ref",
    "rtc_extraction_script_ref",
    "case_manifest_ref",
    "yaml_ref",
    "yaml_sha256",
    "stats_ref",
    "stats_sha256",
    "yaml_ef_search",
    "yaml_queue_size",
    "yaml_warmup_size",
    "yaml_gt_k",
    "yaml_k_neighbors",
    "yaml_n_query_limit",
    "yaml_n_parallel_query",
    "yaml_mfnns_enable",
    "yaml_early_exit_enable",
    "yaml_lb_et_enable",
    "yaml_row_policy",
    "stats_recall",
    "stats_mem_cycle",
    "stats_num_query",
]

EXPERIMENT_FIELDS = [
    "source_experiment",
    "contribution_points",
    "points_within_xlim",
    "sift1m_r1_points",
    "t2i1m_r1_points",
    "sift1m_r100_points",
    "t2i1m_r100_points",
    "recorded_total_cases",
    "recorded_completed_cases",
    "recorded_failed_cases",
    "role",
    "test_record_ref",
    "test_record_sha256",
    "yaml_generator_ref",
    "yaml_generator_sha256",
    "result_summarizer_ref",
    "result_summarizer_sha256",
    "submission_helper_ref",
    "submission_helper_sha256",
    "case_manifest_ref",
    "case_manifest_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Author MFANNS workspace containing figure/evaluation/RTC and simulator/memory.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_delimited(path: Path, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_delimited(
    path: Path, rows: list[dict[str, object]], fields: list[str], delimiter: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_ref(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source path is outside MFANNS root: {path}") from exc


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def require_float_close(
    actual: float,
    expected: float,
    label: str,
    *,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-9,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return data


def build_rows(
    source_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, str]],
    Path,
    Path,
    Path,
    Path,
]:
    source_root = source_root.resolve()
    memory_root = source_root / "simulator/memory"
    rtc_data = source_root / "figure/evaluation/RTC/data"
    juno_path = rtc_data / "juno_fig8_designs.tsv"
    plot_path = rtc_data / "rtc_recall_qps_plot_data.tsv"
    frontier_path = rtc_data / "mfnns_current_best_frontier.tsv"
    range_path = rtc_data / "frontier_better_ranges.tsv"
    speedup_path = rtc_data / "frontier_speedup_samples.tsv"
    outer_path = source_root / SOURCE_OUTER_FRONTIER

    juno_rows = read_delimited(juno_path, "\t")
    plot_rows = read_delimited(plot_path, "\t")
    frontier_rows = read_delimited(frontier_path, "\t")
    outer_rows = read_delimited(outer_path, "\t")
    require_equal(len(juno_rows), 277, "JUNO++ Fig. 8 raw row count")
    require_equal(
        sum(int(row["plot_default"]) for row in juno_rows),
        164,
        "JUNO++ selected plot-input row count",
    )
    require_equal(len(plot_rows), 353, "Figure 19 plot row count")
    require_equal(len(frontier_rows), 189, "MFNNS frontier row count")

    mfnns_plot_rows = [row for row in plot_rows if row["series"] == "MFNNS frontier"]
    require_equal(len(mfnns_plot_rows), 189, "MFNNS plot-input row count")
    plot_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in mfnns_plot_rows:
        key = (row["dataset"], row["recall_tag"], row["source_detail"])
        if key in plot_lookup:
            raise ValueError(f"Duplicate MFNNS plot row: {key}")
        plot_lookup[key] = row

    frontier_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in frontier_rows:
        key = (row["dataset"], row["recall_tag"], row["case_name"])
        if key in frontier_lookup:
            raise ValueError(f"Duplicate MFNNS frontier row: {key}")
        frontier_lookup[key] = row
    require_equal(set(plot_lookup), set(frontier_lookup), "plot/frontier case set")

    outer_lookup: dict[str, dict[str, str]] = {}
    for row in outer_rows:
        case = row["case_name"]
        if case in outer_lookup:
            raise ValueError(f"Duplicate outer-frontier case: {case}")
        outer_lookup[case] = row
    require_equal(
        set(outer_lookup), {row["case_name"] for row in frontier_rows}, "outer/frontier case set"
    )

    manifests: dict[str, dict[str, dict[str, str]]] = {}
    for experiment in EXPERIMENTS:
        manifest_path = memory_root / experiment / "case_manifest.tsv"
        manifest_rows = read_delimited(manifest_path, "\t")
        manifests[experiment] = {row["case_name"]: row for row in manifest_rows}
        if len(manifests[experiment]) != len(manifest_rows):
            raise ValueError(f"Duplicate case in {manifest_path}")

    provenance: list[dict[str, object]] = []
    for key in sorted(
        frontier_lookup,
        key=lambda item: (PANEL_ORDER.index((item[0], item[1])), int(frontier_lookup[item]["frontier_rank"])),
    ):
        frontier = frontier_lookup[key]
        plot = plot_lookup[key]
        case_name = frontier["case_name"]
        yaml_path = Path(frontier["yaml_path"]).resolve()
        stats_path = Path(frontier["stats_path"]).resolve()
        if not yaml_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(f"Missing YAML/stats for {case_name}")

        yaml_relative = yaml_path.relative_to(memory_root)
        experiment = "/".join(yaml_relative.parts[:2])
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unexpected source experiment for {case_name}: {experiment}")
        experiment_root = memory_root / experiment
        manifest = manifests[experiment].get(case_name)
        if manifest is None:
            raise ValueError(f"{case_name} is absent from {experiment}/case_manifest.tsv")

        config = load_yaml(yaml_path)
        stats = load_yaml(stats_path)
        frontend = config.get("Frontend")
        stats_frontend = stats.get("Frontend")
        if not isinstance(frontend, dict) or not isinstance(stats_frontend, dict):
            raise ValueError(f"Missing Frontend mapping for {case_name}")

        dataset = frontier["dataset"]
        recall_tag = frontier["recall_tag"]
        expected_k = 1 if recall_tag == "r1" else 100
        ef_search = int(frontier["ef_search"])
        queue_size = int(frontier["queue_size"])
        warmup_size = int(frontier["warmup_size"])
        recall = float(frontier["recall"])
        qps = float(frontier["qps_2p4ghz"])
        mem_cycle = int(frontier["s_mem_cycle"])

        require_equal(manifest["method"], "mfnns", f"{case_name} manifest method")
        require_equal(manifest["target_dataset"], dataset, f"{case_name} manifest dataset")
        require_equal(manifest["recall_tag"], recall_tag, f"{case_name} manifest recall tag")
        require_equal(int(manifest["ef_search"]), ef_search, f"{case_name} manifest ef")
        require_equal(int(manifest["queue_size"]), queue_size, f"{case_name} manifest queue")
        require_equal(int(manifest["warmup_size"]), warmup_size, f"{case_name} manifest warmup")
        require_equal(Path(manifest["yaml_path"]).resolve(), yaml_path, f"{case_name} manifest YAML")
        require_equal(Path(manifest["stats_path"]).resolve(), stats_path, f"{case_name} manifest stats")

        require_equal(int(frontend["ef_search"]), ef_search, f"{case_name} YAML ef")
        require_equal(
            int(frontend["dualQueueLowerBoundQueueSize"]),
            queue_size,
            f"{case_name} YAML queue",
        )
        require_equal(
            int(frontend["dualQueueLowerBoundWarmupSize"]),
            warmup_size,
            f"{case_name} YAML warmup",
        )
        require_equal(int(frontend["gt_k"]), 100, f"{case_name} YAML gt_k")
        require_equal(int(frontend["k_neighbors"]), expected_k, f"{case_name} YAML k")
        require_equal(int(frontend["nQueryLimit"]), 1000, f"{case_name} YAML query count")
        require_equal(
            int(frontend["nParallelQuery"]), 1000, f"{case_name} YAML parallel queries"
        )
        require_equal(
            type(frontend["mfnnsEnable"]),
            bool,
            f"{case_name} YAML MFNNS type",
        )
        require_equal(frontend["mfnnsEnable"], True, f"{case_name} YAML MFNNS")
        require_equal(
            type(frontend["earlyExitEnable"]),
            bool,
            f"{case_name} YAML early-exit type",
        )
        require_equal(frontend["earlyExitEnable"], False, f"{case_name} YAML early exit")
        require_equal(
            type(frontend["dualQueueLowerBoundETEnable"]),
            bool,
            f"{case_name} YAML LB-ET type",
        )
        require_equal(
            frontend["dualQueueLowerBoundETEnable"],
            True,
            f"{case_name} YAML LB-ET",
        )
        require_equal(
            Path(frontend["stat_path"]).resolve(), stats_path, f"{case_name} YAML stat_path"
        )

        stats_recall = float(stats_frontend["s_recall_rate"])
        stats_mem_cycle = int(stats_frontend["s_mem_cycle"])
        stats_num_query = int(stats_frontend["s_num_query"])
        require_float_close(stats_recall, recall, f"{case_name} stats recall")
        require_equal(stats_mem_cycle, mem_cycle, f"{case_name} stats memory cycles")
        require_equal(stats_num_query, 1000, f"{case_name} stats query count")
        require_float_close(
            stats_num_query * QPS_FREQUENCY_HZ / stats_mem_cycle,
            qps,
            f"{case_name} 2.4GHz QPS",
            rel_tol=1e-9,
            abs_tol=0.001,
        )
        require_float_close(float(plot["recall"]), recall, f"{case_name} plot recall")
        require_float_close(float(plot["qps"]), qps, f"{case_name} plot QPS", abs_tol=0.001)

        outer = outer_lookup[case_name]
        require_equal(outer["target_dataset"], dataset, f"{case_name} outer dataset")
        require_equal(outer["recall_tag"], recall_tag, f"{case_name} outer recall tag")
        require_equal(outer["method"], "mfnns", f"{case_name} outer method")
        require_equal(outer["group_tag"], frontier["group_tag"], f"{case_name} outer group")
        require_equal(int(outer["ef_search"]), ef_search, f"{case_name} outer ef")
        require_equal(int(outer["queue_size"]), queue_size, f"{case_name} outer queue")
        require_equal(int(outer["warmup_size"]), warmup_size, f"{case_name} outer warmup")
        require_equal(int(outer["gt_k"]), 100, f"{case_name} outer gt_k")
        require_equal(int(outer["k_neighbors"]), expected_k, f"{case_name} outer k")
        require_equal(int(outer["nQueryLimit"]), 1000, f"{case_name} outer query count")
        require_equal(
            int(outer["nParallelQuery"]),
            1000,
            f"{case_name} outer parallel queries",
        )
        require_equal(outer.get("final_status"), "completed", f"{case_name} final status")
        require_equal(int(outer["s_num_query"]), 1000, f"{case_name} outer stats queries")
        require_equal(int(outer["s_mem_cycle"]), mem_cycle, f"{case_name} outer cycles")
        require_equal(
            Path(outer["yaml_path"]).resolve(), yaml_path, f"{case_name} outer YAML"
        )
        require_equal(
            Path(outer["stats_path"]).resolve(), stats_path, f"{case_name} outer stats"
        )
        require_float_close(float(outer["recall"]), recall, f"{case_name} outer recall")
        require_float_close(
            float(outer["qps_2p4ghz"]), qps, f"{case_name} outer QPS", abs_tol=0.001
        )

        row_policy = config["MemorySystem"]["Controller"]["RowPolicy"]["impl"]
        require_equal(row_policy, "OpenRowPolicy", f"{case_name} YAML row policy")
        meta = EXPERIMENTS[experiment]
        provenance.append(
            {
                "panel": PANEL_LABELS[(dataset, recall_tag)],
                "dataset": dataset,
                "recall_tag": recall_tag,
                "paper_metric": frontier["paper_metric"],
                "frontier_rank": int(frontier["frontier_rank"]),
                "case_name": case_name,
                "group_tag": frontier["group_tag"],
                "recall": f"{recall:.12g}",
                "qps_2p4ghz": f"{qps:.12g}",
                "s_mem_cycle": mem_cycle,
                "plot_input": 1,
                "point_within_xlim": int(XLIM_LOW <= recall <= XLIM_HIGH),
                "data_status": "measured_completed_simulator",
                "source_experiment": experiment,
                "test_record_ref": f"simulator/memory/{experiment}/{meta['test_record']}",
                "yaml_generator_ref": (
                    f"simulator/memory/{experiment}/{meta['yaml_generator']}"
                ),
                "result_summarizer_ref": (
                    f"simulator/memory/{experiment}/{meta['result_summarizer']}"
                ),
                "final_selection_script_ref": FINAL_SELECTION_SCRIPT,
                "rtc_extraction_script_ref": RTC_EXTRACTION_SCRIPT,
                "case_manifest_ref": f"simulator/memory/{experiment}/case_manifest.tsv",
                "yaml_ref": source_ref(yaml_path, source_root),
                "yaml_sha256": sha256(yaml_path),
                "stats_ref": source_ref(stats_path, source_root),
                "stats_sha256": sha256(stats_path),
                "yaml_ef_search": int(frontend["ef_search"]),
                "yaml_queue_size": int(frontend["dualQueueLowerBoundQueueSize"]),
                "yaml_warmup_size": int(frontend["dualQueueLowerBoundWarmupSize"]),
                "yaml_gt_k": int(frontend["gt_k"]),
                "yaml_k_neighbors": int(frontend["k_neighbors"]),
                "yaml_n_query_limit": int(frontend["nQueryLimit"]),
                "yaml_n_parallel_query": int(frontend["nParallelQuery"]),
                "yaml_mfnns_enable": int(bool(frontend["mfnnsEnable"])),
                "yaml_early_exit_enable": int(bool(frontend["earlyExitEnable"])),
                "yaml_lb_et_enable": int(bool(frontend["dualQueueLowerBoundETEnable"])),
                "yaml_row_policy": row_policy,
                "stats_recall": f"{stats_recall:.12g}",
                "stats_mem_cycle": stats_mem_cycle,
                "stats_num_query": stats_num_query,
            }
        )

    require_equal(len(provenance), 189, "provenance row count")
    require_equal(
        sum(int(row["point_within_xlim"]) for row in provenance),
        160,
        "MFNNS points within xlim",
    )

    contributions = Counter(str(row["source_experiment"]) for row in provenance)
    experiment_rows: list[dict[str, object]] = []
    for experiment, meta in EXPERIMENTS.items():
        experiment_root = memory_root / experiment
        selected = [row for row in provenance if row["source_experiment"] == experiment]
        test_record = experiment_root / str(meta["test_record"])
        generator = experiment_root / str(meta["yaml_generator"])
        summarizer = experiment_root / str(meta["result_summarizer"])
        manifest = experiment_root / "case_manifest.tsv"
        helper_name = str(meta["submission_helper"])
        helper = experiment_root / helper_name if helper_name else None
        for required in (test_record, generator, summarizer, manifest):
            if not required.is_file():
                raise FileNotFoundError(required)
        if helper is not None and not helper.is_file():
            raise FileNotFoundError(helper)
        test_record_text = test_record.read_text(encoding="utf-8")
        for marker in meta["completion_markers"]:
            if str(marker) not in test_record_text:
                raise ValueError(
                    f"Missing TEST_RECORD completion evidence in {experiment}: {marker}"
                )

        by_panel = Counter((str(row["dataset"]), str(row["recall_tag"])) for row in selected)
        experiment_rows.append(
            {
                "source_experiment": experiment,
                "contribution_points": contributions[experiment],
                "points_within_xlim": sum(
                    int(row["point_within_xlim"]) for row in selected
                ),
                "sift1m_r1_points": by_panel[("sift1m", "r1")],
                "t2i1m_r1_points": by_panel[("t2i1m", "r1")],
                "sift1m_r100_points": by_panel[("sift1m", "r100")],
                "t2i1m_r100_points": by_panel[("t2i1m", "r100")],
                "recorded_total_cases": meta["recorded_total_cases"],
                "recorded_completed_cases": meta["recorded_completed_cases"],
                "recorded_failed_cases": meta["recorded_failed_cases"],
                "role": meta["role"],
                "test_record_ref": source_ref(test_record, source_root),
                "test_record_sha256": sha256(test_record),
                "yaml_generator_ref": source_ref(generator, source_root),
                "yaml_generator_sha256": sha256(generator),
                "result_summarizer_ref": source_ref(summarizer, source_root),
                "result_summarizer_sha256": sha256(summarizer),
                "submission_helper_ref": (
                    source_ref(helper, source_root) if helper is not None else ""
                ),
                "submission_helper_sha256": sha256(helper) if helper is not None else "",
                "case_manifest_ref": source_ref(manifest, source_root),
                "case_manifest_sha256": sha256(manifest),
            }
        )

    return (
        provenance,
        experiment_rows,
        plot_rows,
        juno_path,
        plot_path,
        range_path,
        speedup_path,
    )


def summary_lines(
    provenance: list[dict[str, object]],
    experiments: list[dict[str, object]],
    speedup_path: Path,
) -> list[str]:
    lines = ["metric\tvalue"]
    lines.append("plot_rows_total\t353")
    lines.append("juno_raw_rows\t277")
    lines.append("juno_plot_input_points\t164")
    lines.append("juno_raw_unplotted_points\t113")
    lines.append("mfnns_plot_input_points\t189")
    lines.append(
        "mfnns_points_within_xlim\t"
        f"{sum(int(row['point_within_xlim']) for row in provenance)}"
    )
    lines.append(
        "mfnns_points_below_xlim\t"
        f"{sum(not int(row['point_within_xlim']) for row in provenance)}"
    )
    for dataset, recall_tag in PANEL_ORDER:
        count = sum(
            row["dataset"] == dataset and row["recall_tag"] == recall_tag
            for row in provenance
        )
        lines.append(f"panel_points:{dataset}:{recall_tag}\t{count}")
    for row in experiments:
        lines.append(
            f"source_points:{row['source_experiment']}\t{row['contribution_points']}"
        )

    speedups = read_delimited(speedup_path, "\t")
    for recall in ("0.90", "0.95"):
        selected = [float(row["speedup"]) for row in speedups if row["recall"] == recall]
        require_equal(len(selected), 4, f"speedup sample count at {recall}")
        geomean = math.prod(selected) ** (1.0 / len(selected))
        lines.append(f"speedup_geomean_at_recall:{recall}\t{geomean:.12f}")
        lines.append(f"speedup_min_at_recall:{recall}\t{min(selected):.12f}")
        lines.append(f"speedup_max_at_recall:{recall}\t{max(selected):.12f}")
    return lines


def expected_outputs(
    provenance: list[dict[str, object]],
    experiments: list[dict[str, object]],
    juno_path: Path,
    plot_path: Path,
    range_path: Path,
    speedup_path: Path,
) -> dict[Path, bytes]:
    import io

    outputs: dict[Path, bytes] = {
        DEFAULT_JUNO_OUTPUT: juno_path.read_bytes(),
        DEFAULT_PLOT_OUTPUT: plot_path.read_bytes(),
        DEFAULT_RANGE_OUTPUT: range_path.read_bytes(),
        DEFAULT_SPEEDUP_OUTPUT: speedup_path.read_bytes(),
    }
    for path, rows, fields in (
        (DEFAULT_PROVENANCE_OUTPUT, provenance, PROVENANCE_FIELDS),
        (DEFAULT_EXPERIMENT_OUTPUT, experiments, EXPERIMENT_FIELDS),
    ):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=fields, delimiter=",", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        outputs[path] = buffer.getvalue().encode("utf-8")

    summary = "\n".join(summary_lines(provenance, experiments, speedup_path)) + "\n"
    outputs[DEFAULT_SUMMARY_OUTPUT] = summary.encode("utf-8")
    return outputs


def write_sha_file(paths: list[Path]) -> None:
    lines = [
        f"{sha256(path)}  {os.path.relpath(path, DEFAULT_SHA_OUTPUT.parent)}"
        for path in paths
    ]
    DEFAULT_SHA_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    (
        provenance,
        experiments,
        _,
        juno_path,
        plot_path,
        range_path,
        speedup_path,
    ) = build_rows(args.source_root)
    outputs = expected_outputs(
        provenance,
        experiments,
        juno_path,
        plot_path,
        range_path,
        speedup_path,
    )
    if args.check_only:
        for path, expected in outputs.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.read_bytes() != expected:
                raise ValueError(f"Frozen Figure 19 output differs from source audit: {path}")
        return

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_sha_file(
        [
            DEFAULT_JUNO_OUTPUT,
            DEFAULT_PLOT_OUTPUT,
            DEFAULT_PROVENANCE_OUTPUT,
            DEFAULT_EXPERIMENT_OUTPUT,
            DEFAULT_RANGE_OUTPUT,
            DEFAULT_SPEEDUP_OUTPUT,
        ]
    )
    print(f"Wrote {len(provenance)} MFNNS provenance rows")
    print(f"Wrote {len(experiments)} source-experiment rows")


if __name__ == "__main__":
    main()
