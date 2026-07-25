#!/usr/bin/env python3
"""Shared, portable Recall@10 energy model for paper Figures 16 and 17."""

from __future__ import annotations

import csv
import math
from pathlib import Path


AE_DIR = Path(__file__).resolve().parent
REPO_ROOT = AE_DIR.parent
DEFAULT_TRACE_TABLE = (
    REPO_ROOT
    / "simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10"
    / "execution_traces.csv"
)
DEFAULT_FIGURE14_CSV = AE_DIR / "figure14/data/figure14_results.csv"

DATASET_ORDER = [
    ("deep10m", "Deep10M", "DP"),
    ("glove2m", "GloVe2M", "GV"),
    ("sift1m", "SIFT1M", "SF"),
    ("t2i1m", "T2I1M", "T2I"),
    ("w2v1m", "W2V1M", "W2V"),
    ("wiki1m", "Wiki1M", "WK"),
    ("pubmed", "PubMed", "PM"),
]
TRACE_DESIGNS = ("cpu", "ansmet", "ndp_base", "ndp_fpma", "ndp_et", "mfnns")
METHOD_SPECS = [
    ("CPU", "cpu", "cpu", "std_fp"),
    ("ANSMET", "ansmet", "ansmet", "std_fp"),
    ("NMP-Base", "ndp_base", "ndp_base", "std_fp"),
    ("NMP-FPMA", "ndp_fpma", "ndp_fpma", "fpma"),
    ("NMP-FPSA", "ndp_fpma", "ndp_fpma", "fpsa"),
    ("NMP-Base-ET", "mfnns", "mfnns", "base_et"),
    ("NMP-FPSA-ET", "ndp_et", "ndp_et", "fpsa_et"),
    ("MFNNS", "mfnns", "mfnns", "fpsa_et"),
]
COMPUTE_FORMULAS = {
    "std_fp": "s_fmac_ops * 1.7 pJ",
    "fpma": "s_fmac_ops * 0.6 pJ",
    "fpsa": "s_fmac_ops * 0.46 pJ",
    "base_et": "phase1_ops * 1.7 pJ + phase2_fmac_ops * 1.7 pJ",
    "fpsa_et": "phase1_ops * 2 * 0.06 pJ + phase2_fmac_ops * 0.28 pJ",
}

TRACE_REQUIRED_COLUMNS = {
    "top_k",
    "dataset_key",
    "dataset_label",
    "dataset_short",
    "design",
    "trace_source_design",
    "config_ref",
    "source_stats_ref",
    "source_energy_ref",
    "source_stats_sha256",
    "source_energy_sha256",
    "data_status",
    "s_num_query",
    "s_mem_cycle",
    "recall",
    "s_fmac_ops_sum",
    "phase1_ops_sum",
    "phase2_fmac_ops_sum",
    "raw_memory_energy_nj",
    "memory_energy_scale",
    "memory_energy_nj",
    "memory_energy_terms",
}
FIGURE14_REQUIRED_COLUMNS = {
    "top_k",
    "dataset_key",
    "design",
    "qps",
    "recall",
    "data_status",
    "config_ref",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def require_columns(
    rows: list[dict[str, str]], required: set[str], label: str
) -> None:
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def validate_config(config_ref: str, dataset_key: str, design: str) -> None:
    ref = Path(config_ref)
    if ref.is_absolute():
        raise ValueError(f"Absolute config_ref is forbidden: {config_ref}")
    expected = Path(
        "simulator/run_case/figure14_recall_gt0895/configs/final/k10"
    ) / dataset_key / f"{design}.yaml"
    if ref != expected:
        raise ValueError(f"Unexpected config_ref: {ref} != {expected}")
    path = REPO_ROOT / ref
    if not path.is_file():
        raise FileNotFoundError(f"Missing final YAML: {path}")
    text = path.read_text(encoding="utf-8")
    expected_stat = (
        "stat_path: simulator/run_case/figure14_recall_gt0895/results/"
        f"k10/{dataset_key}/{design}_stats.yml"
    )
    required_fragments = (
        expected_stat,
        "nQueryLimit: 1000",
        "nParallelQuery: 1000",
        "drampower_enable: true",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(f"{config_ref} is missing required settings: {missing}")


def load_trace_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    require_columns(rows, TRACE_REQUIRED_COLUMNS, "Execution trace table")
    expected = {
        (dataset_key, design)
        for dataset_key, _, _ in DATASET_ORDER
        for design in TRACE_DESIGNS
    }
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset_key"], row["design"])
        if key in lookup:
            raise ValueError(f"Duplicate execution trace row: {key}")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError(
            "Execution trace matrix mismatch: "
            f"missing={sorted(expected-set(lookup))}, "
            f"extra={sorted(set(lookup)-expected)}"
        )

    for (dataset_key, design), row in lookup.items():
        if row["top_k"] != "k10":
            raise ValueError(f"Non-k10 trace row: {(dataset_key, design)}")
        if int(row["s_num_query"]) != 1000:
            raise ValueError(f"Trace is not nq1000: {(dataset_key, design)}")
        for field in (
            "config_ref",
            "source_stats_ref",
            "source_energy_ref",
        ):
            if Path(row[field]).is_absolute():
                raise ValueError(f"Absolute {field} is forbidden: {row[field]}")
        for field in ("source_stats_sha256", "source_energy_sha256"):
            digest = row[field]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid SHA-256 in {field}: {digest}")
        validate_config(row["config_ref"], dataset_key, design)

        raw = float(row["raw_memory_energy_nj"])
        scale = float(row["memory_energy_scale"])
        memory = float(row["memory_energy_nj"])
        if not math.isclose(raw * scale, memory, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Memory-energy scale mismatch: {(dataset_key, design)}")
        expected_terms = 1 if design == "cpu" else 32
        if int(row["memory_energy_terms"]) != expected_terms:
            raise ValueError(
                f"Unexpected memory-energy term count: {(dataset_key, design)}"
            )
        if design == "cpu":
            if not math.isclose(scale, 30.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"CPU historical scale is not x30: {dataset_key}")
        elif not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Non-CPU energy must be direct: {(dataset_key, design)}")
    return lookup


def load_figure14_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    require_columns(rows, FIGURE14_REQUIRED_COLUMNS, "Figure 14 results")
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["top_k"] != "k10":
            continue
        key = (row["dataset_key"], row["design"])
        if key in lookup:
            raise ValueError(f"Duplicate Figure 14 row: {key}")
        lookup[key] = row
    required_designs = {spec[2] for spec in METHOD_SPECS} | {"gpu_cagra", "bang"}
    expected = {
        (dataset_key, design)
        for dataset_key, _, _ in DATASET_ORDER
        for design in required_designs
    }
    missing = expected - set(lookup)
    if missing:
        raise ValueError(f"Missing Figure 14 Recall@10 rows: {sorted(missing)}")
    for key in expected:
        row = lookup[key]
        qps = float(row["qps"])
        recall = float(row["recall"])
        if not math.isfinite(qps) or qps <= 0:
            raise ValueError(f"Invalid Figure 14 QPS for {key}: {qps}")
        if (
            row["design"] not in {"gpu_cagra", "bang"}
            and (not math.isfinite(recall) or recall <= 0.895)
        ):
            raise ValueError(f"Figure 14 recall is not >0.895 for {key}: {recall}")
        if Path(row["config_ref"]).is_absolute():
            raise ValueError(f"Absolute Figure 14 config_ref for {key}")
    return lookup


def compute_energy_nj(trace: dict[str, str], model: str) -> float:
    fmac = int(trace["s_fmac_ops_sum"])
    phase1 = int(trace["phase1_ops_sum"])
    phase2 = int(trace["phase2_fmac_ops_sum"])
    if model == "std_fp":
        energy_pj = fmac * 1.7
    elif model == "fpma":
        energy_pj = fmac * 0.6
    elif model == "fpsa":
        energy_pj = fmac * 0.46
    elif model == "base_et":
        energy_pj = phase1 * 1.7 + phase2 * 1.7
    elif model == "fpsa_et":
        energy_pj = phase1 * 2 * 0.06 + phase2 * 0.28
    else:
        raise ValueError(f"Unknown compute model: {model}")
    return energy_pj / 1000.0


def build_energy_rows(
    trace_path: Path = DEFAULT_TRACE_TABLE,
    figure14_path: Path = DEFAULT_FIGURE14_CSV,
) -> list[dict[str, object]]:
    traces = load_trace_lookup(trace_path)
    figure14 = load_figure14_lookup(figure14_path)
    rows: list[dict[str, object]] = []
    for dataset_key, dataset_label, dataset_short in DATASET_ORDER:
        for method, trace_design, qps_design, model in METHOD_SPECS:
            trace = traces[(dataset_key, trace_design)]
            qps_row = figure14[(dataset_key, qps_design)]
            compute = compute_energy_nj(trace, model)
            memory = float(trace["memory_energy_nj"])
            total = compute + memory
            qps = float(qps_row["qps"])
            power_w = total * qps / 1e12
            rows.append(
                {
                    "top_k": "k10",
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "dataset_short": dataset_short,
                    "method": method,
                    "trace_design": trace_design,
                    "trace_source_design": trace["trace_source_design"],
                    "qps_source_design": qps_design,
                    "qps": qps,
                    "recall": float(qps_row["recall"]),
                    "config_ref": trace["config_ref"],
                    "trace_table_ref": str(trace_path.relative_to(REPO_ROOT)),
                    "data_status": trace["data_status"],
                    "s_num_query": int(trace["s_num_query"]),
                    "s_mem_cycle": int(trace["s_mem_cycle"]),
                    "compute_formula": COMPUTE_FORMULAS[model],
                    "compute_energy_nj": compute,
                    "memory_energy_nj": memory,
                    "total_energy_nj": total,
                    "power_w": power_w,
                    "energy_efficiency_qps_per_w": 1e12 / total,
                }
            )
    validate_base_et_mfnns_pair(rows)
    return rows


def validate_base_et_mfnns_pair(rows: list[dict[str, object]]) -> None:
    """Base-ET reuses the MFNNS trace and changes only compute energy."""

    by_key = {
        (str(row["dataset_key"]), str(row["method"])): row for row in rows
    }
    for dataset_key, _, _ in DATASET_ORDER:
        base = by_key[(dataset_key, "NMP-Base-ET")]
        mfnns = by_key[(dataset_key, "MFNNS")]
        shared_fields = (
            "trace_design",
            "trace_source_design",
            "qps_source_design",
            "qps",
            "recall",
            "config_ref",
            "s_num_query",
            "s_mem_cycle",
            "memory_energy_nj",
        )
        for field in shared_fields:
            if base[field] != mfnns[field]:
                raise ValueError(
                    f"{dataset_key}: Base-ET/MFNNS mismatch in {field}"
                )
        if float(base["compute_energy_nj"]) <= float(mfnns["compute_energy_nj"]):
            raise ValueError(
                f"{dataset_key}: Base-ET compute energy must exceed MFNNS"
            )
        if float(base["total_energy_nj"]) <= float(mfnns["total_energy_nj"]):
            raise ValueError(
                f"{dataset_key}: Base-ET total energy must exceed MFNNS"
            )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))
