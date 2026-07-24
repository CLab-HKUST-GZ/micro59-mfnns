#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURE14_CSV = SCRIPT_DIR.parent / "figure14" / "data" / "figure14_results.csv"
DEFAULT_SPECS_CSV = SCRIPT_DIR / "data" / "area_specs.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "data" / "figure15_area_efficiency.csv"
DEFAULT_SUMMARY_TSV = SCRIPT_DIR / "output" / "figure15_summary.tsv"

TOP_K = "k10"
DATASET_ORDER = [
    ("deep10m", "Deep10M", "DP"),
    ("glove2m", "GloVe2M", "GV"),
    ("sift1m", "SIFT1M", "SF"),
    ("t2i1m", "T2I1M", "T2I"),
    ("w2v1m", "W2V1M", "W2V"),
    ("wiki1m", "Wiki1M", "WK"),
    ("pubmed", "PubMed", "PM"),
]
EXPECTED_METHODS = [
    "ANSMET",
    "NMP-Base",
    "NMP-FPMA",
    "NMP-FPSA",
    "NMP-Base-ET",
    "MFNNS",
]
EXPECTED_SOURCE_DESIGNS = {"ansmet", "ndp_base", "ndp_fpma", "mfnns"}
EXPECTED_FPMA_FALLBACK_DATASETS = {"wiki1m", "pubmed"}
EXPECTED_SPECS = {
    "ANSMET": ("ansmet", 0.020913242),
    "NMP-Base": ("ndp_base", 0.020913242),
    "NMP-FPMA": ("ndp_fpma", 0.01208361),
    "NMP-FPSA": ("ndp_fpma", 0.009678554),
    "NMP-Base-ET": ("mfnns", 0.020913242),
    "MFNNS": ("mfnns", 0.009678554),
}

OUTPUT_FIELDS = [
    "top_k",
    "dataset_key",
    "dataset_label",
    "dataset_short",
    "method",
    "source_design",
    "qps",
    "recall",
    "data_status",
    "figure14_source",
    "figure14_stats_ref",
    "figure14_config_ref",
    "figure14_note",
    "area_mm2",
    "area_efficiency_kqps_per_mm2",
    "normalized_area_efficiency_vs_ansmet",
    "derivation_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 15 Recall@10 area-efficiency data from the canonical "
            "Figure 14 results and the synthesized DPE area specification."
        )
    )
    parser.add_argument("--figure14", type=Path, default=DEFAULT_FIGURE14_CSV)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and derived metrics without writing files.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: set[str], label: str) -> None:
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_specs(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    require_columns(
        rows,
        {"method", "source_design", "area_mm2", "derivation_note"},
        "Area specification",
    )
    methods = [row["method"] for row in rows]
    if methods != EXPECTED_METHODS:
        raise ValueError(f"Unexpected method order: {methods}")
    if {row["source_design"] for row in rows} != EXPECTED_SOURCE_DESIGNS:
        raise ValueError("Area specification has unexpected Figure 14 sources")
    for row in rows:
        area = float(row["area_mm2"])
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"Invalid area for {row['method']}: {row['area_mm2']}")
        expected_design, expected_area = EXPECTED_SPECS[row["method"]]
        if row["source_design"] != expected_design:
            raise ValueError(
                f"Unexpected QPS source for {row['method']}: "
                f"{row['source_design']} != {expected_design}"
            )
        if not math.isclose(area, expected_area, rel_tol=0, abs_tol=1e-15):
            raise ValueError(
                f"Unexpected area for {row['method']}: {area} != {expected_area}"
            )
    return rows


def load_figure14_lookup(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(path)
    require_columns(
        rows,
        {
            "top_k",
            "dataset_key",
            "design",
            "qps",
            "recall",
            "data_status",
            "source",
            "stats_ref",
            "config_ref",
            "note",
        },
        "Figure 14 results",
    )

    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["top_k"], row["dataset_key"], row["design"])
        if key in lookup:
            raise ValueError(f"Duplicate Figure 14 row: {key}")
        lookup[key] = row

    required_keys = {
        (TOP_K, dataset_key, design)
        for dataset_key, _, _ in DATASET_ORDER
        for design in EXPECTED_SOURCE_DESIGNS
    }
    missing = sorted(required_keys - set(lookup))
    if missing:
        raise ValueError(f"Missing Figure 14 inputs: {missing}")

    for key in sorted(required_keys):
        row = lookup[key]
        qps = float(row["qps"])
        recall = float(row["recall"])
        if not math.isfinite(qps) or qps <= 0:
            raise ValueError(f"Invalid Figure 14 QPS for {key}: {row['qps']}")
        if not math.isfinite(recall) or recall <= 0.895:
            raise ValueError(f"Figure 15 source recall is not > 0.895 for {key}: {recall}")
        for field in ("stats_ref", "config_ref"):
            value = row[field].strip()
            if value and Path(value).is_absolute():
                raise ValueError(f"Absolute {field} in Figure 14 row {key}: {value}")

    fpma_rows = [
        lookup[(TOP_K, dataset_key, "ndp_fpma")]
        for dataset_key, _, _ in DATASET_ORDER
    ]
    fallback_datasets = {
        row["dataset_key"]
        for row in fpma_rows
        if row["data_status"] == "derived_copy_from_ndp_base"
    }
    if fallback_datasets != EXPECTED_FPMA_FALLBACK_DATASETS:
        raise ValueError(
            "Unexpected Recall@10 NMP-FPMA fallback set: "
            f"{sorted(fallback_datasets)}"
        )
    direct_count = sum(
        row["data_status"] != "derived_copy_from_ndp_base" for row in fpma_rows
    )
    if direct_count != 5:
        raise ValueError(f"Expected five direct k10 NMP-FPMA rows, found {direct_count}")
    for dataset_key, _, _ in DATASET_ORDER:
        fpma = lookup[(TOP_K, dataset_key, "ndp_fpma")]
        if dataset_key in fallback_datasets:
            base = lookup[(TOP_K, dataset_key, "ndp_base")]
            if fpma["qps"] != base["qps"] or fpma["recall"] != base["recall"]:
                raise ValueError(
                    f"NMP-FPMA fallback does not match NMP-Base for {dataset_key}"
                )
            if "copied_from=latest_ndp_base" not in fpma["note"]:
                raise ValueError(
                    f"NMP-FPMA fallback note is incomplete for {dataset_key}"
                )
        elif not fpma["data_status"].startswith("measured"):
            raise ValueError(
                f"Direct NMP-FPMA row is not marked measured for {dataset_key}: "
                f"{fpma['data_status']}"
            )

    return lookup


def build_rows(
    specs: list[dict[str, str]],
    figure14: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []

    for dataset_key, dataset_label, dataset_short in DATASET_ORDER:
        dataset_rows: list[dict[str, object]] = []
        ansmet_efficiency: float | None = None

        for spec in specs:
            source_design = spec["source_design"]
            source = figure14[(TOP_K, dataset_key, source_design)]
            qps = float(source["qps"])
            recall = float(source["recall"])
            area = float(spec["area_mm2"])
            efficiency = (qps / 1000.0) / area

            row: dict[str, object] = {
                "top_k": TOP_K,
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "dataset_short": dataset_short,
                "method": spec["method"],
                "source_design": source_design,
                "qps": qps,
                "recall": recall,
                "data_status": source["data_status"],
                "figure14_source": source["source"],
                "figure14_stats_ref": source["stats_ref"],
                "figure14_config_ref": source["config_ref"],
                "figure14_note": source["note"],
                "area_mm2": area,
                "area_efficiency_kqps_per_mm2": efficiency,
                "normalized_area_efficiency_vs_ansmet": None,
                "derivation_note": spec["derivation_note"],
            }
            dataset_rows.append(row)
            if spec["method"] == "ANSMET":
                ansmet_efficiency = efficiency

        if ansmet_efficiency is None:
            raise RuntimeError(f"Missing ANSMET row for {dataset_key}")
        for row in dataset_rows:
            row["normalized_area_efficiency_vs_ansmet"] = (
                float(row["area_efficiency_kqps_per_mm2"]) / ansmet_efficiency
            )
            result.append(row)

    if len(result) != len(DATASET_ORDER) * len(EXPECTED_METHODS):
        raise RuntimeError(f"Unexpected Figure 15 row count: {len(result)}")
    return result


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def build_summary(rows: list[dict[str, object]]) -> list[tuple[str, str, str]]:
    lookup = {
        (str(row["dataset_key"]), str(row["method"])): row
        for row in rows
    }
    dataset_keys = [dataset_key for dataset_key, _, _ in DATASET_ORDER]

    def ratios(numerator: str, denominator: str) -> list[float]:
        return [
            float(lookup[(dataset, numerator)]["area_efficiency_kqps_per_mm2"])
            / float(lookup[(dataset, denominator)]["area_efficiency_kqps_per_mm2"])
            for dataset in dataset_keys
        ]

    mfnns_vs_ansmet = ratios("MFNNS", "ANSMET")
    minimum = min(zip(mfnns_vs_ansmet, dataset_keys))
    maximum = max(zip(mfnns_vs_ansmet, dataset_keys))
    recalls = [float(row["recall"]) for row in rows]
    fpma_sources = [
        row for row in rows if str(row["method"]) == "NMP-FPMA"
    ]
    fallback_count = sum(
        row["data_status"] == "derived_copy_from_ndp_base" for row in fpma_sources
    )

    metrics = [
        (
            "mfnns_vs_ansmet_geomean",
            f"{geometric_mean(mfnns_vs_ansmet):.15f}",
            f"{geometric_mean(mfnns_vs_ansmet):.2f}x",
        ),
        ("mfnns_vs_ansmet_min", f"{minimum[0]:.15f}", minimum[1]),
        ("mfnns_vs_ansmet_max", f"{maximum[0]:.15f}", maximum[1]),
        (
            "nmp_fpma_vs_nmp_base_geomean",
            f"{geometric_mean(ratios('NMP-FPMA', 'NMP-Base')):.15f}",
            "1.73x",
        ),
        (
            "nmp_fpsa_vs_nmp_fpma_geomean",
            f"{geometric_mean(ratios('NMP-FPSA', 'NMP-FPMA')):.15f}",
            "1.25x",
        ),
        (
            "mfnns_vs_nmp_fpsa_geomean",
            f"{geometric_mean(ratios('MFNNS', 'NMP-FPSA')):.15f}",
            "2.06x",
        ),
        (
            "mfnns_vs_nmp_base_et_geomean",
            f"{geometric_mean(ratios('MFNNS', 'NMP-Base-ET')):.15f}",
            "2.16x",
        ),
        ("source_recall_min", f"{min(recalls):.15f}", "-"),
        ("source_recall_max", f"{max(recalls):.15f}", "-"),
        ("ndp_fpma_direct_count", str(len(fpma_sources) - fallback_count), "-"),
        ("ndp_fpma_fallback_count", str(fallback_count), "-"),
    ]
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "exact_value", "display_or_dataset"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    specs = load_specs(args.specs)
    figure14 = load_figure14_lookup(args.figure14)
    rows = build_rows(specs, figure14)
    summary = build_summary(rows)

    if not args.check_only:
        write_csv(args.output, rows)
        write_summary(args.summary, summary)

    summary_lookup = {metric: value for metric, value, _ in summary}
    print(
        "CHECK_OK "
        f"rows={len(rows)} "
        f"mfnns_vs_ansmet={summary_lookup['mfnns_vs_ansmet_geomean']} "
        f"fpma_direct={summary_lookup['ndp_fpma_direct_count']} "
        f"fpma_fallback={summary_lookup['ndp_fpma_fallback_count']}"
    )


if __name__ == "__main__":
    main()
