#!/usr/bin/env python3
"""Validate the portable 126-point Figure 14 YAML matrix."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[2]
FINAL_ROOT = CASE_ROOT / "configs/final"
MANIFEST = CASE_ROOT / "manifests/final_cases.tsv"
K100_SOURCES = CASE_ROOT / "manifests/k100_sources.tsv"
DATASETS = (
    "deep10m",
    "glove2m",
    "sift1m",
    "t2i1m",
    "w2v1m",
    "wiki1m",
    "pubmed",
)
DESIGNS = ("cpu", "ansmet", "ndp_base", "ndp_fpma", "ndp_et", "mfnns")


def scalar(text: str, field: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(field)}:\s*([^#\s]+)", text, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"missing YAML field: {field}")
    return match.group(1)


def rewrite_unique(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected one derived-config field, found {count}")
    return updated


def main() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 126:
        raise RuntimeError(f"expected 126 manifest rows, found {len(rows)}")

    expected = {
        (top_k, dataset, design)
        for top_k in (5, 10, 100)
        for dataset in DATASETS
        for design in DESIGNS
    }
    seen: set[tuple[int, str, str]] = set()
    paths: set[Path] = set()
    status_counts: dict[str, int] = {}
    for row in rows:
        top_k = int(row["top_k"])
        dataset = row["dataset"]
        design = row["design"]
        key = (top_k, dataset, design)
        if key in seen:
            raise RuntimeError(f"duplicate manifest key: {key}")
        seen.add(key)
        if row["normalization"] != "normalized":
            raise RuntimeError(f"non-normalized manifest row: {key}")

        relative = Path(row["config_path"])
        if relative.is_absolute():
            raise RuntimeError(f"absolute config path: {relative}")
        path = REPO_ROOT / relative
        expected_path = FINAL_ROOT / f"k{top_k}" / dataset / f"{design}.yaml"
        if path != expected_path or not path.is_file():
            raise RuntimeError(f"bad config mapping for {key}: {relative}")
        paths.add(path)

        text = path.read_text()
        absolute_scalars = re.findall(
            r"^\s*[A-Za-z0-9_]+:\s*(/[^\s#]+)", text, re.MULTILINE
        )
        if absolute_scalars:
            raise RuntimeError(f"{relative}: absolute paths {absolute_scalars}")
        if int(scalar(text, "gt_k")) != top_k:
            raise RuntimeError(f"{key}: gt_k mismatch")
        if int(scalar(text, "k_neighbors")) != top_k:
            raise RuntimeError(f"{key}: k_neighbors mismatch")
        if int(scalar(text, "nQueryLimit")) != 1000:
            raise RuntimeError(f"{key}: nQueryLimit mismatch")
        if int(scalar(text, "nParallelQuery")) != 1000:
            raise RuntimeError(f"{key}: nParallelQuery mismatch")
        if int(scalar(text, "ef_search")) != int(row["ef_search"]):
            raise RuntimeError(f"{key}: ef_search mismatch")
        if row["queue_size"] != "NA":
            if int(scalar(text, "dualQueueLowerBoundQueueSize")) != int(
                row["queue_size"]
            ):
                raise RuntimeError(f"{key}: queue_size mismatch")

        base = f"mfnns_hnswlib/cpu_index/{dataset}"
        expected_paths = {
            "model_path": f"{base}/hnsw_index_M32_ef100.bin",
            "query_path": f"{base}/query_vectors_n1000_seed42.bin",
            "gt_path": f"{base}/gt_labels_topk{top_k}_n1000_seed42.bin",
            "stat_path": (
                "simulator/run_case/figure14_recall_gt0895/results/"
                f"k{top_k}/{dataset}/{design}_stats.yml"
            ),
        }
        for field, expected_value in expected_paths.items():
            value = scalar(text, field)
            if value != expected_value:
                raise RuntimeError(
                    f"{key}: {field}={value}, expected {expected_value}"
                )

        status = row["historical_result_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    actual_paths = set(FINAL_ROOT.glob("k*/*/*.yaml"))
    if seen != expected or paths != actual_paths:
        raise RuntimeError("portable Figure 14 matrix is incomplete or has extras")
    expected_statuses = {
        "completed_bundled": 47,
        "completed_external": 34,
        "not_run_cancelled": 3,
        "matched_direct": 33,
        "figure_copied_from_ndp_base": 7,
        "figure_manual_override": 2,
    }
    if status_counts != expected_statuses:
        raise RuntimeError(
            f"unexpected historical status counts: {status_counts}"
        )
    with K100_SOURCES.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    source_keys = {
        (row["dataset"], row["design"])
        for row in source_rows
    }
    expected_source_keys = {
        (dataset, design)
        for dataset in DATASETS
        for design in DESIGNS
    }
    if len(source_rows) != 42 or source_keys != expected_source_keys:
        raise RuntimeError("k100 source map is incomplete or has duplicates")
    manifest_by_key = {
        (int(row["top_k"]), row["dataset"], row["design"]): row
        for row in rows
    }
    source_status_counts: dict[str, int] = {}
    for row in source_rows:
        for field in ("source_yaml_ref", "source_stats_ref", "config_path"):
            if Path(row[field]).is_absolute():
                raise RuntimeError(f"absolute k100 source reference: {row[field]}")
        for field in ("source_yaml_sha256", "source_stats_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise RuntimeError(f"invalid k100 source checksum: {row[field]}")
        manifest_row = manifest_by_key[(100, row["dataset"], row["design"])]
        for source_field, manifest_field in (
            ("match_status", "historical_result_status"),
            ("ef_search", "ef_search"),
            ("config_path", "config_path"),
        ):
            if row[source_field] != manifest_row[manifest_field]:
                raise RuntimeError(
                    f"k100 source/manifest mismatch: "
                    f"{row['dataset']}/{row['design']}/{source_field}"
                )
        status = row["match_status"]
        source_status_counts[status] = source_status_counts.get(status, 0) + 1
        if status in {"matched_direct", "figure_manual_override"}:
            if not math.isclose(
                float(row["figure_qps"]),
                float(row["source_result_qps"]),
                rel_tol=2e-6,
                abs_tol=0.01,
            ):
                raise RuntimeError(
                    f"k100 QPS mismatch: {row['dataset']}/{row['design']}"
                )
            if not math.isclose(
                float(row["figure_recall"]),
                float(row["source_result_recall"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise RuntimeError(
                    f"k100 recall mismatch: {row['dataset']}/{row['design']}"
                )
    expected_source_statuses = {
        "matched_direct": 33,
        "figure_copied_from_ndp_base": 7,
        "figure_manual_override": 2,
    }
    if source_status_counts != expected_source_statuses:
        raise RuntimeError(
            f"unexpected k100 source statuses: {source_status_counts}"
        )

    for dataset in DATASETS:
        base_path = FINAL_ROOT / "k100" / dataset / "ndp_base.yaml"
        fpma_path = FINAL_ROOT / "k100" / dataset / "ndp_fpma.yaml"
        expected_fpma = base_path.read_text()
        expected_fpma = rewrite_unique(
            expected_fpma,
            r"^  stat_path:\s*.*$",
            "  stat_path: "
            "simulator/run_case/figure14_recall_gt0895/results/"
            f"k100/{dataset}/ndp_fpma_stats.yml",
        )
        for field, value in (
            ("cyclesPerFMAC", 13),
            ("multiplierLatencyCycles", 13),
            ("dualQueueCoarseFinalizeCycles", 12),
            ("dualQueueFineFinalizeCycles", 12),
            ("stdFpFinalizeCycles", 0),
        ):
            expected_fpma = rewrite_unique(
                expected_fpma,
                rf"^    {field}:\s*.*$",
                f"    {field}: {value}",
            )
        if fpma_path.read_text() != expected_fpma:
            raise RuntimeError(
                f"k100 derived NDP-FPMA delta differs for {dataset}"
            )

    print(
        "PASS: 126 normalized, repository-relative Figure 14 YAMLs; "
        "k100 status counts 33 direct + 2 manual + 7 copied"
    )


if __name__ == "__main__":
    main()
