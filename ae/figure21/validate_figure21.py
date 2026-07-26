#!/usr/bin/env python3
"""Read-only validation for the Figure 21 AE bundle."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import struct
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP = SCRIPT_DIR / "data/figure21_sweep_results.tsv"
PROVENANCE = SCRIPT_DIR / "data/simulator_provenance.tsv"
PLOT_DATA = SCRIPT_DIR / "output/figure21_plot_data.tsv"
INPUT_MANIFEST = SCRIPT_DIR / "data/input_manifest.tsv"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MODEL_REF = "mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin"
QUERY_REF = "ae/figure21/inputs/query_vectors_n100_seed42.bin"
GT_REF = "ae/figure21/inputs/gt_labels_topk32_n100_seed42.bin"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(text: str, key: str) -> str:
    match = re.search(
        rf"^\s{{2}}{re.escape(key)}:\s*({FLOAT_PATTERN})\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Missing {key}")
    return match.group(1)


def extract_text(text: str, key: str) -> str:
    match = re.search(
        rf"^\s{{2}}{re.escape(key)}:\s*([^\s#]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Missing {key}")
    return match.group(1)


def validate_normalized_query_and_gt() -> None:
    query_path = SCRIPT_DIR / "inputs/query_vectors_n100_seed42.bin"
    with query_path.open("rb") as stream:
        rows, dimensions = struct.unpack("<II", stream.read(8))
        payload = stream.read()
    if (rows, dimensions) != (100, 200):
        raise ValueError(f"Unexpected Figure 21 query shape: {rows}x{dimensions}")
    if len(payload) != rows * dimensions * 4:
        raise ValueError("Figure 21 query payload size mismatch")
    values = struct.unpack(f"<{rows * dimensions}f", payload)
    for row in range(rows):
        offset = row * dimensions
        norm = math.sqrt(
            math.fsum(
                value * value
                for value in values[offset : offset + dimensions]
            )
        )
        if not math.isfinite(norm) or abs(norm - 1.0) > 2.0e-5:
            raise ValueError(
                f"Figure 21 query row {row} is not L2-normalized: {norm}"
            )

    gt_path = SCRIPT_DIR / "inputs/gt_labels_topk32_n100_seed42.bin"
    with gt_path.open("rb") as stream:
        gt_rows, gt_k = struct.unpack("<II", stream.read(8))
        gt_payload = stream.read()
    if (gt_rows, gt_k) != (rows, 32):
        raise ValueError(f"Unexpected Figure 21 GT shape: {gt_rows}x{gt_k}")
    if len(gt_payload) != gt_rows * gt_k * 4:
        raise ValueError("Figure 21 GT payload size mismatch")
    labels = struct.unpack(f"<{gt_rows * gt_k}I", gt_payload)
    if any(label >= 1_000_000 for label in labels):
        raise ValueError("Figure 21 GT contains an out-of-range T2I1M label")


def main() -> None:
    sweep = read_tsv(SWEEP)
    if len(sweep) != 243:
        raise ValueError(f"Expected 243 sweep rows, found {len(sweep)}")
    expected_keys = {
        (ef_search, queue_size)
        for ef_search in (20, 30, 40)
        for queue_size in range(20, 101)
    }
    actual_keys = {
        (int(row["ef_search"]), int(row["queue_size"])) for row in sweep
    }
    if actual_keys != expected_keys:
        raise ValueError("MFNNS sweep grid is incomplete or duplicated")
    if any(row["final_status"] != "PASS" for row in sweep):
        raise ValueError("The frozen MFNNS sweep contains non-PASS rows")

    provenance = read_tsv(PROVENANCE)
    if len(provenance) != 246:
        raise ValueError(f"Expected 246 provenance rows, found {len(provenance)}")
    expected_provenance_keys = {
        ("mfnns", ef_search, queue_size)
        for ef_search, queue_size in expected_keys
    } | {("ansmet", ef_search, 33) for ef_search in (20, 30, 40)}
    actual_provenance_keys = {
        (row["method"], int(row["ef_search"]), int(row["queue_size"]))
        for row in provenance
    }
    if actual_provenance_keys != expected_provenance_keys:
        raise ValueError("Simulator provenance grid is incomplete or duplicated")

    frozen = {
        (int(row["ef_search"]), int(row["queue_size"])): row for row in sweep
    }
    for row in provenance:
        config = SCRIPT_DIR.parents[1] / row["config_ref"]
        if sha256(config) != row["portable_yaml_sha256"]:
            raise ValueError(f"Portable YAML digest mismatch: {config}")
        if row["final_status"] != "PASS":
            raise ValueError(f"Non-PASS provenance row: {row['case_name']}")
        text = config.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"YAML root is not a mapping: {config}")
        if int(extract(text, "ef_search")) != int(row["ef_search"]):
            raise ValueError(f"ef_search mismatch: {config}")
        if int(extract(text, "dualQueueLowerBoundQueueSize")) != int(
            row["queue_size"]
        ):
            raise ValueError(f"queue-size mismatch: {config}")
        if int(extract(text, "dualQueueLowerBoundWarmupSize")) != int(
            row["warmup_size"]
        ):
            raise ValueError(f"warmup-size mismatch: {config}")
        expected_paths = {
            "model_path": MODEL_REF,
            "query_path": QUERY_REF,
            "gt_path": GT_REF,
            "stat_path": (
                f"stats/{row['method']}/{config.stem}_stats.yml"
            ),
        }
        for key, expected_value in expected_paths.items():
            if extract_text(text, key) != expected_value:
                raise ValueError(f"Non-portable {key}: {config}")

        if row["method"] == "mfnns":
            source = frozen[(int(row["ef_search"]), int(row["queue_size"]))]
            for field in (
                "config_ref",
                "warmup_size",
                "slurm_job_id",
                "final_status",
            ):
                if row[field] != source[field]:
                    raise ValueError(
                        f"Sweep/provenance {field} mismatch: {row['case_name']}"
                    )
            if float(row["recall"]) != float(source["recall"]):
                raise ValueError(
                    f"Sweep/provenance recall mismatch: {row['case_name']}"
                )
            if float(row["s_mem_cycle"]) != float(source["s_mem_cycle"]):
                raise ValueError(
                    f"Sweep/provenance cycle mismatch: {row['case_name']}"
                )
        else:
            stats = (
                SCRIPT_DIR
                / "data/ansmet_stats"
                / f"{row['case_name']}_stats.yml"
            )
            if sha256(stats) != row["source_stats_sha256"]:
                raise ValueError(f"ANSMET stats digest mismatch: {stats}")
            stats_text = stats.read_text(encoding="utf-8")
            if float(extract(stats_text, "s_recall_rate")) != float(row["recall"]):
                raise ValueError(f"ANSMET recall mismatch: {stats}")
            if int(extract(stats_text, "s_mem_cycle")) != int(row["s_mem_cycle"]):
                raise ValueError(f"ANSMET cycle mismatch: {stats}")

    plot_rows = read_tsv(PLOT_DATA)
    if len(plot_rows) != 243:
        raise ValueError(f"Expected 243 plotted rows, found {len(plot_rows)}")
    plotted_keys = {
        (int(row["ef_search"]), int(row["queue_size"])) for row in plot_rows
    }
    if plotted_keys != expected_keys:
        raise ValueError("Plotted sweep grid is incomplete or duplicated")
    global_max = max(float(row["s_mem_cycle"]) for row in sweep)
    if global_max != 618147:
        raise ValueError(f"Unexpected global maximum cycle: {global_max}")
    for row in plot_rows:
        key = (int(row["ef_search"]), int(row["queue_size"]))
        source = frozen[key]
        if float(row["recall"]) != float(source["recall"]):
            raise ValueError(f"Recall mismatch at {key}")
        if float(row["s_mem_cycle"]) != float(source["s_mem_cycle"]):
            raise ValueError(f"Cycle mismatch at {key}")
        expected_throughput = global_max / float(source["s_mem_cycle"])
        if abs(float(row["norm_throughput"]) - expected_throughput) > 5e-7:
            raise ValueError(f"Normalized throughput mismatch at {key}")

    input_rows = read_tsv(INPUT_MANIFEST)
    expected_inputs = {
        "hnsw_index_M32_ef100.bin": "no",
        "query_vectors_n100_seed42.bin": "yes",
        "gt_labels_topk32_n100_seed42.bin": "yes",
    }
    if {row["artifact"]: row["included"] for row in input_rows} != expected_inputs:
        raise ValueError("Input manifest contents are incomplete or duplicated")
    for row in input_rows:
        if row["included"] != "yes":
            continue
        path = SCRIPT_DIR / "inputs" / row["artifact"]
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Input size mismatch: {path}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Input digest mismatch: {path}")

    validate_normalized_query_and_gt()

    print(
        "CHECK_OK "
        "sweep=243 ansmet=3 configs=246 "
        f"normalized_queries=100 global_max_cycle={global_max:.0f}"
    )


if __name__ == "__main__":
    main()
