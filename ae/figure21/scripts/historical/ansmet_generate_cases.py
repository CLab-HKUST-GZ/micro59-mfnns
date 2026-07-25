#!/usr/bin/env python3
"""ANSMET ef_search sweep for k_neighbors=10, target recall@10 in [0.9, 0.91).

Based on 20260328/ansmet_recall09_efsearch but with k_neighbors=10.
Uses same binary with ef_search unclamped + recall@k denominator fix.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path("/hpc2hdd/home/rmeng603/workspace/MFANNS")
EXP_DIR = ROOT / "simulator/memory/20260329/ansmet_recall09_k10_efsearch"
YAML_DIR = EXP_DIR / "yamls"
LOG_DIR = EXP_DIR / "logs"
MANIFEST_PATH = EXP_DIR / "case_manifest.tsv"
COMMANDS_PATH = EXP_DIR / "commands.sh"
ERROR_LOG_PATH = EXP_DIR / "error_log.md"

BASE_EXP_DIR = ROOT / "simulator/memory/20260326/multidataset_nine_config_fp16_hotrep_cfs"
BASE_YAML_DIR = BASE_EXP_DIR / "yamls"
BUILD_DIR = ROOT / "simulator/build_compute_sysgcc"
RAMULATOR_BIN = BUILD_DIR / "ramulator2"

K_NEIGHBORS = 10
TARGET_RECALL_LO = 0.9
TARGET_RECALL_HI = 0.91

MANIFEST_FIELDS = [
    "case_order", "dataset", "variant", "search_role", "case_name",
    "ef_search", "queue_size", "warmup_size", "k_neighbors",
    "yaml_path", "stats_path", "base_yaml", "note",
]

# ef_search ranges estimated from k=32 results (k=10 needs lower ef than k=32)
SEARCH_SPECS = [
    {
        "dataset": "sift1m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [10, 12, 14, 16, 18, 20, 25],
    },
    {
        "dataset": "deep10m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [10, 12, 14, 16, 18, 20, 25],
    },
    {
        "dataset": "w2v1m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [10, 14, 18, 20, 22, 25, 28],
    },
    {
        "dataset": "wiki1m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [10, 15, 20, 22, 25, 28, 32],
    },
    {
        "dataset": "t2i1m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [10, 15, 20, 25, 28, 30, 35],
    },
    {
        "dataset": "gist1m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [15, 20, 25, 30, 35, 40, 45],
    },
    {
        "dataset": "pubmed",
        "variant": "raw",
        "search_role": "trim_ef",
        "ef_candidates": [50, 80, 100, 150, 200, 250],
    },
    {
        "dataset": "glove2m",
        "variant": "normalized",
        "search_role": "trim_ef",
        "ef_candidates": [100, 200, 300, 400, 500, 600],
    },
]


def load_yaml(path: Path) -> dict:
    with path.open() as fin:
        return yaml.safe_load(fin)


def dump_yaml(data: dict, path: Path) -> None:
    with path.open("w") as fout:
        yaml.safe_dump(data, fout, sort_keys=False)


def base_yaml_path(dataset: str, variant: str) -> Path:
    return BASE_YAML_DIR / f"{dataset}_{variant}_ansmet.yaml"


def configure_yaml(data: dict, ef_search: int, stats_path: Path) -> None:
    front = data["Frontend"]
    front["ef_search"] = ef_search
    front["k_neighbors"] = K_NEIGHBORS
    front["stat_path"] = str(stats_path)
    front["perQuerySummaryEnable"] = False
    front["debugIssueTraceEnable"] = False
    front["debugDuplicateAcceptEnable"] = False
    # Set nFMAC=16 as in k=32 experiment
    inner = front["Frontend"]
    inner["nFMAC"] = 16


def main() -> None:
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    case_order = 0

    for spec in SEARCH_SPECS:
        dataset = spec["dataset"]
        variant = spec["variant"]
        base_yaml = base_yaml_path(dataset, variant)

        for ef_search in spec["ef_candidates"]:
            case_order += 1
            case_name = f"{dataset}_{variant}_k{K_NEIGHBORS}_ef{ef_search}"
            yaml_path = YAML_DIR / f"{case_name}.yaml"
            stats_path = LOG_DIR / f"{case_name}_stats.yml"

            data = load_yaml(base_yaml)
            configure_yaml(data, ef_search, stats_path)
            dump_yaml(data, yaml_path)

            rows.append({
                "case_order": str(case_order),
                "dataset": dataset,
                "variant": variant,
                "search_role": spec["search_role"],
                "case_name": case_name,
                "ef_search": str(ef_search),
                "queue_size": "33",
                "warmup_size": "32",
                "k_neighbors": str(K_NEIGHBORS),
                "yaml_path": str(yaml_path),
                "stats_path": str(stats_path),
                "base_yaml": str(base_yaml),
                "note": f"ansmet_k{K_NEIGHBORS}_nfmac16;target_recall_[{TARGET_RECALL_LO},{TARGET_RECALL_HI})",
            })

    with MANIFEST_PATH.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"generated_cases\t{len(rows)}")
    print(f"manifest\t{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
