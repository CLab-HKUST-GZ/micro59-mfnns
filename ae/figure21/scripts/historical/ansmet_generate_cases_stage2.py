#!/usr/bin/env python3
"""Stage 2 refinement for ANSMET k=10 ef_search sweep.

Stage 1 results:
  sift1m:  ef=10→0.924 (can't go below k=10, unreachable)
  deep10m: ef=10→0.875, ef=12→0.900 → try ef=11
  w2v1m:   ef=25→0.899, ef=28→0.910 → try ef=26,27
  wiki1m:  ef=15→0.883, ef=20→0.932 → try ef=16,17,18,19
  t2i1m:   ef=25→0.909 ✅ IN BAND (done)
  gist1m:  ef=40→0.898, ef=45→0.907 → try ef=41,42,43,44
  pubmed:  ef=200→0.895, ef=250→0.915 → try ef=210,220,230,240
  glove2m: ef=400→0.890, ef=500→0.904, ef=600→0.908 → try ef=450,480
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path("/hpc2hdd/home/rmeng603/workspace/MFANNS")
EXP_DIR = ROOT / "simulator/memory/20260329/ansmet_recall09_k10_efsearch"
YAML_DIR = EXP_DIR / "yamls_stage2"
LOG_DIR = EXP_DIR / "logs_stage2"
MANIFEST_PATH = EXP_DIR / "case_manifest_stage2.tsv"

BASE_EXP_DIR = ROOT / "simulator/memory/20260326/multidataset_nine_config_fp16_hotrep_cfs"
BASE_YAML_DIR = BASE_EXP_DIR / "yamls"

K_NEIGHBORS = 10
TARGET_RECALL_LO = 0.9
TARGET_RECALL_HI = 0.91

MANIFEST_FIELDS = [
    "case_order", "dataset", "variant", "search_role", "case_name",
    "ef_search", "queue_size", "warmup_size", "k_neighbors",
    "yaml_path", "stats_path", "base_yaml", "note",
]

SEARCH_SPECS = [
    {"dataset": "deep10m", "variant": "normalized", "search_role": "refine",
     "ef_candidates": [11]},
    {"dataset": "w2v1m", "variant": "normalized", "search_role": "refine",
     "ef_candidates": [26, 27]},
    {"dataset": "wiki1m", "variant": "normalized", "search_role": "refine",
     "ef_candidates": [16, 17, 18, 19]},
    {"dataset": "gist1m", "variant": "normalized", "search_role": "refine",
     "ef_candidates": [41, 42, 43, 44]},
    {"dataset": "pubmed", "variant": "raw", "search_role": "refine",
     "ef_candidates": [210, 220, 230, 240]},
    {"dataset": "glove2m", "variant": "normalized", "search_role": "refine",
     "ef_candidates": [450, 480]},
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
                "note": f"ansmet_k{K_NEIGHBORS}_stage2_refine",
            })

    with MANIFEST_PATH.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"generated_cases\t{len(rows)}")
    print(f"manifest\t{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
