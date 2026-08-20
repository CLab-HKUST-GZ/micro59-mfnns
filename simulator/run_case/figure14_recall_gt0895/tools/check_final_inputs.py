#!/usr/bin/env python3
"""Validate the generated index/query/GT inputs used by Figure 14 YAMLs."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import yaml


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[2]
FINAL_ROOT = CASE_ROOT / "configs/final"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "mfnns_hnswlib/cpu_index"
HNSW_HEADER = struct.Struct("<QQQQQQiIQQQdQ")
HNSW_FIELDS = (
    "offset_level0",
    "max_elements",
    "current_elements",
    "size_data_per_element",
    "label_offset",
    "offset_data",
    "max_level",
    "entrypoint",
    "maxM",
    "maxM0",
    "M",
    "mult",
    "ef_construction",
)
DATASET_SPECS = {
    "deep10m": (10_000_000, 96),
    "glove2m": (2_196_017, 300),
    "sift1m": (1_000_000, 128),
    "t2i1m": (1_000_000, 200),
    "w2v1m": (1_000_000, 300),
    "wiki1m": (1_000_000, 768),
    "pubmed": (500_000, 768),
}
TOP_KS = (5, 10, 100)


def config_selections(configs: list[str]) -> dict[str, set[int]]:
    if not configs:
        return {dataset: set(TOP_KS) for dataset in DATASET_SPECS}

    selections: dict[str, set[int]] = defaultdict(set)
    for value in configs:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"config path must be repository-relative: {value}")
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"missing config: {value}")
        try:
            path.resolve().relative_to(FINAL_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"config is outside the final YAML tree: {value}") from exc
        parsed = yaml.safe_load(path.read_text())
        frontend = parsed.get("Frontend") if isinstance(parsed, dict) else None
        if not isinstance(frontend, dict):
            raise ValueError(f"missing Frontend mapping: {value}")

        model_path = Path(str(frontend.get("model_path", "")))
        query_path = Path(str(frontend.get("query_path", "")))
        gt_path = Path(str(frontend.get("gt_path", "")))
        try:
            top_k = int(frontend["gt_k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid gt_k: {value}") from exc
        dataset = model_path.parent.name
        if dataset not in DATASET_SPECS or top_k not in TOP_KS:
            raise ValueError(f"unsupported Figure 14 dataset/top-k: {value}")

        base = Path("mfnns_hnswlib/cpu_index") / dataset
        expected = {
            "model_path": base / "hnsw_index_M32_ef100.bin",
            "query_path": base / "query_vectors_n1000_seed42.bin",
            "gt_path": base / f"gt_labels_topk{top_k}_n1000_seed42.bin",
        }
        actual = {
            "model_path": model_path,
            "query_path": query_path,
            "gt_path": gt_path,
        }
        for field, wanted in expected.items():
            if actual[field] != wanted:
                raise ValueError(
                    f"{value}: {field}={actual[field]}, expected {wanted}"
                )
        selections[dataset].add(top_k)
    return dict(selections)


def read_hnsw_header(path: Path) -> dict[str, int | float]:
    with path.open("rb") as stream:
        raw = stream.read(HNSW_HEADER.size)
    if len(raw) != HNSW_HEADER.size:
        raise ValueError(f"short HNSW header ({len(raw)} bytes): {path}")
    return dict(zip(HNSW_FIELDS, HNSW_HEADER.unpack(raw)))


def validate_hnsw(path: Path, expected_elements: int) -> list[str]:
    if not path.is_file():
        return [f"missing index: {path}"]
    try:
        values = read_hnsw_header(path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    expected = {
        "max_elements": expected_elements,
        "current_elements": expected_elements,
        "maxM": 32,
        "maxM0": 64,
        "M": 32,
        "ef_construction": 100,
    }
    errors = []
    for field, wanted in expected.items():
        if values[field] != wanted:
            errors.append(f"{path}: {field}={values[field]} expected={wanted}")
    wanted_mult = 1.0 / math.log(32)
    if not math.isclose(
        float(values["mult"]), wanted_mult, rel_tol=0.0, abs_tol=1e-15
    ):
        errors.append(f"{path}: mult={values['mult']} expected={wanted_mult}")
    return errors


def read_matrix(path: Path, rows: int, columns: int) -> bytes:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"short matrix header: {path}")
    actual_rows, actual_columns = struct.unpack("<ii", raw[:8])
    if (actual_rows, actual_columns) != (rows, columns):
        raise ValueError(
            f"{path}: shape={(actual_rows, actual_columns)} "
            f"expected={(rows, columns)}"
        )
    expected_bytes = 8 + rows * columns * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path}: bytes={len(raw)} expected={expected_bytes}"
        )
    return raw[8:]


def validate_queries(path: Path, dimension: int) -> list[str]:
    if not path.is_file():
        return [f"missing query matrix: {path}"]
    try:
        payload = read_matrix(path, 1000, dimension)
    except (OSError, ValueError) as exc:
        return [str(exc)]

    errors = []
    values = struct.iter_unpack("<f", payload)
    for row in range(1000):
        squared_norm = 0.0
        row_has_nonfinite = False
        for _ in range(dimension):
            value = next(values)[0]
            if not math.isfinite(value):
                row_has_nonfinite = True
            else:
                squared_norm += value * value
        if row_has_nonfinite:
            errors.append(f"{path}: query row {row} contains a non-finite value")
        else:
            norm = math.sqrt(squared_norm)
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2.0e-4):
                errors.append(
                    f"{path}: query row {row} norm={norm:.9g}, expected 1"
                )
        if len(errors) >= 10:
            errors.append(f"{path}: stopping after 10 query errors")
            break
    return errors


def read_and_validate_gt(
    path: Path, top_k: int, element_count: int
) -> tuple[list[str], list[int] | None]:
    if not path.is_file():
        return [f"missing ground truth: {path}"], None
    try:
        payload = read_matrix(path, 1000, top_k)
    except (OSError, ValueError) as exc:
        return [str(exc)], None
    labels = [item[0] for item in struct.iter_unpack("<I", payload)]
    errors = []
    for row in range(1000):
        begin = row * top_k
        row_labels = labels[begin : begin + top_k]
        if any(label >= element_count for label in row_labels):
            errors.append(f"{path}: row {row} contains an out-of-range label")
        if len(set(row_labels)) != top_k:
            errors.append(f"{path}: row {row} contains duplicate labels")
        if len(errors) >= 10:
            errors.append(f"{path}: stopping after 10 GT errors")
            break
    return errors, labels


def validate_dataset(
    artifact_root: Path, dataset: str, top_ks: set[int]
) -> list[str]:
    element_count, dimension = DATASET_SPECS[dataset]
    root = artifact_root / dataset
    errors = validate_hnsw(
        root / "hnsw_index_M32_ef100.bin", element_count
    )
    errors.extend(
        validate_queries(root / "query_vectors_n1000_seed42.bin", dimension)
    )

    labels_by_k: dict[int, list[int]] = {}
    for top_k in sorted(top_ks):
        gt_errors, labels = read_and_validate_gt(
            root / f"gt_labels_topk{top_k}_n1000_seed42.bin",
            top_k,
            element_count,
        )
        errors.extend(gt_errors)
        if labels is not None:
            labels_by_k[top_k] = labels

    if 100 in labels_by_k:
        widest = labels_by_k[100]
        for top_k in (5, 10):
            if top_k not in labels_by_k:
                continue
            prefix = labels_by_k[top_k]
            for row in range(1000):
                narrow_labels = set(prefix[row * top_k : (row + 1) * top_k])
                widest_labels = set(widest[row * 100 : (row + 1) * 100])
                if not narrow_labels.issubset(widest_labels):
                    errors.append(
                        f"{dataset}: top-{top_k} GT is not contained in top-100 "
                        f"at row {row}"
                    )
                    break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        help="repository-relative final YAMLs; omit to check all 35 artifacts",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="flat <dataset>/ artifact root (default: mfnns_hnswlib/cpu_index)",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root
    if not artifact_root.is_absolute():
        artifact_root = REPO_ROOT / artifact_root

    try:
        selections = config_selections(args.configs)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    errors = []
    for dataset, top_ks in selections.items():
        errors.extend(validate_dataset(artifact_root, dataset, top_ks))
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(
            f"INPUTS_FAILED datasets={len(selections)} errors={len(errors)}",
            file=sys.stderr,
        )
        return 1

    gt_count = sum(len(top_ks) for top_ks in selections.values())
    print(
        f"INPUTS_OK datasets={len(selections)} indexes={len(selections)} "
        f"queries={len(selections)} gt={gt_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
