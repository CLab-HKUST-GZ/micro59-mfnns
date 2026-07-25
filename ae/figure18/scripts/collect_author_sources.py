#!/usr/bin/env python3
"""Build the Figure 18 AE data, YAML bundle, and provenance manifest.

This is an author-workspace collection tool.  The generated Figure 18 bundle
is portable, but this collector intentionally scans the historical MFANNS
memory tree to preserve source provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = Path("/hpc2hdd/home/rmeng603/workspace/MFANNS")
DEFAULT_WORKSPACE_ROOT = DEFAULT_SOURCE_ROOT.parent
DEFAULT_PLOT_CSV = (
    DEFAULT_SOURCE_ROOT
    / "figure/evaluation/recall_qps_curve/"
    "mfnns_recall_qps_2x2_singlecol_data.csv"
)
DEFAULT_MEMORY_ROOT = DEFAULT_SOURCE_ROOT / "simulator/memory"
DEFAULT_DATA_OUT = FIGURE_DIR / "data/figure18_recall_qps.csv"
DEFAULT_PROVENANCE_OUT = FIGURE_DIR / "data/simulator_provenance.tsv"
DEFAULT_CONFIG_ROOT = FIGURE_DIR / "configs"

SIMULATOR_METHODS = {"ansmet", "mfnns"}
EXPECTED_SIMULATOR_ROWS = 108
EXPECTED_STATUS_COUNTS = {
    "verified_original_yaml_stats": 88,
    "original_yaml_cycle_mismatch": 1,
    "reconstructed_from_same_panel_template": 19,
}

YAML_FIELD_PATTERN = (
    r"^\s+(model_path|query_path|gt_path|gt_k|stat_path|nQueryLimit|"
    r"nParallelQuery|ef_search|k_neighbors|dualQueueLowerBoundQueueSize|"
    r"dualQueueLowerBoundWarmupSize|mfnnsEnable):"
)
STAT_FIELDS = ("s_recall_rate", "s_mem_cycle", "s_num_query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_DATA_OUT)
    parser.add_argument("--provenance-out", type=Path, default=DEFAULT_PROVENANCE_OUT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def workspace_ref(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def repo_ref(path: Path) -> str:
    return str(path.resolve().relative_to(FIGURE_DIR.parents[1].resolve()))


def recall_2dp(text: str) -> str:
    return str(Decimal(text).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def parse_scalar(text: str) -> str:
    value = text.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def collect_yaml_fields(memory_root: Path) -> dict[Path, dict[str, str]]:
    command = [
        "rg",
        "-n",
        YAML_FIELD_PATTERN,
        str(memory_root),
        "-g",
        "*.yaml",
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    configs: dict[Path, dict[str, str]] = {}
    line_pattern = re.compile(
        r"^(.*?):([0-9]+):\s+([A-Za-z_]+):\s*(.*?)\s*$"
    )
    for line in result.stdout.splitlines():
        match = line_pattern.match(line)
        if not match:
            continue
        raw_path, _, field, value = match.groups()
        configs.setdefault(Path(raw_path), {})[field] = parse_scalar(value)
    return configs


def first_stat(text: str, key: str) -> float | None:
    match = re.search(
        rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def read_stats(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    values: dict[str, float] = {}
    for key in STAT_FIELDS:
        value = first_stat(text, key)
        if value is not None:
            values[key] = value
    return values


def classify_config(path: Path, fields: dict[str, str]) -> dict[str, object] | None:
    required = {
        "model_path",
        "ef_search",
        "k_neighbors",
        "dualQueueLowerBoundQueueSize",
    }
    if not required.issubset(fields):
        return None
    model = fields["model_path"].lower()
    if "deep1b" in model:
        dataset = "deep1b"
    elif "text2img1b" in model or "t2i1b" in model:
        dataset = "t2i1b"
    else:
        return None
    method = "mfnns" if fields.get("mfnnsEnable", "").lower() == "true" else "ansmet"
    stats_path = Path(fields["stat_path"]) if fields.get("stat_path") else None
    stats = read_stats(stats_path) if stats_path else {}
    return {
        "path": path,
        "fields": fields,
        "dataset": dataset,
        "method": method,
        "k": int(fields["k_neighbors"]),
        "ef": int(fields["ef_search"]),
        "lbq": int(fields["dualQueueLowerBoundQueueSize"]),
        "stats_path": stats_path,
        "stats": stats,
    }


def candidate_distance(
    candidate: dict[str, object], target_ef: int, target_lbq: int
) -> float:
    ef = int(candidate["ef"])
    lbq = int(candidate["lbq"])
    return abs(math.log((ef + 1) / (target_ef + 1))) + 0.25 * abs(
        math.log((lbq + 1) / (target_lbq + 1))
    )


def semantic_candidates(
    configs: list[dict[str, object]],
    row: dict[str, str],
) -> list[dict[str, object]]:
    k = 10 if row["recall_tag"] == "r10" else 100
    ef = int(row["ef"])
    candidates = [
        config
        for config in configs
        if config["dataset"] == row["dataset"]
        and config["method"] == row["method"]
        and config["k"] == k
        and config["ef"] == ef
    ]
    if row["method"] == "mfnns":
        lbq = int(row["lbq"])
        candidates = [config for config in candidates if config["lbq"] == lbq]
    return candidates


def exact_cycle_candidates(
    configs: list[dict[str, object]],
    row: dict[str, str],
) -> list[dict[str, object]]:
    target_cycle = int(row["cycle"])
    return [
        config
        for config in semantic_candidates(configs, row)
        if int(config["stats"].get("s_mem_cycle", -1)) == target_cycle
    ]


def choose_closest_recall(
    candidates: list[dict[str, object]], target_recall: float
) -> dict[str, object]:
    return min(
        candidates,
        key=lambda config: abs(
            float(config["stats"].get("s_recall_rate", -100.0)) - target_recall
        ),
    )


def choose_template(
    configs: list[dict[str, object]],
    row: dict[str, str],
) -> dict[str, object]:
    k = 10 if row["recall_tag"] == "r10" else 100
    target_ef = int(row["ef"])
    target_lbq = int(row["lbq"] or 0)
    candidates = [
        config
        for config in configs
        if config["dataset"] == row["dataset"]
        and config["method"] == row["method"]
        and config["k"] == k
    ]
    if not candidates:
        raise RuntimeError(
            "No same-panel template for "
            f"{row['dataset']} {row['recall_tag']} {row['method']}"
        )
    with_stats = [
        candidate
        for candidate in candidates
        if "s_mem_cycle" in candidate["stats"]
        and "s_recall_rate" in candidate["stats"]
    ]
    return min(
        with_stats or candidates,
        key=lambda candidate: candidate_distance(candidate, target_ef, target_lbq),
    )


def replace_top_level_int(text: str, key: str, value: int) -> str:
    pattern = re.compile(rf"^(  {re.escape(key)}:)\s*.*$", flags=re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1> {value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one top-level {key}, found {count}")
    return updated


def portable_config_text(
    source_path: Path,
    row: dict[str, str],
    status: str,
    relative_stat_path: str,
    source_ref: str,
) -> str:
    text = source_path.read_text(encoding="utf-8")
    if status == "reconstructed_from_same_panel_template":
        text = replace_top_level_int(text, "ef_search", int(row["ef"]))
        if row["method"] == "mfnns":
            lbq = int(row["lbq"])
            text = replace_top_level_int(
                text, "dualQueueLowerBoundQueueSize", lbq
            )
            text = replace_top_level_int(
                text, "dualQueueLowerBoundWarmupSize", max(0, lbq - 1)
            )
    stat_pattern = re.compile(r"^(  stat_path:)\s*.*$", flags=re.MULTILINE)
    text, count = stat_pattern.subn(
        rf"\g<1> {relative_stat_path}", text, count=1
    )
    if count != 1:
        raise RuntimeError(f"Expected one top-level stat_path in {source_path}")
    header = (
        "# Figure 18 AE simulator configuration\n"
        f"# provenance_status: {status}\n"
        f"# source_yaml: {source_ref}\n"
        "# stat_path is portable and resolved relative to the runtime workdir.\n"
    )
    if status == "reconstructed_from_same_panel_template":
        header += (
            "# ef_search/LBQueue were reconstructed from the same-panel template;\n"
            "# this file is a rerun recipe, not evidence of the frozen plotted point.\n"
        )
    elif status == "original_yaml_cycle_mismatch":
        header += (
            "# An original same-parameter YAML exists, but its recorded cycle differs\n"
            "# from the frozen plotting table; rerun before replacing frozen data.\n"
        )
    return header + text


def normalized_config_text(text: str, ignored_fields: set[str]) -> str:
    frontend = text.find("Frontend:")
    if frontend < 0:
        raise ValueError("Config is missing Frontend:")
    normalized = text[frontend:]
    for field in sorted(ignored_fields):
        pattern = re.compile(
            rf"^(  {re.escape(field)}:)\s*.*$", flags=re.MULTILINE
        )
        normalized, count = pattern.subn(
            rf"\g<1> <normalized>", normalized, count=1
        )
        if count != 1:
            raise ValueError(f"Config is missing top-level {field}")
    return normalized


def config_relative_path(row: dict[str, str], source_lbq: int) -> Path:
    base = Path(row["dataset"]) / row["recall_tag"] / row["method"]
    ef = int(row["ef"])
    if row["method"] == "mfnns":
        name = f"mfnns_ef{ef:04d}_lbq{int(row['lbq']):04d}.yaml"
    else:
        name = f"ansmet_ef{ef:04d}_queue{source_lbq:04d}.yaml"
    return base / name


def point_id(row: dict[str, str], source_lbq: int) -> str:
    suffix = (
        f"lbq{int(row['lbq']):04d}"
        if row["method"] == "mfnns"
        else f"queue{source_lbq:04d}"
    )
    return (
        f"{row['dataset']}.{row['recall_tag']}.{row['method']}."
        f"ef{int(row['ef']):04d}.{suffix}"
    )


def write_bundle(
    rows: list[dict[str, str]],
    configs: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    args.config_root.mkdir(parents=True, exist_ok=True)
    provenance_rows: list[dict[str, str]] = []
    manifest_lookup: dict[tuple[str, ...], dict[str, str]] = {}

    for row in rows:
        if row["method"] not in SIMULATOR_METHODS:
            continue
        target_recall = float(row["recall"])
        exact = exact_cycle_candidates(configs, row)
        semantic = semantic_candidates(configs, row)
        if exact:
            selected = choose_closest_recall(exact, target_recall)
            status = "verified_original_yaml_stats"
            note = "Original YAML and stats match dataset/method/k/ef/lbq/cycle."
        elif semantic:
            selected = choose_closest_recall(semantic, target_recall)
            status = "original_yaml_cycle_mismatch"
            note = (
                "Original same-parameter YAML exists, but its recorded cycle "
                "does not match the frozen plot row."
            )
        else:
            selected = choose_template(configs, row)
            status = "reconstructed_from_same_panel_template"
            note = (
                "No original same-parameter YAML was found; ef/LBQueue are "
                "reconstructed from the nearest same-panel template."
            )

        source_path = Path(selected["path"])
        source_stats_path = selected["stats_path"]
        source_stats = selected["stats"]
        source_lbq = int(selected["lbq"])
        relative_config = config_relative_path(row, source_lbq)
        portable_path = args.config_root / relative_config
        portable_path.parent.mkdir(parents=True, exist_ok=True)
        stat_name = portable_path.stem + "_stats.yml"
        relative_stat_path = f"stats/{stat_name}"
        source_yaml_ref = workspace_ref(source_path, args.workspace_root)
        config_text = portable_config_text(
            source_path,
            row,
            status,
            relative_stat_path,
            source_yaml_ref,
        )
        ignored_fields = {"stat_path"}
        if status == "reconstructed_from_same_panel_template":
            ignored_fields.update(
                {
                    "ef_search",
                    "dualQueueLowerBoundQueueSize",
                    "dualQueueLowerBoundWarmupSize",
                }
            )
        source_text = source_path.read_text(encoding="utf-8")
        if normalized_config_text(
            source_text, ignored_fields
        ) != normalized_config_text(config_text, ignored_fields):
            raise RuntimeError(
                f"Unexpected non-portable config change for {source_path}"
            )
        portable_path.write_text(config_text, encoding="utf-8")

        source_cycle_value = source_stats.get("s_mem_cycle")
        source_recall_value = source_stats.get("s_recall_rate")
        source_nq = source_stats.get("s_num_query")
        source_qps = (
            source_nq * 2_400_000_000.0 / source_cycle_value
            if source_nq and source_cycle_value
            else None
        )
        source_stats_ref = (
            workspace_ref(Path(source_stats_path), args.workspace_root)
            if source_stats_path
            else ""
        )
        source_stats_sha = (
            sha256(Path(source_stats_path))
            if source_stats_path and Path(source_stats_path).is_file()
            else ""
        )
        manifest_row = {
            "point_id": point_id(row, source_lbq),
            "dataset": row["dataset"],
            "recall_tag": row["recall_tag"],
            "method": row["method"],
            "plot_recall_raw": row["recall"],
            "plot_recall_2dp": recall_2dp(row["recall"]),
            "plot_qps": row["qps"],
            "plot_cycle": row["cycle"],
            "ef": row["ef"],
            "lbq": row["lbq"],
            "source_queue": str(source_lbq),
            "config_status": status,
            "portable_config_ref": repo_ref(portable_path),
            "portable_config_sha256": sha256_bytes(config_text.encode("utf-8")),
            "source_yaml_ref": source_yaml_ref,
            "source_yaml_sha256": sha256(source_path),
            "source_stats_ref": source_stats_ref,
            "source_stats_sha256": source_stats_sha,
            "source_recall": (
                "" if source_recall_value is None else f"{source_recall_value:.10g}"
            ),
            "source_cycle": (
                "" if source_cycle_value is None else str(int(source_cycle_value))
            ),
            "source_qps_2p4ghz": (
                "" if source_qps is None else f"{source_qps:.6f}"
            ),
            "recall_delta_plot_minus_source": (
                ""
                if source_recall_value is None
                else f"{target_recall-source_recall_value:.10g}"
            ),
            "cycle_delta_plot_minus_source": (
                ""
                if source_cycle_value is None
                else str(int(row["cycle"]) - int(source_cycle_value))
            ),
            "note": note,
        }
        provenance_rows.append(manifest_row)
        key = (
            row["dataset"],
            row["recall_tag"],
            row["method"],
            row["cycle"],
            row["ef"],
            row["lbq"],
        )
        if key in manifest_lookup:
            raise RuntimeError(f"Duplicate simulator plotting key: {key}")
        manifest_lookup[key] = manifest_row

    if len(provenance_rows) != EXPECTED_SIMULATOR_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SIMULATOR_ROWS} simulator rows, "
            f"found {len(provenance_rows)}"
        )
    actual_status_counts: dict[str, int] = {}
    for row in provenance_rows:
        status = row["config_status"]
        actual_status_counts[status] = actual_status_counts.get(status, 0) + 1
    if actual_status_counts != EXPECTED_STATUS_COUNTS:
        raise RuntimeError(
            f"Unexpected provenance counts: {actual_status_counts}; "
            f"expected {EXPECTED_STATUS_COUNTS}"
        )

    data_rows: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["dataset"],
            row["recall_tag"],
            row["method"],
            row["cycle"],
            row["ef"],
            row["lbq"],
        )
        provenance = manifest_lookup.get(key)
        data_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "recall_tag": row["recall_tag"],
                "recall_label": row["recall_label"],
                "method": row["method"],
                "method_label": row["method_label"],
                "recall_raw": row["recall"],
                "recall_2dp": recall_2dp(row["recall"]),
                "qps": row["qps"],
                "cycle": row["cycle"],
                "ef": row["ef"],
                "lbq": row["lbq"],
                "data_status": (
                    provenance["config_status"]
                    if provenance
                    else "external_frozen_plot_input"
                ),
                "config_ref": (
                    provenance["portable_config_ref"] if provenance else ""
                ),
            }
        )
    return data_rows, provenance_rows


def write_csv(path: Path, rows: list[dict[str, str]], delimiter: str = ",") -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.plot_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No plotting rows in {args.plot_csv}")

    raw_configs = collect_yaml_fields(args.memory_root)
    configs = [
        classified
        for path, fields in raw_configs.items()
        if (classified := classify_config(path, fields)) is not None
    ]
    data_rows, provenance_rows = write_bundle(rows, configs, args)
    write_csv(args.data_out, data_rows)
    write_csv(args.provenance_out, provenance_rows, delimiter="\t")
    print(f"Wrote {len(data_rows)} plotting rows to {args.data_out}")
    print(f"Wrote {len(provenance_rows)} simulator rows to {args.provenance_out}")
    print(f"Wrote {len(provenance_rows)} YAMLs under {args.config_root}")


if __name__ == "__main__":
    main()
