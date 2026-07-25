#!/usr/bin/env python3
"""Collect the historical Figure 21 sweep into a portable AE bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

MFNNS_EXPERIMENT = Path(
    "simulator/memory/20260329/"
    "mfnns_t2i_k10_ef20_30_40_dualq_q20_100"
)
ANSMET_EXPERIMENT = Path(
    "simulator/memory/20260329/ansmet_recall09_k10_efsearch"
)
INPUT_CACHE = Path(
    "recall_analysis/memory/vectordb_recall_20260306/"
    "cache/t2i1m/normalized"
)

MODEL_NAME = "hnsw_index_M32_ef100.bin"
QUERY_NAME = "query_vectors_n100_seed42.bin"
GT_NAME = "gt_labels_topk32_n100_seed42.bin"

PORTABLE_MODEL = f"mfnns_hnswlib/cpu_index/t2i1m/{MODEL_NAME}"
PORTABLE_QUERY = f"ae/figure21/inputs/{QUERY_NAME}"
PORTABLE_GT = f"ae/figure21/inputs/{GT_NAME}"

PROVENANCE_FIELDS = [
    "method",
    "case_name",
    "ef_search",
    "queue_size",
    "warmup_size",
    "config_ref",
    "source_yaml_ref",
    "source_yaml_sha256",
    "portable_yaml_sha256",
    "source_stats_ref",
    "source_stats_sha256",
    "slurm_job_id",
    "recall",
    "s_mem_cycle",
    "final_status",
]

SWEEP_FIELDS = [
    "case_order",
    "method",
    "config_ref",
    "ef_search",
    "queue_size",
    "warmup_size",
    "recall",
    "s_mem_cycle",
    "final_status",
    "slurm_job_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-root",
        type=Path,
        required=True,
        help="Path to the full MFANNS author workspace.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def extract_scalar(text: str, key: str) -> str:
    match = re.search(
        rf"^\s{{2}}{re.escape(key)}:\s*([^\s#]+)",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Missing top-level Frontend key: {key}")
    return match.group(1)


def replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"^(\s{{2}}{re.escape(key)}:)\s*.*$",
        flags=re.MULTILINE,
    )
    updated, count = pattern.subn(rf"\1 {value}", text, count=1)
    if count != 1:
        raise ValueError(f"Expected one top-level Frontend key: {key}")
    return updated


def portable_yaml(
    source: Path,
    source_ref: str,
    method: str,
    stat_name: str,
) -> str:
    source_digest = sha256(source)
    text = source.read_text(encoding="utf-8")
    text = replace_scalar(text, "model_path", PORTABLE_MODEL)
    text = replace_scalar(text, "query_path", PORTABLE_QUERY)
    text = replace_scalar(text, "gt_path", PORTABLE_GT)
    text = replace_scalar(text, "stat_path", f"stats/{method}/{stat_name}")
    header = "\n".join(
        [
            "# Figure 21 AE simulator configuration",
            f"# source_yaml: MFANNS/{source_ref}",
            f"# source_yaml_sha256: {source_digest}",
            "# Input paths are repository-relative. The Figure 21 runner resolves",
            "# them to absolute paths in a task-specific runtime copy.",
            "",
        ]
    )
    return header + text


def source_ref(path: Path, author_root: Path) -> str:
    return path.resolve().relative_to(author_root.resolve()).as_posix()


def find_job_id(experiment: Path, yaml_name: str) -> str:
    matches: list[tuple[float, str]] = []
    for record_path in experiment.glob("runs/**/run_record.tsv"):
        record: dict[str, str] = {}
        for line in record_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            record[key] = value
        if Path(record.get("yaml_path", "")).name != yaml_name:
            continue
        job_id = record.get("slurm_job_id", "NA")
        matches.append((record_path.stat().st_mtime, job_id))
    if not matches:
        return "NA"
    matches.sort()
    return matches[-1][1]


def collect_main_sweep(
    author_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    experiment = author_root / MFNNS_EXPERIMENT
    summary_rows = read_tsv(experiment / "summary_latest.tsv")
    if len(summary_rows) != 243:
        raise ValueError(f"Expected 243 MFNNS rows, found {len(summary_rows)}")

    config_dir = PACKAGE_DIR / "configs/mfnns"
    config_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []

    for case_order, row in enumerate(summary_rows, 1):
        case_name = row["case_name"]
        source_yaml = experiment / "yamls" / f"{case_name}.yaml"
        source_stats = experiment / "logs" / f"{case_name}_stats.yml"
        config_path = config_dir / source_yaml.name
        config_ref = config_path.relative_to(PACKAGE_DIR.parents[1]).as_posix()
        config_path.write_text(
            portable_yaml(
                source_yaml,
                source_ref(source_yaml, author_root),
                "mfnns",
                source_stats.name,
            ),
            encoding="utf-8",
        )

        sweep_rows.append(
            {
                "case_order": str(case_order),
                "method": "mfnns",
                "config_ref": config_ref,
                "ef_search": row["ef_search"],
                "queue_size": row["queue_size"],
                "warmup_size": row["warmup_size"],
                "recall": row["recall"],
                "s_mem_cycle": row["s_mem_cycle"],
                "final_status": row["final_status"],
                "slurm_job_id": row["slurm_job_id"],
            }
        )
        provenance_rows.append(
            {
                "method": "mfnns",
                "case_name": case_name,
                "ef_search": row["ef_search"],
                "queue_size": row["queue_size"],
                "warmup_size": row["warmup_size"],
                "config_ref": config_ref,
                "source_yaml_ref": f"MFANNS/{source_ref(source_yaml, author_root)}",
                "source_yaml_sha256": sha256(source_yaml),
                "portable_yaml_sha256": sha256(config_path),
                "source_stats_ref": f"MFANNS/{source_ref(source_stats, author_root)}",
                "source_stats_sha256": sha256(source_stats),
                "slurm_job_id": row["slurm_job_id"],
                "recall": row["recall"],
                "s_mem_cycle": row["s_mem_cycle"],
                "final_status": row["final_status"],
            }
        )

    return sweep_rows, provenance_rows


def collect_ansmet(author_root: Path) -> list[dict[str, str]]:
    experiment = author_root / ANSMET_EXPERIMENT
    config_dir = PACKAGE_DIR / "configs/ansmet"
    stats_dir = PACKAGE_DIR / "data/ansmet_stats"
    config_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for ef_search in (20, 30, 40):
        case_name = f"t2i1m_normalized_k10_ef{ef_search}"
        source_yaml = experiment / "yamls" / f"{case_name}.yaml"
        source_stats = experiment / "logs" / f"{case_name}_stats.yml"
        config_path = config_dir / source_yaml.name
        config_ref = config_path.relative_to(PACKAGE_DIR.parents[1]).as_posix()
        config_path.write_text(
            portable_yaml(
                source_yaml,
                source_ref(source_yaml, author_root),
                "ansmet",
                source_stats.name,
            ),
            encoding="utf-8",
        )
        shutil.copy2(source_stats, stats_dir / source_stats.name)
        yaml_text = source_yaml.read_text(encoding="utf-8")
        stats_text = source_stats.read_text(encoding="utf-8")
        rows.append(
            {
                "method": "ansmet",
                "case_name": case_name,
                "ef_search": str(ef_search),
                "queue_size": extract_scalar(
                    yaml_text, "dualQueueLowerBoundQueueSize"
                ),
                "warmup_size": extract_scalar(
                    yaml_text, "dualQueueLowerBoundWarmupSize"
                ),
                "config_ref": config_ref,
                "source_yaml_ref": f"MFANNS/{source_ref(source_yaml, author_root)}",
                "source_yaml_sha256": sha256(source_yaml),
                "portable_yaml_sha256": sha256(config_path),
                "source_stats_ref": f"MFANNS/{source_ref(source_stats, author_root)}",
                "source_stats_sha256": sha256(source_stats),
                "slurm_job_id": find_job_id(experiment, source_yaml.name),
                "recall": extract_scalar(stats_text, "s_recall_rate"),
                "s_mem_cycle": extract_scalar(stats_text, "s_mem_cycle"),
                "final_status": "PASS",
            }
        )
    return rows


def collect_inputs(author_root: Path) -> None:
    source_dir = author_root / INPUT_CACHE
    input_dir = PACKAGE_DIR / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for name, included, purpose in [
        (MODEL_NAME, False, "T2I1M normalized HNSW M32 efConstruction100 index"),
        (QUERY_NAME, True, "Exact 100 normalized query vectors used by Figure 21"),
        (GT_NAME, True, "Exact top-32 ground truth used by Figure 21"),
    ]:
        source = source_dir / name
        if included:
            shutil.copy2(source, input_dir / name)
        rows.append(
            {
                "artifact": name,
                "included": "yes" if included else "no",
                "size_bytes": str(source.stat().st_size),
                "sha256": sha256(source),
                "source_ref": f"MFANNS/{source_ref(source, author_root)}",
                "purpose": purpose,
            }
        )

    write_tsv(
        PACKAGE_DIR / "data/input_manifest.tsv",
        [
            "artifact",
            "included",
            "size_bytes",
            "sha256",
            "source_ref",
            "purpose",
        ],
        rows,
    )


def collect_historical_scripts(author_root: Path) -> None:
    destinations = {
        MFNNS_EXPERIMENT / "commands.sh": "mfnns_commands.sh",
        MFNNS_EXPERIMENT / "generate_cases.py": "mfnns_generate_cases.py",
        MFNNS_EXPERIMENT / "summarize_results.py": "mfnns_summarize_results.py",
        MFNNS_EXPERIMENT / "case_manifest.tsv": "mfnns_case_manifest.tsv",
        MFNNS_EXPERIMENT / "report.md": "mfnns_report.md",
        MFNNS_EXPERIMENT / "error_log.md": "mfnns_error_log.md",
        ANSMET_EXPERIMENT / "commands.sh": "ansmet_commands.sh",
        ANSMET_EXPERIMENT / "generate_cases.py": "ansmet_generate_cases.py",
        ANSMET_EXPERIMENT / "generate_cases_stage2.py": "ansmet_generate_cases_stage2.py",
        ANSMET_EXPERIMENT / "summarize_results.py": "ansmet_summarize_results.py",
        ANSMET_EXPERIMENT / "case_manifest.tsv": "ansmet_case_manifest.tsv",
        ANSMET_EXPERIMENT / "report_final.md": "ansmet_report_final.md",
        ANSMET_EXPERIMENT / "error_log.md": "ansmet_error_log.md",
        Path("figure/evaluation/et_para/plot_t2i_lbq_recall_throughput.py"):
            "author_plot_t2i_lbq_recall_throughput.py",
    }
    output_dir = PACKAGE_DIR / "scripts/historical"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_relative, output_name in destinations.items():
        source = author_root / source_relative
        shutil.copy2(source, output_dir / output_name)

    checksum_lines = [
        f"{sha256(path)}  {path.name}\n"
        for path in sorted(output_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (output_dir / "SHA256SUMS").write_text(
        "".join(checksum_lines), encoding="utf-8"
    )


def write_config_checksums() -> None:
    config_dir = PACKAGE_DIR / "configs"
    yaml_paths = sorted(config_dir.glob("*/*.yaml"))
    if len(yaml_paths) != 246:
        raise ValueError(f"Expected 246 portable YAMLs, found {len(yaml_paths)}")
    lines = [
        f"{sha256(path)}  {path.relative_to(config_dir).as_posix()}\n"
        for path in yaml_paths
    ]
    (config_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    author_root = args.author_root.resolve()
    if not (author_root / MFNNS_EXPERIMENT).is_dir():
        raise FileNotFoundError(f"Missing author experiment under {author_root}")

    sweep_rows, provenance_rows = collect_main_sweep(author_root)
    provenance_rows.extend(collect_ansmet(author_root))
    collect_inputs(author_root)
    collect_historical_scripts(author_root)
    write_config_checksums()

    write_tsv(
        PACKAGE_DIR / "data/figure21_sweep_results.tsv",
        SWEEP_FIELDS,
        sweep_rows,
    )
    write_tsv(
        PACKAGE_DIR / "data/simulator_provenance.tsv",
        PROVENANCE_FIELDS,
        provenance_rows,
    )
    print(
        "COLLECT_OK "
        f"sweep={len(sweep_rows)} provenance={len(provenance_rows)} "
        "configs=246 inputs=2"
    )


if __name__ == "__main__":
    main()
