#!/usr/bin/env python3
"""Summarize ANSMET k=10 ef_search sweep results."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

TARGET_RECALL_LO = 0.9
TARGET_RECALL_HI = 0.91
MANIFEST_KEY = "yaml_path"
SUMMARY_FIELDS = [
    "dataset", "variant", "search_role", "case_name",
    "ef_search", "queue_size", "warmup_size", "k_neighbors",
    "slurm_job_id", "slurm_state", "exit_code", "final_status",
    "recall", "recall_gap_lo", "recall_gap_hi", "in_band",
    "avg_total_latency", "s_mem_cycle", "s_mem_read_req",
    "yaml_path", "stats_path", "slurm_stdout_log", "slurm_stderr_log", "note",
]


def read_tsv(path: Path) -> list[dict]:
    with path.open() as fin:
        return list(csv.DictReader(fin, delimiter="\t"))


def parse_top_frontend_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    lines = path.read_text(errors="ignore").splitlines()
    in_top = False
    for line in lines:
        if not in_top:
            if line == "Frontend:":
                in_top = True
            continue
        if line.startswith("  Frontend:"):
            break
        if not line.startswith("  "):
            break
        match = re.match(r"^  ([A-Za-z0-9_]+):\s*(.+?)\s*$", line)
        if match:
            stats[match.group(1)] = match.group(2)
    return stats


def safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_sacct(job_ids: list[str]) -> dict[str, tuple[str, str]]:
    if not job_ids:
        return {}
    cmd = ["sacct", "-X", "-P", "-n", "-j", ",".join(job_ids), "--format=JobID,State,ExitCode"]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result: dict[str, tuple[str, str]] = {}
    if proc.returncode != 0:
        return result
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        job_id, state, exit_code = parts
        if not job_id or "." in job_id:
            continue
        if job_id not in result:
            result[job_id] = (state, exit_code)
    return result


def final_status(stats_exists: bool, slurm_state: str, exit_code: str, recall: float | None) -> str:
    state = slurm_state.upper()
    if stats_exists and state.startswith("COMPLETED") and (recall is None or recall != 0.0):
        return "PASS"
    if stats_exists and (state.startswith("FAILED") or state.startswith("CANCELLED") or exit_code not in {"0:0", "0"}):
        return "STATS_PRESENT_NONZERO_EXIT"
    if stats_exists and recall == 0.0:
        return "BAD_RECALL"
    if not stats_exists and state.startswith("COMPLETED"):
        return "MISSING_STATS"
    if state.startswith("PENDING") or state.startswith("RUNNING") or state.startswith("COMPLETING"):
        return "INCOMPLETE"
    if state:
        return state
    return "UNKNOWN"


def dataset_key(row: dict) -> str:
    return f"{row['dataset']}:{row['variant']}"


def choose_best(rows: list[dict]) -> dict:
    passed = [row for row in rows if row["final_status"] == "PASS"]
    in_band = [row for row in passed if row["in_band"] == "yes" and safe_float(row["s_mem_cycle"]) is not None]
    if in_band:
        return min(in_band, key=lambda r: float(r["s_mem_cycle"]))
    near = [row for row in passed if safe_float(row["recall"]) is not None]
    if near:
        def closeness(r):
            recall = float(r["recall"])
            if recall < TARGET_RECALL_LO:
                return (TARGET_RECALL_LO - recall, float(r["s_mem_cycle"]) if safe_float(r["s_mem_cycle"]) else float("inf"))
            else:
                return (recall - TARGET_RECALL_HI, float(r["s_mem_cycle"]) if safe_float(r["s_mem_cycle"]) else float("inf"))
        return min(near, key=closeness)
    return rows[0]


def build_report(rows: list[dict]) -> str:
    lines = []
    lines.append(f"# ANSMET recall@10 [{TARGET_RECALL_LO}, {TARGET_RECALL_HI}) ef_search parameter search")
    lines.append("")
    lines.append(f"- **k_neighbors**: 10")
    lines.append(f"- **Target recall band**: `[{TARGET_RECALL_LO}, {TARGET_RECALL_HI})`")
    lines.append("- **Objective**: minimize `s_mem_cycle` within the target recall band")
    lines.append("- **nFMAC**: 16")
    lines.append("- **nQueryLimit**: 100")
    lines.append("")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(dataset_key(row), []).append(row)

    lines.append("## Best Candidates")
    lines.append("")
    lines.append("| Dataset | ef_search | recall | s_mem_cycle | Status |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for key in sorted(grouped):
        best = choose_best(grouped[key])
        status = "✅ IN BAND" if best["in_band"] == "yes" else "❌ closest"
        lines.append(f"| **{best['dataset']}** | {best['ef_search']} | {best['recall']} | {best['s_mem_cycle']} | {status} |")
    lines.append("")

    lines.append("## Dataset Details")
    lines.append("")
    for key in sorted(grouped):
        dataset, variant = key.split(":", 1)
        lines.append(f"### {dataset} / {variant}")
        lines.append("")
        lines.append("| ef | recall | s_mem_cycle | in_band | status |")
        lines.append("| ---: | ---: | ---: | --- | --- |")
        ordered = sorted(grouped[key], key=lambda r: int(r["ef_search"]))
        for row in ordered:
            lines.append(f"| {row['ef_search']} | {row['recall']} | {row['s_mem_cycle']} | {row['in_band']} | {row['final_status']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_tsv(args.manifest)
    manifest = {row[MANIFEST_KEY]: row for row in manifest_rows}
    run_rows = read_tsv(args.run_summary)
    sacct_rows = query_sacct([row["slurm_job_id"] for row in run_rows if row.get("slurm_job_id") and row["slurm_job_id"] != "NA"])

    summary_rows: list[dict] = []
    for run_row in run_rows:
        meta = manifest.get(run_row["yaml_path"])
        if meta is None:
            continue
        stats_path = Path(meta["stats_path"])
        stats = parse_top_frontend_stats(stats_path)
        recall = safe_float(stats.get("s_recall_rate"))
        s_mem_cycle = safe_float(stats.get("s_mem_cycle"))
        slurm_state, exit_code = sacct_rows.get(run_row.get("slurm_job_id", ""), ("UNKNOWN", "NA"))
        recall_gap_lo = None if recall is None else recall - TARGET_RECALL_LO
        recall_gap_hi = None if recall is None else recall - TARGET_RECALL_HI
        in_band = recall is not None and TARGET_RECALL_LO <= recall < TARGET_RECALL_HI

        summary_rows.append({
            "dataset": meta["dataset"],
            "variant": meta["variant"],
            "search_role": meta["search_role"],
            "case_name": meta["case_name"],
            "ef_search": meta["ef_search"],
            "queue_size": meta["queue_size"],
            "warmup_size": meta["warmup_size"],
            "k_neighbors": meta.get("k_neighbors", "10"),
            "slurm_job_id": run_row.get("slurm_job_id", "NA"),
            "slurm_state": slurm_state,
            "exit_code": exit_code,
            "final_status": final_status(stats_path.exists(), slurm_state, exit_code, recall),
            "recall": stats.get("s_recall_rate", "NA"),
            "recall_gap_lo": "NA" if recall_gap_lo is None else f"{recall_gap_lo:.6f}",
            "recall_gap_hi": "NA" if recall_gap_hi is None else f"{recall_gap_hi:.6f}",
            "in_band": "yes" if in_band else "no",
            "avg_total_latency": stats.get("s_avg_total_latency", "NA"),
            "s_mem_cycle": stats.get("s_mem_cycle", "NA"),
            "s_mem_read_req": stats.get("s_mem_read_req", "NA"),
            "yaml_path": meta["yaml_path"],
            "stats_path": meta["stats_path"],
            "slurm_stdout_log": run_row.get("slurm_stdout_log", "NA"),
            "slurm_stderr_log": run_row.get("slurm_stderr_log", "NA"),
            "note": meta["note"],
        })

    with args.output_summary.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    args.output_report.write_text(build_report(summary_rows))
    print(f"summary\t{args.output_summary}")
    print(f"report\t{args.output_report}")


if __name__ == "__main__":
    main()
