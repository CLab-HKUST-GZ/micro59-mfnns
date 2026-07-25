#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

SUMMARY_FIELDS = [
    'dataset',
    'variant',
    'case_name',
    'ef_search',
    'queue_size',
    'warmup_size',
    'slurm_job_id',
    'slurm_state',
    'exit_code',
    'final_status',
    'recall',
    'avg_total_latency',
    's_mem_cycle',
    'query_per_kcycle',
    's_mem_read_req',
    'yaml_path',
    'stats_path',
    'slurm_stdout_log',
    'slurm_stderr_log',
    'note',
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as fin:
        return list(csv.DictReader(fin, delimiter='\t'))


def parse_top_frontend_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    lines = path.read_text(errors='ignore').splitlines()
    in_top = False
    for line in lines:
        if not in_top:
            if line == 'Frontend:':
                in_top = True
            continue
        if line.startswith('  Frontend:'):
            break
        if not line.startswith('  '):
            break
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            stats[parts[0].strip()] = parts[1].strip()
    return stats


def safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == 'NA':
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_sacct(job_ids: list[str]) -> dict[str, tuple[str, str]]:
    if not job_ids:
        return {}
    cmd = ['sacct', '-X', '-P', '-n', '-j', ','.join(job_ids), '--format=JobID,State,ExitCode']
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result: dict[str, tuple[str, str]] = {}
    if proc.returncode != 0:
        return result
    for line in proc.stdout.splitlines():
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
        job_id, state, exit_code = parts
        if not job_id or '.' in job_id:
            continue
        result.setdefault(job_id, (state, exit_code))
    return result


def final_status(stats_exists: bool, slurm_state: str, recall: float | None) -> str:
    state = slurm_state.upper()
    if stats_exists and state.startswith('COMPLETED') and (recall is None or recall != 0.0):
        return 'PASS'
    if not stats_exists and state.startswith('COMPLETED'):
        return 'MISSING_STATS'
    if stats_exists and recall == 0.0:
        return 'BAD_RECALL'
    if state.startswith('PENDING') or state.startswith('RUNNING'):
        return 'INCOMPLETE'
    if state:
        return state
    return 'UNKNOWN'


def pick_min_queue(rows: list[dict[str, str]], threshold: float) -> dict[str, str] | None:
    eligible = [row for row in rows if safe_float(row['recall']) is not None and safe_float(row['recall']) >= threshold]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            int(row['queue_size']),
            safe_float(row['s_mem_cycle']) if safe_float(row['s_mem_cycle']) is not None else float('inf'),
        ),
    )


def pick_best_cycle(rows: list[dict[str, str]], threshold: float | None = None) -> dict[str, str] | None:
    eligible = rows
    if threshold is not None:
        eligible = [row for row in rows if safe_float(row['recall']) is not None and safe_float(row['recall']) >= threshold]
    eligible = [row for row in eligible if safe_float(row['s_mem_cycle']) is not None]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            safe_float(row['s_mem_cycle']),
            int(row['queue_size']),
            int(row['ef_search']),
        ),
    )


def build_report(rows: list[dict[str, str]]) -> str:
    lines = [
        '# MFNNS t2i k=10 ef_search={20,30,40} dualQueueLowerBoundQueueSize Sweep',
        '',
        '## Experiment Configuration',
        '',
        '- Dataset: `t2i1m / normalized`',
        '- Reference: `20260329/mfnns_etopt_hotrep_k10_ef_sweep` t2i setup',
        '- Fixed params: `k_neighbors=10`, `nQueryLimit=100`, `nParallelQuery=100`, `mfnnsEnable=true`, `dualQueueLowerBoundETEnable=true`',
        '- Sweep params: `ef_search in {20, 30, 40}`, `dualQueueLowerBoundQueueSize in [20, 100]`',
        '- Warmup rule: `dualQueueLowerBoundWarmupSize = dualQueueLowerBoundQueueSize - 1`',
        f'- Total rows: `{len(rows)}`',
        f'- PASS rows: `{sum(1 for row in rows if row["final_status"] == "PASS")}`',
        '',
    ]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row['ef_search']].append(row)

    lines.extend([
        '## Threshold Summary',
        '',
        '| ef_search | first q with recall>=0.90 | first q with recall>=0.91 | best s_mem_cycle | best s_mem_cycle with recall>=0.90 |',
        '| ---: | --- | --- | --- | --- |',
    ])
    for ef_search in sorted(grouped, key=int):
        items = sorted(grouped[ef_search], key=lambda row: int(row['queue_size']))
        q90 = pick_min_queue(items, 0.90)
        q91 = pick_min_queue(items, 0.91)
        best_any = pick_best_cycle(items)
        best_90 = pick_best_cycle(items, 0.90)

        def fmt_threshold(row: dict[str, str] | None) -> str:
            if row is None:
                return 'NA'
            return f"q={row['queue_size']} (recall={float(row['recall']):.3f}, cycle={row['s_mem_cycle']})"

        def fmt_best(row: dict[str, str] | None) -> str:
            if row is None:
                return 'NA'
            return f"q={row['queue_size']} (recall={float(row['recall']):.3f}, cycle={row['s_mem_cycle']})"

        lines.append(
            f"| {ef_search} | {fmt_threshold(q90)} | {fmt_threshold(q91)} | {fmt_best(best_any)} | {fmt_best(best_90)} |"
        )

    lines.extend(['', '## Full Data', '', '| ef_search | queue | recall | s_mem_cycle | avg_total_latency | s_mem_read_req | status |', '| ---: | ---: | ---: | ---: | ---: | ---: | --- |'])
    for row in rows:
        lines.append(
            f"| {row['ef_search']} | {row['queue_size']} | {row['recall']} | {row['s_mem_cycle']} | {row['avg_total_latency']} | {row['s_mem_read_req']} | {row['final_status']} |"
        )

    lines.extend(['', '## Output Files', '', '- `summary_latest.tsv`: all sweep rows with Slurm/run metadata', '- `case_manifest.tsv`: generated YAML/stat mapping', '- `runs/formal/summary.tsv`: raw launcher summary from `run_yaml_case.py`', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-summary', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_tsv(args.manifest)
    run_rows = read_tsv(args.run_summary)
    manifest = {row['yaml_path']: row for row in manifest_rows}
    sacct_rows = query_sacct([row['slurm_job_id'] for row in run_rows if row['slurm_job_id'] != 'NA'])

    summary_rows: list[dict[str, str]] = []
    for run_row in run_rows:
        meta = manifest[run_row['yaml_path']]
        stats_path = Path(meta['stats_path'])
        stats = parse_top_frontend_stats(stats_path)
        recall = safe_float(stats.get('s_recall_rate'))
        s_mem_cycle = safe_float(stats.get('s_mem_cycle'))
        query_per_kcycle = 'NA'
        if s_mem_cycle not in (None, 0.0):
            query_per_kcycle = f'{100.0 * 1000.0 / s_mem_cycle:.6f}'
        slurm_state, exit_code = sacct_rows.get(run_row['slurm_job_id'], ('UNKNOWN', 'NA'))
        summary_rows.append(
            {
                'dataset': meta['dataset'],
                'variant': meta['variant'],
                'case_name': meta['case_name'],
                'ef_search': meta['ef_search'],
                'queue_size': meta['queue_size'],
                'warmup_size': meta['warmup_size'],
                'slurm_job_id': run_row['slurm_job_id'],
                'slurm_state': slurm_state,
                'exit_code': exit_code,
                'final_status': final_status(stats_path.exists(), slurm_state, recall),
                'recall': stats.get('s_recall_rate', 'NA'),
                'avg_total_latency': stats.get('s_avg_total_latency', 'NA'),
                's_mem_cycle': stats.get('s_mem_cycle', 'NA'),
                'query_per_kcycle': query_per_kcycle,
                's_mem_read_req': stats.get('s_mem_read_req', 'NA'),
                'yaml_path': run_row['yaml_path'],
                'stats_path': meta['stats_path'],
                'slurm_stdout_log': run_row['slurm_stdout_log'],
                'slurm_stderr_log': run_row['slurm_stderr_log'],
                'note': meta.get('note', ''),
            }
        )

    summary_rows.sort(key=lambda row: (int(row['ef_search']), int(row['queue_size'])))
    with args.output_summary.open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(summary_rows)

    args.output_report.write_text(build_report(summary_rows))


if __name__ == '__main__':
    main()
