#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc2hdd/home/rmeng603/workspace/MFANNS
EXP=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260329/mfnns_t2i_k10_ef20_30_40_dualq_q20_100
BUILD_DIR=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/build_compute_sysgcc
RAMULATOR_BIN=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/build_compute_sysgcc/ramulator2
RUN_ROOT="$EXP/runs/formal"

python3 "$EXP/generate_cases.py"

mapfile -t YAMLS < <(python3 - <<'PY'
import csv
from pathlib import Path
manifest = Path('/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260329/mfnns_t2i_k10_ef20_30_40_dualq_q20_100/case_manifest.tsv')
with manifest.open() as fin:
    for row in csv.DictReader(fin, delimiter='\t'):
        print(row['yaml_path'])
PY
)

python3 "$ROOT/simulator/memory/run_yaml_case.py" \
  --launcher sbatch \
  --partition i64m512u \
  --time-limit 02:00:00 \
  --mem 128G \
  --cpus-per-task 4 \
  --exclude cpu1-73 \
  --skip-build \
  --build-dir "$BUILD_DIR" \
  --ramulator-bin "$RAMULATOR_BIN" \
  --result-root "$RUN_ROOT" \
  --job-name-prefix t2idq \
  "${YAMLS[@]}"

python3 - <<'PY'
import csv
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

summary = Path('/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260329/mfnns_t2i_k10_ef20_30_40_dualq_q20_100/runs/formal/summary.tsv')
with summary.open() as fin:
    job_ids = {
        row['slurm_job_id']
        for row in csv.DictReader(fin, delimiter='\t')
        if row['slurm_job_id'] != 'NA'
    }

user = os.environ.get('USER', '')
while True:
    proc = subprocess.run(
        ['squeue', '-h', '-u', user, '-o', '%i %T %j'],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active_rows = []
    states = Counter()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, state, *name_parts = line.split()
        if job_id not in job_ids:
            continue
        active_rows.append((job_id, state, ' '.join(name_parts)))
        states[state] += 1

    print(f"{datetime.now().strftime('%F %T')}\tactive={len(active_rows)}\tstates={dict(states)}", flush=True)
    if not active_rows:
        break
    time.sleep(30)
PY

sleep 10
python3 "$EXP/summarize_results.py" \
  --manifest "$EXP/case_manifest.tsv" \
  --run-summary "$RUN_ROOT/summary.tsv" \
  --output-summary "$EXP/summary_latest.tsv" \
  --output-report "$EXP/report.md"
